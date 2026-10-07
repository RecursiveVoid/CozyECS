/**
 * World.dispose(): the whole-world teardown level above destroy() / removeSystem() / clear().
 *
 * Two kinds of test live here:
 *  1. BEHAVIOUR -- what dispose() tears down, what throws afterwards, what stays safe. These run
 *     everywhere.
 *  2. COLLECTABILITY -- the actual point of the API: after dispose() and dropping the reference,
 *     the World, its archetypes and their ArrayBuffers must be unreachable. Proven with WeakRef
 *     plus `global.gc` and with `process.memoryUsage().arrayBuffers` over many build/dispose
 *     cycles, so they need jest under `--expose-gc` and skip cleanly (with a note) without it.
 *
 * Worlds here stay small (<= 20k entities, ~1 MB of table bytes each) on purpose: a leak shows up
 * in the number of cycles, not in the size of one world.
 */
import { describe, it, expect } from '@jest/globals';
import { World, component, tag, f32, f64, System, registerWorldDisposeHook } from '../src/index';
import type { Archetype, ComponentType, SystemHandle } from '../src/index';
import { _setTrampolineShapeLimit, _trampolineStats } from '../src/query';
import { _disposeHookCount } from '../src/world';

const Position = component({ x: f32, y: f32 }, { name: 'Position' });
const Velocity = component({ vx: f64, vy: f64 }, { name: 'Velocity' });
const Marker = tag({ name: 'Marker' });
const Fading = component({ t: f32 }, { name: 'Fading', enableable: true });

const ENTITIES = 20000;

/** A world with entities, archetypes, queries, listeners and systems, after a few ticks. */
function buildWorld(n: number = 2000): { world: World; arch: Archetype; first: number } {
  const world = new World({ initialCapacity: 256 });
  const arch = world.archetype(Position, Velocity);
  world.spawnMany(arch, n, (chunk, row, i) => {
    const pos = chunk.col(Position)!;
    pos.x[row] = i;
    pos.y[row] = -i;
  });
  const moving = world.query({ all: [Position, Velocity] });
  world.system('move', { query: moving }, (q, dt) => {
    q.forEach([Position, Velocity], (_e, _chunk, row, pos, vel) => {
      pos.x[row] += vel.vx[row] * dt;
    });
  });
  world.onAdd(Marker, () => {});
  moving.onEnter(() => {});
  world.update(1 / 60);
  const first = arch.entities[0];
  return { world, arch, first };
}

/** `global.gc` when jest runs under --expose-gc, else undefined. */
function gcOrNull(): (() => void) | undefined {
  const g = (globalThis as unknown as { gc?: () => void }).gc;
  return typeof g === 'function' ? g : undefined;
}

// eslint-disable-next-line no-console
const note = (s: string): void => console.log(`[dispose] ${s}`);

/**
 * Collects as hard as a test can: WeakRef targets survive until the end of the current job, so
 * every gc() must be followed by yielding to the macrotask queue.
 */
async function collect(gc: () => void): Promise<void> {
  for (let i = 0; i < 4; i++) {
    gc();
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
  }
}

// ---------------------------------------------------------------------------
// Idempotence and the post-dispose rule
// ---------------------------------------------------------------------------

describe('dispose() idempotence', () => {
  it('is a no-op the second time and flips `disposed`', () => {
    const { world } = buildWorld(64);
    expect(world.disposed).toBe(false);
    world.dispose();
    expect(world.disposed).toBe(true);
    expect(() => world.dispose()).not.toThrow();
    expect(() => world.dispose()).not.toThrow();
    expect(world.disposed).toBe(true);
  });

  it('refuses to run during iteration', () => {
    const { world } = buildWorld(16);
    const q = world.query({ all: [Position] });
    let thrown: unknown = null;
    q.forEach(() => {
      try {
        world.dispose();
      } catch (e) {
        thrown = e;
      }
    });
    expect(String(thrown)).toMatch(/cannot run during iteration/);
    expect(world.disposed).toBe(false);
    world.dispose();
  });
});

describe('after dispose(), every mutating call throws and names dispose()', () => {
  it('throws from every structural entry point', () => {
    const { world, arch, first } = buildWorld(64);
    world.dispose();

    const calls: [string, () => unknown][] = [
      ['spawn', () => world.spawn()],
      ['spawn', () => world.spawn(arch)],
      ['spawn', () => world.spawn([Position])],
      ['spawnMany', () => world.spawnMany(arch, 4)],
      ['destroy', () => world.destroy(first)],
      ['add', () => world.add(first, Marker)],
      ['add', () => world.add(first, Position, { x: 1 })],
      ['remove', () => world.remove(first, Position)],
      ['set', () => world.set(first, Position, { x: 1 })],
      ['enable', () => world.enable(first, Fading, false)],
      ['archetype', () => world.archetype(Position)],
      ['query', () => world.query({ all: [Position] })],
      ['onAdd', () => world.onAdd(Position, () => {})],
      ['onRemove', () => world.onRemove(Position, () => {})],
      ['system', () => world.system('s', {}, () => {})],
      ['addSystem', () => world.addSystem(class extends System { onUpdate(): void {} })],
      ['update', () => world.update(1 / 60)],
      ['compact', () => world.compact()],
      ['clear', () => world.clear()],
    ];
    for (const [method, call] of calls) {
      let message = '';
      try {
        call();
      } catch (e) {
        message = String(e);
      }
      expect(message).toContain(`world.${method}()`);
      expect(message).toContain('dispose()');
    }
  });

  it('throws from query subscriptions', () => {
    const { world } = buildWorld(16);
    const q = world.query({ all: [Position] });
    world.dispose();
    expect(() => q.onEnter(() => {})).toThrow(/query\.onEnter\(\).*dispose\(\)/);
    expect(() => q.onExit(() => {})).toThrow(/query\.onExit\(\).*dispose\(\)/);
  });

  it('keeps read-only calls safe', () => {
    const { world, arch, first } = buildWorld(128);
    const q = world.query({ all: [Position, Velocity] });
    expect(q.count()).toBe(128);
    world.dispose();

    expect(world.isAlive(first)).toBe(false);
    expect(world.isAlive(0)).toBe(false);
    expect(world.has(first, Position)).toBe(false);
    expect(world.get(first, Position)).toBeUndefined();
    expect(world.isEnabled(first, Fading)).toBe(false);
    expect(q.count()).toBe(0);
    expect(q.chunks.length).toBe(0);
    let visited = 0;
    q.forEach(() => visited++);
    q.forEach([Position], () => visited++);
    q.forEachChunk([Position], () => visited++);
    expect(visited).toBe(0);
    expect(() => world.flush()).not.toThrow();
    expect(arch.count).toBe(0);
  });

  it('zeroes memory()', () => {
    const { world } = buildWorld(1000);
    const before = world.memory();
    expect(before.entities).toBe(1000);
    expect(before.tables.reserved).toBeGreaterThan(0);
    world.dispose();
    const after = world.memory();
    expect(after.entities).toBe(0);
    expect(after.tables).toEqual({ used: 0, reserved: 0 });
    expect(after.entityIndex).toBe(0);
    expect(after.total).toBe(0);
    expect(after.archetypes).toEqual([]);
    // The string table keeps only its structural id 0 ('').
    expect(after.strings.count).toBe(1);
  });

  it('leaves earlier unsubscribe functions callable', () => {
    const { world } = buildWorld(16);
    const q = world.query({ all: [Position] });
    const offHook = world.onAdd(Marker, () => {});
    const offEnter = q.onEnter(() => {});
    world.dispose();
    expect(() => offHook()).not.toThrow();
    expect(() => offEnter()).not.toThrow();
    expect(() => offHook()).not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

describe('dispose() tears systems down', () => {
  it('calls onDestroy once per class system, in reverse registration order', () => {
    const world = new World();
    const order: string[] = [];
    class A extends System {
      onUpdate(): void {}
      onDestroy(): void {
        order.push('A');
      }
    }
    class B extends System {
      onUpdate(): void {}
      onDestroy(): void {
        order.push('B');
      }
    }
    class C extends System {
      onUpdate(): void {}
      onDestroy(): void {
        order.push('C');
      }
    }
    world.addSystem(A);
    world.addSystem(B, { group: 'render' });
    world.addSystem(C, { order: -10 });
    world.dispose();
    expect(order).toEqual(['C', 'B', 'A']);
    world.dispose();
    expect(order).toEqual(['C', 'B', 'A']);
  });

  it('does not call onDestroy twice for a system already removed', () => {
    const world = new World();
    let destroyed = 0;
    class S extends System {
      onUpdate(): void {}
      onDestroy(): void {
        destroyed++;
      }
    }
    const s = world.addSystem(S);
    world.removeSystem(s);
    expect(destroyed).toBe(1);
    world.dispose();
    expect(destroyed).toBe(1);
  });

  it('survives an onDestroy that removes itself or throws', () => {
    const world = new World();
    const seen: string[] = [];
    class SelfRemoving extends System {
      onUpdate(): void {}
      onDestroy(): void {
        seen.push('self');
        this.world.removeSystem(this);
      }
    }
    class Throwing extends System {
      onUpdate(): void {}
      onDestroy(): void {
        seen.push('throw');
        throw new Error('onDestroy blew up');
      }
    }
    world.addSystem(SelfRemoving);
    world.addSystem(Throwing);
    // The error is rethrown, but only after the world is fully torn down.
    expect(() => world.dispose()).toThrow(/onDestroy blew up/);
    expect(seen).toEqual(['throw', 'self']);
    expect(world.disposed).toBe(true);
    expect(world.memory().total).toBe(0);
    expect(() => world.spawn()).toThrow(/dispose\(\)/);
  });

  it('disables and unregisters function systems', () => {
    const world = new World();
    let ran = 0;
    const handle: SystemHandle = world.system('tick', {}, () => {
      ran++;
    });
    world.update(1 / 60);
    expect(ran).toBe(1);
    world.dispose();
    expect(handle.enabled).toBe(false);
    expect(() => world.update(1 / 60)).toThrow(/dispose\(\)/);
    expect(ran).toBe(1);
  });

  it('makes removeSystem() a no-op afterwards', () => {
    const world = new World();
    const handle = world.system('tick', {}, () => {});
    world.dispose();
    expect(() => world.removeSystem(handle)).not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// Pending commands and tables
// ---------------------------------------------------------------------------

describe('dispose() discards pending commands without running them', () => {
  it('drops the queue left behind by a throwing listener', () => {
    const world = new World({ initialCapacity: 16 });
    const arch = world.archetype(Position);
    world.spawnMany(arch, 8);
    const q = world.query({ all: [Position] });
    let added = 0;
    world.onAdd(Marker, () => {
      added++;
      if (added === 1) throw new Error('listener blew up');
    });
    // The adds are deferred by forEach; the flush at the end of the iteration applies the first
    // one, the listener throws, and the rest stay queued (documented flush() behaviour).
    expect(() => q.forEach((e) => world.add(e, Marker))).toThrow(/listener blew up/);
    expect(world._commands.length).toBe(7);
    expect(added).toBe(1);

    world.dispose();
    expect(world._commands.length).toBe(0);
    // Nothing queued was applied: the listener never fired again.
    expect(added).toBe(1);
  });
});

describe('dispose() releases the tables', () => {
  it('deflates every archetype to capacity 0 and clears its edges', () => {
    const { world } = buildWorld(1000);
    const pv = world.archetype(Position, Velocity);
    const pvm = world.archetype(Position, Velocity, Marker);
    expect(pv.capacity).toBeGreaterThan(0);
    expect(pv.buffer.byteLength).toBeGreaterThan(0);
    const e = pv.entities[0];
    world.add(e, Marker); // populates the add/remove transition edges
    expect(pv.edgesAdd.size).toBeGreaterThan(0);

    world.dispose();

    for (const a of [pv, pvm]) {
      expect(a.count).toBe(0);
      expect(a.capacity).toBe(0);
      expect(a.buffer.byteLength).toBe(0);
      expect(a.entities.length).toBe(0);
      expect(a.col(Position)!.x.length).toBe(0);
      expect(a.edgesAdd.size).toBe(0);
      expect(a.edgesRemove.size).toBe(0);
    }
  });
});

// ---------------------------------------------------------------------------
// The teardown-hook seam (what cozyecs/gpu registers into)
// ---------------------------------------------------------------------------

describe('registerWorldDisposeHook()', () => {
  it('runs hooks with the world still readable, then unregisters cleanly', () => {
    const seen: { entities: number; archetypes: number; world: boolean }[] = [];
    const before = _disposeHookCount();
    const off = registerWorldDisposeHook((w) => {
      const mem = w.memory();
      seen.push({ entities: mem.entities, archetypes: mem.archetypes.length, world: w instanceof World });
    });
    expect(_disposeHookCount()).toBe(before + 1);

    const { world } = buildWorld(512);
    world.dispose();
    expect(seen).toEqual([{ entities: 512, archetypes: seen[0]?.archetypes ?? 0, world: true }]);
    expect(seen[0].archetypes).toBeGreaterThan(0);

    // A second dispose() of the same world does not re-run hooks.
    world.dispose();
    expect(seen.length).toBe(1);

    off();
    off();
    expect(_disposeHookCount()).toBe(before);
    const other = buildWorld(16);
    other.world.dispose();
    expect(seen.length).toBe(1);
  });

  it('disposes the world completely even when a hook throws, and rethrows after', () => {
    const off = registerWorldDisposeHook(() => {
      throw new Error('hook blew up');
    });
    try {
      const { world } = buildWorld(256);
      expect(() => world.dispose()).toThrow(/hook blew up/);
      expect(world.disposed).toBe(true);
      expect(world.memory().total).toBe(0);
      expect(() => world.spawn()).toThrow(/dispose\(\)/);
    } finally {
      off();
    }
  });

  it('rejects a non-function hook', () => {
    expect(() => registerWorldDisposeHook(undefined as unknown as () => void)).toThrow(TypeError);
  });
});

// ---------------------------------------------------------------------------
// Collectability (the point of the API)
// ---------------------------------------------------------------------------

interface Refs {
  world: WeakRef<World>;
  arch: WeakRef<Archetype>;
  buffer: WeakRef<ArrayBufferLike>;
  bytes: number;
}

/**
 * Builds a fully exercised world, takes weak references to it, to one archetype and to that
 * archetype's table buffer, disposes it and returns only the weak references -- no strong
 * reference to anything inside survives this call.
 */
/**
 * Like {@link buildAndDispose}, but over caller-supplied components so the forEach loop
 * shape is unique to one test. Built in a plain function on purpose: locals of an async
 * test body stay reachable through its suspended frame, which would mask collectability.
 */
function buildAndDisposeWith(A: ComponentType, B: ComponentType, n: number): Refs {
  const world = new World({ initialCapacity: 256 });
  const arch = world.archetype(A, B);
  world.spawnMany(arch, n, () => {});
  const q = world.query({ all: [A, B] });
  // One stable callback: a plan (and therefore a shape build) is only attempted once the
  // same function object is seen again, so a fresh arrow per tick would never get there.
  const body = (): void => {};
  world.system('oddMove', { query: q }, (query) => {
    query.forEach([A, B], body);
  });
  // Several ticks: a compiled loop is only attempted once the same callback has been
  // seen again, so one tick would never reach the shape-limit check.
  for (let i = 0; i < 4; i++) world.update(1 / 60);
  const refs: Refs = {
    world: new WeakRef(world),
    arch: new WeakRef(arch),
    buffer: new WeakRef(arch.buffer),
    bytes: world.memory().total,
  };
  world.dispose();
  return refs;
}

function buildAndDispose(n: number, dispose: boolean = true): Refs {
  const { world, arch } = buildWorld(n);
  const refs: Refs = {
    world: new WeakRef(world),
    arch: new WeakRef(arch),
    buffer: new WeakRef(arch.buffer),
    bytes: world.memory().total,
  };
  if (dispose) world.dispose();
  return refs;
}

describe('a disposed world becomes collectable', () => {
  it('leaves no strong reference to the world, its archetypes or their buffers', async () => {
    const gc = gcOrNull();
    if (!gc) {
      note('collectability skipped: global.gc is unavailable (run jest under --expose-gc)');
      return;
    }
    const refs = buildAndDispose(ENTITIES);
    expect(refs.bytes).toBeGreaterThan(500_000);
    await collect(gc);
    expect(refs.world.deref()).toBeUndefined();
    expect(refs.arch.deref()).toBeUndefined();
    expect(refs.buffer.deref()).toBeUndefined();
  });

  it('collects a world whose forEach took the generic per-chunk fallback', async () => {
    const gc = gcOrNull();
    if (!gc) {
      note('ONE_CHUNK collectability skipped: global.gc is unavailable');
      return;
    }
    // Shape bound 0 forces runPlan onto the generic loops, which pass chunks through the
    // module-level ONE_CHUNK scratch array. A chunk left there would outlive the world.
    // Components unique to this test, so the loop shape is one no earlier test has
    // compiled: with the limit at 0 the build is refused and runPlan falls back to the
    // generic per-chunk path. (Reusing Position/Velocity would hit the shape cache and
    // never reject, which made this assertion order-dependent.)
    const Odd = component({ a: f32, b: f32 }, { name: 'DisposeOddA' });
    const Ball = component({ c: f32, d: f32, e: f32 }, { name: 'DisposeOddB' });
    const prev = _setTrampolineShapeLimit(0);
    const rejectionsBefore = _trampolineStats().shapeRejections;
    let refs: Refs;
    try {
      refs = buildAndDisposeWith(Odd, Ball, 2000);
      expect(_trampolineStats().shapeRejections).toBeGreaterThan(rejectionsBefore);
    } finally {
      _setTrampolineShapeLimit(prev);
    }
    await collect(gc);
    expect(refs.arch.deref()).toBeUndefined();
    expect(refs.world.deref()).toBeUndefined();
    expect(refs.buffer.deref()).toBeUndefined();
  });

  it('does not grow ArrayBuffer memory over many build/dispose cycles', async () => {
    const gc = gcOrNull();
    if (!gc) {
      note('cycle test skipped: global.gc is unavailable');
      return;
    }
    const CYCLES = 30;
    let perWorld = 0;
    for (let i = 0; i < 3; i++) perWorld = buildAndDispose(ENTITIES).bytes; // warm up

    gc();
    gc();
    const base = process.memoryUsage().arrayBuffers;
    for (let i = 0; i < CYCLES; i++) buildAndDispose(ENTITIES);
    await collect(gc);
    const after = process.memoryUsage().arrayBuffers;
    const growth = after - base;
    note(
      `${CYCLES} build+dispose cycles of ${ENTITIES} entities (${(perWorld / 1024).toFixed(0)} KiB of tables each): ` +
        `arrayBuffers ${(base / 1024).toFixed(0)} -> ${(after / 1024).toFixed(0)} KiB (${(growth / 1024).toFixed(0)} KiB)`,
    );
    // Leaking even one world per cycle would be ~CYCLES * perWorld bytes.
    expect(growth).toBeLessThan(perWorld * 3);
  });
});
