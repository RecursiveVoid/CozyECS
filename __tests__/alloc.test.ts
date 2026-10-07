/// <reference types="node" />
/**
 * Allocation guards for the steady-state tick.
 *
 * CozyECS promises that a tick over an unchanging entity set allocates nothing: component storage
 * lives in pre-allocated archetype tables, column objects and `world.get()` views are cached, and
 * the compiled chunk trampolines are reused once a query's shape has settled. That promise is easy
 * to break by accident -- a cache keyed on something that differs every tick, a per-row view
 * object, a closure built per chunk -- and the damage shows up in a real game as GC pauses, never
 * as a failing unit test. Hence this file.
 *
 * Two independent signals, because each catches what the other misses:
 *
 *  1. ALLOCATION RATE. `process.memoryUsage().heapUsed` grows monotonically between collections, so
 *     its delta across a window in which NO collection happened is the number of bytes allocated in
 *     that window -- garbage included. That last part is what matters here: per-tick garbage is
 *     invisible to any after-the-fact heap measurement, yet it is exactly what causes frame-time
 *     spikes. The window is validated two ways: a PerformanceObserver asserts no GC ran inside it
 *     (a scavenge would reset `heapUsed` and silently turn the measurement into an underestimate),
 *     and a control loop that allocates one small object per iteration must be measured at its real
 *     cost, which proves the meter works in this environment rather than reporting zero for
 *     everything.
 *
 *  2. RETAINED HEAP (`global.gc()` + `heapUsed` around blocks of ticks). Blind to garbage, but the
 *     only signal that catches something which ACCUMULATES: a Map that grows per tick, trampolines
 *     recompiled and kept, listener arrays that never shrink.
 *
 * Both need `global.gc`, so run them with `npm run test:alloc` (jest under `--expose-gc`); they
 * skip themselves with a note otherwise.
 *
 * Reference numbers from this fixture (10k entities, 3 systems, Apple M4, node 22):
 *
 *     empty arithmetic loop ............................   0.6 -  1.4 bytes/iteration
 *     a correct tick ...................................   1.2 -  1.9 bytes/tick
 *     the same tick with an inline forEachChunk kernel
 *       and an inline component list ...................   121 -  175 bytes/tick
 *     control: one 3-field object per iteration ........          64 bytes/iteration
 *
 * The systems below cover the three iteration styles, which carry different allocation risks: the
 * plain chunk loop (`q.chunks` + `chunk.col`), `forEachChunk` (compiled trampolines), and
 * `forEach(components, fn)` over a query with an enableable component (per-row enable checks).
 *
 * See docs/INTERNALS.md, "What allocates", for per-call measurements made with the same meter.
 */
import { describe, test, expect } from '@jest/globals';
import { PerformanceObserver } from 'node:perf_hooks';
import { World, component, tag, f32, i16, u8 } from '../src';
import type { Archetype, ColumnsOf } from '../src';

// ---------------------------------------------------------------------------- components

const posSchema = { x: f32, y: f32 } as const;
const velSchema = { dx: f32, dy: f32 } as const;
const healthSchema = { hp: i16 } as const;
const spriteSchema = { frame: u8 } as const;

const Position = component(posSchema, { name: 'AllocPosition' });
const Velocity = component(velSchema, { name: 'AllocVelocity' });
const Health = component(healthSchema, { name: 'AllocHealth', enableable: true });
const Sprite = component(spriteSchema, { name: 'AllocSprite' });
const Frozen = tag({ name: 'AllocFrozen' });

type PosCols = ColumnsOf<typeof posSchema>;
type VelCols = ColumnsOf<typeof velSchema>;
type HealthCols = ColumnsOf<typeof healthSchema>;

// ---------------------------------------------------------------------------- hot-loop constants
//
// Everything a tick touches is hoisted to module scope on purpose:
//  - the kernels, because the trampoline cache is keyed on function IDENTITY, so a lambda written
//    inline in the system body is a new function every tick and the plan is rebuilt forever;
//  - the component-list arrays, because `[Position, Velocity]` at the call site is a fresh array
//    every tick -- small, but exactly the kind of allocation this file exists to catch.

const PV = [Position, Velocity] as const;
const H = [Health] as const;

/** forEachChunk kernel: Position += Velocity, with the columns constant-folded by the trampoline. */
const integrate = (n: number, pos: PosCols, vel: VelCols): void => {
  const x = pos.x;
  const y = pos.y;
  const dx = vel.dx;
  const dy = vel.dy;
  for (let i = 0; i < n; i++) {
    x[i] += dx[i];
    y[i] += dy[i];
  }
};

/** Row callback for forEach(components, fn) over an enableable component (disabled rows skipped). */
let healthSeen = 0;
const regenerate = (_e: number, _chunk: Archetype, row: number, hp: HealthCols): void => {
  const a = hp.hp;
  const v = a[row] + 1;
  a[row] = v > 100 ? 100 : v;
  healthSeen++;
};

// ---------------------------------------------------------------------------- world fixture

const ENTITIES = 10_000;

function buildWorld(): { world: World; tick: () => void } {
  const world = new World({ initialCapacity: 1024 });

  const plain = world.archetype(Position, Velocity);
  const living = world.archetype(Position, Velocity, Health);
  const decor = world.archetype(Position, Velocity, Sprite, Frozen);

  world.spawnMany(plain, 6000);
  world.spawnMany(living, 3000);
  world.spawnMany(decor, ENTITIES - 9000);

  const moving = world.query({ all: [Position, Velocity] });
  const healthy = world.query({ all: [Health] });

  // Seed velocities, and disable a third of the Health components so the forEach loop really
  // exercises the per-row enable check instead of visiting every row.
  let i = 0;
  moving.forEach((_e, chunk, row) => {
    const vel = chunk.col(Velocity);
    vel.dx[row] = 1e-3;
    vel.dy[row] = -1e-3;
  });
  healthy.forEach((e, chunk, row) => {
    chunk.col(Health).hp[row] = 50;
    if (i++ % 3 === 0) world.enable(e, Health, false);
  });
  world.flush();

  // System 1: the plain chunk loop, reading `q.chunks` and `chunk.col()` directly.
  world.system('move', { query: moving }, (q) => {
    const chunks = q.chunks;
    for (let c = 0; c < chunks.length; c++) {
      const chunk = chunks[c];
      const n = chunk.count;
      if (n === 0) continue;
      const pos = chunk.col(Position);
      const x = pos.x;
      const y = pos.y;
      for (let r = 0; r < n; r++) {
        if (x[r] > 1) x[r] = 0;
        if (y[r] < -1) y[r] = 0;
      }
    }
  });

  // System 2: forEachChunk with a stable kernel (compiled trampolines).
  world.system('integrate', { query: moving }, (q) => {
    q.forEachChunk(PV, integrate);
  });

  // System 3: forEach(components, fn) over a query with an enableable component.
  world.system('regen', { query: healthy }, (q) => {
    q.forEach(H, regenerate);
  });

  return { world, tick: () => world.update(1 / 60) };
}

// ---------------------------------------------------------------------------- measurement

/** `global.gc` when jest runs under --expose-gc, else undefined. */
function gcOrNull(): (() => void) | undefined {
  const g = (globalThis as unknown as { gc?: () => void }).gc;
  return typeof g === 'function' ? g : undefined;
}

// eslint-disable-next-line no-console
const note = (s: string): void => console.log(`[alloc] ${s}`);

interface AllocMeasurement {
  /** Bytes allocated while the body ran (garbage included). */
  bytes: number;
  bytesPerIteration: number;
  /** Collections observed inside the window; must be 0 for `bytes` to be a true total. */
  collections: number;
}

/**
 * Bytes allocated by `iters` runs of `body`. Starts from a collected heap so the window has the
 * whole young generation to fill without triggering a scavenge, then reads `heapUsed` before and
 * after. GC entries are delivered on a later turn of the event loop, so the count is read after
 * yielding -- reading it synchronously would always report 0.
 */
async function measureAllocation(
  gc: () => void,
  iters: number,
  body: (iters: number) => void,
): Promise<AllocMeasurement> {
  let collections = 0;
  const observer = new PerformanceObserver((list) => {
    collections += list.getEntries().length;
  });
  gc();
  gc();
  observer.observe({ entryTypes: ['gc'] });
  const before = process.memoryUsage().heapUsed;
  body(iters);
  const after = process.memoryUsage().heapUsed;
  await new Promise((resolve) => setTimeout(resolve, 0));
  observer.disconnect();
  const bytes = after - before;
  return { bytes, bytesPerIteration: bytes / iters, collections };
}

/**
 * `measureAllocation`, retried while a collection lands inside the window. One would invalidate the
 * delta (a scavenge resets `heapUsed`, turning the measurement into an underestimate), and nothing
 * measured here comes close to filling the young generation, so a hit means the environment was
 * busy rather than that the code allocates. Returns the last attempt either way, so a window that
 * cannot be measured cleanly reports `collections !== 0` and fails loudly instead of passing
 * vacuously.
 */
async function measureAllocationStable(
  gc: () => void,
  iters: number,
  body: (iters: number) => void,
): Promise<AllocMeasurement> {
  let last = await measureAllocation(gc, iters, body);
  for (let attempt = 1; attempt < 3 && last.collections !== 0; attempt++) {
    note(`re-measuring: a collection landed inside the window (attempt ${attempt + 1} of 3)`);
    last = await measureAllocation(gc, iters, body);
  }
  return last;
}

/** Retained heap after two collections (the second sweeps what the first finalized). */
function retainedHeap(gc: () => void): number {
  gc();
  gc();
  return process.memoryUsage().heapUsed;
}

function runTicks(tick: () => void, n: number): void {
  for (let i = 0; i < n; i++) tick();
}

/**
 * Ticks run before anything is measured. V8 needs a few thousand to tier the system bodies, the row
 * loops and the compiled trampolines up to optimized code, and every one of those compilations is
 * an allocation: measured on this fixture, the first 8000-tick window reports ~395 bytes/tick after
 * 1000 warmup ticks, 3.8 after 2000 and 1.2 from 4000 on. A discarded measurement window follows,
 * so a slower machine that has not finished tiering by then still reaches steady state before the
 * number that gets asserted.
 */
const WARMUP = 4000;
const TICKS = 8000;
/**
 * A tick may allocate no more than this: ~8x the steady-state measurement (1.2-1.9 B/tick, itself
 * barely above the empty loop's noise floor) and ~8x below the cheapest real mistake (an inline
 * forEachChunk kernel, 121-175 B/tick). One 3-field object per tick costs 64 bytes, so a single
 * stray allocation per tick does not fit under this bound.
 */
const MAX_BYTES_PER_TICK = 16;

let sink: unknown = null;

// ---------------------------------------------------------------------------- tests

describe('a steady-state tick allocates nothing', () => {
  test('a tick allocates no bytes', async () => {
    const gc = gcOrNull();
    if (!gc) {
      note('allocation rate skipped: global.gc is unavailable (run `npm run test:alloc`)');
      return;
    }

    const { world, tick } = buildWorld();
    expect(world.query({ all: [Position, Velocity] }).count()).toBe(ENTITIES);

    // Warm up first: lazily created views, compiled row loops, chunk trampolines and V8's own
    // optimized code are one-off allocations that must happen BEFORE anything is measured.
    runTicks(tick, WARMUP);
    const settling = await measureAllocationStable(gc, TICKS, (n) => runTicks(tick, n));

    // Control: one 3-field object per iteration. V8 lays that out at 64 bytes, so a meter that
    // works must report roughly that. If this came back near zero, every other number below would
    // be meaningless -- which is precisely how a weaker meter (V8's sampling heap profiler, which
    // does not observe inline new-space allocation here) silently passes everything.
    const control = await measureAllocationStable(gc, TICKS, (n) => {
      for (let i = 0; i < n; i++) sink = { a: i, b: i, c: i };
    });

    // Noise floor: a loop of the same length that provably allocates nothing.
    let acc = 0;
    const baseline = await measureAllocationStable(gc, TICKS, (n) => {
      for (let i = 0; i < n; i++) acc += i & 7;
    });
    expect(acc).toBeGreaterThan(0);

    const measured = await measureAllocationStable(gc, TICKS, (n) => runTicks(tick, n));

    const fmt = (m: AllocMeasurement): string =>
      `${m.bytes}B total, ${m.bytesPerIteration.toFixed(2)} B/iter, ${m.collections} collections`;
    note(
      `${TICKS} iterations over ${ENTITIES} entities / 3 systems:\n` +
        `         tick     ${fmt(measured)}\n` +
        `         settling ${fmt(settling)}  (discarded: the window right after warmup)\n` +
        `         baseline ${fmt(baseline)}\n` +
        `         control  ${fmt(control)}  (one 3-field object per iteration)`,
    );

    // The meter is honest: it sees the control's allocations...
    expect(control.bytesPerIteration).toBeGreaterThan(32);
    // ...and no collection reset heapUsed inside any of the three windows, so the deltas are true
    // totals rather than underestimates.
    expect(control.collections).toBe(0);
    expect(baseline.collections).toBe(0);
    expect(measured.collections).toBe(0);
    expect(baseline.bytesPerIteration).toBeLessThan(MAX_BYTES_PER_TICK);

    // The actual guard.
    expect(measured.bytesPerIteration).toBeLessThan(MAX_BYTES_PER_TICK);
    // The systems really ran (guards against a fixture that silently iterates nothing).
    expect(healthSeen).toBeGreaterThan(0);
    expect(sink).not.toBeNull();
  }, 120_000);

  test('retained heap is flat across blocks of ticks', () => {
    const gc = gcOrNull();
    if (!gc) {
      note('retained heap skipped: global.gc is unavailable (run `npm run test:alloc`)');
      return;
    }

    const BLOCK = 2000;
    const BLOCKS = 4;
    // 256 KiB per 2000-tick block == 128 bytes/tick. V8's heap wobbles by tens of KiB between
    // collections even when nothing is retained (measured drift on this fixture: < 1 B/tick), so
    // this bound cannot be tight; the allocation-rate test above is the sensitive one. What this
    // catches is growth that REPEATS block after block.
    const ALLOWANCE = 256 * 1024;

    const { tick } = buildWorld();
    runTicks(tick, WARMUP);

    const samples: number[] = [retainedHeap(gc)];
    for (let b = 0; b < BLOCKS; b++) {
      runTicks(tick, BLOCK);
      samples.push(retainedHeap(gc));
    }

    const deltas = samples.slice(1).map((v, i) => v - samples[i]);
    const total = samples[samples.length - 1] - samples[0];
    note(
      `${BLOCKS} x ${BLOCK} ticks over ${ENTITIES} entities: ` +
        `heapUsed ${samples.map((s) => (s / 1048576).toFixed(2) + 'MB').join(' -> ')}; ` +
        `per-block delta ${deltas.map((d) => (d / 1024).toFixed(1) + 'KiB').join(', ')}; ` +
        `total ${(total / 1024).toFixed(1)}KiB = ${(total / (BLOCK * BLOCKS)).toFixed(2)} B/tick`,
    );

    // No single block may grow by more than the allowance...
    for (let i = 0; i < deltas.length; i++) expect(deltas[i]).toBeLessThan(ALLOWANCE);
    // ...and the run as a whole must not drift upward block after block.
    expect(total).toBeLessThan(ALLOWANCE);
  }, 120_000);
});
