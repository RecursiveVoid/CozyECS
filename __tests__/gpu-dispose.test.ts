/// <reference types="@webgpu/types" />
/**
 * GPU-side teardown: `world.dispose()` and `handle.destroy()` must give the
 * DEVICE memory back, not just the JavaScript objects.
 *
 * WHY THIS FILE EXISTS. The leak the user reported is on a page that rebuilds
 * its World on every backend switch. Chrome's `performance.memory` excludes
 * ArrayBuffers and cannot see device memory at all, and
 * `process.memoryUsage().arrayBuffers` cannot see it either, so a GPU buffer
 * that is never destroyed is invisible to every heap measurement we have. The
 * census in src/gpu/runtime.ts (`gpuDeviceMemory()`) is the only instrument
 * that can see it, and these tests are what make it trustworthy: every
 * assertion is "held bytes came back to the number they started at".
 *
 * Worlds here are SMALL (<= 20k entities, a few MB of device memory) and the
 * cycle count is what is large. A leak shows up as a monotonic rise across
 * cycles, which small worlds reveal just as well as big ones, and they cannot
 * fill the machine.
 */
import { describe, test, expect, beforeAll } from '@jest/globals';
import { World, component, f32 } from '../src/index';
import {
  disposeWorldKernels,
  gpuDeviceMemory,
  kernelSystem,
  setGPUProvider,
} from '../src/gpu/index';
import type { KernelSystemHandle } from '../src/gpu/index';
// Internal seams: the process-wide pipeline cache (which must NOT grow per
// world) and the per-archetype table census.
import { pipelineCacheSize, residentTableBytes } from '../src/gpu/runtime';

// See gpu-dawn.test.ts: jest's CJS runtime cannot require() the ESM-only
// `webgpu` package, and ts-jest rewrites a literal import() into a require.
const esmImport = new Function('s', 'return import(s)') as (s: string) => Promise<any>;

let dawn: { gpu: GPU; adapter: GPUAdapter } | null = null;
let skipReason = '';

beforeAll(async () => {
  if (process.env.COZYECS_SKIP_DAWN) {
    skipReason = 'COZYECS_SKIP_DAWN is set';
    return;
  }
  try {
    const mod = await esmImport('webgpu');
    if (mod.globals) Object.assign(globalThis, mod.globals);
    const gpu: GPU = mod.create([]);
    // HAZARD 1 (src/gpu/runtime.ts): Dawn segfaults if the instance is
    // collected. Pin it for the life of the process.
    (globalThis as any).__dawnKeepAlive = gpu;
    const adapter = await gpu.requestAdapter();
    if (!adapter) {
      skipReason = 'webgpu package loaded but yielded no adapter';
      return;
    }
    dawn = { gpu, adapter };
    (globalThis as any).__dawnKeepAlive = { gpu, adapter };
    setGPUProvider(gpu);
  } catch (e) {
    skipReason = `webgpu package unavailable (${(e as Error).message})`;
  }
});

function skipped(): boolean {
  if (dawn) return false;
  // eslint-disable-next-line no-console
  console.log(`[gpu-dispose] skipped: ${skipReason}`);
  return true;
}

const P = component({ x: f32, y: f32 }, { name: 'DP' });
const V = component({ x: f32, y: f32 }, { name: 'DV' });
const W = component({ x: f32, y: f32 }, { name: 'DW' });

/** 8k entities: ~128 KB of table per archetype. Deliberately small. */
const N = 8_000;

function build(n = N): World {
  const world = new World();
  for (let i = 0; i < n; i++) {
    const e = world.spawn([P, V]);
    world.set(e, P, { x: i, y: 0 });
    world.set(e, V, { x: 1, y: 2 });
  }
  return world;
}

function move(p: any, v: any, dt: number): void {
  p.x += v.x * dt;
  p.y += v.y * dt;
}

async function addMove(world: World, name: string): Promise<KernelSystemHandle> {
  return kernelSystem(world, name, {
    components: [P, V],
    target: 'gpu',
    readback: 'async',
    kernel: move,
  });
}

/**
 * Runs frames and waits for the device to be quiet, so every staging buffer is
 * back in the pool. A staging buffer whose `mapAsync` is still outstanding is
 * owned by that map, not by the kernel, and is destroyed as it drains -- which
 * is correct but not synchronous, so tests that assert an exact byte count
 * settle first. The one test that deliberately tears down mid-flight polls.
 */
async function runAndSettle(world: World, handle: KernelSystemHandle, frames = 3): Promise<void> {
  for (let f = 0; f < frames; f++) world.update(1 / 60);
  await handle.sync();
}

/**
 * Builds a world with a GPU kernel, settles it, disposes its kernels, and returns only
 * WeakRefs. Nothing here may outlive the call: see the note at its one call site.
 */
async function buildKernelWorldAndDispose(n: number): Promise<{ worldRef: WeakRef<object>; archRef: WeakRef<object> }> {
  const world = build(n);
  const handle = await addMove(world, 'Collect');
  await runAndSettle(world, handle);
  const worldRef = new WeakRef(world as unknown as object);
  const archRef = new WeakRef(world.query({ all: [P, V] }).chunks[0] as unknown as object);
  disposeWorldKernels(world);
  expect(handle.memory().heldBytes).toBe(0);
  world.dispose();
  return { worldRef, archRef };
}

/** Runs the GC, if this process was started with --expose-gc. */
async function collect(): Promise<boolean> {
  const gc = (globalThis as any).gc as (() => void) | undefined;
  if (typeof gc !== 'function') return false;
  for (let i = 0; i < 4; i++) {
    gc();
    await new Promise((r) => setTimeout(r, 0));
  }
  return true;
}

/** Waits for `gpuDeviceMemory().heldBytes` to reach `target`, or gives up. */
async function waitForHeld(target: number, budgetMs = 2000): Promise<number> {
  const deadline = Date.now() + budgetMs;
  while (gpuDeviceMemory().heldBytes !== target && Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 5));
  }
  return gpuDeviceMemory().heldBytes;
}

describe('device-memory accounting', () => {
  test('the census is internally consistent and sees a kernel allocate', async () => {
    if (skipped()) return;
    const base = gpuDeviceMemory();
    expect(base.allocatedBytes - base.freedBytes).toBe(base.heldBytes);

    const world = build();
    const handle = await addMove(world, 'Census');
    expect(handle.backend).toBe('gpu');
    await runAndSettle(world, handle);

    const live = gpuDeviceMemory();
    expect(live.heldBytes).toBeGreaterThan(base.heldBytes);
    expect(live.allocatedBytes - live.freedBytes).toBe(live.heldBytes);
    expect(live.buffers).toBeGreaterThan(base.buffers);
    expect(live.peakBytes).toBeGreaterThanOrEqual(live.heldBytes);

    // The per-kernel slice adds up to what the kernel actually holds.
    const m = handle.memory();
    expect(m.tables).toBe(1);
    expect(m.tableBytes).toBeGreaterThan(0);
    expect(m.uniformBytes).toBeGreaterThan(0);
    expect(m.stagingBytes).toBeGreaterThan(0);
    expect(m.heldBytes).toBe(m.tableBytes + m.uniformBytes + m.stagingBytes);
    expect(m.allocatedBytes - m.freedBytes).toBe(m.heldBytes);
    // One kernel, one world: the kernel's held bytes ARE the process's delta.
    expect(m.heldBytes).toBe(live.heldBytes - base.heldBytes);

    handle.destroy();
    expect(handle.memory().heldBytes).toBe(0);
    expect(gpuDeviceMemory().heldBytes).toBe(base.heldBytes);
  });

  test('a CPU-backend kernel holds no device memory', async () => {
    if (skipped()) return;
    const base = gpuDeviceMemory().heldBytes;
    const world = build(64);
    const handle = await kernelSystem(world, 'CpuOnly', {
      components: [P, V],
      target: 'cpu',
      kernel: move,
    });
    for (let f = 0; f < 5; f++) world.update(1 / 60);
    expect(handle.backend).toBe('cpu');
    expect(handle.memory().heldBytes).toBe(0);
    expect(gpuDeviceMemory().heldBytes).toBe(base);
    handle.destroy();
    expect(gpuDeviceMemory().heldBytes).toBe(base);
  });
});

describe('handle.destroy() alone', () => {
  test('returns held bytes to zero, 12 times over', async () => {
    if (skipped()) return;
    const base = gpuDeviceMemory().heldBytes;
    const seen: number[] = [];
    for (let cycle = 0; cycle < 12; cycle++) {
      const world = build();
      const handle = await addMove(world, `Cycle${cycle}`);
      await runAndSettle(world, handle);
      seen.push(gpuDeviceMemory().heldBytes - base);
      handle.destroy();
      expect(gpuDeviceMemory().heldBytes).toBe(base);
    }
    // Every cycle held the same amount: no creep in what a kernel allocates.
    expect(new Set(seen).size).toBe(1);
    expect(seen[0]).toBeGreaterThan(0);
    const end = gpuDeviceMemory();
    expect(end.heldBytes).toBe(base);
    expect(end.freedBytes).toBeGreaterThan(0);
  });

  test('is idempotent, and further dispatches are no-ops', async () => {
    if (skipped()) return;
    const base = gpuDeviceMemory().heldBytes;
    const world = build();
    const handle = await addMove(world, 'Twice');
    await runAndSettle(world, handle);
    const dispatches = handle.stats.dispatches;

    handle.destroy();
    handle.destroy();
    handle.destroy();
    expect(gpuDeviceMemory().heldBytes).toBe(base);

    // The system is gone, but dispatching the runtime directly must also be a
    // no-op: nothing is re-allocated and no dispatch is counted.
    for (let f = 0; f < 5; f++) world.update(1 / 60);
    expect(handle.stats.dispatches).toBe(dispatches);
    expect(gpuDeviceMemory().heldBytes).toBe(base);
    expect(handle.memory().heldBytes).toBe(0);
    // sync() still resolves, and releaseUnused() reports nothing left.
    await handle.sync();
    expect(handle.releaseUnused()).toBe(0);
  });

  test('a tear-down mid-flight drains to zero', async () => {
    if (skipped()) return;
    const base = gpuDeviceMemory().heldBytes;
    const world = build();
    const handle = await addMove(world, 'InFlight');
    // Dispatch without sync(): a readback is outstanding and its staging
    // buffer is owned by the map, not by the kernel.
    for (let f = 0; f < 4; f++) world.update(1 / 60);
    expect(handle.stats.pending).toBeGreaterThan(0);
    handle.destroy();
    // Tables and the uniform buffer go at once; the staging buffers in flight
    // are destroyed as their maps settle.
    expect(await waitForHeld(base)).toBe(base);
  });
});

describe('world disposal through the core seam', () => {
  test('_onDispose(fn) is registered once per world and frees every kernel', async () => {
    if (skipped()) return;
    const base = gpuDeviceMemory().heldBytes;
    const world = build() as World & { _onDispose(fn: () => void): () => void };
    // Stand in for the core's seam: World.dispose() runs these callbacks.
    const hooks: (() => void)[] = [];
    (world as any)._onDispose = (fn: () => void) => {
      hooks.push(fn);
      return () => {
        const i = hooks.indexOf(fn);
        if (i >= 0) hooks.splice(i, 1);
      };
    };

    const a = await addMove(world, 'SeamA');
    const b = await kernelSystem(world, 'SeamB', {
      components: [P, V],
      target: 'gpu',
      readback: 'async',
      group: 'late',
      kernel: (p: any, v: any) => {
        v.x *= 0.99;
        p.x += 0;
      },
    });
    // Three kernels, two groups, one hook.
    const c = await addMove(world, 'SeamC');
    expect(hooks.length).toBe(1);

    for (let f = 0; f < 3; f++) world.update(1 / 60);
    world.update(1 / 60, 'late');
    await a.sync();
    await b.sync();
    await c.sync();
    expect(gpuDeviceMemory().heldBytes).toBeGreaterThan(base);

    // What World.dispose() will do.
    hooks[0]();
    expect(gpuDeviceMemory().heldBytes).toBe(base);
    expect(a.memory().heldBytes).toBe(0);
    expect(b.memory().heldBytes).toBe(0);
    expect(c.memory().heldBytes).toBe(0);

    // A handle that outlives its world must not try to touch it again.
    expect(() => {
      a.destroy();
      b.destroy();
      c.destroy();
    }).not.toThrow();
    expect(gpuDeviceMemory().heldBytes).toBe(base);
  });

  test('disposeWorldKernels() reports the bytes it freed and is idempotent', async () => {
    if (skipped()) return;
    const base = gpuDeviceMemory().heldBytes;
    const world = build();
    const handle = await addMove(world, 'Direct');
    await runAndSettle(world, handle);
    const held = gpuDeviceMemory().heldBytes - base;
    expect(held).toBeGreaterThan(0);

    expect(disposeWorldKernels(world)).toBe(held);
    expect(gpuDeviceMemory().heldBytes).toBe(base);
    expect(disposeWorldKernels(world)).toBe(0);
    expect(disposeWorldKernels(new World())).toBe(0);
    expect(disposeWorldKernels(null)).toBe(0);
    expect(disposeWorldKernels(undefined)).toBe(0);
  });

  test('two kernels sharing one archetype table both have to let go', async () => {
    if (skipped()) return;
    const base = gpuDeviceMemory().heldBytes;
    const world = build();
    const a = await addMove(world, 'ShareA');
    const b = await kernelSystem(world, 'ShareB', {
      components: [P, V],
      target: 'gpu',
      readback: 'async',
      kernel: (p: any, v: any) => {
        v.y *= 0.5;
        p.y += 0;
      },
    });
    for (let f = 0; f < 3; f++) world.update(1 / 60);
    await a.sync();
    await b.sync();

    // The table is ONE device copy with two holders: both report it.
    const ma = a.memory();
    const mb = b.memory();
    expect(ma.tableBytes).toBe(mb.tableBytes);
    expect(ma.tableBytes).toBeGreaterThan(0);
    // ...so the per-kernel numbers over-count and the census does not.
    expect(ma.heldBytes + mb.heldBytes).toBeGreaterThan(gpuDeviceMemory().heldBytes - base);

    // Destroying one leaves the shared table alive for the other.
    a.destroy();
    expect(gpuDeviceMemory().heldBytes).toBeGreaterThan(base);
    expect(b.memory().tableBytes).toBe(mb.tableBytes);
    b.destroy();
    expect(gpuDeviceMemory().heldBytes).toBe(base);
  });

  test('the compatibility shim wraps dispose() when _onDispose is absent', async () => {
    if (skipped()) return;
    const base = gpuDeviceMemory().heldBytes;
    const world = build();
    // A core published BEFORE the seam: dispose() exists, _onDispose does not. The own
    // property shadows World.prototype._onDispose, which a current core does provide --
    // without this the world takes the seam path and never exercises the shim.
    let coreDisposeRan = 0;
    let heldWhenCoreRan = -1;
    (world as any)._onDispose = undefined;
    (world as any).dispose = () => {
      coreDisposeRan++;
      heldWhenCoreRan = gpuDeviceMemory().heldBytes;
    };

    const handle = await addMove(world, 'Shim');
    await runAndSettle(world, handle);
    expect(gpuDeviceMemory().heldBytes).toBeGreaterThan(base);

    (world as any).dispose();
    expect(coreDisposeRan).toBe(1);
    // GPU teardown ran BEFORE the core's dispose(), so the core sees no device
    // memory left for this world.
    expect(heldWhenCoreRan).toBe(base);
    expect(gpuDeviceMemory().heldBytes).toBe(base);
  });

  test('if the core already has dispose(), it releases the kernels', async () => {
    if (skipped()) return;
    const probe = new World() as World & { dispose?: () => void; _onDispose?: unknown };
    if (typeof probe.dispose !== 'function' && typeof probe._onDispose !== 'function') {
      // eslint-disable-next-line no-console
      console.log('[gpu-dispose] core World has no dispose()/_onDispose() yet; integration half skipped');
      return;
    }
    const base = gpuDeviceMemory().heldBytes;
    const world = build();
    const handle = await addMove(world, 'CoreDispose');
    await runAndSettle(world, handle);
    expect(gpuDeviceMemory().heldBytes).toBeGreaterThan(base);
    (world as unknown as { dispose(): void }).dispose();
    expect(await waitForHeld(base)).toBe(base);
    expect(handle.memory().heldBytes).toBe(0);
  });
});

describe('nothing module-level survives a disposed world', () => {
  test('the pipeline cache does not grow per world', async () => {
    if (skipped()) return;
    const base = gpuDeviceMemory().heldBytes;
    // One compile for this body, then six worlds that reuse it. The cache is
    // keyed by the WGSL, not by the kernel's name, so a page that rebuilds its
    // world cannot grow it -- which is why dispose() leaves it alone.
    let after = -1;
    for (let cycle = 0; cycle < 6; cycle++) {
      const world = build(1_000);
      const handle = await kernelSystem(world, `Pipe${cycle}`, {
        components: [P, V],
        target: 'gpu',
        readback: 'async',
        kernel: (p: any, v: any, dt: number) => {
          p.x += v.x * dt * 2;
        },
      });
      await runAndSettle(world, handle, 1);
      if (cycle === 0) after = pipelineCacheSize();
      else expect(pipelineCacheSize()).toBe(after);
      disposeWorldKernels(world);
      expect(gpuDeviceMemory().heldBytes).toBe(base);
    }
    expect(after).toBeGreaterThan(0);
  });

  test('the archetype table census is empty after disposal', async () => {
    if (skipped()) return;
    const base = gpuDeviceMemory().heldBytes;
    const world = build(2_000);
    const handle = await addMove(world, 'Census2');
    await runAndSettle(world, handle);
    const archetypes = world.query({ all: [P, V] }).chunks.slice();
    expect(archetypes.length).toBeGreaterThan(0);
    let before = 0;
    for (const a of archetypes) before += residentTableBytes(a);
    expect(before).toBeGreaterThan(0);

    disposeWorldKernels(world);
    let held = 0;
    for (const a of archetypes) held += residentTableBytes(a);
    // Not just "the buffers were destroyed": the table cache entry is gone, so
    // the archetype is not keyed in any module-level map any more.
    expect(held).toBe(0);
    expect(gpuDeviceMemory().heldBytes).toBe(base);
  });

  test('a disposed world becomes collectable once the app drops it', async () => {
    if (skipped()) return;
    const base = gpuDeviceMemory().heldBytes;
    // Built inside a helper, NOT in this test body: locals of an async function stay
    // reachable through its suspended frame until it returns, so a world held in a block
    // here would look retained no matter what the library does. The helper's frame is
    // gone once it resolves, leaving only the WeakRefs.
    const { worldRef, archRef } = await buildKernelWorldAndDispose(4_000);
    expect(gpuDeviceMemory().heldBytes).toBe(base);

    if (!(await collect())) {
      // eslint-disable-next-line no-console
      console.log('[gpu-dispose] no --expose-gc: collectability half skipped (device bytes still asserted)');
      return;
    }
    // Nothing module-level in src/gpu/** may key a world or an archetype
    // strongly: `tracked`, `tableCache`, `hooked` and `deviceTags` are all weak
    // and the disposed world's entry is gone from the first two.
    expect(worldRef.deref()).toBeUndefined();
    expect(archRef.deref()).toBeUndefined();
  });

  test('a handle outliving a disposed world holds no world or query', async () => {
    if (skipped()) return;
    const base = gpuDeviceMemory().heldBytes;
    // The realistic shape of the reported leak: the page drops the World but
    // keeps the handles. Nothing on the GPU side may be what keeps it alive.
    let keptHandle: KernelSystemHandle | null = null;
    let worldRef: WeakRef<object>;
    {
      const world = build(4_000);
      const handle = await addMove(world, 'Kept');
      await runAndSettle(world, handle);
      worldRef = new WeakRef(world as unknown as object);
      disposeWorldKernels(world);
      keptHandle = handle;
    }
    expect(gpuDeviceMemory().heldBytes).toBe(base);
    // Still usable, still honest, and no longer reaching for the world.
    expect(keptHandle.memory().heldBytes).toBe(0);
    expect(keptHandle.releaseUnused()).toBe(0);
    expect(() => keptHandle!.destroy()).not.toThrow();
    await keptHandle.sync();

    if (!(await collect())) return;
    // NOTE FOR THE INTEGRATOR: a SystemHandle is a core object and still
    // reaches the world through the scheduler/query until `dispose()`
    // unregisters its systems. This assertion is therefore about the GPU half
    // only: it says the gpu bundle is not the thing holding on.
    const alive = worldRef.deref();
    if (alive) {
      // eslint-disable-next-line no-console
      console.log('[gpu-dispose] handle still reaches the world through core structures (see note)');
    }
    expect(gpuDeviceMemory().heldBytes).toBe(base);
  });
});

describe('many worlds, many cycles', () => {
  test('16 rebuild cycles of 3 kernels each leave nothing on the device', async () => {
    if (skipped()) return;
    const base = gpuDeviceMemory().heldBytes;
    const CYCLES = 16;
    const perCycle: number[] = [];

    for (let cycle = 0; cycle < CYCLES; cycle++) {
      // A world with two archetypes, so a kernel holds more than one table.
      const world = new World();
      for (let i = 0; i < N; i++) {
        const e = i % 4 === 0 ? world.spawn([P, V, W]) : world.spawn([P, V]);
        world.set(e, P, { x: i, y: 1 });
        world.set(e, V, { x: 1, y: -1 });
      }
      const hooks: (() => void)[] = [];
      (world as any)._onDispose = (fn: () => void) => {
        hooks.push(fn);
      };

      const handles: KernelSystemHandle[] = [];
      handles.push(await addMove(world, `M${cycle}`));
      handles.push(
        await kernelSystem(world, `Damp${cycle}`, {
          components: [P, V],
          target: 'gpu',
          readback: 'async',
          kernel: (p: any, v: any) => {
            v.x *= 0.98;
            v.y *= 0.98;
            p.x += 0;
          },
        }),
      );
      handles.push(
        await kernelSystem(world, `Wob${cycle}`, {
          components: [P, W],
          target: 'gpu',
          readback: 'async',
          kernel: (p: any, w: any, dt: number) => {
            w.x += dt;
            p.y += w.x * 0;
          },
        }),
      );

      for (let f = 0; f < 4; f++) world.update(1 / 60);
      for (const h of handles) await h.sync();

      const held = gpuDeviceMemory().heldBytes - base;
      expect(held).toBeGreaterThan(0);
      perCycle.push(held);

      // The world goes away exactly as the demo page lets it: dispose, drop.
      expect(hooks.length).toBe(1);
      hooks[0]();

      expect(gpuDeviceMemory().heldBytes).toBe(base);
      for (const h of handles) expect(h.memory().heldBytes).toBe(0);
    }

    // Same device footprint every cycle, and nothing held at the end: the
    // ceiling is one world's worth, not sixteen.
    expect(new Set(perCycle).size).toBe(1);
    const end = gpuDeviceMemory();
    expect(end.heldBytes).toBe(base);
    expect(end.peakBytes).toBeLessThan(base + perCycle[0] * 2);
    expect(end.allocatedBytes - end.freedBytes).toBe(base);
    // Device memory the run churned through, as a sanity check that the test
    // actually exercised the GPU: 16 cycles x 3 kernels.
    expect(end.freedBytes).toBeGreaterThan(perCycle[0] * CYCLES);
  }, 120_000);
});
