/// <reference types="@webgpu/types" />
/**
 * GPU-side memory reclamation on a real Dawn device.
 *
 * Covers the two halves of the contract added alongside `World.compact()` /
 * `World.clear()`:
 *
 *  1. CORRECTNESS WITHOUT RECLAMATION. Compacting or clearing a world between
 *     dispatches replaces an archetype's ArrayBuffer and moves every column
 *     inside it. The device copy is reused at its old (now oversized) size, so
 *     the dispatch path must re-upload, re-read every field offset, and never
 *     dispatch a row count or read a buffer range that belonged to the old
 *     layout. The subtle case is a table emptied and REFILLED to the same row
 *     count in the same buffer: `chunk.buffer` and `chunk.count` both come back
 *     to what the device holds, and only row 0's entity handle says otherwise.
 *
 *  2. RECLAMATION ON DEMAND. `handle.releaseUnused()` actually destroys device
 *     buffers, reports the bytes, and leaves the kernel able to dispatch again
 *     by re-uploading from the CPU tables.
 *
 * Dawn comes from the optional `webgpu` dev dependency, exactly as in
 * gpu-dawn.test.ts; every test returns early with a note when there is no
 * adapter or `COZYECS_SKIP_DAWN` is set, because "no WebGPU here" is not a
 * CozyECS bug.
 */
import { describe, test, expect, jest, beforeAll, afterAll } from '@jest/globals';
import { World, component, f32 } from '../src/index';
import type { Archetype } from '../src/archetype';
import { kernelSystem, setGPUProvider, getGPUContext } from '../src/gpu/index';
import type { KernelSystemHandle } from '../src/gpu/index';
import { residentTableBytes, residentTableCount } from '../src/gpu/runtime';

// ts-jest rewrites a literal `import()` into require(), which cannot load the
// ESM-only `webgpu` package; building it through `new Function` reaches Node's
// real ESM loader (jest runs with --experimental-vm-modules).
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
    // HAZARD 1 (src/gpu/runtime.ts): Dawn segfaults if the instance is collected.
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
  console.log(`[gpu-memory] skipped: ${skipReason}`);
  return true;
}

const P = component({ x: f32, y: f32 }, { name: 'MemP' });
const V = component({ x: f32, y: f32 }, { name: 'MemV' });

/** `p.x += v.x` per dispatch: the value proves WHICH upload the GPU ran over. */
const STEP = (p: any, v: any) => {
  p.x += v.x;
  p.y += v.y;
};

interface Fixture {
  world: World;
  handle: KernelSystemHandle;
  ids: number[];
  arch: Archetype;
}

/** Spawns `n` entities with `p.x = base + i`, `v.x = 1`, and a GPU-pinned kernel. */
async function fixture(n: number, base = 0, options?: { shared?: boolean }): Promise<Fixture> {
  const world = new World(options);
  const ids = fill(world, n, base);
  const handle = await kernelSystem(world, 'MemStep', {
    components: [P, V],
    target: 'gpu',
    readback: 'sync-frame',
    kernel: STEP,
  });
  const chunks = world.query({ all: [P, V] }).chunks;
  expect(chunks.length).toBe(1);
  return { world, handle, ids, arch: chunks[0] };
}

function fill(world: World, n: number, base: number): number[] {
  const ids: number[] = [];
  for (let i = 0; i < n; i++) {
    const e = world.spawn([P, V]);
    world.set(e, P, { x: base + i, y: 0 });
    world.set(e, V, { x: 1, y: 2 });
    ids.push(e);
  }
  return ids;
}

/** `p.x` for every id, as the CPU tables see it. */
function xs(world: World, ids: readonly number[]): number[] {
  return ids.map((e) => world.getField(e, P, 'x'));
}

async function step(f: Fixture): Promise<void> {
  f.world.update(1 / 60);
  await f.handle.sync();
}

/**
 * The `x` values N dispatches produce from `p.x = base + i`, `v.x = 1`.
 * Comparing against this catches a dispatch that ran over a stale device image
 * (too many increments) or over no upload at all (too few).
 */
function expected(n: number, base: number, dispatches: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < n; i++) out.push(base + i + dispatches);
  return out;
}

const warn = jest.spyOn(console, 'warn').mockImplementation(() => {});
afterAll(() => warn.mockRestore());

describe('dawn is available for the memory tests', () => {
  test('a device can be acquired', async () => {
    if (skipped()) return;
    const ctx = await getGPUContext();
    expect(ctx).not.toBeNull();
  });
});

// ---------------------------------------------------------------------------
// 1. Correctness across compaction, with no reclamation call at all.
// ---------------------------------------------------------------------------

describe('world.compact() between dispatches (Dawn)', () => {
  test('GPU results stay correct after the table is compacted under the kernel', async () => {
    if (skipped()) return;
    const N = 20_000;
    const KEEP = 300;
    const f = await fixture(N);
    expect(f.handle.backend).toBe('gpu');

    await step(f);
    expect(xs(f.world, f.ids)).toEqual(expected(N, 0, 1));

    // Destroy all but the first KEEP entities, then compact: the archetype's
    // capacity collapses, its ArrayBuffer is replaced and every column moves.
    // `destroy` swap-removes, so walk from the back to keep rows 0..KEEP-1.
    for (let i = N - 1; i >= KEEP; i--) f.world.destroy(f.ids[i]);
    const kept = f.ids.slice(0, KEEP);
    const before = f.arch.buffer;
    const capBefore = f.arch.capacity;
    const deviceBefore = residentTableBytes(f.arch);
    const stats = f.world.compact();
    expect(stats.archetypes).toBeGreaterThan(0);
    expect(f.arch.buffer).not.toBe(before);
    expect(f.arch.capacity).toBeLessThan(capBefore);
    expect(f.arch.count).toBe(KEEP);
    // Same archetype OBJECT, still the query's only chunk, same id.
    expect(f.world.query({ all: [P, V] }).chunks[0]).toBe(f.arch);

    // The device copy is untouched by compaction: oversized, and stale.
    expect(residentTableBytes(f.arch)).toBe(deviceBefore);

    // Two more dispatches over the compacted layout.
    await step(f);
    await step(f);
    expect(xs(f.world, kept)).toEqual(expected(KEEP, 0, 3));
    expect(f.handle.stats.staleReadbacks).toBe(0);
    f.handle.destroy();
  }, 60_000);

  test('a compacted table is re-uploaded, not re-read from the device image', async () => {
    if (skipped()) return;
    const N = 8_000;
    const f = await fixture(N);
    await step(f);

    // Destroy everything but row 0 and compact, then overwrite row 0 on the CPU
    // side. A dispatch that skipped the re-upload would use the device's copy of
    // the OLD layout and produce 2 instead of 1001.
    for (let i = N - 1; i >= 1; i--) f.world.destroy(f.ids[i]);
    f.world.compact();
    f.world.set(f.ids[0], P, { x: 1000, y: 0 });
    f.handle.markCpuDirty(); // the documented way to publish a CPU write
    await step(f);
    expect(f.world.getField(f.ids[0], P, 'x')).toBe(1001);
    f.handle.destroy();
  }, 60_000);
});

// ---------------------------------------------------------------------------
// 2. The residency hole clear() opens: same buffer, same count, new rows.
// ---------------------------------------------------------------------------

describe('world.clear() under a kernel (Dawn)', () => {
  test("clear({ compact: false }) then refilling to the same count does not reuse the device image", async () => {
    if (skipped()) return;
    const N = 5_000;
    const f = await fixture(N);
    await step(f);
    expect(xs(f.world, f.ids)).toEqual(expected(N, 0, 1));

    const buffer = f.arch.buffer;
    const capacity = f.arch.capacity;
    f.world.clear({ compact: false });
    expect(f.world.isAlive(f.ids[0])).toBe(false);
    expect(f.arch.count).toBe(0);
    // The trap: nothing about the table's identity or size changed.
    expect(f.arch.buffer).toBe(buffer);
    expect(f.arch.capacity).toBe(capacity);

    // Refill to EXACTLY the same row count, with new values, and dispatch
    // without any intervening tick. `chunk.buffer` and `chunk.count` are both
    // what the device holds; only row 0's handle differs.
    const ids = fill(f.world, N, 1000);
    expect(f.arch.buffer).toBe(buffer);
    expect(f.arch.count).toBe(N);
    expect(ids[0]).not.toBe(f.ids[0]);
    await step(f);
    expect(xs(f.world, ids)).toEqual(expected(N, 1000, 1));
    f.handle.destroy();
  }, 60_000);

  test('clear() deflates the table to zero capacity and spawning still works', async () => {
    if (skipped()) return;
    const N = 4_000;
    const f = await fixture(N);
    await step(f);

    f.world.clear();
    expect(f.arch.count).toBe(0);
    expect(f.arch.capacity).toBe(0);
    expect(f.arch.buffer.byteLength).toBe(0);

    // A dispatch over an empty, zero-capacity table is a no-op, not a crash.
    f.world.update(1 / 60);
    await f.handle.sync();
    expect(f.handle.stats.lastEntities).toBe(0);
    expect(f.handle.backend).toBe('gpu');

    const ids = fill(f.world, N, 500);
    expect(f.arch.capacity).toBeGreaterThanOrEqual(N);
    await step(f);
    expect(xs(f.world, ids)).toEqual(expected(N, 500, 1));
    f.handle.destroy();
  }, 60_000);
});

// ---------------------------------------------------------------------------
// 3. releaseUnused(): the bytes really go, and the next dispatch re-uploads.
// ---------------------------------------------------------------------------

describe('releaseUnused() (Dawn)', () => {
  test('frees an emptied archetype and the next dispatch re-uploads correctly', async () => {
    if (skipped()) return;
    const N = 30_000;
    const f = await fixture(N);
    await step(f);
    const resident = residentTableBytes(f.arch);
    expect(resident).toBeGreaterThan(0);
    expect(f.handle.bufferFor(f.arch)).not.toBeNull();
    expect(residentTableCount(f.arch)).toBe(1);

    f.world.clear();
    const freed = f.handle.releaseUnused();
    expect(freed).toBeGreaterThanOrEqual(resident);
    expect(residentTableBytes(f.arch)).toBe(0);
    expect(f.handle.bufferFor(f.arch)).toBeNull();

    // Idempotent: nothing left to free.
    expect(f.handle.releaseUnused()).toBe(0);

    const ids = fill(f.world, N, 7);
    await step(f);
    expect(residentTableBytes(f.arch)).toBeGreaterThan(0);
    expect(xs(f.world, ids)).toEqual(expected(N, 7, 1));
    f.handle.destroy();
  }, 60_000);

  test('gives back the slack world.compact() leaves on the device', async () => {
    if (skipped()) return;
    const N = 60_000;
    const KEEP = 200;
    const f = await fixture(N);
    await step(f);
    const big = residentTableBytes(f.arch);
    expect(big).toBeGreaterThan(0);

    for (let i = N - 1; i >= KEEP; i--) f.world.destroy(f.ids[i]);
    const kept = f.ids.slice(0, KEEP);
    f.world.compact();
    // Device memory is unchanged by compaction on its own...
    expect(residentTableBytes(f.arch)).toBe(big);

    // ...and releasing it reports real bytes.
    const freed = f.handle.releaseUnused();
    expect(freed).toBeGreaterThanOrEqual(big);
    expect(residentTableBytes(f.arch)).toBe(0);

    // The next dispatch reallocates at the COMPACTED size, which is far smaller,
    // and the values are still right.
    await step(f);
    const small = residentTableBytes(f.arch);
    expect(small).toBeGreaterThan(0);
    expect(small).toBeLessThan(big / 4);
    expect(xs(f.world, kept)).toEqual(expected(KEEP, 0, 2));
    f.handle.destroy();
  }, 60_000);

  test('keeps a live, right-sized table and returns 0', async () => {
    if (skipped()) return;
    const f = await fixture(10_000);
    await step(f);
    const resident = residentTableBytes(f.arch);
    const buffer = f.handle.bufferFor(f.arch);
    // Nothing is empty, stale or oversized: a well-behaved reclaim is a no-op.
    expect(f.handle.releaseUnused()).toBe(0);
    expect(residentTableBytes(f.arch)).toBe(resident);
    expect(f.handle.bufferFor(f.arch)).toBe(buffer);
    await step(f);
    expect(xs(f.world, f.ids)).toEqual(expected(10_000, 0, 2));
    f.handle.destroy();
  }, 60_000);

  test('compaction and release each cost exactly one re-upload, and residency returns', async () => {
    if (skipped()) return;
    const N = 20_000;
    const KEEP = 1_000;
    const f = await fixture(N);
    await step(f);
    const afterFirst = f.handle.stats.bytesUploaded;
    expect(afterFirst).toBeGreaterThan(0);

    // Residency: a steady-state dispatch uploads nothing at all.
    await step(f);
    expect(f.handle.stats.bytesUploaded).toBe(afterFirst);

    // Compaction replaces the buffer, so the next dispatch re-uploads once...
    for (let i = N - 1; i >= KEEP; i--) f.world.destroy(f.ids[i]);
    const kept = f.ids.slice(0, KEEP);
    expect(f.world.compact().archetypes).toBeGreaterThan(0);
    await step(f);
    const afterCompact = f.handle.stats.bytesUploaded;
    expect(afterCompact).toBeGreaterThan(afterFirst);
    // ...and residency is restored, not permanently broken.
    await step(f);
    expect(f.handle.stats.bytesUploaded).toBe(afterCompact);

    // A release costs one more upload, and only one.
    expect(f.handle.releaseUnused()).toBeGreaterThan(0);
    await step(f);
    const afterRelease = f.handle.stats.bytesUploaded;
    expect(afterRelease).toBeGreaterThan(afterCompact);
    await step(f);
    expect(f.handle.stats.bytesUploaded).toBe(afterRelease);

    expect(xs(f.world, kept)).toEqual(expected(KEEP, 0, 6));
    f.handle.destroy();
  }, 60_000);

  test('returns 0 after destroy() and on a CPU-pinned kernel', async () => {
    if (skipped()) return;
    const f = await fixture(5_000);
    await step(f);
    f.handle.destroy();
    expect(f.handle.releaseUnused()).toBe(0);

    const world = new World();
    fill(world, 100, 0);
    const cpu = await kernelSystem(world, 'MemStepCPU', {
      components: [P, V],
      target: 'cpu',
      readback: 'sync-frame',
      kernel: STEP,
    });
    world.update(1 / 60);
    expect(cpu.backend).toBe('cpu');
    expect(cpu.releaseUnused()).toBe(0);
    cpu.destroy();
  }, 60_000);
});

// ---------------------------------------------------------------------------
// 4. SharedArrayBuffer tables (HAZARD 2) survive compaction and release.
// ---------------------------------------------------------------------------

describe('shared (SharedArrayBuffer) tables (Dawn)', () => {
  test('compaction keeps the table shared, and upload/release/re-upload still work', async () => {
    if (skipped()) return;
    if (typeof SharedArrayBuffer === 'undefined') {
      // eslint-disable-next-line no-console
      console.log('[gpu-memory] no SharedArrayBuffer here; skipped');
      return;
    }
    const N = 20_000;
    const KEEP = 150;
    const f = await fixture(N, 0, { shared: true });
    expect(f.arch.shared).toBe(true);
    expect(f.arch.buffer).toBeInstanceOf(SharedArrayBuffer);
    await step(f);
    expect(xs(f.world, f.ids)).toEqual(expected(N, 0, 1));

    for (let i = N - 1; i >= KEEP; i--) f.world.destroy(f.ids[i]);
    const kept = f.ids.slice(0, KEEP);
    f.world.compact();
    expect(f.arch.shared).toBe(true);
    expect(f.arch.buffer).toBeInstanceOf(SharedArrayBuffer);

    // Dispatch across the compaction, then release and dispatch again: the
    // upload path has to rebuild its plain-ArrayBuffer staging copy both times
    // (a writeBuffer straight out of a SharedArrayBuffer is a segfault).
    await step(f);
    f.handle.releaseUnused();
    await step(f);
    expect(xs(f.world, kept)).toEqual(expected(KEEP, 0, 3));
    f.handle.destroy();
  }, 60_000);
});

// ---------------------------------------------------------------------------
// 5. Two kernels over one archetype share its table: releasing is shared too.
// ---------------------------------------------------------------------------

describe('two kernels over one archetype (Dawn)', () => {
  test('the first releaseUnused() reports the bytes, the second finds none, both recover', async () => {
    if (skipped()) return;
    const N = 15_000;
    const world = new World();
    const ids = fill(world, N, 0);
    const a = await kernelSystem(world, 'MemShareA', {
      components: [P, V],
      target: 'gpu',
      readback: 'sync-frame',
      order: 0,
      kernel: (p: any, v: any) => {
        p.x += v.x;
      },
    });
    const b = await kernelSystem(world, 'MemShareB', {
      components: [P, V],
      target: 'gpu',
      readback: 'sync-frame',
      order: 1,
      kernel: (p: any, v: any) => {
        p.y += v.y;
      },
    });
    const arch = world.query({ all: [P, V] }).chunks[0];
    world.update(1 / 60);
    await a.sync();
    await b.sync();
    expect(ids.map((e) => world.getField(e, P, 'x'))).toEqual(expected(N, 0, 1));
    expect(world.getField(ids[0], P, 'y')).toBe(2);

    // ONE shared device table for both kernels, so exactly one of the two calls
    // can free it; the other must not double-destroy or report the bytes twice.
    expect(residentTableBytes(arch)).toBeGreaterThan(0);
    world.clear();
    const freedA = a.releaseUnused();
    const freedB = b.releaseUnused();
    expect(freedA).toBeGreaterThan(0);
    expect(freedB).toBe(0);
    expect(residentTableBytes(arch)).toBe(0);
    expect(a.bufferFor(arch)).toBeNull();
    expect(b.bufferFor(arch)).toBeNull();

    // Both kernels recover: the table is recreated once and both see the same
    // image, so B's writes still follow A's within one update (docs/GPU.md 3.5).
    const refilled = fill(world, N, 0);
    world.update(1 / 60);
    await a.sync();
    await b.sync();
    expect(residentTableBytes(arch)).toBeGreaterThan(0);
    expect(refilled.map((e) => world.getField(e, P, 'x'))).toEqual(expected(N, 0, 1));
    expect(world.getField(refilled[0], P, 'y')).toBe(2);
    a.destroy();
    b.destroy();
  }, 60_000);
});
