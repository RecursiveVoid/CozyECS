/// <reference types="@webgpu/types" />
import { describe, test, expect, jest, beforeAll, afterAll } from '@jest/globals';
import * as core from '../src/index';
import { World, component, f32, tag } from '../src/index';
import { kernelSystem, setGPUProvider, getGPUContext, flushKernels, calibrateAuto, getAutoThresholds, setAutoThresholds } from '../src/gpu/index';
import type { KernelSystemHandle } from '../src/gpu/index';

// Jest's CJS runtime cannot require() the ESM-only `webgpu` package, and
// ts-jest rewrites a literal `import()` into that require. Building the import
// through `new Function` hides it from both, so it reaches Node's real ESM
// loader (the `test` script runs jest with --experimental-vm-modules).
const esmImport = new Function('s', 'return import(s)') as (s: string) => Promise<any>;

/**
 * Acquires a Dawn device through the `webgpu` package, or returns null -- the
 * package is an optional dev dependency, and CI machines without a GPU get no
 * adapter. Every test here then returns early with a note instead of failing:
 * "no WebGPU here" is not a CozyECS bug.
 */
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
  console.log(`[gpu-dawn] skipped: ${skipReason}`);
  return true;
}

describe('dawn under jest', () => {
  test('can get a device through setGPUProvider', async () => {
    if (skipped()) return;
    const ctx = await getGPUContext();
    expect(ctx).not.toBeNull();
    expect(ctx!.capabilities.maxComputeWorkgroupSizeX).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// CPU-vs-GPU parity fuzz.
//
// Random kernels over two f32 components are generated as SOURCE TEXT twice:
// once as the kernel handed to kernelSystem, once as an ORACLE in which every
// operation is wrapped in Math.fround -- i.e. the f32 semantics both backends
// promise (docs/GPU.md, "Parity"). The GPU backend (Dawn) and, once
// src/gpu/cpu.ts is implemented, the CPU backend are each compared to the
// oracle, which transitively compares them to each other.
//
// Tolerance (docs/GPU.md): arithmetic <= 1e-6 relative, sqrt <= 1e-5 relative;
// the GPU may fuse a multiply-add (<= 1 ulp per op), which cancellation can
// amplify, so values are compared relative to max(|a|, |b|, 1) at 1e-5. The
// generator keeps inputs in [-2, 2] and expressions shallow so magnitudes stay
// small, and it avoids discontinuous ops (floor/round/comparisons on computed
// values) whose output can legitimately flip on a 1-ulp difference.
// ---------------------------------------------------------------------------

const P = component({ x: f32, y: f32 }, { name: 'P' });
const V = component({ x: f32, y: f32 }, { name: 'V' });
const Split = tag({ name: 'Split' });

const REL_TOL = 1e-5;

/** mulberry32: a deterministic PRNG so every run fuzzes the same kernels. */
function prng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

interface Gen {
  kernel: string;
  oracle: string;
}

const LEAVES = ['p.x', 'p.y', 'v.x', 'v.y', 'dt', 'u.k'];

function genExpr(r: () => number, depth: number): Gen {
  if (depth <= 0 || r() < 0.25) {
    if (r() < 0.2) {
      const c = (Math.round((r() * 4 - 2) * 100) / 100).toFixed(2);
      return { kernel: c, oracle: `F(${c})` };
    }
    const leaf = LEAVES[Math.floor(r() * LEAVES.length)];
    return { kernel: leaf, oracle: leaf };
  }
  const a = genExpr(r, depth - 1);
  const b = genExpr(r, depth - 1);
  const pick = Math.floor(r() * 9);
  switch (pick) {
    case 0:
      return { kernel: `(${a.kernel} + ${b.kernel})`, oracle: `F(${a.oracle} + ${b.oracle})` };
    case 1:
      return { kernel: `(${a.kernel} - ${b.kernel})`, oracle: `F(${a.oracle} - ${b.oracle})` };
    case 2:
    case 3:
      return { kernel: `(${a.kernel} * ${b.kernel})`, oracle: `F(${a.oracle} * ${b.oracle})` };
    case 4:
      return { kernel: `Math.min(${a.kernel}, ${b.kernel})`, oracle: `Math.min(${a.oracle}, ${b.oracle})` };
    case 5:
      return { kernel: `Math.max(${a.kernel}, ${b.kernel})`, oracle: `Math.max(${a.oracle}, ${b.oracle})` };
    case 6:
      return { kernel: `Math.abs(${a.kernel})`, oracle: `Math.abs(${a.oracle})` };
    case 7:
      return { kernel: `Math.sqrt(Math.abs(${a.kernel}))`, oracle: `F(Math.sqrt(Math.abs(${a.oracle})))` };
    default:
      return {
        kernel: `(${a.kernel} / (1 + Math.abs(${b.kernel})))`,
        oracle: `F(${a.oracle} / F(1 + Math.abs(${b.oracle})))`,
      };
  }
}

const TARGETS = ['p.x', 'p.y', 'v.x', 'v.y'];

/** A kernel body of 1-3 statements: a local, then assignments (compound too). */
function genKernel(r: () => number): Gen {
  const k: string[] = [];
  const o: string[] = [];
  const t = genExpr(r, 2);
  k.push(`const t = ${t.kernel};`);
  o.push(`const t = ${t.oracle};`);
  const n = 1 + Math.floor(r() * 3);
  for (let i = 0; i < n; i++) {
    const target = TARGETS[Math.floor(r() * TARGETS.length)];
    const e = genExpr(r, 3);
    const ek = r() < 0.3 ? `${e.kernel} * t` : e.kernel;
    const eo = ek === e.kernel ? e.oracle : `F(${e.oracle} * t)`;
    if (r() < 0.3) {
      k.push(`${target} += ${ek};`);
      o.push(`${target} = F(${target} + ${eo});`);
    } else {
      k.push(`${target} = ${ek};`);
      o.push(`${target} = F(${eo});`);
    }
  }
  return {
    kernel: `(p, v, dt, u) => { ${k.join(' ')} }`,
    oracle: `(p, v, dt, u) => { const F = Math.fround; ${o.join(' ')} }`,
  };
}

function close(a: number, b: number): boolean {
  if (Number.isNaN(a) || Number.isNaN(b)) return Number.isNaN(a) && Number.isNaN(b);
  return Math.abs(a - b) <= REL_TOL * Math.max(1, Math.abs(a), Math.abs(b));
}

interface Row {
  p: { x: number; y: number };
  v: { x: number; y: number };
}

/** Builds a world with `n` random rows split across two archetypes. */
function buildWorld(r: () => number, n: number): { world: World; ids: number[]; rows: Row[] } {
  const world = new World();
  const ids: number[] = [];
  const rows: Row[] = [];
  const f = Math.fround;
  for (let i = 0; i < n; i++) {
    const e = r() < 0.3 ? world.spawn([P, V, Split]) : world.spawn([P, V]);
    const row: Row = {
      p: { x: f(r() * 4 - 2), y: f(r() * 4 - 2) },
      v: { x: f(r() * 4 - 2), y: f(r() * 4 - 2) },
    };
    world.set(e, P, row.p);
    world.set(e, V, row.v);
    ids.push(e);
    rows.push({ p: { ...row.p }, v: { ...row.v } });
  }
  return { world, ids, rows };
}

async function runBackend(
  target: 'gpu' | 'cpu',
  gen: Gen,
  seed: number,
  n: number,
  k: number,
  dt: number,
): Promise<{ backend: string; out: Row[]; handle: KernelSystemHandle }> {
  const r = prng(seed);
  const { world, ids } = buildWorld(r, n);
  const kernel = new Function(`return (${gen.kernel});`)() as (...a: any[]) => void;
  const handle = await kernelSystem(world, `Fuzz${seed}`, {
    components: [P, V],
    uniforms: { k },
    target,
    readback: 'sync-frame',
    kernel,
  });
  world.update(dt);
  await handle.sync();
  // world.get returns a reused scratch object: copy it.
  const out = ids.map((e) => ({ p: { ...world.get(e, P) }, v: { ...world.get(e, V) } }));
  const backend = handle.backend;
  handle.destroy();
  return { backend, out, handle };
}

function oracleRun(gen: Gen, seed: number, n: number, k: number, dt: number): Row[] {
  const r = prng(seed);
  const { rows } = buildWorld(r, n);
  const fn = new Function(`return (${gen.oracle});`)() as (p: any, v: any, dt: number, u: any) => void;
  const f = Math.fround;
  for (const row of rows) {
    fn(row.p, row.v, f(dt), { k: f(k) });
    row.p.x = f(row.p.x);
    row.p.y = f(row.p.y);
    row.v.x = f(row.v.x);
    row.v.y = f(row.v.y);
  }
  return rows;
}

function mismatches(label: string, got: Row[], want: Row[], src: string): string[] {
  const bad: string[] = [];
  for (let i = 0; i < want.length && bad.length < 3; i++) {
    for (const c of ['p', 'v'] as const) {
      for (const f of ['x', 'y'] as const) {
        if (!close(got[i][c][f], want[i][c][f])) {
          bad.push(`${label} row ${i} ${c}.${f}: got ${got[i][c][f]}, oracle ${want[i][c][f]} -- ${src}`);
        }
      }
    }
  }
  return bad;
}

describe('CPU-vs-GPU parity fuzz (Dawn)', () => {
  const warn = jest.spyOn(console, 'warn').mockImplementation(() => {});
  afterAll(() => warn.mockRestore());

  const KERNELS = 200;

  test(`${KERNELS} random kernels agree with the f32 oracle within ${REL_TOL} relative`, async () => {
    if (skipped()) return;
    const failures: string[] = [];
    let cpuCompared = 0;
    for (let s = 1; s <= KERNELS; s++) {
      const r = prng(s * 7919);
      const gen = genKernel(r);
      const n = 1 + Math.floor(r() * 700);
      const k = Math.fround(r() * 4 - 2);
      const dt = Math.fround(1 / 60);
      const dataSeed = s * 104729;
      const want = oracleRun(gen, dataSeed, n, k, dt);

      const gpu = await runBackend('gpu', gen, dataSeed, n, k, dt);
      expect(gpu.backend).toBe('gpu');
      failures.push(...mismatches('gpu', gpu.out, want, gen.kernel));

      const cpu = await runBackend('cpu', gen, dataSeed, n, k, dt);
      if (cpu.backend === 'cpu') {
        cpuCompared++;
        failures.push(...mismatches('cpu', cpu.out, want, gen.kernel));
      }
    }
    if (cpuCompared === 0) {
      // eslint-disable-next-line no-console
      console.log('[gpu-dawn] CPU backend not available yet (src/gpu/cpu.ts); compared GPU to the oracle only.');
    }
    expect(failures).toEqual([]);
  }, 120_000);
});

// ---------------------------------------------------------------------------
// 'async' back-pressure on a real device. 60 frames are dispatched without a
// single await, so at most MAX_READBACKS_IN_FLIGHT staging buffers exist and
// most frames' readbacks are coalesced. flushKernels() must still resolve
// only once the tables hold all 60 frames (regression: x = 3007 vs 3067).
// ---------------------------------------------------------------------------

describe("readback 'async' under back-pressure (Dawn)", () => {
  const warn = jest.spyOn(console, 'warn').mockImplementation(() => {});
  afterAll(() => warn.mockRestore());

  test('200k entities, 60 frames without awaiting, then flushKernels: exact values', async () => {
    if (skipped()) return;
    const N = 200_000;
    const FRAMES = 60;
    const world = new World();
    const ids: number[] = [];
    for (let i = 0; i < N; i++) {
      // Two archetypes, so the catch-up readback has to cover both tables.
      const e = i % 3 === 0 ? world.spawn([P, V, Split]) : world.spawn([P, V]);
      world.set(e, P, { x: i, y: 0 });
      world.set(e, V, { x: 0, y: 0 });
      ids.push(e);
    }
    const handle = await kernelSystem(world, 'Backpressure', {
      components: [P, V],
      target: 'gpu',
      readback: 'async',
      kernel: (p: any, v: any) => {
        p.x += 1;
        p.y -= 2;
        v.x += 3;
      },
    });
    expect(handle.backend).toBe('gpu');
    for (let f = 0; f < FRAMES; f++) world.update(1 / 60);
    expect(handle.stats.dispatches).toBe(FRAMES);
    await flushKernels(world);

    expect(handle.stats.pending).toBe(0);
    expect(handle.stats.completed).toBe(FRAMES);
    expect(handle.stats.staleReadbacks).toBe(0);
    expect(handle.stats.coalescedReadbacks).toBeGreaterThan(0);
    const bad: string[] = [];
    for (let i = 0; i < N && bad.length < 5; i++) {
      const p = world.get(ids[i], P);
      const px = p.x;
      const py = p.y;
      const vx = world.get(ids[i], V).x;
      if (px !== i + FRAMES || py !== -2 * FRAMES || vx !== 3 * FRAMES) {
        bad.push(`row ${i}: p=(${px}, ${py}) v.x=${vx}, want (${i + FRAMES}, ${-2 * FRAMES}) ${3 * FRAMES}`);
      }
    }
    expect(bad).toEqual([]);
    handle.destroy();
  }, 120_000);
});

describe('calibrateAuto (Dawn)', () => {
  test('measures a finite break-even and applies it only when asked', async () => {
    if (skipped()) return;
    const before = getAutoThresholds();
    const r = await calibrateAuto(core, { apply: false, frames: 20, repeats: 3 });
    expect(r).not.toBeNull();
    const { thresholds, breakEven, fits } = r!;
    // Sane band for any real GPU vs a JIT-compiled CPU loop.
    expect(thresholds.fireAndForget).toBeGreaterThan(1_000);
    expect(thresholds.fireAndForget).toBeLessThan(1_000_000);
    expect(thresholds.synchronous).toBeGreaterThanOrEqual(thresholds.fireAndForget);
    expect(breakEven.none).toBeGreaterThan(0);
    // GPU timings wait for completion, so the GPU has a real fixed cost per frame.
    expect(fits.none.fixedMs).toBeGreaterThan(fits.cpu.fixedMs);
    expect(fits.cpu.perEntityNs).toBeGreaterThan(fits.none.perEntityNs);
    expect(getAutoThresholds()).toEqual(before);

    const applied = await calibrateAuto(core, { frames: 10, repeats: 1 });
    expect(applied!.applied).toBe(true);
    expect(getAutoThresholds()).toEqual(applied!.thresholds);
    setAutoThresholds(before);
  }, 60_000);
});
