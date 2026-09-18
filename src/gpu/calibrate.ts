/**
 * `calibrateAuto()`: measures where the GPU starts to beat the CPU on THIS
 * machine and sets the `target: 'auto'` thresholds from it.
 *
 * The built-in defaults (`DEFAULT_AUTO_THRESHOLDS`) were measured on one Apple
 * M4. A different GPU, a slower CPU or a browser instead of Node moves the
 * break-even by a large factor, so an app that ships to many devices should
 * call this once at startup, before creating its kernels.
 *
 * Method. A branch-free probe kernel (gravity + drag + integrate) runs on a
 * throwaway world at two sizes,
 * on the CPU backend and on the GPU with each readback mode. Every GPU timing
 * waits for the work to FINISH (`queue.onSubmittedWorkDone()` for `'none'`,
 * a drain for `'async'`, `sync()` every frame for `'sync-frame'`), so it
 * measures GPU time, not just submission. Each backend is fitted with
 * `ms/frame = fixed + perEntity * n`, and the break-even is where the lines
 * cross. `'none'` and `'async'` share one threshold, so the fire-and-forget
 * threshold is the geometric mean of their two break-evens (the same rule the
 * defaults use). The probe's break-even is finally normalized to a
 * baseline-cost kernel, which is what `autoPrefersGPU` scales per kernel.
 */

import type { World as WorldType } from '../world';
import type { component as componentFn, ComponentType } from '../component';
import type { f32 as f32Token } from '../types';
import { getGPUContext } from './device';
import { BASELINE_CPU_NS_PER_ENTITY, cpuCostEstimate, flushKernels, setAutoThresholds } from './runtime';
import type { AutoThresholds, KernelTarget, ReadbackMode } from './runtime';
import { kernelSystem } from './index';

export interface CalibrateOptions {
  /** Two entity counts to measure at. Default `[32_768, 262_144]`. */
  sizes?: readonly [number, number];
  /** Timed frames per measurement. Default 60. */
  frames?: number;
  /** Repeats per measurement; the median is used. Default 5. */
  repeats?: number;
  /** Apply the result with `setAutoThresholds`. Default true. */
  apply?: boolean;
}

export interface CalibrationResult {
  /** Thresholds for a baseline-cost kernel, as passed to `setAutoThresholds`. */
  thresholds: AutoThresholds;
  /** Measured break-even of the probe kernel, in entities (`Infinity`: the GPU never wins). */
  breakEven: { none: number; async: number; syncFrame: number };
  /** Fitted per-frame costs, `fixed` in ms and `perEntity` in ns. */
  fits: Record<'cpu' | 'none' | 'async' | 'syncFrame', { fixedMs: number; perEntityNs: number }>;
  /** Whether the thresholds were applied. */
  applied: boolean;
  /** Wall time the calibration took, ms. */
  elapsedMs: number;
}

/**
 * The parts of the core `cozyecs` module the calibration needs. Passed in
 * (`import * as cozy from 'cozyecs'; calibrateAuto(cozy)`) because this entry
 * point imports the core for types only: bundling a second copy of the core
 * here would also give it a second component-id counter.
 */
export interface CoreModule {
  World: new (options?: { initialCapacity?: number }) => WorldType;
  component: typeof componentFn;
  f32: typeof f32Token;
}

/** Used when the GPU never wins within reason: effectively "always CPU". */
const NEVER = 1_000_000_000;

interface Probe {
  core: CoreModule;
  Pos: ComponentType;
  Vel: ComponentType;
}

let probeComponents: Probe | null = null;

/** Probe components, created once per core module. */
function probeFor(core: CoreModule): Probe {
  if (probeComponents && probeComponents.core === core) return probeComponents;
  const Pos = core.component({ x: core.f32, y: core.f32 }, { name: 'CalibratePos' });
  const Vel = core.component({ dx: core.f32, dy: core.f32 }, { name: 'CalibrateVel' });
  probeComponents = { core, Pos, Vel };
  return probeComponents;
}

/**
 * Branch-free on purpose: a data-dependent branch makes the CPU side depend on
 * how predictable the probe's data happens to be (up to ~2x on an M4), while
 * the calibration should measure the MACHINE. Per-kernel cost differences are
 * the estimator's job (`cpuCostEstimate`).
 */
const probeKernel = (p: any, v: any, dt: number, u: any): void => {
  v.dy += u.g * dt;
  v.dx = v.dx * u.drag;
  p.x += v.dx * dt;
  p.y += v.dy * dt;
};
const probeUniforms = { g: -9.8, drag: 0.999 };

function now(): number {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}

function median(xs: number[]): number {
  const s = xs.slice().sort((a, b) => a - b);
  return s[s.length >> 1];
}

function makeWorld({ core, Pos, Vel }: Probe, n: number): WorldType {
  const world = new core.World({ initialCapacity: n });
  world.spawnMany(world.archetype(Pos, Vel), n, (chunk, row, i) => {
    const p = chunk.col(Pos) as Record<string, Float32Array>;
    const v = chunk.col(Vel) as Record<string, Float32Array>;
    p.x[row] = i % 1000;
    p.y[row] = 50 + (i % 97);
    v.dx[row] = 1;
    v.dy[row] = 0;
  });
  return world;
}

/** ms/frame for one backend at one size (median of `repeats`). */
async function measure(
  probe: Probe,
  n: number,
  target: KernelTarget,
  readback: ReadbackMode,
  frames: number,
  repeats: number,
  queue: GPUQueue,
): Promise<number> {
  const world = makeWorld(probe, n);
  const handle = await kernelSystem(world, 'calibrate', {
    components: [probe.Pos, probe.Vel],
    target,
    readback,
    uniforms: probeUniforms,
    kernel: probeKernel,
  });
  try {
    const dt = 1 / 60;
    const settle = async (): Promise<void> => {
      if (target === 'cpu') return;
      if (readback === 'none') await queue.onSubmittedWorkDone();
      else await flushKernels(world);
    };
    // Warm-up: JIT, pipeline, first upload.
    for (let f = 0; f < 5; f++) {
      world.update(dt);
      if (readback === 'sync-frame' && target !== 'cpu') await handle.sync();
    }
    await settle();
    const samples: number[] = [];
    for (let r = 0; r < repeats; r++) {
      const t0 = now();
      for (let f = 0; f < frames; f++) {
        world.update(dt);
        if (readback === 'sync-frame' && target !== 'cpu') await handle.sync();
      }
      await settle();
      samples.push((now() - t0) / frames);
    }
    return median(samples);
  } finally {
    handle.destroy();
  }
}

/** Line through the two measured points, clamped to non-negative terms. */
function fit(n0: number, t0: number, n1: number, t1: number): { fixedMs: number; perEntityNs: number } {
  let per = (t1 - t0) / (n1 - n0); // ms per entity
  if (!(per > 0)) per = 0;
  let fixed = t0 - per * n0;
  if (!(fixed > 0)) fixed = 0;
  return { fixedMs: fixed, perEntityNs: per * 1e6 };
}

/** Entities where `gpu` becomes cheaper than `cpu`. */
function crossing(cpu: { fixedMs: number; perEntityNs: number }, gpu: { fixedMs: number; perEntityNs: number }): number {
  const dPer = (cpu.perEntityNs - gpu.perEntityNs) / 1e6; // ms per entity the GPU saves
  const dFixed = gpu.fixedMs - cpu.fixedMs; // ms per frame the GPU costs extra
  if (dFixed <= 0) return 0;
  if (dPer <= 0) return Infinity;
  return dFixed / dPer;
}

/**
 * Measures the CPU/GPU break-even on this machine and (by default) applies it
 * with `setAutoThresholds`. Affects kernels created AFTER it resolves, so call
 * it at startup before `kernelSystem`. Takes about a second on a desktop GPU.
 *
 * Resolves `null`, changing nothing, when no WebGPU device is available (the
 * CPU backend is then used regardless). Never throws for device problems.
 *
 * ```js
 * import * as cozy from 'cozyecs';
 * import { calibrateAuto, kernelSystem } from 'cozyecs/gpu';
 * const cal = await calibrateAuto(cozy);   // e.g. { thresholds: { fireAndForget: 41_000, ... } }
 * const move = await kernelSystem(world, 'Move', { ..., target: 'auto' });
 * ```
 */
export async function calibrateAuto(core: CoreModule, options: CalibrateOptions = {}): Promise<CalibrationResult | null> {
  if (!core || typeof core.World !== 'function' || typeof core.component !== 'function' || !core.f32) {
    throw new TypeError("calibrateAuto(core): pass the core module, e.g. import * as cozy from 'cozyecs'; calibrateAuto(cozy)");
  }
  const started = now();
  const probe = probeFor(core);
  const context = await getGPUContext();
  if (!context) return null;
  const queue = context.device.queue;
  const [n0, n1] = options.sizes || [32_768, 262_144];
  if (!(n0 > 0 && n1 > n0)) throw new RangeError('calibrateAuto: sizes must be two increasing positive counts');
  const frames = Math.max(1, options.frames ?? 60);
  const repeats = Math.max(1, options.repeats ?? 5);

  const line = async (target: KernelTarget, readback: ReadbackMode) =>
    fit(
      n0,
      await measure(probe, n0, target, readback, frames, repeats, queue),
      n1,
      await measure(probe, n1, target, readback, frames, repeats, queue),
    );

  let fits: CalibrationResult['fits'];
  try {
    fits = {
      cpu: await line('cpu', 'async'),
      none: await line('gpu', 'none'),
      async: await line('gpu', 'async'),
      syncFrame: await line('gpu', 'sync-frame'),
    };
  } catch {
    return null; // a device lost mid-calibration: keep the current thresholds
  }

  const breakEven = {
    none: crossing(fits.cpu, fits.none),
    async: crossing(fits.cpu, fits.async),
    syncFrame: crossing(fits.cpu, fits.syncFrame),
  };

  // Normalize the probe's break-even to a baseline-cost kernel: autoPrefersGPU
  // uses `threshold * BASELINE / estimate`, so threshold = n * estimate / BASELINE.
  const probeWorld = new core.World();
  probeWorld.spawn([probe.Pos, probe.Vel]);
  const probeHandle = await kernelSystem(probeWorld, 'calibrateProbe', {
    components: [probe.Pos, probe.Vel],
    target: 'cpu',
    uniforms: probeUniforms,
    kernel: probeKernel,
  });
  const scale = cpuCostEstimate(probeHandle.ir) / BASELINE_CPU_NS_PER_ENTITY;
  probeHandle.destroy();

  const toThreshold = (n: number): number => (Number.isFinite(n) ? Math.min(NEVER, Math.round(n * scale)) : NEVER);
  const ff =
    Number.isFinite(breakEven.none) && Number.isFinite(breakEven.async)
      ? Math.sqrt(Math.max(1, breakEven.none) * Math.max(1, breakEven.async))
      : Math.min(breakEven.none, breakEven.async);
  const thresholds: AutoThresholds = {
    fireAndForget: toThreshold(ff),
    synchronous: toThreshold(breakEven.syncFrame),
  };

  const apply = options.apply !== false;
  if (apply) setAutoThresholds(thresholds);
  return { thresholds, breakEven, fits, applied: apply, elapsedMs: now() - started };
}
