/// <reference types="@webgpu/types" />
import { describe, test, expect, jest, afterAll, beforeEach, afterEach } from '@jest/globals';
import {
  getGPUContext,
  hasGPUContext,
  peekGPUContext,
  setGPUProvider,
} from '../src/gpu/index';

// The gpu entry must never throw on a host without WebGPU: `kernelSystem` is
// supposed to fall back to the CPU backend, and a rejected promise here would
// take the whole world down at registration time.
describe('gpu device acquisition', () => {
  const warn = jest.spyOn(console, 'warn').mockImplementation(() => {});
  afterAll(() => warn.mockRestore());

  test('resolves to null instead of throwing when no provider exists', async () => {
    expect((globalThis as { navigator?: { gpu?: unknown } }).navigator?.gpu).toBeUndefined();
    await expect(getGPUContext()).resolves.toBeNull();
  });

  test('warns exactly once across repeated calls', async () => {
    warn.mockClear();
    await getGPUContext();
    await getGPUContext();
    await getGPUContext();
    expect(warn).toHaveBeenCalledTimes(0); // already warned on the first test's call
  });

  test('hasGPUContext/peekGPUContext report no device rather than "attempted"', async () => {
    await getGPUContext();
    expect(hasGPUContext()).toBe(false);
    expect(peekGPUContext()).toBeNull();
  });

  test('setGPUProvider resets a previously memoized failure', async () => {
    const fake = {
      requestAdapter: async (): Promise<null> => null,
    } as unknown as GPU;
    setGPUProvider(fake);
    expect(peekGPUContext()).toBeNull();
    await expect(getGPUContext()).resolves.toBeNull();
  });

  test('surfaces granted limits, not adapter maxima, as capabilities', async () => {
    // A device is only granted the limits it asked for, so codegen budgets have
    // to come from device.limits. Assert we read the device, not the adapter.
    const limits = {
      maxStorageBuffersPerShaderStage: 8,
      maxComputeWorkgroupSizeX: 256,
      maxComputeInvocationsPerWorkgroup: 256,
      maxBufferSize: 268435456,
      maxStorageBufferBindingSize: 134217728,
      minStorageBufferOffsetAlignment: 256,
      minUniformBufferOffsetAlignment: 256,
      maxComputeWorkgroupsPerDimension: 65535,
    };
    const fake = {
      requestAdapter: async (): Promise<unknown> => ({
        limits: { ...limits, maxStorageBuffersPerShaderStage: 10, maxComputeWorkgroupSizeX: 1024 },
        features: new Set<string>(['subgroups']),
        requestDevice: async (): Promise<unknown> => ({
          limits,
          features: new Set<string>(['subgroups']),
        }),
      }),
    } as unknown as GPU;
    setGPUProvider(fake);
    const ctx = await getGPUContext();
    expect(ctx).not.toBeNull();
    expect(ctx!.capabilities.maxStorageBuffersPerShaderStage).toBe(8);
    expect(ctx!.capabilities.maxComputeWorkgroupSizeX).toBe(256);
    expect(ctx!.capabilities.subgroups).toBe(true);
    expect(ctx!.capabilities.timestampQuery).toBe(false);
    expect(hasGPUContext()).toBe(true);
    expect(peekGPUContext()).toBe(ctx);
  });
});

// ---------------------------------------------------------------------------
// kernelSystem end to end, and the runtime's backend resolution.
//
// Every test below loads the gpu entry in an ISOLATED module registry, so the
// device module's once-per-process warning and memoized context start fresh.
// ---------------------------------------------------------------------------

import { World, component, f32, f64 } from '../src/index';

type GpuEntry = typeof import('../src/gpu/index');

async function freshEntry(): Promise<GpuEntry> {
  let mod: GpuEntry | null = null;
  await jest.isolateModulesAsync(async () => {
    mod = await import('../src/gpu/index');
  });
  return mod as unknown as GpuEntry;
}

/**
 * True once src/gpu/cpu.ts is implemented. Until then compileCPUKernel throws
 * "not implemented", every CPU-target kernel reports backend 'none' (which is
 * itself asserted below), and the value-level CPU tests are skipped.
 */
function cpuBackendReady(): boolean {
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const cpu = require('../src/gpu/cpu') as typeof import('../src/gpu/cpu');
  try {
    cpu.generateCPUSource(null as never);
  } catch (e) {
    return !/not implemented/.test((e as Error).message);
  }
  return true;
}
const CPU_READY = cpuBackendReady();
const cpuTest = CPU_READY ? test : test.skip;

const Pos = component({ x: f32, y: f32 }, { name: 'Pos' });
const Vel = component({ x: f32, y: f32 }, { name: 'Vel' });
const Wide = component({ x: f64 }, { name: 'Wide' });

function spawnMovers(world: World, n: number): number[] {
  const ids: number[] = [];
  for (let i = 0; i < n; i++) {
    const e = world.spawn([Pos, Vel]);
    world.set(e, Pos, { x: i, y: 10 });
    world.set(e, Vel, { x: 1, y: 0 });
    ids.push(e);
  }
  return ids;
}

const moveKernel = (p: any, v: any, dt: number, u: any) => {
  v.y += u.gravity * dt;
  p.x += v.x * dt;
  p.y += v.y * dt;
  if (p.y < 0) {
    p.y = 0;
    v.y = -v.y * 0.5;
  }
};

/** A fresh console.warn spy per test (spies from other describe blocks restore on their own schedule). */
function spyWarn() {
  return jest.spyOn(console, 'warn').mockImplementation(() => {});
}

function warnings(spy: { mock: { calls: unknown[][] } }, re: RegExp): number {
  return spy.mock.calls.filter((c) => re.test(String(c[0]))).length;
}

describe('kernelSystem: target cpu', () => {
  let warn: ReturnType<typeof spyWarn>;
  beforeEach(() => (warn = spyWarn()));
  afterEach(() => warn.mockRestore());

  test('registers without throwing and never requests a device', async () => {
    const gpu = await freshEntry();
    const world = new World();
    spawnMovers(world, 4);
    const h = await gpu.kernelSystem(world, 'MoveCPU', {
      components: [Pos, Vel],
      uniforms: { gravity: -10 },
      target: 'cpu',
      kernel: moveKernel,
    });
    expect(['cpu', 'none']).toContain(h.backend);
    expect(gpu.hasGPUContext()).toBe(false);
    expect(warnings(warn, /No WebGPU device/)).toBe(0);
    expect(() => world.update(0.1)).not.toThrow();
    await expect(h.sync()).resolves.toBeUndefined();
  });

  cpuTest('integrates positions like the kernel says, reading uniforms and dt per frame', async () => {
    const gpu = await freshEntry();
    const world = new World();
    const ids = spawnMovers(world, 3);
    const h = await gpu.kernelSystem(world, 'MoveCPU2', {
      components: [Pos, Vel],
      uniforms: { gravity: -10 },
      target: 'cpu',
      kernel: moveKernel,
    });
    expect(h.backend).toBe('cpu');
    world.update(0.5);
    await h.sync();
    // v.y = -5, p.x = i + 0.5, p.y = 10 - 2.5
    for (let i = 0; i < ids.length; i++) {
      expect(world.get(ids[i], Pos)).toEqual({ x: i + 0.5, y: 7.5 });
      expect(world.get(ids[i], Vel)).toEqual({ x: 1, y: -5 });
    }
    h.setUniform('gravity', 0);
    world.update(1);
    expect(world.get(ids[0], Pos)).toEqual({ x: 1.5, y: 2.5 });
    expect(h.stats.lastBackend).toBe('cpu');
    expect(h.stats.lastEntities).toBe(3);
  });

  cpuTest('bounce branch runs and entities added later are picked up', async () => {
    const gpu = await freshEntry();
    const world = new World();
    const [a] = spawnMovers(world, 1);
    world.set(a, Pos, { x: 0, y: 0.1 });
    world.set(a, Vel, { x: 0, y: -4 });
    await gpu.kernelSystem(world, 'Bounce', { components: [Pos, Vel], uniforms: { gravity: 0 }, target: 'cpu', kernel: moveKernel });
    world.update(1);
    expect(world.get(a, Pos).y).toBe(0);
    expect(world.get(a, Vel).y).toBe(2);
    const later = spawnMovers(world, 2000); // grows the archetype
    world.update(1);
    expect(world.get(later[1999], Pos).x).toBe(2000);
  });
});

describe('kernelSystem: target auto with no device', () => {
  let warn: ReturnType<typeof spyWarn>;
  beforeEach(() => (warn = spyWarn()));
  afterEach(() => warn.mockRestore());

  test('never throws, warns about the missing device exactly once across kernels', async () => {
    const gpu = await freshEntry();
    expect((globalThis as { navigator?: { gpu?: unknown } }).navigator?.gpu).toBeUndefined();
    const world = new World();
    spawnMovers(world, 8);
    const a = await gpu.kernelSystem(world, 'AutoA', { components: [Pos, Vel], uniforms: { gravity: -1 }, kernel: moveKernel });
    const b = await gpu.kernelSystem(world, 'AutoB', {
      components: [Pos],
      target: 'auto',
      kernel: (p: any, dt: number) => {
        p.x += dt;
      },
    });
    expect(warnings(warn, /No WebGPU device/)).toBe(1);
    for (const h of [a, b]) expect(h.backend).toBe(CPU_READY ? 'cpu' : 'none');
    expect(() => {
      world.update(1 / 60);
      world.update(1 / 60);
    }).not.toThrow();
    expect(warnings(warn, /No WebGPU device/)).toBe(1);
    await expect(gpu.flushKernels(world)).resolves.toBeUndefined();
  });
});

describe('backend "none" is reported, never silent', () => {
  let warn: ReturnType<typeof spyWarn>;
  beforeEach(() => (warn = spyWarn()));
  afterEach(() => warn.mockRestore());

  test('CPU compile failure + no GPU: backend none, one warning, dispatch is a no-op', async () => {
    let gpu: GpuEntry | null = null;
    await jest.isolateModulesAsync(async () => {
      jest.doMock('../src/gpu/cpu', () => ({
        ...(jest.requireActual('../src/gpu/cpu') as object),
        compileCPUKernel: () => {
          throw new Error('synthetic codegen failure');
        },
      }));
      gpu = await import('../src/gpu/index');
    });
    jest.dontMock('../src/gpu/cpu');
    const g = gpu as unknown as GpuEntry;
    const world = new World();
    const [e] = spawnMovers(world, 1);
    const h = await g.kernelSystem(world, 'Broken', { components: [Pos, Vel], uniforms: { gravity: -1 }, target: 'cpu', kernel: moveKernel });
    expect(h.backend).toBe('none');
    expect(h.stats.lastBackend).toBe('none');
    const msgs = warn.mock.calls.map((c) => String(c[0])).filter((m) => /Broken/.test(m));
    expect(msgs).toHaveLength(1);
    expect(msgs[0]).toMatch(/no usable backend/);
    expect(msgs[0]).toMatch(/synthetic codegen failure/);
    expect(msgs[0]).toMatch(/'none'/);
    world.update(1);
    world.update(1);
    expect(world.get(e, Pos)).toEqual({ x: 0, y: 10 });
    expect(h.stats.dispatches).toBe(0);
    expect(warn.mock.calls.filter((c) => /Broken/.test(String(c[0])))).toHaveLength(1);
  });
});

describe('free identifiers', () => {
  let warn: ReturnType<typeof spyWarn>;
  beforeEach(() => (warn = spyWarn()));
  afterEach(() => warn.mockRestore());

  test('a closure variable is E_UNKNOWN_IDENTIFIER naming it and suggesting uniforms', async () => {
    const gpu = await freshEntry();
    const world = new World();
    const gravity = -9.8;
    void gravity;
    let err: unknown = null;
    try {
      await gpu.kernelSystem(world, 'Leaky', {
        components: [Pos],
        target: 'cpu',
        kernel: (p: any, dt: number) => {
          p.y += gravity * dt;
        },
      });
    } catch (e) {
      err = e;
    }
    expect(err).toBeInstanceOf(gpu.KernelError);
    const ke = err as InstanceType<GpuEntry['KernelError']>;
    expect(ke.code).toBe('E_UNKNOWN_IDENTIFIER');
    const text = `${ke.message}\n${(ke as { hint?: string }).hint ?? ''}`;
    expect(text).toMatch(/gravity/);
    expect(text).toMatch(/uniforms/);
    expect(text).toMatch(/u\.gravity/);
    // Registration failed before a system was added: an update is a no-op.
    expect(() => world.update(1)).not.toThrow();
  });

  test('Math.random and world access are rejected at registration', async () => {
    const gpu = await freshEntry();
    const world = new World();
    await expect(
      gpu.kernelSystem(world, 'Rnd', { components: [Pos], target: 'cpu', kernel: (p: any) => { p.x = Math.random(); } }),
    ).rejects.toBeInstanceOf(gpu.KernelError);
    await expect(
      gpu.kernelSystem(world, 'W', { components: [Pos], target: 'cpu', kernel: (p: any) => { p.x = (world as any).tick; } }),
    ).rejects.toMatchObject({ code: 'E_UNKNOWN_IDENTIFIER' });
  });
});

describe('f64 fields', () => {
  let warn: ReturnType<typeof spyWarn>;
  beforeEach(() => (warn = spyWarn()));
  afterEach(() => warn.mockRestore());

  test('warns once per kernel that f64 is computed in f32', async () => {
    const gpu = await freshEntry();
    const world = new World();
    world.spawn([Wide]);
    await gpu.kernelSystem(world, 'Wide1', { components: [Wide], target: 'cpu', kernel: (w: any, dt: number) => { w.x += dt; } });
    const msgs = warn.mock.calls.map((c) => String(c[0])).filter((m) => /Wide1/.test(m) && /f64/.test(m));
    expect(msgs).toHaveLength(1);
    expect(msgs[0]).toMatch(/f32/);
  });
});

describe('auto break-even', () => {
  test('defaults: ~45k fire-and-forget, ~700k sync-frame (Dawn/M4), scaled by estimated CPU cost', async () => {
    const gpu = await freshEntry();
    const t = { ...gpu.DEFAULT_AUTO_THRESHOLDS };
    expect(t).toEqual({ fireAndForget: 45_000, synchronous: 700_000 });
    expect(gpu.GPU_FIXED_OVERHEAD_NS.fireAndForget).toBeCloseTo(45_000 * gpu.BASELINE_CPU_NS_PER_ENTITY);
    expect(gpu.GPU_FIXED_OVERHEAD_NS.synchronous).toBeCloseTo(700_000 * gpu.BASELINE_CPU_NS_PER_ENTITY);
    expect(gpu.BASELINE_CPU_NS_PER_ENTITY).toBe(1.14);
    const ir = { opCount: 12 } as never;
    const base = gpu.BASELINE_CPU_NS_PER_ENTITY;
    expect(gpu.autoPrefersGPU(ir, 44_999, 'async', t, false, base)).toBe(false);
    expect(gpu.autoPrefersGPU(ir, 45_000, 'async', t, false, base)).toBe(true);
    expect(gpu.autoPrefersGPU(ir, 45_000, 'none', t, false, base)).toBe(true);
    expect(gpu.autoPrefersGPU(ir, 699_999, 'sync-frame', t, false, base)).toBe(false);
    expect(gpu.autoPrefersGPU(ir, 700_000, 'sync-frame', t, false, base)).toBe(true);
    // A kernel 4x as expensive per entity pays back the overhead 4x sooner.
    expect(gpu.autoPrefersGPU(ir, 11_250, 'async', t, false, base * 4)).toBe(true);
    expect(gpu.autoPrefersGPU(ir, 11_249, 'async', t, false, base * 4)).toBe(false);
    // Clamped: an absurdly cheap estimate never pushes the threshold past 4x.
    expect(gpu.autoPrefersGPU(ir, 180_000, 'async', t, false, 1e-9)).toBe(true);
    expect(gpu.autoPrefersGPU(ir, 179_999, 'async', t, false, 1e-9)).toBe(false);
    // Hysteresis: once on the GPU it stays until threshold / 1.25.
    expect(gpu.autoPrefersGPU(ir, 36_000, 'async', t, true, base)).toBe(true);
    expect(gpu.autoPrefersGPU(ir, 35_999, 'async', t, true, base)).toBe(false);
  });

  test('user override: per-kernel thresholds and setAutoThresholds', async () => {
    const gpu = await freshEntry();
    const ir = { opCount: 12 } as never;
    const base = gpu.BASELINE_CPU_NS_PER_ENTITY;
    expect(gpu.autoPrefersGPU(ir, 100, 'async', { fireAndForget: 100, synchronous: 1 }, false, base)).toBe(true);
    gpu.setAutoThresholds({ fireAndForget: 1 });
    expect(gpu.DEFAULT_AUTO_THRESHOLDS.fireAndForget).toBe(45_000); // defaults stay frozen
  });

  test('cpuCostEstimate falls back to the op-count proxy and is always positive', async () => {
    const gpu = await freshEntry();
    const ns = gpu.cpuCostEstimate({ opCount: 24 } as never);
    expect(ns).toBeGreaterThan(0);
    expect(Number.isFinite(ns)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// 'async' readback under back-pressure, on a scripted mock device.
//
// The mock executes the one kernel used here (`c.x += 1` over an f32 field)
// on the CPU when a command buffer is submitted, copies buffers for real, and
// HOLDS every mapAsync until the test releases it -- so the runtime runs out
// of staging buffers exactly like Dawn does at 1M entities. Regression test
// for "sync() resolves with the tables behind the GPU": every frame must land.
// ---------------------------------------------------------------------------

import { MAX_READBACKS_IN_FLIGHT } from '../src/gpu/runtime';

interface MockBuffer {
  size: number;
  data: Uint8Array;
  destroyed: boolean;
  mapAsync(mode: number, offset?: number, size?: number): Promise<void>;
  getMappedRange(offset?: number, size?: number): ArrayBuffer;
  unmap(): void;
  destroy(): void;
}

function mockDevice() {
  const heldMaps: (() => void)[] = [];
  const limits = {
    maxStorageBuffersPerShaderStage: 8,
    maxComputeWorkgroupSizeX: 256,
    maxComputeInvocationsPerWorkgroup: 256,
    maxBufferSize: 268435456,
    maxStorageBufferBindingSize: 134217728,
    minStorageBufferOffsetAlignment: 256,
    minUniformBufferOffsetAlignment: 256,
    maxComputeWorkgroupsPerDimension: 65535,
  };
  const createBuffer = (desc: { size: number }): MockBuffer => {
    const b: MockBuffer = {
      size: desc.size,
      data: new Uint8Array(desc.size),
      destroyed: false,
      mapAsync: () => new Promise<void>((resolve, reject) => heldMaps.push(() => (b.destroyed ? reject(new Error('destroyed')) : resolve()))),
      getMappedRange: (offset = 0, size?: number) => b.data.slice(offset, size === undefined ? undefined : offset + size).buffer,
      unmap: () => {},
      destroy: () => {
        b.destroyed = true;
      },
    };
    return b;
  };
  type Op =
    | { kind: 'dispatch'; bind: { entries: { binding: number; resource: { buffer: MockBuffer; offset?: number } }[] } }
    | { kind: 'copy'; src: MockBuffer; srcOff: number; dst: MockBuffer; dstOff: number; size: number };
  const createCommandEncoder = () => {
    const ops: Op[] = [];
    let bind: any = null;
    return {
      beginComputePass: () => ({
        setPipeline: () => {},
        setBindGroup: (_i: number, bg: unknown) => {
          bind = bg;
        },
        dispatchWorkgroups: () => ops.push({ kind: 'dispatch', bind }),
        end: () => {},
      }),
      copyBufferToBuffer: (src: MockBuffer, srcOff: number, dst: MockBuffer, dstOff: number, size: number) =>
        ops.push({ kind: 'copy', src, srcOff, dst, dstOff, size }),
      finish: () => ops,
    };
  };
  const submit = (buffers: Op[][]) => {
    for (const ops of buffers) {
      for (const op of ops) {
        if (op.kind === 'copy') {
          op.dst.data.set(op.src.data.subarray(op.srcOff, op.srcOff + op.size), op.dstOff);
          continue;
        }
        // uniform layout (ir.uniformLayout): count @4, base @8, first field base @16.
        const entries = op.bind.entries;
        const table = entries[0].resource.buffer;
        const uni = entries[entries.length - 1].resource;
        const u32 = new Uint32Array(uni.buffer.data.buffer, uni.offset ?? 0, 8);
        const count = u32[1];
        const fieldBase = u32[4];
        const f = new Float32Array(table.data.buffer);
        for (let i = 0; i < count; i++) f[fieldBase + i] += 1;
      }
    }
  };
  const device = {
    limits,
    features: new Set<string>(),
    lost: new Promise(() => {}),
    createBuffer,
    createShaderModule: () => ({}),
    createComputePipelineAsync: async () => ({ getBindGroupLayout: () => ({}) }),
    createBindGroup: (desc: unknown) => desc,
    createCommandEncoder,
    queue: {
      writeBuffer: (b: MockBuffer, off: number, src: Uint8Array, srcOff = 0, size?: number) => {
        const n = size ?? src.byteLength - srcOff;
        b.data.set(new Uint8Array(src.buffer, src.byteOffset + srcOff, n), off);
      },
      submit,
    },
  };
  const gpu = {
    requestAdapter: async () => ({ limits, features: new Set<string>(), requestDevice: async () => device }),
  } as unknown as GPU;
  return {
    gpu,
    /** Resolves every held map (and any issued while doing so), one tick at a time. */
    async releaseAll(): Promise<void> {
      for (let guard = 0; guard < 1000 && heldMaps.length > 0; guard++) {
        heldMaps.shift()!();
        await new Promise((r) => setTimeout(r, 0));
      }
    },
    held: () => heldMaps.length,
  };
}

describe("readback 'async' under back-pressure (mock device)", () => {
  const G = globalThis as Record<string, unknown>;
  const saved = { usage: G.GPUBufferUsage, map: G.GPUMapMode };
  let warn: ReturnType<typeof spyWarn>;
  beforeEach(() => {
    warn = spyWarn();
    G.GPUBufferUsage ??= { MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128 };
    G.GPUMapMode ??= { READ: 1 };
  });
  afterEach(() => {
    warn.mockRestore();
    G.GPUBufferUsage = saved.usage;
    G.GPUMapMode = saved.map;
  });

  const Cnt = component({ x: f32 }, { name: 'Cnt' });

  test('skipped readbacks are coalesced, never lost: sync() sees every frame', async () => {
    const gpu = await freshEntry();
    const mock = mockDevice();
    gpu.setGPUProvider(mock.gpu);
    const world = new World();
    const N = 100;
    const ids: number[] = [];
    for (let i = 0; i < N; i++) {
      const e = world.spawn([Cnt]);
      world.set(e, Cnt, { x: i });
      ids.push(e);
    }
    const h = await gpu.kernelSystem(world, 'Count', {
      components: [Cnt],
      target: 'gpu',
      readback: 'async',
      kernel: (c: any) => {
        c.x += 1;
      },
    });
    expect(h.backend).toBe('gpu');

    const FRAMES = MAX_READBACKS_IN_FLIGHT + 5;
    for (let f = 0; f < FRAMES; f++) world.update(1 / 60);
    // Every staging buffer is held: the last 5 frames could not get one.
    expect(mock.held()).toBe(MAX_READBACKS_IN_FLIGHT);
    expect(h.stats.pending).toBe(FRAMES);
    expect(h.stats.coalescedReadbacks).toBe(5);
    expect(warnings(warn, /coalescing frames/)).toBe(1);
    expect(warnings(warn, /none are lost/)).toBe(1);

    let synced = false;
    const done = h.sync().then(() => {
      synced = true;
    });
    // Applying the first readbacks must not resolve sync(): the tables would
    // still be FRAMES - 1 frames behind the device.
    await mock.releaseAll();
    await done;
    expect(synced).toBe(true);
    for (let i = 0; i < N; i++) expect(world.get(ids[i], Cnt).x).toBe(i + FRAMES);
    expect(h.stats.pending).toBe(0);
    expect(h.stats.completed).toBe(FRAMES);
    expect(h.stats.coalescedReadbacks).toBeGreaterThan(0);
    expect(h.stats.staleReadbacks).toBe(0);
    expect(mock.held()).toBe(0);

    // sync() waits for the dispatches submitted BEFORE the call only.
    world.update(1 / 60);
    const s2 = h.sync();
    world.update(1 / 60); // not part of s2's target
    await mock.releaseAll();
    await s2;
    await gpu.flushKernels(world);
    for (let i = 0; i < N; i++) expect(world.get(ids[i], Cnt).x).toBe(i + FRAMES + 2);
    expect(h.stats.pending).toBe(0);
    expect(h.stats.staleReadbacks).toBe(0);
  });

  test('destroy with a catch-up owed resolves waiters and drains without throwing', async () => {
    const gpu = await freshEntry();
    const mock = mockDevice();
    gpu.setGPUProvider(mock.gpu);
    const world = new World();
    for (let i = 0; i < 10; i++) world.spawn([Cnt]);
    const h = await gpu.kernelSystem(world, 'CountD', {
      components: [Cnt],
      target: 'gpu',
      readback: 'async',
      kernel: (c: any) => {
        c.x += 1;
      },
    });
    for (let f = 0; f < MAX_READBACKS_IN_FLIGHT + 3; f++) world.update(1 / 60);
    const s = h.sync();
    h.destroy();
    await expect(s).resolves.toBeUndefined();
    await expect(mock.releaseAll()).resolves.toBeUndefined();
    expect(h.stats.pending).toBe(0);
  });
});

describe("target 'auto': estimator calibration and backend prediction (mock device)", () => {
  const G = globalThis as Record<string, unknown>;
  const saved = { usage: G.GPUBufferUsage, map: G.GPUMapMode };
  let warn: ReturnType<typeof spyWarn>;
  beforeEach(() => {
    warn = spyWarn();
    G.GPUBufferUsage ??= { MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128 };
    G.GPUMapMode ??= { READ: 1 };
  });
  afterEach(() => {
    warn.mockRestore();
    G.GPUBufferUsage = saved.usage;
    G.GPUMapMode = saved.map;
  });

  const P = component({ x: f32, y: f32 }, { name: 'AP' });
  const V = component({ dx: f32, dy: f32 }, { name: 'AV' });

  async function autoKernel(gpu: Awaited<ReturnType<typeof freshEntry>>, world: World, name: string, thresholds?: { fireAndForget?: number }) {
    return gpu.kernelSystem(world, name, {
      components: [P, V],
      target: 'auto',
      readback: 'none',
      thresholds: thresholds as never,
      kernel: (p: any, v: any) => {
        p.x += v.dx;
        p.y += v.dy;
      },
    });
  }

  test('the simple benchmark kernel switches near its measured break-even (49k none / 64k async), not at ~90k', async () => {
    const gpu = await freshEntry();
    const mock = mockDevice();
    gpu.setGPUProvider(mock.gpu);
    const h = await autoKernel(gpu, new World(), 'SimpleAuto');
    const t = { ...gpu.DEFAULT_AUTO_THRESHOLDS };
    const ns = gpu.cpuCostEstimate(h.ir);
    expect(gpu.autoPrefersGPU(h.ir, 45_000, 'none', t, false, ns)).toBe(false);
    expect(gpu.autoPrefersGPU(h.ir, 70_000, 'none', t, false, ns)).toBe(true);
    expect(gpu.autoPrefersGPU(h.ir, 70_000, 'async', t, false, ns)).toBe(true);
    h.destroy();
  });

  test('before the first dispatch, backend reports the prediction for the current query size', async () => {
    const gpu = await freshEntry();
    const mock = mockDevice();
    gpu.setGPUProvider(mock.gpu);
    const world = new World();
    for (let i = 0; i < 100; i++) world.spawn([P, V]);

    // 100 entities is far below the break-even: the first dispatch will run on the CPU.
    const small = await autoKernel(gpu, world, 'SmallAuto');
    expect(small.backend).toBe('cpu');
    expect(small.stats.lastBackend).toBe('cpu');
    world.update(1 / 60);
    expect(small.stats.lastBackend).toBe('cpu');
    expect(small.backend).toBe('cpu');
    small.destroy();

    // Same query, threshold below its size: predicted and dispatched on the GPU.
    const big = await autoKernel(gpu, world, 'BigAuto', { fireAndForget: 10 });
    expect(big.backend).toBe('gpu');
    expect(big.stats.lastBackend).toBe('gpu');
    world.update(1 / 60);
    expect(big.stats.lastBackend).toBe('gpu');
    big.destroy();
    await mock.releaseAll();
  });

  test('the prediction follows spawns that happen after registration, until the first dispatch', async () => {
    const gpu = await freshEntry();
    const mock = mockDevice();
    gpu.setGPUProvider(mock.gpu);
    const world = new World();
    const h = await autoKernel(gpu, world, 'GrowAuto', { fireAndForget: 50 });
    expect(h.backend).toBe('cpu'); // empty query
    for (let i = 0; i < 100; i++) world.spawn([P, V]);
    expect(h.backend).toBe('gpu'); // re-predicted from the new size
    world.update(1 / 60);
    expect(h.stats.lastBackend).toBe('gpu');
    h.destroy();
    await mock.releaseAll();
  });

  test("pinned targets are unaffected: 'gpu' reports gpu even for a tiny query", async () => {
    const gpu = await freshEntry();
    const mock = mockDevice();
    gpu.setGPUProvider(mock.gpu);
    const world = new World();
    world.spawn([P, V]);
    const h = await gpu.kernelSystem(world, 'PinnedGpu', {
      components: [P, V],
      target: 'gpu',
      readback: 'none',
      kernel: (p: any, v: any) => {
        p.x += v.dx;
      },
    });
    expect(h.backend).toBe('gpu');
    h.destroy();
    await mock.releaseAll();
  });
});

describe('calibrateAuto without a device', () => {
  let warn: ReturnType<typeof spyWarn>;
  beforeEach(() => {
    warn = spyWarn();
  });
  afterEach(() => warn.mockRestore());

  test('resolves null and leaves the thresholds alone', async () => {
    const gpu = await freshEntry();
    const core = await import('../src/index');
    const before = gpu.getAutoThresholds();
    await expect(gpu.calibrateAuto(core)).resolves.toBeNull();
    expect(gpu.getAutoThresholds()).toEqual(before);
  });

  test('rejects a missing core module with a TypeError that says what to pass', async () => {
    const gpu = await freshEntry();
    await expect(gpu.calibrateAuto(undefined as never)).rejects.toThrow(/import \* as cozy from 'cozyecs'/);
  });
});
