// WebGPU device acquisition for the optional `cozyecs/gpu` entry point.
//
// Nothing here is imported by the core bundle: `src/index.ts` never references
// `src/gpu/**`, so a consumer that only imports `cozyecs` pays nothing for it.
//
// Two host-specific hazards are handled here rather than in the codegen:
//
//  1. Default device limits are far below what the adapter supports. Dawn (and
//     browsers) hand back the *spec default* limits unless you ask for more --
//     on an M4 that is 8 storage buffers and workgroup_size 256, against an
//     adapter maximum of 10 and 1024. Codegen budgets have to come from the
//     device we actually got, so we request the adapter's maxima up front and
//     then report what was granted.
//
//  2. The `webgpu` (Dawn) Node binding schedules `ProcessEvents` on the libuv
//     check phase for as long as async GPU work is outstanding. V8 liveness GC
//     will collect the `GPU` instance once the last *textual* use of it has
//     passed -- even a module-level `const` -- and the already-scheduled
//     callback then runs against freed Dawn memory and segfaults the process.
//     Measured at ~30% of runs on a 1M-entity dispatch. `keepAlive` below is
//     the fix: it is read at teardown, so the instance stays reachable.

/** What the codegen is allowed to assume about the device it got. */
export interface GPUCapabilities {
  /** Storage buffers bindable in one compute stage. Caps fields per dispatch. */
  maxStorageBuffersPerShaderStage: number;
  /** Upper bound on workgroup_size(X). */
  maxComputeWorkgroupSizeX: number;
  /** Upper bound on the product of workgroup_size dimensions. */
  maxComputeInvocationsPerWorkgroup: number;
  /** Largest single buffer allocation, in bytes. */
  maxBufferSize: number;
  /** Largest range bindable as a storage buffer, in bytes. */
  maxStorageBufferBindingSize: number;
  /** Required alignment of a storage binding's offset, in bytes. Always 256 in practice. */
  minStorageBufferOffsetAlignment: number;
  /** Required alignment of a uniform binding's offset, in bytes. */
  minUniformBufferOffsetAlignment: number;
  /** Workgroups dispatchable per dimension; entities beyond this need a 2D dispatch. */
  maxComputeWorkgroupsPerDimension: number;
  /** True when the `subgroups` feature was granted. */
  subgroups: boolean;
  /** True when `timestamp-query` was granted (benchmarks only). */
  timestampQuery: boolean;
  /** True when `shader-f16` was granted. */
  shaderF16: boolean;
}

export interface GPUContext {
  device: GPUDevice;
  adapter: GPUAdapter;
  capabilities: GPUCapabilities;
}

/**
 * Strong references to every Dawn object whose collection would crash the
 * process. Never read for its value; its only job is to be reachable. See
 * hazard 2 above.
 */
const keepAlive: unknown[] = [];

let contextPromise: Promise<GPUContext | null> | null = null;
let resolved: GPUContext | null = null;
let warned = false;

/** One-time warning when no device could be obtained; callers fall back to CPU. */
function warnNoDevice(reason: string): void {
  if (warned) return;
  warned = true;
  // eslint-disable-next-line no-console
  console.warn(
    `[cozyecs/gpu] No WebGPU device (${reason}). Kernels will run on the CPU backend.`,
  );
}

function readCapabilities(adapter: GPUAdapter, device: GPUDevice): GPUCapabilities {
  const l = device.limits;
  return {
    maxStorageBuffersPerShaderStage: l.maxStorageBuffersPerShaderStage,
    maxComputeWorkgroupSizeX: l.maxComputeWorkgroupSizeX,
    maxComputeInvocationsPerWorkgroup: l.maxComputeInvocationsPerWorkgroup,
    maxBufferSize: Number(l.maxBufferSize),
    maxStorageBufferBindingSize: Number(l.maxStorageBufferBindingSize),
    minStorageBufferOffsetAlignment: l.minStorageBufferOffsetAlignment,
    minUniformBufferOffsetAlignment: l.minUniformBufferOffsetAlignment,
    maxComputeWorkgroupsPerDimension: l.maxComputeWorkgroupsPerDimension,
    subgroups: device.features.has('subgroups'),
    timestampQuery: device.features.has('timestamp-query'),
    shaderF16: device.features.has('shader-f16'),
  };
}

/** The `GPU` object for this host: `navigator.gpu` in browsers and Deno, or an
 *  instance created by the `webgpu` Node package when one has been installed
 *  via {@link setGPUProvider}. */
let provider: GPU | null = null;

/**
 * Supplies the `GPU` instance to use. Node has no `navigator.gpu`, so a host
 * that wants GPU kernels there calls this once with the object from the
 * `webgpu` package:
 *
 * ```js
 * import { create, globals } from 'webgpu';
 * Object.assign(globalThis, globals);
 * setGPUProvider(create([]));
 * ```
 *
 * The instance is pinned for the lifetime of the process (hazard 2).
 */
export function setGPUProvider(gpu: GPU): void {
  provider = gpu;
  keepAlive.push(gpu);
  contextPromise = null;
  resolved = null;
}

function resolveProvider(): GPU | null {
  if (provider) return provider;
  const nav = (globalThis as { navigator?: { gpu?: GPU } }).navigator;
  return nav?.gpu ?? null;
}

/**
 * Acquires (once) an adapter and a device with the adapter's maximum limits and
 * whatever optional features it offers. Resolves to `null` -- never throws --
 * when WebGPU is unavailable, so `kernelSystem` can fall back to the CPU
 * backend after a single warning.
 */
export function getGPUContext(): Promise<GPUContext | null> {
  if (contextPromise) return contextPromise;
  contextPromise = (async () => {
    const gpu = resolveProvider();
    if (!gpu) {
      warnNoDevice('no navigator.gpu; call setGPUProvider() on Node');
      return null;
    }
    let adapter: GPUAdapter | null;
    try {
      adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
    } catch (e) {
      warnNoDevice(`requestAdapter threw: ${(e as Error).message}`);
      return null;
    }
    if (!adapter) {
      warnNoDevice('requestAdapter returned null');
      return null;
    }

    // Ask for the adapter's maxima on the limits codegen budgets against. A
    // device is only granted limits it is asked for; the spec defaults are much
    // lower than any real desktop GPU supports.
    const wanted: Record<string, number> = {};
    for (const key of [
      'maxStorageBuffersPerShaderStage',
      'maxComputeWorkgroupSizeX',
      'maxComputeInvocationsPerWorkgroup',
      'maxComputeWorkgroupStorageSize',
      'maxStorageBufferBindingSize',
      'maxBufferSize',
      'maxUniformBufferBindingSize',
      'maxBindingsPerBindGroup',
      'maxComputeWorkgroupsPerDimension',
    ] as const) {
      const v = (adapter.limits as unknown as Record<string, number>)[key];
      if (typeof v === 'number' || typeof v === 'bigint') wanted[key] = Number(v);
    }
    const optional = (
      ['subgroups', 'subgroup-size-control', 'timestamp-query', 'shader-f16'] as const
    ).filter((f) => adapter!.features.has(f));

    let device: GPUDevice;
    try {
      device = await adapter.requestDevice({
        requiredFeatures: optional as unknown as GPUFeatureName[],
        requiredLimits: wanted,
      });
    } catch {
      // A driver may refuse a limit or feature we asked for. Retry bare rather
      // than losing the GPU entirely; codegen reads the granted limits anyway.
      try {
        device = await adapter.requestDevice();
      } catch (e) {
        warnNoDevice(`requestDevice threw: ${(e as Error).message}`);
        return null;
      }
    }
    keepAlive.push(gpu, adapter, device);
    resolved = { device, adapter, capabilities: readCapabilities(adapter, device) };
    return resolved;
  })();
  return contextPromise;
}

/**
 * True once a device has actually been acquired. False both before the first
 * {@link getGPUContext} call and after one that found no device, so a caller
 * can branch on it synchronously without re-triggering acquisition.
 */
export function hasGPUContext(): boolean {
  return resolved !== null;
}

/** The acquired context, or `null` if acquisition has not finished or failed.
 *  Synchronous companion to {@link getGPUContext} for per-frame hot paths. */
export function peekGPUContext(): GPUContext | null {
  return resolved;
}
