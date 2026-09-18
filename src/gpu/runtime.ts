/// <reference types="@webgpu/types" />
/**
 * Device, pipeline and buffer management; dispatch, readback and the CPU/GPU
 * target decision.
 *
 * OWNER: developer C. This is the only file allowed to touch a GPUDevice
 * besides ./device. It must not parse JavaScript and must not build WGSL: it
 * asks ./wgsl for a module and ./cpu for a loop and drives whichever one it
 * got.
 *
 * ---------------------------------------------------------------------------
 * THE THREE HAZARDS THIS FILE EXISTS TO CONTAIN
 *
 *  1. DAWN SEGFAULTS IF THE GPU INSTANCE IS COLLECTED. Measured 12/40 runs
 *     crashing with SIGSEGV in `InstanceBase::ProcessEvents` without a
 *     keepalive, 0/40 with one. ./device holds gpu/adapter/device in a module
 *     array that is still reachable at teardown. THIS FILE MUST NOT create its
 *     own GPU instance, and any benchmark or test outside ./device that touches
 *     WebGPU has to pin its objects the same way.
 *
 *  2. `queue.writeBuffer()` WITH A SharedArrayBuffer-BACKED VIEW IS A HARD
 *     CRASH (SIGSEGV, exit 139, 100% reproducible). `WorldOptions.shared` makes
 *     every archetype table a SharedArrayBuffer, so the upload path MUST copy
 *     out of the SAB into a plain ArrayBuffer staging view first. `Archetype`
 *     exposes `chunk.shared` -- branch on it, and keep one reusable staging
 *     ArrayBuffer per archetype rather than allocating per frame. Views over a
 *     normal ArrayBuffer, aligned or not, are fine.
 *
 *  3. DEFAULT DEVICE LIMITS ARE FAR BELOW THE ADAPTER'S. A device is granted
 *     only what it asked for. ./device asks for the adapter maxima and reports
 *     what was GRANTED in {@link GPUCapabilities}. Budget against that; never
 *     read `adapter.limits` here.
 *
 * ---------------------------------------------------------------------------
 * RESIDENCY IS THE POINT
 *
 * Re-uploading the data every frame erases the GPU win: `writeBuffer` of 4 MB
 * costs 0.29 ms, so four columns at 1M entities is ~1.2 ms against a 1.14 ms
 * CPU loop. Readback is nearly free (~0.05 ms over a bare dispatch) and the
 * encode itself is a flat ~0.013 ms regardless of entity count. So: upload
 * once, keep the data on the device, and only read back what the kernel wrote.
 *
 * Between dispatches the GPU owns every field the kernel writes. The runtime
 * re-uploads an archetype only when it can tell the CPU side changed:
 *   - first dispatch for that archetype;
 *   - `chunk.buffer` identity changed (the table grew: `Archetype._allocate`
 *     replaces the buffer and every view);
 *   - `chunk.count` changed since the last dispatch (rows were added or
 *     swap-removed, both of which are CPU-side writes);
 *   - the app called {@link KernelRuntime.markCpuDirty}.
 * Any other CPU write to a kernel-written field is LOST. That is the documented
 * bargain (docs/GPU.md, "Who owns the data").
 * ---------------------------------------------------------------------------
 */

import type { Archetype } from '../archetype';
import type { ComponentType } from '../component';
import type { Query } from '../query';
import { getGPUContext } from './device';
import type { GPUCapabilities, GPUContext } from './device';
import type { FieldAccess, KernelIR, StorageViewType, UniformLayout } from './ir';
import {
  FIELD_KINDS,
  KernelError,
  WORKGROUP_SIZE,
  gpuBlockers,
  maxRowsPerDispatch,
  storageViews,
  uniformLayout,
  validateIR,
} from './ir';
import { compileCPUKernel, estimateCPUNanosPerEntity } from './cpu';
import type { CompiledCPUKernel } from './cpu';
import { checkGPUSupport, generateWGSL, wgslCacheKey } from './wgsl';
import type { WGSLModule } from './wgsl';

/** Where a kernel runs. `'auto'` is resolved per dispatch (see {@link autoPrefersGPU}). */
export type KernelTarget = 'gpu' | 'cpu' | 'auto';

/**
 * The backend a kernel actually runs on. `'none'` means neither backend is
 * usable (the CPU loop failed to compile and no GPU pipeline exists): every
 * dispatch is a no-op. It is reported once through `console.warn` and is
 * always inspectable here -- never silent.
 */
export type KernelBackend = 'gpu' | 'cpu' | 'none';

/**
 * What happens to the kernel's writes.
 *
 *  - `'async'` (default): the readback is issued after the dispatch and applied
 *    to the archetype tables when the map resolves -- typically DURING THE NEXT
 *    FRAME. CPU code reading those fields sees last frame's values. Cheapest
 *    mode that still keeps the CPU in the loop (GPU wins above ~45k entities
 *    for a baseline-cost kernel). When the app dispatches faster than the GPU
 *    completes readbacks, frames are COALESCED, never lost: a later readback
 *    copies the full resident state, so `sync()` still sees every dispatch.
 *  - `'sync-frame'`: same dispatch, but the app is expected to
 *    `await handle.sync()` (or {@link flushKernels}) before it presents the
 *    frame. Adds a ~0.2 ms fixed readback round trip; GPU wins above ~700k entities
 *    for a baseline-cost kernel (heavier kernels much sooner).
 *  - `'none'`: never read back. The data stays GPU-resident and is meant to be
 *    consumed by rendering through {@link KernelRuntime.bufferFor}. CPU reads of
 *    those fields are stale from the first dispatch onwards.
 */
export type ReadbackMode = 'async' | 'sync-frame' | 'none';

/**
 * Break-even entity counts for a kernel of baseline CPU cost
 * ({@link BASELINE_CPU_NS_PER_ENTITY}), measured on an Apple M4 (Metal 3, Dawn).
 * Below these the CPU backend is faster including all overheads; above them
 * the GPU is. `auto` scales them by kernel cost (see
 * {@link autoPrefersGPU}) and the app can override them.
 *
 * Supporting numbers (Dawn on M4, full per-frame cost including upload-once
 * residency and readback): a simple integrate kernel breaks even at ~65k
 * entities with `readback: 'none'`, ~74k with `'async'` and above 1M with
 * `'sync-frame'`; the heavier gravity+bounce kernel at ~26k / ~32k / ~530k.
 * Normalized by each kernel's CPU cost, the GPU's fixed per-frame cost is
 * ~50-68 us fire-and-forget (a ~0.04 ms floor plus ~0.25 ns/entity), i.e.
 * ~45k entities at 1.14 ns/entity. The sync-frame default of 700k sits
 * between the two kernels' measurements.
 */
export interface AutoThresholds {
  /** Entities above which GPU wins for `'async'` / `'none'`. Default 45_000. */
  fireAndForget: number;
  /** Entities above which GPU wins for `'sync-frame'`. Default 700_000. */
  synchronous: number;
}

/** The process-wide defaults. Mutated only through {@link setAutoThresholds}. */
export const DEFAULT_AUTO_THRESHOLDS: Readonly<AutoThresholds> = Object.freeze({
  fireAndForget: 45_000,
  synchronous: 700_000,
});

/** Live defaults: {@link DEFAULT_AUTO_THRESHOLDS} until {@link setAutoThresholds} is called. */
let currentThresholds: AutoThresholds = {
  fireAndForget: DEFAULT_AUTO_THRESHOLDS.fireAndForget,
  synchronous: DEFAULT_AUTO_THRESHOLDS.synchronous,
};

/**
 * Overrides the defaults for kernels created afterwards. Provided because the
 * constants above are one machine's measurement: a slower integrated GPU or a
 * faster CPU moves them. An app that ships to many devices should measure once
 * at startup and call this.
 */
export function setAutoThresholds(thresholds: Partial<AutoThresholds>): void {
  const next: AutoThresholds = {
    fireAndForget: currentThresholds.fireAndForget,
    synchronous: currentThresholds.synchronous,
  };
  if (thresholds) {
    if (typeof thresholds.fireAndForget === 'number' && thresholds.fireAndForget >= 0) {
      next.fireAndForget = thresholds.fireAndForget;
    }
    if (typeof thresholds.synchronous === 'number' && thresholds.synchronous >= 0) {
      next.synchronous = thresholds.synchronous;
    }
  }
  currentThresholds = next;
}

/** The thresholds kernels created from now on will use (defaults, or the last {@link setAutoThresholds}). */
export function getAutoThresholds(): AutoThresholds {
  return { fireAndForget: currentThresholds.fireAndForget, synchronous: currentThresholds.synchronous };
}

/** @internal Resets the process-wide thresholds. Test seam. */
export function resetAutoThresholds(): void {
  currentThresholds = {
    fireAndForget: DEFAULT_AUTO_THRESHOLDS.fireAndForget,
    synchronous: DEFAULT_AUTO_THRESHOLDS.synchronous,
  };
}

/** Op count of the kernel the break-even numbers were measured with. */
const BASELINE_OPS = 12;
/**
 * CPU cost of that baseline kernel, ns per entity (M4: 1.140 ms at 1M
 * entities). Must agree with the calibration inside `estimateCPUNanosPerEntity`
 * (./cpu): the ratio `BASELINE_CPU_NS_PER_ENTITY / estimate` is what scales the
 * thresholds, so a kernel estimated at the baseline cost gets them unchanged.
 */
export const BASELINE_CPU_NS_PER_ENTITY = 1.14;
/**
 * The GPU's fixed per-frame cost the CPU has to "pay back" before the GPU wins,
 * in ns, as implied by the default thresholds at the baseline CPU cost:
 *  - fire-and-forget (`'async'` / `'none'`): 45k x 1.14 ns = ~51 us (encode,
 *    uniform upload, submit, dispatch latency, the async map issue; measured
 *    ~50-68 us per frame on Dawn/M4 once normalized by kernel cost);
 *  - `'sync-frame'`: 700k x 1.14 ns = ~800 us (dominated by the round trip:
 *    submit, wait for the queue, map, copy back; measured ~530k entities for
 *    the gravity kernel and >1M for the simple one).
 * The GPU's own per-entity cost is folded into these. The model is therefore
 *   `GPU wins  <=>  entities * cpuNsPerEntity >= GPU_FIXED_OVERHEAD_NS[mode]`,
 * which is exactly `entities >= threshold * (BASELINE_CPU_NS_PER_ENTITY / cpuNs)`.
 * Override the entity thresholds (not these) via `setAutoThresholds` or the
 * `thresholds` kernel option.
 */
export const GPU_FIXED_OVERHEAD_NS: Readonly<{ fireAndForget: number; synchronous: number }> = Object.freeze({
  fireAndForget: DEFAULT_AUTO_THRESHOLDS.fireAndForget * BASELINE_CPU_NS_PER_ENTITY,
  synchronous: DEFAULT_AUTO_THRESHOLDS.synchronous * BASELINE_CPU_NS_PER_ENTITY,
});
/** Hysteresis band: on the GPU we stay there down to `threshold / HYSTERESIS`. */
const HYSTERESIS = 1.25;

/**
 * Per-entity CPU cost used by `auto` for `ir`: `estimateCPUNanosPerEntity`
 * from ./cpu, or -- if that estimator is unavailable or returns nonsense --
 * the op-count proxy `BASELINE_CPU_NS_PER_ENTITY * opCount / BASELINE_OPS`.
 * Computed once per kernel, never per dispatch.
 */
export function cpuCostEstimate(ir: KernelIR): number {
  try {
    const ns = estimateCPUNanosPerEntity(ir);
    if (typeof ns === 'number' && Number.isFinite(ns) && ns > 0) return ns;
  } catch {
    /* fall through to the op-count proxy */
  }
  const ops = ir && ir.opCount > 0 ? ir.opCount : BASELINE_OPS;
  return (BASELINE_CPU_NS_PER_ENTITY * ops) / BASELINE_OPS;
}

/**
 * The `auto` decision for one dispatch.
 *
 * `entities` is the total row count across the query's chunks. The threshold is
 * scaled by estimated kernel cost, because a heavier kernel pays the GPU's fixed
 * overhead back sooner:
 * `threshold * clamp(BASELINE_CPU_NS_PER_ENTITY / cpuNsPerEntity, 1/8, 4)`
 * (see {@link GPU_FIXED_OVERHEAD_NS} for the model). `cpuNsPerEntity` defaults
 * to {@link cpuCostEstimate}`(ir)`; the runtime passes its cached value.
 *
 * Applies hysteresis: once on the GPU, a dispatch stays on the GPU until the
 * count drops below `threshold / 1.25`, and vice versa, so a count oscillating
 * around the threshold does not flip backends (and re-upload) every frame.
 */
export function autoPrefersGPU(
  ir: KernelIR,
  entities: number,
  readback: ReadbackMode,
  thresholds: AutoThresholds,
  currentlyGPU: boolean,
  cpuNsPerEntity?: number,
): boolean {
  const base = readback === 'sync-frame' ? thresholds.synchronous : thresholds.fireAndForget;
  const ns = cpuNsPerEntity !== undefined && cpuNsPerEntity > 0 ? cpuNsPerEntity : cpuCostEstimate(ir);
  let scale = BASELINE_CPU_NS_PER_ENTITY / ns;
  if (scale < 0.125) scale = 0.125;
  else if (scale > 4) scale = 4;
  const threshold = base * scale;
  return entities >= (currentlyGPU ? threshold / HYSTERESIS : threshold);
}

/** Per-kernel counters. Cheap to read; meant for a debug overlay and for tests. */
export interface KernelStats {
  /** Dispatches encoded since creation. */
  readonly dispatches: number;
  /**
   * Dispatches whose results are in the tables: covered by an APPLIED readback
   * (which copies the full resident state, so it covers every dispatch
   * submitted before it), or finished synchronously (CPU backend, `'none'`).
   */
  readonly completed: number;
  /** `dispatches - completed`. `0` means the tables are current. */
  readonly pending: number;
  /** Bytes uploaded since creation. A number that keeps growing means residency is broken. */
  readonly bytesUploaded: number;
  /** Bytes read back since creation. */
  readonly bytesReadBack: number;
  /**
   * Per-archetype readback groups DISCARDED: the archetype changed shape or
   * buffer (or the CPU marked it dirty) between encode and apply, or the map
   * failed. Nonzero means some GPU results were genuinely dropped.
   */
  readonly staleReadbacks: number;
  /**
   * Readbacks skipped or superseded under 'latest-wins' coalescing: a frame
   * found every staging buffer in flight (a later catch-up readback covers
   * it), or a readback arrived after a newer one had already been applied.
   * No data is lost; this only says the app dispatches faster than the GPU
   * completes readbacks.
   */
  readonly coalescedReadbacks: number;
  /** Which backend the last dispatch used. */
  readonly lastBackend: KernelBackend;
  /** Entities processed by the last dispatch. */
  readonly lastEntities: number;
}

/** @internal Mutable twin of {@link KernelStats}. */
interface MutableStats {
  dispatches: number;
  completed: number;
  pending: number;
  bytesUploaded: number;
  bytesReadBack: number;
  staleReadbacks: number;
  coalescedReadbacks: number;
  lastBackend: KernelBackend;
  lastEntities: number;
}

export interface KernelRuntimeOptions {
  readonly target: KernelTarget;
  readonly readback: ReadbackMode;
  /** Component ids whose enabled flag gates a row. See {@link KernelRuntime.create}. */
  readonly enableIds?: readonly number[];
  /** Overrides {@link DEFAULT_AUTO_THRESHOLDS} for this kernel only. */
  readonly thresholds?: Partial<AutoThresholds>;
  /** Overrides the workgroup size. Defaults to `WORKGROUP_SIZE` (256) from ./ir. */
  readonly workgroupSize?: number;
}

// ---------------------------------------------------------------------------
// Process-wide pipeline cache
// ---------------------------------------------------------------------------

interface CachedPipeline {
  readonly device: GPUDevice;
  readonly pipeline: GPUComputePipeline;
  readonly module: WGSLModule;
}

/** Keyed by `device tag + wgslCacheKey(ir, opts)`; see {@link pipelineCacheSize}. */
const pipelineCache = new Map<string, CachedPipeline>();

/** Devices get a small integer so the cache key cannot collide across devices. */
const deviceTags = new WeakMap<GPUDevice, number>();
let nextDeviceTag = 0;

function deviceTag(device: GPUDevice): number {
  let t = deviceTags.get(device);
  if (t === undefined) {
    t = nextDeviceTag++;
    deviceTags.set(device, t);
  }
  return t;
}

/**
 * @internal Test seam: the process-wide pipeline cache, so a test can assert
 * that two kernels with the same body share a pipeline and that adding an
 * archetype does not compile a second one.
 */
export function pipelineCacheSize(): number {
  return pipelineCache.size;
}

/** @internal Test seam: drops the pipeline cache. */
export function clearPipelineCache(): void {
  pipelineCache.clear();
}

// ---------------------------------------------------------------------------
// Per-archetype device state
// ---------------------------------------------------------------------------

/** One byte range of a table to copy back, in table-buffer coordinates. */
interface ByteRange {
  offset: number;
  size: number;
  /** Index into `_views`: which device-side copy of the table holds it. */
  view: number;
}

/** Fixed slot order for a table's per-view-type device copies. */
const VIEW_ORDER: readonly StorageViewType[] = ['f32', 'i32', 'u32'];

/**
 * The device-side image of ONE archetype table, SHARED BY EVERY KERNEL that
 * dispatches over that archetype on that device.
 *
 * Sharing is not an optimization, it is what makes docs/GPU.md section 3.5 true:
 * "a kernel sees the previous kernel's writes within the same world.update()".
 * With a private copy per kernel, two kernels in one frame would each start
 * from the last upload and the second readback would simply overwrite the
 * first -- measured: `x*2` then `x+1` over x=3 produced 4, not 7. Sharing also
 * means the table is uploaded once no matter how many kernels read it.
 *
 * `buffers` is indexed by {@link VIEW_ORDER}, not by a kernel's own view list,
 * because two kernels over the same archetype may need different view types.
 * Entries are created on demand and are byte-identical copies of each other.
 *
 * WHY SEVERAL COPIES AND NOT ONE BUFFER BOUND SEVERAL TIMES: WebGPU forbids
 * binding the same buffer range to two WRITABLE storage bindings in one bind
 * group, and ./wgsl declares every view `read_write`. Dawn rejects it outright:
 * "Writable storage buffer binding aliasing found between ... binding index 0
 * and ... binding index 1 with overlapping ranges". Distinct buffers is the fix
 * available on this side of the contract, and it costs nothing in the common
 * case: a kernel over f32 fields only has ONE view type. Fields of different
 * view types occupy disjoint bytes, so each copy is authoritative for its own
 * fields and a readback takes each field from the copy that owns it.
 */
interface SharedTable {
  readonly arch: Archetype;
  readonly device: GPUDevice;
  /** Per-view-type copies, indexed by {@link VIEW_ORDER}; null until needed. */
  buffers: (GPUBuffer | null)[];
  /** Bytes allocated for every non-null entry of `buffers`. */
  size: number;
  /** `chunk.buffer` identity at the last upload; a change means the table was reallocated. */
  srcBuffer: ArrayBufferLike | null;
  /** `chunk.count` at the last upload. */
  lastCount: number;
  /** Forces the next dispatch to re-upload. */
  dirty: boolean;
  /**
   * Bumped on every upload and on `markCpuDirty`. A readback carrying an older
   * generation is discarded: the CPU has written since it was issued.
   */
  generation: number;
  /** Bumped whenever a buffer is created or replaced, invalidating bind groups. */
  epoch: number;
  /** Plain-ArrayBuffer staging copy, only for `chunk.shared` tables (hazard 2). */
  sabStaging: Uint8Array | null;
  /** Kernels holding this table. At zero the buffers are destroyed. */
  refs: number;
}

/** One kernel's per-archetype bookkeeping around a {@link SharedTable}. */
interface ArchState {
  readonly arch: Archetype;
  readonly table: SharedTable;
  /**
   * First slot of this archetype's reservation in the kernel's ONE uniform
   * buffer, or -1 before it has one. Reservations are permanent, so the bind
   * groups below stay valid frame after frame.
   */
  slotBase: number;
  /** Slots reserved: one per dispatch pass. */
  slots: number;
  /** Bind groups, one per reserved slot. */
  bindGroups: GPUBindGroup[];
  /** `table.epoch` the bind groups were built against. */
  bindEpoch: number;
}

/** Device tables, keyed by archetype then by device tag. */
const tableCache = new WeakMap<Archetype, Map<number, SharedTable>>();

/** @internal Test seam: how many archetype tables are resident on the device. */
export function residentTableCount(arch: Archetype): number {
  const byDevice = tableCache.get(arch);
  return byDevice ? byDevice.size : 0;
}

/** One chunk's share of a batched readback. */
interface ReadbackGroup {
  readonly state: ArchState;
  /** Offset of this chunk's bytes inside the staging buffer. */
  readonly start: number;
  readonly bytes: number;
  readonly ranges: readonly ByteRange[];
  /** Stamps that must still hold when the map resolves, or the data is stale. */
  readonly srcBuffer: ArrayBufferLike;
  readonly count: number;
  readonly generation: number;
}

/**
 * One dispatch's readback, in flight. ONE staging buffer and ONE `mapAsync` per
 * dispatch, not per chunk: a `mapAsync` call costs ~4 us of encode time, which
 * at 32 chunks was more than the rest of the frame's GPU work.
 */
interface PendingReadback {
  readonly buffer: GPUBuffer;
  /** Bytes actually copied (<= the buffer's size). */
  readonly bytes: number;
  readonly groups: readonly ReadbackGroup[];
  /**
   * `stats.dispatches` when this readback was encoded. The copy is queued after
   * every earlier submit, so applying it brings the tables up to date with all
   * of those dispatches (latest-wins coalescing).
   */
  readonly coveredUpTo: number;
}

const UNIFORM_SLOT_ALIGN = 256;

function alignUp(value: number, align: number): number {
  return (value + align - 1) - ((value + align - 1) % align);
}

// ---------------------------------------------------------------------------
// Runtime registry (for flushKernels)
// ---------------------------------------------------------------------------

const tracked = new WeakMap<object, Map<string, Set<KernelRuntime>>>();

/**
 * @internal Registers a runtime so {@link flushKernels} can find it. Called by
 * index.ts at registration and again with `false` on removal.
 */
export function trackRuntime(world: unknown, group: string, runtime: KernelRuntime, active: boolean): void {
  if (!world || typeof world !== 'object') return;
  const key = world as object;
  let groups = tracked.get(key);
  if (!groups) {
    if (!active) return;
    groups = new Map();
    tracked.set(key, groups);
  }
  let set = groups.get(group);
  if (active) {
    if (!set) {
      set = new Set();
      groups.set(group, set);
    }
    set.add(runtime);
    runtime._trackedIn(key, group);
  } else if (set) {
    set.delete(runtime);
    if (set.size === 0) groups.delete(group);
  }
}

/**
 * Awaits every kernel of `world` (optionally only those in `group`). The
 * companion to `world.update()` for apps using `'sync-frame'`:
 *
 * ```js
 * world.update(dt);          // encodes, never blocks
 * await flushKernels(world); // only when you need this frame's results now
 * ```
 */
export function flushKernels(world: unknown, group?: string): Promise<void> {
  if (!world || typeof world !== 'object') return Promise.resolve();
  const groups = tracked.get(world as object);
  if (!groups) return Promise.resolve();
  const waits: Promise<void>[] = [];
  if (group === undefined) {
    for (const set of groups.values()) for (const runtime of set) waits.push(runtime.sync());
  } else {
    const set = groups.get(group);
    if (set) for (const runtime of set) waits.push(runtime.sync());
  }
  if (waits.length === 0) return Promise.resolve();
  return Promise.all(waits).then(noop);
}

function noop(): void {
  /* nothing */
}

function warn(message: string): void {
  // eslint-disable-next-line no-console
  console.warn(`[cozyecs/gpu] ${message}`);
}

/**
 * Owns everything one kernel needs at run time. One instance per
 * `kernelSystem` call; `index.ts` wraps it in a `SystemHandle`.
 *
 * INVARIANTS
 *  - `dispatch()` NEVER blocks and never awaits. It encodes and submits. This
 *    is what lets a kernel be an ordinary system inside the synchronous
 *    `world.update()`.
 *  - `dispatch()` never throws for a device-side reason. Anything that would
 *    (device lost, unsupported field, pipeline creation failure) degrades to
 *    the CPU backend after one `console.warn`, permanently.
 *  - The CPU backend is always constructed, even when the GPU one is chosen, so
 *    the degradation path needs no allocation at the moment it is taken.
 *  - The device-side table of an archetype is SHARED by every kernel over that
 *    archetype ({@link SharedTable}, refcounted): it is uploaded once, and a
 *    kernel sees the previous kernel's writes within the same frame, which is
 *    what docs/GPU.md section 3.5 promises. Per kernel and archetype the
 *    runtime then keeps only a uniform-slot reservation and its bind groups.
 *  - Per kernel there is ONE uniform buffer for every (archetype, pass) slot
 *    and ONE readback staging buffer per dispatch, because `queue.writeBuffer`
 *    and `mapAsync` each cost ~5 us of encode time per CALL: batching them took
 *    a 32-chunk frame from 0.33 ms to 0.20 ms.
 *  - Pipelines are cached PROCESS-WIDE by `wgslCacheKey(ir)`, so two kernels
 *    with the same body share one GPUComputePipeline, and an archetype added
 *    later costs a bind group, not a compile.
 */
export class KernelRuntime {
  /** The IR this runtime was built for. */
  readonly ir: KernelIR;
  /** Components in IR order, for `forEachChunk`. */
  readonly components: readonly ComponentType[];
  readonly readback: ReadbackMode;
  /** Which backend the next dispatch will use. `'auto'` flips this per dispatch. */
  backend: KernelBackend;
  readonly stats: KernelStats;
  /** The device context, or null when running on the CPU backend. */
  readonly context: GPUContext | null;
  /**
   * Live user-uniform values, index-aligned with `ir.uniforms`. `index.ts`
   * installs this array right after {@link KernelRuntime.create} and mutates
   * elements in place from `handle.setUniform`; every dispatch reads it (into
   * the uniform buffer on the GPU, into the compiled loop's state on the CPU).
   * Never replace the array -- mutate it.
   */
  uniformValues: Float64Array;

  /** @internal resolved target ('auto' stays 'auto'; a degradation pins 'cpu'). */
  private _target: KernelTarget;
  private readonly _thresholds: AutoThresholds;
  private readonly _stats: MutableStats;
  private readonly _cpu: CompiledCPUKernel | null;
  private _cpuBrokenWarned: boolean;
  /** Why the CPU loop failed to compile, for the 'none' warning. */
  private _cpuError: string | null = null;
  /** `cpuCostEstimate(ir)`, computed once for the `auto` decision. */
  private readonly _cpuNs: number;
  /** GPU side, all null on the CPU backend. */
  private _pipeline: GPUComputePipeline | null;
  private _module: WGSLModule | null;
  private _layout: UniformLayout | null;
  private _views: readonly StorageViewType[];
  private _fields: readonly FieldAccess[];
  private _writeFields: readonly FieldAccess[];
  private _workgroupSize: number;
  private _maxRows: number;
  /**
   * ONE uniform buffer for the whole kernel, sliced into 256-byte slots -- one
   * per (archetype, dispatch pass). `queue.writeBuffer` costs ~6 us per call
   * regardless of size, so a per-chunk uniform buffer made the encode scale
   * with chunk count (32 chunks: 0.20 ms). Filling a host-side image of every
   * slot and uploading the touched span in ONE call makes it flat again.
   */
  private _uniBuffer: GPUBuffer | null;
  /** Slots the buffer (and the host image) can hold. */
  private _uniCapacity: number;
  /** Next unreserved slot. Reservations are permanent so bind groups stay valid. */
  private _uniNext: number;
  private _uniHost: Uint8Array | null;
  private _uniF32: Float32Array | null;
  private _uniU32: Uint32Array | null;
  private _uniformStride: number;
  /** Slot span written during the current dispatch, for the single writeBuffer. */
  private _slotLo: number;
  private _slotHi: number;
  /** Readback staging buffers, shared by every archetype of this kernel. */
  private readonly _readPool: GPUBuffer[];
  private _readBusy: number;
  private readonly _states: Map<number, ArchState>;
  /** `sync()` callers: each resolves once `completed >= target`. */
  private readonly _waiters: { target: number; resolve: () => void }[];
  /**
   * Set when a frame's readback was skipped (no free staging buffer) or a map
   * failed: the next readback to finish immediately issues a readback-only
   * catch-up copy of every live table, which covers the skipped dispatches.
   */
  private _catchUpNeeded: boolean;
  /** Consecutive map failures; bounds the retry so a dying device cannot loop. */
  private _mapFailures: number;
  private readonly _trackedAt: { world: object; group: string }[];
  private _destroyed: boolean;
  /** True once {@link dispatch} has run; until then `auto` reports a prediction. */
  private _dispatched: boolean;
  private _degraded: boolean;
  /** Scratch reused per dispatch so the hot path allocates nothing. */
  private readonly _touched: { state: ArchState; count: number }[];
  /** How many entries of `_touched` the current dispatch filled. */
  private _touchedCount: number;
  private _rangeScratch: ByteRange[];
  /** Scratch list of (state, rows) handed to `_encodeReadback`. */
  private readonly _readList: { state: ArchState; count: number }[];
  private _poolWarned: boolean;

  /** @internal Use {@link KernelRuntime.create}. */
  private constructor(
    ir: KernelIR,
    components: readonly ComponentType[],
    options: KernelRuntimeOptions,
    context: GPUContext | null,
    cpu: CompiledCPUKernel | null,
  ) {
    this.ir = ir;
    this.components = components;
    this.readback = options.readback;
    this.context = context;
    this.backend = context ? 'gpu' : 'cpu';
    this._target = options.target;
    this._thresholds = {
      fireAndForget:
        options.thresholds && typeof options.thresholds.fireAndForget === 'number'
          ? options.thresholds.fireAndForget
          : currentThresholds.fireAndForget,
      synchronous:
        options.thresholds && typeof options.thresholds.synchronous === 'number'
          ? options.thresholds.synchronous
          : currentThresholds.synchronous,
    };
    this._stats = {
      dispatches: 0,
      completed: 0,
      pending: 0,
      bytesUploaded: 0,
      bytesReadBack: 0,
      staleReadbacks: 0,
      coalescedReadbacks: 0,
      lastBackend: context ? 'gpu' : 'cpu',
      lastEntities: 0,
    };
    this.stats = this._stats as KernelStats;
    this._cpu = cpu;
    this._cpuBrokenWarned = false;
    this._cpuNs = cpuCostEstimate(ir);
    this._pipeline = null;
    this._module = null;
    this._layout = null;
    this._views = [];
    this._fields = [];
    this._writeFields = [];
    this._workgroupSize = options.workgroupSize || WORKGROUP_SIZE;
    this._maxRows = maxRowsPerDispatch(
      context ? context.capabilities.maxComputeWorkgroupsPerDimension : 65535,
    );
    this._uniBuffer = null;
    this._uniCapacity = 0;
    this._uniNext = 0;
    this._uniHost = null;
    this._uniF32 = null;
    this._uniU32 = null;
    this._uniformStride = UNIFORM_SLOT_ALIGN;
    this._slotLo = 0;
    this._slotHi = -1;
    this._readPool = [];
    this._readBusy = 0;
    this._states = new Map();
    this._waiters = [];
    this._catchUpNeeded = false;
    this._mapFailures = 0;
    this._trackedAt = [];
    this._destroyed = false;
    this._dispatched = false;
    this._degraded = false;
    this._touched = [];
    this._touchedCount = 0;
    this._rangeScratch = [];
    this._readList = [];
    this._poolWarned = false;
    const values = new Float64Array(ir.uniforms.length);
    for (let i = 0; i < ir.uniforms.length; i++) values[i] = ir.uniforms[i].initial;
    this.uniformValues = values;
  }

  /**
   * Builds a runtime for `ir`.
   *
   * Resolution order, none of which throws for a capability reason:
   *  1. `target: 'cpu'` -> CPU backend, no device requested.
   *  2. `getGPUContext()` resolves null -> CPU backend (./device already warned
   *     once).
   *  3. `checkGPUSupport(ir, caps)` fails -> CPU backend, one `console.warn`
   *     naming the blocking field or limit.
   *  4. `enableIds` is non-empty -> CPU backend, one `console.warn`. v1 cannot
   *     skip disabled rows on the GPU: the enabled flags are `Uint8Array`
   *     columns and the GPU backend addresses 4-byte views only. The CPU
   *     backend honours them exactly.
   *  5. otherwise -> GPU backend (or `auto`, decided per dispatch).
   *
   * If the CPU loop fails to compile (a library bug, reported with one
   * `console.warn`), a working GPU pipeline becomes the only backend and `auto`
   * is pinned to it. If there is no GPU pipeline either -- or `target` was
   * `'cpu'` -- the runtime reports `backend === 'none'`, warns once more saying
   * dispatches are no-ops, and `dispatch()` does nothing. Never silent.
   *
   * @throws KernelError only for E_INVALID_IR (a bug, not a capability).
   */
  static async create(
    ir: KernelIR,
    components: readonly ComponentType[],
    options: KernelRuntimeOptions,
  ): Promise<KernelRuntime> {
    validateIR(ir);

    // Kernel arithmetic is f32 on BOTH backends (that is what makes them agree),
    // so an f64 field keeps its storage but loses precision on every write.
    // Say so once per kernel, whatever the target.
    const f64 = safeBlockers(ir).filter((b) => b.kind === 'f64');
    if (f64.length > 0) {
      warn(
        `kernel "${ir.name}" touches f64 field(s) ${f64.map((b) => `${b.component}.${b.field}`).join(', ')}; ` +
          'kernels compute in f32, so values written there are rounded to f32 precision ' +
          '(and f64 fields keep the kernel on the CPU backend).',
      );
    }

    // The CPU backend is ALWAYS built: degradation must never need to allocate
    // or compile at the moment it is taken.
    let cpu: CompiledCPUKernel | null = null;
    let cpuError: string | null = null;
    try {
      cpu = compileCPUKernel(ir, components, { enableIds: options.enableIds });
    } catch (e) {
      if (e instanceof KernelError && e.code === 'E_INVALID_IR') throw e;
      // A broken CPU backend is a library bug, not a capability problem. Keep
      // the GPU path usable if there is one; otherwise report backend 'none'.
      // Either way exactly one warning, emitted below once the outcome is known.
      cpuError = (e as Error).message;
    }

    const wantsGPU = options.target !== 'cpu';
    let context: GPUContext | null = null;
    if (wantsGPU) {
      const enableIds = options.enableIds;
      if (enableIds && enableIds.length > 0) {
        warn(
          `kernel "${ir.name}" queries enableable components, which the v1 GPU backend cannot test; ` +
            `running on the CPU backend (disabled rows are skipped exactly).`,
        );
      } else {
        context = await getGPUContext();
        if (context) {
          const support = safeCheckSupport(ir, context.capabilities, options.workgroupSize);
          if (!support.ok) {
            warn(
              `kernel "${ir.name}" cannot run on this device (${support.reasons
                .map((r) => `${r.code}: ${r.message}`)
                .join('; ')}); running on the CPU backend.`,
            );
            context = null;
          }
        }
      }
    }

    const runtime = new KernelRuntime(ir, components, options, context, cpu);
    if (context) {
      const ok = await runtime._initGPU(context, options);
      if (!ok) runtime._degradeToCPU(null);
    }
    runtime._cpuError = cpuError;
    if (runtime._pipeline === null) {
      runtime._target = 'cpu';
      runtime._setBackend(cpu ? 'cpu' : 'none');
    } else if (!cpu) {
      // GPU only: `auto` has nothing to fall back to, so pin the GPU.
      runtime._target = 'gpu';
      warn(
        `the CPU backend for "${ir.name}" could not be compiled (${cpuError}); ` +
          `running on the GPU only, with no fallback if the device is lost.`,
      );
    }
    return runtime;
  }

  /** Builds (or reuses) the pipeline. Returns false to fall back to the CPU. */
  private async _initGPU(context: GPUContext, options: KernelRuntimeOptions): Promise<boolean> {
    const device = context.device;
    const wgslOpts = options.workgroupSize ? { workgroupSize: options.workgroupSize } : undefined;
    let key: string;
    try {
      key = `${deviceTag(device)}:${wgslCacheKey(this.ir, wgslOpts)}`;
    } catch (e) {
      warn(`WGSL cache key for "${this.ir.name}" failed (${(e as Error).message}); using the CPU backend.`);
      return false;
    }

    const cached = pipelineCache.get(key);
    if (cached && cached.device === device) {
      this._pipeline = cached.pipeline;
      this._module = cached.module;
    } else {
      let module: WGSLModule;
      try {
        module = generateWGSL(this.ir, wgslOpts);
      } catch (e) {
        warn(`WGSL generation for "${this.ir.name}" failed (${(e as Error).message}); using the CPU backend.`);
        return false;
      }
      try {
        const shader = device.createShaderModule({ code: module.code, label: `cozy:${this.ir.name}` });
        const pipeline = await device.createComputePipelineAsync({
          label: `cozy:${this.ir.name}`,
          layout: 'auto',
          compute: { module: shader, entryPoint: module.entryPoint },
        });
        this._pipeline = pipeline;
        this._module = module;
        pipelineCache.set(key, { device, pipeline, module });
      } catch (e) {
        warn(
          `pipeline creation for "${this.ir.name}" failed (${(e as Error).message}); using the CPU backend.`,
        );
        return false;
      }
    }

    const module = this._module as WGSLModule;
    this._workgroupSize = module.workgroupSize;
    this._views = module.views && module.views.length ? module.views : storageViews(this.ir);
    this._layout = module.layout || uniformLayout(this.ir);
    this._fields = this._layout.fields.map((f) => f.access);
    this._writeFields = dedupeAccesses(this.ir.writes);
    this._maxRows = maxRowsPerDispatch(context.capabilities.maxComputeWorkgroupsPerDimension);
    this._uniformStride = alignUp(this._layout.size, UNIFORM_SLOT_ALIGN);

    // A lost device must not take down world.update(): degrade on the spot.
    device.lost.then((info: GPUDeviceLostInfo) => {
      if (this._destroyed) return;
      this._degradeToCPU(`device lost: ${info && info.message ? info.message : 'unknown reason'}`);
    }, noop);
    return true;
  }

  /** @internal Remembers where {@link trackRuntime} registered us, so destroy can unregister. */
  _trackedIn(world: object, group: string): void {
    for (const t of this._trackedAt) if (t.world === world && t.group === group) return;
    this._trackedAt.push({ world, group });
  }

  /**
   * Runs one frame of the kernel over `query`. Synchronous and non-blocking.
   *
   * GPU path, per matching non-empty chunk:
   *   1. ensure the table buffer exists and is large enough (`chunk.capacity *
   *      rowBytes`, rounded up); (re)upload when the residency rules at the top
   *      of this file say the CPU side changed -- copying out of a
   *      SharedArrayBuffer first (hazard 2);
   *   2. write the uniform buffer: `dt`, `count`, `base`, `countOther`, the
   *      field base offsets (`view.byteOffset / bytesPerElement`, re-read every
   *      frame because growth moves them) and the user uniforms, using the byte
   *      offsets from `uniformLayout(ir)`;
   *   3. encode the compute pass, `ceil(count / workgroupSize)` workgroups,
   *      splitting into several passes with `base` advanced when that exceeds
   *      `caps.maxComputeWorkgroupsPerDimension` (16.7M entities per pass at
   *      workgroup 256);
   *   4. for readback modes other than `'none'`, copy only the byte ranges of
   *      the WRITTEN fields into a staging buffer and start `mapAsync`.
   * One `queue.submit`, one `queue.writeBuffer` for all uniform slots and one
   * `mapAsync` per dispatch -- not per chunk.
   *
   * CPU path: `query.forEachChunk(components, compiled.fn)` after writing
   * `dt` and the uniform values into the compiled kernel's state object.
   */
  /**
   * Under `target: 'auto'`, before the first dispatch: sets {@link backend} (and
   * `stats.lastBackend`) to what the first dispatch would choose for `query`'s
   * current entity count, instead of reporting `'gpu'` just because a pipeline
   * exists. Also means the first dispatch starts from the right side of the
   * hysteresis band (a kernel that starts on the CPU must reach the full
   * threshold to move). A no-op after the first dispatch, for pinned targets,
   * and when the GPU path is unavailable.
   */
  predictBackend(query: Query): KernelBackend {
    if (this._dispatched || this._destroyed || this._target !== 'auto' || this._pipeline === null || this.backend === 'none') {
      return this.backend;
    }
    const chunks = query.chunks;
    let entities = 0;
    for (let i = 0; i < chunks.length; i++) entities += chunks[i].count;
    const backend: KernelBackend = autoPrefersGPU(this.ir, entities, this.readback, this._thresholds, false, this._cpuNs)
      ? 'gpu'
      : 'cpu';
    this.backend = backend;
    this._stats.lastBackend = backend;
    return backend;
  }

  dispatch(query: Query, dt: number): void {
    if (this._destroyed) return;
    if (!this._dispatched) {
      this.predictBackend(query);
      this._dispatched = true;
    }
    const chunks = query.chunks;
    let entities = 0;
    for (let i = 0; i < chunks.length; i++) entities += chunks[i].count;

    if (this.backend === 'none') return;
    let useGPU = this.backend === 'gpu';
    if (this._target === 'cpu' || this._pipeline === null) useGPU = false;
    else if (this._target === 'gpu') useGPU = true;
    else useGPU = autoPrefersGPU(this.ir, entities, this.readback, this._thresholds, this.backend === 'gpu', this._cpuNs);
    this.backend = useGPU ? 'gpu' : 'cpu';

    const stats = this._stats;
    stats.dispatches++;
    stats.lastBackend = this.backend;
    stats.lastEntities = entities;

    if (!useGPU) {
      this._dispatchCPU(query, dt);
      this._settle();
      return;
    }

    try {
      this._dispatchGPU(query, dt, entities);
    } catch (e) {
      // Nothing device-side may escape into world.update().
      this._degradeToCPU(`dispatch failed: ${(e as Error).message}`);
      this._dispatchCPU(query, dt);
      stats.lastBackend = this.backend;
      this._settle();
    }
  }

  private _dispatchCPU(query: Query, dt: number): void {
    const cpu = this._cpu;
    if (!cpu) {
      // Only reachable if the GPU path threw mid-dispatch and degraded; the
      // degrade already switched us to 'none' and warned.
      this._setBackend('none');
      return;
    }
    const state = cpu.state;
    state.dt = dt;
    const u = state.u;
    const values = this.uniformValues;
    const n = u.length < values.length ? u.length : values.length;
    for (let i = 0; i < n; i++) u[i] = values[i];
    // Pairs are formed within a chunk, so the compiled loop uses its own
    // `count` for both sides; countOther carries the query total for kernels
    // that want it as a scale factor.
    state.countOther = this._stats.lastEntities;
    query.forEachChunk(cpu.components as never, cpu.fn as never);
  }

  private _dispatchGPU(query: Query, dt: number, entities: number): void {
    const stats = this._stats;
    if (entities === 0) {
      this._settle();
      return;
    }

    const context = this.context as GPUContext;
    const device = context.device;
    const pipeline = this._pipeline as GPUComputePipeline;
    const layout = this._layout as UniformLayout;
    const chunks = query.chunks;
    const touched = this._touched;
    let touchedCount = 0;
    this._slotLo = Number.MAX_SAFE_INTEGER;
    this._slotHi = -1;

    // Pass 1: device-side storage and the host image of every uniform slot.
    // Both can reallocate, so nothing is encoded until they have settled.
    const wg = this._workgroupSize;
    const maxRows = this._maxRows;
    for (let c = 0; c < chunks.length; c++) {
      const chunk = chunks[c];
      const count = chunk.count;
      if (count === 0) continue;
      const state = this._stateFor(chunk);
      this._ensureTable(state, chunk);
      this._upload(state, chunk, count);
      const passes = count > maxRows ? Math.ceil(count / maxRows) : 1;
      this._reserveSlots(state, passes, device);
      for (let p = 0; p < passes; p++) {
        this._writeUniform(state, chunk, dt, count, p * maxRows, state.slotBase + p, layout);
      }
      if (touchedCount < touched.length) {
        touched[touchedCount].state = state;
        touched[touchedCount].count = count;
      } else {
        touched.push({ state, count });
      }
      touchedCount++;
    }
    this._touchedCount = touchedCount;
    if (touchedCount === 0) {
      this._settle();
      return;
    }

    // One uniform upload for the whole dispatch. Queue operations run in issue
    // order, so this lands before the submit below.
    if (this._slotHi >= this._slotLo) {
      const stride = this._uniformStride;
      const from = this._slotLo * stride;
      const size = (this._slotHi - this._slotLo + 1) * stride;
      device.queue.writeBuffer(this._uniBuffer as GPUBuffer, from, this._uniHost as Uint8Array<ArrayBuffer>, from, size);
    }

    // Pass 2: encode. Bind groups exist by now and cannot move under us.
    const encoder = device.createCommandEncoder({ label: `cozy:${this.ir.name}` });
    const pass = encoder.beginComputePass({ label: `cozy:${this.ir.name}` });
    pass.setPipeline(pipeline);
    for (let i = 0; i < touchedCount; i++) {
      const state = touched[i].state;
      const count = touched[i].count;
      const passes = count > maxRows ? Math.ceil(count / maxRows) : 1;
      this._ensureBindGroups(state, device, pipeline);
      for (let p = 0; p < passes; p++) {
        const base = p * maxRows;
        const rows = count - base < maxRows ? count - base : maxRows;
        pass.setBindGroup(0, state.bindGroups[p]);
        pass.dispatchWorkgroups(Math.ceil(rows / wg));
      }
    }
    pass.end();

    if (this.readback === 'none') {
      device.queue.submit([encoder.finish()]);
      this._settle();
      return;
    }
    // Collect the touched states into the scratch list `_encodeReadback` reads.
    const list = this._readList;
    list.length = 0;
    for (let i = 0; i < touchedCount; i++) list.push(touched[i]);
    const pending = this._encodeReadback(device, encoder, list, stats.dispatches);
    device.queue.submit([encoder.finish()]);
    if (pending === SKIPPED) {
      // Every staging buffer is in flight. Latest-wins: the table stays resident
      // on the device, so the catch-up readback issued when the next one
      // finishes copies this frame's results too. This dispatch stays pending.
      this._catchUpNeeded = true;
      stats.coalescedReadbacks++;
      this._warnCoalescing();
    } else if (pending) {
      this._startReadback(pending);
    }
    this._settle();
  }

  private _warnCoalescing(): void {
    if (this._poolWarned) return;
    this._poolWarned = true;
    warn(
      `kernel "${this.ir.name}" has ${MAX_READBACKS_IN_FLIGHT} readbacks in flight; coalescing frames: ` +
        `this frame's results arrive with a later readback (none are lost; stats.coalescedReadbacks counts them). ` +
        `The app is dispatching faster than the GPU completes readbacks.`,
    );
  }

  /**
   * Copies the written byte ranges of every listed chunk into ONE staging
   * buffer and returns the readback to start once `encoder` is submitted.
   * Returns null when there is nothing to read back, or {@link SKIPPED} when
   * no staging buffer is available right now.
   *
   * Stamps (buffer identity, count, generation) are captured NOW: they are
   * what the tables must still look like when the map resolves.
   */
  private _encodeReadback(
    device: GPUDevice,
    encoder: GPUCommandEncoder,
    list: readonly { state: ArchState; count: number }[],
    coveredUpTo: number,
  ): PendingReadback | null | typeof SKIPPED {
    const groups: ReadbackGroup[] = [];
    let total = 0;
    for (let i = 0; i < list.length; i++) {
      const state = list[i].state;
      const ranges = this._writtenRanges(state.arch, list[i].count);
      if (ranges.length === 0) continue;
      let bytes = 0;
      for (let r = 0; r < ranges.length; r++) bytes += ranges[r].size;
      groups.push({
        state,
        start: total,
        bytes,
        ranges: ranges.slice(),
        srcBuffer: state.arch.buffer,
        count: list[i].count,
        generation: state.table.generation,
      });
      total += bytes;
    }
    if (total === 0) return null;
    const dst = this._takeReadBuffer(device, total);
    if (!dst) return SKIPPED;
    for (let g = 0; g < groups.length; g++) {
      const group = groups[g];
      let at = group.start;
      for (let r = 0; r < group.ranges.length; r++) {
        const range = group.ranges[r];
        encoder.copyBufferToBuffer(this._bufferOf(group.state.table, range.view), range.offset, dst, at, range.size);
        at += range.size;
      }
    }
    return { buffer: dst, bytes: total, groups, coveredUpTo };
  }

  /**
   * Issues a readback-only submit covering every dispatch so far: a copy of
   * the written ranges of every live table whose device image still matches
   * the CPU table's layout (same buffer, same row count; a table that does
   * not match is re-uploaded by the next dispatch anyway). Called when a
   * readback finishes and `_catchUpNeeded` is set, so a staging buffer has
   * just been freed. Clears the flag unless no staging buffer could be had;
   * then the next finishing readback (or {@link _settle}) tries again.
   */
  private _submitCatchUp(): void {
    if (!this._catchUpNeeded || this._destroyed || this._degraded || this._pipeline === null) return;
    const context = this.context;
    if (!context) return;
    const device = context.device;
    const list = this._readList;
    list.length = 0;
    for (const state of this._states.values()) {
      const arch = state.arch;
      const table = state.table;
      if (table.srcBuffer !== arch.buffer || table.lastCount !== arch.count || arch.count === 0) continue;
      let hasBuffers = true;
      for (let v = 0; v < this._views.length; v++) if (!this._bufferOf(table, v)) hasBuffers = false;
      if (!hasBuffers) continue;
      list.push({ state, count: arch.count });
    }
    let pending: PendingReadback | null | typeof SKIPPED = null;
    let encoder: GPUCommandEncoder | null = null;
    try {
      encoder = device.createCommandEncoder({ label: `cozy:${this.ir.name}:catch-up` });
      pending = this._encodeReadback(device, encoder, list, this._stats.dispatches);
      if (pending === SKIPPED) return; // retried by the next finishing readback / _settle
      if (pending) device.queue.submit([encoder.finish()]);
    } catch {
      // A failing device: give the staging buffer back and let degrade/destroy
      // settle the waiters. Never throw out of a map callback.
      if (pending && pending !== SKIPPED) this._releaseReadBuffer(pending.buffer);
      this._catchUpNeeded = false;
      return;
    }
    this._catchUpNeeded = false;
    if (pending) this._startReadback(pending);
  }

  private _startReadback(p: PendingReadback): void {
    p.buffer.mapAsync(GPUMapMode.READ, 0, p.bytes).then(
      () => {
        this._applyReadback(p);
      },
      () => {
        // Device lost, destroyed buffer, or a cancelled map. On a live runtime
        // this readback's data is gone but the device still has it: ask for a
        // catch-up, at most twice in a row so a dying device cannot loop.
        if (!this._destroyed && !this._degraded) {
          this._stats.staleReadbacks += p.groups.length;
          if (this._mapFailures++ < MAX_MAP_RETRIES) this._catchUpNeeded = true;
          else this._markCovered(p.coveredUpTo);
        }
        this._finishReadback(p, false);
      },
    );
  }

  private _applyReadback(p: PendingReadback): void {
    this._mapFailures = 0;
    if (this._destroyed || this._degraded) {
      // The CPU backend owns the tables now; old GPU data must not overwrite it.
      this._finishReadback(p, true);
      return;
    }
    if (p.coveredUpTo <= this._stats.completed) {
      // A newer readback was applied first; this one would roll the tables back.
      this._stats.coalescedReadbacks++;
      this._finishReadback(p, true);
      return;
    }
    try {
      const mapped = new Uint8Array(p.buffer.getMappedRange(0, p.bytes));
      for (let g = 0; g < p.groups.length; g++) {
        const group = p.groups[g];
        const arch = group.state.arch;
        // The rows this readback describes must still be the rows in the table.
        const fresh =
          !this._destroyed &&
          arch.buffer === group.srcBuffer &&
          arch.count === group.count &&
          group.state.table.generation === group.generation;
        if (!fresh) {
          this._stats.staleReadbacks++;
          continue;
        }
        const dst = new Uint8Array(arch.buffer as ArrayBuffer);
        let at = group.start;
        for (let r = 0; r < group.ranges.length; r++) {
          const range = group.ranges[r];
          dst.set(mapped.subarray(at, at + range.size), range.offset);
          at += range.size;
        }
        this._stats.bytesReadBack += group.bytes;
      }
    } catch {
      this._stats.staleReadbacks += p.groups.length;
    }
    // Stale groups are still "covered": their archetype changed on the CPU
    // side, and the next dispatch re-uploads it from the CPU tables.
    this._markCovered(p.coveredUpTo);
    this._finishReadback(p, true);
  }

  private _finishReadback(p: PendingReadback, mapped: boolean): void {
    this._readBusy--;
    if (mapped) {
      try {
        p.buffer.unmap();
      } catch {
        /* already unmapped or destroyed */
      }
    }
    this._releaseReadBuffer(p.buffer);
    // A frame was skipped while this one was in flight: its staging buffer is
    // free now, so the catch-up goes out before the next dispatch can take it.
    if (this._catchUpNeeded) this._submitCatchUp();
    this._settle();
  }

  /** Returns a staging buffer to the pool, or destroys it on a dead runtime. */
  private _releaseReadBuffer(buffer: GPUBuffer): void {
    if (!this._destroyed && !this._degraded && this._readPool.length < MAX_READBACKS_IN_FLIGHT) {
      this._readPool.push(buffer);
      return;
    }
    try {
      buffer.destroy();
    } catch {
      /* ignore */
    }
  }

  private _markCovered(upTo: number): void {
    if (upTo > this._stats.completed) this._stats.completed = upTo;
  }

  /**
   * Recomputes `completed`/`pending` and resolves the `sync()` callers whose
   * target is reached. With no readback in flight and no catch-up owed, every
   * dispatch is reflected in the tables (each later one either finished
   * synchronously or would itself be in flight / owed), so `completed` jumps
   * to `dispatches`. A catch-up owed with nothing in flight -- only possible
   * if a staging buffer could not be allocated -- is attempted here once and
   * then abandoned, so `sync()` can never hang on it.
   */
  private _settle(): void {
    const stats = this._stats;
    if (this._destroyed || this._degraded) {
      this._catchUpNeeded = false;
      stats.completed = stats.dispatches;
    } else if (this._readBusy === 0) {
      if (this._catchUpNeeded) this._submitCatchUp();
      if (this._readBusy === 0) {
        this._catchUpNeeded = false;
        stats.completed = stats.dispatches;
      }
    }
    stats.pending = stats.dispatches - stats.completed;
    const waiters = this._waiters;
    if (waiters.length === 0) return;
    let kept = 0;
    const ready: (() => void)[] = [];
    for (let i = 0; i < waiters.length; i++) {
      const w = waiters[i];
      if (stats.completed >= w.target) ready.push(w.resolve);
      else waiters[kept++] = w;
    }
    waiters.length = kept;
    for (let i = 0; i < ready.length; i++) ready[i]();
  }

  // -------------------------------------------------------------------------
  // Device-side bookkeeping
  // -------------------------------------------------------------------------

  private _stateFor(arch: Archetype): ArchState {
    let state = this._states.get(arch.id);
    if (!state || state.arch !== arch) {
      const device = (this.context as GPUContext).device;
      let byDevice = tableCache.get(arch);
      if (!byDevice) {
        byDevice = new Map();
        tableCache.set(arch, byDevice);
      }
      const tag = deviceTag(device);
      let table = byDevice.get(tag);
      if (!table) {
        table = {
          arch,
          device,
          buffers: [null, null, null],
          size: 0,
          srcBuffer: null,
          lastCount: -1,
          dirty: true,
          generation: 0,
          epoch: 0,
          sabStaging: null,
          refs: 0,
        };
        byDevice.set(tag, table);
      }
      table.refs++;
      state = { arch, table, slotBase: -1, slots: 0, bindGroups: [], bindEpoch: -1 };
      this._states.set(arch.id, state);
    }
    return state;
  }

  /**
   * Makes sure the shared table has a big-enough device copy for every view
   * type THIS kernel binds. Growing or adding a copy bumps `table.epoch`, which
   * invalidates the bind groups of every kernel using the table.
   */
  private _ensureTable(state: ArchState, chunk: Archetype): void {
    const device = (this.context as GPUContext).device;
    const table = state.table;
    const raw = chunk.buffer.byteLength;
    const need = raw < 4 ? 4 : alignUp(raw, 4);

    if (table.size < need) {
      // The table grew: every copy is the wrong size now.
      for (let i = 0; i < table.buffers.length; i++) {
        const b = table.buffers[i];
        if (!b) continue;
        try {
          b.destroy();
        } catch {
          /* ignore */
        }
        table.buffers[i] = null;
      }
      table.size = need;
      table.epoch++;
      table.dirty = true;
      table.sabStaging = null;
    } else if (table.srcBuffer !== chunk.buffer) {
      // Same size class, reallocated source: re-upload, keep the buffers.
      table.dirty = true;
    }

    for (let v = 0; v < this._views.length; v++) {
      const slot = VIEW_ORDER.indexOf(this._views[v]);
      if (slot < 0 || table.buffers[slot]) continue;
      table.buffers[slot] = device.createBuffer({
        label: `cozy:arch${chunk.id}:${VIEW_ORDER[slot]}`,
        size: table.size,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
      table.epoch++;
      // A copy created now holds nothing: the table has to be re-uploaded.
      table.dirty = true;
    }
  }

  /** Uploads the table if the residency rules say the CPU side changed. */
  private _upload(state: ArchState, chunk: Archetype, count: number): void {
    const table = state.table;
    const src = chunk.buffer;
    if (!table.dirty && table.srcBuffer === src && table.lastCount === count) return;
    const device = (this.context as GPUContext).device;
    const size = src.byteLength < table.size ? src.byteLength : table.size;
    const aligned = size - (size % 4);
    if (aligned > 0) {
      let bytes: Uint8Array;
      if (chunk.shared) {
        // HAZARD 2: writeBuffer with a SharedArrayBuffer-backed view segfaults.
        let staging = table.sabStaging;
        if (!staging || staging.byteLength < aligned) {
          staging = new Uint8Array(new ArrayBuffer(aligned));
          table.sabStaging = staging;
        }
        staging.set(new Uint8Array(src as ArrayBuffer, 0, aligned));
        bytes = staging.byteLength === aligned ? staging : staging.subarray(0, aligned);
      } else {
        bytes = new Uint8Array(src as ArrayBuffer, 0, aligned);
      }
      let copies = 0;
      for (let i = 0; i < table.buffers.length; i++) {
        const b = table.buffers[i];
        if (!b) continue;
        // `bytes` is always backed by a plain ArrayBuffer here: a SharedArrayBuffer
        // table was copied into `sabStaging` above (hazard 2).
        device.queue.writeBuffer(b, 0, bytes as Uint8Array<ArrayBuffer>, 0, aligned);
        copies++;
      }
      this._stats.bytesUploaded += aligned * copies;
    }
    table.srcBuffer = src;
    table.lastCount = count;
    table.dirty = false;
    table.generation++;
  }

  /**
   * Gives `state` a permanent reservation of `passes` uniform slots, growing
   * the kernel's single uniform buffer (and its host image) when needed.
   * Reservations never move, so a bind group built once stays valid; growing
   * the BUFFER does invalidate every bind group, which is why that is rare
   * (capacity doubles) and handled by dropping them all.
   */
  private _reserveSlots(state: ArchState, passes: number, device: GPUDevice): void {
    if (state.slots < passes) {
      state.slotBase = this._uniNext;
      state.slots = passes;
      this._uniNext += passes;
      state.bindGroups.length = 0;
    }
    if (this._uniCapacity >= this._uniNext && this._uniBuffer !== null) return;

    let capacity = this._uniCapacity > 0 ? this._uniCapacity : 8;
    while (capacity < this._uniNext) capacity *= 2;
    const stride = this._uniformStride;
    const host = new Uint8Array(capacity * stride);
    if (this._uniHost) host.set(this._uniHost);
    this._uniHost = host;
    this._uniF32 = new Float32Array(host.buffer);
    this._uniU32 = new Uint32Array(host.buffer);
    if (this._uniBuffer) {
      try {
        this._uniBuffer.destroy();
      } catch {
        /* ignore */
      }
    }
    this._uniBuffer = device.createBuffer({
      label: `cozy:${this.ir.name}:uniforms`,
      size: capacity * stride,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._uniCapacity = capacity;
    // Every bind group referenced the old buffer.
    for (const other of this._states.values()) other.bindGroups.length = 0;
  }

  private _ensureBindGroups(state: ArchState, device: GPUDevice, pipeline: GPUComputePipeline): void {
    const table = state.table;
    if (state.bindEpoch !== table.epoch) state.bindGroups.length = 0;
    if (state.bindGroups.length >= state.slots) return;
    const layout = pipeline.getBindGroupLayout(0);
    const stride = this._uniformStride;
    const size = (this._layout as UniformLayout).size;
    for (let p = state.bindGroups.length; p < state.slots; p++) {
      const entries: GPUBindGroupEntry[] = [];
      for (let v = 0; v < this._views.length; v++) {
        entries.push({ binding: v, resource: { buffer: this._bufferOf(table, v) } });
      }
      entries.push({
        binding: this._views.length,
        resource: { buffer: this._uniBuffer as GPUBuffer, offset: (state.slotBase + p) * stride, size },
      });
      state.bindGroups.push(device.createBindGroup({ label: `cozy:${this.ir.name}:bg`, layout, entries }));
    }
    state.bindEpoch = table.epoch;
  }

  /** The shared table's device copy for this kernel's view index `v`. */
  private _bufferOf(table: SharedTable, v: number): GPUBuffer {
    return table.buffers[VIEW_ORDER.indexOf(this._views[v])] as GPUBuffer;
  }

  /** Fills one uniform slot in the host image; the upload happens once per dispatch. */
  private _writeUniform(
    state: ArchState,
    chunk: Archetype,
    dt: number,
    count: number,
    base: number,
    slot: number,
    layout: UniformLayout,
  ): void {
    const f32 = this._uniF32 as Float32Array;
    const u32 = this._uniU32 as Uint32Array;
    const at = (slot * this._uniformStride) >> 2;
    f32[at + (layout.dt.offset >> 2)] = dt;
    u32[at + (layout.count.offset >> 2)] = count;
    u32[at + (layout.base.offset >> 2)] = base;
    u32[at + (layout.countOther.offset >> 2)] = count;
    const fields = layout.fields;
    for (let i = 0; i < fields.length; i++) {
      const a = fields[i].access;
      const col = chunk.columns[this.components[a.component].id];
      const view = col ? (col[a.field] as unknown as { byteOffset: number; BYTES_PER_ELEMENT: number }) : null;
      // Re-read every frame: growth moves every column.
      u32[at + (fields[i].offset >> 2)] = view ? view.byteOffset / view.BYTES_PER_ELEMENT : 0;
    }
    const uniforms = layout.uniforms;
    const values = this.uniformValues;
    for (let i = 0; i < uniforms.length; i++) f32[at + (uniforms[i].offset >> 2)] = values[i];
    if (slot < this._slotLo) this._slotLo = slot;
    if (slot > this._slotHi) this._slotHi = slot;
    void state;
  }

  /**
   * The byte ranges of the fields the kernel writes, in table coordinates,
   * tagged with the device buffer they must be copied out of and merged where
   * two written columns are adjacent in the same buffer.
   */
  private _writtenRanges(chunk: Archetype, count: number): ByteRange[] {
    const out = this._rangeScratch;
    out.length = 0;
    const fields = this._writeFields;
    for (let i = 0; i < fields.length; i++) {
      const a = fields[i];
      const col = chunk.columns[this.components[a.component].id];
      if (!col) continue;
      const view = col[a.field] as unknown as { byteOffset: number; BYTES_PER_ELEMENT: number } | undefined;
      if (!view) continue;
      const size = count * view.BYTES_PER_ELEMENT;
      if (size <= 0) continue;
      const kind = FIELD_KINDS[a.fieldKind].view;
      const vi = kind === null ? 0 : this._views.indexOf(kind);
      out.push({ offset: view.byteOffset, size, view: vi < 0 ? 0 : vi });
    }
    if (out.length < 2) return out;
    out.sort(byViewThenOffset);
    const merged: ByteRange[] = [];
    let cur = { offset: out[0].offset, size: out[0].size, view: out[0].view };
    for (let i = 1; i < out.length; i++) {
      const r = out[i];
      if (r.view === cur.view && r.offset <= cur.offset + cur.size) {
        const end = r.offset + r.size;
        if (end > cur.offset + cur.size) cur.size = end - cur.offset;
      } else {
        merged.push(cur);
        cur = { offset: r.offset, size: r.size, view: r.view };
      }
    }
    merged.push(cur);
    this._rangeScratch = merged;
    return merged;
  }

  /**
   * A MAP_READ staging buffer of at least `bytes`. Sizes are rounded up to a
   * power of two so the pool stabilizes instead of churning as the row count
   * drifts. Returns null when too many readbacks are already in flight (or
   * allocation failed); the caller then owes a catch-up readback.
   */
  private _takeReadBuffer(device: GPUDevice, bytes: number): GPUBuffer | null {
    const need = poolSize(bytes);
    for (let i = 0; i < this._readPool.length; i++) {
      if (this._readPool[i].size >= need) {
        const buffer = this._readPool.splice(i, 1)[0];
        this._readBusy++;
        return buffer;
      }
    }
    if (this._readBusy >= MAX_READBACKS_IN_FLIGHT) return null;
    // Nothing reusable: drop an undersized idle buffer to bound the pool.
    const idle = this._readPool.pop();
    if (idle) {
      try {
        idle.destroy();
      } catch {
        /* ignore */
      }
    }
    let buffer: GPUBuffer;
    try {
      buffer = device.createBuffer({
        label: `cozy:${this.ir.name}:readback`,
        size: need,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
    } catch {
      return null;
    }
    this._readBusy++;
    return buffer;
  }

  /** One-way switch to the CPU backend. `reason` null means already reported. */
  private _degradeToCPU(reason: string | null): void {
    if (this._degraded) return;
    this._degraded = true;
    if (reason) warn(`kernel "${this.ir.name}" fell back to the CPU backend (${reason}).`);
    this._target = 'cpu';
    this._pipeline = null;
    this._releaseDeviceState();
    this._setBackend(this._cpu ? 'cpu' : 'none');
    // In-flight readbacks drain into _finishReadback without touching the
    // tables; waiters resolve now, because the CPU tables are authoritative.
    this._settle();
  }

  /**
   * Sets {@link backend}. Entering `'none'` warns exactly once per runtime: a
   * kernel that silently stops running reads as a physics bug, not a library
   * one.
   */
  private _setBackend(backend: KernelBackend): void {
    this.backend = backend;
    this._stats.lastBackend = backend;
    if (backend === 'none' && !this._cpuBrokenWarned) {
      this._cpuBrokenWarned = true;
      warn(
        `kernel "${this.ir.name}" has no usable backend: the CPU loop failed to compile` +
          `${this._cpuError ? ` (${this._cpuError})` : ''} and no GPU pipeline is available. ` +
          `Every dispatch is a no-op; handle.backend === 'none'.`,
      );
    }
  }

  private _releaseDeviceState(): void {
    for (const state of this._states.values()) {
      state.bindGroups.length = 0;
      state.slotBase = -1;
      state.slots = 0;
      state.bindEpoch = -1;
      releaseTable(state.table);
    }
    this._states.clear();
    if (this._uniBuffer) {
      try {
        this._uniBuffer.destroy();
      } catch {
        /* ignore */
      }
    }
    this._uniBuffer = null;
    this._uniCapacity = 0;
    this._uniNext = 0;
    this._uniHost = null;
    this._uniF32 = null;
    this._uniU32 = null;
    for (const b of this._readPool) {
      try {
        b.destroy();
      } catch {
        /* ignore */
      }
    }
    this._readPool.length = 0;
  }

  /**
   * Resolves when every dispatch submitted so far has been read back and
   * applied to the archetype tables. Resolves immediately for `'none'` and for
   * the CPU backend. This is how a `'sync-frame'` kernel is awaited, and the
   * only supported way to observe a kernel's results deterministically.
   */
  sync(): Promise<void> {
    const target = this._stats.dispatches;
    if (this._stats.completed >= target) return Promise.resolve();
    return new Promise<void>((resolve) => {
      this._waiters.push({ target, resolve });
      // Nothing in flight but a catch-up owed: issue it now rather than wait.
      this._settle();
    });
  }

  /**
   * Declares that the CPU wrote fields this kernel owns, so the next dispatch
   * re-uploads. Pass an archetype to invalidate one table, or nothing for all.
   * Cheap: it only sets a flag.
   */
  markCpuDirty(archetype?: Archetype): void {
    if (archetype) {
      const state = this._states.get(archetype.id);
      if (state && state.arch === archetype) {
        state.table.dirty = true;
        state.table.generation++;
      }
      return;
    }
    for (const state of this._states.values()) {
      state.table.dirty = true;
      state.table.generation++;
    }
  }

  /**
   * The GPUBuffer holding `archetype`'s table on the device, for rendering
   * straight out of it with `readback: 'none'`. Null when the archetype has
   * never been dispatched or the kernel is on the CPU backend. The buffer is
   * REPLACED when the archetype grows -- re-fetch it after structural changes,
   * exactly like a column TypedArray.
   *
   * Field offsets inside it are `chunk.col(C)[field].byteOffset`, the same as
   * on the CPU side: the buffer is a byte-for-byte image of the table.
   */
  bufferFor(archetype: Archetype): GPUBuffer | null {
    if (!archetype) return null;
    const state = this._states.get(archetype.id);
    if (!state || state.arch !== archetype || this._views.length === 0) return null;
    return state.table.buffers[VIEW_ORDER.indexOf(this._views[0])];
  }

  /**
   * Releases every GPU buffer and drops the pipeline references. Called by
   * `world.removeSystem` through the handle. Safe to call twice. Pending
   * readbacks are abandoned (their `sync()` promises still resolve).
   */
  destroy(): void {
    if (this._destroyed) return;
    this._destroyed = true;
    this._releaseDeviceState();
    this._pipeline = null;
    for (const t of this._trackedAt) trackRuntime(t.world, t.group, this, false);
    this._trackedAt.length = 0;
    // In-flight readbacks (including a catch-up) drain into _finishReadback,
    // which destroys their staging buffers and touches no table.
    this._catchUpNeeded = false;
    this._stats.pending = 0;
    this._stats.completed = this._stats.dispatches;
    const waiters = this._waiters.slice();
    this._waiters.length = 0;
    for (let i = 0; i < waiters.length; i++) waiters[i].resolve();
  }
}

/** Readbacks in flight per kernel before the runtime starts coalescing frames. */
export const MAX_READBACKS_IN_FLIGHT = 4;

/** Consecutive failed maps that still trigger a catch-up (bounds the retry). */
const MAX_MAP_RETRIES = 2;

/** `_encodeReadback`'s "no staging buffer right now" result. */
const SKIPPED = 1 as const;

/** Staging buffers are sized in powers of two (>= 4 KiB) so the pool settles. */
function poolSize(bytes: number): number {
  let size = 4096;
  while (size < bytes) size *= 2;
  return size;
}

/** Drops one kernel's hold on a shared table, freeing it when the last goes. */
function releaseTable(table: SharedTable): void {
  if (--table.refs > 0) return;
  for (let i = 0; i < table.buffers.length; i++) {
    const b = table.buffers[i];
    if (!b) continue;
    try {
      b.destroy();
    } catch {
      /* ignore */
    }
    table.buffers[i] = null;
  }
  table.size = 0;
  table.srcBuffer = null;
  table.lastCount = -1;
  table.dirty = true;
  table.epoch++;
  const byDevice = tableCache.get(table.arch);
  if (byDevice) {
    byDevice.delete(deviceTag(table.device));
    if (byDevice.size === 0) tableCache.delete(table.arch);
  }
}

function byViewThenOffset(a: ByteRange, b: ByteRange): number {
  return a.view !== b.view ? a.view - b.view : a.offset - b.offset;
}

function dedupeAccesses(list: readonly FieldAccess[]): FieldAccess[] {
  const seen = new Set<string>();
  const out: FieldAccess[] = [];
  for (const a of list) {
    const key = `${a.role}:${a.component}.${a.field}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(a);
  }
  return out;
}

/** `checkGPUSupport` is another developer's file: a throw from it means "no GPU", not a crash. */
function safeBlockers(ir: KernelIR): { component: string; field: string; kind: string }[] {
  try {
    return gpuBlockers(ir);
  } catch {
    return [];
  }
}

function safeCheckSupport(
  ir: KernelIR,
  caps: GPUCapabilities,
  workgroupSize: number | undefined,
): { ok: boolean; reasons: readonly { code: string; message: string }[] } {
  try {
    const support = checkGPUSupport(ir, caps, workgroupSize ? { workgroupSize } : undefined);
    return { ok: support.ok, reasons: support.reasons || [] };
  } catch (e) {
    return { ok: false, reasons: [{ code: 'E_DEVICE_LIMIT', message: (e as Error).message }] };
  }
}

/** @internal Re-exported so index.ts can report capabilities without importing ./device twice. */
export type { GPUCapabilities };
