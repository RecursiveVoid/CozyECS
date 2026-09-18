/// <reference types="@webgpu/types" />
/**
 * `cozyecs/gpu` -- the optional entry point.
 *
 * OWNER: architect. Developers A-D own ./parse, ./wgsl, ./runtime and ./cpu and
 * should not need to change this file; if a signature here is wrong, say so
 * rather than working around it.
 *
 * Write a system as a plain JavaScript kernel; CozyECS turns it into WGSL:
 *
 * ```js
 * import { kernelSystem } from 'cozyecs/gpu';
 *
 * const move = await kernelSystem(world, 'Move', {
 *   components: [Position, Velocity],   // kernel parameters, in order
 *   uniforms: { gravity: -9.8 },        // every outside value comes through here
 *   readback: 'async',
 *   kernel: (p, v, dt, u) => {
 *     v.y += u.gravity * dt;
 *     p.x += v.x * dt; p.y += v.y * dt;
 *     if (p.y < 0) { p.y = 0; v.y = -v.y * 0.5; }
 *   },
 * });
 *
 * world.update(dt);           // the kernel dispatches here, without blocking
 * ```
 *
 * The core (`cozyecs`) never imports anything in this directory, and this
 * directory imports the core for TYPES ONLY, so neither bundle carries the
 * other. See docs/GPU.md for the subset grammar, the error catalogue and the
 * CPU/GPU parity contract.
 */

import type { ComponentType } from '../component';
import type { QueryDesc } from '../query';
import type { SystemHandle } from '../system';
import type { World } from '../world';
import type { Archetype } from '../archetype';

import type { IRComponent, IRFieldKind, IRUniform, KernelIR } from './ir';
import { KernelError, validateIR } from './ir';
import { kernelSource, parseKernel } from './parse';
import { KernelRuntime, trackRuntime } from './runtime';
import type { AutoThresholds, KernelBackend, KernelStats, KernelTarget, ReadbackMode } from './runtime';

export { getGPUContext, hasGPUContext, peekGPUContext, setGPUProvider } from './device';
export type { GPUContext, GPUCapabilities } from './device';

export { KernelError, describeIR, IR_VERSION, BUILTINS, WORKGROUP_SIZE } from './ir';
export type { KernelErrorCode, KernelIR, Span } from './ir';

export {
  BASELINE_CPU_NS_PER_ENTITY,
  DEFAULT_AUTO_THRESHOLDS,
  GPU_FIXED_OVERHEAD_NS,
  autoPrefersGPU,
  cpuCostEstimate,
  flushKernels,
  getAutoThresholds,
  setAutoThresholds,
} from './runtime';
export { calibrateAuto } from './calibrate';
export type { CalibrateOptions, CalibrationResult, CoreModule } from './calibrate';
export type { AutoThresholds, KernelBackend, KernelStats, KernelTarget, ReadbackMode } from './runtime';

/** Options for {@link kernelSystem}. */
export interface KernelSystemOptions {
  /**
   * Components the kernel takes, in parameter order. Every one is implicitly
   * part of the system's query `all`.
   */
  readonly components: readonly ComponentType[];
  /**
   * Components the kernel writes. Optional: the writes are inferred from the
   * kernel's assignments. When given, it must be a SUPERSET of the inferred set
   * (E_WRITE_NOT_DECLARED otherwise); the extra components are still read back,
   * which is occasionally useful to keep a field the kernel does not touch in
   * sync.
   */
  readonly write?: readonly ComponentType[];
  /**
   * Values the kernel reads as `u.<name>`. This is the ONLY way into a kernel
   * from outside: a closure variable is invisible to `Function.prototype.toString`,
   * so referencing one is E_UNKNOWN_IDENTIFIER. Update them later with
   * {@link KernelSystemHandle.setUniform}.
   */
  readonly uniforms?: Readonly<Record<string, number>>;
  /** `'gpu'`, `'cpu'` or `'auto'` (default): see docs/GPU.md, "Choosing a target". */
  readonly target?: KernelTarget;
  /** What happens to the writes. Default `'async'`. */
  readonly readback?: ReadbackMode;
  /** Scheduler group, like any system. Default `'update'`. */
  readonly group?: string;
  /** Scheduler order within the group. Default 0. */
  readonly order?: number;
  /**
   * Extra query filters. `all` is merged with `components`; `any`/`none` are
   * used as given. Enableable components in `all` force the CPU backend (v1),
   * with a one-time warning.
   */
  readonly query?: Omit<QueryDesc, 'all'> & { all?: readonly ComponentType[] };
  /**
   * Pairwise (n-body / boids) form: `kernel(self, other, dt, u)` running over
   * every ordered pair in a chunk. `self` and `other` expose the fields of all
   * `components` merged into one namespace, so no two of them may share a field
   * name (E_FIELD_COLLISION). O(n^2): read the warning in docs/GPU.md before
   * using it, and note that pairs are formed WITHIN a chunk only.
   */
  readonly pairwise?: boolean;
  /** Overrides the auto break-even for this kernel. */
  readonly thresholds?: Partial<AutoThresholds>;
  /** Overrides the workgroup size (default 256). Rarely useful. */
  readonly workgroupSize?: number;
  /**
   * Trip cap for a loop whose bound the parser cannot prove, and the ceiling
   * for one it can. Default 4096. Both backends emit the cap, so a kernel bug
   * cannot hang a GPU and both backends stop at the same iteration.
   */
  readonly maxLoopIterations?: number;
  /**
   * The kernel. Parameters are matched BY POSITION: one per component, then
   * optionally `dt`, then optionally `u`. Names do not matter (minifiers rename
   * them); field names do.
   */
  readonly kernel: (...args: any[]) => void;
}

/** What {@link kernelSystem} adds to an ordinary `SystemHandle`. */
export interface KernelExtras {
  /** The parsed kernel. Stable; safe to snapshot with `describeIR`. */
  readonly ir: KernelIR;
  /**
   * Which backend the next dispatch will use. Flips per dispatch under `'auto'`;
   * before the first dispatch it is the prediction for the query's current size.
   * `'none'` means neither backend is usable and dispatches are no-ops (one
   * warning was printed); see docs/GPU.md section 5.
   */
  readonly backend: KernelBackend;
  readonly readback: ReadbackMode;
  readonly stats: KernelStats;
  /**
   * Resolves when every dispatch submitted so far has been applied to the
   * archetype tables. Resolves immediately on the CPU backend and for
   * `readback: 'none'`. This is the only deterministic way to observe results.
   */
  sync(): Promise<void>;
  /** Sets a uniform for the next dispatch. Throws E_UNKNOWN_UNIFORM for an undeclared name. */
  setUniform(name: string, value: number): void;
  /** Current value of a uniform. */
  getUniform(name: string): number;
  /** Declares that the CPU wrote fields this kernel owns, forcing a re-upload. */
  markCpuDirty(archetype?: Archetype): void;
  /**
   * The device buffer holding an archetype's table, for rendering out of it
   * under `readback: 'none'`. Null on the CPU backend or before the first
   * dispatch. Replaced when the archetype grows.
   */
  bufferFor(archetype: Archetype): GPUBuffer | null;
  /** Unregisters the system and releases its GPU resources. Idempotent. */
  destroy(): void;
}

/** The handle `kernelSystem` returns: a normal system plus {@link KernelExtras}. */
export type KernelSystemHandle = SystemHandle & KernelExtras;

/** Builds the backend-facing component description from a core ComponentType. */
function toIRComponent(c: ComponentType, index: number): IRComponent {
  const fields = c.keys.map((name, i) => ({
    name,
    kind: c.tokens[i].kind as IRFieldKind,
    index: i,
  }));
  return { index, id: c.id, name: c.name, fields };
}

/** Validates and numbers the declared uniforms. */
function toIRUniforms(values: Readonly<Record<string, number>> | undefined, kernelName: string): IRUniform[] {
  const out: IRUniform[] = [];
  if (!values) return out;
  const names = Object.keys(values);
  for (let i = 0; i < names.length; i++) {
    const v = values[names[i]];
    if (typeof v !== 'number' || !Number.isFinite(v)) {
      throw new KernelError('E_BAD_UNIFORM_VALUE', `uniform "${names[i]}" is ${String(v)}; uniforms must be finite numbers`, {
        kernelName,
        hint: 'Uniforms are uploaded as f32. Pass a number, and update it per frame with handle.setUniform().',
      });
    }
    out.push({ name: names[i], index: i, initial: v });
  }
  return out;
}

/**
 * Registers a kernel as a system.
 *
 * Asynchronous because acquiring a WebGPU device is. Register kernels during
 * startup, before the first `world.update()`; the returned handle is an
 * ordinary `SystemHandle` (so `enabled`, `order` and `world.removeSystem` all
 * work) with the GPU-specific members of {@link KernelExtras} added.
 *
 * NEVER THROWS FOR A CAPABILITY REASON. No WebGPU device, a field the GPU
 * backend cannot address, a disabled-row query: each produces one
 * `console.warn` and the CPU backend. It throws {@link KernelError} only for
 * things the author must fix -- a kernel outside the supported subset, an
 * unknown field or uniform, a bad `write` list.
 */
export async function kernelSystem(
  world: World,
  name: string,
  options: KernelSystemOptions,
): Promise<KernelSystemHandle> {
  const components = options.components || [];
  if (components.length === 0) {
    throw new KernelError('E_PARAM_COUNT', 'a kernel needs at least one component', { kernelName: name });
  }
  const form = options.pairwise ? 'pairwise' : 'per-entity';
  const irComponents = components.map(toIRComponent);
  const uniforms = toIRUniforms(options.uniforms, name);
  const source = kernelSource(options.kernel, name);

  const ir: KernelIR = parseKernel(options.kernel, {
    name,
    form,
    components: irComponents,
    uniformNames: uniforms.map((u) => u.name),
    source,
    maxLoopIterations: options.maxLoopIterations,
  });
  validateIR(ir);

  // A declared `write` list must cover everything the kernel assigns to.
  if (options.write) {
    const declared = new Set(options.write.map((c) => c.id));
    for (const w of ir.writes) {
      const c = components[w.component];
      if (!declared.has(c.id)) {
        throw new KernelError(
          'E_WRITE_NOT_DECLARED',
          `the kernel assigns to ${c.name}.${w.field}, but "${c.name}" is missing from the write list`,
          { kernelName: name, source, hint: `Add ${c.name} to write, or drop the write option and let it be inferred.` },
        );
      }
    }
  }

  const desc = options.query || {};
  const all: ComponentType[] = components.slice();
  if (desc.all) for (const c of desc.all) if (all.indexOf(c) === -1) all.push(c);
  const query = world.query({ all, any: desc.any, none: desc.none });
  const enableIds = all.filter((c) => c.enableable).map((c) => c.id);

  const runtime = await KernelRuntime.create(ir, components, {
    target: options.target || 'auto',
    readback: options.readback || 'async',
    enableIds,
    thresholds: options.thresholds,
    workgroupSize: options.workgroupSize,
  });

  const handle = world.system(name, { group: options.group, order: options.order, query }, (q, dt) => {
    runtime.dispatch(q, dt);
  }) as KernelSystemHandle;

  // So `flushKernels(world, group?)` can await this kernel without the app
  // holding onto the handle. `handle.group` is the group the scheduler resolved.
  trackRuntime(world, handle.group, runtime, true);
  runtime.predictBackend(query);

  // The runtime reads uniform values from here on every dispatch.
  const uniformValues = new Float64Array(uniforms.length);
  for (let i = 0; i < uniforms.length; i++) uniformValues[i] = uniforms[i].initial;
  runtime.uniformValues = uniformValues;

  let destroyed = false;
  const extras: KernelExtras & ThisType<KernelSystemHandle> = {
    ir,
    get backend() {
      // Before the first dispatch, 'auto' reports its prediction for the
      // query's current size rather than "a GPU exists".
      return runtime.predictBackend(query);
    },
    get readback() {
      return runtime.readback;
    },
    get stats() {
      return runtime.stats;
    },
    sync: () => runtime.sync(),
    setUniform: (uname: string, value: number) => {
      const u = ir.uniforms.find((x) => x.name === uname);
      if (!u) {
        throw new KernelError('E_UNKNOWN_UNIFORM', `no uniform "${uname}"; declared: ${ir.uniforms.map((x) => x.name).join(', ') || 'none'}`, {
          kernelName: name,
        });
      }
      if (typeof value !== 'number' || !Number.isFinite(value)) {
        throw new KernelError('E_BAD_UNIFORM_VALUE', `uniform "${uname}" set to ${String(value)}`, { kernelName: name });
      }
      uniformValues[u.index] = value;
    },
    getUniform: (uname: string) => {
      const u = ir.uniforms.find((x) => x.name === uname);
      if (!u) throw new KernelError('E_UNKNOWN_UNIFORM', `no uniform "${uname}"`, { kernelName: name });
      return uniformValues[u.index];
    },
    markCpuDirty: (archetype?: Archetype) => runtime.markCpuDirty(archetype),
    bufferFor: (archetype: Archetype) => runtime.bufferFor(archetype),
    destroy: () => {
      if (destroyed) return;
      destroyed = true;
      world.removeSystem(handle);
      trackRuntime(world, handle.group, runtime, false);
      runtime.destroy();
    },
  };

  for (const key of Object.keys(extras) as (keyof KernelExtras)[]) {
    const d = Object.getOwnPropertyDescriptor(extras, key);
    if (d) Object.defineProperty(handle, key, d);
  }
  return handle;
}
