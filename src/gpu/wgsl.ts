/**
 * {@link KernelIR} -> WGSL source + binding layout descriptor.
 *
 * OWNER: developer B. Pure string generation: this file must not import
 * `./runtime`, must not touch a GPUDevice and must not allocate anything. Its
 * whole output is decided by the IR plus a {@link WGSLOptions}, which is what
 * lets its tests be plain string assertions with no GPU present.
 *
 * SHAPE OF THE GENERATED MODULE (the contract runtime.ts binds against; every
 * number in it comes from `uniformLayout`/`storageViews` in ./ir):
 *
 *   // cozyecs/gpu generated compute kernel -- do not edit.
 *   struct U {
 *     dt: f32,
 *     count: u32,
 *     base: u32,
 *     countOther: u32,
 *     o0_Position_x: u32,      // field bases, in ELEMENTS of that field's view
 *     o0_Velocity_y: u32,
 *     u_gravity: f32,
 *   }
 *   @group(0) @binding(0) var<storage, read_write> t_f32: array<f32>;
 *   @group(0) @binding(1) var<uniform> u: U;                         // binding = views.length
 *   fn cozy_rand(seed: f32) -> f32 { ... }                           // only when ir.usesRand
 *   @compute @workgroup_size(256)
 *   fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
 *     let row: u32 = gid.x + u.base;
 *     if (row >= u.count) { return; }
 *     ...body...
 *   }
 *
 * WHY ONE BUFFER PER ARCHETYPE, NOT ONE PER FIELD: a device grants only 8
 * storage buffers per compute stage by default (10 on an M4 adapter), and
 * `Position{x,y,z} + Velocity{x,y,z}` already burns 6. Binding the archetype's
 * single table buffer once per 4-byte view type and addressing fields by base
 * offset removes the cap entirely, matches CozyECS's storage exactly, and
 * measured never slower (1M-entity move kernel: per-field 0.93/0.95/0.68/0.59 ms
 * vs one-buffer 0.74/0.69/0.64/0.58 ms). Bind-group OFFSETS must be 256-byte
 * aligned, which a field column is not -- computed indices inside the shader
 * have no such rule, which is the second reason offsets live in the uniform.
 *
 * EXACTLY ONE STORAGE BINDING, ALWAYS. The original plan -- bind the table once
 * per 4-byte view type (`storageViews(ir)`) -- is invalid: WebGPU rejects two
 * writable storage bindings whose ranges overlap, and two views of the same
 * table overlap completely. Dawn/Metal, verbatim:
 *
 *   Writable storage buffer binding aliasing found between [BindGroup] binding
 *   index 0, and [BindGroup] binding index 1, with overlapping ranges
 *   (offset: 0, size: 8272) and (offset: 0, size: 8272) in [Buffer].
 *
 * So a kernel that touches only one view type declares that view (the common
 * all-f32 case, byte-identical to the original plan), and a kernel that mixes
 * view types declares ONE `array<u32>` binding and reinterprets:
 * `bitcast<f32>(t_u32[i])` to read, `t_u32[i] = bitcast<u32>(v)` to write. Every
 * eligible field is 4 bytes wide, so the element index is the same either way,
 * and a bitcast is free on the GPU (measured identical, see the module notes).
 *
 * CONSEQUENCE FOR runtime.ts: bind from `WGSLModule.views` and
 * `WGSLModule.uniformBinding`, NOT from `storageViews(ir)`/`uniformBinding(ir)`
 * -- the ./ir helpers still describe the per-view-type plan, which would build a
 * bind group WebGPU refuses. `views.length` is always 0 or 1, so the uniform
 * block is at binding 0 or 1.
 *
 * The one storage binding is declared `read_write` even for a read-only kernel,
 * so one bind group layout serves every kernel.
 *
 * TYPE RULES (from ./ir design rule 1): all arithmetic is f32. An i32/u32 field
 * is read as `f32(t_i32[...])` and written as `t_i32[...] = cozy_store_i32(x)`
 * (rules below). Fields whose kind has no 4-byte view (f64, i8/u8/i16/u16,
 * bool, str) never reach this file: {@link checkGPUSupport} rejects them first
 * and the kernel stays on the CPU backend.
 *
 * ===========================================================================
 * PARITY CONTRACT -- the two rules the CPU backend must reproduce BIT FOR BIT
 * ===========================================================================
 * Both are defined for EVERY f32 input, NaN and +-Infinity included, using
 * only integer bit operations and exact float operations, so neither depends
 * on WGSL's "indeterminate" out-of-range float->int conversion or on a
 * driver's fast-math NaN handling. The WGSL is emitted verbatim from
 * {@link COZY_RAND_WGSL} / {@link COZY_STORE_WGSL}; the JS twins are
 * {@link cozyRandRef} / {@link storeI32Ref} / {@link storeU32Ref}, and
 * __tests__/gpu-wgsl.test.ts runs the WGSL on Dawn against those twins.
 * `x` below always means the f32 the expression evaluated to (on the CPU:
 * after `Math.fround`).
 *
 * 1. rand(seed)  -- stateless hash, result in [0, 1 - 2^-24], exact multiple of 2^-24
 *      k = Math.fround(seed) | 0          // ECMAScript ToInt32: truncate toward 0,
 *                                         // wrap mod 2^32; NaN, +-Inf, |x| < 1 -> 0
 *      h = (k ^ 0x9e3779b9) >>> 0
 *      h = Math.imul(h ^ (h >>> 16), 0x21f0aaad) >>> 0
 *      h = Math.imul(h ^ (h >>> 15), 0x735a2d97) >>> 0
 *      h = (h ^ (h >>> 15)) >>> 0
 *      return (h >>> 8) / 16777216        // == (h >>> 8) * 2^-24, exact in f32 and f64
 *    The WGSL computes ToInt32 from the f32's bits (`cozy_toint32`), NOT with
 *    `i32(seed)`: WGSL leaves out-of-range/NaN conversion indeterminate, JS wraps.
 *    NOTE: the final scale must be exactly 2^-24. `5.9604645e-8` (the constant in
 *    ./ir `cozy_rand`) is 2^-24 only once rounded to f32; in f64 it is off by
 *    3.8e-9 relative, so a CPU twin using it must `Math.fround` the result.
 *
 * 2. integer-field write  (`=` and every compound `op=` on an i32/u32 field)
 *      i32:  NaN -> 0;  x >= 2^31 -> 2147483647;  x <= -2^31 -> -2147483648;
 *            otherwise Math.round(x)     (round half toward +Infinity: 2.5 -> 3,
 *                                         -2.5 -> -2, -0.4 -> 0)
 *      u32:  NaN -> 0;  x >= 2^32 -> 4294967295;  x <= 0 -> 0;
 *            otherwise Math.round(x)
 *    i.e. saturate to the type's TRUE range (FIELD_KINDS[kind].range), never
 *    wrap the way a bare TypedArray store does. The WGSL rounds with
 *    `r = floor(x); x - r >= 0.5 ? r + 1 : r`, which is exact in f32 (x - floor(x)
 *    is always representable) and therefore equals JS `Math.round` on the same
 *    f32 -- unlike `floor(x + 0.5)`, whose f32 add rounds (0.49999997 -> 1,
 *    8388609 -> 8388610). Saturation is decided on x BEFORE rounding, which is
 *    equivalent: no f32 lies in (2^31 - 128, 2^31) or (2^32 - 256, 2^32).
 *
 *    Integer READS (`f32(t_i32[i])`) are exact for |v| <= 2^24. Beyond that the
 *    CPU rounds to nearest-even (`Math.fround`), while WGSL lets the GPU pick
 *    either f32 neighbour: at most one f32 ulp apart, documented, not policed.
 */

import type { GPUCapabilities } from './device';
import type {
  BlockStmt,
  Expr,
  FieldAccess,
  FieldRef,
  IRFieldKind,
  KernelErrorCode,
  KernelIR,
  Span,
  Stmt,
  StorageViewType,
  UniformLayout,
} from './ir';
import {
  BUILTINS,
  IR_VERSION,
  KernelError,
  WORKGROUP_SIZE,
  accessKey,
  gpuBlockers,
  offsetMemberName,
  storageViews,
  touchedFields,
  uniformLayout,
  uniformMemberName,
  validateIR,
} from './ir';

/** Knobs the runtime may pass; all default to the values pinned in ./ir. */
export interface WGSLOptions {
  /** Override {@link import('./ir').WORKGROUP_SIZE}. Must divide the device's invocation limit. */
  readonly workgroupSize?: number;
  /** Emit `// line` comments mapping generated WGSL back to kernel source. Debug only. */
  readonly annotate?: boolean;
}

/** Everything runtime.ts needs to create a pipeline and a bind group. */
export interface WGSLModule {
  /** The complete shader source. Deterministic for a given (ir, options). */
  readonly code: string;
  /** Always `'main'`. */
  readonly entryPoint: string;
  readonly workgroupSize: number;
  /**
   * The storage bindings to create, in binding order: `views[i]` is
   * `@binding(i)`, bound to the archetype's table buffer.
   *
   * At most ONE entry (empty only for a kernel that touches no field): WebGPU
   * forbids overlapping writable storage bindings, so a kernel mixing view
   * types gets a single `array<u32>` binding it bitcasts through. This is
   * therefore NOT `storageViews(ir)`, which still lists one view per type --
   * bind from this field. See the note at the top of this file.
   */
  readonly views: readonly StorageViewType[];
  /**
   * `@binding` of the uniform block; equals `views.length`, so 0 or 1. Use this
   * rather than `uniformBinding(ir)`, for the same reason as {@link views}.
   */
  readonly uniformBinding: number;
  /** Byte layout of the uniform block. Equals `uniformLayout(ir)`. */
  readonly layout: UniformLayout;
  /**
   * Pipeline cache key: identical strings mean identical `code`, so the runtime
   * can reuse a GPUComputePipeline across archetypes and across kernels.
   * Contains no archetype- or device-specific data.
   */
  readonly cacheKey: string;
}

/** Why a kernel cannot run on the GPU. Empty `reasons` means it can. */
export interface GPUSupport {
  readonly ok: boolean;
  readonly reasons: readonly { code: KernelErrorCode; message: string }[];
}

// ---------------------------------------------------------------------------
// Emission tables
// ---------------------------------------------------------------------------

/** Module-scope name of the storage binding for each view type. */
const VIEW_VAR: Readonly<Record<StorageViewType, string>> = Object.freeze({
  f32: 't_f32',
  i32: 't_i32',
  u32: 't_u32',
});

/**
 * Bumped whenever the emitted code changes shape for an unchanged IR, so a
 * pipeline cache keyed on {@link wgslCacheKey} can never serve stale code.
 * 2: exact-parity rand + saturating integer stores (see PARITY CONTRACT).
 */
export const WGSL_CODEGEN_REVISION = 2;

/**
 * `rand(seed)` as emitted into every module that uses it. Parity rule 1 in the
 * file header; {@link cozyRandRef} is the JS twin. `cozy_toint32` is ECMAScript
 * ToInt32 of an f32, computed from its bits.
 */
export const COZY_RAND_WGSL = `// rand(seed): bit-exact twin of JS
//   h = ((Math.fround(seed) | 0) ^ 0x9e3779b9) >>> 0;
//   h = Math.imul(h ^ (h >>> 16), 0x21f0aaad) >>> 0;
//   h = Math.imul(h ^ (h >>> 15), 0x735a2d97) >>> 0;
//   h = (h ^ (h >>> 15)) >>> 0;  return (h >>> 8) / 16777216;
// cozy_toint32 is JS ToInt32 (truncate, wrap mod 2^32, NaN/Inf -> 0) on the f32 bits.
fn cozy_toint32(x: f32) -> u32 {
  let b: u32 = bitcast<u32>(x);
  let e: u32 = (b >> 23u) & 0xffu;
  if (e < 127u || e == 255u) { return 0u; }
  let m: u32 = (b & 0x7fffffu) | 0x800000u;
  var v: u32 = 0u;
  if (e < 150u) {
    v = m >> (150u - e);
  } else if (e < 182u) {
    v = m << (e - 150u);
  }
  if ((b >> 31u) != 0u) { v = 0u - v; }
  return v;
}
fn cozy_rand(seed: f32) -> f32 {
  var h: u32 = cozy_toint32(seed) ^ 0x9e3779b9u;
  h = (h ^ (h >> 16u)) * 0x21f0aaadu;
  h = (h ^ (h >> 15u)) * 0x735a2d97u;
  h = h ^ (h >> 15u);
  return f32(h >> 8u) * 5.9604644775390625e-8;
}`;

/**
 * The integer-field write helpers, emitted into every module that writes an
 * i32 or u32 field. Parity rule 2 in the file header; {@link storeI32Ref} and
 * {@link storeU32Ref} are the JS twins. The NaN test is on the bits so no
 * fast-math mode can fold it away.
 */
export const COZY_STORE_WGSL = `// Integer-field write: bit-exact twin of JS (x = the stored f32)
//   i32: x !== x ? 0 : x >= 2147483648 ? 2147483647 : x <= -2147483648 ? -2147483648 : Math.round(x)
//   u32: x !== x || x <= 0 ? 0 : x >= 4294967296 ? 4294967295 : Math.round(x)
// Math.round = round half toward +Infinity; exact here because x - floor(x) is exact in f32.
fn cozy_round(x: f32) -> f32 {
  let r: f32 = floor(x);
  return select(r, r + 1.0, x - r >= 0.5);
}
fn cozy_store_i32(x: f32) -> i32 {
  if ((bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u) { return 0i; }
  if (x >= 2147483648.0) { return 2147483647i; }
  if (x <= -2147483648.0) { return bitcast<i32>(0x80000000u); }
  return i32(cozy_round(x));
}
fn cozy_store_u32(x: f32) -> u32 {
  if ((bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u) { return 0u; }
  if (x >= 4294967296.0) { return 0xffffffffu; }
  if (x <= 0.0) { return 0u; }
  return u32(cozy_round(x));
}`;

/** JS twin of `cozy_rand` in {@link COZY_RAND_WGSL}: identical bits for every input. */
export function cozyRandRef(seed: number): number {
  let h = ((Math.fround(seed) | 0) ^ 0x9e3779b9) >>> 0;
  h = Math.imul(h ^ (h >>> 16), 0x21f0aaad) >>> 0;
  h = Math.imul(h ^ (h >>> 15), 0x735a2d97) >>> 0;
  h = (h ^ (h >>> 15)) >>> 0;
  return (h >>> 8) / 16777216;
}

/** JS twin of `cozy_store_i32`: the value an i32 field holds after `field = x`. */
export function storeI32Ref(x: number): number {
  const f = Math.fround(x);
  if (f !== f) return 0;
  if (f >= 2147483648) return 2147483647;
  if (f <= -2147483648) return -2147483648;
  return Math.round(f) | 0;
}

/** JS twin of `cozy_store_u32`: the value a u32 field holds after `field = x`. */
export function storeU32Ref(x: number): number {
  const f = Math.fround(x);
  if (f !== f || f <= 0) return 0;
  if (f >= 4294967296) return 4294967295;
  return Math.round(f);
}

/**
 * The storage binding a kernel declares, and how fields are addressed through
 * it. `raw` means one `array<u32>` binding reinterpreted with `bitcast`,
 * which is what a kernel mixing view types must use (see the file header).
 */
interface BindingPlan {
  /** The declared views, in binding order. Length 0 or 1. */
  readonly views: readonly StorageViewType[];
  /** True when accesses go through `array<u32>` + `bitcast`. */
  readonly raw: boolean;
}

/**
 * Decides the storage binding for `ir`. One binding at most: WebGPU refuses a
 * bind group with two overlapping writable storage bindings, and every view of
 * an archetype table overlaps every other.
 */
function bindingPlan(ir: KernelIR): BindingPlan {
  const needed = storageViews(ir);
  if (needed.length <= 1) return { views: needed, raw: false };
  return { views: ['u32'], raw: true };
}

/** Storage index variable for each role: the dispatched row, or the inner pair. */
const ROLE_INDEX: readonly [string, string] = ['row', 'j'];

const INDENT = '  ';

// ---------------------------------------------------------------------------
// Identifiers
// ---------------------------------------------------------------------------

/**
 * Turns an arbitrary schema name into a WGSL identifier. WGSL allows
 * `[A-Za-z_][A-Za-z0-9_]*` but reserves names beginning with `__` and the bare
 * `_`; a JS field name may legally contain `$` or start with a digit, which
 * WGSL does not accept.
 */
function sanitizeIdent(raw: string): string {
  let s = raw.replace(/[^A-Za-z0-9_]/g, '_');
  if (s === '' || /^[0-9]/.test(s)) s = `v${s}`;
  if (s.startsWith('__')) s = `v${s}`;
  if (s === '_') s = 'v_';
  return s;
}

/**
 * WGSL member names for the uniform block.
 *
 * They are `offsetMemberName`/`uniformMemberName` from ./ir for every schema a
 * user can realistically write. The disambiguation below only fires for names
 * WGSL cannot spell or for two components that share a sanitized name, and it
 * is invisible to runtime.ts: the runtime writes uniform members by BYTE OFFSET
 * (`uniformLayout(ir)`), never by name. Deterministic, so the cache key holds.
 */
interface MemberNames {
  /** `accessKey(access)` -> member name. */
  readonly field: Map<string, string>;
  /** Uniform name -> member name. */
  readonly uniform: Map<string, string>;
}

function memberNames(ir: KernelIR): MemberNames {
  const used = new Set<string>(['dt', 'count', 'base', 'countOther', 'row', 'j', 'u', 'gid', 'main']);
  const take = (preferred: string): string => {
    const base = sanitizeIdent(preferred);
    if (!used.has(base)) {
      used.add(base);
      return base;
    }
    for (let n = 2; ; n++) {
      const cand = `${base}_${n}`;
      if (!used.has(cand)) {
        used.add(cand);
        return cand;
      }
    }
  };
  const field = new Map<string, string>();
  for (const a of touchedFields(ir)) field.set(accessKey(a), take(offsetMemberName(ir, a)));
  const uniform = new Map<string, string>();
  for (const u of ir.uniforms) uniform.set(u.name, take(uniformMemberName(u.name)));
  return { field, uniform };
}

/** Mangled name of a kernel local. `locals[i].id === i`, so this is injective. */
function localName(ir: KernelIR, id: number): string {
  return `l${id}_${sanitizeIdent(ir.locals[id].name)}`;
}

// ---------------------------------------------------------------------------
// Literals
// ---------------------------------------------------------------------------

function invalid(ir: KernelIR, message: string, span?: Span): never {
  throw new KernelError('E_INVALID_IR', message, { kernelName: ir.name, span, source: ir.source });
}

/**
 * A WGSL float literal that is exactly the f32 the value rounds to, so the
 * shader compiler performs no further rounding and the emitted constant is the
 * one the CPU backend's `Math.fround` produces.
 */
function f32Literal(ir: KernelIR, value: number, span?: Span): string {
  if (!Number.isFinite(value)) {
    invalid(ir, `numeric literal ${String(value)} cannot be expressed in WGSL`, span);
  }
  const f = Math.fround(value);
  if (!Number.isFinite(f)) {
    invalid(ir, `numeric literal ${value} is outside the f32 range (max 3.4028235e38)`, span);
  }
  if (Object.is(f, -0)) return '-0.0';
  let s = String(f);
  if (!/[.eE]/.test(s)) s += '.0';
  return s;
}

// ---------------------------------------------------------------------------
// The generator
// ---------------------------------------------------------------------------

class WGSLWriter {
  private readonly lines: string[] = [];
  private readonly names: MemberNames;
  private trips = 0;

  constructor(
    private readonly ir: KernelIR,
    private readonly annotate: boolean,
    private readonly plan: BindingPlan,
  ) {
    this.names = memberNames(ir);
  }

  // -- output ---------------------------------------------------------------

  /** Appends one line at `depth` levels of indentation. */
  line(depth: number, text: string): void {
    this.lines.push(text === '' ? '' : INDENT.repeat(depth) + text);
  }

  /** Appends a blank separator line. */
  blank(): void {
    this.lines.push('');
  }

  text(): string {
    return this.lines.join('\n');
  }

  /** One-line excerpt of the kernel source a node came from, for `annotate`. */
  private note(depth: number, span: Span | undefined): void {
    if (!this.annotate || !span || !this.ir.source) return;
    const raw = this.ir.source.slice(span.start, span.end).replace(/\s+/g, ' ').trim();
    if (raw === '') return;
    this.line(depth, `// ${raw.length > 72 ? `${raw.slice(0, 69)}...` : raw}`);
  }

  // -- storage addressing ---------------------------------------------------

  /** `u.<offset member> + <row|j>` -- the element index of a field access. */
  private indexOf(a: FieldAccess | FieldRef): string {
    const member = this.names.field.get(accessKey(a));
    if (member === undefined) {
      invalid(this.ir, `field "${this.ir.components[a.component].name}.${a.field}" is not in ir.reads/ir.writes`);
    }
    return `u.${member} + ${ROLE_INDEX[a.role]}`;
  }

  /** Reads a field as an f32, converting from the storage view's type. */
  private loadField(f: FieldRef): string {
    const view = viewOf(this.ir, f);
    if (!this.plan.raw) {
      const slot = `${VIEW_VAR[view]}[${this.indexOf(f)}]`;
      return view === 'f32' ? slot : `f32(${slot})`;
    }
    // One u32 binding: reinterpret the 4 bytes, then widen to the f32 the
    // subset computes in. `bitcast` compiles to nothing on Metal.
    const slot = `${VIEW_VAR.u32}[${this.indexOf(f)}]`;
    if (view === 'u32') return `f32(${slot})`;
    if (view === 'f32') return `bitcast<f32>(${slot})`;
    return `f32(bitcast<i32>(${slot}))`;
  }

  /**
   * Stores an f32 expression into a field. Integer fields go through
   * `cozy_store_i32`/`cozy_store_u32` (parity rule 2 in the file header):
   * NaN -> 0, saturate to the type's true range, else round half toward
   * +Infinity -- never the truncate-and-wrap of a bare TypedArray store.
   */
  private storeField(f: FieldRef, value: string): string {
    const view = viewOf(this.ir, f);
    const slot = `${VIEW_VAR[this.plan.raw ? 'u32' : view]}[${this.indexOf(f)}]`;
    const stored = view === 'f32' ? value : `cozy_store_${view}(${value})`;
    // Through the single raw u32 binding, f32 and i32 bit patterns are
    // reinterpreted; a u32 needs nothing.
    const bits = this.plan.raw && view !== 'u32' ? `bitcast<u32>(${stored})` : stored;
    return `${slot} = ${bits};`;
  }

  // -- expressions ----------------------------------------------------------

  expr(e: Expr): string {
    switch (e.kind) {
      case 'num':
        return f32Literal(this.ir, e.value, e.span);
      case 'boollit':
        return e.value ? 'true' : 'false';
      case 'field':
        return this.loadField(e);
      case 'uniform': {
        const member = this.names.uniform.get(e.name);
        if (member === undefined) invalid(this.ir, `uniform "${e.name}" is not declared`, e.span);
        return `u.${member}`;
      }
      case 'local':
        return localName(this.ir, e.id);
      case 'builtinValue':
        // `row` and `u.count` are u32; the subset is all-f32, so convert here.
        return e.name === 'dt' ? 'u.dt' : e.name === 'index' ? 'f32(row)' : 'f32(u.count)';
      case 'unary':
        return e.op === '!' ? `!(${this.expr(e.arg)})` : `-(${this.expr(e.arg)})`;
      case 'binary':
      case 'logical':
        // Fully parenthesized: JS and WGSL precedence differ in places (WGSL
        // does not even allow unparenthesized mixed `&&`/`||`), and the IR
        // already carries the tree the user wrote.
        return `(${this.expr(e.left)} ${e.op} ${this.expr(e.right)})`;
      case 'cond':
        // WGSL has no `?:`. `select(false, true, cond)` evaluates both arms,
        // which the subset makes unobservable: expressions have no effects.
        return `select(${this.expr(e.alt)}, ${this.expr(e.then)}, ${this.expr(e.test)})`;
      case 'call': {
        const info = BUILTINS[e.callee];
        if (!info) invalid(this.ir, `unknown builtin "${String(e.callee)}"`, e.span);
        // Emitted through the shared template in ./ir so the two backends
        // cannot drift apart.
        return info.wgsl(e.args.map((a) => this.expr(a)));
      }
      default:
        return invalid(this.ir, `unknown expression node "${(e as { kind: string }).kind}"`);
    }
  }

  // -- statements -----------------------------------------------------------

  /** Emits the statements of a block WITHOUT braces of its own. */
  blockBody(b: BlockStmt, depth: number): void {
    for (const s of b.body) this.stmt(s, depth);
  }

  /** Emits a block as a braced WGSL compound statement (its own scope). */
  private braced(b: BlockStmt, depth: number, head: string, tail = '}'): void {
    this.line(depth, head);
    this.blockBody(b, depth + 1);
    this.line(depth, tail);
  }

  stmt(s: Stmt, depth: number): void {
    switch (s.kind) {
      case 'block':
        this.braced(s, depth, '{');
        return;

      case 'decl': {
        this.note(depth, s.span);
        const l = this.ir.locals[s.id];
        if (!l) invalid(this.ir, `declaration of local #${s.id}, which does not exist`, s.span);
        // `var` is assignable, `let` is not -- exactly JS `let` vs `const`.
        const keyword = l.mutable ? 'var' : 'let';
        this.line(depth, `${keyword} ${localName(this.ir, s.id)}: ${l.type} = ${this.expr(s.init)};`);
        return;
      }

      case 'assign': {
        this.note(depth, s.span);
        if (s.target.kind === 'local') {
          // WGSL has the same compound assignment operators, including `%=`.
          this.line(depth, `${localName(this.ir, s.target.id)} ${s.op} ${this.expr(s.value)};`);
          return;
        }
        const target = s.target;
        if (target.role !== 0) {
          // Every pair is visited from both sides, so a write to `other` races
          // with itself. parse.ts rejects this; belt and braces.
          invalid(
            this.ir,
            `kernel writes to "other.${target.field}"; a pairwise kernel may only write through "self"`,
            s.span,
          );
        }
        const value =
          s.op === '='
            ? this.expr(s.value)
            : `(${this.loadField(target)} ${s.op.slice(0, 1)} ${this.expr(s.value)})`;
        this.line(depth, this.storeField(target, value));
        return;
      }

      case 'if':
        this.note(depth, s.span);
        this.line(depth, `if (${this.expr(s.test)}) {`);
        this.blockBody(s.then, depth + 1);
        if (s.alt) {
          this.line(depth, '} else {');
          this.blockBody(s.alt, depth + 1);
        }
        this.line(depth, '}');
        return;

      case 'for': {
        this.note(depth, s.span);
        // A WGSL `for` cannot carry the trip guard, so every loop becomes a
        // `loop` with the guard as its first statement: a kernel bug can cost
        // `maxIterations` iterations, never a hung device or a TDR.
        const trip = `cozy_trip${this.trips++}`;
        const cap = tripCap(this.ir, s.maxIterations, s.span);
        this.line(depth, '{');
        if (s.init) this.stmt(s.init, depth + 1);
        this.line(depth + 1, `var ${trip}: u32 = 0u;`);
        this.line(depth + 1, 'loop {');
        this.line(depth + 2, `if (${trip} >= ${cap}u) { break; }`);
        this.line(depth + 2, `${trip} = ${trip} + 1u;`);
        if (s.test) this.line(depth + 2, `if (!(${this.expr(s.test)})) { break; }`);
        this.blockBody(s.body, depth + 2);
        if (s.update) {
          // `continuing` is where a `continue` lands, which is what makes the
          // update run for a `continue`d iteration, as in JS.
          this.line(depth + 2, 'continuing {');
          this.stmt(s.update, depth + 3);
          this.line(depth + 2, '}');
        }
        this.line(depth + 1, '}');
        this.line(depth, '}');
        return;
      }

      case 'while': {
        this.note(depth, s.span);
        const trip = `cozy_trip${this.trips++}`;
        const cap = tripCap(this.ir, s.maxIterations, s.span);
        this.line(depth, '{');
        this.line(depth + 1, `var ${trip}: u32 = 0u;`);
        this.line(depth + 1, 'loop {');
        this.line(depth + 2, `if (${trip} >= ${cap}u) { break; }`);
        this.line(depth + 2, `${trip} = ${trip} + 1u;`);
        this.line(depth + 2, `if (!(${this.expr(s.test)})) { break; }`);
        this.blockBody(s.body, depth + 2);
        this.line(depth + 1, '}');
        this.line(depth, '}');
        return;
      }

      case 'break':
        this.line(depth, 'break;');
        return;

      case 'continue':
        this.line(depth, 'continue;');
        return;

      default:
        invalid(this.ir, `unknown statement node "${(s as { kind: string }).kind}"`);
    }
  }

  // -- module ---------------------------------------------------------------

  /** `struct U { ... }` -- one 4-byte scalar per `uniformLayout` member. */
  uniformStruct(layout: UniformLayout): void {
    this.line(0, 'struct U {');
    this.line(1, `dt: f32,          // offset ${layout.dt.offset}`);
    this.line(1, `count: u32,       // offset ${layout.count.offset}  rows in the chunk`);
    this.line(1, `base: u32,        // offset ${layout.base.offset}  first row of this dispatch`);
    this.line(1, `countOther: u32,  // offset ${layout.countOther.offset}`);
    for (const m of layout.fields) {
      const member = this.names.field.get(accessKey(m.access));
      this.line(1, `${member}: u32,  // offset ${m.offset}  ${this.ir.components[m.access.component].name}.${m.access.field} base, in elements`);
    }
    for (const m of layout.uniforms) {
      this.line(1, `${this.names.uniform.get(m.name)}: f32,  // offset ${m.offset}  uniforms.${m.name}`);
    }
    // No trailing `;`: a struct declaration in current WGSL does not take one.
    this.line(0, '}');
  }
}

/** The view an f32/i32/u32 field is addressed through, or a hard error. */
function viewOf(ir: KernelIR, f: { fieldKind: IRFieldKind; field: string; component: number; span?: Span }): StorageViewType {
  switch (f.fieldKind) {
    case 'f32':
    case 'i32':
    case 'u32':
      return f.fieldKind;
    default:
      return invalid(
        ir,
        `field "${ir.components[f.component].name}.${f.field}" is ${f.fieldKind}, which the GPU backend cannot address; ` +
          'checkGPUSupport() should have kept this kernel on the CPU backend',
        f.span,
      );
  }
}

/** Static trip cap of a loop, as a u32 literal body. */
function tripCap(ir: KernelIR, maxIterations: number, span?: Span): number {
  const n = Math.floor(maxIterations);
  if (!(n > 0) || !Number.isFinite(n)) invalid(ir, `loop has no usable iteration cap (${maxIterations})`, span);
  // 2^32-1 is the largest u32; a cap that big is already "forever".
  return Math.min(n, 4294967295);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Decides whether `ir` can run on a device with `caps`, WITHOUT generating
 * anything. runtime.ts calls this once per kernel; a `false` result becomes a
 * one-time `console.warn` and a permanent CPU target -- never a throw.
 *
 * Reasons v1 reports:
 *  - a touched field whose kind has no 4-byte view (`gpuBlockers(ir)`): f64,
 *    i8/u8/i16/u16, bool, str  -> E_DEVICE_LIMIT
 *  - `views.length + 1 > caps.maxStorageBuffersPerShaderStage` (cannot happen:
 *    a kernel declares at most one storage binding, but checked anyway;
 *    the `+ 1` is deliberately conservative -- the uniform block is charged
 *    against `maxUniformBuffersPerShaderStage`, not this budget)
 *  - `uniformLayout(ir).size > 65536`  -> E_DEVICE_LIMIT
 *  - `opts.workgroupSize > caps.maxComputeWorkgroupSizeX` or
 *    `> caps.maxComputeInvocationsPerWorkgroup`
 *  - a write to a role-1 (`other`) field in a pairwise kernel, which would race
 *    -> E_INVALID_IR, reported here rather than thrown so a parser bug costs a
 *    fallback instead of a crash inside `world.update()`.
 *
 * Note it reads DEVICE limits (`GPUCapabilities` is built from `device.limits`),
 * never adapter maxima: a device is granted only the limits it asked for.
 *
 * Never throws: a malformed IR is itself a reason.
 */
export function checkGPUSupport(ir: KernelIR, caps: GPUCapabilities, opts?: WGSLOptions): GPUSupport {
  const reasons: { code: KernelErrorCode; message: string }[] = [];
  const add = (code: KernelErrorCode, message: string): void => {
    reasons.push({ code, message });
  };
  try {
    for (const b of gpuBlockers(ir)) {
      add(
        'E_DEVICE_LIMIT',
        `field "${b.component}.${b.field}" is ${b.kind}, which has no 4-byte GPU view ` +
          '(f64 is 8 bytes; i8/i16/u8/u16/bool are sub-word; str is an interned id)',
      );
    }

    const views = bindingPlan(ir).views;
    if (views.length + 1 > caps.maxStorageBuffersPerShaderStage) {
      add(
        'E_DEVICE_LIMIT',
        `kernel needs ${views.length} storage bindings, device grants ${caps.maxStorageBuffersPerShaderStage} per compute stage`,
      );
    }

    const layout = uniformLayout(ir);
    if (layout.size > 65536) {
      add('E_DEVICE_LIMIT', `uniform block is ${layout.size} bytes, over the 65536-byte binding limit`);
    }

    const wg = resolveWorkgroupSize(opts);
    if (!Number.isInteger(wg) || wg < 1) {
      add('E_DEVICE_LIMIT', `workgroupSize must be a positive integer, got ${String(opts && opts.workgroupSize)}`);
    } else {
      if (wg > caps.maxComputeWorkgroupSizeX) {
        add('E_DEVICE_LIMIT', `workgroup_size(${wg}) over the device's maxComputeWorkgroupSizeX of ${caps.maxComputeWorkgroupSizeX}`);
      }
      if (wg > caps.maxComputeInvocationsPerWorkgroup) {
        add(
          'E_DEVICE_LIMIT',
          `workgroup_size(${wg}) over the device's maxComputeInvocationsPerWorkgroup of ${caps.maxComputeInvocationsPerWorkgroup}`,
        );
      }
    }

    if (ir.form === 'pairwise') {
      for (const w of ir.writes) {
        if (w.role === 1) {
          add(
            'E_INVALID_IR',
            `kernel writes to "other.${w.field}"; every pair is evaluated from both sides, so that would race. ` +
              'Accumulate into "self" only.',
          );
        }
      }
    } else {
      for (const a of touchedFields(ir)) {
        if (a.role !== 0) add('E_INVALID_IR', `role-1 field "${a.field}" in a per-entity kernel`);
      }
    }
  } catch (e) {
    add('E_INVALID_IR', `kernel IR could not be inspected: ${(e as Error).message}`);
  }
  return { ok: reasons.length === 0, reasons };
}

/**
 * Generates the compute module for `ir`.
 *
 * CONTRACT
 *  - Pure and deterministic: same (ir, opts) gives a byte-identical `code`.
 *    Tests snapshot it.
 *  - Calls `validateIR(ir)` first; a bad IR throws E_INVALID_IR rather than
 *    emitting broken WGSL.
 *  - Never emits an unbounded loop: a `for`/`while` becomes a WGSL `loop` with
 *    an added `if (trip >= <maxIterations>) { break; }` guard, so a kernel bug
 *    cannot hang the device.
 *  - Bounds check before any access: `if (row >= u.count) { return; }`.
 *  - Reads/writes go through `u.<offsetMemberName(ir, access)> + row`; the
 *    generator never computes a byte offset itself.
 *  - Emits {@link COZY_RAND_WGSL} verbatim when `ir.usesRand`, and
 *    {@link COZY_STORE_WGSL} when the kernel writes an i32/u32 field, so both
 *    are bit-identical to their JS twins (PARITY CONTRACT, file header).
 *  - Pairwise (`ir.form === 'pairwise'`): the body is wrapped in
 *    `for (var j = 0u; j < u.countOther; j = j + 1u)` with role-1 field
 *    accesses indexed by `j` and role-0 by `row`, plus `if (j == row) { continue; }`.
 *    Writes to role-1 fields are rejected (E_INVALID_IR): they would race.
 *  - The kernel's NAME is not in the output (only under `annotate`), so two
 *    kernels with the same body share one pipeline. See {@link wgslCacheKey}.
 *
 * @throws KernelError E_INVALID_IR, or E_DEVICE_LIMIT if called for an IR that
 *         {@link checkGPUSupport} rejects (callers check first).
 */
export function generateWGSL(ir: KernelIR, opts?: WGSLOptions): WGSLModule {
  validateIR(ir);

  const annotate = !!(opts && opts.annotate);
  const workgroupSize = resolveWorkgroupSize(opts);
  if (!Number.isInteger(workgroupSize) || workgroupSize < 1) {
    throw new KernelError('E_DEVICE_LIMIT', `workgroupSize must be a positive integer, got ${String(opts && opts.workgroupSize)}`, {
      kernelName: ir.name,
      source: ir.source,
    });
  }

  const plan = bindingPlan(ir);
  const views = plan.views;
  const layout = uniformLayout(ir);
  // NOT uniformBinding(ir): that counts the per-view-type bindings WebGPU
  // refuses to alias. See the file header.
  const binding = views.length;
  const w = new WGSLWriter(ir, annotate, plan);

  // --- header --------------------------------------------------------------
  emitHeader(w, ir, annotate, views, workgroupSize);

  // --- uniform block -------------------------------------------------------
  w.uniformStruct(layout);
  w.blank();

  // --- bindings ------------------------------------------------------------
  for (let i = 0; i < views.length; i++) {
    const note = plan.raw ? '  // every field bitcast through one u32 view (no writable aliasing)' : '';
    w.line(0, `@group(0) @binding(${i}) var<storage, read_write> ${VIEW_VAR[views[i]]}: array<${views[i]}>;${note}`);
  }
  w.line(0, `@group(0) @binding(${binding}) var<uniform> u: U;`);
  w.blank();

  // --- helpers -------------------------------------------------------------
  if (ir.usesRand) {
    for (const line of COZY_RAND_WGSL.split('\n')) w.line(0, line);
    w.blank();
  }
  if (writesIntegerField(ir)) {
    for (const line of COZY_STORE_WGSL.split('\n')) w.line(0, line);
    w.blank();
  }

  // --- entry point ---------------------------------------------------------
  w.line(0, `@compute @workgroup_size(${workgroupSize})`);
  w.line(0, 'fn main(@builtin(global_invocation_id) gid: vec3<u32>) {');
  w.line(1, 'let row: u32 = gid.x + u.base;');
  w.line(1, 'if (row >= u.count) { return; }');
  if (ir.form === 'pairwise') {
    w.line(1, 'for (var j: u32 = 0u; j < u.countOther; j = j + 1u) {');
    w.line(2, 'if (j == row) { continue; }');
    w.blockBody(ir.body, 2);
    w.line(1, '}');
  } else {
    w.blockBody(ir.body, 1);
  }
  w.line(0, '}');

  return {
    code: `${w.text()}\n`,
    entryPoint: 'main',
    workgroupSize,
    views,
    uniformBinding: binding,
    layout,
    cacheKey: wgslCacheKey(ir, opts),
  };
}

/**
 * The `cacheKey` of the module `generateWGSL(ir, opts)` would produce, without
 * generating it. Must be cheap: the runtime calls it on every archetype the
 * first time it dispatches. Two IRs with the same key MUST produce identical
 * code -- derive it from the fields the generator actually reads (form,
 * components' field kinds, uniform names, the body, workgroup size), not from
 * `ir.name` or `ir.source`.
 *
 * It is an exact canonical rendering of those inputs, not a hash: a hash
 * collision here would silently run the WRONG shader, and the cache holds one
 * entry per distinct kernel body, so the few hundred bytes are free.
 *
 * `ir.name` is deliberately absent, which is also why the generated code never
 * contains it -- except under `annotate`, which is therefore part of the key.
 */
export function wgslCacheKey(ir: KernelIR, opts?: WGSLOptions): string {
  const annotate = !!(opts && opts.annotate);
  const parts: string[] = [
    `cozy-wgsl/${IR_VERSION}.${WGSL_CODEGEN_REVISION}`,
    `form=${ir.form}`,
    `wg=${resolveWorkgroupSize(opts)}`,
    `ann=${annotate ? (ir.source ? `1:${ir.name}` : '1') : '0'}`,
    // Component names reach the output through the uniform member names, and
    // field kinds decide the view and the conversions.
    `c=[${ir.components.map((c) => `${c.name}(${c.fields.map((f) => `${f.name}:${f.kind}`).join(',')})`).join('|')}]`,
    `u=[${ir.uniforms.map((x) => x.name).join(',')}]`,
    `l=[${ir.locals.map((l) => `${l.name}:${l.type}${l.mutable ? '*' : ''}`).join(',')}]`,
    `rand=${ir.usesRand ? 1 : 0}`,
    `body=${canonicalStmt(ir.body)}`,
  ];
  return parts.join(';');
}

/** Injective textual rendering of a statement tree. Cache-key use only. */
function canonicalStmt(s: Stmt): string {
  switch (s.kind) {
    case 'block':
      return `{${s.body.map(canonicalStmt).join('')}}`;
    case 'decl':
      return `d${s.id}=${canonicalExpr(s.init)};`;
    case 'assign':
      return `a(${canonicalExpr(s.target)})${s.op}(${canonicalExpr(s.value)});`;
    case 'if':
      return `if(${canonicalExpr(s.test)})${canonicalStmt(s.then)}${s.alt ? `else${canonicalStmt(s.alt)}` : ''}`;
    case 'for':
      return `for(${s.init ? canonicalStmt(s.init) : ''};${s.test ? canonicalExpr(s.test) : ''};${
        s.update ? canonicalStmt(s.update) : ''
      };max${s.maxIterations})${canonicalStmt(s.body)}`;
    case 'while':
      return `while(${canonicalExpr(s.test)};max${s.maxIterations})${canonicalStmt(s.body)}`;
    case 'break':
      return 'brk;';
    case 'continue':
      return 'cnt;';
    default:
      return `?${(s as { kind: string }).kind};`;
  }
}

/** Injective textual rendering of an expression tree. Cache-key use only. */
function canonicalExpr(e: Expr): string {
  switch (e.kind) {
    case 'num':
      // The emitted literal is the f32, so two IRs differing below f32
      // precision really do produce the same code.
      return `n${Object.is(Math.fround(e.value), -0) ? '-0' : String(Math.fround(e.value))}`;
    case 'boollit':
      return e.value ? 'true' : 'false';
    case 'field':
      return `f${e.component}.${e.field}:${e.fieldKind}@${e.role}`;
    case 'uniform':
      return `u.${e.name}`;
    case 'local':
      return `v${e.id}`;
    case 'builtinValue':
      return `b.${e.name}`;
    case 'unary':
      return `${e.op}(${canonicalExpr(e.arg)})`;
    case 'binary':
    case 'logical':
      return `(${canonicalExpr(e.left)}${e.op}${canonicalExpr(e.right)})`;
    case 'cond':
      return `(${canonicalExpr(e.test)}?${canonicalExpr(e.then)}:${canonicalExpr(e.alt)})`;
    case 'call':
      return `${e.callee}(${e.args.map(canonicalExpr).join(',')})`;
    default:
      return `?${(e as { kind: string }).kind}`;
  }
}

/** True when the kernel assigns an i32/u32 field, i.e. needs {@link COZY_STORE_WGSL}. */
function writesIntegerField(ir: KernelIR): boolean {
  return ir.writes.some((w) => w.fieldKind === 'i32' || w.fieldKind === 'u32');
}

function resolveWorkgroupSize(opts?: WGSLOptions): number {
  const wg = opts && opts.workgroupSize;
  return wg === undefined || wg === null ? WORKGROUP_SIZE : wg;
}

function emitHeader(
  w: WGSLWriter,
  ir: KernelIR,
  annotate: boolean,
  views: readonly StorageViewType[],
  workgroupSize: number,
): void {
  w.line(0, '// Generated by cozyecs/gpu -- do not edit.');
  w.line(
    0,
    `// form: ${ir.form}, storage: ${views.join('+') || 'none'}, workgroup_size: ${workgroupSize}, ops: ${ir.opCount}`,
  );
  if (annotate) {
    w.line(0, `// kernel: ${ir.name.replace(/[\r\n]+/g, ' ')}`);
    for (const line of ir.source.split('\n')) w.line(0, `//   ${line.replace(/\r/g, '')}`);
  }
  w.blank();
}
