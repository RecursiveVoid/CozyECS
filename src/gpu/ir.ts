/**
 * Kernel IR: the contract between the parser and the two backends.
 *
 *   kernel.toString()  --parse.ts-->  KernelIR  --wgsl.ts--> WGSL + bindings
 *                                             \--cpu.ts---> compiled JS chunk loop
 *
 * Nothing here imports the core (`src/*.ts`) or WebGPU. The IR is plain data:
 * structurally cloneable, JSON-serializable (drop `source` for size) and safe
 * to snapshot in tests. Both backends are pure functions of it, which is what
 * makes CPU/GPU parity testable.
 *
 * Design rules that the whole module depends on:
 *
 *  1. ARITHMETIC IS f32. Every expression is `f32` or `bool`; there is no
 *     integer arithmetic in the subset. An `i32`/`u32`/`i16`/... field is read
 *     as f32 and written back with round-to-nearest + clamp to the field's
 *     range. This removes the type lattice from both backends and makes
 *     CPU/GPU results comparable (see docs/GPU.md, "parity contract").
 *
 *  2. PARAMETERS ARE POSITIONAL. Minifiers rename identifiers and closure
 *     variables are invisible to `Function.prototype.toString`, so a kernel's
 *     component parameters are matched by position against `components`, never
 *     by name. Field names (`p.x`) are property names and do survive.
 *
 *  3. FIELD REFERENCES ARE INDICES. `FieldRef.component` indexes
 *     `KernelIR.components`, not `ComponentType.id`. The backends never see a
 *     ComponentType, so they never need the core.
 *
 *  4. ONE STORAGE BUFFER PER ARCHETYPE. CozyECS stores an archetype as a
 *     single ArrayBuffer with one typed-array view per field, so the GPU
 *     backend binds that buffer once per 4-byte view type and addresses fields
 *     through base offsets passed in the uniform block. {@link uniformLayout}
 *     and {@link storageViews} pin that layout so wgsl.ts and runtime.ts agree
 *     byte for byte without talking to each other.
 */

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

/**
 * Stable error codes. Every rejection a user can hit carries one; they are the
 * documented catalogue in docs/GPU.md and the assertion surface of the tests.
 * Never renumber or reuse a code.
 */
export type KernelErrorCode =
  // --- parse: shape of the kernel function -------------------------------
  /** `kernel` is not a function, or is native/bound so `toString` gives no body. */
  | 'E_NOT_A_FUNCTION'
  /** Parameter count is not components.length, +1 (dt) or +2 (dt, u). */
  | 'E_PARAM_COUNT'
  /** `async`, generator, or a getter/setter used as a kernel. */
  | 'E_KERNEL_ASYNC'
  /** The body could not be tokenized/parsed at all (syntax the subset parser does not know). */
  | 'E_SYNTAX'
  // --- parse: unsupported subset -----------------------------------------
  /** A free identifier that is not a parameter, a local, `Math` or a builtin. */
  | 'E_UNKNOWN_IDENTIFIER'
  /** String literal, template literal, regexp, object/array literal, `new`. */
  | 'E_UNSUPPORTED_LITERAL'
  /** A statement form outside the subset (try, switch, labeled break, do/while, return, ...). */
  | 'E_UNSUPPORTED_STATEMENT'
  /** An expression form outside the subset (bit ops, comma, optional chaining, `in`, ...). */
  | 'E_UNSUPPORTED_EXPRESSION'
  /** A call to something that is not a listed builtin (including Math.random and user functions). */
  | 'E_UNSUPPORTED_CALL'
  /** A property read/write that is not `<param>.<field>` or `u.<name>`. */
  | 'E_UNSUPPORTED_MEMBER'
  /** `for`/`while` whose iteration count cannot be bounded statically. */
  | 'E_UNBOUNDED_LOOP'
  /** Assignment to a `const` local, to `dt`, or to a uniform. */
  | 'E_ASSIGN_TO_CONST'
  /** A local used before its declaration, or redeclared in the same scope. */
  | 'E_BAD_LOCAL'
  // --- registration: kernel vs. world ------------------------------------
  /** `p.zz` where the component has no field `zz`. */
  | 'E_UNKNOWN_FIELD'
  /** `u.g` with no `g` in `uniforms`. */
  | 'E_UNKNOWN_UNIFORM'
  /** Two components in a pairwise kernel expose the same field name. */
  | 'E_FIELD_COLLISION'
  /** The declared `write` list omits a component the kernel assigns to. */
  | 'E_WRITE_NOT_DECLARED'
  /** A uniform value is not a finite number. */
  | 'E_BAD_UNIFORM_VALUE'
  // --- backends -----------------------------------------------------------
  /** The IR failed {@link validateIR}: a backend bug or a hand-built IR. */
  | 'E_INVALID_IR'
  /** The kernel needs more of the device than it grants (buffer size, dispatch size). */
  | 'E_DEVICE_LIMIT'
  /** `new Function` is unavailable (CSP). The CPU backend falls back to an interpreter-free path. */
  | 'E_NO_CODEGEN';

/** Half-open character range in the kernel source, for code frames. */
export interface Span {
  readonly start: number;
  readonly end: number;
}

/**
 * Every rejection thrown by the gpu entry. Registration-time problems throw
 * this; run-time capability problems (no device, unsupported field width)
 * warn once and fall back to the CPU backend instead of throwing.
 */
export class KernelError extends Error {
  readonly code: KernelErrorCode;
  /** Kernel name, as passed to `kernelSystem`. */
  readonly kernelName: string;
  /** Location in `source`, when the problem is a piece of syntax. */
  readonly span: Span | null;
  /** The kernel source the span refers to. */
  readonly source: string;

  constructor(code: KernelErrorCode, message: string, opts?: { kernelName?: string; span?: Span; source?: string; hint?: string }) {
    const kernelName = (opts && opts.kernelName) || '';
    const source = (opts && opts.source) || '';
    const span = (opts && opts.span) || null;
    let text = `[cozyecs/gpu] ${code}${kernelName ? ` in kernel "${kernelName}"` : ''}: ${message}`;
    if (span && source) text += `\n\n${codeFrame(source, span)}`;
    if (opts && opts.hint) text += `\n\nHint: ${opts.hint}`;
    super(text);
    this.name = 'KernelError';
    this.code = code;
    this.kernelName = kernelName;
    this.span = span;
    this.source = source;
  }
}

/**
 * Renders `source` around `span` as a two-line-context code frame with a caret
 * run under the offending characters. Tabs become single spaces so the caret
 * lines up. Used by KernelError; exported for tests and for backends that want
 * to point at a node.
 */
export function codeFrame(source: string, span: Span, context = 2): string {
  const text = source.replace(/\t/g, ' ');
  const lines = text.split('\n');
  const starts: number[] = [];
  for (let i = 0, at = 0; i < lines.length; i++) {
    starts.push(at);
    at += lines[i].length + 1;
  }
  let line = 0;
  while (line + 1 < starts.length && starts[line + 1] <= span.start) line++;
  const col = span.start - starts[line];
  const width = Math.max(1, Math.min(span.end - span.start, lines[line].length - col));
  const from = Math.max(0, line - context);
  const to = Math.min(lines.length - 1, line + context);
  const gutter = String(to + 1).length;
  const out: string[] = [];
  for (let i = from; i <= to; i++) {
    const n = String(i + 1).padStart(gutter, ' ');
    out.push(`${i === line ? '>' : ' '} ${n} | ${lines[i]}`);
    if (i === line) out.push(`  ${' '.repeat(gutter)} | ${' '.repeat(Math.max(0, col))}${'^'.repeat(width)}`);
  }
  return out.join('\n');
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Type of an IR expression. There is no integer arithmetic (design rule 1). */
export type ExprType = 'f32' | 'bool';

/**
 * Storage kind of a component field. Mirrors `FieldKind` in `src/types.ts`,
 * duplicated so the gpu entry never imports the core.
 */
export type IRFieldKind = 'f32' | 'f64' | 'i8' | 'i16' | 'i32' | 'u8' | 'u16' | 'u32' | 'bool' | 'str';

/** Per-kind storage facts both backends need. */
export interface FieldKindInfo {
  /** Bytes per element in the archetype table. */
  readonly bytes: number;
  /** The 4-byte WGSL view a field of this kind can be addressed through, or null if it cannot. */
  readonly view: StorageViewType | null;
  /** Inclusive value range after a write, or null for floats (no clamping). */
  readonly range: readonly [number, number] | null;
  /** Writes are rounded to the nearest integer before clamping. */
  readonly integer: boolean;
}

/** The 4-byte typed views the GPU backend can bind an archetype table through. */
export type StorageViewType = 'f32' | 'i32' | 'u32';

/**
 * Storage facts per field kind. `view === null` means the GPU backend cannot
 * address the field (f64 is 8 bytes; i8/u8/i16/u16/bool are sub-word; str is
 * an interned id that means nothing on the GPU). A kernel touching such a
 * field is CPU-only -- runtime.ts warns once and keeps the CPU backend, it
 * does not throw. See docs/GPU.md, "GPU-eligible fields".
 */
export const FIELD_KINDS: Readonly<Record<IRFieldKind, FieldKindInfo>> = Object.freeze({
  f32: { bytes: 4, view: 'f32', range: null, integer: false },
  f64: { bytes: 8, view: null, range: null, integer: false },
  i8: { bytes: 1, view: null, range: [-128, 127], integer: true },
  i16: { bytes: 2, view: null, range: [-32768, 32767], integer: true },
  i32: { bytes: 4, view: 'i32', range: [-2147483648, 2147483647], integer: true },
  u8: { bytes: 1, view: null, range: [0, 255], integer: true },
  u16: { bytes: 2, view: null, range: [0, 65535], integer: true },
  u32: { bytes: 4, view: 'u32', range: [0, 4294967295], integer: true },
  bool: { bytes: 1, view: null, range: [0, 1], integer: true },
  str: { bytes: 4, view: null, range: [0, 4294967295], integer: true },
});

/** True when a kernel touching this field kind can run on the GPU backend. */
export function isGPUEligible(kind: IRFieldKind): boolean {
  return FIELD_KINDS[kind].view !== null;
}

// ---------------------------------------------------------------------------
// Builtins
// ---------------------------------------------------------------------------

/** Name of a builtin callable inside a kernel. */
export type BuiltinName =
  | 'sin' | 'cos' | 'tan' | 'asin' | 'acos' | 'atan' | 'atan2'
  | 'sqrt' | 'abs' | 'min' | 'max' | 'floor' | 'ceil' | 'round'
  | 'pow' | 'exp' | 'log' | 'sign' | 'hypot' | 'rand';

/** Name of a builtin *value* usable as a bare identifier inside a kernel. */
export type BuiltinValueName = 'index' | 'count';

export interface BuiltinInfo {
  /** Accepted argument counts (inclusive range). */
  readonly arity: readonly [number, number];
  /** Written as `Math.<name>` in the kernel (all except `rand`). */
  readonly onMath: boolean;
  /**
   * Emits the WGSL for a call. `args` are already-emitted f32 expressions.
   * `rand` expands to the helper wgsl.ts injects (see {@link RAND_WGSL}).
   */
  readonly wgsl: (args: readonly string[]) => string;
  /** Emits the JS for a call in the compiled CPU loop. Mirrors `wgsl` exactly. */
  readonly js: (args: readonly string[]) => string;
}

const b = (
  min: number,
  max: number,
  onMath: boolean,
  wgsl: (a: readonly string[]) => string,
  js: (a: readonly string[]) => string,
): BuiltinInfo => Object.freeze({ arity: [min, max] as const, onMath, wgsl, js });

const simple = (name: string, min: number, max = min): BuiltinInfo =>
  b(min, max, true, (a) => `${name}(${a.join(', ')})`, (a) => `Math.${name}(${a.join(', ')})`);

/**
 * The single source of truth for what a kernel may call and how each backend
 * emits it. parse.ts validates against `arity`/`onMath`; wgsl.ts and cpu.ts
 * emit through the templates, so the two backends cannot drift apart.
 *
 * `sign`: WGSL `sign` and `Math.sign` agree on -0 and NaN closely enough for
 * the f32 tolerance in docs/GPU.md.
 * `log` is the natural log in both. `pow(x, y)` with negative `x` is undefined
 * in WGSL -- documented, not policed.
 */
export const BUILTINS: Readonly<Record<BuiltinName, BuiltinInfo>> = Object.freeze({
  sin: simple('sin', 1),
  cos: simple('cos', 1),
  tan: simple('tan', 1),
  asin: simple('asin', 1),
  acos: simple('acos', 1),
  atan: simple('atan', 1),
  atan2: b(2, 2, true, (a) => `atan2(${a[0]}, ${a[1]})`, (a) => `Math.atan2(${a[0]}, ${a[1]})`),
  sqrt: simple('sqrt', 1),
  abs: simple('abs', 1),
  min: b(2, 2, true, (a) => `min(${a[0]}, ${a[1]})`, (a) => `Math.min(${a[0]}, ${a[1]})`),
  max: b(2, 2, true, (a) => `max(${a[0]}, ${a[1]})`, (a) => `Math.max(${a[0]}, ${a[1]})`),
  floor: simple('floor', 1),
  ceil: simple('ceil', 1),
  round: b(1, 1, true, (a) => `floor(${a[0]} + 0.5)`, (a) => `Math.floor(${a[0]} + 0.5)`),
  pow: b(2, 2, true, (a) => `pow(${a[0]}, ${a[1]})`, (a) => `Math.pow(${a[0]}, ${a[1]})`),
  exp: simple('exp', 1),
  log: simple('log', 1),
  sign: simple('sign', 1),
  hypot: b(2, 2, true, (a) => `length(vec2<f32>(${a[0]}, ${a[1]}))`, (a) => `Math.hypot(${a[0]}, ${a[1]})`),
  rand: b(1, 1, false, (a) => `cozy_rand(${a[0]})`, (a) => `cozy_rand(${a[0]})`),
});

/**
 * `Math.round` is deliberately NOT `Math.round`: WGSL `round` is round-half-to-even
 * and JS `Math.round` is round-half-up, so both backends emit `floor(x + 0.5)`
 * and agree. Kept as a named export so the docs and tests can point at it.
 */
export const ROUND_IS_FLOOR_PLUS_HALF = true;

/**
 * `rand(seed)`: a stateless hash, not a stream. Same seed, same value, on both
 * backends, bit for bit -- it is the one part of the parity contract that is
 * exact, because it is pure u32 integer arithmetic. Seed it per entity, e.g.
 * `rand(index + u.frame * 7919)`.
 */
export const RAND_WGSL = `fn cozy_rand(seed: f32) -> f32 {
  var h: u32 = bitcast<u32>(i32(seed)) ^ 0x9e3779b9u;
  h = (h ^ (h >> 16u)) * 0x21f0aaadu;
  h = (h ^ (h >> 15u)) * 0x735a2d97u;
  h = h ^ (h >> 15u);
  return f32(h >> 8u) * 5.9604645e-8;
}`;

/** The JS twin of {@link RAND_WGSL}. cpu.ts injects it into the compiled loop's scope. */
export function cozy_rand(seed: number): number {
  let h = ((seed | 0) ^ 0x9e3779b9) >>> 0;
  h = Math.imul(h ^ (h >>> 16), 0x21f0aaad) >>> 0;
  h = Math.imul(h ^ (h >>> 15), 0x735a2d97) >>> 0;
  h = (h ^ (h >>> 15)) >>> 0;
  return (h >>> 8) * 5.9604645e-8;
}

/** Builtin values readable as bare identifiers: the row index and the chunk row count. */
export const BUILTIN_VALUES: Readonly<Record<BuiltinValueName, ExprType>> = Object.freeze({
  index: 'f32',
  count: 'f32',
});

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------

export interface NumLit {
  readonly kind: 'num';
  readonly value: number;
  readonly type: 'f32';
  readonly span?: Span;
}

export interface BoolLit {
  readonly kind: 'boollit';
  readonly value: boolean;
  readonly type: 'bool';
  readonly span?: Span;
}

/**
 * A component field access. `component` indexes {@link KernelIR.components};
 * `role` is 0 for the per-entity row (and for `self` in a pairwise kernel) and
 * 1 for `other`.
 */
export interface FieldRef {
  readonly kind: 'field';
  readonly component: number;
  readonly field: string;
  readonly fieldKind: IRFieldKind;
  readonly role: 0 | 1;
  readonly type: 'f32';
  readonly span?: Span;
}

/** `u.<name>`. Every uniform is an f32 (design rule 1). */
export interface UniformRef {
  readonly kind: 'uniform';
  readonly name: string;
  readonly type: 'f32';
  readonly span?: Span;
}

/** A `const`/`let` declared inside the kernel. `id` indexes {@link KernelIR.locals}. */
export interface LocalRef {
  readonly kind: 'local';
  readonly id: number;
  readonly type: ExprType;
  readonly span?: Span;
}

/** `dt`, or one of {@link BUILTIN_VALUES}. */
export interface BuiltinRef {
  readonly kind: 'builtinValue';
  readonly name: 'dt' | BuiltinValueName;
  readonly type: 'f32';
  readonly span?: Span;
}

export interface UnaryExpr {
  readonly kind: 'unary';
  readonly op: '-' | '!';
  readonly arg: Expr;
  readonly type: ExprType;
  readonly span?: Span;
}

/** Arithmetic (`f32`) and comparison (`bool`) operators. */
export type BinaryOp = '+' | '-' | '*' | '/' | '%' | '<' | '<=' | '>' | '>=' | '==' | '!=';

export interface BinaryExpr {
  readonly kind: 'binary';
  readonly op: BinaryOp;
  readonly left: Expr;
  readonly right: Expr;
  readonly type: ExprType;
  readonly span?: Span;
}

/**
 * `&&` / `||`. Both backends evaluate BOTH sides: the subset has no side
 * effects inside an expression, so short-circuiting is unobservable, and WGSL's
 * `&&`/`||` are short-circuiting anyway. Documented for completeness.
 */
export interface LogicalExpr {
  readonly kind: 'logical';
  readonly op: '&&' | '||';
  readonly left: Expr;
  readonly right: Expr;
  readonly type: 'bool';
  readonly span?: Span;
}

export interface CondExpr {
  readonly kind: 'cond';
  readonly test: Expr;
  readonly then: Expr;
  readonly alt: Expr;
  readonly type: ExprType;
  readonly span?: Span;
}

export interface CallExpr {
  readonly kind: 'call';
  readonly callee: BuiltinName;
  readonly args: readonly Expr[];
  readonly type: 'f32';
  readonly span?: Span;
}

export type Expr =
  | NumLit
  | BoolLit
  | FieldRef
  | UniformRef
  | LocalRef
  | BuiltinRef
  | UnaryExpr
  | BinaryExpr
  | LogicalExpr
  | CondExpr
  | CallExpr;

// ---------------------------------------------------------------------------
// Statements
// ---------------------------------------------------------------------------

export interface BlockStmt {
  readonly kind: 'block';
  readonly body: readonly Stmt[];
  readonly span?: Span;
}

export interface DeclStmt {
  readonly kind: 'decl';
  readonly id: number;
  readonly init: Expr;
  readonly span?: Span;
}

/** `=` or a compound assignment. `%=` is included; bitwise compounds are not. */
export type AssignOp = '=' | '+=' | '-=' | '*=' | '/=' | '%=';

export interface AssignStmt {
  readonly kind: 'assign';
  readonly target: FieldRef | LocalRef;
  readonly op: AssignOp;
  readonly value: Expr;
  readonly span?: Span;
}

export interface IfStmt {
  readonly kind: 'if';
  readonly test: Expr;
  readonly then: BlockStmt;
  readonly alt: BlockStmt | null;
  readonly span?: Span;
}

/**
 * A counted loop. `maxIterations` is a static upper bound the parser proved (or
 * the user pinned); the GPU backend emits it as the loop's trip cap so a shader
 * can never hang the device, and the CPU backend emits the same cap so both
 * backends stop at the same place.
 */
export interface ForStmt {
  readonly kind: 'for';
  readonly init: DeclStmt | null;
  readonly test: Expr | null;
  readonly update: AssignStmt | null;
  readonly body: BlockStmt;
  readonly maxIterations: number;
  readonly span?: Span;
}

export interface WhileStmt {
  readonly kind: 'while';
  readonly test: Expr;
  readonly body: BlockStmt;
  readonly maxIterations: number;
  readonly span?: Span;
}

/** Unlabeled only. Labeled break/continue are rejected (E_UNSUPPORTED_STATEMENT). */
export interface BreakStmt {
  readonly kind: 'break';
  readonly span?: Span;
}

export interface ContinueStmt {
  readonly kind: 'continue';
  readonly span?: Span;
}

export type Stmt = BlockStmt | DeclStmt | AssignStmt | IfStmt | ForStmt | WhileStmt | BreakStmt | ContinueStmt;

export type Node = Expr | Stmt;

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

/**
 * Per-entity: `kernel(c0, c1, ..., dt?, u?)`, one parameter per component.
 * Pairwise: `kernel(self, other, dt?, u?)`, where `self`/`other` expose the
 * fields of every component in `components` merged into one namespace (a field
 * name used by two components is E_FIELD_COLLISION).
 */
export type KernelForm = 'per-entity' | 'pairwise';

/** One field of a component, as the backends see it. */
export interface IRField {
  readonly name: string;
  readonly kind: IRFieldKind;
  /** Index of the field in the component's schema order. */
  readonly index: number;
}

/** A component the kernel operates on. Carries no reference to the core's ComponentType. */
export interface IRComponent {
  /** Index in {@link KernelIR.components}; equals the kernel parameter index for per-entity. */
  readonly index: number;
  /** `ComponentType.id`, so runtime.ts can look the column up on a chunk. */
  readonly id: number;
  /** `ComponentType.name`, for messages only. */
  readonly name: string;
  readonly fields: readonly IRField[];
}

/** A `uniforms` entry. Order is declaration order and fixes the uniform buffer layout. */
export interface IRUniform {
  readonly name: string;
  readonly index: number;
  /** The value at registration time. runtime.ts may overwrite it per frame. */
  readonly initial: number;
}

/** A `const`/`let` inside the kernel. */
export interface IRLocal {
  readonly id: number;
  /** Source name; used verbatim in generated code after mangling to `l<id>_<name>`. */
  readonly name: string;
  readonly type: ExprType;
  readonly mutable: boolean;
}

/** A unique (component, field, role) touched by the kernel. */
export interface FieldAccess {
  readonly component: number;
  readonly field: string;
  readonly fieldKind: IRFieldKind;
  readonly role: 0 | 1;
}

/** Bump when the IR shape changes in a way a backend must notice. */
export const IR_VERSION = 1;

export interface KernelIR {
  readonly version: typeof IR_VERSION;
  /** The `name` passed to `kernelSystem`. Appears in errors, WGSL comments and system names. */
  readonly name: string;
  readonly form: KernelForm;
  readonly components: readonly IRComponent[];
  readonly uniforms: readonly IRUniform[];
  /** Index-addressed by `LocalRef.id`; `locals[i].id === i`. */
  readonly locals: readonly IRLocal[];
  readonly body: BlockStmt;
  /** Every field the kernel reads, deduped, in first-use order. */
  readonly reads: readonly FieldAccess[];
  /** Every field the kernel assigns, deduped, in first-use order. Drives readback. */
  readonly writes: readonly FieldAccess[];
  readonly usesDt: boolean;
  readonly usesIndex: boolean;
  readonly usesCount: boolean;
  readonly usesRand: boolean;
  /**
   * Static cost estimate: one per arithmetic op, comparison, builtin call and
   * field access, with loop bodies multiplied by `maxIterations`. Feeds the
   * `target: 'auto'` heuristic (see docs/GPU.md, "auto"). Not a cycle count.
   */
  readonly opCount: number;
  /** `kernel.toString()`, retained for code frames. */
  readonly source: string;
}

// ---------------------------------------------------------------------------
// Constructors
//
// Terse on purpose: parse.ts builds thousands of these. Every one computes the
// node's type, so a backend can read `.type` without inferring anything.
// ---------------------------------------------------------------------------

export function num(value: number, span?: Span): NumLit {
  return { kind: 'num', value, type: 'f32', span };
}

export function boolLit(value: boolean, span?: Span): BoolLit {
  return { kind: 'boollit', value, type: 'bool', span };
}

export function field(component: number, name: string, fieldKind: IRFieldKind, role: 0 | 1 = 0, span?: Span): FieldRef {
  return { kind: 'field', component, field: name, fieldKind, role, type: 'f32', span };
}

export function uniform(name: string, span?: Span): UniformRef {
  return { kind: 'uniform', name, type: 'f32', span };
}

export function local(id: number, type: ExprType, span?: Span): LocalRef {
  return { kind: 'local', id, type, span };
}

export function builtinValue(name: 'dt' | BuiltinValueName, span?: Span): BuiltinRef {
  return { kind: 'builtinValue', name, type: 'f32', span };
}

export function unary(op: '-' | '!', arg: Expr, span?: Span): UnaryExpr {
  return { kind: 'unary', op, arg, type: op === '!' ? 'bool' : 'f32', span };
}

/** Comparison operators yield `bool`; the rest yield `f32`. */
export function binary(op: BinaryOp, left: Expr, right: Expr, span?: Span): BinaryExpr {
  const cmp = op === '<' || op === '<=' || op === '>' || op === '>=' || op === '==' || op === '!=';
  return { kind: 'binary', op, left, right, type: cmp ? 'bool' : 'f32', span };
}

export function logical(op: '&&' | '||', left: Expr, right: Expr, span?: Span): LogicalExpr {
  return { kind: 'logical', op, left, right, type: 'bool', span };
}

/** `then`/`alt` must agree; the result takes `then`'s type. */
export function cond(test: Expr, then: Expr, alt: Expr, span?: Span): CondExpr {
  return { kind: 'cond', test, then, alt, type: then.type, span };
}

export function call(callee: BuiltinName, args: readonly Expr[], span?: Span): CallExpr {
  return { kind: 'call', callee, args, type: 'f32', span };
}

export function block(body: readonly Stmt[], span?: Span): BlockStmt {
  return { kind: 'block', body, span };
}

export function decl(id: number, init: Expr, span?: Span): DeclStmt {
  return { kind: 'decl', id, init, span };
}

export function assign(target: FieldRef | LocalRef, op: AssignOp, value: Expr, span?: Span): AssignStmt {
  return { kind: 'assign', target, op, value, span };
}

export function ifStmt(test: Expr, then: BlockStmt, alt: BlockStmt | null, span?: Span): IfStmt {
  return { kind: 'if', test, then, alt, span };
}

export function forStmt(
  init: DeclStmt | null,
  test: Expr | null,
  update: AssignStmt | null,
  body: BlockStmt,
  maxIterations: number,
  span?: Span,
): ForStmt {
  return { kind: 'for', init, test, update, body, maxIterations, span };
}

export function whileStmt(test: Expr, body: BlockStmt, maxIterations: number, span?: Span): WhileStmt {
  return { kind: 'while', test, body, maxIterations, span };
}

export const BREAK: BreakStmt = Object.freeze({ kind: 'break' });
export const CONTINUE: ContinueStmt = Object.freeze({ kind: 'continue' });

// ---------------------------------------------------------------------------
// Traversal and derived facts
// ---------------------------------------------------------------------------

/** Children of `node` in evaluation order. */
export function childrenOf(node: Node): Node[] {
  switch (node.kind) {
    case 'num':
    case 'boollit':
    case 'field':
    case 'uniform':
    case 'local':
    case 'builtinValue':
    case 'break':
    case 'continue':
      return [];
    case 'unary':
      return [node.arg];
    case 'binary':
    case 'logical':
      return [node.left, node.right];
    case 'cond':
      return [node.test, node.then, node.alt];
    case 'call':
      return node.args.slice();
    case 'block':
      return node.body.slice();
    case 'decl':
      return [node.init];
    case 'assign':
      return [node.target, node.value];
    case 'if':
      return node.alt === null ? [node.test, node.then] : [node.test, node.then, node.alt];
    case 'for': {
      const out: Node[] = [];
      if (node.init) out.push(node.init);
      if (node.test) out.push(node.test);
      if (node.update) out.push(node.update);
      out.push(node.body);
      return out;
    }
    case 'while':
      return [node.test, node.body];
    default:
      return [];
  }
}

/** Pre-order walk. Return `false` from `visit` to skip a node's children. */
export function walk(node: Node, visit: (n: Node) => boolean | void): void {
  if (visit(node) === false) return;
  const kids = childrenOf(node);
  for (let i = 0; i < kids.length; i++) walk(kids[i], visit);
}

/** Stable identity of a field access: `"<component>.<field>@<role>"`. */
export function accessKey(a: { component: number; field: string; role: 0 | 1 }): string {
  return `${a.component}.${a.field}@${a.role}`;
}

/**
 * Recomputes `reads`/`writes`/`uses*`/`opCount` from a body. parse.ts calls
 * this instead of tracking them by hand, and {@link validateIR} calls it to
 * check that a hand-built IR is self-consistent.
 *
 * A compound assignment (`p.x += v`) counts its target as BOTH a read and a
 * write. A plain `=` to a field counts as a write only.
 */
export function deriveFacts(body: BlockStmt): {
  reads: FieldAccess[];
  writes: FieldAccess[];
  usesDt: boolean;
  usesIndex: boolean;
  usesCount: boolean;
  usesRand: boolean;
  opCount: number;
} {
  const reads = new Map<string, FieldAccess>();
  const writes = new Map<string, FieldAccess>();
  let usesDt = false;
  let usesIndex = false;
  let usesCount = false;
  let usesRand = false;

  const noteRead = (f: FieldRef): void => {
    const key = accessKey(f);
    if (!reads.has(key)) reads.set(key, { component: f.component, field: f.field, fieldKind: f.fieldKind, role: f.role });
  };
  const noteWrite = (f: FieldRef): void => {
    const key = accessKey(f);
    if (!writes.has(key)) writes.set(key, { component: f.component, field: f.field, fieldKind: f.fieldKind, role: f.role });
  };

  // Cost of a subtree, with loop bodies scaled by their static trip cap.
  const cost = (n: Node): number => {
    let own = 0;
    switch (n.kind) {
      case 'unary':
      case 'binary':
      case 'logical':
      case 'cond':
      case 'field':
        own = 1;
        break;
      case 'call':
        own = n.callee === 'sqrt' || n.callee === 'abs' || n.callee === 'min' || n.callee === 'max' ? 1 : 4;
        break;
      case 'assign':
        own = 1;
        break;
      default:
        own = 0;
    }
    if (n.kind === 'for' || n.kind === 'while') {
      const trips = Math.max(1, n.maxIterations);
      const head = n.kind === 'for' ? (n.init ? cost(n.init) : 0) + (n.test ? cost(n.test) : 0) + (n.update ? cost(n.update) : 0) : cost(n.test);
      return head + trips * cost(n.body);
    }
    const kids = childrenOf(n);
    for (let i = 0; i < kids.length; i++) own += cost(kids[i]);
    return own;
  };

  walk(body, (n) => {
    if (n.kind === 'assign') {
      if (n.target.kind === 'field') {
        noteWrite(n.target);
        if (n.op !== '=') noteRead(n.target);
      }
      // Skip the target so the generic field-read case below does not see it.
      walk(n.value, (m) => {
        if (m.kind === 'field') noteRead(m);
        else if (m.kind === 'builtinValue') {
          if (m.name === 'dt') usesDt = true;
          else if (m.name === 'index') usesIndex = true;
          else usesCount = true;
        } else if (m.kind === 'call' && m.callee === 'rand') usesRand = true;
      });
      return false;
    }
    if (n.kind === 'field') noteRead(n);
    else if (n.kind === 'builtinValue') {
      if (n.name === 'dt') usesDt = true;
      else if (n.name === 'index') usesIndex = true;
      else usesCount = true;
    } else if (n.kind === 'call' && n.callee === 'rand') usesRand = true;
    return true;
  });

  return {
    reads: [...reads.values()],
    writes: [...writes.values()],
    usesDt,
    usesIndex,
    usesCount,
    usesRand,
    opCount: cost(body),
  };
}

/** Assembles a KernelIR from a body plus declarations, deriving the rest. */
export function makeKernelIR(parts: {
  name: string;
  form: KernelForm;
  components: readonly IRComponent[];
  uniforms: readonly IRUniform[];
  locals: readonly IRLocal[];
  body: BlockStmt;
  source: string;
}): KernelIR {
  const facts = deriveFacts(parts.body);
  return {
    version: IR_VERSION,
    name: parts.name,
    form: parts.form,
    components: parts.components,
    uniforms: parts.uniforms,
    locals: parts.locals,
    body: parts.body,
    reads: facts.reads,
    writes: facts.writes,
    usesDt: facts.usesDt,
    usesIndex: facts.usesIndex,
    usesCount: facts.usesCount,
    usesRand: facts.usesRand,
    opCount: facts.opCount,
    source: parts.source,
  };
}

/** Every field the kernel touches (read or written), deduped. */
export function touchedFields(ir: KernelIR): FieldAccess[] {
  const out = new Map<string, FieldAccess>();
  for (const a of ir.reads) out.set(accessKey(a), a);
  for (const a of ir.writes) out.set(accessKey(a), a);
  return [...out.values()];
}

/** Components the kernel writes to, as indices into `ir.components`, ascending. */
export function writtenComponents(ir: KernelIR): number[] {
  const seen = new Set<number>();
  for (const w of ir.writes) seen.add(w.component);
  return [...seen].sort((x, y) => x - y);
}

/**
 * Field kinds the kernel touches that the GPU backend cannot address (f64,
 * sub-word ints, str). Empty means the kernel is GPU-eligible as far as its
 * fields go. runtime.ts turns a non-empty result into a one-time warning and a
 * permanent CPU target -- never a throw.
 */
export function gpuBlockers(ir: KernelIR): { component: string; field: string; kind: IRFieldKind }[] {
  const out: { component: string; field: string; kind: IRFieldKind }[] = [];
  for (const a of touchedFields(ir)) {
    if (!isGPUEligible(a.fieldKind)) {
      out.push({ component: ir.components[a.component].name, field: a.field, kind: a.fieldKind });
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Binding layout
//
// Pinned here so wgsl.ts (which writes the struct) and runtime.ts (which fills
// the buffer) cannot disagree. Changing anything below is an IR_VERSION bump.
// ---------------------------------------------------------------------------

/**
 * The 4-byte views the kernel needs over the archetype table, in binding
 * order. The same GPUBuffer is bound once per view type (WebGPU permits the
 * same buffer in several bindings; the fields they address are disjoint).
 * Binding `n` of the returned array is `@binding(n)`; the uniform block is
 * `@binding(views.length)`.
 */
export function storageViews(ir: KernelIR): StorageViewType[] {
  const seen: StorageViewType[] = [];
  for (const a of touchedFields(ir)) {
    const v = FIELD_KINDS[a.fieldKind].view;
    if (v !== null && seen.indexOf(v) === -1) seen.push(v);
  }
  // Order f32, i32, u32 regardless of first use, so two kernels over the same
  // fields always produce the same layout (pipeline cache key stability).
  const order: StorageViewType[] = ['f32', 'i32', 'u32'];
  return order.filter((v) => seen.indexOf(v) !== -1);
}

/** Binding index of the uniform block. */
export function uniformBinding(ir: KernelIR): number {
  return storageViews(ir).length;
}

/** One 4-byte member of the uniform block. */
export interface UniformMember {
  /** WGSL member name, also used as the key runtime.ts writes by. */
  readonly member: string;
  /** Byte offset inside the uniform buffer. */
  readonly offset: number;
  readonly type: 'f32' | 'u32';
}

/**
 * Exact byte layout of the kernel's uniform block.
 *
 *   offset  0  dt      f32
 *   offset  4  count   u32   rows in this dispatch
 *   offset  8  base    u32   first row of this dispatch (0 unless split)
 *   offset 12  countOther u32 rows of the `other` table in a pairwise kernel
 *                              (equal to `count` for a per-entity kernel)
 *   offset 16  o_<c>_<field>  u32 ... one per touched field: the field column's
 *                             base offset in ELEMENTS of that field's view
 *   then       u_<name>       f32 ... one per declared uniform, in order
 *   size rounded up to a multiple of 16
 *
 * Only 4-byte scalars are used: a `array<u32, N>` member in the uniform address
 * space has a 16-byte stride in WGSL, which would waste 4x the space and is easy
 * to get wrong. Named scalars have none of that.
 */
export interface UniformLayout {
  readonly size: number;
  readonly dt: UniformMember;
  readonly count: UniformMember;
  readonly base: UniformMember;
  readonly countOther: UniformMember;
  /** Field base offsets, index-aligned with {@link touchedFields}. */
  readonly fields: readonly (UniformMember & { access: FieldAccess })[];
  /** User uniforms, index-aligned with `ir.uniforms`. */
  readonly uniforms: readonly (UniformMember & { name: string })[];
  /** Every member, in offset order, for the WGSL struct writer. */
  readonly members: readonly UniformMember[];
}

/** WGSL member name of a field's base-offset slot. Deterministic; both backends call this. */
export function offsetMemberName(ir: KernelIR, a: FieldAccess): string {
  return `o${a.role}_${ir.components[a.component].name.replace(/[^A-Za-z0-9_]/g, '_')}_${a.field}`;
}

/** WGSL member name of a user uniform. */
export function uniformMemberName(name: string): string {
  return `u_${name}`;
}

export function uniformLayout(ir: KernelIR): UniformLayout {
  const members: UniformMember[] = [];
  const dt: UniformMember = { member: 'dt', offset: 0, type: 'f32' };
  const count: UniformMember = { member: 'count', offset: 4, type: 'u32' };
  const base: UniformMember = { member: 'base', offset: 8, type: 'u32' };
  const countOther: UniformMember = { member: 'countOther', offset: 12, type: 'u32' };
  members.push(dt, count, base, countOther);
  let at = 16;
  const fields: (UniformMember & { access: FieldAccess })[] = [];
  for (const a of touchedFields(ir)) {
    const m = { member: offsetMemberName(ir, a), offset: at, type: 'u32' as const, access: a };
    fields.push(m);
    members.push(m);
    at += 4;
  }
  const uniforms: (UniformMember & { name: string })[] = [];
  for (const u of ir.uniforms) {
    const m = { member: uniformMemberName(u.name), offset: at, type: 'f32' as const, name: u.name };
    uniforms.push(m);
    members.push(m);
    at += 4;
  }
  return { size: (at + 15) & ~15, dt, count, base, countOther, fields, uniforms, members };
}

/**
 * Workgroup size for every kernel. 256 is the maximum the WebGPU *default*
 * limits grant, so no raised limits are needed, and 64/128/256 measured within
 * noise of each other on an M4 (0.54-0.65 ms for a 1M-entity integrate kernel)
 * while 32 was clearly slower.
 */
export const WORKGROUP_SIZE = 256;

/**
 * Entities per 1D dispatch: `maxComputeWorkgroupsPerDimension` (65535 in
 * practice) x {@link WORKGROUP_SIZE}. Beyond this a chunk is dispatched in
 * several passes with `base` advanced (see {@link UniformLayout}).
 */
export function maxRowsPerDispatch(maxWorkgroupsPerDimension: number): number {
  return maxWorkgroupsPerDimension * WORKGROUP_SIZE;
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

function fail(ir: KernelIR, message: string, span?: Span): never {
  throw new KernelError('E_INVALID_IR', message, { kernelName: ir.name, span, source: ir.source });
}

/**
 * Structural and type check of an IR. parse.ts runs it before returning, and
 * both backends run it (cheaply, once per kernel) before codegen, so a bad IR
 * fails with a code frame instead of producing broken WGSL or JS.
 *
 * Checks: version; component/field references resolve; uniform references are
 * declared; local ids are in range, declared before use and not reassigned when
 * immutable; assignment target types; operand types; builtin arity; loops carry
 * a positive `maxIterations`; `break`/`continue` appear only inside a loop;
 * `reads`/`writes`/`opCount` match {@link deriveFacts}.
 */
export function validateIR(ir: KernelIR): void {
  if (ir.version !== IR_VERSION) fail(ir, `IR version ${ir.version}, expected ${IR_VERSION}`);
  for (let i = 0; i < ir.components.length; i++) {
    if (ir.components[i].index !== i) fail(ir, `components[${i}].index is ${ir.components[i].index}`);
  }
  for (let i = 0; i < ir.locals.length; i++) {
    if (ir.locals[i].id !== i) fail(ir, `locals[${i}].id is ${ir.locals[i].id}`);
  }
  if (ir.form === 'per-entity') {
    for (const a of touchedFields(ir)) {
      if (a.role !== 0) fail(ir, `role 1 field "${a.field}" in a per-entity kernel`);
    }
  }

  const declared = new Set<number>();
  let loopDepth = 0;

  const checkExpr = (e: Expr): void => {
    switch (e.kind) {
      case 'num':
      case 'boollit':
        return;
      case 'field': {
        const c = ir.components[e.component];
        if (!c) fail(ir, `field "${e.field}" on component index ${e.component}, which does not exist`, e.span);
        const f = c.fields.find((x) => x.name === e.field);
        if (!f) fail(ir, `component "${c.name}" has no field "${e.field}"`, e.span);
        if (f.kind !== e.fieldKind) fail(ir, `field "${c.name}.${e.field}" is ${f.kind}, IR says ${e.fieldKind}`, e.span);
        return;
      }
      case 'uniform':
        if (!ir.uniforms.some((u) => u.name === e.name)) fail(ir, `uniform "${e.name}" is not declared`, e.span);
        return;
      case 'local': {
        const l = ir.locals[e.id];
        if (!l) fail(ir, `local #${e.id} does not exist`, e.span);
        if (!declared.has(e.id)) fail(ir, `local "${l.name}" used before its declaration`, e.span);
        if (l.type !== e.type) fail(ir, `local "${l.name}" is ${l.type}, reference says ${e.type}`, e.span);
        return;
      }
      case 'builtinValue':
        return;
      case 'unary':
        checkExpr(e.arg);
        if (e.op === '!' && e.arg.type !== 'bool') fail(ir, '"!" needs a bool operand', e.span);
        if (e.op === '-' && e.arg.type !== 'f32') fail(ir, 'unary "-" needs a numeric operand', e.span);
        return;
      case 'binary':
        checkExpr(e.left);
        checkExpr(e.right);
        if (e.op === '==' || e.op === '!=') {
          if (e.left.type !== e.right.type) fail(ir, `"${e.op}" compares ${e.left.type} with ${e.right.type}`, e.span);
        } else if (e.left.type !== 'f32' || e.right.type !== 'f32') {
          fail(ir, `"${e.op}" needs numeric operands`, e.span);
        }
        return;
      case 'logical':
        checkExpr(e.left);
        checkExpr(e.right);
        if (e.left.type !== 'bool' || e.right.type !== 'bool') fail(ir, `"${e.op}" needs bool operands`, e.span);
        return;
      case 'cond':
        checkExpr(e.test);
        checkExpr(e.then);
        checkExpr(e.alt);
        if (e.test.type !== 'bool') fail(ir, 'the condition of "?:" must be a bool', e.span);
        if (e.then.type !== e.alt.type) fail(ir, `"?:" branches are ${e.then.type} and ${e.alt.type}`, e.span);
        return;
      case 'call': {
        const info = BUILTINS[e.callee];
        if (!info) fail(ir, `unknown builtin "${e.callee}"`, e.span);
        if (e.args.length < info.arity[0] || e.args.length > info.arity[1]) {
          fail(ir, `"${e.callee}" takes ${info.arity[0]}..${info.arity[1]} arguments, got ${e.args.length}`, e.span);
        }
        for (const a of e.args) {
          checkExpr(a);
          if (a.type !== 'f32') fail(ir, `"${e.callee}" takes numeric arguments`, e.span);
        }
        return;
      }
      default:
        fail(ir, `unknown expression node "${(e as { kind: string }).kind}"`);
    }
  };

  const checkStmt = (s: Stmt): void => {
    switch (s.kind) {
      case 'block':
        for (const st of s.body) checkStmt(st);
        return;
      case 'decl': {
        const l = ir.locals[s.id];
        if (!l) fail(ir, `declaration of local #${s.id}, which does not exist`, s.span);
        if (declared.has(s.id)) fail(ir, `local "${l.name}" declared twice`, s.span);
        checkExpr(s.init);
        if (s.init.type !== l.type) fail(ir, `local "${l.name}" is ${l.type} but its initializer is ${s.init.type}`, s.span);
        declared.add(s.id);
        return;
      }
      case 'assign': {
        checkExpr(s.value);
        if (s.target.kind === 'local') {
          const l = ir.locals[s.target.id];
          if (!l) fail(ir, `assignment to local #${s.target.id}, which does not exist`, s.span);
          if (!declared.has(s.target.id)) fail(ir, `local "${l.name}" assigned before its declaration`, s.span);
          if (!l.mutable) fail(ir, `local "${l.name}" is const`, s.span);
          if (s.op !== '=' && l.type !== 'f32') fail(ir, `"${s.op}" needs a numeric target`, s.span);
          if (s.value.type !== l.type) fail(ir, `local "${l.name}" is ${l.type} but the value is ${s.value.type}`, s.span);
        } else {
          checkExpr(s.target);
          if (s.value.type !== 'f32') fail(ir, 'a field can only be assigned a numeric value', s.span);
        }
        return;
      }
      case 'if':
        checkExpr(s.test);
        if (s.test.type !== 'bool') fail(ir, 'the condition of "if" must be a bool', s.span);
        checkStmt(s.then);
        if (s.alt) checkStmt(s.alt);
        return;
      case 'for':
        if (!(s.maxIterations > 0)) fail(ir, 'a for loop needs a positive maxIterations', s.span);
        if (s.init) checkStmt(s.init);
        if (s.test) {
          checkExpr(s.test);
          if (s.test.type !== 'bool') fail(ir, 'the condition of "for" must be a bool', s.span);
        }
        if (s.update) checkStmt(s.update);
        loopDepth++;
        checkStmt(s.body);
        loopDepth--;
        return;
      case 'while':
        if (!(s.maxIterations > 0)) fail(ir, 'a while loop needs a positive maxIterations', s.span);
        checkExpr(s.test);
        if (s.test.type !== 'bool') fail(ir, 'the condition of "while" must be a bool', s.span);
        loopDepth++;
        checkStmt(s.body);
        loopDepth--;
        return;
      case 'break':
      case 'continue':
        if (loopDepth === 0) fail(ir, `"${s.kind}" outside a loop`, s.span);
        return;
      default:
        fail(ir, `unknown statement node "${(s as { kind: string }).kind}"`);
    }
  };

  checkStmt(ir.body);

  const facts = deriveFacts(ir.body);
  const same = (a: readonly FieldAccess[], bs: readonly FieldAccess[]): boolean => {
    if (a.length !== bs.length) return false;
    const keys = new Set(a.map(accessKey));
    return bs.every((x) => keys.has(accessKey(x)));
  };
  if (!same(facts.reads, ir.reads)) fail(ir, 'ir.reads does not match the body');
  if (!same(facts.writes, ir.writes)) fail(ir, 'ir.writes does not match the body');
  if (facts.opCount !== ir.opCount) fail(ir, `ir.opCount is ${ir.opCount}, body says ${facts.opCount}`);
}

// ---------------------------------------------------------------------------
// Debug rendering
// ---------------------------------------------------------------------------

/** Pretty-prints an expression as JS-like text. Test snapshots and error messages only. */
export function exprToString(ir: KernelIR, e: Expr): string {
  switch (e.kind) {
    case 'num':
      return Number.isInteger(e.value) ? `${e.value}.0` : `${e.value}`;
    case 'boollit':
      return `${e.value}`;
    case 'field':
      return `${e.role === 0 ? '' : 'other:'}${ir.components[e.component].name}.${e.field}`;
    case 'uniform':
      return `u.${e.name}`;
    case 'local':
      return ir.locals[e.id].name;
    case 'builtinValue':
      return e.name;
    case 'unary':
      return `${e.op}${exprToString(ir, e.arg)}`;
    case 'binary':
    case 'logical':
      return `(${exprToString(ir, e.left)} ${e.op} ${exprToString(ir, e.right)})`;
    case 'cond':
      return `(${exprToString(ir, e.test)} ? ${exprToString(ir, e.then)} : ${exprToString(ir, e.alt)})`;
    case 'call':
      return `${e.callee}(${e.args.map((a) => exprToString(ir, a)).join(', ')})`;
    default:
      return '?';
  }
}

/** Pretty-prints a whole kernel. Stable across runs: usable as a test snapshot. */
export function describeIR(ir: KernelIR): string {
  const out: string[] = [];
  const write = (depth: number, s: string): void => {
    out.push(`${'  '.repeat(depth)}${s}`);
  };
  const stmt = (s: Stmt, depth: number): void => {
    switch (s.kind) {
      case 'block':
        for (const st of s.body) stmt(st, depth);
        return;
      case 'decl':
        write(depth, `${ir.locals[s.id].mutable ? 'let' : 'const'} ${ir.locals[s.id].name} = ${exprToString(ir, s.init)};`);
        return;
      case 'assign':
        write(depth, `${exprToString(ir, s.target)} ${s.op} ${exprToString(ir, s.value)};`);
        return;
      case 'if':
        write(depth, `if (${exprToString(ir, s.test)}) {`);
        stmt(s.then, depth + 1);
        if (s.alt) {
          write(depth, '} else {');
          stmt(s.alt, depth + 1);
        }
        write(depth, '}');
        return;
      case 'for':
        write(depth, `for (...; max ${s.maxIterations}) {`);
        stmt(s.body, depth + 1);
        write(depth, '}');
        return;
      case 'while':
        write(depth, `while (${exprToString(ir, s.test)}; max ${s.maxIterations}) {`);
        stmt(s.body, depth + 1);
        write(depth, '}');
        return;
      default:
        write(depth, `${s.kind};`);
    }
  };
  write(0, `kernel ${ir.name} [${ir.form}] ops=${ir.opCount}`);
  write(0, `components: ${ir.components.map((c) => `${c.index}:${c.name}(${c.fields.map((f) => `${f.name}:${f.kind}`).join(',')})`).join(' ')}`);
  if (ir.uniforms.length) write(0, `uniforms: ${ir.uniforms.map((u) => u.name).join(', ')}`);
  write(0, `reads: ${ir.reads.map((a) => `${ir.components[a.component].name}.${a.field}`).join(', ') || '-'}`);
  write(0, `writes: ${ir.writes.map((a) => `${ir.components[a.component].name}.${a.field}`).join(', ') || '-'}`);
  write(0, 'body {');
  stmt(ir.body, 1);
  write(0, '}');
  return out.join('\n');
}
