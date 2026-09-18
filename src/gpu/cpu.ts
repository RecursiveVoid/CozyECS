/**
 * {@link KernelIR} -> a compiled JavaScript chunk loop.
 *
 * OWNER: developer D (who also owns the two `src/query.ts` carry-overs, since
 * this backend rides on that machinery).
 *
 * This is not just the fallback for "no WebGPU". It is the fast path for every
 * kernel below the GPU break-even (~45k entities with readback 'async' or
 * 'none', ~700k when the frame awaits readback with 'sync-frame'), which is
 * most kernels in most games. It is also the parity oracle the GPU backend is
 * tested against.
 *
 * WHY IT IS FAST: the generated function is passed to `Query.forEachChunk`,
 * which -- for a callback whose identity is stable across frames -- runs it
 * from per-chunk trampolines in which each column TypedArray is a closure
 * CONSTANT. V8 then embeds the array's data pointer instead of reloading it per
 * row (measured 262k -> 382k ops/sec on a 4 x 1000-row `Position += Velocity`
 * loop; see docs/INTERNALS.md, "Why the plain chunk loop trails closure-constant
 * loops"). A kernel user gets that for free: `compileCPUKernel` returns ONE
 * function object and the system reuses it forever.
 *
 * IMPORTANT for the owner: `fn` identity must be stable. Do not wrap it, do not
 * rebind it per frame, and do not pass a fresh arrow into `forEachChunk` --
 * that silently drops every kernel back onto the generic loop.
 *
 * SHAPE OF THE GENERATED CODE (per-entity, components [Position, Velocity],
 * writes Position.x/y and Velocity.y, one uniform `gravity`):
 *
 *   "use strict";
 *   const st = state, U = st.u, fr = Math.fround, rand = cozy_rand;
 *   return function cozyKernel_Move(count, c0, c1, chunk) {
 *     const dt = fr(st.dt), u0_gravity = fr(U[0]);
 *     const h0 = fr(u0_gravity * dt);                         // loop-invariant, hoisted
 *     const a0_0 = c0["x"], a0_1 = c0["y"], a1_0 = c1["x"], a1_1 = c1["y"];
 *     for (let i = 0; i < count; i++) {
 *       const r1_0 = a1_0[i];                                 // each field loaded once
 *       let r0_0 = a0_0[i], r0_1 = a0_1[i], r1_1 = a1_1[i];
 *       {
 *         a1_1[i] = r1_1 = fr(r1_1 + h0);                     // store-through, value kept
 *         ...
 *       }
 *     }
 *   };
 *
 * ROW CACHE: every `self` field the body reads is loaded into a local once at
 * the top of the row and every write stores through (`a[i] = r = v`), so the
 * body never re-reads a value it just wrote. Together with HOISTING below this
 * took the gravity kernel from 1.07-1.11x the time of the equivalent
 * hand-written `forEachChunk` loop to 0.94-1.00x (10k/100k/1M entities, M4,
 * median of 5 isolated runs). The cached value is exactly what
 * a re-read would return: `v` is already f32-valued (or clamped, for integer
 * fields), so the store is lossless. Disabled when two kernel components share
 * a component id (their columns would alias). Pairwise kernels load the cache
 * once per `i`, outside the `j` loop; `other` reads stay array reads.
 *
 * HOISTING: any compound expression built only from literals, uniforms, `dt`
 * and `count` is evaluated once per chunk. Every such expression is pure
 * (Math builtins and `rand` have no side effects), so evaluating it when a
 * branch is not taken is unobservable.
 *
 * TYPE RULES: identical to the GPU backend (see ./ir design rule 1 and the
 * PARITY CONTRACT in ./wgsl). All arithmetic is f32-shaped: every arithmetic
 * result, uniform, `dt` and non-f32 field read is rounded through `Math.fround`
 * exactly where WGSL rounds, so the two backends agree to the tolerance
 * documented in docs/GPU.md (transcendentals may differ by a few ulp).
 * `rand(seed)` is the bit-exact twin `cozyRandRef` from ./wgsl. Writes to an
 * integer field saturate to the field's range (NaN -> 0) and round half toward
 * +Infinity (`Math.round` on the f32), bit-identical to `cozy_store_i32/u32`
 * -- the TypedArray would truncate and wrap instead. The `round` builtin is
 * `Math.floor(fround(x + 0.5))`, the f32 twin of WGSL `floor(x + 0.5)`.
 */

import type { ComponentType } from '../component';
import type { AssignStmt, BlockStmt, Expr, FieldRef, IRFieldKind, KernelIR, Stmt } from './ir';
import { BUILTINS, FIELD_KINDS, KernelError, deriveFacts, touchedFields, validateIR } from './ir';
import { cozyRandRef } from './wgsl';

/** Per-frame inputs the compiled loop reads. Mutate in place; never replace. */
export interface CPUKernelState {
  /** `dt` as passed to `world.update(dt)`. */
  dt: number;
  /** User uniform values, indexed by `ir.uniforms[i].index`. */
  readonly u: Float64Array;
  /** Rows of the `other` side for a pairwise kernel. Ignored for per-entity. */
  countOther: number;
}

export interface CompileCPUOptions {
  /**
   * Component ids whose per-entity enabled flag must be 1 for a row to run
   * (the query's enableable `all` components). The loop reads
   * `chunk.enabled[id]` once per chunk and tests it per row. Empty by default.
   */
  readonly enableIds?: readonly number[];
  /**
   * Round every intermediate through `Math.fround`. Default true: it is what
   * makes CPU results match the GPU's f32 arithmetic. Set false only in a
   * benchmark that is explicitly measuring the cost of parity.
   */
  readonly fround?: boolean;
  /** Emit `// source` comments into the generated function. Debug only. */
  readonly annotate?: boolean;
}

/** A compiled kernel, ready to hand to `Query.forEachChunk`. */
export interface CompiledCPUKernel {
  /**
   * The chunk callback. Signature matches `ChunkKernelFn`:
   * `(count, ...columnObjects, chunk)`. IDENTITY IS STABLE for the lifetime of
   * this object and must stay that way -- see the note above.
   */
  readonly fn: (count: number, ...args: unknown[]) => void;
  /** The state object the loop reads. Write `dt`/`u` here before each call. */
  readonly state: CPUKernelState;
  /** Components to pass as `forEachChunk`'s first argument, in IR order. */
  readonly components: readonly ComponentType[];
  /** The generated source, for debugging and for the `E_NO_CODEGEN` message. */
  readonly source: string;
  /** False when `new Function` was unavailable and the interpreting path is in use. */
  readonly compiled: boolean;
}

/**
 * Compiles `ir` into a chunk loop.
 *
 * CONTRACT
 *  - `components` are the core ComponentTypes matching `ir.components` by
 *    index; the caller (index.ts) guarantees that. This file uses them only as
 *    an opaque token to hand back to `forEachChunk` -- it must not read their
 *    schemas, which are already in the IR.
 *  - Calls `validateIR(ir)` before generating.
 *  - Generation is deterministic: same (ir, opts) gives the same `source`.
 *  - If `new Function` throws (CSP), falls back to a closure-tree evaluator
 *    built from the IR (correct, ~5-10x slower) and sets `compiled: false`.
 *    NEVER throws E_NO_CODEGEN out of this function: a kernel must keep working
 *    under CSP.
 *  - The loop reads `count` from its parameter, never `chunk.count`, so rows
 *    appended during iteration are not visited.
 *  - Structural changes are impossible from a kernel by construction (the
 *    subset has no world access), so the loop needs no command-buffer handling.
 *  - Pairwise: the generated loop is `for (i) for (j) if (j !== i) { body }`
 *    over the SAME chunk, with role-1 accesses indexed by `j`. Writes to role-1
 *    fields are rejected by `validateIR`'s caller (index.ts) before compiling
 *    (and again here, as E_INVALID_IR). A row whose enable flags are off takes
 *    part in no pair, on either side.
 *
 * @throws KernelError E_INVALID_IR only.
 */
export function compileCPUKernel(
  ir: KernelIR,
  components: readonly ComponentType[],
  opts?: CompileCPUOptions,
): CompiledCPUKernel {
  const source = generateCPUSource(ir, opts); // validates
  const u = new Float64Array(ir.uniforms.length);
  for (const x of ir.uniforms) u[x.index] = x.initial;
  const state: CPUKernelState = { dt: 0, u, countOther: 0 };
  let fn: (count: number, ...args: unknown[]) => void;
  let compiled = true;
  try {
    fn = new Function('state', 'cozy_rand', source)(state, cozyRandRef) as typeof fn;
    if (typeof fn !== 'function') throw new TypeError('codegen returned no function');
  } catch {
    fn = interpretKernel(ir, state, opts);
    compiled = false;
  }
  return { fn, state, components: components.slice(), source, compiled };
}

/**
 * The JavaScript the compiler would emit, without compiling it. Exported so
 * tests can assert the generated loop (including the fround placement and the
 * integer write clamps) as a string, with no dependence on codegen being
 * available in the test environment.
 *
 * The text is the BODY of `new Function('state', 'cozy_rand', source)`, which
 * returns the chunk callback.
 */
export function generateCPUSource(ir: KernelIR, opts?: CompileCPUOptions): string {
  validateIR(ir);
  rejectOtherWrites(ir);
  return new JSWriter(ir, opts).module();
}

/**
 * Fixed CPU cost per entity (or per pair), in ns, that every kernel pays
 * regardless of its body: the loop, the row index, and moving the row's
 * columns through the cache. See {@link estimateCPUNanosPerEntity}.
 */
export const CPU_NS_PER_ENTITY_FIXED = 0.52;

/** Marginal CPU cost of one IR op, in ns. See {@link estimateCPUNanosPerEntity}. */
export const CPU_NS_PER_OP = 0.065;

/**
 * Estimated per-entity cost in nanoseconds, used by `target: 'auto'` to decide
 * CPU vs GPU without measuring: `CPU_NS_PER_ENTITY_FIXED + CPU_NS_PER_OP * ops`.
 *
 * The model is affine, not proportional. A purely per-op model (the round-1
 * `0.095 * ops`) rated the 6-op `simple` kernel 3.3x cheaper than the 20-op
 * `gravity` kernel, while the measured CPU cost ratio in the break-even range
 * (25k-100k entities) is ~2x: a cheap kernel still pays for the loop and the
 * memory traffic. That made `auto` keep `simple` on the CPU up to ~90k entities
 * against a measured break-even of ~49k (`'none'`).
 *
 * Calibration (Apple M4 + Dawn, benchmarks/RESULTS.md "GPU / kernel", round 2):
 * the constants put the effective `'async'`/`'none'` threshold at the geometric
 * mean of each kernel's measured `'none'` and `'async'` break-even:
 *   simple  (6 ops):  0.91 ns -> ~56k entities (measured 49k none / 64k async)
 *   gravity (20 ops): 1.82 ns -> ~28k entities (measured 23k none / 34k async)
 * The absolute values are normalized to `BASELINE_CPU_NS_PER_ENTITY` in
 * ./runtime; only the ratio between kernels matters. If the GPU benchmark is
 * re-run, re-fit both constants together.
 *
 * Ops are counted from the IR body (`deriveFacts`, loop bodies scaled by their
 * trip cap). For a pairwise kernel the result is per PAIR: multiply by the
 * chunk's row count for a per-entity figure. Never below one op.
 */
export function estimateCPUNanosPerEntity(ir: KernelIR): number {
  const ops = deriveFacts(ir.body).opCount;
  return CPU_NS_PER_ENTITY_FIXED + CPU_NS_PER_OP * (ops > 1 ? ops : 1);
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

function invalid(ir: KernelIR, message: string): never {
  throw new KernelError('E_INVALID_IR', message, { kernelName: ir.name, source: ir.source });
}

function rejectOtherWrites(ir: KernelIR): void {
  for (const w of ir.writes) {
    if (w.role !== 0) {
      invalid(ir, `kernel writes to "other.${w.field}"; a pairwise kernel may only write through "self"`);
    }
  }
}

/** Kinds whose stored value is already an f32 (or a small integer exactly representable as one). */
function readIsExactF32(kind: IRFieldKind): boolean {
  return kind === 'f32' || kind === 'i8' || kind === 'u8' || kind === 'i16' || kind === 'u16' || kind === 'bool';
}

/** Name of the integer-store helper for `kind`, e.g. `s_i32`. */
function storeHelperName(kind: IRFieldKind): string {
  return `s_${kind}`;
}

/**
 * Source of the saturating integer store for `kind` (parity rule 2 in ./wgsl):
 * NaN -> 0, round half toward +Infinity, saturate to the field's range. For
 * i32/u32 this is bit-identical to `storeI32Ref`/`storeU32Ref`.
 */
function storeHelperSource(kind: IRFieldKind): string {
  const range = FIELD_KINDS[kind].range as readonly [number, number];
  const lo = range[0];
  const hi = range[1];
  return (
    `const ${storeHelperName(kind)} = (x) => { if (x !== x) return 0; const r = Math.round(x); ` +
    `return r < ${lo} ? ${lo} : r > ${hi} ? ${hi} : r; };`
  );
}

/** A deterministic JS numeric literal for the f32 `value` rounds to (parenthesized when negative). */
function numLiteral(value: number, fround: boolean): string {
  const f = fround ? Math.fround(value) : value;
  if (f !== f) return 'NaN';
  if (f === Infinity) return 'Infinity';
  if (f === -Infinity) return '(-Infinity)';
  if (Object.is(f, -0)) return '(-0)';
  const s = String(f);
  return f < 0 ? `(${s})` : s;
}

function sanitize(raw: string): string {
  const s = raw.replace(/[^A-Za-z0-9_$]/g, '_').slice(0, 48);
  return s === '' ? 'kernel' : s;
}

/** Math builtins whose result is exact for an f32 argument (no rounding needed). */
const EXACT_CALLS: ReadonlySet<string> = new Set(['abs', 'min', 'max', 'floor', 'ceil', 'sign']);

// ---------------------------------------------------------------------------
// Code generation
// ---------------------------------------------------------------------------

class JSWriter {
  private readonly lines: string[] = [];
  private readonly fr: boolean;
  private readonly annotate: boolean;
  private readonly enableIds: readonly number[];
  /** accessKey without role -> column local name. */
  private readonly cols = new Map<string, string>();
  /** accessKey without role -> row-cache local name (role-0 fields the body reads). */
  private readonly cache = new Map<string, string>();
  /** Loop-invariant expressions hoisted to once per chunk: source text -> const name. */
  private readonly hoisted = new Map<string, string>();
  private trips = 0;

  constructor(
    private readonly ir: KernelIR,
    opts?: CompileCPUOptions,
  ) {
    this.fr = !(opts && opts.fround === false);
    this.annotate = !!(opts && opts.annotate);
    this.enableIds = (opts && opts.enableIds) || [];
  }

  private line(depth: number, text: string): void {
    this.lines.push('  '.repeat(depth) + text);
  }

  private note(depth: number, span: { start: number; end: number } | undefined): void {
    if (!this.annotate || !span || !this.ir.source) return;
    const raw = this.ir.source.slice(span.start, span.end).replace(/\s+/g, ' ').replace(/\*\//g, '* /').trim();
    if (raw !== '') this.line(depth, `// ${raw.length > 72 ? `${raw.slice(0, 69)}...` : raw}`);
  }

  /** `fr(x)` when rounding is on, else `x`. */
  private round(x: string): string {
    return this.fr ? `fr(${x})` : x;
  }

  private colName(f: { component: number; field: string }): string {
    const key = `${f.component}.${f.field}`;
    const name = this.cols.get(key);
    if (name === undefined) invalid(this.ir, `field "${this.ir.components[f.component].name}.${f.field}" is not in ir.reads/ir.writes`);
    return name;
  }

  private localName(id: number): string {
    return `l${id}_${sanitize(this.ir.locals[id].name)}`;
  }

  private uniformName(name: string): string {
    const u = this.ir.uniforms.find((x) => x.name === name);
    if (!u) invalid(this.ir, `uniform "${name}" is not declared`);
    return `u${u.index}_${sanitize(name)}`;
  }

  // -- expressions ------------------------------------------------------------

  private readField(f: FieldRef): string {
    const cached = f.role === 1 ? undefined : this.cache.get(`${f.component}.${f.field}`);
    const slot = cached !== undefined ? cached : `${this.colName(f)}[${f.role === 1 ? 'j' : 'i'}]`;
    return readIsExactF32(f.fieldKind) ? slot : this.round(slot);
  }

  /**
   * An expression, with any compound sub-expression that depends only on
   * literals, uniforms, `dt` and `count` hoisted to a per-chunk const.
   */
  expr(e: Expr): string {
    if (e.kind !== 'num' && e.kind !== 'boollit' && e.kind !== 'uniform' && e.kind !== 'builtinValue' && isInvariant(e)) {
      const text = this.exprText(e);
      let name = this.hoisted.get(text);
      if (name === undefined) {
        name = `h${this.hoisted.size}`;
        this.hoisted.set(text, name);
      }
      return name;
    }
    return this.exprText(e);
  }

  private exprText(e: Expr): string {
    switch (e.kind) {
      case 'num':
        return numLiteral(e.value, this.fr);
      case 'boollit':
        return e.value ? 'true' : 'false';
      case 'field':
        return this.readField(e);
      case 'uniform':
        return this.uniformName(e.name);
      case 'local':
        return this.localName(e.id);
      case 'builtinValue':
        return e.name === 'dt' ? 'dt' : e.name === 'index' ? this.round('i') : 'nCount';
      case 'unary':
        return e.op === '!' ? `(!${this.expr(e.arg)})` : `(-${this.expr(e.arg)})`;
      case 'binary': {
        const l = this.expr(e.left);
        const r = this.expr(e.right);
        if (e.type === 'bool') {
          const op = e.op === '==' ? '===' : e.op === '!=' ? '!==' : e.op;
          return `(${l} ${op} ${r})`;
        }
        return this.round(`${l} ${e.op} ${r}`);
      }
      case 'logical':
        return `(${this.expr(e.left)} ${e.op} ${this.expr(e.right)})`;
      case 'cond':
        return `(${this.expr(e.test)} ? ${this.expr(e.then)} : ${this.expr(e.alt)})`;
      case 'call': {
        const args = e.args.map((a) => this.expr(a));
        if (e.callee === 'round') return `Math.floor(${this.round(`${args[0]} + 0.5`)})`;
        if (e.callee === 'rand') return `rand(${args[0]})`;
        const info = BUILTINS[e.callee];
        if (!info) invalid(this.ir, `unknown builtin "${String(e.callee)}"`);
        const js = info.js(args);
        return EXACT_CALLS.has(e.callee) ? js : this.round(js);
      }
      default:
        return invalid(this.ir, `unknown expression node "${(e as { kind: string }).kind}"`);
    }
  }

  // -- statements -------------------------------------------------------------

  private block(b: BlockStmt, depth: number): void {
    for (const s of b.body) this.stmt(s, depth);
  }

  private storeField(f: FieldRef, value: string): string {
    const slot = `${this.colName(f)}[i]`;
    // With a row cache, store through and keep the stored value in the local:
    // `a[i] = r = v`. `r` must equal what re-reading `a[i]` would give, so an
    // f32 column's value is rounded here when fround is off (when it is on,
    // every f32-typed expression is already f32-valued).
    const cached = this.cache.get(`${f.component}.${f.field}`);
    const to = cached !== undefined ? `${slot} = ${cached}` : slot;
    if (FIELD_KINDS[f.fieldKind].integer) return `${to} = ${storeHelperName(f.fieldKind)}(${value});`;
    if (cached !== undefined && !this.fr && f.fieldKind === 'f32') return `${to} = fr(${value});`;
    return `${to} = ${value};`;
  }

  /** One assignment as a single JS statement (ends in `;`). */
  private assignText(s: AssignStmt): string {
    const rhs = this.expr(s.value);
    if (s.target.kind === 'local') {
      const name = this.localName(s.target.id);
      const v = s.op === '=' ? rhs : this.round(`${name} ${s.op.slice(0, 1)} ${rhs}`);
      return `${name} = ${v};`;
    }
    const t = s.target;
    if (t.role !== 0) invalid(this.ir, `kernel writes to "other.${t.field}"; a pairwise kernel may only write through "self"`);
    const v = s.op === '=' ? rhs : this.round(`${this.readField(t)} ${s.op.slice(0, 1)} ${rhs}`);
    return this.storeField(t, v);
  }

  stmt(s: Stmt, depth: number): void {
    switch (s.kind) {
      case 'block':
        this.line(depth, '{');
        this.block(s, depth + 1);
        this.line(depth, '}');
        return;
      case 'decl':
        this.note(depth, s.span);
        this.line(depth, `${this.ir.locals[s.id].mutable ? 'let' : 'const'} ${this.localName(s.id)} = ${this.expr(s.init)};`);
        return;
      case 'assign':
        this.note(depth, s.span);
        this.line(depth, this.assignText(s));
        return;
      case 'if':
        this.note(depth, s.span);
        this.line(depth, `if (${this.expr(s.test)}) {`);
        this.block(s.then, depth + 1);
        if (s.alt) {
          this.line(depth, '} else {');
          this.block(s.alt, depth + 1);
        }
        this.line(depth, '}');
        return;
      case 'for': {
        // Same trip guard and order as the WGSL `loop`: cap, count, test, body,
        // update (a `continue` runs the update, as in JS and WGSL `continuing`).
        this.note(depth, s.span);
        const t = `t${this.trips++}`;
        this.line(depth, '{');
        if (s.init) this.stmt(s.init, depth + 1);
        this.line(depth + 1, `let ${t} = 0;`);
        const update = s.update ? this.assignText(s.update).replace(/;$/, '') : '';
        this.line(depth + 1, `for (;; ${update}) {`);
        this.line(depth + 2, `if (${t} >= ${tripCap(this.ir, s.maxIterations)}) break;`);
        this.line(depth + 2, `${t}++;`);
        if (s.test) this.line(depth + 2, `if (!${this.expr(s.test)}) break;`);
        this.block(s.body, depth + 2);
        this.line(depth + 1, '}');
        this.line(depth, '}');
        return;
      }
      case 'while': {
        this.note(depth, s.span);
        const t = `t${this.trips++}`;
        this.line(depth, '{');
        this.line(depth + 1, `let ${t} = 0;`);
        this.line(depth + 1, 'for (;;) {');
        this.line(depth + 2, `if (${t} >= ${tripCap(this.ir, s.maxIterations)}) break;`);
        this.line(depth + 2, `${t}++;`);
        this.line(depth + 2, `if (!${this.expr(s.test)}) break;`);
        this.block(s.body, depth + 2);
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

  // -- module -----------------------------------------------------------------

  module(): string {
    const ir = this.ir;
    const m = ir.components.length;
    const pairwise = ir.form === 'pairwise';

    // Which uniforms the body reads (declaration order), and which integer kinds it writes.
    const usedUniforms = new Set<string>();
    const selfReads = new Set<string>(); // role-0 fields the body reads (incl. compound-assign targets)
    const walkE = (e: Expr): void => {
      if (e.kind === 'uniform') usedUniforms.add(e.name);
      else if (e.kind === 'field') {
        if (e.role === 0) selfReads.add(`${e.component}.${e.field}`);
      } else if (e.kind === 'unary') walkE(e.arg);
      else if (e.kind === 'binary' || e.kind === 'logical') {
        walkE(e.left);
        walkE(e.right);
      } else if (e.kind === 'cond') {
        walkE(e.test);
        walkE(e.then);
        walkE(e.alt);
      } else if (e.kind === 'call') e.args.forEach(walkE);
    };
    const walkS = (s: Stmt): void => {
      switch (s.kind) {
        case 'block':
          s.body.forEach(walkS);
          return;
        case 'decl':
          walkE(s.init);
          return;
        case 'assign':
          walkE(s.value);
          if (s.op !== '=' && s.target.kind !== 'local' && s.target.role === 0) selfReads.add(`${s.target.component}.${s.target.field}`);
          return;
        case 'if':
          walkE(s.test);
          walkS(s.then);
          if (s.alt) walkS(s.alt);
          return;
        case 'for':
          if (s.init) walkS(s.init);
          if (s.test) walkE(s.test);
          if (s.update) walkS(s.update);
          walkS(s.body);
          return;
        case 'while':
          walkE(s.test);
          walkS(s.body);
          return;
        default:
          return;
      }
    };
    walkS(ir.body);
    const intKinds: IRFieldKind[] = [];
    for (const w of ir.writes) {
      if (FIELD_KINDS[w.fieldKind].integer && intKinds.indexOf(w.fieldKind) < 0) intKinds.push(w.fieldKind);
    }
    intKinds.sort();

    // Column locals: one per (component, field), role 0 and 1 share it (same chunk).
    const colDecls: string[] = [];
    const fields = touchedFields(ir)
      .map((a) => ({ component: a.component, field: a.field, index: fieldIndex(ir, a.component, a.field) }))
      .sort((x, y) => x.component - y.component || x.index - y.index);
    for (const f of fields) {
      const key = `${f.component}.${f.field}`;
      if (this.cols.has(key)) continue;
      const name = `a${f.component}_${f.index}`;
      this.cols.set(key, name);
      colDecls.push(`${name} = c${f.component}[${JSON.stringify(f.field)}]`);
    }

    // Row cache: one local per role-0 field the body reads, loaded once per row.
    // Two kernel components with one id would alias columns: no cache then.
    const ids = new Set(ir.components.map((c) => c.id));
    const cacheOk = ids.size === ir.components.length;
    const written = new Set(ir.writes.map((w) => `${w.component}.${w.field}`));
    const cacheConst: string[] = [];
    const cacheLet: string[] = [];
    if (cacheOk) {
      for (const f of fields) {
        const key = `${f.component}.${f.field}`;
        if (!selfReads.has(key) || this.cache.has(key)) continue;
        const name = `r${f.component}_${f.index}`;
        this.cache.set(key, name);
        (written.has(key) ? cacheLet : cacheConst).push(`${name} = ${this.cols.get(key)}[i]`);
      }
    }

    const params = ['count'];
    for (let c = 0; c < m; c++) params.push(`c${c}`);
    params.push('chunk');

    // Body first (it fixes the trip counter names), then the frame around it.
    const bodyDepth = pairwise ? 4 : 3;
    const head = this.lines.splice(0);
    this.block(ir.body, bodyDepth);
    const body = this.lines.splice(0);
    this.lines.push(...head);

    this.line(0, '"use strict";');
    this.line(0, `// cozyecs/gpu CPU backend: kernel "${ir.name.replace(/[\r\n*/]+/g, ' ')}", form ${ir.form}, ops ${ir.opCount}`);
    this.line(0, `const st = state, U = st.u, fr = Math.fround, rand = cozy_rand;`);
    for (const k of intKinds) this.line(0, storeHelperSource(k));
    this.line(0, `return function cozyKernel_${sanitize(ir.name)}(${params.join(', ')}) {`);
    const perChunk: string[] = [];
    if (ir.usesDt) perChunk.push(`dt = ${this.round('st.dt')}`);
    if (ir.usesCount) perChunk.push(`nCount = ${this.round('count')}`);
    for (const u of ir.uniforms) {
      if (usedUniforms.has(u.name)) perChunk.push(`${this.uniformName(u.name)} = ${this.round(`U[${u.index}]`)}`);
    }
    if (perChunk.length > 0) this.line(1, `const ${perChunk.join(', ')};`);
    if (this.hoisted.size > 0) {
      this.line(1, `const ${[...this.hoisted].map(([text, name]) => `${name} = ${text}`).join(', ')};`);
    }
    if (colDecls.length > 0) this.line(1, `const ${colDecls.join(', ')};`);
    const gids = this.enableIds;
    if (gids.length > 0) {
      this.line(1, `const ${gids.map((id, k) => `g${k} = chunk.enabled[${id | 0}]`).join(', ')};`);
    }
    const skip = (row: string): string => gids.map((_, k) => `g${k}[${row}] === 0`).join(' || ');
    this.line(1, 'for (let i = 0; i < count; i++) {');
    if (gids.length > 0) this.line(2, `if (${skip('i')}) continue;`);
    if (cacheConst.length > 0) this.line(2, `const ${cacheConst.join(', ')};`);
    if (cacheLet.length > 0) this.line(2, `let ${cacheLet.join(', ')};`);
    if (pairwise) {
      this.line(2, 'for (let j = 0; j < count; j++) {');
      this.line(3, gids.length > 0 ? `if (j === i || ${skip('j')}) continue;` : 'if (j === i) continue;');
      // The body sits one level deeper inside a labelled-free block so its own
      // `continue`s (only ever inside kernel loops) cannot reach the j loop.
      this.line(3, '{');
      this.lines.push(...body);
      this.line(3, '}');
      this.line(2, '}');
    } else {
      this.line(2, '{');
      this.lines.push(...body);
      this.line(2, '}');
    }
    this.line(1, '}');
    this.line(0, '};');
    return `${this.lines.join('\n')}\n`;
  }
}

/**
 * True when `e` has the same value for every row of a chunk: built only from
 * literals, uniforms, `dt` and `count` (not `index`, fields or locals). All
 * such expressions are pure, so they can be evaluated once per chunk.
 */
function isInvariant(e: Expr): boolean {
  switch (e.kind) {
    case 'num':
    case 'boollit':
    case 'uniform':
      return true;
    case 'builtinValue':
      return e.name !== 'index';
    case 'unary':
      return isInvariant(e.arg);
    case 'binary':
    case 'logical':
      return isInvariant(e.left) && isInvariant(e.right);
    case 'cond':
      return isInvariant(e.test) && isInvariant(e.then) && isInvariant(e.alt);
    case 'call':
      return e.args.every(isInvariant);
    default:
      return false;
  }
}

function fieldIndex(ir: KernelIR, component: number, field: string): number {
  const f = ir.components[component].fields.find((x) => x.name === field);
  return f ? f.index : 0;
}

function tripCap(ir: KernelIR, maxIterations: number): number {
  const n = Math.floor(maxIterations);
  if (!(n > 0) || !Number.isFinite(n)) invalid(ir, `loop has no usable iteration cap (${maxIterations})`);
  return Math.min(n, 4294967295);
}

// ---------------------------------------------------------------------------
// Closure-tree interpreter (no codegen: CSP without 'unsafe-eval')
// ---------------------------------------------------------------------------

/** Mutable per-call evaluation environment of the interpreter. */
interface Env {
  i: number;
  j: number;
  count: number;
  dt: number;
  /** Rounded uniforms by declaration index. */
  u: number[];
  /** Column arrays by slot. */
  arrs: ArrayLike<number>[];
  locals: unknown[];
}

type EvalFn = (env: Env) => number | boolean;
/** 0 = normal, 1 = break, 2 = continue. */
type ExecFn = (env: Env) => number;

const NORMAL = 0;
const BREAK = 1;
const CONTINUE = 2;

/**
 * Builds the same semantics as the generated loop from closures: every rounding
 * point, trip cap, enable skip and store rule is shared with the codegen path.
 */
function interpretKernel(ir: KernelIR, state: CPUKernelState, opts?: CompileCPUOptions): (count: number, ...args: unknown[]) => void {
  const doRound = !(opts && opts.fround === false);
  const fr = doRound ? Math.fround : (x: number): number => x;
  const enableIds = ((opts && opts.enableIds) || []).slice();
  const m = ir.components.length;

  // Column slots, same dedupe as the codegen path.
  const slots = new Map<string, number>();
  const slotRefs: { component: number; field: string }[] = [];
  for (const a of touchedFields(ir)) {
    const key = `${a.component}.${a.field}`;
    if (!slots.has(key)) {
      slots.set(key, slotRefs.length);
      slotRefs.push({ component: a.component, field: a.field });
    }
  }
  const slotOf = (f: FieldRef): number => slots.get(`${f.component}.${f.field}`) as number;

  const stores = new Map<IRFieldKind, (x: number) => number>();
  const storeFor = (kind: IRFieldKind): ((x: number) => number) | null => {
    const info = FIELD_KINDS[kind];
    if (!info.integer) return null;
    let s = stores.get(kind);
    if (!s) {
      const [lo, hi] = info.range as readonly [number, number];
      s = (x: number): number => {
        if (x !== x) return 0;
        const r = Math.round(x);
        return r < lo ? lo : r > hi ? hi : r;
      };
      stores.set(kind, s);
    }
    return s;
  };

  const readField = (f: FieldRef): EvalFn => {
    const slot = slotOf(f);
    const exact = readIsExactF32(f.fieldKind);
    if (f.role === 1) return exact ? (env) => env.arrs[slot][env.j] : (env) => fr(env.arrs[slot][env.j]);
    return exact ? (env) => env.arrs[slot][env.i] : (env) => fr(env.arrs[slot][env.i]);
  };

  const arith = (op: string, a: EvalFn, b: EvalFn): EvalFn => {
    switch (op) {
      case '+':
        return (env) => fr((a(env) as number) + (b(env) as number));
      case '-':
        return (env) => fr((a(env) as number) - (b(env) as number));
      case '*':
        return (env) => fr((a(env) as number) * (b(env) as number));
      case '/':
        return (env) => fr((a(env) as number) / (b(env) as number));
      case '%':
        return (env) => fr((a(env) as number) % (b(env) as number));
      default:
        return invalid(ir, `unknown operator "${op}"`);
    }
  };

  const ev = (e: Expr): EvalFn => {
    switch (e.kind) {
      case 'num': {
        const v = doRound ? Math.fround(e.value) : e.value;
        return () => v;
      }
      case 'boollit': {
        const v = e.value;
        return () => v;
      }
      case 'field':
        return readField(e);
      case 'uniform': {
        const u = ir.uniforms.find((x) => x.name === e.name);
        if (!u) return invalid(ir, `uniform "${e.name}" is not declared`);
        const k = u.index;
        return (env) => env.u[k];
      }
      case 'local': {
        const id = e.id;
        return (env) => env.locals[id] as number | boolean;
      }
      case 'builtinValue':
        if (e.name === 'dt') return (env) => env.dt;
        if (e.name === 'index') return (env) => fr(env.i);
        return (env) => fr(env.count);
      case 'unary': {
        const a = ev(e.arg);
        return e.op === '!' ? (env) => !a(env) : (env) => -(a(env) as number);
      }
      case 'binary': {
        const a = ev(e.left);
        const b = ev(e.right);
        switch (e.op) {
          case '<':
            return (env) => a(env) < b(env);
          case '<=':
            return (env) => a(env) <= b(env);
          case '>':
            return (env) => a(env) > b(env);
          case '>=':
            return (env) => a(env) >= b(env);
          case '==':
            return (env) => a(env) === b(env);
          case '!=':
            return (env) => a(env) !== b(env);
          default:
            return arith(e.op, a, b);
        }
      }
      case 'logical': {
        const a = ev(e.left);
        const b = ev(e.right);
        return e.op === '&&' ? (env) => !!a(env) && !!b(env) : (env) => !!a(env) || !!b(env);
      }
      case 'cond': {
        const t = ev(e.test);
        const a = ev(e.then);
        const b = ev(e.alt);
        return (env) => (t(env) ? a(env) : b(env));
      }
      case 'call': {
        const args = e.args.map(ev);
        const a0 = args[0];
        const a1 = args[1];
        const n = (f: EvalFn, env: Env): number => f(env) as number;
        switch (e.callee) {
          case 'rand':
            return (env) => cozyRandRef(n(a0, env));
          case 'round':
            return (env) => Math.floor(fr(n(a0, env) + 0.5));
          case 'abs':
            return (env) => Math.abs(n(a0, env));
          case 'floor':
            return (env) => Math.floor(n(a0, env));
          case 'ceil':
            return (env) => Math.ceil(n(a0, env));
          case 'sign':
            return (env) => Math.sign(n(a0, env));
          case 'min':
            return (env) => Math.min(n(a0, env), n(a1, env));
          case 'max':
            return (env) => Math.max(n(a0, env), n(a1, env));
          case 'atan2':
            return (env) => fr(Math.atan2(n(a0, env), n(a1, env)));
          case 'pow':
            return (env) => fr(Math.pow(n(a0, env), n(a1, env)));
          case 'hypot':
            return (env) => fr(Math.hypot(n(a0, env), n(a1, env)));
          default: {
            const f = (Math as unknown as Record<string, (x: number) => number>)[e.callee];
            if (typeof f !== 'function') return invalid(ir, `unknown builtin "${String(e.callee)}"`);
            return (env) => fr(f(n(a0, env)));
          }
        }
      }
      default:
        return invalid(ir, `unknown expression node "${(e as { kind: string }).kind}"`);
    }
  };

  const assignValue = (op: string, read: EvalFn, value: EvalFn): EvalFn => (op === '=' ? value : arith(op.slice(0, 1), read, value));

  const ex = (s: Stmt): ExecFn => {
    switch (s.kind) {
      case 'block': {
        const list = s.body.map(ex);
        const k = list.length;
        return (env) => {
          for (let q = 0; q < k; q++) {
            const r = list[q](env);
            if (r !== NORMAL) return r;
          }
          return NORMAL;
        };
      }
      case 'decl': {
        const id = s.id;
        const init = ev(s.init);
        return (env) => {
          env.locals[id] = init(env);
          return NORMAL;
        };
      }
      case 'assign': {
        if (s.target.kind === 'local') {
          const id = s.target.id;
          const v = assignValue(s.op, (env) => env.locals[id] as number, ev(s.value));
          return (env) => {
            env.locals[id] = v(env);
            return NORMAL;
          };
        }
        const t = s.target;
        if (t.role !== 0) return invalid(ir, `kernel writes to "other.${t.field}"; a pairwise kernel may only write through "self"`);
        const slot = slotOf(t);
        const v = assignValue(s.op, readField(t), ev(s.value));
        const store = storeFor(t.fieldKind);
        if (store) {
          return (env) => {
            (env.arrs[slot] as unknown as number[])[env.i] = store(v(env) as number);
            return NORMAL;
          };
        }
        return (env) => {
          (env.arrs[slot] as unknown as number[])[env.i] = v(env) as number;
          return NORMAL;
        };
      }
      case 'if': {
        const t = ev(s.test);
        const a = ex(s.then);
        const b = s.alt ? ex(s.alt) : null;
        return (env) => (t(env) ? a(env) : b ? b(env) : NORMAL);
      }
      case 'for':
      case 'while': {
        const cap = tripCap(ir, s.maxIterations);
        const init = s.kind === 'for' && s.init ? ex(s.init) : null;
        const test = s.kind === 'for' ? (s.test ? ev(s.test) : null) : ev(s.test);
        const update = s.kind === 'for' && s.update ? ex(s.update) : null;
        const body = ex(s.body);
        return (env) => {
          if (init) init(env);
          for (let trips = 0; ; ) {
            if (trips >= cap) break;
            trips++;
            if (test && !test(env)) break;
            const r = body(env);
            if (r === BREAK) break;
            if (update) update(env);
          }
          return NORMAL;
        };
      }
      case 'break':
        return () => BREAK;
      case 'continue':
        return () => CONTINUE;
      default:
        return invalid(ir, `unknown statement node "${(s as { kind: string }).kind}"`);
    }
  };

  const body = ex(ir.body);
  const pairwise = ir.form === 'pairwise';
  const env: Env = { i: 0, j: 0, count: 0, dt: 0, u: [], arrs: [], locals: new Array(ir.locals.length) };
  const flags: ArrayLike<number>[] = [];

  return function cozyKernelInterpreted(count: number): void {
    // eslint-disable-next-line prefer-rest-params
    const args = arguments;
    const chunk = args[1 + m] as { enabled: (ArrayLike<number> | undefined)[] } | undefined;
    env.count = count;
    env.dt = fr(state.dt);
    for (const u of ir.uniforms) env.u[u.index] = fr(state.u[u.index]);
    for (let q = 0; q < slotRefs.length; q++) {
      const col = args[1 + slotRefs[q].component] as Record<string, ArrayLike<number>>;
      env.arrs[q] = col[slotRefs[q].field];
    }
    flags.length = 0;
    for (const id of enableIds) flags.push((chunk as { enabled: ArrayLike<number>[] }).enabled[id]);
    const off = (row: number): boolean => {
      for (let q = 0; q < flags.length; q++) if (flags[q][row] === 0) return true;
      return false;
    };
    for (let i = 0; i < count; i++) {
      if (flags.length > 0 && off(i)) continue;
      env.i = i;
      if (!pairwise) {
        body(env);
        continue;
      }
      for (let j = 0; j < count; j++) {
        if (j === i || (flags.length > 0 && off(j))) continue;
        env.j = j;
        body(env);
      }
    }
  };
}
