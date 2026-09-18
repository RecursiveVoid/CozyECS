/**
 * JS subset -> {@link KernelIR}.
 *
 * OWNER: developer A. Nothing outside this file may parse JavaScript.
 *
 * The parser is hand-written and dependency-free (a tokenizer plus a Pratt
 * expression parser is roughly 500 lines) because the core promises zero
 * dependencies and the gpu entry must stay small enough to be worth shipping.
 * It parses only what {@link KernelIR} can represent; everything else is a
 * {@link KernelError} with a code frame, never a silent miscompile.
 *
 * WHY POSITIONAL MATCHING: `kernel.toString()` gives the source as written,
 * but a bundler may have renamed every identifier, and a closure variable is
 * invisible from the source. So component parameters are matched by POSITION
 * against `spec.components`, and `dt`/`u` are the parameters after them.
 * Property names (`p.x`, `u.gravity`) survive minification and are matched by
 * name. Any other free identifier is E_UNKNOWN_IDENTIFIER, which is what makes
 * "all outside values must come through `uniforms`" enforceable.
 *
 * GRAMMAR: docs/GPU.md, "Kernel subset". Keep the two in sync; the doc is the
 * user-facing contract and this file is its implementation.
 *
 * PURITY: this file touches no global state -- in particular it never calls
 * `console.warn`. A kernel that touches an f64 or sub-word field parses
 * normally; `gpuBlockers(ir)` reports it and runtime.ts issues the one-time
 * warning, so the same IR can be parsed in a test without printing anything.
 */

import type {
  AssignOp,
  BinaryOp,
  BlockStmt,
  BuiltinName,
  DeclStmt,
  Expr,
  ExprType,
  FieldRef,
  IRComponent,
  IRFieldKind,
  IRLocal,
  IRUniform,
  KernelErrorCode,
  KernelForm,
  KernelIR,
  LocalRef,
  Span,
  Stmt,
  UniformRef,
} from './ir';
import {
  BREAK,
  BUILTINS,
  BUILTIN_VALUES,
  CONTINUE,
  KernelError,
  assign,
  binary,
  block,
  boolLit,
  builtinValue,
  call,
  cond,
  decl,
  field,
  forStmt,
  ifStmt,
  local,
  logical,
  makeKernelIR,
  num,
  unary,
  uniform,
  validateIR,
  walk,
  whileStmt,
} from './ir';

/** Default static trip cap for a loop whose bound the parser cannot prove. */
export const DEFAULT_MAX_LOOP_ITERATIONS = 4096;

/** Everything the parser needs besides the function itself. */
export interface ParseSpec {
  /** Kernel name, for errors and generated-code comments. */
  readonly name: string;
  readonly form: KernelForm;
  /**
   * Components in the order the kernel's parameters take them. For
   * `form: 'pairwise'` the kernel has two parameters (`self`, `other`) that
   * both expose every component's fields merged into one namespace.
   */
  readonly components: readonly IRComponent[];
  /** Declared uniform names. A `u.<name>` outside this list is E_UNKNOWN_UNIFORM. */
  readonly uniformNames: readonly string[];
  /**
   * Registration-time value of each uniform, index-aligned with
   * `uniformNames`. Optional: when omitted every `IRUniform.initial` is 0 and
   * the live values live in `KernelRuntime.uniformValues` (which is what every
   * dispatch reads). Pass it when you want `ir.uniforms` to describe the
   * kernel as registered.
   */
  readonly uniformInitials?: readonly number[];
  /**
   * Source to parse. Defaults to `fn.toString()`; pass it explicitly only in
   * tests, where a hand-written string is easier to assert spans against.
   */
  readonly source?: string;
  /** Cap for loops with no provable bound. Default {@link DEFAULT_MAX_LOOP_ITERATIONS}. */
  readonly maxLoopIterations?: number;
}

/**
 * Parses `fn` into a validated IR.
 *
 * CONTRACT
 *  - Pure: no globals touched, same input gives the same IR (deep-equal), so
 *    `describeIR` output is a usable test snapshot.
 *  - Never returns an invalid IR: it calls `validateIR` before returning.
 *  - Every rejection is a {@link KernelError} carrying a code from the
 *    catalogue, a span into `source` and a hint. Never throws a bare Error.
 *  - `reads`/`writes`/`opCount` come from `deriveFacts`; do not compute them by
 *    hand.
 *  - Accepts arrow functions, function expressions, methods and concise bodies
 *    (`(p, v) => p.x += v.x`). Rejects `async`, generators, default parameter
 *    values, destructuring and rest parameters (E_PARAM_COUNT / E_SYNTAX).
 *  - Parameter arity: `components.length`, `+1` (dt) or `+2` (dt, u) for
 *    per-entity; `2`, `3` or `4` for pairwise. Anything else is E_PARAM_COUNT
 *    with a message naming the expected shape.
 *  - A `for`/`while` gets `maxIterations` from a proved bound when the loop is
 *    `for (let i = 0; i < <literal|uniform-free constant>; i++)`, otherwise
 *    `spec.maxLoopIterations`. The bound is a CAP, not a promise: both backends
 *    stop there, so a runaway loop cannot hang a GPU.
 *
 * @throws KernelError for every unsupported construct.
 */
export function parseKernel(fn: unknown, spec: ParseSpec): KernelIR {
  const source = spec.source !== undefined ? spec.source : kernelSource(fn, spec.name);
  const parser = new Parser(source, spec);
  const ir = parser.parse();
  validateIR(ir);
  return ir;
}

/**
 * `fn.toString()` with the checks that make the result parseable: rejects
 * native/bound functions (`E_NOT_A_FUNCTION`), `async` and generators
 * (`E_KERNEL_ASYNC`). Exported so `kernelSystem` can fail fast before building
 * a spec, and so tests can assert the message without a full parse.
 */
export function kernelSource(fn: unknown, name: string): string {
  if (typeof fn !== 'function') {
    throw new KernelError('E_NOT_A_FUNCTION', `kernel is ${describeValue(fn)}, not a function`, {
      kernelName: name,
      hint: 'Pass the kernel as a plain arrow function, e.g. kernel: (p, v, dt, u) => { ... }.',
    });
  }
  let source: string;
  try {
    source = Function.prototype.toString.call(fn);
  } catch {
    throw new KernelError('E_NOT_A_FUNCTION', 'the kernel\'s source could not be read', {
      kernelName: name,
      hint: 'A native or Proxy-wrapped function has no readable source. Write the kernel inline.',
    });
  }
  if (/\{\s*\[\s*native code\s*\]\s*\}/.test(source)) {
    throw new KernelError('E_NOT_A_FUNCTION', 'the kernel is a native or bound function, so it has no readable source', {
      kernelName: name,
      source,
      hint: 'Do not call .bind() on a kernel; pass the function itself. Uniforms replace bound arguments.',
    });
  }
  if (/^\s*class[\s{]/.test(source)) {
    throw new KernelError('E_NOT_A_FUNCTION', 'the kernel is a class, not a function', { kernelName: name, source });
  }
  if (/^\s*async[\s(]/.test(source)) {
    throw new KernelError('E_KERNEL_ASYNC', 'an async kernel cannot run on the GPU', {
      kernelName: name,
      source,
      hint: 'A kernel is a pure per-entity computation: no await, no promises.',
    });
  }
  if (/^\s*(function\s*\*|\*\s*[A-Za-z_$])/.test(source)) {
    throw new KernelError('E_KERNEL_ASYNC', 'a generator cannot be used as a kernel', { kernelName: name, source });
  }
  return source;
}

/**
 * Parameter names in order, from a kernel source string. Used for error
 * messages only -- never for matching (see WHY POSITIONAL MATCHING above).
 * Returns `[]` for a source it cannot split, rather than throwing.
 */
export function parameterNames(source: string): string[] {
  try {
    const tokens = tokenizeWithCtx(source, { name: '', source });
    const head = parseHeader(tokens, { name: '', source });
    return head.params.map((p) => p.name);
  } catch {
    return [];
  }
}

/**
 * The merged field namespace of a pairwise kernel: field name -> the component
 * that owns it. Throws E_FIELD_COLLISION naming both components when two
 * components expose the same field name.
 *
 * `kernelSystem` calls this at registration so the collision is reported before
 * parsing, with a message about the components rather than about a token.
 */
export function mergedFieldNamespace(
  components: readonly IRComponent[],
  kernelName: string,
): Map<string, { component: number; kind: IRComponent['fields'][number]['kind'] }> {
  const out = new Map<string, { component: number; kind: IRFieldKind }>();
  for (const c of components) {
    for (const f of c.fields) {
      const prev = out.get(f.name);
      if (prev !== undefined) {
        throw new KernelError(
          'E_FIELD_COLLISION',
          `components "${components[prev.component].name}" and "${c.name}" both have a field "${f.name}", ` +
            'so a pairwise kernel cannot tell them apart',
          {
            kernelName,
            hint: `A pairwise kernel merges every component into one namespace (self.${f.name}). Rename one of the fields, or split the kernel.`,
          },
        );
      }
      out.set(f.name, { component: c.index, kind: f.kind });
    }
  }
  return out;
}

/**
 * @internal Test hook: the token stream, so tokenizer bugs can be pinned
 * without going through a full parse. Not exported from `cozyecs/gpu`.
 */
export interface Token {
  readonly type: 'num' | 'ident' | 'punct' | 'keyword';
  readonly value: string;
  readonly span: Span;
}

/** @internal */
export function tokenize(source: string): Token[] {
  return tokenizeWithCtx(source, { name: '', source });
}

// ---------------------------------------------------------------------------
// Diagnostics plumbing
// ---------------------------------------------------------------------------

interface Ctx {
  readonly name: string;
  readonly source: string;
}

function fail(ctx: Ctx, code: KernelErrorCode, message: string, span?: Span, hint?: string): never {
  throw new KernelError(code, message, { kernelName: ctx.name, span, source: ctx.source, hint });
}

function describeValue(v: unknown): string {
  if (v === null) return 'null';
  if (v === undefined) return 'undefined';
  return `a ${typeof v}`;
}

/** Cheap "did you mean" for field and uniform names. */
function suggest(name: string, options: readonly string[]): string | null {
  let best: string = null;
  let bestScore = 3;
  const lower = name.toLowerCase();
  for (const o of options) {
    if (o.toLowerCase() === lower) return o;
    const ol = o.toLowerCase();
    // A truncation (`u.grav` for `gravity`) is the common typo and is far
    // outside an edit-distance window, so match prefixes explicitly.
    if (lower.length >= 2 && (ol.indexOf(lower) === 0 || lower.indexOf(ol) === 0)) return o;
    const d = editDistance(name, o);
    if (d < bestScore) {
      bestScore = d;
      best = o;
    }
  }
  return best;
}

function editDistance(a: string, b: string): number {
  const m = a.length;
  const n = b.length;
  if (Math.abs(m - n) > 2) return 99;
  let prev = new Array<number>(n + 1);
  let cur = new Array<number>(n + 1);
  for (let j = 0; j <= n; j++) prev[j] = j;
  for (let i = 1; i <= m; i++) {
    cur[0] = i;
    for (let j = 1; j <= n; j++) {
      const cost = a.charCodeAt(i - 1) === b.charCodeAt(j - 1) ? 0 : 1;
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost);
    }
    const t = prev;
    prev = cur;
    cur = t;
  }
  return prev[n];
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

const KEYWORDS = new Set([
  'var', 'const', 'let', 'if', 'else', 'for', 'while', 'do', 'break', 'continue', 'return',
  'function', 'class', 'new', 'delete', 'typeof', 'void', 'instanceof', 'in', 'switch', 'case',
  'default', 'try', 'catch', 'finally', 'throw', 'this', 'super', 'null', 'true', 'false',
  'yield', 'with', 'export', 'import', 'extends',
]);

/** Longest-match first. Everything the subset rejects is still tokenized, so the error can point at it. */
const PUNCTUATORS: string[] = [
  '>>>=', '...', '===', '!==', '**=', '<<=', '>>=', '>>>', '&&=', '||=', '??=',
  '=>', '==', '!=', '<=', '>=', '&&', '||', '??', '?.', '++', '--', '+=', '-=', '*=', '/=',
  '%=', '&=', '|=', '^=', '**', '<<', '>>',
  '{', '}', '(', ')', '[', ']', ';', ',', '<', '>', '+', '-', '*', '/', '%', '&', '|', '^',
  '!', '~', '?', ':', '=', '.', '@', '#',
];

function isIdentStart(c: number): boolean {
  return (
    (c >= 97 && c <= 122) || // a-z
    (c >= 65 && c <= 90) || // A-Z
    c === 95 || // _
    c === 36 || // $
    c > 127 // let minifiers use unicode identifiers
  );
}

function isIdentPart(c: number): boolean {
  return isIdentStart(c) || (c >= 48 && c <= 57);
}

function isDigit(c: number): boolean {
  return c >= 48 && c <= 57;
}

function tokenizeWithCtx(source: string, ctx: Ctx): Token[] {
  const out: Token[] = [];
  const n = source.length;
  let i = 0;
  while (i < n) {
    const c = source.charCodeAt(i);
    // whitespace
    if (c === 32 || c === 9 || c === 10 || c === 13 || c === 12 || c === 11 || c === 0xfeff) {
      i++;
      continue;
    }
    // comments
    if (c === 47 /* / */ && i + 1 < n) {
      const c2 = source.charCodeAt(i + 1);
      if (c2 === 47) {
        while (i < n && source.charCodeAt(i) !== 10) i++;
        continue;
      }
      if (c2 === 42) {
        const end = source.indexOf('*/', i + 2);
        if (end === -1) fail(ctx, 'E_SYNTAX', 'unterminated block comment', { start: i, end: n });
        i = end + 2;
        continue;
      }
    }
    // string / template literals: never part of the subset
    if (c === 34 || c === 39 || c === 96) {
      const quote = source[i];
      let j = i + 1;
      while (j < n && source[j] !== quote) j += source[j] === '\\' ? 2 : 1;
      const span = { start: i, end: Math.min(n, j + 1) };
      fail(
        ctx,
        'E_UNSUPPORTED_LITERAL',
        c === 96 ? 'template literals are not part of the kernel subset' : 'string literals are not part of the kernel subset',
        span,
        'A kernel computes numbers only. Pass anything else in through `uniforms` as a number.',
      );
    }
    // numbers
    if (isDigit(c) || (c === 46 /* . */ && i + 1 < n && isDigit(source.charCodeAt(i + 1)))) {
      const start = i;
      if (c === 48 && i + 1 < n && /[xXbBoO]/.test(source[i + 1])) {
        i += 2;
        while (i < n && /[0-9a-fA-F_]/.test(source[i])) i++;
      } else {
        while (i < n && (isDigit(source.charCodeAt(i)) || source[i] === '_')) i++;
        if (i < n && source[i] === '.') {
          i++;
          while (i < n && (isDigit(source.charCodeAt(i)) || source[i] === '_')) i++;
        }
        if (i < n && (source[i] === 'e' || source[i] === 'E')) {
          const save = i;
          i++;
          if (i < n && (source[i] === '+' || source[i] === '-')) i++;
          if (i < n && isDigit(source.charCodeAt(i))) {
            while (i < n && isDigit(source.charCodeAt(i))) i++;
          } else {
            i = save;
          }
        }
      }
      if (i < n && source[i] === 'n') {
        fail(ctx, 'E_UNSUPPORTED_LITERAL', 'BigInt literals are not part of the kernel subset', { start, end: i + 1 }, 'All kernel arithmetic is f32.');
      }
      out.push({ type: 'num', value: source.slice(start, i), span: { start, end: i } });
      continue;
    }
    // identifiers and keywords
    if (isIdentStart(c)) {
      const start = i;
      i++;
      while (i < n && isIdentPart(source.charCodeAt(i))) i++;
      const value = source.slice(start, i);
      out.push({ type: KEYWORDS.has(value) ? 'keyword' : 'ident', value, span: { start, end: i } });
      continue;
    }
    // punctuators
    let matched: string = null;
    for (let k = 0; k < PUNCTUATORS.length; k++) {
      const p = PUNCTUATORS[k];
      if (source.startsWith(p, i)) {
        matched = p;
        break;
      }
    }
    if (matched === null) {
      fail(ctx, 'E_SYNTAX', `unexpected character ${JSON.stringify(source[i])}`, { start: i, end: i + 1 });
    }
    out.push({ type: 'punct', value: matched, span: { start: i, end: i + matched.length } });
    i += matched.length;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Header (signature) parsing
// ---------------------------------------------------------------------------

interface HeaderParam {
  readonly name: string;
  readonly span: Span;
}

interface Header {
  readonly params: HeaderParam[];
  /** Index of the first token of the body. */
  readonly bodyAt: number;
  /** A concise arrow body (`(p, v) => p.x += v.x`) has no braces. */
  readonly concise: boolean;
}

function parseHeader(tokens: Token[], ctx: Ctx): Header {
  let i = 0;
  const tok = (k = 0): Token => tokens[i + k];
  const isP = (v: string, k = 0): boolean => {
    const t = tok(k);
    return t !== undefined && t.type === 'punct' && t.value === v;
  };
  const isK = (v: string, k = 0): boolean => {
    const t = tok(k);
    return t !== undefined && t.type === 'keyword' && t.value === v;
  };

  if (tokens.length === 0) fail(ctx, 'E_SYNTAX', 'the kernel source is empty');

  const first = tokens[0];
  if (first.type === 'ident' && first.value === 'async') {
    fail(ctx, 'E_KERNEL_ASYNC', 'an async kernel cannot run on the GPU', first.span);
  }
  if (isK('class')) fail(ctx, 'E_NOT_A_FUNCTION', 'the kernel is a class, not a function', first.span);

  if (isK('function')) {
    i++;
    if (isP('*')) fail(ctx, 'E_KERNEL_ASYNC', 'a generator cannot be used as a kernel', tok().span);
    if (tok() !== undefined && tok().type === 'ident') i++;
  } else if (
    first.type === 'ident' &&
    (first.value === 'get' || first.value === 'set') &&
    tok(1) !== undefined &&
    (tok(1).type === 'ident' || tok(1).type === 'keyword') &&
    isP('(', 2)
  ) {
    fail(ctx, 'E_KERNEL_ASYNC', `a ${first.value}ter cannot be used as a kernel`, first.span, 'Pass a plain function.');
  } else if (first.type === 'ident' && isP('=>', 1)) {
    // single-parameter arrow: `p => ...`
    const params = [{ name: first.value, span: first.span }];
    return finishArrow(params, 2);
  } else if (first.type === 'ident' && isP('(', 1)) {
    // method shorthand: `move(p, v) { ... }`
    i++;
  } else if (first.type === 'ident' && isP('.', 1)) {
    fail(ctx, 'E_SYNTAX', 'expected a function, got a member expression', first.span);
  } else if (!isP('(')) {
    fail(ctx, 'E_SYNTAX', `expected a function or arrow function, got "${first.value}"`, first.span);
  }

  if (!isP('(')) fail(ctx, 'E_SYNTAX', 'expected a parameter list', tok() ? tok().span : undefined);
  i++; // '('
  const params: HeaderParam[] = [];
  const seen = new Set<string>();
  while (!isP(')')) {
    const t = tok();
    if (t === undefined) fail(ctx, 'E_SYNTAX', 'unterminated parameter list');
    if (t.type === 'punct' && t.value === '...') {
      fail(ctx, 'E_PARAM_COUNT', 'rest parameters are not supported', t.span, 'A kernel takes one parameter per component, then dt, then u.');
    }
    if (t.type === 'punct' && (t.value === '[' || t.value === '{')) {
      fail(ctx, 'E_PARAM_COUNT', 'destructuring parameters are not supported', t.span, 'Parameters are matched by position; take the component itself, e.g. (p, v) => { p.x += v.x; }.');
    }
    if (t.type !== 'ident') fail(ctx, 'E_SYNTAX', `expected a parameter name, got "${t.value}"`, t.span);
    if (seen.has(t.value)) fail(ctx, 'E_SYNTAX', `duplicate parameter name "${t.value}"`, t.span);
    seen.add(t.value);
    params.push({ name: t.value, span: t.span });
    i++;
    if (isP('=')) {
      fail(ctx, 'E_PARAM_COUNT', `parameter "${t.value}" has a default value, which is not supported`, tok().span, 'Parameters are supplied by the runtime, never by you.');
    }
    if (isP(',')) {
      i++;
      continue;
    }
    if (!isP(')')) fail(ctx, 'E_SYNTAX', `expected "," or ")" in the parameter list, got "${tok().value}"`, tok().span);
  }
  i++; // ')'

  if (isP('=>')) return finishArrow(params, i + 1);
  if (isP('{')) return { params, bodyAt: i, concise: false };
  fail(ctx, 'E_SYNTAX', 'expected a function body', tok() ? tok().span : undefined);

  function finishArrow(ps: HeaderParam[], at: number): Header {
    const t = tokens[at];
    if (t === undefined) fail(ctx, 'E_SYNTAX', 'the arrow function has no body');
    return { params: ps, bodyAt: at, concise: !(t.type === 'punct' && t.value === '{') };
  }
}

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

type ParamRole = 'component' | 'dt' | 'uniforms';

interface ParamInfo {
  readonly name: string;
  readonly span: Span;
  readonly role: ParamRole;
  /** Field namespace, for a component parameter. */
  readonly fields?: Map<string, { component: number; kind: IRFieldKind }>;
  /** `FieldRef.role`: 0 for the entity's own row (and `self`), 1 for `other`. */
  readonly fieldRole?: 0 | 1;
  /** Human label used in messages ("Position", "self"). */
  readonly label?: string;
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

const ASSIGN_OPS = new Set(['=', '+=', '-=', '*=', '/=', '%=']);
const BITWISE_OPS = new Set(['|', '&', '^', '<<', '>>', '>>>']);
const REJECTED_ASSIGN_OPS = new Set(['&=', '|=', '^=', '<<=', '>>=', '>>>=', '**=', '&&=', '||=', '??=']);

class Parser {
  private readonly ctx: Ctx;
  private readonly tokens: Token[];
  private readonly spec: ParseSpec;
  private readonly cap: number;
  private pos = 0;

  private params: ParamInfo[] = [];
  private paramByName = new Map<string, ParamInfo>();
  private locals: IRLocal[] = [];
  private scopes: Map<string, number>[] = [];
  private loopDepth = 0;

  constructor(source: string, spec: ParseSpec) {
    this.ctx = { name: spec.name, source };
    this.spec = spec;
    this.tokens = tokenizeWithCtx(source, this.ctx);
    const cap = spec.maxLoopIterations === undefined ? DEFAULT_MAX_LOOP_ITERATIONS : spec.maxLoopIterations;
    if (!(cap >= 1) || !Number.isFinite(cap)) {
      fail(this.ctx, 'E_UNBOUNDED_LOOP', `maxLoopIterations must be a positive number, got ${String(spec.maxLoopIterations)}`);
    }
    this.cap = Math.floor(cap);
  }

  parse(): KernelIR {
    const header = parseHeader(this.tokens, this.ctx);
    this.bindParams(header.params);
    this.pos = header.bodyAt;
    this.scopes.push(new Map());

    let body: BlockStmt;
    if (header.concise) {
      const out: Stmt[] = [];
      const start = this.peek() ? this.peek().span.start : 0;
      this.parseConciseBody(out);
      body = block(out, { start, end: this.prevEnd() });
    } else {
      body = this.parseBlock();
      if (this.pos < this.tokens.length) {
        // Trailing tokens after the body: a bound/decorated function, or a bug.
        const t = this.peek();
        if (!(t.type === 'punct' && (t.value === ';' || t.value === ')'))) {
          fail(this.ctx, 'E_SYNTAX', `unexpected "${t.value}" after the kernel body`, t.span);
        }
      }
    }
    this.scopes.pop();

    const uniforms: IRUniform[] = this.spec.uniformNames.map((name, index) => ({
      name,
      index,
      initial: this.spec.uniformInitials && Number.isFinite(this.spec.uniformInitials[index]) ? this.spec.uniformInitials[index] : 0,
    }));

    return makeKernelIR({
      name: this.spec.name,
      form: this.spec.form,
      components: this.spec.components,
      uniforms,
      locals: this.locals,
      body,
      source: this.ctx.source,
    });
  }

  // -- parameters ----------------------------------------------------------

  private bindParams(header: HeaderParam[]): void {
    const comps = this.spec.components;
    const pairwise = this.spec.form === 'pairwise';
    const nComponents = pairwise ? 2 : comps.length;
    const min = nComponents;
    const max = nComponents + 2;
    if (header.length < min || header.length > max) {
      const shape = pairwise
        ? '(self, other, dt?, u?)'
        : `(${comps.map((c) => c.name.toLowerCase()).join(', ')}, dt?, u?)`;
      fail(
        this.ctx,
        'E_PARAM_COUNT',
        `the kernel takes ${header.length} parameter${header.length === 1 ? '' : 's'}, but ${
          pairwise ? 'a pairwise kernel' : `a kernel over ${comps.length} component${comps.length === 1 ? '' : 's'}`
        } takes ${min}, ${min + 1} or ${max}`,
        header.length ? { start: header[0].span.start, end: header[header.length - 1].span.end } : undefined,
        `Expected ${shape}. Parameters are matched by POSITION, not by name: one per component in the order you listed them, then dt, then u.`,
      );
    }

    const merged = pairwise ? mergedFieldNamespace(comps, this.spec.name) : null;
    for (let i = 0; i < header.length; i++) {
      const h = header[i];
      let info: ParamInfo;
      if (i < nComponents) {
        if (pairwise) {
          info = { name: h.name, span: h.span, role: 'component', fields: merged, fieldRole: i === 0 ? 0 : 1, label: i === 0 ? 'self' : 'other' };
        } else {
          const c = comps[i];
          const fields = new Map<string, { component: number; kind: IRFieldKind }>();
          for (const f of c.fields) fields.set(f.name, { component: c.index, kind: f.kind });
          info = { name: h.name, span: h.span, role: 'component', fields, fieldRole: 0, label: c.name };
        }
      } else if (i === nComponents) {
        info = { name: h.name, span: h.span, role: 'dt', label: 'dt' };
      } else {
        info = { name: h.name, span: h.span, role: 'uniforms', label: 'u' };
      }
      this.params.push(info);
      this.paramByName.set(h.name, info);
    }
  }

  // -- token helpers -------------------------------------------------------

  private peek(k = 0): Token {
    return this.tokens[this.pos + k];
  }

  private prevEnd(): number {
    const t = this.tokens[this.pos - 1];
    return t ? t.span.end : 0;
  }

  private atPunct(v: string, k = 0): boolean {
    const t = this.peek(k);
    return t !== undefined && t.type === 'punct' && t.value === v;
  }

  private atKeyword(v: string, k = 0): boolean {
    const t = this.peek(k);
    return t !== undefined && t.type === 'keyword' && t.value === v;
  }

  private eatPunct(v: string): boolean {
    if (this.atPunct(v)) {
      this.pos++;
      return true;
    }
    return false;
  }

  private expectPunct(v: string, what: string): Token {
    if (!this.atPunct(v)) {
      const t = this.peek();
      fail(this.ctx, 'E_SYNTAX', `expected "${v}" ${what}, got ${t ? `"${t.value}"` : 'the end of the kernel'}`, t ? t.span : this.eofSpan());
    }
    const t = this.peek();
    this.pos++;
    return t;
  }

  private eofSpan(): Span {
    const n = this.ctx.source.length;
    return { start: Math.max(0, n - 1), end: n };
  }

  private spanFrom(start: number): Span {
    return { start, end: this.prevEnd() };
  }

  // -- statements ----------------------------------------------------------

  private parseBlock(): BlockStmt {
    const open = this.expectPunct('{', 'to open a block');
    this.scopes.push(new Map());
    const out: Stmt[] = [];
    while (!this.atPunct('}')) {
      if (this.peek() === undefined) fail(this.ctx, 'E_SYNTAX', 'unterminated block: expected "}"', open.span);
      this.parseStatement(out);
    }
    this.pos++; // '}'
    this.scopes.pop();
    return block(out, { start: open.span.start, end: this.prevEnd() });
  }

  /** A block, or a single statement used as one (`if (c) p.x = 0;`). */
  private parseBody(): BlockStmt {
    if (this.atPunct('{')) return this.parseBlock();
    const start = this.peek() ? this.peek().span.start : 0;
    this.scopes.push(new Map());
    const out: Stmt[] = [];
    this.parseStatement(out);
    this.scopes.pop();
    return block(out, { start, end: this.prevEnd() });
  }

  private parseConciseBody(out: Stmt[]): void {
    const t = this.peek();
    if (t === undefined) fail(this.ctx, 'E_SYNTAX', 'the arrow function has no body');
    const start = t.span.start;
    try {
      this.parseSequenceInto(out);
    } catch (e) {
      if (e instanceof KernelError && e.code === 'E_UNSUPPORTED_STATEMENT' && out.length === 0) {
        fail(
          this.ctx,
          'E_UNSUPPORTED_STATEMENT',
          'a concise kernel body must be an assignment',
          { start, end: this.ctx.source.length },
          'A kernel returns nothing; it writes fields. Write `(p, v) => p.x += v.x`, or use a braced body.',
        );
      }
      throw e;
    }
    this.eatPunct(';');
    if (this.pos < this.tokens.length) {
      const extra = this.peek();
      fail(this.ctx, 'E_UNSUPPORTED_STATEMENT', `unexpected "${extra.value}" after the kernel body`, extra.span, 'Use a braced body for more than one statement.');
    }
  }

  private parseStatement(out: Stmt[]): void {
    const t = this.peek();
    if (t === undefined) fail(this.ctx, 'E_SYNTAX', 'unexpected end of the kernel');

    if (t.type === 'punct') {
      if (t.value === ';') {
        this.pos++;
        return;
      }
      if (t.value === '{') {
        out.push(this.parseBlock());
        return;
      }
    }

    if (t.type === 'keyword') {
      switch (t.value) {
        case 'const':
        case 'let':
          this.parseDeclaration(out, t.value === 'const');
          return;
        case 'var':
          fail(this.ctx, 'E_UNSUPPORTED_STATEMENT', '`var` is not part of the kernel subset', t.span, 'Use `let` or `const`.');
          break;
        case 'if':
          out.push(this.parseIf());
          return;
        case 'for':
          out.push(this.parseFor());
          return;
        case 'while':
          out.push(this.parseWhile());
          return;
        case 'break':
        case 'continue': {
          this.pos++;
          const next = this.peek();
          if (next !== undefined && next.type === 'ident') {
            fail(this.ctx, 'E_UNSUPPORTED_STATEMENT', `labeled \`${t.value}\` is not part of the kernel subset`, { start: t.span.start, end: next.span.end }, 'Restructure with a flag, or use an unlabeled break/continue.');
          }
          if (this.loopDepth === 0) {
            fail(this.ctx, 'E_UNSUPPORTED_STATEMENT', `\`${t.value}\` outside a loop`, t.span);
          }
          this.eatPunct(';');
          out.push(t.value === 'break' ? BREAK : CONTINUE);
          return;
        }
        case 'return':
          fail(
            this.ctx,
            'E_UNSUPPORTED_STATEMENT',
            '`return` is not part of the kernel subset',
            t.span,
            'A kernel produces no value: assign to a component field instead. To skip an entity, wrap the work in `if (...) { ... }`.',
          );
          break;
        case 'do':
          fail(this.ctx, 'E_UNSUPPORTED_STATEMENT', '`do`/`while` is not part of the kernel subset', t.span, 'Use `while (cond) { ... }`; every loop needs a statically bounded trip count.');
          break;
        case 'switch':
          fail(this.ctx, 'E_UNSUPPORTED_STATEMENT', '`switch` is not part of the kernel subset', t.span, 'Use `if`/`else if`.');
          break;
        case 'try':
        case 'catch':
        case 'finally':
        case 'throw':
          fail(this.ctx, 'E_UNSUPPORTED_STATEMENT', `\`${t.value}\` is not part of the kernel subset`, t.span, 'A kernel cannot throw: there is no exception mechanism on a GPU.');
          break;
        case 'function':
        case 'class':
          fail(this.ctx, 'E_UNSUPPORTED_STATEMENT', `a nested ${t.value} is not part of the kernel subset`, t.span, 'Inline the helper by hand; a kernel is a single flat computation.');
          break;
        default:
          fail(this.ctx, 'E_UNSUPPORTED_STATEMENT', `\`${t.value}\` is not part of the kernel subset`, t.span);
      }
    }

    // labeled statement: `outer: for (...)`
    if (t.type === 'ident' && this.atPunct(':', 1)) {
      fail(this.ctx, 'E_UNSUPPORTED_STATEMENT', 'labeled statements are not part of the kernel subset', { start: t.span.start, end: this.peek(1).span.end }, 'Labels are rejected by design (WGSL has no labeled break/continue). Restructure with a flag, or use an unlabeled break/continue.');
    }

    this.parseSequenceInto(out);
    this.eatPunct(';');
  }

  // -- statement-level expression forms ------------------------------------
  //
  // MINIFIED INPUT. A bundler rewrites a kernel body before it ever reaches
  // `toString()`, and the two rewrites that matter are worth supporting
  // exactly because positional matching exists to survive minification:
  //
  //   p.x += v.x * dt; p.y += v.y * dt;      ->  p.x += v.x*dt, p.y += v.y*dt
  //   if (p.y < 0) { p.y = 0; v.y = 0; }     ->  p.y < 0 && (p.y = 0, v.y = 0)
  //
  // Both are accepted ONLY in statement position, where they are exactly
  // equivalent to the statements they came from. A comma or a `&&` whose value
  // is used is still E_UNSUPPORTED_EXPRESSION -- see docs/GPU.md 2.5.

  /** `item (',' item)*` in statement position. */
  private parseSequenceInto(out: Stmt[]): void {
    this.parseStatementItem(out);
    while (this.eatPunct(',')) this.parseStatementItem(out);
  }

  private parseStatementItem(out: Stmt[]): void {
    const t = this.peek();
    if (t === undefined) fail(this.ctx, 'E_SYNTAX', 'unexpected end of the kernel', this.eofSpan());

    const stmt = this.tryParseAssignmentStatement();
    if (stmt !== null) {
      out.push(stmt);
      return;
    }

    // `(a = 1, b = 2)` as a whole statement (a minified `if` body).
    let speculativeError: KernelError = null;
    if (t.type === 'punct' && t.value === '(') {
      const group = this.speculate(() => {
        this.pos++;
        const inner: Stmt[] = [];
        this.parseSequenceInto(inner);
        this.expectPunct(')', 'to close a grouped statement');
        const next = this.peek();
        // If something follows that continues an expression, this was a
        // parenthesized condition, not a group of statements.
        if (next !== undefined && !(next.type === 'punct' && (next.value === ',' || next.value === ';' || next.value === ')' || next.value === '}'))) {
          fail(this.ctx, 'E_SYNTAX', 'not a statement group', next.span);
        }
        return inner;
      });
      if (group !== null) {
        for (const s of group) out.push(s);
        return;
      }
      speculativeError = this.lastSpeculativeError;
    }

    // `cond && (...)`, `cond || (...)`, `cond ? (...) : (...)`: parse the
    // condition with the operators below `&&` so the tail stays available.
    const start = t.span.start;
    let test: Expr;
    try {
      test = this.parseEquality();
    } catch (e) {
      // Both readings failed. Report the one that got further into the source:
      // `(p.x = wrong)` should complain about `wrong`, not about a missing ")".
      throw pickBetterError(speculativeError, e);
    }
    const op = this.peek();
    if (op !== undefined && op.type === 'punct' && (op.value === '&&' || op.value === '||' || op.value === '?')) {
      this.requireBool(test, `the left side of "${op.value}"`, { start, end: this.prevEnd() });
      this.pos++;
      if (op.value === '?') {
        const then = this.branchOf();
        this.expectPunct(':', 'in a `?:` statement');
        const alt = this.branchOf();
        out.push(ifStmt(test, then, alt, { start, end: this.prevEnd() }));
        return;
      }
      const branch = this.branchOf();
      const span = { start, end: this.prevEnd() };
      out.push(op.value === '&&' ? ifStmt(test, branch, null, span) : ifStmt(unary('!', test), branch, null, span));
      return;
    }

    if (speculativeError !== null) throw speculativeError;
    fail(
      this.ctx,
      'E_UNSUPPORTED_STATEMENT',
      'this statement computes a value but does not store it anywhere',
      { start, end: this.prevEnd() },
      test.kind === 'binary' && test.op === '=='
        ? 'This is a comparison, not an assignment: did you mean a single "="?'
        : 'A kernel has no side effects other than assigning to component fields and locals.',
    );
  }

  /** The right-hand side of a statement-level `&&` / `||` / `?:`, as a block. */
  private branchOf(): BlockStmt {
    const start = this.peek() ? this.peek().span.start : this.prevEnd();
    const inner: Stmt[] = [];
    this.parseStatementItem(inner);
    return block(inner, { start, end: this.prevEnd() });
  }

  /**
   * Runs `fn`, restoring the parser (token position AND the locals table) if it
   * throws. Used only where two statement shapes start with the same token.
   */
  private speculate<T>(fn: () => T): T {
    const pos = this.pos;
    const nLocals = this.locals.length;
    const depth = this.scopes.length;
    this.lastSpeculativeError = null;
    try {
      return fn();
    } catch (e) {
      if (!(e instanceof KernelError)) throw e;
      this.pos = pos;
      this.locals.length = nLocals;
      this.scopes.length = depth;
      this.lastSpeculativeError = e.code === 'E_SYNTAX' ? null : e;
      return null;
    }
  }

  /** Why the last {@link speculate} gave up, when it is worth reporting. */
  private lastSpeculativeError: KernelError = null;

  private parseDeclaration(out: Stmt[], isConst: boolean): void {
    const kw = this.peek();
    this.pos++;
    for (;;) {
      const nameTok = this.peek();
      if (nameTok === undefined || nameTok.type !== 'ident') {
        if (nameTok !== undefined && nameTok.type === 'punct' && (nameTok.value === '[' || nameTok.value === '{')) {
          fail(this.ctx, 'E_UNSUPPORTED_STATEMENT', 'destructuring declarations are not part of the kernel subset', nameTok.span, 'Declare one numeric local at a time.');
        }
        fail(this.ctx, 'E_SYNTAX', 'expected a variable name', nameTok ? nameTok.span : this.eofSpan());
      }
      this.pos++;
      if (!this.atPunct('=')) {
        const t = this.peek();
        fail(
          this.ctx,
          'E_SYNTAX',
          `local "${nameTok.value}" must be initialized`,
          t ? { start: nameTok.span.start, end: t.span.end } : nameTok.span,
          'Every local is a number or a boolean and must have a value: `let t = 0;`.',
        );
      }
      this.pos++; // '='
      const init = this.parseExpr();
      const id = this.declareLocal(nameTok.value, init.type, !isConst, nameTok.span);
      out.push(decl(id, init, { start: kw.span.start, end: this.prevEnd() }));
      if (this.eatPunct(',')) continue;
      this.eatPunct(';');
      return;
    }
  }

  private declareLocal(name: string, type: ExprType, mutable: boolean, span: Span): number {
    const scope = this.scopes[this.scopes.length - 1];
    if (scope.has(name)) {
      fail(this.ctx, 'E_BAD_LOCAL', `"${name}" is declared twice in the same scope`, span);
    }
    if (this.paramByName.has(name)) {
      // Shadowing is legal JS but always a mistake here: parameters are matched
      // by position, so a shadowed one is silently unreachable from that point.
      fail(this.ctx, 'E_BAD_LOCAL', `local "${name}" shadows a kernel parameter`, span, 'Rename the local; parameters are matched by position and cannot be reused.');
    }
    const id = this.locals.length;
    this.locals.push({ id, name, type, mutable });
    scope.set(name, id);
    return id;
  }

  private lookupLocal(name: string): number {
    for (let i = this.scopes.length - 1; i >= 0; i--) {
      const id = this.scopes[i].get(name);
      if (id !== undefined) return id;
    }
    return -1;
  }

  private parseIf(): Stmt {
    const kw = this.peek();
    this.pos++;
    this.expectPunct('(', 'after `if`');
    const test = this.parseExpr();
    this.requireBool(test, 'the condition of `if`');
    this.expectPunct(')', 'after the `if` condition');
    const then = this.parseBody();
    let alt: BlockStmt = null;
    if (this.atKeyword('else')) {
      this.pos++;
      if (this.atKeyword('if')) {
        const inner = this.parseIf();
        alt = block([inner], inner.span);
      } else {
        alt = this.parseBody();
      }
    }
    return ifStmt(test, then, alt, this.spanFrom(kw.span.start));
  }

  private parseWhile(): Stmt {
    const kw = this.peek();
    this.pos++;
    this.expectPunct('(', 'after `while`');
    const test = this.parseExpr();
    this.requireBool(test, 'the condition of `while`');
    this.expectPunct(')', 'after the `while` condition');
    this.loopDepth++;
    const body = this.parseBody();
    this.loopDepth--;
    const span = this.spanFrom(kw.span.start);
    this.requireProgress(test, null, body, span, 'while');
    return whileStmt(test, body, this.cap, span);
  }

  private parseFor(): Stmt {
    const kw = this.peek();
    this.pos++;
    this.expectPunct('(', 'after `for`');
    this.scopes.push(new Map());

    let init: DeclStmt = null;
    if (!this.atPunct(';')) {
      if (this.atKeyword('const') || this.atKeyword('let')) {
        const decls: Stmt[] = [];
        this.parseDeclaration(decls, this.peek().value === 'const');
        if (decls.length !== 1 || decls[0].kind !== 'decl') {
          fail(this.ctx, 'E_UNSUPPORTED_STATEMENT', 'a `for` initializer declares exactly one local', kw.span);
        }
        init = decls[0] as DeclStmt;
      } else if (this.atKeyword('var')) {
        fail(this.ctx, 'E_UNSUPPORTED_STATEMENT', '`var` is not part of the kernel subset', this.peek().span, 'Use `let i = 0`.');
      } else {
        const t = this.peek();
        fail(this.ctx, 'E_UNSUPPORTED_STATEMENT', 'a `for` initializer must declare its counter', t.span, 'Write `for (let i = 0; i < n; i += 1) { ... }`.');
      }
    } else {
      this.pos++; // the ';' that parseDeclaration would otherwise have eaten
    }

    let test: Expr = null;
    if (!this.atPunct(';')) {
      test = this.parseExpr();
      this.requireBool(test, 'the condition of `for`');
    }
    this.expectPunct(';', 'after the `for` condition');

    let update: Stmt = null;
    if (!this.atPunct(')')) {
      update = this.tryParseAssignmentStatement();
      if (update === null) {
        const t = this.peek();
        fail(this.ctx, 'E_UNSUPPORTED_STATEMENT', 'the update clause of a `for` must be an assignment', t ? t.span : this.eofSpan(), 'Write `i += 1` (or `i++`).');
      }
    }
    this.expectPunct(')', 'after the `for` clauses');

    this.loopDepth++;
    const body = this.parseBody();
    this.loopDepth--;
    this.scopes.pop();

    const span = this.spanFrom(kw.span.start);
    const updateAssign = update !== null && update.kind === 'assign' ? update : null;
    if (update !== null && updateAssign === null) {
      fail(this.ctx, 'E_UNSUPPORTED_STATEMENT', 'the update clause of a `for` must be an assignment', span);
    }
    const trips = this.boundFor(init, test, updateAssign, body, span);
    return forStmt(init, test, updateAssign, body, trips, span);
  }

  // -- assignments ---------------------------------------------------------

  /**
   * Parses `<target> <op> <expr>`, `<target>++` or `++<target>` if that is what
   * comes next, and returns null (restoring the position) if it is not. All the
   * "you cannot assign to that" diagnostics live here.
   */
  private tryParseAssignmentStatement(): Stmt {
    const save = this.pos;
    const t = this.peek();
    if (t === undefined) return null;

    // prefix ++/--
    if (t.type === 'punct' && (t.value === '++' || t.value === '--')) {
      this.pos++;
      const target = this.parseAssignTarget();
      if (target === null) {
        this.pos = save;
        return null;
      }
      return this.makeUpdate(target, t.value, this.spanFrom(t.span.start));
    }

    if (t.type !== 'ident') return null;
    const isMember = this.atPunct('.', 1);
    const opTok = this.peek(isMember ? 3 : 1);
    if (opTok === undefined || opTok.type !== 'punct') return null;
    if (opTok.value === '++' || opTok.value === '--') {
      const target = this.parseAssignTarget();
      if (target === null) {
        this.pos = save;
        return null;
      }
      this.pos++; // the ++/--
      return this.makeUpdate(target, opTok.value, this.spanFrom(t.span.start));
    }
    if (REJECTED_ASSIGN_OPS.has(opTok.value)) {
      fail(
        this.ctx,
        'E_UNSUPPORTED_EXPRESSION',
        `the "${opTok.value}" operator is not part of the kernel subset`,
        opTok.span,
        opTok.value === '**=' ? 'Use `x = Math.pow(x, n)`.' : 'All kernel arithmetic is f32: there are no bitwise or logical-assignment operators.',
      );
    }
    if (!ASSIGN_OPS.has(opTok.value)) return null;

    const target = this.parseAssignTarget();
    if (target === null) {
      this.pos = save;
      return null;
    }
    this.pos++; // the operator
    const op = opTok.value as AssignOp;
    const value = this.parseExpr();
    return this.makeAssign(target, op, value, this.spanFrom(t.span.start), opTok.span);
  }

  private makeUpdate(target: FieldRef | LocalRef, op: string, span: Span): Stmt {
    if (target.type !== 'f32') {
      fail(this.ctx, 'E_UNSUPPORTED_EXPRESSION', `"${op}" needs a numeric target`, span);
    }
    return assign(target, op === '++' ? '+=' : '-=', num(1), span);
  }

  private makeAssign(target: FieldRef | LocalRef, op: AssignOp, value: Expr, span: Span, opSpan: Span): Stmt {
    if (target.kind === 'local') {
      const l = this.locals[target.id];
      if (op !== '=' && l.type !== 'f32') {
        fail(this.ctx, 'E_UNSUPPORTED_EXPRESSION', `"${op}" needs a numeric target, but "${l.name}" is a boolean`, opSpan);
      }
      if (value.type !== l.type) {
        fail(
          this.ctx,
          'E_UNSUPPORTED_EXPRESSION',
          `"${l.name}" is a ${l.type === 'f32' ? 'number' : 'boolean'} but the value assigned is a ${value.type === 'f32' ? 'number' : 'boolean'}`,
          span,
          'A local keeps the type it was declared with; there is no implicit conversion.',
        );
      }
    } else if (value.type !== 'f32') {
      fail(
        this.ctx,
        'E_UNSUPPORTED_EXPRESSION',
        'a component field can only be assigned a number, but this value is a boolean',
        span,
        'Use a ternary to pick a number: `p.flag = (v.x > 0) ? 1 : 0;`.',
      );
    }
    return assign(target, op, value, span);
  }

  /**
   * `ident` or `ident.field` in assignment position. Returns null only when the
   * tokens are not shaped like a target at all; everything that is shaped like
   * one but cannot be written to throws.
   */
  private parseAssignTarget(): FieldRef | LocalRef {
    const t = this.peek();
    if (t === undefined || t.type !== 'ident') return null;
    const name = t.value;
    const hasMember = this.atPunct('.', 1);
    const propTok = hasMember ? this.peek(2) : undefined;
    if (hasMember && (propTok === undefined || (propTok.type !== 'ident' && propTok.type !== 'keyword'))) return null;

    const param = this.paramByName.get(name);
    const localId = this.lookupLocal(name);

    if (localId >= 0) {
      if (hasMember) {
        fail(this.ctx, 'E_UNSUPPORTED_MEMBER', `"${name}" is a local number, not an object`, { start: t.span.start, end: propTok.span.end });
      }
      const l = this.locals[localId];
      if (!l.mutable) {
        fail(this.ctx, 'E_ASSIGN_TO_CONST', `"${name}" is declared const`, t.span, 'Declare it with `let` if the kernel needs to change it.');
      }
      this.pos++;
      return local(localId, l.type, t.span);
    }

    if (param === undefined) {
      this.unknownIdentifier(t);
    }

    if (param.role === 'dt') {
      fail(this.ctx, 'E_ASSIGN_TO_CONST', 'the `dt` parameter is read-only', t.span, 'Copy it into a local first: `let step = dt;`.');
    }
    if (param.role === 'uniforms') {
      const span = hasMember ? { start: t.span.start, end: propTok.span.end } : t.span;
      fail(this.ctx, 'E_ASSIGN_TO_CONST', 'uniforms are read-only inside a kernel', span, 'Change a uniform from the host with handle.setUniform(name, value).');
    }
    // component parameter
    if (!hasMember) {
      fail(
        this.ctx,
        'E_ASSIGN_TO_CONST',
        `"${name}" is a component, not a value`,
        t.span,
        `Assign to one of its fields instead, e.g. ${name}.${firstFieldName(param)} = ...`,
      );
    }
    const ref = this.resolveField(param, t, propTok);
    if (ref.role === 1) {
      fail(
        this.ctx,
        'E_UNSUPPORTED_STATEMENT',
        `a pairwise kernel cannot write to "${name}.${propTok.value}" (the *other* entity)`,
        { start: t.span.start, end: propTok.span.end },
        'Every pair is evaluated from both sides, so writing to `other` would race. Accumulate into `self`; the symmetric half happens when the roles swap.',
      );
    }
    this.pos += 3;
    return ref;
  }

  // -- expressions (precedence climbing) -----------------------------------

  private parseExpr(): Expr {
    return this.parseConditional();
  }

  private parseConditional(): Expr {
    const test = this.parseOr();
    if (!this.atPunct('?')) return test;
    const q = this.peek();
    this.pos++;
    this.requireBool(test, 'the condition of `?:`');
    const then = this.parseExpr();
    this.expectPunct(':', 'in a `?:` expression');
    const alt = this.parseExpr();
    if (then.type !== alt.type) {
      fail(
        this.ctx,
        'E_UNSUPPORTED_EXPRESSION',
        `the branches of \`?:\` are a ${then.type === 'f32' ? 'number' : 'boolean'} and a ${alt.type === 'f32' ? 'number' : 'boolean'}`,
        q.span,
        'Both branches must have the same type.',
      );
    }
    return cond(test, then, alt, { start: test.span ? test.span.start : q.span.start, end: this.prevEnd() });
  }

  private parseOr(): Expr {
    let left = this.parseAnd();
    while (this.atPunct('||')) {
      const op = this.peek();
      this.pos++;
      const right = this.parseAnd();
      this.requireBool(left, '`||`', op.span);
      this.requireBool(right, '`||`', op.span);
      left = logical('||', left, right, this.joinSpan(left, op));
    }
    if (this.atPunct('??')) {
      fail(this.ctx, 'E_UNSUPPORTED_EXPRESSION', 'the "??" operator is not part of the kernel subset', this.peek().span, 'Every kernel value is defined; there is nothing to default.');
    }
    const t = this.peek();
    if (t !== undefined && t.type === 'punct' && BITWISE_OPS.has(t.value)) {
      fail(
        this.ctx,
        'E_UNSUPPORTED_EXPRESSION',
        `the "${t.value}" operator is not part of the kernel subset`,
        t.span,
        'All kernel arithmetic is f32; there are no integers to operate on bitwise. Use Math.floor / Math.pow instead of `| 0` and shifts.',
      );
    }
    return left;
  }

  private parseAnd(): Expr {
    let left = this.parseEquality();
    while (this.atPunct('&&')) {
      const op = this.peek();
      this.pos++;
      const right = this.parseEquality();
      this.requireBool(left, '`&&`', op.span);
      this.requireBool(right, '`&&`', op.span);
      left = logical('&&', left, right, this.joinSpan(left, op));
    }
    return left;
  }

  private parseEquality(): Expr {
    let left = this.parseRelational();
    for (;;) {
      const t = this.peek();
      if (t === undefined || t.type !== 'punct') break;
      let op: BinaryOp;
      if (t.value === '==' || t.value === '===') op = '==';
      else if (t.value === '!=' || t.value === '!==') op = '!=';
      else break;
      this.pos++;
      const right = this.parseRelational();
      if (left.type !== right.type) {
        fail(
          this.ctx,
          'E_UNSUPPORTED_EXPRESSION',
          `"${t.value}" compares a ${left.type === 'f32' ? 'number' : 'boolean'} with a ${right.type === 'f32' ? 'number' : 'boolean'}`,
          t.span,
        );
      }
      left = binary(op, left, right, this.joinSpan(left, t));
    }
    return left;
  }

  private parseRelational(): Expr {
    let left = this.parseAdditive();
    for (;;) {
      const t = this.peek();
      if (t === undefined) break;
      if (t.type === 'keyword' && (t.value === 'in' || t.value === 'instanceof')) {
        fail(this.ctx, 'E_UNSUPPORTED_EXPRESSION', `the "${t.value}" operator is not part of the kernel subset`, t.span);
      }
      if (t.type !== 'punct') break;
      if (t.value !== '<' && t.value !== '<=' && t.value !== '>' && t.value !== '>=') break;
      this.pos++;
      const right = this.parseAdditive();
      this.requireNum(left, `"${t.value}"`, t.span);
      this.requireNum(right, `"${t.value}"`, t.span);
      left = binary(t.value as BinaryOp, left, right, this.joinSpan(left, t));
    }
    return left;
  }

  private parseAdditive(): Expr {
    let left = this.parseMultiplicative();
    for (;;) {
      const t = this.peek();
      if (t === undefined || t.type !== 'punct') break;
      if (t.value !== '+' && t.value !== '-') break;
      this.pos++;
      const right = this.parseMultiplicative();
      this.requireNum(left, `"${t.value}"`, t.span);
      this.requireNum(right, `"${t.value}"`, t.span);
      left = binary(t.value as BinaryOp, left, right, this.joinSpan(left, t));
    }
    return left;
  }

  private parseMultiplicative(): Expr {
    let left = this.parseUnary();
    for (;;) {
      const t = this.peek();
      if (t === undefined || t.type !== 'punct') break;
      if (t.value === '**') {
        fail(this.ctx, 'E_UNSUPPORTED_EXPRESSION', 'the "**" operator is not part of the kernel subset', t.span, 'Use Math.pow(x, y).');
      }
      if (t.value !== '*' && t.value !== '/' && t.value !== '%') break;
      this.pos++;
      const right = this.parseUnary();
      this.requireNum(left, `"${t.value}"`, t.span);
      this.requireNum(right, `"${t.value}"`, t.span);
      left = binary(t.value as BinaryOp, left, right, this.joinSpan(left, t));
    }
    return left;
  }

  private parseUnary(): Expr {
    const t = this.peek();
    if (t === undefined) fail(this.ctx, 'E_SYNTAX', 'unexpected end of the kernel', this.eofSpan());
    if (t.type === 'punct') {
      if (t.value === '-') {
        this.pos++;
        const arg = this.parseUnary();
        this.requireNum(arg, 'unary "-"', t.span);
        if (arg.kind === 'num') return num(-arg.value, { start: t.span.start, end: this.prevEnd() });
        return unary('-', arg, { start: t.span.start, end: this.prevEnd() });
      }
      if (t.value === '+') {
        this.pos++;
        const arg = this.parseUnary();
        this.requireNum(arg, 'unary "+"', t.span);
        return arg;
      }
      if (t.value === '!') {
        this.pos++;
        // Minifiers write `true`/`false` as `!0`/`!1`.
        const lit = this.peek();
        if (lit !== undefined && lit.type === 'num' && (lit.value === '0' || lit.value === '1')) {
          this.pos++;
          return boolLit(lit.value === '0', { start: t.span.start, end: lit.span.end });
        }
        const arg = this.parseUnary();
        this.requireBool(arg, 'the operand of "!"', t.span);
        return unary('!', arg, { start: t.span.start, end: this.prevEnd() });
      }
      if (t.value === '~') {
        fail(this.ctx, 'E_UNSUPPORTED_EXPRESSION', 'bitwise operators are not part of the kernel subset', t.span, 'All kernel arithmetic is f32; use Math.floor for truncation.');
      }
      if (t.value === '++' || t.value === '--') {
        fail(this.ctx, 'E_UNSUPPORTED_EXPRESSION', `"${t.value}" is only allowed as a statement or as a \`for\` update`, t.span, 'Write `x += 1` inside an expression.');
      }
    }
    if (t.type === 'keyword' && (t.value === 'typeof' || t.value === 'void' || t.value === 'delete')) {
      fail(this.ctx, 'E_UNSUPPORTED_EXPRESSION', `the "${t.value}" operator is not part of the kernel subset`, t.span);
    }
    return this.parsePostfix();
  }

  private parsePostfix(): Expr {
    const e = this.parsePrimary();
    for (;;) {
      const t = this.peek();
      if (t === undefined || t.type !== 'punct') return e;
      if (t.value === '++' || t.value === '--') {
        fail(this.ctx, 'E_UNSUPPORTED_EXPRESSION', `"${t.value}" is only allowed as a statement or as a \`for\` update`, t.span, 'Write `x += 1`.');
      }
      if (t.value === '?.') {
        fail(this.ctx, 'E_UNSUPPORTED_EXPRESSION', 'optional chaining is not part of the kernel subset', t.span);
      }
      if (t.value === '[') {
        fail(this.ctx, 'E_UNSUPPORTED_MEMBER', 'indexing is not part of the kernel subset', t.span, 'A kernel sees one entity at a time; there are no arrays.');
      }
      if (t.value === '.') {
        const prop = this.peek(1);
        fail(
          this.ctx,
          'E_UNSUPPORTED_MEMBER',
          'only `<parameter>.<field>` and `u.<uniform>` property accesses are supported',
          { start: t.span.start, end: prop ? prop.span.end : t.span.end },
        );
      }
      if (t.value === '(') {
        fail(this.ctx, 'E_UNSUPPORTED_CALL', 'only Math.* builtins and rand(seed) can be called from a kernel', t.span);
      }
      return e;
    }
  }

  /** True when the `(` at the cursor opens an arrow function's parameter list. */
  private looksLikeArrow(): boolean {
    let depth = 0;
    for (let i = this.pos; i < this.tokens.length; i++) {
      const t = this.tokens[i];
      if (t.type !== 'punct') continue;
      if (t.value === '(') depth++;
      else if (t.value === ')') {
        depth--;
        if (depth === 0) {
          const next = this.tokens[i + 1];
          return next !== undefined && next.type === 'punct' && next.value === '=>';
        }
      }
    }
    return false;
  }

  private parsePrimary(): Expr {
    const t = this.peek();
    if (t === undefined) fail(this.ctx, 'E_SYNTAX', 'unexpected end of the kernel', this.eofSpan());

    if (t.type === 'num') {
      this.pos++;
      const value = Number(t.value.replace(/_/g, ''));
      if (!Number.isFinite(value)) fail(this.ctx, 'E_SYNTAX', `"${t.value}" is not a finite number`, t.span);
      return num(value, t.span);
    }

    if (t.type === 'keyword') {
      if (t.value === 'true' || t.value === 'false') {
        this.pos++;
        return boolLit(t.value === 'true', t.span);
      }
      if (t.value === 'new') {
        fail(this.ctx, 'E_UNSUPPORTED_LITERAL', '`new` is not part of the kernel subset', t.span, 'A kernel allocates nothing.');
      }
      if (t.value === 'null') {
        fail(this.ctx, 'E_UNSUPPORTED_LITERAL', '`null` is not part of the kernel subset', t.span, 'Every kernel value is a number or a boolean.');
      }
      if (t.value === 'this' || t.value === 'super') {
        fail(this.ctx, 'E_UNKNOWN_IDENTIFIER', `\`${t.value}\` is not available inside a kernel`, t.span, 'A kernel has no receiver; pass values through `uniforms`.');
      }
      if (t.value === 'function') {
        fail(this.ctx, 'E_UNSUPPORTED_STATEMENT', 'nested functions are not part of the kernel subset', t.span, 'Inline the helper by hand.');
      }
      fail(this.ctx, 'E_UNSUPPORTED_EXPRESSION', `\`${t.value}\` is not part of the kernel subset`, t.span);
    }

    if (t.type === 'punct') {
      if (t.value === '(') {
        if (this.looksLikeArrow()) {
          fail(this.ctx, 'E_UNSUPPORTED_STATEMENT', 'nested functions are not part of the kernel subset', t.span, 'Inline the helper by hand; a kernel is one flat computation over one entity.');
        }
        this.pos++;
        const e = this.parseExpr();
        if (this.atPunct(',')) {
          fail(this.ctx, 'E_UNSUPPORTED_EXPRESSION', 'the comma operator is not part of the kernel subset', this.peek().span, 'Write one statement per line.');
        }
        this.expectPunct(')', 'to close a parenthesized expression');
        return e;
      }
      if (t.value === '{') {
        fail(this.ctx, 'E_UNSUPPORTED_LITERAL', 'object literals are not part of the kernel subset', t.span, 'Every kernel value is a number or a boolean; pass structured data as separate numeric `uniforms` (e.g. u.windX, u.windY).');
      }
      if (t.value === '[') {
        fail(this.ctx, 'E_UNSUPPORTED_LITERAL', 'array literals are not part of the kernel subset', t.span, 'A kernel sees one entity at a time; there are no arrays.');
      }
      fail(this.ctx, 'E_SYNTAX', `unexpected "${t.value}"`, t.span);
    }

    // identifier
    const name = t.value;
    if (this.atPunct('=>', 1)) {
      fail(this.ctx, 'E_UNSUPPORTED_STATEMENT', 'nested functions are not part of the kernel subset', { start: t.span.start, end: this.peek(1).span.end }, 'Inline the helper by hand.');
    }
    if (this.atPunct('?.', 1)) {
      fail(this.ctx, 'E_UNSUPPORTED_EXPRESSION', 'optional chaining is not part of the kernel subset', this.peek(1).span, 'Component fields always exist; write `p.x`.');
    }
    const isCall = this.atPunct('(', 1);
    const isMember = this.atPunct('.', 1);
    const propTok = isMember ? this.peek(2) : undefined;

    // Math.<fn>(...)
    if (name === 'Math') {
      if (!isMember) {
        fail(this.ctx, 'E_UNKNOWN_IDENTIFIER', '`Math` can only be used as `Math.<fn>(...)`', t.span, `Available: ${mathBuiltinList()}.`);
      }
      if (propTok === undefined || (propTok.type !== 'ident' && propTok.type !== 'keyword')) {
        fail(this.ctx, 'E_UNSUPPORTED_MEMBER', 'expected a Math function name', t.span);
      }
      return this.parseBuiltinCall(propTok.value, t, propTok, true);
    }

    const param = this.paramByName.get(name);
    const localId = this.lookupLocal(name);

    if (localId >= 0) {
      if (isMember) {
        fail(this.ctx, 'E_UNSUPPORTED_MEMBER', `"${name}" is a local number, not an object`, { start: t.span.start, end: propTok.span.end });
      }
      if (isCall) {
        fail(this.ctx, 'E_UNSUPPORTED_CALL', `"${name}" is a local number, not a function`, t.span);
      }
      this.pos++;
      return local(localId, this.locals[localId].type, t.span);
    }

    if (param !== undefined) {
      if (param.role === 'dt') {
        if (isMember) {
          fail(
            this.ctx,
            'E_UNSUPPORTED_MEMBER',
            `"${name}" is the dt parameter, which is a number, not an object`,
            { start: t.span.start, end: propTok.span.end },
            `Parameters are matched BY POSITION, not by name: "${name}" is parameter #${this.params.indexOf(param) + 1}, which is the dt slot. ` +
              `The expected shape is (${this.paramShape()}).`,
          );
        }
        this.pos++;
        return builtinValue('dt', t.span);
      }
      if (param.role === 'uniforms') {
        if (!isMember) {
          fail(this.ctx, 'E_UNSUPPORTED_MEMBER', 'the uniform block can only be read as `u.<name>`', t.span, `Declared uniforms: ${this.uniformList()}.`);
        }
        if (propTok === undefined || (propTok.type !== 'ident' && propTok.type !== 'keyword')) {
          fail(this.ctx, 'E_UNSUPPORTED_MEMBER', 'expected a uniform name after `u.`', t.span);
        }
        return this.parseUniformRef(t, propTok);
      }
      // component parameter
      if (!isMember) {
        fail(
          this.ctx,
          'E_UNSUPPORTED_MEMBER',
          `"${name}" is a component, not a number`,
          t.span,
          `Read one of its fields, e.g. ${name}.${firstFieldName(param)}.`,
        );
      }
      if (propTok === undefined || (propTok.type !== 'ident' && propTok.type !== 'keyword')) {
        fail(this.ctx, 'E_UNSUPPORTED_MEMBER', `expected a field name after "${name}."`, t.span);
      }
      const ref = this.resolveField(param, t, propTok);
      this.pos += 3;
      return ref;
    }

    // builtin values
    if (!isMember && !isCall && Object.prototype.hasOwnProperty.call(BUILTIN_VALUES, name)) {
      this.pos++;
      return builtinValue(name as 'index' | 'count', t.span);
    }

    // rand(seed)
    if (name === 'rand' && isCall) {
      return this.parseBuiltinCall('rand', t, t, false);
    }

    if (isCall) {
      const onMath = Object.prototype.hasOwnProperty.call(BUILTINS, name) && (BUILTINS as Record<string, { onMath: boolean }>)[name].onMath;
      fail(
        this.ctx,
        'E_UNSUPPORTED_CALL',
        `cannot call "${name}" from a kernel`,
        { start: t.span.start, end: this.peek(1).span.end },
        onMath
          ? `Did you mean Math.${name}(...)? Builtins live on Math; only rand(seed) is bare.`
          : 'A kernel can only call Math builtins and rand(seed). Inline your helper by hand -- a kernel is one flat computation.',
      );
    }

    this.unknownIdentifier(t);
  }

  private parseUniformRef(objTok: Token, propTok: Token): UniformRef {
    const name = propTok.value;
    const span = { start: objTok.span.start, end: propTok.span.end };
    if (this.spec.uniformNames.indexOf(name) === -1) {
      const near = suggest(name, this.spec.uniformNames);
      fail(
        this.ctx,
        'E_UNKNOWN_UNIFORM',
        `no uniform "${name}"; declared: ${this.uniformList()}`,
        span,
        near ? `Did you mean u.${near}?` : `Add it at registration: uniforms: { ${name}: <value> }.`,
      );
    }
    this.pos += 3;
    if (this.atPunct('(')) {
      fail(this.ctx, 'E_UNSUPPORTED_CALL', `"u.${name}" is a number, not a function`, this.peek().span);
    }
    return uniform(name, span);
  }

  private resolveField(param: ParamInfo, objTok: Token, propTok: Token): FieldRef {
    const name = propTok.value;
    const info = param.fields.get(name);
    const span = { start: objTok.span.start, end: propTok.span.end };
    if (info === undefined) {
      const names = [...param.fields.keys()];
      const near = suggest(name, names);
      const where =
        this.spec.form === 'pairwise'
          ? `the pairwise namespace (${this.spec.components.map((c) => c.name).join(' + ')})`
          : `component "${param.label}"`;
      let hint = near ? `Did you mean ${objTok.value}.${near}?` : `Available fields: ${names.join(', ') || 'none'}.`;
      if (objTok.value === 'u' || objTok.value === 'dt') {
        hint +=
          ` Note that parameters are matched BY POSITION: "${objTok.value}" is parameter #${this.params.indexOf(param) + 1}, ` +
          `so it is the component ${param.label}. Add the missing component parameters, or reorder them.`;
      }
      fail(this.ctx, 'E_UNKNOWN_FIELD', `${where} has no field "${name}"`, span, hint);
    }
    return field(info.component, name, info.kind, param.fieldRole, span);
  }

  private parseBuiltinCall(name: string, startTok: Token, nameTok: Token, onMath: boolean): Expr {
    const info = (BUILTINS as Record<string, { arity: readonly [number, number]; onMath: boolean }>)[name];
    const nameSpan = { start: startTok.span.start, end: nameTok.span.end };
    if (info === undefined || info.onMath !== onMath) {
      if (onMath && (name === 'random')) {
        fail(
          this.ctx,
          'E_UNSUPPORTED_CALL',
          'Math.random() is not available inside a kernel',
          nameSpan,
          'Use the built-in rand(seed): a stateless hash that is bit-identical on both backends, e.g. rand(index + u.frame * 7919).',
        );
      }
      if (onMath && (name === 'PI' || name === 'E' || name === 'LN2' || name === 'SQRT2')) {
        fail(
          this.ctx,
          'E_UNSUPPORTED_MEMBER',
          `Math.${name} is not available inside a kernel`,
          nameSpan,
          `Write the literal (${name === 'PI' ? '3.141592653589793' : 'its value'}) or pass it as a uniform.`,
        );
      }
      fail(
        this.ctx,
        'E_UNSUPPORTED_CALL',
        onMath ? `Math.${name} is not one of the kernel builtins` : `"${name}" is not one of the kernel builtins`,
        nameSpan,
        `Available: ${mathBuiltinList()}, and rand(seed).`,
      );
    }
    // consume `Math . name (` or `rand (`
    this.pos += onMath ? 3 : 1;
    this.expectPunct('(', `after ${onMath ? `Math.${name}` : name}`);
    const args: Expr[] = [];
    if (!this.atPunct(')')) {
      for (;;) {
        const a = this.parseExpr();
        this.requireNum(a, `an argument of ${onMath ? `Math.${name}` : name}`, a.span || nameSpan);
        args.push(a);
        if (this.eatPunct(',')) {
          if (this.atPunct(')')) break; // trailing comma
          continue;
        }
        break;
      }
    }
    this.expectPunct(')', `to close ${onMath ? `Math.${name}` : name}(`);
    if (args.length < info.arity[0] || args.length > info.arity[1]) {
      const want = info.arity[0] === info.arity[1] ? `${info.arity[0]}` : `${info.arity[0]}..${info.arity[1]}`;
      fail(
        this.ctx,
        'E_UNSUPPORTED_CALL',
        `${onMath ? `Math.${name}` : name} takes ${want} argument${info.arity[1] === 1 ? '' : 's'}, got ${args.length}`,
        { start: nameSpan.start, end: this.prevEnd() },
        name === 'hypot' && args.length > 2 ? 'The kernel subset supports the 2-argument form only; nest it: Math.hypot(Math.hypot(x, y), z).' : undefined,
      );
    }
    return call(name as BuiltinName, args, { start: nameSpan.start, end: this.prevEnd() });
  }

  private unknownIdentifier(t: Token): never {
    const name = t.value;
    if (name === 'await') {
      fail(this.ctx, 'E_KERNEL_ASYNC', '`await` cannot be used inside a kernel', t.span, 'A kernel runs synchronously for every entity in one dispatch; read async results before the frame and pass them in through `uniforms`.');
    }
    let hint =
      `Values from outside the kernel must come through \`uniforms\`: register with uniforms: { ${name}: <value> } and read it as \`u.${name}\`. ` +
      'A closure variable is invisible to Function.prototype.toString, so it cannot be captured.';
    if (name === 'dt') {
      hint = `\`dt\` is a parameter, not a global: add it after the ${this.spec.form === 'pairwise' ? 'self/other' : 'component'} parameters, e.g. (${this.paramShape()}).`;
    } else if (name === 'u') {
      hint = `The uniform block is a parameter: add it last, e.g. (${this.paramShape()}).`;
    } else if (name === 'world' || name === 'entity' || name === 'entities') {
      hint = 'A kernel cannot touch the world: no spawning, no add/remove, no destroy. Do structural changes in a normal system.';
    }
    fail(this.ctx, 'E_UNKNOWN_IDENTIFIER', `unknown identifier "${name}"`, t.span, hint);
  }

  private paramShape(): string {
    const names = this.spec.form === 'pairwise' ? ['self', 'other'] : this.spec.components.map((c) => c.name.toLowerCase());
    return [...names, 'dt', 'u'].join(', ');
  }

  private uniformList(): string {
    return this.spec.uniformNames.length ? this.spec.uniformNames.join(', ') : 'none';
  }

  private joinSpan(left: Expr, op: Token): Span {
    return { start: left.span ? left.span.start : op.span.start, end: this.prevEnd() };
  }

  private requireBool(e: Expr, what: string, span?: Span): void {
    if (e.type === 'bool') return;
    fail(
      this.ctx,
      'E_UNSUPPORTED_EXPRESSION',
      `${what} must be a boolean, but this is a number`,
      span || e.span,
      'There is no truthiness in the kernel subset: compare explicitly, e.g. `if (p.y != 0)`.',
    );
  }

  private requireNum(e: Expr, what: string, span?: Span): void {
    if (e.type === 'f32') return;
    fail(
      this.ctx,
      'E_UNSUPPORTED_EXPRESSION',
      `${what} needs a number, but this is a boolean`,
      span || e.span,
      'Turn the boolean into a number with a ternary: `(a < b) ? 1 : 0`.',
    );
  }

  // -- loop bounds ---------------------------------------------------------

  /**
   * The static trip cap for a `for`. A counted loop over constants proves its
   * own bound; anything else gets `maxLoopIterations`, and a loop that cannot
   * make progress at all is E_UNBOUNDED_LOOP.
   */
  private boundFor(init: DeclStmt, test: Expr, update: Stmt, body: BlockStmt, span: Span): number {
    if (test === null) {
      fail(this.ctx, 'E_UNBOUNDED_LOOP', 'a `for` with no condition never terminates', span, 'Give it a bound: `for (let i = 0; i < 8; i += 1)`.');
    }
    const proved = this.proveCountedLoop(init, test, update, body);
    if (proved === 'diverges') {
      let canBreak = false;
      walk(body, (n) => {
        if (n.kind === 'break') canBreak = true;
        return true;
      });
      if (!canBreak) {
        fail(
          this.ctx,
          'E_UNBOUNDED_LOOP',
          'the counter of this `for` moves away from its bound, so the loop never finishes',
          span,
          'Check the direction of the update: `for (let i = 0; i < n; i += 1)` counts up, `for (let i = n; i > 0; i -= 1)` counts down.',
        );
      }
      return this.cap;
    }
    if (proved !== null) {
      if (proved > this.cap) {
        fail(
          this.ctx,
          'E_UNBOUNDED_LOOP',
          `this loop runs ${proved} times, over the cap of ${this.cap}`,
          span,
          'Raise it with maxLoopIterations, or make the loop shorter. Both backends emit the cap, so a kernel can never hang the device.',
        );
      }
      return Math.max(1, proved);
    }
    this.requireProgress(test, update, body, span, 'for');
    return this.cap;
  }

  /**
   * The exact trip count of `for (let i = c0; i <op> K; i += step)`, `'diverges'`
   * when it is a counted loop whose counter can never reach the bound, or null
   * when it is not a counted loop at all.
   */
  private proveCountedLoop(init: DeclStmt, test: Expr, update: Stmt, body: BlockStmt): number | 'diverges' {
    if (init === null || update === null || update.kind !== 'assign' || update.target.kind !== 'local') return null;
    const id = init.id;
    if (update.target.id !== id) return null;
    const from = constValue(init.init);
    if (from === null) return null;

    // the counter must not be touched anywhere else
    let clobbered = false;
    walk(body, (n) => {
      if (n.kind === 'assign' && n.target.kind === 'local' && n.target.id === id) clobbered = true;
      return true;
    });
    if (clobbered) return null;

    let step: number = null;
    if (update.op === '+=') step = constValue(update.value);
    else if (update.op === '-=') {
      const s = constValue(update.value);
      step = s === null ? null : -s;
    } else if (update.op === '=') {
      // i = i + c | i = c + i | i = i - c
      const v = update.value;
      if (v.kind === 'binary' && (v.op === '+' || v.op === '-')) {
        if (v.left.kind === 'local' && v.left.id === id) {
          const c = constValue(v.right);
          if (c !== null) step = v.op === '+' ? c : -c;
        } else if (v.op === '+' && v.right.kind === 'local' && v.right.id === id) {
          const c = constValue(v.left);
          if (c !== null) step = c;
        }
      }
    }
    if (step === null) return null;

    if (test.kind !== 'binary') return null;
    let op = test.op;
    let limit: number = null;
    if (test.left.kind === 'local' && test.left.id === id) {
      limit = constValue(test.right);
    } else if (test.right.kind === 'local' && test.right.id === id) {
      limit = constValue(test.left);
      op = mirrorOp(op);
    }
    if (limit === null) return null;

    if (step === 0) return 'diverges'; // the counter never advances
    let trips: number;
    switch (op) {
      case '<':
        if (step < 0) return from < limit ? 'diverges' : 0;
        trips = Math.ceil((limit - from) / step);
        break;
      case '<=':
        if (step < 0) return from <= limit ? 'diverges' : 0;
        trips = Math.floor((limit - from) / step) + 1;
        break;
      case '>':
        if (step > 0) return from > limit ? 'diverges' : 0;
        trips = Math.ceil((from - limit) / -step);
        break;
      case '>=':
        if (step > 0) return from >= limit ? 'diverges' : 0;
        trips = Math.floor((from - limit) / -step) + 1;
        break;
      case '!=': {
        const d = (limit - from) / step;
        if (!Number.isInteger(d)) return 'diverges';
        if (d < 0) return 'diverges';
        trips = d;
        break;
      }
      default:
        return null;
    }
    if (!Number.isFinite(trips)) return null;
    return Math.max(0, trips);
  }

  /**
   * A loop whose condition can never change, and that cannot `break`, would
   * spin until the trip cap on every entity of every frame. That is always a
   * bug, so it is rejected instead of silently costing `cap` iterations.
   */
  private requireProgress(test: Expr, update: Stmt, body: BlockStmt, span: Span, what: 'for' | 'while'): void {
    const readLocals = new Set<number>();
    const readFields = new Set<string>();
    walk(test, (n) => {
      if (n.kind === 'local') readLocals.add(n.id);
      else if (n.kind === 'field') readFields.add(`${n.component}.${n.field}@${n.role}`);
      return true;
    });

    let progress = false;
    const noteAssign = (n: Stmt): void => {
      if (n.kind !== 'assign') return;
      if (n.target.kind === 'local' && readLocals.has(n.target.id)) progress = true;
      if (n.target.kind === 'field' && readFields.has(`${n.target.component}.${n.target.field}@${n.target.role}`)) progress = true;
    };
    if (update !== null) noteAssign(update);
    walk(body, (n) => {
      if (n.kind === 'break') progress = true;
      else if (n.kind === 'assign') noteAssign(n);
      return true;
    });

    if (!progress) {
      fail(
        this.ctx,
        'E_UNBOUNDED_LOOP',
        `this \`${what}\` loop can never finish: nothing in its condition changes, and it has no \`break\``,
        span,
        'Advance a counter in the loop, or break out of it. Every loop also carries a static trip cap so a kernel cannot hang the device.',
      );
    }
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Constant-folds literal arithmetic. Uniforms and fields are not constants. */
function constValue(e: Expr): number {
  if (e === null || e === undefined) return null;
  switch (e.kind) {
    case 'num':
      return e.value;
    case 'unary': {
      if (e.op !== '-') return null;
      const v = constValue(e.arg);
      return v === null ? null : -v;
    }
    case 'binary': {
      const l = constValue(e.left);
      const r = constValue(e.right);
      if (l === null || r === null) return null;
      switch (e.op) {
        case '+':
          return l + r;
        case '-':
          return l - r;
        case '*':
          return l * r;
        case '/':
          return r === 0 ? null : l / r;
        case '%':
          return r === 0 ? null : l % r;
        default:
          return null;
      }
    }
    default:
      return null;
  }
}

/**
 * Of two failed readings of the same tokens, the more useful error is the one
 * that got further into the source (the classic "furthest failure" rule).
 */
function pickBetterError(speculative: KernelError, thrown: unknown): unknown {
  if (speculative === null || !(thrown instanceof KernelError)) return thrown;
  const a = speculative.span ? speculative.span.start : -1;
  const b = thrown.span ? thrown.span.start : -1;
  return a > b ? speculative : thrown;
}

function mirrorOp(op: BinaryOp): BinaryOp {
  switch (op) {
    case '<':
      return '>';
    case '<=':
      return '>=';
    case '>':
      return '<';
    case '>=':
      return '<=';
    default:
      return op;
  }
}

function firstFieldName(param: ParamInfo): string {
  if (param.fields) {
    for (const k of param.fields.keys()) return k;
  }
  return 'x';
}

function mathBuiltinList(): string {
  return Object.keys(BUILTINS)
    .filter((k) => (BUILTINS as Record<string, { onMath: boolean }>)[k].onMath)
    .map((k) => `Math.${k}`)
    .join(', ');
}
