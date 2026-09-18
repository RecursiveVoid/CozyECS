import type { Archetype } from './archetype';
import type { ComponentType } from './component';
import { componentsKey, createMask, maskContains, maskIntersects, normalizeComponents } from './component';
import type { ColumnsOf, EntityCallback, Unsubscribe } from './types';
import type { World } from './world';

const EMPTY_LIST: readonly ComponentType[] = [];
const EMPTY_LISTENERS: EntityCallback[] = [];

/** Copy-on-write append. */
function appendListener(list: EntityCallback[], cb: EntityCallback): EntityCallback[] {
  const out = list.slice();
  out.push(cb);
  return out;
}

/** Copy-on-write removal of one occurrence of `cb`. */
function removeListener(list: EntityCallback[], cb: EntityCallback): EntityCallback[] {
  const i = list.indexOf(cb);
  if (i === -1) return list;
  const out = list.slice();
  out.splice(i, 1);
  return out;
}

/** Callback for `Query.forEach(fn)`. */
export type QueryForEachFn = (entity: number, chunk: Archetype, row: number) => void;

/** Column objects (`chunk.col(C)` results) for a tuple of component types, in the same order. */
export type ColumnsTuple<T extends readonly ComponentType<any>[]> = {
  [I in keyof T]: T[I] extends ComponentType<infer S> ? ColumnsOf<S> : never;
};

/**
 * Callback for `Query.forEach(components, fn)`: receives the column objects of `components`
 * for the row's chunk after (entity, chunk, row), resolved once per chunk.
 */
export type QueryForEachColumnsFn<T extends readonly ComponentType<any>[]> = (
  entity: number,
  chunk: Archetype,
  row: number,
  ...columns: ColumnsTuple<T>
) => void;

/**
 * Callback for `Query.forEachChunk(components, fn)`: called once per non-empty chunk with the
 * chunk's row count, the column objects of `components` (in order) and the chunk itself.
 */
export type ChunkKernelFn<T extends readonly ComponentType<any>[]> = (
  count: number,
  ...args: [...ColumnsTuple<T>, Archetype]
) => void;

/** Any forEach callback (both forms), as seen by the row loops. */
type AnyForEachFn = (entity: number, chunk: Archetype, row: number, ...columns: any[]) => void;

/** Any forEachChunk callback, as seen by the chunk loops. */
type AnyChunkFn = (count: number, ...args: any[]) => void;

/**
 * A row loop. `comps` is null for the plain `forEach(fn)` form; otherwise the columns of
 * `comps` are resolved per chunk and passed after (entity, chunk, row).
 */
type RowLoop = (chunks: Archetype[], ids: Int32Array, comps: readonly ComponentType[] | null, fn: AnyForEachFn) => void;

/** Largest component list passed as separate generated parameters (longer lists use the generic loop). */
const MAX_COLUMN_PARAMS = 8;

/**
 * Generic forEach row loop for the plain form (also the fallback when runtime codegen is
 * unavailable). Rows are visited in REVERSE order; empty chunks are skipped; rows where any
 * enableable component of `ids` is disabled are skipped. Structural changes are deferred while
 * iterating (flush() is a no-op at depth > 0), so a chunk's count, entities, enabled arrays and
 * column arrays cannot change inside its row loop and are read once per chunk (enabled FLAGS
 * are still read per row, so enable/disable inside fn is observed).
 * `rowLoopSource` emits the same loop as source text for specialized copies: keep them in sync.
 */
function forEachRows(chunks: Archetype[], ids: Int32Array, comps: readonly ComponentType[] | null, fn: AnyForEachFn): void {
  if (comps !== null) {
    forEachRowsColumns(chunks, ids, comps, fn);
    return;
  }
  const nChunks = chunks.length;
  const k = ids.length;
  if (k === 0) {
    for (let c = 0; c < nChunks; c++) {
      const chunk = chunks[c];
      const n = chunk.count;
      if (n === 0) continue;
      const ents = chunk.entities;
      for (let row = n - 1; row >= 0; row--) fn(ents[row], chunk, row);
    }
  } else if (k === 1) {
    const id = ids[0];
    for (let c = 0; c < nChunks; c++) {
      const chunk = chunks[c];
      const n = chunk.count;
      if (n === 0) continue;
      const ents = chunk.entities;
      const flags = chunk.enabled[id] as Uint8Array;
      for (let row = n - 1; row >= 0; row--) {
        if (flags[row] !== 0) fn(ents[row], chunk, row);
      }
    }
  } else {
    const flags: Uint8Array[] = [];
    const placeholder = new Uint8Array(0);
    for (let j = 0; j < k; j++) flags.push(placeholder);
    for (let c = 0; c < nChunks; c++) {
      const chunk = chunks[c];
      const n = chunk.count;
      if (n === 0) continue;
      const ents = chunk.entities;
      const enabled = chunk.enabled;
      for (let j = 0; j < k; j++) flags[j] = enabled[ids[j]] as Uint8Array;
      outer: for (let row = n - 1; row >= 0; row--) {
        for (let j = 0; j < k; j++) {
          if (flags[j][row] === 0) continue outer;
        }
        fn(ents[row], chunk, row);
      }
    }
  }
}

/** Generic loop for the `forEach(components, fn)` form: same visiting rules, arguments via apply. */
function forEachRowsColumns(chunks: Archetype[], ids: Int32Array, comps: readonly ComponentType[], fn: AnyForEachFn): void {
  const nChunks = chunks.length;
  const k = ids.length;
  const m = comps.length;
  const args: unknown[] = [0, null, 0];
  for (let j = 0; j < m; j++) args.push(undefined);
  const flags: Uint8Array[] = [];
  for (let j = 0; j < k; j++) flags.push(new Uint8Array(0));
  for (let c = 0; c < nChunks; c++) {
    const chunk = chunks[c];
    const n = chunk.count;
    if (n === 0) continue;
    const ents = chunk.entities;
    const enabled = chunk.enabled;
    for (let j = 0; j < k; j++) flags[j] = enabled[ids[j]] as Uint8Array;
    args[1] = chunk;
    for (let j = 0; j < m; j++) args[3 + j] = chunk.col(comps[j]);
    outer: for (let row = n - 1; row >= 0; row--) {
      for (let j = 0; j < k; j++) {
        if (flags[j][row] === 0) continue outer;
      }
      args[0] = ents[row];
      args[2] = row;
      (fn as (...a: unknown[]) => void).apply(undefined, args);
    }
  }
}

/**
 * Source text of a specialized row loop (parameters: chunks, ids, comps, fn). `m` is -1 for the
 * plain form (fn gets entity, chunk, row), otherwise the number of components (0..MAX_COLUMN_PARAMS)
 * whose columns are loaded into locals once per chunk and passed as extra arguments. Built from
 * string literals rather than `forEachRows.toString()`, so minifiers, bundlers and coverage
 * instrumentation cannot change what gets compiled.
 */
function rowLoopSource(m: number): string {
  let cols = '';
  let args = 'ents[row], chunk, row';
  for (let j = 0; j < m; j++) {
    cols += `    const c${j} = chunk.col(comps[${j}]);\n`;
    args += `, c${j}`;
  }
  const head =
    '    const chunk = chunks[c];\n' +
    '    const n = chunk.count;\n' +
    '    if (n === 0) continue;\n' +
    '    const ents = chunk.entities;\n' +
    cols;
  return (
    'const nChunks = chunks.length;\n' +
    'const k = ids.length;\n' +
    'if (k === 0) {\n' +
    '  for (let c = 0; c < nChunks; c++) {\n' +
    head +
    `    for (let row = n - 1; row >= 0; row--) fn(${args});\n` +
    '  }\n' +
    '} else if (k === 1) {\n' +
    '  const id = ids[0];\n' +
    '  for (let c = 0; c < nChunks; c++) {\n' +
    head +
    '    const flags = chunk.enabled[id];\n' +
    '    for (let row = n - 1; row >= 0; row--) {\n' +
    `      if (flags[row] !== 0) fn(${args});\n` +
    '    }\n' +
    '  }\n' +
    '} else {\n' +
    '  const flags = [];\n' +
    '  const placeholder = new Uint8Array(0);\n' +
    '  for (let j = 0; j < k; j++) flags.push(placeholder);\n' +
    '  for (let c = 0; c < nChunks; c++) {\n' +
    head +
    '    const enabled = chunk.enabled;\n' +
    '    for (let j = 0; j < k; j++) flags[j] = enabled[ids[j]];\n' +
    '    outer: for (let row = n - 1; row >= 0; row--) {\n' +
    '      for (let j = 0; j < k; j++) {\n' +
    '        if (flags[j][row] === 0) continue outer;\n' +
    '      }\n' +
    `      fn(${args});\n` +
    '    }\n' +
    '  }\n' +
    '}\n'
  );
}

/*
 * Per-callback specialization. The shared generic loops have one megamorphic `fn(...)` call
 * site, so V8 cannot inline callbacks into them. When a callback is passed to forEach for the
 * second time, a private copy of the loop is compiled for it (new Function over `rowLoopSource`)
 * and cached in a WeakMap keyed by the callback, making that call site monomorphic (measured
 * ~2-3x faster forEach for small callbacks). Fresh closures created per call are never seen
 * twice, so they stay on the generic loop and cost no compilation. Codegen is probed lazily on
 * first need and disabled for good when unavailable (CSP without 'unsafe-eval'). At most
 * MAX_SPECIALIZED copies are ever compiled.
 *
 * Module state and the call-site warm-up are created lazily by `initQueryModule()` on the first
 * Query construction, so importing this module has no side effects (package.json declares
 * "sideEffects": false, so bundlers may drop import-time code).
 */
const MAX_SPECIALIZED = 1024;
let specializedCount = 0;
/** 0 = not probed yet, 1 = available, -1 = unavailable. */
let codegenState = 0;
/** Unique suffix per compiled copy (see `compileRowLoop`). */
let serial = 0;
let initialized = false;
/** A compiled loop and the column count it was compiled for (-1 = plain form). */
interface SpecializedLoop {
  readonly m: number;
  readonly loop: RowLoop;
}
let specialized: WeakMap<AnyForEachFn, SpecializedLoop>;
let seenOnce: WeakSet<AnyForEachFn>;

/**
 * Compiles a private copy of the row loop for column count `m` (-1 = plain form). Every copy
 * needs its own feedback vector: V8's eval cache shares compiled code (and feedback) for
 * identical source strings, so each copy gets a unique trailing comment.
 */
function compileRowLoop(m: number): RowLoop {
  return new Function('chunks', 'ids', 'comps', 'fn', rowLoopSource(m) + '// cozyecs forEach #' + serial++) as RowLoop;
}

/** Checks that codegen works and the compiled loops behave like the generic ones on a fake chunk. */
function probeCodegen(): void {
  codegenState = -1;
  try {
    const plain = compileRowLoop(-1);
    const withCols = compileRowLoop(1);
    const colA = { v: 3 };
    const fake = [
      {
        count: 2,
        entities: new Uint32Array([7, 9]),
        enabled: [new Uint8Array([1, 0]), new Uint8Array([1, 1])],
        col: (c: unknown) => (c === probeComps[0] ? colA : undefined),
      },
    ] as unknown as Archetype[];
    const probeComps = [{}] as unknown as ComponentType[];
    let seen = 0;
    const visit: AnyForEachFn = (e, chunk, row, col?: { v: number }) => {
      seen = seen * 10 + (chunk === fake[0] ? e + row : 0) + (col === undefined ? 0 : col.v);
    };
    plain(fake, new Int32Array(0), null, visit); // reverse: 9 + 1, 7 + 0
    plain(fake, new Int32Array(1), null, visit); // row 1 disabled: 7
    plain(fake, new Int32Array([0, 1]), null, visit); // row 1 disabled: 7
    withCols(fake, new Int32Array(1), probeComps, visit); // row 1 disabled: 7 + 3
    if (seen !== 107780) return;
    if (!probeTrampolines()) return;
    codegenState = 1;
  } catch {
    // codegen unavailable: keep the generic loop
  }
}

/**
 * Loop to use for `fn` with column count `m` (-1 = plain form): its specialized copy if one
 * exists for `m` or `fn` was seen before (compiling one), otherwise the generic `forEachRows`
 * (remembering `fn` as seen).
 */
function resolveLoop(fn: AnyForEachFn, m: number): RowLoop {
  if (codegenState < 0 || m > MAX_COLUMN_PARAMS) return forEachRows;
  const entry = specialized.get(fn);
  if (entry !== undefined && entry.m === m) return entry.loop;
  if (specializedCount >= MAX_SPECIALIZED) return forEachRows;
  if (entry === undefined && !seenOnce.has(fn)) {
    seenOnce.add(fn);
    return forEachRows;
  }
  if (codegenState === 0) probeCodegen();
  if (codegenState < 0) return forEachRows;
  let compiled: RowLoop;
  try {
    compiled = compileRowLoop(m);
  } catch {
    codegenState = -1;
    return forEachRows;
  }
  specializedCount++;
  specialized.set(fn, { m, loop: compiled });
  return compiled;
}

/**
 * Calls `loop`. Its call site is made megamorphic by `initQueryModule`, so TurboFan never inlines
 * a row loop into forEach: code inlined into a try/finally region is compiled much worse
 * (measured ~1.7x slower tight loops in V8), while a megamorphic call costs a few ns per forEach.
 */
function invokeLoop(
  loop: RowLoop,
  chunks: Archetype[],
  ids: Int32Array,
  comps: readonly ComponentType[] | null,
  fn: AnyForEachFn,
): void {
  loop(chunks, ids, comps, fn);
}

/*
 * Compiled chunk trampolines (forEachChunk and the columns form of forEach).
 *
 * A tight loop over typed arrays compiles much better in V8 when the arrays are compile-time
 * constants: TurboFan embeds their data pointer and length instead of reloading them per row
 * (measured 262k -> 382k ops/sec for a 4 x 1000 row Position += Velocity loop; see
 * docs/INTERNALS.md, "Why the plain chunk loop trails closure-constant loops"). Arrays loaded
 * from a chunk at run time can never be constants, so for a callback passed a second time (same
 * identity) the query's chunks get trampolines, each covering a run of up to CHUNK_BATCH
 * consecutive chunks:
 *
 *   const f = fn, c0 = chs[0], e0 = ents[0], k0_0 = { "x": arrs[0], "y": arrs[1] }, c1 = ...;
 *   return function () {
 *     if (c0.entities !== e0 || c1.entities !== e1 ...) return false;  // storage reallocated
 *     n = c0.count; if (n !== 0) f(n, k0_0, ..., c0);                    // forEachChunk
 *     n = c0.count; if (n !== 0) { ents = c0.entities; for (row = n - 1 ...) f(ents[row], c0, row, k0_0, ...); }  // forEach
 *     ...
 *     return true;
 *   };
 *
 * TWO TIERS, ONE COMPILATION PER SHAPE. The source above depends only on the run's SHAPE: kind
 * (chunk / rows), column count, enable-check count, chunks in the run and the field names of
 * every column object (the snapshot objects are built by an object literal in the factory, so
 * one shape = one hidden class). It is compiled ONCE per distinct shape into a factory
 * (`new Function(fn, chs, ents, arrs, ids)`), cached process-wide in `shapes`, and every
 * trampoline is an INSTANCE of it (a factory call: no parsing, no compilation). Archetype growth
 * (every reallocation replaces `entities`, a public, unmangled name), chunks appended to a run
 * and new queries or callbacks with the same shape all re-instantiate from the cached factory.
 * Distinct shapes are bounded by MAX_TRAMPOLINE_SHAPES (4096): the 4097th distinct shape runs on
 * the generic loops, shapes already compiled keep working forever. Real programs have a handful
 * of shapes (one per kernel arity x field list x run length 1..8), so the bound is never reached
 * in practice; it only stops a pathological program from compiling without limit.
 *
 * Instances of one factory are closures of one SharedFunctionInfo, which V8 does NOT specialize
 * to their context: their columns are ordinary context loads, measured ~40% slower than a
 * trampoline that is the only closure of its own function (262k vs 435k ops/sec, 4 x 1000 rows,
 * median of 5 isolated runs). So an instance that has run SPECIALIZE_AFTER times without a
 * rebuild is replaced by a private copy compiled from the SAME cached source plus a unique
 * comment (V8's eval cache would otherwise share the function), which is the only closure of
 * its SharedFunctionInfo: TurboFan then embeds `f`, `c0` and the column arrays as constants and
 * inlines the callback per chunk. This compile is deferred to a steady run, so growth itself
 * never compiles; it happens at most once per run between two reallocations of its chunks, and
 * reallocations are geometric (capacity x2, then x1.25), so a run is re-specialized
 * O(log(max rows)) times over its whole life. It is not counted against the shape bound, and a
 * failure just keeps the instance.
 *
 * Plans are per (callback, query, kind, component list), held in a WeakMap keyed by the callback
 * (then by the query), so plans and their trampolines are released with the callback or the
 * query. They are only created for callbacks seen at least twice (fresh closures stay on the
 * generic loops). Without codegen (CSP) the generic loops are used.
 */
/** Bound on distinct trampoline shapes (one compiled factory each) per process. */
export const MAX_TRAMPOLINE_SHAPES = 4096;
let maxShapes = MAX_TRAMPOLINE_SHAPES;
/** Successful calls of a factory instance, without rebuild, before it gets a private specialized copy. */
const SPECIALIZE_AFTER = 16;
/** Chunks per trampoline. */
const CHUNK_BATCH = 8;
/** Largest number of enableable `all` components inlined into a forEach trampoline. */
const MAX_ENABLE_PARAMS = 8;
const KIND_CHUNK = 0;
const KIND_ROWS = 1;

/** Runs its chunks and returns true, or returns false (running nothing) if any chunk's storage was reallocated. */
type Trampoline = () => boolean;
/** Instantiates a trampoline of one shape: (fn, chunks, entities arrays, flat column arrays, enable ids). */
type TrampolineFactory = (fn: unknown, chs: readonly unknown[], ents: readonly unknown[], arrs: readonly unknown[], ids: Int32Array) => Trampoline;

/** A compiled shape: its source (for specialized copies) and factory. */
interface Shape {
  readonly source: string;
  readonly factory: TrampolineFactory;
}

/** Compiled shapes by key (see `shapeKey`). Process-wide; bounded by `maxShapes`. */
let shapes: Map<string, Shape>;

/** Counters exposed to tests through `_trampolineStats`. */
const tstats = { shapes: 0, instances: 0, specializations: 0, shapeRejections: 0 };

/** Trampolines of one (callback, query, kind, component list). */
interface ChunkPlan {
  readonly kind: number;
  /** Component count (columns passed to the callback). */
  readonly m: number;
  /** Copy of the component list. */
  readonly comps: ComponentType[];
  readonly fn: AnyForEachFn | AnyChunkFn;
  /** `tramps[g]` covers chunks [g * CHUNK_BATCH, g * CHUNK_BATCH + sizes[g]) (null = none). Packed. */
  readonly tramps: (Trampoline | null)[];
  readonly sizes: number[];
  /** Successful calls of `tramps[g]` since it was built; -1 once specialized (or not specializable). */
  readonly hits: number[];
}

let plansByFn: WeakMap<AnyForEachFn | AnyChunkFn, WeakMap<Query, ChunkPlan[]>>;
let seenForPlan: WeakSet<AnyForEachFn | AnyChunkFn>;

/** True if `plan` was built for `comps` (length m). */
function sameComps(plan: ChunkPlan, comps: readonly ComponentType[], m: number): boolean {
  if (plan.m !== m) return false;
  const own = plan.comps;
  for (let j = 0; j < m; j++) if (own[j] !== comps[j]) return false;
  return true;
}

/**
 * Source of a trampoline factory (parameters: fn, chs, ents, arrs, ids) over `fields.length`
 * chunks passing `m` columns; `fields[i][j]` is the key list of chunk i's column object j, or
 * null when that column is undefined. KIND_ROWS visits rows in reverse and skips rows where any of
 * the `k` enabled-flag arrays `c.enabled[ids[j]]` is 0 (flag values read per row, arrays once per
 * chunk), like `forEachRows`. Deterministic: the same shape always gives the same text.
 */
function trampolineSource(kind: number, m: number, k: number, fields: readonly (readonly (readonly string[] | null)[])[]): string {
  const s = fields.length;
  let decl = 'const f = fn';
  for (let j = 0; j < k; j++) decl += `, i${j} = ids[${j}]`;
  let check = '';
  let body = '';
  let a = 0;
  for (let i = 0; i < s; i++) {
    const c = `c${i}`;
    decl += `, ${c} = chs[${i}], e${i} = ents[${i}]`;
    let cols = '';
    for (let j = 0; j < m; j++) {
      const keys = fields[i][j];
      let init = 'undefined';
      if (keys !== null) {
        init = '{ ';
        for (let q = 0; q < keys.length; q++) init += `${q === 0 ? '' : ', '}${JSON.stringify(keys[q])}: arrs[${a++}]`;
        init += ' }';
      }
      decl += `, k${i}_${j} = ${init}`;
      cols += `, k${i}_${j}`;
    }
    check += (i === 0 ? '' : ' || ') + `${c}.entities !== e${i}`;
    body += `  n = ${c}.count;\n`;
    if (kind === KIND_CHUNK) {
      body += `  if (n !== 0) f(n${cols}, ${c});\n`;
      continue;
    }
    let flags = '';
    let cond = '';
    for (let j = 0; j < k; j++) {
      flags += ` const g${j} = ${c}.enabled[i${j}];`;
      cond += (j === 0 ? '' : ' && ') + `g${j}[row] !== 0`;
    }
    const call = `f(ents[row], ${c}, row${cols});`;
    body +=
      `  if (n !== 0) {\n    const ents = ${c}.entities;${flags}\n` +
      `    for (let row = n - 1; row >= 0; row--) ${k === 0 ? call : `{ if (${cond}) ${call} }`}\n  }\n`;
  }
  return decl + ';\nreturn function () {\n' + `  if (${check}) return false;\n` + '  let n = 0;\n' + body + '  return true;\n};\n';
}

/** Compiles factory source; `tag` makes the text (and so V8's SharedFunctionInfo) unique. Throws without codegen. */
function compileFactory(source: string, tag: string): TrampolineFactory {
  return new Function('fn', 'chs', 'ents', 'arrs', 'ids', source + tag) as TrampolineFactory;
}

/** Checks that trampolines compile and behave as specified on fake chunks. */
function probeTrampolines(): boolean {
  const fake = {
    count: 2,
    entities: new Uint32Array([7, 9]),
    enabled: [new Uint8Array([1, 0]), new Uint8Array([1, 1])],
  };
  const empty = { count: 0, entities: new Uint32Array(0), enabled: [] as Uint8Array[] };
  let seen = 0;
  const rows = (e: number, chunk: unknown, row: number, col: { v: number }) => {
    seen = seen * 10 + (chunk === fake ? e + row : 0) + col.v;
  };
  const kernel = (n: number, col: { v: number }, chunk: unknown) => {
    seen = seen * 10 + n + col.v + (chunk === fake ? 1 : 0);
  };
  const two = [fake, empty];
  const fields = [[['v']], [['v']]];
  const arrs = [3, 3];
  const ents = [fake.entities, empty.entities];
  const tag = '// cozyecs probe';
  const rowsF = compileFactory(trampolineSource(KIND_ROWS, 1, 0, fields), tag);
  const rowsFlags = compileFactory(trampolineSource(KIND_ROWS, 1, 2, fields), tag);
  const chunkF = compileFactory(trampolineSource(KIND_CHUNK, 1, 0, fields), tag);
  // rows, no flags: 9 + 1 + 3, 7 + 0 + 3 (reverse); empty chunk skipped
  if (rowsF(rows, two, ents, arrs, new Int32Array(0))() !== true) return false;
  // rows, flags 0 and 1: row 1 disabled -> 7 + 3
  if (rowsFlags(rows, two, ents, arrs, new Int32Array([0, 1]))() !== true) return false;
  // chunks: 2 + 3 + 1, empty chunk skipped
  const chunks = chunkF(kernel, two, ents, arrs, new Int32Array(0));
  if (chunks() !== true) return false;
  empty.entities = new Uint32Array(0);
  if (chunks() !== false) return false; // stale: runs nothing
  return seen === 14106;
}

/**
 * Plan for (query, fn, kind, comps) or null (generic loops): an existing plan, or a new one when
 * `fn` was seen before and codegen works.
 */
function resolvePlan(
  query: Query,
  fn: AnyForEachFn | AnyChunkFn,
  kind: number,
  comps: readonly ComponentType[],
  k: number,
): ChunkPlan | null {
  const m = comps.length;
  if (codegenState < 0 || m > MAX_COLUMN_PARAMS || k > MAX_ENABLE_PARAMS) return null;
  const byQuery = plansByFn.get(fn);
  let list = byQuery !== undefined ? byQuery.get(query) : undefined;
  if (list !== undefined) {
    for (let i = 0; i < list.length; i++) {
      const p = list[i];
      if (p.kind === kind && sameComps(p, comps, m)) return p;
    }
  }
  if (byQuery === undefined && !seenForPlan.has(fn)) {
    seenForPlan.add(fn);
    return null;
  }
  if (codegenState === 0) probeCodegen();
  if (codegenState < 0) return null;
  const own: ComponentType[] = [];
  for (let j = 0; j < m; j++) own.push(comps[j]);
  const plan: ChunkPlan = { kind, m, comps: own, fn, tramps: [], sizes: [], hits: [] };
  let map = byQuery;
  if (map === undefined) plansByFn.set(fn, (map = new WeakMap()));
  if (list === undefined) map.set(query, (list = []));
  list.push(plan);
  return plan;
}

/** Calls a trampoline. Kept megamorphic (see initQueryModule) so trampolines are never inlined here. */
function callTrampoline(t: Trampoline): boolean {
  return t();
}

/** Reused one-element chunk list for per-chunk generic fallbacks (read once per loop). */
const ONE_CHUNK: Archetype[] = [];

/** Generic forEachChunk loop (no codegen, fresh callbacks, shape bound reached). */
function forEachChunkGeneric(chunks: Archetype[], comps: readonly ComponentType[], fn: AnyChunkFn): void {
  const m = comps.length;
  const nChunks = chunks.length;
  for (let c = 0; c < nChunks; c++) {
    const chunk = chunks[c];
    const n = chunk.count;
    if (n === 0) continue;
    if (m === 0) fn(n, chunk);
    else if (m === 1) fn(n, chunk.col(comps[0]), chunk);
    else if (m === 2) fn(n, chunk.col(comps[0]), chunk.col(comps[1]), chunk);
    else if (m === 3) fn(n, chunk.col(comps[0]), chunk.col(comps[1]), chunk.col(comps[2]), chunk);
    else {
      const args: unknown[] = [n];
      for (let j = 0; j < m; j++) args.push(chunk.col(comps[j]));
      args.push(chunk);
      (fn as (...a: unknown[]) => void).apply(undefined, args);
    }
  }
}

/**
 * Runs `plan` over `chunks` (`ids`: the query's enableable ids, KIND_ROWS): one trampoline call per
 * run of CHUNK_BATCH chunks, re-instantiating a run's trampoline when it reports reallocated storage
 * or chunks were appended to the run, and specializing it once it has been stable for
 * SPECIALIZE_AFTER calls. Runs that cannot get a trampoline use the generic loops.
 */
function runPlan(plan: ChunkPlan, chunks: Archetype[], ids: Int32Array): void {
  const tramps = plan.tramps;
  const sizes = plan.sizes;
  const hits = plan.hits;
  // Chunks appended during iteration are empty, so the length is read once.
  const nChunks = chunks.length;
  for (let g = 0, start = 0; start < nChunks; g++, start += CHUNK_BATCH) {
    const left = nChunks - start;
    const size = left < CHUNK_BATCH ? left : CHUNK_BATCH;
    const t = g < tramps.length ? tramps[g] : null;
    if (t !== null && sizes[g] === size && callTrampoline(t)) {
      if (hits[g] >= 0 && ++hits[g] >= SPECIALIZE_AFTER) buildTrampoline(plan, chunks, g, start, size, ids, true);
      continue;
    }
    const fresh = buildTrampoline(plan, chunks, g, start, size, ids, false);
    if (fresh !== null && callTrampoline(fresh)) continue;
    for (let c = start; c < start + size; c++) {
      ONE_CHUNK[0] = chunks[c];
      if (plan.kind === KIND_CHUNK) forEachChunkGeneric(ONE_CHUNK, plan.comps, plan.fn as AnyChunkFn);
      else forEachRows(ONE_CHUNK, ids, plan.comps, plan.fn as AnyForEachFn);
    }
  }
}

/**
 * Builds and stores the trampoline of run `g` (chunks [start, start + size)); null if not possible.
 * `specialize` = false: an instance of the shape's cached factory (compiling the factory only for a
 * shape never seen before, within the shape bound). `specialize` = true: a private copy compiled
 * from the cached source, replacing a stable instance (see the section comment).
 */
function buildTrampoline(
  plan: ChunkPlan,
  chunks: Archetype[],
  g: number,
  start: number,
  size: number,
  ids: Int32Array,
  specialize: boolean,
): Trampoline | null {
  if (codegenState < 0) return null;
  const m = plan.m;
  const k = plan.kind === KIND_CHUNK ? 0 : ids.length;
  const chs: Archetype[] = [];
  const ents: Uint32Array[] = [];
  const arrs: unknown[] = [];
  const fields: (string[] | null)[][] = [];
  for (let c = start; c < start + size; c++) {
    const chunk = chunks[c];
    const own: (string[] | null)[] = [];
    for (let j = 0; j < m; j++) {
      const col = chunk.col(plan.comps[j]) as Record<string, unknown> | undefined;
      if (col === undefined) {
        own.push(null);
        continue;
      }
      const keys = Object.keys(col);
      for (let q = 0; q < keys.length; q++) arrs.push(col[keys[q]]);
      own.push(keys);
    }
    chs.push(chunk);
    ents.push(chunk.entities);
    fields.push(own);
  }
  const key = JSON.stringify([plan.kind, m, k, fields]);
  let shape = shapes.get(key);
  const tramps = plan.tramps;
  const sizes = plan.sizes;
  const hits = plan.hits;
  let t: Trampoline;
  try {
    if (shape === undefined) {
      if (specialize) return null; // cannot happen: a specialized run was built from its shape
      if (tstats.shapes >= maxShapes) {
        tstats.shapeRejections++;
        return null;
      }
      const source = trampolineSource(plan.kind, m, k, fields);
      shape = { source, factory: compileFactory(source, '// cozyecs chunks shape #' + tstats.shapes) };
      shapes.set(key, shape);
      tstats.shapes++;
    }
    if (specialize) {
      t = compileFactory(shape.source, '// cozyecs chunks #' + serial++)(plan.fn, chs, ents, arrs, ids);
      tstats.specializations++;
    } else {
      t = shape.factory(plan.fn, chs, ents, arrs, ids);
      tstats.instances++;
    }
  } catch {
    if (specialize) {
      hits[g] = -1; // keep the working instance
      return null;
    }
    codegenState = -1;
    return null;
  }
  while (tramps.length <= g) {
    tramps.push(null);
    sizes.push(-1);
    hits.push(0);
  }
  tramps[g] = t;
  sizes[g] = size;
  hits[g] = specialize ? -1 : 0;
  return t;
}

/**
 * @internal Test hook: trampoline counters. `shapes` = factories compiled (one per distinct
 * shape, never more than the bound), `instances` = factory instantiations (builds and rebuilds),
 * `specializations` = private copies compiled for stable runs, `shapeRejections` = builds refused
 * because the shape bound was reached.
 */
export function _trampolineStats(): { shapes: number; instances: number; specializations: number; shapeRejections: number } {
  return { shapes: tstats.shapes, instances: tstats.instances, specializations: tstats.specializations, shapeRejections: tstats.shapeRejections };
}

/** @internal Test hook: sets the distinct-shape bound (default MAX_TRAMPOLINE_SHAPES); returns the previous one. */
export function _setTrampolineShapeLimit(limit: number): number {
  const prev = maxShapes;
  maxShapes = limit;
  return prev;
}

/** Sink for the warm-up loops below (a side effect so minifiers keep the calls). */
let warmupSink = 0;

/** One-time lazy module init (first Query construction): specialization caches + call-site warm-up. */
function initQueryModule(): void {
  initialized = true;
  specialized = new WeakMap();
  seenOnce = new WeakSet();
  plansByFn = new WeakMap();
  seenForPlan = new WeakSet();
  shapes = new Map();
  const warm: RowLoop[] = [
    (c) => { warmupSink = c.length + 1; },
    (c) => { warmupSink = c.length + 2; },
    (c) => { warmupSink = c.length + 3; },
    (c) => { warmupSink = c.length + 4; },
    (c) => { warmupSink = c.length + 5; },
    (c) => { warmupSink = c.length + 6; },
    (c) => { warmupSink = c.length + 7; },
    (c) => { warmupSink = c.length + 8; },
  ];
  const chunks: Archetype[] = [];
  const ids = new Int32Array(0);
  const fn: AnyForEachFn = () => {};
  // Enough calls for V8 to allocate a feedback vector and record > 4 targets.
  for (let r = 0; r < 64; r++) for (let i = 0; i < warm.length; i++) invokeLoop(warm[i], chunks, ids, null, fn);
  const tramps: Trampoline[] = [
    () => (warmupSink = 1) > 0,
    () => (warmupSink = 2) > 0,
    () => (warmupSink = 3) > 0,
    () => (warmupSink = 4) > 0,
    () => (warmupSink = 5) > 0,
    () => (warmupSink = 6) > 0,
    () => (warmupSink = 7) > 0,
    () => (warmupSink = 8) > 0,
  ];
  for (let r = 0; r < 64; r++) for (let i = 0; i < tramps.length; i++) callTrampoline(tramps[i]);
}

/** Query description. All lists are optional; an empty description matches every archetype. */
export interface QueryDesc {
  /** Archetype must contain every one of these. */
  all?: readonly ComponentType[];
  /** Archetype must contain at least one of these (ignored when empty). */
  any?: readonly ComponentType[];
  /** Archetype must contain none of these. */
  none?: readonly ComponentType[];
}

/**
 * A cached set of archetypes matching a description. Create via `world.query(desc)`,
 * which dedupes by `Query.keyOf(desc)`.
 *
 * Fast path: iterate `chunks` directly:
 *   for (const c of q.chunks) { const p = c.col(Position); for (let i = 0; i < c.count; i++) p.x[i] += 1; }
 * Chunk loops do NOT skip disabled rows (use `chunk.enabledArray(C)`) and do NOT defer
 * structural changes; use `forEach` or run inside a system for that.
 * Fastest for small hot loops: `q.forEachChunk([Position, Velocity], kernel)` with a kernel defined
 * once (compiled per chunk run with constant columns; see forEachChunk).
 */
export class Query {
  readonly world: World;
  /** Dedupe key, see `keyOf`. */
  readonly key: string;
  /** Normalized (sorted, deduped) component lists. */
  readonly all: readonly ComponentType[];
  readonly any: readonly ComponentType[];
  readonly none: readonly ComponentType[];
  /**
   * Matching archetypes in creation order (including ones with count === 0; loops should
   * skip them cheaply with `if (chunk.count === 0) continue`).
   * Append-only and stable: the same array object for the query's lifetime, updated in place
   * (World appends via `_addArchetype` when it creates an archetype), so it can be cached.
   */
  readonly chunks: Archetype[];

  /** @internal mask of `all` (length may be 0). */
  readonly _allMask: Uint32Array;
  /** @internal mask of `any`, or null when `any` is empty. */
  readonly _anyMask: Uint32Array | null;
  /** @internal mask of `none`, or null when `none` is empty. */
  readonly _noneMask: Uint32Array | null;
  /** @internal enableable components of `all`; forEach skips rows where any is disabled. */
  readonly _enableable: readonly ComponentType[];
  /** @internal enter listeners. Copy-on-write: replaced (never mutated) on unsubscribe. */
  _enter: EntityCallback[];
  /** @internal exit listeners. Copy-on-write. */
  _exit: EntityCallback[];
  /** @internal ids of `_enableable`, for the forEach hot loop. */
  private readonly _enableIds: Int32Array;
  /** @internal total enter + exit subscriptions. */
  private _listenerCount: number;
  /** @internal callback whose specialized loop is cached in `_loop`. */
  private _loopFn: AnyForEachFn | null;
  /** @internal column count `_loop` was compiled for (-1 = plain form). */
  private _loopM: number;
  /** @internal specialized row loop for (`_loopFn`, `_loopM`). */
  private _loop: RowLoop;
  /** @internal last per-chunk trampoline plan used by forEach / forEachChunk (memo). */
  private _plan: ChunkPlan | null;

  /**
   * Normalizes the lists, builds masks and scans `world._archetypes` once, pushing matches
   * into `chunks`. Does NOT register itself with the world (World.query does).
   */
  constructor(world: World, desc: QueryDesc) {
    if (!initialized) initQueryModule();
    const all = normalizeComponents(desc.all || EMPTY_LIST);
    const any = normalizeComponents(desc.any || EMPTY_LIST);
    const none = normalizeComponents(desc.none || EMPTY_LIST);
    this.world = world;
    this.all = all;
    this.any = any;
    this.none = none;
    this.key = `a:${componentsKey(all)}|y:${componentsKey(any)}|n:${componentsKey(none)}`;
    this.chunks = [];
    this._allMask = createMask(all);
    this._anyMask = any.length > 0 ? createMask(any) : null;
    this._noneMask = none.length > 0 ? createMask(none) : null;
    const enableable: ComponentType[] = [];
    for (let i = 0; i < all.length; i++) if (all[i].enableable) enableable.push(all[i]);
    this._enableable = enableable;
    const ids = new Int32Array(enableable.length);
    for (let i = 0; i < enableable.length; i++) ids[i] = enableable[i].id;
    this._enableIds = ids;
    this._enter = EMPTY_LISTENERS;
    this._exit = EMPTY_LISTENERS;
    this._listenerCount = 0;
    this._loopFn = null;
    this._loopM = -1;
    this._loop = forEachRows;
    this._plan = null;
    const archetypes = world._archetypes;
    for (let i = 0; i < archetypes.length; i++) {
      const a = archetypes[i];
      if (a !== undefined && this.matches(a)) this.chunks.push(a);
    }
  }

  /**
   * Canonical key: `a:<ids>|y:<ids>|n:<ids>` where ids are the sorted deduped
   * component ids joined by ',' (e.g. `a:0,3|y:|n:5`).
   */
  static keyOf(desc: QueryDesc): string {
    return `a:${componentsKey(normalizeComponents(desc.all || EMPTY_LIST))}|y:${componentsKey(
      normalizeComponents(desc.any || EMPTY_LIST),
    )}|n:${componentsKey(normalizeComponents(desc.none || EMPTY_LIST))}`;
  }

  /** all ⊆ mask, (any empty or any ∩ mask ≠ ∅), none ∩ mask = ∅. */
  matches(archetype: Archetype): boolean {
    const mask = archetype.mask;
    const anyMask = this._anyMask;
    const noneMask = this._noneMask;
    return (
      maskContains(mask, this._allMask) &&
      (anyMask === null || maskIntersects(mask, anyMask)) &&
      (noneMask === null || !maskIntersects(mask, noneMask))
    );
  }

  /** Sum of `count` over chunks (disabled rows included). */
  count(): number {
    const chunks = this.chunks;
    let total = 0;
    for (let i = 0; i < chunks.length; i++) total += chunks[i].count;
    return total;
  }

  /**
   * Calls `fn(entity, chunk, row)` for every row, chunk by chunk, rows in REVERSE order,
   * skipping rows where any enableable component of `all` is disabled.
   * Wraps the loop in world._beginIteration() / world._endIteration() (try/finally), so
   * structural changes made inside are deferred and flushed when the outermost
   * iteration ends.
   */
  forEach(fn: QueryForEachFn): void;
  /**
   * Like `forEach(fn)`, but also passes the column objects of `components` (what
   * `chunk.col(C)` returns), resolved once per chunk instead of once per row:
   *   q.forEach([Position, Velocity], (e, chunk, row, pos, vel) => { pos.x[row] += vel.dx[row]; });
   * Measured 1.7-3.9x faster than calling `chunk.col(C)` inside the callback. When the same `fn`
   * (same identity: define it once) is passed again, its row loops are compiled with the columns
   * as constants (see `forEachChunk`), e.g. 1.4x over the generic loop for 4 chunks of
   * Position += Velocity. A column is undefined for chunks without that component (or for tags),
   * so list components of `all`. The column objects are snapshots valid during the call only: do
   * not retain or modify them. The array is read on every call and not retained; an inline
   * literal is fine.
   */
  forEach<const T extends readonly ComponentType<any>[]>(components: T, fn: QueryForEachColumnsFn<T>): void;
  forEach(a: QueryForEachFn | readonly ComponentType[], b?: AnyForEachFn): void {
    let fn: AnyForEachFn;
    let comps: readonly ComponentType[] | null;
    let m: number;
    if (typeof a === 'function') {
      fn = a;
      comps = null;
      m = -1;
    } else {
      fn = b as AnyForEachFn;
      comps = a;
      m = a.length;
    }
    const world = this.world;
    // Columns form: compiled chunk trampolines (constant columns). The plain form stays on the
    // per-callback loops: its callbacks call chunk.col() per row, which trampolines cannot fold
    // (measured no gain, up to ~10% slower).
    let plan = comps === null ? null : this._plan;
    if (comps !== null && (plan === null || plan.fn !== fn || plan.kind !== KIND_ROWS || !sameComps(plan, comps, m))) {
      plan = resolvePlan(this, fn, KIND_ROWS, comps, this._enableIds.length);
      if (plan !== null) this._plan = plan;
    }
    if (plan !== null) {
      if (world._iterDepth > 0) {
        runPlan(plan, this.chunks, this._enableIds);
        return;
      }
      world._beginIteration();
      try {
        runPlan(plan, this.chunks, this._enableIds);
      } finally {
        world._endIteration();
      }
      return;
    }
    let loop = this._loop;
    if (this._loopFn !== fn || this._loopM !== m) {
      loop = resolveLoop(fn, m);
      // Cache only specialized loops, so a callback first seen here can still be upgraded.
      if (loop !== forEachRows) {
        this._loopFn = fn;
        this._loopM = m;
        this._loop = loop;
      }
    }
    world._beginIteration();
    try {
      // Chunks appended during iteration are empty, so the length is read once (in the loop).
      // Called through `invokeLoop` so the loop is never inlined into this try block.
      invokeLoop(loop, this.chunks, this._enableIds, comps, fn);
    } finally {
      world._endIteration();
    }
  }

  /**
   * Calls `fn(count, ...columns, chunk)` once per non-empty chunk, where `columns` are the column
   * objects of `components` (same keys and TypedArrays as `chunk.col(C)`), in order:
   *   q.forEachChunk([Position, Velocity], (n, pos, vel) => {
   *     const x = pos.x, dx = vel.dx;
   *     for (let i = 0; i < n; i++) x[i] += dx[i];
   *   });
   * This is a chunk loop: rows are NOT filtered by enabled flags (use `chunk.enabledArray(C)`).
   * Structural changes made inside are deferred like in `forEach`.
   *
   * Performance: when the same `fn` (same function identity: define it once, not inline per call)
   * is passed again, the query's chunks get compiled trampolines in which the columns are
   * constants for V8, so small kernels run faster than the equivalent `for (const chunk of
   * q.chunks)` loop (measured 1.5x for 4 chunks x 1000 rows of Position += Velocity, 1.4x for 200
   * chunks x 10 rows, 1.03-1.05x for single-array loops). Keep kernels small (V8 must inline them).
   * The column objects are snapshots valid during the call only: do not retain or modify them
   * (their TypedArrays go stale when the chunk grows). A column is undefined for chunks without
   * that component (or for tags), so list components of `all`. `components` is read on every call
   * and not retained; an inline literal is fine. Trampolines are compiled with `new Function`;
   * without runtime codegen (CSP) the same semantics run on a generic loop.
   */
  forEachChunk<const T extends readonly ComponentType<any>[]>(components: T, fn: ChunkKernelFn<T>): void {
    const kernel = fn as unknown as AnyChunkFn;
    const m = components.length;
    let plan = this._plan;
    if (plan === null || plan.fn !== kernel || plan.kind !== KIND_CHUNK || !sameComps(plan, components, m)) {
      plan = resolvePlan(this, kernel, KIND_CHUNK, components, 0);
      if (plan !== null) this._plan = plan;
    }
    const world = this.world;
    if (world._iterDepth > 0) {
      // Already iterating (e.g. inside a system): changes are deferred and the outer iteration
      // flushes, so no begin/end (and no try/finally) is needed.
      if (plan !== null) runPlan(plan, this.chunks, this._enableIds);
      else forEachChunkGeneric(this.chunks, components, kernel);
      return;
    }
    world._beginIteration();
    try {
      if (plan !== null) runPlan(plan, this.chunks, this._enableIds);
      else forEachChunkGeneric(this.chunks, components, kernel);
    } finally {
      world._endIteration();
    }
  }

  /**
   * Fires `cb(entity)` after an entity starts matching (spawn into / move into a matching
   * archetype). Registers with world._setQueryListening(this, true) on the first listener.
   * @returns unsubscribe (idempotent); calls world._setQueryListening(this, false) when the
   * last enter/exit listener is removed.
   */
  onEnter(cb: EntityCallback): Unsubscribe {
    this._enter = appendListener(this._enter, cb);
    return this._subscribed(cb, true);
  }

  /** Like onEnter, fired after an entity stops matching (destroy / move out). */
  onExit(cb: EntityCallback): Unsubscribe {
    this._exit = appendListener(this._exit, cb);
    return this._subscribed(cb, false);
  }

  /** @internal Called by World for each newly created archetype; pushes it to `chunks` if it matches. */
  _addArchetype(archetype: Archetype): void {
    if (this.matches(archetype)) this.chunks.push(archetype);
  }

  /**
   * @internal Called by World after a structural change is applied, only while this query
   * has listeners. `from`/`to` are the old/new archetypes (null for spawn/destroy).
   * Fires exit listeners if matches(from) && !matches(to), enter listeners if the reverse.
   */
  _transition(entity: number, from: Archetype | null, to: Archetype | null): void {
    const wasIn = from !== null && this.matches(from);
    const isIn = to !== null && this.matches(to);
    if (wasIn === isIn) return;
    // Read the listener array once; unsubscribing during dispatch replaces it (copy-on-write).
    const list = wasIn ? this._exit : this._enter;
    for (let i = 0; i < list.length; i++) list[i](entity);
  }

  /** @internal Bumps the listener count, notifies the world on 0 -> 1, returns unsubscribe. */
  private _subscribed(cb: EntityCallback, enter: boolean): Unsubscribe {
    if (this._listenerCount++ === 0) this.world._setQueryListening(this, true);
    let active = true;
    return () => {
      if (!active) return;
      active = false;
      if (enter) this._enter = removeListener(this._enter, cb);
      else this._exit = removeListener(this._exit, cb);
      if (--this._listenerCount === 0) this.world._setQueryListening(this, false);
    };
  }
}
