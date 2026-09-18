# CozyECS internals

Contract between modules. Stubs in `src/` hold the exact signatures; this file
holds the rules. If code and this file disagree, raise it; do not guess.

## Module dependency graph (runtime imports)

```
types.ts      <- component.ts
strings.ts
entities.ts
archetype.ts  -> (type) component, types
query.ts      -> component (mask helpers); (type) archetype, world
system.ts     -> (type) query, world
world.ts      -> archetype, component (helpers), entities, query, system, strings
index.ts      -> everything public
```

`query.ts` and `system.ts` must only `import type` from `world.ts` (no runtime cycle).
Private helper fields may be added freely inside a file; everything listed in
the stubs (including `_`-prefixed members) is a cross-module contract.

## Hot-path rules

- No allocation, closures, `Map` lookups or string-keyed dynamic lookups per entity in
  `Query.forEach`, `Archetype.pushRow/swapRemove/copyRowTo`, `World._move`.
  (`Map` lookups per structural op on `edgesAdd/edgesRemove` are allowed.)
- Arrays indexed by component id (`columns`, `enabled`, `_onAdd`, `_onRemove`, `_views`)
  must stay packed: when extending, fill intermediate slots with `undefined` via a loop
  (never write past `length`).
- Listener arrays are copy-on-write: subscribe/unsubscribe build a NEW array and assign it;
  dispatch loops read the array reference once and iterate it.

## Component ids and masks (component.ts — implemented)

- `component()` assigns `id` from a module-global counter (dense, from 0). Components are
  shared by all worlds.
- Mask: `Uint32Array`, word `id >>> 5`, bit `1 << (id & 31)`. Masks have variable length;
  missing words are zero. Use only `createMask`, `maskHas`, `maskContains(sup, sub)`,
  `maskIntersects(a, b)`.
- `normalizeComponents(list)` dedupes and sorts by id (new array). `componentsKey(sorted)`
  = ids joined by `,` (`''` for none). Archetype key == `componentsKey(normalized)`.

## Entities (entities.ts)

- Handle = `((gen << 20) | index) >>> 0`. Generation wraps at 12 bits. Handle `0` is valid;
  `-1` means "no entity".
- **Per-index state is one `Int32Array`, 4 B per index** (was 8 B: Int16 archetype +
  Uint32 row + Uint16 generation). `slot[index] = (tag << 20) | low`:

  | tag (`slot >>> 20`) | meaning | low 20 bits |
  |---|---|---|
  | `0..4092` | placed in archetype `tag` | row |
  | `4093` (`TAG_BIG`) | placed in archetype `bigAid[index]` (ids 4093..32767) | row |
  | `4094` (`TAG_FREE`) | free (destroyed, or never handed out) | generation the next handle gets |
  | `4095` (`TAG_PENDING`) | alive, spawned while deferring, no row yet | current generation |

  Decoded archetype id (`archetypeOf(index)`): `tag < 4093 ? tag : tag === 4093 ?
  bigAid[index] : tag - 4096` (pending -> `-1`, free -> `-2`).
- **Generation of a placed entity is not stored in the allocator.** The table row holds the full
  handle, so liveness is `archetypes[tag].entities[row] === e` (a stale handle has another
  generation and fails). `locate(e)` returns the archetype id, `-1` (pending) or `-2` (not
  alive); `isAlive(e)` is `locate(e) !== -2`. The allocator holds a reference to
  `world._archetypes` for this.
- `bigAid: Int16Array` is empty until the World creates archetype id 4093 (`enableBig()`), then
  kept as long as `slot`. The 32768-archetype limit is unchanged.
- `create()`: pop `freeStack` (LIFO) else `next++`; generation = low bits of the free slot (already
  bumped on release) or 0; `slot = TAG_PENDING << 20 | gen`; `aliveCount++`; throw `Error` at
  `MAX_ENTITIES`.
- `release(e)`: `slot = TAG_FREE << 20 | ((gen + 1) & 0xfff)`, push index on `freeStack`,
  `aliveCount--`. `freeStack` (Int32Array) grows on demand: it costs 4 B per *simultaneously
  free* index, nothing for a world that never destroys.
- `setLocation(e, aid, row)` takes the HANDLE. Hot paths write the slot directly:
  `slot[idx] = (aid << 20) | row` when `aid < TAG_BIG`, else `setLocation`. The row of an entity
  moved by `swapRemove` keeps its tag: `slot[m] = (s & ~INDEX_MASK) | r`, where `s` is the slot of the
  entity leaving row `r` (same archetype, so same tag bits; no extra read).
- Growth: x2 below 65536 indices, then x1.25 rounded up to a multiple of 4096, same as archetype
  tables, so at 100k entities both are 102,400. Arrays are replaced on growth: always read
  `alloc.slot` fresh, never cache across anything that may allocate ids.
- `World.add` / `World.remove` inline the `locate` fast path (tag `< TAG_BIG`, then
  `from.entities[row] === e`, reusing `from` for the move) and queue the command inline when
  deferring; only dead, pending and big-id cases go through `_addSlow` / `_removeSlow` (which
  call `locate`). Keep them small enough for V8 to inline into caller loops (bytecode budget):
  growing `add()` past it cost ~15% on add_remove in round 3, and routing the deferred push
  through `_addSlow` cost ~4% on the systems variant. `_flushMoveRun` caches `from.entities`
  while consecutive commands stay in the same source archetype.
- Measured (round 3, paired isolated runs, 9 repeats, vs the 8 B/index allocator):
  entity_cycle systems 1.08x / direct 1.12x; add_remove systems 1.02x / direct 1.02x.
- Memory (benchmarks/memory.js, 100k Position+Velocity f32x4): table 20 B/row x 102,400 +
  allocator 4 B x 102,400 = 24.6 B/entity of ArrayBuffers; the probe reports 26.2 B/entity
  because ~100 KB of its heap delta is Node's lazy `performance` initialization inside the
  probe itself (25.2 B/entity when `performance.now()` is called before the baseline sample).

## Archetypes (archetype.ts)

- `id` is its index in `world._archetypes`; id 0 is the empty archetype, created by the
  World constructor.
- Constructor receives an already-normalized component list and its key; builds one column
  object per data component (`columns[c.id]`, keys in schema order, one TypedArray per field),
  `enabled[c.id]: Uint8Array` for every enableable component (tags included). Tags without
  enableable get nothing.
- **Single table buffer.** All per-row data of an archetype (`entities`, every field column,
  every enabled column) are views over ONE `ArrayBuffer` (a `SharedArrayBuffer` when
  `WorldOptions.shared` is true and SAB exists; `archetype.buffer`, `archetype.shared`).
  Layout groups columns by element width, descending: `[f64][entities, f32/i32/u32/str]
  [i16/u16][i8/u8/bool][enabled flags]`, so every column's byte offset is a multiple of its
  width and no padding is needed. `rowBytes` = sum of widths (Position+Velocity f32x4: 20 B).
- Row ops (`pushRow`, `pushRowFrom`, `swapRemove`, `copyRowTo`) go through four whole-buffer
  views (`_v64/_v32/_v16/_v8`) plus per-slot base offsets, one loop per width (monomorphic,
  bit-exact: f32 copied as Int32). Transfers use a precomputed per-(source, target) plan
  (`Int32Array`, cached on the target, one-entry memo).
- **Row packing invariant:** rows `[0, count)` are live and contiguous; `entities[row]` is
  the handle stored at `row`. Content beyond `count` is garbage.
- `pushRow(e)`: grow if `count === capacity`; zero every field at the row; set every enabled
  flag to 1; `entities[row] = e`; `count++`; return row.
- `swapRemove(row)`: `last = count - 1`; if `row !== last` copy every field, enabled flag and
  `entities[last]` into `row`; `count--`; return the moved handle or `-1` if `row === last`.
- `copyRowTo(row, target, targetRow)`: for each component in `this` that `target.has`, copy
  every field and enabled flag. Does not change counts or `entities`.
- `pushRowFrom(source, sourceRow, e)`: append a row initialized from `source[sourceRow]`
  (shared components copied, others zeroed / enabled) without double writes.
- `grow()`: capacity *= 2: allocates ONE new buffer and copies `[0, count)` of each column once.
  Column OBJECT identity stays the same (new views assigned to its properties).
  `ensureCapacity(n)` computes the final doubled size and reallocates once.
  **Every TypedArray obtained from `col()`, `enabledArray()` or `entities` is invalid after
  growth** (it still points at the old buffer): re-fetch after structural changes; systems
  re-fetch per tick.
- `col(c)` = `columns[c.id]` (undefined for absent or tags). `has(c)` = mask test.
  `isEnabled(c, row)`: absent -> false; not enableable -> true; else `enabled[c.id][row] === 1`.
- Archetypes never touch entity locations. `edgesAdd/edgesRemove` are written only by World.

## Entity location update rules (world.ts)

Every structural op leaves `alloc.slot` (archetype tag + row) consistent before any callback fires.

- **place (spawn)** `_place(e, A)`: `row = A.pushRow(e)`; `setLocation(e, A.id, row)`; events(null -> A).
- **move (add/remove)** `_move(e, B)` from A at `r`:
  1. `r2 = B.pushRowFrom(A, r, e)`
  2. (values of `add` are written after the move)
  3. `m = A.swapRemove(r)`; if `m !== -1`: row bits of `slot[m & INDEX_MASK]` = `r` (tag unchanged)
  4. `setLocation(e, B.id, r2)`
  5. events(A -> B)
- **destroy** `_destroyNow(e)`: if placed at A/r: `m = A.swapRemove(r)`, fix `m` as above;
  `alloc.release(e)`; events(A -> null). A pending entity just gets released, events(null -> null)
  = nothing.
- **spawnMany(A, n, init)**: `A.ensureCapacity(A.count + n)`, `alloc.reserve(n)`, then per
  entity `create` + `pushRow` + `setLocation` + `init?.(A, row, i)`; events per entity only when
  listeners exist (check once before the loop).
- `add(e, C, v)` when entity already has C: just `set(e, C, v)` (no events). `remove` of an absent
  component: no-op.
- Transition archetypes are looked up in `world._edgeAdd[A.id][C.id]` /
  `_edgeRemove[A.id][C.id]` (packed arrays mirroring the Maps, including self-edges), falling
  back to `_archetypeWith(A, C)` = `A.edgesAdd.get(C.id)` or
  `_getOrCreateArchetype([...A.components, C])`; on creation/lookup miss set
  `A.edgesAdd(C.id) = B` and `B.edgesRemove(C.id) = A`. Symmetric for `_archetypeWithout`.

## Archetype creation and query notification

`_getOrCreateArchetype(list)`: normalize, key, map lookup. On miss:
`new Archetype(_archetypes.length, sorted, key, _initialCapacity)`, push to `_archetypes`,
set in `_archetypeByKey`, then **synchronously** call `q._addArchetype(a)` for every query in
`_queries` (in creation order). This is the only way `Query.chunks` grows. It may happen at any
time, including during iteration (new chunks have `count === 0` then, so this is harmless).

`world.query(desc)`: `key = Query.keyOf(desc)`; return cached, or `new Query(world, desc)`
(which scans `world._archetypes` once), push to `_queries`, set in `_queryByKey`.

## Query (query.ts)

- `matches(a)`: `maskContains(a.mask, _allMask) && (!_anyMask || maskIntersects(a.mask, _anyMask))
  && (!_noneMask || !maskIntersects(a.mask, _noneMask))`.
- `forEach(fn)`: `world._beginIteration()`; `try` { for each chunk: `n = chunk.count`, rows
  `n-1 .. 0`; skip row if any `_enableable[k]` has `chunk.enabled[id][row] === 0`; else
  `fn(chunk.entities[row], chunk, row)` } `finally` { `world._endIteration()` }.
  Specialize the no-enableable case (separate loop) to keep it branch-free.
  The row loop is called through a megamorphic `invokeLoop` (never inlined into the try). A
  callback passed to forEach a second time gets its own copy of the loop compiled with
  `new Function` (WeakMap cache, max 1024 copies, probed once; falls back to the shared loop
  when codegen is unavailable, e.g. CSP), making the `fn` call site monomorphic.
- `forEachChunk(comps, fn)` and `forEach(comps, fn)`: for a callback seen a second time, runs of up
  to 8 chunks go through compiled TRAMPOLINES that pass each column object as a closure constant
  (see the next section for why that matters). Trampolines are cached by SHAPE, not by chunk:
  the shape key is (kind, column count, enable-check count, chunks in the run, field names of
  every column object). Each distinct shape is compiled ONCE into a factory
  (`new Function(fn, chs, ents, arrs, ids)`, process-wide `Map`), and every trampoline is an
  instance of it. Archetype growth (reallocation replaces `chunk.entities`, which the trampoline
  checks), chunks appended to a run, new queries and new callbacks of the same shape all
  re-instantiate from the cached factory: no parsing, no compilation. The only bound is
  `MAX_TRAMPOLINE_SHAPES = 4096` DISTINCT shapes per process; a shape beyond it runs on the
  generic loop, while shapes compiled earlier keep working. There is no lifetime budget any more
  (the old `MAX_TRAMPOLINES = 4096` counted every rebuild, so a long-running app silently fell
  back to generic loops: 5000 reallocations took `forEachChunk` from 355k to 246k ops/sec; with
  the shape cache it stays at 348k, median of 5 isolated runs, 4 x 1000 rows).
  Factory instances share one SharedFunctionInfo, which V8 does not context-specialize (their
  columns are context loads, measured 251k vs 435k ops/sec for a private copy), so an instance that
  runs 16 times without a rebuild is replaced by a private copy compiled from the cached source
  with a unique comment (V8's eval cache would otherwise share it). That compile happens on a
  steady frame, never on the growth frame, at most once per run between reallocations
  (geometric, so O(log rows) per run over its life), and is not counted against the shape bound.
  Plans live in `WeakMap<callback, WeakMap<Query, plan[]>>`, so plans and trampolines are released
  with the callback or the query. Test hooks: `_trampolineStats()` (`shapes`, `instances`,
  `specializations`, `shapeRejections`) and `_setTrampolineShapeLimit(n)`.
- `onEnter/onExit`: copy-on-write append; when total listeners goes 0 -> 1 call
  `world._setQueryListening(this, true)`; when 1 -> 0, `false`. Unsubscribe is idempotent.
- `_transition(e, from, to)`: `wasIn = from !== null && matches(from)`, `isIn = to !== null && matches(to)`;
  `wasIn && !isIn` -> exit listeners; `!wasIn && isIn` -> enter listeners.

## Why the plain chunk loop trails closure-constant loops

The loop users write by hand,

```js
for (const c of q.chunks) {
  const p = c.col(Position), v = c.col(Velocity);
  const x = p.x, dx = v.dx;
  for (let i = 0; i < c.count; i++) x[i] += dx[i];
}
```

is ~17% behind bitECS in `simple_iter`, while the same work through `forEachChunk` is ~20% ahead.
The difference is not CozyECS's storage: it is whether V8 knows the typed arrays at compile time.
bitECS's arrays (`Position.x`) are module constants and `forEachChunk` trampolines bind columns as
closure constants of a function that is the only closure of its SharedFunctionInfo, so TurboFan
embeds each array's data pointer and length. Arrays read from a chunk at run time are ordinary
values whose backing-store pointer and length must be loaded and checked at run time; the table
below shows that this costs 17-65% on a two-add loop and that nothing about the holder objects
changes it.

Measured (Apple M4, Node 22.14, median of 5 isolated processes, load average < 20; 4 chunks x
1000 rows, `x += dx; y += dy`, ops/sec):

| variant | ops/sec | vs plain |
| --- | ---: | ---: |
| plain chunk loop (`c.col(C)` per chunk, column object literal) | 263.7k | 1.00 |
| per-chunk frozen column objects (`Object.freeze`) | 249.6k | 0.95 |
| stable hidden class for column holders (`class PosCols { x; y }`) | 260.4k | 0.99 |
| trampoline instance of a SHARED factory (many closures, one SFI) | 250.6k | 0.95 |
| same, after 50 growth rebuilds | 251.2k | 0.95 |
| bitECS-shaped loop: module-constant arrays through an entity list | 307.6k | 1.17 |
| real bitECS 0.3 (`defineQuery`, `Pos.x[e] += Vel.dx[e]`) | 221.0k | 0.84 |
| trampoline, private compiled copy (columns are closure constants) | 435.4k | 1.65 |

2026-09-18 (`benchmarks/gpu.js`, `simple` kernel, median of 5 isolated processes): the plain loop is 0.77-0.88x bitECS at 1k-100k entities and 1.24x at 1M; `kernelSystem({ target: 'cpu' })` is 1.14-1.30x. A second run later that day (round 2, sizes 1k-1M, `benchmarks/RESULTS.md`) agrees: the plain loop is 0.78-0.87x at 1k-100k and 1.23x at 1M, and `kernelSystem` is 1.00-1.30x at 1k-100k and 1.17x at 1M.

Chunks are already monomorphic (every chunk is an `Archetype`; every column object of a component
has the same keys in the same order), so neither freezing nor a class for the column holders
changes anything: the per-row cost is the reload of the arrays' data pointers, which only a
compile-time constant removes. A library cannot make a user's own loop variable a constant
without compiling that loop, i.e. without taking the loop out of the user's hands. The full
benchmark agrees (`simple_iter`, median of 5 isolated runs): `cozyecs (direct)` 240k, bitECS
283-290k, `cozyecs` (systems + `forEachChunk`) 337-383k ops/sec.

Recommended patterns for hot loops, fastest first:

1. `kernelSystem(world, name, { target: 'cpu', ... })` from `cozyecs/gpu`: the kernel is compiled
   into a flat chunk loop with every column hoisted, and runs through the same trampolines
   (use `target: 'auto'` to let large queries move to the GPU).
2. `q.forEachChunk([A, B], kernel)` with `kernel` defined ONCE (stable identity): from the second
   call the chunks run through trampolines, specialized after 16 steady frames.
3. `q.forEach([A, B], (e, chunk, row, a, b) => ...)`, same identity rule, for per-row callbacks.
4. The plain chunk loop: still the right tool for one-off or cold code; hoist `c.col(C)` and the
   arrays out of the row loop as above.

## Enter/exit and add/remove hook rules

- Fire only AFTER the structural change is fully applied (locations updated). On destroy
  the entity is already dead: callbacks receive the handle but cannot read its data.
- Enable/disable never fires anything. `set` never fires anything.
- Zero cost when unused: `_fireEvents` returns immediately if `_hookCount === 0 &&
  _eventQueries.length === 0`.
- Order within `_fireEvents(e, from, to)`:
  1. `onRemove` for each component in `from` not in `to` (component id order)
  2. `onAdd` for each component in `to` not in `from`
  3. `q._transition(e, from, to)` for each query in `_eventQueries` (exits and enters, per query,
     in `_eventQueries` order)
  For single-component `add/remove`, hooks may be dispatched directly for `C` (same result).
- Callbacks may perform structural changes. If the world is deferring (inside iteration or
  flush) they are queued; otherwise they apply immediately (re-entrant `_fireEvents` is fine
  because locations are consistent).

## Iteration and command buffer semantics

- `_iterDepth` counts nested `forEach` calls and system runs. `_isIterating = _iterDepth > 0`.
- **Deferral predicate:** structural ops (`spawn`, `spawnMany`, `destroy`, `add`, `remove`)
  are queued iff `_deferring` (`_iterDepth > 0 || _flushing`). Queuing during flush keeps
  command order and avoids touching entities whose spawn is still queued.
- `_endIteration()`: `--_iterDepth`; if it is now 0, `flush()`.
- `world.update(dt = 0, group = 'update')`: `list = _scheduler.group(group)`; ONE try/finally
  around the group; for each enabled `s`: `_iterDepth = base + 1; s._run(dt); _iterDepth = base`,
  then flush if `base === 0` and commands are pending. On throw the finally restores the depth
  and flushes. `_run` call sites are warmed to megamorphic at module load so system bodies are
  not inlined into the try region.
- Command buffer: parallel typed arrays `codes` (`ref << 3 | op`), `ents`, `slots` (index into
  the `vals` payload array), grown by doubling, never shrunk.
- `flush()`: if `_flushing` or iterating return. Set `_flushing = true`; apply commands with the
  length re-read (commands queued by callbacks are applied in the same flush); without
  listeners a tight `_flushPlain` loop is used; then `clear()`; `finally` reset
  `_flushing = false` (on throw, drop applied commands and the failing one, keep the rest).
- Validation happens twice:
  - at call time: `add`/`set` on a dead entity throw; `destroy`/`remove` on dead are ignored
    (not queued).
  - at flush time: any command whose entity is no longer alive is silently skipped
    (e.g. `add` after a queued `destroy`).
- `spawn` while deferring: `e = alloc.create()` (pending, alive), queue `CMD_SPAWN(e, A)`,
  return `e`. Until flushed: `isAlive(e)` true, `has` false, `get` undefined, `set`/`enable`
  throw (no component yet). A queued destroy of a pending entity is valid.
- `destroy` while deferring: queued; the entity stays alive (and readable) until flush.
- `add(e, C, values)` while deferring stores the `values` reference; it is read at flush.
- Immediate ops (`set`, `get`, `getField`, `enable`, `isEnabled`, `has`, `archetype`, `query`)
  are never deferred.
- `spawn(ComponentType[])` resolves the archetype immediately (allocates a key string; pass an
  `Archetype` on hot paths).

## Values conversion (world.set / world.get)

- `set`: for each key of `C.keys` whose value in `values` is not `undefined`: code
  `FIELD_STR` -> `strings.intern(v)`, `FIELD_BOOL` -> `v ? 1 : 0`, otherwise the number.
- `get`: `_views[C.id]` is created once per component with all keys; overwritten each call;
  `FIELD_STR` -> `strings.get(id)`, `FIELD_BOOL` -> `x !== 0`. Tags return the (empty) view.

## Systems (system.ts)

- `world.system(name, opts, fn)`: resolve `opts.query` (Query instance as-is, desc via
  `world.query`), create `FunctionSystem(world, name, group ?? 'update', order ?? 0, q, fn,
  _systemSeq++)`, `_scheduler.add`, return it.
- `world.addSystem(Ctor, opts)`: `s = new Ctor(world)`; `name = Ctor.name || 'System'`, set group,
  order, `_seq = _systemSeq++`; `_scheduler.add(s)`; `s.onCreate()`; return `s`.
- `world.removeSystem(s)`: if `_scheduler.remove(s)`: `s.enabled = false`; if `s instanceof System`
  call `s.onDestroy()`.
- `System` constructor sets `world`, `enabled = true`, `name = ''`, `group = 'update'`,
  `order = 0`, `_seq = 0` (overwritten by addSystem). Field initializers like
  `q = this.query(...)` run after `super(world)`, so `this.world` is available.
- `Scheduler` keeps `Map<group, Runnable[]>`; `add` builds a new array with the system inserted
  after all entries with `(order, _seq) <=` its own; `remove` builds a new array without it.
  `group(name)` returns the stored array or a shared frozen empty array.
