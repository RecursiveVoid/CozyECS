# CozyECS API: queries, iteration and systems

This page covers the iteration API — `Query`, chunks (`Archetype`), `forEach` and systems — and
the memory API: [measuring and reclaiming storage](#memory-measuring-and-reclaiming-it).
For how things work inside, see [INTERNALS.md](./INTERNALS.md).

## Queries

```ts
const q = world.query({ all: [Position, Velocity], any: [A, B], none: [Frozen] });
```

- `world.query(desc)` returns a cached `Query`. Descriptions that are equal after sorting and
  removing duplicates return the same instance.
- `q.chunks: Archetype[]` lists the matching archetypes in creation order. It is the **same array
  for the query's lifetime**. New matching archetypes are appended to it, so you can keep a
  reference to it.
- `chunks` can include empty archetypes (`count === 0`). The loop skips them cheaply.
- `q.count()` is the number of rows across all chunks, disabled rows included.
- `q.onEnter(cb)` / `q.onExit(cb)` subscribe to entities starting or stopping to match. Each
  returns an unsubscribe function.

## Chunks (`Chunk = Archetype`)

| member | meaning |
|---|---|
| `count` | live rows |
| `entities` | `Uint32Array` of entity handles per row (length is the capacity, read only `[0, count)`) |
| `col(C)` | column object of `C`: one TypedArray per field, keys in schema order (`undefined` for tags or absent components) |
| `enabledArray(C)` | `Uint8Array` of enabled flags (1 = enabled) for enableable components, else `undefined` |
| `has(C)` | whether the archetype contains `C` |

**Column arrays go stale when a table is reallocated.** All typed columns of an archetype are
views over one buffer, and that buffer is replaced when the table grows — or shrinks, under
[`shrinkToFit` / `world.compact()` / `world.clear()` / `world.dispose()`](#memory-measuring-and-reclaiming-it). After
a spawn, add, remove or destroy that runs immediately, and after a compaction, arrays you got
earlier from `col(C).x`, `enabledArray(C)` or `entities` may point to the old buffer. The column
*object* returned by `col(C)` keeps its identity; only its properties are replaced.

So fetch the arrays again in every tick or loop, as in the examples below. Never keep them
across structural changes. `world.dispose()` is the end state of the same rule: every table is
deflated to a zero-length buffer, so a column array taken before it is a view over a buffer the
world no longer uses, and there is no later call that will refresh it.

`new World({ shared: true })` allocates each archetype's buffer as a `SharedArrayBuffer` (when
the runtime provides one), so columns can be handed to workers; `chunk.buffer` is the current
buffer and `chunk.shared` tells which kind it is. The default is a plain `ArrayBuffer`.

## Iterating

There are three idioms, from fastest to most convenient.

### 1. Chunk loops (fastest)

```ts
world.system('move', { query: q }, (q) => {
  const chunks = q.chunks;
  for (let c = 0; c < chunks.length; c++) {
    const ch = chunks[c];
    const p = ch.col(Position), v = ch.col(Velocity);
    const x = p.x, y = p.y, dx = v.dx, dy = v.dy;
    for (let i = 0, n = ch.count; i < n; i++) {
      x[i] += dx[i];
      y[i] += dy[i];
    }
  }
});
```

- Chunk loops do **not** skip disabled rows. Check `ch.enabledArray(C)` yourself.
- Outside a system or `forEach`, a chunk loop does **not** defer structural changes. Inside a
  system they are deferred, and flushed after the system returns.

### 2. `forEach(components, fn)`: rows with columns resolved per chunk

```ts
q.forEach([Position, Velocity], (entity, chunk, row, pos, vel) => {
  pos.x[row] += vel.dx[row];
  pos.y[row] += vel.dy[row];
});
```

- The callback gets the column objects of `components` (what `chunk.col(C)` returns) after
  `(entity, chunk, row)`, in the order listed.
- Columns are resolved **once per chunk** instead of once per row. The TypeScript types are
  inferred from the list, so `pos.x` is a `Float32Array`.
- Rows are visited in reverse order per chunk. Rows where an enableable component of the
  query's `all` list is disabled are skipped. Structural changes are deferred until the
  outermost iteration ends.
- A column is `undefined` for chunks that don't have that component, and always for tags. List
  components from the query's `all`.
- The list is read on every call and not kept, so an inline array literal is fine.
- Up to 8 components take the fully specialized path. Longer lists still work, through a
  slower generic loop.

Measured speed-up over calling `chunk.col(C)` inside a plain `forEach` callback (Node 22,
Apple M4, isolated runs, same machine load; see the round-3 report):

| scenario | `forEach(fn)` | `forEach(components, fn)` | speed-up |
|---|--:|--:|--:|
| packed_5 (5 queries x 1000 rows) | ~112k ops/s | ~187k ops/s | ~1.7x |
| simple_iter (4 chunks x 1000 rows, 2 components) | ~52k ops/s | ~123k ops/s | ~2.4x |
| frag_iter (26 chunks x 100 rows) | ~200k ops/s | ~347k ops/s | ~1.7x |

### 3. `forEach(fn)`: plain rows

```ts
q.forEach((entity, chunk, row) => {
  chunk.col(Position).x[row] += 1;
});
```

It follows the same visiting rules as the columns form. It is convenient, but `chunk.col(C)`
runs again for every row.

### How forEach is optimized

A callback passed to `forEach` a second time gets its own compiled copy of the row loop, with
a WeakMap cache and at most 1024 copies. That makes the call monomorphic, so V8 can inline the
callback.

- **Fresh closures:** a closure created anew on every call is never specialized. Hoist
  callbacks out of hot paths.
- **Compiled chunk trampolines are cached by shape.** `forEachChunk` and the columns form of
  `forEach` compile per-chunk trampolines keyed by (column count, field names, chunk count),
  not by chunk, so archetype growth, new archetypes and new queries reuse compiled code. At
  most 4096 distinct shapes are compiled per process (`MAX_TRAMPOLINE_SHAPES`); later shapes
  use the generic loop while earlier ones keep their fast path. Details:
  [INTERNALS.md](INTERNALS.md#why-the-plain-chunk-loop-trails-closure-constant-loops) and the
  section before it.
- **No runtime code generation:** where it isn't allowed (CSP without `'unsafe-eval'`), both
  forms fall back to a shared generic loop.

## Systems

```ts
world.system(name, { query, group = 'update', order = 0 }, (q, dt, world) => { ... });
world.addSystem(MySystemClass, { group, order });
world.update(dt, group);
```

- Systems run in `(order, registration)` order.
- Each system runs as one iteration level: structural changes it makes are queued and flushed
  right after it returns.
- The system body is called through a megamorphic call site on purpose. V8 then never inlines
  the body into `world.update`'s `try/finally`, where tight loops compile about 1.7x slower.
  Per-tick overhead is about 14 ns per `update()` with one system.
- Fetch columns inside the system body on every tick. They go stale after the table grows
  **or shrinks**, as described above: a `world.compact()` or `world.clear()` between two ticks
  replaces the buffer under a column you fetched in the first one.

## Memory: measuring and reclaiming it

Tables only ever grow on their own. A wave of destroys, a level teardown or a spawn burst that
is over leaves capacity behind, and CozyECS never takes it back by itself.

**The ECS provides mechanism, never policy.** There is no background GC, no heuristic, no
reference tracing and nothing that runs on a tick. Your engine decides *when* to reclaim; these
calls are what make it possible, because the buffers are private.

All of them must be called **between ticks**. `compact()`, `clear()` and `dispose()` throw if
they are reached from inside a system, a `forEach`, or a hook that runs during a flush;
`compact()` and `clear()` additionally refuse to run with structural commands still queued
(`dispose()` discards those instead):

```
CozyECS: world.compact() cannot run during iteration; call it between ticks.
```

### `world.memory(): WorldMemory`

A snapshot of the storage the world owns.

```ts
const m = world.memory();
m.entities;            // live entities (queued spawns included)
m.tables.used;         // sum of count * rowBytes
m.tables.reserved;     // sum of buffer.byteLength  -- reserved - used is the slack
m.entityIndex;         // bytes in the allocator's slot / bigAid / free-stack arrays
m.strings.count;       // interned strings, '' included
m.total;               // tables.reserved + entityIndex
m.archetypes;          // [{ id, components, count, capacity, rowBytes, bytes }, ...] in id order
```

Sizes are bytes of ECS-owned typed-array storage. JS object overhead (archetype descriptors,
queries, systems, the interned strings themselves) is not counted.

`archetypes` lists **every** archetype the world has created, including the empty root archetype
(id 0, no components) that `spawn([...])` routes through. At a large `initialCapacity` that root
table is not negligible — 400 KB at `initialCapacity: 100_000`, since its `rowBytes` is the 4-byte
entity column alone — and `compact()` deflates it like any other, so `stats.archetypes` can count
an archetype you never spawned into.

It is **diagnostic, so it allocates** — one result object, plus one entry and one name array per
archetype (~1.4 KB for four archetypes). Call it from a debug overlay or a test, never per frame.

### `world.compact(options?): CompactStats`

Deflates every archetype to the capacity the growth policy would have reached for its live row
count, so the next spawn does not immediately re-grow it. An empty archetype goes to a
zero-length buffer.

```ts
const stats = world.compact();                  // { archetypes, bytesFreed }
world.compact({ minBytes: 0 });                 // deflate even a nearly-full table
world.compact({ strings: true });               // also rebuild the string table
```

| option | default | meaning |
|---|---|---|
| `minBytes` | `4096` | skip an archetype unless deflating it frees at least this many bytes, so compaction never churns tables that are nearly full |
| `strings` | `false` | also rebuild the `StringTable`, dropping strings no live row references, and rewrite every `str` column with the new ids |

The target is the growth policy's own capacity for `count` rows, run forward from the world's
`initialCapacity` — so `initialCapacity` is the floor for any **non-empty** table. A world created
with `{ initialCapacity: 1_000_000 }` and ten live rows in a table frees nothing from it, by
design: that is the capacity the table would have had all along. Only an archetype whose `count`
is 0 goes to a zero-length buffer.

What survives: archetype **objects**, their `id`, mask, key, component list, transition edges and
their membership in every query's `chunks`. Live entities keep their component values, their
enabled bits and their handles. Only buffers are replaced — so the typed arrays from `col(C)`,
`enabledArray(C)` and `entities` go stale exactly as they do on growth.

With `{ strings: true }`, **interned ids are not stable across the call.** Every `str` column
inside the world is rewritten in the same pass, so stored values are fine, but an id you are
holding yourself — from `world.strings.intern(...)` or read with `getField(e, C, 'name')` — is
meaningless afterwards. `stats.strings` reports `{ before, after }`. Interning a dropped string
again simply gets a fresh id.

### `world.clear(options?): void`

Destroys every entity in one pass instead of one at a time: table counts are reset, every live
index goes back to the allocator with its generation **bumped**, and (by default) every table is
deflated.

```ts
world.clear();                        // fast path: no events, tables deflated
world.clear({ compact: false });      // keep the capacities for an imminent refill
world.clear({ events: true });        // fire onRemove / onExit per entity
```

| option | default | meaning |
|---|---|---|
| `events` | `false` | **No `onRemove` hook and no query `onExit` listener fires.** Skipping the per-entity work is the point of the fast path. `true` fires them with the same ordering rules as `destroy()` — hooks in component-id order, then query exits — after every row has already been removed |
| `compact` | `true` | deflate every table to a zero-length buffer afterwards |

If a listener owns something the ECS does not — a sprite, a socket, a GPU resource keyed by
entity — pass `{ events: true }` or tear those down yourself before calling.

Handles taken before the call stay dead, so `isAlive` keeps answering correctly. Archetypes,
queries, systems, component registrations, transition edges and the string table all survive:
spawning works immediately afterwards, and queries report 0 until it happens.

Note that `clear()` can make `memory().entityIndex` **grow**: the free stack is sized to hold
every index that has ever been handed out. The entity index never shrinks, which is what keeps
old handles reading as dead.

### `world.dispose(): void`

The last rung: the world itself, not its contents. Use it when a world is being thrown away —
a level change, a backend switch, a test teardown — and nothing is going to spawn into it again.

```ts
world.dispose();
world.disposed;        // true
```

In order: every registered `registerWorldDisposeHook` callback runs first, with the world still
fully readable; every system is unregistered and `onDestroy()` called on class systems in reverse
registration order; every cached query drops its archetypes, its `onEnter`/`onExit` listeners and
its compiled loops; `onAdd`/`onRemove` listeners are dropped; **queued structural commands are
discarded without being applied**; every table is deflated to a zero-length buffer and the
archetype list is emptied; the entity index is released, which is what makes every handle read as
dead; and the view cache, component registry and interned strings are dropped.

| after `dispose()` | behaviour |
|---|---|
| `spawn`, `spawnMany`, `destroy`, `add`, `remove`, `set`, `enable`, `archetype`, `query`, `onAdd`, `onRemove`, `system`, `addSystem`, `update`, `compact`, `clear`, `query.onEnter`, `query.onExit` | throw an `Error` naming `dispose()` |
| `isAlive` → `false`, `has` → `false`, `get` → `undefined`, `query.count()` → `0`, `forEach` iterates nothing, `memory()` all zeros | safe, answering for an empty world |
| `dispose()`, `removeSystem()`, `flush()`, unsubscribe functions handed out earlier | no-ops |

Measured on a world of 20,000 `Position + Velocity` entities (`initialCapacity: 1024`):
`memory().tables.reserved` goes 404,096 → 0 (400,000 for the table plus 4,096 for the empty root
archetype), `memory().total` → 0, `memory().archetypes` → `[]` and `world.strings.size` → 1 (the
structural `''`).

**Column references are invalidated, exactly as by growth and compaction** — and finally. Any
`col(C)`, `enabledArray(C)` or `entities` array taken before the call now views a buffer the
world has let go of, and nothing will hand you a new one.

`dispose()` is **idempotent**, and a hook or an `onDestroy()` that throws does not abort it: the
world is disposed completely and the first error is rethrown afterwards. Components are *not*
touched — a `component()` descriptor is process-global and shared by every world, so the same
components keep working in the next world you build.

Dropping the world without disposing it is also fine: nothing in the library holds a world, so it
is collectable on its own (measured in `__tests__/leak.test.ts`). `dispose()` is what gives the
bytes back *now* rather than at the GC's convenience, and what lets attached resources —
`cozyecs/gpu`'s device buffers, through the hook — be released with the world.

### `archetype.shrinkToFit(minCapacity = 0): boolean`

The single-table primitive the two calls above are built on. Reallocates the table to
`max(count, minCapacity)` rows and returns whether it actually reallocated.

"Deflate, don't delete": the archetype object and everything attached to it survive; only the
storage is replaced. A target of 0 leaves a zero-length buffer and zero-length views; such a
table re-grows from capacity 1 on the next `pushRow`, so call `ensureCapacity` first when you
know the size of the next burst.

### A level teardown, end to end

```ts
// The world is being refilled: keep it, empty it.
function unloadLevel(world: World, kernel: KernelSystemHandle) {
  // Between ticks: not inside a system, and nothing queued.
  world.clear();                         // every entity gone, every table deflated
  world.compact({ strings: true });      // give the string table back too
  kernel.releaseUnused();                // and the device buffers (see docs/GPU.md)
}

// The world is going away: one call does all of it, device buffers included.
function unloadGame(world: World) {
  world.dispose();                       // hooks, systems, queries, tables, entity index
}
```

Both `compact()` and `clear()` are cheap to call and free to skip. Reclaiming nothing costs one
pass over the archetype list and a ~40-byte stats object. `dispose()` is the one that is not
optional-shaped: it ends the world, and there is no call that brings it back.
