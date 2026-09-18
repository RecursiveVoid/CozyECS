# CozyECS API: queries, iteration and systems

This page covers the iteration API: `Query`, chunks (`Archetype`), `forEach` and systems.
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

**Column arrays go stale when a table grows.** All typed columns of an archetype are views
over one buffer, and that buffer is replaced when the table grows. After a spawn, add, remove
or destroy that runs immediately, arrays you got earlier from `col(C).x`, `enabledArray(C)` or
`entities` may point to the old buffer. The column *object* returned by `col(C)` keeps its
identity; only its properties are replaced.

So fetch the arrays again in every tick or loop, as in the examples below. Never keep them
across structural changes.

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
- Fetch columns inside the system body on every tick. They go stale after the table grows,
  as described above.
