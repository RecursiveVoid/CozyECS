# CozyECS guide

The full guide to the core library: concepts, the API reference, performance tips and limits.
For the GPU entry point see [GPU.md](GPU.md); for how the internals work see
[INTERNALS.md](INTERNALS.md); for benchmark methodology see
[../benchmarks/RESULTS.md](../benchmarks/RESULTS.md).

## Contents

- [Concepts](#concepts)
  - [Components and tags](#components-and-tags)
  - [Archetypes and spawning](#archetypes-and-spawning)
  - [Entity handles and generations](#entity-handles-and-generations)
  - [Queries and iteration](#queries-and-iteration)
  - [Systems](#systems)
  - [Deferred structural changes](#deferred-structural-changes)
  - [Events](#events)
  - [Shared buffers and column invalidation](#shared-buffers-and-column-invalidation)
  - [GPU kernels (experimental)](#gpu-kernels-experimental)
- [API reference](#api-reference)
- [Performance tips](#performance-tips)
- [Limits](#limits)

## Concepts

The snippets below build on the Quick start: they reuse `world`, `Position`, `Velocity` and
`Player`.

### Components and tags

A component is a schema: field names mapped to field tokens. Each field is stored in its own
typed-array column.

| token | column | `set` / `get` value |
|---|---|---|
| `f32`, `f64` | `Float32Array`, `Float64Array` | number |
| `i8`, `i16`, `i32` | `Int8Array`, `Int16Array`, `Int32Array` | number |
| `u8`, `u16`, `u32` | `Uint8Array`, `Uint16Array`, `Uint32Array` | number |
| `bool` | `Uint8Array` (0/1) | boolean |
| `str` | `Uint32Array` of interned string ids | string |

```ts
import { component, tag, i32, u8, bool, str } from 'cozyecs';

const Health = component({ hp: i32, max: i32 }, { name: 'Health' });
const Name = component({ value: str }, { name: 'Name' });
const Sprite = component({ visible: bool, layer: u8 }, { name: 'Sprite' });

// A tag has no fields and costs no memory per entity.
const Frozen = tag({ name: 'Frozen' });

// An enableable component can be switched off per entity without moving the entity.
const Ai = component({ state: u8 }, { name: 'Ai', enableable: true });
```

- Components do not belong to a world. Ids come from one counter for the whole process, so
  define each component once, at module level, and use it with any number of worlds.
- `name` is only used in error messages.
- **`str` fields** store ids from `world.strings`, which is a `StringTable`. `set` and `get`
  convert for you. Raw columns and `getField` give you the id. Id `0` is always `''`. Interned
  strings are never evicted, so do not use `str` for unbounded unique values.

```ts
const hero = world.spawn(world.archetype(Position, Name, Health, Sprite));
world.set(hero, Name, { value: 'Ayla' });
world.set(hero, Health, { hp: 30, max: 30 });
world.set(hero, Sprite, { visible: true, layer: 2 });

const heroName = world.get(hero, Name)!.value;          // 'Ayla'
const nameId = world.getField(hero, Name, 'value');     // interned id (a number)
const sameName = world.strings.get(nameId);             // 'Ayla'
const visible = world.get(hero, Sprite)!.visible;       // true
```

- **Enableable components** use one extra byte per entity for the flag. A disabled component
  stays on the entity: `has` stays true and the entity keeps its archetype. Enabling or
  disabling is immediate and never fires events. `forEach` skips rows where a component from the
  query's `all` list is disabled. Chunk loops do not skip them: check `chunk.enabledArray(C)`.

```ts
const guard = world.spawn(world.archetype(Position, Ai));
world.enable(guard, Ai, false);
const guardAiOn = world.isEnabled(guard, Ai);   // false
const guardHasAi = world.has(guard, Ai);        // true
world.enable(guard, Ai);                        // on again (default `true`)
```

### Archetypes and spawning

An archetype is the table for one exact set of components. Entities with the same set share a
table, one row per entity. The rows are packed with no holes, and the columns are contiguous.

```ts
const Bullet = component({ ttl: f32 }, { name: 'Bullet' });
const bulletArch = world.archetype(Position, Velocity, Bullet); // order and duplicates ignored

// One entity: all fields start at zero.
const bullet = world.spawn(bulletArch);
world.set(bullet, Bullet, { ttl: 1 });

// Many entities: storage grows once, and `init` writes straight into the columns.
world.spawnMany(bulletArch, 1000, (chunk, row, i) => {
  chunk.col(Position).x[row] = i;
  chunk.col(Velocity).dx[row] = 10;
  chunk.col(Bullet).ttl[row] = 2;
});
```

**Spawn directly into the full archetype.** Each `add` or `remove` moves the entity to another
table, copying its row. `world.spawn(bulletArch)` places the entity once. `spawn()` followed by
three `add` calls creates it in the empty archetype and then moves it three times.

- `world.archetype(...)` is cached: the same set always returns the same `Archetype`. Look it up
  once and keep it.
- `world.spawn([A, B])` (the array form) also works, but it builds a key string on every call.
  Pass an `Archetype` on hot paths.
- `world.spawn()` with no argument creates an entity with no components.

### Entity handles and generations

An entity is a plain `number` (a `u32`): a 20-bit index plus a 12-bit generation. Destroying an
entity frees its index for reuse, and the reused index gets the next generation. A stale handle
therefore stays dead even after its index has been reused.

```ts
const temp = world.spawn(world.archetype(Position));
world.destroy(temp);
const tempAlive = world.isAlive(temp);                     // false
const reused = world.spawn(world.archetype(Position));     // reuses temp's index
const sameIndex = (reused & 0xfffff) === (temp & 0xfffff); // true
const staleStillDead = world.isAlive(temp);                // false: different generation
```

- `destroy` on a dead handle is ignored. So is `remove`. `add`, `set` and `enable` throw on a
  dead handle. `get` returns `undefined` for one.
- Handles are only meaningful inside the world that created them.
- The generation wraps after 4096 reuses of the same index. Do not keep handles forever without
  checking `isAlive`.

### Queries and iteration

```ts
const movers = world.query({ all: [Position, Velocity], none: [Frozen] });
const moverCount = movers.count();
```

- `all`: an archetype must have every one of these components. `any`: at least one of them, if
  the list is not empty. `none`: none of them.
- Queries are cached. Any description with the same sets returns the same `Query`, whatever the
  order of the components.
- `query.chunks` holds the matching archetypes. It is **the same array for the query's whole
  life**, and new matching archetypes are appended to it. It can contain empty chunks
  (`count === 0`).
- `count()` adds up the rows of all chunks, disabled rows included.

There are four ways to iterate, from most direct to most convenient. For raw speed on small loop
bodies, `forEachChunk` with a callback defined once (or a CPU `kernelSystem`) beats the plain chunk
loop by about 17%, because only a compiled loop lets V8 treat the column arrays as constants (see
[Performance tips](#performance-tips)). On heavier bodies and large tables the difference fades.

**1. Chunk loops.** A plain loop over typed arrays. Rows are `0..count-1` in each chunk.

```ts
for (const chunk of movers.chunks) {
  const p = chunk.col(Position), v = chunk.col(Velocity);
  const x = p.x, dx = v.dx, ents = chunk.entities;
  for (let i = chunk.count - 1; i >= 0; i--) {
    if (x[i] > 10_000) world.destroy(ents[i]);
    else x[i] += dx[i];
  }
}
```

A destroy inside a chunk loop that is *not* running in a system applies immediately. It swaps
the chunk's last row into row `i` and shrinks `count`. That is why the loop above visits rows in
reverse order. A forward loop would skip the swapped-in row. You can also run the loop inside a
system, where changes are deferred, or collect the handles first.

Chunk loops ignore enabled flags. Check them yourself:

```ts
const thinkers = world.query({ all: [Ai] });
for (const chunk of thinkers.chunks) {
  const on = chunk.enabledArray(Ai)!;
  const state = chunk.col(Ai).state;
  for (let i = 0; i < chunk.count; i++) if (on[i] === 1) state[i]++;
}
```

**2. `forEachChunk(components, fn)`.** A chunk loop with the columns resolved for you. `fn`
receives `(count, ...columns, chunk)`. If you define `fn` once and pass the same function every
time, CozyECS compiles a specialized loop for it, which is faster than a hand-written chunk loop
(1.16–1.30x bitecs vs 0.78–0.86x for the plain loop on a 2-add body, 1k–50k entities).

```ts
const integrate = (n: number, pos: { x: Float32Array; y: Float32Array }, vel: { dx: Float32Array; dy: Float32Array }) => {
  const x = pos.x, y = pos.y, dx = vel.dx, dy = vel.dy;
  for (let i = 0; i < n; i++) { x[i] += dx[i]; y[i] += dy[i]; }
};
movers.forEachChunk([Position, Velocity], integrate);
```

**3. `forEach(components, fn)`.** A callback per row, with the columns resolved once per chunk.

```ts
movers.forEach([Position, Velocity], (entity, chunk, row, pos, vel) => {
  pos.y[row] += vel.dy[row];
});
```

**4. `forEach(fn)`.** A callback per row with `(entity, chunk, row)`.

```ts
movers.forEach((entity, chunk, row) => {
  if (chunk.col(Position).y[row] < 0) world.add(entity, Frozen); // deferred until forEach returns
});
```

Both `forEach` forms visit rows in reverse order within each chunk. They skip rows where an
enableable component from `all` is disabled, and they defer structural changes (see
[Deferred structural changes](#deferred-structural-changes)). A column is `undefined` for a tag,
or for a component that the chunk does not have, so list only components from `all`.

### Systems

A system is a named unit of per-tick work. It belongs to a **group** (default `'update'`) and
has an **order** (default `0`; lower runs first, and ties run in registration order).
`world.update(dt, group)` runs every enabled system of that group.

**Function systems** receive `(query, dt, world)`:

```ts
const drawn: number[] = [];

const gravity = world.system('gravity', { query: { all: [Velocity] }, order: -10 }, (q, dt) => {
  for (const chunk of q.chunks) {
    const dy = chunk.col(Velocity).dy;
    for (let i = 0; i < chunk.count; i++) dy[i] -= 9.8 * dt;
  }
});

world.system('draw', { group: 'render', query: { all: [Position, Player] } }, (q) => {
  for (const chunk of q.chunks) for (let i = 0; i < chunk.count; i++) drawn.push(chunk.entities[i]);
});
```

**Class systems** extend `System`. Field initializers can already use `this.query(...)` and
`this.world`:

```ts
import { System } from 'cozyecs';

class Lifetime extends System {
  bullets = this.query({ all: [Bullet] });
  expired = 0;

  onCreate() {
    // called once, right after registration
  }

  onUpdate(dt: number) {
    for (const chunk of this.bullets.chunks) {
      const ttl = chunk.col(Bullet).ttl;
      const ents = chunk.entities;
      for (let i = 0; i < chunk.count; i++) {
        ttl[i] -= dt;
        if (ttl[i] <= 0) {
          this.world.destroy(ents[i]); // queued; applied right after this system returns
          this.expired++;
        }
      }
    }
  }

  onDestroy() {
    // called by world.removeSystem
  }
}

const lifetime = world.addSystem(Lifetime, { order: 10 });
```

A game loop runs its groups separately:

```ts
world.update(1 / 60);          // 'update' group: gravity (-10), move (0), Lifetime (10)
world.update(0, 'render');     // 'render' group: draw

gravity.enabled = false;       // skipped by update() until re-enabled
world.removeSystem(lifetime);  // unregisters it and calls onDestroy()
```

### Deferred structural changes

`spawn`, `spawnMany`, `destroy`, `add` and `remove` are **structural changes**, because they move
rows between tables. Inside a system run, a `forEach` or a `forEachChunk`, they are queued in a
command buffer instead of being applied. The buffer is flushed:

- right after each system returns (inside `world.update`),
- when the outermost `forEach` or `forEachChunk` returns,
- when you call `world.flush()` outside any iteration.

Outside those contexts, structural changes apply immediately.

While a change is queued:

- `spawn` returns a handle that is already alive, but it has no components yet: `has` is false,
  `get` is `undefined`, and `set` throws.
- `destroy` leaves the entity alive and readable until the flush.
- `add(e, C, values)` keeps a reference to `values` and reads it at flush time, so do not reuse
  or mutate that object.
- A command whose entity died before the flush is skipped.

`set`, `get`, `getField`, `enable`, `isEnabled` and `has` are never deferred.

```ts
let shot = -1;
let shotPlacedInside = true;
world.system('spawner', {}, (_q, _dt, w) => {
  shot = w.spawn(bulletArch);                  // alive now, placed after this system returns
  shotPlacedInside = w.has(shot, Bullet);      // false here
});
world.update(0);
const shotPlacedAfter = world.has(shot, Bullet); // true
```

### Events

```ts
let added = 0, removed = 0, entered = 0, exited = 0;

const offAdd = world.onAdd(Health, (e) => { added++; });       // add, spawn, spawnMany
const offRemove = world.onRemove(Health, (e) => { removed++; }); // remove, destroy
const offEnter = movers.onEnter((e) => { entered++; });        // starts matching the query
const offExit = movers.onExit((e) => { exited++; });           // stops matching the query

const npc = world.spawn(world.archetype(Position, Velocity, Health)); // added = 1, entered = 1
world.add(npc, Frozen);                                               // exited = 1
world.destroy(npc);                                                   // removed = 1

offAdd(); offRemove(); offEnter(); offExit(); // unsubscribe (safe to call twice)
```

- Events fire **after** the change has been applied. On destroy, the handle is already dead, so
  its data can no longer be read.
- For one change, the order is: `onRemove` hooks, then `onAdd` hooks, then query exit/enter.
- `set` and `enable` never fire events.
- A callback may make structural changes itself. They are deferred if the world is iterating
  or flushing.
- Events cost nothing when nobody listens. With listeners, flushes take a slower path.

### Shared buffers and column invalidation

All columns of an archetype (`entities`, every field and every enabled flag) are views over one
buffer. `new World({ shared: true })` allocates these buffers as `SharedArrayBuffer`, so columns
can be sent to workers without copying. If `SharedArrayBuffer` is not available (for example in
a browser page without cross-origin isolation), CozyECS uses a plain `ArrayBuffer`.

```ts
const sharedWorld = new World({ shared: true, initialCapacity: 4096 });
const particles = sharedWorld.archetype(Position, Velocity);
sharedWorld.spawnMany(particles, 1000);

const isShared = particles.shared;         // true in Node and cross-origin-isolated pages
const xs = particles.col(Position).x;      // Float32Array over particles.buffer
// worker.postMessage({ xs, count: particles.count }); // the worker sees the same memory
```

**Columns go stale when a table grows.** When a table runs out of capacity, CozyECS allocates a
new, larger buffer and copies the rows. Typed arrays you got earlier from `col(C).x`,
`enabledArray(C)` or `entities` still point at the old buffer. The object returned by `col(C)`
keeps its identity; only its properties are replaced.

```ts
sharedWorld.spawnMany(particles, 10_000);        // 11,000 rows > capacity 4096: the table grows
const stale = xs !== particles.col(Position).x;  // true: re-fetch after structural changes
```

- Fetch column arrays inside each system run or loop. Never keep them across spawns, adds,
  removes or destroys that apply immediately.
- After growth, a worker holding old arrays keeps reading the old buffer. Send it the new
  arrays, or set `initialCapacity` so the table never grows.
- `shared: true` costs no memory. It measured within ±2% in most scenarios and 6–9% slower in
  one fragmented-iteration case ([details](../benchmarks/RESULTS.md)).
  Keep the default (`false`) unless you use workers.

### GPU kernels (experimental)

> **EXPERIMENTAL.** The API, the kernel subset and the break-even constants may
> change in any minor release. Full documentation: [docs/GPU.md](GPU.md).

`cozyecs/gpu` is a separate, opt-in entry point. It lets you write a system as
a plain JavaScript function. CozyECS parses it (no dependencies), compiles it to
WGSL for WebGPU, and also compiles it to a flat CPU chunk loop. The core
`cozyecs` bundle never imports it.

```js
import { kernelSystem } from 'cozyecs/gpu';

// Position { x, y } and Velocity { dx, dy } from the quick start.
const fall = await kernelSystem(world, 'Fall', {
  components: [Position, Velocity],  // kernel parameters, matched BY POSITION
  uniforms: { gravity: -9.8 },       // the only way to pass outside values in
  target: 'auto',                    // 'gpu' | 'cpu' | 'auto'
  readback: 'async',                 // 'async' | 'sync-frame' | 'none'
  kernel: (p, v, dt, u) => {
    v.dy += u.gravity * dt;
    p.x += v.dx * dt; p.y += v.dy * dt;
    if (p.y < 0) { p.y = 0; v.dy = -v.dy * 0.5; }
  },
});
world.update(1 / 60);  // dispatches inside the system's group, never blocks
```

- **Free identifiers are errors.** A closure variable cannot be read from
  `Function.prototype.toString`, so `p.y += gravity * dt` fails at registration
  with `E_UNKNOWN_IDENTIFIER` telling you to use `uniforms: { gravity }` and
  `u.gravity`. `Math.random`, strings, objects, your own function calls and any
  `world` access are also rejected.
- **Readback.** `'async'` (default) applies GPU results **next frame**, so CPU
  code reads last frame's values. `'sync-frame'` lets you `await fall.sync()` (or
  `flushKernels(world, group)`) before the frame ends: slower, but current.
  `'none'` never reads back; use `fall.bufferFor(archetype)` to render straight
  from the GPU buffer. If you dispatch faster than the GPU completes readbacks,
  frames are coalesced, never lost: `sync()` still waits for every frame.
- **`target: 'auto'`** picks CPU or GPU on every dispatch. The defaults are
  45k entities (`'async'`/`'none'`) and 700k (`'sync-frame'`) for a
  baseline-cost kernel (1.14 ns/entity on the CPU), scaled by the kernel's
  estimated CPU cost. They are one machine's numbers (an M4 through Dawn);
  override them with `setAutoThresholds()` or the `thresholds` option.
- **Measured break-even** (entities above which the GPU beats the compiled CPU
  loop, M4 + Dawn, median of 5 isolated runs, every cell checked row by row):

  | kernel | `'none'` | `'async'` | `'sync-frame'` |
  |---|---:|---:|---:|
  | simple (`p.x += v.dx; p.y += v.dy`) | ~49k | ~64k | never, up to 1M |
  | gravity (integrate + bounce) | ~23k | ~34k | ~554k |

  `auto` keeps smaller queries on the CPU. At 1M entities the GPU with `'none'` runs the
  simple kernel in 0.35 ms against 1.80 ms on the CPU, but below ~20k it is
  always slower (a ~0.06 ms per-frame floor). `'sync-frame'` rarely pays off.
- **The CPU backend is a speedup on its own.** `kernelSystem(world, name,
  { target: 'cpu', ... })` compiles the kernel into a loop with every column
  as a constant: 1.08–1.30x bitecs on the simple kernel up to 50k entities and
  1.04–1.15x on gravity (1k–1M, leaving out an outlier bitecs size).
- **Never throws for a capability reason.** If there is no WebGPU device (e.g.
  Node without `setGPUProvider`), the kernel uses a GPU-incompatible field, or
  the query includes an **enableable** component, the kernel falls back to the
  CPU backend and warns once. The CPU backend skips disabled rows exactly. If
  no backend can run at all, `fall.backend === 'none'` and you get one warning;
  the failure is never silent.
- **`f64` fields are computed in f32** (one warning per kernel).
- **Pairwise kernels** (`pairwise: true`, `(self, other, dt, u) => ...`) are
  **O(n²)** and only form pairs within one archetype.
- **Calibrate `auto` on the user's machine.** The built-in switch points were
  measured on one Apple M4. `await calibrateAuto(cozy)` (with
  `import * as cozy from 'cozyecs'`) measures the CPU/GPU break-even on the
  current device in ~0.7 s and applies it to kernels created afterwards (see
  [docs/GPU.md, "Choosing a target"](GPU.md#4-choosing-a-target)).
- **Status: experimental.** The parser, CPU backend and GPU backend are covered
  by parity and fuzz tests, in Node (Dawn over Metal) and in Chrome's WebGPU.

## API reference

### Components

| function | description |
|---|---|
| `component(schema, { name?, enableable? })` | Defines a component type. Returns a frozen `ComponentType` with `id`, `name`, `schema`, `keys`, `tokens`, `isTag` and `enableable`. |
| `tag({ name?, enableable? })` | Defines a component with no fields. |
| `f32` `f64` `i8` `i16` `i32` `u8` `u16` `u32` `bool` `str` | Field type tokens. |

### `World`

| member | description |
|---|---|
| `new World({ initialCapacity?, shared? })` | `initialCapacity` sets the starting rows per archetype and the size of the entity arrays (default 64). `shared` allocates tables as `SharedArrayBuffer` (default `false`). |
| `strings: StringTable` | Intern table for `str` fields: `intern(s)`, `get(id)`, `size`. |
| `archetype(...components): Archetype` | Gets or creates the archetype for a component set. Cached. |
| `spawn(archetype? \| components[]): number` | Creates an entity with all fields zeroed. Deferred while iterating. |
| `spawnMany(archetype, count, init?)` | Creates `count` entities and grows storage once. `init(chunk, row, i)` runs for each new row. Deferred while iterating. |
| `destroy(entity)` | Destroys an entity. Ignored if it is dead. Deferred while iterating. |
| `isAlive(entity): boolean` | True for live handles, including a spawn that is still queued. |
| `add(entity, C, values?)` | Adds `C`, optionally with values. If the entity already has `C`, only sets the values. Throws if the entity is dead. Deferred while iterating. |
| `remove(entity, C)` | Removes `C`. No-op if the entity is dead or lacks `C`. Deferred while iterating. |
| `has(entity, C): boolean` | True if the entity is alive, placed, and has `C`. |
| `set(entity, C, values)` | Writes the given fields immediately. Keys that are not in the schema are ignored. Throws if the entity is dead or lacks `C`. |
| `get(entity, C)` | Reads all fields into a view object that is cached per component and **overwritten on the next call**, so copy what you need. `undefined` if the entity is dead or lacks `C`. |
| `getField(entity, C, key): number` | Raw column value (`str` gives the id, `bool` gives 0/1). The entity must have `C`. |
| `enable(entity, C, on = true)` | Sets an enableable component's flag. Throws if `C` is not enableable, or if the entity is dead or lacks `C`. |
| `isEnabled(entity, C): boolean` | False if the entity is dead or lacks `C`. True if `C` is not enableable. Otherwise the flag. |
| `onAdd(C, cb)` / `onRemove(C, cb)` | Component hooks. Each returns an unsubscribe function. |
| `query({ all?, any?, none? }): Query` | Gets or creates a cached query. |
| `system(name, { query?, group?, order? }, fn): SystemHandle` | Registers a function system `fn(query, dt, world)`. `query` can be a `Query` or a description. |
| `addSystem(SystemClass, { group?, order? })` | Constructs a class system, registers it, calls `onCreate()` and returns the instance. |
| `removeSystem(system)` | Unregisters a system, sets `enabled = false` and calls `onDestroy()` for class systems. |
| `update(dt = 0, group = 'update')` | Runs the group's enabled systems in order, flushing after each one. If a system throws, pending commands are flushed and the error propagates. |
| `flush()` | Applies queued structural changes. No-op while iterating or already flushing. |

### `Query`

| member | description |
|---|---|
| `chunks: Archetype[]` | Matching archetypes, in creation order. The same array for the query's whole life. |
| `all`, `any`, `none` | The normalized component lists. |
| `count(): number` | Total rows across all chunks, disabled rows included. |
| `matches(archetype): boolean` | Whether an archetype matches the query. |
| `forEach(fn)` | `fn(entity, chunk, row)` for every enabled row. |
| `forEach(components, fn)` | `fn(entity, chunk, row, ...columns)`. Columns are resolved once per chunk. |
| `forEachChunk(components, fn)` | `fn(count, ...columns, chunk)` once per non-empty chunk. Does not filter by enabled flags. |
| `onEnter(cb)` / `onExit(cb)` | `cb(entity)` when an entity starts or stops matching. Each returns an unsubscribe function. |

### `Archetype` (also exported as the type `Chunk`)

| member | description |
|---|---|
| `count: number` | Live rows. Rows `0..count-1` are packed. |
| `capacity: number` | Allocated rows. |
| `entities: Uint32Array` | The handle in each row. Read only `[0, count)`. |
| `col(C)` | Column object of `C`, with one typed array per field. `undefined` for a tag or a component the archetype lacks. |
| `enabledArray(C): Uint8Array \| undefined` | Enabled flags (1 = on) for an enableable `C`. |
| `has(C): boolean` | Whether the archetype contains `C`. |
| `isEnabled(C, row): boolean` | The flag for one row. |
| `components` | The component types, sorted by id. |
| `buffer`, `shared`, `rowBytes` | The backing buffer, whether it is a `SharedArrayBuffer`, and bytes per row. |

The typed arrays from `entities`, `col(C)` and `enabledArray(C)` are replaced when the table
grows (see [column invalidation](#shared-buffers-and-column-invalidation)).

### Systems

| member | description |
|---|---|
| `SystemHandle` | Returned by `world.system`. Fields: `name`, `group`, `order`, `query`, `fn` and a writable `enabled`. |
| `abstract class System` | Fields: `world`, `name`, `group`, `order` and `enabled`. `query(desc)` is shorthand for `world.query`. Implement `onUpdate(dt)`. `onCreate()` and `onDestroy()` are optional. |

Exported types: `WorldOptions`, `SpawnInitFn`, `SystemOptions`, `FunctionSystemOptions`,
`SystemFn`, `SystemClass`, `ComponentType`, `ComponentOptions`, `Schema`, `ColumnsOf`,
`ValuesOf`, `FieldToken`, `TypedArray`, `EntityCallback`, `QueryDesc`, `QueryForEachFn`,
`QueryForEachColumnsFn`, `ColumnsTuple` and `Chunk`.

More detail: [docs/API.md](API.md) covers iteration and systems, and
[docs/INTERNALS.md](INTERNALS.md) covers storage, entities and the command buffer.

## Performance tips

1. **Iterate chunks, not entities, and let the library compile the loop.** The fastest forms
   are `kernelSystem(world, name, { target: 'cpu', ... })` from `cozyecs/gpu` and
   `forEachChunk` with a callback defined once: both run a compiled loop in which V8 treats
   the column arrays as constants. A plain chunk loop you write yourself
   (`for (let i = 0; i < chunk.count; i++) x[i] += dx[i]`) is about 17% slower on tiny loop
   bodies, because V8 cannot treat its arrays as constants; see
   [Why the plain chunk loop trails closure-constant loops](INTERNALS.md#why-the-plain-chunk-loop-trails-closure-constant-loops).
   Use `forEach` where convenience matters more.
2. **Define callbacks once.** `forEach` and `forEachChunk` compile a specialized loop for a
   callback they see a second time. A closure created fresh on every call never gets one.
   Compiled loops are cached by *shape* (column count and field names), so archetype growth,
   new archetypes and new queries reuse existing code. The only bound is 4096 distinct shapes
   per process (`MAX_TRAMPOLINE_SHAPES`), plus 1024 specialized `forEach` copies; past it, new
   shapes run on the generic loop and everything compiled earlier keeps its fast path. Earlier
   builds counted every rebuild against a lifetime cap, so a long-running app could silently
   lose its compiled loops; that is fixed.
3. **Spawn into the final archetype**, and use `spawnMany` with an `init` for batches. Avoid
   `spawn()` followed by a chain of `add` calls.
4. **Keep hot structural changes in systems or `forEach`.** Queued commands are applied in
   batches. The entity_cycle and add_remove benchmarks run 1.6–4x faster than the best
   competitor either way.
5. **Prefer enableable components to add/remove** for state that toggles often. A flip is one
   byte write and does not move the row.
6. **Pass `Archetype` objects, not arrays,** to `spawn` on hot paths. The array form builds a key
   string.
7. **Re-fetch columns after structural changes.** A table that grows replaces its buffer.
8. **Don't use `get` in hot loops.** It converts every field into an object. Use `getField` or
   the columns.
9. **Set `initialCapacity`** when you know roughly how many entities you will have, to avoid
   repeated growth.
10. **Leave `shared` off** unless you actually hand columns to workers.

## Limits

- 1,048,576 (2^20) live entities per world.
- 32,768 archetypes per world.
- 12-bit generations: a handle's index is reused with a new generation, and the generation wraps
  after 4096 reuses.
- Structural changes and `forEach` must run on the thread that owns the world. With
  `shared: true`, other threads may read and write column values.
