# CozyECS GPU kernels (`cozyecs/gpu`)

> **EXPERIMENTAL.** `cozyecs/gpu` is an opt-in entry point whose API, subset
> grammar and break-even constants may change in any minor release. The core
> `cozyecs` package does not import it and is unaffected when you do not.
> Things to know before relying on it:
>
> - **Readback is next-frame by default** (`'async'`): CPU code sees last
>   frame's kernel results. Use `'sync-frame'` + `await` for this frame's
>   values, or `'none'` to keep data GPU-resident for rendering (§3.3).
> - **Pairwise kernels are O(n²)** within a chunk (§3.8).
> - **Enableable components force the CPU backend** (with one warning), which
>   skips disabled rows exactly (§3.6).
> - **`f64` fields are computed in f32** on both backends, with a one-time
>   warning per kernel (§2.5).

Write a system as a plain JavaScript function. CozyECS reads that function, turns
it into WGSL, and runs it on the GPU — or compiles it to a tight CPU loop when
that is faster. You never write shader code, and the core library is untouched
when you do not import this entry point.

```js
import { World, component, f32 } from 'cozyecs';
import { kernelSystem } from 'cozyecs/gpu';

const Position = component({ x: f32, y: f32 }, { name: 'Position' });
const Velocity = component({ x: f32, y: f32 }, { name: 'Velocity' });

const move = await kernelSystem(world, 'Move', {
  components: [Position, Velocity],   // kernel parameters, in order
  uniforms: { gravity: -9.8 },        // every outside value comes through here
  readback: 'async',
  group: 'update',
  kernel: (p, v, dt, u) => {
    v.y += u.gravity * dt;
    p.x += v.x * dt;
    p.y += v.y * dt;
    if (p.y < 0) { p.y = 0; v.y = -v.y * 0.5; }
  },
});

world.update(1 / 60);   // the kernel dispatches here, without blocking
```

Contents: [Install and requirements](#install-and-requirements) ·
[1. Architecture](#1-architecture) · [2. Kernel subset](#2-kernel-subset) ·
[3. Semantics](#3-semantics) · [4. Choosing a target](#4-choosing-a-target) ·
[5. Falling back to the CPU](#5-falling-back-to-the-cpu) ·
[6. CPU/GPU parity contract](#6-cpugpu-parity-contract) ·
[7. Error catalogue](#7-error-catalogue) · [8. Device limits](#8-device-limits) ·
[9. Reference](#9-reference) · [10. Running the GPU benchmarks](#10-running-the-gpu-benchmarks)

---

## Install and requirements

`cozyecs/gpu` ships inside the `cozyecs` package (`dist/gpu/index.{esm.js,cjs,d.ts}`,
exported as `"./gpu"`). There is nothing extra to install in a browser. The core
entry never imports it, so an app that does not import `cozyecs/gpu` pays nothing.

| environment | how the GPU is found | notes |
|---|---|---|
| Browser with WebGPU (Chromium-based browsers, and recent Safari/Firefox where WebGPU is enabled) | `navigator.gpu`, automatically | Needs a secure context (HTTPS or `localhost`). |
| Web worker | `navigator.gpu` in the worker, where the browser exposes it | Same rules as the page. |
| Node | `setGPUProvider(create([]))` from the optional peer dependency `webgpu` (Dawn); see [§1](#1-architecture) | `npm install webgpu`. Tested with `webgpu@0.6.1` on macOS/Metal. Without it there is no `navigator.gpu` in Node. |
| Deno, Bun, other runtimes | `navigator.gpu` if the runtime provides one, otherwise `setGPUProvider(gpu)` with any object implementing the WebGPU `GPU` interface | Not tested. |
| No WebGPU at all (older browsers, server-side rendering, CI, Node without `webgpu`) | nothing found, or no adapter | `kernelSystem` still works: the **CPU backend** is used, with one warning (§5). |

TypeScript: `dist/gpu/index.d.ts` starts with
`/// <reference types="@webgpu/types" />` (the `GPUBuffer` returned by
`bufferFor`, the `GPU` passed to `setGPUProvider`). Install the optional peer
`@webgpu/types` (`npm install -D @webgpu/types`), or type-check with
`skipLibCheck`. The core `cozyecs` declarations do not reference it.

Use `await getGPUContext()` (resolves to `null`, never throws, when there is no
device) or, after that, the synchronous `hasGPUContext()` to find out whether a
device was acquired, and `handle.backend` to see which backend a kernel uses.

---

## 1. Architecture

```
  kernel.toString()
        |
        v
  src/gpu/parse.ts     JS subset -> typed IR. Hand-written tokenizer + Pratt
        |              parser, no dependencies. Every rejection is a
        |              KernelError with a code frame.
        v
  src/gpu/ir.ts        The IR, its validator, and the binding layout that
       / \             both backends agree on. Pure data, no imports.
      /   \
     v     v
 wgsl.ts  cpu.ts       Two pure code generators.
     |     |
     v     v
  runtime.ts           Device, pipelines, buffers, dispatch, readback,
        |              and the CPU/GPU target decision.
        v
  index.ts             kernelSystem(): registers an ordinary system whose
                       body is one dispatch.
```

Four properties hold this together:

- **The core never imports `src/gpu/**`.** `dist/index.esm.js` and
  `dist/index.cjs` contain no reference to `gpu`, `webgpu` or `navigator`
  (checked after each build; ~38 KB unminified). The GPU code lives only in
  `dist/gpu/`.
- **`src/gpu/**` imports the core for types only.** No core code is duplicated
  into `dist/gpu/index.esm.js`.
- **Both backends are pure functions of the IR.** That is what makes the parity
  tests possible: the same IR produces a WGSL string and a JS string, and the
  two are run against each other.
- **`webgpu` is an optional peer dependency.** In a browser, `navigator.gpu` is
  used. In Node, install `webgpu` and hand the instance over once:

  ```js
  import { create, globals } from 'webgpu';
  import { setGPUProvider } from 'cozyecs/gpu';
  Object.assign(globalThis, globals);
  setGPUProvider(create([]));
  ```

### One storage buffer per archetype

CozyECS already stores an archetype as a single `ArrayBuffer` with one typed
array view per field. The GPU backend binds **that buffer**, once per 4-byte view
type, and addresses fields through base offsets passed in the uniform block.

The obvious alternative — one storage buffer per field — caps a kernel at 8
fields on a default device (10 on an M4 adapter); `Position{x,y,z} +
Velocity{x,y,z}` already burns 6. The one-buffer form has no such cap, mirrors
the CPU layout exactly, and measured **never slower** (1M-entity move kernel,
4 repeats: per-field 0.925 / 0.946 / 0.682 / 0.587 ms, one-buffer 0.742 / 0.688 /
0.639 / 0.578 ms).

Generated module shape:

```wgsl
struct U {
  dt: f32, count: u32, base: u32, countOther: u32,
  o0_Position_x: u32, o0_Position_y: u32, o0_Velocity_x: u32, o0_Velocity_y: u32,
  u_gravity: f32,
};
@group(0) @binding(0) var<storage, read_write> t_f32: array<f32>;
@group(0) @binding(1) var<uniform> u: U;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x + u.base;
  if (row >= u.count) { return; }
  /* body */
}
```

Field offsets are in **elements** of that field's view, taken from the live
typed array (`view.byteOffset / view.BYTES_PER_ELEMENT`) every frame, because a
table that grows is reallocated and every offset moves. Bind-group *offsets*
must be 256-byte aligned and a field column is not, which is the second reason
offsets live in the uniform rather than in the binding.

---

## 2. Kernel subset

### 2.1 Signature and parameter matching

Parameters are matched **by position**, never by name. A bundler renames
identifiers, and a closure variable does not appear in `Function.prototype.toString()`
at all. Property names (`p.x`, `u.gravity`) do survive minification and are
matched by name.

| form | signature | arity |
|---|---|---|
| per-entity (default) | `(c0, c1, ..., dt?, u?)` | `components.length`, `+1`, or `+2` |
| pairwise | `(self, other, dt?, u?)` | 2, 3, or 4 |

`dt` is the value passed to `world.update(dt)`. `u` is the uniform block.

In a **pairwise** kernel, `self` and `other` expose the fields of *all*
`components` merged into one namespace, so two components may not share a field
name (`E_FIELD_COLLISION`).

### 2.2 Grammar

```
Body      := Stmt*
Stmt      := Decl | Assign | If | For | While | Block | 'break' | 'continue'
Decl      := ('const' | 'let') Ident '=' Expr ';'
Assign    := LValue AssignOp Expr ';'
           | LValue ('++' | '--') ';'   // statement or `for` update only,
           | ('++' | '--') LValue ';'   // never inside an expression
LValue    := Param '.' Ident            // a component field
           | Ident                      // a local declared above
AssignOp  := '=' | '+=' | '-=' | '*=' | '/=' | '%='
If        := 'if' '(' Expr ')' Block ('else' (Block | If))?
For       := 'for' '(' Decl? ';' Expr? ';' Assign? ')' Block
While     := 'while' '(' Expr ')' Block
Block     := '{' Stmt* '}' | Stmt

Expr      := Cond
Cond      := Or ('?' Expr ':' Expr)?
Or        := And ('||' And)*
And       := Cmp ('&&' Cmp)*
Cmp       := Add (('<' | '<=' | '>' | '>=' | '==' | '===' | '!=' | '!==') Add)*
Add       := Mul (('+' | '-') Mul)*
Mul       := Unary (('*' | '/' | '%') Unary)*
Unary     := ('-' | '!' | '+')? Primary
Primary   := Number | 'true' | 'false'
           | Param '.' Ident            // component field read
           | 'u' '.' Ident              // uniform read
           | Ident                      // local, or dt, or a builtin value
           | 'Math' '.' BuiltinFn '(' Args ')'
           | 'rand' '(' Expr ')'
           | '(' Expr ')'
```

`===`/`!==` are accepted and mean `==`/`!=`; there is one numeric type, so the
distinction is empty.

### 2.3 Types

**All arithmetic is `f32`.** Every expression is either `f32` or `bool`. There is
no integer arithmetic in the subset, and no bit operators.

An integer field (`i32`, `u32`, and on the CPU backend also `i8`/`i16`/`u8`/`u16`/
`bool`) is **read** as an `f32` and **written** as `floor(x + 0.5)` clamped to the
field's range. Note that this differs from what a plain TypedArray store would
do (truncate and wrap); the kernel semantics are the WGSL ones on both backends.

### 2.4 Builtins

| callable | notes |
|---|---|
| `Math.sin` `cos` `tan` `asin` `acos` `atan` | 1 argument |
| `Math.atan2` `min` `max` `pow` `hypot` | 2 arguments |
| `Math.sqrt` `abs` `floor` `ceil` `exp` `log` `sign` | 1 argument; `log` is the natural log |
| `Math.round` | emitted as `floor(x + 0.5)` on **both** backends: WGSL `round` is half-to-even, `Math.round` is half-up, and they would disagree on `.5` |
| `rand(seed)` | stateless hash, **bit-identical** on both backends. Not a stream: same seed, same value. Seed it per entity, e.g. `rand(index + u.frame * 7919)` |

| bare value | type | meaning |
|---|---|---|
| `dt` | f32 | the kernel's `dt` parameter |
| `index` | f32 | the entity's row index in its chunk |
| `count` | f32 | rows in the chunk |

`Math.random` is **not** available (`E_UNSUPPORTED_CALL`): it has no GPU
equivalent and no reproducibility. Use `rand`.

### 2.5 Not supported

Every item below is a registration-time error with a code frame, never a silent
miscompile:

strings and template literals · objects, arrays and `new` · closures and nested
functions · calls to your own functions (inline them) · `world.*`, spawning,
`add`/`remove`/`destroy` — no structural changes from a kernel · `Math.random` ·
`async`/`await` and generators · `try`/`catch` · `switch` · `do`/`while` ·
labeled `break`/`continue` (unlabeled are fine) · `return` · bit operators (`&`
`|` `^` `<<` `>>` `~`) · `++`/`--` inside an expression (`x = y++`; as a
statement or a `for` update they are fine) · comma expressions · optional
chaining · `in`/`instanceof` · destructuring, default or rest parameters ·
`let` without an initializer.

`f64` fields and sub-word fields (`i8`/`i16`/`u8`/`u16`/`bool`/`str`) are legal in
a kernel, but they force the **CPU backend** — see §5.

**`f64` is computed in f32.** Kernel arithmetic is f32 on both backends (that is
what makes them agree, §6), so an `f64` field keeps its 8-byte storage but every
value a kernel writes to it is rounded to f32 precision (~7 significant digits).
Registering such a kernel prints one warning naming the fields, whatever the
`target`. Keep high-precision accumulators (world coordinates far from the
origin, simulation time) in an ordinary system, or store an f32 offset from an
f64 origin.

### 2.6 Loops

Every loop carries a static trip cap. `for (let i = 0; i < 8; i++)` proves its
own bound; a loop whose bound cannot be proved gets `maxLoopIterations` (default
4096, settable per kernel). A loop the parser cannot analyse at all, or one whose
proved bound exceeds the cap, is `E_UNBOUNDED_LOOP`. Both backends emit the cap,
so:

- a GPU kernel can never hang the device;
- a loop that would have run longer stops at the same iteration on both backends.

---

## 3. Semantics

### 3.1 When the dispatch happens

A kernel is an ordinary system. `kernelSystem` registers it with `world.system()`
in the given `group` and `order`, so it runs where the scheduler says it runs, in
`(order, registration)` order, inside the synchronous `world.update(dt, group)`.

The system body **encodes and submits; it never awaits**. Measured cost of the
encode is a flat **~0.013 ms regardless of entity count** — 0.013 ms at 20k, 0.016 ms
at 1M, 0.017 ms at 2M. That is the entire per-frame cost a kernel adds to
`world.update()`.

Per dispatch, for each matching non-empty chunk, the runtime:

1. ensures the device-side table buffer exists and is large enough;
2. re-uploads it **only if the CPU side changed** (see §3.2);
3. writes `dt`, `count`, `base`, `countOther`, the field base offsets and the
   user uniforms into the uniform buffer;
4. encodes `ceil(count / 256)` workgroups, splitting into several passes when
   that exceeds the device's `maxComputeWorkgroupsPerDimension` (16.7M entities
   per pass at workgroup 256);
5. for readback modes other than `'none'`, copies the byte ranges of the
   **written** fields into a staging buffer and starts `mapAsync`.

One `queue.submit` per dispatch, not one per chunk.

### 3.2 Who owns the data

**GPU residency is not an optimization, it is the precondition.** `writeBuffer`
of 4 MB costs 0.29 ms, so re-uploading four columns at 1M entities costs ~1.2 ms
against a 1.14 ms CPU loop — the GPU win is gone before the shader starts.
Readback, by contrast, is nearly free (~0.05 ms over a bare dispatch).

So between dispatches, **the GPU owns every field the kernel writes.** The
runtime re-uploads an archetype only when it can tell the CPU side changed:

- the first dispatch for that archetype;
- `chunk.buffer` identity changed (the table grew and was reallocated);
- `chunk.count` changed (rows were added or swap-removed);
- you called `handle.markCpuDirty(archetype?)`.

Any other CPU write to a kernel-written field **is lost**. If you set a position
from CPU code between frames, call `handle.markCpuDirty()` — it only sets a flag.

`handle.stats.bytesUploaded` is the diagnostic: if it keeps climbing frame after
frame, residency is broken somewhere.

### 3.3 Readback modes

| mode | when results land | `auto` default | measured break-even | use for |
|---|---|---:|---:|---|
| `'async'` (default) | **next frame** | 45k entities | 34k–64k | simulation the CPU reads a frame late |
| `'sync-frame'` | this frame, after you `await` | 700k entities | ~554k, or never up to 1M | CPU logic that must see this frame's values |
| `'none'` | never (GPU-resident) | 45k entities | 23k–49k | data consumed by rendering |

(The default is for a kernel of baseline CPU cost; `auto` scales it by the
kernel's estimated cost. The measured ranges span the two benchmark kernels on
an M4 through Dawn. See [section 4](#4-choosing-a-target).)

`'async'` is the honest default: `world.update()` cannot block, so the map
resolves on a later turn of the event loop and the values you read from
`chunk.col(Position).x` during the next frame are one frame old. For most
simulation that is invisible; when it is not, use `'sync-frame'`.

`'sync-frame'` does not make `world.update()` blocking — nothing can, WebGPU has
no synchronous map. It means *you* await before presenting the frame:

```js
world.update(dt);              // encodes every kernel, never blocks
await flushKernels(world);     // or: await move.sync()
render();                      // tables now hold this frame's values
```

If you never await, `'sync-frame'` degrades to `'async'` — correct, just later.

**Back-pressure: frames are coalesced, never lost.** Each kernel keeps at most
4 readbacks in flight. If the app dispatches faster than the GPU completes them
(many `world.update()` calls without yielding, or a very large table), a frame
that finds no free staging buffer skips its own readback and stays pending. The
table is resident on the device, so nothing is lost: when an in-flight readback
finishes, the runtime immediately issues one **catch-up readback** that copies
the full current state of every table the kernel writes, which covers every
skipped frame at once ("latest wins"). `sync()` / `flushKernels()` resolve only
once the tables reflect every dispatch submitted before the call. The first time
this happens the kernel warns once, and `stats.coalescedReadbacks` counts the
skipped or superseded readbacks.

`'none'` keeps everything on the device. `handle.bufferFor(archetype)` gives you
the `GPUBuffer`; field offsets inside it are exactly
`chunk.col(C)[field].byteOffset`, because the buffer is a byte-for-byte image of
the table. The buffer is **replaced when the archetype grows**, so re-fetch it
after structural changes, like a column typed array.

### 3.4 How a dispatch reports completion

`kernelSystem` returns a normal `SystemHandle` with these added:

```ts
handle.backend        // 'gpu' | 'cpu' | 'none' — the backend in use (see below)
handle.readback       // the mode
handle.stats          // { dispatches, completed, pending, bytesUploaded, bytesReadBack,
                      //   staleReadbacks, coalescedReadbacks, lastBackend, lastEntities }
await handle.sync()   // resolves when every dispatch submitted before the call
                      // is reflected in the archetype tables
handle.setUniform(name, value)
handle.getUniform(name)
handle.markCpuDirty(archetype?)
handle.bufferFor(archetype)
handle.destroy()      // unregister + free GPU resources; idempotent
```

Under `target: 'auto'` the backend is decided **per dispatch** from the matched
entity count. Before the first `world.update()`, `handle.backend` (and
`stats.lastBackend`) report the backend the first dispatch **will** use for the
query's current size: a kernel registered over 100 entities reports `'cpu'`,
not `'gpu'`, even when a device exists. The prediction follows spawns until the
first dispatch; after it, both report what the last dispatch actually did.

`handle.stats.pending === 0` means the tables are current. `stats.completed`
counts dispatches covered by an **applied** readback (a readback copies the full
resident state, so applying it covers every dispatch submitted before it was
encoded), plus dispatches that finish synchronously (CPU backend, `'none'`).
`sync()` resolves immediately on the CPU backend and for `'none'`. It is the
only deterministic way to observe a kernel's results — do not poll a column and
hope.

The two readback counters mean different things:

- `stats.staleReadbacks` — per-archetype readback data **discarded**: the
  archetype changed shape while the readback was in flight (buffer reallocated,
  row count changed, or `markCpuDirty`), so applying it would write rows that
  have since moved, or the map itself failed. The next dispatch re-uploads from
  the CPU tables. A handful after a spawn burst is normal; a steadily climbing
  count means the world is churning faster than the GPU frame.
- `stats.coalescedReadbacks` — readbacks **skipped or superseded** under
  back-pressure (see section 3.3). No data is lost; a later readback covers
  them. A climbing count only says the app dispatches faster than the GPU
  completes readbacks.

### 3.5 Ordering against other systems

- **GPU kernel after GPU kernel, same frame:** ordered. All dispatches go to one
  device queue in submission order, so a kernel sees the previous kernel's
  writes within the same `world.update()` even without a readback.
- **CPU system after a GPU kernel, same frame:** it sees **stale** data (last
  frame's, under `'async'`). Either move that logic into a kernel, or use
  `'sync-frame'` and `await flushKernels(world, group)` between the two groups:

  ```js
  world.update(dt, 'simulate');   // GPU kernels
  await flushKernels(world, 'simulate');
  world.update(dt, 'react');      // CPU systems that need the results
  ```
- **GPU kernel after a CPU system that wrote its fields, same frame:** call
  `handle.markCpuDirty()` from that system, otherwise the write is overwritten
  by the resident GPU copy.

### 3.6 Enableable components and disabled rows

A dispatch runs over rows `[0, count)` of a chunk. `enableable` components add a
`Uint8Array` flag column, and the v1 GPU backend addresses 4-byte views only, so
it cannot test those flags.

**If any component in the kernel's query `all` is `enableable`, the kernel runs
on the CPU backend**, with one `console.warn` at registration. The CPU backend
honours the flags exactly, skipping disabled rows the way `Query.forEach` does.
Behaviour is therefore always correct; only the backend differs. If you want a
kernel on the GPU, model the on/off state as a plain field the kernel reads
(`if (e.active > 0) { ... }`) instead of an enableable component.

### 3.7 Structural changes

A kernel cannot spawn, destroy, add or remove — the subset has no way to express
it, so there is nothing to defer and no command buffer involvement. Do structural
work in an ordinary system before or after the kernel's group.

### 3.8 Pairwise kernels (experimental)

```js
const gravity = await kernelSystem(world, 'NBody', {
  components: [Position, Velocity, Mass],
  pairwise: true,
  uniforms: { g: 6.674e-11, eps: 1e-3 },
  kernel: (self, other, dt, u) => {
    const dx = other.x - self.x;
    const dy = other.y - self.y;
    const r2 = dx * dx + dy * dy + u.eps;
    const f = (u.g * other.m) / (r2 * Math.sqrt(r2));
    self.vx += f * dx * dt;
    self.vy += f * dy * dt;
  },
});
```

**This is O(n²).** 10k entities is 100M pair evaluations per frame — fine on a
GPU, ruinous on a CPU. Above ~50k entities you want a spatial structure, which
this module does not provide.

Two v1 restrictions:

- **Pairs are formed within a chunk**, i.e. within an archetype. Entities in
  different archetypes never see each other. Keep the interacting bodies in one
  archetype (same component set) or the simulation is wrong, not just partial.
- **Writes to `other` are rejected.** Every pair is evaluated by both
  participants, so a write to `other` would race. Accumulate into `self` only;
  the symmetric half happens when the roles swap.

---

## 4. Choosing a target

`target: 'auto'` (the default) picks per dispatch, from the total matched entity
count. The raw costs below were measured on an Apple M4 (Metal 3, macOS 26.5) with a
12-op gravity+integrate kernel:

| N | CPU | GPU encode only | GPU encode + sync |
|---:|---:|---:|---:|
| 20k | 0.023 ms | 0.013 ms | 0.209 ms |
| 100k | 0.115 ms | 0.012 ms | 0.244 ms |
| 200k | 0.446 ms | 0.013 ms | 0.291 ms |
| 1M | 1.140 ms | 0.016 ms | 0.504 ms |
| 2M | 2.300 ms | 0.017 ms | 0.805 ms |

Fixed overhead floor: an empty dispatch plus sync is 0.154 ms.

Those isolate the GPU work. The defaults come from the **full per-frame
comparison** in `benchmarks/gpu.js` (Dawn through the `webgpu` package on the
same M4; `world.update()` plus one event-loop yield, queued work drained before
the clock stops), against the CPU backend (`kernel-cpu`):

| kernel | `'none'` | `'async'` | `'sync-frame'` |
|---|---:|---:|---:|
| simple (2 adds) | ~49k (round 1: ~65k) | ~64k (~74k) | > 1M (never wins up to 1M) |
| gravity (integrate + bounce) | ~23k (~26k) | ~34k (~32k) | ~554k (~530k) |

Round 2 (2026-09-18) is the current build, median of 5 isolated processes,
every GPU cell checked row by row against the CPU backend. Break-evens are
interpolated log-linearly between the measured sizes (1k, 10k, 30k, 50k, 100k,
300k, 1M), so treat them as rough. Full tables:
[benchmarks/RESULTS.md, "GPU / kernel"](../benchmarks/RESULTS.md#gpu--kernel).

Selected ms/frame (round 2; the machine was loaded, so compare ratios):

| kernel | N | `kernel-cpu` | bitecs | GPU `'none'` | GPU `'async'` | GPU `'sync-frame'` |
|---|---:|---:|---:|---:|---:|---:|
| simple | 10k | 0.011 | 0.015 | 0.057 | 0.073 | 0.365 |
| simple | 100k | 0.149 | 0.149 | 0.057 | 0.091 | 0.455 |
| simple | 1M | 1.80 | 2.12 | 0.350 | 1.18 | 2.06 |
| gravity | 10k | 0.025 | 0.028 | 0.057 | 0.083 | 0.350 |
| gravity | 100k | 0.271 | 0.282 | 0.058 | 0.111 | 0.450 |
| gravity | 1M | 2.70 | 3.12 | 0.364 | 1.40 | 2.30 |

A GPU frame has a fixed floor, flat from 1k to 50k entities: ~0.057 ms for
`'none'`, ~0.07-0.08 ms for `'async'` and ~0.35 ms for `'sync-frame'` (the
readback round trip) in round 2; ~0.04 / ~0.04 / ~0.22 ms on the less loaded
round-1 run. `'async'` at 1M is much slower than `'none'` because every frame's
results really are copied back (and the 1M `'async'` cells spread 0.5-1.5 ms
across repeats, depending on how many readbacks coalesce).
Multiplying each break-even by the kernel's CPU cost gives ~50-68 µs per frame
for both kernels in fire-and-forget modes, so one constant scaled by kernel cost
fits. At the baseline 1.14 ns/entity:

```
readback 'none' | 'async'   ->  GPU above ~45 000 entities
readback 'sync-frame'       ->  GPU above ~700 000 entities
```

Those are `DEFAULT_AUTO_THRESHOLDS`. They encode a simple cost model:

```
GPU wins  <=>  entities × cpuNsPerEntity  >=  GPU_FIXED_OVERHEAD_NS[mode]

GPU_FIXED_OVERHEAD_NS.fireAndForget = 45 000 × 1.14 ns ≈ 51 µs    ('async' / 'none')
GPU_FIXED_OVERHEAD_NS.synchronous   = 700 000 × 1.14 ns ≈ 798 µs   ('sync-frame')
BASELINE_CPU_NS_PER_ENTITY          = 1.14 ns  (the 12-op kernel above, 1.140 ms at 1M)
```

`cpuNsPerEntity` is `estimateCPUNanosPerEntity(ir)` from the CPU backend,
computed once per kernel as `0.52 + 0.065 × opCount` ns: a fixed per-entity
cost (loop, index, memory traffic) plus a per-op cost, fitted to the measured
`simple` and `gravity` break-evens below (an op-count proxy, `1.14 × opCount / 12`,
is used if the estimator is unavailable; `cpuCostEstimate(ir)` exposes the value). So the
effective threshold is
`threshold × clamp(BASELINE_CPU_NS_PER_ENTITY / cpuNsPerEntity, 1/8, 4)`: a
kernel estimated at 4× the baseline cost switches to the GPU at ~11k entities
instead of 45k. Hysteresis (×1.25 / ÷1.25) keeps a count oscillating around the
threshold from flapping between backends. `autoPrefersGPU(ir, entities,
readback, thresholds, currentlyGPU, cpuNs?)` is exported so you can check a
decision without dispatching. The GPU's own per-entity cost is folded into the
fixed overhead.

They are **one machine's numbers**. An app shipping to many devices should
measure on the device at startup with `calibrateAuto`, before creating its
kernels:

```js
import * as cozy from 'cozyecs';
import { calibrateAuto, kernelSystem } from 'cozyecs/gpu';

const cal = await calibrateAuto(cozy);   // ~0.7 s; null (and no change) without a GPU
// cal.thresholds -> e.g. { fireAndForget: 24_963, synchronous: 226_721 }, already applied
```

It runs a branch-free probe kernel on a throwaway world at two sizes, on the CPU
backend and on the GPU in every readback mode. Each GPU timing waits for the
work to **finish** (`queue.onSubmittedWorkDone()` for `'none'`, a drain for
`'async'`, `sync()` for `'sync-frame'`), so it measures GPU time, not
submission. Each backend is fitted as `ms/frame = fixed + perEntity × n`, the
break-evens are where the lines cross, and the result is normalized to a
baseline-cost kernel and passed to `setAutoThresholds` (skip that with
`{ apply: false }`). It affects kernels created afterwards. The core module is
passed in because `cozyecs/gpu` imports the core for types only: a second
bundled copy would also carry a second component-id counter.

Expect roughly ±30% between runs on a busy machine; cache the result (e.g. in
`localStorage`) and pass it to `setAutoThresholds` on later launches. Measured
on the same M4: `fireAndForget` ~18k-27k in Node (Dawn) and ~25k in Chrome 152,
against the 45k built-in default.

You can also set thresholds by hand, globally or per kernel:

```js
import { setAutoThresholds } from 'cozyecs/gpu';
setAutoThresholds({ fireAndForget: 50_000, synchronous: 400_000 });
```

or per kernel with `thresholds: { ... }`.

**Branches over real data cost the CPU more than the benchmark shows.** The
benchmark gives every entity the same values, so a branch such as the gravity
kernel's bounce is perfectly predictable. With varied data (heights spread
over a range) the same kernel measured ~5.7 ns/entity on the CPU instead of
~2.7, because of branch mispredictions; the GPU does not care. The estimator
counts ops and cannot see this, so a branchy kernel over irregular data
switches to the GPU later than it should. If that is your case, lower that
kernel's `thresholds`, or pin `target: 'gpu'` above a few tens of thousands of
entities.

**Calibration.** With that estimator the effective thresholds for the two
benchmark kernels are:

| kernel | estimate | `'none'` / `'async'` threshold | measured `'none'` / `'async'` | `'sync-frame'` threshold | measured |
|---|---:|---:|---:|---:|---:|
| simple (6 ops) | 0.91 ns | ~56k | ~49k / ~64k | ~877k | never, up to 1M |
| gravity (20 ops) | 1.82 ns | ~28k | ~23k / ~34k | ~438k | ~554k |

Every switch point is within ~0.8-1.25x of the measured break-even. The one
remaining miss is small: a `simple`-sized kernel under `'sync-frame'` moves to
the GPU from ~877k entities, where the GPU is still ~13% slower than the CPU at
1M (2.06 vs 1.80 ms). An earlier estimator (`0.095 × opCount`, no fixed term)
rated `simple` 3.3× cheaper than `gravity` against a measured ~2×, and kept
`simple` on the CPU up to ~90k entities under `'none'`; that is fixed.

`workgroup_size` is **256** for every kernel: 64, 128 and 256 measured within
noise of each other (0.54–0.65 ms at 1M), 32 was clearly slower, and 256 is the
maximum the WebGPU *default* limits grant, so no raised limits are needed.

---

## 5. Falling back to the CPU

`kernelSystem` **never throws for a capability reason.** Each of these produces
exactly one `console.warn` and the CPU backend:

- no WebGPU device (no `navigator.gpu`, no `setGPUProvider`, adapter or device
  request failed);
- the kernel touches a field the GPU backend cannot address: `f64` (8 bytes),
  `i8`/`i16`/`u8`/`u16`/`bool` (sub-word), `str` (an interned id, meaningless on
  a GPU);
- the query has an `enableable` component in `all` (§3.6);
- a device limit the kernel cannot fit in.

The CPU backend is not a consolation prize. It compiles the same IR into a flat
chunk loop passed to `Query.forEachChunk`, which — for a callback with stable
identity, which this always is — compiles a per-chunk trampoline where every
column typed array is a closure **constant**. That is the fastest iteration
CozyECS has (measured 262k → 382k ops/sec on a 4 × 1000-row `Position +=
Velocity` loop versus a hand-written chunk loop). Below the break-even it is
also faster than the GPU.

Under a CSP that blocks `new Function`, the CPU backend falls back again to a
closure-tree evaluator: correct, roughly 5–10× slower, still no throw.

### When neither backend is usable: `backend === 'none'`

If the CPU loop itself fails to compile (a library bug, not a capability
problem) and there is no GPU pipeline to fall back to — or `target` is
`'cpu'` — the kernel does **not** silently stop. `kernelSystem` still resolves
(it never throws for this), prints exactly one warning that includes the
compiler's error, and reports it through the handle:

```js
const h = await kernelSystem(world, 'Move', { ... });
if (h.backend === 'none') { /* every dispatch is a no-op; see the warning */ }
h.stats.lastBackend;   // also 'none'
```

If the CPU loop fails but a GPU pipeline exists, the kernel runs on the GPU
only (`target: 'auto'` is pinned to `'gpu'`), with one warning that there is no
fallback if the device is later lost.

---

## 6. CPU/GPU parity contract

**Bit-exactness is not guaranteed.** The two backends are held to this instead:

| what | guarantee |
|---|---|
| `rand(seed)` | **bit-identical**. Pure `u32` integer hash, same constants in WGSL and JS. |
| integer field writes | identical: `floor(x + 0.5)`, clamped to the field's range. |
| `Math.round` | identical: `floor(x + 0.5)` on both (not `Math.round`). |
| arithmetic-only kernels (`+ - * / %`, comparisons, `min`/`max`/`abs`/`floor`/`ceil`/`sign`) | relative error ≤ **1e-6**, usually exact. |
| `sqrt`, `pow`, `exp`, `log`, `hypot` | relative error ≤ **1e-5**. |
| trigonometric functions | relative error ≤ **1e-4** for arguments in ±100; unbounded beyond that — WGSL's accuracy requirements degrade with magnitude. Range-reduce yourself if it matters. |
| `NaN` / `Infinity` | **undefined**. WGSL implementations may assume they do not occur. Do not rely on NaN propagation, and guard divisions. |
| order of operations | as written. Neither backend reassociates, but the GPU may fuse a multiply-add, which costs ≤ 1 ulp. |

Why it cannot be exact: the CPU backend rounds every intermediate through
`Math.fround`, so it computes in f32 like the shader, but a GPU may contract
`a * b + c` into an FMA (one rounding instead of two) and its transcendental
functions are only required to be accurate to a tabulated ULP count, which
differs per vendor.

**Practical rule:** a kernel that must give identical results everywhere (lockstep
netcode, replays) should pin `target: 'cpu'`. Anything where a 1e-5 drift per
frame is invisible — which is nearly all rendering and simulation — can use
`'auto'`.

Tests assert parity at these tolerances by running the same IR through both
backends over the same data. `__tests__/gpu-dawn.test.ts` fuzzes 200
deterministic random kernels (arithmetic, `min`/`max`/`abs`/`sqrt`, guarded
division, locals, compound assignment) over up to 700 rows split across two
archetypes. Each is run on the GPU (Dawn, via the `webgpu` package) and on the
CPU backend, and both are compared at 1e-5 relative (floor 1) to an f32 oracle,
which is the same kernel with every operation wrapped in `Math.fround`. At
tolerance 0 the same run flags ~490 rows (1-ulp `sqrt`/FMA differences), so the
comparison is not vacuous. The suite returns early with a note when the
`webgpu` package is missing or yields no adapter.

---

## 7. Error catalogue

Every registration-time rejection is a `KernelError` with a `.code` from this
table, a code frame pointing into the kernel source, and a hint.

```js
import { KernelError } from 'cozyecs/gpu';
try { await kernelSystem(world, 'Move', { ... }); }
catch (e) { if (e instanceof KernelError && e.code === 'E_UNKNOWN_IDENTIFIER') { ... } }
```

| code | meaning |
|---|---|
| `E_NOT_A_FUNCTION` | `kernel` is not a function, or is native/bound so its source cannot be read. |
| `E_PARAM_COUNT` | Parameter count is not `components.length`, `+1` or `+2` (2–4 for pairwise), or a parameter uses destructuring, a default value or rest syntax. |
| `E_KERNEL_ASYNC` | The kernel is `async` or a generator. |
| `E_SYNTAX` | The body could not be parsed: a token the subset does not know (a regexp literal, for example), or a `let`/`const` without an initializer. |
| `E_UNKNOWN_IDENTIFIER` | A free identifier that is not a parameter, a local, `Math` or a builtin — **most often a closure variable that should be a uniform**. |
| `E_UNSUPPORTED_LITERAL` | String or template literal (also the `'x'` in `'x' in p`), object/array literal, or `new`. |
| `E_UNSUPPORTED_STATEMENT` | `try`, `switch`, `do`/`while`, `return`, labeled `break`/`continue`, … |
| `E_UNSUPPORTED_EXPRESSION` | Bit operators, `++`/`--`, comma, optional chaining, `in`, `instanceof`, … |
| `E_UNSUPPORTED_CALL` | A call that is not a listed builtin, including `Math.random` and your own functions. |
| `E_UNSUPPORTED_MEMBER` | A property access that is not `<param>.<field>` or `u.<name>`. |
| `E_UNBOUNDED_LOOP` | A loop the parser cannot analyse at all (no test, or a counter that does not advance), or one whose proved bound exceeds `maxLoopIterations`. |
| `E_ASSIGN_TO_CONST` | Assignment to a `const` local, to `dt`, or to a uniform. |
| `E_BAD_LOCAL` | A local that shadows a kernel parameter (including `dt` and `u`), or is declared twice in the same scope. A local used before its declaration is `E_UNKNOWN_IDENTIFIER`. |
| `E_UNKNOWN_FIELD` | `p.zz` where the component has no field `zz`. |
| `E_UNKNOWN_UNIFORM` | `u.g` with no `g` in `uniforms` (also thrown by `setUniform`). |
| `E_FIELD_COLLISION` | Two components of a pairwise kernel expose the same field name. |
| `E_WRITE_NOT_DECLARED` | The declared `write` list omits a component the kernel assigns to. |
| `E_BAD_UNIFORM_VALUE` | A uniform value is not a finite number. |
| `E_INVALID_IR` | The IR failed validation — a bug in this module, or a hand-built IR. |
| `E_DEVICE_LIMIT` | The kernel needs more of the device than it grants. Reported as a warning + CPU fallback, not a throw, when it comes from capability detection. |
| `E_NO_CODEGEN` | `new Function` is unavailable (CSP). Never surfaces as a throw: the CPU backend degrades to an evaluator. |

Example:

```
[cozyecs/gpu] E_UNKNOWN_IDENTIFIER in kernel "Move": unknown identifier "gravity"

  1 | (p, v, dt) => {
> 2 |   v.y += gravity * dt;
    |          ^^^^^^^
  3 | }

Hint: Values from outside the kernel must come through `uniforms`: register with uniforms: { gravity: <value> } and read it as `u.gravity`. A closure variable is invisible to Function.prototype.toString, so it cannot be captured.
```

---

## 8. Device limits

Measured on the reference machine (Apple M4, Metal 3, macOS 26.5.2) with the
`webgpu` Dawn bindings. **A device is granted only the limits it asks for**: the
column that matters is the granted one, not the adapter maximum. `cozyecs/gpu`
requests the adapter's maxima at device creation and reports what was granted as
`GPUCapabilities`; codegen budgets against that.

| limit | spec default | M4 adapter max | used by |
|---|---:|---:|---|
| `maxStorageBuffersPerShaderStage` | 8 | 10 | ≤ 3 needed (one per view type) |
| `maxComputeWorkgroupSizeX` | 256 | 1024 | 256 |
| `maxComputeInvocationsPerWorkgroup` | 256 | 1024 | 256 |
| `maxBufferSize` | 256 MB | 4 GB − 1 | one archetype table |
| `maxStorageBufferBindingSize` | 128 MB | 4 GB − 4 | one archetype table |
| `maxComputeWorkgroupStorageSize` | 16 KB | 32 KB | unused (no workgroup memory) |
| `maxComputeWorkgroupsPerDimension` | 65 535 | 65 535 | 16.7M entities per pass |
| `min{Storage,Uniform}BufferOffsetAlignment` | 256 | 256 | why offsets live in the uniform |

Features requested when offered: `subgroups`, `subgroup-size-control`,
`timestamp-query` (benchmarks), `shader-f16` (unused in v1). Subgroup size on the
M4 is fixed at 32.

At 26.2 bytes per entity, a 128 MB storage binding holds roughly 5M entities of a
typical archetype in one table; beyond that the world will have split into
several archetypes anyway, and each is dispatched separately.

### Host hazards this module handles for you

1. **Dawn segfaults if the `GPU` instance is garbage-collected** — measured 12/40
   runs crashing without a keepalive, 0/40 with one. `src/gpu/device.ts` pins
   gpu/adapter/device in an array that stays reachable at teardown. If you write
   a benchmark or test that touches WebGPU outside this module, pin them too.
2. **`queue.writeBuffer()` with a `SharedArrayBuffer`-backed view is a hard
   crash** (SIGSEGV, 100% reproducible). `WorldOptions.shared` makes every table
   a SAB, so the upload path copies into a plain `ArrayBuffer` first. Shared
   worlds work; they just pay one memcpy per upload.
3. **Default limits are far below the adapter's** — see the table above.

---

## 9. Reference

```ts
kernelSystem(world, name, {
  components,          // ComponentType[]  — kernel parameters, in order (required)
  kernel,              // (…) => void      — the kernel (required)
  write?,              // ComponentType[]  — must cover every assigned component
  uniforms?,           // Record<string, number>
  target?,             // 'gpu' | 'cpu' | 'auto'          (default 'auto')
  readback?,           // 'async' | 'sync-frame' | 'none' (default 'async')
  group?, order?,      // scheduler placement, like any system
  query?,              // extra { all?, any?, none? } filters
  pairwise?,           // boolean (experimental)
  thresholds?,         // { fireAndForget?, synchronous? }
  workgroupSize?,      // rarely useful
  maxLoopIterations?,  // loop trip cap, default 4096 (§2.6)
}): Promise<KernelSystemHandle>

// KernelSystemHandle = SystemHandle & {
//   ir, readback, stats,
//   backend: 'gpu' | 'cpu' | 'none',   // 'none': no usable backend (§5)
//   sync(), setUniform(name, v), getUniform(name),
//   markCpuDirty(archetype?), bufferFor(archetype), destroy() }

flushKernels(world, group?): Promise<void>
setAutoThresholds({ fireAndForget?, synchronous? }): void
getAutoThresholds(): AutoThresholds  // what kernels created from now on will use
calibrateAuto(core, { sizes?, frames?, repeats?, apply? }?): Promise<CalibrationResult | null>
autoPrefersGPU(ir, entities, readback, thresholds, currentlyGPU, cpuNs?): boolean
cpuCostEstimate(ir): number          // ns/entity used by 'auto'
DEFAULT_AUTO_THRESHOLDS, GPU_FIXED_OVERHEAD_NS, BASELINE_CPU_NS_PER_ENTITY
setGPUProvider(gpu): void
getGPUContext(): Promise<GPUContext | null>
hasGPUContext(): boolean
peekGPUContext(): GPUContext | null
KernelError                          // .code is one of the codes in §7
describeIR(ir), IR_VERSION, BUILTINS, WORKGROUP_SIZE   // introspection
```

`kernelSystem` is asynchronous because acquiring a device is. Register kernels
during startup, before the first `world.update()`.

---

## 10. Running the GPU benchmarks

`benchmarks/gpu.js` compares, per kernel and entity count: `kernelSystem` on the
CPU backend (`kernel-cpu`), a hand-written `forEachChunk`, a plain chunk loop,
bitecs, and the GPU backend in all three readback modes. It needs a build and,
for the GPU columns, the `webgpu` package (a dev dependency of this repo):

```
npm run build
npm run benchmark:gpu                                  # everything (2 kernels x 7 variants x 7 sizes x 5)
node benchmarks/gpu.js --kernels=simple --sizes=10000,100000 --repeats=3
node benchmarks/gpu.js --no-gpu                        # CPU variants only
node benchmarks/gpu.js --write                         # replace the section in RESULTS.md
```

| option | default | meaning |
|---|---|---|
| `--kernels=` | `simple,gravity` | which kernels |
| `--variants=` | all | e.g. `kernel-cpu,gpu-none` |
| `--sizes=` | `1000,10000,30000,50000,100000,300000,1000000` | entity counts |
| `--repeats=` | 5 | isolated processes per cell; the median is reported |
| `--time=` / `--warmup=` | 600 / 200 | ms measured / warmed up per process |
| `--job-timeout=` | 180 | seconds before a process is killed |
| `--no-gpu` | | skip the `gpu-*` variants |
| `--json=path` / `--from-json=a,b` | | write raw results / render earlier runs without measuring |
| `--write` | | replace the `GPU / kernel` section of `benchmarks/RESULTS.md` |

Every cell is gated on correctness: CPU cells are checked against an f32
reference, and every GPU cell is drained with `flushKernels` and compared row
by row with the CPU backend run for the same number of frames (`'none'` is read
straight out of `bufferFor`). A cell that differs prints `FAIL` instead of a
time. The script also prints the break-even table and a `BREAK_EVEN {...}` JSON
line. To calibrate `auto` for your own hardware, run it on the target machine
and pass the break-evens to `setAutoThresholds` (scaled to the baseline cost as
in [section 4](#4-choosing-a-target)).
