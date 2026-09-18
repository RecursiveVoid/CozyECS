<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/RecursiveVoid/CozyECS/main/docs/assets/hero-dark.svg">
    <img alt="CozyECS: a tiny, archetype-based ECS for JavaScript with GPU kernels. #1 in all 5 benchmark scenarios, 26 bytes per entity, 7.2x faster on the GPU at 1M entities, zero dependencies." src="https://raw.githubusercontent.com/RecursiveVoid/CozyECS/main/docs/assets/hero-light.svg" width="100%">
  </picture>
</p>

<p align="center">
  <a href="https://www.npmjs.com/package/cozyecs"><img alt="npm" src="https://img.shields.io/npm/v/cozyecs?color=2a78d6"></a>
  <a href="https://github.com/RecursiveVoid/CozyECS/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/RecursiveVoid/CozyECS/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="gzipped core" src="https://img.shields.io/badge/core-12%20KB%20gzipped-1baf7a">
  <img alt="dependencies" src="https://img.shields.io/badge/dependencies-0-1baf7a">
  <img alt="TypeScript" src="https://img.shields.io/badge/types-included-2a78d6">
  <a href="LICENSE"><img alt="MIT license" src="https://img.shields.io/badge/license-MIT-lightgrey"></a>
</p>

<p align="center"><b><a href="https://recursivevoid.github.io/CozyECS/">▶ Live demo: a million particles running on your GPU</a></b></p>

**CozyECS** is an Entity Component System for TypeScript and JavaScript. It stores
components in packed typed arrays grouped by archetype, runs your systems over flat memory,
and can compile a system written as a plain JavaScript function into a WebGPU compute shader.

- 🏎️ **Fast.** #1 in all five scenarios of the standard ECS benchmark against nine other libraries,
  including bitecs, wolf-ecs and harmony-ecs. Adding, removing and spawning are 1.3–3.4× faster than
  the next best.
- 🪶 **Small.** 26 bytes per entity for Position + Velocity (2–37× less than the others), a 12 KB
  gzipped core, and no runtime dependencies.
- 🎮 **GPU kernels.** Write `(p, v, dt) => { p.x += v.dx * dt }`. CozyECS turns it into WGSL, runs it
  on WebGPU, and falls back to a compiled CPU loop when there is no GPU.
- 🧩 **Typed.** Component schemas give typed columns: `chunk.col(Position).x` is a `Float32Array`.

## Install

```bash
npm install cozyecs
```

ESM, CommonJS and type declarations are included. The GPU module is a separate entry point,
`cozyecs/gpu`, so the core bundle never contains GPU code.

## Quick start

```ts
import { World, component, tag, f32 } from 'cozyecs';

// 1. Components: a schema of typed fields.
const Position = component({ x: f32, y: f32 }, { name: 'Position' });
const Velocity = component({ dx: f32, dy: f32 }, { name: 'Velocity' });
const Player = tag({ name: 'Player' });

// 2. A world and an entity.
const world = new World();
const player = world.spawn(world.archetype(Position, Velocity, Player));
world.set(player, Velocity, { dx: 1, dy: 2 });

// 3. A system: a query plus a function that loops over matching chunks.
world.system('move', { query: { all: [Position, Velocity] } }, (q, dt) => {
  const chunks = q.chunks;
  for (let c = 0; c < chunks.length; c++) {
    const chunk = chunks[c];
    const p = chunk.col(Position), v = chunk.col(Velocity);
    const x = p.x, y = p.y, dx = v.dx, dy = v.dy;
    for (let i = 0, n = chunk.count; i < n; i++) {
      x[i] += dx[i] * dt;
      y[i] += dy[i] * dt;
    }
  }
});

// 4. Run a tick.
world.update(0.5);
console.log(world.get(player, Position)); // { x: 0.5, y: 1 }
```

Class-based systems, `forEach`, enableable components, string fields, events and system groups
are covered in the [guide](docs/GUIDE.md).

## Systems on the GPU

Write the system once as a plain function. CozyECS parses it, checks it against a small, safe
subset of JavaScript, and compiles it twice: to a **WGSL compute shader** and to a **compiled CPU
loop**. You never write shader code.

```js
import * as cozy from 'cozyecs';
import { kernelSystem, calibrateAuto } from 'cozyecs/gpu';

await calibrateAuto(cozy); // optional: measure where the GPU wins on this device (~0.7 s)

const fall = await kernelSystem(world, 'Fall', {
  components: [Position, Velocity],  // kernel parameters, matched by position
  uniforms: { gravity: -9.8 },       // outside values come in through `u`
  target: 'auto',                    // 'gpu' | 'cpu' | 'auto'
  readback: 'async',                 // 'async' | 'sync-frame' | 'none'
  kernel: (p, v, dt, u) => {
    v.dy += u.gravity * dt;
    p.x += v.dx * dt;
    p.y += v.dy * dt;
    if (p.y < 0) { p.y = 0; v.dy = -v.dy * 0.5; }
  },
});

world.update(1 / 60); // dispatches with the other systems and never blocks
```

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/RecursiveVoid/CozyECS/main/docs/assets/gpu-dark.svg">
  <img alt="Milliseconds per frame for the same kernel on CPU and GPU from 1k to 1M entities. The GPU with readback 'none' wins above about 22k entities and is 7.2 times faster than the CPU at 1M entities (0.374 ms vs 2.69 ms)." src="https://raw.githubusercontent.com/RecursiveVoid/CozyECS/main/docs/assets/gpu-light.svg" width="100%">
</picture>

- **`target: 'auto'`** keeps small queries on the CPU and moves large ones to the GPU. The switch
  point comes from the kernel's estimated cost, and `calibrateAuto()` measures it on the user's device.
- **The CPU backend is a speedup on its own.** `target: 'cpu'` matches or beats bitecs (1.0–1.3×).
- **Never throws for a missing GPU.** Without WebGPU, the kernel runs on the CPU and you get one warning.
- **Tested for parity** in Node (Dawn over Metal) and in Chrome's WebGPU: CPU and GPU agree to f32 rounding.
- In Node, install the `webgpu` package and pass it in with `setGPUProvider(create([]))`. With
  TypeScript, add `@webgpu/types` as a dev dependency for the WebGPU types.

**[Try it live](https://recursivevoid.github.io/CozyECS/):** 10k to 1M particles as entities, one kernel,
and a CPU/GPU switch, with a live view of the archetype's memory layout, the systems running each tick,
entity memory, and five tracked entities whose values are read back from GPU memory. At 1M particles on an Apple M4, the CPU backend uses ~15–20 ms of main-thread
time per frame; the GPU backend uses ~0.04 ms and draws straight from the kernel's buffer.

The GPU module is **experimental**: its API may change in a minor release. The subset grammar,
error codes, readback modes and break-even tables are in [docs/GPU.md](docs/GPU.md).

## Benchmarks

The five scenarios of the [ddmills/js-ecs-benchmarks](https://github.com/ddmills/js-ecs-benchmarks)
suite, each run in its own Node process, 5 shuffled repeats, median shown. Every competitor's
adapter was reviewed in a separate fairness pass, and every run checks its results for correctness.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/RecursiveVoid/CozyECS/main/docs/assets/throughput-dark.svg">
  <img alt="Throughput relative to CozyECS in five scenarios. CozyECS is 1.10x the best competitor in packed_5, 1.20x in simple_iter, 1.13x in frag_iter, 3.41x in entity_cycle and 1.28x in add_remove." src="https://raw.githubusercontent.com/RecursiveVoid/CozyECS/main/docs/assets/throughput-light.svg" width="100%">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/RecursiveVoid/CozyECS/main/docs/assets/memory-dark.svg">
  <img alt="Memory per entity for 100,000 entities: CozyECS 26.2 bytes, harmony-ecs 54.2, wolf-ecs 72.4, becsy 170, bitecs 0.4 253, bitecs 0.3 302.1, geotic 398.9, ecsy 978.7." src="https://raw.githubusercontent.com/RecursiveVoid/CozyECS/main/docs/assets/memory-light.svg" width="100%">
</picture>

**Where the margins are narrow:** the iteration leads (1.1–1.2×) are close to run-to-run noise.
In `packed_5`, harmony-ecs won one of the five repeats. Not every CozyECS API wins every
scenario: the fastest forms are systems, `forEachChunk` and CPU kernels. The full tables, the
methodology, the fairness fixes and every case where CozyECS is not first are in
[benchmarks/RESULTS.md](benchmarks/RESULTS.md).

```bash
npm run build && npm run benchmark -- --paired --repeats=5
```

## How it works

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/RecursiveVoid/CozyECS/main/docs/assets/how-it-works-dark.svg">
  <img alt="Left: entities with the same components share a table whose fields are typed-array views over one ArrayBuffer. Right: a kernel function is parsed into an IR and compiled to a WGSL compute shader and to a compiled JavaScript loop." src="https://raw.githubusercontent.com/RecursiveVoid/CozyECS/main/docs/assets/how-it-works-light.svg" width="100%">
</picture>

- **Archetype tables.** Each set of components gets its own table, and each field is a typed array
  over that table's single `ArrayBuffer`. Iterating a query means walking a few dense arrays.
- **Entity handles** are plain numbers with a generation counter, so stale handles are detected.
- **Cached queries** remember their matching archetypes. A new archetype updates them once, not every tick.
- **Deferred structural changes.** Spawning, destroying and adding or removing components during a
  system are queued and applied in a batch after it.
- **Compiled loops.** `forEachChunk` and CPU kernels generate a loop in which V8 treats the column
  arrays as constants.

## Documentation

| | |
|---|---|
| [Guide](docs/GUIDE.md) | Concepts, the full API reference, performance tips and limits |
| [Queries and systems](docs/API.md) | Chunks, the three iteration styles and how `forEach` is optimized |
| [GPU kernels](docs/GPU.md) | The kernel subset, readback modes, choosing a target, error codes |
| [Internals](docs/INTERNALS.md) | Storage layout, entity handles, the command buffer, compiled loops |
| [Benchmark results](benchmarks/RESULTS.md) | Every table, the methodology and the fairness audit |

## Development

```bash
npm ci
npm run typecheck
npm test               # 387 tests: unit, reference-model fuzz, GPU parser, codegen and parity
npm run build
npm run benchmark      # CPU suite vs 13 library variants
npm run benchmark:gpu  # CPU vs GPU from 1k to 1M entities
```

The README graphics are generated from the benchmark numbers by `node docs/assets/generate.mjs`.
The live demo lives in `demo/` and is deployed to GitHub Pages by `.github/workflows/pages.yml`.

## License

MIT © M. Ergin Turk
