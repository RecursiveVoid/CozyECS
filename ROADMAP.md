# Roadmap

What 1.0 means here: **the API is frozen**, not "enough features shipped". The gate is a real
consumer — the engine being built on CozyECS — surviving a release without needing breaking
changes. Everything below is sequenced around that.

Dates are deliberately absent. The order is the commitment; the pace follows the engine.

| Milestone | Theme | Status |
|---|---|---|
| [0.2](#02--memory-and-lifecycle) | Memory and lifecycle | released |
| [0.3](#03--persistence-and-structure) | Persistence and structure | next |
| [0.4](#04--reactivity-and-scheduling) | Reactivity and scheduling | planned |
| [0.5](#05--parallelism) | Parallelism | planned |
| [1.0](#10--the-freeze) | The freeze | gated on the engine |

---

## 0.2 — Memory and lifecycle

Released. `world.memory()`, `world.compact()`, `world.clear()`, `world.dispose()`,
`Archetype.shrinkToFit()`, `handle.releaseUnused()`, `handle.memory()`, `gpuDeviceMemory()`,
plus leak and allocation regression suites.

The design rule this established, and the one the rest of the roadmap follows: **the ECS provides
mechanism, the engine owns policy.** No automatic collection, no hidden work on a tick.

## 0.3 — Persistence and structure

The two things an engine needs before anything else.

- **Serialization / snapshots.** Save and restore a world, or one archetype, as a binary blob.
  The storage layout makes this close to a buffer copy: a table is already flat. Needs a stable
  component-id mapping in the file (ids are process-global, so a snapshot must carry its own
  table), versioning, and a documented format. Unlocks save games, networking, editor undo and
  hot reload.
- **Relations (`cozyecs/relations`).** `ChildOf`-style links with cascade destroy, as an optional
  entry point rather than core. Parent/child transforms are the first thing a scene graph wants,
  and entity-to-entity references are currently just numbers the ECS does not understand.
- **Prefabs / blueprints.** Spawn templates with default values, built on the archetype path so a
  template spawn is still one `spawnMany`.

## 0.4 — Reactivity and scheduling

- **Change detection.** Per-component change ticks so a query can ask "what changed since last
  frame". Drives dirty-flag rendering and network deltas. Must be opt-in per component: it costs
  a write barrier, and the whole point of this library is that the hot loop stays bare.
- **System dependencies.** Systems currently run in registration order within a group. Add
  explicit `after`/`before` ordering and declared read/write sets — the latter is also the
  prerequisite for 0.5.
- **Singletons / resources.** World-level data (time, input, config) without faking a one-entity
  archetype.
- **`onSet` observers**, once change detection exists.

## 0.5 — Parallelism

- **Worker-parallel systems.** Run systems with disjoint write sets on several threads over
  `SharedArrayBuffer` storage. The storage half already exists (`new World({ shared: true })`,
  verified through compaction); nothing uses it yet. On a 10-core machine this is a multiple, not
  a percentage, and combined with GPU kernels it is the thing no other JavaScript ECS offers.
- **GPU entry point decision.** Either freeze `cozyecs/gpu`'s API and promote it to stable, or
  split it into its own `0.x` package so an experimental module stops gating the core's version.

## 1.0 — The freeze

Ships when the engine has shipped something real on 0.5, plus:

- A written **API stability and deprecation policy**.
- A **CHANGELOG** covering every release.
- A **browser matrix**: Safari and Firefox WebGPU are untested today — only Chrome and Dawn
  (Node) are verified.
- A **bundle-size budget** enforced in CI (the core is 12 KB gzipped; that number is a promise).
- A **CLA** before the first outside pull request is merged, so relicensing stays possible.

## Not planned

Deliberate omissions, so nobody has to ask twice:

- **Automatic garbage collection or heuristic shrinking.** Predictable frame times are the point;
  reclamation stays explicit.
- **`component.destroy()`.** Component ids are process-global and baked into archetype masks;
  freeing one individually would corrupt existing masks. `world.dispose()` is the reclaim call.
- **Scene graphs, asset lifetimes, TTL components.** Engine concerns. The ECS provides
  `onAdd`/`onRemove` as the handoff.
- **A tracing collector over entity references.** It would mean a second GC inside a runtime that
  already has one, and would cost exactly the predictability this library sells.
