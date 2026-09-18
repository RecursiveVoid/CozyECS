# CozyECS benchmark results

Run on 2026-09-17 (label `audit_final`), after the benchmark fairness audit.

> `npm run benchmark` rewrites this file with freshly generated tables only. The notes in the
> sections "Summary", "Fairness fixes", "Shared buffer", "Where CozyECS is not #1" and "Caveats"
> were written by hand for this run.

## Summary

CozyECS's fastest form is #1 in all five scenarios. The margins are smaller than in the
2026-09-16 results, because the audit removed several harness effects that slowed down
competitors (see [Fairness fixes](#fairness-fixes)).

"Paired" is the median over 5 repeats of (CozyECS's best form ÷ the fastest competitor measured
in the same repeat). This ratio cancels out slow drift in machine load.

| scenario | best CozyECS (ops/sec) | best competitor (ops/sec) | paired |
|---|---:|---:|---:|
| packed_5 | systems 487,645 / direct 480,170 | harmony-ecs 430,455 | 1.12 |
| simple_iter | systems 415,145 | bitecs 346,897 | 1.20 |
| frag_iter | systems 912,828 / direct 903,068 | harmony-ecs 822,210 | 1.13 |
| entity_cycle | direct 70,947 | wolf-ecs 16,387 | 4.11 |
| add_remove | direct 49,691 | wolf-ecs 31,798 | 1.56 |

Memory at 100k entities (Position+Velocity, one query): CozyECS 26.2 B/entity, next best
harmony-ecs at 54.2 and wolf-ecs at 72.4.

**Not every CozyECS API wins every scenario.** `cozyecs (direct)` is slower than bitecs and
wolf-ecs on simple_iter. See [Where CozyECS is not #1](#where-cozyecs-is-not-1).

## Environment

- Node v22.14.0 (darwin/arm64), Apple M4 (10 cores: 4 performance, 6 efficiency)
- `NODE_ENV=production` for every benchmark and memory process
- Load average about 10 during the run (busy laptop: iCloud and media daemons running). The
  load average at the end was 3.72 / 6.13 / 7.36.
- Library versions: bitecs 0.3.40, bitecs 0.4.0 (`bitecs4`), wolf-ecs 2.1.3, harmony-ecs 0.0.12,
  @lastolivegames/becsy 0.15.5 (perf build), @javelin/ecs 1.0.0-alpha.13, miniplex 2.0.0,
  ecsy 0.4.3, geotic 4.3.2, perform-ecs 0.7.8

## Methodology

```
node benchmarks/benchmark.js --paired --repeats=5 --time=2000 --warmup=500 --min-ref=0.85
```

- **One process per job.** Every (scenario, variant) job runs in its own `node` process
  (`benchmarks/worker.js`), so JIT state and type feedback never leak between libraries.
- **Benchmark first, verify afterwards.** Each job benchmarks the first world created in its
  process. Steady-state scenarios are re-checked after the benchmark. The variant is then
  verified on a fresh world (3 ticks, then the full state is checked). A job that fails
  verification is reported as "wrong result".
- **Timing:** 500 ms warm-up, then 2000 ms of measurement. `step()` is called through the
  `mega` batch runner, which is the same for all libraries.
- **Repeats:** the full list of 70 jobs is run 5 times, in a freshly shuffled order each time
  (seed 513185147). The tables report the **median** of the 5 repeats.
- **Paired ratios:** in each repeat, every variant's ops/sec is divided by the fastest correct
  non-CozyECS variant from the same repeat. The median of those ratios is reported.
- **Slow-core guard:** every job also times a fixed CPU reference kernel. A job whose reference
  speed falls below 0.85 x the best seen in the run (because it landed on an efficiency core or
  a contended CPU) is re-run, up to 2 times.
- **Idiomatic adapters.** Each adapter iterates the way that library's README shows. Typed
  libraries use `f32` fields and object libraries use JS numbers. Entity counts and the work per
  op are the same across adapters.

### Scenarios

- **packed_5**: 1000 entities with A..E ({value}). 5 systems, each doubling one component.
- **simple_iter**: 4 archetypes x 1000 entities (P+V, P+V+A, P+V+B, P+V+A+B). 1 system does P += V.
- **frag_iter**: 26 components A..Z, with 100 entities per letter, each having [Letter, Data].
  1 system doubles Data (2600 entities in 26 archetypes).
- **entity_cycle**: 1000 entities with A. Per op: spawn one B-entity per A, then destroy every
  B-entity.
- **add_remove**: 1000 entities with A. Per op: add B to every A, then remove B from every B.

### CozyECS variants

- `cozyecs`: function systems with chunk loops, run by `world.update()`. Structural changes go
  through the command buffer.
- `cozyecs (direct)`: raw chunk loops with immediate structural changes.
- `cozyecs (forEach)`: `query.forEach` callbacks.

### Extra competitor variant

- `bitecs4 (cached query)`: bitecs 0.4 rebuilds a string key (map/sort/join) on every
  `query(world, terms)` call. This variant keeps the stable array returned by `query()` and calls
  the exported `commitRemovals()` once per frame. `bitecs4` is the documented usage.

## Throughput (ops/sec, median of 5)

| variant | packed_5 | simple_iter | frag_iter | entity_cycle | add_remove |
|---|---:|---:|---:|---:|---:|
| cozyecs | **487,645** | **415,145** | **912,828** | 58,191 | 40,704 |
| cozyecs (direct) | 480,170 | 279,162 | 903,068 | **70,947** | **49,691** |
| cozyecs (forEach) | 369,450 | 360,747 | 680,811 | 55,969 | 43,067 |
| bitecs | 349,240 | 346,897 | 727,961 | 3,227 | 5,106 |
| bitecs4 | 234,902 | 278,517 | 707,125 | 3,505 | 2,895 |
| bitecs4 (cached query) | 370,655 | 341,384 | 718,808 | 3,544 | 2,611 |
| wolf-ecs | 379,934 | 323,361 | 715,302 | 16,387 | 31,798 |
| harmony-ecs | 430,455 | 272,589 | 822,210 | 8,274 | 5,805 |
| becsy | 82,966 | 13,282 | 144,742 | 6,801 | 16,618 |
| javelin | 236,279 | 145,304 | 380,015 | 1,429 | 1,347 |
| miniplex | 38,658 | 45,385 | 36,999 | 2,714 | 1,303 |
| ecsy | 17,935 | 30,898 | 58,894 | 828 | 1,691 |
| geotic | 77,422 | 89,235 | 67,946 | 760 | 1,453 |
| perform-ecs | 158,614 | 173,394 | 58,670 | 2,857 | 2,488 (wrong result) |

### Paired ratio vs best competitor (median over 5 shuffled repeats)

A value is the variant's ops/sec divided by the fastest correct non-CozyECS variant in the same
repeat. A value of 1.00 or more means the variant beat every competitor.

| variant | packed_5 | simple_iter | frag_iter | entity_cycle | add_remove |
|---|---:|---:|---:|---:|---:|
| cozyecs | 1.10 | 1.20 | 1.13 | 3.41 | 1.28 |
| cozyecs (direct) | 1.12 | 0.81 | 1.11 | 4.11 | 1.56 |
| cozyecs (forEach) | 0.85 | 1.04 | 0.87 | 3.39 | 1.37 |
| bitecs | 0.81 | 1.00 | 0.89 | 0.19 | n/a |
| bitecs4 | 0.54 | 0.81 | 0.86 | 0.20 | 0.09 |
| bitecs4 (cached query) | 0.85 | 0.98 | 0.90 | 0.21 | 0.08 |
| wolf-ecs | 0.89 | 0.94 | 0.89 | 1.00 | 1.00 |
| harmony-ecs | 1.00 | 0.78 | 1.00 | 0.51 | 0.20 |
| becsy | 0.18 | 0.04 | 0.17 | 0.41 | 0.52 |
| javelin | 0.53 | 0.41 | 0.47 | 0.08 | 0.04 |
| miniplex | 0.10 | 0.13 | 0.05 | 0.16 | 0.04 |
| ecsy | 0.04 | 0.09 | 0.07 | 0.05 | 0.05 |
| geotic | 0.18 | 0.26 | 0.09 | 0.05 | 0.05 |
| perform-ecs | 0.36 | 0.50 | 0.07 | 0.17 | n/a |

For competitors, the ratio is taken against the fastest *other* competitor, so the leading
competitor in each scenario shows 1.00.

### Relative to the fastest correct variant in each scenario (medians)

| variant | packed_5 | simple_iter | frag_iter | entity_cycle | add_remove |
|---|---:|---:|---:|---:|---:|
| cozyecs | 100% | 100% | 100% | 82% | 82% |
| cozyecs (direct) | 98% | 67% | 99% | 100% | 100% |
| cozyecs (forEach) | 76% | 87% | 75% | 79% | 87% |
| bitecs | 72% | 84% | 80% | 5% | 10% |
| bitecs4 | 48% | 67% | 77% | 5% | 6% |
| bitecs4 (cached query) | 76% | 82% | 79% | 5% | 5% |
| wolf-ecs | 78% | 78% | 78% | 23% | 64% |
| harmony-ecs | 88% | 66% | 90% | 12% | 12% |
| becsy | 17% | 3% | 16% | 10% | 33% |
| javelin | 48% | 35% | 42% | 2% | 3% |
| miniplex | 8% | 11% | 4% | 4% | 3% |
| ecsy | 4% | 7% | 6% | 1% | 3% |
| geotic | 16% | 21% | 7% | 1% | 3% |
| perform-ecs | 33% | 42% | 6% | 4% | 5% |

### Spread across repeats (min .. max)

| variant | packed_5 | simple_iter | frag_iter | entity_cycle | add_remove |
|---|---:|---:|---:|---:|---:|
| cozyecs | 446,591 .. 501,440 | 339,780 .. 426,168 | 896,392 .. 928,382 | 48,853 .. 58,543 | 39,379 .. 44,212 |
| cozyecs (direct) | 427,067 .. 488,458 | 264,496 .. 283,601 | 841,724 .. 916,565 | 64,519 .. 71,715 | 46,199 .. 52,479 |
| cozyecs (forEach) | 349,454 .. 378,778 | 337,825 .. 365,383 | 583,149 .. 707,711 | 52,025 .. 58,345 | 42,136 .. 44,992 |
| bitecs | 336,301 .. 366,840 | 342,821 .. 353,456 | 700,839 .. 736,283 | 3,140 .. 3,386 | 4,671 .. 5,151 |
| bitecs4 | 206,658 .. 241,549 | 259,093 .. 310,013 | 647,815 .. 719,624 | 3,288 .. 3,664 | 2,592 .. 2,919 |
| bitecs4 (cached query) | 320,643 .. 382,370 | 320,374 .. 345,487 | 696,425 .. 760,946 | 3,411 .. 3,629 | 2,426 .. 2,623 |
| wolf-ecs | 364,925 .. 382,209 | 303,320 .. 338,535 | 672,674 .. 732,007 | 16,275 .. 17,887 | 27,292 .. 32,491 |
| harmony-ecs | 427,055 .. 468,142 | 255,927 .. 274,617 | 753,434 .. 876,097 | 8,236 .. 8,527 | 5,745 .. 6,730 |
| becsy | 77,883 .. 84,850 | 10,743 .. 13,945 | 123,415 .. 152,366 | 6,620 .. 7,023 | 14,168 .. 16,796 |
| javelin | 203,614 .. 242,089 | 133,031 .. 153,178 | 338,118 .. 399,795 | 1,289 .. 1,496 | 1,241 .. 1,394 |
| miniplex | 31,911 .. 43,609 | 42,376 .. 48,074 | 22,295 .. 37,629 | 2,478 .. 2,763 | 1,292 .. 1,390 |
| ecsy | 17,287 .. 18,490 | 29,654 .. 33,250 | 51,919 .. 62,342 | 764 .. 835 | 1,484 .. 1,752 |
| geotic | 76,727 .. 78,805 | 88,242 .. 92,874 | 55,292 .. 78,764 | 739 .. 780 | 1,333 .. 1,477 |
| perform-ecs | 151,281 .. 165,742 | 150,985 .. 183,574 | 53,990 .. 61,592 | 2,716 .. 2,871 | 2,368 .. 2,587 |

## Memory: 100,000 entities with Position{x,y} + Velocity{dx,dy}

Each library runs in its own `node --expose-gc` process, and the table shows the median of the
repeats. `NODE_ENV=production`. Each process first builds a 10-entity world to warm up the code
path. `gc()` runs before and after creation. The numbers are the retained JS heap plus
ArrayBuffer (external) bytes of the world **and one held Position+Velocity query**.

| library | total MB | heap MB | buffers MB | bytes/entity | create time |
|---|---:|---:|---:|---:|---:|
| cozyecs | 2.50 | 0.16 | 2.34 | 26.2 | 12 ms |
| harmony-ecs | 5.17 | 2.88 | 2.29 | 54.2 | 26 ms |
| wolf-ecs | 6.91 | 5.38 | 1.53 | 72.4 | 12 ms |
| javelin | 16.11 | 16.11 | 0.00 | 168.9 | 114 ms |
| becsy | 16.21 | 6.51 | 9.70 | 170.0 | 42 ms |
| miniplex | 20.33 | 20.33 | 0.00 | 213.2 | 38 ms |
| bitecs4 | 24.13 | 22.60 | 1.53 | 253.0 | 39 ms |
| bitecs | 28.81 | 26.90 | 1.91 | 302.1 | 40 ms |
| perform-ecs | 36.11 | 36.11 | 0.00 | 378.7 | 53 ms |
| geotic | 38.04 | 38.04 | 0.00 | 398.9 | 788 ms |
| ecsy | 93.33 | 93.33 | 0.00 | 978.7 | 858 ms |

- bitecs4, wolf-ecs, harmony-ecs and becsy have fixed-capacity stores, and they are sized to
  exactly 100,000 entities here. bitecs 0.3 uses its default size of 100k. CozyECS grows its
  storage on demand (capacity 102,400 at this count), so its figure is not flattered by a tight
  preallocation.
- CozyECS: a table of 20 B/row plus 4 B/index of entity state gives 24.6 B/entity of
  ArrayBuffers. About 100 KB of the measured heap delta is Node initializing `performance`
  lazily inside the probe itself.
- Holding a query is now part of the measurement for every library. It raised the figures for
  libraries whose queries store every matching entity: bitecs to 302, bitecs4 to 253, becsy to
  170 and ecsy to 979 B/entity.

## Fairness fixes

The benchmarks were audited before this run. Nothing in `src/` changed; all fixes are in
`benchmarks/`. To see how much each fix mattered, the old and new harness were run back to back
(3 rounds each, 1 s per job):

| job | old harness | new harness | change |
|---|---:|---:|---:|
| becsy packed_5 | 8,719 | 82,004 | 9.4x |
| perform-ecs packed_5 | 20,304 | 169,640 | 8.4x |
| geotic packed_5 | 14,998 | 77,265 | 5.2x |
| ecsy packed_5 | 10,815 | 18,439 | 1.7x |
| bitecs simple_iter | 248,640 | 342,308 | 1.38x |
| wolf-ecs simple_iter | 247,697 | 344,938 | 1.39x |
| becsy simple_iter | 10,060 | 14,006 | 1.39x |
| cozyecs (all variants), harmony-ecs, bitecs4, javelin | | | about 1.00x |

1. **Verification world polluted type feedback (the biggest effect).** Each job used to verify
   correctness on its own world *before* building the benchmark world. V8 shares type feedback
   between closures and classes created by the same source line, so the verification world
   left the benchmark code polymorphic. This hurt libraries with a class per component (becsy,
   ecsy, geotic, perform-ecs) and adapters that capture typed arrays in closures (bitecs,
   wolf-ecs). CozyECS defines its components once for the whole process, so it was never
   affected. Now the benchmark runs on the first world created, and verification runs afterwards
   on a fresh world.
2. **Shared system body in packed_5.** The adapters for becsy, ecsy, geotic, perform-ecs and
   javelin built their 5 systems from one loop. That single line of code saw 5 component classes,
   which made property access there slow. Each of these adapters now has 5 hand-written systems,
   as a real app would.
3. **bitecs 0.4 query cost.** Added the `bitecs4 (cached query)` variant (see above) next to the
   documented usage. On packed_5 it goes from 234.9k to 370.7k ops/sec.
4. **Development-mode checks.** ecsy runs extra checks unless `NODE_ENV=production`, so every
   benchmark and memory process now sets it. ecsy still reads `process.env` on some paths, which
   costs it a little in Node. It is far behind in every scenario either way.
5. **Memory warm-up.** bitecs 0.3 was the only library measured without the small warm-up world,
   because its entity ids are global to the process. It now calls `resetGlobals()`, so the
   warm-up can run.
6. **Memory queries.** Some adapters used to hold a query in the memory test and others did not.
   Every library now holds one Position+Velocity query.
7. **Checked, no change needed:**
   - Field types: every typed library uses `f32`, and object libraries use JS numbers.
   - Entity counts and work per op match across adapters. In add_remove, javelin takes two steps
     per op because its changes apply one step late. That is the same amount of work.
   - Every job gets the same process isolation, warm-up and batch runner.
   - Adapters iterate the way each library's README shows. becsy uses its perf build.
   - CozyECS adapters have the strictest verification of all adapters: every value, the chunk
     layout and the live entity count are checked. Their structural changes really happen.
   - perform-ecs's add_remove failure is a genuine leak in that library:
     `removeComponentsFromEntity` never removes the entry from `entity.components`, which held
     6,888 entries after the benchmark.

## Shared buffer (`new World({ shared: true })`)

`shared: true` allocates each archetype table as a `SharedArrayBuffer` instead of an
`ArrayBuffer`. The two builds were compared with isolated, shuffled, paired runs (5 repeats in
round 4, 4 repeats in round 3, 1 s per job). Each cell is the median of shared ÷ plain:

| scenario | `cozyecs` r4 | `cozyecs (direct)` r4 | `cozyecs` r3 | `cozyecs (direct)` r3 |
|---|---:|---:|---:|---:|
| packed_5 | 1.00 | 1.00 | 1.00 | 0.99 |
| simple_iter | 1.00 | 1.00 | 1.00 | 1.00 |
| frag_iter | 1.00 | **0.91** | **0.94** | 0.99 |
| entity_cycle | 1.00 | 1.02 | 1.00 | 1.02 |
| add_remove | 1.00 | 0.99 | 1.00 | 0.94 |

- Most cells are within ±2%, which is run-to-run noise. frag_iter lost 6–9% in one variant in
  each round, though not in the same variant both times. In round 4 the loss was consistent:
  `direct` measured 827k–846k shared against about 920k plain in 4 of 5 repeats. The 0.94 on
  add_remove (direct) in round 3 did not show up again in round 4. Treat many small shared
  chunks as possibly a few percent slower.
- Memory is the same: 26.2 B/entity and 12 ms creation time with or without `shared`.
- The default is `shared: false`. Turn it on only if you hand columns to workers.

## Where CozyECS is not #1

- **simple_iter, `cozyecs (direct)`: 0.81 of bitecs.** The raw chunk loop outside a system is
  slower than bitecs (346.9k) and wolf-ecs (323.4k) here. The cause is in `src/`, not in the
  adapter. The same loop inside a function system reaches 415k. The `forEach` form wins this
  scenario only by 1.04.
- **packed_5 and frag_iter, `cozyecs (forEach)`: 0.85 and 0.87.** harmony-ecs, and on packed_5
  also wolf-ecs and `bitecs4 (cached query)`, beat the callback form. Use chunk loops or
  `forEachChunk` when this matters.
- **Structural changes inside systems cost 15–20%.** On entity_cycle and add_remove the systems
  variant reaches 82% of `direct`, because commands are queued and then flushed. It is still
  3.4x (entity_cycle) and 1.28x (add_remove) ahead of the fastest competitor.
- **packed_5: harmony-ecs won one repeat outright.** Taking the best CozyECS form in each repeat
  and dividing by the fastest competitor in that same repeat gives 1.168, 0.984, 1.124, 1.103 and
  1.135 (median 1.12). In repeat 2, harmony-ecs at 468.1k beat every CozyECS form. The other four
  scenarios stay above 1.00 in all five repeats.
- **Leads are narrow in iteration.** The margins are 1.10–1.20 on packed_5, simple_iter and
  frag_iter, and the spread across repeats is about 10%, so a single repeat can swing the order.
  The claim these numbers support is that CozyECS's best form is fastest in each scenario at the
  median. They do not support "every CozyECS API is the fastest", nor a reliable win in any one
  iteration run.

## Caveats

- The machine was loaded (load average about 10). The shuffled pairing and the slow-core guard
  reduce the effect, but the function-system variant still varied from 340k to 426k on
  simple_iter, and entity_cycle and add_remove repeats varied by 10–15%.
- The `cozyecs` systems variant measured 415k on simple_iter here, against 282k in an earlier
  run the same day. `src/` probably changed in between, so do not compare these numbers with
  runs from earlier builds.
- These are micro-benchmarks of five fixed workloads on one CPU and one Node version. Measure
  your own workload.

## Reproducing

```
npm run build
npm run benchmark -- --paired --repeats=5
```

Use `--scenario=`, `--lib=` and `--variant=` to run a subset, and `--quick` for a noisy smoke
test. `benchmarks/benchmark.js` lists every option. Running the benchmark regenerates this file.

<!-- gpu-kernel:start -->
## GPU / kernel

Generated by `node benchmarks/gpu.js` at 2026-09-18T08:15:30.767Z (merged from 2 runs with `--from-json`).

- Machine: Apple M4 (10 cores), Darwin 25.5.0, Node v22.14.0
- Load average (1/5/15 min): start 4.6 / 8.9 / 14.5, end 5.0 / 5.3 / 8.6
- Median of 5 isolated processes per cell, 600 ms measured per process after 200 ms warm-up. ms/frame, lower is better.
- Layout: 4 archetypes (Pos+Vel, +A, +B, +A+B), n/4 entities each. `simple`: `p.x += v.dx; p.y += v.dy`. `gravity`: gravity + integrate + bounce, with `dt` and a uniform.
- Correctness gate: every CPU cell checks every entity against an f32 reference. Every GPU cell (including `none`, read from the device buffer) is drained with `flushKernels`, then compared row by row with the CPU backend run for the same number of frames; a cell that differs in any repeat shows `FAIL (failed/runs)` instead of a time.
- `gpu-none` / `gpu-async` ms/frame include the drain (`flushKernels`) after every timed batch, so the final drain is paid for.

### simple

| entities | kernel-cpu | forEachChunk | plain | bitecs | gpu-none | gpu-async | gpu-sync |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 0.0013 | 0.0012 | 0.0018 | 0.0014 | 0.058 | 0.071 | 0.352 |
| 10,000 | 0.011 | 0.011 | 0.017 | 0.015 | 0.057 | 0.073 | 0.365 |
| 30,000 | 0.034 | 0.034 | 0.051 | 0.044 | 0.058 | 0.075 | 0.414 |
| 50,000 | 0.059 | 0.059 | 0.085 | 0.073 | 0.057 | 0.078 | 0.405 |
| 100,000 | 0.149 | 0.142 | 0.170 | 0.149 | 0.057 | 0.091 | 0.455 |
| 300,000 | 0.541 | 0.539 | 0.510 | 1.43 | 0.074 | 0.315 | 0.767 |
| 1,000,000 | 1.80 | 1.80 | 1.72 | 2.12 | 0.350 | 1.18 | 2.06 |

Speed vs bitecs (bitecs ms / variant ms, >1 = faster than bitecs):

| entities | kernel-cpu | forEachChunk | plain |
|---:|---:|---:|---:|
| 1,000 | 1.08x | 1.16x | 0.78x |
| 10,000 | 1.29x | 1.30x | 0.86x |
| 30,000 | 1.30x | 1.30x | 0.86x |
| 50,000 | 1.24x | 1.23x | 0.86x |
| 100,000 | 1.00x | 1.05x | 0.87x |
| 300,000 | 2.65x | 2.66x | 2.81x |
| 1,000,000 | 1.17x | 1.18x | 1.23x |

### gravity

| entities | kernel-cpu | forEachChunk | plain | bitecs | gpu-none | gpu-async | gpu-sync |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 0.0026 | 0.0025 | 0.0028 | 0.0028 | 0.058 | 0.082 | 0.351 |
| 10,000 | 0.025 | 0.025 | 0.028 | 0.028 | 0.057 | 0.083 | 0.350 |
| 30,000 | 0.076 | 0.076 | 0.083 | 0.084 | 0.057 | 0.085 | 0.368 |
| 50,000 | 0.129 | 0.132 | 0.138 | 0.140 | 0.057 | 0.092 | 0.402 |
| 100,000 | 0.271 | 0.286 | 0.275 | 0.282 | 0.058 | 0.111 | 0.450 |
| 300,000 | 0.813 | 0.857 | 0.829 | 2.11 | 0.112 | 0.272 | 0.960 |
| 1,000,000 | 2.70 | 2.85 | 2.76 | 3.12 | 0.364 | 1.40 | 2.30 |

Speed vs bitecs (bitecs ms / variant ms, >1 = faster than bitecs):

| entities | kernel-cpu | forEachChunk | plain |
|---:|---:|---:|---:|
| 1,000 | 1.08x | 1.10x | 0.98x |
| 10,000 | 1.13x | 1.12x | 1.03x |
| 30,000 | 1.11x | 1.10x | 1.02x |
| 50,000 | 1.09x | 1.06x | 1.02x |
| 100,000 | 1.04x | 0.99x | 1.03x |
| 300,000 | 2.60x | 2.47x | 2.55x |
| 1,000,000 | 1.15x | 1.09x | 1.13x |

### gpu-async readback stats (median per process)

| kernel | entities | dispatches | coalescedReadbacks | staleReadbacks | final drain ms | gate |
|---|---:|---:|---:|---:|---:|---|
| simple | 1,000 | 11051 | 7223 | 0 | 0.230 | ok |
| simple | 10,000 | 11084 | 7223 | 0 | 0.277 | ok |
| simple | 30,000 | 10544 | 6594 | 0 | 0.464 | ok |
| simple | 50,000 | 9930 | 6062 | 0 | 0.236 | ok |
| simple | 100,000 | 8629 | 4399 | 0 | 0.270 | ok |
| simple | 300,000 | 2936 | 389 | 0 | 0.460 | ok |
| simple | 1,000,000 | 1126 | 680 | 0 | 0.727 | ok |
| gravity | 1,000 | 9646 | 4903 | 0 | 0.216 | ok |
| gravity | 10,000 | 9633 | 5457 | 0 | 0.251 | ok |
| gravity | 30,000 | 9629 | 4896 | 0 | 0.203 | ok |
| gravity | 50,000 | 8682 | 3899 | 0 | 0.305 | ok |
| gravity | 100,000 | 7242 | 1230 | 0 | 0.212 | ok |
| gravity | 300,000 | 3114 | 425 | 0 | 0.430 | ok |
| gravity | 1,000,000 | 977 | 584 | 0 | 0.948 | ok |

### GPU break-even vs kernel-cpu (entities)

| kernel | gpu-none | gpu-async | gpu-sync |
|---|---:|---:|---:|
| simple | 48741 | 64390 | > 1000000 |
| gravity | 22676 | 34164 | 553988 |

<!-- gpu-kernel:end -->

### GPU / kernel: notes (written by hand for the 2026-09-18 round-2 run)

**The section above is regenerated.** `node benchmarks/gpu.js --write` replaces everything between
the `gpu-kernel` markers. These notes are outside the markers, so they are kept. The round-2 table
was measured as two invocations (`--kernels=simple --json=...`, then `--kernels=gravity`) and
rendered with `--from-json=simple.json,gravity.json --write`.

**GPU:** Dawn, through the `webgpu` npm package (0.6.1) on the M4's integrated GPU. A GPU frame
calls `world.update(dt)` and then yields once to the event loop. `sync-frame` also awaits
`flushKernels(world)`. The `none` and `async` runs are drained with `flushKernels` after every
timed batch, inside the clock, so the final drain is included in their ms/frame.

**Absolute times are about 1.7-2x the round-1 figures on every variant, CPU and GPU alike**
(for example `kernel-cpu` simple at 1k: 1.3 µs vs 0.6 µs, and the GPU `none` floor: 0.057 ms vs
0.037 ms). The spread across the 5 repeats is tight (usually under 3%), so the machine ran
uniformly slower during this run (freshly restarted, other agents active). Ratios and
break-evens are therefore comparable with round 1. Absolute ms are not.

**Correctness gate: every GPU cell passes, at every size, in every repeat.** That includes
`async` from 1k to 1M. Each GPU process is drained with `flushKernels`, and then every row
(x, y, dx, dy) is compared with the CPU backend run for the same number of frames; `none` is read
back from the device buffer (`handle.bufferFor`). The round-1 `async` failures at 100k-1M (tables
missing the last ~40-60 frames after a skipped readback) are fixed. Now `stats.staleReadbacks` is
0 in every `async` process, and frames that could not get their own readback are counted in
`stats.coalescedReadbacks` (for example 7,223 of 11,051 dispatches at 1k and 680 of 1,126 at 1M
for simple). The next readback carries their results.

Because every `async` frame's results are now actually read back, `async` at 300k-1M costs much
more than in round 1 (simple at 1M: 1.18 ms vs 0.27 ms; that 0.27 ms was an undercount, because
most readbacks were skipped). The 1M `async` cells also spread widely across repeats (0.51-1.48
ms for simple, 0.92-1.44 ms for gravity), depending on how many readbacks coalesce.

**Break-even vs `kernel-cpu`.** Interpolated log-linearly between the measured sizes (1k, 10k,
30k, 50k, 100k, 300k, 1M), so treat these as rough figures:

| kernel | none | async | sync-frame |
|---|---:|---:|---:|
| simple (2 adds) | ~49k (round 1: ~65k) | ~64k (~74k) | > 1M (> 1M) |
| gravity (integrate + bounce) | ~23k (~26k) | ~34k (~32k) | ~554k (~530k) |

- The GPU's fixed floor is about 0.057 ms for `none`, 0.07-0.08 ms for `async` and 0.35 ms for
  `sync-frame`, flat from 1k to 50k (about 0.037 / 0.043 / 0.22 ms in round 1; the whole machine
  was slower this run).
- `sync-frame` breaks even only for gravity (~554k). For simple it is still 1.14x slower than the
  CPU at 1M (2.06 vs 1.80 ms).
- **Defaults in `src/gpu/runtime.ts` (`fireAndForget = 45_000`, `synchronous = 700_000`).**
  Compared with the raw break-evens, 45k is 1.08x the simple `none` break-even, 1.43x simple
  `async`, 0.76x gravity `async` and **0.50x gravity `none`** (outside 0.6-1.6x). 700k is 0.79x
  gravity `sync-frame` and below the simple break-even (> 1M). `auto` does not use the raw
  defaults, though. It scales them by `BASELINE_CPU_NS_PER_ENTITY / estimateCPUNanosPerEntity(ir)`,
  and the estimator gives simple 0.57 ns (scale 2.0) and gravity 1.90 ns (scale 0.6). So the
  effective thresholds are simple 90k / 1.4M and gravity 27k / 420k. Against those, gravity is
  0.84x (`none`), 1.27x (`async`) and 1.32x (`sync-frame`), and simple `async` is 0.72x, all
  inside 0.6-1.6x. Simple `none` is **0.54x** (49k measured vs 90k effective), just outside.
  The cause is the estimator, not the constants: it rates simple at half the baseline cost,
  while the measured CPU cost ratio gravity/simple is only 1.5-1.8x, not the estimator's 3.3x.
  With `none`, `auto` therefore keeps a simple kernel on the CPU from ~49k to 90k entities,
  where the GPU would be faster by up to about 2x (a tie at 50k: 0.059 vs 0.057 ms; at 100k,
  just past the band, 0.149 vs 0.057 ms). No constant change is needed for `async` or `sync-frame`.
- **Fixed after this run (estimator re-fit, no re-measurement).** `estimateCPUNanosPerEntity` is now
  `0.52 + 0.065 × ops` ns (a fixed per-entity term plus a per-op term) instead of `0.095 × ops`.
  Effective thresholds: simple ~56k / ~877k, gravity ~28k / ~438k, i.e. measured/effective of
  0.87 (simple `none`), 1.14 (simple `async`), 0.80 (gravity `none`), 1.21 (gravity `async`) and
  1.26 (gravity `sync-frame`). All inside 0.6-1.6x. The thresholds constants are unchanged.
- **Confirmation run (2026-09-18, after the re-fit, machine load ~2.5).** Re-measured break-evens vs
  kernel-cpu: simple 49.7k (`none`) / 65.4k (`async`) / > 1M (`sync-frame`); gravity 22.4k / 33.6k /
  645k. Against the effective thresholds (simple ~56k / ~877k, gravity ~28k / ~438k) that is
  measured/effective 0.88, 1.16, 0.79, 1.19 and 1.47 (gravity `sync-frame`): every switch point
  within 0.6-1.6x. All GPU cells passed the row-by-row correctness gate; `staleReadbacks` 0 everywhere.

**CPU backend.** `kernelSystem({ target: 'cpu' })` matches hand-written `forEachChunk` on
`simple` (within 5% at every size) and is 1.08-1.30x faster than bitecs up to 50k, 1.00x at 100k
and 1.17x at 1M. On `gravity` it is now 1.04-1.15x bitecs. It is level with hand-written `forEachChunk` up to 30k
and 2-5% faster from 50k (2.70 vs 2.85 ms at 1M). In round 1 it was 0.94-0.97x bitecs and behind
`forEachChunk`, so the branch/bounce gap is closed.

**bitecs at 300k is an outlier:** 1.43 ms (simple) and 2.11 ms (gravity), which is 2.6x slower
than every CozyECS variant and slower per entity than bitecs at 1M. It reproduces in all 5 repeats
(1.38-2.15 ms). The likely cause is the harness's `setDefaultSize(1.1 n)` sizing of bitecs's sparse
set at that size. It is not a CozyECS result, so the 300k "vs bitecs" ratios (2.5-2.8x) are left
out of the claims here.

**Plain chunk loop (carryover issue B).** The plain `for (i < chunk.count)` loop over
`chunk.col()` is 0.78-0.87x bitecs on `simple` from 1k to 100k and 1.23x at 1M, where memory
bandwidth dominates (round 1: 0.77-0.88x and 1.24x). On `gravity`, the heavier body hides the gap
(0.98-1.03x up to 100k, 1.13x at 1M). `kernelSystem` and `forEachChunk` stay the recommended fast
paths (docs/INTERNALS.md, "Why the plain chunk loop trails closure-constant loops").

### Regression rerun of the 5 scenarios + memory (2026-09-18 round 2, same build as the GPU run above)

`node benchmarks/benchmark.js --paired --repeats=5 --no-write` (defaults: 2000 ms measured after a
500 ms warm-up, slow-core guard at 0.85x the reference kernel). There were 58 slow-core retries.
The tables at the top of this file are from the 2026-09-17 audit run and were left unchanged.
Absolute ops/sec are about half of the round-1 rerun on every library, CozyECS and competitors
alike: the reference kernel itself ran at 315k (the same slower machine state as the GPU run). So
compare the paired ratios, not the ops/sec.

| scenario | best CozyECS (ops/sec) | best competitor (ops/sec) | paired, round 2 [min..max] | round 1 (09-18) | 09-17 |
|---|---:|---:|---:|---:|---:|
| packed_5 | systems 260,772 | harmony-ecs 254,840 | 1.02 [0.98..1.03] | 1.03 | 1.12 |
| simple_iter | systems 221,860 | bitecs4 (cached) 183,992 | 1.21 [1.20..1.22] | 1.19 | 1.20 |
| frag_iter | systems 493,042 | harmony-ecs 471,934 | 1.06 [1.04..1.10] | 1.31 | 1.13 |
| entity_cycle | direct 36,967 | wolf-ecs 9,169 | 4.01 [3.25..4.18] | 4.13 | 4.11 |
| add_remove | direct 27,748 | wolf-ecs 17,065 | 1.63 [1.62..1.64] | 1.63 | 1.56 |

CozyECS is still #1 in all five scenarios by median paired ratio. packed_5 is effectively a tie
with harmony-ecs, as in the round-1 rerun. The median is 1.024 and the worst repeat is 0.975. frag_iter's margin (1.06) is back near the 09-17 value (1.13), and round 1's 1.31 looks like
the outlier. Memory is unchanged at **26.2 B/entity** (harmony-ecs 54.2, wolf-ecs 72.4, bitecs4
253.0, bitecs 302.0). perform-ecs still fails verification on add_remove, as before.
