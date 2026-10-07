/// <reference types="node" />
/**
 * Does CozyECS leak? Measured in Node, where the answer can be definitive.
 *
 * The complaint this file exists to settle: a long-running page that rebuilds its World
 * (level change, backend switch, entity-count switch) grows without bound. The browser
 * cannot answer that question -- Chrome's `performance.memory` largely EXCLUDES ArrayBuffer
 * memory, and ALL of CozyECS's entity data lives in ArrayBuffers (one per archetype table,
 * plus the entity index). A JS-heap graph that looks flat says nothing about the bytes that
 * actually matter here.
 *
 * Node does answer it:
 *
 *   `process.memoryUsage().arrayBuffers` -- bytes in ArrayBuffers V8 has not yet freed. THE
 *      number for this library: archetype tables, `entities` arrays, the entity index.
 *   `process.memoryUsage().heapUsed` ----- JS objects: archetypes, queries, systems, closures,
 *      listener arrays, Maps, compiled trampolines.
 *   `global.gc()` ----------------------- removes "it just has not been collected yet" as an
 *      explanation, so a delta that survives it is retention, not lag.
 *   `WeakRef` --------------------------- proves the object itself is gone, not merely that the
 *      heap looks similar. The only unambiguous evidence of collectability.
 *
 * Five scenarios, each reporting bytes/cycle (printed as a table at the end of the run):
 *
 *   1. build -> run -> dispose -> drop, x20       both meters must return to baseline
 *   2. build -> run -> drop, x20 (NO dispose)     is a dropped world collectable on its own?
 *   3. WeakRef proof                              World, Archetype, raw ArrayBuffer
 *   4. churn inside one long-lived world          then compact(); string table + archetype bounds
 *   5. system / query / listener churn            nothing may accumulate
 *
 * Worlds are SMALL on purpose (20k entities, ~2 MB of tables) and the cycle count is high: a
 * leak shows up as a slope, and 20 cycles x 2 MB is a loud signal at ~40 MB of worst-case
 * retention. A 1M-entity world would prove nothing extra and could fill the machine.
 *
 * Needs `--expose-gc`: run `npm run test:leak`. Without it the measuring tests skip with a
 * note (the behavioural tests still run).
 *
 * MEASUREMENT HONESTY. Two things are asserted for every scenario, never just the total:
 * the least-squares slope per cycle (a leak is linear in cycles; harness noise is not) and
 * the worst single-sample excursion from baseline. A 1-cycle total can hide a leak behind
 * one lucky collection; a slope over 20 cannot.
 */
import { describe, test, expect, afterAll } from '@jest/globals';
import { World, System, component, tag, f32, i16, str } from '../src/index';
import type { Archetype, ColumnsOf, EntityCallback, SystemHandle } from '../src/index';

// ---------------------------------------------------------------------------- the API under test
//
// `World.dispose()` is the API this file was written against. It is reached through a cast
// rather than `world.dispose()` so that THIS FILE TYPE-CHECKS BOTH BEFORE AND AFTER dispose()
// lands in src/world.ts (tests and src share one tsconfig; a hard reference would break
// `npm run typecheck` for every other suite while the method is missing). The runtime
// behaviour is not softened: a missing dispose() fails the test that needs it, loudly.

interface MaybeDisposable {
  dispose?: () => void;
}

/** `world.dispose` bound, or undefined when the method does not exist yet. */
function disposeOf(world: World): (() => void) | undefined {
  const d = (world as unknown as MaybeDisposable).dispose;
  return typeof d === 'function' ? d.bind(world) : undefined;
}

const HAS_DISPOSE = disposeOf(new World()) !== undefined;

const MISSING_DISPOSE =
  'World.dispose() is not implemented in src/world.ts. This leak suite is written against the ' +
  'specified API (dispose(): tears the world down, releases every table buffer and the entity ' +
  'index, is idempotent, and leaves the World collectable). Nothing here papers over its absence.';

function requireDispose(world: World): () => void {
  const d = disposeOf(world);
  if (d === undefined) throw new Error(MISSING_DISPOSE);
  return d;
}

// ---------------------------------------------------------------------------- components
//
// Module scope, because component ids come from a module-global counter: creating components
// per cycle would grow every id-indexed array in every world and would be measuring the test,
// not the library. (That components are process-global and have no destroy() is by design.)

const posSchema = { x: f32, y: f32 } as const;
const velSchema = { dx: f32, dy: f32 } as const;
const wideSchema = { a: f32, b: f32, c: f32, d: f32, e: f32, f: f32, g: f32, h: f32 } as const;

const Position = component(posSchema, { name: 'LeakPosition' });
const Velocity = component(velSchema, { name: 'LeakVelocity' });
/** 32 bytes a row: makes the ArrayBuffer footprint of one world unmistakable at 20k rows. */
const Wide = component(wideSchema, { name: 'LeakWide' });
const Health = component({ hp: i16 }, { name: 'LeakHealth', enableable: true });
const Label = component({ text: str }, { name: 'LeakLabel' });
const Frozen = tag({ name: 'LeakFrozen' });

type PosCols = ColumnsOf<typeof posSchema>;
type VelCols = ColumnsOf<typeof velSchema>;

/** Module-level kernel identity: the trampoline cache is keyed on function identity. */
const integrate = (n: number, pos: PosCols, vel: VelCols): void => {
  const x = pos.x;
  const y = pos.y;
  const dx = vel.dx;
  const dy = vel.dy;
  for (let i = 0; i < n; i++) {
    x[i] += dx[i];
    y[i] += dy[i];
  }
};

const PV = [Position, Velocity] as const;

/** Query descriptors hoisted: `world.query` dedupes by descriptor, so the set of keys stays fixed. */
const DESC_MOVERS = { all: [Position, Velocity], none: [Frozen] } as const;
const DESC_WIDE = { all: [Wide] } as const;
const DESC_LABELLED = { all: [Label] } as const;

// ---------------------------------------------------------------------------- fixture

/** A class system, so dispose()'s onDestroy path and `addSystem` bookkeeping are exercised. */
class DecaySystem extends System {
  readonly movers = this.query(DESC_MOVERS);
  destroyed = 0;
  onUpdate(): void {
    for (let c = 0; c < this.movers.chunks.length; c++) {
      const chunk = this.movers.chunks[c];
      if (chunk.count === 0) continue;
      const hp = chunk.col(Health).hp;
      for (let r = 0; r < chunk.count; r++) hp[r] -= 1;
    }
  }
  onDestroy(): void {
    this.destroyed++;
  }
}

interface Fixture {
  world: World;
  /** A handful of live handles, for churn. */
  ids: number[];
}

/**
 * A world with everything a real one has attached to it: several archetype tables, cached
 * queries, a function system with a compiled trampoline, a class system, component hooks,
 * query enter/exit listeners and a string table.
 */
function buildWorld(entities: number): Fixture {
  const world = new World({ initialCapacity: 1024 });
  const main = world.archetype(Position, Velocity, Wide, Health);
  world.spawnMany(main, entities, (chunk, row, i) => {
    const p = chunk.col(Position);
    const v = chunk.col(Velocity);
    p.x[row] = i;
    p.y[row] = -i;
    v.dx[row] = 1;
    v.dy[row] = -1;
    chunk.col(Health).hp[row] = 100;
  });

  // A second table with a `str` column, and a third with a tag, so the world holds more than
  // one buffer and the string table has content.
  const labelled = world.archetype(Position, Label);
  world.spawnMany(labelled, 512, (chunk, row, i) => {
    chunk.col(Position).x[row] = i;
    chunk.col(Label).text[row] = world.strings.intern(`label-${i & 15}`);
  });
  const frozen = world.archetype(Position, Velocity, Frozen);
  world.spawnMany(frozen, 256);

  const movers = world.query(DESC_MOVERS);
  world.query(DESC_WIDE);
  world.query(DESC_LABELLED);

  // Listeners that close over per-world state (the shape most likely to pin a world alive).
  const seen = { added: 0, removed: 0, entered: 0, exited: 0 };
  world.onAdd(Position, () => {
    seen.added++;
  });
  world.onRemove(Position, () => {
    seen.removed++;
  });
  movers.onEnter(() => {
    seen.entered++;
  });
  movers.onExit(() => {
    seen.exited++;
  });

  world.system('integrate', { query: movers }, (q) => {
    q.forEachChunk(PV, integrate);
  });
  world.system('plain', { query: world.query(DESC_WIDE), order: 1 }, (q) => {
    for (let c = 0; c < q.chunks.length; c++) {
      const chunk = q.chunks[c];
      const col = chunk.col(Wide);
      for (let r = 0; r < chunk.count; r++) col.a[r] += 1;
    }
  });
  world.addSystem(DecaySystem, { order: 2 });

  const ids: number[] = [];
  for (let r = 0; r < 64 && r < main.count; r++) ids.push(main.entities[r]);
  return { world, ids };
}

/** Structural churn: moves rows between archetypes and allocates/frees entity indices. */
function churn(world: World, ids: number[], rounds: number): void {
  for (let i = 0; i < rounds; i++) {
    const e = ids[i % ids.length];
    if (!world.isAlive(e)) continue;
    world.add(e, Frozen);
    world.remove(e, Frozen);
    world.set(e, Position, { x: i, y: -i });
    const fresh = world.spawn([Position, Label]);
    world.set(fresh, Label, { text: `churn-${i & 7}` });
    world.destroy(fresh);
  }
}

/**
 * Leaves structural commands QUEUED but not applied, as a world torn down mid-tick would:
 * `_iterDepth > 0` is exactly the state a system body runs in, and the commands (one of them
 * holding a values object) stay in the buffer when it drops back to 0 without a flush.
 * dispose() must discard them without running them.
 */
function queuePending(world: World, ids: number[]): void {
  world._iterDepth = 1;
  for (let i = 0; i < 8 && i < ids.length; i++) {
    if (!world.isAlive(ids[i])) continue;
    world.add(ids[i], Label, { text: `pending-${i}` });
    world.destroy(ids[i]);
  }
  world._iterDepth = 0;
}

// ---------------------------------------------------------------------------- measurement

/** `global.gc` when the process ran with --expose-gc, else undefined. */
function gcOrNull(): (() => void) | undefined {
  const g = (globalThis as unknown as { gc?: () => void }).gc;
  return typeof g === 'function' ? g : undefined;
}

const GC = gcOrNull();
/** The measuring tests need --expose-gc (`npm run test:leak`); they skip with a note without it. */
const measureTest = GC === undefined ? test.skip : test;

// eslint-disable-next-line no-console
const note = (s: string): void => console.log(`[leak] ${s}`);

interface Sample {
  heapUsed: number;
  arrayBuffers: number;
}

function usage(): Sample {
  const m = process.memoryUsage();
  return { heapUsed: m.heapUsed, arrayBuffers: m.arrayBuffers };
}

/**
 * Drives the heap to a quiet state. Yields to the macrotask queue first and again between
 * collections: a WeakRef is not cleared during the turn it was created in, finalizers run on
 * a later turn, and freeing the backing store of an ArrayBuffer can take a second cycle after
 * the JS wrapper dies.
 */
async function settle(gc: () => void): Promise<void> {
  await new Promise<void>((resolve) => setImmediate(resolve));
  gc();
  gc();
  await new Promise<void>((resolve) => setImmediate(resolve));
  gc();
}

/** Least-squares slope per cycle: the shape a leak has and harness noise does not. */
function perCycle(values: readonly number[]): number {
  const n = values.length;
  if (n < 2) return 0;
  let sx = 0;
  let sy = 0;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sx += i;
    sy += values[i];
    sxx += i * i;
    sxy += i * values[i];
  }
  const denom = n * sxx - sx * sx;
  return denom === 0 ? 0 : (n * sxy - sx * sy) / denom;
}

/** Largest excursion above baseline across the run (a slope near 0 with a step is still a leak). */
function maxExcess(values: readonly number[], baseline: number): number {
  let worst = 0;
  for (let i = 0; i < values.length; i++) if (values[i] - baseline > worst) worst = values[i] - baseline;
  return worst;
}

interface CycleResult {
  heapPerCycle: number;
  bufPerCycle: number;
  heapWorst: number;
  bufWorst: number;
  heapTotal: number;
  bufTotal: number;
  samples: number;
  /** Per-cycle readings relative to baseline, printed when COZYECS_LEAK_VERBOSE=1. */
  heap: readonly number[];
  bufs: readonly number[];
}

function summarize(heap: readonly number[], bufs: readonly number[], base: Sample): CycleResult {
  return {
    heapPerCycle: perCycle(heap),
    bufPerCycle: perCycle(bufs),
    heapWorst: maxExcess(heap, base.heapUsed),
    bufWorst: maxExcess(bufs, base.arrayBuffers),
    heapTotal: heap[heap.length - 1] - base.heapUsed,
    bufTotal: bufs[bufs.length - 1] - base.arrayBuffers,
    samples: heap.length,
    heap: heap.map((v) => v - base.heapUsed),
    bufs: bufs.map((v) => v - base.arrayBuffers),
  };
}

/**
 * Runs `body` `cycles` times, collecting after each, and reports both meters per cycle.
 * `warmup` cycles run before the baseline is taken: the first world of the process compiles
 * the row loops and trampolines, tiers up the system bodies and grows the command buffer, and
 * none of that repeats per cycle.
 */
async function measureCycles(gc: () => void, cycles: number, warmup: number, body: () => void): Promise<CycleResult> {
  for (let i = 0; i < warmup; i++) body();
  await settle(gc);
  const base = usage();
  const heap: number[] = [];
  const bufs: number[] = [];
  for (let i = 0; i < cycles; i++) {
    body();
    await settle(gc);
    const u = usage();
    heap.push(u.heapUsed);
    bufs.push(u.arrayBuffers);
  }
  return summarize(heap, bufs, base);
}

const kb = (bytes: number): string => `${(bytes / 1024).toFixed(1)} KiB`;

/** Collected for the summary table printed after the run. */
const REPORT: { scenario: string; heapPerCycle: number; bufPerCycle: number; verdict: string }[] = [];

function record(scenario: string, r: CycleResult, verdict: string): void {
  REPORT.push({ scenario, heapPerCycle: r.heapPerCycle, bufPerCycle: r.bufPerCycle, verdict });
  if (VERBOSE) {
    note(`${scenario}: arrayBuffers over baseline (KiB) ${r.bufs.map((v) => (v / 1024).toFixed(0)).join(' ')}`);
    note(`${scenario}: heapUsed over baseline (KiB)     ${r.heap.map((v) => (v / 1024).toFixed(0)).join(' ')}`);
  }
  note(
    `${scenario}: arrayBuffers ${r.bufPerCycle.toFixed(0)} B/cycle (worst +${kb(r.bufWorst)}, ` +
      `total ${kb(r.bufTotal)}), heapUsed ${r.heapPerCycle.toFixed(0)} B/cycle ` +
      `(worst +${kb(r.heapWorst)}, total ${kb(r.heapTotal)}) over ${r.samples} cycles`,
  );
}

afterAll(() => {
  if (REPORT.length === 0) return;
  note('--- bytes per cycle ------------------------------------------------');
  for (const row of REPORT) {
    note(
      `${row.scenario.padEnd(34)} arrayBuffers ${row.bufPerCycle.toFixed(0).padStart(10)} B  ` +
        `heapUsed ${row.heapPerCycle.toFixed(0).padStart(10)} B  ${row.verdict}`,
    );
  }
});

// ---------------------------------------------------------------------------- budget
//
// 20k entities in the main table: 20000 rows x (8 + 8 + 32 + 2 + 1 enabled + 4 entities) = ~1.1 MB
// of live data, in a buffer grown by doubling from 1024 (so ~1.8 MB reserved), plus the two
// smaller tables and the entity index. One leaked world is therefore ~2 MB of arrayBuffers --
// two orders of magnitude above the tolerances below, and 20 leaked worlds are ~40 MB, well
// inside the memory budget this suite is allowed.

/**
 * Sizes are env-tunable so a suspicious slope can be re-measured over more cycles without
 * editing the file: a real leak keeps its slope at 20 cycles and at 100, while V8's own one-off
 * bookkeeping saturates. `COZYECS_LEAK_VERBOSE=1` prints every per-cycle reading.
 */
function envInt(name: string, fallback: number): number {
  const raw = process.env[name];
  const n = raw === undefined ? Number.NaN : Number.parseInt(raw, 10);
  return n > 0 ? n : fallback;
}

const VERBOSE = process.env.COZYECS_LEAK_VERBOSE === '1';
const ENTITIES = envInt('COZYECS_LEAK_ENTITIES', 20_000);
const CYCLES = envInt('COZYECS_LEAK_CYCLES', 20);
/** Blocks of churn in scenarios 4 and 5 (each block is thousands of operations). */
const BLOCKS = envInt('COZYECS_LEAK_BLOCKS', 10);
const WARMUP_CYCLES = 3;
const TICKS_PER_CYCLE = 12;

/**
 * Tolerances. One leaked world is ~2 MB of arrayBuffers and ~200 KB of heap, so 64 KiB/cycle
 * cannot hide one: it is 3% of a leaked table set. heapUsed is given more room than
 * arrayBuffers because V8's own bookkeeping (compilation cache entries for the per-world
 * `new Function` row loops, feedback vectors, string internalization) lands there and is not
 * CozyECS state; the ArrayBuffer meter is the one with no such excuse.
 */
const MAX_BUFFER_BYTES_PER_CYCLE = 64 * 1024;
const MAX_HEAP_BYTES_PER_CYCLE = 192 * 1024;
/** A single excursion is allowed one world's worth of lag, no more. */
const MAX_BUFFER_EXCURSION = 4 * 1024 * 1024;

const TIMEOUT = 180_000;

/** One build -> run -> (dispose) -> drop cycle. Nothing may escape this function. */
function cycle(withDispose: boolean): void {
  const fixture = buildWorld(ENTITIES);
  const world = fixture.world;
  for (let t = 0; t < TICKS_PER_CYCLE; t++) world.update(1 / 60);
  churn(world, fixture.ids, 200);
  world.flush();
  queuePending(world, fixture.ids);
  if (withDispose) requireDispose(world)();
}

// ---------------------------------------------------------------------------- 1. dispose + drop

describe('1. build -> run -> dispose -> drop', () => {
  measureTest(
    `returns both meters to baseline over ${CYCLES} cycles`,
    async () => {
      const gc = GC as () => void;
      if (!HAS_DISPOSE) {
        // Fail with the reason, rather than reporting a tolerance met by a method that is absent.
        throw new Error(MISSING_DISPOSE);
      }
      const r = await measureCycles(gc, CYCLES, WARMUP_CYCLES, () => cycle(true));
      record('1. dispose + drop', r, r.bufPerCycle < MAX_BUFFER_BYTES_PER_CYCLE ? 'flat' : 'LEAK');
      expect(r.bufPerCycle).toBeLessThan(MAX_BUFFER_BYTES_PER_CYCLE);
      expect(r.heapPerCycle).toBeLessThan(MAX_HEAP_BYTES_PER_CYCLE);
      expect(r.bufWorst).toBeLessThan(MAX_BUFFER_EXCURSION);
    },
    TIMEOUT,
  );

  measureTest(
    'releases the table buffers synchronously, before anything is collected',
    async () => {
      const gc = GC as () => void;
      const fixture = buildWorld(ENTITIES);
      const world = fixture.world;
      world.update(1 / 60);
      const before = world.memory();
      expect(before.tables.reserved).toBeGreaterThan(1_000_000);
      const live = usage();
      requireDispose(world)();
      // No gc(): dispose() deflates the tables itself, so the ArrayBuffer bytes must be gone
      // from the process as soon as the (now unreferenced) buffers are collected, and the
      // world's own accounting must report nothing reserved immediately.
      const after = world.memory();
      expect(after.tables.reserved).toBe(0);
      expect(after.total).toBe(0);
      await settle(gc);
      const freed = live.arrayBuffers - usage().arrayBuffers;
      note(`single dispose(): world reported ${kb(before.total)} reserved, process gave back ${kb(freed)}`);
      expect(freed).toBeGreaterThan(before.tables.reserved * 0.5);
    },
    TIMEOUT,
  );
});

// ---------------------------------------------------------------------------- 2. no dispose

describe('2. build -> run -> drop, without dispose()', () => {
  measureTest(
    `a dropped world is collectable on its own over ${CYCLES} cycles`,
    async () => {
      const gc = GC as () => void;
      const r = await measureCycles(gc, CYCLES, WARMUP_CYCLES, () => cycle(false));
      const verdict = r.bufPerCycle < MAX_BUFFER_BYTES_PER_CYCLE ? 'collectable' : 'RETAINED';
      record('2. drop, no dispose', r, verdict);
      // If this fails, a module-level reference is pinning dropped worlds: that IS the leak the
      // user reported, and the retaining path must be chased before anything else.
      expect(r.bufPerCycle).toBeLessThan(MAX_BUFFER_BYTES_PER_CYCLE);
      expect(r.heapPerCycle).toBeLessThan(MAX_HEAP_BYTES_PER_CYCLE);
      expect(r.bufWorst).toBeLessThan(MAX_BUFFER_EXCURSION);
    },
    TIMEOUT,
  );
});

// ---------------------------------------------------------------------------- 3. WeakRef proof

interface Refs {
  world: WeakRef<World>;
  archetype: WeakRef<Archetype>;
  buffer: WeakRef<object>;
  reservedBytes: number;
}

/** Builds a world, keeps only weak references to it, its main table and that table's buffer. */
function weakRefsFor(withDispose: boolean): Refs {
  const fixture = buildWorld(4096);
  const world = fixture.world;
  for (let t = 0; t < 4; t++) world.update(1 / 60);
  const main = world.archetype(Position, Velocity, Wide, Health);
  const reservedBytes = main.buffer.byteLength;
  const refs: Refs = {
    world: new WeakRef(world),
    archetype: new WeakRef(main),
    // The raw backing store, not a view: the bytes `process.memoryUsage().arrayBuffers` counts.
    buffer: new WeakRef(main.buffer as object),
    reservedBytes,
  };
  if (withDispose) requireDispose(world)();
  return refs;
}

describe('3. WeakRef proof', () => {
  measureTest(
    'after dispose() and dropping the reference, World, Archetype and the ArrayBuffer are gone',
    async () => {
      const gc = GC as () => void;
      if (!HAS_DISPOSE) throw new Error(MISSING_DISPOSE);
      const refs = weakRefsFor(true);
      expect(refs.reservedBytes).toBeGreaterThan(0);
      await settle(gc);
      await settle(gc);
      expect(refs.world.deref()).toBeUndefined();
      expect(refs.archetype.deref()).toBeUndefined();
      expect(refs.buffer.deref()).toBeUndefined();
      note(`WeakRef (with dispose): World, Archetype and its ${kb(refs.reservedBytes)} buffer all collected`);
    },
    TIMEOUT,
  );

  measureTest(
    'without dispose(), dropping the reference alone is enough (no module-level registry)',
    async () => {
      const gc = GC as () => void;
      const refs = weakRefsFor(false);
      await settle(gc);
      await settle(gc);
      const alive = [
        refs.world.deref() !== undefined ? 'World' : '',
        refs.archetype.deref() !== undefined ? 'Archetype' : '',
        refs.buffer.deref() !== undefined ? 'ArrayBuffer' : '',
      ].filter((s) => s !== '');
      note(
        alive.length === 0
          ? 'WeakRef (no dispose): World, Archetype and buffer all collected -- nothing in src pins a dropped world'
          : `WeakRef (no dispose): STILL REACHABLE: ${alive.join(', ')} -- a module-level reference retains them`,
      );
      expect(alive).toEqual([]);
    },
    TIMEOUT,
  );
});

// ---------------------------------------------------------------------------- 4. churn in one world

describe('4. churn inside one long-lived world', () => {
  measureTest(
    'spawn/destroy/add/remove churn then compact() comes back to baseline',
    async () => {
      const gc = GC as () => void;
      const fixture = buildWorld(2000);
      const world = fixture.world;
      const ids = fixture.ids;

      // Baseline: the world at rest, after the first churn round has created every archetype
      // the churn can reach and compiled every loop it uses.
      churn(world, ids, 200);
      world.update(1 / 60);
      world.compact({ minBytes: 0 });
      await settle(gc);
      const base = usage();
      const baseMem = world.memory();
      const baseArchetypes = baseMem.archetypes.length;
      const baseStrings = baseMem.strings.count;

      const ROUNDS = 2000;
      const heap: number[] = [];
      const bufs: number[] = [];
      for (let b = 0; b < BLOCKS; b++) {
        churn(world, ids, ROUNDS);
        for (let t = 0; t < 4; t++) world.update(1 / 60);
        world.compact({ minBytes: 0 });
        await settle(gc);
        const u = usage();
        heap.push(u.heapUsed);
        bufs.push(u.arrayBuffers);
      }

      const r = summarize(heap, bufs, base);
      record(`4. churn x${BLOCKS * ROUNDS} + compact`, r, r.bufPerCycle < MAX_BUFFER_BYTES_PER_CYCLE ? 'flat' : 'LEAK');

      const after = world.memory();
      // The archetype registry is bounded by the component sets the churn actually reaches.
      expect(after.archetypes.length).toBe(baseArchetypes);
      // `churn` interns 8 distinct strings; the table is append-only, so it may not grow past them.
      expect(after.strings.count).toBe(baseStrings);
      expect(r.bufPerCycle).toBeLessThan(MAX_BUFFER_BYTES_PER_CYCLE);
      expect(r.heapPerCycle).toBeLessThan(MAX_HEAP_BYTES_PER_CYCLE);
      // 20000 churn rounds must not inflate the tables: compact() gives the slack back.
      expect(after.tables.reserved).toBeLessThan(baseMem.tables.reserved + 256 * 1024);
    },
    TIMEOUT,
  );

  measureTest(
    'the string table grows only with distinct strings, and compact({strings:true}) gives them back',
    async () => {
      const gc = GC as () => void;
      const world = new World({ initialCapacity: 256 });
      const arch = world.archetype(Label);
      const keep = world.spawn(arch);
      world.set(keep, Label, { text: 'kept' });

      // Append-only by documented design: N distinct strings => N new ids, and no more.
      const DISTINCT = 5000;
      const before = world.strings.size;
      for (let i = 0; i < DISTINCT; i++) {
        const e = world.spawn(arch);
        world.set(e, Label, { text: `unique-${i}` });
        world.destroy(e);
      }
      expect(world.strings.size).toBe(before + DISTINCT);
      // Re-interning the same strings adds nothing.
      for (let i = 0; i < DISTINCT; i++) world.strings.intern(`unique-${i}`);
      expect(world.strings.size).toBe(before + DISTINCT);

      const stats = world.compact({ strings: true, minBytes: 0 });
      expect(stats.strings).toBeDefined();
      // Only '' and the one live row's string survive.
      expect(world.strings.size).toBe(2);
      expect(world.get(keep, Label).text).toBe('kept');
      await settle(gc);
      note(`string table: ${DISTINCT} dead strings interned, compact({strings:true}) left ${world.strings.size}`);
    },
    TIMEOUT,
  );
});

// ---------------------------------------------------------------------------- 5. registration churn

/**
 * Registers `rounds` function systems, each closing over 64 KiB of ArrayBuffer, runs a tick and
 * removes them again. Returns weak references to every handle and every buffer: after this
 * function returns, nothing but `world` should be able to reach any of them.
 */
function registerAndRemoveSystems(world: World, rounds: number): WeakRef<object>[] {
  const refs: WeakRef<object>[] = [];
  for (let i = 0; i < rounds; i++) {
    const big = new Float64Array(8192); // 64 KiB a system, visible in arrayBuffers
    const handle = world.system('tmp', { query: DESC_MOVERS }, () => {
      big[0] += 1;
    });
    world.update(1 / 60);
    refs.push(new WeakRef(handle as unknown as object));
    refs.push(new WeakRef(big.buffer as object));
    world.removeSystem(handle);
  }
  return refs;
}

describe('5. system / query / listener churn', () => {
  measureTest(
    'registering and removing thousands of times accumulates nothing',
    async () => {
      const gc = GC as () => void;
      const fixture = buildWorld(1000);
      const world = fixture.world;
      const ids = fixture.ids;
      const movers = world.query(DESC_MOVERS);

      const baseQueries = world._queries.length;
      const baseSystems = world._scheduler.group('update').length;
      const baseHooks = world._hookCount;
      const baseEventQueries = world._eventQueries.length;

      const tick: EntityCallback = () => {};
      const ROUNDS = 500;

      const round = (i: number): void => {
        const handles: SystemHandle[] = [];
        handles.push(world.system('churn-fn', { query: DESC_MOVERS }, () => {}));
        handles.push(world.system('churn-wide', { query: DESC_WIDE, group: 'churn', order: i & 3 }, () => {}));
        const cls = world.addSystem(DecaySystem, { group: 'churn' });
        const unsubs = [
          world.onAdd(Position, tick),
          world.onRemove(Position, tick),
          world.onAdd(Label, tick),
          movers.onEnter(tick),
          movers.onExit(tick),
          world.query(DESC_LABELLED).onEnter(tick),
        ];
        // Exercise them, so listener arrays and scheduler groups are actually walked.
        world.update(1 / 60, 'churn');
        const e = world.spawn([Position, Velocity]);
        world.destroy(e);
        for (let u = 0; u < unsubs.length; u++) unsubs[u]();
        for (let h = 0; h < handles.length; h++) world.removeSystem(handles[h]);
        world.removeSystem(cls);
        if ((i & 63) === 0) churn(world, ids, 8);
      };

      for (let i = 0; i < 100; i++) round(i); // warm-up: first compiles, first group arrays
      await settle(gc);
      const base = usage();
      const heap: number[] = [];
      const bufs: number[] = [];
      for (let b = 0; b < BLOCKS; b++) {
        for (let i = 0; i < ROUNDS; i++) round(b * ROUNDS + i);
        await settle(gc);
        const u = usage();
        heap.push(u.heapUsed);
        bufs.push(u.arrayBuffers);
      }

      const r = summarize(heap, bufs, base);
      record(
        `5. registration churn x${BLOCKS * ROUNDS}`,
        r,
        r.heapPerCycle < MAX_HEAP_BYTES_PER_CYCLE ? 'flat' : 'LEAK',
      );

      // Bookkeeping must be exactly back where it started, not merely "small".
      expect(world._scheduler.group('update').length).toBe(baseSystems);
      expect(world._scheduler.group('churn').length).toBe(0);
      expect(world._hookCount).toBe(baseHooks);
      expect(world._eventQueries.length).toBe(baseEventQueries);
      expect(movers._enter.length).toBe(1); // the fixture's own listener, nothing else
      expect(movers._exit.length).toBe(1);
      // Queries are cached per descriptor and never removed (documented): the bound is the
      // number of DISTINCT descriptors used, which this loop does not grow.
      expect(world._queries.length).toBe(baseQueries);
      expect(r.heapPerCycle).toBeLessThan(MAX_HEAP_BYTES_PER_CYCLE);
      expect(r.bufPerCycle).toBeLessThan(MAX_BUFFER_BYTES_PER_CYCLE);
    },
    TIMEOUT,
  );

  measureTest(
    'a removed system and its closures are collectable',
    async () => {
      const gc = GC as () => void;
      const world = new World({ initialCapacity: 64 });
      world.spawnMany(world.archetype(Position, Velocity), 128);
      // The loop lives in its own function on purpose: locals of the LAST iteration stay
      // reachable from a live stack frame, which would report one handle and one buffer as
      // retained even in a perfectly clean library. Returning discards that frame.
      const refs = registerAndRemoveSystems(world, 32);
      await settle(gc);
      await settle(gc);
      const alive = refs.filter((r) => r.deref() !== undefined).length;
      note(`removed systems: ${alive} of ${refs.length} handles/buffers still reachable (expected 0)`);
      expect(alive).toBe(0);
    },
    TIMEOUT,
  );
});

// ---------------------------------------------------------------------------- post-dispose contract
//
// The rule asserted here (stated in the report for the integrator to reconcile):
//   MUTATING calls throw an Error naming dispose(); READ-ONLY calls stay safe and answer as if
//   the world were empty; dispose() itself is idempotent.

describe('post-dispose contract', () => {
  test('dispose() discards queued commands without running them', () => {
    const fixture = buildWorld(256);
    const world = fixture.world;
    let removed = 0;
    world.onRemove(Position, () => {
      removed++;
    });
    queuePending(world, fixture.ids);
    expect(world._commands.length).toBeGreaterThan(0);
    const before = removed;
    requireDispose(world)();
    expect(removed).toBe(before);
    expect(world._commands.length).toBe(0);
  });

  test('class systems get onDestroy() exactly once', () => {
    const world = new World({ initialCapacity: 32 });
    const sys = world.addSystem(DecaySystem);
    const dispose = requireDispose(world);
    dispose();
    expect(sys.destroyed).toBe(1);
    dispose(); // idempotent
    expect(sys.destroyed).toBe(1);
  });

  test('mutating calls throw and read-only calls stay safe', () => {
    const fixture = buildWorld(256);
    const world = fixture.world;
    const entity = fixture.ids[0];
    const movers = world.query(DESC_MOVERS);
    const arch = world.archetype(Position, Velocity, Wide, Health);
    requireDispose(world)();

    const naming = /dispose/i;
    expect(() => world.spawn([Position])).toThrow(naming);
    expect(() => world.spawnMany(arch, 1)).toThrow(naming);
    expect(() => world.add(entity, Frozen)).toThrow(naming);
    expect(() => world.remove(entity, Frozen)).toThrow(naming);
    expect(() => world.set(entity, Position, { x: 1 })).toThrow(naming);
    expect(() => world.update(1 / 60)).toThrow(naming);
    expect(() => world.compact()).toThrow(naming);
    expect(() => world.clear()).toThrow(naming);
    expect(() => world.system('late', {}, () => {})).toThrow(naming);
    expect(() => world.addSystem(DecaySystem)).toThrow(naming);

    // Read-only: safe, and answering as an empty world.
    expect(world.isAlive(entity)).toBe(false);
    const mem = world.memory();
    expect(mem.entities).toBe(0);
    expect(mem.tables.reserved).toBe(0);
    expect(mem.tables.used).toBe(0);
    expect(mem.total).toBe(0);
    expect(movers.count()).toBe(0);
  });
});
