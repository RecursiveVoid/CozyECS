import { describe, it, expect } from '@jest/globals';
import { World, component, tag, f32, f64, i32, u8, bool, str } from '../src/index';
import { capacityFor, nextCapacity } from '../src/archetype';
import type { ClearOptions, CompactOptions, CompactStats, WorldMemory } from '../src/world';

const Position = component({ x: f32, y: f32 }, { name: 'Position' });
const Velocity = component({ vx: f64, vy: f64 }, { name: 'Velocity' });
const Label = component({ text: str }, { name: 'Label' });
const Wide = component({ a: i32, b: u8, c: bool, d: f64 }, { name: 'Wide' });
const Fading = component({ t: f32 }, { name: 'Fading', enableable: true });
const Player = tag({ name: 'Player' });

/** Spawns `n` entities with Position + Velocity, seeded from their index. */
function seed(w: World, n: number): number[] {
  const arch = w.archetype(Position, Velocity);
  const ids: number[] = [];
  for (let i = 0; i < n; i++) {
    const e = w.spawn(arch);
    w.set(e, Position, { x: i, y: -i });
    w.set(e, Velocity, { vx: i * 2, vy: i * 3 });
    ids.push(e);
  }
  return ids;
}

/** Reads back every seeded entity and asserts its values survived. */
function expectSeedIntact(w: World, ids: number[]): void {
  for (let i = 0; i < ids.length; i++) {
    const e = ids[i];
    expect(w.isAlive(e)).toBe(true);
    expect(w.get(e, Position)).toEqual({ x: i, y: -i });
    expect(w.get(e, Velocity)).toEqual({ vx: i * 2, vy: i * 3 });
  }
}

// ---------------------------------------------------------------------------
// capacityFor
// ---------------------------------------------------------------------------

describe('capacityFor()', () => {
  it('returns the capacity a table grown one row at a time would have', () => {
    // Replays the growth policy from `initial` and compares.
    for (const initial of [1, 4, 64, 100]) {
      let cap = initial;
      for (let n = 0; n <= 5000; n++) {
        if (n > cap) cap = nextCapacity(cap);
        expect(capacityFor(n, initial)).toBe(cap);
      }
    }
  });

  it('never returns less than 1 and never less than n', () => {
    expect(capacityFor(0, 64)).toBe(64);
    expect(capacityFor(0, 0)).toBe(1);
    expect(capacityFor(1, 0)).toBe(1);
    expect(capacityFor(70000, 64)).toBeGreaterThanOrEqual(70000);
  });
});

// ---------------------------------------------------------------------------
// Archetype.shrinkToFit
// ---------------------------------------------------------------------------

describe('Archetype.shrinkToFit()', () => {
  it('preserves rows, handles, values and enabled bits', () => {
    const w = new World({ initialCapacity: 1024 });
    const arch = w.archetype(Position, Fading, Player);
    const ids: number[] = [];
    for (let i = 0; i < 10; i++) {
      const e = w.spawn(arch);
      w.set(e, Position, { x: i, y: i * 10 });
      w.set(e, Fading, { t: i / 2 });
      if (i % 2 === 0) w.enable(e, Fading, false);
      ids.push(e);
    }
    const before = arch.buffer;
    expect(arch.shrinkToFit()).toBe(true);
    expect(arch.capacity).toBe(10);
    expect(arch.count).toBe(10);
    expect(arch.buffer).not.toBe(before);
    expect(arch.entities.length).toBe(10);
    for (let i = 0; i < 10; i++) {
      expect(arch.entities[i]).toBe(ids[i]);
      expect(w.get(ids[i], Position)).toEqual({ x: i, y: i * 10 });
      expect(w.get(ids[i], Fading)).toEqual({ t: i / 2 });
      expect(w.isEnabled(ids[i], Fading)).toBe(i % 2 !== 0);
    }
  });

  it('honours minCapacity and reports no-ops', () => {
    const w = new World({ initialCapacity: 64 });
    const arch = w.archetype(Position);
    for (let i = 0; i < 5; i++) w.spawn(arch);
    expect(arch.shrinkToFit(32)).toBe(true);
    expect(arch.capacity).toBe(32);
    expect(arch.shrinkToFit(32)).toBe(false);
    expect(arch.shrinkToFit(3)).toBe(true); // clamped up to count
    expect(arch.capacity).toBe(5);
  });

  it('deflates an empty table to a zero-length buffer and re-grows from it', () => {
    const w = new World({ initialCapacity: 64 });
    const arch = w.archetype(Position, Velocity);
    const ids = seed(w, 3);
    for (const e of ids) w.destroy(e);
    expect(arch.count).toBe(0);
    expect(arch.shrinkToFit(0)).toBe(true);
    expect(arch.capacity).toBe(0);
    expect(arch.buffer.byteLength).toBe(0);
    expect(arch.entities.length).toBe(0);
    expect(arch.col(Position).x.length).toBe(0);

    // nextCapacity(0) === 1, so pushRow re-grows a zero-capacity table.
    const again = seed(w, 40);
    expect(arch.count).toBe(40);
    expect(arch.capacity).toBeGreaterThanOrEqual(40);
    expectSeedIntact(w, again);
  });

  it('keeps using a SharedArrayBuffer when the world is shared', () => {
    if (typeof SharedArrayBuffer === 'undefined') return;
    const w = new World({ initialCapacity: 256, shared: true });
    const arch = w.archetype(Position);
    for (let i = 0; i < 4; i++) w.set(w.spawn(arch), Position, { x: i, y: i });
    expect(arch.shared).toBe(true);
    expect(arch.buffer).toBeInstanceOf(SharedArrayBuffer);
    expect(arch.shrinkToFit(0)).toBe(true);
    expect(arch.buffer).toBeInstanceOf(SharedArrayBuffer);
    expect(arch.col(Position).x[2]).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// World.memory
// ---------------------------------------------------------------------------

describe('World.memory()', () => {
  it('reports entities, per-archetype bytes and totals', () => {
    const w = new World({ initialCapacity: 16 });
    const arch = w.archetype(Position, Velocity);
    seed(w, 10);
    const m: WorldMemory = w.memory();
    expect(m.entities).toBe(10);
    expect(m.strings.count).toBe(1); // '' only
    expect(m.archetypes.length).toBe(w._archetypes.length);

    const row = m.archetypes.find((a) => a.id === arch.id)!;
    expect(row.count).toBe(10);
    expect(row.capacity).toBe(arch.capacity);
    expect(row.rowBytes).toBe(arch.rowBytes);
    expect(row.bytes).toBe(arch.buffer.byteLength);
    expect(row.components).toEqual(['Position', 'Velocity']);

    let reserved = 0;
    let used = 0;
    for (const a of m.archetypes) {
      reserved += a.bytes;
      used += a.count * a.rowBytes;
    }
    expect(m.tables.reserved).toBe(reserved);
    expect(m.tables.used).toBe(used);
    expect(m.tables.used).toBeLessThan(m.tables.reserved);
    expect(m.entityIndex).toBeGreaterThan(0);
    expect(m.total).toBe(m.tables.reserved + m.entityIndex);
  });

  it('tracks strings and shrinks after a compact', () => {
    const w = new World({ initialCapacity: 64 });
    const arch = w.archetype(Label);
    for (let i = 0; i < 200; i++) w.set(w.spawn(arch), Label, { text: `n${i}` });
    const before = w.memory();
    expect(before.strings.count).toBe(201);
    expect(before.tables.reserved).toBeGreaterThan(before.tables.used);

    w.compact({ minBytes: 0 });
    const after = w.memory();
    expect(after.tables.reserved).toBeLessThan(before.tables.reserved);
    expect(after.tables.used).toBe(before.tables.used);
    expect(arch.capacity).toBe(capacityFor(200, 64));
  });
});

// ---------------------------------------------------------------------------
// World.compact
// ---------------------------------------------------------------------------

describe('World.compact()', () => {
  it('deflates to the growth-policy capacity and preserves everything live', () => {
    const w = new World({ initialCapacity: 64 });
    const arch = w.archetype(Position, Velocity);
    const ids = seed(w, 300);
    // Destroy the tail, keeping [0, 100) intact (swapRemove from the end never moves them).
    for (let i = 299; i >= 100; i--) w.destroy(ids[i]);
    const kept = ids.slice(0, 100);
    const capBefore = arch.capacity;
    const bytesBefore = arch.buffer.byteLength;

    const stats: CompactStats = w.compact({ minBytes: 0 });
    expect(stats.archetypes).toBeGreaterThanOrEqual(1);
    expect(stats.bytesFreed).toBeGreaterThanOrEqual(bytesBefore - arch.buffer.byteLength);
    expect(stats.strings).toBeUndefined();
    expect(arch.capacity).toBe(capacityFor(100, 64));
    expect(arch.capacity).toBeLessThan(capBefore);
    expect(arch.count).toBe(100);
    expectSeedIntact(w, kept);
    expect(w.query({ all: [Position, Velocity] }).count()).toBe(100);
  });

  it('reports the bytes it actually freed', () => {
    const w = new World({ initialCapacity: 64 });
    const arch = w.archetype(Position, Velocity);
    const ids = seed(w, 300);
    for (let i = 299; i >= 100; i--) w.destroy(ids[i]);
    const before = w.memory().tables.reserved;
    const stats = w.compact({ minBytes: 0 });
    const after = w.memory().tables.reserved;
    expect(stats.bytesFreed).toBe(before - after);
    expect(stats.bytesFreed).toBeGreaterThan(0);
    expect(arch.count).toBe(100);
  });

  it('never grows a table and is idempotent', () => {
    const w = new World({ initialCapacity: 64 });
    const arch = w.archetype(Position);
    for (let i = 0; i < 300; i++) w.spawn(arch);
    const first = w.compact({ minBytes: 0 });
    const cap = arch.capacity;
    const second = w.compact({ minBytes: 0 });
    expect(second.archetypes).toBe(0);
    expect(second.bytesFreed).toBe(0);
    expect(arch.capacity).toBe(cap);
    expect(first.archetypes).toBeGreaterThanOrEqual(0);
  });

  it('skips tables whose slack is below minBytes (default 4096)', () => {
    const w = new World({ initialCapacity: 4 });
    const arch = w.archetype(Position); // rowBytes = 4 (entities) + 4 + 4 (x, y) = 12
    const ids: number[] = [];
    for (let i = 0; i < 300; i++) ids.push(w.spawn(arch));
    for (let i = 299; i >= 250; i--) w.destroy(ids[i]);
    const cap = arch.capacity;
    const target = capacityFor(250, 4);
    expect(target).toBeLessThan(cap);
    expect((cap - target) * arch.rowBytes).toBeLessThan(4096); // 256 * 12 = 3072

    expect(w.compact().archetypes).toBe(0); // default minBytes 4096: no churn
    expect(arch.capacity).toBe(cap);
    expect(w.compact({ minBytes: 3072 }).archetypes).toBeGreaterThanOrEqual(1);
    expect(arch.capacity).toBe(target);
  });

  it('keeps archetype ids, query chunk lists and transition edges valid', () => {
    const w = new World({ initialCapacity: 128 });
    const q = w.query({ all: [Position] });
    const chunksArray = q.chunks;
    const pv = w.archetype(Position, Velocity);
    const pvId = pv.id;
    const pvKey = pv.key;
    const ids = seed(w, 200);
    const edgeAdd = pv.edgesAdd.size;

    // Warm the add/remove transition edges (this also creates the +Player archetype), then compact.
    w.add(ids[0], Player);
    w.remove(ids[0], Player);
    expect(pv.edgesAdd.size).toBeGreaterThan(edgeAdd);
    const addEdge = pv.edgesAdd.get(Player.id)!;
    const archetypeIds = w._archetypes.map((a) => a.id);
    const chunkIds = q.chunks.map((c) => c.id);
    const chunkObjects = q.chunks.slice();

    w.compact({ minBytes: 0 });

    expect(w._archetypes[pvId]).toBe(pv);
    expect(pv.id).toBe(pvId);
    expect(pv.key).toBe(pvKey);
    expect(w._archetypes.map((a) => a.id)).toEqual(archetypeIds);
    expect(q.chunks).toBe(chunksArray); // the same array object, updated in place
    expect(q.chunks.map((c) => c.id)).toEqual(chunkIds);
    for (let i = 0; i < chunkObjects.length; i++) expect(q.chunks[i]).toBe(chunkObjects[i]);
    expect(pv.edgesAdd.get(Player.id)).toBe(addEdge);
    expect(q.count()).toBe(200);

    // Transitions still work through the cached edges after the buffers were replaced.
    w.add(ids[5], Player);
    expect(w.has(ids[5], Player)).toBe(true);
    expect(w.get(ids[5], Position)).toEqual({ x: 5, y: -5 });
    expect(q.count()).toBe(200);
  });

  it('lets forEach and forEachChunk run over compacted chunks', () => {
    const w = new World({ initialCapacity: 8 });
    const all = seed(w, 400);
    for (let i = 399; i >= 120; i--) w.destroy(all[i]);
    const ids = all.slice(0, 120);
    const q = w.query({ all: [Position, Velocity] });
    // ONE kernel identity, called repeatedly, so the query compiles trampolines that hold the
    // column arrays as constants: after a compact they must notice the reallocation and rebuild.
    let seen = 0;
    const kernel = (n: number, p: { x: Float32Array }, v: { vx: Float64Array }): void => {
      for (let r = 0; r < n; r++) {
        p.x[r] += v.vx[r] * 0;
        seen++;
      }
    };
    for (let k = 0; k < 24; k++) q.forEachChunk([Position, Velocity], kernel);
    expect(seen).toBe(120 * 24);
    const buffer = w.archetype(Position, Velocity).buffer;
    w.compact({ minBytes: 0 });
    expect(w.archetype(Position, Velocity).buffer).not.toBe(buffer);
    seen = 0;
    for (let k = 0; k < 24; k++) q.forEachChunk([Position, Velocity], kernel);
    expect(seen).toBe(120 * 24);
    let rows = 0;
    const rowFn = (): void => { rows++; };
    for (let k = 0; k < 24; k++) { rows = 0; q.forEach([Position], rowFn); }
    expect(rows).toBe(120);
    expectSeedIntact(w, ids);
  });

  it('spawning after a compact works and re-grows the table', () => {
    const w = new World({ initialCapacity: 64 });
    const arch = w.archetype(Position, Velocity);
    const ids = seed(w, 300);
    for (const e of ids) w.destroy(e);
    w.compact({ minBytes: 0 });
    expect(arch.capacity).toBe(0);
    const fresh = seed(w, 500);
    expect(arch.count).toBe(500);
    expectSeedIntact(w, fresh);
    expect(w.query({ all: [Position, Velocity] }).count()).toBe(500);
  });

  it('throws during iteration, during a flush and with queued commands', () => {
    const w = new World({ initialCapacity: 64 });
    seed(w, 4);
    const q = w.query({ all: [Position] });
    let thrownCompact: unknown = null;
    let thrownClear: unknown = null;
    q.forEach([Position], () => {
      try { w.compact(); } catch (err) { thrownCompact = err; }
      try { w.clear(); } catch (err) { thrownClear = err; }
    });
    expect(String(thrownCompact)).toContain('world.compact() cannot run during iteration; call it between ticks.');
    expect(String(thrownClear)).toContain('world.clear() cannot run during iteration; call it between ticks.');

    // Queued commands outside iteration: the same guard, naming flush().
    const e = w.spawn(w.archetype(Position));
    w._iterDepth = 1;
    w.destroy(e);
    w._iterDepth = 0;
    expect(w._commands.length).toBe(1);
    expect(() => w.compact()).toThrow('cannot run during iteration; call it between ticks.');
    expect(() => w.compact()).toThrow('call world.flush() first');
    expect(() => w.clear()).toThrow('call world.flush() first');
    w.flush();
    expect(() => w.compact()).not.toThrow();
  });

  it('a system body cannot compact', () => {
    const w = new World({ initialCapacity: 64 });
    seed(w, 4);
    let err: unknown = null;
    w.system('bad', { query: { all: [Position] } }, () => {
      try { w.compact(); } catch (e) { err = e; }
    });
    w.update(1 / 60);
    expect(String(err)).toContain('cannot run during iteration');
  });
});

// ---------------------------------------------------------------------------
// World.compact({ strings: true })
// ---------------------------------------------------------------------------

describe('World.compact({ strings: true })', () => {
  it('drops unreferenced strings and rewrites the columns', () => {
    const w = new World({ initialCapacity: 64 });
    const arch = w.archetype(Label);
    const ids: number[] = [];
    for (let i = 0; i < 50; i++) {
      const e = w.spawn(arch);
      w.set(e, Label, { text: `tag-${i}` });
      ids.push(e);
    }
    expect(w.strings.size).toBe(51);
    // Destroy the odd half; swapRemove keeps the survivors' values with their handles.
    const survivors: number[] = [];
    const texts = new Map<number, string>();
    for (let i = 0; i < 50; i++) {
      if (i % 2 === 0) { survivors.push(ids[i]); texts.set(ids[i], `tag-${i}`); }
      else w.destroy(ids[i]);
    }
    const stats = w.compact({ minBytes: 0, strings: true });
    expect(stats.strings).toEqual({ before: 51, after: 26 }); // '' + 25 survivors
    expect(w.strings.size).toBe(26);
    for (const e of survivors) expect(w.get(e, Label)!.text).toBe(texts.get(e));
    expect(arch.count).toBe(25);

    // The table still interns and resolves normally; a dropped string gets a fresh id.
    const fresh = w.strings.intern('tag-1');
    expect(w.strings.size).toBe(27);
    expect(w.strings.get(fresh)).toBe('tag-1');
    expect(w.strings.get(0)).toBe('');
    // Re-interning a survivor returns its (new) live id, and reading it back round-trips.
    const live = w.strings.intern('tag-0');
    expect(w.strings.get(live)).toBe('tag-0');
    expect(w.strings.size).toBe(27);
  });

  it('keeps strings referenced by rows in other archetypes', () => {
    const w = new World({ initialCapacity: 64 });
    const a = w.archetype(Label);
    const b = w.archetype(Label, Player);
    const keepA = w.spawn(a);
    w.set(keepA, Label, { text: 'alpha' });
    const keepB = w.spawn(b);
    w.set(keepB, Label, { text: 'beta' });
    const dead = w.spawn(a);
    w.set(dead, Label, { text: 'gamma' });
    w.destroy(dead);

    const stats = w.compact({ minBytes: 0, strings: true });
    expect(stats.strings!.before).toBe(4);
    expect(stats.strings!.after).toBe(3);
    expect(w.get(keepA, Label)!.text).toBe('alpha');
    expect(w.get(keepB, Label)!.text).toBe('beta');
  });

  it('leaves the empty-string id at 0 and handles a world with no strings', () => {
    const w = new World({ initialCapacity: 64 });
    seed(w, 10);
    const stats = w.compact({ minBytes: 0, strings: true });
    expect(stats.strings).toEqual({ before: 1, after: 1 });
    expect(w.strings.get(0)).toBe('');
  });
});

// ---------------------------------------------------------------------------
// World.clear
// ---------------------------------------------------------------------------

describe('World.clear()', () => {
  it('empties the world, kills old handles and keeps spawning working', () => {
    const w = new World({ initialCapacity: 64 });
    const arch = w.archetype(Position, Velocity);
    const q = w.query({ all: [Position, Velocity] });
    const ids = seed(w, 250);
    const chunksArray = q.chunks;

    w.clear();

    expect(q.count()).toBe(0);
    expect(arch.count).toBe(0);
    expect(arch.capacity).toBe(0);
    expect(arch.buffer.byteLength).toBe(0);
    expect(w.memory().entities).toBe(0);
    expect(w.memory().tables.used).toBe(0);
    for (const e of ids) {
      expect(w.isAlive(e)).toBe(false);
      expect(w.has(e, Position)).toBe(false);
      expect(w.get(e, Position)).toBeUndefined();
    }
    expect(q.chunks).toBe(chunksArray);

    const again = seed(w, 100);
    expect(q.count()).toBe(100);
    expectSeedIntact(w, again);
    // Reused indices carry bumped generations, so no old handle aliases a new entity.
    for (const e of ids) expect(again.indexOf(e)).toBe(-1);
    for (const e of ids) expect(w.isAlive(e)).toBe(false);
  });

  it('bumps generations for pending and already-free indices too', () => {
    const w = new World({ initialCapacity: 64 });
    const arch = w.archetype(Position);
    const a = w.spawn(arch);
    const b = w.spawn(arch);
    w.destroy(b); // b's index goes back on the free list before the clear
    w.clear();
    expect(w.isAlive(a)).toBe(false);
    expect(w.isAlive(b)).toBe(false);
    const fresh: number[] = [];
    for (let i = 0; i < 4; i++) fresh.push(w.spawn(arch));
    expect(fresh.indexOf(a)).toBe(-1);
    expect(fresh.indexOf(b)).toBe(-1);
    for (const e of fresh) expect(w.isAlive(e)).toBe(true);
  });

  it('{ compact: false } keeps the capacities for a refill', () => {
    const w = new World({ initialCapacity: 64 });
    const arch = w.archetype(Position, Velocity);
    seed(w, 300);
    const cap = arch.capacity;
    const bytes = arch.buffer;
    w.clear({ compact: false });
    expect(arch.count).toBe(0);
    expect(arch.capacity).toBe(cap);
    expect(arch.buffer).toBe(bytes);
    const ids = seed(w, 300);
    expect(arch.capacity).toBe(cap); // no reallocation was needed
    expectSeedIntact(w, ids);
  });

  it('fires nothing by default', () => {
    const w = new World({ initialCapacity: 64 });
    const arch = w.archetype(Position, Velocity);
    seed(w, 20);
    const q = w.query({ all: [Position] });
    let removed = 0;
    let exited = 0;
    w.onRemove(Position, () => { removed++; });
    q.onExit(() => { exited++; });
    w.clear();
    expect(removed).toBe(0);
    expect(exited).toBe(0);
    expect(arch.count).toBe(0);
    expect(q.count()).toBe(0);
  });

  it('{ events: true } fires onRemove and query onExit once per entity', () => {
    const w = new World({ initialCapacity: 64 });
    const q = w.query({ all: [Position] });
    const ids = seed(w, 20);
    const removed: number[] = [];
    const exited: number[] = [];
    let addedDuringClear = 0;
    w.onRemove(Position, (e) => { removed.push(e); });
    w.onRemove(Velocity, (e) => { removed.push(e); });
    w.onAdd(Position, () => { addedDuringClear++; });
    q.onExit((e) => { exited.push(e); });
    w.clear({ events: true });
    expect(removed.length).toBe(40); // Position + Velocity per entity
    expect(exited.length).toBe(20);
    expect(exited.slice().sort((a, b) => a - b)).toEqual(ids.slice().sort((a, b) => a - b));
    expect(addedDuringClear).toBe(0);
    expect(q.count()).toBe(0);
    expect(w.memory().entities).toBe(0);
  });

  it('{ events: true } reports the entities as dead to its own listeners', () => {
    const w = new World({ initialCapacity: 64 });
    const ids = seed(w, 5);
    const alive: boolean[] = [];
    w.onRemove(Position, (e) => { alive.push(w.isAlive(e)); });
    w.clear({ events: true });
    expect(alive).toEqual([false, false, false, false, false]);
    expect(ids.length).toBe(5);
  });

  it('survives a listener that spawns during { events: true }', () => {
    const w = new World({ initialCapacity: 64 });
    const arch = w.archetype(Position, Velocity);
    seed(w, 10);
    let n = 0;
    const off = w.onRemove(Velocity, () => {
      if (n++ === 0) w.set(w.spawn(arch), Position, { x: 99, y: 99 });
    });
    w.clear({ events: true });
    off();
    expect(arch.count).toBe(1);
    expect(arch.col(Position).x[0]).toBe(99);
    expect(w.memory().entities).toBe(1);
    expect(arch.capacity).toBeGreaterThanOrEqual(1);
  });

  it('keeps the string table and lets a compact follow', () => {
    const w = new World({ initialCapacity: 64 });
    const arch = w.archetype(Label);
    w.set(w.spawn(arch), Label, { text: 'keep-me' });
    const size = w.strings.size;
    w.clear();
    expect(w.strings.size).toBe(size);
    const stats = w.compact({ minBytes: 0, strings: true });
    expect(stats.strings).toEqual({ before: size, after: 1 });
    const e = w.spawn(arch);
    w.set(e, Label, { text: 'keep-me' });
    expect(w.get(e, Label)!.text).toBe('keep-me');
  });

  it('works on a shared world and keeps SharedArrayBuffer tables', () => {
    if (typeof SharedArrayBuffer === 'undefined') return;
    const w = new World({ initialCapacity: 128, shared: true });
    const arch = w.archetype(Position);
    for (let i = 0; i < 50; i++) w.spawn(arch);
    w.clear();
    expect(arch.buffer).toBeInstanceOf(SharedArrayBuffer);
    expect(arch.buffer.byteLength).toBe(0);
    w.spawn(arch);
    expect(arch.buffer).toBeInstanceOf(SharedArrayBuffer);
    expect(arch.count).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// Option objects are plain and optional
// ---------------------------------------------------------------------------

describe('option handling', () => {
  it('accepts empty / undefined option objects', () => {
    const w = new World({ initialCapacity: 8 });
    seed(w, 3);
    const c: CompactOptions = {};
    const cl: ClearOptions = {};
    expect(() => w.compact(c)).not.toThrow();
    expect(() => w.compact(undefined)).not.toThrow();
    expect(() => w.clear(cl)).not.toThrow();
    expect(() => w.clear(undefined)).not.toThrow();
    expect(w.memory().entities).toBe(0);
  });

  it('treats a non-numeric minBytes as the default', () => {
    const w = new World({ initialCapacity: 64 });
    const arch = w.archetype(Position);
    for (let i = 0; i < 5; i++) w.spawn(arch);
    const cap = arch.capacity;
    expect(w.compact({ minBytes: NaN }).archetypes).toBe(0);
    expect(arch.capacity).toBe(cap);
  });
});

// ---------------------------------------------------------------------------
// Steady state allocates nothing
// ---------------------------------------------------------------------------

describe('steady-state ticks allocate nothing', () => {
  const gc = (globalThis as unknown as { gc?: () => void }).gc;
  // Needs --expose-gc; `npm run test:alloc` supplies it (see package.json).
  const maybeIt = typeof gc === 'function' ? it : it.skip;

  maybeIt('a steady tick over 10k entities retains nothing', () => {
    const forceGc = gc as () => void;
    const w = new World({ initialCapacity: 16384 });
    const arch = w.archetype(Position, Velocity, Wide);
    w.spawnMany(arch, 10000, (chunk, row, i) => {
      const p = chunk.col(Position);
      const v = chunk.col(Velocity);
      p.x[row] = i;
      p.y[row] = -i;
      v.vx[row] = 1;
      v.vy[row] = 2;
    });
    const q = w.query({ all: [Position, Velocity] });
    // One kernel identity, so the query compiles its trampolines once during the warm-up.
    const kernel = (n: number, p: { x: Float32Array; y: Float32Array }, v: { vx: Float64Array; vy: Float64Array }): void => {
      for (let r = 0; r < n; r++) {
        p.x[r] += v.vx[r] * (1 / 60);
        p.y[r] += v.vy[r] * (1 / 60);
      }
    };
    w.system('move', { query: q }, (query) => {
      query.forEachChunk([Position, Velocity], kernel);
    });

    const heap = (): number => {
      forceGc();
      forceGc();
      forceGc();
      return process.memoryUsage().heapUsed;
    };
    const run = (n: number): void => {
      for (let t = 0; t < n; t++) w.update(1 / 60);
    };

    // A tick that allocates would make the heap delta scale with the tick count, so measure a
    // short run and a 10x longer one: what is left is the harness's own GC noise, which does not.
    run(200); // warm-up: compiles the trampolines, grows the command buffer
    const a0 = heap();
    run(200);
    const short = heap() - a0;
    const a1 = heap();
    run(2000);
    const long = heap() - a1;

    expect(long).toBeLessThan(short + 512 * 1024);
    expect(long / 2000).toBeLessThan(64);
    expect(arch.count).toBe(10000);
    // 2400 ticks of vx = 1 at dt = 1/60, from x = 1 (entity index 1).
    expect(arch.col(Position).x[1]).toBeCloseTo(1 + 2400 / 60, 1);
  });
});

describe('reclaiming at scale', () => {
  it('compact() and clear() give the table bytes back', () => {
    const w = new World({ initialCapacity: 64 });
    const arch = w.archetype(Position, Velocity, Wide);
    w.spawnMany(arch, 50000);
    const full = w.memory();
    expect(full.tables.reserved).toBeGreaterThan(1_000_000);
    expect(full.entities).toBe(50000);

    const ids: number[] = [];
    for (let r = 0; r < arch.count; r++) ids.push(arch.entities[r]);
    for (let i = ids.length - 1; i >= 1000; i--) w.destroy(ids[i]);
    expect(arch.count).toBe(1000);
    const freed = w.compact().bytesFreed;
    expect(freed).toBeGreaterThan(1_000_000);
    expect(arch.capacity).toBe(capacityFor(1000, 64));
    const trimmed = w.memory();
    expect(trimmed.tables.reserved).toBe(full.tables.reserved - freed);
    expect(trimmed.tables.used).toBe(1000 * arch.rowBytes);

    w.clear();
    const empty = w.memory();
    expect(empty.tables.reserved).toBe(0);
    expect(empty.tables.used).toBe(0);
    expect(empty.entities).toBe(0);
    // The entity index is deliberately NEVER shrunk, so a refill reallocates nothing.
    expect(empty.entityIndex).toBeGreaterThanOrEqual(full.entityIndex);
    expect(empty.total).toBe(empty.entityIndex);

    // And the whole thing refills.
    w.spawnMany(arch, 50000);
    expect(arch.count).toBe(50000);
    expect(w.memory().entities).toBe(50000);
    expect(w.query({ all: [Position, Velocity] }).count()).toBe(50000);
  });
});
