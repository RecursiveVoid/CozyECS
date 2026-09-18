import { describe, it, expect } from '@jest/globals';
import { World, Archetype, component, tag, f32, f64, i8, i16, i32, u8, u16, u32, bool, str } from '../src/index';

const Position = component({ x: f32, y: f32 }, { name: 'Position' });
const Velocity = component({ vx: f64, vy: f64 }, { name: 'Velocity' });
const Mixed = component({ a: i8, b: i16, c: i32, d: u8, e: u16, f: u32, g: bool, h: str }, { name: 'Mixed' });
const Frozen = component({ t: f32 }, { name: 'Frozen', enableable: true });
const Player = tag({ name: 'Player' });
const Hidden = tag({ name: 'Hidden', enableable: true });

describe('world.archetype()', () => {
  it('get-or-create by key, independent of order and duplicates', () => {
    const w = new World();
    const a = w.archetype(Position, Velocity);
    const b = w.archetype(Velocity, Position, Velocity);
    expect(a).toBe(b);
    expect(a).toBeInstanceOf(Archetype);
    const ids = [Position.id, Velocity.id].sort((x, y) => x - y);
    expect(a.key).toBe(ids.join(','));
    expect(a.components.map((c) => c.id)).toEqual(ids);
  });

  it('empty archetype exists and is used for bare entities', () => {
    const w = new World();
    const empty = w.archetype();
    expect(empty.key).toBe('');
    expect(empty.components.length).toBe(0);
    const e = w.spawn();
    expect(empty.count).toBe(1);
    expect(empty.entities[0]).toBe(e);
    expect(w.spawn([])).toBeDefined();
    expect(empty.count).toBe(2);
  });

  it('has() and col()', () => {
    const w = new World();
    const a = w.archetype(Position, Player);
    expect(a.has(Position)).toBe(true);
    expect(a.has(Player)).toBe(true);
    expect(a.has(Velocity)).toBe(false);
    const p = a.col(Position);
    expect(p.x).toBeInstanceOf(Float32Array);
    expect(p.y).toBeInstanceOf(Float32Array);
    expect(a.col(Velocity)).toBeUndefined();
    expect(a.col(Player)).toBeUndefined();
  });

  it('allocates the right typed arrays for every field type', () => {
    const w = new World();
    const a = w.archetype(Mixed);
    const m = a.col(Mixed);
    expect(m.a).toBeInstanceOf(Int8Array);
    expect(m.b).toBeInstanceOf(Int16Array);
    expect(m.c).toBeInstanceOf(Int32Array);
    expect(m.d).toBeInstanceOf(Uint8Array);
    expect(m.e).toBeInstanceOf(Uint16Array);
    expect(m.f).toBeInstanceOf(Uint32Array);
    expect(m.g).toBeInstanceOf(Uint8Array);
    expect(m.h).toBeInstanceOf(Uint32Array);
  });

  it('respects initialCapacity option (default 64)', () => {
    expect(new World().archetype(Position).capacity).toBe(64);
    expect(new World({ initialCapacity: 4 }).archetype(Position).capacity).toBe(4);
  });

  it('enabledArray only for enableable components', () => {
    const w = new World();
    const a = w.archetype(Position, Frozen, Hidden);
    expect(a.enabledArray(Position)).toBeUndefined();
    expect(a.enabledArray(Frozen)).toBeInstanceOf(Uint8Array);
    expect(a.enabledArray(Hidden)).toBeInstanceOf(Uint8Array);
    expect(a.enabledArray(Player)).toBeUndefined();
  });
});

describe('Archetype row operations', () => {
  it('pushRow zero-inits fields and sets enabled=1', () => {
    const w = new World();
    const a = w.archetype(Position, Frozen);
    const r0 = a.pushRow(123);
    a.col(Position).x[r0] = 5;
    a.enabledArray(Frozen)![r0] = 0;
    a.swapRemove(r0);
    const r1 = a.pushRow(456);
    expect(r1).toBe(0);
    expect(a.col(Position).x[r1]).toBe(0);
    expect(a.enabledArray(Frozen)![r1]).toBe(1);
    expect(a.isEnabled(Frozen, r1)).toBe(true);
    expect(a.entities[r1]).toBe(456);
    expect(a.count).toBe(1);
  });

  it('swapRemove moves last row into removed slot and returns moved entity or -1', () => {
    const w = new World();
    const a = w.archetype(Position, Frozen);
    for (let i = 0; i < 3; i++) {
      const r = a.pushRow(100 + i);
      a.col(Position).x[r] = i * 10;
      a.enabledArray(Frozen)![r] = i === 2 ? 0 : 1;
    }
    expect(a.swapRemove(0)).toBe(102);
    expect(a.count).toBe(2);
    expect(a.entities[0]).toBe(102);
    expect(a.col(Position).x[0]).toBe(20);
    expect(a.isEnabled(Frozen, 0)).toBe(false);
    expect(a.swapRemove(1)).toBe(-1);
    expect(a.count).toBe(1);
  });

  it('copyRowTo copies shared fields and enabled bits', () => {
    const w = new World();
    const src = w.archetype(Position, Velocity, Frozen);
    const dst = w.archetype(Position, Frozen, Player);
    const r = src.pushRow(1);
    src.col(Position).x[r] = 3;
    src.col(Position).y[r] = 4;
    src.col(Velocity).vx[r] = 9;
    src.col(Frozen).t[r] = 7;
    src.enabledArray(Frozen)![r] = 0;
    const r2 = dst.pushRow(1);
    src.copyRowTo(r, dst, r2);
    expect(dst.col(Position).x[r2]).toBe(3);
    expect(dst.col(Position).y[r2]).toBe(4);
    expect(dst.col(Frozen).t[r2]).toBe(7);
    expect(dst.isEnabled(Frozen, r2)).toBe(false);
  });

  it('grow() doubles capacity preserving data', () => {
    const w = new World({ initialCapacity: 2 });
    const a = w.archetype(Position, Hidden);
    a.pushRow(1);
    a.col(Position).x[0] = 1.5;
    a.enabledArray(Hidden)![0] = 0;
    a.grow();
    expect(a.capacity).toBe(4);
    expect(a.col(Position).x.length).toBe(4);
    expect(a.entities.length).toBe(4);
    expect(a.col(Position).x[0]).toBe(1.5);
    expect(a.entities[0]).toBe(1);
    expect(a.isEnabled(Hidden, 0)).toBe(false);
  });

  it('pushRow grows automatically beyond capacity', () => {
    const w = new World({ initialCapacity: 1 });
    const a = w.archetype(Position);
    for (let i = 0; i < 20; i++) {
      const r = a.pushRow(i);
      a.col(Position).x[r] = i;
    }
    expect(a.count).toBe(20);
    expect(a.capacity).toBeGreaterThanOrEqual(20);
    for (let i = 0; i < 20; i++) {
      expect(a.col(Position).x[i]).toBe(i);
      expect(a.entities[i]).toBe(i);
    }
  });

  it('isEnabled: non-enableable present -> true, absent -> false', () => {
    const w = new World();
    const a = w.archetype(Position);
    a.pushRow(1);
    expect(a.isEnabled(Position, 0)).toBe(true);
    expect(a.isEnabled(Velocity, 0)).toBe(false);
  });

  it('edge caches are populated by add/remove transitions', () => {
    const w = new World();
    const e = w.spawn([Position]);
    w.add(e, Velocity);
    const from = w.archetype(Position);
    const to = w.archetype(Position, Velocity);
    expect(from.edgesAdd.get(Velocity.id)).toBe(to);
    w.remove(e, Velocity);
    expect(to.edgesRemove.get(Velocity.id)).toBe(from);
  });
});

describe('>32 components', () => {
  it('query all:[C] matches for components whose id has bit 31 of its mask word (id % 32 === 31)', () => {
    const comps = [];
    for (let i = 0; i < 64; i++) comps.push(component({ v: i32 }));
    const c31 = comps.find((c) => (c.id & 31) === 31)!;
    const w = new World();
    const q = w.query({ all: [c31] });
    const e = w.spawn([c31]);
    expect(w.has(e, c31)).toBe(true);
    expect(q.matches(w.archetype(c31))).toBe(true);
    expect(q.count()).toBe(1);
    let visited = 0;
    q.forEach(() => visited++);
    expect(visited).toBe(1);
  });


  it('archetypes, queries, add/remove work with ids spanning multiple mask words', () => {
    const comps = [];
    for (let i = 0; i < 100; i++) comps.push(component({ v: i32 }));
    const w = new World();
    const lo = comps[0];
    const mid = comps[40];
    const hi = comps[99];
    const qHi = w.query({ all: [hi] });
    const qLoHi = w.query({ all: [lo, hi] });
    const qNoneMid = w.query({ all: [lo], none: [mid] });
    const qAny = w.query({ any: [mid, hi] });

    const e = w.spawn([lo, hi]);
    w.set(e, hi, { v: 99 });
    w.set(e, lo, { v: 1 });
    expect(w.has(e, hi)).toBe(true);
    expect(w.has(e, mid)).toBe(false);
    expect(qHi.count()).toBe(1);
    expect(qLoHi.count()).toBe(1);
    expect(qNoneMid.count()).toBe(1);
    expect(qAny.count()).toBe(1);

    w.add(e, mid, { v: 40 });
    expect(qNoneMid.count()).toBe(0);
    expect(w.get(e, hi)!.v).toBe(99);
    expect(w.get(e, lo)!.v).toBe(1);
    expect(w.get(e, mid)!.v).toBe(40);

    w.remove(e, hi);
    expect(qHi.count()).toBe(0);
    expect(qAny.count()).toBe(1);
    expect(w.get(e, mid)!.v).toBe(40);

    const e2 = w.spawn([lo]);
    expect(qNoneMid.count()).toBe(1);
    expect(qAny.count()).toBe(1);
    expect(w.isAlive(e2)).toBe(true);
  });
});

describe('shared table buffer', () => {
  const SP = component({ x: f32, y: f32 }, { name: 'SBPos' });
  const SI = component({ big: f64, small: i8, mid: u16, id: u32 }, { name: 'SBInfo', enableable: true });

  for (const shared of [false, true]) {
    it(`all columns are views over one buffer that survives growth (shared=${shared})`, () => {
      const w = new World({ initialCapacity: 2, shared });
      const arch = w.archetype(SP, SI);
      const ents: number[] = [];
      for (let i = 0; i < 37; i++) {
        const e = w.spawn(arch);
        w.set(e, SP, { x: i + 0.5, y: 0 - i || 0 });
        w.set(e, SI, { big: i * 1e10, small: 0 - i || 0, mid: 1000 + i, id: 0xfffffff0 + (i % 15) });
        if (i % 3 === 0) w.enable(e, SI, false);
        ents.push(e);
      }
      expect(arch.capacity).toBeGreaterThanOrEqual(37);
      const buf = arch.buffer;
      const useSAB = shared && typeof SharedArrayBuffer !== 'undefined';
      expect(arch.shared).toBe(useSAB);
      expect(buf instanceof (useSAB ? SharedArrayBuffer : ArrayBuffer)).toBe(true);
      const p = arch.col(SP);
      const s = arch.col(SI);
      for (const v of [p.x, p.y, s.big, s.small, s.mid, s.id, arch.entities, arch.enabledArray(SI) as Uint8Array]) {
        expect(v.buffer).toBe(buf);
        expect(v.length).toBe(arch.capacity);
        expect(v.byteOffset % v.BYTES_PER_ELEMENT).toBe(0);
      }
      for (let i = 0; i < 37; i++) {
        const e = ents[i];
        expect(w.get(e, SP)).toEqual({ x: i + 0.5, y: 0 - i || 0 });
        expect(w.get(e, SI)).toEqual({ big: i * 1e10, small: 0 - i || 0, mid: 1000 + i, id: 0xfffffff0 + (i % 15) });
        expect(w.isEnabled(e, SI)).toBe(i % 3 !== 0);
      }
      // move rows out and back (plan copy across tables)
      for (const e of ents) w.remove(e, SP);
      for (const e of ents) w.add(e, SP, { x: 7 });
      for (let i = 0; i < 37; i++) {
        expect(w.get(ents[i], SI)!.id).toBe(0xfffffff0 + (i % 15));
        expect(w.get(ents[i], SP)).toEqual({ x: 7, y: 0 });
        expect(w.isEnabled(ents[i], SI)).toBe(i % 3 !== 0);
      }
    });
  }
});
