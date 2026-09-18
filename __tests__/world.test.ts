import { describe, it, expect } from '@jest/globals';
import { World, component, tag, f32, f64, i8, u8, u16, i32, bool, str } from '../src/index';

const Position = component({ x: f32, y: f32 }, { name: 'Position' });
const Velocity = component({ vx: f32, vy: f32 }, { name: 'Velocity' });
const Health = component({ hp: i32, alive: bool }, { name: 'Health' });
const Info = component({ name: str, level: u16, ratio: f64, small: i8, flag: u8 }, { name: 'Info' });
const Frozen = component({ t: f32 }, { name: 'Frozen', enableable: true });
const Player = tag({ name: 'Player' });
const Hidden = tag({ name: 'Hidden', enableable: true });

describe('entities', () => {
  it('handles: low 20 bits index, high 12 bits generation', () => {
    const w = new World();
    const e0 = w.spawn();
    const e1 = w.spawn();
    expect(e0 & 0xfffff).toBe(0);
    expect(e1 & 0xfffff).toBe(1);
    expect(e0 >>> 20).toBe(0);
    w.destroy(e0);
    const e2 = w.spawn();
    expect(e2 & 0xfffff).toBe(0); // index recycled
    expect(e2 >>> 20).toBe(1); // generation bumped
    expect(e2).not.toBe(e0);
  });

  it('isAlive false for stale ids after reuse', () => {
    const w = new World();
    const e = w.spawn([Position]);
    expect(w.isAlive(e)).toBe(true);
    w.destroy(e);
    expect(w.isAlive(e)).toBe(false);
    const e2 = w.spawn([Position]);
    expect(w.isAlive(e2)).toBe(true);
    expect(w.isAlive(e)).toBe(false);
    // stale id operations
    expect(w.has(e, Position)).toBe(false);
    expect(w.get(e, Position)).toBeUndefined();
    expect(() => w.destroy(e)).not.toThrow();
    expect(w.isAlive(e2)).toBe(true);
    expect(() => w.remove(e, Position)).not.toThrow();
    expect(w.has(e2, Position)).toBe(true);
    expect(() => w.add(e, Velocity)).toThrow();
    expect(() => w.set(e, Position, { x: 1 })).toThrow();
    expect(w.has(e2, Velocity)).toBe(false);
  });

  it('isAlive false for never-allocated ids and garbage', () => {
    const w = new World();
    expect(w.isAlive(0)).toBe(false);
    expect(w.isAlive(12345)).toBe(false);
    expect(w.isAlive(-1)).toBe(false);
  });

  it('generation wraps within 12 bits', () => {
    const w = new World();
    let e = w.spawn();
    const first = e;
    for (let i = 0; i < 4095; i++) {
      w.destroy(e);
      e = w.spawn();
    }
    expect(e >>> 20).toBe(4095);
    expect(e & 0xfffff).toBe(0);
    expect(e).toBeGreaterThan(0); // uint32 handle, not negative
    expect(w.isAlive(e)).toBe(true);
    expect(w.isAlive(first)).toBe(false);
    w.destroy(e);
    const wrapped = w.spawn();
    expect(wrapped >>> 20).toBe(0);
    expect(wrapped & 0xfffff).toBe(0);
    expect(w.isAlive(wrapped)).toBe(true);
    expect(w.isAlive(e)).toBe(false);
  });

  it('free list recycles indices (LIFO) and grows beyond initial capacity', () => {
    const w = new World({ initialCapacity: 2 });
    const es: number[] = [];
    for (let i = 0; i < 1000; i++) es.push(w.spawn([Position]));
    expect(new Set(es.map((e) => e & 0xfffff)).size).toBe(1000);
    for (const e of es) expect(w.isAlive(e)).toBe(true);
    w.destroy(es[10]);
    w.destroy(es[20]);
    const a = w.spawn();
    const b = w.spawn();
    expect(new Set([a & 0xfffff, b & 0xfffff])).toEqual(new Set([10, 20]));
    const c = w.spawn();
    expect(c & 0xfffff).toBe(1000);
  });
});

describe('spawn', () => {
  it('spawn with component list or archetype zero-inits fields', () => {
    const w = new World();
    const e1 = w.spawn([Position, Health]);
    const e2 = w.spawn(w.archetype(Position, Health));
    for (const e of [e1, e2]) {
      expect(w.has(e, Position)).toBe(true);
      expect(w.has(e, Health)).toBe(true);
      expect(w.get(e, Position)).toEqual({ x: 0, y: 0 });
      expect(w.get(e, Health)).toEqual({ hp: 0, alive: false });
    }
  });

  it('bare spawn has no components', () => {
    const w = new World();
    const e = w.spawn();
    expect(w.isAlive(e)).toBe(true);
    expect(w.has(e, Position)).toBe(false);
  });

  it('spawn reuses zeroed rows after destroy', () => {
    const w = new World();
    const e = w.spawn([Position, Frozen]);
    w.set(e, Position, { x: 5, y: 6 });
    w.enable(e, Frozen, false);
    w.destroy(e);
    const e2 = w.spawn([Position, Frozen]);
    expect(w.get(e2, Position)).toEqual({ x: 0, y: 0 });
    expect(w.isEnabled(e2, Frozen)).toBe(true);
  });

  it('rejects an archetype from a different world', () => {
    const w1 = new World();
    const w2 = new World();
    w2.archetype(Velocity);
    const foreign = w2.archetype(Position, Player);
    expect(() => w1.spawn(foreign)).toThrow();
  });
});

describe('spawnMany', () => {
  it('spawns count entities and calls init(chunk, row, i)', () => {
    const w = new World({ initialCapacity: 4 });
    const arch = w.archetype(Position);
    const seen: number[] = [];
    w.spawnMany(arch, 100, (chunk, row, i) => {
      expect(chunk).toBe(arch);
      chunk.col(Position).x[row] = i;
      seen.push(i);
    });
    expect(arch.count).toBe(100);
    expect(seen).toEqual(Array.from({ length: 100 }, (_, i) => i));
    expect(arch.capacity).toBeGreaterThanOrEqual(100);
    for (let r = 0; r < 100; r++) {
      const e = arch.entities[r];
      expect(w.isAlive(e)).toBe(true);
      expect(w.get(e, Position)!.x).toBe(arch.col(Position).x[r]);
    }
    const q = w.query({ all: [Position] });
    expect(q.count()).toBe(100);
  });

  it('works without init and with count 0', () => {
    const w = new World();
    const arch = w.archetype(Position, Velocity);
    w.spawnMany(arch, 0);
    expect(arch.count).toBe(0);
    w.spawnMany(arch, 70);
    expect(arch.count).toBe(70);
    expect(w.get(arch.entities[69], Velocity)).toEqual({ vx: 0, vy: 0 });
  });

  it('grows storage once (single reallocation)', () => {
    const w = new World({ initialCapacity: 2 });
    const arch = w.archetype(Position);
    w.spawnMany(arch, 1000);
    expect(arch.capacity).toBeGreaterThanOrEqual(1000);
    expect(arch.capacity).toBeLessThan(2048 + 1);
  });

  it('recycles freed ids', () => {
    const w = new World();
    const arch = w.archetype(Position);
    const e = w.spawn(arch);
    w.destroy(e);
    w.spawnMany(arch, 3);
    const idx = Array.from(arch.entities.subarray(0, 3)).map((x) => x & 0xfffff);
    expect(idx).toContain(0);
    expect(w.isAlive(e)).toBe(false);
  });
});

describe('destroy', () => {
  it('swap-remove keeps other entities data intact', () => {
    const w = new World();
    const es: number[] = [];
    for (let i = 0; i < 10; i++) {
      const e = w.spawn([Position, Health, Info]);
      w.set(e, Position, { x: i, y: -i });
      w.set(e, Health, { hp: i * 100, alive: i % 2 === 0 });
      w.set(e, Info, { name: `n${i}`, level: i, ratio: i / 3, small: -i, flag: i });
      es.push(e);
    }
    w.destroy(es[0]);
    w.destroy(es[5]);
    w.destroy(es[9]);
    for (let i = 0; i < 10; i++) {
      if (i === 0 || i === 5 || i === 9) {
        expect(w.isAlive(es[i])).toBe(false);
        continue;
      }
      expect(w.get(es[i], Position)).toEqual({ x: i, y: -i });
      expect(w.get(es[i], Health)).toEqual({ hp: i * 100, alive: i % 2 === 0 });
      expect(w.get(es[i], Info)).toEqual({ name: `n${i}`, level: i, ratio: i / 3, small: -i, flag: i });
    }
    expect(w.archetype(Position, Health, Info).count).toBe(7);
  });

  it('swap-remove keeps enabled flags with their entities', () => {
    const w = new World();
    const a = w.spawn([Frozen, Hidden]);
    const b = w.spawn([Frozen, Hidden]);
    const c = w.spawn([Frozen, Hidden]);
    w.enable(c, Frozen, false);
    w.enable(b, Hidden, false);
    w.destroy(a);
    expect(w.isEnabled(c, Frozen)).toBe(false);
    expect(w.isEnabled(c, Hidden)).toBe(true);
    expect(w.isEnabled(b, Frozen)).toBe(true);
    expect(w.isEnabled(b, Hidden)).toBe(false);
  });

  it('destroying twice is a silent no-op', () => {
    const w = new World();
    const e = w.spawn([Position]);
    const other = w.spawn([Position]);
    w.destroy(e);
    w.destroy(e);
    expect(w.isAlive(other)).toBe(true);
    expect(w.archetype(Position).count).toBe(1);
  });
});

describe('add / remove / has', () => {
  it('archetype transitions preserve data', () => {
    const w = new World();
    const e = w.spawn([Position, Health, Info]);
    w.set(e, Position, { x: 1, y: 2 });
    w.set(e, Health, { hp: 50, alive: true });
    w.set(e, Info, { name: 'x', level: 3, ratio: 0.5, small: -3, flag: 255 });
    w.add(e, Velocity, { vx: 3 });
    expect(w.has(e, Velocity)).toBe(true);
    expect(w.get(e, Velocity)).toEqual({ vx: 3, vy: 0 });
    expect(w.get(e, Position)).toEqual({ x: 1, y: 2 });
    expect(w.get(e, Health)).toEqual({ hp: 50, alive: true });
    expect(w.get(e, Info)).toEqual({ name: 'x', level: 3, ratio: 0.5, small: -3, flag: 255 });

    w.add(e, Player);
    w.remove(e, Health);
    expect(w.has(e, Health)).toBe(false);
    expect(w.has(e, Player)).toBe(true);
    expect(w.get(e, Position)).toEqual({ x: 1, y: 2 });
    expect(w.get(e, Velocity)).toEqual({ vx: 3, vy: 0 });
    expect(w.get(e, Info)!.name).toBe('x');

    w.remove(e, Position);
    w.remove(e, Velocity);
    w.remove(e, Player);
    w.remove(e, Info);
    expect(w.isAlive(e)).toBe(true);
    expect(w.archetype().entities.subarray(0, w.archetype().count)).toContain(e);
  });

  it('transitions preserve enabled bits', () => {
    const w = new World();
    const e = w.spawn([Frozen]);
    w.enable(e, Frozen, false);
    w.add(e, Position);
    expect(w.isEnabled(e, Frozen)).toBe(false);
    w.remove(e, Position);
    expect(w.isEnabled(e, Frozen)).toBe(false);
  });

  it('moving out of an archetype keeps remaining entities there intact', () => {
    const w = new World();
    const a = w.spawn([Position]);
    const b = w.spawn([Position]);
    const c = w.spawn([Position]);
    w.set(a, Position, { x: 1 });
    w.set(b, Position, { x: 2 });
    w.set(c, Position, { x: 3 });
    w.add(a, Velocity);
    expect(w.get(b, Position)!.x).toBe(2);
    expect(w.get(c, Position)!.x).toBe(3);
    expect(w.get(a, Position)!.x).toBe(1);
    w.remove(a, Velocity);
    expect(w.get(a, Position)!.x).toBe(1);
    expect(w.get(b, Position)!.x).toBe(2);
    expect(w.get(c, Position)!.x).toBe(3);
  });

  it('add of existing component only sets values (no move)', () => {
    const w = new World();
    const e = w.spawn([Position]);
    w.set(e, Position, { x: 1, y: 2 });
    const arch = w.archetype(Position);
    w.add(e, Position, { y: 9 });
    expect(w.archetype(Position)).toBe(arch);
    expect(arch.count).toBe(1);
    expect(w.get(e, Position)).toEqual({ x: 1, y: 9 });
    w.add(e, Position);
    expect(w.get(e, Position)).toEqual({ x: 1, y: 9 });
  });

  it('remove of absent component is a no-op', () => {
    const w = new World();
    const e = w.spawn([Position]);
    w.remove(e, Velocity);
    expect(w.has(e, Position)).toBe(true);
  });

  it('add throws on dead entity, remove silently ignores', () => {
    const w = new World();
    const e = w.spawn();
    w.destroy(e);
    expect(() => w.add(e, Position)).toThrow(Error);
    expect(() => w.remove(e, Position)).not.toThrow();
  });

  it('has false for dead entity', () => {
    const w = new World();
    const e = w.spawn([Position]);
    w.destroy(e);
    expect(w.has(e, Position)).toBe(false);
  });

  it('tags carry no column memory', () => {
    const w = new World();
    const e = w.spawn();
    w.add(e, Player);
    const arch = w.archetype(Player);
    expect(arch.col(Player)).toBeUndefined();
    expect(w.has(e, Player)).toBe(true);
  });
});

describe('set / get / getField', () => {
  it('set writes partial values', () => {
    const w = new World();
    const e = w.spawn([Position]);
    w.set(e, Position, { x: 3 });
    expect(w.get(e, Position)).toEqual({ x: 3, y: 0 });
    w.set(e, Position, { y: 4 });
    expect(w.get(e, Position)).toEqual({ x: 3, y: 4 });
  });

  it('bool fields convert to 0/1 and read back as boolean', () => {
    const w = new World();
    const e = w.spawn([Health]);
    w.set(e, Health, { alive: true });
    expect(w.getField(e, Health, 'alive')).toBe(1);
    expect(w.get(e, Health)!.alive).toBe(true);
    w.set(e, Health, { alive: false });
    expect(w.getField(e, Health, 'alive')).toBe(0);
    expect(w.get(e, Health)!.alive).toBe(false);
  });

  it('f32 fields store with float32 precision', () => {
    const w = new World();
    const e = w.spawn([Position]);
    w.set(e, Position, { x: 0.1 });
    expect(w.get(e, Position)!.x).toBe(Math.fround(0.1));
  });

  it('get returns undefined for absent component or dead entity', () => {
    const w = new World();
    const e = w.spawn([Position]);
    expect(w.get(e, Velocity)).toBeUndefined();
    w.destroy(e);
    expect(w.get(e, Position)).toBeUndefined();
  });

  it('get returns a per-component cached view overwritten on each call', () => {
    const w = new World();
    const a = w.spawn([Position]);
    const b = w.spawn([Position]);
    w.set(a, Position, { x: 1 });
    w.set(b, Position, { x: 2 });
    const va = w.get(a, Position)!;
    expect(va.x).toBe(1);
    const vb = w.get(b, Position)!;
    expect(vb).toBe(va);
    expect(va.x).toBe(2);
    // different components have different views
    const e = w.spawn([Position, Velocity]);
    expect(w.get(e, Velocity)).not.toBe(w.get(e, Position));
  });

  it('get on a tag returns a (empty) view object', () => {
    const w = new World();
    const e = w.spawn([Player]);
    expect(w.get(e, Player)).toEqual({});
  });

  it('getField returns raw column values', () => {
    const w = new World();
    const e = w.spawn([Info]);
    w.set(e, Info, { level: 7, ratio: 1.25, small: -5, flag: 200, name: 'hi' });
    expect(w.getField(e, Info, 'level')).toBe(7);
    expect(w.getField(e, Info, 'ratio')).toBe(1.25);
    expect(w.getField(e, Info, 'small')).toBe(-5);
    expect(w.getField(e, Info, 'flag')).toBe(200);
    expect(w.getField(e, Info, 'name')).toBe(w.strings.intern('hi'));
  });

  it('set throws for dead entity', () => {
    const w = new World();
    const e = w.spawn([Position]);
    w.destroy(e);
    expect(() => w.set(e, Position, { x: 1 })).toThrow(Error);
  });

  it('set writes are visible via chunk columns', () => {
    const w = new World();
    const e = w.spawn([Position]);
    w.set(e, Position, { x: 8, y: 9 });
    const c = w.archetype(Position);
    expect(c.col(Position).x[0]).toBe(8);
    expect(c.col(Position).y[0]).toBe(9);
  });
});

describe('enable / isEnabled', () => {
  it('toggles enableable components', () => {
    const w = new World();
    const e = w.spawn([Frozen, Hidden, Position]);
    expect(w.isEnabled(e, Frozen)).toBe(true);
    w.enable(e, Frozen, false);
    expect(w.isEnabled(e, Frozen)).toBe(false);
    w.enable(e, Frozen);
    expect(w.isEnabled(e, Frozen)).toBe(true);
    w.enable(e, Hidden, false);
    expect(w.isEnabled(e, Hidden)).toBe(false);
    expect(w.isEnabled(e, Frozen)).toBe(true);
  });

  it('isEnabled for non-enableable present component is true, absent is false', () => {
    const w = new World();
    const e = w.spawn([Position]);
    expect(w.isEnabled(e, Position)).toBe(true);
    expect(w.isEnabled(e, Velocity)).toBe(false);
  });

  it('enabled data survives growth', () => {
    const w = new World({ initialCapacity: 2 });
    const es: number[] = [];
    for (let i = 0; i < 50; i++) {
      const e = w.spawn([Frozen]);
      w.enable(e, Frozen, i % 3 !== 0);
      es.push(e);
    }
    es.forEach((e, i) => expect(w.isEnabled(e, Frozen)).toBe(i % 3 !== 0));
  });
});

describe('growth', () => {
  it('data preserved when archetypes grow beyond initial capacity', () => {
    const w = new World({ initialCapacity: 3 });
    const es: number[] = [];
    for (let i = 0; i < 500; i++) {
      const e = w.spawn([Position, Info]);
      w.set(e, Position, { x: i, y: i * 2 });
      w.set(e, Info, { name: `e${i % 7}`, level: i });
      es.push(e);
    }
    es.forEach((e, i) => {
      expect(w.get(e, Position)).toEqual({ x: i, y: i * 2 });
      expect(w.get(e, Info)!.name).toBe(`e${i % 7}`);
      expect(w.get(e, Info)!.level).toBe(i);
    });
    // transitions into a small target archetype also grow
    for (const e of es) w.add(e, Velocity);
    es.forEach((e, i) => expect(w.get(e, Position)).toEqual({ x: i, y: i * 2 }));
    expect(w.archetype(Position, Info).count).toBe(0);
    expect(w.archetype(Position, Info, Velocity).count).toBe(500);
  });
});

describe('initialCapacity option', () => {
  it('default is 64', () => {
    expect(new World().archetype(Position).capacity).toBe(64);
  });
});
