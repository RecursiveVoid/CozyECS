import { describe, it, expect } from '@jest/globals';
import * as api from '../src/index';
import { component, tag, f32, f64, i8, i16, i32, u8, u16, u32, bool, str } from '../src/index';
import { createMask, maskContains, maskHas, maskIntersects, normalizeComponents, componentsKey } from '../src/component';

describe('public exports', () => {
  it('exports every runtime API from the spec', () => {
    for (const name of ['World', 'System', 'component', 'tag', 'f32', 'f64', 'i8', 'i16', 'i32', 'u8', 'u16', 'u32', 'bool', 'str']) {
      expect((api as Record<string, unknown>)[name]).toBeDefined();
    }
  });

  it('maps each field token to the right TypedArray constructor', () => {
    expect(f32.ctor).toBe(Float32Array);
    expect(f64.ctor).toBe(Float64Array);
    expect(i8.ctor).toBe(Int8Array);
    expect(i16.ctor).toBe(Int16Array);
    expect(i32.ctor).toBe(Int32Array);
    expect(u8.ctor).toBe(Uint8Array);
    expect(u16.ctor).toBe(Uint16Array);
    expect(u32.ctor).toBe(Uint32Array);
    expect(bool.ctor).toBe(Uint8Array);
    expect(str.ctor).toBe(Uint32Array);
  });
});

describe('component()', () => {
  it('creates a component type with schema metadata', () => {
    const Position = component({ x: f32, y: f32 }, { name: 'Position' });
    expect(Position.name).toBe('Position');
    expect(Position.keys).toEqual(['x', 'y']);
    expect(Position.tokens).toEqual([f32, f32]);
    expect(Position.schema.x).toBe(f32);
    expect(Position.isTag).toBe(false);
    expect(Position.enableable).toBe(false);
    expect(typeof Position.id).toBe('number');
  });

  it('assigns unique increasing ids from a module-global counter', () => {
    const A = component({ a: i32 });
    const B = component({ b: i32 });
    const C = tag();
    expect(B.id).toBe(A.id + 1);
    expect(C.id).toBe(B.id + 1);
  });

  it('defaults the name when not given', () => {
    const A = component({ a: u8 });
    expect(typeof A.name).toBe('string');
    expect(A.name.length).toBeGreaterThan(0);
  });

  it('supports enableable option', () => {
    const A = component({ a: u8 }, { enableable: true });
    expect(A.enableable).toBe(true);
  });

  it('rejects invalid field tokens', () => {
    expect(() => component({ a: 'nope' as unknown as typeof f32 })).toThrow();
  });
});

describe('tag()', () => {
  it('creates an empty-schema component', () => {
    const Player = tag({ name: 'Player' });
    expect(Player.name).toBe('Player');
    expect(Player.isTag).toBe(true);
    expect(Player.keys).toEqual([]);
    expect(Player.tokens).toEqual([]);
    expect(Player.enableable).toBe(false);
  });

  it('can be enableable', () => {
    expect(tag({ enableable: true }).enableable).toBe(true);
  });
});

describe('mask helpers', () => {
  it('uses id>>>5 words and supports ids beyond 32', () => {
    const comps = [];
    for (let i = 0; i < 70; i++) comps.push(tag());
    const hi = comps[69];
    const lo = comps[0];
    const m = createMask([lo, hi]);
    expect(m.length).toBe((Math.max(lo.id, hi.id) >>> 5) + 1);
    expect(maskHas(m, hi.id)).toBe(true);
    expect(maskHas(m, lo.id)).toBe(true);
    expect(maskHas(m, comps[40].id)).toBe(false);
    expect(maskHas(new Uint32Array(0), hi.id)).toBe(false);

    const sub = createMask([hi]);
    expect(maskContains(m, sub)).toBe(true);
    expect(maskContains(createMask([lo]), sub)).toBe(false);
    expect(maskContains(m, new Uint32Array(0))).toBe(true);
    expect(maskIntersects(createMask([lo]), sub)).toBe(false);
    expect(maskIntersects(m, sub)).toBe(true);
  });

  it('bit 31 masks work (sign bit)', () => {
    const comps = [];
    for (let i = 0; i < 64; i++) comps.push(tag());
    const c = comps.find((x) => (x.id & 31) === 31)!;
    const m = createMask([c]);
    expect(maskHas(m, c.id)).toBe(true);
    expect(maskContains(m, createMask([c]))).toBe(true);
    expect(maskIntersects(m, createMask([c]))).toBe(true);
  });

  it('normalizes and keys component lists', () => {
    const A = tag();
    const B = tag();
    const n = normalizeComponents([B, A, B]);
    expect(n).toEqual([A, B]);
    expect(componentsKey(n)).toBe(`${A.id},${B.id}`);
    expect(componentsKey([])).toBe('');
  });
});
