import { describe, it, expect } from '@jest/globals';
import { World, Query, component, tag, f32 } from '../src/index';
import { MAX_TRAMPOLINE_SHAPES, _setTrampolineShapeLimit, _trampolineStats } from '../src/query';

const A = component({ v: f32 }, { name: 'A' });
const B = component({ v: f32 }, { name: 'B' });
const C = component({ v: f32 }, { name: 'C' });
const E = component({ v: f32 }, { name: 'E', enableable: true });
const T = tag({ name: 'T', enableable: true });

describe('world.query', () => {
  it('queries are cached/deduped by normalized key', () => {
    const w = new World();
    const q1 = w.query({ all: [A, B], none: [C] });
    const q2 = w.query({ all: [B, A, A], none: [C], any: [] });
    expect(q1).toBe(q2);
    expect(q1).toBeInstanceOf(Query);
    expect(w.query({ all: [A] })).not.toBe(q1);
    expect(w.query({ any: [A] })).not.toBe(w.query({ all: [A] }));
    expect(w.query({ none: [A] })).not.toBe(w.query({ all: [A] }));
  });

  it('queries are per world', () => {
    const w1 = new World();
    const w2 = new World();
    expect(w1.query({ all: [A] })).not.toBe(w2.query({ all: [A] }));
  });

  it('all/any/none matching', () => {
    const w = new World();
    const aAB = w.archetype(A, B);
    const aA = w.archetype(A);
    const aBC = w.archetype(B, C);
    const aABC = w.archetype(A, B, C);
    const empty = w.archetype();

    const qAll = w.query({ all: [A, B] });
    expect(qAll.matches(aAB)).toBe(true);
    expect(qAll.matches(aABC)).toBe(true);
    expect(qAll.matches(aA)).toBe(false);

    const qAny = w.query({ any: [A, C] });
    expect(qAny.matches(aA)).toBe(true);
    expect(qAny.matches(aBC)).toBe(true);
    expect(qAny.matches(empty)).toBe(false);

    const qNone = w.query({ all: [B], none: [C] });
    expect(qNone.matches(aAB)).toBe(true);
    expect(qNone.matches(aBC)).toBe(false);
    expect(qNone.matches(aABC)).toBe(false);

    const qEmpty = w.query({});
    expect(qEmpty.matches(empty)).toBe(true);
    expect(qEmpty.matches(aABC)).toBe(true);
  });

  it('chunks include existing and later-created matching archetypes', () => {
    const w = new World();
    const before = w.archetype(A);
    const q = w.query({ all: [A] });
    expect(q.chunks).toContain(before);
    const after = w.archetype(A, B);
    expect(q.chunks).toContain(after);
    expect(q.chunks).not.toContain(w.archetype(B));
    // archetypes created implicitly via add
    const e = w.spawn([A]);
    w.add(e, C);
    expect(q.chunks).toContain(w.archetype(A, C));
    expect(new Set(q.chunks).size).toBe(q.chunks.length);
  });

  it('count() sums entities across chunks', () => {
    const w = new World();
    const q = w.query({ all: [A] });
    expect(q.count()).toBe(0);
    w.spawn([A]);
    w.spawn([A, B]);
    w.spawn([A, B]);
    w.spawn([B]);
    expect(q.count()).toBe(3);
    const e = w.spawn([A]);
    w.destroy(e);
    expect(q.count()).toBe(3);
  });

  it('chunk loop example from spec', () => {
    const w = new World();
    for (let i = 0; i < 10; i++) w.spawn([A, B]);
    for (let i = 0; i < 5; i++) w.spawn([A]);
    const q = w.query({ all: [A] });
    for (const chunk of q.chunks) {
      const p = chunk.col(A);
      for (let i = 0; i < chunk.count; i++) p.v[i] += 2;
    }
    let total = 0;
    q.forEach((e) => (total += w.get(e, A)!.v));
    expect(total).toBe(30);
  });
});

describe('Query.forEach', () => {
  it('visits every matching entity with (entity, chunk, row)', () => {
    const w = new World();
    const es = new Set<number>();
    for (let i = 0; i < 5; i++) es.add(w.spawn([A]));
    for (let i = 0; i < 5; i++) es.add(w.spawn([A, B]));
    w.spawn([B]);
    const q = w.query({ all: [A] });
    const seen = new Set<number>();
    q.forEach((e, chunk, row) => {
      expect(chunk.entities[row]).toBe(e);
      expect(chunk.has(A)).toBe(true);
      seen.add(e);
    });
    expect(seen).toEqual(es);
  });

  it('iterates rows in reverse order per chunk', () => {
    const w = new World();
    for (let i = 0; i < 5; i++) w.spawn([A]);
    const rows: number[] = [];
    w.query({ all: [A] }).forEach((_e, _c, row) => rows.push(row));
    expect(rows).toEqual([4, 3, 2, 1, 0]);
  });

  it('destroy during forEach visits every entity exactly once and destroys all', () => {
    const w = new World();
    const es: number[] = [];
    for (let i = 0; i < 20; i++) {
      const e = w.spawn([A]);
      w.set(e, A, { v: i });
      es.push(e);
    }
    const q = w.query({ all: [A] });
    const visited: number[] = [];
    q.forEach((e) => {
      visited.push(e);
      w.destroy(e);
    });
    expect(visited.length).toBe(20);
    expect(new Set(visited).size).toBe(20);
    expect(q.count()).toBe(0);
    for (const e of es) expect(w.isAlive(e)).toBe(false);
  });

  it('destroy of other entities during forEach does not break iteration', () => {
    const w = new World();
    const es: number[] = [];
    for (let i = 0; i < 10; i++) es.push(w.spawn([A]));
    const q = w.query({ all: [A] });
    let visits = 0;
    q.forEach((e) => {
      visits++;
      w.destroy(es[0]);
      w.destroy(e);
    });
    expect(visits).toBe(10);
    expect(q.count()).toBe(0);
  });

  it('skips rows where an enableable component in all is disabled', () => {
    const w = new World();
    const on = w.spawn([A, E, T]);
    const offE = w.spawn([A, E, T]);
    const offT = w.spawn([A, E, T]);
    w.enable(offE, E, false);
    w.enable(offT, T, false);
    const seen: number[] = [];
    w.query({ all: [A, E, T] }).forEach((e) => seen.push(e));
    expect(seen).toEqual([on]);

    // Only components listed in `all` filter
    const seenA: number[] = [];
    w.query({ all: [A] }).forEach((e) => seenA.push(e));
    expect(seenA.sort()).toEqual([on, offE, offT].sort());

    const seenE: number[] = [];
    w.query({ all: [E] }).forEach((e) => seenE.push(e));
    expect(seenE.sort()).toEqual([on, offT].sort());

    // any/none do not filter by enabled state
    const seenAny: number[] = [];
    w.query({ any: [E] }).forEach((e) => seenAny.push(e));
    expect(seenAny.length).toBe(3);

    // count() includes disabled
    expect(w.query({ all: [A, E, T] }).count()).toBe(3);
  });

  it('chunk loops do not filter disabled rows; enabledArray available', () => {
    const w = new World();
    const e1 = w.spawn([E]);
    w.spawn([E]);
    w.enable(e1, E, false);
    const q = w.query({ all: [E] });
    let rows = 0;
    let enabled = 0;
    for (const c of q.chunks) {
      const en = c.enabledArray(E)!;
      for (let i = 0; i < c.count; i++) {
        rows++;
        if (en[i]) enabled++;
      }
    }
    expect(rows).toBe(2);
    expect(enabled).toBe(1);
  });

  it('add/remove during forEach is deferred until forEach ends', () => {
    const w = new World();
    for (let i = 0; i < 10; i++) w.spawn([A]);
    const q = w.query({ all: [A] });
    const qB = w.query({ all: [B] });
    let visits = 0;
    q.forEach((e) => {
      visits++;
      w.add(e, B, { v: 5 });
      expect(w.has(e, B)).toBe(false);
    });
    expect(visits).toBe(10);
    expect(qB.count()).toBe(10);
    q.forEach((e) => {
      expect(w.get(e, B)!.v).toBe(5);
      w.remove(e, A);
    });
    expect(q.count()).toBe(0);
  });

  it('spawn during forEach returns a live id and is placed after forEach', () => {
    const w = new World();
    for (let i = 0; i < 3; i++) w.spawn([A]);
    const q = w.query({ all: [A] });
    const spawned: number[] = [];
    let visits = 0;
    q.forEach(() => {
      visits++;
      const e = w.spawn([A]);
      expect(w.isAlive(e)).toBe(true);
      spawned.push(e);
    });
    expect(visits).toBe(3);
    expect(q.count()).toBe(6);
    for (const e of spawned) {
      expect(w.isAlive(e)).toBe(true);
      expect(w.has(e, A)).toBe(true);
    }
    expect(new Set(spawned).size).toBe(3);
  });

  it('spawned ids during forEach are unique vs existing and can be targeted by deferred ops', () => {
    const w = new World();
    w.spawn([A]);
    const q = w.query({ all: [A] });
    let child = -1;
    q.forEach(() => {
      child = w.spawn([B]);
      w.add(child, C, { v: 3 });
    });
    expect(w.has(child, B)).toBe(true);
    expect(w.get(child, C)!.v).toBe(3);
  });

  it('destroying a spawned-during-iteration entity before flush results in dead entity', () => {
    const w = new World();
    w.spawn([A]);
    const q = w.query({ all: [A] });
    let child = -1;
    q.forEach(() => {
      child = w.spawn([B]);
      w.destroy(child);
    });
    expect(w.isAlive(child)).toBe(false);
    expect(w.archetype(B).count).toBe(0);
  });

  it('nested forEach flushes after the outermost ends', () => {
    const w = new World();
    w.spawn([A]);
    w.spawn([B]);
    const qA = w.query({ all: [A] });
    const qB = w.query({ all: [B] });
    let innerDone = false;
    qA.forEach((ea) => {
      qB.forEach((eb) => {
        w.destroy(eb);
      });
      innerDone = true;
      expect(qB.count()).toBe(1); // still deferred
      w.destroy(ea);
    });
    expect(innerDone).toBe(true);
    expect(qA.count()).toBe(0);
    expect(qB.count()).toBe(0);
  });

  it('set/get/enable are immediate inside forEach', () => {
    const w = new World();
    const e = w.spawn([A, E]);
    w.query({ all: [A] }).forEach((x) => {
      w.set(x, A, { v: 9 });
      expect(w.get(x, A)!.v).toBe(9);
      w.enable(x, E, false);
      expect(w.isEnabled(x, E)).toBe(false);
    });
    expect(w.get(e, A)!.v).toBe(9);
  });

  it('spawnMany during forEach is deferred', () => {
    const w = new World();
    w.spawn([A]);
    const q = w.query({ all: [A] });
    q.forEach(() => {
      w.spawnMany(w.archetype(A), 5);
      expect(q.count()).toBe(1);
    });
    expect(q.count()).toBe(6);
  });

  it('exception in forEach callback still ends iteration (later changes are immediate)', () => {
    const w = new World();
    w.spawn([A]);
    const q = w.query({ all: [A] });
    expect(() =>
      q.forEach(() => {
        throw new Error('boom');
      }),
    ).toThrow('boom');
    const e = w.spawn([A]);
    expect(w.has(e, A)).toBe(true);
  });

  it('explicit flush() applies pending commands', () => {
    const w = new World();
    expect(() => w.flush()).not.toThrow();
    const e = w.spawn([A]);
    w.destroy(e);
    w.flush();
    expect(w.isAlive(e)).toBe(false);
  });
});

describe('command buffer ordering', () => {
  it('applies deferred commands in call order', () => {
    const w = new World();
    const e = w.spawn([A]);
    const e2 = w.spawn([A]);
    w.query({ all: [A] }).forEach((x) => {
      if (x !== e) return;
      w.add(e, B, { v: 1 });
      w.remove(e, B);
      w.remove(e2, A);
      w.add(e2, A, { v: 4 });
    });
    expect(w.has(e, B)).toBe(false);
    expect(w.has(e2, A)).toBe(true);
    expect(w.get(e2, A)!.v).toBe(4);
  });

  it('ops queued after a deferred destroy are silently skipped for that entity', () => {
    const w = new World();
    const e = w.spawn([A]);
    expect(() =>
      w.query({ all: [A] }).forEach((x) => {
        w.destroy(x);
        w.add(x, B);
        w.remove(x, A);
        w.destroy(x);
      }),
    ).not.toThrow();
    expect(w.isAlive(e)).toBe(false);
    expect(w.archetype(A, B).count).toBe(0);
  });

  it('hooks firing during flush that make structural changes are applied in the same flush', () => {
    const w = new World();
    w.onAdd(B, (x) => w.add(x, C, { v: 2 }));
    const e = w.spawn([A]);
    w.query({ all: [A] }).forEach((x) => w.add(x, B));
    expect(w.has(e, C)).toBe(true);
    expect(w.get(e, C)!.v).toBe(2);
  });

  it('getField and get follow entity after moves caused by other entities', () => {
    const w = new World();
    const es: number[] = [];
    for (let i = 0; i < 6; i++) {
      const e = w.spawn([A]);
      w.set(e, A, { v: i });
      es.push(e);
    }
    w.query({ all: [A] }).forEach((x) => {
      if (w.getField(x, A, 'v') % 2 === 0) w.add(x, B);
    });
    es.forEach((e, i) => {
      expect(w.getField(e, A, 'v')).toBe(i);
      expect(w.has(e, B)).toBe(i % 2 === 0);
    });
  });
});

describe('compiled trampolines: shape cache (no lifetime budget)', () => {
  const P = component({ x: f32, y: f32 }, { name: 'TP' });
  const V = component({ dx: f32, dy: f32 }, { name: 'TV' });

  /** Sum of P.x over a query's chunks. */
  const sumX = (q: Query): number => {
    let s = 0;
    for (const c of q.chunks) {
      const x = c.col(P).x;
      for (let i = 0; i < c.count; i++) s += x[i];
    }
    return s;
  };

  it('10,000 grow/rebuild cycles of one shape compile at most one factory', () => {
    const w = new World();
    const arch = w.archetype(P, V);
    for (let i = 0; i < 4; i++) w.set(w.spawn(arch), V, { dx: 1, dy: 2 });
    const q = w.query({ all: [P, V] });
    const move = (n: number, p: { x: Float32Array }, v: { dx: Float32Array }) => {
      const x = p.x, dx = v.dx;
      for (let i = 0; i < n; i++) x[i] += dx[i];
    };
    q.forEachChunk([P, V], move); // first sighting: generic loop
    const before = _trampolineStats();
    q.forEachChunk([P, V], move); // plan + trampoline
    let expected = 8;
    expect(sumX(q)).toBe(expected);
    let live = arch.entities;
    for (let cycle = 0; cycle < 10000; cycle++) {
      if (cycle < 12) {
        arch.grow(); // real reallocation: new buffer, new column arrays, new entities view
        live = arch.entities;
      } else if (cycle % 100 === 0) {
        // a fresh world: new archetype, new query, same shape
        const w2 = new World();
        const e2 = w2.spawn(w2.archetype(P, V));
        w2.set(e2, V, { dx: 5, dy: 0 });
        const q2 = w2.query({ all: [P, V] });
        q2.forEachChunk([P, V], move);
        expect(sumX(q2)).toBe(5);
      } else {
        // what every reallocation does as far as the trampoline can see: `entities` replaced
        (arch as unknown as { entities: Uint32Array }).entities = live.slice();
      }
      q.forEachChunk([P, V], move);
      expected += 4;
    }
    (arch as unknown as { entities: Uint32Array }).entities = live; // restore the real view
    expect(sumX(q)).toBe(expected);
    const after = _trampolineStats();
    expect(after.shapes - before.shapes).toBeLessThanOrEqual(1);
    expect(after.instances - before.instances).toBeGreaterThanOrEqual(10000);
    // every rebuild came one call after the previous one: nothing was stable long enough to specialize
    expect(after.specializations).toBe(before.specializations);
    expect(after.shapeRejections).toBe(before.shapeRejections);
  });

  it('a run that stays stable is specialized once, then left alone', () => {
    const w = new World();
    for (let i = 0; i < 3; i++) w.set(w.spawn([P, V]), V, { dx: 1, dy: 0 });
    const q = w.query({ all: [P, V] });
    const k = (n: number, p: { x: Float32Array }, v: { dx: Float32Array }) => {
      for (let i = 0; i < n; i++) p.x[i] += v.dx[i];
    };
    const before = _trampolineStats();
    for (let f = 0; f < 200; f++) q.forEachChunk([P, V], k);
    expect(sumX(q)).toBe(600);
    const after = _trampolineStats();
    expect(after.specializations - before.specializations).toBe(1);
    expect(after.shapes - before.shapes).toBeLessThanOrEqual(1);
  });

  it('many distinct callbacks with the same shape share one factory', () => {
    const w = new World();
    for (let i = 0; i < 3; i++) w.set(w.spawn([P, V]), V, { dx: 1, dy: 0 });
    const q = w.query({ all: [P, V] });
    const make = (scale: number) => (n: number, p: { x: Float32Array }, v: { dx: Float32Array }) => {
      for (let i = 0; i < n; i++) p.x[i] += v.dx[i] * scale;
    };
    const warm = make(0);
    q.forEachChunk([P, V], warm);
    q.forEachChunk([P, V], warm); // the shape exists from here on
    const before = _trampolineStats();
    let expected = 0;
    for (let c = 1; c <= 50; c++) {
      const fn = make(c);
      q.forEachChunk([P, V], fn);
      q.forEachChunk([P, V], fn);
      expected += 2 * 3 * c;
    }
    expect(sumX(q)).toBe(expected);
    const after = _trampolineStats();
    expect(after.shapes).toBe(before.shapes);
    expect(after.instances - before.instances).toBe(50);
  });

  it('falls back to the generic loop only for shapes beyond the bound', () => {
    expect(MAX_TRAMPOLINE_SHAPES).toBe(4096);
    const comps = ['s0', 's1', 's2', 's3'].map((f) => component({ [f]: f32 }, { name: f.toUpperCase() }));
    const w = new World();
    for (const C of comps) w.spawn([C]);
    const base = _trampolineStats();
    const prev = _setTrampolineShapeLimit(base.shapes + 2);
    try {
      const hits: number[] = [0, 0, 0, 0];
      const kernels = comps.map((C, idx) => (n: number, col: Record<string, Float32Array>) => {
        const a = col[`s${idx}`];
        for (let i = 0; i < n; i++) a[i] += 1;
        hits[idx] += n;
      });
      for (let r = 0; r < 3; r++) comps.forEach((C, idx) => w.query({ all: [C] }).forEachChunk([C], kernels[idx] as never));
      const s = _trampolineStats();
      // two new shapes compiled, the other two refused (and re-refused each call) but still run
      expect(s.shapes - base.shapes).toBe(2);
      expect(s.shapeRejections - base.shapeRejections).toBeGreaterThanOrEqual(2);
      expect(hits).toEqual([3, 3, 3, 3]);
      comps.forEach((C, idx) => expect(w.query({ all: [C] }).chunks[0].col(C)[`s${idx}` as never]).toBeDefined());
      // shapes compiled before the bound was reached keep getting new instances
      const w2 = new World();
      w2.spawn([comps[0]]);
      const inst = _trampolineStats().instances;
      const q2 = w2.query({ all: [comps[0]] });
      q2.forEachChunk([comps[0]], kernels[0] as never);
      expect(_trampolineStats().instances).toBe(inst + 1);
      expect(_trampolineStats().shapes - base.shapes).toBe(2);
    } finally {
      _setTrampolineShapeLimit(prev);
    }
  });
});
