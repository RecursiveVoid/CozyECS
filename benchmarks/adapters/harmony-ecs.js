// harmony-ecs 0.0.x adapter. Idiomatic per its README/API docs: binary (TypedArray, SoA) schemas,
// Query.make() records iterated as `for (const [entities, [a, b]] of query)`; archetype rows are
// swap-removed, so loops that change structure iterate backwards.
import { World, Schema, Entity, Query, Format } from 'harmony-ecs';

const N = 1000;
const SIZE = 10000; // binary column capacity per archetype (rows, not entity ids)
const LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');

function count(q) {
  let n = 0;
  for (let i = 0; i < q.length; i++) n += q[i][0].length;
  return n;
}

// Fairness audit (r3): iteration scenarios check the exact entity count, the archetype layout
// (non-empty records) and EVERY entity's value, like the cozyecs/bitecs4 adapters.
function checkEach(q, pick, want, count, sizes) {
  const recs = [];
  for (let i = 0; i < q.length; i++) if (q[i][0].length > 0) recs.push(q[i]);
  const got = recs.map((r) => r[0].length).join(',');
  if (got !== sizes) return `record sizes: expected [${sizes}], got [${got}]`;
  let n = 0;
  for (const [entities, cols] of recs) {
    const a = pick(cols);
    for (let j = 0; j < entities.length; j++, n++) if (a[j] !== want) return `row ${j}: expected ${want}, got ${a[j]}`;
  }
  return n === count ? null : `count: expected ${count}, got ${n}`;
}
const sizesOf = (k, size) => Array(k).fill(size).join(',');

function packed5() {
  const world = World.make(SIZE);
  const comps = [0, 1, 2, 3, 4].map(() => Schema.makeBinary(world, { value: Format.float32 }));
  for (let i = 0; i < N; i++) Entity.make(world, comps, comps.map(() => ({ value: 1 })));
  const queries = comps.map((C) => Query.make(world, [C]));
  const makeSystem = (q) => () => {
    for (const [entities, [c]] of q) {
      const v = c.value;
      for (let j = 0; j < entities.length; j++) v[j] *= 2;
    }
  };
  const [s0, s1, s2, s3, s4] = queries.map(makeSystem);
  return {
    step() {
      s0(); s1(); s2(); s3(); s4();
    },
    check(ticks) {
      for (const q of queries) {
        let s = 0;
        for (const [entities, [c]] of q) for (let j = 0; j < entities.length; j++) s += c.value[j];
        if (s !== N * 2 ** ticks) return `sum: expected ${N * 2 ** ticks}, got ${s}`;
        const err = checkEach(q, (c) => c[0].value, 2 ** ticks, N, sizesOf(1, N));
        if (err) return err;
      }
      return null;
    },
  };
}

function simpleIter() {
  const world = World.make(SIZE);
  const Position = Schema.makeBinary(world, { x: Format.float32, y: Format.float32 });
  const Velocity = Schema.makeBinary(world, { dx: Format.float32, dy: Format.float32 });
  const A = Schema.makeBinary(world, { value: Format.float32 });
  const B = Schema.makeBinary(world, { value: Format.float32 });
  for (const extra of [[], [A], [B], [A, B]]) {
    for (let i = 0; i < N; i++) Entity.make(world, [Position, Velocity, ...extra], [, { dx: 1, dy: 2 }]);
  }
  const kinetics = Query.make(world, [Position, Velocity]);
  const movement = () => {
    for (const [entities, [p, v]] of kinetics) {
      const { x, y } = p;
      const { dx, dy } = v;
      for (let j = 0; j < entities.length; j++) {
        x[j] += dx[j];
        y[j] += dy[j];
      }
    }
  };
  return {
    step: movement,
    check(ticks) {
      let sx = 0, sy = 0, n = 0;
      for (const [entities, [p]] of kinetics) {
        for (let j = 0; j < entities.length; j++) {
          sx += p.x[j];
          sy += p.y[j];
          n++;
        }
      }
      if (n !== 4 * N || sx !== 4 * N * ticks || sy !== 8 * N * ticks) return `n=${n} sums x=${sx} y=${sy}`;
      return (
        checkEach(kinetics, (c) => c[0].x, ticks, 4 * N, sizesOf(4, N)) ||
        checkEach(kinetics, (c) => c[0].y, 2 * ticks, 4 * N, sizesOf(4, N))
      );
    },
  };
}

function fragIter() {
  const world = World.make(SIZE);
  const Data = Schema.makeBinary(world, { value: Format.float32 });
  for (const _ of LETTERS) {
    const L = Schema.makeBinary(world, { value: Format.float32 });
    for (let i = 0; i < 100; i++) Entity.make(world, [L, Data], [, { value: 1 }]);
  }
  const q = Query.make(world, [Data]);
  const dataSystem = () => {
    for (const [entities, [d]] of q) {
      const v = d.value;
      for (let j = 0; j < entities.length; j++) v[j] *= 2;
    }
  };
  return {
    step: dataSystem,
    check(ticks) {
      let s = 0, n = 0;
      for (const [entities, [d]] of q) for (let j = 0; j < entities.length; j++, n++) s += d.value[j];
      if (n !== 2600 || s !== 2600 * 2 ** ticks) return `count=${n} sum=${s}`;
      return checkEach(q, (c) => c[0].value, 2 ** ticks, 2600, sizesOf(26, 100));
    },
  };
}

function entityCycle() {
  const world = World.make(SIZE);
  const A = Schema.makeBinary(world, { value: Format.float32 });
  const B = Schema.makeBinary(world, { value: Format.float32 });
  for (let i = 0; i < N; i++) Entity.make(world, [A]);
  const qA = Query.make(world, [A]);
  const qB = Query.make(world, [B]);
  const typeB = [B];
  const spawnB = () => {
    for (const [entities] of qA) {
      for (let j = entities.length - 1; j >= 0; j--) Entity.make(world, typeB);
    }
  };
  const killB = () => {
    for (const [entities] of qB) {
      for (let j = entities.length - 1; j >= 0; j--) Entity.destroy(world, entities[j]);
    }
  };
  return {
    step() {
      spawnB();
      killB();
    },
    check() {
      const a = count(qA), b = count(qB);
      if (a !== N || b !== 0) return `A=${a} B=${b}`;
      return null;
    },
  };
}

function addRemove() {
  const world = World.make(SIZE);
  const A = Schema.makeBinary(world, { value: Format.float32 });
  const B = Schema.makeBinary(world, { value: Format.float32 });
  for (let i = 0; i < N; i++) Entity.make(world, [A]);
  const typeB = [B];
  const qA = Query.make(world, [A]);
  const qAnoB = Query.make(world, [A], Query.not(typeB));
  const qB = Query.make(world, typeB);
  const addB = () => {
    for (const [entities] of qAnoB) {
      for (let j = entities.length - 1; j >= 0; j--) Entity.set(world, entities[j], typeB);
    }
  };
  const removeB = () => {
    for (const [entities] of qB) {
      for (let j = entities.length - 1; j >= 0; j--) Entity.unset(world, entities[j], typeB);
    }
  };
  return {
    step() {
      addB();
      removeB();
    },
    check() {
      const a = count(qA), b = count(qB);
      if (a !== N || b !== 0) return `A=${a} B=${b}`;
      return null;
    },
    checkHalf() {
      addB();
      const b = count(qB);
      removeB();
      return b === N ? null : `B after add: expected ${N}, got ${b}`;
    },
  };
}

export const variants = {
  packed_5: { 'harmony-ecs': packed5 },
  simple_iter: { 'harmony-ecs': simpleIter },
  frag_iter: { 'harmony-ecs': fragIter },
  entity_cycle: { 'harmony-ecs': entityCycle },
  add_remove: { 'harmony-ecs': addRemove },
};

export function memory(n) {
  // Binary columns are preallocated to the world size, so size it to exactly n.
  const world = World.make(n);
  const Position = Schema.makeBinary(world, { x: Format.float32, y: Format.float32 });
  const Velocity = Schema.makeBinary(world, { dx: Format.float32, dy: Format.float32 });
  const type = [Position, Velocity];
  for (let i = 0; i < n; i++) Entity.make(world, type, [{ x: i, y: i }, { dx: 1, dy: 1 }]);
  const q = Query.make(world, type);
  return { world, q };
}
