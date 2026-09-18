// wolf-ecs 2.x adapter. Idiomatic per its README: typed components from `types`, queries from
// createQuery(), and the "more performant" manual archetype loop (query.a[i].e, iterated backwards).
import { ECS, types, not } from 'wolf-ecs';

const N = 1000;
const LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');

function sum(q, arr) {
  let s = 0, n = 0;
  for (let i = 0; i < q.a.length; i++) {
    const ents = q.a[i].e;
    for (let j = ents.length - 1; j >= 0; j--) {
      s += arr[ents[j]];
      n++;
    }
  }
  return [s, n];
}

function count(q) {
  let n = 0;
  for (let i = 0; i < q.a.length; i++) n += q.a[i].e.length;
  return n;
}

function packed5() {
  const ecs = new ECS();
  const comps = [0, 1, 2, 3, 4].map(() => ecs.defineComponent({ value: types.f32 }));
  const queries = comps.map((C) => ecs.createQuery(C));
  for (let i = 0; i < N; i++) {
    const e = ecs.createEntity();
    for (const C of comps) {
      ecs.addComponent(e, C);
      C.value[e] = 1;
    }
  }
  const makeSystem = (C, q) => () => {
    const v = C.value;
    for (let i = 0; i < q.a.length; i++) {
      const arch = q.a[i].e;
      for (let j = arch.length - 1; j >= 0; j--) v[arch[j]] *= 2;
    }
  };
  const [s0, s1, s2, s3, s4] = comps.map((C, i) => makeSystem(C, queries[i]));
  return {
    step() {
      s0(); s1(); s2(); s3(); s4();
    },
    check(ticks) {
      for (let i = 0; i < 5; i++) {
        const [s] = sum(queries[i], comps[i].value);
        if (s !== N * 2 ** ticks) return `sum: expected ${N * 2 ** ticks}, got ${s}`;
      }
      return null;
    },
  };
}

function simpleIter() {
  const ecs = new ECS();
  const Position = ecs.defineComponent({ x: types.f32, y: types.f32 });
  const Velocity = ecs.defineComponent({ dx: types.f32, dy: types.f32 });
  const A = ecs.defineComponent({ value: types.f32 });
  const B = ecs.defineComponent({ value: types.f32 });
  const q = ecs.createQuery(Position, Velocity);
  for (const extra of [[], [A], [B], [A, B]]) {
    for (let i = 0; i < N; i++) {
      const e = ecs.createEntity();
      ecs.addComponent(e, Position);
      ecs.addComponent(e, Velocity);
      for (const C of extra) ecs.addComponent(e, C);
      Position.x[e] = 0;
      Position.y[e] = 0;
      Velocity.dx[e] = 1;
      Velocity.dy[e] = 2;
    }
  }
  const movement = () => {
    const { x, y } = Position;
    const { dx, dy } = Velocity;
    for (let i = 0; i < q.a.length; i++) {
      const arch = q.a[i].e;
      for (let j = arch.length - 1; j >= 0; j--) {
        const e = arch[j];
        x[e] += dx[e];
        y[e] += dy[e];
      }
    }
  };
  return {
    step: movement,
    check(ticks) {
      const [sx, n] = sum(q, Position.x);
      const [sy] = sum(q, Position.y);
      if (n !== 4 * N || sx !== 4 * N * ticks || sy !== 8 * N * ticks) return `n=${n} sums x=${sx} y=${sy}`;
      return null;
    },
  };
}

function fragIter() {
  const ecs = new ECS();
  const Data = ecs.defineComponent({ value: types.f32 });
  const letters = LETTERS.map(() => ecs.defineComponent({ value: types.f32 }));
  const q = ecs.createQuery(Data);
  for (const L of letters) {
    for (let i = 0; i < 100; i++) {
      const e = ecs.createEntity();
      ecs.addComponent(e, L);
      ecs.addComponent(e, Data);
      Data.value[e] = 1;
    }
  }
  const dataSystem = () => {
    const v = Data.value;
    for (let i = 0; i < q.a.length; i++) {
      const arch = q.a[i].e;
      for (let j = arch.length - 1; j >= 0; j--) v[arch[j]] *= 2;
    }
  };
  return {
    step: dataSystem,
    check(ticks) {
      const [s, n] = sum(q, Data.value);
      if (n !== 2600 || s !== 2600 * 2 ** ticks) return `count=${n} sum=${s}`;
      return null;
    },
  };
}

function entityCycle() {
  const ecs = new ECS();
  const A = ecs.defineComponent({ value: types.f32 });
  const B = ecs.defineComponent({ value: types.f32 });
  const qA = ecs.createQuery(A);
  const qB = ecs.createQuery(B);
  for (let i = 0; i < N; i++) ecs.addComponent(ecs.createEntity(), A);
  const spawnB = () => {
    for (let i = 0; i < qA.a.length; i++) {
      const arch = qA.a[i].e;
      for (let j = arch.length - 1; j >= 0; j--) ecs.addComponent(ecs.createEntity(), B);
    }
  };
  const killB = () => {
    for (let i = 0; i < qB.a.length; i++) {
      const arch = qB.a[i].e;
      for (let j = arch.length - 1; j >= 0; j--) ecs.destroyEntity(arch[j]);
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
  const ecs = new ECS();
  const A = ecs.defineComponent({ value: types.f32 });
  const B = ecs.defineComponent({ value: types.f32 });
  const qA = ecs.createQuery(A);
  const qAnoB = ecs.createQuery(A, not(B));
  const qB = ecs.createQuery(B);
  for (let i = 0; i < N; i++) ecs.addComponent(ecs.createEntity(), A);
  const addB = () => {
    for (let i = 0; i < qAnoB.a.length; i++) {
      const arch = qAnoB.a[i].e;
      for (let j = arch.length - 1; j >= 0; j--) ecs.addComponent(arch[j], B);
    }
  };
  const removeB = () => {
    for (let i = 0; i < qB.a.length; i++) {
      const arch = qB.a[i].e;
      for (let j = arch.length - 1; j >= 0; j--) ecs.removeComponent(arch[j], B);
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
  packed_5: { 'wolf-ecs': packed5 },
  simple_iter: { 'wolf-ecs': simpleIter },
  frag_iter: { 'wolf-ecs': fragIter },
  entity_cycle: { 'wolf-ecs': entityCycle },
  add_remove: { 'wolf-ecs': addRemove },
};

export function memory(n) {
  // Component arrays are allocated for the entity limit, so size it to exactly n.
  const ecs = new ECS(n);
  const Position = ecs.defineComponent({ x: types.f32, y: types.f32 });
  const Velocity = ecs.defineComponent({ dx: types.f32, dy: types.f32 });
  const q = ecs.createQuery(Position, Velocity);
  for (let i = 0; i < n; i++) {
    const e = ecs.createEntity();
    ecs.addComponent(e, Position);
    ecs.addComponent(e, Velocity);
    Position.x[e] = i; Position.y[e] = i; Velocity.dx[e] = 1; Velocity.dy[e] = 1;
  }
  return { ecs, q, Position, Velocity }; // wolf-ecs does not retain component arrays itself
}
