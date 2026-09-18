// bitECS 0.4.x adapter (installed as the alias `bitecs4`).
// Idiomatic per docs/Intro.md: components are user-owned SoA stores (typed arrays for numeric data),
// systems are plain functions calling `query(world, [...])` every frame.
import {
  createWorld,
  addEntity,
  removeEntity,
  addComponent,
  removeComponent,
  query,
  Not,
  commitRemovals,
} from 'bitecs4';

const N = 1000;
const CAP = 10000; // entity ids are recycled immediately, so a small store suffices for the scenarios
const LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
const scalar = (cap = CAP) => ({ value: new Float32Array(cap) });

function sumOf(world, C) {
  let s = 0;
  const ents = query(world, [C]);
  for (let i = 0; i < ents.length; i++) s += C.value[ents[i]];
  return s;
}

// Fairness audit (r3): iteration scenarios check the exact entity count and EVERY entity's value
// (bitECS has no archetypes, so there is no layout to check), like the cozyecs/harmony-ecs adapters.
function checkEach(world, terms, arr, want, count, label) {
  const ents = query(world, terms);
  if (ents.length !== count) return `${label} count: expected ${count}, got ${ents.length}`;
  for (let i = 0; i < ents.length; i++) if (arr[ents[i]] !== want) return `${label}[${ents[i]}]: expected ${want}, got ${arr[ents[i]]}`;
  return null;
}

function packed5() {
  const world = createWorld();
  const comps = [scalar(), scalar(), scalar(), scalar(), scalar()];
  for (let i = 0; i < N; i++) {
    const e = addEntity(world);
    for (const C of comps) {
      addComponent(world, e, C);
      C.value[e] = 1;
    }
  }
  const systems = comps.map((C) => (world) => {
    const v = C.value;
    const ents = query(world, [C]);
    for (let i = 0; i < ents.length; i++) v[ents[i]] *= 2;
  });
  const [s0, s1, s2, s3, s4] = systems;
  return {
    step() {
      s0(world);
      s1(world);
      s2(world);
      s3(world);
      s4(world);
    },
    check(ticks) {
      for (const C of comps) {
        const s = sumOf(world, C);
        if (s !== N * 2 ** ticks) return `sum: expected ${N * 2 ** ticks}, got ${s}`;
        const err = checkEach(world, [C], C.value, 2 ** ticks, N, 'value');
        if (err) return err;
      }
      return null;
    },
  };
}

function simpleIter() {
  const world = createWorld();
  const Position = { x: new Float32Array(CAP), y: new Float32Array(CAP) };
  const Velocity = { dx: new Float32Array(CAP), dy: new Float32Array(CAP) };
  const A = scalar();
  const B = scalar();
  for (const extra of [[], [A], [B], [A, B]]) {
    for (let i = 0; i < N; i++) {
      const e = addEntity(world);
      addComponent(world, e, Position);
      addComponent(world, e, Velocity);
      for (const C of extra) addComponent(world, e, C);
      Velocity.dx[e] = 1;
      Velocity.dy[e] = 2;
    }
  }
  const movement = (world) => {
    const { x, y } = Position;
    const { dx, dy } = Velocity;
    const ents = query(world, [Position, Velocity]);
    for (let i = 0; i < ents.length; i++) {
      const e = ents[i];
      x[e] += dx[e];
      y[e] += dy[e];
    }
  };
  return {
    step() {
      movement(world);
    },
    check(ticks) {
      let sx = 0, sy = 0;
      for (const e of query(world, [Position, Velocity])) {
        sx += Position.x[e];
        sy += Position.y[e];
      }
      if (sx !== 4 * N * ticks || sy !== 8 * N * ticks) return `sums x=${sx} y=${sy}`;
      return (
        checkEach(world, [Position, Velocity], Position.x, ticks, 4 * N, 'x') ||
        checkEach(world, [Position, Velocity], Position.y, 2 * ticks, 4 * N, 'y')
      );
    },
  };
}

function fragIter() {
  const world = createWorld();
  const Data = scalar();
  for (const _ of LETTERS) {
    const L = scalar();
    for (let i = 0; i < 100; i++) {
      const e = addEntity(world);
      addComponent(world, e, L);
      addComponent(world, e, Data);
      Data.value[e] = 1;
    }
  }
  const dataSystem = (world) => {
    const v = Data.value;
    const ents = query(world, [Data]);
    for (let i = 0; i < ents.length; i++) v[ents[i]] *= 2;
  };
  return {
    step() {
      dataSystem(world);
    },
    check(ticks) {
      const n = query(world, [Data]).length;
      const s = sumOf(world, Data);
      if (n !== 2600 || s !== 2600 * 2 ** ticks) return `count=${n} sum=${s}`;
      return checkEach(world, [Data], Data.value, 2 ** ticks, 2600, 'Data');
    },
  };
}

function entityCycle() {
  const world = createWorld();
  const A = scalar();
  const B = scalar();
  for (let i = 0; i < N; i++) addComponent(world, addEntity(world), A);
  const spawnB = (world) => {
    const as = query(world, [A]);
    for (let i = 0; i < as.length; i++) addComponent(world, addEntity(world), B);
  };
  const killB = (world) => {
    const bs = query(world, [B]);
    for (let i = bs.length - 1; i >= 0; i--) removeEntity(world, bs[i]);
  };
  return {
    step() {
      spawnB(world);
      killB(world);
    },
    check() {
      const a = query(world, [A]).length, b = query(world, [B]).length;
      if (a !== N || b !== 0) return `A=${a} B=${b}`;
      return null;
    },
  };
}

function addRemove() {
  const world = createWorld();
  const A = scalar();
  const B = scalar();
  for (let i = 0; i < N; i++) addComponent(world, addEntity(world), A);
  const addB = (world) => {
    const ents = query(world, [A, Not(B)]);
    for (let i = ents.length - 1; i >= 0; i--) addComponent(world, ents[i], B);
  };
  const removeB = (world) => {
    const ents = query(world, [B]);
    for (let i = ents.length - 1; i >= 0; i--) removeComponent(world, ents[i], B);
  };
  return {
    step() {
      addB(world);
      removeB(world);
    },
    check() {
      const a = query(world, [A]).length, b = query(world, [B]).length;
      if (a !== N || b !== 0) return `A=${a} B=${b}`;
      return null;
    },
    checkHalf() {
      addB(world);
      const b = query(world, [B]).length;
      removeB(world);
      return b === N ? null : `B after add: expected ${N}, got ${b}`;
    },
  };
}

// ---------------------------------------------------------------- 'bitecs4 (cached query)'
// Fairness audit (r5): query(world, terms) re-derives the query on EVERY call (terms.find/filter,
// modifier scan, and a queryHash string built with map/sort/join before the Map lookup), a fixed cost
// of several hundred ns per system call that dominates small workloads such as packed_5. This variant
// calls query() once per system at setup and keeps the array it returns: that array is the query's
// own dense sparse-set list (a stable object, updated in place). Each frame it only runs the exported
// commitRemovals(world), which is the remaining work query() does, so the results are identical.
function cachedQuery(world, terms) {
  const dense = query(world, terms);
  return () => {
    commitRemovals(world);
    return dense;
  };
}

function packed5Cached() {
  const world = createWorld();
  const comps = [scalar(), scalar(), scalar(), scalar(), scalar()];
  for (let i = 0; i < N; i++) {
    const e = addEntity(world);
    for (const C of comps) {
      addComponent(world, e, C);
      C.value[e] = 1;
    }
  }
  const systems = comps.map((C) => {
    const q = cachedQuery(world, [C]);
    const v = C.value;
    return () => {
      const ents = q();
      for (let i = 0; i < ents.length; i++) v[ents[i]] *= 2;
    };
  });
  const [s0, s1, s2, s3, s4] = systems;
  return {
    step() {
      s0();
      s1();
      s2();
      s3();
      s4();
    },
    check(ticks) {
      for (const C of comps) {
        const s = sumOf(world, C);
        if (s !== N * 2 ** ticks) return `sum: expected ${N * 2 ** ticks}, got ${s}`;
        const err = checkEach(world, [C], C.value, 2 ** ticks, N, 'value');
        if (err) return err;
      }
      return null;
    },
  };
}

function simpleIterCached() {
  const world = createWorld();
  const Position = { x: new Float32Array(CAP), y: new Float32Array(CAP) };
  const Velocity = { dx: new Float32Array(CAP), dy: new Float32Array(CAP) };
  const A = scalar();
  const B = scalar();
  for (const extra of [[], [A], [B], [A, B]]) {
    for (let i = 0; i < N; i++) {
      const e = addEntity(world);
      addComponent(world, e, Position);
      addComponent(world, e, Velocity);
      for (const C of extra) addComponent(world, e, C);
      Velocity.dx[e] = 1;
      Velocity.dy[e] = 2;
    }
  }
  const q = cachedQuery(world, [Position, Velocity]);
  const { x, y } = Position;
  const { dx, dy } = Velocity;
  const movement = () => {
    const ents = q();
    for (let i = 0; i < ents.length; i++) {
      const e = ents[i];
      x[e] += dx[e];
      y[e] += dy[e];
    }
  };
  return {
    step: movement,
    check(ticks) {
      return (
        checkEach(world, [Position, Velocity], Position.x, ticks, 4 * N, 'x') ||
        checkEach(world, [Position, Velocity], Position.y, 2 * ticks, 4 * N, 'y')
      );
    },
  };
}

function fragIterCached() {
  const world = createWorld();
  const Data = scalar();
  for (const _ of LETTERS) {
    const L = scalar();
    for (let i = 0; i < 100; i++) {
      const e = addEntity(world);
      addComponent(world, e, L);
      addComponent(world, e, Data);
      Data.value[e] = 1;
    }
  }
  const q = cachedQuery(world, [Data]);
  const v = Data.value;
  return {
    step() {
      const ents = q();
      for (let i = 0; i < ents.length; i++) v[ents[i]] *= 2;
    },
    check(ticks) {
      return checkEach(world, [Data], Data.value, 2 ** ticks, 2600, 'Data');
    },
  };
}

function entityCycleCached() {
  const world = createWorld();
  const A = scalar();
  const B = scalar();
  for (let i = 0; i < N; i++) addComponent(world, addEntity(world), A);
  const qA = cachedQuery(world, [A]);
  const qB = cachedQuery(world, [B]);
  return {
    step() {
      const as = qA();
      for (let i = 0; i < as.length; i++) addComponent(world, addEntity(world), B);
      const bs = qB();
      for (let i = bs.length - 1; i >= 0; i--) removeEntity(world, bs[i]);
    },
    check() {
      const a = query(world, [A]).length, b = query(world, [B]).length;
      if (a !== N || b !== 0) return `A=${a} B=${b}`;
      return null;
    },
  };
}

function addRemoveCached() {
  const world = createWorld();
  const A = scalar();
  const B = scalar();
  for (let i = 0; i < N; i++) addComponent(world, addEntity(world), A);
  const qAnoB = cachedQuery(world, [A, Not(B)]);
  const qB = cachedQuery(world, [B]);
  const addB = () => {
    const ents = qAnoB();
    for (let i = ents.length - 1; i >= 0; i--) addComponent(world, ents[i], B);
  };
  const removeB = () => {
    const ents = qB();
    for (let i = ents.length - 1; i >= 0; i--) removeComponent(world, ents[i], B);
  };
  return {
    step() {
      addB();
      removeB();
    },
    check() {
      const a = query(world, [A]).length, b = query(world, [B]).length;
      if (a !== N || b !== 0) return `A=${a} B=${b}`;
      return null;
    },
    checkHalf() {
      addB();
      const b = query(world, [B]).length;
      removeB();
      return b === N ? null : `B after add: expected ${N}, got ${b}`;
    },
  };
}

export const variants = {
  packed_5: { bitecs4: packed5, 'bitecs4 (cached query)': packed5Cached },
  simple_iter: { bitecs4: simpleIter, 'bitecs4 (cached query)': simpleIterCached },
  frag_iter: { bitecs4: fragIter, 'bitecs4 (cached query)': fragIterCached },
  entity_cycle: { bitecs4: entityCycle, 'bitecs4 (cached query)': entityCycleCached },
  add_remove: { bitecs4: addRemove, 'bitecs4 (cached query)': addRemoveCached },
};

export function memory(n) {
  const world = createWorld();
  // Typed SoA stores sized for exactly n entities (ids start at 1).
  const Position = { x: new Float32Array(n + 1), y: new Float32Array(n + 1) };
  const Velocity = { dx: new Float32Array(n + 1), dy: new Float32Array(n + 1) };
  for (let i = 0; i < n; i++) {
    const e = addEntity(world);
    addComponent(world, e, Position);
    addComponent(world, e, Velocity);
    Position.x[e] = i; Position.y[e] = i; Velocity.dx[e] = 1; Velocity.dy[e] = 1;
  }
  // Fairness audit (r5): every memory probe holds a Position+Velocity query (what a movement system
  // needs), so libraries whose queries index entities (sparse sets, cached arrays) pay for it everywhere.
  return { world, Position, Velocity, n: query(world, [Position, Velocity]).length };
}
