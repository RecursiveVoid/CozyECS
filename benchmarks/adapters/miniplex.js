// miniplex 2.x adapter. Idiomatic per its README: entities are plain objects, components are
// properties, queries come from world.with()/without() and are iterated with for...of
// (which iterates in reverse, so removing entities/components while iterating is safe).
import { World } from 'miniplex';

const N = 1000;
const LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');

function packed5() {
  const world = new World();
  for (let i = 0; i < N; i++) world.add({ A: 1, B: 1, C: 1, D: 1, E: 1 });
  const qA = world.with('A');
  const qB = world.with('B');
  const qC = world.with('C');
  const qD = world.with('D');
  const qE = world.with('E');
  const systems = [
    () => { for (const e of qA) e.A *= 2; },
    () => { for (const e of qB) e.B *= 2; },
    () => { for (const e of qC) e.C *= 2; },
    () => { for (const e of qD) e.D *= 2; },
    () => { for (const e of qE) e.E *= 2; },
  ];
  const [s0, s1, s2, s3, s4] = systems;
  return {
    step() {
      s0(); s1(); s2(); s3(); s4();
    },
    check(ticks) {
      for (const [k, q] of [['A', qA], ['B', qB], ['C', qC], ['D', qD], ['E', qE]]) {
        let s = 0;
        for (const e of q) s += e[k];
        if (s !== N * 2 ** ticks) return `sum ${k}: expected ${N * 2 ** ticks}, got ${s}`;
      }
      return null;
    },
  };
}

function simpleIter() {
  const world = new World();
  for (let i = 0; i < N; i++) world.add({ position: { x: 0, y: 0 }, velocity: { x: 1, y: 2 } });
  for (let i = 0; i < N; i++) world.add({ position: { x: 0, y: 0 }, velocity: { x: 1, y: 2 }, A: 0 });
  for (let i = 0; i < N; i++) world.add({ position: { x: 0, y: 0 }, velocity: { x: 1, y: 2 }, B: 0 });
  for (let i = 0; i < N; i++) world.add({ position: { x: 0, y: 0 }, velocity: { x: 1, y: 2 }, A: 0, B: 0 });
  const moving = world.with('position', 'velocity');
  const movementSystem = () => {
    for (const { position, velocity } of moving) {
      position.x += velocity.x;
      position.y += velocity.y;
    }
  };
  return {
    step: movementSystem,
    check(ticks) {
      let sx = 0, sy = 0;
      for (const { position } of moving) {
        sx += position.x;
        sy += position.y;
      }
      if (sx !== 4 * N * ticks || sy !== 8 * N * ticks) return `sums x=${sx} y=${sy}`;
      return null;
    },
  };
}

function fragIter() {
  const world = new World();
  for (const L of LETTERS) {
    for (let i = 0; i < 100; i++) world.add({ [L]: 0, Data: 1 });
  }
  const withData = world.with('Data');
  const dataSystem = () => {
    for (const e of withData) e.Data *= 2;
  };
  return {
    step: dataSystem,
    check(ticks) {
      let s = 0, n = 0;
      for (const e of withData) {
        s += e.Data;
        n++;
      }
      if (n !== 2600 || s !== 2600 * 2 ** ticks) return `count=${n} sum=${s}`;
      return null;
    },
  };
}

function entityCycle() {
  const world = new World();
  for (let i = 0; i < N; i++) world.add({ A: 0 });
  const withA = world.with('A');
  const withB = world.with('B');
  const spawnB = () => {
    for (const _ of withA) world.add({ B: 0 });
  };
  const killB = () => {
    for (const e of withB) world.remove(e);
  };
  return {
    step() {
      spawnB();
      killB();
    },
    check() {
      const a = withA.size, b = withB.size;
      if (a !== N || b !== 0) return `A=${a} B=${b}`;
      return null;
    },
  };
}

function addRemove() {
  const world = new World();
  for (let i = 0; i < N; i++) world.add({ A: 0 });
  const withA = world.with('A');
  const aNoB = world.with('A').without('B');
  const withB = world.with('B');
  const addB = () => {
    for (const e of aNoB) world.addComponent(e, 'B', 0);
  };
  const removeB = () => {
    for (const e of withB) world.removeComponent(e, 'B');
  };
  return {
    step() {
      addB();
      removeB();
    },
    check() {
      const a = withA.size, b = withB.size;
      if (a !== N || b !== 0) return `A=${a} B=${b}`;
      return null;
    },
    checkHalf() {
      addB();
      const b = withB.size;
      removeB();
      return b === N ? null : `B after add: expected ${N}, got ${b}`;
    },
  };
}

export const variants = {
  packed_5: { miniplex: packed5 },
  simple_iter: { miniplex: simpleIter },
  frag_iter: { miniplex: fragIter },
  entity_cycle: { miniplex: entityCycle },
  add_remove: { miniplex: addRemove },
};

export function memory(n) {
  const world = new World();
  for (let i = 0; i < n; i++) world.add({ position: { x: i, y: i }, velocity: { x: 1, y: 1 } });
  // A connected query is what a real app would hold (it indexes the entities).
  const moving = world.with('position', 'velocity').connect();
  return { world, moving };
}
