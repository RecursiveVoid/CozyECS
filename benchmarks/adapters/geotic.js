// geotic 4.x adapter (object entities, BigInt bitmask queries, component instances).
import { Engine, Component } from 'geotic';

const N = 1000;
const LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');

// geotic stores the component bit on the class prototype at registration, so every
// scenario defines fresh classes and its own Engine.
function valueComponent(name) {
  const C = { [name]: class extends Component {} }[name];
  C.properties = { value: 0 };
  return C;
}

function setup(components) {
  const engine = new Engine();
  for (const C of components) engine.registerComponent(C);
  return engine.createWorld();
}

function packed5() {
  const comps = ['A', 'B', 'C', 'D', 'E'].map(valueComponent);
  const keys = ['a', 'b', 'c', 'd', 'e'];
  const world = setup(comps);
  for (let i = 0; i < N; i++) {
    const e = world.createEntity();
    for (const C of comps) e.add(C, { value: 1 });
  }
  const queries = comps.map((C) => world.createQuery({ all: [C] }));
  const [qA, qB, qC, qD, qE] = queries;
  // Fairness audit (r5): one hand-written system per component, as an application would have. A single
  // system body shared by all 5 systems (created in a loop) made its property accesses megamorphic
  // (5 component classes at one source site), which only hurts object/accessor-based libraries.
  const s0 = () => { const ents = qA.get(); for (let i = 0; i < ents.length; i++) ents[i].a.value *= 2; };
  const s1 = () => { const ents = qB.get(); for (let i = 0; i < ents.length; i++) ents[i].b.value *= 2; };
  const s2 = () => { const ents = qC.get(); for (let i = 0; i < ents.length; i++) ents[i].c.value *= 2; };
  const s3 = () => { const ents = qD.get(); for (let i = 0; i < ents.length; i++) ents[i].d.value *= 2; };
  const s4 = () => { const ents = qE.get(); for (let i = 0; i < ents.length; i++) ents[i].e.value *= 2; };
  return {
    step() {
      s0(); s1(); s2(); s3(); s4();
    },
    check(ticks) {
      for (let k = 0; k < 5; k++) {
        let s = 0;
        for (const e of queries[k].get()) s += e[keys[k]].value;
        if (s !== N * 2 ** ticks) return `sum ${keys[k]}=${s}`;
      }
      return null;
    },
  };
}

function simpleIter() {
  class Position extends Component {}
  Position.properties = { x: 0, y: 0 };
  class Velocity extends Component {}
  Velocity.properties = { dx: 0, dy: 0 };
  const A = valueComponent('A');
  const B = valueComponent('B');
  const world = setup([Position, Velocity, A, B]);
  for (const extra of [[], [A], [B], [A, B]]) {
    for (let i = 0; i < N; i++) {
      const e = world.createEntity();
      e.add(Position, { x: 0, y: 0 });
      e.add(Velocity, { dx: 1, dy: 2 });
      for (const C of extra) e.add(C);
    }
  }
  const q = world.createQuery({ all: [Position, Velocity] });
  return {
    step() {
      const ents = q.get();
      for (let i = 0; i < ents.length; i++) {
        const e = ents[i];
        e.position.x += e.velocity.dx;
        e.position.y += e.velocity.dy;
      }
    },
    check(ticks) {
      let sx = 0, sy = 0;
      for (const e of q.get()) { sx += e.position.x; sy += e.position.y; }
      return sx === 4 * N * ticks && sy === 8 * N * ticks ? null : `sums x=${sx} y=${sy}`;
    },
  };
}

function fragIter() {
  const letters = LETTERS.map(valueComponent);
  const Data = valueComponent('Data');
  const world = setup([...letters, Data]);
  for (const L of letters) {
    for (let i = 0; i < 100; i++) {
      const e = world.createEntity();
      e.add(L);
      e.add(Data, { value: 1 });
    }
  }
  const q = world.createQuery({ all: [Data] });
  return {
    step() {
      const ents = q.get();
      for (let i = 0; i < ents.length; i++) ents[i].data.value *= 2;
    },
    check(ticks) {
      let s = 0;
      for (const e of q.get()) s += e.data.value;
      return s === 2600 * 2 ** ticks ? null : `sum=${s}`;
    },
  };
}

function entityCycle() {
  const A = valueComponent('A');
  const B = valueComponent('B');
  const world = setup([A, B]);
  for (let i = 0; i < N; i++) world.createEntity().add(A);
  const qA = world.createQuery({ all: [A] });
  const qB = world.createQuery({ all: [B] });
  return {
    step() {
      const as = qA.get();
      for (let i = 0; i < as.length; i++) world.createEntity().add(B);
      const bs = qB.get();
      // destroy() splices the live query cache: iterate backwards.
      for (let i = bs.length - 1; i >= 0; i--) bs[i].destroy();
    },
    check() {
      const a = qA.get().length, b = qB.get().length, alive = world._entities.size;
      return a === N && b === 0 && alive === N ? null : `A=${a} B=${b} alive=${alive}`;
    },
  };
}

function addRemove() {
  const A = valueComponent('A');
  const B = valueComponent('B');
  const world = setup([A, B]);
  for (let i = 0; i < N; i++) world.createEntity().add(A);
  const qA = world.createQuery({ all: [A] });
  const qB = world.createQuery({ all: [B] });
  return {
    step() {
      const as = qA.get();
      for (let i = 0; i < as.length; i++) as[i].add(B);
      const bs = qB.get();
      for (let i = bs.length - 1; i >= 0; i--) bs[i].remove(bs[i].b);
    },
    check() {
      const a = qA.get().length, b = qB.get().length;
      return a === N && b === 0 ? null : `A=${a} B=${b}`;
    },
  };
}

export const variants = {
  packed_5: { geotic: packed5 },
  simple_iter: { geotic: simpleIter },
  frag_iter: { geotic: fragIter },
  entity_cycle: { geotic: entityCycle },
  add_remove: { geotic: addRemove },
};

export function memory(n) {
  class Position extends Component {}
  Position.properties = { x: 0, y: 0 };
  class Velocity extends Component {}
  Velocity.properties = { dx: 0, dy: 0 };
  const world = setup([Position, Velocity]);
  for (let i = 0; i < n; i++) {
    const e = world.createEntity();
    e.add(Position, { x: i, y: i });
    e.add(Velocity, { dx: 1, dy: 1 });
  }
  // Fairness audit (r5): every memory probe holds a Position+Velocity query (what a movement system
  // needs), so libraries whose queries index entities (sparse sets, cached arrays) pay for it everywhere.
  const q = world.createQuery({ all: [Position, Velocity] });
  return { world, q, n: q.get().length };
}
