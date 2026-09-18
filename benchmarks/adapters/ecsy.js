// ecsy 0.4.x adapter (object components, systems with static queries).
import { World, System, Component, Types } from 'ecsy';

const N = 1000;
const LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');

function valueComponent(name) {
  const C = { [name]: class extends Component {} }[name];
  C.schema = { value: { type: Types.Number, default: 0 } };
  return C;
}

function makeWorld(components) {
  const world = new World({ entityPoolSize: 1000 });
  for (const C of components) world.registerComponent(C);
  return world;
}

function querySystem(name, components, execute) {
  const S = { [name]: class extends System {
    execute() { execute(this.queries.q.results, this.world); }
  } }[name];
  S.queries = { q: { components } };
  return S;
}

let t = 0;
const tick = (world) => world.execute(1, ++t);

function sum(world, C, key) {
  let s = 0;
  for (const e of world.entityManager._entities) {
    const c = e.getComponent(C);
    if (c) s += c[key];
  }
  return s;
}

function packed5() {
  const comps = ['A', 'B', 'C', 'D', 'E'].map(valueComponent);
  const world = makeWorld(comps);
  const [A, B, C, D, E] = comps;
  // Fairness audit (r5): one hand-written system per component, as an application would have. A single
  // system body shared by all 5 systems (created in a loop) made its property accesses megamorphic
  // (5 component classes at one source site), which only hurts object/accessor-based libraries.
  class Double0 extends System {
    execute() { const r = this.queries.q.results; for (let j = 0; j < r.length; j++) r[j].getMutableComponent(A).value *= 2; }
  }
  Double0.queries = { q: { components: [A] } };
  class Double1 extends System {
    execute() { const r = this.queries.q.results; for (let j = 0; j < r.length; j++) r[j].getMutableComponent(B).value *= 2; }
  }
  Double1.queries = { q: { components: [B] } };
  class Double2 extends System {
    execute() { const r = this.queries.q.results; for (let j = 0; j < r.length; j++) r[j].getMutableComponent(C).value *= 2; }
  }
  Double2.queries = { q: { components: [C] } };
  class Double3 extends System {
    execute() { const r = this.queries.q.results; for (let j = 0; j < r.length; j++) r[j].getMutableComponent(D).value *= 2; }
  }
  Double3.queries = { q: { components: [D] } };
  class Double4 extends System {
    execute() { const r = this.queries.q.results; for (let j = 0; j < r.length; j++) r[j].getMutableComponent(E).value *= 2; }
  }
  Double4.queries = { q: { components: [E] } };
  for (const S of [Double0, Double1, Double2, Double3, Double4]) world.registerSystem(S);
  for (let i = 0; i < N; i++) {
    const e = world.createEntity();
    for (const K of comps) e.addComponent(K, { value: 1 });
  }
  return {
    step: () => tick(world),
    check(ticks) {
      for (const K of comps) {
        const s = sum(world, K, 'value');
        if (s !== N * 2 ** ticks) return `sum ${K.name}=${s}`;
      }
      return null;
    },
  };
}

function simpleIter() {
  class Position extends Component {}
  Position.schema = { x: { type: Types.Number }, y: { type: Types.Number } };
  class Velocity extends Component {}
  Velocity.schema = { dx: { type: Types.Number }, dy: { type: Types.Number } };
  const A = valueComponent('A');
  const B = valueComponent('B');
  const world = makeWorld([Position, Velocity, A, B]);
  world.registerSystem(
    querySystem('Move', [Position, Velocity], (results) => {
      for (let i = 0; i < results.length; i++) {
        const e = results[i];
        const p = e.getMutableComponent(Position);
        const v = e.getComponent(Velocity);
        p.x += v.dx;
        p.y += v.dy;
      }
    }),
  );
  for (const extra of [[], [A], [B], [A, B]]) {
    for (let i = 0; i < N; i++) {
      const e = world.createEntity().addComponent(Position, { x: 0, y: 0 }).addComponent(Velocity, { dx: 1, dy: 2 });
      for (const C of extra) e.addComponent(C);
    }
  }
  return {
    step: () => tick(world),
    check(ticks) {
      const sx = sum(world, Position, 'x'), sy = sum(world, Position, 'y');
      if (sx !== 4 * N * ticks || sy !== 8 * N * ticks) return `sums x=${sx} y=${sy}`;
      return null;
    },
  };
}

function fragIter() {
  const letters = LETTERS.map(valueComponent);
  const Data = valueComponent('Data');
  const world = makeWorld([...letters, Data]);
  world.registerSystem(
    querySystem('DataSys', [Data], (results) => {
      for (let i = 0; i < results.length; i++) results[i].getMutableComponent(Data).value *= 2;
    }),
  );
  for (const L of letters) {
    for (let i = 0; i < 100; i++) world.createEntity().addComponent(L).addComponent(Data, { value: 1 });
  }
  return {
    step: () => tick(world),
    check(ticks) {
      const s = sum(world, Data, 'value');
      return s === 2600 * 2 ** ticks ? null : `sum=${s}`;
    },
  };
}

function entityCycle() {
  const A = valueComponent('A');
  const B = valueComponent('B');
  const world = makeWorld([A, B]);
  world.registerSystem(
    querySystem('SpawnB', [A], (results, w) => {
      for (let i = 0; i < results.length; i++) w.createEntity().addComponent(B);
    }),
  );
  world.registerSystem(
    querySystem('KillB', [B], (results) => {
      // entity.remove() splices the query results immediately: iterate backwards.
      for (let i = results.length - 1; i >= 0; i--) results[i].remove();
    }),
  );
  for (let i = 0; i < N; i++) world.createEntity().addComponent(A);
  const qA = world.getSystems()[0].queries.q;
  const qB = world.getSystems()[1].queries.q;
  return {
    step: () => tick(world),
    check() {
      const a = qA.results.length, b = qB.results.length, alive = world.entityManager._entities.length;
      return a === N && b === 0 && alive === N ? null : `A=${a} B=${b} alive=${alive}`;
    },
  };
}

function addRemove() {
  const A = valueComponent('A');
  const B = valueComponent('B');
  const world = makeWorld([A, B]);
  world.registerSystem(
    querySystem('AddB', [A], (results) => {
      for (let i = 0; i < results.length; i++) results[i].addComponent(B);
    }),
  );
  world.registerSystem(
    querySystem('RemB', [B], (results) => {
      for (let i = results.length - 1; i >= 0; i--) results[i].removeComponent(B);
    }),
  );
  for (let i = 0; i < N; i++) world.createEntity().addComponent(A);
  const qA = world.getSystems()[0].queries.q;
  const qB = world.getSystems()[1].queries.q;
  return {
    step: () => tick(world),
    check() {
      const a = qA.results.length, b = qB.results.length;
      return a === N && b === 0 ? null : `A=${a} B=${b}`;
    },
  };
}

export const variants = {
  packed_5: { ecsy: packed5 },
  simple_iter: { ecsy: simpleIter },
  frag_iter: { ecsy: fragIter },
  entity_cycle: { ecsy: entityCycle },
  add_remove: { ecsy: addRemove },
};

export function memory(n) {
  class Position extends Component {}
  Position.schema = { x: { type: Types.Number }, y: { type: Types.Number } };
  class Velocity extends Component {}
  Velocity.schema = { dx: { type: Types.Number }, dy: { type: Types.Number } };
  const world = makeWorld([Position, Velocity]);
  // Fairness audit (r5): every memory probe holds a Position+Velocity query (what a movement system
  // needs), so libraries whose queries index entities (sparse sets, cached arrays) pay for it everywhere.
  world.registerSystem(querySystem('Move', [Position, Velocity], () => {}));
  for (let i = 0; i < n; i++) {
    world.createEntity().addComponent(Position, { x: i, y: i }).addComponent(Velocity, { dx: 1, dy: 1 });
  }
  return world;
}
