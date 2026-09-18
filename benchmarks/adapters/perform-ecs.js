// perform-ecs 0.7.x adapter (mixin model: component fields are written onto the entity object).
// perform-ecs component ids are process-global and hashed into a 32-bit int, so at most 31
// component classes can exist: they are defined once here and shared by all scenarios.
// Because fields live on the entity itself, every component gets a distinct field name.
import PECS from 'perform-ecs';

const { ECS, System, Component, EntityViewFactory, makeComponent } = PECS;
const N = 1000;
const LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');

function define(name, reset) {
  const C = { [name]: class extends Component {} }[name];
  C.prototype.reset = reset;
  makeComponent(C);
  return C;
}

const Letters = LETTERS.map((l) => {
  const key = l.toLowerCase();
  return define(l, (obj, v = 0) => { obj[key] = v; });
});
const [A, B, C, D, E] = Letters;
const Data = define('Data', (obj, v = 0) => { obj.data = v; });
const Position = define('Position', (obj) => { obj.x = 0; obj.y = 0; });
const Velocity = define('Velocity', (obj, dx = 0, dy = 0) => { obj.dx = dx; obj.dy = dy; });

function viewSystem(components, update) {
  const S = class extends System {
    constructor() {
      super();
      this.view = EntityViewFactory.createView({ components });
    }
    update(dt) { update(this.view.entities, this.ecs); }
  };
  return new S();
}

function packed5() {
  const ecs = new ECS();
  const comps = [A, B, C, D, E];
  const keys = ['a', 'b', 'c', 'd', 'e'];
  // Fairness audit (r5): one hand-written system per component, as an application would have. A single
  // system body shared by all 5 systems (created in a loop) made its property accesses megamorphic
  // (5 component classes at one source site), which only hurts object/accessor-based libraries.
  const systems = [
    ecs.registerSystem(viewSystem([A], (ents) => { for (let i = 0; i < ents.length; i++) ents[i].a *= 2; })),
    ecs.registerSystem(viewSystem([B], (ents) => { for (let i = 0; i < ents.length; i++) ents[i].b *= 2; })),
    ecs.registerSystem(viewSystem([C], (ents) => { for (let i = 0; i < ents.length; i++) ents[i].c *= 2; })),
    ecs.registerSystem(viewSystem([D], (ents) => { for (let i = 0; i < ents.length; i++) ents[i].d *= 2; })),
    ecs.registerSystem(viewSystem([E], (ents) => { for (let i = 0; i < ents.length; i++) ents[i].e *= 2; })),
  ];
  for (let i = 0; i < N; i++) ecs.createEntity(comps.map((K) => ({ component: K, args: [1] })));
  return {
    step: () => ecs.update(1),
    check(ticks) {
      for (let k = 0; k < 5; k++) {
        const ents = systems[k].view.entities;
        let s = 0;
        for (const e of ents) s += e[keys[k]];
        if (ents.length !== N || s !== N * 2 ** ticks) return `${keys[k]}: count=${ents.length} sum=${s}`;
      }
      return null;
    },
  };
}

function simpleIter() {
  const ecs = new ECS();
  const move = ecs.registerSystem(viewSystem([Position, Velocity], (ents) => {
    for (let i = 0; i < ents.length; i++) {
      const e = ents[i];
      e.x += e.dx;
      e.y += e.dy;
    }
  }));
  for (const extra of [[], [A], [B], [A, B]]) {
    for (let i = 0; i < N; i++) {
      ecs.createEntity([{ component: Position }, { component: Velocity, args: [1, 2] }, ...extra.map((K) => ({ component: K }))]);
    }
  }
  return {
    step: () => ecs.update(1),
    check(ticks) {
      let sx = 0, sy = 0;
      for (const e of move.view.entities) { sx += e.x; sy += e.y; }
      return sx === 4 * N * ticks && sy === 8 * N * ticks ? null : `sums x=${sx} y=${sy} n=${move.view.entities.length}`;
    },
  };
}

function fragIter() {
  const ecs = new ECS();
  const sys = ecs.registerSystem(viewSystem([Data], (ents) => {
    for (let i = 0; i < ents.length; i++) ents[i].data *= 2;
  }));
  for (const L of Letters) {
    for (let i = 0; i < 100; i++) ecs.createEntity([{ component: L }, { component: Data, args: [1] }]);
  }
  return {
    step: () => ecs.update(1),
    check(ticks) {
      let s = 0;
      for (const e of sys.view.entities) s += e.data;
      return s === 2600 * 2 ** ticks ? null : `sum=${s}`;
    },
  };
}

function entityCycle() {
  const ecs = new ECS();
  const spawn = ecs.registerSystem(viewSystem([A], (ents, ecs) => {
    for (let i = 0; i < ents.length; i++) ecs.createEntity([{ component: B }]);
  }));
  const kill = ecs.registerSystem(viewSystem([B], (ents, ecs) => {
    for (let i = ents.length - 1; i >= 0; i--) ecs.removeEntity(ents[i]);
  }));
  for (let i = 0; i < N; i++) ecs.createEntity([{ component: A }]);
  return {
    step: () => ecs.update(1),
    check() {
      const a = spawn.view.entities.length, b = kill.view.entities.length;
      return a === N && b === 0 ? null : `A=${a} B=${b}`;
    },
  };
}

function addRemove() {
  const ecs = new ECS();
  const add = ecs.registerSystem(viewSystem([A], (ents, ecs) => {
    for (let i = 0; i < ents.length; i++) ecs.addComponentsToEntity(ents[i], [{ component: B }]);
  }));
  const rem = ecs.registerSystem(viewSystem([B], (ents, ecs) => {
    for (let i = ents.length - 1; i >= 0; i--) ecs.removeComponentsFromEntity(ents[i], B);
  }));
  for (let i = 0; i < N; i++) ecs.createEntity([{ component: A }]);
  return {
    step: () => ecs.update(1),
    check() {
      const a = add.view.entities.length, b = rem.view.entities.length;
      const leaked = add.view.entities[0].components.length;
      if (a !== N || b !== 0) return `A=${a} B=${b}`;
      if (leaked !== 1) return `entity.components grows on every add/remove cycle (length ${leaked})`;
      return null;
    },
  };
}

export const variants = {
  packed_5: { 'perform-ecs': packed5 },
  simple_iter: { 'perform-ecs': simpleIter },
  frag_iter: { 'perform-ecs': fragIter },
  entity_cycle: { 'perform-ecs': entityCycle },
  add_remove: { 'perform-ecs': addRemove },
};

export function memory(n) {
  const ecs = new ECS();
  ecs.registerSystem(viewSystem([Position, Velocity], () => {}));
  for (let i = 0; i < n; i++) ecs.createEntity([{ component: Position }, { component: Velocity, args: [1, 1] }]);
  return ecs;
}
