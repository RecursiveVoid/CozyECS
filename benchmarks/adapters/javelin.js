// @javelin/ecs 1.0.0-alpha adapter. Idiomatic per its README/docs: schemas of `number` fields,
// pooled object components created with component(), systems registered with world.addSystem()
// and run by world.step(); queries iterated per archetype as `for (const [entities, [a]] of query)`.
// All structural changes (create/attach/detach/destroy) are deferred and applied at the start of
// the next world.step(), so results lag one step behind; checks account for that.
import { createWorld, createQuery, component, number } from '@javelin/ecs';

const N = 1000;
const LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
const valueSchema = () => ({ value: number });

function count(q) {
  let n = 0;
  for (const [entities] of q) n += entities.length;
  return n;
}

function packed5() {
  const comps = [0, 1, 2, 3, 4].map(valueSchema);
  const queries = comps.map((C) => createQuery(C));
  const world = createWorld();
  for (let i = 0; i < N; i++) world.create(...comps.map((C) => component(C, { value: 1 })));
  world.step(); // apply the deferred creates
  const [qA, qB, qC, qD, qE] = queries;
  // Fairness audit (r5): one hand-written system per component, as an application would have. A single
  // system body shared by all 5 systems (created in a loop) made its property accesses megamorphic
  // (5 component classes at one source site), which only hurts object/accessor-based libraries.
  world.addSystem(() => { for (const [entities, [c]] of qA) for (let j = 0; j < entities.length; j++) c[j].value *= 2; });
  world.addSystem(() => { for (const [entities, [c]] of qB) for (let j = 0; j < entities.length; j++) c[j].value *= 2; });
  world.addSystem(() => { for (const [entities, [c]] of qC) for (let j = 0; j < entities.length; j++) c[j].value *= 2; });
  world.addSystem(() => { for (const [entities, [c]] of qD) for (let j = 0; j < entities.length; j++) c[j].value *= 2; });
  world.addSystem(() => { for (const [entities, [c]] of qE) for (let j = 0; j < entities.length; j++) c[j].value *= 2; });
  const bound = queries.map((q) => q.bind(world));
  return {
    step() {
      world.step();
    },
    check(ticks) {
      for (const q of bound) {
        let s = 0;
        for (const [entities, [c]] of q) for (let j = 0; j < entities.length; j++) s += c[j].value;
        if (s !== N * 2 ** ticks) return `sum: expected ${N * 2 ** ticks}, got ${s}`;
      }
      return null;
    },
  };
}

function simpleIter() {
  const Position = { x: number, y: number };
  const Velocity = { dx: number, dy: number };
  const A = valueSchema();
  const B = valueSchema();
  const world = createWorld();
  for (const extra of [[], [A], [B], [A, B]]) {
    for (let i = 0; i < N; i++) {
      world.create(component(Position), component(Velocity, { dx: 1, dy: 2 }), ...extra.map((C) => component(C)));
    }
  }
  world.step();
  const kinetics = createQuery(Position, Velocity);
  world.addSystem(() => {
    for (const [entities, [p, v]] of kinetics) {
      for (let j = 0; j < entities.length; j++) {
        const pj = p[j];
        const vj = v[j];
        pj.x += vj.dx;
        pj.y += vj.dy;
      }
    }
  });
  const bound = kinetics.bind(world);
  return {
    step() {
      world.step();
    },
    check(ticks) {
      let sx = 0, sy = 0, n = 0;
      for (const [entities, [p]] of bound) {
        for (let j = 0; j < entities.length; j++, n++) {
          sx += p[j].x;
          sy += p[j].y;
        }
      }
      if (n !== 4 * N || sx !== 4 * N * ticks || sy !== 8 * N * ticks) return `n=${n} sums x=${sx} y=${sy}`;
      return null;
    },
  };
}

function fragIter() {
  const Data = valueSchema();
  const world = createWorld();
  for (const _ of LETTERS) {
    const L = valueSchema();
    for (let i = 0; i < 100; i++) world.create(component(L), component(Data, { value: 1 }));
  }
  world.step();
  const q = createQuery(Data);
  world.addSystem(() => {
    for (const [entities, [d]] of q) {
      for (let j = 0; j < entities.length; j++) d[j].value *= 2;
    }
  });
  const bound = q.bind(world);
  return {
    step() {
      world.step();
    },
    check(ticks) {
      let s = 0, n = 0;
      for (const [entities, [d]] of bound) for (let j = 0; j < entities.length; j++, n++) s += d[j].value;
      if (n !== 2600 || s !== 2600 * 2 ** ticks) return `count=${n} sum=${s}`;
      return null;
    },
  };
}

function entityCycle() {
  const A = valueSchema();
  const B = valueSchema();
  const world = createWorld();
  for (let i = 0; i < N; i++) world.create(component(A));
  world.step();
  const qA = createQuery(A);
  const qB = createQuery(B);
  world.addSystem((world) => {
    for (const [entities] of qA) {
      for (let j = 0; j < entities.length; j++) world.create(component(B));
    }
  });
  world.addSystem((world) => {
    for (const [entities] of qB) {
      for (let j = 0; j < entities.length; j++) world.destroy(entities[j]);
    }
  });
  const bA = qA.bind(world);
  const bB = qB.bind(world);
  let steps = 0;
  return {
    step() {
      world.step();
      steps++;
    },
    // Steady state after >= 2 steps: the N B-entities created last step are live (their destroy is
    // queued), i.e. every step creates N and destroys N entities.
    check() {
      const a = count(bA), b = count(bB);
      const wantB = steps >= 2 ? N : 0;
      if (a !== N || b !== wantB) return `A=${a} B=${b} (expected B=${wantB} with one-step deferral)`;
      return null;
    },
  };
}

function addRemove() {
  const A = valueSchema();
  const B = valueSchema();
  const world = createWorld();
  for (let i = 0; i < N; i++) world.create(component(A));
  world.step();
  const qA = createQuery(A);
  const qAnoB = createQuery(A).not(B);
  const qB = createQuery(B);
  world.addSystem((world) => {
    for (const [entities] of qAnoB) {
      for (let j = 0; j < entities.length; j++) world.attach(entities[j], component(B));
    }
  });
  world.addSystem((world) => {
    for (const [entities] of qB) {
      for (let j = 0; j < entities.length; j++) world.detach(entities[j], B);
    }
  });
  const bA = qA.bind(world);
  const bB = qB.bind(world);
  let ops = 0;
  return {
    // Changes land one world.step() later, so the add pass and the remove pass alternate between
    // world steps; two world steps = N attaches + N detaches applied, the same work as one op of
    // the other libraries.
    step() {
      world.step();
      world.step();
      ops++;
    },
    // After every op the N attaches are applied and N detaches are queued for the next step.
    check() {
      const a = count(bA), b = count(bB);
      const wantB = ops >= 1 ? N : 0;
      if (a !== N || b !== wantB) return `A=${a} B=${b} (expected B=${wantB} with one-step deferral)`;
      return null;
    },
  };
}

export const variants = {
  packed_5: { javelin: packed5 },
  simple_iter: { javelin: simpleIter },
  frag_iter: { javelin: fragIter },
  entity_cycle: { javelin: entityCycle },
  add_remove: { javelin: addRemove },
};

export function memory(n) {
  const Position = { x: number, y: number };
  const Velocity = { dx: number, dy: number };
  const world = createWorld();
  for (let i = 0; i < n; i++) world.create(component(Position, { x: i, y: i }), component(Velocity, { dx: 1, dy: 1 }));
  world.step();
  const q = createQuery(Position, Velocity).bind(world);
  return { world, q };
}
