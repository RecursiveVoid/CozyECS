// bitECS 0.3.x adapter (SoA typed arrays, sparse-set queries).
import {
  createWorld,
  defineComponent,
  Types,
  addEntity,
  addComponent,
  removeComponent,
  removeEntity,
  defineQuery,
  resetGlobals,
} from 'bitecs';

const N = 1000;
const LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
const val = () => defineComponent({ value: Types.f32 });

function packed5() {
  const world = createWorld();
  const comps = [val(), val(), val(), val(), val()];
  for (let i = 0; i < N; i++) {
    const e = addEntity(world);
    for (const C of comps) {
      addComponent(world, C, e);
      C.value[e] = 1;
    }
  }
  const systems = comps.map((C) => {
    const q = defineQuery([C]);
    const v = C.value;
    return (world) => {
      const ents = q(world);
      for (let i = 0; i < ents.length; i++) v[ents[i]] *= 2;
    };
  });
  return {
    step() {
      for (let i = 0; i < 5; i++) systems[i](world);
    },
    check(ticks) {
      for (const C of comps) {
        const ents = defineQuery([C])(world);
        let s = 0;
        for (const e of ents) s += C.value[e];
        if (s !== N * 2 ** ticks) return `sum: expected ${N * 2 ** ticks}, got ${s}`;
      }
      return null;
    },
  };
}

function simpleIter() {
  const world = createWorld();
  const Pos = defineComponent({ x: Types.f32, y: Types.f32 });
  const Vel = defineComponent({ dx: Types.f32, dy: Types.f32 });
  const A = val();
  const B = val();
  const kinds = [[], [A], [B], [A, B]];
  for (const extra of kinds) {
    for (let i = 0; i < N; i++) {
      const e = addEntity(world);
      addComponent(world, Pos, e);
      addComponent(world, Vel, e);
      for (const C of extra) addComponent(world, C, e);
      Vel.dx[e] = 1;
      Vel.dy[e] = 2;
    }
  }
  const q = defineQuery([Pos, Vel]);
  const { x, y } = Pos;
  const { dx, dy } = Vel;
  return {
    step() {
      const ents = q(world);
      for (let i = 0; i < ents.length; i++) {
        const e = ents[i];
        x[e] += dx[e];
        y[e] += dy[e];
      }
    },
    check(ticks) {
      let sx = 0, sy = 0;
      for (const e of q(world)) { sx += x[e]; sy += y[e]; }
      if (sx !== 4 * N * ticks || sy !== 8 * N * ticks) return `sums x=${sx} y=${sy}`;
      return null;
    },
  };
}

function fragIter() {
  const world = createWorld();
  const Data = val();
  for (const _ of LETTERS) {
    const L = val();
    for (let i = 0; i < 100; i++) {
      const e = addEntity(world);
      addComponent(world, L, e);
      addComponent(world, Data, e);
      Data.value[e] = 1;
    }
  }
  const q = defineQuery([Data]);
  const v = Data.value;
  return {
    step() {
      const ents = q(world);
      for (let i = 0; i < ents.length; i++) v[ents[i]] *= 2;
    },
    check(ticks) {
      let s = 0;
      const ents = q(world);
      for (const e of ents) s += v[e];
      if (ents.length !== 2600 || s !== 2600 * 2 ** ticks) return `count=${ents.length} sum=${s}`;
      return null;
    },
  };
}

function entityCycle() {
  const world = createWorld();
  const A = val();
  const B = val();
  for (let i = 0; i < N; i++) addComponent(world, A, addEntity(world));
  const qA = defineQuery([A]);
  const qB = defineQuery([B]);
  return {
    step() {
      const as = qA(world);
      for (let i = 0; i < as.length; i++) addComponent(world, B, addEntity(world));
      const bs = qB(world);
      for (let i = bs.length - 1; i >= 0; i--) removeEntity(world, bs[i]);
    },
    check() {
      const a = qA(world).length, b = qB(world).length;
      if (a !== N || b !== 0) return `A=${a} B=${b}`;
      return null;
    },
  };
}

function addRemove() {
  const world = createWorld();
  const A = val();
  const B = val();
  for (let i = 0; i < N; i++) addComponent(world, A, addEntity(world));
  const qA = defineQuery([A]);
  const qB = defineQuery([B]);
  return {
    step() {
      const as = qA(world);
      for (let i = 0; i < as.length; i++) addComponent(world, B, as[i]);
      const bs = qB(world);
      for (let i = bs.length - 1; i >= 0; i--) removeComponent(world, B, bs[i]);
    },
    check() {
      const a = qA(world).length, b = qB(world).length;
      if (a !== N || b !== 0) return `A=${a} B=${b}`;
      return null;
    },
  };
}

export const variants = {
  packed_5: { bitecs: packed5 },
  simple_iter: { bitecs: simpleIter },
  frag_iter: { bitecs: fragIter },
  entity_cycle: { bitecs: entityCycle },
  add_remove: { bitecs: addRemove },
};

export function memory(n) {
  // Entity ids are process-global in bitECS 0.3 and capped at the default size (100k). Reset them so
  // memory.js can warm the code path with a small world first, like every other library
  // (fairness audit r5: bitecs used to be the only library measured without warm-up).
  resetGlobals();
  const world = createWorld();
  const Pos = defineComponent({ x: Types.f32, y: Types.f32 });
  const Vel = defineComponent({ dx: Types.f32, dy: Types.f32 });
  for (let i = 0; i < n; i++) {
    const e = addEntity(world);
    addComponent(world, Pos, e);
    addComponent(world, Vel, e);
    Pos.x[e] = i; Pos.y[e] = i; Vel.dx[e] = 1; Vel.dy[e] = 1;
  }
  // Fairness audit (r5): every memory probe holds a Position+Velocity query (what a movement system
  // needs), so libraries whose queries index entities (sparse sets, cached arrays) pay for it everywhere.
  const q = defineQuery([Pos, Vel]);
  return { world, Pos, Vel, q, n: q(world).length };
}
