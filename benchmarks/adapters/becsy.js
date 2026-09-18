// @lastolivegames/becsy 0.15 adapter. Uses the `perf` build (checks stripped, as recommended for
// production); set BECSY_CHECKS=1 to run against the checked dev build instead.
// Idiomatic per its docs: component classes with a static `schema`, System subclasses declaring
// queries/access in field initializers, entities created before the first frame with
// world.createEntity(), frames run with `await world.execute()` (async API).
// Verification uses a Checker system that is stopped during benchmarking and restarted (with the
// workload systems stopped) only when check() runs, so it adds no per-frame cost.
const becsy = await import(process.env.BECSY_CHECKS ? '@lastolivegames/becsy/index.js' : '@lastolivegames/becsy/perf.js');
const { World, System, Type } = becsy;

const N = 1000;
const LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');

function valueComponent(name) {
  const C = { [name]: class {} }[name];
  C.schema = { value: Type.float32 };
  return C;
}

function named(name, cls) {
  Object.defineProperty(cls, 'name', { value: name });
  return cls;
}

// Build a world whose workload systems can be paused so a Checker system can inspect the state.
async function makeWorld({ components, systems, checkQuery, checkFn, options }) {
  const result = { value: null };
  const Checker = named(
    'Checker',
    class extends System {
      q = this.query(checkQuery);
      execute() {
        result.value = checkFn(this.q.current);
      }
    },
  );
  const world = await World.create({ defs: [...components, ...systems, Checker], ...options });
  world.control({ stop: [Checker], restart: [] });
  return {
    world,
    async check(...args) {
      world.control({ stop: systems, restart: [Checker] });
      result.value = undefined;
      await world.execute();
      world.control({ stop: [Checker], restart: systems });
      const fn = result.value;
      return typeof fn === 'function' ? fn(...args) : fn;
    },
  };
}

async function packed5() {
  const comps = ['A', 'B', 'C', 'D', 'E'].map(valueComponent);
  const [A, B, C, D, E] = comps;
  // Fairness audit (r5): one hand-written system per component, as an application would have. A single
  // system body shared by all 5 systems (created in a loop) made its property accesses megamorphic
  // (5 component classes at one source site), which only hurts object/accessor-based libraries.
  const systems = [
    named('Double0', class extends System {
      q = this.query((q) => q.current.with(A).write);
      execute() { for (const entity of this.q.current) entity.write(A).value *= 2; }
    }),
    named('Double1', class extends System {
      q = this.query((q) => q.current.with(B).write);
      execute() { for (const entity of this.q.current) entity.write(B).value *= 2; }
    }),
    named('Double2', class extends System {
      q = this.query((q) => q.current.with(C).write);
      execute() { for (const entity of this.q.current) entity.write(C).value *= 2; }
    }),
    named('Double3', class extends System {
      q = this.query((q) => q.current.with(D).write);
      execute() { for (const entity of this.q.current) entity.write(D).value *= 2; }
    }),
    named('Double4', class extends System {
      q = this.query((q) => q.current.with(E).write);
      execute() { for (const entity of this.q.current) entity.write(E).value *= 2; }
    }),
  ];
  const { world, check } = await makeWorld({
    components: comps,
    systems,
    checkQuery: (q) => q.current.with(...comps),
    checkFn: (entities) => {
      const sums = comps.map((K) => entities.reduce((s, e) => s + e.read(K).value, 0));
      return (ticks) => {
        for (const s of sums) if (s !== N * 2 ** ticks) return `sum: expected ${N * 2 ** ticks}, got ${s}`;
        return null;
      };
    },
    options: { maxEntities: 2 * N, maxShapeChangesPerFrame: 16 * N },
  });
  const init = [];
  for (const K of comps) init.push(K, { value: 1 });
  for (let i = 0; i < N; i++) world.createEntity(...init);
  return { async: true, step: () => world.execute(), check };
}

async function simpleIter() {
  const Position = named('Position', class { static schema = { x: Type.float32, y: Type.float32 }; });
  const Velocity = named('Velocity', class { static schema = { dx: Type.float32, dy: Type.float32 }; });
  const A = valueComponent('A');
  const B = valueComponent('B');
  const Move = named(
    'Move',
    class extends System {
      q = this.query((q) => q.current.with(Velocity).with(Position).write);
      execute() {
        for (const entity of this.q.current) {
          const v = entity.read(Velocity);
          const p = entity.write(Position);
          p.x += v.dx;
          p.y += v.dy;
        }
      }
    },
  );
  const { world, check } = await makeWorld({
    components: [Position, Velocity, A, B],
    systems: [Move],
    checkQuery: (q) => q.current.with(Position, Velocity),
    checkFn: (entities) => {
      let sx = 0, sy = 0;
      for (const e of entities) {
        const p = e.read(Position);
        sx += p.x;
        sy += p.y;
      }
      const n = entities.length;
      return (ticks) => (n !== 4 * N || sx !== 4 * N * ticks || sy !== 8 * N * ticks ? `n=${n} sums x=${sx} y=${sy}` : null);
    },
    options: { maxEntities: 4 * N + 16, maxShapeChangesPerFrame: 32 * N },
  });
  for (const extra of [[], [A], [B], [A, B]]) {
    for (let i = 0; i < N; i++) world.createEntity(Position, Velocity, { dx: 1, dy: 2 }, ...extra);
  }
  return { async: true, step: () => world.execute(), check };
}

async function fragIter() {
  const Data = valueComponent('Data');
  const letters = LETTERS.map(valueComponent);
  const DataSystem = named(
    'DataSystem',
    class extends System {
      q = this.query((q) => q.current.with(Data).write);
      execute() {
        for (const entity of this.q.current) entity.write(Data).value *= 2;
      }
    },
  );
  const { world, check } = await makeWorld({
    components: [Data, ...letters],
    systems: [DataSystem],
    checkQuery: (q) => q.current.with(Data),
    checkFn: (entities) => {
      const n = entities.length;
      const s = entities.reduce((a, e) => a + e.read(Data).value, 0);
      return (ticks) => (n !== 2600 || s !== 2600 * 2 ** ticks ? `count=${n} sum=${s}` : null);
    },
    options: { maxEntities: 2600 + 16, maxShapeChangesPerFrame: 16 * 2600 },
  });
  for (const L of letters) for (let i = 0; i < 100; i++) world.createEntity(L, Data, { value: 1 });
  return { async: true, step: () => world.execute(), check };
}

async function entityCycle() {
  const A = valueComponent('A');
  const B = valueComponent('B');
  const SpawnB = named(
    'SpawnB',
    class extends System {
      qa = this.query((q) => q.current.with(A));
      qb = this.query((q) => q.using(B).write);
      execute() {
        const as = this.qa.current;
        for (let i = 0; i < as.length; i++) this.createEntity(B);
      }
    },
  );
  const KillB = named(
    'KillB',
    class extends System {
      sched = this.schedule((s) => s.after(SpawnB));
      qb = this.query((q) => q.current.with(B).write);
      execute() {
        for (const entity of this.qb.current) entity.delete();
      }
    },
  );
  const { world, check } = await makeWorld({
    components: [A, B],
    systems: [SpawnB, KillB],
    checkQuery: (q) => q.current.withAny(A, B),
    checkFn: (entities) => {
      let a = 0, b = 0;
      for (const e of entities) {
        if (e.has(A)) a++;
        if (e.has(B)) b++;
      }
      return () => (a !== N || b !== 0 ? `A=${a} B=${b}` : null);
    },
    // deleted entities and their components stay in limbo for a couple of frames
    options: { maxEntities: 8 * N, maxLimboComponents: 8 * N, maxShapeChangesPerFrame: 16 * N },
  });
  for (let i = 0; i < N; i++) world.createEntity(A);
  return { async: true, step: () => world.execute(), check };
}

async function addRemove() {
  const A = valueComponent('A');
  const B = valueComponent('B');
  const AddB = named(
    'AddB',
    class extends System {
      q = this.query((q) => q.current.with(A).without(B).using(B).write);
      execute() {
        for (const entity of this.q.current) entity.add(B);
      }
    },
  );
  const RemoveB = named(
    'RemoveB',
    class extends System {
      sched = this.schedule((s) => s.after(AddB));
      q = this.query((q) => q.current.with(B).write);
      execute() {
        for (const entity of this.q.current) entity.remove(B);
      }
    },
  );
  const { world, check } = await makeWorld({
    components: [A, B],
    systems: [AddB, RemoveB],
    checkQuery: (q) => q.current.withAny(A, B),
    checkFn: (entities) => {
      let a = 0, b = 0;
      for (const e of entities) {
        if (e.has(A)) a++;
        if (e.has(B)) b++;
      }
      return () => (a !== N || b !== 0 ? `A=${a} B=${b}` : null);
    },
    options: { maxEntities: 2 * N, maxLimboComponents: 8 * N, maxShapeChangesPerFrame: 16 * N },
  });
  for (let i = 0; i < N; i++) world.createEntity(A);
  return { async: true, step: () => world.execute(), check };
}

export const variants = {
  packed_5: { becsy: packed5 },
  simple_iter: { becsy: simpleIter },
  frag_iter: { becsy: fragIter },
  entity_cycle: { becsy: entityCycle },
  add_remove: { becsy: addRemove },
};

export async function memory(n) {
  const Position = named('Position', class { static schema = { x: Type.float32, y: Type.float32 }; });
  const Velocity = named('Velocity', class { static schema = { dx: Type.float32, dy: Type.float32 }; });
  // Fairness audit (r5): every memory probe holds a Position+Velocity query (what a movement system
  // needs), so libraries whose queries index entities (sparse sets, cached arrays) pay for it everywhere.
  const Move = named('Move', class extends System {
    q = this.query((q) => q.current.with(Velocity).with(Position).write);
  });
  const world = await World.create({ defs: [Position, Velocity, Move], maxEntities: n + 16, maxShapeChangesPerFrame: 4 * n + 64 });
  for (let i = 0; i < n; i++) world.createEntity(Position, { x: i, y: i }, Velocity, { dx: 1, dy: 1 });
  await world.execute();
  return world;
}
