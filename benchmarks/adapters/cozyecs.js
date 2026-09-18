// CozyECS adapters. Three flavours per scenario:
//   'cozyecs'           idiomatic: function systems + chunk loops, world.update()
//                       (structural changes inside systems go through the command buffer)
//   'cozyecs (direct)'  no systems: raw chunk loops, immediate structural changes
//   'cozyecs (forEach)' query.forEach callbacks (structural changes deferred until forEach ends)
import { World, component, f32 } from '../../dist/index.esm.js';

const N = 1000;
const LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');

// Components are world-independent in CozyECS, define them once.
const Pos = component({ x: f32, y: f32 }, { name: 'Position' });
const Vel = component({ dx: f32, dy: f32 }, { name: 'Velocity' });
const Letters = LETTERS.map((l) => component({ value: f32 }, { name: l }));
const [A, B, C, D, E] = Letters;
const Data = component({ value: f32 }, { name: 'Data' });

function sumCol(q, C, key) {
  let s = 0;
  for (const ch of q.chunks) {
    const a = ch.col(C)[key];
    for (let i = 0; i < ch.count; i++) s += a[i];
  }
  return s;
}

function expectEq(label, got, want) {
  return got === want ? null : `${label}: expected ${want}, got ${got}`;
}

// Fairness audit (r3): the iteration scenarios verify the same things in cozyecs, bitecs4 and
// harmony-ecs: exact entity count, the chunk/archetype layout where the library has one, and EVERY
// entity's value (not only the sum, which could hide one row updated twice and another skipped).
function checkEach(q, C, key, want, count, chunkSizes) {
  const chunks = q.chunks.filter((ch) => ch.count > 0);
  const sizes = chunks.map((ch) => ch.count).join(',');
  if (sizes !== chunkSizes) return `${C.name} chunk sizes: expected [${chunkSizes}], got [${sizes}]`;
  let n = 0;
  for (const ch of chunks) {
    const a = ch.col(C)[key];
    for (let i = 0; i < ch.count; i++, n++) if (a[i] !== want) return `${C.name}.${key}[${i}]: expected ${want}, got ${a[i]}`;
  }
  return expectEq(`${C.name} count`, n, count) || expectEq(`${C.name} q.count()`, q.count(), count);
}
const sizesOf = (k, size) => Array(k).fill(size).join(',');

// ---------------------------------------------------------------- packed_5
function packed5(mode) {
  const w = new World();
  const comps = [A, B, C, D, E];
  const arch = w.archetype(...comps);
  w.spawnMany(arch, N, (chunk, row) => {
    for (const K of comps) chunk.col(K).value[row] = 1;
  });
  const qs = comps.map((K) => w.query({ all: [K] }));
  const makeChunkLoop = (K) => (q) => {
    const chunks = q.chunks;
    for (let c = 0; c < chunks.length; c++) {
      const ch = chunks[c];
      const v = ch.col(K).value;
      for (let i = 0, n = ch.count; i < n; i++) v[i] *= 2;
    }
  };
  let step;
  if (mode === 'systems') {
    const kern = (n, c) => { const v = c.value; for (let i = 0; i < n; i++) v[i] *= 2; };
    comps.forEach((K, i) => { const L = [K]; w.system('double' + i, { query: qs[i] }, (q) => q.forEachChunk(L, kern)); });
    step = () => w.update(0);
  } else if (mode === 'direct') {
    const loops = comps.map(makeChunkLoop);
    step = () => {
      for (let i = 0; i < 5; i++) loops[i](qs[i]);
    };
  } else {
    // forEach(components, fn): columns resolved once per chunk, not per row.
    const cols = comps.map((K) => [K]);
    const fn = (e, ch, row, c) => {
      c.value[row] *= 2;
    };
    step = () => {
      for (let i = 0; i < 5; i++) qs[i].forEach(cols[i], fn);
    };
  }
  return {
    step,
    check(ticks) {
      for (let i = 0; i < 5; i++) {
        const err =
          expectEq(`sum ${comps[i].name}`, sumCol(qs[i], comps[i], 'value'), N * 2 ** ticks) ||
          checkEach(qs[i], comps[i], 'value', 2 ** ticks, N, sizesOf(1, N));
        if (err) return err;
      }
      return null;
    },
  };
}

// ---------------------------------------------------------------- simple_iter
function simpleIter(mode) {
  const w = new World();
  const archs = [w.archetype(Pos, Vel), w.archetype(Pos, Vel, A), w.archetype(Pos, Vel, B), w.archetype(Pos, Vel, A, B)];
  for (const a of archs) {
    w.spawnMany(a, N, (ch, row) => {
      const v = ch.col(Vel);
      v.dx[row] = 1;
      v.dy[row] = 2;
    });
  }
  const q = w.query({ all: [Pos, Vel] });
  const move = (q) => {
    const chunks = q.chunks;
    for (let c = 0; c < chunks.length; c++) {
      const ch = chunks[c];
      const p = ch.col(Pos);
      const v = ch.col(Vel);
      const x = p.x, y = p.y, dx = v.dx, dy = v.dy;
      for (let i = 0, n = ch.count; i < n; i++) {
        x[i] += dx[i];
        y[i] += dy[i];
      }
    }
  };
  let step;
  if (mode === 'systems') {
    const PV2 = [Pos, Vel];
    const kern = (n, p, v) => { const x = p.x, y = p.y, dx = v.dx, dy = v.dy; for (let i = 0; i < n; i++) { x[i] += dx[i]; y[i] += dy[i]; } };
    w.system('move', { query: q }, (q) => q.forEachChunk(PV2, kern));
    step = () => w.update(0);
  } else if (mode === 'direct') {
    step = () => move(q);
  } else {
    const PV = [Pos, Vel];
    const fn = (e, ch, row, p, v) => {
      p.x[row] += v.dx[row];
      p.y[row] += v.dy[row];
    };
    step = () => q.forEach(PV, fn);
  }
  return {
    step,
    check(ticks) {
      return (
        expectEq('sum x', sumCol(q, Pos, 'x'), 4 * N * ticks) ||
        expectEq('sum y', sumCol(q, Pos, 'y'), 4 * N * 2 * ticks) ||
        checkEach(q, Pos, 'x', ticks, 4 * N, sizesOf(4, N)) ||
        checkEach(q, Pos, 'y', 2 * ticks, 4 * N, sizesOf(4, N))
      );
    },
  };
}

// ---------------------------------------------------------------- frag_iter
function fragIter(mode) {
  const w = new World();
  for (const L of Letters) {
    w.spawnMany(w.archetype(L, Data), 100, (ch, row) => {
      ch.col(Data).value[row] = 1;
    });
  }
  const q = w.query({ all: [Data] });
  const dbl = (q) => {
    const chunks = q.chunks;
    for (let c = 0; c < chunks.length; c++) {
      const ch = chunks[c];
      const v = ch.col(Data).value;
      for (let i = 0, n = ch.count; i < n; i++) v[i] *= 2;
    }
  };
  let step;
  if (mode === 'systems') {
    const DD = [Data];
    const kern = (n, d) => { const v = d.value; for (let i = 0; i < n; i++) v[i] *= 2; };
    w.system('data', { query: q }, (q) => q.forEachChunk(DD, kern));
    step = () => w.update(0);
  } else if (mode === 'direct') {
    step = () => dbl(q);
  } else {
    const D = [Data];
    const fn = (e, ch, row, d) => {
      d.value[row] *= 2;
    };
    step = () => q.forEach(D, fn);
  }
  return {
    step,
    check(ticks) {
      return (
        expectEq('sum Data', sumCol(q, Data, 'value'), 26 * 100 * 2 ** ticks) ||
        expectEq('count', q.count(), 2600) ||
        checkEach(q, Data, 'value', 2 ** ticks, 2600, sizesOf(26, 100))
      );
    },
  };
}

// ---------------------------------------------------------------- entity_cycle
function entityCycle(mode) {
  const w = new World();
  const archA = w.archetype(A);
  const archB = w.archetype(B);
  w.spawnMany(archA, N);
  const qA = w.query({ all: [A] });
  const qB = w.query({ all: [B] });
  let step;
  if (mode === 'systems') {
    w.system('spawnB', { query: qA }, (q, dt, world) => {
      const chunks = q.chunks;
      for (let c = 0; c < chunks.length; c++) {
        for (let i = 0, n = chunks[c].count; i < n; i++) world.spawn(archB);
      }
    });
    w.system('killB', { query: qB }, (q, dt, world) => {
      const chunks = q.chunks;
      for (let c = 0; c < chunks.length; c++) {
        const ch = chunks[c];
        const ents = ch.entities;
        for (let i = ch.count - 1; i >= 0; i--) world.destroy(ents[i]);
      }
    });
    step = () => w.update(0);
  } else if (mode === 'direct') {
    step = () => {
      const ca = qA.chunks;
      for (let c = 0; c < ca.length; c++) {
        for (let i = 0, n = ca[c].count; i < n; i++) w.spawn(archB);
      }
      const cb = qB.chunks;
      for (let c = 0; c < cb.length; c++) {
        const ch = cb[c];
        // Reverse: destroy swap-removes the last row into i, which was already visited.
        for (let i = ch.count - 1; i >= 0; i--) w.destroy(ch.entities[i]);
      }
    };
  } else {
    const spawnB = () => {
      w.spawn(archB);
    };
    const kill = (e) => {
      w.destroy(e);
    };
    step = () => {
      qA.forEach(spawnB);
      qB.forEach(kill);
    };
  }
  return {
    step,
    check() {
      return expectEq('A count', qA.count(), N) || expectEq('B count', qB.count(), 0) || expectEq('alive', w._entities.aliveCount, N);
    },
  };
}

// ---------------------------------------------------------------- add_remove
function addRemove(mode) {
  const w = new World();
  w.spawnMany(w.archetype(A), N);
  const qA = w.query({ all: [A] });
  const qAnoB = w.query({ all: [A], none: [B] });
  const qB = w.query({ all: [B] });
  let step;
  const addB = (q, world) => {
    const chunks = q.chunks;
    for (let c = 0; c < chunks.length; c++) {
      const ch = chunks[c];
      for (let i = ch.count - 1; i >= 0; i--) world.add(ch.entities[i], B);
    }
  };
  const remB = (q, world) => {
    const chunks = q.chunks;
    for (let c = 0; c < chunks.length; c++) {
      const ch = chunks[c];
      for (let i = ch.count - 1; i >= 0; i--) world.remove(ch.entities[i], B);
    }
  };
  if (mode === 'systems') {
    w.system('addB', { query: qAnoB }, (q, dt, world) => addB(q, world));
    w.system('remB', { query: qB }, (q, dt, world) => remB(q, world));
    step = () => w.update(0);
  } else if (mode === 'direct') {
    step = () => {
      addB(qAnoB, w);
      remB(qB, w);
    };
  } else {
    const add = (e) => w.add(e, B);
    const rem = (e) => w.remove(e, B);
    step = () => {
      qAnoB.forEach(add);
      qB.forEach(rem);
    };
  }
  return {
    step,
    check() {
      return expectEq('A count', qA.count(), N) || expectEq('B count', qB.count(), 0);
    },
    // Extra check: half a step (only the add) must give 1000 B.
    checkHalf() {
      if (mode !== 'direct') return null;
      addB(qAnoB, w);
      const err = expectEq('B after add', qB.count(), N);
      remB(qB, w);
      return err;
    },
  };
}

function modes(factory) {
  return {
    cozyecs: () => factory('systems'),
    'cozyecs (direct)': () => factory('direct'),
    'cozyecs (forEach)': () => factory('forEach'),
  };
}

export const variants = {
  packed_5: modes(packed5),
  simple_iter: modes(simpleIter),
  frag_iter: modes(fragIter),
  entity_cycle: modes(entityCycle),
  add_remove: modes(addRemove),
};

export function memory(n) {
  const w = new World();
  const arch = w.archetype(Pos, Vel);
  for (let i = 0; i < n; i++) {
    const e = w.spawn(arch);
    w.set(e, Pos, { x: i, y: i });
    w.set(e, Vel, { dx: 1, dy: 1 });
  }
  // Fairness audit (r5): every memory probe holds a Position+Velocity query (what a movement system
  // needs), so libraries whose queries index entities (sparse sets, cached arrays) pay for it everywhere.
  const q = w.query({ all: [Pos, Vel] });
  return { w, q, n: q.count() };
}
