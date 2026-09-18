import { describe, test, expect } from '@jest/globals';
import { World, component, f32, i32 } from '../src/index';
import type { ComponentType } from '../src/component';
import { compileCPUKernel, estimateCPUNanosPerEntity, generateCPUSource, CPU_NS_PER_OP, CPU_NS_PER_ENTITY_FIXED } from '../src/gpu/cpu';
import type { CompiledCPUKernel } from '../src/gpu/cpu';
import { parseKernel } from '../src/gpu/parse';
import type { ParseSpec } from '../src/gpu/parse';
import type { IRComponent, IRFieldKind, KernelIR } from '../src/gpu/ir';
import { cozyRandRef, storeI32Ref, storeU32Ref } from '../src/gpu/wgsl';

// Kernels are given as SOURCE STRINGS: jest instruments functions for coverage,
// which would change what `toString()` returns.

const comp = (index: number, id: number, name: string, fields: [string, IRFieldKind][]): IRComponent => ({
  index,
  id,
  name,
  fields: fields.map(([n, kind], i) => ({ name: n, kind, index: i })),
});

const Pos = comp(0, 1, 'Position', [['x', 'f32'], ['y', 'f32']]);
const Vel = comp(1, 2, 'Velocity', [['x', 'f32'], ['y', 'f32']]);
const Stats = comp(0, 3, 'Stats', [['hp', 'i32'], ['gold', 'u32'], ['lvl', 'u8'], ['wide', 'f64']]);
const Body = comp(0, 10, 'Body', [['x', 'f32'], ['y', 'f32']]);
const Mot = comp(1, 11, 'Mot', [['vx', 'f32'], ['vy', 'f32']]);

function parse(source: string, opts: Partial<ParseSpec> = {}): KernelIR {
  const names = opts.uniformNames || [];
  return parseKernel(null, {
    name: opts.name || 'K',
    form: opts.form || 'per-entity',
    components: opts.components || [Pos, Vel],
    uniformNames: names,
    uniformInitials: opts.uniformInitials || names.map(() => 0),
    source,
    maxLoopIterations: opts.maxLoopIterations,
  });
}

/** Opaque ComponentType tokens (compileCPUKernel only hands them back). */
const tokens = (n: number): ComponentType[] => Array.from({ length: n }, () => ({}) as ComponentType);

/** Seeded PRNG (mulberry32). */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

type Cols = Record<string, Float32Array | Float64Array | Int32Array | Uint32Array | Uint8Array>;

function randomF32(n: number, r: () => number, lo = -10, hi = 10): Float32Array {
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = lo + (hi - lo) * r();
  return a;
}

const cloneCols = (cols: Cols[]): Cols[] =>
  cols.map((c) => {
    const o: Cols = {};
    for (const k of Object.keys(c)) o[k] = c[k].slice();
    return o;
  });

/** Runs a compiled kernel over one fake chunk. */
function run(k: CompiledCPUKernel, count: number, cols: Cols[], chunk: unknown = { enabled: [] }): void {
  k.fn(count, ...cols, chunk);
}

const fr = Math.fround;

// ---------------------------------------------------------------------------
// generateCPUSource
// ---------------------------------------------------------------------------

describe('generateCPUSource', () => {
  test('per-entity move kernel: hoisted columns, row cache, fround at every f32 rounding point', () => {
    const ir = parse('(p, v, dt, u) => { v.y += u.gravity * dt; p.x += v.x * dt; p.y += v.y * dt; }', {
      name: 'Move',
      uniformNames: ['gravity'],
    });
    expect(generateCPUSource(ir)).toBe(
      [
        '"use strict";',
        '// cozyecs/gpu CPU backend: kernel "Move", form per-entity, ops 11',
        'const st = state, U = st.u, fr = Math.fround, rand = cozy_rand;',
        'return function cozyKernel_Move(count, c0, c1, chunk) {',
        '  const dt = fr(st.dt), u0_gravity = fr(U[0]);',
        '  const h0 = fr(u0_gravity * dt);',
        '  const a0_0 = c0["x"], a0_1 = c0["y"], a1_0 = c1["x"], a1_1 = c1["y"];',
        '  for (let i = 0; i < count; i++) {',
        '    const r1_0 = a1_0[i];',
        '    let r0_0 = a0_0[i], r0_1 = a0_1[i], r1_1 = a1_1[i];',
        '    {',
        '      a1_1[i] = r1_1 = fr(r1_1 + h0);',
        '      a0_0[i] = r0_0 = fr(r0_0 + fr(r1_0 * dt));',
        '      a0_1[i] = r0_1 = fr(r0_1 + fr(r1_1 * dt));',
        '    }',
        '  }',
        '};',
        '',
      ].join('\n'),
    );
  });

  test('integer writes use saturating round-half-up helpers; non-f32 reads are rounded', () => {
    const ir = parse('(s) => { s.hp = s.hp * 1.5; s.gold -= 1; s.lvl += 0.5; s.wide = s.wide * 0.1; }', {
      components: [Stats],
    });
    const src = generateCPUSource(ir);
    expect(src).toContain(
      'const s_i32 = (x) => { if (x !== x) return 0; const r = Math.round(x); return r < -2147483648 ? -2147483648 : r > 2147483647 ? 2147483647 : r; };',
    );
    expect(src).toContain('const s_u32 = (x) => { if (x !== x) return 0; const r = Math.round(x); return r < 0 ? 0 : r > 4294967295 ? 4294967295 : r; };');
    expect(src).toContain('const s_u8 = (x) => { if (x !== x) return 0; const r = Math.round(x); return r < 0 ? 0 : r > 255 ? 255 : r; };');
    // the row cache holds the stored (clamped) integer; i32/u32 reads round it
    expect(src).toContain('a0_0[i] = r0_0 = s_i32(fr(fr(r0_0) * 1.5));');
    expect(src).toContain('a0_1[i] = r0_1 = s_u32(fr(fr(r0_1) - 1));');
    // u8 reads are exact in f32: no fround on the read
    expect(src).toContain('a0_2[i] = r0_2 = s_u8(fr(r0_2 + 0.5));');
    // f64 is computed in f32 like everything else; the literal is the f32 nearest 0.1
    expect(src).toContain('a0_3[i] = r0_3 = fr(fr(r0_3) * 0.10000000149011612);');
    expect(src).not.toContain('s_f64');
  });

  test('deterministic, fround can be switched off, enable checks read once per chunk', () => {
    const ir = parse('(p, v) => { p.x = p.x + v.x; }');
    expect(generateCPUSource(ir)).toBe(generateCPUSource(ir));
    // fround off: the f32 row cache still holds what the Float32Array stored
    expect(generateCPUSource(ir, { fround: false })).toContain('a0_0[i] = r0_0 = fr(r0_0 + r1_0);');
    const en = generateCPUSource(ir, { enableIds: [4, 9] });
    expect(en).toContain('const g0 = chunk.enabled[4], g1 = chunk.enabled[9];');
    expect(en).toContain('if (g0[i] === 0 || g1[i] === 0) continue;');
  });

  test('loops carry the same trip cap and order as the WGSL loop; round is floor(fround(x + 0.5))', () => {
    const ir = parse('(p) => { let s = 0; for (let k = 0; k < 4; k++) { s += Math.round(p.x); } p.y = s; }', {
      components: [Pos],
    });
    const src = generateCPUSource(ir);
    expect(src).toMatch(/let t0 = 0;\n\s+for \(;; l\d+_k = fr\(l\d+_k \+ 1\)\) \{\n\s+if \(t0 >= 4\) break;\n\s+t0\+\+;\n\s+if \(!\(l\d+_k < 4\)\) break;/);
    expect(src).toMatch(/Math\.floor\(fr\(r0_0 \+ 0\.5\)\)/);
  });

  test('pairwise: nested i/j loops over the same chunk, role 1 indexed by j', () => {
    const ir = parse('(s, o, dt, u) => { s.vx += (o.x - s.x) * u.k; }', {
      form: 'pairwise',
      components: [Body, Mot],
      uniformNames: ['k'],
    });
    const src = generateCPUSource(ir, { enableIds: [10] });
    expect(src).toContain('for (let j = 0; j < count; j++) {');
    expect(src).toContain('if (j === i || g0[j] === 0) continue;');
    expect(src).toContain('a1_0[i] = r1_0 = fr(r1_0 + fr(fr(a0_0[j] - r0_0) * u0_k));');
    // self fields are loaded once per i, outside the j loop; other fields stay array reads
    expect(src.indexOf('let r1_0 = a1_0[i];')).toBeLessThan(src.indexOf('for (let j = 0'));
    expect(src).toContain('const r0_0 = a0_0[i];');
  });

  test('gravity: uniforms, dt and invariant products hoisted out of the row loop; each field loaded at most once per row', () => {
    const ir = parse(
      '(p, v, dt, u) => { v.y += u.gravity * dt; p.x += v.x * dt; p.y += v.y * dt; if (p.y < 0) { p.y = 0; v.y = -v.y * 0.5; } }',
      { name: 'Move', uniformNames: ['gravity'] },
    );
    const src = generateCPUSource(ir);
    const loopAt = src.indexOf('for (let i = 0; i < count; i++) {');
    expect(loopAt).toBeGreaterThan(0);
    const head = src.slice(0, loopAt);
    const loop = src.slice(loopAt);
    // dt, the uniform and u.gravity * dt are per-chunk constants
    expect(head).toContain('const dt = fr(st.dt), u0_gravity = fr(U[0]);');
    expect(head).toContain('const h0 = fr(u0_gravity * dt);');
    expect(loop).not.toContain('st.dt');
    expect(loop).not.toContain('U[');
    expect(loop).not.toContain('u0_gravity');
    expect(loop).toContain('fr(r1_1 + h0)');
    // every column is read at most once per row (as the row-cache load); writes store through
    for (const col of ['a0_0', 'a0_1', 'a1_0', 'a1_1']) {
      // a store target is `col[i] = ...`; anything else is a load
      const reads = loop.match(new RegExp(`\\b${col}\\[i\\](?! = )`, 'g')) || [];
      expect(reads.length).toBeLessThanOrEqual(1);
    }
    expect(loop).toContain('const r1_0 = a1_0[i];');
    expect(loop).toContain('let r0_0 = a0_0[i], r0_1 = a0_1[i], r1_1 = a1_1[i];');
    expect(loop).toContain('if ((r0_1 < 0)) {');
    expect(loop).toContain('a0_1[i] = r0_1 = 0;');
    expect(loop).toContain('a1_1[i] = r1_1 = fr((-r1_1) * 0.5);');
    // no integer coercions or enable checks on an all-f32, non-enableable kernel
    expect(loop).not.toMatch(/\|\s*0\b/);
    expect(loop).not.toContain('chunk.enabled');
    expect(loop).not.toContain('continue');
  });

  test('row cache is disabled when two kernel components share an id (aliased columns)', () => {
    const ir = parse('(a, b) => { a.x += 1; b.x += 1; }', { components: [Pos, { ...Pos, index: 1 }] });
    const src = generateCPUSource(ir);
    expect(src).not.toMatch(/\br\d+_\d+\b/);
    expect(src).toContain('a0_0[i] = fr(a0_0[i] + 1);');
    expect(src).toContain('a1_0[i] = fr(a1_0[i] + 1);');
    const k = compileCPUKernel(ir, tokens(2));
    const x = new Float32Array([1, 2]);
    run(k, 2, [{ x, y: new Float32Array(2) }, { x, y: new Float32Array(2) }]);
    expect(Array.from(x)).toEqual([3, 4]);
  });

  test('annotate adds source comments without changing the code', () => {
    const ir = parse('(p, v) => { for (let k = 0; k < 2; k++) { p.x += v.x; } }');
    const plain = generateCPUSource(ir);
    const noted = generateCPUSource(ir, { annotate: true });
    expect(noted).toContain('// ');
    const strip = (s: string) =>
      s
        .split('\n')
        .filter((l) => !/^\s*\/\/ /.test(l))
        .join('\n');
    expect(strip(noted)).toBe(strip(plain));
  });
});

// ---------------------------------------------------------------------------
// Execution parity against plain JS references
// ---------------------------------------------------------------------------

interface ParityCase {
  name: string;
  source: string;
  components?: IRComponent[];
  uniforms?: Record<string, number>;
  dt?: number;
  /** Builds the random columns for `n` rows. */
  make: (n: number, r: () => number) => Cols[];
  /** Plain JS reference, one row at a time, mirroring the f32 rounding points. */
  ref: (i: number, cols: Cols[], dt: number, u: Record<string, number>, count: number) => void;
}

const posVel = (n: number, r: () => number): Cols[] => [
  { x: randomF32(n, r), y: randomF32(n, r) },
  { x: randomF32(n, r), y: randomF32(n, r) },
];

const CASES: ParityCase[] = [
  {
    name: 'arithmetic',
    source: '(p, v, dt, u) => { p.x = p.x * v.x + v.y / 3 - p.y % 2; p.y = -p.y + u.k * dt; v.x -= 0.1; v.y *= v.y; }',
    uniforms: { k: 0.37 },
    dt: 1 / 60,
    make: posVel,
    ref: (i, [p, v], dt, u) => {
      const DT = fr(dt), K = fr(u.k);
      p.x[i] = fr(fr(fr(p.x[i] * v.x[i]) + fr(v.y[i] / 3)) - fr(p.y[i] % 2));
      p.y[i] = fr(-p.y[i] + fr(K * DT));
      v.x[i] = fr(v.x[i] - fr(0.1));
      v.y[i] = fr(v.y[i] * v.y[i]);
    },
  },
  {
    name: 'if/else, ternary, logical',
    source:
      '(p, v) => { if (p.x > 0 && v.x < 0.5) { p.y = 1; } else if (!(p.x > 0) || v.y === 2) { p.y = p.x > -1 ? 2 : 3; } else { p.y = 4; } }',
    make: posVel,
    ref: (i, [p, v]) => {
      if (p.x[i] > 0 && v.x[i] < 0.5) p.y[i] = 1;
      else if (!(p.x[i] > 0) || v.y[i] === 2) p.y[i] = p.x[i] > -1 ? 2 : 3;
      else p.y[i] = 4;
    },
  },
  {
    name: 'bounded loops with break/continue',
    source:
      '(p, v) => { let s = 0; for (let k = 0; k < 5; k++) { if (k === 3) continue; s += p.x * k; } let t = 0; while (t < v.x) { t += 1.5; if (t > 6) break; } p.x = s; p.y = t; }',
    make: posVel,
    ref: (i, [p, v]) => {
      let s = 0;
      for (let k = 0; k < 5; k++) {
        if (k === 3) continue;
        s = fr(s + fr(p.x[i] * k));
      }
      let t = 0;
      while (t < v.x[i]) {
        t = fr(t + 1.5);
        if (t > 6) break;
      }
      p.x[i] = s;
      p.y[i] = t;
    },
  },
  {
    name: 'loop trip cap stops a runaway while',
    source: '(p) => { let t = 0; while (t < p.x + 1000) { t += 1; } p.y = t; }',
    components: [Pos],
    make: (n, r) => [{ x: randomF32(n, r, 0, 1), y: randomF32(n, r) }],
    ref: (i, [p]) => {
      p.y[i] = 37; // maxLoopIterations below
    },
  },
  {
    name: 'Math builtins',
    source:
      '(p, v) => { const a = Math.sin(p.x) + Math.cos(v.x) * Math.tan(p.y * 0.1); const b = Math.sqrt(Math.abs(p.x)) + Math.atan2(p.y, v.y); ' +
      'const c = Math.min(p.x, v.x) - Math.max(p.y, v.y) + Math.floor(p.x) + Math.ceil(v.y) + Math.round(v.x) + Math.sign(p.y); ' +
      'const d = Math.pow(Math.abs(v.x) + 1, 1.5) + Math.exp(p.x * 0.1) + Math.log(Math.abs(p.y) + 1) + Math.hypot(p.x, v.y); ' +
      'const e = Math.asin(Math.min(1, Math.max(-1, p.x * 0.1))) + Math.acos(Math.min(1, Math.max(-1, v.x * 0.1))) + Math.atan(v.y); ' +
      'p.x = a + b; p.y = c + d + e; }',
    make: posVel,
    ref: (i, [p, v]) => {
      const x = p.x[i], y = p.y[i], vx = v.x[i], vy = v.y[i];
      const a = fr(fr(Math.sin(x)) + fr(fr(Math.cos(vx)) * fr(Math.tan(fr(y * fr(0.1))))));
      const b = fr(fr(Math.sqrt(Math.abs(x))) + fr(Math.atan2(y, vy)));
      const c = fr(fr(fr(fr(fr(Math.min(x, vx) - Math.max(y, vy)) + Math.floor(x)) + Math.ceil(vy)) + Math.floor(fr(vx + 0.5))) + Math.sign(y));
      const d = fr(
        fr(fr(fr(Math.pow(fr(Math.abs(vx) + 1), 1.5)) + fr(Math.exp(fr(x * fr(0.1))))) + fr(Math.log(fr(Math.abs(y) + 1)))) +
          fr(Math.hypot(x, vy)),
      );
      const e = fr(
        fr(fr(Math.asin(Math.min(1, Math.max(-1, fr(x * fr(0.1)))))) + fr(Math.acos(Math.min(1, Math.max(-1, fr(vx * fr(0.1))))))) +
          fr(Math.atan(vy)),
      );
      p.x[i] = fr(a + b);
      p.y[i] = fr(fr(c + d) + e);
    },
  },
  {
    name: 'rand, index and count',
    source: '(p, v, dt, u) => { p.x = rand(index + u.seed); p.y = rand(p.y * 1000) + count; v.x = index; }',
    uniforms: { seed: 7919 },
    make: posVel,
    ref: (i, [p, v], dt, u, count) => {
      p.x[i] = cozyRandRef(fr(i + fr(u.seed)));
      p.y[i] = fr(cozyRandRef(fr(p.y[i] * 1000)) + count);
      v.x[i] = i;
    },
  },
  {
    name: 'integer fields saturate and round half up',
    source: '(s, v) => { s.hp = v.x * v.y; s.gold = v.x * 1000 - 500; s.lvl = v.y * 40; s.wide = s.wide + v.x; }',
    components: [Stats, { ...Vel, index: 1 }],
    make: (n, r) => {
      const vx = randomF32(n, r, -3e9, 3e9);
      const vy = randomF32(n, r, -2, 2);
      // hand-picked edge values
      const edge = [NaN, 2.5, -2.5, -0.4, 1e30, -1e30, 0.49999997, 8388609];
      for (let q = 0; q < edge.length && q < n; q++) {
        vx[q] = edge[q];
        vy[q] = 1;
      }
      return [
        { hp: new Int32Array(n), gold: new Uint32Array(n), lvl: new Uint8Array(n), wide: new Float64Array(n).fill(0.25) },
        { x: vx, y: vy },
      ];
    },
    ref: (i, [s, v]) => {
      s.hp[i] = storeI32Ref(fr(v.x[i] * v.y[i]));
      s.gold[i] = storeU32Ref(fr(fr(v.x[i] * 1000) - 500));
      const l = fr(v.y[i] * 40);
      s.lvl[i] = l !== l ? 0 : Math.min(255, Math.max(0, Math.round(l)));
      s.wide[i] = fr(fr(s.wide[i]) + v.x[i]);
    },
  },
];

function compileCase(c: ParityCase): { ir: KernelIR; k: CompiledCPUKernel } {
  const names = Object.keys(c.uniforms || {});
  const ir = parse(c.source, {
    components: c.components || [Pos, Vel],
    uniformNames: names,
    uniformInitials: names.map((n) => (c.uniforms as Record<string, number>)[n]),
    maxLoopIterations: 37,
  });
  const k = compileCPUKernel(ir, tokens(ir.components.length));
  k.state.dt = c.dt === undefined ? 0.016 : c.dt;
  return { ir, k };
}

function expectSameColumns(got: Cols[], want: Cols[]): void {
  for (let c = 0; c < want.length; c++) {
    for (const key of Object.keys(want[c])) {
      expect({ col: `${c}.${key}`, v: Array.from(got[c][key]) }).toEqual({ col: `${c}.${key}`, v: Array.from(want[c][key]) });
    }
  }
}

describe('compiled loop parity with plain JS references (random data)', () => {
  for (const c of CASES) {
    test(c.name, () => {
      const { k } = compileCase(c);
      expect(k.compiled).toBe(true);
      const r = rng(0xc0ffee ^ c.name.length);
      for (const n of [1, 7, 64, 513]) {
        const cols = c.make(n, r);
        const want = cloneCols(cols);
        run(k, n, cols);
        for (let i = 0; i < n; i++) c.ref(i, want, k.state.dt, c.uniforms || {}, n);
        expectSameColumns(cols, want);
      }
    });
  }

  test('uniforms and dt are read from state on every call', () => {
    const ir = parse('(p, v, dt, u) => { p.x = u.a * dt; }', { uniformNames: ['a'], uniformInitials: [2] });
    const k = compileCPUKernel(ir, tokens(2));
    expect(Array.from(k.state.u)).toEqual([2]);
    const cols: Cols[] = [{ x: new Float32Array(2), y: new Float32Array(2) }, { x: new Float32Array(2), y: new Float32Array(2) }];
    k.state.dt = 0.5;
    run(k, 2, cols);
    expect(Array.from(cols[0].x)).toEqual([1, 1]);
    k.state.u[0] = 0.1;
    k.state.dt = 3;
    run(k, 2, cols);
    expect(cols[0].x[0]).toBe(fr(fr(0.1) * 3));
  });

  test('rows past the count parameter are not visited', () => {
    const ir = parse('(p) => { p.x = 5; }', { components: [Pos] });
    const k = compileCPUKernel(ir, tokens(1));
    const cols: Cols[] = [{ x: new Float32Array(4), y: new Float32Array(4) }];
    run(k, 2, cols);
    expect(Array.from(cols[0].x)).toEqual([5, 5, 0, 0]);
  });
});

// ---------------------------------------------------------------------------
// Enableable skip, through a real World and Query.forEachChunk
// ---------------------------------------------------------------------------

describe('enableable components', () => {
  test('disabled rows are skipped (compiled loop through forEachChunk trampolines)', () => {
    const P = component({ x: f32, y: f32 }, { name: 'CpuP' });
    const Live = component({ hp: i32 }, { name: 'CpuLive', enableable: true });
    const w = new World();
    const es: number[] = [];
    for (let i = 0; i < 40; i++) {
      const e = w.spawn([P, Live]);
      w.set(e, Live, { hp: i });
      es.push(e);
    }
    for (let i = 0; i < 40; i += 3) w.enable(es[i], Live, false);
    const ir = parse('(p, l, dt) => { p.x += 1; l.hp = l.hp + 2; }', {
      components: [comp(0, P.id, 'CpuP', [['x', 'f32'], ['y', 'f32']]), comp(1, Live.id, 'CpuLive', [['hp', 'i32']])],
    });
    const k = compileCPUKernel(ir, [P, Live], { enableIds: [Live.id] });
    expect(k.compiled).toBe(true);
    const q = w.query({ all: [P, Live] });
    const frames = 40; // generic loop, then a trampoline instance, then its specialized copy
    for (let f = 0; f < frames; f++) q.forEachChunk(k.components as never, k.fn as never);
    for (let i = 0; i < 40; i++) {
      const off = i % 3 === 0;
      expect({ i, x: w.getField(es[i], P, 'x'), hp: w.getField(es[i], Live, 'hp') }).toEqual({
        i,
        x: off ? 0 : frames,
        hp: off ? i : i + 2 * frames,
      });
    }
  });

  test('pairwise: a disabled row takes part in no pair', () => {
    const ir = parse('(s, o) => { s.vx += o.x; }', { form: 'pairwise', components: [Body, Mot] });
    const k = compileCPUKernel(ir, tokens(2), { enableIds: [10] });
    const cols: Cols[] = [
      { x: new Float32Array([1, 10, 100, 1000]), y: new Float32Array(4) },
      { vx: new Float32Array(4), vy: new Float32Array(4) },
    ];
    run(k, 4, cols, { enabled: Object.assign([], { 10: new Uint8Array([1, 0, 1, 1]) }) });
    expect(Array.from(cols[1].vx)).toEqual([1100, 0, 1001, 101]);
  });
});

// ---------------------------------------------------------------------------
// Pairwise
// ---------------------------------------------------------------------------

describe('pairwise kernels', () => {
  test('n-body style accumulation matches a nested-loop reference', () => {
    const src =
      '(s, o, dt, u) => { const dx = o.x - s.x; const dy = o.y - s.y; const d2 = dx * dx + dy * dy + u.eps; const inv = 1 / (d2 * Math.sqrt(d2)); s.vx += dx * inv * dt; s.vy += dy * inv * dt; }';
    const ir = parse(src, { form: 'pairwise', components: [Body, Mot], uniformNames: ['eps'], uniformInitials: [0.01] });
    const k = compileCPUKernel(ir, tokens(2));
    k.state.dt = 0.02;
    const r = rng(42);
    for (const n of [2, 5, 33]) {
      const cols: Cols[] = [
        { x: randomF32(n, r), y: randomF32(n, r) },
        { vx: randomF32(n, r, -1, 1), vy: randomF32(n, r, -1, 1) },
      ];
      const want = cloneCols(cols);
      run(k, n, cols);
      const DT = fr(0.02), EPS = fr(0.01);
      const [b, m] = want;
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          if (j === i) continue;
          const dx = fr(b.x[j] - b.x[i]);
          const dy = fr(b.y[j] - b.y[i]);
          const d2 = fr(fr(fr(dx * dx) + fr(dy * dy)) + EPS);
          const inv = fr(1 / fr(d2 * fr(Math.sqrt(d2))));
          m.vx[i] = fr(m.vx[i] + fr(fr(dx * inv) * DT));
          m.vy[i] = fr(m.vy[i] + fr(fr(dy * inv) * DT));
        }
      }
      expectSameColumns(cols, want);
    }
  });
});

// ---------------------------------------------------------------------------
// Interpreter fallback (no codegen)
// ---------------------------------------------------------------------------

describe('closure-tree interpreter (new Function unavailable)', () => {
  function withoutFunction<T>(body: () => T): T {
    const g = globalThis as unknown as { Function: FunctionConstructor };
    const real = g.Function;
    g.Function = function () {
      throw new EvalError('Refused to evaluate a string as JavaScript (CSP)');
    } as unknown as FunctionConstructor;
    try {
      return body();
    } finally {
      g.Function = real;
    }
  }

  test('falls back with compiled:false, same fn identity and state object across calls', () => {
    const ir = parse('(p, v) => { p.x += v.x; }');
    const k = withoutFunction(() => compileCPUKernel(ir, tokens(2)));
    expect(k.compiled).toBe(false);
    expect(typeof k.fn).toBe('function');
    expect(k.source).toBe(generateCPUSource(ir));
    const cols: Cols[] = [{ x: new Float32Array([1, 2]), y: new Float32Array(2) }, { x: new Float32Array([3, 4]), y: new Float32Array(2) }];
    const fn = k.fn;
    run(k, 2, cols);
    run(k, 2, cols);
    expect(k.fn).toBe(fn);
    expect(Array.from(cols[0].x)).toEqual([7, 10]);
  });

  for (const c of CASES) {
    test(`interpreter matches the compiled loop bit for bit: ${c.name}`, () => {
      const { ir, k } = compileCase(c);
      const slow = withoutFunction(() => compileCPUKernel(ir, tokens(ir.components.length)));
      expect(slow.compiled).toBe(false);
      slow.state.dt = k.state.dt;
      const r = rng(7 + c.name.length);
      const n = 97;
      const a = c.make(n, r);
      const b = cloneCols(a);
      run(k, n, a);
      run(slow, n, b);
      expectSameColumns(b, a);
    });
  }

  test('interpreter honours enable flags and pairwise form', () => {
    const ir = parse('(s, o) => { s.vx += o.x; }', { form: 'pairwise', components: [Body, Mot] });
    const k = withoutFunction(() => compileCPUKernel(ir, tokens(2), { enableIds: [10] }));
    expect(k.compiled).toBe(false);
    const cols: Cols[] = [
      { x: new Float32Array([1, 10, 100, 1000]), y: new Float32Array(4) },
      { vx: new Float32Array(4), vy: new Float32Array(4) },
    ];
    run(k, 4, cols, { enabled: Object.assign([], { 10: new Uint8Array([1, 0, 1, 1]) }) });
    expect(Array.from(cols[1].vx)).toEqual([1100, 0, 1001, 101]);
  });
});

// ---------------------------------------------------------------------------
// Cost estimate
// ---------------------------------------------------------------------------

describe('estimateCPUNanosPerEntity', () => {
  test('a fixed per-entity cost plus a per-op cost, counted from the IR body', () => {
    expect(CPU_NS_PER_ENTITY_FIXED).toBe(0.52);
    expect(CPU_NS_PER_OP).toBe(0.065);
    const ir = parse('(p, v, dt, u) => { v.y += u.g * dt; p.x += v.x * dt; p.y += v.y * dt; }', { uniformNames: ['g'] });
    expect(estimateCPUNanosPerEntity(ir)).toBeCloseTo(0.52 + ir.opCount * 0.065, 10);
    const tiny = parse('(p) => { p.x = 1; }', { components: [Pos] });
    expect(tiny.opCount).toBe(2); // one assignment + one field access
    expect(estimateCPUNanosPerEntity(tiny)).toBeCloseTo(0.52 + 0.13, 10);
    const looped = parse('(p) => { for (let k = 0; k < 10; k++) { p.x += 1; } }', { components: [Pos] });
    expect(estimateCPUNanosPerEntity(looped)).toBeGreaterThan(0.52 + 10 * 0.065);
  });

  test('matches the measured cost ratio of the benchmark kernels (~2x, not 3.3x)', () => {
    const simple = parse('(p, v) => { p.x += v.x; p.y += v.y; }');
    const gravity = parse(
      '(p, v, dt, u) => { v.y += u.g * dt; p.x += v.x * dt; p.y += v.y * dt; if (p.y < 0) { p.y = 0; v.y = -v.y * 0.5; } }',
      { uniformNames: ['g'] },
    );
    const ratio = estimateCPUNanosPerEntity(gravity) / estimateCPUNanosPerEntity(simple);
    expect(ratio).toBeGreaterThan(1.5);
    expect(ratio).toBeLessThan(2.5);
  });
});
