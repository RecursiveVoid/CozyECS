import { describe, test, expect } from '@jest/globals';
import {
  assign,
  binary,
  block,
  builtinValue,
  call,
  cond,
  decl,
  field,
  forStmt,
  ifStmt,
  local,
  logical,
  makeKernelIR,
  BUILTINS,
  num,
  storageViews,
  unary,
  uniform,
  uniformBinding,
  uniformLayout,
  whileStmt,
  BREAK,
  CONTINUE,
  cozy_rand,
} from '../src/gpu/ir';
import type { BuiltinName, Expr, IRComponent, IRLocal, KernelIR } from '../src/gpu/ir';
import {
  COZY_RAND_WGSL,
  COZY_STORE_WGSL,
  checkGPUSupport,
  cozyRandRef,
  generateWGSL,
  storeI32Ref,
  storeU32Ref,
  wgslCacheKey,
} from '../src/gpu/wgsl';
import type { WGSLModule, WGSLOptions } from '../src/gpu/wgsl';
import type { GPUCapabilities } from '../src/gpu/device';

const Position: IRComponent = {
  index: 0,
  id: 0,
  name: 'Position',
  fields: [
    { name: 'x', kind: 'f32', index: 0 },
    { name: 'y', kind: 'f32', index: 1 },
  ],
};
const Velocity: IRComponent = {
  index: 1,
  id: 1,
  name: 'Velocity',
  fields: [
    { name: 'x', kind: 'f32', index: 0 },
    { name: 'y', kind: 'f32', index: 1 },
  ],
};
const Health: IRComponent = {
  index: 1,
  id: 2,
  name: 'Health',
  fields: [
    { name: 'hp', kind: 'i32', index: 0 },
    { name: 'team', kind: 'u32', index: 1 },
  ],
};
const Wide: IRComponent = {
  index: 1,
  id: 3,
  name: 'Wide',
  fields: [
    { name: 'big', kind: 'f64', index: 0 },
    { name: 'small', kind: 'i16', index: 1 },
  ],
};

/** The reference kernel from docs/GPU.md. */
function moveIR(): KernelIR {
  const px = field(0, 'x', 'f32');
  const py = field(0, 'y', 'f32');
  const vx = field(1, 'x', 'f32');
  const vy = field(1, 'y', 'f32');
  const dt = builtinValue('dt');
  return makeKernelIR({
    name: 'Move',
    form: 'per-entity',
    components: [Position, Velocity],
    uniforms: [{ name: 'gravity', index: 0, initial: -9.8 }],
    locals: [],
    body: block([
      assign(vy, '+=', binary('*', uniform('gravity'), dt)),
      assign(px, '+=', binary('*', vx, dt)),
      assign(py, '+=', binary('*', vy, dt)),
      ifStmt(binary('<', py, num(0)), block([assign(py, '=', num(0)), assign(vy, '=', binary('*', unary('-', vy), num(0.5)))]), null),
    ]),
    source: '(p, v, dt, u) => { v.y += u.gravity * dt; p.x += v.x * dt; p.y += v.y * dt; if (p.y < 0) { p.y = 0; v.y = -v.y * 0.5; } }',
  });
}

/** f32 + i32 + u32 in one kernel: the case that must collapse to one binding. */
function mixedIR(): KernelIR {
  return makeKernelIR({
    name: 'Damage',
    form: 'per-entity',
    components: [Position, Health],
    uniforms: [{ name: 'dmg', index: 0, initial: 0.6 }],
    locals: [],
    body: block([
      assign(field(1, 'hp', 'i32'), '-=', uniform('dmg')),
      assign(field(1, 'team', 'u32'), '=', binary('+', field(1, 'team', 'u32'), num(1))),
      assign(field(0, 'x', 'f32'), '=', binary('*', field(1, 'hp', 'i32'), num(2))),
    ]),
    source: '(p, h, dt, u) => { h.hp -= u.dmg; h.team = h.team + 1; p.x = h.hp * 2; }',
  });
}

function loopsIR(): KernelIR {
  const locals: IRLocal[] = [
    { id: 0, name: 's', type: 'f32', mutable: true },
    { id: 1, name: 'i', type: 'f32', mutable: true },
    { id: 2, name: 'k', type: 'f32', mutable: true },
  ];
  const s = local(0, 'f32');
  const i = local(1, 'f32');
  const k = local(2, 'f32');
  return makeKernelIR({
    name: 'Loops',
    form: 'per-entity',
    components: [Position],
    uniforms: [],
    locals,
    body: block([
      decl(0, num(0)),
      forStmt(
        decl(1, num(0)),
        binary('<', i, num(8)),
        assign(i, '+=', num(1)),
        block([
          ifStmt(binary('==', i, num(3)), block([CONTINUE]), null),
          ifStmt(binary('>', i, num(5)), block([BREAK]), null),
          assign(s, '+=', binary('*', i, num(2))),
        ]),
        8,
      ),
      decl(2, num(0)),
      whileStmt(binary('<', k, num(1000)), block([assign(k, '+=', num(1))]), 7),
      assign(field(0, 'x', 'f32'), '=', s),
      assign(field(0, 'y', 'f32'), '=', k),
    ]),
    source: '(p) => { /* loops */ }',
  });
}

const PPos: IRComponent = { index: 0, id: 20, name: 'PPos', fields: [{ name: 'px', kind: 'f32', index: 0 }] };
const PVel: IRComponent = { index: 1, id: 21, name: 'PVel', fields: [{ name: 'vx', kind: 'f32', index: 0 }] };

function pairwiseIR(writeOther = false): KernelIR {
  return makeKernelIR({
    name: writeOther ? 'BadPair' : 'Pair',
    form: 'pairwise',
    components: [PPos, PVel],
    uniforms: [],
    locals: [],
    body: block([
      assign(
        field(1, 'vx', 'f32', writeOther ? 1 : 0),
        '+=',
        binary('-', field(0, 'px', 'f32', 1), field(0, 'px', 'f32', 0)),
      ),
    ]),
    source: '(self, other) => { self.vx += other.px - self.px; }',
  });
}

const CAPS: GPUCapabilities = {
  maxStorageBuffersPerShaderStage: 10,
  maxComputeWorkgroupSizeX: 1024,
  maxComputeInvocationsPerWorkgroup: 1024,
  maxBufferSize: 4e9,
  maxStorageBufferBindingSize: 4e9,
  minStorageBufferOffsetAlignment: 256,
  minUniformBufferOffsetAlignment: 256,
  maxComputeWorkgroupsPerDimension: 65535,
  subgroups: true,
  timestampQuery: true,
  shaderF16: true,
};

describe('generateWGSL: module shape', () => {
  const mod = generateWGSL(moveIR());

  test('entry point, workgroup size and bindings', () => {
    expect(mod.entryPoint).toBe('main');
    expect(mod.workgroupSize).toBe(256);
    expect(mod.views).toEqual(['f32']);
    expect(mod.uniformBinding).toBe(1);
    expect(mod.code).toContain('@group(0) @binding(0) var<storage, read_write> t_f32: array<f32>;');
    expect(mod.code).toContain('@group(0) @binding(1) var<uniform> u: U;');
    expect(mod.code).toContain('@compute @workgroup_size(256)');
  });

  test('an all-f32 kernel still agrees with the ./ir binding helpers', () => {
    expect(mod.views).toEqual(storageViews(moveIR()));
    expect(mod.uniformBinding).toBe(uniformBinding(moveIR()));
    expect(mod.layout).toEqual(uniformLayout(moveIR()));
  });

  test('the bounds check comes before any storage access', () => {
    expect(mod.code).toContain('let row: u32 = gid.x + u.base;');
    expect(mod.code).toContain('if (row >= u.count) { return; }');
    expect(mod.code.indexOf('if (row >= u.count)')).toBeLessThan(mod.code.indexOf('t_f32['));
  });

  test('uniform members are named by ./ir and addressed as base + row', () => {
    expect(mod.code).toContain('o0_Position_x: u32');
    expect(mod.code).toContain('u_gravity: f32');
    expect(mod.code).toContain('t_f32[u.o0_Position_x + row]');
  });

  test('a compound field assignment expands to read-op-write', () => {
    expect(mod.code).toContain(
      't_f32[u.o0_Position_x + row] = (t_f32[u.o0_Position_x + row] + (t_f32[u.o0_Velocity_x + row] * u.dt));',
    );
  });

  test('no rand helper unless the kernel uses it', () => {
    expect(mod.code).not.toContain('cozy_rand');
  });
});

describe('generateWGSL: determinism and the pipeline cache key', () => {
  test('same IR, byte-identical code', () => {
    expect(generateWGSL(moveIR()).code).toBe(generateWGSL(moveIR()).code);
  });

  test('wgslCacheKey agrees with the generated module', () => {
    expect(wgslCacheKey(moveIR())).toBe(generateWGSL(moveIR()).cacheKey);
  });

  test('the kernel name is in neither the code nor the key, so bodies share a pipeline', () => {
    const renamed: KernelIR = { ...moveIR(), name: 'Move2', source: '/* different */' };
    expect(generateWGSL(renamed).code).toBe(generateWGSL(moveIR()).code);
    expect(wgslCacheKey(renamed)).toBe(wgslCacheKey(moveIR()));
    expect(generateWGSL(moveIR()).code).not.toContain('Move');
  });

  test('a changed literal changes both the code and the key', () => {
    const other = makeKernelIR({
      name: 'Move',
      form: 'per-entity',
      components: [Position, Velocity],
      uniforms: [{ name: 'gravity', index: 0, initial: -9.8 }],
      locals: [],
      body: block([assign(field(0, 'y', 'f32'), '+=', num(0.25))]),
      source: '',
    });
    expect(wgslCacheKey(other)).not.toBe(wgslCacheKey(moveIR()));
    expect(generateWGSL(other).code).not.toBe(generateWGSL(moveIR()).code);
  });

  test('workgroupSize and annotate are part of the key', () => {
    expect(wgslCacheKey(moveIR(), { workgroupSize: 64 })).not.toBe(wgslCacheKey(moveIR()));
    expect(wgslCacheKey(moveIR(), { annotate: true })).not.toBe(wgslCacheKey(moveIR()));
    expect(generateWGSL(moveIR(), { workgroupSize: 64 }).code).toContain('@workgroup_size(64)');
  });
});

describe('generateWGSL: field kinds', () => {
  const mod = generateWGSL(mixedIR());

  test('mixing view types collapses to ONE u32 binding', () => {
    // WebGPU rejects two writable storage bindings over overlapping ranges, and
    // two views of one archetype table overlap completely. Verified against
    // Dawn/Metal: "Writable storage buffer binding aliasing found between ...".
    expect(mod.views).toEqual(['u32']);
    expect(mod.uniformBinding).toBe(1);
    expect(mod.code.match(/@binding\(\d+\) var<storage/g)).toHaveLength(1);
    // ... which is exactly where this deviates from the ./ir helpers.
    expect(storageViews(mixedIR())).toEqual(['f32', 'i32', 'u32']);
  });

  test('integer reads convert to f32 through a bitcast', () => {
    expect(mod.code).toContain('f32(bitcast<i32>(t_u32[u.o0_Health_hp + row]))');
  });

  test('integer writes go through the saturating parity helpers', () => {
    expect(mod.code).toContain(COZY_STORE_WGSL);
    expect(mod.code).toContain('t_u32[u.o0_Health_hp + row] = bitcast<u32>(cozy_store_i32(');
    expect(mod.code).toContain('t_u32[u.o0_Health_team + row] = cozy_store_u32(');
    // An f32 field never goes through a store helper.
    expect(mod.code).toContain('t_u32[u.o0_Position_x + row] = bitcast<u32>((f32(bitcast<i32>(');
  });

  test('a kernel writing no integer field carries no store helpers', () => {
    expect(generateWGSL(moveIR()).code).not.toContain('cozy_store_');
  });
});

describe('generateWGSL: control flow', () => {
  const mod = generateWGSL(loopsIR());

  test('every loop carries its static trip cap, so a kernel cannot hang the device', () => {
    expect(mod.code).toContain('if (cozy_trip0 >= 8u) { break; }');
    expect(mod.code).toContain('if (cozy_trip1 >= 7u) { break; }');
  });

  test('a for update lands in a continuing block, so continue still runs it', () => {
    expect(mod.code).toContain('continuing {');
    expect(mod.code).toContain('continue;');
    expect(mod.code).toContain('break;');
  });

  test('let vs var follows const vs let in the kernel', () => {
    expect(mod.code).toContain('var l0_s: f32 = 0.0;');
  });
});

function builtinsIR(): KernelIR {
  return makeKernelIR({
    name: 'B',
    form: 'per-entity',
    components: [Position, Velocity],
    uniforms: [],
    locals: [],
    body: block([
      assign(
        field(0, 'x', 'f32'),
        '=',
        binary(
          '+',
          binary('+', call('round', [field(1, 'x', 'f32')]), call('hypot', [field(1, 'x', 'f32'), field(1, 'y', 'f32')])),
          call('rand', [builtinValue('index')]),
        ),
      ),
      assign(field(0, 'y', 'f32'), '=', cond(binary('>', field(1, 'y', 'f32'), num(0)), builtinValue('count'), num(-1))),
    ]),
    source: '',
  });
}

describe('generateWGSL: builtins', () => {
  const mod = generateWGSL(builtinsIR());

  test('rand is the exact-parity helper, verbatim', () => {
    expect(mod.code).toContain(COZY_RAND_WGSL);
    expect(mod.code).toContain('cozy_rand(f32(row))');
  });

  test('Math.round is floor(x + 0.5), never WGSL round (half-to-even)', () => {
    expect(mod.code).toContain('floor(t_f32[u.o0_Velocity_x + row] + 0.5)');
    expect(mod.code).not.toMatch(/\bround\(/);
  });

  test('hypot and the ternary use their WGSL equivalents', () => {
    expect(mod.code).toContain('length(vec2<f32>(');
    expect(mod.code).toContain('select(');
  });

  test('index and count widen to f32', () => {
    expect(mod.code).toContain('f32(row)');
    expect(mod.code).toContain('f32(u.count)');
  });
});

describe('generateWGSL: pairwise', () => {
  const mod = generateWGSL(pairwiseIR());

  test('the body runs inside a bounded loop over the other rows, skipping self', () => {
    expect(mod.code).toContain('for (var j: u32 = 0u; j < u.countOther; j = j + 1u) {');
    expect(mod.code).toContain('if (j == row) { continue; }');
  });

  test('role 1 is indexed by j and gets its own offset member', () => {
    expect(mod.code).toContain('t_f32[u.o1_PPos_px + j]');
    expect(mod.code).toContain('t_f32[u.o0_PPos_px + row]');
  });

  test('a write to other is rejected rather than raced', () => {
    expect(() => generateWGSL(pairwiseIR(true))).toThrow(/E_INVALID_IR/);
    expect(() => generateWGSL(pairwiseIR(true))).toThrow(/other\.vx/);
  });
});

describe('checkGPUSupport', () => {
  test('accepts the kernels the GPU can run', () => {
    expect(checkGPUSupport(moveIR(), CAPS).ok).toBe(true);
    expect(checkGPUSupport(mixedIR(), CAPS).ok).toBe(true);
    expect(checkGPUSupport(loopsIR(), CAPS).ok).toBe(true);
    expect(checkGPUSupport(pairwiseIR(), CAPS).ok).toBe(true);
  });

  test('reports one reason per field with no 4-byte view', () => {
    const wide = makeKernelIR({
      name: 'W',
      form: 'per-entity',
      components: [Position, Wide],
      uniforms: [],
      locals: [],
      body: block([assign(field(0, 'x', 'f32'), '=', binary('+', field(1, 'big', 'f64'), field(1, 'small', 'i16')))]),
      source: '',
    });
    const r = checkGPUSupport(wide, CAPS);
    expect(r.ok).toBe(false);
    expect(r.reasons).toHaveLength(2);
    expect(r.reasons.every((x) => x.code === 'E_DEVICE_LIMIT')).toBe(true);
    expect(r.reasons[0].message).toContain('Wide.big');
  });

  test('reports a workgroup size over the device limit', () => {
    expect(checkGPUSupport(moveIR(), { ...CAPS, maxComputeWorkgroupSizeX: 128 }).ok).toBe(false);
    expect(checkGPUSupport(moveIR(), { ...CAPS, maxComputeInvocationsPerWorkgroup: 64 }).ok).toBe(false);
    expect(checkGPUSupport(moveIR(), CAPS, { workgroupSize: 64 }).ok).toBe(true);
    expect(checkGPUSupport(moveIR(), CAPS, { workgroupSize: 0 }).ok).toBe(false);
  });

  test('one storage binding plus the uniform always fits', () => {
    expect(checkGPUSupport(mixedIR(), { ...CAPS, maxStorageBuffersPerShaderStage: 2 }).ok).toBe(true);
    expect(checkGPUSupport(mixedIR(), { ...CAPS, maxStorageBuffersPerShaderStage: 1 }).ok).toBe(false);
  });

  test('a write to other is a reason, not a throw', () => {
    const r = checkGPUSupport(pairwiseIR(true), CAPS);
    expect(r.ok).toBe(false);
    expect(r.reasons[0].code).toBe('E_INVALID_IR');
  });

  test('never throws, even for a malformed IR', () => {
    const junk = { version: 1, name: 'junk', form: 'per-entity' } as unknown as KernelIR;
    expect(() => checkGPUSupport(junk, CAPS)).not.toThrow();
    expect(checkGPUSupport(junk, CAPS).ok).toBe(false);
  });
});

describe('generateWGSL: validation', () => {
  test('a bad IR fails before any code is emitted', () => {
    const bad = { ...moveIR(), opCount: 999 };
    expect(() => generateWGSL(bad)).toThrow(/E_INVALID_IR/);
  });

  test('annotate carries the kernel source into the shader', () => {
    const code = generateWGSL(moveIR(), { annotate: true }).code;
    expect(code).toContain('// kernel: Move');
    expect(code).toContain('v.y += u.gravity * dt');
  });
});

/** Every builtin, every binary/logical operator and both unary ops in one kernel. */
function everythingIR(): KernelIR {
  const x = field(0, 'x', 'f32');
  const y = field(0, 'y', 'f32');
  let acc: Expr = num(0);
  for (const name of Object.keys(BUILTINS) as BuiltinName[]) {
    const n = BUILTINS[name].arity[0];
    acc = binary('+', acc, call(name, n === 1 ? [x] : [x, y]));
  }
  for (const op of ['-', '*', '/', '%'] as const) acc = binary(op, acc, y);
  const t = logical(
    '||',
    logical('&&', binary('<', x, y), binary('>=', x, num(1))),
    unary('!', logical('||', binary('==', x, y), binary('!=', binary('<=', x, num(2)), binary('>', y, num(3))))),
  );
  return makeKernelIR({
    name: 'Everything',
    form: 'per-entity',
    components: [Position],
    uniforms: [{ name: 'a', index: 0, initial: 1 }, { name: 'b', index: 1, initial: 2 }],
    locals: [{ id: 0, name: 'acc', type: 'f32', mutable: true }],
    body: block([
      decl(0, binary('*', acc, uniform('a'))),
      ifStmt(t, block([assign(local(0, 'f32'), '*=', uniform('b'))]), block([assign(local(0, 'f32'), '-=', num(1e-7))])),
      assign(x, '=', unary('-', local(0, 'f32'))),
      assign(y, '/=', builtinValue('dt')),
    ]),
    source: '',
  });
}

describe('parity reference functions (the CPU backend\'s spec)', () => {
  test('cozyRandRef equals ./ir cozy_rand rounded to f32, for every in-range seed', () => {
    // ./ir's JS twin scales by the f64 literal 5.9604645e-8, which is 2^-24 only
    // once rounded to f32 -- so it matches after Math.fround, not before.
    const seeds = [0, 1, -1, 2, 7919, -7919, 123456, 16777216, -16777216, 2147483520, -2147483648, 0.5, -0.99, 3.75];
    for (let i = 0; i < 2000; i++) seeds.push(i * 7919 - 5_000_000);
    for (const s of seeds) expect(cozyRandRef(s)).toBe(Math.fround(cozy_rand(Math.fround(s))));
  });

  test('cozyRandRef is an exact multiple of 2^-24 in [0, 1)', () => {
    for (let s = -5000; s < 5000; s++) {
      const r = cozyRandRef(s);
      expect(r).toBeGreaterThanOrEqual(0);
      expect(r).toBeLessThan(1);
      expect(Number.isInteger(r * 16777216)).toBe(true);
      expect(Math.fround(r)).toBe(r);
    }
  });

  test('cozyRandRef: NaN, Infinity and |x| < 1 all hash seed 0; big seeds wrap like ToInt32', () => {
    const r0 = cozyRandRef(0);
    for (const s of [NaN, Infinity, -Infinity, 0.99, -0.99, -0]) expect(cozyRandRef(s)).toBe(r0);
    expect(cozyRandRef(2 ** 32 + 5 * 512)).toBe(cozyRandRef(5 * 512)); // f32 ulp at 2^32 is 512
    expect(cozyRandRef(2 ** 31)).toBe(cozyRandRef(-(2 ** 31)));
  });

  test('integer stores: NaN -> 0, saturate to the true range, round half toward +Infinity', () => {
    const i32: [number, number][] = [
      [NaN, 0], [Infinity, 2147483647], [-Infinity, -2147483648], [2 ** 31, 2147483647], [2 ** 40, 2147483647],
      [2147483520, 2147483520], [-(2 ** 31), -2147483648], [-(2 ** 40), -2147483648],
      [2.5, 3], [-2.5, -2], [-0.4, 0], [0.49999997, 0], [8388609, 8388609], [1.5, 2], [-1.5, -1], [-0.6, -1],
    ];
    for (const [x, want] of i32) expect(storeI32Ref(x)).toBe(want);
    const u32: [number, number][] = [
      [NaN, 0], [Infinity, 4294967295], [-Infinity, 0], [2 ** 32, 4294967295], [4294967040, 4294967040],
      [-5, 0], [-0.4, 0], [0.5, 1], [2.5, 3], [0.49999997, 0], [16777217, 16777216],
    ];
    for (const [x, want] of u32) expect(storeU32Ref(x)).toBe(want);
  });
});

// ---------------------------------------------------------------------------
// Dawn: compile every generated shader, and run the parity helpers on the GPU
// ---------------------------------------------------------------------------

// Jest's CJS runtime cannot require() the ESM-only `webgpu` package, and
// ts-jest rewrites a literal `import()` into a require. `new Function` hides it
// from both so it reaches Node's real ESM loader (needs --experimental-vm-modules).
const esmImport = new Function('s', 'return import(s)') as (s: string) => Promise<any>;

interface Dawn {
  device: GPUDevice;
  usage: { STORAGE: number; COPY_SRC: number; COPY_DST: number; MAP_READ: number };
  mapRead: number;
}

let dawnPromise: Promise<Dawn | null> | null = null;
function getDawn(): Promise<Dawn | null> {
  if (!dawnPromise) {
    dawnPromise = (async () => {
      try {
        const mod = await esmImport('webgpu');
        const gpu = mod.create([]);
        // Pinned for the life of the process: Dawn's event pump segfaults if the
        // GPU instance is collected with work outstanding (see src/gpu/device.ts).
        (globalThis as any).__cozyWgslDawn = [gpu];
        const adapter = await gpu.requestAdapter();
        if (!adapter) return null;
        const device: GPUDevice = await adapter.requestDevice();
        (globalThis as any).__cozyWgslDawn.push(adapter, device);
        const g = mod.globals;
        return { device, usage: g.GPUBufferUsage, mapRead: g.GPUMapMode.READ };
      } catch {
        return null;
      }
    })();
  }
  return dawnPromise;
}

/** Every module the suites above generate, plus the option variants. */
function allModules(): { name: string; mod: WGSLModule }[] {
  const irs: [string, () => KernelIR][] = [
    ['move', moveIR],
    ['mixed', mixedIR],
    ['loops', loopsIR],
    ['builtins', builtinsIR],
    ['pairwise', () => pairwiseIR()],
    ['everything', everythingIR],
  ];
  const opts: [string, WGSLOptions | undefined][] = [
    ['default', undefined],
    ['wg64', { workgroupSize: 64 }],
    ['annotate', { annotate: true }],
  ];
  const out: { name: string; mod: WGSLModule }[] = [];
  for (const [n, f] of irs) for (const [o, opt] of opts) out.push({ name: `${n}/${o}`, mod: generateWGSL(f(), opt) });
  return out;
}

async function compileErrors(device: GPUDevice, code: string): Promise<string[]> {
  device.pushErrorScope('validation');
  const sm = device.createShaderModule({ code });
  const info = await sm.getCompilationInfo();
  const scoped = await device.popErrorScope();
  const msgs = info.messages
    .filter((m) => m.type === 'error')
    .map((m) => `${m.lineNum}:${m.linePos} ${m.message}`);
  if (scoped) msgs.push(`validation: ${scoped.message}`);
  return msgs;
}

/** Runs the parity helpers on `inputs` (raw f32 bit patterns) on the GPU. */
async function runHelpers(d: Dawn, bits: Uint32Array<ArrayBuffer>): Promise<{ rand: Float32Array; i32: Int32Array; u32: Uint32Array }> {
  const { device, usage } = d;
  const code = `${COZY_RAND_WGSL}
${COZY_STORE_WGSL}
@group(0) @binding(0) var<storage, read> inp: array<u32>;
@group(0) @binding(1) var<storage, read_write> outp: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&inp)) { return; }
  let x = bitcast<f32>(inp[i]);
  outp[3u * i] = bitcast<u32>(cozy_rand(x));
  outp[3u * i + 1u] = bitcast<u32>(cozy_store_i32(x));
  outp[3u * i + 2u] = cozy_store_u32(x);
}`;
  expect(await compileErrors(device, code)).toEqual([]);
  const n = bits.length;
  const inBuf = device.createBuffer({ size: n * 4, usage: usage.STORAGE | usage.COPY_DST });
  device.queue.writeBuffer(inBuf, 0, bits);
  const outBuf = device.createBuffer({ size: n * 12, usage: usage.STORAGE | usage.COPY_SRC });
  const readBuf = device.createBuffer({ size: n * 12, usage: usage.MAP_READ | usage.COPY_DST });
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: device.createShaderModule({ code }), entryPoint: 'main' },
  });
  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: inBuf } },
      { binding: 1, resource: { buffer: outBuf } },
    ],
  });
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(Math.ceil(n / 64));
  pass.end();
  enc.copyBufferToBuffer(outBuf, 0, readBuf, 0, n * 12);
  device.queue.submit([enc.finish()]);
  await readBuf.mapAsync(d.mapRead);
  const raw = new Uint32Array(readBuf.getMappedRange().slice(0));
  readBuf.unmap();
  inBuf.destroy();
  outBuf.destroy();
  readBuf.destroy();
  const rand = new Float32Array(n);
  const i32 = new Int32Array(n);
  const u32 = new Uint32Array(n);
  const rv = new Uint32Array(rand.buffer);
  for (let i = 0; i < n; i++) {
    rv[i] = raw[3 * i];
    i32[i] = raw[3 * i + 1] | 0;
    u32[i] = raw[3 * i + 2];
  }
  return { rand, i32, u32 };
}

/** Edge inputs plus a deterministic sweep of raw f32 bit patterns. */
function parityInputs(): Uint32Array<ArrayBuffer> {
  const f = new Float32Array(1);
  const u = new Uint32Array(f.buffer);
  const out: number[] = [];
  const addF = (x: number): void => {
    f[0] = x;
    out.push(u[0]);
  };
  for (const x of [
    0, -0, 0.25, 0.5, -0.5, 0.49999997, -0.49999997, 0.99999994, 1, -1, 1.5, -1.5, 2.5, -2.5, 8388609, -8388609,
    8388608.5, 16777217, 2147483520, 2 ** 31, -(2 ** 31), -(2 ** 31) - 256, 4294967040, 2 ** 32, 2 ** 32 + 512,
    2 ** 40 + 3 * 2 ** 17, -(2 ** 40), 2 ** 63, 3.4028235e38, -3.4028235e38, Infinity, -Infinity, NaN, 1e-45, -1e-45,
    1.17549435e-38, 7919, -7919, 123456.789,
  ]) addF(x);
  out.push(0x7fc00001, 0xffc00000, 0x7f800001, 0xff812345); // NaN payloads, both signs
  for (let i = -3000; i <= 3000; i++) addF(i * 0.25);
  for (let i = 0; i < 4096; i++) addF(i * 7919 - 16_000_000);
  // xorshift32 over raw bit patterns: every exponent, sign and mantissa shape.
  let s = 0x12345678;
  for (let i = 0; i < 40000; i++) {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    out.push(s >>> 0);
  }
  return new Uint32Array(out);
}

describe('Dawn (skipped cleanly without a WebGPU adapter)', () => {
  test('every generated shader compiles with no errors and builds a pipeline', async () => {
    const d = await getDawn();
    if (!d) {
      console.warn('gpu-wgsl: no WebGPU adapter; shader validation skipped');
      return;
    }
    const mods = allModules();
    expect(mods.length).toBe(18);
    const failures: string[] = [];
    for (const { name, mod } of mods) {
      const errs = await compileErrors(d.device, mod.code);
      if (errs.length) {
        failures.push(`${name}:\n  ${errs.join('\n  ')}\n${mod.code}`);
        continue;
      }
      // The binding contract runtime.ts relies on: views then the uniform block.
      d.device.pushErrorScope('validation');
      const p = await d.device.createComputePipelineAsync({
        layout: 'auto',
        compute: { module: d.device.createShaderModule({ code: mod.code }), entryPoint: mod.entryPoint },
      });
      p.getBindGroupLayout(0);
      const e = await d.device.popErrorScope();
      if (e) failures.push(`${name}: pipeline: ${e.message}`);
    }
    expect(failures).toEqual([]);
  });

  test('cozy_rand and the integer stores match their JS twins bit for bit on the GPU', async () => {
    const d = await getDawn();
    if (!d) {
      console.warn('gpu-wgsl: no WebGPU adapter; GPU parity check skipped');
      return;
    }
    const bits = parityInputs();
    const gpu = await runHelpers(d, bits);
    const xs = new Float32Array(bits.buffer.slice(0));
    const bad: string[] = [];
    const rv = new Uint32Array(gpu.rand.buffer);
    const rb = new Uint32Array(1);
    const rf = new Float32Array(rb.buffer);
    for (let i = 0; i < bits.length && bad.length < 20; i++) {
      const x = xs[i];
      rf[0] = cozyRandRef(x);
      if (rv[i] !== rb[0]) bad.push(`rand(0x${bits[i].toString(16)} = ${x}): gpu ${gpu.rand[i]} js ${rf[0]}`);
      if (gpu.i32[i] !== storeI32Ref(x)) bad.push(`i32(0x${bits[i].toString(16)} = ${x}): gpu ${gpu.i32[i]} js ${storeI32Ref(x)}`);
      if (gpu.u32[i] !== storeU32Ref(x)) bad.push(`u32(0x${bits[i].toString(16)} = ${x}): gpu ${gpu.u32[i]} js ${storeU32Ref(x)}`);
    }
    expect(bad).toEqual([]);
    expect(bits.length).toBeGreaterThan(50000);
  });
});
