import { describe, test, expect } from '@jest/globals';
import {
  BUILTINS,
  IR_VERSION,
  KernelError,
  WORKGROUP_SIZE,
  accessKey,
  assign,
  binary,
  block,
  boolLit,
  builtinValue,
  call,
  codeFrame,
  cond,
  cozy_rand,
  decl,
  deriveFacts,
  describeIR,
  field,
  forStmt,
  gpuBlockers,
  ifStmt,
  isGPUEligible,
  local,
  makeKernelIR,
  maxRowsPerDispatch,
  num,
  offsetMemberName,
  storageViews,
  touchedFields,
  uniform,
  uniformBinding,
  uniformLayout,
  unary,
  validateIR,
  writtenComponents,
} from '../src/gpu/ir';
import type { IRComponent, IRLocal, KernelIR } from '../src/gpu/ir';

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
  index: 2,
  id: 2,
  name: 'Health',
  fields: [
    { name: 'hp', kind: 'i16', index: 0 },
    { name: 'team', kind: 'u32', index: 1 },
  ],
};

/**
 * The reference kernel from docs/GPU.md:
 *   v.y += u.gravity * dt;
 *   p.x += v.x * dt; p.y += v.y * dt;
 *   if (p.y < 0) { p.y = 0; v.y = -v.y * 0.5; }
 */
function moveIR(): KernelIR {
  const py = field(0, 'y', 'f32');
  const vy = field(1, 'y', 'f32');
  return makeKernelIR({
    name: 'Move',
    form: 'per-entity',
    components: [Position, Velocity],
    uniforms: [{ name: 'gravity', index: 0, initial: -9.8 }],
    locals: [],
    body: block([
      assign(vy, '+=', binary('*', uniform('gravity'), builtinValue('dt'))),
      assign(field(0, 'x', 'f32'), '+=', binary('*', field(1, 'x', 'f32'), builtinValue('dt'))),
      assign(py, '+=', binary('*', vy, builtinValue('dt'))),
      ifStmt(
        binary('<', py, num(0)),
        block([assign(py, '=', num(0)), assign(vy, '=', binary('*', unary('-', vy), num(0.5)))]),
        null,
      ),
    ]),
    source: 'kernel source placeholder',
  });
}

describe('gpu IR', () => {
  test('derives reads, writes and op count from the body', () => {
    const ir = moveIR();
    expect(ir.version).toBe(IR_VERSION);
    expect(ir.usesDt).toBe(true);
    expect(ir.usesRand).toBe(false);
    // Every field is both read and written: all three assignments are compound
    // or read the field on the right-hand side.
    expect(ir.writes.map((w) => `${ir.components[w.component].name}.${w.field}`).sort()).toEqual([
      'Position.x',
      'Position.y',
      'Velocity.y',
    ]);
    expect(ir.reads.map((r) => `${ir.components[r.component].name}.${r.field}`).sort()).toEqual([
      'Position.x',
      'Position.y',
      'Velocity.x',
      'Velocity.y',
    ]);
    expect(ir.opCount).toBeGreaterThan(0);
    expect(writtenComponents(ir)).toEqual([0, 1]);
    validateIR(ir);
  });

  test('a plain "=" to a field is a write but not a read', () => {
    const facts = deriveFacts(block([assign(field(0, 'x', 'f32'), '=', num(1))]));
    expect(facts.writes).toHaveLength(1);
    expect(facts.reads).toHaveLength(0);
  });

  test('accessKey separates self from other', () => {
    expect(accessKey({ component: 0, field: 'x', role: 0 })).not.toBe(accessKey({ component: 0, field: 'x', role: 1 }));
  });

  test('loop bodies are weighted by their trip cap in opCount', () => {
    const one = deriveFacts(block([assign(field(0, 'x', 'f32'), '+=', num(1))])).opCount;
    const looped = deriveFacts(
      block([forStmt(null, binary('<', num(0), num(1)), null, block([assign(field(0, 'x', 'f32'), '+=', num(1))]), 10)]),
    ).opCount;
    expect(looped).toBeGreaterThanOrEqual(one * 10);
  });
});

describe('gpu IR validation', () => {
  test('rejects a field the component does not have', () => {
    const ir = makeKernelIR({
      name: 'Bad',
      form: 'per-entity',
      components: [Position],
      uniforms: [],
      locals: [],
      body: block([assign(field(0, 'z', 'f32'), '=', num(1))]),
      source: 'p.z = 1',
    });
    expect(() => validateIR(ir)).toThrow(KernelError);
    try {
      validateIR(ir);
    } catch (e) {
      expect((e as KernelError).code).toBe('E_INVALID_IR');
      expect((e as KernelError).message).toContain('Position');
    }
  });

  test('rejects an undeclared uniform', () => {
    const ir = makeKernelIR({
      name: 'Bad',
      form: 'per-entity',
      components: [Position],
      uniforms: [],
      locals: [],
      body: block([assign(field(0, 'x', 'f32'), '+=', uniform('nope'))]),
      source: '',
    });
    expect(() => validateIR(ir)).toThrow(/uniform "nope"/);
  });

  test('rejects assignment to a const local and use before declaration', () => {
    const locals: IRLocal[] = [{ id: 0, name: 'k', type: 'f32', mutable: false }];
    const constAssign = makeKernelIR({
      name: 'Bad',
      form: 'per-entity',
      components: [Position],
      uniforms: [],
      locals,
      body: block([decl(0, num(1)), assign(local(0, 'f32'), '=', num(2))]),
      source: '',
    });
    expect(() => validateIR(constAssign)).toThrow(/is const/);

    const early = makeKernelIR({
      name: 'Bad',
      form: 'per-entity',
      components: [Position],
      uniforms: [],
      locals: [{ id: 0, name: 'k', type: 'f32', mutable: true }],
      body: block([assign(field(0, 'x', 'f32'), '=', local(0, 'f32')), decl(0, num(1))]),
      source: '',
    });
    expect(() => validateIR(early)).toThrow(/before its declaration/);
  });

  test('rejects a non-bool if condition and mismatched ternary branches', () => {
    const badIf = makeKernelIR({
      name: 'Bad',
      form: 'per-entity',
      components: [Position],
      uniforms: [],
      locals: [],
      body: block([ifStmt(num(1), block([]), null)]),
      source: '',
    });
    expect(() => validateIR(badIf)).toThrow(/must be a bool/);

    const badCond = makeKernelIR({
      name: 'Bad',
      form: 'per-entity',
      components: [Position],
      uniforms: [],
      locals: [],
      body: block([assign(field(0, 'x', 'f32'), '=', cond(boolLit(true), num(1), boolLit(false)) as never)]),
      source: '',
    });
    expect(() => validateIR(badCond)).toThrow();
  });

  test('rejects wrong builtin arity', () => {
    const ir = makeKernelIR({
      name: 'Bad',
      form: 'per-entity',
      components: [Position],
      uniforms: [],
      locals: [],
      body: block([assign(field(0, 'x', 'f32'), '=', call('atan2', [num(1)]))]),
      source: '',
    });
    expect(() => validateIR(ir)).toThrow(/atan2/);
  });

  test('rejects break outside a loop and an unbounded loop', () => {
    const noLoop = makeKernelIR({
      name: 'Bad',
      form: 'per-entity',
      components: [Position],
      uniforms: [],
      locals: [],
      body: block([{ kind: 'break' }]),
      source: '',
    });
    expect(() => validateIR(noLoop)).toThrow(/outside a loop/);

    const unbounded = makeKernelIR({
      name: 'Bad',
      form: 'per-entity',
      components: [Position],
      uniforms: [],
      locals: [],
      body: block([forStmt(null, boolLit(true), null, block([]), 0)]),
      source: '',
    });
    expect(() => validateIR(unbounded)).toThrow(/maxIterations/);
  });

  test('rejects a role-1 field in a per-entity kernel', () => {
    const ir = makeKernelIR({
      name: 'Bad',
      form: 'per-entity',
      components: [Position],
      uniforms: [],
      locals: [],
      body: block([assign(field(0, 'x', 'f32'), '+=', field(0, 'y', 'f32', 1))]),
      source: '',
    });
    expect(() => validateIR(ir)).toThrow(/per-entity/);
  });
});

describe('gpu binding layout', () => {
  test('binds one view per 4-byte type, in a stable order, uniform last', () => {
    const ir = moveIR();
    expect(storageViews(ir)).toEqual(['f32']);
    expect(uniformBinding(ir)).toBe(1);

    const mixed = makeKernelIR({
      name: 'Mixed',
      form: 'per-entity',
      components: [Position, Health],
      uniforms: [],
      locals: [],
      body: block([
        assign(field(0, 'x', 'f32'), '+=', num(1)),
        assign(field(1, 'team', 'u32'), '=', num(3)),
      ]),
      source: '',
    });
    expect(storageViews(mixed)).toEqual(['f32', 'u32']);
    expect(uniformBinding(mixed)).toBe(2);
  });

  test('uniform layout is 4-byte scalars from offset 16, size a multiple of 16', () => {
    const ir = moveIR();
    const layout = uniformLayout(ir);
    expect(layout.dt.offset).toBe(0);
    expect(layout.count.offset).toBe(4);
    expect(layout.base.offset).toBe(8);
    expect(layout.countOther.offset).toBe(12);
    expect(layout.fields).toHaveLength(touchedFields(ir).length);
    expect(layout.fields[0].offset).toBe(16);
    // Members are contiguous and ascending.
    for (let i = 1; i < layout.members.length; i++) {
      expect(layout.members[i].offset).toBe(layout.members[i - 1].offset + 4);
    }
    expect(layout.size % 16).toBe(0);
    expect(layout.size).toBeGreaterThanOrEqual(layout.members[layout.members.length - 1].offset + 4);
    expect(layout.uniforms[0].member).toBe('u_gravity');
    expect(offsetMemberName(ir, ir.writes[0])).toMatch(/^o0_/);
  });

  test('dispatch budget matches the measured device limit', () => {
    expect(WORKGROUP_SIZE).toBe(256);
    expect(maxRowsPerDispatch(65535)).toBe(65535 * 256);
  });
});

describe('gpu field eligibility', () => {
  test('only 4-byte views can reach the GPU', () => {
    expect(isGPUEligible('f32')).toBe(true);
    expect(isGPUEligible('i32')).toBe(true);
    expect(isGPUEligible('u32')).toBe(true);
    for (const k of ['f64', 'i8', 'i16', 'u8', 'u16', 'bool', 'str'] as const) {
      expect(isGPUEligible(k)).toBe(false);
    }
  });

  test('gpuBlockers names the offending component and field', () => {
    const ir = makeKernelIR({
      name: 'Damage',
      form: 'per-entity',
      components: [Health],
      uniforms: [],
      locals: [],
      body: block([assign(field(0, 'hp', 'i16'), '-=', num(1))]),
      source: '',
    });
    expect(gpuBlockers(ir)).toEqual([{ component: 'Health', field: 'hp', kind: 'i16' }]);
    expect(gpuBlockers(moveIR())).toEqual([]);
  });
});

describe('gpu builtins', () => {
  test('round is floor(x + 0.5) on both backends, not Math.round', () => {
    expect(BUILTINS.round.wgsl(['x'])).toBe('floor(x + 0.5)');
    expect(BUILTINS.round.js(['x'])).toBe('Math.floor(x + 0.5)');
  });

  test('every builtin emits for both backends with a declared arity', () => {
    for (const name of Object.keys(BUILTINS) as (keyof typeof BUILTINS)[]) {
      const info = BUILTINS[name];
      const args = ['a', 'b'].slice(0, info.arity[0]);
      expect(info.wgsl(args)).toContain('a');
      expect(info.js(args)).toContain('a');
      expect(info.arity[0]).toBeLessThanOrEqual(info.arity[1]);
    }
  });

  test('rand is deterministic and inside [0, 1)', () => {
    for (let s = 0; s < 1000; s++) {
      const v = cozy_rand(s);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
      expect(cozy_rand(s)).toBe(v);
    }
    // Neighbouring seeds must not correlate: a hash, not a counter.
    expect(Math.abs(cozy_rand(1) - cozy_rand(2))).toBeGreaterThan(0.01);
  });
});

describe('gpu diagnostics', () => {
  test('code frame points at the offending span', () => {
    const src = 'const f = (p, v) => {\n  p.x += v.zz;\n};';
    const at = src.indexOf('zz');
    const frame = codeFrame(src, { start: at, end: at + 2 });
    expect(frame).toContain('> 2 |   p.x += v.zz;');
    expect(frame).toMatch(/\^\^/);
    // The caret sits under the "zz".
    const caretLine = frame.split('\n').find((l) => l.includes('^'))!;
    expect(caretLine.indexOf('^')).toBe(frame.split('\n')[1].indexOf('zz'));
  });

  test('KernelError carries its code and renders the frame', () => {
    const e = new KernelError('E_UNKNOWN_IDENTIFIER', 'unknown identifier "gravity"', {
      kernelName: 'Move',
      source: 'p.x += gravity;',
      span: { start: 7, end: 14 },
      hint: 'pass it through uniforms',
    });
    expect(e.code).toBe('E_UNKNOWN_IDENTIFIER');
    expect(e.kernelName).toBe('Move');
    expect(e.message).toContain('[cozyecs/gpu] E_UNKNOWN_IDENTIFIER in kernel "Move"');
    expect(e.message).toContain('Hint: pass it through uniforms');
    expect(e instanceof Error).toBe(true);
  });

  test('describeIR is stable and readable', () => {
    const text = describeIR(moveIR());
    expect(text).toContain('kernel Move [per-entity]');
    expect(text).toContain('Velocity.y += (u.gravity * dt);');
    expect(text).toBe(describeIR(moveIR()));
  });
});
