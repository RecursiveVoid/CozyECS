import { describe, test, expect } from '@jest/globals';
import { DEFAULT_MAX_LOOP_ITERATIONS, kernelSource, mergedFieldNamespace, parameterNames, parseKernel, tokenize } from '../src/gpu/parse';
import type { ParseSpec } from '../src/gpu/parse';
import { KernelError, describeIR, exprToString, gpuBlockers, validateIR } from '../src/gpu/ir';
import type { IRComponent, IRFieldKind, KernelErrorCode, KernelIR } from '../src/gpu/ir';

// --- fixtures ---------------------------------------------------------------

const comp = (index: number, id: number, name: string, fields: [string, IRFieldKind][]): IRComponent => ({
  index,
  id,
  name,
  fields: fields.map(([n, k], i) => ({ name: n, kind: k, index: i })),
});

const Position = comp(0, 1, 'Position', [['x', 'f32'], ['y', 'f32'], ['z', 'f32']]);
const Velocity = comp(1, 2, 'Velocity', [['x', 'f32'], ['y', 'f32'], ['z', 'f32']]);
const Health = comp(0, 3, 'Health', [['hp', 'i32'], ['alive', 'bool']]);
const Body = comp(0, 10, 'Body', [['x', 'f32'], ['y', 'f32']]);
const Mot = comp(1, 11, 'Mot', [['vx', 'f32'], ['vy', 'f32'], ['m', 'f32']]);

type Opts = Partial<ParseSpec> & { source?: string };

function parse(fn: unknown, opts: Opts = {}): KernelIR {
  return parseKernel(fn, {
    name: opts.name || 'K',
    form: opts.form || 'per-entity',
    components: opts.components || [Position, Velocity],
    uniformNames: opts.uniformNames || [],
    source: opts.source,
    uniformInitials: opts.uniformInitials,
    maxLoopIterations: opts.maxLoopIterations,
  });
}

/** Parses and expects a KernelError with `code`; returns it for message assertions. */
function reject(code: KernelErrorCode, source: string, opts: Opts = {}): KernelError {
  let thrown: unknown = null;
  try {
    parse(null, { source, ...opts });
  } catch (e) {
    thrown = e;
  }
  expect(thrown).toBeInstanceOf(KernelError);
  expect({ code: (thrown as KernelError).code, source }).toEqual({ code, source });
  return thrown as KernelError;
}

const names = (ir: KernelIR, list: readonly { component: number; field: string }[]): string[] =>
  list.map((a) => `${ir.components[a.component].name}.${a.field}`);

// --- the canonical kernel ---------------------------------------------------

describe('parseKernel: the documented example', () => {
  const move = (p: any, v: any, dt: number, u: any) => {
    v.y += u.gravity * dt;
    p.x += v.x * dt;
    p.y += v.y * dt;
    if (p.y < 0) {
      p.y = 0;
      v.y = -v.y * 0.5;
    }
  };

  test('infers reads, writes and the uses* flags', () => {
    const ir = parse(move, { name: 'Move', uniformNames: ['gravity'] });
    expect(names(ir, ir.writes)).toEqual(['Velocity.y', 'Position.x', 'Position.y']);
    expect(names(ir, ir.reads)).toEqual(['Velocity.y', 'Position.x', 'Velocity.x', 'Position.y']);
    expect(ir.usesDt).toBe(true);
    expect(ir.usesRand).toBe(false);
    expect(ir.opCount).toBeGreaterThan(0);
    validateIR(ir);
  });

  test('is deterministic', () => {
    const a = describeIR(parse(move, { uniformNames: ['gravity'] }));
    const b = describeIR(parse(move, { uniformNames: ['gravity'] }));
    expect(a).toBe(b);
  });

  test('declares every uniform, in order, whether used or not', () => {
    const ir = parse(move, { uniformNames: ['gravity', 'unused'], uniformInitials: [-9.8, 3] });
    expect(ir.uniforms).toEqual([
      { name: 'gravity', index: 0, initial: -9.8 },
      { name: 'unused', index: 1, initial: 3 },
    ]);
  });

  test('parses without touching the console', () => {
    const spies = (['warn', 'log', 'error'] as const).map((k) => {
      const orig = console[k];
      let calls = 0;
      console[k] = (() => {
        calls++;
      }) as never;
      return { k, orig, calls: () => calls };
    });
    try {
      parse((h: any) => {
        h.hp += 1;
      }, { components: [Health] });
    } finally {
      for (const s of spies) console[s.k] = s.orig;
    }
    expect(spies.map((s) => s.calls())).toEqual([0, 0, 0]);
  });
});

// --- function shapes --------------------------------------------------------

describe('function forms', () => {
  test('function expression, named function and method shorthand', () => {
    expect(parse(function (p: any, v: any) {
      p.x += v.x;
    }).writes).toHaveLength(1);
    expect(parse(function moveIt(p: any, v: any) {
      p.x += v.x;
    }).writes).toHaveLength(1);
    const o = {
      move(p: any, v: any) {
        p.x += v.x;
      },
    };
    expect(parse(o.move).writes).toHaveLength(1);
  });

  test('concise arrow body and single-parameter arrow', () => {
    expect(parse((p: any, v: any) => (p.x += v.x)).body.body).toHaveLength(1);
    expect(parse((p: any) => {
      p.x += 1;
    }, { components: [Position] }).writes).toHaveLength(1);
  });

  test('arity may be components, +dt or +dt,u', () => {
    expect(() => parse(null, { source: '(p,v)=>{p.x+=v.x}' })).not.toThrow();
    expect(() => parse(null, { source: '(p,v,t)=>{p.x+=v.x*t}' })).not.toThrow();
    expect(() => parse(null, { source: '(p,v,t,q)=>{p.x+=v.x*t*q.k}', uniformNames: ['k'] })).not.toThrow();
  });

  test('kernelSource rejects non-functions, bound functions, async and generators', () => {
    const code = (fn: () => unknown): string => {
      try {
        fn();
        return 'none';
      } catch (e) {
        return (e as KernelError).code;
      }
    };
    expect(code(() => kernelSource(42, 'K'))).toBe('E_NOT_A_FUNCTION');
    expect(code(() => kernelSource(((p: any) => p).bind(null), 'K'))).toBe('E_NOT_A_FUNCTION');
    expect(code(() => kernelSource(async (): Promise<void> => undefined, 'K'))).toBe('E_KERNEL_ASYNC');
    expect(code(() => kernelSource(function* (): Generator<number> { yield 1; }, 'K'))).toBe('E_KERNEL_ASYNC');
  });
});

// --- minified input ---------------------------------------------------------
//
// The reason parameters are matched by position is that a bundler rewrites the
// kernel before `toString()` sees it. These are the rewrites terser actually
// emits for the documented example.

describe('minified input', () => {
  test('single-letter parameters and a comma sequence', () => {
    const ir = parse(null, { source: '(a,b,c,d)=>{b.y+=d.gravity*c,a.x+=b.x*c,a.y+=b.y*c}', uniformNames: ['gravity'] });
    expect(names(ir, ir.writes)).toEqual(['Velocity.y', 'Position.x', 'Position.y']);
    expect(ir.body.body).toHaveLength(3);
  });

  test('`cond && (a, b)` becomes an if', () => {
    const ir = parse(null, { source: '(a,b)=>{a.y<0&&(a.y=0,b.y=-.5*b.y)}' });
    const s = ir.body.body[0];
    expect(s.kind).toBe('if');
    expect(s.kind === 'if' && s.then.body).toHaveLength(2);
    expect(s.kind === 'if' && s.alt).toBeNull();
  });

  test('`cond || (a)` becomes if(!cond), and a ternary becomes if/else', () => {
    const or = parse(null, { source: '(a,b)=>{a.x>0||(a.x=1)}' }).body.body[0];
    expect(or.kind === 'if' && or.test.kind).toBe('unary');
    const tern = parse(null, { source: '(a,b)=>{a.x>0?a.y=1:a.y=2}' }).body.body[0];
    expect(tern.kind === 'if' && tern.alt.body).toHaveLength(1);
  });

  test('concise body holding a comma sequence', () => {
    expect(parse(null, { source: '(a,b,c)=>(a.x+=b.x*c,a.y+=b.y*c)' }).body.body).toHaveLength(2);
  });

  test('!0 / !1, leading-dot and exponent literals, ASI, comments, CRLF', () => {
    expect(parse(null, { source: '(a,b)=>{let f=!0;if(f){a.x=1}}' }).locals[0].type).toBe('bool');
    expect(parse(null, { source: '(a,b)=>{a.x+=.5\na.y+=1e3}' }).body.body).toHaveLength(2);
    expect(parse(null, { source: '(p,v)=>{ /* c */ p.x+=v.x; // done\n }' }).body.body).toHaveLength(1);
    expect(parse(null, { source: '(p, v) => {\r\n\tp.x += v.x;\r\n}' }).body.body).toHaveLength(1);
  });

  test('a comma or && whose value is used is still rejected', () => {
    reject('E_UNSUPPORTED_EXPRESSION', '(p,v)=>{p.x=(v.x,v.y)}');
    reject('E_UNSUPPORTED_EXPRESSION', '(p,v)=>{p.x=v.x&&v.y}');
  });
});

// --- terser output of the documented example --------------------------------
//
// Hand-reproduced from what terser emits for the task's `Move` kernel: params
// and locals renamed to t/e/o/s/n, whitespace stripped, statements joined with
// commas, the trailing `if` folded into `&&(...)`. The IR must be identical to
// the readable kernel's -- which proves parameters are matched BY POSITION and
// that only field and uniform names (which survive minification) matter.

describe('minified input: terser output of the documented example', () => {
  const readable = `(p, v, dt, u) => {
    const g = u.gravity * dt;
    v.y += g;
    p.x += v.x * dt;
    p.y += v.y * dt;
    if (p.y < 0) {
      p.y = 0;
      v.y = -v.y * 0.5;
    }
  }`;
  const minified: [string, string][] = [
    ['arrow, block body', '(t,e,o,s)=>{const n=s.gravity*o;e.y+=n,t.x+=e.x*o,t.y+=e.y*o,t.y<0&&(t.y=0,e.y=-e.y*.5)}'],
    ['arrow, let local', '(t,e,o,s)=>{let n=s.gravity*o;e.y+=n,t.x+=e.x*o,t.y+=e.y*o,t.y<0&&(t.y=0,e.y=-e.y*.5)}'],
    ['anonymous function', 'function(t,e,o,s){const n=s.gravity*o;e.y+=n,t.x+=e.x*o,t.y+=e.y*o,t.y<0&&(t.y=0,e.y=-e.y*.5)}'],
    ['named function', 'function a(t,e,o,s){const n=s.gravity*o;e.y+=n,t.x+=e.x*o,t.y+=e.y*o,t.y<0&&(t.y=0,e.y=-e.y*.5)}'],
    ['if with braces, semicolons', '(t,e,o,s)=>{const n=s.gravity*o;e.y+=n;t.x+=e.x*o;t.y+=e.y*o;if(t.y<0){t.y=0;e.y=-e.y*.5}}'],
    // Adversarial renaming: the minifier reused the *readable* names in other slots.
    ['names shuffled across slots', '(u,dt,v,p)=>{const x=p.gravity*v;dt.y+=x,u.x+=dt.x*v,u.y+=dt.y*v,u.y<0&&(u.y=0,dt.y=-dt.y*.5)}'],
  ];

  /** describeIR with local names normalized, since those are the only thing a minifier may change. */
  const shape = (ir: KernelIR): string => describeIR({ ...ir, name: 'K', locals: ir.locals.map((l, i) => ({ ...l, name: `L${i}` })) });
  const opts: Opts = { uniformNames: ['gravity'] };
  const reference = shape(parse(null, { source: readable, ...opts }));

  test.each(minified)('%s parses to the same IR as the readable kernel', (_label, source) => {
    const ir = parse(null, { source, ...opts });
    validateIR(ir);
    const want = source.includes('let n') ? reference.replace('const L0', 'let L0') : reference;
    expect(shape(ir)).toBe(want);
    expect(names(ir, ir.writes)).toEqual(['Velocity.y', 'Position.x', 'Position.y']);
    expect(ir.usesDt).toBe(true);
  });

  test('a real minified Function (not just a string) goes through kernelSource', () => {
    // eslint-disable-next-line no-new-func
    const fn = new Function('t', 'e', 'o', 's', 'const n=s.gravity*o;e.y+=n,t.x+=e.x*o,t.y+=e.y*o,t.y<0&&(t.y=0,e.y=-e.y*.5)');
    expect(shape(parse(fn, opts))).toBe(reference);
  });

  test('parameters bind to components by position, never by name', () => {
    // Same text, components swapped: the first parameter is now Velocity.
    const src = '(p,v,o,s)=>{v.y+=s.gravity*o,p.x+=v.x*o}';
    const straight = parse(null, { source: src, ...opts });
    const swapped = parse(null, { source: src, ...opts, components: [{ ...Velocity, index: 0 }, { ...Position, index: 1 }] });
    expect(names(straight, straight.writes)).toEqual(['Velocity.y', 'Position.x']);
    expect(names(swapped, swapped.writes)).toEqual(['Position.y', 'Velocity.x']);
    // A parameter literally called `v` bound to slot 0 is Position.
    const misleading = parse(null, { source: '(v,p)=>{v.x+=p.x}' });
    expect(names(misleading, misleading.writes)).toEqual(['Position.x']);
    expect(names(misleading, misleading.reads)).toEqual(['Position.x', 'Velocity.x']);
    // dt and u are positional too: slot 2 is dt and slot 3 the uniform block whatever they are called.
    const dtu = parse(null, { source: '(a,b,gravity,dt)=>{a.x+=dt.gravity*gravity}', ...opts });
    expect(dtu.usesDt).toBe(true);
    expect(dtu.body.body[0].kind === 'assign' && exprToString(dtu, dtu.body.body[0].value)).toBe('(u.gravity * dt)');
  });

  test('a free identifier in minified code is still named and pointed at uniforms', () => {
    const e = reject('E_UNKNOWN_IDENTIFIER', '(t,e,o)=>{e.y+=r*o,t.x+=e.x*o}');
    expect(e.message).toMatch(/unknown identifier "r"/);
    expect(e.message).toMatch(/uniforms/);
    expect(e.message).toMatch(/u\.r/);
  });
});

// --- every unsupported construct has a clear message ------------------------

describe('rejection messages', () => {
  const cases: [string, KernelErrorCode, string, RegExp][] = [
    ['string', 'E_UNSUPPORTED_LITERAL', '(t,e)=>{t.x="a"}', /string literals are not part of the kernel subset[\s\S]*Hint:.*uniforms/],
    ['template string', 'E_UNSUPPORTED_LITERAL', '(t,e)=>{t.x=`a`}', /not part of the kernel subset/],
    ['object', 'E_UNSUPPORTED_LITERAL', '(t,e)=>{const n={b:1};t.x=1}', /object literals are not part of the kernel subset[\s\S]*Hint:.*uniforms/],
    ['array', 'E_UNSUPPORTED_LITERAL', '(t,e)=>{const n=[1,2];t.x=1}', /array literals are not part of the kernel subset[\s\S]*Hint:/],
    ['new', 'E_UNSUPPORTED_LITERAL', '(t,e)=>{const n=new Float32Array(2);t.x=1}', /`new` is not part of the kernel subset[\s\S]*Hint:/],
    ['closure (arrow)', 'E_UNSUPPORTED_STATEMENT', '(t,e)=>{const n=o=>o*2;t.x=1}', /nested functions are not part of the kernel subset[\s\S]*Hint: Inline/],
    ['closure (parenthesized arrow)', 'E_UNSUPPORTED_STATEMENT', '(t,e)=>{const n=(o)=>o;t.x=1}', /nested functions[\s\S]*Hint: Inline/],
    ['closure (function expr)', 'E_UNSUPPORTED_STATEMENT', '(t,e)=>{const n=function(){};t.x=1}', /nested functions[\s\S]*Hint: Inline/],
    ['closure (declaration)', 'E_UNSUPPORTED_STATEMENT', '(t,e)=>{function n(){}t.x=1}', /nested function is not part of the kernel subset[\s\S]*Hint: Inline/],
    ['user call', 'E_UNSUPPORTED_CALL', '(t,e)=>{t.x=helper(e.x)}', /cannot call "helper" from a kernel[\s\S]*Hint:.*Inline/],
    ['Math.random', 'E_UNSUPPORTED_CALL', '(t,e)=>{t.x=Math.random()}', /Math\.random\(\) is not available inside a kernel[\s\S]*rand\(seed\)/],
    ['async arrow', 'E_KERNEL_ASYNC', 'async(t,e)=>{t.x=1}', /an async kernel cannot run on the GPU/],
    ['async function', 'E_KERNEL_ASYNC', 'async function a(t,e){t.x=1}', /an async kernel cannot run on the GPU/],
    ['await', 'E_KERNEL_ASYNC', '(t,e)=>{t.x=await e.x}', /`await` cannot be used inside a kernel[\s\S]*uniforms/],
    ['try', 'E_UNSUPPORTED_STATEMENT', '(t,e)=>{try{t.x=1}catch(n){}}', /`try` is not part of the kernel subset[\s\S]*Hint:/],
    ['label', 'E_UNSUPPORTED_STATEMENT', '(t,e)=>{n:for(let o=0;o<2;o++){t.x+=1}}', /labeled statements are not part of the kernel subset[\s\S]*unlabeled break\/continue/],
    ['labeled break', 'E_UNSUPPORTED_STATEMENT', '(t,e)=>{for(let o=0;o<2;o++){break n}}', /labeled `break` is not part of the kernel subset[\s\S]*unlabeled/],
    ['labeled continue', 'E_UNSUPPORTED_STATEMENT', '(t,e)=>{for(let o=0;o<2;o++){continue n}}', /labeled `continue` is not part of the kernel subset[\s\S]*unlabeled/],
    ['free identifier', 'E_UNKNOWN_IDENTIFIER', '(t,e,o)=>{e.y+=gravity*o}', /unknown identifier "gravity"[\s\S]*`uniforms`[\s\S]*uniforms: \{ gravity: <value> \}[\s\S]*u\.gravity/],
    ['free identifier (closure var)', 'E_UNKNOWN_IDENTIFIER', 'function(t,e,o){e.y+=speed*o}', /unknown identifier "speed"[\s\S]*uniforms: \{ speed/],
  ];

  test.each(cases)('%s', (_label, code, source, message) => {
    const e = reject(code, source);
    expect(e.message).toMatch(/^\[cozyecs\/gpu\] E_[A-Z_]+ in kernel "K": /);
    expect(e.message).toMatch(message);
    expect(e.span).toBeTruthy();
    expect(e.span.end).toBeGreaterThan(e.span.start);
    expect(e.span.end).toBeLessThanOrEqual(source.length);
  });

  test('a real closure variable cannot leak in: the captured name is reported', () => {
    const speed = 3;
    const e = (() => {
      try {
        parse((p: any, v: any, dt: number) => {
          p.x += v.x * speed * dt;
        });
      } catch (err) {
        return err as KernelError;
      }
      return null;
    })();
    expect(e).toBeInstanceOf(KernelError);
    expect(e.code).toBe('E_UNKNOWN_IDENTIFIER');
    expect(e.message).toMatch(/"speed"[\s\S]*uniforms: \{ speed: <value> \}/);
  });
});

// --- locals, control flow, builtins ----------------------------------------

describe('locals and control flow', () => {
  test('const/let with inferred types and inner-scope declarations', () => {
    const ir = parse(null, { source: '(p,v)=>{const s=2;let t=p.x*s;{const r=t+1;t=r}p.x=t}' });
    expect(ir.locals.map((l) => [l.name, l.type, l.mutable])).toEqual([
      ['s', 'f32', false],
      ['t', 'f32', true],
      ['r', 'f32', false],
    ]);
  });

  test('a boolean local keeps its type', () => {
    const ir = parse(null, { source: '(p,v)=>{const hit=p.y<0;if(hit){p.y=0}}' });
    expect(ir.locals[0].type).toBe('bool');
  });

  test('several declarators in one statement', () => {
    expect(parse(null, { source: '(p,v)=>{let a=1,b=2;p.x=a+b}' }).locals).toHaveLength(2);
  });

  test('builtins: Math functions, rand, index and count', () => {
    const ir = parse(null, { source: '(p,v)=>{p.x=Math.sin(p.y)+Math.hypot(v.x,v.y)+Math.round(v.z)+rand(index)*count}' });
    expect([ir.usesRand, ir.usesIndex, ir.usesCount]).toEqual([true, true, true]);
  });

  test('if / else-if / else, with and without braces', () => {
    const ir = parse(null, { source: '(p,v)=>{if(p.x<0){p.x=0}else if(p.x>1){p.x=1}else{p.x=.5}}' });
    const s = ir.body.body[0];
    expect(s.kind === 'if' && s.alt.body[0].kind).toBe('if');
    const bare = parse(null, { source: '(p,v)=>{if(p.x<0)p.x=0;else p.x=1}' }).body.body[0];
    expect(bare.kind === 'if' && bare.then.body).toHaveLength(1);
  });

  test('integer and sub-word fields parse, and are reported as GPU blockers', () => {
    const ir = parse(null, { source: '(h)=>{h.hp-=1;if(h.hp<1){h.alive=0}}', components: [Health] });
    expect(gpuBlockers(ir).map((b) => b.field)).toEqual(['alive']);
  });
});

describe('loop bounds', () => {
  /** The trip cap of the first loop in the kernel body. */
  const trips = (source: string, opts: Opts = {}): number => {
    for (const s of parse(null, { source, ...opts }).body.body) {
      if (s.kind === 'for' || s.kind === 'while') return s.maxIterations;
    }
    return -1;
  };

  test('a counted loop proves its own bound', () => {
    expect(trips('(p,v)=>{for(let i=0;i<8;i++){p.x+=i}}')).toBe(8);
    expect(trips('(p,v)=>{for(let i=0;i<9;i+=2){p.x+=i}}')).toBe(5);
    expect(trips('(p,v)=>{for(let i=10;i>0;i-=1){p.x+=i}}')).toBe(10);
    expect(trips('(p,v)=>{for(let i=0;i<=4-1;i+=1){p.x+=i}}')).toBe(4);
    expect(trips('(p,v)=>{for(let i=0;i<5;i=i+1){p.x+=i}}')).toBe(5);
    expect(trips('(p,v)=>{for(let i=0;i<3;++i){p.x+=i}}')).toBe(3);
  });

  test('an unprovable bound gets the cap, which is configurable', () => {
    expect(trips('(p,v)=>{for(let i=0;i<p.z;i+=1){p.x+=i}}')).toBe(DEFAULT_MAX_LOOP_ITERATIONS);
    expect(trips('(p,v)=>{for(let i=0;i<p.z;i+=1){p.x+=i}}', { maxLoopIterations: 32 })).toBe(32);
    expect(trips('(p,v)=>{let i=0;while(i<p.z){i+=1;p.x+=1}}')).toBe(DEFAULT_MAX_LOOP_ITERATIONS);
  });

  test('a loop that cannot terminate is rejected', () => {
    reject('E_UNBOUNDED_LOOP', '(p,v)=>{for(;;){p.x+=1}}');
    reject('E_UNBOUNDED_LOOP', '(p,v)=>{let i=0;while(i<10){p.x+=1}}');
    reject('E_UNBOUNDED_LOOP', '(p,v)=>{for(let i=0;i<10;i-=1){p.x+=1}}');
    reject('E_UNBOUNDED_LOOP', '(p,v)=>{for(let i=0;i<100000;i+=1){p.x+=1}}');
  });

  test('a loop with a break is accepted even when the bound is unprovable', () => {
    expect(trips('(p,v)=>{let i=0;while(i<10){if(p.x>0){break}p.x+=1}}')).toBe(DEFAULT_MAX_LOOP_ITERATIONS);
  });

  test('nested loops multiply the op count', () => {
    const ir = parse(null, { source: '(p,v)=>{for(let i=0;i<4;i++){for(let j=0;j<3;j++){p.x+=1}}}' });
    expect(ir.opCount).toBeGreaterThanOrEqual(12);
  });
});

// --- pairwise ---------------------------------------------------------------

describe('pairwise kernels', () => {
  const pw: Opts = { form: 'pairwise', components: [Body, Mot], uniformNames: ['g', 'eps'] };

  test('the documented n-body kernel', () => {
    const ir = parse(
      (self: any, other: any, dt: number, u: any) => {
        const dx = other.x - self.x;
        const dy = other.y - self.y;
        const r2 = dx * dx + dy * dy + u.eps;
        const f = (u.g * other.m) / (r2 * Math.sqrt(r2));
        self.vx += f * dx * dt;
        self.vy += f * dy * dt;
      },
      pw,
    );
    expect(ir.reads.filter((r) => r.role === 1).map((r) => r.field)).toEqual(['x', 'y', 'm']);
    expect(ir.writes.every((w) => w.role === 0)).toBe(true);
    validateIR(ir);
  });

  test('writing to `other` is rejected', () => {
    const e = reject('E_UNSUPPORTED_STATEMENT', '(s,o)=>{o.vx+=1}', pw);
    expect(e.message).toMatch(/other/);
  });

  test('arity is 2..4', () => {
    expect(() => parse(null, { source: '(s,o)=>{s.vx+=o.x}', ...pw })).not.toThrow();
    expect(() => parse(null, { source: '(s,o,t)=>{s.vx+=o.x*t}', ...pw })).not.toThrow();
    expect(() => parse(null, { source: '(s,o,t,q)=>{s.vx+=o.x*t*q.g}', ...pw })).not.toThrow();
    reject('E_PARAM_COUNT', '(s)=>{s.vx+=1}', pw);
  });

  test('two components sharing a field name collide', () => {
    const A = comp(0, 20, 'A', [['x', 'f32']]);
    const B = comp(1, 21, 'B', [['x', 'f32']]);
    let e: KernelError = null;
    try {
      mergedFieldNamespace([A, B], 'K');
    } catch (err) {
      e = err as KernelError;
    }
    expect(e.code).toBe('E_FIELD_COLLISION');
    expect(e.message).toMatch(/"A" and "B"/);
    reject('E_FIELD_COLLISION', '(s,o)=>{s.x+=o.x}', { form: 'pairwise', components: [A, B] });
  });

  test('the namespace merges every component', () => {
    expect([...mergedFieldNamespace([Body, Mot], 'K').keys()]).toEqual(['x', 'y', 'vx', 'vy', 'm']);
  });
});

// --- the error catalogue ----------------------------------------------------

describe('rejections (docs/GPU.md 2.5 and the error catalogue)', () => {
  const cases: [KernelErrorCode, string][] = [
    ['E_UNKNOWN_IDENTIFIER', '(p,v,dt)=>{v.y+=gravity*dt}'],
    ['E_UNKNOWN_IDENTIFIER', '(p,v)=>{p.x=this.x}'],
    ['E_UNKNOWN_FIELD', '(p,v)=>{p.w=1}'],
    ['E_UNKNOWN_UNIFORM', '(p,v,dt,u)=>{p.x=u.nope}'],
    ['E_UNSUPPORTED_LITERAL', '(p,v)=>{p.x="a"}'],
    ['E_UNSUPPORTED_LITERAL', '(p,v)=>{p.x=`a`}'],
    ['E_UNSUPPORTED_LITERAL', '(p,v)=>{const a=[1,2];p.x=1}'],
    ['E_UNSUPPORTED_LITERAL', '(p,v)=>{const a={b:1};p.x=1}'],
    ['E_UNSUPPORTED_LITERAL', '(p,v)=>{const a=new Thing();p.x=1}'],
    ['E_UNSUPPORTED_STATEMENT', '(p,v)=>{return}'],
    ['E_UNSUPPORTED_STATEMENT', '(p,v)=>{try{p.x=1}catch(e){}}'],
    ['E_UNSUPPORTED_STATEMENT', '(p,v)=>{switch(p.x){}}'],
    ['E_UNSUPPORTED_STATEMENT', '(p,v)=>{do{p.x+=1}while(p.x<3)}'],
    ['E_UNSUPPORTED_STATEMENT', '(p,v)=>{outer:for(let i=0;i<2;i++){p.x+=1}}'],
    ['E_UNSUPPORTED_STATEMENT', '(p,v)=>{for(let i=0;i<2;i++){break outer}}'],
    ['E_UNSUPPORTED_STATEMENT', '(p,v)=>{var a=1;p.x=a}'],
    ['E_UNSUPPORTED_STATEMENT', '(p,v)=>{function h(){}p.x=1}'],
    ['E_UNSUPPORTED_STATEMENT', '(p,v)=>{const f=(a)=>a;p.x=1}'],
    ['E_UNSUPPORTED_STATEMENT', '(p,v)=>{p.x}'],
    ['E_UNSUPPORTED_EXPRESSION', '(p,v)=>{p.x=p.y|0}'],
    ['E_UNSUPPORTED_EXPRESSION', '(p,v)=>{p.x=p.y&3}'],
    ['E_UNSUPPORTED_EXPRESSION', '(p,v)=>{p.x=~p.y}'],
    ['E_UNSUPPORTED_EXPRESSION', '(p,v)=>{p.x=p.y<<1}'],
    ['E_UNSUPPORTED_EXPRESSION', '(p,v)=>{p.x=p.y**2}'],
    ['E_UNSUPPORTED_EXPRESSION', '(p,v)=>{p.x=(p.y++)+1}'],
    ['E_UNSUPPORTED_EXPRESSION', '(p,v)=>{p.x=p.y??1}'],
    ['E_UNSUPPORTED_EXPRESSION', '(p,v)=>{p.x=p?.y}'],
    ['E_UNSUPPORTED_EXPRESSION', '(p,v)=>{p.x=typeof p.y}'],
    ['E_UNSUPPORTED_EXPRESSION', '(p,v)=>{if(p.x){p.y=1}}'],
    ['E_UNSUPPORTED_EXPRESSION', '(p,v)=>{p.x=(p.y<1)?1:true}'],
    ['E_UNSUPPORTED_EXPRESSION', '(p,v)=>{let b=p.x<1;b+=1;p.y=1}'],
    ['E_UNSUPPORTED_EXPRESSION', '(p,v)=>{p.x=p.y<1}'],
    ['E_UNSUPPORTED_CALL', '(p,v)=>{p.x=Math.random()}'],
    ['E_UNSUPPORTED_CALL', '(p,v)=>{p.x=Math.cbrt(p.y)}'],
    ['E_UNSUPPORTED_CALL', '(p,v)=>{p.x=helper(p.y)}'],
    ['E_UNSUPPORTED_CALL', '(p,v)=>{p.x=sin(p.y)}'],
    ['E_UNSUPPORTED_CALL', '(p,v)=>{p.x=Math.sin(p.y,1)}'],
    ['E_UNSUPPORTED_MEMBER', '(p,v)=>{p.x=p.y.z}'],
    ['E_UNSUPPORTED_MEMBER', '(p,v)=>{p.x=v[0]}'],
    ['E_UNSUPPORTED_MEMBER', '(p,v)=>{p.x=Math.PI}'],
    ['E_ASSIGN_TO_CONST', '(p,v)=>{const a=1;a=2;p.x=a}'],
    ['E_ASSIGN_TO_CONST', '(p,v,dt)=>{dt=1;p.x=1}'],
    ['E_ASSIGN_TO_CONST', '(p,v,dt,u)=>{u.g=1;p.x=1}'],
    ['E_ASSIGN_TO_CONST', '(p,v)=>{p=v}'],
    ['E_BAD_LOCAL', '(p,v)=>{let a=1;let a=2;p.x=a}'],
    ['E_BAD_LOCAL', '(p,v)=>{let p=1;v.x=p}'],
    ['E_PARAM_COUNT', '(p)=>{p.x=1}'],
    ['E_PARAM_COUNT', '(p,v,dt,u,extra)=>{p.x=1}'],
    ['E_PARAM_COUNT', '(p,...rest)=>{p.x=1}'],
    ['E_PARAM_COUNT', '({x},v)=>{v.x=1}'],
    ['E_PARAM_COUNT', '(p,v=1)=>{p.x=1}'],
    ['E_SYNTAX', '(p,v)=>{p.x=}'],
    ['E_SYNTAX', '(p,v)=>{let a;p.x=1}'],
  ];

  test.each(cases)('%s for %s', (code, source) => {
    const e = reject(code, source, { uniformNames: ['g'] });
    expect(e.message.startsWith('[cozyecs/gpu]')).toBe(true);
    if (e.span) expect(e.message).toContain('^');
  });
});

describe('error quality', () => {
  test('an unknown identifier is named, spanned and pointed at uniforms', () => {
    const e = reject('E_UNKNOWN_IDENTIFIER', '(p,v,dt)=>{\n  v.y += gravity * dt;\n}');
    expect(e.message).toMatch(/unknown identifier "gravity"/);
    expect(e.message).toMatch(/uniforms: \{ gravity/);
    expect(e.source.slice(e.span.start, e.span.end)).toBe('gravity');
    expect(e.message).toMatch(/\^{7}/);
  });

  test('a mistyped field or uniform gets a suggestion', () => {
    expect(reject('E_UNKNOWN_FIELD', '(p,v)=>{p.yy=1}').message).toMatch(/Did you mean p\.y\?/);
    expect(reject('E_UNKNOWN_UNIFORM', '(p,v,dt,u)=>{p.x=u.grav}', { uniformNames: ['gravity', 'k'] }).message).toMatch(
      /declared: gravity, k[\s\S]*Did you mean u\.gravity\?/,
    );
  });

  test('a parameter in the wrong slot is explained as positional matching', () => {
    expect(reject('E_UNSUPPORTED_MEMBER', '(p,v,u)=>{p.x+=u.gravity}', { uniformNames: ['gravity'] }).message).toMatch(/BY POSITION/);
    expect(reject('E_UNKNOWN_FIELD', '(p,u)=>{p.x+=u.gravity}', { uniformNames: ['gravity'] }).message).toMatch(/BY POSITION/);
  });

  test('a grouped statement reports the real problem, not a missing paren', () => {
    expect(reject('E_UNKNOWN_IDENTIFIER', '(p,v)=>{(p.x=wrong)}').message).toMatch(/"wrong"/);
  });

  test('every span points inside the source', () => {
    const source = '(p,v)=>{\n  p.x = v.nope;\n}';
    const e = reject('E_UNKNOWN_FIELD', source);
    expect(source.slice(e.span.start, e.span.end)).toBe('v.nope');
  });
});

// --- the small exported helpers --------------------------------------------

describe('helpers', () => {
  test('parameterNames reports names and never throws', () => {
    expect(parameterNames('(p, v, dt, u) => { p.x += v.x; }')).toEqual(['p', 'v', 'dt', 'u']);
    expect(parameterNames('function move(a,b){}')).toEqual(['a', 'b']);
    expect(parameterNames('p=>p.x')).toEqual(['p']);
    expect(parameterNames('not a function at all')).toEqual([]);
    expect(parameterNames('')).toEqual([]);
  });

  test('tokenize exposes the stream with spans', () => {
    const t = tokenize('p.x += 1.5');
    expect(t.map((x) => x.type)).toEqual(['ident', 'punct', 'ident', 'punct', 'num']);
    expect(t[4].value).toBe('1.5');
    expect(t[0].span).toEqual({ start: 0, end: 1 });
  });
});

// --- fuzz -------------------------------------------------------------------

describe('fuzz', () => {
  test('300 generated kernels parse, validate and are deterministic', () => {
    let seed = 12345;
    const rnd = (): number => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
    const pick = <T>(a: T[]): T => a[Math.floor(rnd() * a.length) % a.length];
    const term = (d: number): string => {
      if (d <= 0) return pick(['p.x', 'v.y', 'dt', 'u.g', '1.5', 'index', 'count', 'rand(index)']);
      const f = pick(['bin', 'call', 'neg', 'paren']);
      if (f === 'bin') return term(d - 1) + pick([' + ', ' - ', ' * ', ' / ']) + term(d - 1);
      if (f === 'call') return pick(['Math.sin(', 'Math.abs(', 'Math.sqrt(']) + term(d - 1) + ')';
      if (f === 'neg') return '-(' + term(d - 1) + ')';
      return '(' + term(d - 1) + ')';
    };
    for (let i = 0; i < 300; i++) {
      const body: string[] = [];
      const n = 1 + Math.floor(rnd() * 4);
      for (let j = 0; j < n; j++) {
        const shape = pick(['assign', 'if', 'for', 'let']);
        if (shape === 'assign') body.push(pick(['p.x', 'p.y', 'v.x', 'v.y']) + pick([' = ', ' += ', ' -= ', ' *= ']) + term(2) + ';');
        else if (shape === 'if') body.push(`if (${term(1)} < ${term(1)}) { p.z = ${term(1)}; } else { p.z = 0; }`);
        else if (shape === 'for') body.push(`for (let q${j} = 0; q${j} < 3; q${j} += 1) { p.x += ${term(1)}; }`);
        else body.push(`let w${j} = ${term(1)}; p.y += w${j};`);
      }
      const source = `(p, v, dt, u) => {${body.join('\n')}}`;
      const ir = parse(null, { source, uniformNames: ['g'] });
      validateIR(ir);
      expect(describeIR(ir)).toBe(describeIR(parse(null, { source, uniformNames: ['g'] })));
    }
  });

  test('a 200-statement body and a 200-deep expression still parse', () => {
    const lines: string[] = [];
    for (let i = 0; i < 200; i++) lines.push(`p.x+=v.x*${i};`);
    expect(parse(null, { source: `(p,v)=>{${lines.join('')}}` }).body.body).toHaveLength(200);
    let e = 'p.x';
    for (let i = 0; i < 200; i++) e = `(${e}+1)`;
    expect(parse(null, { source: `(p,v)=>{p.y=${e}}` }).opCount).toBeGreaterThan(200);
  });
});
