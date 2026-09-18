// Measures ONE (library, scenario, variant) in the current process and prints one JSON line.
//
//   node benchmarks/worker.js --lib=cozyecs --scenario=packed_5 --variant="cozyecs (direct)" [--time=2000] [--warmup=500] [--runner=mega]
//
// benchmark.js spawns this once per job (process isolation keeps JIT state and GC pressure of one
// library from skewing another). It can also be imported: `measure(opts)` returns the same object.
//
// Adapter instance contract (see adapters/index.js):
//   factory() -> inst | Promise<inst>
//   inst.step()             one op; may return a Promise only if inst.async === true
//   inst.check(ticks)       -> null | error string (may be async); ticks = steps done so far
//   inst.checkHalf?()       optional extra verification
//
// Timing loop / JIT treatment (--runner=, default "mega"):
//   Earlier the batch loop lived inside measure() itself, so V8 decided per library whether step()
//   (and everything it inlines) was OSR-compiled INTO measure(). CPU profiles showed 13-61% of self
//   time attributed to measure() for some jobs (harmony-ecs, cozyecs direct) and ~0% for others, i.e.
//   each library got different JIT treatment from the harness. Now every job calls step() through a
//   tiny batch runner that is compiled per job with new Function():
//     mega    (default) the runner's call site is made megamorphic before the job starts, so V8 never
//             inlines step() into the harness; every library's step() is optimized as its own
//             compilation unit. Cost: one indirect call per op (a few ns vs >= 2 us per op).
//     mono    fresh per-job runner, call site left monomorphic (V8 may inline step() into it)
//     inline  old behaviour: the loop inside measure()
import { fileURLToPath } from 'node:url';
// Libraries such as ecsy keep development-only checks behind NODE_ENV; benchmark production mode
// (benchmark.js also passes it to every spawned process).
if (!process.env.NODE_ENV) process.env.NODE_ENV = 'production';
import { ADAPTERS } from './adapters/index.js';

export const VERIFY_TICKS = 3;
export const STEADY = new Set(['entity_cycle', 'add_remove']);

const errText = (e) => String((e && e.stack) || e).split('\n').slice(0, 3).join(' | ');

// Builds the per-job batch runner. new Function gives a fresh SharedFunctionInfo (own feedback vector)
// per call, so runners never share type feedback, even with --in-process.
export function makeRunner(isAsync, mode = 'mega') {
  const body = isAsync
    ? 'return (async () => { for (let i = 0; i < n; i++) await step(); })();'
    : 'for (let i = 0; i < n; i++) step();';
  const run = new Function('step', 'n', `'use strict'; ${body}`);
  if (mode === 'mega') {
    // Distinct functions (distinct SharedFunctionInfos) -> megamorphic call feedback -> no inlining.
    // Called 64x so the runner's (lazily allocated) feedback vector exists and records the targets.
    const dummies = [];
    for (let k = 0; k < 8; k++) dummies.push(new Function(`return ${k};`));
    for (let round = 0; round < 8; round++) {
      for (let k = 0; k < 8; k++) {
        const r = run(dummies[k], 2);
        if (isAsync) r.catch(() => {});
      }
    }
  }
  return run;
}

// CPU reference probe: a fixed Float32Array kernel timed for `ms` through its own mega runner.
// It runs right before and right after the measured loop; refHz = the lower of the two. On a loaded
// machine with performance + efficiency cores a whole job can land on an efficiency core (~2x slower,
// and libraries do not slow down by the same factor), which a low refHz reveals. benchmark.js uses it to
// retry such jobs (--min-ref). Allocation-free, independent of the library under test.
export function refProbe(ms = 100) {
  const x = new Float32Array(4096);
  const dx = new Float32Array(4096).fill(1);
  const kernel = new Function('x', 'dx', 'return () => { for (let i = 0; i < x.length; i++) x[i] += dx[i]; };')(x, dx);
  const run = makeRunner(false, 'mega');
  const t0 = performance.now();
  let n = 0;
  while (performance.now() - t0 < ms / 2) {
    run(kernel, 64);
    n += 64;
  }
  const s = performance.now();
  let m = 0;
  while (performance.now() - s < ms / 2) {
    run(kernel, 64);
    m += 64;
  }
  return (m * 1000) / (performance.now() - s);
}

export async function measure({ lib, scenario, variant, time = 2000, warmup = 500, minSamples = 10, runner = 'mega' }) {
  const out = { lib, scenario, variant, runner, hz: 0, rme: 0, samples: 0, ops: 0, errors: [] };
  const entry = ADAPTERS.find((a) => a.lib === lib);
  if (!entry) return { ...out, errors: [{ stage: 'load', message: 'unknown lib' }] };
  let mod;
  try {
    mod = await entry.load();
  } catch (e) {
    return { ...out, errors: [{ stage: 'load', message: errText(e) }] };
  }
  const factory = mod.variants[scenario] && mod.variants[scenario][variant];
  if (!factory) return { ...out, errors: [{ stage: 'load', message: 'no such variant' }] };

  const origWarn = console.warn;
  const origLog = console.log;
  const quiet = () => {
    console.warn = () => {};
    console.log = () => {};
  };
  const loud = () => {
    console.warn = origWarn;
    console.log = origLog;
  };

  // 1) benchmark on the FIRST instance created in this process.
  // Fairness audit (r5): verification used to run first, on its own instance. V8 shares type feedback
  // between closures/classes created at the same source site, so for libraries whose adapters create
  // component classes or accessor closures per instance (becsy, ecsy, geotic, ...) the verification
  // world made the benchmark's property accesses polymorphic before it started. Typed-array
  // libraries were unaffected, so the order penalised object-based libraries only. Verification now
  // runs after the measurement, on a fresh instance.
  let inst;
  try {
    quiet();
    inst = await factory();
  } catch (e) {
    loud();
    out.errors.push({ stage: 'setup', message: errText(e) });
    return out;
  }
  const step = inst.step;
  const isAsync = !!inst.async;
  const run = runner === 'inline' ? null : makeRunner(isAsync, runner);
  let ticks = 0;
  const refBefore = refProbe();
  try {
    // warm-up + batch calibration
    let t0 = performance.now();
    let n = 0;
    let batch = 1;
    while (performance.now() - t0 < warmup) {
      if (run) {
        if (isAsync) await run(step, batch);
        else run(step, batch);
      } else if (isAsync) for (let i = 0; i < batch; i++) await step();
      else for (let i = 0; i < batch; i++) step();
      n += batch;
      if (batch < 1 << 20) batch *= 2;
    }
    ticks += n;
    const perMs = n / (performance.now() - t0);
    // ~50 ms per sample, at least 1 op
    batch = Math.max(1, Math.round(perMs * 50));

    const hzs = [];
    let totalOps = 0;
    let totalMs = 0;
    const tStart = performance.now();
    while (performance.now() - tStart < time || hzs.length < minSamples) {
      const s = performance.now();
      if (run) {
        if (isAsync) await run(step, batch);
        else run(step, batch);
      } else if (isAsync) for (let i = 0; i < batch; i++) await step();
      else for (let i = 0; i < batch; i++) step();
      const ms = performance.now() - s;
      hzs.push((batch * 1000) / ms);
      totalOps += batch;
      totalMs += ms;
      if (hzs.length > 10000) break;
    }
    ticks += totalOps;
    const refAfter = refProbe();
    out.refHz = Math.min(refBefore, refAfter);
    out.refHzBefore = refBefore;
    out.refHzAfter = refAfter;
    const mean = hzs.reduce((a, b) => a + b, 0) / hzs.length;
    const variance = hzs.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(1, hzs.length - 1);
    const sem = Math.sqrt(variance / hzs.length);
    out.hz = (totalOps * 1000) / totalMs;
    out.meanSampleHz = mean;
    out.rme = mean ? ((1.96 * sem) / mean) * 100 : 0;
    out.samples = hzs.length;
    // Sample-hz quantiles. On a loaded machine with performance + efficiency cores (Apple M-series) a
    // job can be scheduled on an efficiency core for part or all of its run, which shows up as a
    // bimodal sample distribution; p90SampleHz estimates the fast-core speed (reported in the JSON only; ranking uses hz).
    const sorted = [...hzs].sort((x, y) => x - y);
    const q = (f) => sorted[Math.min(sorted.length - 1, Math.floor(f * sorted.length))];
    out.p10SampleHz = q(0.1);
    out.p50SampleHz = q(0.5);
    out.p90SampleHz = q(0.9);
    out.maxSampleHz = sorted[sorted.length - 1];
    out.ops = totalOps;
  } catch (e) {
    loud();
    out.errors.push({ stage: 'benchmark', message: errText(e) });
    return out;
  } finally {
    loud();
  }

  // 2) steady-state scenarios must still be correct after thousands of ops
  if (STEADY.has(scenario)) {
    try {
      quiet();
      const err = await inst.check(ticks);
      if (err) out.errors.push({ stage: 'after benchmark', message: String(err) });
    } catch (e) {
      out.errors.push({ stage: 'after benchmark', message: 'check crashed: ' + errText(e) });
    } finally {
      loud();
    }
  }

  // 3) correctness on a fresh instance (after the measurement, see step 1)
  try {
    quiet();
    const vinst = await factory();
    for (let t = 0; t < VERIFY_TICKS; t++) await vinst.step();
    let err = await vinst.check(VERIFY_TICKS);
    if (!err && vinst.checkHalf) err = await vinst.checkHalf();
    if (err) out.errors.push({ stage: 'verify', message: String(err) });
  } catch (e) {
    out.errors.push({ stage: 'verify', message: errText(e) });
  } finally {
    loud();
  }
  return out;
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  const args = Object.fromEntries(
    process.argv.slice(2).map((a) => {
      const i = a.indexOf('=');
      return i < 0 ? [a.replace(/^--/, ''), true] : [a.slice(0, i).replace(/^--/, ''), a.slice(i + 1)];
    }),
  );
  const res = await measure({
    lib: args.lib,
    scenario: args.scenario,
    variant: args.variant,
    time: Number(args.time || 2000),
    warmup: Number(args.warmup || 500),
    runner: typeof args.runner === 'string' ? args.runner : 'mega',
  });
  process.stdout.write('\n' + JSON.stringify(res) + '\n');
  process.exit(0);
}
