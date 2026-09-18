// CozyECS benchmark runner (js-ecs-benchmarks style scenarios).
//
//   npm run build && npm run benchmark
//
// By default every (scenario, variant) job runs in its own `node` process (benchmarks/worker.js),
// the whole job list is repeated N times (order reversed on odd repeats) and the MEDIAN ops/sec
// of the repeats is reported. This removes most of the suite-order / shared-JIT noise.
//
// Options:
//   --scenario=packed_5,frag_iter   run only these scenarios
//   --lib=cozyecs,bitecs            run only these libraries (adapter ids)
//   --variant=cozyecs,bitecs4       run only these variant names (exact match)
//   --repeats=3                     repeats per job (median reported)
//   --time=2000                     measuring time per job in ms (after warm-up)
//   --warmup=500                    warm-up time per job in ms
//   --quick                         repeats=1, time=400, warmup=200 (noisy, for smoke tests)
//   --in-process                    run all jobs in this process (no isolation; old behaviour)
//   --job-timeout=120               hard timeout per isolated job in seconds
//   --no-memory                     skip the memory probe
//   --memory-count=100000           entity count for the memory probe
//   --no-write                      do not write benchmarks/RESULTS.md
//   --json[=path]                   also write machine-readable results (default benchmarks/results/<label>.json)
//   --label=name                    label stored in the JSON (default: date-time)
//   --runner=mega                   worker batch runner: mega (default) | mono | inline; see worker.js
//   --paired                        paired comparison mode (default repeats 7): every repeat runs the whole
//                                   job list in a freshly SHUFFLED order; for each scenario and repeat, each
//                                   variant's ops/sec is divided by the best COMPETITOR (fastest correct
//                                   variant of a library other than --home) measured in that SAME repeat,
//                                   so slow drifts in machine load cancel out. Reports the median ratio over
//                                   repeats (plus median ops/sec) in the console, RESULTS.md and JSON `paired`.
//   --min-ref=0.85                  slow-core guard: each isolated job also times a fixed CPU reference kernel
//                                   (worker.js refHz). A job whose refHz is below this fraction of the highest
//                                   refHz seen so far in the run (i.e. it ran on an efficiency core or a
//                                   contended CPU) is re-run, up to --retries times; the fastest-ref attempt is
//                                   kept and flagged `slowCore` if still below. --min-ref=0 disables.
//   --retries=2                     max re-runs per job for --min-ref
//   --home=cozyecs                  library whose variants are compared against competitors in --paired
//   --seed=N                        seed for the --paired shuffle (default: random, stored in the JSON)
import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { ADAPTERS, SCENARIOS } from './adapters/index.js';

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
// Production mode for every library (ecsy runs extra checks unless NODE_ENV === 'production').
const childEnv = { ...process.env, NODE_ENV: process.env.NODE_ENV || 'production' };
const args = Object.fromEntries(
  process.argv.slice(2).map((a) => {
    const i = a.indexOf('=');
    return i < 0 ? [a.replace(/^--/, ''), true] : [a.slice(0, i).replace(/^--/, ''), a.slice(i + 1)];
  }),
);
const list = (key) => (typeof args[key] === 'string' ? args[key].split(',') : null);
const onlyScenarios = list('scenario');
const onlyLibs = list('lib');
const onlyVariants = list('variant');
const quick = !!args.quick;
const paired = !!args.paired;
const home = typeof args.home === 'string' ? args.home : 'cozyecs';
const runner = typeof args.runner === 'string' ? args.runner : 'mega';
const seed = args.seed !== undefined ? Number(args.seed) >>> 0 : (Math.random() * 2 ** 32) >>> 0;
const minRef = args['min-ref'] !== undefined ? Number(args['min-ref']) : 0.85;
const retries = Number(args.retries ?? 2);
const repeats = Number(args.repeats || (quick ? 1 : paired ? 7 : 3));
const time = Number(args.time || (quick ? 400 : 2000));
const warmup = Number(args.warmup || (quick ? 200 : 500));
const inProcess = !!args['in-process'];
const jobTimeout = Number(args['job-timeout'] || 120) * 1000;
const memCount = Number(args['memory-count'] || 100000);
const label = typeof args.label === 'string' ? args.label : new Date().toISOString().replace(/[:.]/g, '-');

const scenarios = SCENARIOS.filter((s) => !onlyScenarios || onlyScenarios.includes(s.id));
const adapters = ADAPTERS.filter((a) => !onlyLibs || onlyLibs.includes(a.lib));

const fmt = (n) => (n >= 100 ? Math.round(n).toLocaleString('en-US') : n.toFixed(n >= 10 ? 1 : 2));
// mulberry32: small seeded PRNG so a --paired run's job orders are reproducible with --seed
function rng(a) {
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const random = rng(seed);
const shuffle = (xs) => {
  const a = [...xs];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
};
const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length === 0 ? 0 : s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

// ------------------------------------------------------------------ discover jobs
const skipped = []; // { lib, reason }
const jobs = []; // { scenario, lib, variant }
const libsLoaded = [];
for (const a of adapters) {
  let mod;
  try {
    mod = await a.load();
  } catch (e) {
    const reason = String(e && e.message).split('\n')[0];
    skipped.push({ lib: a.lib, reason });
    console.log(`SKIP ${a.lib}: ${reason}`);
    continue;
  }
  libsLoaded.push(a.lib);
  for (const sc of scenarios) {
    const vs = mod.variants[sc.id];
    if (!vs) continue;
    for (const variant of Object.keys(vs)) {
      if (onlyVariants && !onlyVariants.includes(variant)) continue;
      jobs.push({ scenario: sc.id, lib: a.lib, variant });
    }
  }
}

let measureInProcess = null;
if (inProcess) measureInProcess = (await import('./worker.js')).measure;

function runJob(job) {
  const r = spawnSync(
    process.execPath,
    [here('./worker.js'), `--lib=${job.lib}`, `--scenario=${job.scenario}`, `--variant=${job.variant}`, `--time=${time}`, `--warmup=${warmup}`, `--runner=${runner}`],
    { encoding: 'utf8', timeout: jobTimeout, maxBuffer: 64 * 1024 * 1024, env: childEnv },
  );
  const line = (r.stdout || '').trim().split('\n').filter((l) => l.startsWith('{')).pop();
  try {
    return JSON.parse(line);
  } catch {
    const why = r.error ? String(r.error.message) : `status ${r.status} signal ${r.signal}`;
    return { ...job, hz: 0, rme: 0, samples: 0, errors: [{ stage: 'process', message: `no result (${why}) ${(r.stderr || '').trim().split('\n').slice(-2).join(' | ')}` }] };
  }
}

// ------------------------------------------------------------------ run
const runs = new Map(); // key -> [result]
const key = (j) => `${j.scenario}\u0000${j.variant}`;
const t0 = Date.now();
let maxRef = 0; // highest reference-kernel speed seen in this run
let retried = 0;
let slowJobs = 0;
if (minRef > 0) {
  // Seed maxRef with a few probes in this process so the first jobs already have a baseline.
  const { refProbe } = await import('./worker.js');
  for (let i = 0; i < 5; i++) maxRef = Math.max(maxRef, refProbe(100));
  console.log(`reference kernel: ${fmt(maxRef)} ops/sec (slow-core guard at ${minRef} x that, ${retries} retries)`);
}
console.log(
  `${jobs.length} jobs x ${repeats} repeats, ${inProcess ? 'in-process' : 'isolated processes'}, time ${time} ms, warm-up ${warmup} ms, runner ${runner}` +
    (paired ? `, PAIRED (shuffled, seed ${seed}, home ${home})` : ''),
);
for (let rep = 0; rep < repeats; rep++) {
  const order = paired ? shuffle(jobs) : rep % 2 ? [...jobs].reverse() : jobs;
  console.log(`\n# repeat ${rep + 1}/${repeats}`);
  for (const job of order) {
    let res = inProcess ? await measureInProcess({ ...job, time, warmup, runner }) : runJob(job);
    // slow-core guard (see --min-ref)
    for (let a = 0; minRef > 0 && res.refHz && a < retries; a++) {
      if (res.refHz >= minRef * maxRef) break;
      console.log(`  ${job.scenario.padEnd(13)} ${job.variant.padEnd(20)} ref ${fmt(res.refHz)} < ${minRef} x ${fmt(maxRef)}: slow core, retrying`);
      const again = inProcess ? await measureInProcess({ ...job, time, warmup, runner }) : runJob(job);
      again.attempts = (res.attempts || 1) + 1;
      if (!again.refHz || again.refHz < res.refHz) {
        res.attempts = again.attempts;
      } else res = again;
      retried++;
    }
    if (res.refHz) {
      maxRef = Math.max(maxRef, res.refHz);
      if (minRef > 0 && res.refHz < minRef * maxRef) res.slowCore = true;
    }
    res.rep = rep;
    if (!runs.has(key(job))) runs.set(key(job), []);
    runs.get(key(job)).push(res);
    const err = res.errors && res.errors.length ? `  !! ${res.errors.map((e) => `${e.stage}: ${e.message}`).join('; ')}` : '';
    const slow = res.slowCore ? '  [slow core]' : '';
    console.log(`  ${job.scenario.padEnd(13)} ${job.variant.padEnd(20)} ${fmt(res.hz).padStart(12)} ops/sec ±${(res.rme || 0).toFixed(2)}% (${res.samples} samples, ref ${fmt(res.refHz || 0)})${slow}${err}`);
  }
}

// ------------------------------------------------------------------ aggregate
// Re-flag against the final maxRef (a job measured before the fastest probe was seen may be slow too).
if (minRef > 0) for (const r of [...runs.values()].flat()) if (r.refHz && r.refHz < minRef * maxRef) r.slowCore = true;
slowJobs = [...runs.values()].flat().filter((r) => r.slowCore).length;
const results = {}; // scenario -> [{ lib, variant, hz, min, max, runs, rme, errors }]
const failures = [];
for (const sc of scenarios) {
  results[sc.id] = [];
  for (const job of jobs.filter((j) => j.scenario === sc.id)) {
    const rs = runs.get(key(job)) || [];
    const hzs = rs.map((r) => r.hz);
    const errors = [];
    for (const r of rs) for (const e of r.errors || []) if (!errors.some((x) => x.stage === e.stage)) errors.push(e);
    for (const e of errors) failures.push({ scenario: sc.id, variant: job.variant, stage: e.stage, message: e.message });
    results[sc.id].push({
      lib: job.lib,
      variant: job.variant,
      hz: median(hzs),
      min: Math.min(...hzs),
      max: Math.max(...hzs),
      runs: hzs,
      rme: median(rs.map((r) => r.rme || 0)),
      errors,
    });
  }
}

console.log(`\n## Median of ${repeats} run(s) (ops/sec)`);
for (const sc of scenarios) {
  console.log(`\n${sc.id}`);
  const rs = [...results[sc.id]].sort((a, b) => b.hz - a.hz);
  const best = rs.length ? rs[0].hz : 0;
  for (const r of rs) {
    const bad = r.errors.length ? '  (WRONG RESULT / ERROR)' : '';
    console.log(`  ${r.variant.padEnd(20)} ${fmt(r.hz).padStart(12)}  ${((r.hz / best) * 100).toFixed(0).padStart(4)}%  [${fmt(r.min)} .. ${fmt(r.max)}]${bad}`);
  }
}

// ------------------------------------------------------------------ paired ratios
// paired[scenario] = { competitors, bestPerRepeat: [{variant, hz}], variants: { [variant]: { lib, hz: [..], ratio: [..], medianHz, medianRatio, minRatio, maxRatio } } }
// ratio[r] = hz of the variant in repeat r / hz of the fastest correct competitor in repeat r.
const pairedOut = {};
if (paired) {
  console.log(`\n## Paired: median ratio vs best competitor in the same repeat (${repeats} repeats, home ${home})`);
  for (const sc of scenarios) {
    const scJobs = jobs.filter((j) => j.scenario === sc.id);
    const comp = scJobs.filter((j) => j.lib !== home);
    const entry = { competitors: comp.map((j) => j.variant), bestPerRepeat: [], variants: {} };
    for (let rep = 0; rep < repeats; rep++) {
      let best = null;
      for (const j of comp) {
        const r = (runs.get(key(j)) || [])[rep];
        if (r && !r.slowCore && !(r.errors && r.errors.length) && r.hz > 0 && (!best || r.hz > best.hz)) best = { variant: j.variant, hz: r.hz };
      }
      entry.bestPerRepeat.push(best);
    }
    for (const j of scJobs) {
      const rs = runs.get(key(j)) || [];
      const hz = rs.map((r) => r.hz);
      // Runs flagged slowCore (see --min-ref) are left out of both sides of the ratio.
      const ratio = rs
        .map((r, rep) => (entry.bestPerRepeat[rep] && !r.slowCore && !(r.errors && r.errors.length) ? r.hz / entry.bestPerRepeat[rep].hz : null))
        .filter((x) => x !== null);
      entry.variants[j.variant] = {
        lib: j.lib,
        hz,
        ratio,
        medianHz: median(hz),
        slowCoreRuns: rs.filter((r) => r.slowCore).length,
        medianRatio: ratio.length ? median(ratio) : null,
        minRatio: ratio.length ? Math.min(...ratio) : null,
        maxRatio: ratio.length ? Math.max(...ratio) : null,
      };
    }
    const bestNames = {};
    for (const b of entry.bestPerRepeat) if (b) bestNames[b.variant] = (bestNames[b.variant] || 0) + 1;
    entry.bestCompetitorCounts = bestNames;
    pairedOut[sc.id] = entry;
    console.log(`\n${sc.id}  (best competitor per repeat: ${Object.entries(bestNames).map(([v, n]) => `${v} x${n}`).join(', ') || 'none'})`);
    const vs = Object.entries(entry.variants).sort((a, b) => (b[1].medianRatio ?? -1) - (a[1].medianRatio ?? -1));
    for (const [v, e] of vs) {
      const rr = e.medianRatio === null ? '   n/a' : e.medianRatio.toFixed(3);
      const span = e.medianRatio === null ? '' : `[${e.minRatio.toFixed(3)} .. ${e.maxRatio.toFixed(3)}]`;
      console.log(`  ${v.padEnd(20)} ${fmt(e.medianHz).padStart(12)}  ratio ${rr} ${span}`);
    }
  }
}

// ------------------------------------------------------------------ memory
const memory = [];
if (!args['no-memory']) {
  console.log(`\n## memory: ${memCount.toLocaleString('en-US')} entities with Position + Velocity (median of ${repeats})`);
  for (const lib of libsLoaded) {
    const rows = [];
    for (let rep = 0; rep < repeats; rep++) {
      const r = spawnSync(process.execPath, ['--expose-gc', here('./memory.js'), lib, String(memCount)], {
        encoding: 'utf8',
        timeout: 300000,
        env: childEnv,
      });
      const line = (r.stdout || '').trim().split('\n').filter((l) => l.startsWith('{')).pop();
      let row;
      try {
        row = JSON.parse(line);
      } catch {
        row = { lib, error: `no output (status ${r.status}) ${(r.stderr || '').split('\n')[0]}` };
      }
      rows.push(row);
      if (row.error) break;
    }
    const good = rows.filter((x) => !x.error);
    let row;
    if (!good.length) row = rows[rows.length - 1];
    else {
      const bpe = median(good.map((x) => x.bytesPerEntity));
      row = { ...good.find((x) => x.bytesPerEntity === bpe) || good[0] };
      row.bytesPerEntity = bpe;
      row.total = median(good.map((x) => x.total));
      row.heap = median(good.map((x) => x.heap));
      row.external = median(good.map((x) => x.external));
      row.createMs = median(good.map((x) => x.createMs));
      row.runs = good.map((x) => x.bytesPerEntity);
    }
    memory.push(row);
    if (row.error) console.log(`  ${lib}: ERROR ${row.error}`);
    else
      console.log(
        `  ${lib.padEnd(12)} total ${(row.total / 1048576).toFixed(2)} MB (heap ${(row.heap / 1048576).toFixed(2)} MB, buffers ${(row.external / 1048576).toFixed(2)} MB) = ${row.bytesPerEntity.toFixed(1)} B/entity, created in ${row.createMs.toFixed(0)} ms`,
      );
  }
}

const env = {
  node: process.version,
  platform: `${process.platform}/${process.arch}`,
  cpu: os.cpus()[0]?.model ?? 'unknown CPU',
  cores: os.cpus().length,
  loadavg: os.loadavg().map((x) => +x.toFixed(2)),
};
console.log(`\nDone in ${((Date.now() - t0) / 1000).toFixed(0)} s; load average now ${env.loadavg.join(' ')}; ${retried} slow-core retries, ${slowJobs} job(s) still flagged slow`);

// ------------------------------------------------------------------ JSON
if (args.json) {
  const file = typeof args.json === 'string' ? path.resolve(args.json) : here(`./results/${label}.json`);
  mkdirSync(path.dirname(file), { recursive: true });
  writeFileSync(
    file,
    JSON.stringify(
      {
        label,
        date: new Date().toISOString(),
        env,
        options: { repeats, time, warmup, isolated: !inProcess, memCount, runner, paired, minRef, retries, ...(paired ? { home, seed } : {}) },
        scenarios,
        skipped,
        results,
        memory,
        failures,
        slowCore: { retried, flagged: slowJobs, maxRefHz: maxRef, jobs: [...runs.values()].flat().filter((r) => r.slowCore).map((r) => ({ scenario: r.scenario, variant: r.variant, rep: r.rep, refHz: r.refHz })) },
        ...(paired ? { paired: pairedOut } : {}),
      },
      null,
      2,
    ),
  );
  console.log(`Wrote ${file}`);
}

// ------------------------------------------------------------------ RESULTS.md
if (!args['no-write']) {
  const out = [];
  out.push('# CozyECS benchmark results', '');
  out.push(`Generated by \`npm run benchmark\` on ${new Date().toISOString().slice(0, 10)}.`, '');
  out.push(`- Node ${env.node} (${env.platform}), ${env.cpu}, ${env.cores} cores, load average at end ${env.loadavg.join(' ')}`);
  out.push(
    `- ops/sec (higher is better): ${inProcess ? 'all jobs in one process' : 'each job in its own node process'}, ${warmup} ms warm-up + ${time} ms measuring, **median of ${repeats} repeat(s)**${paired ? ` in shuffled order (paired, seed ${seed})` : ''}; best per scenario in **bold**; step() called through the \`${runner}\` batch runner (see worker.js)`,
  );
  out.push('- Each variant is benchmarked on the first world created in its process (so no verification world pollutes V8 type feedback), steady-state scenarios are re-checked after the benchmark, then the variant is verified on a fresh world (3 ticks, then state checked). NODE_ENV=production.');
  out.push('');
  out.push('## Scenarios', '');
  for (const s of scenarios) out.push(`- **${s.id}**: ${s.desc}`);
  out.push('');
  out.push(
    'CozyECS variants: `cozyecs` = function systems + chunk loops via `world.update()` (structural changes go through the command buffer); `cozyecs (direct)` = raw chunk loops with immediate structural changes; `cozyecs (forEach)` = `query.forEach` callbacks.',
  );
  out.push('');
  out.push('## Throughput (ops/sec, median)', '');
  const variantNames = [];
  for (const s of scenarios) for (const r of results[s.id]) if (!variantNames.includes(r.variant)) variantNames.push(r.variant);
  out.push(`| variant | ${scenarios.map((s) => s.id).join(' | ')} |`);
  out.push(`|---|${scenarios.map(() => '---:').join('|')}|`);
  for (const v of variantNames) {
    const cells = scenarios.map((s) => {
      const rs = results[s.id];
      const r = rs.find((x) => x.variant === v);
      if (!r) return 'n/a';
      const best = Math.max(...rs.filter((x) => !x.errors.length).map((x) => x.hz));
      const txt = fmt(r.hz);
      return (r.hz === best ? `**${txt}**` : txt) + (r.errors.length ? ' (wrong result)' : '');
    });
    out.push(`| ${v} | ${cells.join(' | ')} |`);
  }
  out.push('');
  out.push('### Relative to the fastest correct variant in each scenario', '');
  out.push(`| variant | ${scenarios.map((s) => s.id).join(' | ')} |`);
  out.push(`|---|${scenarios.map(() => '---:').join('|')}|`);
  for (const v of variantNames) {
    const cells = scenarios.map((s) => {
      const rs = results[s.id];
      const r = rs.find((x) => x.variant === v);
      if (!r) return 'n/a';
      const best = Math.max(...rs.filter((x) => !x.errors.length).map((x) => x.hz));
      return `${((r.hz / best) * 100).toFixed(0)}%`;
    });
    out.push(`| ${v} | ${cells.join(' | ')} |`);
  }
  out.push('');
  if (repeats > 1) {
    out.push('### Spread across repeats (min .. max)', '');
    out.push(`| variant | ${scenarios.map((s) => s.id).join(' | ')} |`);
    out.push(`|---|${scenarios.map(() => '---:').join('|')}|`);
    for (const v of variantNames) {
      const cells = scenarios.map((s) => {
        const r = results[s.id].find((x) => x.variant === v);
        return r ? `${fmt(r.min)} .. ${fmt(r.max)}` : 'n/a';
      });
      out.push(`| ${v} | ${cells.join(' | ')} |`);
    }
    out.push('');
  }
  if (paired) {
    out.push(`### Paired ratio vs best competitor (median over ${repeats} shuffled repeats, home \`${home}\`)`, '');
    out.push('Each value is ops/sec divided by the fastest correct non-' + home + ' variant measured in the same repeat; >= 1.00 means #1.', '');
    out.push(`| variant | ${scenarios.map((s) => s.id).join(' | ')} |`);
    out.push(`|---|${scenarios.map(() => '---:').join('|')}|`);
    for (const v of variantNames) {
      const cells = scenarios.map((s) => {
        const e = pairedOut[s.id] && pairedOut[s.id].variants[v];
        return e && e.medianRatio !== null ? e.medianRatio.toFixed(2) : 'n/a';
      });
      out.push(`| ${v} | ${cells.join(' | ')} |`);
    }
    out.push('');
  }
  if (memory.length) {
    out.push(`## Memory: ${memCount.toLocaleString('en-US')} entities with Position{x,y} + Velocity{dx,dy}`, '');
    out.push(
      'Separate `node --expose-gc` process per library (median of repeats), `NODE_ENV=production`, code path warmed with a 10-entity world; `gc()` before and after creation; retained JS heap + ArrayBuffer (external) bytes of the world plus one Position+Velocity query. Numeric fields are `f32` where the library supports typed storage and JS numbers elsewhere. Libraries with fixed-capacity stores (bitecs4, wolf-ecs, harmony-ecs, becsy) are sized to exactly the entity count (bitecs 0.3 uses its 100k default size); CozyECS grows its storage on demand.',
      '',
    );
    out.push('| library | total MB | heap MB | buffers MB | bytes/entity | create time |');
    out.push('|---|---:|---:|---:|---:|---:|');
    for (const m of memory) {
      if (m.error) out.push(`| ${m.lib} | error: ${m.error} | | | | |`);
      else
        out.push(
          `| ${m.lib} | ${(m.total / 1048576).toFixed(2)} | ${(m.heap / 1048576).toFixed(2)} | ${(m.external / 1048576).toFixed(2)} | ${m.bytesPerEntity.toFixed(1)} | ${m.createMs.toFixed(0)} ms |`,
        );
    }
    out.push('');
  }
  if (skipped.length || failures.length) {
    out.push('## Skipped libraries / verification failures', '');
    for (const s of skipped) out.push(`- **${s.lib}** skipped: ${s.reason}`);
    for (const f of failures) out.push(`- **${f.variant}** / ${f.scenario} (${f.stage}): ${f.message}`);
    out.push('');
  }
  // Keep the GPU / kernel section (written by benchmarks/gpu.js) and everything after it.
  const prev = existsSync(here('./RESULTS.md')) ? readFileSync(here('./RESULTS.md'), 'utf8') : '';
  const gpuAt = prev.indexOf('<!-- gpu-kernel:start -->');
  writeFileSync(here('./RESULTS.md'), out.join('\n') + (gpuAt >= 0 ? '\n' + prev.slice(gpuAt) : ''));
  console.log(`Wrote ${here('./RESULTS.md')}`);
}
