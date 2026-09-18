// GPU / kernel benchmark: kernelSystem (cpu + gpu backends) vs hand-written loops vs bitecs.
//
//   npm run build && node benchmarks/gpu.js [options]
//
// Every (kernel, variant, size) job runs in its own `node` process (this file with --child), the job
// list is repeated --repeats times (order reversed on odd repeats) and the MEDIAN ms/frame is reported.
//
// Kernels (same entity layout as simple_iter: 4 archetypes Pos+Vel, +A, +B, +A+B, n/4 entities each):
//   simple    p.x += v.dx; p.y += v.dy
//   gravity   v.dy += u.g * dt; p.x += v.dx * dt; p.y += v.dy * dt; bounce at y < 0
//
// Variants:
//   kernel-cpu     kernelSystem({ target: 'cpu' }) driven by world.update(dt)
//   forEachChunk   world.system + q.forEachChunk([Pos, Vel], hand-written kernel), world.update(dt)
//   plain          world.system + plain `for (i < chunk.count)` loop over chunk.col(), world.update(dt)
//   bitecs         bitECS 0.3 sparse-set query + closure-constant columns, called directly
//   gpu-none       kernelSystem({ target: 'gpu', readback: 'none' })     (data stays GPU-resident)
//   gpu-async      kernelSystem({ target: 'gpu', readback: 'async' })    (results land next frame)
//   gpu-sync       kernelSystem({ target: 'gpu', readback: 'sync-frame' }), frame = update + flushKernels
//   GPU frames yield to the event loop once per frame (setImmediate) so Dawn can complete map callbacks;
//   the async/none runs drain with flushKernels() after every timed batch INSIDE the clock, so the
//   reported ms/frame includes the final drain (whose own time is also reported as finalDrainMs).
//
// Correctness gate (every GPU mode and size): after the timed loop and the final flushKernels(), the
// child rebuilds the same world with kernelSystem({ target: 'cpu' }), runs it for the same number of
// frames and compares every row (x, y, dx, dy). gpu-none is read straight out of the device buffer
// (handle.bufferFor + copy + mapAsync). A differing cell is printed as FAIL instead of a time. CPU
// cells are checked against an f32 scalar reference. 'async' cells also report
// stats.coalescedReadbacks / stats.staleReadbacks ('n/a' when the runtime does not expose a field).
//
// Options:
//   --kernels=simple,gravity  --variants=...  --sizes=1000,10000,30000,50000,100000,300000,1000000
//   --repeats=5  --time=600 (ms measured per job)  --warmup=200  --job-timeout=180 (s)
//   --no-gpu            skip gpu-* variants
//   --write             replace the "GPU / kernel" section of benchmarks/RESULTS.md
//   --json=path         also write raw results
//   --from-json=a,b     do not measure: merge earlier --json outputs and render (with --write) them
import { spawnSync } from 'node:child_process';
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import os from 'node:os';
import { fileURLToPath } from 'node:url';

if (!process.env.NODE_ENV) process.env.NODE_ENV = 'production';
const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const args = Object.fromEntries(
  process.argv.slice(2).map((a) => {
    const i = a.indexOf('=');
    return i < 0 ? [a.replace(/^--/, ''), true] : [a.slice(0, i).replace(/^--/, ''), a.slice(i + 1)];
  }),
);
const list = (k, d) => (typeof args[k] === 'string' ? args[k].split(',') : d);
const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length === 0 ? NaN : s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

const fmtOpt = (x) => (typeof x === 'number' && Number.isFinite(x) ? (Number.isInteger(x) || Math.abs(x) >= 100 ? x.toFixed(0) : x.toFixed(3)) : 'n/a');
const statOpt = (st, k) => (st && typeof st[k] === 'number' ? String(st[k]) : 'n/a');
const asyncStats = (st) => `coalesced=${statOpt(st, 'coalescedReadbacks')} stale=${statOpt(st, 'staleReadbacks')} dispatches=${statOpt(st, 'dispatches')}`;
const isFail = (check) => typeof check === 'string' && check !== 'ok' && !check.startsWith('skipped');

const CPU_VARIANTS = ['kernel-cpu', 'forEachChunk', 'plain', 'bitecs'];
const GPU_VARIANTS = ['gpu-none', 'gpu-async', 'gpu-sync'];
const DT = 1 / 60;
const G = -9.8;

// ======================================================================== child
async function child() {
  const variant = args.variant;
  const kernel = args.kernel;
  const n = Number(args.n);
  const time = Number(args.time || 600);
  const warmup = Number(args.warmup || 200);
  const out = { kernel, variant, n, msPerFrame: 0, frames: 0, error: null, check: null, backend: null };
  try {
    const setup = variant === 'bitecs' ? await setupBitecs(kernel, n) : await setupCozy(kernel, variant, n);
    out.backend = setup.backend || null;
    const frame = setup.frame;
    const isAsync = !!setup.async;
    let frames = 0;
    // warm-up
    let t0 = performance.now();
    while (performance.now() - t0 < warmup || frames < 3) {
      if (isAsync) await frame();
      else frame();
      frames++;
    }
    if (setup.drain) await setup.drain();
    // measure in batches; ms/frame = total / frames (drain included)
    const samples = [];
    let measured = 0;
    let batch = 1;
    const tStart = performance.now();
    while (performance.now() - tStart < time || samples.length < 5) {
      const s = performance.now();
      for (let i = 0; i < batch; i++) {
        if (isAsync) await frame();
        else frame();
      }
      if (setup.drain) {
        const d = performance.now();
        await setup.drain();
        out.finalDrainMs = performance.now() - d; // last batch's drain = the final drain
      }
      const ms = performance.now() - s;
      samples.push(ms / batch);
      measured += batch;
      frames += batch;
      if (ms < 20 && batch < 1 << 16) batch *= 2;
      if (samples.length > 2000) break;
    }
    const total = performance.now() - tStart;
    out.msPerFrame = total / measured;
    out.medianSampleMs = median(samples);
    out.frames = measured;
    if (setup.stats) out.stats = setup.stats();
    out.check = await setup.check(frames);
  } catch (e) {
    out.error = String((e && e.stack) || e).split('\n').slice(0, 4).join(' | ');
  }
  process.stdout.write('\n' + JSON.stringify(out) + '\n');
  process.exit(0);
}

// f32 reference for entity with dx=1, dy=2 (initial x=y=0)
function reference(kernel, frames) {
  const f = Math.fround;
  let x = 0, y = 0, dx = 1, dy = 2;
  for (let t = 0; t < frames; t++) {
    if (kernel === 'simple') {
      x = f(x + dx);
      y = f(y + dy);
    } else {
      dy = f(dy + f(f(G) * f(DT)));
      x = f(x + f(dx * f(DT)));
      y = f(y + f(dy * f(DT)));
      if (y < 0) {
        y = 0;
        dy = f(-dy * 0.5);
      }
    }
  }
  return { x, y };
}
// The final drain before the gate: every queued dispatch applied, device idle.
async function gpuFlush(res) {
  if (res.flushKernels) await res.flushKernels(res.world);
  if (res.handle) await res.handle.sync();
  if (res.ctx) await res.ctx.device.queue.onSubmittedWorkDone();
}
function near(a, b) {
  return Math.abs(a - b) <= 1e-3 * Math.max(1, Math.abs(b));
}

async function setupCozy(kernel, variant, n) {
  const { World, component, f32 } = await import('../dist/index.esm.js');
  const Pos = component({ x: f32, y: f32 }, { name: 'Position' });
  const Vel = component({ dx: f32, dy: f32 }, { name: 'Velocity' });
  const A = component({ value: f32 }, { name: 'A' });
  const B = component({ value: f32 }, { name: 'B' });
  const w = new World();
  const archs = [w.archetype(Pos, Vel), w.archetype(Pos, Vel, A), w.archetype(Pos, Vel, B), w.archetype(Pos, Vel, A, B)];
  for (const a of archs) {
    w.spawnMany(a, n / 4, (ch, row) => {
      const v = ch.col(Vel);
      v.dx[row] = 1;
      v.dy[row] = 2;
    });
  }
  const q = w.query({ all: [Pos, Vel] });
  const res = { async: false };
  // Every row's (x, y, dx, dy) in chunk order. gpu-none reads the device copy of each table.
  res.readRows = async () => {
    const out = new Float32Array(n * 4);
    let k = 0;
    for (const ch of q.chunks) {
      const p = ch.col(Pos), v = ch.col(Vel);
      let x = p.x, y = p.y, dx = v.dx, dy = v.dy;
      if (variant === 'gpu-none') {
        const src = res.handle.bufferFor(ch);
        if (!src) throw new Error(`gpu-none: no device buffer for archetype ${ch.id}`);
        const { device } = res.ctx;
        const size = Math.min(src.size, ch.buffer.byteLength) & ~3;
        const dst = device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const enc = device.createCommandEncoder();
        enc.copyBufferToBuffer(src, 0, dst, 0, size);
        device.queue.submit([enc.finish()]);
        await dst.mapAsync(GPUMapMode.READ);
        const bytes = dst.getMappedRange().slice(0);
        dst.unmap();
        dst.destroy();
        const view = (a) => new Float32Array(bytes, a.byteOffset, ch.count);
        x = view(p.x); y = view(p.y); dx = view(v.dx); dy = view(v.dy);
      }
      for (let i = 0; i < ch.count; i++, k += 4) {
        out[k] = x[i]; out[k + 1] = y[i]; out[k + 2] = dx[i]; out[k + 3] = dy[i];
      }
    }
    if (k !== n * 4) throw new Error(`row count ${k / 4} != ${n}`);
    return out;
  };
  const check = async (frames) => {
    if (res.handle) await res.handle.sync();
    if (variant.startsWith('gpu-')) {
      // Gate: same world on the CPU backend, same number of frames, every row compared.
      await gpuFlush(res);
      const got = await res.readRows();
      const ref = await setupCozy(kernel, 'kernel-cpu', n);
      for (let f = 0; f < frames; f++) ref.frame();
      const want = await ref.readRows();
      let bad = 0, first = -1;
      for (let i = 0; i < want.length; i++) if (!near(got[i], want[i])) { bad++; if (first < 0) first = i; }
      if (bad) {
        const r = (first / 4) | 0, b = r * 4;
        return `FAIL ${bad}/${want.length} values differ from the CPU backend after ${frames} frames ` +
          `(row ${r}: got x=${got[b]} y=${got[b + 1]} dy=${got[b + 3]}, cpu x=${want[b]} y=${want[b + 1]} dy=${want[b + 3]})`;
      }
      return 'ok';
    }
    const want = reference(kernel, frames);
    let bad = 0, count = 0;
    for (const ch of q.chunks) {
      const p = ch.col(Pos);
      for (let i = 0; i < ch.count; i++, count++) if (!near(p.x[i], want.x) || !near(p.y[i], want.y)) bad++;
    }
    if (count !== n) return `FAIL count ${count} != ${n}`;
    if (bad) {
      const p = q.chunks[0].col(Pos);
      return `FAIL ${bad}/${count} rows differ (row0 x=${p.x[0]} y=${p.y[0]}, want ${want.x},${want.y} after ${frames} frames)`;
    }
    return 'ok';
  };
  res.check = check;

  if (variant === 'forEachChunk') {
    const PV = [Pos, Vel];
    const kern =
      kernel === 'simple'
        ? (cnt, p, v) => {
            const x = p.x, y = p.y, dx = v.dx, dy = v.dy;
            for (let i = 0; i < cnt; i++) {
              x[i] += dx[i];
              y[i] += dy[i];
            }
          }
        : (cnt, p, v) => {
            // forEachChunk passes no dt; the benchmark dt is the constant DT
            const x = p.x, y = p.y, dx = v.dx, dy = v.dy;
            const dt = DT, gdt = G * DT;
            for (let i = 0; i < cnt; i++) {
              dy[i] += gdt;
              x[i] += dx[i] * dt;
              y[i] += dy[i] * dt;
              if (y[i] < 0) {
                y[i] = 0;
                dy[i] = -dy[i] * 0.5;
              }
            }
          };
    w.system('move', { query: q }, (q, dt) => q.forEachChunk(PV, kern));
    res.frame = () => w.update(DT);
    return res;
  }
  if (variant === 'plain') {
    const body =
      kernel === 'simple'
        ? (q) => {
            const chunks = q.chunks;
            for (let c = 0; c < chunks.length; c++) {
              const ch = chunks[c];
              const p = ch.col(Pos), v = ch.col(Vel);
              for (let i = 0; i < ch.count; i++) {
                p.x[i] += v.dx[i];
                p.y[i] += v.dy[i];
              }
            }
          }
        : (q, dt) => {
            const chunks = q.chunks;
            for (let c = 0; c < chunks.length; c++) {
              const ch = chunks[c];
              const p = ch.col(Pos), v = ch.col(Vel);
              for (let i = 0; i < ch.count; i++) {
                v.dy[i] += G * dt;
                p.x[i] += v.dx[i] * dt;
                p.y[i] += v.dy[i] * dt;
                if (p.y[i] < 0) {
                  p.y[i] = 0;
                  v.dy[i] = -v.dy[i] * 0.5;
                }
              }
            }
          };
    w.system('move', { query: q }, body);
    res.frame = () => w.update(DT);
    return res;
  }

  // kernelSystem variants
  const gpu = await import('../dist/gpu/index.esm.js');
  const isGpu = variant.startsWith('gpu-');
  if (isGpu) {
    let mod;
    try {
      mod = await import('webgpu');
    } catch (e) {
      throw new Error('webgpu package not available: ' + e.message);
    }
    const g = mod.create([]);
    globalThis.__dawnKeepAlive = g;
    Object.assign(globalThis, mod.globals || {});
    gpu.setGPUProvider(g);
    const ctx = await gpu.getGPUContext();
    if (!ctx) throw new Error('no WebGPU device');
    res.ctx = ctx;
    res.flushKernels = gpu.flushKernels;
    res.world = w;
  }
  const readback = variant === 'gpu-none' ? 'none' : variant === 'gpu-sync' ? 'sync-frame' : 'async';
  const opts = {
    components: [Pos, Vel],
    target: isGpu ? 'gpu' : 'cpu',
    readback: isGpu ? readback : 'async',
    kernel:
      kernel === 'simple'
        ? (p, v) => {
            p.x += v.dx;
            p.y += v.dy;
          }
        : (p, v, dt, u) => {
            v.dy += u.g * dt;
            p.x += v.dx * dt;
            p.y += v.dy * dt;
            if (p.y < 0) {
              p.y = 0;
              v.dy = -v.dy * 0.5;
            }
          },
  };
  if (kernel === 'gravity') opts.uniforms = { g: G };
  const origWarn = console.warn;
  const warnings = [];
  console.warn = (...a) => warnings.push(a.join(' '));
  let handle;
  try {
    handle = await gpu.kernelSystem(w, 'Move', opts);
  } finally {
    console.warn = origWarn;
  }
  res.handle = handle;
  res.backend = handle.backend + (warnings.length ? ` (warn: ${warnings[0].slice(0, 80)})` : '');
  res.stats = () => {
    // Explicit copy so getters survive JSON; fields the runtime lacks stay undefined -> 'n/a'.
    const st = handle.stats || {};
    const o = {};
    for (const k of ['dispatches', 'completed', 'pending', 'bytesUploaded', 'bytesReadBack', 'staleReadbacks', 'coalescedReadbacks', 'lastBackend', 'lastEntities'])
      if (st[k] !== undefined) o[k] = st[k];
    return o;
  };
  if (isGpu && handle.backend !== 'gpu') throw new Error(`fell back to ${handle.backend}: ${warnings.join(' | ')}`);
  if (!isGpu) {
    res.frame = () => w.update(DT);
    return res;
  }
  res.async = true;
  const tick = () => new Promise((r) => setImmediate(r));
  if (readback === 'sync-frame') {
    res.frame = async () => {
      w.update(DT);
      await gpu.flushKernels(w);
    };
  } else {
    res.frame = async () => {
      w.update(DT);
      await tick();
    };
    res.drain = () => gpu.flushKernels(w);
  }
  return res;
}

async function setupBitecs(kernel, n) {
  const bitecs = await import('bitecs');
  const { createWorld, defineComponent, Types, addEntity, addComponent, defineQuery, setDefaultSize } = bitecs;
  if (n > 90000 && setDefaultSize) setDefaultSize(Math.ceil(n * 1.1));
  const world = createWorld();
  const Pos = defineComponent({ x: Types.f32, y: Types.f32 });
  const Vel = defineComponent({ dx: Types.f32, dy: Types.f32 });
  const A = defineComponent({ value: Types.f32 });
  const B = defineComponent({ value: Types.f32 });
  for (const extra of [[], [A], [B], [A, B]]) {
    for (let i = 0; i < n / 4; i++) {
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
  const frame =
    kernel === 'simple'
      ? () => {
          const ents = q(world);
          for (let i = 0; i < ents.length; i++) {
            const e = ents[i];
            x[e] += dx[e];
            y[e] += dy[e];
          }
        }
      : () => {
          const ents = q(world);
          const dt = DT;
          for (let i = 0; i < ents.length; i++) {
            const e = ents[i];
            dy[e] += G * dt;
            x[e] += dx[e] * dt;
            y[e] += dy[e] * dt;
            if (y[e] < 0) {
              y[e] = 0;
              dy[e] = -dy[e] * 0.5;
            }
          }
        };
  return {
    frame,
    check(frames) {
      const want = reference(kernel, frames);
      const ents = q(world);
      let bad = 0;
      for (let i = 0; i < ents.length; i++) if (!near(x[ents[i]], want.x) || !near(y[ents[i]], want.y)) bad++;
      if (ents.length !== n) return `FAIL count ${ents.length}`;
      return bad ? `FAIL ${bad} rows differ` : 'ok';
    },
  };
}

// ======================================================================== parent
// --from-json=a.json,b.json: no measuring; merge the raw runs of earlier --json outputs (e.g. one
// file per kernel, run as separate invocations) and render / --write them as one section.
function loadRuns(files) {
  const runs = new Map();
  const unavailable = new Map();
  const kernels = [], sizes = [], variants = [];
  const add = (xs, x) => xs.includes(x) || xs.push(x);
  let started = null, load0 = null, load1 = null, repeats = 0, time = 600, warmup = 200;
  for (const f of files) {
    const j = JSON.parse(readFileSync(f, 'utf8'));
    if (!started || j.started < started) { started = j.started; load0 = j.load0; }
    load1 = j.load1;
    if (j.options) ({ time, warmup } = j.options);
    for (const [k, rs] of Object.entries(j.raw)) {
      const [kernel, n, variant] = k.split('|');
      add(kernels, kernel); add(sizes, Number(n)); add(variants, variant);
      runs.set(k, rs);
      repeats = Math.max(repeats, rs.length);
      for (const r of rs) if (r.error && /no WebGPU device|webgpu package not available|fell back/.test(r.error)) unavailable.set(variant, r.error);
    }
  }
  sizes.sort((a, b) => a - b);
  const order = [...CPU_VARIANTS, ...GPU_VARIANTS];
  variants.sort((a, b) => order.indexOf(a) - order.indexOf(b));
  return { runs, unavailable, kernels, sizes, variants, repeats, time, warmup, load0, load1, started: new Date(started) };
}

async function parent() {
  const merged = typeof args['from-json'] === 'string' ? loadRuns(args['from-json'].split(',')) : null;
  const kernels = merged ? merged.kernels : list('kernels', ['simple', 'gravity']);
  const sizes = merged ? merged.sizes : list('sizes', ['1000', '10000', '30000', '50000', '100000', '300000', '1000000']).map(Number);
  const variants = merged ? merged.variants : list('variants', args['no-gpu'] ? CPU_VARIANTS : [...CPU_VARIANTS, ...GPU_VARIANTS]);
  const repeats = merged ? merged.repeats : Number(args.repeats || 5);
  const time = merged ? merged.time : Number(args.time || 600);
  const warmup = merged ? merged.warmup : Number(args.warmup || 200);
  const jobTimeout = Number(args['job-timeout'] || 180) * 1000;
  const load0 = merged ? merged.load0 : os.loadavg();
  const started = merged ? merged.started : new Date();

  const jobs = [];
  for (const kernel of kernels) for (const n of sizes) for (const variant of variants) jobs.push({ kernel, n, variant });
  const runs = merged ? merged.runs : new Map();
  const key = (j) => `${j.kernel}|${j.n}|${j.variant}`;
  // variant -> reason (skip after a hard failure like "no WebGPU device")
  const unavailable = merged ? merged.unavailable : new Map();
  if (!merged) console.log(`${jobs.length} jobs x ${repeats} repeats, isolated processes, time ${time} ms, load ${load0.map((l) => l.toFixed(1)).join(' ')}`);
  for (let rep = 0; rep < (merged ? 0 : repeats); rep++) {
    const order = rep % 2 ? [...jobs].reverse() : jobs;
    console.log(`# repeat ${rep + 1}/${repeats}`);
    for (const job of order) {
      if (unavailable.has(job.variant)) continue;
      const r = spawnSync(
        process.execPath,
        [here('./gpu.js'), '--child', `--kernel=${job.kernel}`, `--variant=${job.variant}`, `--n=${job.n}`, `--time=${time}`, `--warmup=${warmup}`],
        { encoding: 'utf8', timeout: jobTimeout, maxBuffer: 16 * 1024 * 1024, env: { ...process.env, NODE_ENV: 'production' } },
      );
      const line = (r.stdout || '').trim().split('\n').filter((l) => l.startsWith('{')).pop();
      let res;
      try {
        res = JSON.parse(line);
      } catch {
        res = { ...job, error: `no result (status ${r.status} signal ${r.signal}) ${(r.stderr || '').trim().split('\n').slice(-2).join(' | ')}` };
      }
      if (res.error && /no WebGPU device|webgpu package not available|fell back/.test(res.error)) unavailable.set(job.variant, res.error);
      if (!runs.has(key(job))) runs.set(key(job), []);
      runs.get(key(job)).push(res);
      const failed = !res.error && res.check !== 'ok' && !String(res.check).startsWith('skipped');
      const tail = res.error ? `  !! ${res.error.slice(0, 160)}` : failed ? `  [${res.check}]` : '';
      const extra = job.variant === 'gpu-async' && !res.error ? `  ${asyncStats(res.stats)} drain=${fmtOpt(res.finalDrainMs)}ms` : '';
      const shown = res.error ? '-'.padStart(10) : failed ? 'FAIL'.padStart(10) : res.msPerFrame.toFixed(4).padStart(10);
      console.log(`  ${job.kernel.padEnd(8)} ${String(job.n).padStart(8)} ${job.variant.padEnd(13)} ${shown} ms/frame${extra}${tail}`);
    }
  }
  const load1 = merged ? merged.load1 : os.loadavg();

  // aggregate
  const table = {}; // kernel -> n -> variant -> { ms, min, max, check, error }
  for (const job of jobs) {
    const rs = runs.get(key(job)) || [];
    const ok = rs.filter((r) => !r.error);
    const fails = ok.filter((r) => isFail(r.check));
    const medOpt = (f) => {
      const xs = ok.map(f).filter((x) => typeof x === 'number' && Number.isFinite(x));
      return xs.length ? median(xs) : undefined;
    };
    const cell = {
      // A cell that failed the correctness gate in ANY repeat gets no time (and no break-even).
      ms: ok.length && !fails.length ? median(ok.map((r) => r.msPerFrame)) : NaN,
      fail: fails.length,
      finalDrainMs: medOpt((r) => r.finalDrainMs),
      coalesced: medOpt((r) => r.stats && r.stats.coalescedReadbacks),
      stale: medOpt((r) => r.stats && r.stats.staleReadbacks),
      dispatches: medOpt((r) => r.stats && r.stats.dispatches),
      min: ok.length ? Math.min(...ok.map((r) => r.msPerFrame)) : NaN,
      max: ok.length ? Math.max(...ok.map((r) => r.msPerFrame)) : NaN,
      runs: ok.length,
      check: [...new Set(ok.map((r) => r.check))].join('; '),
      error: rs.find((r) => r.error)?.error || unavailable.get(job.variant) || null,
    };
    ((table[job.kernel] ||= {})[job.n] ||= {})[job.variant] = cell;
  }

  // break-even: smallest n where gpu-mode ms/frame < kernel-cpu ms/frame (log-linear interpolation)
  const breakEven = {};
  for (const kernel of kernels) {
    for (const gv of GPU_VARIANTS.filter((v) => variants.includes(v))) {
      const pts = sizes.map((n) => ({ n, ratio: table[kernel][n][gv].ms / table[kernel][n]['kernel-cpu'].ms })).filter((p) => Number.isFinite(p.ratio));
      let be = null;
      for (let i = 0; i < pts.length; i++) {
        if (pts[i].ratio < 1) {
          if (i === 0) be = `<= ${pts[0].n}`;
          else {
            const a = pts[i - 1], b = pts[i];
            const t = Math.log(a.ratio) / (Math.log(a.ratio) - Math.log(b.ratio));
            be = Math.round(Math.exp(Math.log(a.n) + t * (Math.log(b.n) - Math.log(a.n))));
          }
          break;
        }
      }
      if (be === null && pts.length) be = `> ${pts[pts.length - 1].n}`;
      (breakEven[kernel] ||= {})[gv] = be;
    }
  }

  // print
  const fmt = (x) => (Number.isFinite(x) ? (x < 0.01 ? x.toFixed(4) : x < 1 ? x.toFixed(3) : x.toFixed(2)) : 'n/a');
  const md = [];
  const machine = `${os.cpus()[0]?.model || os.arch()} (${os.cpus().length} cores), ${os.type()} ${os.release()}, Node ${process.version}`;
  md.push('## GPU / kernel', '');
  md.push(`Generated by \`node benchmarks/gpu.js\` at ${started.toISOString()}${merged ? ` (merged from ${args['from-json'].split(',').length} runs with \`--from-json\`)` : ''}.`, '');
  md.push(`- Machine: ${machine}`);
  md.push(`- Load average (1/5/15 min): start ${load0.map((l) => l.toFixed(1)).join(' / ')}, end ${load1.map((l) => l.toFixed(1)).join(' / ')}`);
  md.push(`- Median of ${repeats} isolated processes per cell, ${time} ms measured per process after ${warmup} ms warm-up. ms/frame, lower is better.`);
  md.push('- Layout: 4 archetypes (Pos+Vel, +A, +B, +A+B), n/4 entities each. `simple`: `p.x += v.dx; p.y += v.dy`. `gravity`: gravity + integrate + bounce, with `dt` and a uniform.');
  md.push('- Correctness gate: every CPU cell checks every entity against an f32 reference. Every GPU cell (including `none`, read from the device buffer) is drained with `flushKernels`, then compared row by row with the CPU backend run for the same number of frames; a cell that differs in any repeat shows `FAIL (failed/runs)` instead of a time.');
  md.push('- `gpu-none` / `gpu-async` ms/frame include the drain (`flushKernels`) after every timed batch, so the final drain is paid for.');
  md.push('');
  for (const kernel of kernels) {
    md.push(`### ${kernel}`, '');
    md.push(`| entities | ${variants.join(' | ')} |`);
    md.push(`|---:|${variants.map(() => '---:').join('|')}|`);
    for (const n of sizes) {
      const row = variants.map((v) => {
        const c = table[kernel][n][v];
        if (!c.runs) return c.error ? 'n/a' : '-';
        return c.fail ? `FAIL (${c.fail}/${c.runs})` : fmt(c.ms);
      });
      md.push(`| ${n.toLocaleString('en-US')} | ${row.join(' | ')} |`);
    }
    md.push('');
    const base = (n) => table[kernel][n]['bitecs']?.ms;
    const rel = variants.filter((v) => v !== 'bitecs' && !v.startsWith('gpu-'));
    if (variants.includes('bitecs')) {
      md.push(`Speed vs bitecs (bitecs ms / variant ms, >1 = faster than bitecs):`, '');
      md.push(`| entities | ${rel.join(' | ')} |`);
      md.push(`|---:|${rel.map(() => '---:').join('|')}|`);
      for (const n of sizes) md.push(`| ${n.toLocaleString('en-US')} | ${rel.map((v) => (base(n) / table[kernel][n][v].ms).toFixed(2) + 'x').join(' | ')} |`);
      md.push('');
    }
  }
  const gv = GPU_VARIANTS.filter((v) => variants.includes(v));
  if (variants.includes('gpu-async')) {
    md.push('### gpu-async readback stats (median per process)', '');
    md.push('| kernel | entities | dispatches | coalescedReadbacks | staleReadbacks | final drain ms | gate |', '|---|---:|---:|---:|---:|---:|---|');
    for (const kernel of kernels)
      for (const n of sizes) {
        const c = table[kernel][n]['gpu-async'];
        if (!c.runs) continue;
        md.push(`| ${kernel} | ${n.toLocaleString('en-US')} | ${fmtOpt(c.dispatches)} | ${fmtOpt(c.coalesced)} | ${fmtOpt(c.stale)} | ${fmtOpt(c.finalDrainMs)} | ${c.fail ? `FAIL ${c.fail}/${c.runs}` : 'ok'} |`);
      }
    md.push('');
  }
  if (gv.length) {
    md.push('### GPU break-even vs kernel-cpu (entities)', '');
    md.push(`| kernel | ${gv.join(' | ')} |`, `|---|${gv.map(() => '---:').join('|')}|`);
    for (const kernel of kernels) md.push(`| ${kernel} | ${gv.map((v) => String(breakEven[kernel][v])).join(' | ')} |`);
    md.push('');
    const errs = gv.map((v) => [v, unavailable.get(v)]).filter(([, e]) => e);
    if (errs.length) md.push(...errs.map(([v, e]) => `- ${v} unavailable: \`${e.slice(0, 200)}\``), '');
  }
  const problems = [];
  for (const kernel of kernels)
    for (const n of sizes)
      for (const v of variants) {
        const c = table[kernel][n][v];
        if (c.check && c.check.split('; ').some(isFail)) problems.push(`- ${kernel} ${n} ${v}: ${c.check.slice(0, 400)}`);
        else if (c.error && !unavailable.has(v)) problems.push(`- ${kernel} ${n} ${v}: ${c.error.slice(0, 200)}`);
      }
  if (problems.length) md.push('### Verification failures / errors', '', ...problems, '');
  const section = md.join('\n');
  console.log('\n' + section);
  console.log('BREAK_EVEN ' + JSON.stringify(breakEven));

  if (typeof args.json === 'string') writeFileSync(args.json, JSON.stringify({ started, load0, load1, machine, options: { repeats, time, warmup }, table, breakEven, raw: Object.fromEntries(runs) }, null, 1));
  if (args.write) {
    const file = here('./RESULTS.md');
    const START = '<!-- gpu-kernel:start -->';
    const END = '<!-- gpu-kernel:end -->';
    const block = `${START}\n${section}\n${END}`;
    let text = existsSync(file) ? readFileSync(file, 'utf8') : '# CozyECS benchmark results\n';
    const a = text.indexOf(START), b = text.indexOf(END);
    text = a >= 0 && b > a ? text.slice(0, a) + block + text.slice(b + END.length) : text.replace(/\s*$/, '\n\n') + block + '\n';
    writeFileSync(file, text);
    console.log(`Wrote GPU / kernel section to ${file}`);
  }
}

if (args.child) await child();
else await parent();
