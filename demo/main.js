// CozyECS live demo: N particles as ECS entities, moved by one JavaScript
// kernel that runs either on the CPU (a compiled JS loop) or on the GPU (the
// same function compiled to a WGSL compute shader). In GPU mode the particles
// never come back to JavaScript: the renderer draws straight from the kernel's
// storage buffer (handle.bufferFor).

import * as cozy from './dist/index.esm.js';
import * as G from './dist/gpu/index.esm.js';

const { World, component, f32 } = cozy;

// ---------------------------------------------------------------------------
// ECS setup
// ---------------------------------------------------------------------------

const Position = component({ x: f32, y: f32 }, { name: 'Position' });
const Velocity = component({ vx: f32, vy: f32 }, { name: 'Velocity' });

// The kernel. This exact function is parsed and compiled for both backends.
const kernel = (p, v, dt, u) => {
  const dx = u.mx - p.x;
  const dy = u.my - p.y;
  const d2 = dx * dx + dy * dy + 0.02;
  const f = u.pull / (d2 * Math.sqrt(d2));
  v.vx += dx * f * dt;
  v.vy += (dy * f - u.gravity) * dt;
  v.vx *= u.drag;
  v.vy *= u.drag;
  p.x += v.vx * dt;
  p.y += v.vy * dt;
  if (p.x < -u.ax) { p.x = -u.ax; v.vx = -v.vx * 0.7; }
  if (p.x > u.ax) { p.x = u.ax; v.vx = -v.vx * 0.7; }
  if (p.y < -1) { p.y = -1; v.vy = -v.vy * 0.7; }
  if (p.y > 1) { p.y = 1; v.vy = -v.vy * 0.7; }
};

// ax = screen width / height: the world is [-ax, ax] x [-1, 1], so shapes stay round.
const UNIFORMS = { mx: 0, my: 0, pull: 0.12, gravity: 0.25, drag: 0.996, ax: 1 };

// ---------------------------------------------------------------------------
// DOM
// ---------------------------------------------------------------------------

const $ = (id) => document.getElementById(id);
const canvas = $('stage');
const ui = {
  fps: $('fps'), ms: $('ms'), n: $('n'), be: $('be'), autoNote: $('autoNote'),
  calibrate: $('calibrate'), reset: $('reset'), banner: $('banner'), hint: $('hint'), src: $('src'),
  backend: $('backend'), count: $('count'),
  mem: $('mem'), bpe: $('bpe'), memNote: $('memNote'), reclaim: $('reclaim'), reclaimNote: $('reclaimNote'),
  overlay: $('overlay'), layout: $('layout'), archmeta: $('archmeta'), tick: $('tick'), watch: $('watch'), watchsrc: $('watchsrc'),
};
const overlayCtx = ui.overlay.getContext('2d');
const WATCH_COLORS = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181'];
const fmt = (n) => n.toLocaleString('en-US');
// Browsers round performance.now() (often to 0.1 ms), so per-system times are averaged over frames.
function ema(key, value) {
  const t = state.times;
  t[key] = t[key] ? t[key] + 0.05 * (value - t[key]) : value;
}

function banner(msg) {
  ui.banner.textContent = msg;
  ui.banner.style.display = msg ? 'block' : 'none';
}

function highlight(src) {
  const esc = src.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  return esc
    .replace(/\b(const|if|let|for|return)\b/g, '<span class="tok-k">$1</span>')
    .replace(/\b(Math\.\w+)\b/g, '<span class="tok-p">$1</span>')
    .replace(/(?<![\w.])(\d+(?:\.\d+)?)\b/g, '<span class="tok-n">$1</span>');
}
ui.src.innerHTML = highlight(kernel.toString().replace(/^\s+/gm, (m) => m.replace(/ {2}/g, '  ')));

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const state = {
  backend: 'gpu', // 'cpu' | 'gpu'
  count: 1_000_000,
  world: null,
  arch: null,
  handle: null,
  building: false,
  gpu: null, // { device, context, format, pipeline, layout, uniform, cpuBuf, bindGroup, boundBuffer }
  ctx2d: null,
  pointer: { active: false, x: 0, y: 0 },
  tick: 0,
  // Simulation clock in seconds: the sum of the dt actually handed to the kernel.
  // Marker extrapolation measures sample age against this, not wall time, so a
  // frame-rate dip (dt is clamped to 1/30) cannot make it overshoot.
  simTime: 0,
  times: { particles: 0, watch: 0, render: 0 },
  // A few entities we follow on screen and in the panel.
  watch: {
    rows: [],
    ents: [],
    samples: new Float32Array(20), // the newest sample: x, y, vx, vy per marker
    incoming: new Float32Array(20), // scratch for one arriving readback
    pos: new Float32Array(10), // what we draw this frame: extrapolated x, y
    trails: [],
    hasSample: false,
    sampleTime: 0, // the simTime the sample in `samples` belongs to
    seq: 0,
    lastSeq: 0,
    pool: [], // rotating staging buffers: { buf, pending, dead, seq, simTime, tick, wall }
    ageFrames: 0, // ema, for the panel
    ageSec: 0, // ema of the same age in seconds, caps the extrapolation
    src: '',
    // Honest before/after numbers; see cozyDemo.markerStats().
    diag: { samples: 0, ageFrames: 0, ageMs: 0, rawPx: 0, extraPx: 0 },
  },
  mem: null, // last world.memory() snapshot (or an estimate), refreshed with the panel
  reclaimNote: '',
  reclaiming: false,
  disposed: 0, // worlds torn down so far (one per rebuild after the first)
};

// Live knobs so the two fixes can be measured against the old behaviour:
// cozyDemo.markerTuning.pool = 1; cozyDemo.markerTuning.extrapolate = false;
const markerTuning = { pool: 3, extrapolate: true };

function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Fills an (empty) archetype with n particles on a spinning disc. */
function spawnParticles(world, arch, n) {
  const r = rng(7);
  world.spawnMany(arch, n, (chunk, row) => {
    const p = chunk.col(Position);
    const v = chunk.col(Velocity);
    // A disc around the center, spinning.
    const a = r() * Math.PI * 2;
    const d = Math.sqrt(r()) * 0.7;
    const x = Math.cos(a) * d, y = Math.sin(a) * d;
    p.x[row] = x;
    p.y[row] = y;
    v.vx[row] = -y * 0.9 + (r() - 0.5) * 0.05;
    v.vy[row] = x * 0.9 + (r() - 0.5) * 0.05;
  });
}

/**
 * Hands one world's memory back, completely, before the next one is built.
 *
 * Every backend or count switch builds a fresh World, so without this the page
 * would keep the old one alive through `state` AND through the GPU objects that
 * name its buffers -- a bind group holds its storage buffer, and the staging
 * buffers hold device memory the JS heap never shows. Order matters: the demo's
 * own references go first, then the kernel's device copies, then the world.
 *
 * Returns the `world.memory().total` of the world that was released, or 0.
 */
function teardown() {
  const world = state.world, handle = state.handle;
  const was = world ? memorySnapshot() : null;

  // 1. What the demo itself holds. The staging buffers read the kernel's
  //    storage buffer; the bind group and `boundBuffer` name it directly; the
  //    CPU-backend upload buffer is sized for the old table, so a 1M -> 10k
  //    switch would otherwise strand its device memory for the whole session.
  releasePool();
  const g = state.gpu;
  if (g) {
    g.bindGroup = null;
    g.boundBuffer = null;
    if (g.cpuBuf) {
      try {
        g.cpuBuf.destroy();
      } catch {
        /* already gone */
      }
      g.cpuBuf = null;
    }
  }
  resetWatch();
  state.mem = null;
  state.handle = null;
  state.world = null;
  state.arch = null;

  // 2. The kernel: its device copy of the table, its readback pool and its
  //    pipeline references. Idempotent, and world.dispose() would do it too.
  if (handle) {
    try {
      handle.destroy();
    } catch (e) {
      banner('Kernel teardown failed: ' + e.message);
    }
  }

  // 3. The world: systems, query caches, listeners, pending commands, every
  //    archetype table and the entity index. dispose() is newer than the
  //    published build, so a demo running against an older cozyecs falls back
  //    to clear() + compact(), which at least gives the table buffers back.
  if (world) {
    try {
      if (typeof world.dispose === 'function') {
        world.dispose();
      } else if (typeof world.clear === 'function' && typeof world.compact === 'function') {
        world.clear();
        world.compact({ strings: true, minBytes: 0 });
      }
    } catch (e) {
      banner('World teardown failed: ' + e.message);
    }
  }
  if (was) state.disposed++;
  return was ? was.total : 0;
}

async function build() {
  if (state.building) return;
  state.building = true;
  try {
    // The old world goes away BEFORE the new one allocates, so the page never
    // holds two worlds' tables at once (at 1M that is ~23 MB either way).
    const freed = teardown();
    const n = state.count;
    const world = new World({ initialCapacity: n });
    const arch = world.archetype(Position, Velocity);
    spawnParticles(world, arch, n);
    const handle = await G.kernelSystem(world, 'particles', {
      components: [Position, Velocity],
      target: state.backend,
      // GPU: the data stays on the GPU and is rendered from there.
      readback: state.backend === 'gpu' ? 'none' : 'async',
      uniforms: UNIFORMS,
      kernel,
    });
    if (state.backend === 'gpu' && handle.backend !== 'gpu') {
      banner('The GPU backend is not available here, so the kernel runs on the CPU.');
    }
    // Two more systems in a second group, so one frame is:
    //   world.update(dt)            -> group 'update': the particles kernel
    //   world.update(dt, 'render')  -> group 'render': watch, then render
    world.system('watch', { group: 'render' }, () => {
      const t = performance.now();
      sampleWatched();
      advanceMarkers();
      ema('watch', performance.now() - t);
    });
    world.system('render', { group: 'render' }, () => {
      const t = performance.now();
      if (state.gpu) renderGPU();
      else render2D();
      ema('render', performance.now() - t);
    });

    state.world = world;
    state.arch = arch;
    state.handle = handle;
    state.tick = 0;
    state.simTime = 0;
    state.times = { particles: 0, watch: 0, render: 0 };
    setupWatch(arch, n);
    drawLayout(arch);
    syncReclaimButton();
    ui.n.textContent = n >= 1e6 ? n / 1e6 + 'M' : n / 1e3 + 'k';
    updateAutoNote();
    if (freed > 0) noteRebuild(freed);
  } finally {
    state.building = false;
  }
}

/**
 * One fading line in the memory section: the previous world was disposed and
 * how much it was holding. Honest about what the number is -- the reserved
 * bytes world.memory() reported for that world, not a heap measurement.
 */
function noteRebuild(freed) {
  const note =
    `Rebuild ${fmt(state.disposed)}: the previous world was <strong>disposed</strong>` +
    `, giving back the ${mb(freed)} it reserved.`;
  state.reclaimNote = note;
  setTimeout(() => {
    if (state.reclaimNote === note) {
      state.reclaimNote = '';
      updatePanel();
    }
  }, 5000);
}

// ---------------------------------------------------------------------------
// 'auto' explanation
// ---------------------------------------------------------------------------

function autoThreshold(ir) {
  let scale = G.BASELINE_CPU_NS_PER_ENTITY / G.cpuCostEstimate(ir);
  scale = Math.min(4, Math.max(0.125, scale));
  return G.getAutoThresholds().fireAndForget * scale;
}

function updateAutoNote(calibrated) {
  if (!state.handle) return;
  const thr = autoThreshold(state.handle.ir);
  const k = thr >= 1e6 ? (thr / 1e6).toFixed(1) + 'M' : Math.round(thr / 1000) + 'k';
  ui.autoNote.innerHTML = state.gpu
    ? `With <code>target: 'auto'</code>, this kernel would switch to the GPU above <strong>~${k} entities</strong>` +
      (calibrated ? ' (measured on this device).' : ' (built-in estimate, measured on an Apple M4).')
    : 'WebGPU is not available in this browser, so only the CPU backend runs here.';
}

ui.calibrate.addEventListener('click', async () => {
  ui.calibrate.disabled = true;
  const label = ui.calibrate.textContent;
  ui.calibrate.textContent = 'Measuring…';
  try {
    const result = await G.calibrateAuto(cozy);
    if (!result) banner('No WebGPU device: nothing to calibrate.');
    updateAutoNote(!!result);
  } catch (e) {
    banner('Calibration failed: ' + e.message);
  } finally {
    ui.calibrate.disabled = false;
    ui.calibrate.textContent = label;
  }
});

// ---------------------------------------------------------------------------
// WebGPU renderer: instanced quads read straight from the table buffer
// ---------------------------------------------------------------------------

const RENDER_WGSL = /* wgsl */ `
struct R {
  xOff: u32, yOff: u32, vxOff: u32, vyOff: u32,
  sizeX: f32, sizeY: f32, alpha: f32, invAx: f32,
};
@group(0) @binding(0) var<storage, read> data: array<f32>;
@group(0) @binding(1) var<uniform> r: R;

struct VO { @builtin(position) pos: vec4f, @location(0) color: vec4f };

const CORNERS = array<vec2f, 6>(
  vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
  vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));

@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
  let p = vec2f(data[r.xOff + ii], data[r.yOff + ii]);
  let v = vec2f(data[r.vxOff + ii], data[r.vyOff + ii]);
  let c = CORNERS[vi];
  var o: VO;
  o.pos = vec4f(p.x * r.invAx + c.x * r.sizeX, p.y + c.y * r.sizeY, 0.0, 1.0);
  let t = clamp(length(v) * 0.9, 0.0, 1.0);
  let blue = vec3f(0.22, 0.53, 0.90);
  let orange = vec3f(0.85, 0.35, 0.15);
  let pale = vec3f(1.0, 0.86, 0.55);
  let col = select(mix(orange, pale, (t - 0.6) / 0.4), mix(blue, orange, t / 0.6), t < 0.6);
  o.color = vec4f(col, r.alpha);
  return o;
}

@fragment fn fs(i: VO) -> @location(0) vec4f { return i.color; }
`;

async function initGPU() {
  const ctx = await G.getGPUContext();
  if (!ctx) return null;
  const device = ctx.device;
  const context = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });
  const layout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
    ],
  });
  const module = device.createShaderModule({ code: RENDER_WGSL });
  const additive = { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' };
  const pipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
    vertex: { module, entryPoint: 'vs' },
    fragment: { module, entryPoint: 'fs', targets: [{ format, blend: { color: additive, alpha: additive } }] },
    primitive: { topology: 'triangle-list' },
  });
  const uniform = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  return { device, context, format, pipeline, layout, uniform, cpuBuf: null, bindGroup: null, boundBuffer: null };
}

function columnOffsets(arch) {
  const p = arch.col(Position), v = arch.col(Velocity);
  return [p.x.byteOffset / 4, p.y.byteOffset / 4, v.vx.byteOffset / 4, v.vy.byteOffset / 4];
}

/** One empty render pass: clears the canvas when there is nothing to draw. */
function clearGPU() {
  const g = state.gpu;
  const enc = g.device.createCommandEncoder();
  enc
    .beginRenderPass({
      colorAttachments: [{ view: g.context.getCurrentTexture().createView(), clearValue: { r: 0.027, g: 0.035, b: 0.05, a: 1 }, loadOp: 'clear', storeOp: 'store' }],
    })
    .end();
  g.device.queue.submit([enc.finish()]);
}

function renderGPU() {
  const g = state.gpu, arch = state.arch, handle = state.handle;
  if (!arch || !handle) return;
  const n = arch.count;
  // After world.clear() + world.compact() the table is deflated to zero bytes:
  // there is no row to read and no buffer to bind, so just clear the canvas.
  if (n === 0 || arch.buffer.byteLength === 0) {
    g.boundBuffer = null;
    clearGPU();
    return;
  }
  let source;
  if (handle.backend === 'gpu') {
    source = handle.bufferFor(arch); // the kernel's own storage buffer
    if (!source) {
      clearGPU(); // before the first dispatch
      return;
    }
  } else {
    // CPU backend: upload the four columns the renderer reads.
    const bytes = arch.buffer.byteLength;
    if (!g.cpuBuf || g.cpuBuf.size < bytes) {
      if (g.cpuBuf) g.cpuBuf.destroy();
      g.cpuBuf = g.device.createBuffer({ size: Math.ceil(bytes / 4) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    }
    const p = arch.col(Position), v = arch.col(Velocity);
    for (const col of [p.x, p.y, v.vx, v.vy]) {
      g.device.queue.writeBuffer(g.cpuBuf, col.byteOffset, col.buffer, col.byteOffset, n * 4);
    }
    source = g.cpuBuf;
  }
  if (g.boundBuffer !== source) {
    g.bindGroup = g.device.createBindGroup({
      layout: g.layout,
      entries: [{ binding: 0, resource: { buffer: source } }, { binding: 1, resource: { buffer: g.uniform } }],
    });
    g.boundBuffer = source;
  }
  const [xo, yo, vxo, vyo] = columnOffsets(arch);
  // Half-size of each dot in device pixels: smaller dots for bigger crowds keep fill cost down.
  const px = Math.max(0.75, Math.min(2.0, 2.0 - Math.log10(n / 10000) * 0.6)) * Math.min(devicePixelRatio, 2);
  const alpha = Math.min(0.9, Math.max(0.16, 60000 / n));
  const u = new ArrayBuffer(32);
  new Uint32Array(u, 0, 4).set([xo, yo, vxo, vyo]);
  new Float32Array(u, 16, 4).set([px / canvas.width, px / canvas.height, alpha, 1 / aspect()]);
  g.device.queue.writeBuffer(g.uniform, 0, u);

  const enc = g.device.createCommandEncoder();
  const pass = enc.beginRenderPass({
    colorAttachments: [{ view: g.context.getCurrentTexture().createView(), clearValue: { r: 0.027, g: 0.035, b: 0.05, a: 1 }, loadOp: 'clear', storeOp: 'store' }],
  });
  pass.setPipeline(g.pipeline);
  pass.setBindGroup(0, g.bindGroup);
  pass.draw(6, n);
  pass.end();
  g.device.queue.submit([enc.finish()]);
}

// ---------------------------------------------------------------------------
// Canvas 2D fallback (no WebGPU): CPU backend, additive pixels
// ---------------------------------------------------------------------------

let img = null, pix = null;
function render2D() {
  const arch = state.arch;
  if (!arch) return;
  const ctx = state.ctx2d, W = canvas.width, H = canvas.height;
  if (!img || img.width !== W || img.height !== H) {
    img = ctx.createImageData(W, H);
    pix = new Uint8ClampedArray(img.data.buffer);
  }
  pix.fill(0);
  for (let i = 3; i < pix.length; i += 4) pix[i] = 255;
  const p = arch.col(Position), v = arch.col(Velocity), n = arch.count, inv = 1 / aspect();
  const add = Math.max(24, Math.min(160, (60000 / n) * 160)) | 0;
  for (let i = 0; i < n; i++) {
    const x = ((p.x[i] * inv + 1) * 0.5 * W) | 0, y = ((1 - p.y[i]) * 0.5 * H) | 0;
    if (x < 0 || y < 0 || x >= W || y >= H) continue;
    const t = Math.min(1, Math.hypot(v.vx[i], v.vy[i]) * 0.9);
    const o = (y * W + x) * 4;
    pix[o] += (0.22 + 0.63 * t) * add;
    pix[o + 1] += (0.53 - 0.18 * t) * add;
    pix[o + 2] += (0.9 - 0.75 * t) * add;
  }
  ctx.putImageData(img, 0, 0);
}

// ---------------------------------------------------------------------------
// Input, sizing, loop
// ---------------------------------------------------------------------------

function resize() {
  const dpr = Math.min(devicePixelRatio || 1, 2);
  const scale = state.gpu ? dpr : Math.min(dpr, 1); // keep the 2D fallback cheap
  canvas.width = Math.max(1, Math.floor(innerWidth * scale));
  canvas.height = Math.max(1, Math.floor(innerHeight * scale));
  ui.overlay.width = Math.max(1, Math.floor(innerWidth * dpr));
  ui.overlay.height = Math.max(1, Math.floor(innerHeight * dpr));
}
addEventListener('resize', resize);

function aspect() {
  return innerWidth / Math.max(1, innerHeight);
}

function toClip(e) {
  return [(e.clientX / innerWidth) * 2 - 1, 1 - (e.clientY / innerHeight) * 2];
}
canvas.addEventListener('pointerdown', (e) => {
  state.pointer.active = true;
  [state.pointer.x, state.pointer.y] = toClip(e);
  canvas.setPointerCapture(e.pointerId);
  ui.hint.style.opacity = 0;
});
canvas.addEventListener('pointermove', (e) => {
  if (state.pointer.active) [state.pointer.x, state.pointer.y] = toClip(e);
});
const release = () => (state.pointer.active = false);
canvas.addEventListener('pointerup', release);
canvas.addEventListener('pointercancel', release);

function setPressed(group, value) {
  for (const b of group.querySelectorAll('button')) b.setAttribute('aria-pressed', String(b.dataset.v === String(value)));
}
ui.backend.addEventListener('click', async (e) => {
  const b = e.target.closest('button');
  if (!b || b.disabled || b.dataset.v === state.backend || state.building || state.reclaiming) return;
  state.backend = b.dataset.v;
  setPressed(ui.backend, state.backend);
  await build();
  resetStats();
});
ui.count.addEventListener('click', async (e) => {
  const b = e.target.closest('button');
  if (!b || b.disabled || Number(b.dataset.v) === state.count || state.building || state.reclaiming) return;
  state.count = Number(b.dataset.v);
  setPressed(ui.count, state.count);
  await build();
  resetStats();
});
ui.reset.addEventListener('click', async () => {
  if (state.reclaiming) return;
  await build();
  resetStats();
});

// ---------------------------------------------------------------------------
// Reclaiming memory: world.clear() -> world.compact() -> respawn
// ---------------------------------------------------------------------------

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/**
 * The reclaim round trip, run between ticks (a click handler, never inside a
 * system): clear() drops every row and deflates the tables, compact() gives the
 * buffers back, releaseUnused() gives the GPU buffers back, and then the same
 * archetype is refilled. The archetype object, its id and the cached query
 * chunk lists survive all of it -- only the buffers go.
 */
/** The button only works on a build that has the memory API; say so if not. */
function syncReclaimButton() {
  const world = state.world;
  const ok = !!world && typeof world.clear === 'function' && typeof world.compact === 'function';
  ui.reclaim.disabled = !ok;
  ui.reclaim.title = ok
    ? 'world.clear() then world.compact(), then spawn the particles again'
    : 'This build of cozyecs predates world.clear() and world.compact().';
}

async function reclaim() {
  const world = state.world, arch = state.arch, h = state.handle;
  if (!world || !arch || state.building || state.reclaiming) return;
  if (typeof world.clear !== 'function' || typeof world.compact !== 'function') return;
  state.reclaiming = true;
  ui.reclaim.disabled = true;
  const label = ui.reclaim.textContent;
  try {
    const n = arch.count;
    const before = memorySnapshot();
    world.clear(); // events: false -> no per-entity onRemove; that is the fast path
    world.compact({ strings: true });
    const after = memorySnapshot();
    // The kernel's device buffers belong to the rows that just went away.
    const gpuFreed = typeof h.releaseUnused === 'function' ? h.releaseUnused() : -1;
    setupWatch(arch, 0);
    drawLayout(arch);
    state.reclaimNote =
      `Gave back <strong>${mb(before.total - after.total)}</strong>` +
      (gpuFreed > 0 ? ` and ${mb(gpuFreed)} on the GPU` : '') +
      `. Respawning…`;
    ui.reclaim.textContent = 'Reclaimed';
    updatePanel();
    // Let a few frames run so the panel and the empty stage are visible.
    await sleep(1200);
    if (state.world !== world || state.arch !== arch) return; // rebuilt meanwhile

    spawnParticles(world, arch, n);
    if (typeof h.markCpuDirty === 'function') h.markCpuDirty(arch); // the CPU wrote kernel-owned fields
    setupWatch(arch, n);
    drawLayout(arch);
    const note = `Respawned ${fmt(n)}: the deflated table re-grew, same archetype, same query.`;
    state.reclaimNote = note;
    setTimeout(() => {
      if (state.reclaimNote === note) {
        state.reclaimNote = '';
        updatePanel();
      }
    }, 5000);
  } catch (e) {
    state.reclaimNote = 'Reclaim failed: ' + e.message;
  } finally {
    state.reclaiming = false;
    ui.reclaim.disabled = false;
    ui.reclaim.textContent = label;
    syncReclaimButton();
    updatePanel();
  }
}
ui.reclaim.addEventListener('click', reclaim);

const stats = { fps: 0, ms: 0, last: 0, shown: 0 };
function resetStats() {
  stats.fps = 0;
  stats.ms = 0;
}

function frame(now) {
  requestAnimationFrame(frame);
  const dtReal = stats.last ? (now - stats.last) / 1000 : 1 / 60;
  stats.last = now;
  const handle = state.handle;
  if (!handle || !state.world || state.building) return;

  // Uniforms: the pointer while dragging, otherwise an attractor on a slow Lissajous path.
  const ax = aspect();
  handle.setUniform('ax', ax);
  if (state.pointer.active) {
    handle.setUniform('mx', state.pointer.x * ax);
    handle.setUniform('my', state.pointer.y);
    handle.setUniform('pull', 0.6);
  } else {
    handle.setUniform('mx', Math.sin(now * 0.00041) * 0.55 * ax);
    handle.setUniform('my', Math.sin(now * 0.00067) * 0.45);
    handle.setUniform('pull', 0.12);
  }

  const dt = Math.min(dtReal, 1 / 30);
  state.simTime += dt;
  const t0 = performance.now();
  state.world.update(dt); // group 'update': the particles kernel
  const t1 = performance.now();
  state.world.update(dt, 'render'); // group 'render': watch + render
  const ms = performance.now() - t0;
  ema('particles', t1 - t0);
  state.tick++;
  drawOverlay(now);

  const a = 0.08;
  stats.fps = stats.fps ? stats.fps + a * (1 / Math.max(dtReal, 1e-3) - stats.fps) : 1 / Math.max(dtReal, 1e-3);
  stats.ms = stats.ms ? stats.ms + a * (ms - stats.ms) : ms;
  if (now - stats.shown > 250) {
    stats.shown = now;
    ui.fps.textContent = Math.round(stats.fps);
    ui.ms.textContent = stats.ms < 1 ? stats.ms.toFixed(2) : stats.ms.toFixed(1);
    const be = handle.backend;
    ui.be.innerHTML = `<span class="dot" style="background:${be === 'gpu' ? 'var(--orange)' : 'var(--blue)'}"></span>${be.toUpperCase()}`;
    updatePanel();
  }
}

// ---------------------------------------------------------------------------
// The ECS made visible: archetype layout, systems, watched entities
// ---------------------------------------------------------------------------

function mb(bytes) {
  return bytes >= 1048576 ? (bytes / 1048576).toFixed(1) + ' MB' : Math.round(bytes / 1024) + ' KB';
}

function drawLayout(arch) {
  if (arch.buffer.byteLength === 0) {
    ui.layout.innerHTML = '<div style="--c:#1b2430; flex:1; color:var(--muted)">deflated · 0 bytes</div>';
    ui.archmeta.textContent = `0 entities · ${arch.rowBytes} bytes per row · the archetype survives, its buffer does not`;
    return;
  }
  const p = arch.col(Position), v = arch.col(Velocity);
  const cols = [
    ['entity', 'u32 handle', arch.entities, '#4b535d'],
    ['x', 'f32', p.x, '#3987e5'],
    ['y', 'f32', p.y, '#2f6fc0'],
    ['vx', 'f32', v.vx, '#199e70'],
    ['vy', 'f32', v.vy, '#12805a'],
  ];
  ui.layout.innerHTML = cols
    .map(([name, type, arr, c]) => `<div style="--c:${c}; flex:${arr.byteLength}" title="${name}: ${type}, ${mb(arr.byteLength)} at byte ${fmt(arr.byteOffset)}">${name}</div>`)
    .join('');
  ui.archmeta.textContent =
    `${fmt(arch.count)} entities · ${arch.rowBytes} bytes per row · ${mb(arch.buffer.byteLength)} in one ArrayBuffer`;
}

/**
 * Forgets every sample, trail and entity handle the markers kept. Nothing here
 * touches the world, so it is also the watch half of `teardown()`: after it the
 * demo holds no entity handle, row index or sampled value from the old world.
 */
function resetWatch() {
  const w = state.watch;
  w.rows = [];
  w.ents = [];
  w.trails = [];
  w.samples.fill(0);
  w.pos.fill(0);
  w.hasSample = false;
  w.sampleTime = 0;
  w.lastSeq = w.seq;
  w.ageFrames = 0;
  w.ageSec = 0;
  w.src = '';
  resetMarkerDiag();
}

/** Picks the watched rows and resets everything that belongs to a sample. */
function setupWatch(arch, n) {
  resetWatch();
  const w = state.watch;
  w.rows = n > 0 ? [0, 0.2, 0.4, 0.6, 0.8].map((f) => Math.floor(f * n)) : [];
  w.ents = w.rows.map((row) => arch.entities[row]);
  w.trails = w.rows.map(() => []);
}

function resetMarkerDiag() {
  state.watch.diag = { samples: 0, ageFrames: 0, ageMs: 0, rawPx: 0, extraPx: 0 };
  history.at = 0;
  history.filled = 0;
}

/** Drops one staging buffer; a map still in flight frees it when it settles. */
function retire(slot) {
  slot.dead = true;
  if (!slot.pending) {
    try {
      slot.buf.destroy();
    } catch {
      /* already gone */
    }
  }
}

/** Throws away the staging buffers (world rebuild, or a pool-size change). */
function releasePool() {
  const w = state.watch;
  for (const slot of w.pool) retire(slot);
  w.pool = [];
}

// A single staging buffer can only hold one readback at a time, so with one
// buffer a fresh sample arrives every 2-4 frames and the markers stutter.
// Rotating a few buffers means one map is always close to resolving.
function syncPool(g) {
  const w = state.watch;
  const want = Math.max(1, Math.min(4, markerTuning.pool | 0));
  if (w.pool.length === want) return;
  if (w.pool.length > want) {
    for (const slot of w.pool.splice(want)) retire(slot);
    return;
  }
  while (w.pool.length < want) {
    w.pool.push({
      buf: g.device.createBuffer({ size: 80, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
      pending: false,
      dead: false,
      seq: 0,
      simTime: 0,
      tick: 0,
      wall: 0,
    });
  }
}

/**
 * What was on screen over the last few frames, so an arriving sample can be
 * checked against the marker that was actually drawn when that sample was
 * taken. `pos` is what we drew; `raw` is where the old code would have drawn
 * it (the newest sample, unextrapolated).
 */
const HISTORY = 32;
const history = { at: 0, filled: 0, time: new Float64Array(HISTORY), pos: new Float32Array(HISTORY * 10), raw: new Float32Array(HISTORY * 10) };

function recordFrame() {
  const w = state.watch, h = history, i = h.at;
  h.time[i] = state.simTime;
  for (let k = 0; k < 5; k++) {
    h.pos[i * 10 + k * 2] = w.pos[k * 2];
    h.pos[i * 10 + k * 2 + 1] = w.pos[k * 2 + 1];
    h.raw[i * 10 + k * 2] = w.samples[k * 4];
    h.raw[i * 10 + k * 2 + 1] = w.samples[k * 4 + 1];
  }
  h.at = (i + 1) % HISTORY;
  h.filled = Math.min(h.filled + 1, HISTORY);
}

function frameAt(simTime) {
  const h = history;
  let best = -1, bestD = Infinity;
  for (let i = 0; i < h.filled; i++) {
    const d = Math.abs(h.time[i] - simTime);
    if (d < bestD) { bestD = d; best = i; }
  }
  return bestD < 0.05 ? best : -1;
}

/**
 * Takes one arriving sample as the new truth, and books how far each marker was
 * from its particle at the instant that sample was taken:
 *   rawPx   - the marker the old code drew there (newest sample, no extrapolation)
 *   extraPx - the marker we actually drew there
 * Same instant, same run, so the two numbers compare directly.
 */
function acceptSample(src, simTime, ageFrames, ageMs) {
  const w = state.watch;
  if (w.hasSample) {
    const i = frameAt(simTime);
    if (i >= 0) {
      const scale = innerHeight / 2; // world units -> CSS pixels, both axes
      let raw = 0, drawn = 0;
      for (let k = 0; k < w.rows.length; k++) {
        const nx = src[k * 4], ny = src[k * 4 + 1];
        raw += Math.hypot(nx - history.raw[i * 10 + k * 2], ny - history.raw[i * 10 + k * 2 + 1]);
        drawn += Math.hypot(nx - history.pos[i * 10 + k * 2], ny - history.pos[i * 10 + k * 2 + 1]);
      }
      const d = w.diag, m = Math.max(1, w.rows.length);
      d.samples++;
      d.ageFrames += ageFrames;
      d.ageMs += ageMs;
      d.rawPx += (raw / m) * scale;
      d.extraPx += (drawn / m) * scale;
    }
    w.ageFrames = w.ageFrames ? w.ageFrames + 0.1 * (ageFrames - w.ageFrames) : ageFrames;
    const sec = ageMs / 1000;
    w.ageSec = w.ageSec ? w.ageSec + 0.1 * (sec - w.ageSec) : sec;
  }
  w.samples.set(src);
  w.sampleTime = simTime;
  w.hasSample = true;
}

/** The 'watch' system: reads the watched entities' components. */
function sampleWatched() {
  const w = state.watch, arch = state.arch, h = state.handle, world = state.world;
  if (!arch || !h || !w.rows.length || arch.count === 0) return;
  if (h.backend !== 'gpu') {
    // CPU backend: the tables are current, so ask the world directly. Exact
    // values, this frame, so there is nothing to extrapolate (age 0).
    const inc = w.incoming;
    w.ents.forEach((e, k) => {
      inc[k * 4] = world.getField(e, Position, 'x');
      inc[k * 4 + 1] = world.getField(e, Position, 'y');
      inc[k * 4 + 2] = world.getField(e, Velocity, 'vx');
      inc[k * 4 + 3] = world.getField(e, Velocity, 'vy');
    });
    acceptSample(inc, state.simTime, 0, 0);
    w.src = 'world.getField()';
    return;
  }
  // GPU backend with readback 'none': the truth lives in GPU memory. Copy just
  // these 5 rows x 4 fields (80 bytes) back into whichever staging buffer is
  // free, so a sample lands almost every frame instead of every 2-4.
  const g = state.gpu;
  syncPool(g);
  const slot = w.pool.find((s) => !s.pending);
  w.src = 'copied back from GPU memory';
  if (!slot) return;
  const src = h.bufferFor(arch);
  if (!src) return;
  const p = arch.col(Position), v = arch.col(Velocity);
  const cols = [p.x, p.y, v.vx, v.vy];
  const enc = g.device.createCommandEncoder();
  w.rows.forEach((row, k) => cols.forEach((col, c) => enc.copyBufferToBuffer(src, col.byteOffset + row * 4, slot.buf, (k * 4 + c) * 4, 4)));
  g.device.queue.submit([enc.finish()]);
  slot.pending = true;
  slot.seq = ++w.seq;
  slot.simTime = state.simTime; // the sim state this copy belongs to
  slot.tick = state.tick;
  slot.wall = performance.now();
  slot.buf.mapAsync(GPUMapMode.READ).then(
    () => {
      slot.pending = false;
      if (slot.dead) {
        retire(slot); // the pool shrank while this copy was in flight
        return;
      }
      w.incoming.set(new Float32Array(slot.buf.getMappedRange()));
      slot.buf.unmap();
      if (slot.seq > w.lastSeq) {
        w.lastSeq = slot.seq;
        acceptSample(w.incoming, slot.simTime, state.tick - slot.tick, performance.now() - slot.wall);
      }
    },
    () => {
      slot.pending = false;
      if (slot.dead) retire(slot);
    },
  );
}

/**
 * Where to draw each marker this frame. The newest sample is 1-2 frames old on
 * the GPU, so step it forward by its own sampled velocity over the measured
 * age: exactly what the kernel does (p += v * dt), which is why the markers sit
 * on their particles instead of trailing them.
 */
function advanceMarkers() {
  const w = state.watch;
  if (!w.hasSample) return;
  // Trust the extrapolation for a few sample ages, never further: if readbacks
  // stall, a marker should stop rather than fly off on a stale velocity.
  const cap = Math.min(0.25, Math.max(0.05, 3 * w.ageSec));
  const age = markerTuning.extrapolate ? Math.min(Math.max(state.simTime - w.sampleTime, 0), cap) : 0;
  const ax = aspect();
  for (let k = 0; k < w.rows.length; k++) {
    let x = w.samples[k * 4] + w.samples[k * 4 + 2] * age;
    let y = w.samples[k * 4 + 1] + w.samples[k * 4 + 3] * age;
    // The kernel bounces off the same walls; clamping keeps a marker inside.
    if (x < -ax) x = -ax; else if (x > ax) x = ax;
    if (y < -1) y = -1; else if (y > 1) y = 1;
    w.pos[k * 2] = x;
    w.pos[k * 2 + 1] = y;
  }
  recordFrame();
  pushTrails();
}

function pushTrails() {
  const w = state.watch;
  // One point per frame (not one per arrival), so the trail stays smooth.
  w.trails.forEach((trail, k) => {
    trail.push(w.pos[k * 2], w.pos[k * 2 + 1]);
    if (trail.length > 240) trail.splice(0, 2);
  });
}

function drawOverlay(now) {
  const ctx = overlayCtx, W = ui.overlay.width, H = ui.overlay.height, w = state.watch;
  ctx.clearRect(0, 0, W, H);
  if (!w.trails.length || !w.hasSample) return;
  const s = W / innerWidth;
  const inv = 1 / aspect();
  const X = (x) => ((x * inv + 1) / 2) * W, Y = (y) => ((1 - y) / 2) * H;
  w.trails.forEach((trail, k) => {
    if (trail.length < 2) return;
    const color = WATCH_COLORS[k];
    // Fading trail. One point per frame is four times as many points as before,
    // so the fade is drawn as a few banded polylines instead of one stroke per
    // segment: same look, fewer canvas calls than the old code made.
    const pts = trail.length >> 1, bands = 12;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2 * s;
    for (let b = 0; b < bands; b++) {
      const i0 = Math.floor((b * (pts - 1)) / bands), i1 = Math.floor(((b + 1) * (pts - 1)) / bands);
      if (i1 <= i0) continue;
      ctx.globalAlpha = ((b + 1) / bands) * 0.9;
      ctx.beginPath();
      ctx.moveTo(X(trail[i0 * 2]), Y(trail[i0 * 2 + 1]));
      for (let i = i0 + 1; i <= i1; i++) ctx.lineTo(X(trail[i * 2]), Y(trail[i * 2 + 1]));
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    // The ring, dot and label sit on this frame's extrapolated position.
    const x = X(w.pos[k * 2]), y = Y(w.pos[k * 2 + 1]);
    // Pulsing ring.
    const r = (7 + 2.5 * Math.sin(now / 180 + k)) * s;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2 * s;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    ctx.arc(x, y, 2 * s, 0, Math.PI * 2);
    ctx.fill();
    // Label pill with the entity's index.
    const label = '#' + fmt(w.ents[k] & 0xfffff);
    ctx.font = `600 ${11 * s}px ui-monospace, SFMono-Regular, Menlo, monospace`;
    const tw = ctx.measureText(label).width;
    const lx = x + 12 * s, ly = y - 12 * s;
    ctx.fillStyle = 'rgba(13,17,23,0.85)';
    ctx.strokeStyle = color;
    ctx.lineWidth = 1 * s;
    ctx.beginPath();
    ctx.roundRect(lx, ly - 13 * s, tw + 12 * s, 18 * s, 5 * s);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = '#f0f6fc';
    ctx.fillText(label, lx + 6 * s, ly);
  });
}

function setSystem(name, ms, desc) {
  $('t-' + name).textContent = ms < 1 ? ms.toFixed(3) + ' ms' : ms.toFixed(2) + ' ms';
  const bar = $('b-' + name);
  bar.style.width = Math.min(100, (ms / 16.7) * 100).toFixed(1) + '%';
  bar.style.background = ms > 8 ? 'var(--orange)' : 'var(--blue)';
  $('d-' + name).textContent = desc;
}

/**
 * world.memory(): a diagnostic snapshot of what the world reserves. It
 * allocates, so the panel asks for it 4x a second, never per frame. Older
 * builds of the library do not have it; then we estimate from the one
 * archetype this demo owns and say so.
 */
function memorySnapshot() {
  const world = state.world, arch = state.arch;
  if (!world || !arch) return null;
  if (typeof world.memory === 'function') {
    const m = world.memory();
    return { ...m, estimated: false };
  }
  const reserved = arch.buffer.byteLength;
  return {
    entities: arch.count,
    tables: { used: arch.count * arch.rowBytes, reserved },
    entityIndex: 0,
    strings: { count: 0 },
    total: reserved,
    archetypes: [],
    estimated: true,
  };
}

/** "11.4 / 22.0 MB": two byte counts in one unit, so they compare at a glance. */
function pair(used, reserved) {
  const big = Math.max(used, reserved);
  const [div, unit] = big >= 1048576 ? [1048576, 'MB'] : [1024, 'KB'];
  const f = (b) => (b === 0 ? '0' : big / div >= 100 ? String(Math.round(b / div)) : (b / div).toFixed(1));
  return `${f(used)} / ${f(reserved)} ${unit}`;
}

function updatePanel() {
  const h = state.handle, w = state.watch, n = state.arch ? state.arch.count : 0;
  if (!h || !state.world || !state.arch) return;
  const gpu = h.backend === 'gpu';
  const count = n >= 1e6 ? n / 1e6 + 'M' : fmt(n);
  ui.tick.textContent = 'tick ' + fmt(state.tick);

  // Entity memory: every component value of every entity lives in the archetype's one ArrayBuffer.
  const m = memorySnapshot();
  state.mem = m;
  // world.memory() sums every table in the world; `tableBytes` is this one.
  const used = m.tables.used, reserved = m.tables.reserved, slack = reserved - used;
  const tableBytes = state.arch.buffer.byteLength;
  ui.mem.textContent = pair(used, reserved);
  ui.bpe.textContent = n ? (tableBytes / n).toFixed(1) + ' B' : '–';
  // Other libraries at their measured bytes/entity (benchmarks/RESULTS.md, 100k-entity memory test), scaled to n.
  const other = (bpe) => mb(bpe * n);
  ui.memNote.innerHTML = n === 0
    ? `The world is empty: <strong>0 B</strong> in tables. <code>world.clear()</code> dropped every row and <code>world.compact()</code> deflated the table to a zero-length buffer — the archetype, its id and its query membership all survived. The ${mb(m.entityIndex)} entity index stays: the allocator never shrinks, so old handles keep reading as dead.`
    : `All ${count} entities' components live in <strong>one ${mb(tableBytes)} ArrayBuffer</strong>` +
      (gpu ? ', mirrored in GPU memory.' : '.') +
      (m.estimated ? ' The table holds' : ' <code>world.memory()</code> reports') +
      ` <strong>${mb(used)} used</strong> of ${mb(reserved)} reserved` +
      (slack > 0 ? `, so <code>world.compact()</code> could hand back ${mb(slack)}` : ' — no slack to reclaim') +
      (m.entityIndex ? `, plus ${mb(m.entityIndex)} for the entity index` : '') +
      (m.estimated ? ' (measured off the archetype: this build predates <code>world.memory()</code>)' : '') +
      `. At their measured bytes per entity, the same data would take about ${other(302.1)} in bitecs and ${other(978.7)} in ecsy.`;
  ui.reclaimNote.innerHTML = state.reclaimNote;
  setSystem('particles', state.times.particles, gpu
    ? `kernelSystem → WGSL compute shader, ${count} entities on the GPU`
    : `kernelSystem → compiled JS loop, ${count} entities on the CPU`);
  setSystem('watch', state.times.watch, gpu
    ? `copies 5 entities (80 bytes) back from the GPU through ${w.pool.length} rotating staging buffers`
    : 'world.getField() on 5 entities');
  setSystem('render', state.times.render, !state.gpu ? 'Canvas 2D pixels' : gpu
    ? `draws ${count} instances straight from bufferFor()`
    : `uploads 4 columns, then draws ${count} instances`);
  ui.watchsrc.textContent = gpu
    ? `${w.src} · ${w.ageFrames.toFixed(1)} frames old${markerTuning.extrapolate ? ', extrapolated' : ''}`
    : `${w.src} · exact`;
  ui.watch.innerHTML = w.ents
    .map((e, k) => {
      const x = w.samples[k * 4], y = w.samples[k * 4 + 1];
      const sp = Math.hypot(w.samples[k * 4 + 2], w.samples[k * 4 + 3]);
      return `<tr><td><span style="--c:${WATCH_COLORS[k]}"></span>#${fmt(e & 0xfffff)} <small>g${e >>> 20}</small></td>` +
        `<td>${x.toFixed(3)}</td><td>${y.toFixed(3)}</td><td>${sp.toFixed(3)}</td></tr>`;
    })
    .join('');
}

// Tabs
for (const tab of document.querySelectorAll('.tabs button')) {
  tab.addEventListener('click', () => {
    for (const t of document.querySelectorAll('.tabs button')) t.setAttribute('aria-selected', String(t === tab));
    $('tab-world').hidden = tab.dataset.tab !== 'world';
    $('tab-kernel').hidden = tab.dataset.tab !== 'kernel';
  });
}

// Console hook for measuring without requestAnimationFrame (e.g. a background tab):
//   await cozyDemo.measure({ backend: 'cpu', count: 1e6, frames: 120 })
// It waits for the GPU to finish, so 'gpu' numbers are real GPU time, not just submission.
window.cozyDemo = {
  state,
  // Marker latency knobs and measurements. To see the old behaviour:
  //   cozyDemo.markerTuning.pool = 1; cozyDemo.markerTuning.extrapolate = false;
  //   cozyDemo.resetMarkerStats(); await new Promise(r => setTimeout(r, 5000));
  //   cozyDemo.markerStats()
  markerTuning,
  resetMarkerStats: resetMarkerDiag,
  /**
   * ageFrames/ageMs: how old a sample is when it lands.
   * rawOffsetPx: how far the marker would be from its particle, in CSS pixels,
   *   if it were drawn at the raw sample (the old behaviour).
   * offsetPx: the same distance for what is actually drawn now.
   * Both are measured at the instant a new sample arrives, so they compare directly.
   */
  markerStats() {
    const d = state.watch.diag, s = d.samples;
    if (!s) return { samples: 0 };
    return {
      samples: s,
      ageFrames: +(d.ageFrames / s).toFixed(2),
      ageMs: +(d.ageMs / s).toFixed(2),
      rawOffsetPx: +(d.rawPx / s).toFixed(2),
      offsetPx: +(d.extraPx / s).toFixed(2),
      pool: state.watch.pool.length,
      extrapolate: markerTuning.extrapolate,
    };
  },
  memory: () => memorySnapshot(),
  reclaim,
  rebuild: build,
  /**
   * Proof that a rebuild hands the old world back. Drives `cycles` rebuilds,
   * rotating the backend and the particle count so both renderer paths run (the
   * kernel's own storage buffer on the GPU, the uploaded copy on the CPU), and
   * steps a few frames each time so the kernel really dispatches and the
   * renderer really builds a bind group over those buffers.
   *
   * `collected` holds the verdict of three WeakRefs on the FIRST world, its
   * archetype and its table ArrayBuffer: all three must be true. `heap` comes
   * from performance.measureUserAgentSpecificMemory() when the page is
   * cross-origin-isolated -- it counts ArrayBuffers and collects before it
   * measures -- and otherwise from performance.memory, which does NOT see
   * ArrayBuffer bytes; each reading says which one it is.
   */
  async leakCheck({ cycles = 16, counts = [10_000, 100_000, 250_000], backends, frames = 4 } = {}) {
    const be = backends || (state.gpu ? ['gpu', 'cpu'] : ['cpu']);
    // measureUserAgentSpecificMemory is the only browser reading that counts
    // ArrayBuffer bytes, but it needs cross-origin isolation, is refused in some
    // embeddings, and is allowed to wait for the next GC -- so it is raced and
    // caught, and performance.memory (JS objects only) is the stated fallback.
    const heap = async () => {
      if (crossOriginIsolated && performance.measureUserAgentSpecificMemory) {
        try {
          const r = await Promise.race([performance.measureUserAgentSpecificMemory(), sleep(20000)]);
          if (r) return { bytes: r.bytes, source: 'measureUserAgentSpecificMemory (ArrayBuffers included)' };
        } catch {
          /* refused here; fall through */
        }
      }
      return {
        bytes: performance.memory ? performance.memory.usedJSHeapSize : -1,
        source: 'performance.memory (ArrayBuffers NOT included)',
      };
    };
    // A WeakRef only reads as collected after a collection has run. Chrome
    // started with --js-flags=--expose-gc gives one directly; otherwise the
    // measurement above collects first; with neither, say the verdict is weak.
    const collect = async () => {
      if (typeof gc === 'function') {
        for (let i = 0; i < 3; i++) {
          gc();
          await sleep(0);
        }
        return 'gc()';
      }
      return 'measurement only';
    };
    const step = async () => {
      for (let i = 0; i < frames; i++) {
        state.simTime += 1 / 60;
        state.world.update(1 / 60);
        state.world.update(1 / 60, 'render');
        state.tick++;
      }
      if (state.gpu) await state.gpu.device.queue.onSubmittedWorkDone();
    };
    await step();
    const first = [new WeakRef(state.world), new WeakRef(state.arch), new WeakRef(state.arch.buffer)];
    const before = { count: state.arch.count, worldTotal: memorySnapshot().total, heap: await heap() };
    const series = [];
    for (let i = 0; i < cycles; i++) {
      state.backend = be[i % be.length];
      state.count = counts[i % counts.length];
      setPressed(ui.backend, state.backend);
      setPressed(ui.count, state.count);
      await build();
      await step();
      series.push({ cycle: i + 1, backend: state.handle.backend, count: state.arch.count, worldTotal: memorySnapshot().total });
    }
    const after = { count: state.arch.count, worldTotal: memorySnapshot().total, heap: await heap() };
    const collectedBy = await collect();
    const [world, arch, buffer] = first.map((r) => r.deref() === undefined);
    return {
      cycles,
      disposed: state.disposed,
      before,
      after,
      collected: { world, archetype: arch, tableBuffer: buffer, by: collectedBy },
      series,
    };
  },
  async measure({ backend = state.backend, count = state.count, frames = 120 } = {}) {
    state.backend = backend;
    state.count = count;
    setPressed(ui.backend, backend);
    setPressed(ui.count, count);
    await build();
    const step = () => {
      state.simTime += 1 / 60;
      state.world.update(1 / 60);
      state.world.update(1 / 60, 'render');
      state.tick++;
    };
    const settle = () => (state.gpu ? state.gpu.device.queue.onSubmittedWorkDone() : Promise.resolve());
    for (let i = 0; i < 10; i++) step();
    await settle();
    let main = 0;
    const t0 = performance.now();
    for (let i = 0; i < frames; i++) {
      const a = performance.now();
      step();
      main += performance.now() - a;
    }
    await settle();
    const total = performance.now() - t0;
    return { backend: state.handle.backend, count, msPerFrame: +(total / frames).toFixed(3), mainThreadMs: +(main / frames).toFixed(3) };
  },
};

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

(async () => {
  // A browser can advertise navigator.gpu and then never settle requestAdapter()
  // (headless Chrome with no adapter does exactly that), which used to hang the
  // boot here for good. Bound the wait and fall back to the CPU path instead.
  const gpuSetup = 'gpu' in navigator
    ? Promise.race([initGPU().catch(() => null), sleep(6000).then(() => null)])
    : Promise.resolve(null);
  state.gpu = await gpuSetup;
  if (!state.gpu) {
    state.backend = 'cpu';
    state.count = 100_000;
    state.ctx2d = canvas.getContext('2d');
    for (const b of ui.backend.querySelectorAll('button')) if (b.dataset.v === 'gpu') b.disabled = true;
    for (const b of ui.count.querySelectorAll('button')) if (Number(b.dataset.v) > 100_000) b.disabled = true;
    banner('WebGPU is not available in this browser, so this runs on the CPU with fewer particles. Try a recent Chrome, Edge or Safari for the GPU version.');
  }
  setPressed(ui.backend, state.backend);
  setPressed(ui.count, state.count);
  resize();
  await build();
  requestAnimationFrame(frame);
})();
