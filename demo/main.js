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
  mem: $('mem'), bpe: $('bpe'), memNote: $('memNote'),
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
  times: { particles: 0, watch: 0, render: 0 },
  // A few entities we follow on screen and in the panel.
  watch: { rows: [], ents: [], samples: new Float32Array(20), trails: [], busy: false, staging: null, src: '' },
};

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

async function build() {
  if (state.building) return;
  state.building = true;
  try {
    if (state.handle) state.handle.destroy();
    state.handle = null;
    const n = state.count;
    const world = new World({ initialCapacity: n });
    const arch = world.archetype(Position, Velocity);
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
      ema('watch', performance.now() - t);
    });
    world.system('render', { group: 'render' }, () => {
      const t = performance.now();
      if (state.gpu) renderGPU();
      else render2D();
      ema('render', performance.now() - t);
    });

    const w = state.watch;
    w.rows = [0, 0.2, 0.4, 0.6, 0.8].map((f) => Math.floor(f * n));
    w.ents = w.rows.map((row) => arch.entities[row]);
    w.trails = w.rows.map(() => []);
    w.busy = false;

    state.world = world;
    state.arch = arch;
    state.handle = handle;
    state.tick = 0;
    state.times = { particles: 0, watch: 0, render: 0 };
    drawLayout(arch);
    ui.n.textContent = n >= 1e6 ? n / 1e6 + 'M' : n / 1e3 + 'k';
    updateAutoNote();
  } finally {
    state.building = false;
  }
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

function renderGPU() {
  const g = state.gpu, arch = state.arch, handle = state.handle;
  if (!arch || !handle) return;
  const n = arch.count;
  let source;
  if (handle.backend === 'gpu') {
    source = handle.bufferFor(arch); // the kernel's own storage buffer
    if (!source) return; // before the first dispatch
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
  if (!b || b.disabled || b.dataset.v === state.backend || state.building) return;
  state.backend = b.dataset.v;
  setPressed(ui.backend, state.backend);
  await build();
  resetStats();
});
ui.count.addEventListener('click', async (e) => {
  const b = e.target.closest('button');
  if (!b || b.disabled || Number(b.dataset.v) === state.count || state.building) return;
  state.count = Number(b.dataset.v);
  setPressed(ui.count, state.count);
  await build();
  resetStats();
});
ui.reset.addEventListener('click', async () => {
  await build();
  resetStats();
});

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
  if (!handle || state.building) return;

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

/** The 'watch' system: reads the watched entities' components. */
function sampleWatched() {
  const w = state.watch, arch = state.arch, h = state.handle, world = state.world;
  if (!arch || !h || !w.rows.length) return;
  if (h.backend !== 'gpu') {
    // CPU backend: the tables are current, so ask the world directly.
    w.ents.forEach((e, k) => {
      w.samples[k * 4] = world.getField(e, Position, 'x');
      w.samples[k * 4 + 1] = world.getField(e, Position, 'y');
      w.samples[k * 4 + 2] = world.getField(e, Velocity, 'vx');
      w.samples[k * 4 + 3] = world.getField(e, Velocity, 'vy');
    });
    w.src = 'world.getField()';
    pushTrails();
    return;
  }
  // GPU backend with readback 'none': the truth lives in GPU memory. Copy just
  // these 5 rows x 4 fields (80 bytes) back, one small async read per frame.
  if (w.busy) return;
  const src = h.bufferFor(arch);
  if (!src) return;
  const g = state.gpu;
  if (!w.staging) w.staging = g.device.createBuffer({ size: 80, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const p = arch.col(Position), v = arch.col(Velocity);
  const cols = [p.x, p.y, v.vx, v.vy];
  const enc = g.device.createCommandEncoder();
  w.rows.forEach((row, k) => cols.forEach((col, c) => enc.copyBufferToBuffer(src, col.byteOffset + row * 4, w.staging, (k * 4 + c) * 4, 4)));
  g.device.queue.submit([enc.finish()]);
  w.busy = true;
  const staging = w.staging;
  staging.mapAsync(GPUMapMode.READ).then(
    () => {
      w.samples.set(new Float32Array(staging.getMappedRange()));
      staging.unmap();
      w.busy = false;
      pushTrails();
    },
    () => (w.busy = false),
  );
  w.src = 'copied back from GPU memory';
}

function pushTrails() {
  const w = state.watch;
  w.trails.forEach((trail, k) => {
    trail.push(w.samples[k * 4], w.samples[k * 4 + 1]);
    if (trail.length > 96) trail.splice(0, 2);
  });
}

function drawOverlay(now) {
  const ctx = overlayCtx, W = ui.overlay.width, H = ui.overlay.height, w = state.watch;
  ctx.clearRect(0, 0, W, H);
  if (!w.trails.length) return;
  const s = W / innerWidth;
  const inv = 1 / aspect();
  const X = (x) => ((x * inv + 1) / 2) * W, Y = (y) => ((1 - y) / 2) * H;
  w.trails.forEach((trail, k) => {
    if (trail.length < 2) return;
    const color = WATCH_COLORS[k];
    // Fading trail.
    for (let i = 2; i < trail.length; i += 2) {
      ctx.globalAlpha = (i / trail.length) * 0.9;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2 * s;
      ctx.beginPath();
      ctx.moveTo(X(trail[i - 2]), Y(trail[i - 1]));
      ctx.lineTo(X(trail[i]), Y(trail[i + 1]));
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    const x = X(trail[trail.length - 2]), y = Y(trail[trail.length - 1]);
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

function updatePanel() {
  const h = state.handle, w = state.watch, n = state.arch ? state.arch.count : 0;
  if (!h) return;
  const gpu = h.backend === 'gpu';
  const count = n >= 1e6 ? n / 1e6 + 'M' : fmt(n);
  ui.tick.textContent = 'tick ' + fmt(state.tick);

  // Entity memory: every component value of every entity lives in the archetype's one ArrayBuffer.
  const bytes = state.arch.buffer.byteLength;
  ui.mem.textContent = mb(bytes);
  ui.bpe.textContent = (bytes / n).toFixed(1) + ' B';
  // Other libraries at their measured bytes/entity (benchmarks/RESULTS.md, 100k-entity memory test), scaled to n.
  const other = (bpe) => mb(bpe * n);
  ui.memNote.innerHTML =
    `All ${count} entities' components live in <strong>one ${mb(bytes)} ArrayBuffer</strong>` +
    (gpu ? ', mirrored in GPU memory.' : '.') +
    ` At their measured bytes per entity, the same data would take about ${other(302.1)} in bitecs and ${other(978.7)} in ecsy.`;
  setSystem('particles', state.times.particles, gpu
    ? `kernelSystem → WGSL compute shader, ${count} entities on the GPU`
    : `kernelSystem → compiled JS loop, ${count} entities on the CPU`);
  setSystem('watch', state.times.watch, gpu ? 'copies 5 entities (80 bytes) back from the GPU' : 'world.getField() on 5 entities');
  setSystem('render', state.times.render, !state.gpu ? 'Canvas 2D pixels' : gpu
    ? `draws ${count} instances straight from bufferFor()`
    : `uploads 4 columns, then draws ${count} instances`);
  ui.watchsrc.textContent = w.src;
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
  async measure({ backend = state.backend, count = state.count, frames = 120 } = {}) {
    state.backend = backend;
    state.count = count;
    setPressed(ui.backend, backend);
    setPressed(ui.count, count);
    await build();
    const step = () => {
      state.world.update(1 / 60);
      state.world.update(1 / 60, 'render');
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
  state.gpu = 'gpu' in navigator ? await initGPU().catch(() => null) : null;
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
