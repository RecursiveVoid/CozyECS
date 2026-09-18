// Generates the README graphics as static SVGs, one light and one dark file each.
//
//   node docs/assets/generate.mjs
//
// The numbers are copied from benchmarks/RESULTS.md (throughput and memory: the
// audited 5-repeat run; GPU: the 2026-09-18 confirmation run, gravity kernel).
// Re-run the benchmarks, update DATA below, then regenerate.
//
// Colors: the dataviz reference palette, slots 1-4 in fixed order (validated
// for colorblind separation in both modes). Two light-mode slots are below 3:1
// against the surface, so every mark carries a direct text label.

import { writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const OUT = (name) => fileURLToPath(new URL(name, import.meta.url));

const DATA = {
  scenarios: [
    // ops/sec, median of 5 isolated runs. cozyecs = idiomatic function systems.
    // paired = median over the 5 repeats of (cozyecs / fastest competitor in that repeat).
    { key: 'packed_5', label: 'packed_5', note: '5 systems, 1k entities', paired: 1.10, cozy: 487645, harmony: 430455, wolf: 379934, bitecs: 349240 },
    { key: 'simple_iter', label: 'simple_iter', note: '4 archetypes, P += V', paired: 1.20, cozy: 415145, harmony: 272589, wolf: 323361, bitecs: 346897 },
    { key: 'frag_iter', label: 'frag_iter', note: '26 archetypes', paired: 1.13, cozy: 912828, harmony: 822210, wolf: 715302, bitecs: 727961 },
    { key: 'entity_cycle', label: 'entity_cycle', note: 'spawn + destroy 1k', paired: 3.41, cozy: 58191, harmony: 8274, wolf: 16387, bitecs: 3227 },
    { key: 'add_remove', label: 'add_remove', note: 'add + remove a component', paired: 1.28, cozy: 40704, harmony: 5805, wolf: 31798, bitecs: 5106 },
  ],
  memory: [
    // bytes/entity, 100k entities with Position{x,y} + Velocity{dx,dy} + one query
    ['CozyECS', 26.2],
    ['harmony-ecs', 54.2],
    ['wolf-ecs', 72.4],
    ['becsy', 170.0],
    ['bitecs 0.4', 253.0],
    ['bitecs 0.3', 302.1],
    ['geotic', 398.9],
    ['ecsy', 978.7],
  ],
  gpu: {
    // gravity kernel, ms/frame, Apple M4 via Dawn (Node), median of 5
    sizes: [1000, 10000, 30000, 50000, 100000, 300000, 1000000],
    cpu: [0.0025, 0.025, 0.075, 0.128, 0.27, 0.808, 2.69],
    none: [0.056, 0.056, 0.056, 0.057, 0.057, 0.127, 0.374],
    async: [0.078, 0.079, 0.083, 0.09, 0.105, 0.269, 1.46],
    sync: [0.354, 0.37, 0.406, 0.41, 0.475, 1.01, 2.37],
    breakEvenNone: 22376,
  },
};

const THEMES = {
  light: {
    surface: '#ffffff', border: '#d0d7de', text: '#1f2328', text2: '#59636e', muted: '#818b98',
    grid: '#eaeef2', axis: '#afb8c1', neutral: '#afb8c1', panel: '#f6f8fa',
    s: ['#2a78d6', '#eb6834', '#1baf7a', '#eda100'],
  },
  dark: {
    surface: '#0d1117', border: '#30363d', text: '#f0f6fc', text2: '#9198a1', muted: '#6e7681',
    grid: '#21262d', axis: '#3d444d', neutral: '#4b535d', panel: '#161b22',
    s: ['#3987e5', '#d95926', '#199e70', '#c98500'],
  },
};

const FONT = `-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif`;
const MONO = `ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, monospace`;

const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const fmt = (n) => n.toLocaleString('en-US');

function svg(w, h, t, title, body) {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-labelledby="t">
<title id="t">${esc(title)}</title>
<rect x="0.5" y="0.5" width="${w - 1}" height="${h - 1}" rx="12" fill="${t.surface}" stroke="${t.border}"/>
<g font-family="${FONT}">
${body}
</g>
</svg>
`;
}

const text = (x, y, s, { size = 13, fill, weight = 400, anchor = 'start', family, extra = '' } = {}) =>
  `<text x="${x}" y="${y}" font-size="${size}" font-weight="${weight}" fill="${fill}" text-anchor="${anchor}"${family ? ` font-family="${family}"` : ''}${extra}>${esc(s)}</text>`;

/** Vertical bar with only the data end rounded (4px), anchored square to the baseline. */
function vbar(x, yTop, w, yBase, fill) {
  const h = yBase - yTop;
  if (h <= 0) return '';
  const r = Math.min(4, h, w / 2);
  return `<path d="M${x},${yBase} V${yTop + r} Q${x},${yTop} ${x + r},${yTop} H${x + w - r} Q${x + w},${yTop} ${x + w},${yTop + r} V${yBase} Z" fill="${fill}"/>`;
}

/** Horizontal bar, data end (right) rounded. */
function hbar(xBase, y, xEnd, h, fill) {
  const w = xEnd - xBase;
  if (w <= 0) return '';
  const r = Math.min(4, w, h / 2);
  return `<path d="M${xBase},${y} H${xEnd - r} Q${xEnd},${y} ${xEnd},${y + r} V${y + h - r} Q${xEnd},${y + h} ${xEnd - r},${y + h} H${xBase} Z" fill="${fill}"/>`;
}

// ---------------------------------------------------------------------------
// 1. Hero banner
// ---------------------------------------------------------------------------
function hero(t, mode) {
  const W = 1200, H = 340;
  let b = '';
  // A small "archetype table" motif: rows of column cells, one highlighted run per row.
  const gx = 800, gy = 58, cw = 26, ch = 16, gap = 4;
  for (let r = 0; r < 6; r++) {
    for (let c = 0; c < 12; c++) {
      const on = c < 4 + ((r * 3) % 7);
      const fill = on ? t.s[r % 4] : t.panel;
      const op = on ? (0.55 + 0.45 * ((c % 4) / 3)).toFixed(2) : 1;
      b += `<rect x="${gx + c * (cw + gap)}" y="${gy + r * (ch + gap)}" width="${cw}" height="${ch}" rx="3" fill="${fill}" opacity="${op}"/>`;
    }
  }
  b += text(gx, gy + 6 * (ch + gap) + 14, 'one ArrayBuffer per archetype · columns as typed-array views', { size: 12, fill: t.muted });

  b += text(56, 104, 'CozyECS', { size: 64, weight: 700, fill: t.text, extra: ' letter-spacing="-1.5"' });
  b += text(58, 142, 'A tiny, archetype-based Entity Component System for JavaScript.', { size: 20, fill: t.text2 });
  b += text(58, 170, 'Typed-array storage, zero dependencies, and systems that can run on the GPU.', { size: 20, fill: t.text2 });

  const tiles = [
    ['#1', 'in all 5 benchmark scenarios', t.s[0]],
    ['26 B', 'per entity (Position + Velocity)', t.s[2]],
    ['7.2×', 'faster on the GPU at 1M entities', t.s[1]],
    ['0', 'dependencies · 12 KB gzipped core', t.s[3]],
  ];
  const tw = 262, th = 92, ty = 218;
  tiles.forEach(([big, small, color], i) => {
    const x = 56 + i * (tw + 16);
    b += `<rect x="${x}" y="${ty}" width="${tw}" height="${th}" rx="10" fill="${t.panel}" stroke="${t.border}"/>`;
    b += `<rect x="${x}" y="${ty + 18}" width="4" height="${th - 36}" rx="2" fill="${color}"/>`;
    b += text(x + 22, ty + 50, big, { size: 34, weight: 700, fill: t.text });
    b += text(x + 22, ty + 74, small, { size: 13.5, fill: t.text2 });
  });
  return svg(W, H, t, 'CozyECS: a tiny archetype ECS for JavaScript with GPU kernels', b);
}

// ---------------------------------------------------------------------------
// 2. Throughput vs the three strongest competitors
// ---------------------------------------------------------------------------
function throughput(t) {
  const W = 1200, H = 520;
  const series = [
    ['cozy', 'CozyECS', 0],
    ['harmony', 'harmony-ecs', 1],
    ['wolf', 'wolf-ecs', 2],
    ['bitecs', 'bitecs', 3],
  ];
  const L = 72, R = 32, T = 132, B = 92;
  const plotW = W - L - R, plotH = H - T - B;
  const yBase = T + plotH;
  let b = '';
  b += text(40, 48, 'Throughput relative to CozyECS', { size: 22, weight: 650, fill: t.text });
  b += text(40, 74, 'ops/sec as a share of CozyECS in each scenario (higher is better). Median of 5 isolated runs, Node 22, Apple M4.', { size: 14, fill: t.text2 });

  // legend
  let lx = 40;
  for (const [, name, si] of series) {
    b += `<rect x="${lx}" y="94" width="12" height="12" rx="3" fill="${t.s[si]}"/>`;
    b += text(lx + 18, 105, name, { size: 13.5, fill: t.text });
    lx += 18 + name.length * 7.6 + 26;
  }

  // grid: 0, 25, 50, 75, 100 %
  for (const p of [0, 25, 50, 75, 100]) {
    const y = yBase - (p / 100) * plotH;
    b += `<line x1="${L}" x2="${W - R}" y1="${y}" y2="${y}" stroke="${p === 0 ? t.axis : t.grid}" stroke-width="1"/>`;
    b += text(L - 10, y + 4, p + '%', { size: 12, fill: t.muted, anchor: 'end' });
  }

  const groups = DATA.scenarios.length;
  const gw = plotW / groups;
  const bw = 30, bgap = 2;
  DATA.scenarios.forEach((sc, gi) => {
    const cx = L + gi * gw + gw / 2;
    const total = series.length * bw + (series.length - 1) * bgap;
    let x = cx - total / 2;
    for (const [key, , si] of series) {
      const pct = (sc[key] / sc.cozy) * 100;
      const yTop = yBase - (pct / 100) * plotH;
      b += vbar(x, yTop, bw, yBase, t.s[si]);
      const label = key === 'cozy' ? fmt(Math.round(sc.cozy / 1000)) + 'k' : Math.round(pct) + '%';
      b += text(x + bw / 2, yTop - 7, label, { size: 11.5, fill: key === 'cozy' ? t.text : t.text2, anchor: 'middle', weight: key === 'cozy' ? 600 : 400 });
      x += bw + bgap;
    }
    b += text(cx, yBase + 24, sc.label, { size: 13.5, weight: 600, fill: t.text, anchor: 'middle', family: MONO });
    b += text(cx, yBase + 43, sc.note, { size: 12, fill: t.muted, anchor: 'middle' });
    b += text(cx, yBase + 64, `${sc.paired.toFixed(2)}× the best competitor`, { size: 12.5, weight: 600, fill: t.text2, anchor: 'middle' });
  });
  return svg(W, H, t, 'Throughput of CozyECS, harmony-ecs, wolf-ecs and bitecs in five benchmark scenarios', b);
}

// ---------------------------------------------------------------------------
// 3. Memory per entity
// ---------------------------------------------------------------------------
function memory(t) {
  const W = 1200, H = 434;
  const L = 170, R = 110, T = 104, rowH = 36, barH = 22;
  const max = 1000;
  const plotW = W - L - R;
  let b = '';
  b += text(40, 48, 'Memory per entity', { size: 22, weight: 650, fill: t.text });
  b += text(40, 74, 'Bytes per entity for 100,000 entities with Position{x,y} + Velocity{dx,dy} and one query (lower is better).', { size: 14, fill: t.text2 });
  for (const v of [0, 250, 500, 750, 1000]) {
    const x = L + (v / max) * plotW;
    b += `<line x1="${x}" x2="${x}" y1="${T - 8}" y2="${T + DATA.memory.length * rowH - 6}" stroke="${v === 0 ? t.axis : t.grid}"/>`;
    b += text(x, T + DATA.memory.length * rowH + 12, v === 0 ? '0' : v + ' B', { size: 12, fill: t.muted, anchor: 'middle' });
  }
  DATA.memory.forEach(([name, bytes], i) => {
    const y = T + i * rowH;
    const isCozy = i === 0;
    b += text(L - 14, y + barH / 2 + 5, name, { size: 14, weight: isCozy ? 650 : 400, fill: isCozy ? t.text : t.text2, anchor: 'end' });
    const xEnd = L + (bytes / max) * plotW;
    b += hbar(L, y, Math.max(L + 3, xEnd), barH, isCozy ? t.s[0] : t.neutral);
    const ratio = isCozy ? '' : `  ·  ${(bytes / DATA.memory[0][1]).toFixed(1)}×`;
    b += text(Math.max(L + 3, xEnd) + 10, y + barH / 2 + 5, `${bytes.toFixed(1)} B${ratio}`, { size: 13, weight: isCozy ? 650 : 400, fill: isCozy ? t.text : t.text2 });
  });
  return svg(W, H, t, 'Memory per entity: CozyECS 26.2 bytes versus 54 to 979 bytes for other libraries', b);
}

// ---------------------------------------------------------------------------
// 4. GPU scaling (log-log)
// ---------------------------------------------------------------------------
function gpu(t) {
  const W = 1200, H = 560;
  const L = 88, R = 190, T = 112, B = 78;
  const plotW = W - L - R, plotH = H - T - B;
  const xMin = Math.log10(1000), xMax = Math.log10(1_000_000);
  const yMin = Math.log10(0.002), yMax = Math.log10(4);
  const X = (n) => L + ((Math.log10(n) - xMin) / (xMax - xMin)) * plotW;
  const Y = (ms) => T + plotH - ((Math.log10(ms) - yMin) / (yMax - yMin)) * plotH;
  const g = DATA.gpu;
  let b = '';
  b += text(40, 48, 'The same kernel on CPU and GPU', { size: 22, weight: 650, fill: t.text });
  b += text(40, 74, 'Milliseconds per frame for a gravity + bounce kernel (lower is better). Both axes are logarithmic. Apple M4, WebGPU via Dawn.', { size: 14, fill: t.text2 });

  for (const ms of [0.01, 0.1, 1]) {
    const y = Y(ms);
    b += `<line x1="${L}" x2="${L + plotW}" y1="${y}" y2="${y}" stroke="${t.grid}"/>`;
    b += text(L - 10, y + 4, ms + ' ms', { size: 12, fill: t.muted, anchor: 'end' });
  }
  for (const n of [1000, 10000, 100000, 1000000]) {
    const x = X(n);
    b += `<line x1="${x}" x2="${x}" y1="${T}" y2="${T + plotH}" stroke="${t.grid}"/>`;
    b += text(x, T + plotH + 22, n >= 1e6 ? '1M' : n / 1000 + 'k', { size: 12, fill: t.muted, anchor: 'middle' });
  }
  b += text(L + plotW / 2, T + plotH + 50, 'entities', { size: 13, fill: t.text2, anchor: 'middle' });
  b += `<line x1="${L}" x2="${L + plotW}" y1="${T + plotH}" y2="${T + plotH}" stroke="${t.axis}"/>`;

  // break-even marker for 'none'
  const bx = X(g.breakEvenNone);
  b += `<line x1="${bx}" x2="${bx}" y1="${T + 6}" y2="${T + plotH}" stroke="${t.muted}" stroke-dasharray="4 4"/>`;
  b += text(bx + 8, T + 18, `GPU wins above ~${Math.round(g.breakEvenNone / 1000)}k entities`, { size: 12.5, fill: t.text2 });

  const lines = [
    ['cpu', 'CPU (compiled kernel)', 0],
    ['none', "GPU, readback 'none'", 1],
    ['async', "GPU, readback 'async'", 2],
    ['sync', "GPU, readback 'sync-frame'", 3],
  ];
  // Direct labels at the line ends, nudged apart so they never overlap.
  const ends = lines.map(([key, name, si]) => ({ key, name, si, y: Y(g[key][g[key].length - 1]) })).sort((a, c) => a.y - c.y);
  for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 18) ends[i].y = ends[i - 1].y + 18;
  for (const [key, , si] of lines) {
    const pts = g.sizes.map((n, i) => `${X(n).toFixed(1)},${Y(g[key][i]).toFixed(1)}`).join(' ');
    b += `<polyline points="${pts}" fill="none" stroke="${t.s[si]}" stroke-width="2.25" stroke-linejoin="round" stroke-linecap="round"/>`;
    g.sizes.forEach((n, i) => {
      b += `<circle cx="${X(n).toFixed(1)}" cy="${Y(g[key][i]).toFixed(1)}" r="4" fill="${t.s[si]}" stroke="${t.surface}" stroke-width="2"/>`;
    });
  }
  for (const e of ends) {
    b += text(L + plotW + 12, e.y + 4, e.name, { size: 12.5, fill: t.text });
    if (e.key === 'none') {
      // headline callout at 1M, under the 'none' label
      b += text(L + plotW + 12, e.y + 24, `${g.none[6]} vs ${g.cpu[6]} ms at 1M`, { size: 12, weight: 650, fill: t.text });
      b += text(L + plotW + 12, e.y + 40, `${(g.cpu[6] / g.none[6]).toFixed(1)}× faster than CPU`, { size: 12, fill: t.text2 });
    }
  }
  return svg(W, H, t, 'Milliseconds per frame for the same kernel on CPU and on GPU, from 1,000 to 1,000,000 entities', b);
}

// ---------------------------------------------------------------------------
// 5. How it works
// ---------------------------------------------------------------------------
function howItWorks(t) {
  const W = 1200, H = 446;
  let b = '';
  // Left: storage
  b += text(40, 48, 'Storage: one table per archetype', { size: 19, weight: 650, fill: t.text });
  b += text(40, 72, 'Entities with the same components share a table.', { size: 13.5, fill: t.text2 });
  b += text(40, 90, 'Each field is a typed-array view over one ArrayBuffer,', { size: 13.5, fill: t.text2 });
  b += text(40, 108, 'so a system loops over flat, packed memory.', { size: 13.5, fill: t.text2 });
  const tables = [
    { name: 'Position + Velocity', cols: [['x', 0], ['y', 0], ['dx', 1], ['dy', 1]], rows: 5 },
    { name: 'Position + Velocity + Player', cols: [['x', 0], ['y', 0], ['dx', 1], ['dy', 1], ['hp', 2]], rows: 3 },
  ];
  let ty = 140;
  for (const tb of tables) {
    b += text(40, ty, tb.name, { size: 13, weight: 600, fill: t.text, family: MONO });
    const cw = 64, rh = 20, x0 = 40, y0 = ty + 10;
    b += `<rect x="${x0 - 6}" y="${y0 - 4}" width="${tb.cols.length * (cw + 4) + 8}" height="${(tb.rows + 1) * (rh + 3) + 6}" rx="8" fill="${t.panel}" stroke="${t.border}"/>`;
    tb.cols.forEach(([name, si], c) => {
      const x = x0 + c * (cw + 4);
      b += text(x + cw / 2, y0 + 14, name, { size: 12, fill: t.text2, anchor: 'middle', family: MONO });
      for (let r = 0; r < tb.rows; r++) {
        b += `<rect x="${x}" y="${y0 + (r + 1) * (rh + 3)}" width="${cw}" height="${rh}" rx="3" fill="${t.s[si]}" opacity="${(0.5 + 0.1 * r).toFixed(2)}"/>`;
      }
    });
    ty = y0 + (tb.rows + 1) * (rh + 3) + 44;
  }

  // Right: kernel pipeline
  const px = 470;
  b += text(px, 48, 'GPU kernels: write JavaScript, run WGSL', { size: 19, weight: 650, fill: t.text });
  b += text(px, 72, 'kernelSystem() reads your function, checks it against a small safe subset,', { size: 13.5, fill: t.text2 });
  b += text(px, 90, 'and compiles it twice. auto picks the faster backend by entity count.', { size: 13.5, fill: t.text2 });
  const box = (x, y, w, h, title, sub, accent, code = false) => {
    let s = `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="10" fill="${t.panel}" stroke="${t.border}"/>`;
    if (accent) s += `<rect x="${x}" y="${y + 14}" width="4" height="${h - 28}" rx="2" fill="${accent}"/>`;
    s += text(x + 18, y + 27, title, { size: 14, weight: 650, fill: t.text });
    sub.forEach((line, i) => (s += text(x + 18, y + 48 + i * 18, line, { size: 12, fill: t.text2, family: code ? MONO : undefined, extra: code ? ' xml:space="preserve"' : '' })));
    return s;
  };
  const arrow = (x1, y1, x2, y2) =>
    `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${t.muted}" stroke-width="1.5"/>` +
    `<path d="M${x2},${y2} l-7,-4 v8 z" fill="${t.muted}" transform="rotate(${(Math.atan2(y2 - y1, x2 - x1) * 180) / Math.PI} ${x2} ${y2})"/>`;
  b += box(px, 122, 210, 88, 'Your kernel', ['(p, v, dt, u) => {', '  p.x += v.dx * dt }'], t.s[0], true);
  b += box(px + 250, 122, 190, 88, 'Parser', ['no dependencies', 'minified code OK'], null);
  b += box(px + 480, 122, 210, 88, 'Kernel IR', ['typed, validated', 'clear errors w/ source'], null);
  b += arrow(px + 212, 166, px + 248, 166);
  b += arrow(px + 442, 166, px + 478, 166);
  b += box(px + 250, 262, 210, 104, 'WGSL compute shader', ['one storage buffer', 'per archetype', "readback: none / async / sync"], t.s[1]);
  b += box(px + 480, 262, 210, 104, 'Compiled JS loop', ['columns as constants', 'CPU fallback, exact parity', '1.1-1.3× bitecs'], t.s[2]);
  b += arrow(px + 560, 212, px + 380, 258);
  b += arrow(px + 585, 212, px + 585, 258);
  b += text(px, 398, "target: 'auto' runs small queries on the CPU and large ones on the GPU.", { size: 12.5, fill: t.muted });
  b += text(px, 416, 'calibrateAuto() measures the switch point on the user\u2019s own device.', { size: 12.5, fill: t.muted });
  return svg(W, H, t, 'How CozyECS works: archetype tables and the kernel compilation pipeline', b);
}

const charts = { hero, throughput, memory, gpu, 'how-it-works': howItWorks };
for (const [name, fn] of Object.entries(charts)) {
  for (const mode of ['light', 'dark']) {
    writeFileSync(OUT(`${name}-${mode}.svg`), fn(THEMES[mode], mode));
  }
}
console.log('wrote', Object.keys(charts).length * 2, 'SVGs to docs/assets/');
