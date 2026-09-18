// Memory probe: node --expose-gc benchmarks/memory.js <lib> [count]
// Creates <count> entities with Position{x,y} + Velocity{dx,dy} and prints a JSON line with
// retained JS heap + ArrayBuffer bytes (process.memoryUsage().arrayBuffers), measured after forced GCs.
// `external` is also reported but not used for the total: it includes unrelated native allocations
// and moves by several MB between runs.
import { ADAPTERS } from './adapters/index.js';

if (!process.env.NODE_ENV) process.env.NODE_ENV = 'production';

const lib = process.argv[2];
const count = Number(process.argv[3] || 100000);

if (typeof globalThis.gc !== 'function') {
  console.log(JSON.stringify({ lib, error: 'run with node --expose-gc' }));
  process.exit(1);
}

function fullGc() {
  for (let i = 0; i < 4; i++) globalThis.gc();
}

const entry = ADAPTERS.find((a) => a.lib === lib);
if (!entry) {
  console.log(JSON.stringify({ lib, error: 'unknown lib' }));
  process.exit(1);
}

try {
  const mod = await entry.load();
  // Warm the code path with a tiny world so JIT/code memory is not attributed to the data.
  if (mod.memoryWarmup !== false) await mod.memory(10);
  fullGc();
  const before = process.memoryUsage();
  const t0 = performance.now();
  let keep = await mod.memory(count);
  const createMs = performance.now() - t0;
  fullGc();
  const after = process.memoryUsage();
  const heap = after.heapUsed - before.heapUsed;
  const arrayBuffers = after.arrayBuffers - before.arrayBuffers;
  const externalAll = after.external - before.external;
  const total = heap + arrayBuffers;
  console.log(
    JSON.stringify({ lib, count, heap, external: arrayBuffers, externalAll, total, bytesPerEntity: total / count, createMs, alive: keep !== undefined }),
  );
  keep = null;
} catch (e) {
  console.log(JSON.stringify({ lib, error: String((e && e.stack) || e).split('\n').slice(0, 3).join(' | ') }));
  process.exit(1);
}
