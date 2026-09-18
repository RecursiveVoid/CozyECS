// Adapter registry. Each adapter module exports:
//   variants: { [scenario]: { [variantName]: () => ({ step(): void, check(ticks): string|null }) } }
//     A factory may be async and an instance may set `async: true` (step() returns a Promise);
//     check() may also return a Promise.
//   memory(n): object (or Promise of one) retaining a world with n Position+Velocity entities
//   memoryWarmup?: false to skip the small warm-up world in memory.js
export const ADAPTERS = [
  { lib: 'cozyecs', load: () => import('./cozyecs.js') },
  { lib: 'bitecs', load: () => import('./bitecs.js') },
  { lib: 'bitecs4', load: () => import('./bitecs4.js') },
  { lib: 'wolf-ecs', load: () => import('./wolf-ecs.js') },
  { lib: 'harmony-ecs', load: () => import('./harmony-ecs.js') },
  { lib: 'becsy', load: () => import('./becsy.js') },
  { lib: 'javelin', load: () => import('./javelin.js') },
  { lib: 'miniplex', load: () => import('./miniplex.js') },
  { lib: 'ecsy', load: () => import('./ecsy.js') },
  { lib: 'geotic', load: () => import('./geotic.js') },
  { lib: 'perform-ecs', load: () => import('./perform-ecs.js') },
];

export const SCENARIOS = [
  { id: 'packed_5', desc: '1000 entities with A..E ({value}); 5 systems each doubling one component' },
  { id: 'simple_iter', desc: '4 archetypes x 1000 entities (P+V, P+V+A, P+V+B, P+V+A+B); 1 system P += V' },
  { id: 'frag_iter', desc: '26 components A..Z, 100 entities per letter each with [Letter, Data]; 1 system doubling Data (2600 entities, 26 archetypes)' },
  { id: 'entity_cycle', desc: '1000 entities with A; per op: spawn one B-entity per A, then destroy every B-entity' },
  { id: 'add_remove', desc: '1000 entities with A; per op: add B to every A, then remove B from every B' },
];
