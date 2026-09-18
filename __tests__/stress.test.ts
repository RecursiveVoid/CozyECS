/// <reference types="node" />
/**
 * Stress / fuzz tests for CozyECS.
 *
 * 1. A deterministic, seeded fuzzer that drives a World with random structural and data
 *    operations (spawn, spawnMany, destroy, add, remove, set, enable, systems and forEach
 *    that mutate while iterating, hook subscribe/unsubscribe) and mirrors every operation
 *    in a naive reference model: Map<entity, Map<component, { vals, en }>>.
 *    Deferred structural changes are mirrored with a model-side command queue that is
 *    applied exactly when the world flushes.
 * 2. A scale test: 1,000,000 entities via spawnMany, iterated with chunk loops and forEach,
 *    with a bytes-per-entity measurement.
 */
import { describe, test, expect } from '@jest/globals';
import { World, System, component, tag, f32, f64, i8, i16, i32, u8, u16, u32, bool, str } from '../src';
import { FIELD_BOOL, FIELD_STR } from '../src/types';

/* eslint-disable @typescript-eslint/no-explicit-any */

// ---------------------------------------------------------------------------------- PRNG

function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------------------------------------------------------------------------------- components

const Pos = component({ x: f32, y: f32 }, { name: 'Pos' });
const Vel = component({ dx: f64 }, { name: 'Vel' });
// Burn ids so later components live in mask words 1 and 2 (ids >= 32 / >= 64).
for (let i = 0; i < 40; i++) component({ pad: u8 }, { name: `Pad${i}` });
const Health = component({ hp: i16, dead: bool }, { name: 'Health', enableable: true });
const Name = component({ s: str }, { name: 'Name' });
const Misc = component({ a: i8, b: u16, c: u32, d: i32, e: u8 }, { name: 'Misc' });
for (let i = 0; i < 25; i++) tag({ name: `PadTag${i}` });
const TagA = tag({ name: 'TagA' });
const TagB = tag({ name: 'TagB', enableable: true });
const ALL: any[] = [Pos, Vel, Health, Name, Misc, TagA, TagB];
const ENABLEABLE: any[] = ALL.filter((c) => c.enableable);

const QUERY_DESCS: any[] = [
  { all: [Pos] },
  { all: [Pos, Vel] },
  { all: [Health] },
  { all: [TagB] },
  { all: [Health, TagB], none: [Misc] },
  { any: [TagA, Name], none: [Vel] },
  {},
  { none: [Pos, Vel, Health, Name, Misc, TagA, TagB] },
  { all: [Name], any: [Health, Misc] },
];

const STRINGS = ['', 'a', 'hello', 'world', 'ünïcødé', 'x'.repeat(40), '0', 'null'];

// ---------------------------------------------------------------------------------- model helpers

type Slot = { vals: Record<string, any>; en: boolean };
type MEnt = Map<any, Slot>;

function zeroVals(C: any): Record<string, any> {
  const o: Record<string, any> = {};
  for (let i = 0; i < C.keys.length; i++) {
    const code = C.tokens[i].code;
    o[C.keys[i]] = code === FIELD_STR ? '' : code === FIELD_BOOL ? false : 0;
  }
  return o;
}

function coerce(token: any, v: any): any {
  if (token.code === FIELD_STR) return typeof v === 'string' ? v : String(v);
  if (token.code === FIELD_BOOL) return !!v;
  const a = new token.ctor(1);
  a[0] = v;
  return a[0];
}

function matchesDesc(ent: MEnt, desc: any): boolean {
  const all = desc.all || [];
  const any = desc.any || [];
  const none = desc.none || [];
  for (const c of all) if (!ent.has(c)) return false;
  if (any.length > 0 && !any.some((c: any) => ent.has(c))) return false;
  for (const c of none) if (ent.has(c)) return false;
  return true;
}

function enabledForDesc(ent: MEnt, desc: any): boolean {
  for (const c of desc.all || []) if (c.enableable && !(ent.get(c) as Slot).en) return false;
  return true;
}

class FuzzFailure extends Error {}
const PROGRESS = process.env.FUZZ_PROGRESS;

// ---------------------------------------------------------------------------------- fuzzer

interface FuzzStats {
  topOps: number;
  innerOps: number;
  spawns: number;
  spawnMany: number;
  destroys: number;
  adds: number;
  removes: number;
  sets: number;
  enables: number;
  updates: number;
  forEaches: number;
  fullChecks: number;
  maxAlive: number;
  expectedThrows: number;
  maxDestroysPerIndex: number;
}

class Fuzzer {
  readonly rng: () => number;
  readonly world: World;
  readonly model = new Map<number, MEnt>();
  /** Committed entities as an array for O(1) random picks. */
  readonly list: number[] = [];
  readonly listIdx = new Map<number, number>();
  /** Spawned while deferring, not yet placed. */
  readonly pending = new Set<number>();
  queue: any[] = [];
  depth = 0;
  readonly stale: number[] = [];
  /** index -> number of destroys (expected generation = count & 0xfff). */
  readonly destroysByIndex = new Map<number, number>();
  readonly log: string[] = [];
  readonly queries: any[];
  readonly stats: FuzzStats = {
    topOps: 0, innerOps: 0, spawns: 0, spawnMany: 0, destroys: 0, adds: 0, removes: 0, sets: 0,
    enables: 0, updates: 0, forEaches: 0, fullChecks: 0, maxAlive: 0, expectedThrows: 0, maxDestroysPerIndex: 0,
  };
  readonly target: number;
  // Event mirrors
  readonly enterQuery: any;
  enterSet: Set<number> | null = new Set();
  unsubEnter: (() => void) | null = null;
  unsubExit: (() => void) | null = null;
  healthSet: Set<number> | null = new Set();
  unsubAdd: (() => void) | null = null;
  unsubRemove: (() => void) | null = null;
  readonly eventErrors: string[] = [];
  systemOrder: string[] = [];

  constructor(readonly seed: number, capacity: number, target: number) {
    this.rng = mulberry32(seed);
    this.world = new World({ initialCapacity: capacity });
    this.target = target;
    this.queries = QUERY_DESCS.map((d) => ({ desc: d, q: this.world.query(d) }));
    this.enterQuery = this.world.query({ all: [Pos, Vel] });
    this.subscribeEvents();
    this.registerSystems();
  }

  // ---------------------------------------------------------------- utils
  fail(msg: string): never {
    throw new FuzzFailure(
      `[seed ${this.seed} op ${this.stats.topOps}] ${msg}\nrecent ops:\n  ${this.log.slice(-25).join('\n  ')}`,
    );
  }
  note(s: string): void {
    this.log.push(s);
    if (this.log.length > 200) this.log.splice(0, 100);
  }
  int(n: number): number {
    return Math.floor(this.rng() * n);
  }
  pick<T>(arr: readonly T[]): T {
    return arr[this.int(arr.length)];
  }
  subset(): any[] {
    const out: any[] = [];
    for (const c of ALL) if (this.rng() < 0.35) out.push(c);
    return out;
  }
  randomValue(token: any): any {
    if (token.code === FIELD_STR) return this.pick(STRINGS);
    if (token.code === FIELD_BOOL) return this.rng() < 0.5;
    const r = this.rng();
    if (r < 0.1) return 0;
    if (r < 0.5) return this.int(200) - 100;
    if (r < 0.8) return (this.rng() - 0.5) * 1e6;
    return Math.floor((this.rng() - 0.5) * 2 ** 34);
  }
  randomValues(C: any): Record<string, any> {
    const o: Record<string, any> = {};
    for (let i = 0; i < C.keys.length; i++) if (this.rng() < 0.7) o[C.keys[i]] = this.randomValue(C.tokens[i]);
    if (this.rng() < 0.05) o.__unknown = 123;
    return o;
  }
  expectThrow(label: string, fn: () => void): void {
    let threw = false;
    try {
      fn();
    } catch {
      threw = true;
    }
    if (!threw) this.fail(`expected throw: ${label}`);
    this.stats.expectedThrows++;
  }
  pickCommitted(): number {
    return this.list.length === 0 ? -1 : this.list[this.int(this.list.length)];
  }
  pickStale(): number {
    if (this.stale.length === 0 || this.rng() < 0.1) return (this.int(0x7fffffff) * 2) >>> 0; // garbage
    return this.stale[this.int(this.stale.length)];
  }

  // ---------------------------------------------------------------- model mutations
  mSpawn(e: number, comps: any[]): void {
    if (this.model.has(e)) this.fail(`model: entity ${e} spawned twice (id collision)`);
    const idx = e & 0xfffff;
    const expGen = (this.destroysByIndex.get(idx) || 0) & 0xfff;
    if (e >>> 20 !== expGen) {
      this.fail(`entity ${e} (index ${idx}) has generation ${e >>> 20}, expected ${expGen} after destroys`);
    }
    const ent: MEnt = new Map();
    for (const C of comps) if (!ent.has(C)) ent.set(C, { vals: zeroVals(C), en: true });
    this.model.set(e, ent);
    this.listIdx.set(e, this.list.length);
    this.list.push(e);
    if (this.model.size > this.stats.maxAlive) this.stats.maxAlive = this.model.size;
  }
  mDestroy(e: number): void {
    this.model.delete(e);
    const idx = e & 0xfffff;
    const n = (this.destroysByIndex.get(idx) || 0) + 1;
    this.destroysByIndex.set(idx, n);
    if (n > this.stats.maxDestroysPerIndex) this.stats.maxDestroysPerIndex = n;
    const i = this.listIdx.get(e) as number;
    const last = this.list.pop() as number;
    if (last !== e) {
      this.list[i] = last;
      this.listIdx.set(last, i);
    }
    this.listIdx.delete(e);
    this.stale.push(e);
    if (this.stale.length > 400) this.stale.splice(0, 200);
  }
  mWrite(slot: Slot, C: any, values: any): void {
    if (values === undefined || values === null) return;
    for (let i = 0; i < C.keys.length; i++) {
      const v = values[C.keys[i]];
      if (v === undefined) continue;
      slot.vals[C.keys[i]] = coerce(C.tokens[i], v);
    }
  }
  mAdd(e: number, C: any, values: any): void {
    const ent = this.model.get(e);
    if (ent === undefined) return;
    let slot = ent.get(C);
    if (slot === undefined) {
      slot = { vals: zeroVals(C), en: true };
      ent.set(C, slot);
    }
    this.mWrite(slot, C, values);
  }
  mRemove(e: number, C: any): void {
    const ent = this.model.get(e);
    if (ent !== undefined) ent.delete(C);
  }
  applyQueue(): void {
    const q = this.queue;
    this.queue = [];
    for (const cmd of q) {
      switch (cmd.t) {
        case 'spawn':
          this.pending.delete(cmd.e);
          this.mSpawn(cmd.e, cmd.comps);
          break;
        case 'spawnMany':
          if (cmd.rec.length !== cmd.count) {
            this.fail(`deferred spawnMany(${cmd.count}) ran init ${cmd.rec.length} times`);
          }
          for (const r of cmd.rec) {
            this.mSpawn(r.e, cmd.comps);
            for (const [C, vals] of r.sets) this.mAdd(r.e, C, vals);
          }
          break;
        case 'destroy':
          if (this.model.has(cmd.e)) this.mDestroy(cmd.e);
          break;
        case 'add':
          this.mAdd(cmd.e, cmd.C, cmd.values);
          break;
        case 'remove':
          this.mRemove(cmd.e, cmd.C);
          break;
      }
    }
    if (this.pending.size !== 0) this.fail(`pending entities left after flush: ${[...this.pending]}`);
  }

  // ---------------------------------------------------------------- events
  subscribeEvents(): void {
    const w = this.world;
    this.enterSet = new Set();
    for (const [e, ent] of this.model) if (matchesDesc(ent, { all: [Pos, Vel] })) this.enterSet.add(e);
    this.unsubEnter = this.enterQuery.onEnter((e: number) => {
      const s = this.enterSet as Set<number>;
      if (s.has(e)) this.eventErrors.push(`onEnter fired twice for ${e}`);
      s.add(e);
    });
    this.unsubExit = this.enterQuery.onExit((e: number) => {
      const s = this.enterSet as Set<number>;
      if (!s.has(e)) this.eventErrors.push(`onExit fired for ${e} which never entered`);
      s.delete(e);
    });
    this.healthSet = new Set();
    for (const [e, ent] of this.model) if (ent.has(Health)) this.healthSet.add(e);
    this.unsubAdd = w.onAdd(Health, (e: number) => {
      const s = this.healthSet as Set<number>;
      if (s.has(e)) this.eventErrors.push(`onAdd(Health) fired twice for ${e}`);
      s.add(e);
    });
    this.unsubRemove = w.onRemove(Health, (e: number) => {
      const s = this.healthSet as Set<number>;
      if (!s.has(e)) this.eventErrors.push(`onRemove(Health) for ${e} without add`);
      s.delete(e);
    });
  }
  toggleEvents(): void {
    if (this.depth > 0) return; // model sets are rebuilt from committed state only
    if (this.unsubEnter) {
      this.note('unsubscribe events');
      this.unsubEnter();
      this.unsubExit!();
      this.unsubAdd!();
      this.unsubRemove!();
      this.unsubEnter = this.unsubExit = this.unsubAdd = this.unsubRemove = null;
      this.enterSet = null;
      this.healthSet = null;
    } else {
      this.note('subscribe events');
      this.subscribeEvents();
    }
  }

  // ---------------------------------------------------------------- systems
  registerSystems(): void {
    const self = this;
    const w = this.world;
    // Runs second (order 1): direct chunk mutation, then ops + forEach.
    w.system('movePos', { query: { all: [Pos] }, order: 1 }, (q, dt, world) => {
      self.systemOrder.push('movePos');
      if (world !== w) self.fail('system got wrong world');
      if (dt !== 0.5) self.fail(`system got dt ${dt}`);
      self.applyQueue(); // previous system's flush
      // chunk loop: p.x += 1 (no structural changes pending in this system yet)
      let seen = 0;
      for (const chunk of q.chunks) {
        const p = chunk.col(Pos);
        for (let i = 0; i < chunk.count; i++) {
          const e = chunk.entities[i];
          const ent = self.model.get(e);
          if (!ent || !ent.has(Pos)) self.fail(`movePos chunk row entity ${e} not in model with Pos`);
          p.x[i] += 1;
          const slot = ent.get(Pos) as Slot;
          slot.vals.x = Math.fround(slot.vals.x + 1);
          if (p.x[i] !== slot.vals.x) self.fail(`chunk write mismatch e=${e}`);
          seen++;
        }
      }
      let expected = 0;
      for (const ent of self.model.values()) if (ent.has(Pos)) expected++;
      if (seen !== expected) self.fail(`movePos chunk rows ${seen} != model ${expected}`);
      self.depth++; // the system body itself defers structural changes
      try {
        const n = self.int(4);
        for (let i = 0; i < n; i++) self.randomOp(true, -1);
        self.runForEach(q, { all: [Pos] }, 0.03);
        const m = self.int(3);
        for (let i = 0; i < m; i++) self.randomOp(true, -1);
      } finally {
        self.depth--;
      }
    });
    class HealthSys extends System {
      q = this.query({ any: [Health, TagA] });
      onUpdate(dt: number): void {
        self.systemOrder.push('health');
        if (dt !== 0.5) self.fail(`class system got dt ${dt}`);
        self.applyQueue();
        self.depth++;
        try {
          self.runForEach(this.q, { any: [Health, TagA] }, 0.03);
          if (self.rng() < 0.5) self.randomOp(true, -1);
        } finally {
          self.depth--;
        }
      }
    }
    w.addSystem(HealthSys, { order: 0 });
    const disabled = w.system('disabled', { order: -5 }, () => self.fail('disabled system ran'));
    disabled.enabled = false;
    w.system('otherGroup', { group: 'late' }, () => self.fail('system from other group ran'));
    // A system that removes itself is not used (would change the fuzz structure).
  }

  /** forEach with random ops; verifies visited set === model expectation (with enabled filter). */
  runForEach(q: any, desc: any, opRate: number): void {
    this.stats.forEaches++;
    const expected = new Set<number>();
    for (const [e, ent] of this.model) if (matchesDesc(ent, desc) && enabledForDesc(ent, desc)) expected.add(e);
    const visited = new Set<number>();
    this.depth++;
    try {
      q.forEach((e: number, chunk: any, row: number) => {
        if (chunk.entities[row] !== e) this.fail(`forEach entity ${e} != chunk.entities[row]`);
        if (visited.has(e)) this.fail(`forEach visited ${e} twice`);
        visited.add(e);
        if (!expected.has(e)) this.fail(`forEach visited unexpected ${e} (desc ${JSON.stringify(descNames(desc))})`);
        if (!this.world.isAlive(e)) this.fail(`forEach visited dead ${e}`);
        if (this.rng() < opRate) this.randomOp(true, e);
        if (this.depth < 3 && this.rng() < 0.0005) {
          const inner = this.pick(this.queries);
          this.runForEach(inner.q, inner.desc, 0); // nested, read-only
        }
      });
    } finally {
      this.depth--;
    }
    if (visited.size !== expected.size) {
      const missing = [...expected].filter((x) => !visited.has(x)).slice(0, 5);
      this.fail(`forEach visited ${visited.size}, expected ${expected.size}; missing e.g. ${missing}`);
    }
    if (this.depth === 0) this.applyQueue();
  }

  // ---------------------------------------------------------------- operations
  get deferring(): boolean {
    return this.depth > 0;
  }

  opSpawn(): void {
    this.stats.spawns++;
    const comps = this.subset();
    const form = this.int(3);
    let e: number;
    if (form === 0 && comps.length === 0) e = this.world.spawn();
    else if (form === 1 || (form === 0 && comps.length > 0)) {
      const list = comps.slice();
      if (list.length > 0 && this.rng() < 0.3) list.push(list[0]); // duplicate
      list.reverse();
      e = this.world.spawn(list);
    } else e = this.world.spawn(this.world.archetype(...comps));
    this.note(`spawn(${names(comps)}) -> ${e}${this.deferring ? ' [deferred]' : ''}`);
    if (!this.world.isAlive(e)) this.fail(`spawned entity ${e} not alive`);
    if (this.model.has(e) || this.pending.has(e)) this.fail(`spawn returned id ${e} that is already alive`);
    if (this.deferring) {
      this.pending.add(e);
      this.queue.push({ t: 'spawn', e, comps });
    } else {
      this.mSpawn(e, comps);
    }
  }

  opSpawnMany(): void {
    this.stats.spawnMany++;
    const comps = this.subset();
    const arch = this.world.archetype(...comps);
    const count = this.rng() < 0.05 && this.model.size < this.target ? 20 + this.int(Math.max(1, this.target >> 3)) : 1 + this.int(8);
    const withInit = this.deferring || this.rng() < 0.6;
    this.note(`spawnMany(${names(comps)}, ${count}, init=${withInit})${this.deferring ? ' [deferred]' : ''}`);
    if (withInit) {
      const rec: any[] = [];
      let expectI = 0;
      const init = (chunk: any, row: number, i: number) => {
        if (chunk !== arch) this.fail('spawnMany init chunk !== archetype');
        if (i !== expectI++) this.fail(`spawnMany init i=${i}, expected ${expectI - 1}`);
        const e = chunk.entities[row];
        if (!this.world.isAlive(e)) this.fail(`spawnMany init entity ${e} not alive`);
        const sets: [any, any][] = [];
        for (const C of comps) {
          if (C.keys.length === 0 || this.rng() < 0.4) continue;
          const cols = chunk.col(C);
          const vals: any = {};
          for (let k = 0; k < C.keys.length; k++) {
            const tok = C.tokens[k];
            const v = this.randomValue(tok);
            const raw = tok.code === FIELD_STR ? this.world.strings.intern(v) : tok.code === FIELD_BOOL ? (v ? 1 : 0) : v;
            cols[C.keys[k]][row] = raw;
            vals[C.keys[k]] = v;
          }
          sets.push([C, vals]);
        }
        rec.push({ e, sets });
      };
      if (this.deferring) {
        const before = arch.count;
        this.world.spawnMany(arch, count, init);
        if (arch.count !== before || rec.length !== 0) this.fail('deferred spawnMany was applied immediately');
        this.queue.push({ t: 'spawnMany', comps, rec, count });
      } else {
        this.world.spawnMany(arch, count, init);
        if (rec.length !== count) this.fail(`spawnMany init called ${rec.length}/${count}`);
        for (const r of rec) {
          this.mSpawn(r.e, comps);
          for (const [C, vals] of r.sets) this.mAdd(r.e, C, vals);
        }
      }
    } else {
      const before = arch.count;
      this.world.spawnMany(arch, count);
      if (arch.count !== before + count) this.fail(`spawnMany count ${arch.count} != ${before + count}`);
      for (let r = before; r < before + count; r++) this.mSpawn(arch.entities[r], comps);
    }
  }

  opDestroy(current: number): void {
    this.stats.destroys++;
    const r = this.rng();
    if (r < 0.1) {
      const s = this.pickStale();
      if (this.model.has(s) || this.pending.has(s)) return;
      this.note(`destroy(stale ${s})`);
      this.world.destroy(s);
      if (this.world.isAlive(s)) this.fail(`stale ${s} alive after destroy`);
      return;
    }
    let e: number;
    if (this.deferring && this.pending.size > 0 && r < 0.2) e = this.pick([...this.pending]);
    else if (current >= 0 && r < 0.5) e = current;
    else e = this.pickCommitted();
    if (e < 0) return;
    this.note(`destroy(${e})${this.deferring ? ' [deferred]' : ''}`);
    this.world.destroy(e);
    if (this.deferring) {
      if (!this.world.isAlive(e)) this.fail(`deferred destroy(${e}) killed it immediately`);
      this.queue.push({ t: 'destroy', e });
    } else {
      if (this.world.isAlive(e)) this.fail(`destroy(${e}) left it alive`);
      this.mDestroy(e);
    }
  }

  opAdd(current: number): void {
    this.stats.adds++;
    const C = this.pick(ALL);
    const values = this.rng() < 0.6 && C.keys.length > 0 ? this.randomValues(C) : undefined;
    const r = this.rng();
    if (r < 0.05) {
      const s = this.pickStale();
      if (this.model.has(s) || this.pending.has(s)) return;
      this.note(`add(stale ${s}, ${C.name}) expect throw`);
      this.expectThrow(`add on dead entity ${s}`, () => this.world.add(s, C, values));
      return;
    }
    let e: number;
    if (this.deferring && this.pending.size > 0 && r < 0.15) e = this.pick([...this.pending]);
    else if (current >= 0 && r < 0.5) e = current;
    else e = this.pickCommitted();
    if (e < 0) return;
    this.note(`add(${e}, ${C.name}, ${JSON.stringify(values)})${this.deferring ? ' [deferred]' : ''}`);
    this.world.add(e, C, values);
    if (this.deferring) this.queue.push({ t: 'add', e, C, values });
    else this.mAdd(e, C, values);
  }

  opRemove(current: number): void {
    this.stats.removes++;
    const C = this.pick(ALL);
    const r = this.rng();
    if (r < 0.05) {
      const s = this.pickStale();
      if (this.model.has(s) || this.pending.has(s)) return;
      this.note(`remove(stale ${s}, ${C.name})`);
      this.world.remove(s, C);
      return;
    }
    let e: number;
    if (this.deferring && this.pending.size > 0 && r < 0.15) e = this.pick([...this.pending]);
    else if (current >= 0 && r < 0.5) e = current;
    else e = this.pickCommitted();
    if (e < 0) return;
    this.note(`remove(${e}, ${C.name})${this.deferring ? ' [deferred]' : ''}`);
    this.world.remove(e, C);
    if (this.deferring) this.queue.push({ t: 'remove', e, C });
    else this.mRemove(e, C);
  }

  opSet(current: number): void {
    this.stats.sets++;
    const C = this.pick(ALL);
    const values = this.randomValues(C);
    const r = this.rng();
    if (r < 0.03) {
      const s = this.pickStale();
      if (this.model.has(s) || this.pending.has(s)) return;
      this.note(`set(stale ${s}) expect throw`);
      this.expectThrow('set on dead entity', () => this.world.set(s, C, values));
      return;
    }
    if (this.deferring && this.pending.size > 0 && r < 0.06) {
      const p = this.pick([...this.pending]);
      this.note(`set(pending ${p}, ${C.name}) expect throw`);
      this.expectThrow('set on pending (unplaced) entity', () => this.world.set(p, C, values));
      if (this.world.has(p, C)) this.fail(`has(pending ${p}, ${C.name}) true before flush`);
      return;
    }
    const e = current >= 0 && r < 0.5 ? current : this.pickCommitted();
    if (e < 0) return;
    const ent = this.model.get(e) as MEnt;
    const slot = ent.get(C);
    this.note(`set(${e}, ${C.name}, ${JSON.stringify(values)})${slot ? '' : ' expect throw'}`);
    if (slot === undefined) {
      this.expectThrow(`set(${C.name}) on entity lacking it`, () => this.world.set(e, C, values));
      return;
    }
    this.world.set(e, C, values);
    this.mWrite(slot, C, values);
  }

  opEnable(current: number): void {
    this.stats.enables++;
    const r = this.rng();
    if (r < 0.03) {
      const C = this.pick(ALL.filter((c) => !c.enableable));
      const e = this.pickCommitted();
      if (e < 0) return;
      this.note(`enable(${e}, non-enableable ${C.name}) expect throw`);
      this.expectThrow('enable on non-enableable', () => this.world.enable(e, C, false));
      return;
    }
    const C = this.pick(ENABLEABLE);
    // Inside forEach only the current (already visited) entity may be toggled, so the
    // expected visit set stays exact.
    const e = this.deferring ? current : this.pickCommitted();
    if (e < 0) return;
    const on = this.rng() < 0.5;
    const slot = (this.model.get(e) as MEnt).get(C);
    this.note(`enable(${e}, ${C.name}, ${on})${slot ? '' : ' expect throw'}`);
    if (slot === undefined) {
      this.expectThrow(`enable(${C.name}) on entity lacking it`, () => this.world.enable(e, C, on));
      return;
    }
    if (this.rng() < 0.2 && on) this.world.enable(e, C);
    else this.world.enable(e, C, on);
    slot.en = on;
  }

  randomOp(inner: boolean, current: number): void {
    if (inner) this.stats.innerOps++;
    const size = this.model.size;
    const grow = size < this.target;
    const r = this.rng() * 100;
    if (size > this.target * 1.5 && r < 10) return this.opDestroy(current);
    if (r < (grow ? 16 : 7)) return this.opSpawn();
    if (r < (grow ? 19 : 10)) return grow ? this.opSpawnMany() : this.opDestroy(current);
    if (r < 34) return this.opDestroy(current);
    if (r < 52) return this.opAdd(current);
    if (r < 64) return this.opRemove(current);
    if (r < 82) return this.opSet(current);
    if (r < 90) return this.opEnable(current);
    if (inner) return;
    if (r < 93) {
      this.stats.updates++;
      this.note('update(0.5)');
      this.systemOrder = [];
      this.world.update(0.5);
      this.applyQueue();
      if (this.systemOrder.join(',') !== 'health,movePos') this.fail(`system order ${this.systemOrder}`);
      return;
    }
    if (r < 96) {
      const { q, desc } = this.pick(this.queries);
      this.note(`forEach(${JSON.stringify(descNames(desc))})`);
      this.runForEach(q, desc, 0.02);
      return;
    }
    if (r < 97) {
      this.note('flush() outside iteration');
      this.world.flush();
      return;
    }
    if (r < 97.3) return this.toggleEvents();
    // stale liveness probe
    const s = this.pickStale();
    if (this.world.isAlive(s) !== (this.model.has(s) || this.pending.has(s))) {
      this.fail(`isAlive(${s}) = ${this.world.isAlive(s)} but model says ${this.model.has(s)}`);
    }
  }

  // ---------------------------------------------------------------- verification
  lightCheck(): void {
    const w: any = this.world;
    if (w._entities.aliveCount !== this.model.size) {
      this.fail(`aliveCount ${w._entities.aliveCount} != model ${this.model.size}`);
    }
    for (const { q, desc } of this.queries) {
      let expected = 0;
      for (const ent of this.model.values()) if (matchesDesc(ent, desc)) expected++;
      const got = q.count();
      if (got !== expected) this.fail(`query ${JSON.stringify(descNames(desc))} count ${got} != model ${expected}`);
    }
    if (this.eventErrors.length) this.fail(`event errors: ${this.eventErrors.slice(0, 5).join('; ')}`);
  }

  fullCheck(): void {
    this.stats.fullChecks++;
    this.lightCheck();
    const w = this.world;
    // 1. every alive entity's components / values / enabled flags
    for (const [e, ent] of this.model) {
      if (!w.isAlive(e)) this.fail(`model entity ${e} not alive in world`);
      for (const C of ALL) {
        const slot = ent.get(C);
        const has = w.has(e, C);
        if (has !== (slot !== undefined)) this.fail(`has(${e}, ${C.name}) = ${has}, model ${slot !== undefined}`);
        const view: any = w.get(e, C);
        if (slot === undefined) {
          if (view !== undefined) this.fail(`get(${e}, ${C.name}) should be undefined`);
          if (w.isEnabled(e, C)) this.fail(`isEnabled(${e}, ${C.name}) true for absent component`);
          continue;
        }
        if (view === undefined) this.fail(`get(${e}, ${C.name}) undefined but model has it`);
        for (let i = 0; i < C.keys.length; i++) {
          const k = C.keys[i];
          const code = C.tokens[i].code;
          const exp = slot.vals[k];
          if (view[k] !== exp && !(Number.isNaN(view[k]) && Number.isNaN(exp))) {
            this.fail(`get(${e}, ${C.name}).${k} = ${view[k]}, model ${exp}`);
          }
          const raw = w.getField(e, C, k);
          const rawExp = code === FIELD_STR ? w.strings.intern(exp) : code === FIELD_BOOL ? (exp ? 1 : 0) : exp;
          if (raw !== rawExp) this.fail(`getField(${e}, ${C.name}, ${k}) = ${raw}, model ${rawExp}`);
        }
        const en = w.isEnabled(e, C);
        const enExp = C.enableable ? slot.en : true;
        if (en !== enExp) this.fail(`isEnabled(${e}, ${C.name}) = ${en}, model ${enExp}`);
      }
    }
    // 2. archetype tables: every row is a unique, alive, model entity with the exact component set
    const seen = new Set<number>();
    const everything = w.query({});
    for (const chunk of everything.chunks) {
      if (chunk.count > chunk.capacity) this.fail(`chunk count > capacity`);
      for (let row = 0; row < chunk.count; row++) {
        const e = chunk.entities[row];
        if (seen.has(e)) this.fail(`entity ${e} appears twice in archetype tables`);
        seen.add(e);
        const ent = this.model.get(e);
        if (ent === undefined) this.fail(`archetype row holds non-model entity ${e} (alive=${w.isAlive(e)})`);
        if (ent.size !== chunk.components.length) this.fail(`entity ${e} chunk components mismatch`);
        for (const C of chunk.components) {
          const slot = ent.get(C);
          if (!slot) this.fail(`entity ${e} chunk has ${C.name}, model does not`);
          if (C.enableable) {
            const arr = chunk.enabledArray(C) as Uint8Array;
            if (arr[row] !== (slot.en ? 1 : 0)) this.fail(`enabledArray mismatch e=${e} ${C.name}`);
            if (chunk.isEnabled(C, row) !== slot.en) this.fail(`chunk.isEnabled mismatch e=${e}`);
          } else if (chunk.enabledArray(C) !== undefined) {
            this.fail(`enabledArray defined for non-enableable ${C.name}`);
          }
        }
      }
    }
    if (seen.size !== this.model.size) this.fail(`archetype rows ${seen.size} != model ${this.model.size}`);
    // 3. queries: chunks match, forEach visits
    for (const { q, desc } of this.queries) {
      for (const chunk of q.chunks) {
        if (!q.matches(chunk)) this.fail('query chunk does not match');
      }
      this.runForEach(q, desc, 0);
    }
    // 4. stale ids
    for (const s of this.stale) {
      if (w.isAlive(s) !== this.model.has(s)) this.fail(`stale ${s}: isAlive ${w.isAlive(s)} model ${this.model.has(s)}`);
      if (!this.model.has(s)) {
        if (w.has(s, Pos) || w.get(s, Pos) !== undefined || w.isEnabled(s, Health)) this.fail(`stale ${s} still readable`);
      }
    }
    // 5. event mirrors
    if (this.enterSet) {
      let n = 0;
      for (const [e, ent] of this.model) {
        if (matchesDesc(ent, { all: [Pos, Vel] })) {
          n++;
          if (!this.enterSet.has(e)) this.fail(`onEnter mirror missing ${e}`);
        }
      }
      if (n !== this.enterSet.size) this.fail(`onEnter/onExit mirror size ${this.enterSet.size} != ${n}`);
    }
    if (this.healthSet) {
      let n = 0;
      for (const [e, ent] of this.model) {
        if (ent.has(Health)) {
          n++;
          if (!this.healthSet.has(e)) this.fail(`onAdd(Health) mirror missing ${e}`);
        }
      }
      if (n !== this.healthSet.size) this.fail(`onAdd/onRemove mirror size ${this.healthSet.size} != ${n}`);
    }
  }

  run(ops: number, fullEvery: number): FuzzStats {
    for (let i = 0; i < ops; i++) {
      this.stats.topOps++;
      this.randomOp(false, -1);
      if (this.depth !== 0 || this.queue.length !== 0) this.fail('model queue not drained at top level');
      if (i % 50 === 0) this.lightCheck();
      if (i % fullEvery === 0) this.fullCheck();
      if (PROGRESS && i % 10000 === 0) require('fs').appendFileSync(PROGRESS, `seed ${this.seed} op ${i} alive ${this.model.size} ${JSON.stringify(this.stats)}\n`);
    }
    this.fullCheck();
    return this.stats;
  }
}

function names(comps: any[]): string {
  return comps.map((c) => c.name).join(',');
}
function descNames(desc: any): any {
  const out: any = {};
  for (const k of ['all', 'any', 'none']) if (desc[k]) out[k] = desc[k].map((c: any) => c.name);
  return out;
}

// ---------------------------------------------------------------------------------- tests

describe('fuzz: world vs reference model', () => {
  const cases: [number, number, number, number][] = [
    // seed, initialCapacity, ops, target population
    [0xc0ffee, 64, 200_000, 1500],
    [12345, 1, 60_000, 300],
    [987654321, 8, 60_000, 40],
  ];
  // Optional extra seeds for longer local runs: FUZZ_EXTRA_SEEDS=20
  const extra = Number(process.env.FUZZ_EXTRA_SEEDS || 0);
  for (let i = 0; i < extra; i++) cases.push([1000 + i * 7919, [1, 2, 64][i % 3], 50_000, [20, 200, 1000][(i >> 1) % 3]]);
  for (const [seed, cap, ops, target] of cases) {
    test(`seed ${seed} cap ${cap}: ${ops} ops`, () => {
      const f = new Fuzzer(seed, cap, target);
      const stats = f.run(ops, 1000);
      // eslint-disable-next-line no-console
      console.log(`fuzz seed=${seed} cap=${cap}`, JSON.stringify(stats));
      expect(stats.topOps).toBe(ops);
    }, 600_000);
  }
});

describe('scale', () => {
  test('1,000,000 entities via spawnMany: iterate + bytes/entity', () => {
    const g: any = (global as any).gc;
    const N = 1_000_000;
    const P = component({ x: f32, y: f32 }, { name: 'SP' });
    const V = component({ dx: f32, dy: f32 }, { name: 'SV' });
    if (g) g();
    const m0 = process.memoryUsage();
    const w = new World();
    const arch = w.archetype(P, V);
    let t = performance.now();
    w.spawnMany(arch, N, (chunk, row, i) => {
      const v = chunk.col(V);
      v.dx[row] = 1;
      v.dy[row] = i & 7;
    });
    const spawnMs = performance.now() - t;
    if (g) g();
    const m1 = process.memoryUsage();
    const bytes = m1.heapUsed + m1.arrayBuffers - (m0.heapUsed + m0.arrayBuffers);
    const bpe = bytes / N;

    expect(arch.count).toBe(N);
    expect(w.query({ all: [P, V] }).count()).toBe(N);

    // chunk loop
    const q = w.query({ all: [P, V] });
    t = performance.now();
    for (let frame = 0; frame < 10; frame++) {
      for (const c of q.chunks) {
        const p = c.col(P);
        const v = c.col(V);
        const n = c.count;
        const px = p.x, py = p.y, vx = v.dx, vy = v.dy;
        for (let i = 0; i < n; i++) {
          px[i] += vx[i];
          py[i] += vy[i];
        }
      }
    }
    const chunkMs = (performance.now() - t) / 10;
    expect(arch.col(P).x[0]).toBe(10);
    expect(arch.col(P).y[N - 1]).toBe(10 * ((N - 1) & 7));

    // forEach
    let visits = 0;
    t = performance.now();
    q.forEach(() => {
      visits++;
    });
    const forEachMs = performance.now() - t;
    expect(visits).toBe(N);

    // destroy half during forEach (deferred)
    t = performance.now();
    let k = 0;
    q.forEach((e) => {
      if (k++ % 2 === 0) w.destroy(e);
    });
    const destroyMs = performance.now() - t;
    expect(q.count()).toBe(N / 2);

    // eslint-disable-next-line no-console
    console.log(
      `scale: N=${N} gc=${!!g} spawnMany=${spawnMs.toFixed(0)}ms bytes/entity=${bpe.toFixed(2)} ` +
        `(heapUsed+arrayBuffers delta ${(bytes / 1048576).toFixed(1)} MB; capacity ${arch.capacity}) ` +
        `chunkLoop=${chunkMs.toFixed(1)}ms/frame forEach=${forEachMs.toFixed(0)}ms destroyHalfDeferred=${destroyMs.toFixed(0)}ms`,
    );
    expect(bpe).toBeLessThan(100);
  }, 120_000);
});
