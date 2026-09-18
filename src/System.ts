import type { Query, QueryDesc } from './query';
import type { World } from './world';

/** Registration options shared by function and class systems. */
export interface SystemOptions {
  /** Group run by `world.update(dt, group)`. Default 'update'. */
  group?: string;
  /** Lower runs first; ties run in registration order. Default 0. */
  order?: number;
}

/** Options for `world.system(name, opts, fn)`. */
export interface FunctionSystemOptions extends SystemOptions {
  /** Query passed as first argument to fn (a QueryDesc is resolved via world.query). */
  query?: Query | QueryDesc;
}

/** Function system body. `q` is undefined when no query was given. */
export type SystemFn = (q: Query, dt: number, world: World) => void;

/** Common shape the Scheduler and World.update rely on (members prefixed with _ are internal). */
export interface Runnable {
  name: string;
  group: string;
  order: number;
  enabled: boolean;
  /** @internal registration sequence number (tie-breaker for ordering). */
  _seq: number;
  /** @internal runs one update tick. */
  _run(dt: number): void;
}

/** Handle returned by `world.system(...)`. */
export interface SystemHandle extends Runnable {
  readonly name: string;
  readonly group: string;
  readonly order: number;
  /** Skipped by update when false. */
  enabled: boolean;
  readonly query: Query | undefined;
  readonly fn: SystemFn;
}

/** @internal Concrete function-system record. Created by world.system(). */
export class FunctionSystem implements SystemHandle {
  readonly world: World;
  readonly name: string;
  readonly group: string;
  readonly order: number;
  enabled: boolean;
  readonly query: Query | undefined;
  readonly fn: SystemFn;
  _seq: number;

  constructor(world: World, name: string, group: string, order: number, query: Query | undefined, fn: SystemFn, seq: number) {
    this.world = world;
    this.name = name;
    this.group = group;
    this.order = order;
    this.enabled = true;
    this.query = query;
    this.fn = fn;
    this._seq = seq;
  }

  /**
   * Calls `fn(query, dt, world)`. The `fn` call site is made megamorphic at module load (see
   * `warmUpSystemCallSites`), so V8 never inlines a system body here: world.update runs systems
   * inside try/finally, and loops inlined into a try region compile ~1.7x slower.
   */
  _run(dt: number): void {
    // `q` is undefined when no query was given (documented on SystemFn).
    this.fn(this.query as Query, dt, this.world);
  }
}

/**
 * Class-style system.
 * @example
 * class Move extends System {
 *   q = this.query({ all: [Position, Velocity] });
 *   onUpdate(dt: number) { for (const c of this.q.chunks) { ... } }
 * }
 * world.addSystem(Move, { order: 1 });
 */
export abstract class System implements Runnable {
  readonly world: World;
  /** Set by world.addSystem (defaults to the constructor name). */
  name: string;
  /** Set by world.addSystem. */
  group: string;
  /** Set by world.addSystem. */
  order: number;
  /** Skipped by update when false. */
  enabled: boolean;
  /** @internal */
  _seq: number;

  /** Stores `world`; name/group/order are assigned by world.addSystem right after construction. */
  constructor(world: World) {
    this.world = world;
    this.name = '';
    this.group = 'update';
    this.order = 0;
    this.enabled = true;
    this._seq = 0;
  }

  /** Shorthand for `this.world.query(desc)`. Safe in field initializers. */
  query(desc: QueryDesc): Query {
    return this.world.query(desc);
  }

  /** Called once after registration. */
  onCreate(): void {}

  /** Called every `world.update(dt, group)` while enabled. */
  abstract onUpdate(dt: number): void;

  /** Called by world.removeSystem. */
  onDestroy(): void {}

  /** @internal calls onUpdate(dt) (call site kept megamorphic, see FunctionSystem._run). */
  _run(dt: number): void {
    this.onUpdate(dt);
  }
}

/*
 * Call-site warm-up. Each `_run` has one call site shared by every system. When an app has a
 * single hot system (or a single class system), V8 records a monomorphic target and inlines the
 * system body into world.update's try/finally, where tight loops compile much worse (measured
 * 1.6-1.7x slower for a 1000-row Position+Velocity loop). Calling each `_run` here with more than
 * four distinct targets makes the call sites megamorphic up front: the system body is then
 * compiled on its own (no try region) at the cost of one indirect call per system per update.
 */
let warmupSink = 0;
function warmUpSystemCallSites(): void {
  const world = null as unknown as World;
  const fns: SystemFn[] = [
    (_q, dt) => { warmupSink = dt + 1; },
    (_q, dt) => { warmupSink = dt + 2; },
    (_q, dt) => { warmupSink = dt + 3; },
    (_q, dt) => { warmupSink = dt + 4; },
    (_q, dt) => { warmupSink = dt + 5; },
    (_q, dt) => { warmupSink = dt + 6; },
    (_q, dt) => { warmupSink = dt + 7; },
    (_q, dt) => { warmupSink = dt + 8; },
  ];
  const fsystems = fns.map((fn) => new FunctionSystem(world, '', '', 0, undefined, fn, 0));
  class W1 extends System { onUpdate(dt: number): void { warmupSink = dt + 1; } }
  class W2 extends System { onUpdate(dt: number): void { warmupSink = dt + 2; } }
  class W3 extends System { onUpdate(dt: number): void { warmupSink = dt + 3; } }
  class W4 extends System { onUpdate(dt: number): void { warmupSink = dt + 4; } }
  class W5 extends System { onUpdate(dt: number): void { warmupSink = dt + 5; } }
  class W6 extends System { onUpdate(dt: number): void { warmupSink = dt + 6; } }
  class W7 extends System { onUpdate(dt: number): void { warmupSink = dt + 7; } }
  class W8 extends System { onUpdate(dt: number): void { warmupSink = dt + 8; } }
  const csystems: System[] = [new W1(world), new W2(world), new W3(world), new W4(world), new W5(world), new W6(world), new W7(world), new W8(world)];
  // Enough calls for V8 to allocate feedback vectors and record > 4 targets per site.
  for (let r = 0; r < 64; r++) {
    for (let i = 0; i < 8; i++) {
      fsystems[i]._run(r);
      csystems[i]._run(r);
    }
  }
}
warmUpSystemCallSites();

/** Constructor type accepted by world.addSystem. */
export type SystemClass<T extends System = System> = new (world: World) => T;

/** Copy-on-write removal from a group; deletes the group when it becomes empty. */
function removeFrom(groups: Map<string, Runnable[]>, name: string, list: Runnable[], system: Runnable): boolean {
  if (list.length === 1) {
    groups.delete(name);
    return true;
  }
  const next = list.slice();
  next.splice(list.indexOf(system), 1);
  groups.set(name, next);
  return true;
}

const EMPTY_GROUP: readonly Runnable[] = Object.freeze([]);

/**
 * @internal Holds registered systems per group, sorted stably by (order, _seq).
 * Group arrays are copy-on-write: add/remove replace the array, so a loop over the
 * array returned by `group()` is unaffected by registrations made during it.
 */
export class Scheduler {
  /** @internal group name -> sorted systems. */
  readonly _groups: Map<string, Runnable[]>;

  constructor() {
    this._groups = new Map();
  }

  /** Inserts `system` into its group keeping (order, _seq) ordering. */
  add(system: Runnable): void {
    const groups = this._groups;
    const list = groups.get(system.group);
    if (list === undefined) {
      groups.set(system.group, [system]);
      return;
    }
    if (list.indexOf(system) !== -1) return;
    const order = system.order;
    const seq = system._seq;
    // Insert after every entry with (order, _seq) <= the new system's (stable).
    let at = list.length;
    while (at > 0) {
      const prev = list[at - 1];
      if (prev.order < order || (prev.order === order && prev._seq <= seq)) break;
      at--;
    }
    const next = list.slice();
    next.splice(at, 0, system);
    groups.set(system.group, next);
  }

  /** Removes `system`. @returns false if it was not registered. */
  remove(system: Runnable): boolean {
    const groups = this._groups;
    const list = groups.get(system.group);
    if (list !== undefined && list.indexOf(system) !== -1) return removeFrom(groups, system.group, list, system);
    // `group` is mutable on class systems: fall back to scanning every group.
    for (const [name, other] of groups) {
      if (other.indexOf(system) !== -1) return removeFrom(groups, name, other, system);
    }
    return false;
  }

  /** Sorted systems of `group` (empty frozen array if none). Do not mutate. */
  group(group: string): readonly Runnable[] {
    const list = this._groups.get(group);
    return list === undefined ? EMPTY_GROUP : list;
  }
}
