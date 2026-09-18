import { Archetype } from './archetype';
import { componentsKey, maskHas, normalizeComponents } from './component';
import type { ComponentType } from './component';
import { EntityAllocator, HIGH_BITS, INDEX_BITS, INDEX_MASK, LOCATION_FREE, LOCATION_PENDING, TAG_BIG, TAG_PENDING } from './entities';
import { Query } from './query';
import type { QueryDesc } from './query';
import { FunctionSystem, Scheduler, System } from './system';
import type { FunctionSystemOptions, SystemClass, SystemFn, SystemHandle, SystemOptions } from './system';
import { StringTable } from './strings';
import { FIELD_BOOL, FIELD_STR } from './types';
import type { EntityCallback, Schema, TypedArray, ValuesOf } from './types';

/** Options for `new World()`. */
export interface WorldOptions {
  /** Initial row capacity of every archetype and of the entity arrays. Default 64. */
  initialCapacity?: number;
  /**
   * Allocate archetype tables as SharedArrayBuffer when available (e.g. to hand columns to
   * workers). Falls back to ArrayBuffer when SharedArrayBuffer is undefined. Default false.
   */
  shared?: boolean;
}

/** Init callback for spawnMany: `row` is the new row in `chunk`, `i` is 0..count-1. */
export type SpawnInitFn = (chunk: Archetype, row: number, i: number) => void;

// Command op codes (plain consts, not const enum, for isolatedModules/ts-jest).
/** @internal */ export const CMD_SPAWN = 0; // ents=entity, ref=archetype id
/** @internal */ export const CMD_SPAWN_MANY = 1; // ents=count, ref=archetype id, slots=SpawnInitFn slot|-1
/** @internal */ export const CMD_DESTROY = 2; // ents=entity
/** @internal */ export const CMD_ADD = 3; // ents=entity, ref=component id, slots=values slot|-1
/** @internal */ export const CMD_REMOVE = 4; // ents=entity, ref=component id
/** @internal */ export const CMD_ADD_NOVAL = 5; // ents=entity, ref=component id (no payload, slots unused)
/** @internal Bits of a command code holding the op; the ref is `code >> CMD_OP_BITS`. */
export const CMD_OP_BITS = 3;
/** @internal */ export const CMD_OP_MASK = 7;

const COMMAND_INITIAL_CAPACITY = 256;

/**
 * @internal Deferred structural commands, stored as parallel typed arrays (no per-command
 * objects, no JS array churn). Arrays grow by doubling and are never shrunk, so a steady
 * workload allocates nothing after warm-up.
 *   codes[i] = (ref << CMD_OP_BITS) | op   ref = archetype id or component id
 *   ents[i]  = entity handle (count for CMD_SPAWN_MANY)
 *   slots[i] = index into `vals` or -1; written ONLY for CMD_ADD / CMD_SPAWN_MANY
 *   (add without values is queued as CMD_ADD_NOVAL and never touches slots/vals)
 * Object payloads (add values, spawnMany init) live in the compact `vals` array; `clear()`
 * overwrites its used prefix with undefined (releases refs, keeps capacity).
 * Arrays are REPLACED on growth: re-read them after anything that may push.
 */
export class CommandBuffer {
  /** Packed op + ref per command. */
  codes: Int32Array;
  /** Entity handle (or count for CMD_SPAWN_MANY). */
  ents: Uint32Array;
  /** Index into `vals`, or -1 (only for CMD_ADD / CMD_SPAWN_MANY; stale otherwise). */
  slots: Int32Array;
  /** Object payloads; entries [0, valCount) are in use. */
  readonly vals: unknown[];
  /** Number of used entries in `vals`. */
  valCount: number;
  /** Number of queued commands. */
  length: number;

  constructor() {
    const cap = COMMAND_INITIAL_CAPACITY;
    this.codes = new Int32Array(cap);
    this.ents = new Uint32Array(cap);
    this.slots = new Int32Array(cap);
    this.vals = [];
    this.valCount = 0;
    this.length = 0;
  }

  /** Appends a command without payload. */
  push(op: number, ent: number, ref: number): void {
    const i = this.length;
    if (i === this.codes.length) this._grow();
    this.codes[i] = (ref << CMD_OP_BITS) | op;
    this.ents[i] = ent;
    this.length = i + 1;
  }

  /** Appends a command with a payload (`val` undefined -> slot -1). */
  pushWithVal(op: number, ent: number, ref: number, val: unknown): void {
    const i = this.length;
    if (i === this.codes.length) this._grow();
    this.codes[i] = (ref << CMD_OP_BITS) | op;
    this.ents[i] = ent;
    let slot = -1;
    if (val !== undefined && val !== null) {
      slot = this.valCount;
      const vals = this.vals;
      if (slot < vals.length) vals[slot] = val;
      else vals.push(val);
      this.valCount = slot + 1;
    }
    this.slots[i] = slot;
    this.length = i + 1;
  }

  /** Drops all commands (releases payload refs for GC, keeps capacity). */
  clear(): void {
    this.length = 0;
    const n = this.valCount;
    if (n !== 0) {
      const vals = this.vals;
      for (let i = 0; i < n; i++) vals[i] = undefined;
      this.valCount = 0;
    }
  }

  /** @internal Drops the first `n` commands, keeping the rest in order (slots stay valid). */
  _dropFront(n: number): void {
    const len = this.length;
    if (n >= len) {
      this.clear();
      return;
    }
    this.codes.copyWithin(0, n, len);
    this.ents.copyWithin(0, n, len);
    this.slots.copyWithin(0, n, len);
    this.length = len - n;
  }

  private _grow(): void {
    const cap = this.codes.length * 2;
    const codes = new Int32Array(cap);
    codes.set(this.codes);
    this.codes = codes;
    const ents = new Uint32Array(cap);
    ents.set(this.ents);
    this.ents = ents;
    const slots = new Int32Array(cap);
    slots.set(this.slots);
    this.slots = slots;
  }
}

const DEFAULT_CAPACITY = 64;

/**
 * The ECS world: owns entities, archetypes, queries, systems and the command buffer.
 */
export class World {
  /** String interning table for `str` fields. */
  readonly strings: StringTable;

  /** @internal */ readonly _initialCapacity: number;
  /** @internal WorldOptions.shared. */
  readonly _shared: boolean;
  /** @internal */ readonly _entities: EntityAllocator;
  /** @internal All archetypes; `_archetypes[a.id] === a`. Append-only. */
  readonly _archetypes: Archetype[];
  /** @internal key -> archetype. */
  readonly _archetypeByKey: Map<string, Archetype>;
  /** @internal The archetype with no components (id 0), created in the constructor. */
  readonly _emptyArchetype: Archetype;
  /** @internal All queries ever created (never removed). */
  readonly _queries: Query[];
  /** @internal Query.keyOf(desc) -> query. */
  readonly _queryByKey: Map<string, Query>;
  /** @internal Queries with >= 1 enter/exit listener. Copy-on-write (replaced on change). */
  _eventQueries: Query[];
  /** @internal onAdd listeners by component id (undefined = none). Inner arrays copy-on-write. */
  _onAdd: (EntityCallback[] | undefined)[];
  /** @internal onRemove listeners by component id. Inner arrays copy-on-write. */
  _onRemove: (EntityCallback[] | undefined)[];
  /** @internal Total number of onAdd + onRemove listeners (0 => skip all hook dispatch). */
  _hookCount: number;
  /** @internal `_hookCount !== 0 || _eventQueries.length !== 0` (kept in sync; one load on hot paths). */
  _hasEvents: boolean;
  /** @internal Nesting depth of forEach / system runs. */
  _iterDepth: number;
  /** @internal True while flush() is applying commands. */
  _flushing: boolean;
  /** @internal Index of the command flush() is applying (valid while flushing / after a throw). */
  _flushCursor: number;
  /** @internal */ readonly _commands: CommandBuffer;
  /** @internal */ readonly _scheduler: Scheduler;
  /** @internal Next system registration sequence number. */
  _systemSeq: number;
  /** @internal Cached get() view objects by component id. */
  readonly _views: (Record<string, unknown> | undefined)[];
  /** @internal Components seen by queued add/remove commands, by id (command refs store ids). */
  readonly _components: (ComponentType | undefined)[];
  /** @internal Last component confirmed registered in `_components` (skips the array check). */
  _cmdComponent: ComponentType | null;
  /**
   * @internal Array mirror of `edgesAdd`, indexed [archetype id][component id]. Also caches
   * self-edges (`[a][c] === a` when `a` already has `c`), so add() needs no mask test.
   * Inner arrays are packed (padded with undefined).
   */
  readonly _edgeAdd: (Archetype | undefined)[][];
  /** @internal Array mirror of `edgesRemove` (self-edge when the component is absent). */
  readonly _edgeRemove: (Archetype | undefined)[][];

  constructor(options?: WorldOptions) {
    const requested = options && options.initialCapacity;
    const cap =
      typeof requested === 'number' && requested >= 1 && requested === requested
        ? Math.floor(requested)
        : DEFAULT_CAPACITY;
    this.strings = new StringTable();
    this._initialCapacity = cap;
    this._shared = !!(options && options.shared);
    this._archetypes = [];
    this._entities = new EntityAllocator(cap, this._archetypes);
    this._archetypeByKey = new Map();
    this._queries = [];
    this._queryByKey = new Map();
    this._eventQueries = [];
    this._onAdd = [];
    this._onRemove = [];
    this._hookCount = 0;
    this._hasEvents = false;
    this._iterDepth = 0;
    this._flushing = false;
    this._flushCursor = 0;
    this._commands = new CommandBuffer();
    this._scheduler = new Scheduler();
    this._systemSeq = 0;
    this._views = [];
    this._components = [];
    this._cmdComponent = null;
    this._edgeAdd = [[]];
    this._edgeRemove = [[]];
    const empty = new Archetype(0, [], '', cap, this._shared);
    this._archetypes.push(empty);
    this._archetypeByKey.set('', empty);
    this._emptyArchetype = empty;
  }

  // ------------------------------------------------------------------ archetypes

  /** Get-or-create the archetype for this component set (order/duplicates ignored). */
  archetype(...components: ComponentType[]): Archetype {
    return this._getOrCreateArchetype(components);
  }

  // ------------------------------------------------------------------ entities

  /**
   * Creates an entity in the given archetype (or component list, or the empty archetype).
   * Fields are zeroed. While deferring, returns a reserved live id immediately and queues
   * the placement.
   */
  spawn(archetypeOrComponents?: Archetype | ComponentType[]): number {
    let arch: Archetype;
    if (archetypeOrComponents === undefined) {
      arch = this._emptyArchetype;
    } else if (Array.isArray(archetypeOrComponents)) {
      arch = this._getOrCreateArchetype(archetypeOrComponents);
    } else {
      arch = archetypeOrComponents;
      this._checkArchetype(arch);
    }
    const e = this._entities.create();
    if (this._iterDepth > 0 || this._flushing) {
      this._commands.push(CMD_SPAWN, e, arch.id);
    } else {
      this._place(e, arch);
    }
    return e;
  }

  /** Creates `count` entities in `archetype`, growing storage once. Deferred while iterating. */
  spawnMany(archetype: Archetype, count: number, init?: SpawnInitFn): void {
    this._checkArchetype(archetype);
    count = Math.floor(count);
    if (!(count > 0)) return;
    if (this._iterDepth > 0 || this._flushing) {
      this._commands.pushWithVal(CMD_SPAWN_MANY, count, archetype.id, init);
    } else {
      this._spawnManyNow(archetype, count, init);
    }
  }

  /** Destroys an entity. Silently ignored if dead. Deferred while iterating. */
  destroy(entity: number): void {
    if (!this._entities.isAlive(entity)) return;
    if (this._iterDepth > 0 || this._flushing) {
      this._commands.push(CMD_DESTROY, entity, 0);
    } else {
      this._destroyNow(entity);
    }
  }

  /** True if the entity is alive (entities spawned while deferring are alive immediately). */
  isAlive(entity: number): boolean {
    return this._entities.isAlive(entity);
  }

  /**
   * Adds component `C`, optionally setting values. If the entity already has `C`, only sets
   * values. Deferred while iterating (values object is read at flush time; do not mutate it).
   * @throws Error if the entity is dead.
   */
  add<S extends Schema>(entity: number, C: ComponentType<S>, values?: Partial<ValuesOf<S>>): void {
    // Fast path: inlined EntityAllocator.locate for a placed entity in an inline-tag archetype
    // (id < TAG_BIG). Kept small (rare cases in _addSlow) so V8 inlines it into callers' loops.
    const alloc = this._entities;
    const idx = entity & INDEX_MASK;
    if (entity >= 0 && idx < alloc.next) {
      const slots = alloc.slot;
      const s = slots[idx];
      const aid = s >>> INDEX_BITS;
      const r = s & INDEX_MASK;
      let from: Archetype;
      if (aid < TAG_BIG && (from = this._archetypes[aid]).entities[r] === entity) {
        const id = C.id;
        if (this._iterDepth > 0 || this._flushing) {
          if (C !== this._cmdComponent) this._registerComponent(C as ComponentType);
          if (values === undefined) this._commands.push(CMD_ADD_NOVAL, entity, id);
          else this._commands.pushWithVal(CMD_ADD, entity, id, values);
          return;
        }
        if (values !== undefined) {
          this._addNow(entity, C as ComponentType, values as Record<string, unknown>);
          return;
        }
        const edges = this._edgeAdd[aid];
        let to = id < edges.length ? edges[id] : undefined;
        if (to === undefined) to = this._archetypeWith(from, C as ComponentType);
        if (to === from) return;
        // _moveRow, inlined by hand (measurably faster than the call on this path).
        const r2 = to.pushRowFrom(from, r, entity);
        const moved = from.swapRemove(r);
        // The moved entity lives in `from` too, so its slot has the same tag bits as `s`.
        if (moved !== -1) slots[moved & INDEX_MASK] = (s & HIGH_BITS) | r;
        const tid = to.id;
        if (tid < TAG_BIG) slots[idx] = (tid << INDEX_BITS) | r2;
        else alloc.setLocation(entity, tid, r2);
        if (this._hasEvents) {
          this._fireEvents(entity, from, to);
        }
        return;
      }
    }
    this._addSlow(entity, C as ComponentType, values as Record<string, unknown> | undefined);
  }

  /** Removes component `C`. No-op if dead or absent. Deferred while iterating. */
  remove(entity: number, C: ComponentType): void {
    // Fast path as in add(); rare cases in _removeSlow.
    const alloc = this._entities;
    const idx = entity & INDEX_MASK;
    if (entity >= 0 && idx < alloc.next) {
      const slots = alloc.slot;
      const s = slots[idx];
      const aid = s >>> INDEX_BITS;
      const r = s & INDEX_MASK;
      let from: Archetype;
      if (aid < TAG_BIG && (from = this._archetypes[aid]).entities[r] === entity) {
        const id = C.id;
        if (this._iterDepth > 0 || this._flushing) {
          if (C !== this._cmdComponent) this._registerComponent(C);
          this._commands.push(CMD_REMOVE, entity, id);
          return;
        }
        const edges = this._edgeRemove[aid];
        let to = id < edges.length ? edges[id] : undefined;
        if (to === undefined) to = this._archetypeWithout(from, C);
        if (to === from) return;
        // _moveRow, inlined by hand.
        const r2 = to.pushRowFrom(from, r, entity);
        const moved = from.swapRemove(r);
        if (moved !== -1) slots[moved & INDEX_MASK] = (s & HIGH_BITS) | r;
        const tid = to.id;
        if (tid < TAG_BIG) slots[idx] = (tid << INDEX_BITS) | r2;
        else alloc.setLocation(entity, tid, r2);
        if (this._hasEvents) {
          this._fireEvents(entity, from, to);
        }
        return;
      }
    }
    this._removeSlow(entity, C);
  }

  /** add() for entities that are dead (throws), pending or in a big-id archetype. */
  private _addSlow(entity: number, C: ComponentType, values: Record<string, unknown> | undefined): void {
    if (this._entities.locate(entity) === LOCATION_FREE) {
      throw new Error(`CozyECS: add(${C.name}) on dead entity ${entity}`);
    }
    if (this._iterDepth > 0 || this._flushing) {
      if (C !== this._cmdComponent) this._registerComponent(C);
      if (values === undefined) this._commands.push(CMD_ADD_NOVAL, entity, C.id);
      else this._commands.pushWithVal(CMD_ADD, entity, C.id, values);
      return;
    }
    this._addNow(entity, C, values);
  }

  /** remove() for entities that are dead (ignored), pending or in a big-id archetype. */
  private _removeSlow(entity: number, C: ComponentType): void {
    const aid = this._entities.locate(entity);
    if (aid === LOCATION_FREE) return;
    if (this._iterDepth > 0 || this._flushing) {
      if (C !== this._cmdComponent) this._registerComponent(C);
      this._commands.push(CMD_REMOVE, entity, C.id);
      return;
    }
    if (aid >= 0) this._removeNow(entity, C);
  }

  /** True if alive, placed, and its archetype contains `C`. */
  has(entity: number, C: ComponentType): boolean {
    const aid = this._entities.locate(entity);
    return aid >= 0 && maskHas(this._archetypes[aid].mask, C.id);
  }

  /**
   * Writes fields immediately. `str` values are interned, `bool` -> 0/1, unknown keys ignored.
   * @throws Error if the entity is dead or does not (yet) have `C`.
   */
  set<S extends Schema>(entity: number, C: ComponentType<S>, values: Partial<ValuesOf<S>>): void {
    const alloc = this._entities;
    const aid = alloc.locate(entity);
    if (aid === LOCATION_FREE) {
      throw new Error(`CozyECS: set(${C.name}) on dead entity ${entity}`);
    }
    const idx = entity & INDEX_MASK;
    if (aid < 0 || !maskHas(this._archetypes[aid].mask, C.id)) {
      throw new Error(`CozyECS: set(${C.name}) on entity ${entity} that does not have it`);
    }
    if (values !== undefined && values !== null) {
      this._writeValues(this._archetypes[aid], alloc.slot[idx] & INDEX_MASK, C as ComponentType, values as Record<string, unknown>);
    }
  }

  /**
   * Reads all fields into a per-component cached view object that is OVERWRITTEN on every
   * call for the same component: copy what you need, do not keep the reference.
   * @returns undefined if dead or absent.
   */
  get<S extends Schema>(entity: number, C: ComponentType<S>): Readonly<ValuesOf<S>> | undefined {
    const alloc = this._entities;
    const aid = alloc.locate(entity);
    if (aid < 0) return undefined;
    const idx = entity & INDEX_MASK;
    const arch = this._archetypes[aid];
    const id = C.id;
    if (!maskHas(arch.mask, id)) return undefined;
    const views = this._views;
    let view = id < views.length ? views[id] : undefined;
    if (view === undefined) view = this._createView(C as ComponentType);
    const cols = arch.columns[id];
    if (cols !== undefined) {
      const row = alloc.slot[idx] & INDEX_MASK;
      const keys = C.keys;
      const tokens = C.tokens;
      for (let i = 0; i < keys.length; i++) {
        const key = keys[i];
        const raw = cols[key][row];
        const code = tokens[i].code;
        view[key] = code === FIELD_STR ? this.strings.get(raw) : code === FIELD_BOOL ? raw !== 0 : raw;
      }
    }
    return view as unknown as Readonly<ValuesOf<S>>;
  }

  /** Raw column value (str -> interned id, bool -> 0/1). Undefined behaviour if absent. */
  getField<S extends Schema>(entity: number, C: ComponentType<S>, key: keyof S & string): number {
    const alloc = this._entities;
    const idx = entity & INDEX_MASK;
    const cols = this._archetypes[alloc.archetypeOf(idx)].columns[C.id] as Record<string, TypedArray>;
    return cols[key][alloc.slot[idx] & INDEX_MASK];
  }

  /**
   * Enables/disables an enableable component. Immediate; never fires enter/exit/add/remove.
   * @throws Error if `C` is not enableable, or the entity is dead or lacks `C`.
   */
  enable(entity: number, C: ComponentType, on: boolean = true): void {
    if (!C.enableable) throw new Error(`CozyECS: component ${C.name} is not enableable`);
    const alloc = this._entities;
    const aid = alloc.locate(entity);
    if (aid === LOCATION_FREE) {
      throw new Error(`CozyECS: enable(${C.name}) on dead entity ${entity}`);
    }
    const idx = entity & INDEX_MASK;
    const flags = aid >= 0 ? this._archetypes[aid].enabled[C.id] : undefined;
    if (flags === undefined) {
      throw new Error(`CozyECS: enable(${C.name}) on entity ${entity} that does not have it`);
    }
    flags[alloc.slot[idx] & INDEX_MASK] = on ? 1 : 0;
  }

  /** False if dead or absent; true if present and not enableable; otherwise the flag. */
  isEnabled(entity: number, C: ComponentType): boolean {
    const alloc = this._entities;
    const aid = alloc.locate(entity);
    if (aid < 0) return false;
    const idx = entity & INDEX_MASK;
    const arch = this._archetypes[aid];
    if (!maskHas(arch.mask, C.id)) return false;
    if (!C.enableable) return true;
    const flags = arch.enabled[C.id];
    return flags !== undefined && flags[alloc.slot[idx] & INDEX_MASK] === 1;
  }

  // ------------------------------------------------------------------ hooks

  /** Fires after `C` is added to an entity (add, spawn, spawnMany). */
  onAdd(C: ComponentType, cb: EntityCallback): () => void {
    return this._subscribeHook(this._onAdd, C.id, cb);
  }

  /** Fires after `C` is removed from an entity (remove, destroy). Data is no longer readable. */
  onRemove(C: ComponentType, cb: EntityCallback): () => void {
    return this._subscribeHook(this._onRemove, C.id, cb);
  }

  // ------------------------------------------------------------------ queries

  /** Get-or-create a cached query. */
  query(desc: QueryDesc): Query {
    const key = Query.keyOf(desc);
    const cached = this._queryByKey.get(key);
    if (cached !== undefined) return cached;
    const q = new Query(this, desc);
    this._queries.push(q);
    this._queryByKey.set(key, q);
    return q;
  }

  // ------------------------------------------------------------------ systems

  /** Registers a function system. */
  system(name: string, options: FunctionSystemOptions, fn: SystemFn): SystemHandle {
    const opts = options || {};
    const qOpt = opts.query;
    const q = qOpt === undefined || qOpt === null ? undefined : qOpt instanceof Query ? qOpt : this.query(qOpt);
    const s = new FunctionSystem(
      this,
      name,
      opts.group === undefined ? 'update' : opts.group,
      opts.order === undefined ? 0 : opts.order,
      q,
      fn,
      this._systemSeq++,
    );
    this._scheduler.add(s);
    return s;
  }

  /** Instantiates and registers a class system, then calls onCreate(). */
  addSystem<T extends System>(SystemCtor: SystemClass<T>, options?: SystemOptions): T {
    const s = new SystemCtor(this);
    s.name = s.name || SystemCtor.name || 'System';
    s.group = options && options.group !== undefined ? options.group : 'update';
    s.order = options && options.order !== undefined ? options.order : 0;
    s._seq = this._systemSeq++;
    this._scheduler.add(s);
    s.onCreate();
    return s;
  }

  /** Unregisters a system (sets enabled = false); calls onDestroy() for class systems. */
  removeSystem(system: SystemHandle | System): void {
    if (!this._scheduler.remove(system)) return;
    system.enabled = false;
    if (system instanceof System) system.onDestroy();
  }

  /**
   * Runs enabled systems of `group` in (order, registration) order. Each system runs as one
   * iteration level, so its structural changes are flushed right after it (when update is not
   * itself nested in an iteration). If a system throws, pending commands are still flushed
   * and the error propagates (remaining systems do not run).
   */
  update(dt: number = 0, group: string = 'update'): void {
    const list = this._scheduler.group(group);
    const n = list.length;
    if (n === 0) return;
    const base = this._iterDepth;
    const cmds = this._commands;
    // One try/finally for the whole group (not per system). `_iterDepth !== base` in the
    // finally block means a system threw while running.
    try {
      for (let i = 0; i < n; i++) {
        const s = list[i];
        if (!s.enabled) continue;
        this._iterDepth = base + 1;
        // system.ts keeps the system-body call sites megamorphic, so bodies are not inlined
        // into this try region (inlined loops there compile ~1.7x slower).
        s._run(dt);
        this._iterDepth = base;
        if (base === 0 && cmds.length !== 0) this.flush();
      }
    } finally {
      if (this._iterDepth !== base) {
        this._iterDepth = base;
        if (base === 0 && cmds.length !== 0) this.flush();
      }
    }
  }

  /**
   * Applies all pending structural commands. No-op while already flushing, and while
   * iterating (the outermost iteration flushes when it ends).
   */
  flush(): void {
    if (this._flushing || this._iterDepth > 0) return;
    const cmds = this._commands;
    if (cmds.length === 0) return;
    this._flushing = true;
    let i = 0;
    try {
      for (;;) {
        const n = cmds.length;
        if (i >= n) break;
        if (!this._hasEvents) {
          i = this._flushPlain(i, n);
        } else {
          this._flushCursor = i;
          this._applyCommand(i);
          i++;
        }
      }
      cmds.clear();
    } finally {
      this._flushing = false;
      // On throw: drop applied commands and the failing one, keep the rest queued.
      // `_flushCursor` is the index of the command being applied (set before applying it).
      if (cmds.length > 0) cmds._dropFront(this._flushCursor + 1);
    }
  }

  /**
   * Applies commands [i, n) while no listeners exist: nothing here runs user code except a
   * spawnMany init and add-values conversion, so no command can be queued and the typed arrays
   * are stable. Stops right after a spawnMany (listeners may have been added). Returns the next
   * index. No try/catch here (code inside a try region compiles measurably slower).
   *
   * Consecutive commands with the same code (op + ref) are applied as one run. `_flushCursor`
   * is published at the start of each run and right before anything that can throw inside a
   * run. If an unexpected error escaped mid-run, the already-applied commands of that run would
   * stay queued; replaying them is harmless (every op re-checks liveness / placement / edges).
   */
  private _flushPlain(i: number, n: number): number {
    const cmds = this._commands;
    const alloc = this._entities;
    const codes = cmds.codes;
    const ents = cmds.ents;
    while (i < n) {
      this._flushCursor = i;
      const code = codes[i];
      const op = code & CMD_OP_MASK;
      if (op === CMD_ADD_NOVAL || op === CMD_REMOVE) {
        i = this._flushMoveRun(i, n, code);
      } else if (op === CMD_DESTROY) {
        i = this._flushDestroyRun(i, n, code);
      } else if (op === CMD_SPAWN) {
        i = this._flushSpawnRun(i, n, code);
      } else if (op === CMD_ADD) {
        const e = ents[i];
        if (alloc.isAlive(e)) {
          const slot = cmds.slots[i];
          this._addNow(
            e,
            this._components[code >> CMD_OP_BITS] as ComponentType,
            slot < 0 ? undefined : (cmds.vals[slot] as Record<string, unknown>),
          );
        }
        i++;
      } else {
        this._applyCommand(i);
        return i + 1;
      }
    }
    return i;
  }

  /** Applies the run of CMD_DESTROY commands starting at `i`. Returns the next index. */
  private _flushDestroyRun(i: number, n: number, code: number): number {
    const cmds = this._commands;
    const codes = cmds.codes;
    const ents = cmds.ents;
    const alloc = this._entities;
    const archetypes = this._archetypes;
    for (; i < n; i++) {
      if (codes[i] !== code) break;
      const e = ents[i];
      const idx = e & INDEX_MASK;
      if (idx >= alloc.next) continue;
      // _destroyNow inlined (no listeners while in _flushPlain). release() may grow freeStack
      // only, so `slots` stays valid within one iteration.
      const slots = alloc.slot;
      const s = slots[idx];
      const tag = s >>> INDEX_BITS;
      if (tag < TAG_BIG) {
        const from = archetypes[tag];
        const r = s & INDEX_MASK;
        if (from.entities[r] !== e) continue;
        const moved = from.swapRemove(r);
        if (moved !== -1) slots[moved & INDEX_MASK] = (s & HIGH_BITS) | r;
        alloc.release(e);
      } else if (alloc._locateSlow(e, idx, s, tag) !== LOCATION_FREE) {
        this._destroyNow(e);
      }
    }
    return i;
  }

  /** Applies the run of CMD_SPAWN commands (same archetype) starting at `i`. Returns the next index. */
  private _flushSpawnRun(i: number, n: number, code: number): number {
    const cmds = this._commands;
    const codes = cmds.codes;
    const ents = cmds.ents;
    const alloc = this._entities;
    const arch = this._archetypes[code >> CMD_OP_BITS];
    const archId = arch.id;
    const pendingBits = TAG_PENDING << INDEX_BITS;
    for (; i < n; i++) {
      if (codes[i] !== code) break;
      const e = ents[i];
      const idx = e & INDEX_MASK;
      // Inlined `locate(e) === LOCATION_PENDING`: tag PENDING and the handle's generation.
      if (idx < alloc.next && alloc.slot[idx] === (pendingBits | (e >>> INDEX_BITS))) {
        alloc.setLocation(e, archId, arch.pushRow(e));
      }
    }
    return i;
  }

  /**
   * Applies the run of CMD_ADD_NOVAL / CMD_REMOVE commands starting at `i` that all equal
   * `code` (same op and component). The (source archetype -> destination) edge is cached for
   * consecutive entities in the same archetype; the entity's archetype is re-read every time.
   * Nothing here allocates entity ids, so the allocator arrays are stable. Returns the next index.
   */
  private _flushMoveRun(i: number, n: number, code: number): number {
    const cmds = this._commands;
    const codes = cmds.codes;
    const ents = cmds.ents;
    const alloc = this._entities;
    const slots = alloc.slot;
    const next = alloc.next;
    const archetypes = this._archetypes;
    const isAdd = (code & CMD_OP_MASK) === CMD_ADD_NOVAL;
    const id = code >> CMD_OP_BITS;
    const table = isAdd ? this._edgeAdd : this._edgeRemove;
    let lastAid = -1;
    let from: Archetype = this._emptyArchetype;
    let to: Archetype = from;
    let fromEnts = from.entities;
    for (; i < n; i++) {
      if (codes[i] !== code) break;
      const e = ents[i];
      const idx = e & INDEX_MASK;
      if (idx >= next) continue;
      const s = slots[idx];
      let aid = s >>> INDEX_BITS;
      if (aid < TAG_BIG) {
        // `fromEnts` is `from.entities` while aid === lastAid (moves only grow `to`).
        if ((aid === lastAid ? fromEnts : archetypes[aid].entities)[s & INDEX_MASK] !== e) continue;
      } else {
        aid = alloc._locateSlow(e, idx, s, aid);
      }
      if (aid < 0) {
        if (aid === LOCATION_FREE || !isAdd) continue;
        this._flushCursor = i;
        const C = this._components[id] as ComponentType;
        throw new Error(`CozyECS: add(${C.name}) on entity ${e} whose spawn is still pending`);
      }
      if (aid !== lastAid) {
        from = archetypes[aid];
        fromEnts = from.entities;
        const edges = table[aid];
        let t = id < edges.length ? edges[id] : undefined;
        if (t === undefined) {
          const C = this._components[id] as ComponentType;
          t = isAdd ? this._archetypeWith(from, C) : this._archetypeWithout(from, C);
        }
        to = t;
        lastAid = aid;
      }
      if (to !== from) {
        const r = s & INDEX_MASK;
        const r2 = to.pushRowFrom(from, r, e);
        const moved = from.swapRemove(r);
        // The moved entity stays in `from` (same tag bits as `s`): only its row changes.
        if (moved !== -1) slots[moved & INDEX_MASK] = (s & HIGH_BITS) | r;
        const tid = to.id;
        if (tid < TAG_BIG) slots[idx] = (tid << INDEX_BITS) | r2;
        else alloc.setLocation(e, tid, r2);
      }
    }
    return i;
  }

  /** Applies command `i` (generic path; callbacks may queue more commands). */
  private _applyCommand(i: number): void {
    const cmds = this._commands;
    const alloc = this._entities;
    const code = cmds.codes[i];
    const ref = code >> CMD_OP_BITS;
    const e = cmds.ents[i];
    switch (code & CMD_OP_MASK) {
      case CMD_SPAWN:
        if (alloc.locate(e) === LOCATION_PENDING) {
          this._place(e, this._archetypes[ref]);
        }
        break;
      case CMD_SPAWN_MANY: {
        const slot = cmds.slots[i];
        this._spawnManyNow(this._archetypes[ref], e, slot < 0 ? undefined : (cmds.vals[slot] as SpawnInitFn));
        break;
      }
      case CMD_DESTROY:
        if (alloc.isAlive(e)) this._destroyNow(e);
        break;
      case CMD_ADD_NOVAL:
        if (alloc.isAlive(e)) this._addNow(e, this._components[ref] as ComponentType, undefined);
        break;
      case CMD_ADD:
        if (alloc.isAlive(e)) {
          const slot = cmds.slots[i];
          this._addNow(
            e,
            this._components[ref] as ComponentType,
            slot < 0 ? undefined : (cmds.vals[slot] as Record<string, unknown>),
          );
        }
        break;
      case CMD_REMOVE:
        if (alloc.isAlive(e)) this._removeNow(e, this._components[ref] as ComponentType);
        break;
    }
  }

  // ------------------------------------------------------------------ internals

  /** @internal True while inside a forEach or a system run. */
  get _isIterating(): boolean {
    return this._iterDepth > 0;
  }

  /** @internal True when structural changes must be queued: `_iterDepth > 0 || _flushing`. */
  get _deferring(): boolean {
    return this._iterDepth > 0 || this._flushing;
  }

  /** @internal `_iterDepth++`. */
  _beginIteration(): void {
    this._iterDepth++;
  }

  /** @internal `_iterDepth--`; when it reaches 0, calls flush() if commands are pending. */
  _endIteration(): void {
    const d = this._iterDepth - 1;
    if (d > 0) {
      this._iterDepth = d;
      return;
    }
    this._iterDepth = 0;
    if (this._commands.length !== 0) this.flush();
  }

  /** @internal Adds/removes `query` from `_eventQueries` (copy-on-write; idempotent). */
  _setQueryListening(query: Query, listening: boolean): void {
    const cur = this._eventQueries;
    const at = cur.indexOf(query);
    if (listening) {
      if (at !== -1) return;
      const next = cur.slice();
      next.push(query);
      this._eventQueries = next;
      this._hasEvents = true;
    } else {
      if (at === -1) return;
      const next = cur.slice();
      next.splice(at, 1);
      this._eventQueries = next;
      this._hasEvents = this._hookCount !== 0 || next.length !== 0;
    }
  }

  /** @internal Get-or-create by normalized list; registers new archetypes with all queries. */
  _getOrCreateArchetype(components: readonly ComponentType[]): Archetype {
    const sorted = normalizeComponents(components);
    const key = componentsKey(sorted);
    const found = this._archetypeByKey.get(key);
    if (found !== undefined) return found;
    const arch = new Archetype(this._archetypes.length, sorted, key, this._initialCapacity, this._shared);
    this._archetypes.push(arch);
    if (arch.id === TAG_BIG) this._entities.enableBig();
    this._edgeAdd.push([]);
    this._edgeRemove.push([]);
    this._archetypeByKey.set(key, arch);
    const queries = this._queries;
    for (let i = 0; i < queries.length; i++) queries[i]._addArchetype(arch);
    return arch;
  }

  /** @internal `from + C` (returns `from` if it already has C). Cached in `_edgeAdd`/`edgesAdd`. */
  _archetypeWith(from: Archetype, C: ComponentType): Archetype {
    const id = C.id;
    const edges = this._edgeAdd[from.id];
    const cached = id < edges.length ? edges[id] : undefined;
    if (cached !== undefined) return cached;
    let to: Archetype;
    if (maskHas(from.mask, id)) {
      to = from;
    } else {
      to = from.edgesAdd.get(id) as Archetype;
      if (to === undefined) {
        const list = from.components.slice();
        list.push(C);
        to = this._getOrCreateArchetype(list);
        from.edgesAdd.set(id, to);
        to.edgesRemove.set(id, from);
        setEdge(this._edgeRemove[to.id], id, from);
      }
    }
    setEdge(edges, id, to);
    return to;
  }

  /** @internal `from - C` (returns `from` if it lacks C). Cached in `_edgeRemove`/`edgesRemove`. */
  _archetypeWithout(from: Archetype, C: ComponentType): Archetype {
    const id = C.id;
    const edges = this._edgeRemove[from.id];
    const cached = id < edges.length ? edges[id] : undefined;
    if (cached !== undefined) return cached;
    let to: Archetype;
    if (!maskHas(from.mask, id)) {
      to = from;
    } else {
      to = from.edgesRemove.get(id) as Archetype;
      if (to === undefined) {
        const src = from.components;
        const list: ComponentType[] = [];
        for (let i = 0; i < src.length; i++) if (src[i] !== C) list.push(src[i]);
        to = this._getOrCreateArchetype(list);
        from.edgesRemove.set(id, to);
        to.edgesAdd.set(id, from);
        setEdge(this._edgeAdd[to.id], id, from);
      }
    }
    setEdge(edges, id, to);
    return to;
  }

  /** @internal Places an already-allocated (pending) entity into `archetype`, then fires events. */
  _place(entity: number, archetype: Archetype): void {
    const row = archetype.pushRow(entity);
    this._entities.setLocation(entity, archetype.id, row);
    if (this._hasEvents) {
      this._fireEvents(entity, null, archetype);
    }
  }

  /** @internal Moves a placed entity to `to`, preserving shared data, then fires events. */
  _move(entity: number, to: Archetype): void {
    const alloc = this._entities;
    const idx = entity & INDEX_MASK;
    const from = this._archetypes[alloc.archetypeOf(idx)];
    if (from === to) return;
    this._moveRow(alloc, idx, entity, from, to);
    if (this._hasEvents) {
      this._fireEvents(entity, from, to);
    }
  }

  /** @internal Removes the entity's row (if placed), releases the id, then fires events. */
  _destroyNow(entity: number): void {
    const alloc = this._entities;
    const idx = entity & INDEX_MASK;
    const aid = alloc.archetypeOf(idx);
    if (aid < 0) {
      alloc.release(entity);
      return;
    }
    const from = this._archetypes[aid];
    const slots = alloc.slot;
    const s = slots[idx];
    const r = s & INDEX_MASK;
    const moved = from.swapRemove(r);
    // The moved entity already lives in `aid` (same tag bits as `s`): only its row changes.
    if (moved !== -1) slots[moved & INDEX_MASK] = (s & HIGH_BITS) | r;
    alloc.release(entity);
    if (this._hasEvents) {
      this._fireEvents(entity, from, null);
    }
  }

  /** @internal Fires onRemove/onAdd hooks, then query exit/enter, for a from -> to change. */
  _fireEvents(entity: number, from: Archetype | null, to: Archetype | null): void {
    if (this._hookCount !== 0) {
      if (from !== null) {
        const hooks = this._onRemove;
        const comps = from.components;
        for (let i = 0; i < comps.length; i++) {
          const id = comps[i].id;
          if (id >= hooks.length) break; // sorted by id: nothing further has hooks
          const list = hooks[id];
          if (list === undefined || (to !== null && maskHas(to.mask, id))) continue;
          for (let k = 0; k < list.length; k++) list[k](entity);
        }
      }
      if (to !== null) {
        const hooks = this._onAdd;
        const comps = to.components;
        for (let i = 0; i < comps.length; i++) {
          const id = comps[i].id;
          if (id >= hooks.length) break;
          const list = hooks[id];
          if (list === undefined || (from !== null && maskHas(from.mask, id))) continue;
          for (let k = 0; k < list.length; k++) list[k](entity);
        }
      }
    }
    const queries = this._eventQueries;
    for (let i = 0; i < queries.length; i++) queries[i]._transition(entity, from, to);
  }

  // ------------------------------------------------------------------ private helpers

  /** Moves the placed entity at `idx` from `from` to `to` (from !== to). No events. */
  private _moveRow(alloc: EntityAllocator, idx: number, entity: number, from: Archetype, to: Archetype): void {
    const slots = alloc.slot;
    const s = slots[idx];
    const r = s & INDEX_MASK;
    const r2 = to.pushRowFrom(from, r, entity);
    const moved = from.swapRemove(r);
    // The moved entity stays in `from` (same tag bits as `s`): only its row changes.
    if (moved !== -1) slots[moved & INDEX_MASK] = (s & HIGH_BITS) | r;
    const tid = to.id;
    if (tid < TAG_BIG) slots[idx] = (tid << INDEX_BITS) | r2;
    else alloc.setLocation(entity, tid, r2);
  }

  private _addNow(entity: number, C: ComponentType, values: Record<string, unknown> | undefined): void {
    const alloc = this._entities;
    const idx = entity & INDEX_MASK;
    const aid = alloc.archetypeOf(idx);
    if (aid < 0) {
      throw new Error(`CozyECS: add(${C.name}) on entity ${entity} whose spawn is still pending`);
    }
    const from = this._archetypes[aid];
    const id = C.id;
    const edges = this._edgeAdd[aid];
    let to = id < edges.length ? edges[id] : undefined;
    if (to === undefined) to = this._archetypeWith(from, C);
    if (to !== from) {
      this._moveRow(alloc, idx, entity, from, to);
      if (values !== undefined && values !== null) this._writeValues(to, alloc.slot[idx] & INDEX_MASK, C, values);
      if (this._hasEvents) {
        this._fireEvents(entity, from, to);
      }
    } else if (values !== undefined && values !== null) {
      this._writeValues(from, alloc.slot[idx] & INDEX_MASK, C, values);
    }
  }

  private _removeNow(entity: number, C: ComponentType): void {
    const alloc = this._entities;
    const idx = entity & INDEX_MASK;
    const aid = alloc.archetypeOf(idx);
    if (aid < 0) return;
    const from = this._archetypes[aid];
    const id = C.id;
    const edges = this._edgeRemove[aid];
    let to = id < edges.length ? edges[id] : undefined;
    if (to === undefined) to = this._archetypeWithout(from, C);
    if (to === from) return;
    this._moveRow(alloc, idx, entity, from, to);
    if (this._hasEvents) {
      this._fireEvents(entity, from, to);
    }
  }

  private _spawnManyNow(arch: Archetype, count: number, init: SpawnInitFn | undefined): void {
    arch.ensureCapacity(arch.count + count);
    const alloc = this._entities;
    alloc.reserve(count);
    const archId = arch.id;
    const events = this._hasEvents;
    if (init === undefined && !events) {
      for (let i = 0; i < count; i++) {
        const e = alloc.create();
        alloc.setLocation(e, archId, arch.pushRow(e));
      }
      return;
    }
    for (let i = 0; i < count; i++) {
      const e = alloc.create();
      const row = arch.pushRow(e);
      alloc.setLocation(e, archId, row);
      if (init !== undefined) init(arch, row, i);
      if (events) {
        const aid = alloc.locate(e);
        if (aid >= 0) this._fireEvents(e, null, this._archetypes[aid]);
      }
    }
  }

  private _writeValues(arch: Archetype, row: number, C: ComponentType, values: Record<string, unknown>): void {
    const cols = arch.columns[C.id];
    if (cols === undefined) return;
    const keys = C.keys;
    const tokens = C.tokens;
    for (let i = 0; i < keys.length; i++) {
      const key = keys[i];
      const v = values[key];
      if (v === undefined) continue;
      const code = tokens[i].code;
      cols[key][row] =
        code === FIELD_STR
          ? this.strings.intern(typeof v === 'string' ? v : String(v))
          : code === FIELD_BOOL
            ? v
              ? 1
              : 0
            : (v as number);
    }
  }

  private _createView(C: ComponentType): Record<string, unknown> {
    const view: Record<string, unknown> = {};
    const keys = C.keys;
    const tokens = C.tokens;
    for (let i = 0; i < keys.length; i++) {
      const code = tokens[i].code;
      view[keys[i]] = code === FIELD_STR ? '' : code === FIELD_BOOL ? false : 0;
    }
    const views = this._views;
    while (views.length <= C.id) views.push(undefined);
    views[C.id] = view;
    return view;
  }

  private _subscribeHook(table: (EntityCallback[] | undefined)[], id: number, cb: EntityCallback): () => void {
    while (table.length <= id) table.push(undefined);
    const cur = table[id];
    const next = cur === undefined ? [] : cur.slice();
    next.push(cb);
    table[id] = next;
    this._hookCount++;
    this._hasEvents = true;
    let active = true;
    return () => {
      if (!active) return;
      active = false;
      const list = table[id];
      if (list === undefined) return;
      const at = list.indexOf(cb);
      if (at === -1) return;
      if (list.length === 1) {
        table[id] = undefined;
      } else {
        const copy = list.slice();
        copy.splice(at, 1);
        table[id] = copy;
      }
      this._hookCount--;
      this._hasEvents = this._hookCount !== 0 || this._eventQueries.length !== 0;
    };
  }

  /** Ensures `_components[C.id] === C` (entries are never removed) and caches C as checked. */
  private _registerComponent(C: ComponentType): void {
    const comps = this._components;
    const id = C.id;
    if (id >= comps.length || comps[id] !== C) {
      while (comps.length <= id) comps.push(undefined);
      comps[id] = C;
    }
    this._cmdComponent = C;
  }

  private _checkArchetype(arch: Archetype): void {
    if (this._archetypes[arch.id] !== arch) {
      throw new Error('CozyECS: archetype belongs to a different World');
    }
  }
}

/** Sets `edges[id] = to`, padding with undefined so the array stays packed. */
function setEdge(edges: (Archetype | undefined)[], id: number, to: Archetype): void {
  while (edges.length <= id) edges.push(undefined);
  edges[id] = to;
}
