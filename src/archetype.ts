import type { ComponentType } from './component';
import { createMask } from './component';
import { MAX_ENTITIES } from './entities';
import type { ColumnsOf, Schema, TypedArray, TypedArrayConstructor } from './types';

// ---------------------------------------------------------------------------
// Storage layout
//
// Every archetype owns ONE ArrayBuffer (or SharedArrayBuffer, see `shared`)
// holding all of its per-row data as structure-of-arrays:
//
//   [f64 columns][u32 entities][f32/i32/u32/str columns][i16/u16 columns]
//   [i8/u8/bool columns][enabled flag columns]
//
// A "slot" is one column. Slots are grouped by element width (8, 4, 2, 1
// bytes) in descending order, so every column's byte offset
// (capacity * bytes-per-row-before-it) is a multiple of its element width and
// typed views can be created over it without padding.
//
// Row operations never touch the per-column views. They go through four views
// over the WHOLE buffer, one per width (`_v64`, `_v32`, `_v16`, `_v8`), plus a
// per-slot unit offset `_base[slot]` (column start in that view's units).
// Copies are bit-exact (f32 is moved as Int32, f64 as Float64), each loop body
// touches a single TypedArray type (monomorphic), and there is no per-row
// string or Map lookup.
//
// Growth allocates one new buffer and copies each column once. Column views
// obtained through `col()` / `enabledArray()` / `entities` are REPLACED on
// growth: re-fetch them after structural changes (systems re-fetch per tick).
// Growth policy (see `nextCapacity`): doubling below 65536 rows, then x1.25
// rounded up to a multiple of 4096 (clamped to MAX_ENTITIES), so large tables
// waste at most ~20% instead of up to 50%.
//
// Limit: at most 32768 archetypes per World (ids 0..32767), because the entity
// allocator stores ids >= 4093 per entity in an Int16Array (EntityAllocator.bigAid).
// ---------------------------------------------------------------------------

/** Width groups, in layout order. */
const G8 = 0;
const G4 = 1;
const G2 = 2;
const G1 = 3;
const GROUPS = 4;

/** Slot kinds (cold code only). */
const S_ENTITY = 0;
const S_FIELD = 1;
const S_ENABLED = 2;

type ViewConstructor = new (buffer: ArrayBufferLike, byteOffset: number, length: number) => TypedArray;

function widthOf(ctor: TypedArrayConstructor): number {
  switch (ctor as unknown) {
    case Float64Array: return 8;
    case Float32Array:
    case Int32Array:
    case Uint32Array: return 4;
    case Int16Array:
    case Uint16Array: return 2;
    case Int8Array:
    case Uint8Array: return 1;
  }
  throw new Error('CozyECS: unsupported column type');
}

function groupOfWidth(w: number): number {
  return w === 8 ? G8 : w === 4 ? G4 : w === 2 ? G2 : G1;
}

const hasSAB = typeof SharedArrayBuffer !== 'undefined';

/** Largest archetype id (EntityAllocator.bigAid is an Int16Array). */
export const MAX_ARCHETYPE_ID = 32767;

/**
 * Growth step for a table/array of `cap` elements: x2 below 65536, otherwise x1.25
 * rounded up to a multiple of 4096. Clamped to MAX_ENTITIES (a table never holds more
 * rows than there are entities) unless `cap` is already at or beyond it.
 */
export function nextCapacity(cap: number): number {
  let next = cap < 65536 ? cap * 2 : Math.ceil((cap * 1.25) / 4096) * 4096;
  if (next > MAX_ENTITIES && cap < MAX_ENTITIES) next = MAX_ENTITIES;
  return next < 1 ? 1 : next;
}

/**
 * Precomputed row transfer plan from a source archetype to a target archetype, packed in
 * one Int32Array of length PLAN_HEADER + 2 * D:
 *   [PLAN_SRC_VER]  source `_layoutVersion` the offsets were resolved for (-1 = never)
 *   [PLAN_DST_VER]  target `_layoutVersion` the offsets were resolved for
 *   [PLAN_MODE]     PLAN_FAST4 if only 4-byte copies / 4-byte zero-inits exist, else PLAN_GENERAL
 *   [PLAN_START]    first index of the RESOLVED region (= PLAN_HEADER + D)
 *   [PLAN_END + k]  end index (in the resolved region) of section k, k = 0..8;
 *                   section k spans [k === 0 ? plan[PLAN_START] : plan[PLAN_END + k - 1], plan[PLAN_END + k])
 *   [PLAN_D]        D, the section data length
 *   [PLAN_ONE_S/D/Z] PLAN_ONE4 only: resolved source copy offset, target copy offset and
 *                   target zero offset. A missing copy/zero points at the `entities` cell
 *                   of the row (written afterwards anyway), so the fast path has no branches.
 * Sections:
 *   k = 0..3  (source, target) pairs to copy, width groups 8/4/2/1
 *   k = 4..7  target cells the source lacks, to zero, width groups 8/4/2/1
 *   k = 8     target enabled cells the source lacks, to set to 1
 * [PLAN_HEADER, PLAN_HEADER + D) holds slot indices (stable across growth);
 * [PLAN_HEADER + D, PLAN_HEADER + 2D) holds the same entries resolved to absolute unit
 * offsets in the whole-buffer width views (`_base[slot]`). The resolved region is rebuilt
 * whenever either archetype's layout version changed (i.e. it grew), so the hot loops are
 * `d[off + r] = s[off' + t]` with no per-cell `_base` indirection.
 */
type RowPlan = Int32Array;
const PLAN_SRC_VER = 0;
const PLAN_DST_VER = 1;
const PLAN_MODE = 2;
const PLAN_START = 3;
const PLAN_END = 4; // 9 section ends: [4, 13)
const PLAN_D = 13;
const PLAN_ONE_S = 14;
const PLAN_ONE_D = 15;
const PLAN_ONE_Z = 16;
const PLAN_HEADER = 17;
/** Any width groups / enabled flags: handled by `_pushRowFromGeneral`. */
const PLAN_GENERAL = 0;
/** Only 4-byte copies followed by 4-byte zero-inits (sections 1 and 5). */
const PLAN_FAST4 = 1;
/** Like PLAN_FAST4 with at most one copy and at most one zero-init (e.g. adding/removing a one-field component or a tag). */
const PLAN_ONE4 = 2;

/**
 * Archetype table: all entities that have exactly the same component set.
 * Rows [0, count) are packed (no holes). Data is structure-of-arrays: one
 * TypedArray view per component field, all views over a single buffer.
 *
 * Archetypes know nothing about World/entity locations: every method that
 * moves rows returns what moved, and World updates EntityAllocator locations.
 */
export class Archetype {
  /** Index in world._archetypes. */
  readonly id: number;
  /** Component bit mask (see component.ts mask helpers). */
  readonly mask: Uint32Array;
  /** Sorted component ids joined by ',' ('' for the empty archetype). */
  readonly key: string;
  /** Components, sorted by ascending id, no duplicates. */
  readonly components: readonly ComponentType[];
  /** Number of live rows. */
  count: number;
  /** Allocated rows in every column. */
  capacity: number;
  /** Entity handle per row. Length === capacity. View over the table buffer; replaced on grow. */
  entities: Uint32Array;
  /**
   * Column objects indexed by component id. `columns[c.id]` is an object with one
   * TypedArray per field (keys in schema order), or undefined if the component is
   * absent or a tag. The column OBJECT identity is stable for the archetype's
   * lifetime; its TypedArray properties are replaced on grow.
   * Must be a packed array (filled with undefined up to max component id).
   */
  readonly columns: (Record<string, TypedArray> | undefined)[];
  /**
   * Enabled flags indexed by component id: Uint8Array (1 = enabled) for enableable
   * components present in this archetype (tags included), otherwise undefined.
   * Replaced on grow. Packed array like `columns`.
   */
  readonly enabled: (Uint8Array | undefined)[];
  /** Transition cache: component id -> archetype with that component added. Maintained by World. */
  readonly edgesAdd: Map<number, Archetype>;
  /** Transition cache: component id -> archetype with that component removed. Maintained by World. */
  readonly edgesRemove: Map<number, Archetype>;
  /** True if the table buffer is a SharedArrayBuffer. */
  readonly shared: boolean;

  /** The single buffer backing every column. Replaced on grow. */
  private _buffer: ArrayBufferLike;
  /** Whole-buffer views by element width. Replaced on grow. */
  private _v64: Float64Array;
  private _v32: Int32Array;
  private _v16: Int16Array;
  private _v8: Uint8Array;
  /** Bytes per row (sum of column widths). */
  private readonly _rowBytes: number;
  /** Per slot: column start, in units of its width view. Rewritten on grow. */
  private readonly _base: Int32Array;
  /** Per slot: bytes per row of all slots before it (layout prefix). */
  private readonly _prefix: Int32Array;
  /** Per slot: element width in bytes. */
  private readonly _width: Int32Array;
  /** Slot ranges per width group: group g = slots [_group[g], _group[g + 1]). */
  private readonly _group: Int32Array;
  /** First enabled-flag slot; enabled slots are [_enStart, slotCount). */
  private readonly _enStart: number;
  /** Per component id: slot of each field (schema order), for plan building. */
  private readonly _slots: (Int32Array | undefined)[];
  /** Per component id: enabled slot, or -1. */
  private readonly _enSlot: Int32Array;
  // Cold per-slot descriptors used to rebuild views on grow.
  private readonly _slotKind: Int32Array;
  private readonly _slotCtor: ViewConstructor[];
  private readonly _slotObj: (Record<string, TypedArray> | null)[];
  private readonly _slotKey: string[];
  private readonly _slotComp: Int32Array;
  /** Incremented whenever `_base` changes (every allocation). Plans compare against it. */
  private _layoutVersion: number;
  /** Plans with this archetype as TARGET, by source; plus a one-entry memo. */
  private _plans: Map<Archetype, RowPlan> | null;
  private _lastSource: Archetype | null;
  private _lastPlan: RowPlan | null;

  /**
   * @param id index in world._archetypes
   * @param components normalized component list (sorted, deduped; see normalizeComponents)
   * @param key componentsKey(components)
   * @param initialCapacity rows to allocate up front (> 0)
   * @param shared allocate the table as a SharedArrayBuffer when available (default false)
   */
  constructor(
    id: number,
    components: readonly ComponentType[],
    key: string,
    initialCapacity: number,
    shared?: boolean,
  ) {
    let cap = initialCapacity | 0;
    if (cap < 1) cap = 1;
    if (id > MAX_ARCHETYPE_ID) throw new Error('CozyECS: too many archetypes (max 32767)');

    this.id = id;
    this.mask = createMask(components);
    this.key = key;
    this.components = components;
    this.count = 0;
    this.capacity = cap;
    this.edgesAdd = new Map();
    this.edgesRemove = new Map();
    this.shared = !!shared && hasSAB;
    this._plans = null;
    this._lastSource = null;
    this._lastPlan = null;

    // Components are sorted, so the last one has the max id.
    const size = components.length === 0 ? 0 : components[components.length - 1].id + 1;
    const columns: (Record<string, TypedArray> | undefined)[] = [];
    const enabled: (Uint8Array | undefined)[] = [];
    const slots: (Int32Array | undefined)[] = [];
    for (let i = 0; i < size; i++) {
      columns.push(undefined);
      enabled.push(undefined);
      slots.push(undefined);
    }
    const enSlot = new Int32Array(size);
    enSlot.fill(-1);

    // --- Layout: collect slots per width group (stable order inside a group).
    interface SlotDesc { kind: number; ctor: ViewConstructor; obj: Record<string, TypedArray> | null; key: string; comp: number; field: number }
    const byGroup: SlotDesc[][] = [[], [], [], []];
    byGroup[G4].push({ kind: S_ENTITY, ctor: Uint32Array, obj: null, key: '', comp: -1, field: -1 });
    const enDescs: SlotDesc[] = [];
    for (let i = 0; i < components.length; i++) {
      const c = components[i];
      const keys = c.keys;
      const tokens = c.tokens;
      if (keys.length > 0) {
        const obj: Record<string, TypedArray> = {};
        columns[c.id] = obj;
        slots[c.id] = new Int32Array(keys.length);
        for (let j = 0; j < keys.length; j++) {
          const ctor = tokens[j].ctor;
          obj[keys[j]] = new ctor(0); // fixes key order (schema order); replaced by _allocate
          byGroup[groupOfWidth(widthOf(ctor))].push({
            kind: S_FIELD, ctor: ctor as unknown as ViewConstructor, obj, key: keys[j], comp: c.id, field: j,
          });
        }
      }
      if (c.enableable) {
        enDescs.push({ kind: S_ENABLED, ctor: Uint8Array, obj: null, key: '', comp: c.id, field: -1 });
      }
    }
    for (let i = 0; i < enDescs.length; i++) byGroup[G1].push(enDescs[i]);

    let slotCount = 0;
    for (let g = 0; g < GROUPS; g++) slotCount += byGroup[g].length;
    const group = new Int32Array(GROUPS + 1);
    const prefix = new Int32Array(slotCount);
    const width = new Int32Array(slotCount);
    const slotKind = new Int32Array(slotCount);
    const slotComp = new Int32Array(slotCount);
    const slotCtor: ViewConstructor[] = [];
    const slotObj: (Record<string, TypedArray> | null)[] = [];
    const slotKey: string[] = [];
    let s = 0;
    let bytes = 0;
    for (let g = 0; g < GROUPS; g++) {
      group[g] = s;
      const w = g === G8 ? 8 : g === G4 ? 4 : g === G2 ? 2 : 1;
      const list = byGroup[g];
      for (let i = 0; i < list.length; i++, s++) {
        const d = list[i];
        prefix[s] = bytes;
        width[s] = w;
        bytes += w;
        slotKind[s] = d.kind;
        slotComp[s] = d.comp;
        slotCtor.push(d.ctor);
        slotObj.push(d.obj);
        slotKey.push(d.key);
        if (d.kind === S_FIELD) (slots[d.comp] as Int32Array)[d.field] = s;
        else if (d.kind === S_ENABLED) enSlot[d.comp] = s;
        // S_ENTITY is always the first slot of the 4-byte group (slot _group[G4]).
      }
    }
    group[GROUPS] = s;

    this.columns = columns;
    this.enabled = enabled;
    this._slots = slots;
    this._enSlot = enSlot;
    this._rowBytes = bytes;
    this._prefix = prefix;
    this._width = width;
    this._group = group;
    this._enStart = slotCount - enDescs.length;
    this._base = new Int32Array(slotCount);
    this._slotKind = slotKind;
    this._slotComp = slotComp;
    this._slotCtor = slotCtor;
    this._slotObj = slotObj;
    this._slotKey = slotKey;
    this._layoutVersion = 0;

    // Placeholders with final types, then the real allocation.
    this._buffer = new ArrayBuffer(0);
    this._v64 = new Float64Array(0);
    this._v32 = new Int32Array(0);
    this._v16 = new Int16Array(0);
    this._v8 = new Uint8Array(0);
    this.entities = new Uint32Array(0);
    this._allocate(cap, 0);
  }

  /** True if the archetype contains component `c` (mask test). */
  has(c: ComponentType): boolean {
    const id = c.id;
    const w = id >>> 5;
    const mask = this.mask;
    return w < mask.length && (mask[w] & (1 << (id & 31))) !== 0;
  }

  /**
   * O(1) column access. Returns undefined (typed as ColumnsOf<S>) if the component is
   * absent or is a tag. Do not cache the TypedArrays across structural changes: they
   * are views over the table buffer, which is replaced when the table grows.
   */
  col<S extends Schema>(c: ComponentType<S>): ColumnsOf<S> {
    const id = c.id;
    return (id < this.columns.length ? this.columns[id] : undefined) as unknown as ColumnsOf<S>;
  }

  /** True if `c` is present and (not enableable, or enabled at `row`). False if absent. */
  isEnabled(c: ComponentType, row: number): boolean {
    const id = c.id;
    if (id >= this.enabled.length) return false;
    const en = this.enabled[id];
    if (en !== undefined) return en[row] === 1;
    return this.has(c);
  }

  /** Enabled flags for `c` (1 = enabled), or undefined if absent / not enableable. */
  enabledArray(c: ComponentType): Uint8Array | undefined {
    const id = c.id;
    return id < this.enabled.length ? this.enabled[id] : undefined;
  }

  /**
   * Appends a row for `entity` at index `count`, growing if count === capacity.
   * All fields of the new row are zeroed and all enabled flags set to 1.
   * @returns the new row index.
   */
  pushRow(entity: number): number {
    const r = this.count;
    if (r === this.capacity) this._resize(nextCapacity(r));
    const g = this._group;
    const b = this._base;
    let k: number, e: number;
    if (g[1] > 0) { const v = this._v64; for (k = 0, e = g[1]; k < e; k++) v[b[k] + r] = 0; }
    // 4-byte group: slot g[1] is `entities`, written below.
    { const v = this._v32; for (k = g[1] + 1, e = g[2]; k < e; k++) v[b[k] + r] = 0; }
    if (g[3] > g[2]) { const v = this._v16; for (k = g[2], e = g[3]; k < e; k++) v[b[k] + r] = 0; }
    if (g[4] > g[3]) {
      const v = this._v8;
      const en = this._enStart;
      for (k = g[3]; k < en; k++) v[b[k] + r] = 0;
      for (e = g[4]; k < e; k++) v[b[k] + r] = 1;
    }
    this.entities[r] = entity;
    this.count = r + 1;
    return r;
  }

  /**
   * Appends a row for `entity` initialized from `source[sourceRow]`: fields and enabled
   * flags of components present in both are copied, the rest are zeroed / enabled.
   * Equivalent to `pushRow(entity)` + `source.copyRowTo(sourceRow, this, row)` without
   * writing the copied cells twice. `source` must not be `this`.
   * @returns the new row index.
   */
  pushRowFrom(source: Archetype, sourceRow: number, entity: number): number {
    // Kept small so TurboFan can inline it into World's move path; anything but the
    // steady-state 4-byte-only cases goes through `_pushRowFromGeneral`.
    const r = this.count;
    const p = this._lastPlan as RowPlan;
    if (
      this._lastSource !== source ||
      r === this.capacity ||
      p[PLAN_SRC_VER] !== source._layoutVersion ||
      p[PLAN_DST_VER] !== this._layoutVersion
    ) {
      return this._pushRowFromGeneral(source, sourceRow, entity);
    }
    const mode = p[PLAN_MODE];
    const d = this._v32;
    if (mode === PLAN_ONE4) {
      d[p[PLAN_ONE_D] + r] = source._v32[p[PLAN_ONE_S] + sourceRow];
      d[p[PLAN_ONE_Z] + r] = 0;
    } else if (mode === PLAN_FAST4) {
      const s = source._v32;
      let i = p[PLAN_START], e: number;
      for (e = p[PLAN_END + 1]; i < e; i += 2) d[p[i + 1] + r] = s[p[i] + sourceRow];
      for (e = p[PLAN_END + 5]; i < e; i++) d[p[i] + r] = 0;
    } else {
      return this._pushRowFromGeneral(source, sourceRow, entity);
    }
    this.entities[r] = entity;
    this.count = r + 1;
    return r;
  }

  /** pushRowFrom slow path: growth, plan lookup / re-resolution, and all width groups. */
  private _pushRowFromGeneral(source: Archetype, sourceRow: number, entity: number): number {
    const r = this.count;
    if (r === this.capacity) this._resize(nextCapacity(r));
    let p: RowPlan;
    if (this._lastSource === source) {
      p = this._lastPlan as RowPlan;
    } else {
      p = source._planFor(this);
      this._lastSource = source;
      this._lastPlan = p;
    }
    if (p[PLAN_SRC_VER] !== source._layoutVersion || p[PLAN_DST_VER] !== this._layoutVersion) {
      source._resolvePlan(p, this);
    }
    const t = sourceRow;
    let i = p[PLAN_START], e: number;
    // Each loop body reads/writes a single TypedArray type.
    if ((e = p[PLAN_END]) > i) { const s = source._v64, d = this._v64; for (; i < e; i += 2) d[p[i + 1] + r] = s[p[i] + t]; }
    if ((e = p[PLAN_END + 1]) > i) { const s = source._v32, d = this._v32; for (; i < e; i += 2) d[p[i + 1] + r] = s[p[i] + t]; }
    if ((e = p[PLAN_END + 2]) > i) { const s = source._v16, d = this._v16; for (; i < e; i += 2) d[p[i + 1] + r] = s[p[i] + t]; }
    if ((e = p[PLAN_END + 3]) > i) { const s = source._v8, d = this._v8; for (; i < e; i += 2) d[p[i + 1] + r] = s[p[i] + t]; }
    if ((e = p[PLAN_END + 4]) > i) { const d = this._v64; for (; i < e; i++) d[p[i] + r] = 0; }
    if ((e = p[PLAN_END + 5]) > i) { const d = this._v32; for (; i < e; i++) d[p[i] + r] = 0; }
    if ((e = p[PLAN_END + 6]) > i) { const d = this._v16; for (; i < e; i++) d[p[i] + r] = 0; }
    if (p[PLAN_END + 8] > i) {
      const d = this._v8;
      for (e = p[PLAN_END + 7]; i < e; i++) d[p[i] + r] = 0;
      for (e = p[PLAN_END + 8]; i < e; i++) d[p[i] + r] = 1;
    }
    this.entities[r] = entity;
    this.count = r + 1;
    return r;
  }

  /**
   * Removes `row` by moving the last row into it (fields, enabled flags, entities),
   * then count--. Vacated slot need not be cleared (pushRow zeroes).
   * @returns the entity handle that now lives at `row`, or -1 if `row` was the last row.
   */
  swapRemove(row: number): number {
    const last = this.count - 1;
    this.count = last;
    if (row === last) return -1;
    return this._moveLastInto(row, last);
  }

  /** swapRemove body: copies every slot of row `last` into `row`; returns the moved entity. */
  private _moveLastInto(row: number, last: number): number {
    const g = this._group;
    const b = this._base;
    let k: number, e: number;
    if (g[1] > 0) { const v = this._v64; for (k = 0, e = g[1]; k < e; k++) { const x = b[k]; v[x + row] = v[x + last]; } }
    // The 4-byte group always contains `entities`.
    { const v = this._v32; for (k = g[1], e = g[2]; k < e; k++) { const x = b[k]; v[x + row] = v[x + last]; } }
    if (g[3] > g[2]) { const v = this._v16; for (k = g[2], e = g[3]; k < e; k++) { const x = b[k]; v[x + row] = v[x + last]; } }
    if (g[4] > g[3]) { const v = this._v8; for (k = g[3], e = g[4]; k < e; k++) { const x = b[k]; v[x + row] = v[x + last]; } }
    return this.entities[row];
  }

  /**
   * Copies fields and enabled flags of every component present in BOTH this archetype
   * and `target` from `this[row]` to `target[targetRow]`. Does not touch `entities`
   * or counts. `targetRow` must already exist (pushRow). Prefer `target.pushRowFrom`.
   */
  copyRowTo(row: number, target: Archetype, targetRow: number): void {
    const p = this._planFor(target);
    if (p[PLAN_SRC_VER] !== this._layoutVersion || p[PLAN_DST_VER] !== target._layoutVersion) {
      this._resolvePlan(p, target);
    }
    const t = row;
    const r = targetRow;
    let i = p[PLAN_START], e: number;
    if ((e = p[PLAN_END]) > i) { const s = this._v64, d = target._v64; for (; i < e; i += 2) d[p[i + 1] + r] = s[p[i] + t]; }
    if ((e = p[PLAN_END + 1]) > i) { const s = this._v32, d = target._v32; for (; i < e; i += 2) d[p[i + 1] + r] = s[p[i] + t]; }
    if ((e = p[PLAN_END + 2]) > i) { const s = this._v16, d = target._v16; for (; i < e; i += 2) d[p[i + 1] + r] = s[p[i] + t]; }
    if ((e = p[PLAN_END + 3]) > i) { const s = this._v8, d = target._v8; for (; i < e; i += 2) d[p[i + 1] + r] = s[p[i] + t]; }
  }

  /**
   * Grows capacity by one growth step (x2 below 65536 rows, then x1.25 rounded up to a
   * multiple of 4096), preserving rows [0, count).
   */
  grow(): void {
    this._resize(nextCapacity(this.capacity));
  }

  /**
   * Ensures capacity >= `minCapacity` with a single reallocation of
   * max(minCapacity, next growth step). No-op if already large enough.
   */
  ensureCapacity(minCapacity: number): void {
    const cap = this.capacity;
    if (cap >= minCapacity) return;
    const step = nextCapacity(cap);
    this._resize(step > minCapacity ? step : Math.ceil(minCapacity));
  }

  /** The buffer backing every column (SharedArrayBuffer if `shared`). Replaced on grow. */
  get buffer(): ArrayBufferLike {
    return this._buffer;
  }

  /** Bytes of table storage per row (entities + fields + enabled flags). */
  get rowBytes(): number {
    return this._rowBytes;
  }

  // -------------------------------------------------------------------------
  // Cold paths
  // -------------------------------------------------------------------------

  /** Reallocates the table buffer to `cap` rows, copying rows [0, count) of every column once. */
  private _resize(cap: number): void {
    this._allocate(cap, this.count);
    this.capacity = cap;
  }

  /**
   * Allocates a buffer for `cap` rows, rebuilds every view and `_base`, and copies the
   * first `n` rows of each column from the previous buffer (n = 0 on construction).
   */
  private _allocate(cap: number, n: number): void {
    // Round to a multiple of 8 so the whole-buffer views cover every byte.
    const byteLength = (cap * this._rowBytes + 7) & ~7;
    const buffer: ArrayBufferLike = this.shared ? new SharedArrayBuffer(byteLength) : new ArrayBuffer(byteLength);
    const v8 = new Uint8Array(buffer, 0, byteLength);
    const old8 = this._v8;
    const oldCap = this.capacity;
    const prefix = this._prefix;
    const width = this._width;
    const base = this._base;
    const kinds = this._slotKind;
    const slotCount = prefix.length;
    for (let s = 0; s < slotCount; s++) {
      const w = width[s];
      const byteOff = cap * prefix[s];
      if (n > 0) {
        const oldOff = oldCap * prefix[s];
        v8.set(old8.subarray(oldOff, oldOff + n * w), byteOff);
      }
      base[s] = byteOff / w;
      const view = new this._slotCtor[s](buffer, byteOff, cap);
      const kind = kinds[s];
      if (kind === S_ENTITY) this.entities = view as Uint32Array;
      else if (kind === S_FIELD) (this._slotObj[s] as Record<string, TypedArray>)[this._slotKey[s]] = view;
      else this.enabled[this._slotComp[s]] = view as Uint8Array;
    }
    this._buffer = buffer;
    this._v8 = v8;
    this._v16 = new Int16Array(buffer, 0, byteLength >>> 1);
    this._v32 = new Int32Array(buffer, 0, byteLength >>> 2);
    this._v64 = new Float64Array(buffer, 0, byteLength >>> 3);
    this._layoutVersion++;
  }

  /** Rewrites the resolved-offset region of plan `p` (this -> target) for the current layouts. */
  private _resolvePlan(p: RowPlan, target: Archetype): void {
    const d = p[PLAN_D];
    const sb = this._base;
    const db = target._base;
    let i = p[PLAN_START];
    for (let k = 0; k < 4; k++) {
      for (const e = p[PLAN_END + k]; i < e; i += 2) {
        p[i] = sb[p[i - d]];
        p[i + 1] = db[p[i + 1 - d]];
      }
    }
    for (const e = p[PLAN_END + 8]; i < e; i++) p[i] = db[p[i - d]];
    if (p[PLAN_MODE] === PLAN_ONE4) {
      const start = p[PLAN_START];
      const copyEnd = p[PLAN_END + 1];
      const entS = sb[this._group[G4]];
      const entD = db[target._group[G4]];
      p[PLAN_ONE_S] = copyEnd > start ? p[start] : entS;
      p[PLAN_ONE_D] = copyEnd > start ? p[start + 1] : entD;
      p[PLAN_ONE_Z] = p[PLAN_END + 5] > copyEnd ? p[copyEnd] : entD;
    }
    p[PLAN_SRC_VER] = this._layoutVersion;
    p[PLAN_DST_VER] = target._layoutVersion;
  }

  /** Returns (building and caching on first use) the row plan from this archetype to `target`. */
  private _planFor(target: Archetype): RowPlan {
    let plans = target._plans;
    if (plans === null) plans = target._plans = new Map();
    const cached = plans.get(this);
    if (cached !== undefined) return cached;

    const copyB: number[][] = [[], [], [], []];
    const initB: number[][] = [[], [], [], [], []];
    const tcomps = target.components;
    for (let i = 0; i < tcomps.length; i++) {
      const c = tcomps[i];
      const id = c.id;
      const dst = target._slots[id];
      const shared = this.has(c);
      if (dst !== undefined) {
        const src = shared ? (this._slots[id] as Int32Array) : null;
        for (let j = 0; j < dst.length; j++) {
          const gi = groupOfWidth(target._width[dst[j]]);
          if (src !== null) copyB[gi].push(src[j], dst[j]);
          else initB[gi].push(dst[j]);
        }
      }
      if (c.enableable) {
        if (shared) copyB[G1].push(this._enSlot[id], target._enSlot[id]);
        else initB[4].push(target._enSlot[id]);
      }
    }
    const sections = copyB.concat(initB);
    let dataLen = 0;
    for (let k = 0; k < sections.length; k++) dataLen += sections[k].length;
    const plan = new Int32Array(PLAN_HEADER + 2 * dataLen);
    const start = PLAN_HEADER + dataLen;
    plan[PLAN_SRC_VER] = -1;
    plan[PLAN_DST_VER] = -1;
    plan[PLAN_START] = start;
    plan[PLAN_D] = dataLen;
    for (let k = 0, w = PLAN_HEADER; k < sections.length; k++) {
      const b = sections[k];
      for (let j = 0; j < b.length; j++) plan[w++] = b[j];
      plan[PLAN_END + k] = w + dataLen;
    }
    let fast = true;
    for (let k = 0; k < sections.length; k++) {
      if (k !== 1 && k !== 5 && sections[k].length !== 0) fast = false;
    }
    plan[PLAN_MODE] = !fast ? PLAN_GENERAL : sections[1].length <= 2 && sections[5].length <= 1 ? PLAN_ONE4 : PLAN_FAST4;
    plans.set(this, plan);
    return plan;
  }
}

/** A chunk is an archetype table viewed as a contiguous batch of rows. */
export type Chunk = Archetype;
