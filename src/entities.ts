/**
 * Entity id allocator.
 *
 * Handle layout (uint32 number): low 20 bits = index, high 12 bits = generation.
 *   handle = ((generation << 20) | index) >>> 0
 * Handle 0 (index 0, generation 0) is a valid entity; use -1 as "no entity".
 *
 * Per-index state is ONE Int32 (4 bytes per index), `slot[index] = (tag << 20) | low`:
 *
 *   tag 0..4092   placed in archetype `tag`;             low = row in that archetype
 *   tag 4093      placed in archetype `bigAid[index]`    low = row
 *                 (archetype ids 4093..32767, see TAG_BIG)
 *   tag 4094      free (never handed out, or destroyed)  low = generation the next handle gets
 *   tag 4095      pending (alive, spawned while deferring, no row yet)
 *                                                        low = current generation
 *   decoded archetype id: tag < 4093 ? tag : tag === 4093 ? bigAid[index] : tag - 4096
 *   (so pending -> -1 = LOCATION_PENDING, free -> -2 = LOCATION_FREE)
 *
 * The generation of a PLACED entity is not stored here: the table row stores the full
 * handle (`archetype.entities[row]`), so liveness of a placed entity is
 *   archetypes[tag].entities[slot & INDEX_MASK] === entity
 * which a stale handle (other generation) can never satisfy. The table load is paid by
 * structural ops anyway (they touch that row next).
 *
 * Rows fit in 20 bits because a table never holds more than MAX_ENTITIES rows.
 * Free indices are recycled LIFO through `freeStack` (grown on demand; its size is the peak
 * number of simultaneously free indices, not the entity count).
 * `bigAid` stays empty until the World creates archetype id 4093 (`enableBig()`); from then
 * on it is kept as long as `slot`.
 *
 * Arrays grow by doubling below 65536 indices, then by x1.25 rounded up to a multiple of
 * 4096 (clamped to MAX_ENTITIES), the same policy as archetype tables. Arrays are REPLACED
 * on growth, so never cache them across calls that may allocate.
 */
import type { Archetype } from './archetype';

export const INDEX_BITS = 20;
export const INDEX_MASK = 0xfffff;
export const GENERATION_MASK = 0xfff;
/** Maximum number of simultaneously allocated indices. */
export const MAX_ENTITIES = 1 << INDEX_BITS;

export const LOCATION_PENDING = -1;
export const LOCATION_FREE = -2;

/** Slot tags (high 12 bits of `slot[index]`, read with `>>> 20`). Ids below TAG_BIG are inline. */
export const TAG_BIG = 4093;
export const TAG_FREE = 4094;
export const TAG_PENDING = 4095;
/** High-bits mask of a slot (and generation mask of a handle), as int32. */
export const HIGH_BITS = ~INDEX_MASK;

/** Index part of a handle. */
export function entityIndex(entity: number): number {
  return entity & INDEX_MASK;
}

/** Generation part of a handle. */
export function entityGeneration(entity: number): number {
  return entity >>> INDEX_BITS;
}

/** Builds a handle from index and generation. */
export function makeEntity(index: number, generation: number): number {
  return ((generation << INDEX_BITS) | index) >>> 0;
}

const EMPTY_I32 = new Int32Array(0);

export class EntityAllocator {
  /** Packed (tag << 20) | low per index (see file header). */
  slot: Int32Array;
  /** Archetype id per index for tag TAG_BIG. Empty until `enableBig()`. */
  bigAid: Int16Array;
  /** Free indices, LIFO; entries [0, freeCount) are valid. */
  freeStack: Int32Array;
  /** Number of indices on the free stack. */
  freeCount: number;
  /** High-water mark: indices [0, next) have been handed out at least once. */
  next: number;
  /** Number of currently alive entities (including pending ones). */
  aliveCount: number;
  /** The World's archetype list (`world._archetypes`), for liveness checks of placed entities. */
  readonly archetypes: readonly Archetype[];

  /**
   * @param initialCapacity initial length of the per-index array (default 64).
   * @param archetypes the owning World's archetype list (read, never modified).
   */
  constructor(initialCapacity: number = 64, archetypes: readonly Archetype[] = []) {
    let cap = initialCapacity | 0;
    if (cap < 1) cap = 1;
    if (cap > MAX_ENTITIES) cap = MAX_ENTITIES;
    this.slot = new Int32Array(cap);
    this.bigAid = new Int16Array(0);
    this.freeStack = EMPTY_I32;
    this.freeCount = 0;
    this.next = 0;
    this.aliveCount = 0;
    this.archetypes = archetypes;
  }

  /**
   * Allocates an entity. Reuses the most recently freed index if any, otherwise
   * `next++`. The entity is pending (tag TAG_PENDING); the caller places it into an
   * archetype with `setLocation`. Grows arrays (see file header) when needed.
   * @throws Error when MAX_ENTITIES indices are in use.
   */
  create(): number {
    let index: number;
    let gen = 0;
    if (this.freeCount > 0) {
      index = this.freeStack[--this.freeCount];
      gen = this.slot[index] & GENERATION_MASK;
    } else {
      index = this.next;
      if (index >= this.slot.length) {
        if (index >= MAX_ENTITIES) {
          throw new Error(`CozyECS: entity limit reached (${MAX_ENTITIES} entities)`);
        }
        this._resize(index + 1);
      }
      this.next = index + 1;
    }
    this.slot[index] = (TAG_PENDING << INDEX_BITS) | gen;
    this.aliveCount++;
    return ((gen << INDEX_BITS) | index) >>> 0;
  }

  /**
   * Frees a live entity: tag TAG_FREE with generation (g + 1) & 0xfff, pushes the index on
   * the free stack, aliveCount--. Caller must have checked liveness and already removed the
   * entity's row from its archetype.
   */
  release(entity: number): void {
    const index = entity & INDEX_MASK;
    this.slot[index] = (TAG_FREE << INDEX_BITS) | (((entity >>> INDEX_BITS) + 1) & GENERATION_MASK);
    let n = this.freeCount;
    if (n === this.freeStack.length) this._growFree();
    this.freeStack[n] = index;
    this.freeCount = n + 1;
    this.aliveCount--;
  }

  /**
   * Archetype id of a live `entity`: >= 0 placed, LOCATION_PENDING (-1) pending, or
   * LOCATION_FREE (-2) when the handle is not alive (stale, free, never allocated, invalid).
   */
  locate(entity: number): number {
    const index = entity & INDEX_MASK;
    if (!(entity >= 0 && index < this.next)) return LOCATION_FREE;
    const s = this.slot[index];
    const tag = s >>> INDEX_BITS;
    if (tag < TAG_BIG) {
      return this.archetypes[tag].entities[s & INDEX_MASK] === entity ? tag : LOCATION_FREE;
    }
    return this._locateSlow(entity, index, s, tag);
  }

  /** `locate` for tags >= TAG_BIG. */
  _locateSlow(entity: number, index: number, s: number, tag: number): number {
    if (tag === TAG_PENDING) {
      return (s & GENERATION_MASK) === entity >>> INDEX_BITS ? LOCATION_PENDING : LOCATION_FREE;
    }
    if (tag === TAG_BIG) {
      const aid = this.bigAid[index];
      return this.archetypes[aid].entities[s & INDEX_MASK] === entity ? aid : LOCATION_FREE;
    }
    return LOCATION_FREE;
  }

  /** True if the handle refers to a currently allocated entity (pending counts as alive). */
  isAlive(entity: number): boolean {
    return this.locate(entity) !== LOCATION_FREE;
  }

  /** Archetype id (or -1 pending / -2 free) stored for `index`, without a liveness check. */
  archetypeOf(index: number): number {
    const tag = this.slot[index] >>> INDEX_BITS;
    return tag < TAG_BIG ? tag : tag === TAG_BIG ? this.bigAid[index] : tag - 4096;
  }

  /** Row of a PLACED entity, by index. */
  rowOf(index: number): number {
    return this.slot[index] & INDEX_MASK;
  }

  /** Records that the live entity `entity` (the HANDLE) is stored in archetype `aid` at `row`. */
  setLocation(entity: number, aid: number, row: number): void {
    const index = entity & INDEX_MASK;
    if (aid < TAG_BIG) {
      this.slot[index] = (aid << INDEX_BITS) | row;
    } else {
      this.slot[index] = (TAG_BIG << INDEX_BITS) | row;
      this.bigAid[index] = aid;
    }
  }

  /** Called by the World when archetype id TAG_BIG is created: allocates `bigAid`. */
  enableBig(): void {
    if (this.bigAid.length < this.slot.length) {
      const big = new Int16Array(this.slot.length);
      big.set(this.bigAid);
      this.bigAid = big;
    }
  }

  /** Ensures per-index arrays can hold `additional` more new indices without growing mid-loop. */
  reserve(additional: number): void {
    const fresh = additional - this.freeCount;
    if (fresh <= 0) return;
    let needed = this.next + fresh;
    if (needed > MAX_ENTITIES) needed = MAX_ENTITIES; // create() throws past the limit
    if (needed > this.slot.length) this._resize(needed);
  }

  /**
   * Reallocates the per-index arrays ONCE to max(minCapacity, next growth step), clamped to
   * MAX_ENTITIES. Growth step: x2 below 65536, else x1.25 rounded up to a multiple of 4096.
   */
  private _resize(minCapacity: number): void {
    const oldCap = this.slot.length;
    let cap = oldCap < 65536 ? oldCap * 2 : Math.ceil((oldCap * 1.25) / 4096) * 4096;
    if (cap < minCapacity) cap = minCapacity;
    if (cap > MAX_ENTITIES) cap = MAX_ENTITIES;
    if (cap <= oldCap) return;

    const slot = new Int32Array(cap);
    slot.set(this.slot);
    this.slot = slot;

    if (this.bigAid.length !== 0) {
      const big = new Int16Array(cap);
      big.set(this.bigAid);
      this.bigAid = big;
    }
  }

  private _growFree(): void {
    const old = this.freeStack;
    const s = new Int32Array(old.length < 32 ? 64 : old.length * 2);
    s.set(old);
    this.freeStack = s;
  }
}
