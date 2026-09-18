import type { EmptySchema, FieldToken, Schema } from './types';

/**
 * A component type. Components are world-independent: ids come from a
 * module-global counter and the same ComponentType can be used by any World.
 */
export interface ComponentType<S extends Schema = Schema> {
  /** Unique, dense, module-global id (0, 1, 2, ...). */
  readonly id: number;
  /** Debug name. */
  readonly name: string;
  /** The schema passed to `component()`. Frozen. */
  readonly schema: S;
  /** Field names in schema declaration order. Frozen. */
  readonly keys: readonly string[];
  /** Field tokens, parallel to `keys`. Frozen. */
  readonly tokens: readonly FieldToken[];
  /** True when the schema has no fields (zero per-entity memory). */
  readonly isTag: boolean;
  /** True when the component can be enabled/disabled per entity. */
  readonly enableable: boolean;
}

/** Options for `component()` and `tag()`. */
export interface ComponentOptions {
  /** Debug name. Defaults to `Component<id>`. */
  name?: string;
  /** Allows `world.enable(e, C, false)`. Costs one byte per entity. Default false. */
  enableable?: boolean;
}

let nextComponentId = 0;

/**
 * Defines a component type.
 * @example const Position = component({ x: f32, y: f32 }, { name: 'Position' });
 */
export function component<S extends Schema>(schema: S, options?: ComponentOptions): ComponentType<S> {
  const id = nextComponentId++;
  const keys = Object.keys(schema);
  const tokens: FieldToken[] = [];
  for (let i = 0; i < keys.length; i++) {
    const t = schema[keys[i]];
    if (!t || typeof t.ctor !== 'function') {
      throw new Error(`CozyECS: invalid field type for "${keys[i]}"`);
    }
    tokens.push(t);
  }
  const frozenSchema = Object.freeze({ ...schema }) as S;
  return Object.freeze({
    id,
    name: (options && options.name) || `Component${id}`,
    schema: frozenSchema,
    keys: Object.freeze(keys),
    tokens: Object.freeze(tokens),
    isTag: keys.length === 0,
    enableable: !!(options && options.enableable),
  });
}

/**
 * Defines a tag component (no data, zero per-entity memory unless enableable).
 * @example const Player = tag({ name: 'Player' });
 */
export function tag(options?: ComponentOptions): ComponentType<EmptySchema> {
  return component({} as EmptySchema, options);
}

// ---------------------------------------------------------------------------
// Internal helpers: component lists, keys and bit masks.
// Masks are Uint32Arrays of variable length; bit for component id `i` is
// word `i >>> 5`, bit `1 << (i & 31)`. Missing words are treated as zero.
// ---------------------------------------------------------------------------

/** @internal Returns a new array with duplicates removed, sorted by ascending id. */
export function normalizeComponents(components: readonly ComponentType[]): ComponentType[] {
  const out: ComponentType[] = [];
  for (let i = 0; i < components.length; i++) {
    const c = components[i];
    if (out.indexOf(c) === -1) out.push(c);
  }
  out.sort((a, b) => a.id - b.id);
  return out;
}

/** @internal Archetype key: ids of an already-normalized list joined by ','. Empty list -> ''. */
export function componentsKey(sorted: readonly ComponentType[]): string {
  let key = '';
  for (let i = 0; i < sorted.length; i++) key += i === 0 ? `${sorted[i].id}` : `,${sorted[i].id}`;
  return key;
}

/** @internal Builds a mask containing every component in the list. Empty list -> length-0 mask. */
export function createMask(components: readonly ComponentType[]): Uint32Array {
  let maxId = -1;
  for (let i = 0; i < components.length; i++) if (components[i].id > maxId) maxId = components[i].id;
  const mask = new Uint32Array(maxId < 0 ? 0 : (maxId >>> 5) + 1);
  for (let i = 0; i < components.length; i++) {
    const id = components[i].id;
    mask[id >>> 5] |= 1 << (id & 31);
  }
  return mask;
}

/** @internal True if `mask` has the bit for component `id`. */
export function maskHas(mask: Uint32Array, id: number): boolean {
  const w = id >>> 5;
  return w < mask.length && (mask[w] & (1 << (id & 31))) !== 0;
}

/** @internal True if every bit set in `sub` is also set in `sup`. */
export function maskContains(sup: Uint32Array, sub: Uint32Array): boolean {
  const n = sub.length;
  const m = sup.length;
  for (let i = 0; i < n; i++) {
    const s = sub[i];
    if (s !== 0 && (i >= m || (s & ~sup[i]) !== 0)) return false;
  }
  return true;
}

/** @internal True if `a` and `b` share at least one set bit. */
export function maskIntersects(a: Uint32Array, b: Uint32Array): boolean {
  const n = a.length < b.length ? a.length : b.length;
  for (let i = 0; i < n; i++) if ((a[i] & b[i]) !== 0) return true;
  return false;
}
