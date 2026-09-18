/**
 * Field type tokens and type-level helpers.
 *
 * A field token describes how one field of a component is stored: every field
 * lives in its own TypedArray column inside an archetype table.
 */

/** Any TypedArray CozyECS can allocate for a column. */
export type TypedArray =
  | Float32Array
  | Float64Array
  | Int8Array
  | Int16Array
  | Int32Array
  | Uint8Array
  | Uint16Array
  | Uint32Array;

/** Constructor of a TypedArray column. */
export type TypedArrayConstructor<A extends TypedArray = TypedArray> = new (length: number) => A;

/** Short names of the supported field kinds. */
export type FieldKind = 'f32' | 'f64' | 'i8' | 'i16' | 'i32' | 'u8' | 'u16' | 'u32' | 'bool' | 'str';

/** Numeric codes of the field kinds (stable; usable in switch statements). */
export const FIELD_F32 = 0;
export const FIELD_F64 = 1;
export const FIELD_I8 = 2;
export const FIELD_I16 = 3;
export const FIELD_I32 = 4;
export const FIELD_U8 = 5;
export const FIELD_U16 = 6;
export const FIELD_U32 = 7;
export const FIELD_BOOL = 8;
export const FIELD_STR = 9;

/**
 * Field type token.
 * @typeParam A TypedArray used for the column.
 * @typeParam V JS value type used by `world.set` / `world.get`.
 */
export interface FieldToken<A extends TypedArray = TypedArray, V = unknown> {
  /** Short name, e.g. `'f32'`. */
  readonly kind: FieldKind;
  /** Numeric code, one of the `FIELD_*` constants. */
  readonly code: number;
  /** Column constructor. */
  readonly ctor: TypedArrayConstructor<A>;
  /** Phantom marker carrying the value type. Never present at runtime. */
  readonly __value?: V;
}

function token<A extends TypedArray, V>(
  kind: FieldKind,
  code: number,
  ctor: TypedArrayConstructor<A>,
): FieldToken<A, V> {
  return Object.freeze({ kind, code, ctor }) as FieldToken<A, V>;
}

/** 32-bit float field. */
export const f32: FieldToken<Float32Array, number> = token('f32', FIELD_F32, Float32Array);
/** 64-bit float field. */
export const f64: FieldToken<Float64Array, number> = token('f64', FIELD_F64, Float64Array);
/** Signed 8-bit integer field. */
export const i8: FieldToken<Int8Array, number> = token('i8', FIELD_I8, Int8Array);
/** Signed 16-bit integer field. */
export const i16: FieldToken<Int16Array, number> = token('i16', FIELD_I16, Int16Array);
/** Signed 32-bit integer field. */
export const i32: FieldToken<Int32Array, number> = token('i32', FIELD_I32, Int32Array);
/** Unsigned 8-bit integer field. */
export const u8: FieldToken<Uint8Array, number> = token('u8', FIELD_U8, Uint8Array);
/** Unsigned 16-bit integer field. */
export const u16: FieldToken<Uint16Array, number> = token('u16', FIELD_U16, Uint16Array);
/** Unsigned 32-bit integer field. */
export const u32: FieldToken<Uint32Array, number> = token('u32', FIELD_U32, Uint32Array);
/** Boolean field, stored as 0/1 in a Uint8Array. */
export const bool: FieldToken<Uint8Array, boolean> = token('bool', FIELD_BOOL, Uint8Array);
/** String field, stored as interned ids (see `world.strings`) in a Uint32Array. 0 = `''`. */
export const str: FieldToken<Uint32Array, string> = token('str', FIELD_STR, Uint32Array);

/** Component schema: field name -> field token. */
export type Schema = Record<string, FieldToken<TypedArray, unknown>>;

/** Schema of a tag component (no fields). */
export type EmptySchema = Record<never, never>;

/** Column object of a component inside a chunk: field name -> TypedArray. */
export type ColumnsOf<S extends Schema> = {
  [K in keyof S]: S[K] extends FieldToken<infer A, unknown> ? A : never;
};

/** Plain JS values of a component: numbers, booleans (bool) and strings (str). */
export type ValuesOf<S extends Schema> = {
  [K in keyof S]: S[K] extends FieldToken<TypedArray, infer V> ? V : never;
};

/** Callback receiving an entity handle. */
export type EntityCallback = (entity: number) => void;

/** Function returned by subscriptions; call it to unsubscribe. Idempotent. */
export type Unsubscribe = () => void;
