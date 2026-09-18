/**
 * String interning table. `str` columns store the ids returned by `intern`.
 * Id 0 is always the empty string. Strings are never evicted.
 */
export class StringTable {
  private readonly _ids = new Map<string, number>();
  private readonly _strings: string[] = [''];

  constructor() {
    this._ids.set('', 0);
  }

  /** Returns the id of `s`, adding it to the table if needed. */
  intern(s: string): number {
    const id = this._ids.get(s);
    if (id !== undefined) return id;
    const next = this._strings.length;
    this._strings.push(s);
    this._ids.set(s, next);
    return next;
  }

  /** Returns the string for `id`, or `''` for an unknown id. */
  get(id: number): string {
    const s = this._strings[id];
    return s === undefined ? '' : s;
  }

  /** Number of interned strings (including `''`). */
  get size(): number {
    return this._strings.length;
  }
}
