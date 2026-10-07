/**
 * String interning table. `str` columns store the ids returned by `intern`.
 * Id 0 is always the empty string.
 *
 * The table is append-only: `intern` never evicts and ids are stable for the table's
 * lifetime. The one exception is `_rebuild`, the seam `World.compact({ strings: true })`
 * uses to drop strings no live row references; it RENUMBERS the surviving ids, so the World
 * rewrites every `str` column in the same pass. Nothing else may renumber the table.
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

  /**
   * @internal Rebuilds the table in place, keeping only the ids marked in `keep` (`keep[id]`
   * non-zero) plus id 0 (`''`). Surviving strings keep their relative order and are
   * RENUMBERED densely from 1.
   *
   * Ids are NOT stable across this call: the caller MUST rewrite every stored id through the
   * returned remap before anything reads the table again. `intern`, `get` and `size` behave
   * exactly as before afterwards (a dropped string simply interns to a fresh id).
   *
   * @param keep marks, indexed by old id; ids at or past its length are dropped.
   * @returns remap indexed by old id: the new id, or 0 (`''`) for a dropped id.
   */
  _rebuild(keep: Uint8Array): Uint32Array {
    const strings = this._strings;
    const n = strings.length;
    const remap = new Uint32Array(n);
    const nKeep = keep.length;
    let w = 1;
    for (let id = 1; id < n; id++) {
      if (id >= nKeep || keep[id] === 0) continue;
      // w <= id always, so the compaction is safe in place.
      strings[w] = strings[id];
      remap[id] = w;
      w++;
    }
    strings.length = w;
    const ids = this._ids;
    ids.clear();
    ids.set('', 0);
    for (let i = 1; i < w; i++) ids.set(strings[i], i);
    return remap;
  }
}
