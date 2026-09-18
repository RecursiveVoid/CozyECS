import { describe, it, expect } from '@jest/globals';
import { StringTable, World, component, str } from '../src/index';

describe('StringTable', () => {
  it('id 0 is the empty string', () => {
    const t = new StringTable();
    expect(t.intern('')).toBe(0);
    expect(t.get(0)).toBe('');
  });

  it('interns strings to stable ids', () => {
    const t = new StringTable();
    const a = t.intern('hello');
    const b = t.intern('world');
    expect(a).not.toBe(0);
    expect(b).not.toBe(a);
    expect(t.intern('hello')).toBe(a);
    expect(t.get(a)).toBe('hello');
    expect(t.get(b)).toBe('world');
  });

  it('handles unicode and look-alike strings distinctly', () => {
    const t = new StringTable();
    const ids = ['a', 'A', 'a ', 'ä', '日本', '0', 'undefined'].map((s) => t.intern(s));
    expect(new Set(ids).size).toBe(ids.length);
    expect(t.get(ids[4])).toBe('日本');
  });
});

describe('str fields in World', () => {
  const Name = component({ label: str }, { name: 'Name' });

  it('each world has its own string table', () => {
    const w1 = new World();
    const w2 = new World();
    expect(w1.strings).toBeInstanceOf(StringTable);
    expect(w1.strings).not.toBe(w2.strings);
  });

  it('set interns automatically; column stores ids; get resolves strings', () => {
    const w = new World();
    const e = w.spawn([Name]);
    w.set(e, Name, { label: 'bob' });
    const id = w.getField(e, Name, 'label');
    expect(id).toBe(w.strings.intern('bob'));
    expect(w.strings.get(id)).toBe('bob');
    expect(w.get(e, Name)!.label).toBe('bob');

    const e2 = w.spawn([Name]);
    w.set(e2, Name, { label: 'bob' });
    expect(w.getField(e2, Name, 'label')).toBe(id);
  });

  it('default str value is empty string (id 0)', () => {
    const w = new World();
    const e = w.spawn([Name]);
    expect(w.getField(e, Name, 'label')).toBe(0);
    expect(w.get(e, Name)!.label).toBe('');
  });

  it('add with string values interns', () => {
    const w = new World();
    const e = w.spawn();
    w.add(e, Name, { label: 'alice' });
    expect(w.get(e, Name)!.label).toBe('alice');
    const arch = w.archetype(Name);
    expect(w.strings.get(arch.col(Name).label[0])).toBe('alice');
  });
});
