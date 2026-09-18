import { describe, it, expect } from '@jest/globals';
import { World, component, tag, f32 } from '../src/index';

const A = component({ v: f32 }, { name: 'A' });
const B = component({ v: f32 }, { name: 'B' });
const C = tag({ name: 'C' });
const E = component({ v: f32 }, { name: 'E', enableable: true });

describe('Query.onEnter / onExit', () => {
  it('fires on spawn and destroy', () => {
    const w = new World();
    const q = w.query({ all: [A] });
    const log: string[] = [];
    q.onEnter((e) => log.push(`enter ${e}`));
    q.onExit((e) => log.push(`exit ${e}`));
    const e = w.spawn([A]);
    const other = w.spawn([B]);
    w.destroy(e);
    w.destroy(other);
    expect(log).toEqual([`enter ${e}`, `exit ${e}`]);
  });

  it('fires on add/remove transitions only when matching changes', () => {
    const w = new World();
    const q = w.query({ all: [A, B], none: [C] });
    const log: string[] = [];
    q.onEnter((e) => log.push(`enter ${e}`));
    q.onExit((e) => log.push(`exit ${e}`));
    const e = w.spawn([A]);
    expect(log).toEqual([]);
    w.add(e, B);
    expect(log).toEqual([`enter ${e}`]);
    w.set(e, A, { v: 1 });
    w.add(e, B, { v: 2 }); // already has: no event
    expect(log).toEqual([`enter ${e}`]);
    w.add(e, C);
    expect(log).toEqual([`enter ${e}`, `exit ${e}`]);
    w.remove(e, C);
    expect(log).toEqual([`enter ${e}`, `exit ${e}`, `enter ${e}`]);
    w.remove(e, A);
    expect(log).toEqual([`enter ${e}`, `exit ${e}`, `enter ${e}`, `exit ${e}`]);
  });

  it('fires after the structural change is applied', () => {
    const w = new World();
    const q = w.query({ all: [A, B] });
    const e = w.spawn([A]);
    w.set(e, A, { v: 7 });
    let seen: unknown = null;
    q.onEnter((x) => {
      seen = { has: w.has(x, B), a: w.get(x, A)!.v, b: w.get(x, B)!.v, count: q.count() };
    });
    let exitSeen: unknown = null;
    q.onExit((x) => {
      exitSeen = { hasB: w.has(x, B), count: q.count() };
    });
    w.add(e, B, { v: 3 });
    expect(seen).toEqual({ has: true, a: 7, b: 3, count: 1 });
    w.remove(e, B);
    expect(exitSeen).toEqual({ hasB: false, count: 0 });
  });

  it('exit on destroy fires after removal', () => {
    const w = new World();
    const q = w.query({ all: [A] });
    const e = w.spawn([A]);
    let info: unknown = null;
    q.onExit((x) => (info = { x, count: q.count(), alive: w.isAlive(x) }));
    w.destroy(e);
    expect(info).toEqual({ x: e, count: 0, alive: false });
  });

  it('enable/disable does not fire enter/exit', () => {
    const w = new World();
    const q = w.query({ all: [E] });
    let n = 0;
    q.onEnter(() => n++);
    q.onExit(() => n++);
    const e = w.spawn([E]);
    expect(n).toBe(1);
    w.enable(e, E, false);
    w.enable(e, E, true);
    expect(n).toBe(1);
  });

  it('unsubscribe stops callbacks; unsubscribe is idempotent', () => {
    const w = new World();
    const q = w.query({ all: [A] });
    let a = 0;
    let b = 0;
    const offA = q.onEnter(() => a++);
    q.onEnter(() => b++);
    w.spawn([A]);
    offA();
    offA();
    w.spawn([A]);
    expect(a).toBe(1);
    expect(b).toBe(2);
  });

  it('multiple listeners fire in registration order', () => {
    const w = new World();
    const q = w.query({ all: [A] });
    const log: number[] = [];
    q.onEnter(() => log.push(1));
    q.onEnter(() => log.push(2));
    q.onEnter(() => log.push(3));
    w.spawn([A]);
    expect(log).toEqual([1, 2, 3]);
  });

  it('unsubscribing during dispatch does not skip other listeners', () => {
    const w = new World();
    const q = w.query({ all: [A] });
    const log: number[] = [];
    let off1: () => void = () => {};
    off1 = q.onEnter(() => {
      log.push(1);
      off1();
    });
    q.onEnter(() => log.push(2));
    w.spawn([A]);
    w.spawn([A]);
    expect(log).toEqual([1, 2, 2]);
  });

  it('events for changes deferred during forEach fire at flush', () => {
    const w = new World();
    const q = w.query({ all: [B] });
    const entered: number[] = [];
    q.onEnter((e) => entered.push(e));
    const e = w.spawn([A]);
    w.query({ all: [A] }).forEach((x) => {
      w.add(x, B);
      expect(entered).toEqual([]);
    });
    expect(entered).toEqual([e]);
  });

  it('spawnMany fires enter for each entity', () => {
    const w = new World();
    const q = w.query({ all: [A] });
    const entered: number[] = [];
    q.onEnter((e) => entered.push(e));
    w.spawnMany(w.archetype(A, B), 5);
    expect(entered.length).toBe(5);
    expect(new Set(entered).size).toBe(5);
  });

  it('entity entering and leaving via different archetypes that all match fires nothing', () => {
    const w = new World();
    const q = w.query({ any: [A, B] });
    let n = 0;
    q.onEnter(() => n++);
    q.onExit(() => n++);
    const e = w.spawn([A]); // enter
    w.add(e, B);
    w.remove(e, A);
    expect(n).toBe(1);
    w.remove(e, B); // exit
    expect(n).toBe(2);
  });
});

describe('world.onAdd / onRemove', () => {
  it('onAdd fires on add, spawn, spawnMany', () => {
    const w = new World();
    const added: number[] = [];
    w.onAdd(A, (e) => added.push(e));
    const e1 = w.spawn([A, B]);
    const e2 = w.spawn();
    w.add(e2, A);
    w.add(e2, A); // already present: no fire
    w.spawnMany(w.archetype(A), 2);
    w.spawn([B]);
    expect(added.length).toBe(4);
    expect(added.slice(0, 2)).toEqual([e1, e2]);
  });

  it('onRemove fires on remove and destroy', () => {
    const w = new World();
    const removed: number[] = [];
    w.onRemove(A, (e) => removed.push(e));
    const e1 = w.spawn([A]);
    const e2 = w.spawn([A, B]);
    w.remove(e1, A);
    w.remove(e1, A); // absent: no fire
    w.destroy(e2);
    w.destroy(e1); // no A anymore
    expect(removed).toEqual([e1, e2]);
  });

  it('onAdd sees values passed to add (fires after change applied)', () => {
    const w = new World();
    const e = w.spawn();
    let v = -1;
    let has = false;
    w.onAdd(A, (x) => {
      has = w.has(x, A);
      v = w.get(x, A)!.v;
    });
    w.add(e, A, { v: 42 });
    expect(has).toBe(true);
    expect(v).toBe(42);
  });

  it('onAdd/onRemove not fired when other components change', () => {
    const w = new World();
    let n = 0;
    w.onAdd(A, () => n++);
    w.onRemove(A, () => n++);
    const e = w.spawn([A]);
    w.add(e, B);
    w.remove(e, B);
    w.add(e, C);
    expect(n).toBe(1);
  });

  it('enable does not fire onAdd/onRemove', () => {
    const w = new World();
    let n = 0;
    w.onAdd(E, () => n++);
    w.onRemove(E, () => n++);
    const e = w.spawn([E]);
    w.enable(e, E, false);
    w.enable(e, E, true);
    expect(n).toBe(1);
  });

  it('unsubscribe works and is idempotent; listeners fire in order', () => {
    const w = new World();
    const log: string[] = [];
    const off1 = w.onAdd(A, () => log.push('1'));
    w.onAdd(A, () => log.push('2'));
    const offR = w.onRemove(A, () => log.push('r'));
    const e = w.spawn([A]);
    expect(log).toEqual(['1', '2']);
    off1();
    off1();
    offR();
    w.remove(e, A);
    w.add(e, A);
    expect(log).toEqual(['1', '2', '2']);
  });

  it('hooks for deferred changes fire at flush', () => {
    const w = new World();
    const log: string[] = [];
    w.onAdd(B, () => log.push('addB'));
    w.onRemove(A, () => log.push('remA'));
    w.spawn([A]);
    w.query({ all: [A] }).forEach((e) => {
      w.add(e, B);
      w.destroy(e);
      expect(log).toEqual([]);
    });
    expect(log).toEqual(['addB', 'remA']);
  });

  it('structural changes from inside hooks are applied', () => {
    const w = new World();
    w.onAdd(A, (e) => w.add(e, B, { v: 1 }));
    const e = w.spawn();
    w.add(e, A);
    expect(w.has(e, B)).toBe(true);
    expect(w.get(e, B)!.v).toBe(1);
    // also during flush
    const e2 = w.spawn();
    w.query({}).forEach(() => {});
    w.spawn([C]);
    w.query({ all: [C] }).forEach(() => w.add(e2, A));
    expect(w.has(e2, A)).toBe(true);
    expect(w.has(e2, B)).toBe(true);
  });

  it('hooks for tags', () => {
    const w = new World();
    let added = 0;
    let removed = 0;
    w.onAdd(C, () => added++);
    w.onRemove(C, () => removed++);
    const e = w.spawn();
    w.add(e, C);
    w.remove(e, C);
    expect([added, removed]).toEqual([1, 1]);
  });
});
