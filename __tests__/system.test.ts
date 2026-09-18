import { describe, it, expect } from '@jest/globals';
import { World, System, Query, component, f32 } from '../src/index';
import type { SystemHandle } from '../src/index';

const Position = component({ x: f32, y: f32 }, { name: 'Position' });
const Velocity = component({ vx: f32, vy: f32 }, { name: 'Velocity' });
const Dead = component({ t: f32 }, { name: 'Dead' });

describe('function systems', () => {
  it('registers with query desc and receives (q, dt, world)', () => {
    const w = new World();
    const e = w.spawn([Position, Velocity]);
    w.set(e, Velocity, { vx: 2, vy: 3 });
    let args: unknown[] = [];
    const h = w.system('move', { query: { all: [Position, Velocity] } }, (q, dt, world) => {
      args = [q, dt, world];
      for (const c of q.chunks) {
        const p = c.col(Position);
        const v = c.col(Velocity);
        for (let i = 0; i < c.count; i++) {
          p.x[i] += v.vx[i] * dt;
          p.y[i] += v.vy[i] * dt;
        }
      }
    });
    expect(h.name).toBe('move');
    expect(h.group).toBe('update');
    expect(h.order).toBe(0);
    expect(h.enabled).toBe(true);
    w.update(0.5);
    expect(args[0]).toBe(w.query({ all: [Velocity, Position] }));
    expect(args[1]).toBe(0.5);
    expect(args[2]).toBe(w);
    expect(w.get(e, Position)).toEqual({ x: 1, y: 1.5 });
  });

  it('accepts an existing Query object', () => {
    const w = new World();
    const q = w.query({ all: [Position] });
    let got: Query | null = null;
    w.system('s', { query: q }, (qq) => (got = qq));
    w.update();
    expect(got).toBe(q);
  });

  it('update default dt is 0', () => {
    const w = new World();
    let dt = -1;
    w.system('s', {}, (_q, d) => (dt = d));
    w.update();
    expect(dt).toBe(0);
  });

  it('runs in order, stable by registration for ties', () => {
    const w = new World();
    const log: string[] = [];
    w.system('c', { order: 1 }, () => log.push('c'));
    w.system('a', { order: -5 }, () => log.push('a'));
    w.system('d', { order: 1 }, () => log.push('d'));
    w.system('b', {}, () => log.push('b'));
    w.system('e', { order: 1 }, () => log.push('e'));
    w.update();
    expect(log).toEqual(['a', 'b', 'c', 'd', 'e']);
  });

  it('groups: update runs only the given group (default "update")', () => {
    const w = new World();
    const log: string[] = [];
    w.system('u', {}, () => log.push('u'));
    w.system('r', { group: 'render' }, () => log.push('r'));
    w.system('u2', { group: 'update' }, () => log.push('u2'));
    w.update(1);
    expect(log).toEqual(['u', 'u2']);
    log.length = 0;
    w.update(1, 'render');
    expect(log).toEqual(['r']);
    log.length = 0;
    w.update(1, 'nonexistent');
    expect(log).toEqual([]);
  });

  it('enabled=false skips the system', () => {
    const w = new World();
    let n = 0;
    const h = w.system('s', {}, () => n++);
    w.update();
    h.enabled = false;
    w.update();
    expect(n).toBe(1);
    h.enabled = true;
    w.update();
    expect(n).toBe(2);
  });

  it('removeSystem unregisters a function system', () => {
    const w = new World();
    let n = 0;
    const h: SystemHandle = w.system('s', {}, () => n++);
    w.update();
    w.removeSystem(h);
    w.update();
    expect(n).toBe(1);
    expect(() => w.removeSystem(h)).not.toThrow();
  });

  it('removing a later system during update prevents it from running', () => {
    const w = new World();
    const log: string[] = [];
    let h2: SystemHandle | null = null;
    w.system('1', {}, () => {
      log.push('1');
      w.removeSystem(h2!);
    });
    h2 = w.system('2', {}, () => log.push('2'));
    w.update();
    expect(log).toEqual(['1']);
  });
});

describe('class systems', () => {
  it('addSystem constructs with world, field query works, onCreate called', () => {
    const w = new World();
    const log: string[] = [];
    class Move extends System {
      q = this.query({ all: [Position, Velocity] });
      onCreate() {
        log.push('create');
      }
      onUpdate(dt: number) {
        log.push(`update ${dt}`);
        for (const c of this.q.chunks) {
          const p = c.col(Position);
          const v = c.col(Velocity);
          for (let i = 0; i < c.count; i++) p.x[i] += v.vx[i] * dt;
        }
      }
      onDestroy() {
        log.push('destroy');
      }
    }
    const e = w.spawn([Position, Velocity]);
    w.set(e, Velocity, { vx: 4 });
    const m = w.addSystem(Move);
    expect(m).toBeInstanceOf(Move);
    expect(m.world).toBe(w);
    expect(m.q).toBe(w.query({ all: [Position, Velocity] }));
    expect(m.group).toBe('update');
    expect(m.order).toBe(0);
    expect(m.enabled).toBe(true);
    expect(typeof m.name).toBe('string');
    expect(log).toEqual(['create']);
    w.update(2);
    expect(w.get(e, Position)!.x).toBe(8);
    w.removeSystem(m);
    expect(log).toEqual(['create', 'update 2', 'destroy']);
    w.update(2);
    expect(log.length).toBe(3);
  });

  it('onCreate/onDestroy are optional', () => {
    const w = new World();
    let n = 0;
    class S extends System {
      onUpdate() {
        n++;
      }
    }
    const s = w.addSystem(S);
    w.update();
    w.removeSystem(s);
    w.update();
    expect(n).toBe(1);
  });

  it('group/order options and mixing with function systems', () => {
    const w = new World();
    const log: string[] = [];
    class A extends System {
      onUpdate() {
        log.push('A');
      }
    }
    class B extends System {
      onUpdate() {
        log.push('B');
      }
    }
    const b = w.addSystem(B, { order: 2 });
    w.system('f', { order: 1 }, () => log.push('f'));
    const a = w.addSystem(A, { order: 2 });
    w.addSystem(A, { group: 'late' });
    expect(b.order).toBe(2);
    w.update();
    expect(log).toEqual(['f', 'B', 'A']);
    log.length = 0;
    a.enabled = false;
    w.update();
    expect(log).toEqual(['f', 'B']);
    log.length = 0;
    w.update(0, 'late');
    expect(log).toEqual(['A']);
  });

  it('onDestroy called exactly once even if removed twice', () => {
    const w = new World();
    let d = 0;
    class S extends System {
      onUpdate() {}
      onDestroy() {
        d++;
      }
    }
    const s = w.addSystem(S);
    w.removeSystem(s);
    w.removeSystem(s);
    expect(d).toBe(1);
  });
});

describe('deferred structural changes in systems', () => {
  it('changes inside a system are flushed after that system finishes (visible to next system)', () => {
    const w = new World();
    for (let i = 0; i < 5; i++) w.spawn([Position]);
    const log: number[] = [];
    w.system('kill', { query: { all: [Position] }, order: 0 }, (q) => {
      for (const c of q.chunks) {
        for (let i = 0; i < c.count; i++) {
          if (i % 2 === 0) w.add(c.entities[i], Dead);
        }
      }
      // raw chunk loop: still deferred inside system
      expect(w.query({ all: [Dead] }).count()).toBe(0);
    });
    w.system('check', { query: { all: [Dead] }, order: 1 }, (q) => {
      log.push(q.count());
    });
    w.update();
    expect(log).toEqual([3]);
  });

  it('destroy in a raw chunk loop inside a system does not corrupt iteration', () => {
    const w = new World();
    const es: number[] = [];
    for (let i = 0; i < 10; i++) es.push(w.spawn([Position]));
    let visited = 0;
    w.system('destroyAll', { query: { all: [Position] } }, (q) => {
      for (const c of q.chunks) {
        for (let i = 0; i < c.count; i++) {
          visited++;
          w.destroy(c.entities[i]);
        }
      }
    });
    w.update();
    expect(visited).toBe(10);
    for (const e of es) expect(w.isAlive(e)).toBe(false);
    expect(w.query({ all: [Position] }).count()).toBe(0);
  });

  it('spawn inside a system returns a live reserved id, placed after the system', () => {
    const w = new World();
    let e = -1;
    let hasDuring = true;
    w.system('spawner', {}, () => {
      e = w.spawn([Position]);
      hasDuring = w.has(e, Position);
      expect(w.isAlive(e)).toBe(true);
    });
    let seenCount = -1;
    w.system('after', { order: 1, query: { all: [Position] } }, (q) => (seenCount = q.count()));
    w.update();
    expect(hasDuring).toBe(false);
    expect(w.has(e, Position)).toBe(true);
    expect(seenCount).toBe(1);
  });

  it('forEach inside a system: flush happens after system, not after forEach', () => {
    const w = new World();
    w.spawn([Position]);
    let countAfterForEach = -1;
    w.system('s', { query: { all: [Position] } }, (q) => {
      q.forEach((e) => w.destroy(e));
      countAfterForEach = q.count();
    });
    w.update();
    expect(countAfterForEach).toBe(1);
    expect(w.query({ all: [Position] }).count()).toBe(0);
  });

  it('exception in a system still flushes and leaves world usable', () => {
    const w = new World();
    let e = -1;
    w.system('bad', {}, () => {
      e = w.spawn([Position]);
      throw new Error('bad');
    });
    expect(() => w.update()).toThrow('bad');
    expect(w.has(e, Position)).toBe(true);
    const e2 = w.spawn([Position]);
    expect(w.has(e2, Position)).toBe(true);
  });

  it('systems added during update do not break the current update', () => {
    const w = new World();
    const log: string[] = [];
    w.system('adder', {}, () => {
      log.push('adder');
      if (log.length === 1) w.system('late', { order: 10 }, () => log.push('late'));
    });
    w.update();
    w.update();
    expect(log[0]).toBe('adder');
    expect(log).toContain('late');
  });
});
