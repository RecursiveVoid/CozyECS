// tests/ecs.test.ts
import { Component } from '../src/Component';
import { ComponentStorage } from '../src/Storage';
import { System } from '../src/System';
import { World } from '../src/World';
import { ComponentDefinition } from '../src/Types';

describe('ECS', () => {
  describe('Component', () => {
    it('should add component definitions', () => {
      const comp = new Component()
        .add('number', 'x')
        .add('boolean', 'active');

      const defs = comp.getDefinitions();
      expect(defs).toEqual([
        { type: 'number', name: 'x' },
        { type: 'boolean', name: 'active' },
      ]);
    });
  });

  describe('ComponentStorage', () => {
    let storage: ComponentStorage;

    beforeEach(() => {
      storage = new ComponentStorage();
    });

    it('should store and retrieve number component', () => {
      const def: ComponentDefinition = { type: 'number', name: 'x' };
      storage.addComponent(def);
      storage.setComponent(1, 'x', 42);
      expect(storage.getComponent(1, 'x')).toBe(42);
    });

    it('should store and retrieve boolean component', () => {
      const def: ComponentDefinition = { type: 'boolean', name: 'active' };
      storage.addComponent(def);
      storage.setComponent(2, 'active', 1);
      expect(storage.getComponent(2, 'active')).toBe(1);
    });
  });

  describe('System', () => {
    let storage: ComponentStorage;
    let system: System;

    beforeEach(() => {
      storage = new ComponentStorage();
      system = new System();

      const defs: ComponentDefinition[] = [
        { type: 'number', name: 'x' },
        { type: 'number', name: 'y' },
      ];

      defs.forEach(def => storage.addComponent(def));
      storage.setComponent(1, 'x', 10);
      storage.setComponent(1, 'y', 20);
    });

    it('should call onUpdate for matching entities', () => {
      const mockFn = jest.fn();
      system.get('x', 'y').onUpdate(mockFn);
      system.runUpdate([1], storage);
      expect(mockFn).toHaveBeenCalledWith(1);
    });

    it('should call onInit for matching entities', () => {
      const mockFn = jest.fn();
      system.get('x', 'y').onInit(mockFn);
      system.runInit([1], storage);
      expect(mockFn).toHaveBeenCalledWith(1);
    });

    it('should call onDestroy for matching entities', () => {
      const mockFn = jest.fn();
      system.get('x', 'y').onDestroy(mockFn);
      system.runDestroy([1], storage);
      expect(mockFn).toHaveBeenCalledWith(1);
    });
  });

  describe('World', () => {
    it('should create entity and assign components', () => {
      const world = new World();
      const comp = new Component().add('number', 'x').add('boolean', 'active');
      world.registerComponent(comp);

      const entity = world.createEntity();
      world.set(entity, 'x', 100).set(entity, 'active', 1);
      // no errors mean components are set properly
    });

    it('should run tick and call system updates', () => {
      const world = new World();
      const comp = new Component().add('number', 'x').add('number', 'y');
      const system = new System();

      const mockUpdate = jest.fn();
      system.get('x', 'y').onUpdate(mockUpdate);

      world.registerComponent(comp).registerSystem(system);

      const entity = world.createEntity();
      world.set(entity, 'x', 5).set(entity, 'y', 10);

      world.tick();
      expect(mockUpdate).toHaveBeenCalledWith(entity);
    });
  });
});
