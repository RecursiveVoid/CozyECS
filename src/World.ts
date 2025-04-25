import { Component } from './Component';
import { System } from './System';
import { ComponentStorage } from './Storage';

export class World {
  private _nextEntityId = 0;
  private _entities = new Set<number>();
  private _storage = new ComponentStorage();
  private _systems: System[] = [];

  public registerComponent(component: Component): this {
    for (const def of component.getDefinitions()) {
      this._storage.addComponent(def);
    }
    return this;
  }

  public registerSystem(system: System): this {
    this._systems.push(system);
    return this;
  }

  public createEntity(): number {
    const id = this._nextEntityId++;
    this._entities.add(id);
    return id;
  }

  public set(entityId: number, name: string, value: any): this {
    this._storage.setComponent(entityId, name, value);
    return this;
  }
  
  public tick(): void {
    const all = Array.from(this._entities);
    for (const system of this._systems) {
      system.runUpdate(all, this._storage);
    }
  }

  public init(): void {
    const all = Array.from(this._entities);
    for (const system of this._systems) {
      system.runInit(all, this._storage);
    }
  }

  public destroy(): void {
    const all = Array.from(this._entities);
    for (const system of this._systems) {
      system.runDestroy(all, this._storage);
    }
  }
}
