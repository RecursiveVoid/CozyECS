import { QueryCallback } from './Types';
import { ComponentStorage } from './Storage';

export class System {
  private _components: string[] = [];
  private _initCb?: QueryCallback;
  private _updateCb?: QueryCallback;
  private _destroyCb?: QueryCallback;

  public get(...components: string[]): this {
    this._components = components;
    return this;
  }

  public onInit(cb: QueryCallback): this {
    this._initCb = cb;
    return this;
  }

  public onUpdate(cb: QueryCallback): this {
    this._updateCb = cb;
    return this;
  }

  public onDestroy(cb: QueryCallback): this {
    this._destroyCb = cb;
    return this;
  }

  public runInit(entities: number[], storage: ComponentStorage): void {
    for (const id of entities) {
      if (this._hasAllComponents(id, storage)) {
        this._initCb?.(id);
      }
    }
  }

  public runUpdate(entities: number[], storage: ComponentStorage): void {
    for (const id of entities) {
      if (this._hasAllComponents(id, storage)) {
        this._updateCb?.(id);
      }
    }
  }

  public runDestroy(entities: number[], storage: ComponentStorage): void {
    for (const id of entities) {
      if (this._hasAllComponents(id, storage)) {
        this._destroyCb?.(id);
      }
    }
  }

  private _hasAllComponents(entityId: number, storage: ComponentStorage): boolean {
    return this._components.every(name => storage.getComponent(entityId, name) !== undefined);
  }
}
