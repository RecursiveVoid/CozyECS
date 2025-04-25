import { ComponentDefinition } from './Types';

export class ComponentStorage {
  private _views = new Map<string, any>();
  private _entityToIndex = new Map<number, number>();
  private _nextIndex = 0;

  public addComponent(def: ComponentDefinition): void {
    const { type, name } = def;
    const capacity = 10000;
    let typedArray: any;

    switch (type) {
      case 'number':
        typedArray = new Float32Array(capacity);
        break;
      case 'boolean':
        typedArray = new Uint8Array(capacity);
        break;
      case 'string':
        typedArray = new Array(capacity);
        break;
      default:
        throw new Error(`Unsupported component type: ${type}`);
    }

    this._views.set(name, typedArray);
  }

  public setComponent(entityId: number, name: string, value: any): void {
    let index = this._entityToIndex.get(entityId);
    if (index === undefined) {
      index = this._nextIndex++;
      this._entityToIndex.set(entityId, index);
    }

    const view = this._views.get(name);
    if (view) view[index] = value;
  }

  public getComponent(entityId: number, name: string): any {
    const index = this._entityToIndex.get(entityId);
    if (index === undefined) return undefined;
    const view = this._views.get(name);
    return view ? view[index] : undefined;
  }
}