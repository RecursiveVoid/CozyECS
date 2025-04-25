import { ComponentDefinition, ComponentType } from './Types';

export class Component {
  private _defs: ComponentDefinition[] = [];

  public add(type: ComponentType, name: string): this {
    this._defs.push({ type, name });
    return this;
  }

  public getDefinitions(): ComponentDefinition[] {
    return this._defs;
  }
}