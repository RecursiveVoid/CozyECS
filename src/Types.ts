export type ComponentType = 'boolean' | 'number' | 'string' | 'null';

export interface ComponentDefinition {
  type: ComponentType;
  name: string;
}

export type QueryCallback = (entityId: number) => void;