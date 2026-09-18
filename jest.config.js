export default {
  preset: 'ts-jest',
  testEnvironment: 'node',
  collectCoverage: true,
  coverageDirectory: 'coverage',
  coverageReporters: ['text', 'lcov'],
  moduleFileExtensions: ['ts', 'js'],
  // GPU tests import the ESM-only `webgpu` package through Node's real loader,
  // which jest's CJS runtime only permits under --experimental-vm-modules. The
  // `test` script sets it; running bare `jest` skips those tests' import.
  transform: {
    '^.+\\.ts$': ['ts-jest', { tsconfig: 'tsconfig.jest.json' }],
  },
};
