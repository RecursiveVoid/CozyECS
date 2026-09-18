import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import typescript from '@rollup/plugin-typescript';
import dts from 'rollup-plugin-dts';
import strip from 'rollup-plugin-strip';
import terser from '@rollup/plugin-terser';
import { visualizer } from 'rollup-plugin-visualizer';
import { readdirSync, rmSync } from 'node:fs';

// The core entry's TypeScript pass declares EVERY file under src/, including
// src/gpu/**, which lands as dist/gpu/{cpu,device,ir,...}.d.ts next to the
// bundled dist/gpu/index.d.ts; the gpu pass leaves its per-file declarations in
// dist/gpu/types/. Both are intermediate. Once the gpu .d.ts is bundled, delete
// them so the published dist/gpu/ is exactly index.{cjs,esm.js,d.ts} + maps.
// src/gpu/** imports the core for TYPES only, via relative paths ('../world').
// Left alone, rollup-plugin-dts inlines COPIES of World, Archetype, Query... into
// dist/gpu/index.d.ts, and because those classes have private members the
// copies are nominally distinct: `kernelSystem(worldFromCozyecs, ...)` would not
// type-check. Resolve every core import to the package itself instead, so the
// gpu declarations reference the one set of core types in dist/index.d.ts.
const coreTypesAsPackage = () => ({
  name: 'core-types-as-package',
  resolveId(source, importer) {
    if (importer && /[\\/]dist[\\/]gpu[\\/]types[\\/]gpu[\\/]/.test(importer) && source.startsWith('../')) {
      return { id: 'cozyecs', external: true };
    }
    return null;
  },
});

const cleanGpuDeclarations = () => ({
  name: 'clean-gpu-declarations',
  writeBundle() {
    rmSync('dist/gpu/types', { recursive: true, force: true });
    for (const f of readdirSync('dist/gpu')) {
      if (f.endsWith('.d.ts') && f !== 'index.d.ts') rmSync(`dist/gpu/${f}`);
    }
  },
});

// Shared plugin stack. `declarationDir` differs per entry so the two bundles'
// .d.ts output cannot collide, and the visualizer only runs for the core entry.
// console.warn/error survive when `keepWarnings` is set. The gpu entry needs
// that: falling back to the CPU backend because no WebGPU device was found is
// silent otherwise, and a silent fallback reads as "the GPU did nothing".
const plugins = ({ declarationDir, visualize = false, keepWarnings = false }) => {
  const dropped = keepWarnings
    ? ['console.log', 'console.info', 'console.debug', 'console.trace']
    : ['console.*'];
  return [
    resolve(),
    commonjs(),
    typescript({
      tsconfig: './tsconfig.json',
      declaration: true,
      declarationDir,
      declarationMap: false,
      noEmitOnError: true,
    }),
    terser({
      compress: {
        drop_console: keepWarnings ? ['log', 'info', 'debug', 'trace'] : true,
        pure_getters: true,
        passes: 2,
      },
    }),
    strip({
      functions: [...dropped, 'assert.*'],
    }),
    ...(visualize ? [visualizer({ open: false })] : []),
  ];
};

export default [
  // Core entry: `cozyecs`. Must not reach into src/gpu/**.
  {
    input: 'src/index.ts',
    output: [
      {
        file: 'dist/index.cjs',
        format: 'cjs',
        sourcemap: true,
      },
      {
        file: 'dist/index.esm.js',
        format: 'esm',
        sourcemap: true,
      },
    ],
    plugins: plugins({ declarationDir: 'dist', visualize: true }),
  },
  {
    input: 'dist/index.d.ts',
    output: [{ file: 'dist/index.d.ts', format: 'es' }],
    plugins: [dts()],
  },
  // Optional entry: `cozyecs/gpu`. Built as its own bundle so the core never
  // carries WGSL codegen, a JS parser or any WebGPU runtime. `webgpu` is left
  // external: on Node the host installs it and hands the instance to
  // setGPUProvider(), and in a browser navigator.gpu is used instead.
  {
    input: 'src/gpu/index.ts',
    output: [
      {
        file: 'dist/gpu/index.cjs',
        format: 'cjs',
        sourcemap: true,
      },
      {
        file: 'dist/gpu/index.esm.js',
        format: 'esm',
        sourcemap: true,
      },
    ],
    external: ['webgpu'],
    plugins: plugins({ declarationDir: 'dist/gpu/types', keepWarnings: true }),
  },
  {
    input: 'dist/gpu/types/gpu/index.d.ts',
    output: [
      {
        file: 'dist/gpu/index.d.ts',
        format: 'es',
        // rollup-plugin-dts drops triple-slash references, but consumers need
        // the ambient WebGPU types for GPUDevice/GPUAdapter to resolve.
        banner: '/// <reference types="@webgpu/types" />',
      },
    ],
    external: ['cozyecs'],
    plugins: [coreTypesAsPackage(), dts(), cleanGpuDeclarations()],
  },
];
