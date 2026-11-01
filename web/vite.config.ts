import react from '@vitejs/plugin-react';
import { defineConfig, type ProxyOptions } from 'vite';

import { mockApi } from './mock/plugin.ts';

// WALKAROUND_API=http://127.0.0.1:8000 sends /api and /healthz to the FastAPI app; without it the mock answers.
const backend = process.env['WALKAROUND_API'];
const proxy: Record<string, ProxyOptions> | undefined = backend
  ? {
      '/api': { target: backend, changeOrigin: false },
      '/healthz': { target: backend, changeOrigin: false },
    }
  : undefined;

export default defineConfig({
  plugins: [react(), ...(backend ? [] : [mockApi()])],
  // Carbon's Sass still uses syntax newer Sass warns about; those warnings are about node_modules, not this app.
  css: { preprocessorOptions: { scss: { quietDeps: true, silenceDeprecations: ['if-function', 'global-builtin', 'import'] } } },
  server: { port: 5173, proxy },
  preview: { port: 4173, proxy },
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: false,
    target: 'es2022',
    // The run page's lazy chunk carries the @meta-sam player and parser (about 0.5 MB, 136 kB gzipped); only that page loads it.
    chunkSizeWarningLimit: 600,
  },
  worker: {
    format: 'es',
  },
});
