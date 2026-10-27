import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

import { mockApiPlugin } from './mock/plugin.ts';

// WALKAROUND_API=http://127.0.0.1:8000 proxies /api and /healthz to the FastAPI app instead of the mock.
const backend = process.env['WALKAROUND_API'];

export default defineConfig({
  plugins: [react(), ...(backend ? [] : [mockApiPlugin()])],
  server: {
    port: 5173,
    proxy: backend
      ? {
          '/api': { target: backend, changeOrigin: false },
          '/healthz': { target: backend, changeOrigin: false },
        }
      : undefined,
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    target: 'es2022',
  },
  worker: {
    format: 'es',
  },
});
