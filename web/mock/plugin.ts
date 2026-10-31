import type { Plugin } from 'vite';

import { createMockApi } from './server.ts';

// Answers /api and /healthz with the mock backend under `vite` and `vite preview` (vite.config.ts leaves it out when
// WALKAROUND_API points at the FastAPI app).
export function mockApi(): Plugin {
  const api = createMockApi();
  return {
    name: 'walkaround-mock-api',
    configureServer(server) {
      server.middlewares.use((req, res, next) => api.handle(req, res, next));
    },
    configurePreviewServer(server) {
      server.middlewares.use((req, res, next) => api.handle(req, res, next));
    },
  };
}
