// `npm run mock`: the built app (dist/) and the mock backend on one port, with no Vite and no FastAPI.
// Run `npm run build` first. PORT defaults to 5174.
import { createReadStream, existsSync, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { dirname, extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

import { createMockApi } from './server.ts';

const DIST = join(dirname(fileURLToPath(import.meta.url)), '..', 'dist');
const PORT = Number(process.env['PORT'] ?? 5174);
const TYPES: Record<string, string> = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript',
  '.css': 'text/css',
  '.woff2': 'font/woff2',
  '.woff': 'font/woff',
  '.svg': 'image/svg+xml',
  '.wasm': 'application/wasm',
};

const api = createMockApi();

createServer((req, res) => {
  api.handle(req, res, () => {
    const path = normalize(decodeURIComponent(new URL(req.url ?? '/', 'http://local').pathname));
    const asset = join(DIST, path);
    const file = path.startsWith('/assets/') && asset.startsWith(DIST) && existsSync(asset) && statSync(asset).isFile() ? asset : join(DIST, 'index.html');
    if (!existsSync(file)) {
      res.writeHead(500, { 'content-type': 'text/plain' });
      res.end('dist/ is missing: run npm run build first');
      return;
    }
    res.writeHead(200, { 'content-type': TYPES[extname(file)] ?? 'application/octet-stream' });
    createReadStream(file).pipe(res);
  });
}).listen(PORT, () => {
  console.log(`walkaround mock: http://localhost:${PORT}/ (dist/ plus the mock API)`);
});
