// The mock backend: the routes of walkaround/app.py over the fixture run in mock/fixture/ (built by build_fixture.py),
// plus simulated live runs so the upload page, the seal card and the live driver work without the FastAPI app.
// Mounted by mock/plugin.ts under `vite` and `vite preview`, and by mock/standalone.ts on its own.
import { createHash, randomBytes } from 'node:crypto';
import { createReadStream, existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { dirname, extname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { replayWaits } from '../src/state/schedule.ts';
import type { Event, ReplayMode, RunManifest, VideoRecord } from '../src/contracts/types.ts';

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURES = join(HERE, 'fixture');
const WALK_CLIP = resolve(HERE, '..', '..', 'data', 'test', 'pingpong.mp4');
const DELAYS = JSON.parse(readFileSync(join(HERE, '..', 'kiosk', 'delays.json'), 'utf8')) as Record<string, unknown>;
const CHUNK_MAX = 9_437_184; // thresholds.json media.chunk_route_max_body_bytes
const CHUNK_BYTES = 8_388_608;
const SAFE_ID = /^[A-Za-z0-9_-]{1,64}$/;
const ROLE = /^(pickup|return|topup_[0-9]+|claim_[0-9]+)$/;
const TERMINAL = new Set(['run.completed', 'run.failed']);
const TYPES: Record<string, string> = {
  '.json': 'application/json',
  '.gz': 'application/gzip',
  '.mp4': 'video/mp4',
  '.m4a': 'audio/mp4',
  '.jpg': 'image/jpeg',
  '.png': 'image/png',
  '.html': 'text/html; charset=utf-8',
  '.ndjson': 'application/x-ndjson',
};

interface StoredRun {
  run: RunManifest;
  dir: string; // where its files live (a live mock run reads the sample's)
  events: Event[]; // emitted so far
  done: boolean;
  listeners: Set<(e: Event) => void>;
  videos: Record<string, VideoRecord>; // uploads of a live mock run
  expected: string[];
}

export type Next = (err?: unknown) => void;

function delays(): Record<string, number> {
  return Object.fromEntries(Object.entries(DELAYS).filter(([k, v]) => !k.startsWith('_') && typeof v === 'number')) as Record<string, number>;
}

function loadFixtures(): Map<string, StoredRun> {
  const runs = new Map<string, StoredRun>();
  if (!existsSync(FIXTURES)) return runs;
  for (const id of readdirSync(FIXTURES).sort()) {
    const dir = join(FIXTURES, id);
    if (!existsSync(join(dir, 'run.json'))) continue;
    const run = JSON.parse(readFileSync(join(dir, 'run.json'), 'utf8')) as RunManifest;
    const events = readFileSync(join(dir, 'events.ndjson'), 'utf8')
      .split('\n')
      .filter(Boolean)
      .map((line) => JSON.parse(line) as Event);
    runs.set(id, { run, dir, events, done: true, listeners: new Set(), videos: {}, expected: run.kind === 'pair' ? ['pickup', 'return'] : ['pickup'] });
  }
  return runs;
}

function json(res: ServerResponse, status: number, body: unknown): void {
  const data = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(data) });
  res.end(data);
}

function empty(res: ServerResponse, status: number): void {
  res.writeHead(status);
  res.end();
}

async function readBody(req: IncomingMessage, limit: number): Promise<Buffer | null> {
  const parts: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const b = chunk as Buffer;
    size += b.length;
    if (size > limit) return null;
    parts.push(b);
  }
  return Buffer.concat(parts);
}

function serveFile(req: IncomingMessage, res: ServerResponse, path: string): void {
  const size = statSync(path).size;
  const type = TYPES[extname(path)] ?? 'application/octet-stream';
  const range = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range ?? '');
  if (range && (range[1] || range[2])) {
    const start = range[1] ? Number(range[1]) : Math.max(0, size - Number(range[2]));
    const end = range[1] && range[2] ? Math.min(size - 1, Number(range[2])) : size - 1;
    if (start >= size || start > end) {
      res.writeHead(416, { 'content-range': `bytes */${size}` });
      res.end();
      return;
    }
    res.writeHead(206, { 'content-type': type, 'content-length': end - start + 1, 'content-range': `bytes ${start}-${end}/${size}`, 'accept-ranges': 'bytes' });
    if (req.method === 'HEAD') res.end();
    else createReadStream(path, { start, end }).pipe(res);
    return;
  }
  res.writeHead(200, { 'content-type': type, 'content-length': size, 'accept-ranges': 'bytes' });
  if (req.method === 'HEAD') res.end();
  else createReadStream(path).pipe(res);
}

function sseLine(e: Event): string {
  return `id: ${e.seq}\nevent: ${e.kind}\ndata: ${JSON.stringify(e)}\n\n`;
}

export interface MockApi {
  handle(req: IncomingMessage, res: ServerResponse, next: Next): void;
}

export function createMockApi(): MockApi {
  const runs = loadFixtures();
  const sample = [...runs.keys()][0] ?? null;
  const chunks = new Map<string, Map<number, Buffer>>();
  let counter = 0;

  const emit = (stored: StoredRun, e: Event) => {
    stored.events.push(e);
    if (e.kind === 'stage' && typeof e.payload['name'] === 'string') stored.run.status = e.payload['name'] as RunManifest['status'];
    if (e.kind === 'seal.manifest') {
      // The uploaded file carries no camera time in the mock, so the third clock says so.
      stored.run.clocks.sealed_at = e.ts;
      stored.run.clocks.sealed_minutes_after_camera = 'camera time not present';
    }
    if (e.kind === 'files.uploaded') stored.run.clocks.first_model_call_at ??= e.ts;
    if (TERMINAL.has(e.kind)) stored.done = true;
    for (const l of stored.listeners) l(e);
  };

  // A live mock run replays the sample's events under its own id, paced like a fixed-delay replay.
  const startPipeline = (stored: StoredRun) => {
    const source = sample ? runs.get(sample)! : null;
    if (!source) return;
    const waits = replayWaits(source.events, 'fixed', delays());
    let i = 0;
    const step = () => {
      if (i >= source.events.length) return;
      const original = source.events[i]!;
      const payload = { ...original.payload };
      if (original.kind === 'upload.done' || original.kind === 'upload.started') {
        const v = stored.videos[String(payload['role'])];
        if (v && original.kind === 'upload.done') Object.assign(payload, { bytes: v.bytes, sha256: v.sha256_server, filename: v.original_filename });
      }
      emit(stored, { ...original, payload, run_id: stored.run.run_id, ts: new Date().toISOString() });
      i += 1;
      if (i < source.events.length) setTimeout(step, waits[i]);
    };
    setTimeout(step, 50);
  };

  const fileFor = (stored: StoredRun, rel: string): string | null => {
    const parts = rel.split('/');
    if (!rel || rel.startsWith('/') || parts.some((p) => p === '..' || p === '' || p.startsWith('.'))) return null;
    if (parts[0] === 'quarantine' || parts[0] === 'uploads' || rel.endsWith('.proxy.mp4') || rel.endsWith('.orig')) return null;
    if (/^media\/[a-z_0-9]+\.proxy_blurred\.mp4$/.test(rel)) return existsSync(WALK_CLIP) ? WALK_CLIP : null;
    const path = join(stored.dir, ...parts);
    return existsSync(path) && statSync(path).isFile() ? path : null;
  };

  const route = async (req: IncomingMessage, res: ServerResponse): Promise<boolean> => {
    const url = new URL(req.url ?? '/', 'http://mock.local');
    const p = url.pathname;
    const m = (re: RegExp) => re.exec(p);
    let hit: RegExpExecArray | null;

    if (p === '/healthz') return json(res, 200, { ok: true, engine: 'mock', probe_ok: true, free_bytes: 10_000_000_000 }), true;
    if (p === '/api/status') return json(res, 200, { paused: false, pickup_left: 6, pairs_left: 1, queue_position: null, queue_eta_minutes: null, engine: 'mock' }), true;
    if (p === '/api/sample') {
      if (!sample) return json(res, 404, { detail: 'no sample pair on this server' }), true;
      const day = runs.get(sample)!.run.created_at.slice(0, 10);
      return json(res, 200, { run_id: sample, precomputed: true, recorded_on: day, badge: `replayed from a run on ${day}`, replay: `/api/runs/${sample}/replay?mode=fixed` }), true;
    }
    if (p === '/api/runs' && req.method === 'GET') return json(res, 200, [...runs.values()].map((r) => ({ ...r.run, thresholds: {} }))), true;
    if (p === '/api/runs' && req.method === 'POST') {
      const body = JSON.parse((await readBody(req, 1_048_576))?.toString() || '{}') as { kind?: string; mode?: string; keep_plate?: boolean };
      const base = sample ? runs.get(sample)!.run : null;
      if (!base) return json(res, 503, { reason: 'the mock has no fixture run to copy' }), true;
      counter += 1;
      const id = `mocklive-${counter}-${randomBytes(3).toString('hex')}`;
      const kind = body.kind === 'pickup_check' ? 'pickup_check' : 'pair';
      const run: RunManifest = {
        ...base,
        run_id: id,
        created_at: new Date().toISOString().replace(/\.\d+Z$/, 'Z'),
        kind,
        status: 'created',
        live_or_replay: 'live',
        keep: false,
        keep_plate: Boolean(body.keep_plate),
        clocks: { sealed_at: null, first_model_call_at: null, sealed_minutes_after_camera: null, camera_time: null },
        events_head_hash: null,
      };
      runs.set(id, { run, dir: runs.get(sample!)!.dir, events: [], done: false, listeners: new Set(), videos: {}, expected: kind === 'pair' ? ['pickup', 'return'] : ['pickup'] });
      return json(res, 201, run), true;
    }

    if ((hit = m(/^\/api\/uploads\/([^/]+)\/chunks\/(\d+)$/)) && req.method === 'PUT') {
      const [, uploadId, n] = hit;
      if (!SAFE_ID.test(uploadId!)) return json(res, 400, { detail: 'bad upload id or chunk index' }), true;
      const data = await readBody(req, CHUNK_MAX);
      if (!data) return json(res, 413, { detail: `chunk over ${CHUNK_MAX} bytes` }), true;
      const map = chunks.get(uploadId!) ?? new Map<number, Buffer>();
      map.set(Number(n), data);
      chunks.set(uploadId!, map);
      return empty(res, 204), true;
    }
    if ((hit = m(/^\/api\/uploads\/([^/]+)\/finalize$/)) && req.method === 'POST') {
      const body = JSON.parse((await readBody(req, 1_048_576))?.toString() || '{}') as { run_id?: string; role?: string; sha256_browser?: string; bytes?: number; filename?: string };
      const stored = runs.get(body.run_id ?? '');
      if (!stored) return json(res, 404, { detail: 'no such run' }), true;
      if (!ROLE.test(body.role ?? '') || !/^[0-9a-f]{64}$/i.test(body.sha256_browser ?? '')) return json(res, 400, { detail: 'bad upload id or role' }), true;
      // orchestrator._check_role: a finalize repeated after a lost response gets the stored record back.
      const existing = stored.videos[body.role!];
      if (existing?.hash_match && existing.sha256_server === body.sha256_browser!.toLowerCase()) return json(res, 200, existing), true;
      if (existing && stored.events.length) return json(res, 400, { detail: `the ${body.role} walk of this run is already sealed; start a new run to send another file` }), true;
      const expected = Math.max(1, Math.ceil((body.bytes ?? 0) / CHUNK_BYTES));
      const map = chunks.get(hit[1]!) ?? new Map<number, Buffer>();
      const missing = Array.from({ length: expected }, (_, i) => i).filter((i) => !map.has(i));
      if (missing.length) return json(res, 400, { detail: 'chunks missing; resume upload', missing }), true;
      const digest = createHash('sha256');
      let size = 0;
      for (let i = 0; i < expected; i += 1) {
        digest.update(map.get(i)!);
        size += map.get(i)!.length;
      }
      chunks.delete(hit[1]!);
      const server = digest.digest('hex');
      const browser = body.sha256_browser!.toLowerCase();
      if (server !== browser) return json(res, 409, { sha256_browser: browser, sha256_server: server, detail: 'hash mismatch; upload again' }), true;
      const record: VideoRecord = {
        role: body.role!,
        original_filename: body.filename ?? 'upload',
        bytes: size,
        sha256_browser: browser,
        sha256_server: server,
        hash_match: true,
        received_at: new Date().toISOString().replace(/\.\d+Z$/, 'Z'),
        chunks_expected: expected,
        chunks_received: expected,
        ffprobe: null,
        proxy: null,
        proxy_blurred: null,
        wav: null,
        files_api: null,
        quarantine_path: `quarantine/${body.role}.orig`,
        quarantine_deleted_at: null,
      };
      stored.videos[body.role!] = record;
      if (stored.run.status === 'created' && stored.events.length === 0 && stored.expected.every((r) => stored.videos[r])) startPipeline(stored);
      return json(res, 200, record), true;
    }

    hit = m(/^\/api\/runs\/([^/]+)(\/.*)?$/);
    if (!hit) return false;
    const stored = runs.get(decodeURIComponent(hit[1]!));
    if (!stored) return json(res, 404, { detail: 'no such run' }), true;
    const rest = hit[2] ?? '';

    if (rest === '' && req.method === 'GET') return json(res, 200, stored.run), true;
    if (rest === '/events') {
      const last = Number(req.headers['last-event-id'] ?? -1);
      const since = Number.isFinite(last) ? last : -1;
      const backlog = stored.events.filter((e) => e.seq > since);
      if (stored.done && !backlog.length) return empty(res, 204), true;
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      for (const e of backlog) res.write(sseLine(e));
      if (stored.done) return res.end(), true;
      const ping = setInterval(() => res.write(': ping\n\n'), 15_000);
      const listener = (e: Event) => {
        res.write(sseLine(e));
        if (TERMINAL.has(e.kind)) close();
      };
      const close = () => {
        clearInterval(ping);
        stored.listeners.delete(listener);
        res.end();
      };
      stored.listeners.add(listener);
      req.on('close', close);
      return true;
    }
    if (rest === '/replay') {
      const mode = (url.searchParams.get('mode') ?? 'fixed') as ReplayMode;
      if (!['fixed', 'recorded', 'instant'].includes(mode)) return json(res, 400, { detail: 'mode is fixed, recorded or instant' }), true;
      return json(res, 200, { run_id: stored.run.run_id, mode, recorded_on: stored.run.created_at.slice(0, 10), delays_ms: mode === 'fixed' ? delays() : {}, events: stored.events }), true;
    }
    if ((hit = /^\/files\/(.+)$/.exec(rest))) {
      const relPath = hit[1]!.split('/').map(decodeURIComponent).join('/');
      const role = /^videos\/([a-z_0-9]+)\.json$/.exec(relPath)?.[1];
      if (role && stored.videos[role]) return json(res, 200, stored.videos[role]), true;
      if (role && stored.run.live_or_replay === 'live') return json(res, 404, { detail: 'no such file' }), true;
      const path = fileFor(stored, relPath);
      if (!path) return json(res, 404, { detail: 'no such file' }), true;
      serveFile(req, res, path);
      return true;
    }
    if (rest === '/sheet') {
      const path = join(stored.dir, 'sheet', 'sheet.html');
      if (!stored.done || !existsSync(path)) return json(res, 404, { detail: 'sheet not written yet' }), true;
      serveFile(req, res, path);
      return true;
    }
    if (rest === '/sheet.json') return json(res, 404, { detail: 'sheet not written yet' }), true;
    if (rest === '/sheet.pdf') return json(res, 404, { detail: 'PDF retry' }), true;
    if (rest === '/bundle.zip') return json(res, 404, { detail: 'bundle not written yet' }), true;
    if (rest === '/topup' && req.method === 'POST') {
      await readBody(req, 1 << 30);
      const coverage = JSON.parse(readFileSync(join(stored.dir, 'derived', 'coverage.json'), 'utf8')) as Record<string, unknown> & { per_role: Record<string, unknown> };
      await new Promise((r) => setTimeout(r, 1500));
      coverage.per_role['topup_1'] = { rear_bumper: { state: 'filmed', state_basis: 'c', seconds: 9.6, best_kf: null, reason: '' } };
      return json(res, 200, { ...coverage, missing: ['roof'], topup_text: 'roof: not run, add 5 seconds?', rerun_seconds: 41.2, odometer: { asked: true, filmed: true, frame_path: null } }), true;
    }
    return false;
  };

  return {
    handle(req, res, next) {
      const url = req.url ?? '/';
      if (url.startsWith('/api/claims') && req.method === 'POST') {
        void (async () => {
          await readBody(req, 60 * 1024 * 1024);
          await new Promise((r) => setTimeout(r, 1200));
          const claim = sample ? join(runs.get(sample)!.dir, 'derived', 'claim.json') : null;
          if (!claim || !existsSync(claim)) json(res, 501, { detail: 'claim-first is not wired yet' });
          else serveFile(req, res, claim);
        })();
        return;
      }
      if (!url.startsWith('/api/') && !url.startsWith('/healthz')) {
        next();
        return;
      }
      route(req, res)
        .then((done) => {
          if (!done) json(res, 404, { detail: 'not a mock route' });
        })
        .catch((err: unknown) => json(res, 500, { detail: err instanceof Error ? err.message : String(err) }));
    },
  };
}
