// Every backend route the browser consumes, in one file, so the integrator can check them against app.py.
import type { Claim, Coverage, ReplayMode, ReplayPayload, RunManifest, SampleResponse, StatusResponse, VideoRecord } from '../contracts/types.ts';

const enc = encodeURIComponent;

// "kiosk" adds ?view=kiosk, which app.py reads as a viewer who is not the run owner (owner-only files under keep_plate 404).
export type FileView = 'owner' | 'kiosk';

export const routes = {
  runs: () => '/api/runs',
  run: (id: string) => `/api/runs/${enc(id)}`,
  events: (id: string) => `/api/runs/${enc(id)}/events`,
  replay: (id: string, mode: ReplayMode) => `/api/runs/${enc(id)}/replay?mode=${mode}`,
  file: (id: string, relPath: string, view: FileView = 'owner') =>
    `/api/runs/${enc(id)}/files/${relPath.split('/').map(enc).join('/')}${view === 'kiosk' ? '?view=kiosk' : ''}`,
  // app.py serves the sheet routes under the files rule, so ?view=kiosk gets a 404 for a keep_plate sheet (it embeds
  // the owner's frames); the kiosk also refuses that scene itself.
  sheetHtml: (id: string, view: FileView = 'owner') => `/api/runs/${enc(id)}/sheet${view === 'kiosk' ? '?view=kiosk' : ''}`,
  sheetJson: (id: string) => `/api/runs/${enc(id)}/sheet.json`,
  sheetPdf: (id: string) => `/api/runs/${enc(id)}/sheet.pdf`,
  bundleZip: (id: string) => `/api/runs/${enc(id)}/bundle.zip`,
  chunk: (uploadId: string, n: number, runId: string) => `/api/uploads/${enc(uploadId)}/chunks/${n}?run_id=${enc(runId)}`,
  finalize: (uploadId: string) => `/api/uploads/${enc(uploadId)}/finalize`,
  topup: (id: string) => `/api/runs/${enc(id)}/topup`,
  claims: () => '/api/claims',
  status: (runId?: string | null) => (runId ? `/api/status?run_id=${enc(runId)}` : '/api/status'),
  sample: () => '/api/sample',
  healthz: () => '/healthz',
} as const;

export class ApiError extends Error {
  readonly status: number;
  readonly body: unknown;
  constructor(status: number, body: unknown) {
    super(detailOf(body) ?? `HTTP ${status}`);
    this.status = status;
    this.body = body;
  }
}

function detailOf(body: unknown): string | null {
  if (body && typeof body === 'object') {
    const b = body as { detail?: unknown; reason?: unknown };
    const d = b.detail ?? b.reason;
    if (typeof d === 'string') return d;
  }
  return typeof body === 'string' && body.length < 200 ? body : null;
}

async function readBody(res: Response): Promise<unknown> {
  const text = await res.text();
  try {
    return text ? JSON.parse(text) : null;
  } catch {
    return text;
  }
}

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url, { headers: { accept: 'application/json' } });
  const body = await readBody(res);
  if (!res.ok) throw new ApiError(res.status, body);
  return body as T;
}

async function send<T>(url: string, init: RequestInit): Promise<T> {
  const res = await fetch(url, { ...init, headers: { accept: 'application/json', ...(init.headers ?? {}) } });
  const body = await readBody(res);
  if (!res.ok) throw new ApiError(res.status, body);
  return body as T;
}

export interface CreateRunBody {
  mode: 'car' | 'apartment';
  kind: 'pair' | 'pickup_check';
  keep_plate: boolean;
}

export interface FinalizeBody {
  run_id: string;
  role: string;
  sha256_browser: string;
  bytes: number;
  filename: string;
  keep_plate: boolean;
}

export const api = {
  status: (runId?: string | null) => getJson<StatusResponse>(routes.status(runId)),
  sample: () => getJson<SampleResponse>(routes.sample()),
  runs: () => getJson<RunManifest[]>(routes.runs()),
  run: (id: string) => getJson<RunManifest>(routes.run(id)),
  replay: (id: string, mode: ReplayMode) => getJson<ReplayPayload>(routes.replay(id, mode)),
  file: <T>(id: string, rel: string, view: FileView = 'owner') => getJson<T>(routes.file(id, rel, view)),
  // A stored text file such as open_weights.ndjson, read as is.
  text: async (id: string, rel: string, view: FileView = 'owner'): Promise<string> => {
    const res = await fetch(routes.file(id, rel, view));
    if (!res.ok) throw new ApiError(res.status, await readBody(res));
    return res.text();
  },
  video: (id: string, role: string, view: FileView = 'owner') => getJson<VideoRecord>(routes.file(id, `videos/${role}.json`, view)),
  createRun: (body: CreateRunBody) =>
    send<RunManifest>(routes.runs(), { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) }),
  putChunk: async (uploadId: string, n: number, blob: Blob, runId: string): Promise<void> => {
    const res = await fetch(routes.chunk(uploadId, n, runId), { method: 'PUT', headers: { 'content-type': 'application/octet-stream' }, body: blob });
    if (!res.ok) throw new ApiError(res.status, await readBody(res));
  },
  finalize: (uploadId: string, body: FinalizeBody) =>
    send<VideoRecord>(routes.finalize(uploadId), { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) }),
  // app.py checks the optional sha256_browser against its hash on receipt and answers 409 with both on a mismatch.
  topup: (id: string, video: File, sha256Browser?: string) => {
    const form = new FormData();
    if (sha256Browser) form.append('sha256_browser', sha256Browser);
    form.append('video', video, video.name);
    return send<Coverage>(routes.topup(id), { method: 'POST', body: form });
  },
  // panel_id is the renter's one tap after exit E2 (two candidate panels) or E2b (both sides shown).
  claim: (body: { run_id: string; image: File; text: string; tire_code?: string; panel_id?: string }) => {
    const form = new FormData();
    form.append('image', body.image, body.image.name);
    form.append('run_id', body.run_id);
    form.append('text', body.text);
    if (body.tire_code) form.append('tire_code', body.tire_code);
    if (body.panel_id) form.append('panel_id', body.panel_id);
    return send<Claim>(routes.claims(), { method: 'POST', body: form });
  },
  // GET with a 1-byte range: FastAPI's FileResponse has no HEAD route, so a 404 means "PDF retry".
  exists: async (url: string): Promise<boolean> => {
    try {
      const res = await fetch(url, { headers: { range: 'bytes=0-0' } });
      return res.ok;
    } catch {
      return false;
    }
  },
};
