// Every backend route the browser consumes, in one file, so the integrator can check them against app.py.
import type { Claim, Coverage, ReplayMode, ReplayPayload, RunManifest, SampleResponse, StatusResponse, VideoRecord } from '../contracts/types.ts';

const enc = encodeURIComponent;

export const routes = {
  createRun: () => '/api/runs',
  runs: () => '/api/runs',
  run: (id: string) => `/api/runs/${enc(id)}`,
  events: (id: string) => `/api/runs/${enc(id)}/events`,
  replay: (id: string, mode: ReplayMode) => `/api/runs/${enc(id)}/replay?mode=${mode}`,
  file: (id: string, relPath: string) => `/api/runs/${enc(id)}/files/${relPath.split('/').map(enc).join('/')}`,
  sheetHtml: (id: string) => `/api/runs/${enc(id)}/sheet`,
  sheetJson: (id: string) => `/api/runs/${enc(id)}/sheet.json`,
  sheetPdf: (id: string) => `/api/runs/${enc(id)}/sheet.pdf`,
  bundleZip: (id: string) => `/api/runs/${enc(id)}/bundle.zip`,
  // run_id is optional on the wire; app.py stores the chunk under the run when it is given.
  chunk: (uploadId: string, n: number, runId?: string) => `/api/uploads/${enc(uploadId)}/chunks/${n}${runId ? `?run_id=${enc(runId)}` : ''}`,
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
  constructor(status: number, body: unknown, message?: string) {
    super(message ?? detailOf(body) ?? `HTTP ${status}`);
    this.status = status;
    this.body = body;
  }
}

function detailOf(body: unknown): string | null {
  if (body && typeof body === 'object') {
    const d = (body as { detail?: unknown; reason?: unknown }).detail ?? (body as { reason?: unknown }).reason;
    if (typeof d === 'string') return d;
  }
  return null;
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

async function postJson<T>(url: string, data: unknown): Promise<T> {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'content-type': 'application/json', accept: 'application/json' },
    body: JSON.stringify(data),
  });
  const body = await readBody(res);
  if (!res.ok) throw new ApiError(res.status, body);
  return body as T;
}

async function postForm<T>(url: string, form: FormData): Promise<T> {
  const res = await fetch(url, { method: 'POST', body: form, headers: { accept: 'application/json' } });
  const body = await readBody(res);
  if (!res.ok) throw new ApiError(res.status, body);
  return body as T;
}

export const api = {
  status: (runId?: string | null) => getJson<StatusResponse>(routes.status(runId)),
  sample: () => getJson<SampleResponse>(routes.sample()),
  runs: () => getJson<RunManifest[]>(routes.runs()),
  run: (id: string) => getJson<RunManifest>(routes.run(id)),
  replay: (id: string, mode: ReplayMode) => getJson<ReplayPayload>(routes.replay(id, mode)),
  derived: <T>(id: string, name: string) => getJson<T>(routes.file(id, `derived/${name}`)),
  video: (id: string, role: string) => getJson<VideoRecord>(routes.file(id, `videos/${role}.json`)),
  createRun: (body: { mode: 'car' | 'apartment'; kind: 'pair' | 'pickup_check'; keep_plate: boolean }) =>
    postJson<RunManifest>(routes.createRun(), body),
  finalize: (
    uploadId: string,
    body: { run_id: string; role: string; sha256_browser: string; bytes: number; filename: string; keep_plate: boolean },
  ) => postJson<VideoRecord>(routes.finalize(uploadId), body),
  putChunk: async (uploadId: string, n: number, blob: Blob, runId?: string): Promise<void> => {
    const res = await fetch(routes.chunk(uploadId, n, runId), {
      method: 'PUT',
      headers: { 'content-type': 'application/octet-stream' },
      body: blob,
    });
    if (!res.ok) throw new ApiError(res.status, await readBody(res));
  },
  topup: (id: string, video: File) => {
    const form = new FormData();
    form.append('video', video, video.name);
    return postForm<Coverage>(routes.topup(id), form);
  },
  claim: (body: { run_id: string; image: File; text: string; tire_code?: string }) => {
    const form = new FormData();
    form.append('image', body.image, body.image.name);
    form.append('run_id', body.run_id);
    form.append('text', body.text);
    if (body.tire_code) form.append('tire_code', body.tire_code);
    return postForm<Claim>(routes.claims(), form);
  },
  // HEAD on the PDF: 404 means "PDF retry" (the HTML sheet is still served).
  pdfReady: async (id: string): Promise<boolean> => {
    try {
      const res = await fetch(routes.sheetPdf(id), { method: 'HEAD' });
      return res.ok;
    } catch {
      return false;
    }
  },
};
