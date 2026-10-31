import { api, ApiError } from '../api/client.ts';
import type { RunManifest, VideoRecord } from '../contracts/types.ts';
import type { HashMessage, HashRequest } from '../workers/protocol.ts';
import { formatBytes } from './format.ts';

// thresholds.json media.*, the fallbacks when a run's frozen thresholds do not carry them
export const CHUNK_BYTES = 8_388_608;
export const CHUNKS_IN_FLIGHT = 3;
export const UPLOAD_MAX_BYTES = 1_073_741_824;
export const UPLOAD_WARN_BYTES = 524_288_000;

export interface UploadLimits {
  chunkBytes: number;
  chunksInFlight: number;
  maxBytes: number;
}

export const DEFAULT_LIMITS: UploadLimits = { chunkBytes: CHUNK_BYTES, chunksInFlight: CHUNKS_IN_FLIGHT, maxBytes: UPLOAD_MAX_BYTES };

export interface UploadProgress {
  hashedChunks: number;
  sentChunks: number;
  totalChunks: number;
  sha256: string | null;
}

export interface UploadResult {
  record: VideoRecord;
  sha256Browser: string;
  uploadId: string;
}

// 409 from finalize: the server deleted what it received and made no model call.
export class HashMismatch extends Error {
  readonly sha256Browser: string;
  readonly sha256Server: string;
  constructor(sha256Browser: string, sha256Server: string) {
    super('hash mismatch');
    this.sha256Browser = sha256Browser;
    this.sha256Server = sha256Server;
  }
}

export interface HashWorkerLike {
  postMessage(message: HashRequest): void;
  addEventListener(type: 'message', listener: (event: MessageEvent<HashMessage>) => void): void;
  removeEventListener(type: 'message', listener: (event: MessageEvent<HashMessage>) => void): void;
}

export function createHashWorker(): Worker {
  return new Worker(new URL('../workers/hash.worker.ts', import.meta.url), { type: 'module' });
}

let shared: Worker | null = null;

// One hash worker for the life of the page. It is never terminated, so an upload keeps going (and finalizes) after the
// upload page unmounts on an in-app navigation.
export function sharedHashWorker(): Worker {
  shared ??= createHashWorker();
  return shared;
}

// app.py accepts [A-Za-z0-9_-]{1,64}. crypto.randomUUID exists only in secure contexts (https or localhost).
export function newUploadId(): string {
  if (typeof crypto.randomUUID === 'function') return crypto.randomUUID();
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
}

export function chunkCount(size: number, chunkBytes = CHUNK_BYTES): number {
  return Math.max(1, Math.ceil(size / chunkBytes));
}

function positive(v: unknown, fallback: number): number {
  return typeof v === 'number' && Number.isFinite(v) && v > 0 ? v : fallback;
}

// The limits the server will assemble with: the run's frozen thresholds (run.json carries them from POST /api/runs).
export function limitsOf(run: Pick<RunManifest, 'thresholds'> | null | undefined): UploadLimits {
  const media = (run?.thresholds?.['media'] ?? {}) as Record<string, unknown>;
  return {
    chunkBytes: positive(media['chunk_bytes'], CHUNK_BYTES),
    chunksInFlight: positive(media['chunks_in_flight'], CHUNKS_IN_FLIGHT),
    maxBytes: positive(media['upload_max_bytes'], UPLOAD_MAX_BYTES),
  };
}

function missingOf(err: unknown): number[] | null {
  if (err instanceof ApiError && err.status === 400 && err.body && typeof err.body === 'object') {
    const missing = (err.body as { missing?: unknown }).missing;
    if (Array.isArray(missing) && missing.every((n) => typeof n === 'number')) return missing as number[];
  }
  return null;
}

// SHA-256 of a whole file in the hash worker, slice by slice, for uploads that do not go through chunks (the top-up).
export function hashFile(file: Blob, worker: HashWorkerLike, chunkBytes = CHUNK_BYTES): Promise<string> {
  const id = newUploadId();
  const digest = digestOf(worker, id);
  worker.postMessage({ type: 'start', id });
  for (let i = 0; i < chunkCount(file.size, chunkBytes); i += 1) {
    worker.postMessage({ type: 'chunk', id, index: i, blob: file.slice(i * chunkBytes, Math.min(file.size, (i + 1) * chunkBytes)) });
  }
  worker.postMessage({ type: 'finish', id });
  return digest;
}

function digestOf(worker: HashWorkerLike, id: string, onChunk?: (index: number) => void): Promise<string> {
  return new Promise<string>((resolve, reject) => {
    const listener = (event: MessageEvent<HashMessage>) => {
      const m = event.data;
      if (m.id !== id) return;
      if (m.type === 'progress') {
        onChunk?.(m.index);
        return;
      }
      worker.removeEventListener('message', listener);
      if (m.type === 'digest') resolve(m.hex);
      else reject(new Error(`hashing failed: ${m.message}`));
    };
    worker.addEventListener('message', listener);
  });
}

// Hash and upload in one loop over the same slices: the worker hashes in index order while three PUTs stay in flight.
// Chunks are idempotent by index, so a finalize that names missing chunks is answered by sending only those.
// resume: true is the retry after a failure: the same uploadId, every slice hashed again, no PUT until finalize names
// what the server lacks (a repeated finalize of bytes already received returns the stored record).
export async function uploadFile(
  file: File,
  options: {
    runId: string;
    role: string;
    keepPlate: boolean;
    worker: HashWorkerLike;
    uploadId?: string;
    resume?: boolean;
    limits?: UploadLimits;
    onProgress?: (p: UploadProgress) => void;
  },
): Promise<UploadResult> {
  const limits = options.limits ?? DEFAULT_LIMITS;
  if (file.size > limits.maxBytes) throw new Error(`This file is over ${formatBytes(limits.maxBytes)}, which the server refuses. Trim the walk and try again.`);
  const uploadId = options.uploadId ?? newUploadId();
  // A fresh hash id per attempt: chunk messages still queued from a failed attempt can never reach this hasher.
  const hashId = newUploadId();
  const totalChunks = chunkCount(file.size, limits.chunkBytes);
  const progress: UploadProgress = { hashedChunks: 0, sentChunks: 0, totalChunks, sha256: null };
  const report = () => options.onProgress?.({ ...progress });

  const digest = digestOf(options.worker, hashId, (index) => {
    progress.hashedChunks = index + 1;
    report();
  });
  digest.catch(() => undefined); // awaited below; this only keeps an early PUT failure from leaving it unhandled
  options.worker.postMessage({ type: 'start', id: hashId });

  const sliceOf = (index: number) => file.slice(index * limits.chunkBytes, Math.min(file.size, (index + 1) * limits.chunkBytes));

  const putWithRetry = async (index: number, blob: Blob): Promise<void> => {
    for (let attempt = 1; ; attempt += 1) {
      try {
        await api.putChunk(uploadId, index, blob, options.runId);
        return;
      } catch (err) {
        if (attempt >= 3 || (err instanceof ApiError && err.status === 413)) throw err;
        await new Promise((r) => setTimeout(r, 500 * attempt));
      }
    }
  };

  // Up to chunksInFlight PUTs at once. beforePut runs as each slice is taken, in index order (the hash needs that).
  const sendAll = async (indices: readonly number[], beforePut?: (index: number, blob: Blob) => void): Promise<void> => {
    let i = 0;
    let failed = false;
    const lane = async () => {
      while (!failed && i < indices.length) {
        const index = indices[i]!;
        i += 1;
        const blob = sliceOf(index);
        beforePut?.(index, blob);
        try {
          await putWithRetry(index, blob);
        } catch (err) {
          failed = true; // the other lanes stop at their next slice instead of sending the rest of the file
          throw err;
        }
        progress.sentChunks += 1;
        report();
      }
    };
    await Promise.all(Array.from({ length: Math.min(limits.chunksInFlight, indices.length) }, lane));
  };

  const all = Array.from({ length: totalChunks }, (_, i) => i);
  if (options.resume) {
    for (const index of all) options.worker.postMessage({ type: 'chunk', id: hashId, index, blob: sliceOf(index) });
  } else {
    try {
      await sendAll(all, (index, blob) => options.worker.postMessage({ type: 'chunk', id: hashId, index, blob }));
    } catch (err) {
      options.worker.postMessage({ type: 'finish', id: hashId }); // frees the worker's hasher
      throw err;
    }
  }
  options.worker.postMessage({ type: 'finish', id: hashId });
  const sha256 = await digest;
  progress.sha256 = sha256;
  report();

  const body = { run_id: options.runId, role: options.role, sha256_browser: sha256, bytes: file.size, filename: file.name, keep_plate: options.keepPlate };
  for (let resent = false; ; resent = true) {
    try {
      const record = await api.finalize(uploadId, body);
      progress.sentChunks = totalChunks;
      report();
      return { record, sha256Browser: sha256, uploadId };
    } catch (err) {
      if (err instanceof ApiError && err.status === 409 && err.body && typeof err.body === 'object') {
        const b = err.body as { sha256_browser?: string; sha256_server?: string };
        throw new HashMismatch(b.sha256_browser ?? sha256, b.sha256_server ?? '');
      }
      const missing = missingOf(err);
      if (!missing?.length || resent) throw err;
      progress.sentChunks = totalChunks - missing.length;
      report();
      await sendAll(missing);
    }
  }
}
