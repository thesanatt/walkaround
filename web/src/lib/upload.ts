import { api, ApiError } from '../api/client.ts';
import type { VideoRecord } from '../contracts/types.ts';
import type { HashMessage, HashRequest } from '../workers/protocol.ts';

export const CHUNK_BYTES = 8_388_608; // thresholds.json media.chunk_bytes
export const CHUNKS_IN_FLIGHT = 3; // media.chunks_in_flight
export const UPLOAD_MAX_BYTES = 1_073_741_824; // media.upload_max_bytes
export const UPLOAD_WARN_BYTES = 524_288_000; // media.upload_warn_bytes

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

export class HashMismatch extends Error {
  constructor(
    readonly sha256Browser: string,
    readonly sha256Server: string,
  ) {
    super('hash mismatch');
  }
}

interface HashWorkerLike {
  postMessage(message: HashRequest): void;
  addEventListener(type: 'message', listener: (event: MessageEvent<HashMessage>) => void): void;
  removeEventListener(type: 'message', listener: (event: MessageEvent<HashMessage>) => void): void;
}

export function createHashWorker(): Worker {
  return new Worker(new URL('../workers/hash.worker.ts', import.meta.url), { type: 'module' });
}

function missingOf(err: unknown): number[] | null {
  if (err instanceof ApiError && err.status === 400 && err.body && typeof err.body === 'object') {
    const missing = (err.body as { missing?: unknown }).missing;
    if (Array.isArray(missing) && missing.every((n) => typeof n === 'number')) return missing as number[];
  }
  return null;
}

// Hash and upload in one loop over the same slices: the worker hashes in order while 3 PUTs stay in flight.
// Chunks are idempotent by index, so a finalize that names missing chunks is answered by sending only those.
export async function uploadFile(
  file: File,
  options: {
    runId: string;
    role: string;
    keepPlate: boolean;
    worker: HashWorkerLike;
    uploadId?: string;
    onProgress?: (p: UploadProgress) => void;
  },
): Promise<UploadResult> {
  if (file.size > UPLOAD_MAX_BYTES) throw new Error('this file is over 1 GiB; the server refuses it');
  const uploadId = options.uploadId ?? crypto.randomUUID();
  const totalChunks = Math.max(1, Math.ceil(file.size / CHUNK_BYTES));
  const progress: UploadProgress = { hashedChunks: 0, sentChunks: 0, totalChunks, sha256: null };
  const report = () => options.onProgress?.({ ...progress });

  const digest = new Promise<string>((resolve, reject) => {
    const listener = (event: MessageEvent<HashMessage>) => {
      const m = event.data;
      if (m.id !== uploadId) return;
      if (m.type === 'progress') {
        progress.hashedChunks = m.index + 1;
        report();
      } else if (m.type === 'digest') {
        options.worker.removeEventListener('message', listener);
        resolve(m.hex);
      } else {
        options.worker.removeEventListener('message', listener);
        reject(new Error(m.message));
      }
    };
    options.worker.addEventListener('message', listener);
  });
  options.worker.postMessage({ type: 'start', id: uploadId });

  const sliceOf = (index: number) => file.slice(index * CHUNK_BYTES, Math.min(file.size, (index + 1) * CHUNK_BYTES));

  const putWithRetry = async (index: number, blob: Blob): Promise<void> => {
    let attempt = 0;
    for (;;) {
      try {
        await api.putChunk(uploadId, index, blob, options.runId);
        return;
      } catch (err) {
        attempt += 1;
        if (attempt >= 3) throw err;
        await new Promise((r) => setTimeout(r, 500 * attempt));
      }
    }
  };

  let next = 0;
  const putOne = async (index: number): Promise<void> => {
    const blob = sliceOf(index);
    options.worker.postMessage({ type: 'chunk', id: uploadId, blob, index });
    await putWithRetry(index, blob);
    progress.sentChunks += 1;
    report();
  };
  const workers = Array.from({ length: Math.min(CHUNKS_IN_FLIGHT, totalChunks) }, async () => {
    while (next < totalChunks) {
      const index = next;
      next += 1;
      await putOne(index);
    }
  });
  await Promise.all(workers);
  options.worker.postMessage({ type: 'finish', id: uploadId });
  const sha256 = await digest;
  progress.sha256 = sha256;
  report();

  const body = {
    run_id: options.runId,
    role: options.role,
    sha256_browser: sha256,
    bytes: file.size,
    filename: file.name,
    keep_plate: options.keepPlate,
  };
  let resumed = false;
  for (;;) {
    try {
      const record = await api.finalize(uploadId, body);
      return { record, sha256Browser: sha256, uploadId };
    } catch (err) {
      if (err instanceof ApiError && err.status === 409 && err.body && typeof err.body === 'object') {
        const b = err.body as { sha256_browser?: string; sha256_server?: string };
        throw new HashMismatch(b.sha256_browser ?? sha256, b.sha256_server ?? '');
      }
      const missing = missingOf(err);
      if (missing && missing.length && !resumed) {
        resumed = true;
        for (const index of missing) await putWithRetry(index, sliceOf(index));
        continue;
      }
      throw err;
    }
  }
}
