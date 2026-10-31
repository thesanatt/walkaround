import { createHash, randomBytes } from 'node:crypto';

import { createSHA256 } from 'hash-wasm';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { ApiError } from '../src/api/client.ts';
import { CHUNK_BYTES, chunkCount, DEFAULT_LIMITS, HashMismatch, hashFile, limitsOf, UPLOAD_MAX_BYTES, uploadFile, type HashWorkerLike } from '../src/lib/upload.ts';
import type { HashMessage, HashRequest } from '../src/workers/protocol.ts';

// The hash worker's logic in-process: the same hash-wasm calls hash.worker.ts makes, in index order.
function inProcessHashWorker(): HashWorkerLike {
  const listeners = new Set<(event: MessageEvent<HashMessage>) => void>();
  const hashers = new Map<string, ReturnType<typeof createSHA256>>();
  let queue = Promise.resolve();
  const post = (m: HashMessage) => {
    for (const l of listeners) l({ data: m } as MessageEvent<HashMessage>);
  };
  return {
    addEventListener: (_t, l) => void listeners.add(l),
    removeEventListener: (_t, l) => void listeners.delete(l),
    postMessage(m: HashRequest) {
      if (m.type === 'start') {
        hashers.set(m.id, createSHA256().then((h) => (h.init(), h)));
        return;
      }
      queue = queue.then(async () => {
        const h = await hashers.get(m.id)!;
        if (m.type === 'chunk') {
          h.update(new Uint8Array(await m.blob.arrayBuffer()));
          post({ type: 'progress', id: m.id, index: m.index });
        } else post({ type: 'digest', id: m.id, hex: h.digest('hex') });
      });
    },
  };
}

// A fake app.py upload route: chunks by index, finalize hashes them in order like orchestrator.finalize_upload.
// refuse maps a chunk index to the status its PUT answers while it is listed.
function fakeServer(options: { dropFirst?: number; corrupt?: boolean; refuse?: Map<number, number>; chunkBytes?: number } = {}) {
  const chunks = new Map<number, Uint8Array>();
  const puts: number[] = [];
  let dropped = false;
  let inFlight = 0;
  let maxInFlight = 0;
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    if (init?.method === 'PUT') {
      const n = Number(/chunks\/(\d+)/.exec(url)![1]);
      expect(url).toContain('run_id=run1');
      inFlight += 1;
      maxInFlight = Math.max(maxInFlight, inFlight);
      await new Promise((r) => setTimeout(r, 2));
      inFlight -= 1;
      puts.push(n);
      const refused = options.refuse?.get(n);
      if (refused) return new Response(JSON.stringify({ detail: 'refused' }), { status: refused });
      if (!(options.dropFirst === n && !dropped)) chunks.set(n, new Uint8Array(await (init.body as Blob).arrayBuffer()));
      else dropped = true;
      return new Response(null, { status: 204 });
    }
    const body = JSON.parse(String(init?.body)) as { sha256_browser: string; bytes: number; run_id: string; role: string };
    const expected = chunkCount(body.bytes, options.chunkBytes);
    const missing = Array.from({ length: expected }, (_, i) => i).filter((i) => !chunks.has(i));
    if (missing.length) return new Response(JSON.stringify({ detail: 'chunks missing; resume upload', missing }), { status: 400 });
    const h = createHash('sha256');
    for (let i = 0; i < expected; i += 1) h.update(chunks.get(i)!);
    const server = options.corrupt ? createHash('sha256').update('other bytes').digest('hex') : h.digest('hex');
    if (server !== body.sha256_browser) return new Response(JSON.stringify({ sha256_browser: body.sha256_browser, sha256_server: server }), { status: 409 });
    return new Response(JSON.stringify({ role: body.role, original_filename: 'pickup.mp4', bytes: body.bytes, sha256_browser: body.sha256_browser, sha256_server: server, hash_match: true }), { status: 200 });
  });
  return { fetchMock, puts, maxInFlight: () => maxInFlight };
}

afterEach(() => vi.unstubAllGlobals());

describe('hash and upload in one loop', () => {
  const bytes = randomBytes(2 * CHUNK_BYTES + 1_234_567);
  const file = new File([bytes], 'pickup.mp4', { type: 'video/mp4' });
  const truth = createHash('sha256').update(bytes).digest('hex');

  it('hashes the 8 MB slices to the same SHA-256 the server computes on receipt', async () => {
    const server = fakeServer();
    vi.stubGlobal('fetch', server.fetchMock);
    const progress: number[] = [];
    const result = await uploadFile(file, { runId: 'run1', role: 'pickup', keepPlate: false, worker: inProcessHashWorker(), onProgress: (p) => progress.push(p.hashedChunks) });
    expect(result.sha256Browser).toBe(truth);
    expect(result.record.sha256_server).toBe(truth);
    expect(server.puts.sort()).toEqual([0, 1, 2]);
    expect(server.maxInFlight()).toBeLessThanOrEqual(3);
    expect(Math.max(...progress)).toBe(3);
  });

  it('resends only the chunks finalize names as missing', async () => {
    const server = fakeServer({ dropFirst: 1 });
    vi.stubGlobal('fetch', server.fetchMock);
    const result = await uploadFile(file, { runId: 'run1', role: 'pickup', keepPlate: false, worker: inProcessHashWorker() });
    expect(result.sha256Browser).toBe(truth);
    expect(server.puts.filter((n) => n === 1)).toHaveLength(2);
    expect(server.puts.filter((n) => n !== 1)).toHaveLength(2);
  });

  it('turns a 409 into a HashMismatch carrying both values', async () => {
    const server = fakeServer({ corrupt: true });
    vi.stubGlobal('fetch', server.fetchMock);
    const err = await uploadFile(file, { runId: 'run1', role: 'pickup', keepPlate: false, worker: inProcessHashWorker() }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(HashMismatch);
    expect((err as HashMismatch).sha256Browser).toBe(truth);
    expect((err as HashMismatch).sha256Server).toMatch(/^[0-9a-f]{64}$/);
  });
});

describe('a failed upload resumes into the same run', () => {
  const bytes = randomBytes(2 * CHUNK_BYTES + 1_234_567);
  const file = new File([bytes], 'pickup.mp4', { type: 'video/mp4' });
  const truth = createHash('sha256').update(bytes).digest('hex');

  it('sends only what the server lacks when the same upload id resumes', async () => {
    const refuse = new Map([[1, 503]]);
    const server = fakeServer({ refuse });
    vi.stubGlobal('fetch', server.fetchMock);
    const worker = inProcessHashWorker();
    const first = await uploadFile(file, { runId: 'run1', role: 'pickup', keepPlate: false, worker, uploadId: 'up-1' }).catch((e: unknown) => e);
    expect(first).toBeInstanceOf(ApiError); // chunk 1 failed three times
    expect(server.puts.filter((n) => n === 1)).toHaveLength(3);
    refuse.clear();
    const before = server.puts.length;
    const progress: number[] = [];
    const result = await uploadFile(file, { runId: 'run1', role: 'pickup', keepPlate: false, worker, uploadId: 'up-1', resume: true, onProgress: (p) => progress.push(p.sentChunks) });
    expect(result.sha256Browser).toBe(truth);
    expect(result.record.sha256_server).toBe(truth);
    expect(server.puts.slice(before)).toEqual([1]); // finalize named chunk 1; chunks 0 and 2 were not sent again
    expect(progress.at(-1)).toBe(3);
  }, 10_000);

  it('stops the other lanes at their next slice once a chunk fails for good', async () => {
    const limits = { chunkBytes: 4096, chunksInFlight: 3, maxBytes: UPLOAD_MAX_BYTES };
    const small = new File([randomBytes(40 * 4096)], 'walk.mp4');
    const server = fakeServer({ refuse: new Map([[2, 413]]), chunkBytes: 4096 });
    vi.stubGlobal('fetch', server.fetchMock);
    const err = await uploadFile(small, { runId: 'run1', role: 'pickup', keepPlate: false, worker: inProcessHashWorker(), limits }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    await new Promise((r) => setTimeout(r, 30));
    expect(server.puts.length).toBeLessThan(8); // not the other 37 slices
  });

  it("sizes the chunks from the run's frozen thresholds", async () => {
    expect(limitsOf({ thresholds: { media: { chunk_bytes: 4_194_304, chunks_in_flight: 2, upload_max_bytes: 10 } } })).toEqual({ chunkBytes: 4_194_304, chunksInFlight: 2, maxBytes: 10 });
    expect(limitsOf({ thresholds: {} })).toEqual(DEFAULT_LIMITS);
    expect(limitsOf(null)).toEqual(DEFAULT_LIMITS);
    const server = fakeServer({ chunkBytes: 4_194_304 });
    vi.stubGlobal('fetch', server.fetchMock);
    const result = await uploadFile(file, { runId: 'run1', role: 'pickup', keepPlate: false, worker: inProcessHashWorker(), limits: limitsOf({ thresholds: { media: { chunk_bytes: 4_194_304 } } }) });
    expect(result.sha256Browser).toBe(truth);
    expect(new Set(server.puts)).toEqual(new Set([0, 1, 2, 3, 4]));
  });

  it('hashes a top-up clip to the SHA-256 the server computes', async () => {
    expect(await hashFile(file, inProcessHashWorker())).toBe(truth);
  });
});
