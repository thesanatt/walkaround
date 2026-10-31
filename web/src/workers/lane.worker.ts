// Fetches one gzipped SAM lane, parses it with @meta-sam/parser and posts per-frame objects. Nothing here runs on the
// main thread. Requests are handled one lane at a time in arrival order.
// TODO(D1, E15): every object of every frame gets its SVG path here, and the strings are copied to the main thread.
// On the fixture ball lane that is about 800 characters and 0.13 ms per mask; a 250 x 200 door mask may be ~7 KB and
// ~7 ms, so a 600-frame door lane ~4 s and ~8 MB. Measure parse ms and path bytes per lane on the E2 lanes, and check
// the main thread for tasks over 50 ms while the snapshots land (the contract's test, not written yet). If either is
// large, build paths lazily for frames near the playhead or downsample the outline.
import { decodeMaskToRaster, parseVideoStream, type SegmentationMask } from '@meta-sam/parser';

import { eventsFromNdjson, FrameCollector, isGzip, iterate, maskPath, type FrameBatch } from './laneParse.ts';
import type { LaneWorkerMessage, ParseRequest, SnapshotObject } from './protocol.ts';

interface WorkerScope {
  postMessage(message: LaneWorkerMessage, transfer?: Transferable[]): void;
  onmessage: ((event: MessageEvent<ParseRequest>) => void) | null;
}

const scope = self as unknown as WorkerScope;
let queue: Promise<void> = Promise.resolve();

async function readLaneText(url: string): Promise<string> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`lane fetch failed: HTTP ${res.status}`);
  const bytes = new Uint8Array(await res.arrayBuffer());
  if (!isGzip(bytes)) return new TextDecoder().decode(bytes);
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  return await new Response(stream).text();
}

function rgbOf(color: string | undefined): [number, number, number] {
  const m = /^#?([0-9a-f]{6})$/i.exec(color ?? '');
  const v = m ? parseInt(m[1]!, 16) : 0x1f7a6d;
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

async function bitmapOf(mask: SegmentationMask, rgb: [number, number, number]): Promise<ImageBitmap> {
  const raster = decodeMaskToRaster(mask);
  const pixels = new Uint8ClampedArray(mask.width * mask.height * 4);
  for (let i = 0; i < raster.length; i += 1) {
    if (raster[i] !== 1) continue;
    pixels.set([rgb[0], rgb[1], rgb[2], 255], i * 4);
  }
  return createImageBitmap(new ImageData(pixels, mask.width, mask.height));
}

async function post(request: ParseRequest, revision: number, batch: FrameBatch, rgb: [number, number, number]): Promise<void> {
  const objects: SnapshotObject[] = [];
  const transfer: Transferable[] = [];
  for (const o of batch.objects) {
    const bitmap = request.bitmaps && o.mask ? await bitmapOf(o.mask, rgb) : null;
    if (bitmap) transfer.push(bitmap);
    objects.push({ id: o.id, box: o.box, c: o.c, path: maskPath(o.mask), rw: o.mask?.width ?? 0, rh: o.mask?.height ?? 0, mask: bitmap });
  }
  scope.postMessage({ type: 'snapshot', role: request.role, noun: request.noun, revision, frame: batch.frame, objects }, transfer);
}

async function run(request: ParseRequest): Promise<void> {
  const started = performance.now();
  const events = eventsFromNdjson(await readLaneText(request.url));
  const parsed = parseVideoStream(iterate(events));
  const collector = new FrameCollector();
  const rgb = rgbOf(request.color);
  const frames = new Set<number>();
  let revision = 0;
  for await (const snapshot of parsed) {
    revision = snapshot.revision;
    for (const batch of collector.take(snapshot.records)) {
      frames.add(batch.frame);
      await post(request, revision, batch, rgb);
    }
  }
  const result = await parsed.finalResult;
  for (const batch of [...collector.take(result.records), ...collector.finish()]) {
    frames.add(batch.frame);
    await post(request, result.revision, batch, rgb);
  }
  scope.postMessage({
    type: 'done',
    role: request.role,
    noun: request.noun,
    frames: frames.size,
    outcome: result.outcome.status,
    diagnostics: result.diagnostics.length,
    parseMs: Math.round(performance.now() - started),
  });
}

scope.onmessage = (event: MessageEvent<ParseRequest>) => {
  const request = event.data;
  if (!request || request.type !== 'parse') return;
  queue = queue.then(() =>
    run(request).catch((err: unknown) => {
      scope.postMessage({ type: 'error', role: request.role, noun: request.noun, message: err instanceof Error ? err.message : String(err) });
    }),
  );
};
