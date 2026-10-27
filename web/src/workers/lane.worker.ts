// Parses one gzipped SAM lane with @meta-sam/parser off the main thread and posts per-frame objects.
import { decodeMaskToRaster, parseVideoStream, type ResponsesEvent } from '@meta-sam/parser';

import { FrameCollector, eventsFromNdjson, isGzip, maskPath, type LaneObject } from './laneParse.ts';
import type { ParseRequest, SnapshotObject, WorkerMessage } from './protocol.ts';

interface WorkerScope {
  postMessage(message: WorkerMessage, transfer?: Transferable[]): void;
  onmessage: ((event: MessageEvent<ParseRequest>) => void) | null;
}

const scope = self as unknown as WorkerScope;

async function readLaneText(url: string): Promise<string> {
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`lane fetch failed: HTTP ${res.status}`);
  const buffer = new Uint8Array(await res.arrayBuffer());
  if (!isGzip(buffer)) return new TextDecoder().decode(buffer);
  const stream = new Blob([buffer]).stream().pipeThrough(new DecompressionStream('gzip'));
  return await new Response(stream).text();
}

async function* iterateEvents(events: readonly ResponsesEvent[]): AsyncIterable<ResponsesEvent> {
  for (const event of events) yield event;
}

function parseColor(color: string | undefined): [number, number, number] {
  const m = /^#?([0-9a-f]{6})$/i.exec(color ?? '');
  if (!m) return [31, 122, 109];
  const v = parseInt(m[1]!, 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

async function bitmapFor(object: LaneObject, rgb: [number, number, number]): Promise<ImageBitmap> {
  const { width, height } = object.mask;
  const raster = decodeMaskToRaster(object.mask);
  const pixels = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < raster.length; i += 1) {
    if (raster[i] !== 1) continue;
    const o = i * 4;
    pixels[o] = rgb[0];
    pixels[o + 1] = rgb[1];
    pixels[o + 2] = rgb[2];
    pixels[o + 3] = 255;
  }
  return createImageBitmap(new ImageData(pixels, width, height));
}

async function run(request: ParseRequest): Promise<void> {
  const started = performance.now();
  const text = await readLaneText(request.url);
  const events = eventsFromNdjson(text);
  const parsed = parseVideoStream(iterateEvents(events));
  const collector = new FrameCollector();
  const rgb = parseColor(request.color);
  const seenFrames = new Set<number>();

  for await (const snapshot of parsed) {
    for (const batch of collector.take(snapshot.records)) {
      const objects: SnapshotObject[] = [];
      const transfer: Transferable[] = [];
      for (const object of batch.objects) {
        const bitmap = request.bitmaps ? await bitmapFor(object, rgb) : null;
        if (bitmap) transfer.push(bitmap);
        objects.push({ id: object.id, box: object.box, c: object.c, mask: bitmap, path: maskPath(object.mask) });
      }
      seenFrames.add(batch.frame);
      scope.postMessage(
        { type: 'snapshot', role: request.role, noun: request.noun, revision: snapshot.revision, frame: batch.frame, objects },
        transfer,
      );
    }
  }
  const result = await parsed.finalResult;
  scope.postMessage({
    type: 'done',
    role: request.role,
    noun: request.noun,
    frames: seenFrames.size,
    outcome: result.outcome.status,
    diagnostics: result.diagnostics.length,
    parseMs: Math.round(performance.now() - started),
  });
}

scope.onmessage = (event: MessageEvent<ParseRequest>) => {
  const request = event.data;
  if (!request || request.type !== 'parse') return;
  run(request).catch((err: unknown) => {
    scope.postMessage({
      type: 'error',
      role: request.role,
      noun: request.noun,
      message: err instanceof Error ? err.message : String(err),
    });
  });
};
