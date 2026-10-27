// Pure helpers shared by the lane worker and the tests: ndjson lines -> Responses events -> per-frame objects.
import {
  decodeMaskToSVGPath,
  frameIndexOf,
  parseVideoStream,
  type ResponseStreamOutcome,
  type ResponsesEvent,
  type SegmentationMask,
  type SegmentationRecord,
} from '@meta-sam/parser';

export interface LaneObject {
  id: string;
  box: [number, number, number, number]; // half-open source pixels
  c: number | null;
  mask: SegmentationMask;
}

export interface FrameBatch {
  frame: number;
  objects: LaneObject[];
}

export function eventsFromNdjson(text: string): ResponsesEvent[] {
  const events: ResponsesEvent[] = [];
  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    try {
      const parsed = JSON.parse(trimmed) as { type?: unknown };
      if (parsed && typeof parsed.type === 'string') events.push(parsed as ResponsesEvent);
    } catch {
      // a torn line in a stream that ended early; the parser reports the lane as incomplete
    }
  }
  return events;
}

export async function* iterate<T>(items: readonly T[]): AsyncIterable<T> {
  for (const item of items) yield item;
}

// Pairs box and mask records into per-frame objects. Records come box first, then mask, per object.
export class FrameCollector {
  private readonly pendingBoxes = new Map<string, { box: [number, number, number, number]; c: number | null }>();
  private consumed = 0;

  // Returns the frames completed by the records added since the last call.
  take(records: readonly SegmentationRecord[]): FrameBatch[] {
    const batches = new Map<number, LaneObject[]>();
    for (let i = this.consumed; i < records.length; i += 1) {
      const record = records[i]!;
      if (record.kind === 'text') continue;
      const frame = frameIndexOf(record) ?? 0;
      const key = `${frame}:${record.objectId}`;
      if (record.kind === 'box') {
        this.pendingBoxes.set(key, {
          box: [record.left, record.top, record.right, record.bottom],
          c: record.confidence ?? null,
        });
        continue;
      }
      const pending = this.pendingBoxes.get(key);
      this.pendingBoxes.delete(key);
      const box: [number, number, number, number] = pending?.box ?? [
        record.bounds.left,
        record.bounds.top,
        record.bounds.right,
        record.bounds.bottom,
      ];
      const object: LaneObject = {
        id: record.objectId,
        box,
        c: record.confidence ?? pending?.c ?? null,
        mask: record.mask,
      };
      const list = batches.get(frame) ?? [];
      list.push(object);
      batches.set(frame, list);
    }
    this.consumed = records.length;
    return [...batches.entries()].map(([frame, objects]) => ({ frame, objects }));
  }
}

export interface ParsedLane {
  frames: Map<number, LaneObject[]>;
  outcome: ResponseStreamOutcome;
  diagnostics: number;
}

// Node-safe: the whole lane at once, no bitmaps. The worker streams the same records through FrameCollector.
export async function parseLaneEvents(events: readonly ResponsesEvent[]): Promise<ParsedLane> {
  const result = await parseVideoStream(iterate(events)).finalResult;
  const collector = new FrameCollector();
  const frames = new Map<number, LaneObject[]>();
  for (const batch of collector.take(result.records)) {
    const existing = frames.get(batch.frame) ?? [];
    frames.set(batch.frame, [...existing, ...batch.objects]);
  }
  return { frames, outcome: result.outcome, diagnostics: result.diagnostics.length };
}

// The exact polygonal outline of a mask in raster pixels (top-left of the raster at the box's top-left).
export function maskPath(mask: SegmentationMask): string {
  return decodeMaskToSVGPath(mask);
}

// Gzip magic check for a lane fetched without Content-Encoding.
export function isGzip(bytes: Uint8Array): boolean {
  return bytes.length >= 2 && bytes[0] === 0x1f && bytes[1] === 0x8b;
}
