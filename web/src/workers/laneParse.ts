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
  box: [number, number, number, number]; // half-open source pixels (the proxy's)
  c: number | null; // absent means absent, never zero
  mask: SegmentationMask | null; // null for a record that carried a box only
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
      // a torn last line of a stream that ended early; the parser then reports the lane as incomplete
    }
  }
  return events;
}

export async function* iterate<T>(items: readonly T[]): AsyncIterable<T> {
  for (const item of items) yield item;
}

// Groups the parser's cumulative records into frames. A frame is handed out once a record of another frame arrives
// or the stream finishes, so its box and mask records (which can land in different snapshots) are both in.
export class FrameCollector {
  private consumed = 0;
  private current: number | null = null;
  private objects = new Map<string, LaneObject>();

  take(records: readonly SegmentationRecord[]): FrameBatch[] {
    const out: FrameBatch[] = [];
    for (let i = this.consumed; i < records.length; i += 1) {
      const record = records[i]!;
      if (record.kind === 'text') continue;
      const frame = frameIndexOf(record) ?? 0;
      if (this.current !== null && frame !== this.current) out.push(this.flushFrame());
      this.current = frame;
      const existing = this.objects.get(record.objectId);
      if (record.kind === 'box') {
        this.objects.set(record.objectId, {
          id: record.objectId,
          box: [record.left, record.top, record.right, record.bottom],
          c: record.confidence ?? existing?.c ?? null,
          mask: existing?.mask ?? null,
        });
      } else {
        this.objects.set(record.objectId, {
          id: record.objectId,
          box: existing?.box ?? [record.bounds.left, record.bounds.top, record.bounds.right, record.bounds.bottom],
          c: existing?.c ?? record.confidence ?? null,
          mask: record.mask,
        });
      }
    }
    this.consumed = records.length;
    return out;
  }

  finish(): FrameBatch[] {
    return this.current === null ? [] : [this.flushFrame()];
  }

  private flushFrame(): FrameBatch {
    const batch = { frame: this.current ?? 0, objects: [...this.objects.values()] };
    this.objects = new Map();
    this.current = null;
    return batch;
  }
}

export interface ParsedLane {
  frames: Map<number, LaneObject[]>;
  outcome: ResponseStreamOutcome;
  diagnostics: number;
}

// The whole lane at once, for tests and Node; the worker streams the same records through FrameCollector.
export async function parseLaneEvents(events: readonly ResponsesEvent[]): Promise<ParsedLane> {
  const result = await parseVideoStream(iterate(events)).finalResult;
  const collector = new FrameCollector();
  const frames = new Map<number, LaneObject[]>();
  for (const batch of [...collector.take(result.records), ...collector.finish()]) {
    frames.set(batch.frame, [...(frames.get(batch.frame) ?? []), ...batch.objects]);
  }
  return { frames, outcome: result.outcome, diagnostics: result.diagnostics.length };
}

// The exact pixel-edge outline of a mask in raster pixels; (0, 0) is the box's top-left corner.
export function maskPath(mask: SegmentationMask | null): string {
  return mask ? decodeMaskToSVGPath(mask) : '';
}

// A lane fetched without Content-Encoding arrives still gzipped; with it, the browser already inflated it.
export function isGzip(bytes: Uint8Array): boolean {
  return bytes.length >= 2 && bytes[0] === 0x1f && bytes[1] === 0x8b;
}
