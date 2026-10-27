import type { SnapshotObject, WorkerMessage } from '../workers/protocol.ts';

export type LaneParseStatus = 'idle' | 'parsing' | 'done' | 'error';

export interface LaneRecord {
  status: LaneParseStatus;
  frames: Map<number, SnapshotObject[]>;
  frameCount: number;
  outcome: string | null;
  error: string | null;
  parseMs: number | null;
  url: string;
}

type Listener = () => void;

// Parsed lanes live outside React state: the canvas reads them per frame; React lists read a version counter.
export class LaneStore {
  private readonly lanes = new Map<string, LaneRecord>();
  private readonly listeners = new Set<Listener>();
  private worker: Worker | null = null;
  // First-seen clock per (role, noun, id) for the bloom; wall or virtual, the player decides.
  readonly firstSeen = new Map<string, number>();
  version = 0;

  private key(role: string, noun: string): string {
    return `${role}\u0000${noun}`;
  }

  private ensureWorker(): Worker {
    if (this.worker) return this.worker;
    const worker = new Worker(new URL('../workers/lane.worker.ts', import.meta.url), { type: 'module' });
    worker.onmessage = (event: MessageEvent<WorkerMessage>) => this.onMessage(event.data);
    worker.onerror = (event) => {
      for (const lane of this.lanes.values()) {
        if (lane.status === 'parsing') {
          lane.status = 'error';
          lane.error = event.message || 'worker error';
        }
      }
      this.bump();
    };
    this.worker = worker;
    return worker;
  }

  private onMessage(message: WorkerMessage): void {
    const lane = this.lanes.get(this.key(message.role, message.noun));
    if (!lane) return;
    if (message.type === 'snapshot') {
      const existing = lane.frames.get(message.frame);
      if (existing) {
        const byId = new Map(existing.map((o) => [o.id, o]));
        for (const o of message.objects) byId.set(o.id, o);
        lane.frames.set(message.frame, [...byId.values()]);
      } else {
        lane.frames.set(message.frame, message.objects);
      }
      lane.frameCount = lane.frames.size;
      // Lists only need a repaint every so often while a lane streams in.
      if (lane.frameCount % 15 === 1) this.bump();
      return;
    }
    if (message.type === 'done') {
      lane.status = 'done';
      lane.outcome = message.outcome;
      lane.parseMs = message.parseMs;
      lane.frameCount = lane.frames.size;
    } else {
      lane.status = 'error';
      lane.error = message.message;
    }
    this.bump();
  }

  private bump(): void {
    this.version += 1;
    for (const l of this.listeners) l();
  }

  subscribe(listener: Listener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  has(role: string, noun: string): boolean {
    return this.lanes.has(this.key(role, noun));
  }

  get(role: string, noun: string): LaneRecord | undefined {
    return this.lanes.get(this.key(role, noun));
  }

  objectsAt(role: string, noun: string, frame: number): SnapshotObject[] {
    return this.lanes.get(this.key(role, noun))?.frames.get(frame) ?? [];
  }

  // Parses a lane once; the worker fetches the gz lane and posts snapshots per frame.
  request(role: string, noun: string, url: string, color?: string): void {
    const key = this.key(role, noun);
    if (this.lanes.has(key)) return;
    this.lanes.set(key, { status: 'parsing', frames: new Map(), frameCount: 0, outcome: null, error: null, parseMs: null, url });
    this.ensureWorker().postMessage({ type: 'parse', url, role, noun, color, bitmaps: false });
    this.bump();
  }

  entries(role?: string): [string, string, LaneRecord][] {
    const out: [string, string, LaneRecord][] = [];
    for (const [key, lane] of this.lanes) {
      const [r, n] = key.split('\u0000') as [string, string];
      if (role && r !== role) continue;
      out.push([r, n, lane]);
    }
    return out;
  }

  dispose(): void {
    this.worker?.terminate();
    this.worker = null;
    this.lanes.clear();
    this.listeners.clear();
  }
}
