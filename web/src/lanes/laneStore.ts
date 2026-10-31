import type { LaneWorkerMessage, ParseRequest, SnapshotObject } from '../workers/protocol.ts';

export type LaneParseStatus = 'parsing' | 'done' | 'error';

export interface LaneEntry {
  status: LaneParseStatus;
  frames: Map<number, SnapshotObject[]>;
  outcome: string | null;
  error: string | null;
  parseMs: number | null;
  url: string;
}

export interface WorkerLike {
  postMessage(message: ParseRequest): void;
  onmessage: ((event: MessageEvent<LaneWorkerMessage>) => void) | null;
  onerror: ((event: ErrorEvent) => void) | null;
  terminate(): void;
}

type Listener = () => void;

const NOTIFY_MS = 150;

function defaultWorker(): WorkerLike {
  return new Worker(new URL('../workers/lane.worker.ts', import.meta.url), { type: 'module' }) as unknown as WorkerLike;
}

// Parsed lanes live outside React state: the canvas reads them per frame, lists read a version counter that moves at
// most every 150 ms however many frames the worker posts.
export class LaneStore {
  private readonly lanes = new Map<string, LaneEntry>();
  private readonly listeners = new Set<Listener>();
  private worker: WorkerLike | null = null;
  private timer: ReturnType<typeof setTimeout> | null = null;
  // First time each (role, lane, id) was drawn, for the bloom. The player's clock writes it.
  readonly firstSeen = new Map<string, number>();
  version = 0;
  private readonly createWorker: () => WorkerLike;

  constructor(createWorker: () => WorkerLike = defaultWorker) {
    this.createWorker = createWorker;
  }

  private static key(role: string, lane: string): string {
    return `${role}|${lane}`;
  }

  private ensureWorker(): WorkerLike {
    if (this.worker) return this.worker;
    const worker = this.createWorker();
    worker.onmessage = (event) => this.onMessage(event.data);
    worker.onerror = (event) => {
      for (const lane of this.lanes.values()) {
        if (lane.status === 'parsing') {
          lane.status = 'error';
          lane.error = event.message || 'the lane worker stopped';
        }
      }
      this.notify(true);
    };
    this.worker = worker;
    return worker;
  }

  private onMessage(message: LaneWorkerMessage): void {
    const lane = this.lanes.get(LaneStore.key(message.role, message.noun));
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
      this.notify(false);
      return;
    }
    if (message.type === 'done') {
      lane.status = 'done';
      lane.outcome = message.outcome;
      lane.parseMs = message.parseMs;
    } else {
      lane.status = 'error';
      lane.error = message.message;
    }
    this.notify(true);
  }

  private notify(now: boolean): void {
    const fire = () => {
      this.timer = null;
      this.version += 1;
      for (const l of this.listeners) l();
    };
    if (now) {
      if (this.timer) clearTimeout(this.timer);
      fire();
    } else if (!this.timer) {
      this.timer = setTimeout(fire, NOTIFY_MS);
    }
  }

  subscribe(listener: Listener): () => void {
    this.listeners.add(listener);
    return () => void this.listeners.delete(listener);
  }

  has(role: string, lane: string): boolean {
    return this.lanes.has(LaneStore.key(role, lane));
  }

  get(role: string, lane: string): LaneEntry | undefined {
    return this.lanes.get(LaneStore.key(role, lane));
  }

  objectsAt(role: string, lane: string, frame: number): readonly SnapshotObject[] {
    return this.lanes.get(LaneStore.key(role, lane))?.frames.get(frame) ?? [];
  }

  // Parses a lane once: the worker fetches the gz lane and posts objects frame by frame.
  request(role: string, lane: string, url: string): void {
    const key = LaneStore.key(role, lane);
    if (this.lanes.has(key)) return;
    this.lanes.set(key, { status: 'parsing', frames: new Map(), outcome: null, error: null, parseMs: null, url });
    this.ensureWorker().postMessage({ type: 'parse', url, role, noun: lane });
    this.notify(true);
  }

  dispose(): void {
    if (this.timer) clearTimeout(this.timer);
    this.worker?.terminate();
    this.worker = null;
    this.lanes.clear();
    this.listeners.clear();
  }
}
