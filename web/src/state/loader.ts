import { api, ApiError, type FileView } from '../api/client.ts';
import type { DeletionRecord, Event, RunManifest } from '../contracts/types.ts';
import type { AnyEvent, SyntheticEvent } from './types.ts';

export type Dispatch = (event: AnyEvent) => void;

// The reads a driver makes for one run. Tests pass a fake; the browser uses runFiles().
export interface RunFiles {
  json<T>(rel: string): Promise<T | null>; // GET /api/runs/{id}/files/<rel>; null when absent
  run(): Promise<RunManifest | null>; // GET /api/runs/{id}
}

export function runFiles(runId: string, view: FileView = 'owner'): RunFiles {
  const soft = async <T>(p: Promise<T>): Promise<T | null> => {
    try {
      return await p;
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) return null;
      throw err;
    }
  };
  return {
    json: <T>(rel: string) => soft(api.file<T>(runId, rel, view)),
    run: () => soft(api.run(runId)),
  };
}

async function settle<T>(p: Promise<T | null>): Promise<T | null> {
  try {
    return await p;
  } catch {
    return null;
  }
}

// What both drivers dispatch before the first event: run.json, then videos/<role>.json for each walk the run expects.
// Events emitted before the seal wait server-side until manifest.json exists, so right after an upload these records
// are the only sign that the file arrived and matched.
export async function initialLoads(files: RunFiles): Promise<SyntheticEvent[] | null> {
  const run = await files.run();
  if (!run) return null;
  const roles = run.kind === 'pair' ? ['pickup', 'return'] : ['pickup'];
  const records = await Promise.all(roles.map((role) => settle(files.json<unknown>(`videos/${role}.json`))));
  const out: SyntheticEvent[] = [{ kind: 'run.loaded', payload: run as unknown as Record<string, unknown> }];
  roles.forEach((role, i) => {
    if (records[i]) out.push({ kind: 'video.loaded', payload: { role, record: records[i] } });
  });
  return out;
}

// The event kinds that reference files the reducer needs.
export const LOADING_KINDS: ReadonlySet<string> = new Set([
  'upload.done',
  'blur.done',
  'seal.manifest',
  'files.uploaded',
  'panels.derived',
  'coverage.updated',
  'topup.requested',
  'marks.derived',
  'narration.derived',
  'staysput.derived',
  'match.derived',
  'run.completed',
  'run.failed',
]);

// Derived JSON referenced by an event becomes "<kind>.loaded" events, so live and replay feed identical sequences.
// A missing file (a partial run writes fewer) yields no event rather than an empty one.
export async function loadFor(event: Event, files: RunFiles): Promise<SyntheticEvent[]> {
  const p = event.payload ?? {};
  const role = typeof p['role'] === 'string' ? p['role'] : null;
  switch (event.kind) {
    case 'upload.done':
    case 'blur.done': {
      if (!role) return [];
      const record = await settle(files.json(`videos/${role}.json`));
      return record ? [{ kind: 'video.loaded', payload: { role, record } }] : [];
    }
    case 'seal.manifest':
    case 'files.uploaded': {
      // run.json gains the three clocks here (sealed_at, then first_model_call_at).
      const run = await settle(files.run());
      return run ? [{ kind: 'run.loaded', payload: run as unknown as Record<string, unknown> }] : [];
    }
    case 'panels.derived': {
      const [tracks, panels, keyframes] = await Promise.all([
        settle(files.json<unknown[]>('derived/tracks.json')),
        settle(files.json<unknown[]>('derived/panels.json')),
        settle(files.json<unknown[]>('derived/keyframes.json')),
      ]);
      const payload: Record<string, unknown> = {};
      if (tracks) payload['tracks'] = tracks;
      if (panels) payload['panels'] = panels;
      if (keyframes) payload['keyframes'] = keyframes;
      return Object.keys(payload).length ? [{ kind: 'panels.derived.loaded', payload }] : [];
    }
    case 'coverage.updated':
    case 'topup.requested': {
      const coverage = await settle(files.json('derived/coverage.json'));
      return coverage ? [{ kind: 'coverage.updated.loaded', payload: { coverage } }] : [];
    }
    case 'marks.derived': {
      // keyframes.json gains conditions and marks_status at this step, so it is read again with the marks.
      const [marks, keyframes] = await Promise.all([settle(files.json<unknown[]>('derived/marks.json')), settle(files.json<unknown[]>('derived/keyframes.json'))]);
      if (!marks) return [];
      return [{ kind: 'marks.derived.loaded', payload: keyframes ? { marks, keyframes } : { marks } }];
    }
    case 'narration.derived': {
      const narration = await settle(files.json('derived/narration.json'));
      return narration ? [{ kind: 'narration.derived.loaded', payload: { narration } }] : [];
    }
    case 'staysput.derived': {
      const tracks = await settle(files.json<unknown[]>('derived/mark_tracks.json'));
      return tracks ? [{ kind: 'staysput.derived.loaded', payload: { mark_tracks: tracks } }] : [];
    }
    case 'match.derived': {
      const [matches, verdicts] = await Promise.all([settle(files.json<unknown[]>('derived/matches.json')), settle(files.json<unknown[]>('derived/verdicts.json'))]);
      return matches || verdicts ? [{ kind: 'match.derived.loaded', payload: { matches: matches ?? [], verdicts: verdicts ?? [] } }] : [];
    }
    case 'run.completed':
    case 'run.failed': {
      const [run, deletions] = await Promise.all([settle(files.run()), settle(files.json<{ rows?: DeletionRecord[] }>('deletions.json'))]);
      const out: SyntheticEvent[] = [];
      if (run) out.push({ kind: 'run.loaded', payload: run as unknown as Record<string, unknown> });
      if (deletions?.rows) out.push({ kind: 'deletions.loaded', payload: { rows: deletions.rows } });
      return out;
    }
    default:
      return [];
  }
}

// Feeds events to the reducer strictly in order: an event that references files waits for its "<kind>.loaded"
// events before the next event is dispatched, so the dispatched sequence never depends on network timing.
// Events that need no file go through synchronously, which keeps React's batching per timer tick.
export class Pipeline {
  private readonly queue: Event[] = [];
  private head = 0;
  private busy = false;
  private stopped = false;
  private idleWaiters: (() => void)[] = [];
  private readonly dispatch: Dispatch;
  private readonly files: RunFiles;
  private readonly fetchFiles: boolean;

  constructor(dispatch: Dispatch, files: RunFiles, fetchFiles = true) {
    this.dispatch = dispatch;
    this.files = files;
    this.fetchFiles = fetchFiles;
  }

  push(event: Event): void {
    if (this.stopped) return;
    this.queue.push(event);
    if (!this.busy) this.drain();
  }

  stop(): void {
    this.stopped = true;
    this.queue.length = 0;
    this.head = 0;
    this.notifyIdle();
  }

  // Resolves once every pushed event and its loads have been dispatched (tests and the kiosk's ?t= capture).
  idle(): Promise<void> {
    if (!this.busy && this.head >= this.queue.length) return Promise.resolve();
    return new Promise((resolve) => this.idleWaiters.push(resolve));
  }

  private drain(): void {
    while (!this.stopped && this.head < this.queue.length) {
      const event = this.queue[this.head]!;
      this.head += 1;
      this.dispatch(event);
      if (this.fetchFiles && LOADING_KINDS.has(event.kind)) {
        this.busy = true;
        void loadFor(event, this.files)
          .then((extra) => {
            if (!this.stopped) for (const e of extra) this.dispatch(e);
          })
          .catch(() => undefined)
          .finally(() => {
            this.busy = false;
            this.drain();
          });
        return;
      }
    }
    if (this.head >= this.queue.length) {
      this.queue.length = 0;
      this.head = 0;
      this.notifyIdle();
    }
  }

  private notifyIdle(): void {
    const waiters = this.idleWaiters;
    this.idleWaiters = [];
    for (const w of waiters) w();
  }
}
