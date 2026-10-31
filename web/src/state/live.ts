import { routes } from '../api/client.ts';
import { EVENT_KINDS, TERMINAL_KINDS, type Event, type RunManifest } from '../contracts/types.ts';
import { initialLoads, Pipeline, type Dispatch, type RunFiles } from './loader.ts';

export interface Driver {
  stop(): void;
}

export type DriverStatus = 'connecting' | 'open' | 'reconnecting' | 'ended' | 'error';

// The slice of EventSource the driver uses, so a test can hand it a fake.
export interface EventSourceLike {
  readonly readyState: number;
  onopen: ((ev: unknown) => void) | null;
  onerror: ((ev: unknown) => void) | null;
  addEventListener(type: string, listener: (ev: MessageEvent<string>) => void): void;
  close(): void;
}

export interface LiveDeps {
  files: RunFiles;
  openEvents?: (url: string) => EventSourceLike;
  onStatus?: (status: DriverStatus, detail?: string) => void;
  onEnd?: () => void;
}

const CLOSED = 2; // EventSource.CLOSED, spelled out so the driver also runs where EventSource is undefined
export const ENDED_STATUSES: ReadonlySet<string> = new Set(['done', 'failed', 'partial']); // orchestrator.TERMINAL_STATUSES

function defaultOpen(url: string): EventSourceLike {
  return new EventSource(url) as unknown as EventSourceLike;
}

// The live driver: GET /api/runs/{id}, then EventSource on /api/runs/{id}/events. Every message carries id: <seq>,
// so after a drop the browser reconnects with Last-Event-ID and app.py resumes from seq + 1 (a closed tab resumes
// the same way from seq 0). app.py ends the stream after run.completed or run.failed; the driver closes the source
// there, or the browser would reconnect to a finished run every few seconds.
export function startLive(runId: string, dispatch: Dispatch, deps: LiveDeps): Driver {
  let stopped = false;
  let source: EventSourceLike | null = null;
  let lastSeq = -1;
  const pipeline = new Pipeline(dispatch, deps.files);
  const status = (s: DriverStatus, detail?: string) => {
    if (!stopped) deps.onStatus?.(s, detail);
  };

  const handle = (message: MessageEvent<string>) => {
    if (stopped || !source) return;
    let event: Event;
    try {
      event = JSON.parse(message.data) as Event;
    } catch {
      return;
    }
    if (typeof event.seq === 'number') {
      if (event.seq <= lastSeq) return;
      lastSeq = event.seq;
    }
    pipeline.push(event);
    if (TERMINAL_KINDS.has(event.kind)) {
      source.close();
      void pipeline.idle().then(() => {
        status('ended');
        if (!stopped) deps.onEnd?.();
      });
    }
  };

  // A run that failed before the seal has no chained event at all (its run.failed waits in events.pending.ndjson), so
  // the stream closes empty. run.json then holds the status and the stage errors, and they are what the page shows.
  const endedWithoutEvents = async () => {
    let run: RunManifest | null = null;
    try {
      run = await deps.files.run();
    } catch {
      run = null;
    }
    if (stopped) return;
    if (run && ENDED_STATUSES.has(run.status)) {
      dispatch({ kind: 'run.ended.loaded', payload: run as unknown as Record<string, unknown> });
      status('ended');
      deps.onEnd?.();
      return;
    }
    status('error', run ? 'the event stream closed before any event arrived' : 'no run with this id on this server');
  };

  const open = () => {
    const s = (deps.openEvents ?? defaultOpen)(routes.events(runId));
    source = s;
    s.onopen = () => status('open');
    s.onerror = () => {
      if (stopped || s !== source) return;
      if (s.readyState !== CLOSED) {
        status('reconnecting'); // the browser reconnects by itself with Last-Event-ID
        return;
      }
      // CLOSED: app.py answers 204 to a reconnect when the run ended with nothing new to send, and 404 for no run.
      // EventSource reports both the same way, so a stream that already delivered events counts as ended.
      if (lastSeq >= 0) void pipeline.idle().then(() => status('ended'));
      else void endedWithoutEvents();
    };
    for (const kind of EVENT_KINDS) s.addEventListener(kind, handle);
    s.addEventListener('message', handle);
  };

  status('connecting');
  initialLoads(deps.files)
    .then((first) => {
      if (stopped) return;
      if (!first) {
        status('error', 'no run with this id on this server');
        return;
      }
      for (const e of first) dispatch(e);
      open();
    })
    .catch((err: unknown) => status('error', err instanceof Error ? err.message : String(err)));

  return {
    stop() {
      stopped = true;
      source?.close();
      pipeline.stop();
    },
  };
}
