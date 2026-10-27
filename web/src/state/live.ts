import { api, routes } from '../api/client.ts';
import { EVENT_KINDS, type Event } from '../contracts/types.ts';
import { LOADING_KINDS, loadForEvent, type Dispatch } from './loader.ts';

export interface Driver {
  stop(): void;
}

export type DriverStatus = 'connecting' | 'open' | 'reconnecting' | 'closed' | 'error';

export interface LiveOptions {
  onStatus?: (status: DriverStatus, detail?: string) => void;
}

// The live driver: GET /api/runs/{id} once, then EventSource on /api/runs/{id}/events.
// The browser resends Last-Event-ID on reconnect (id: <seq> on every message), so a closed tab resumes.
export function startLive(runId: string, dispatch: Dispatch, options: LiveOptions = {}): Driver {
  let stopped = false;
  const status = (s: DriverStatus, detail?: string) => options.onStatus?.(s, detail);

  void api
    .run(runId)
    .then((run) => {
      if (!stopped) dispatch({ kind: 'run.loaded', payload: run as unknown as Record<string, unknown> });
    })
    .catch((err: unknown) => status('error', err instanceof Error ? err.message : String(err)));

  status('connecting');
  const source = new EventSource(routes.events(runId));

  const handle = (raw: MessageEvent<string>) => {
    if (stopped) return;
    let event: Event;
    try {
      event = JSON.parse(raw.data) as Event;
    } catch {
      return;
    }
    dispatch(event);
    if (LOADING_KINDS.has(event.kind)) void loadForEvent(runId, event, dispatch);
  };

  source.onopen = () => status('open');
  source.onerror = () => status(source.readyState === EventSource.CLOSED ? 'closed' : 'reconnecting');
  source.onmessage = handle;
  for (const kind of EVENT_KINDS) source.addEventListener(kind, handle as EventListener);

  return {
    stop() {
      stopped = true;
      source.close();
      status('closed');
    },
  };
}
