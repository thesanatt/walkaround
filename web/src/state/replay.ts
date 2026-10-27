import { api } from '../api/client.ts';
import type { Event, ReplayMode, ReplayPayload } from '../contracts/types.ts';
import kioskDelays from '../../kiosk/delays.json';
import { LOADING_KINDS, loadForEvent, type Dispatch } from './loader.ts';
import { scheduleReplay, type ScheduledEvent } from './schedule.ts';
import type { Driver, DriverStatus } from './live.ts';

export interface ReplayOptions {
  // Apply every event scheduled at or before this virtual time at once, then stop; for the kiosk's ?t=<ms>.
  untilMs?: number;
  onStatus?: (status: DriverStatus, detail?: string) => void;
  onProgress?: (doneCount: number, total: number, atMs: number) => void;
  onEnd?: () => void;
  // Test seam: the payload instead of the network.
  payload?: ReplayPayload;
  fetchFiles?: boolean;
}

export interface Pacer {
  setTimeout(fn: () => void, ms: number): unknown;
  clearTimeout(handle: unknown): void;
  now(): number;
}

const realPacer: Pacer = {
  setTimeout: (fn, ms) => setTimeout(fn, ms),
  clearTimeout: (h) => clearTimeout(h as ReturnType<typeof setTimeout>),
  now: () => performance.now(),
};

// Runs a schedule against a pacer. One timer at a time, re-anchored to the start so drift does not accumulate.
export function runSchedule(
  schedule: ScheduledEvent[],
  emit: (event: Event, atMs: number, index: number) => void,
  options: { pacer?: Pacer; onEnd?: () => void; untilMs?: number } = {},
): Driver {
  const pacer = options.pacer ?? realPacer;
  let index = 0;
  let handle: unknown = null;
  let stopped = false;

  if (options.untilMs !== undefined) {
    const until = options.untilMs;
    while (index < schedule.length && (schedule[index]?.atMs ?? Infinity) <= until) {
      const item = schedule[index]!;
      emit(item.event, item.atMs, index);
      index += 1;
    }
    options.onEnd?.();
    return { stop() { stopped = true; } };
  }

  const start = pacer.now();
  const tick = () => {
    if (stopped) return;
    const elapsed = pacer.now() - start;
    while (index < schedule.length && (schedule[index]?.atMs ?? Infinity) <= elapsed) {
      const item = schedule[index]!;
      emit(item.event, item.atMs, index);
      index += 1;
    }
    if (index >= schedule.length) {
      options.onEnd?.();
      return;
    }
    const next = schedule[index]!.atMs - (pacer.now() - start);
    handle = pacer.setTimeout(tick, Math.max(0, next));
  };
  tick();

  return {
    stop() {
      stopped = true;
      if (handle !== null) pacer.clearTimeout(handle);
    },
  };
}

// The replay driver: GET /api/runs/{id} and GET /api/runs/{id}/replay?mode=..., then the schedule.
export function startReplay(runId: string, mode: ReplayMode, dispatch: Dispatch, options: ReplayOptions = {}): Driver {
  let stopped = false;
  let inner: Driver | null = null;
  const status = (s: DriverStatus, detail?: string) => options.onStatus?.(s, detail);
  const fetchFiles = options.fetchFiles ?? true;

  status('connecting');
  void (async () => {
    try {
      const [run, payload] = await Promise.all([
        api.run(runId).catch(() => null),
        options.payload ? Promise.resolve(options.payload) : api.replay(runId, mode),
      ]);
      if (stopped) return;
      if (run) dispatch({ kind: 'run.loaded', payload: run as unknown as Record<string, unknown> });
      dispatch({ kind: 'replay.started', payload: { mode: payload.mode ?? mode, recorded_on: payload.recorded_on } });
      const delaysMs = Object.keys(payload.delays_ms ?? {}).length ? payload.delays_ms : (kioskDelays as Record<string, number>);
      const schedule = scheduleReplay(payload.events, payload.mode ?? mode, { delaysMs });
      status('open');
      inner = runSchedule(
        schedule,
        (event, atMs, index) => {
          dispatch(event);
          if (fetchFiles && LOADING_KINDS.has(event.kind)) void loadForEvent(runId, event, dispatch);
          options.onProgress?.(index + 1, schedule.length, atMs);
        },
        {
          untilMs: options.untilMs,
          onEnd: () => {
            status('closed');
            options.onEnd?.();
          },
        },
      );
    } catch (err) {
      status('error', err instanceof Error ? err.message : String(err));
    }
  })();

  return {
    stop() {
      stopped = true;
      inner?.stop();
    },
  };
}
