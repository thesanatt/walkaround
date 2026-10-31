import kioskDelays from '../../kiosk/delays.json';
import { api } from '../api/client.ts';
import type { Event, ReplayMode, ReplayPayload } from '../contracts/types.ts';
import { ENDED_STATUSES, type Driver, type DriverStatus } from './live.ts';
import { initialLoads, Pipeline, type Dispatch, type RunFiles } from './loader.ts';
import { RECORDED_CAP_MS, scheduleReplay, totalMs, type ScheduledEvent } from './schedule.ts';
import type { SyntheticEvent } from './types.ts';

export interface Pacer {
  setTimeout(fn: () => void, ms: number): unknown;
  clearTimeout(handle: unknown): void;
  now(): number;
}

export const realPacer: Pacer = {
  setTimeout: (fn, ms) => setTimeout(fn, ms),
  clearTimeout: (h) => clearTimeout(h as ReturnType<typeof setTimeout>),
  now: () => performance.now(),
};

// The checked-in stage budgets (web/kiosk/delays.json), used when a fixed-mode payload arrives without delays_ms.
export const BUNDLED_DELAYS: Record<string, number> = Object.fromEntries(
  Object.entries(kioskDelays as Record<string, unknown>).filter(([k, v]) => !k.startsWith('_') && typeof v === 'number') as [string, number][],
);

// Runs a schedule against a pacer with one timer at a time, re-anchored to the start so drift never accumulates.
// startAtMs emits everything due by then at once and paces the rest from there; untilMs emits everything due by
// then at once and stops (the kiosk's ?t=<ms> frame capture).
export function runSchedule(
  schedule: readonly ScheduledEvent[],
  emit: (event: Event, atMs: number, index: number) => void,
  options: { pacer?: Pacer; onEnd?: () => void; untilMs?: number; startAtMs?: number } = {},
): Driver {
  const pacer = options.pacer ?? realPacer;
  let index = 0;
  let handle: unknown = null;
  let stopped = false;

  const emitDue = (elapsed: number) => {
    while (!stopped && index < schedule.length && schedule[index]!.atMs <= elapsed) {
      const item = schedule[index]!;
      emit(item.event, item.atMs, index);
      index += 1;
    }
  };

  if (options.untilMs !== undefined) {
    emitDue(options.untilMs);
    options.onEnd?.();
    return { stop: () => void (stopped = true) };
  }

  const offset = options.startAtMs ?? 0;
  const start = pacer.now() - offset;
  const tick = () => {
    handle = null;
    if (stopped) return;
    emitDue(pacer.now() - start);
    if (stopped) return;
    if (index >= schedule.length) {
      options.onEnd?.();
      return;
    }
    handle = pacer.setTimeout(tick, Math.max(0, schedule[index]!.atMs - (pacer.now() - start)));
  };
  tick();

  return {
    stop() {
      stopped = true;
      if (handle !== null) pacer.clearTimeout(handle);
    },
  };
}

// When a stage begins in a schedule: its first stage event's time, or 0 when the run never reached it.
export function stageStartMs(schedule: readonly ScheduledEvent[], stage: string): number {
  const hit = schedule.find((s) => s.event.kind === 'stage' && s.event.payload['name'] === stage);
  return hit ? Math.max(0, hit.atMs - 1) : 0;
}

// The recorded-mode cap the run froze (thresholds.orchestrator.replay_recorded_delay_cap_ms), else the default.
function recordedCapOf(first: readonly SyntheticEvent[] | null): number {
  const run = first?.find((e) => e.kind === 'run.loaded')?.payload as { thresholds?: Record<string, unknown> } | undefined;
  const cap = (run?.thresholds?.['orchestrator'] as Record<string, unknown> | undefined)?.['replay_recorded_delay_cap_ms'];
  return typeof cap === 'number' && cap >= 0 ? cap : RECORDED_CAP_MS;
}

export interface ReplayDeps {
  files: RunFiles;
  payload?: () => Promise<ReplayPayload>; // default GET /api/runs/{id}/replay?mode=
  pacer?: Pacer;
  startAtStage?: string; // apply everything before this stage's first event at once, pace the rest
  untilMs?: number; // measured from the start point; freeze there
  delays?: Record<string, number>; // a kiosk scene's own stage budgets, laid over the fixed-mode table
  fetchFiles?: boolean;
  onStatus?: (status: DriverStatus, detail?: string) => void;
  onProgress?: (done: number, total: number, atMs: number, spanMs: number) => void;
  onEnd?: () => void;
}

// The replay driver: GET /api/runs/{id}/replay?mode=, then GET /api/runs/{id}, then the paced events through the same
// pipeline the live driver uses. The badge date comes from the payload's recorded_on.
export function startReplay(runId: string, mode: ReplayMode, dispatch: Dispatch, deps: ReplayDeps): Driver {
  let stopped = false;
  let inner: Driver | null = null;
  const pipeline = new Pipeline(dispatch, deps.files, deps.fetchFiles ?? true);
  const status = (s: DriverStatus, detail?: string) => {
    if (!stopped) deps.onStatus?.(s, detail);
  };

  status('connecting');
  void (async () => {
    try {
      const payload = await (deps.payload ?? (() => api.replay(runId, mode)))();
      const first = await initialLoads(deps.files);
      if (stopped) return;
      const actual = payload.mode ?? mode;
      dispatch({ kind: 'replay.started', payload: { mode: actual, recorded_on: payload.recorded_on } });
      for (const e of first ?? []) dispatch(e);
      // A run that ended before its chain began (failed before the seal) has no event to replay; run.json says how.
      const run = first?.find((e) => e.kind === 'run.loaded')?.payload;
      if (!payload.events.length && run && ENDED_STATUSES.has(String(run['status']))) dispatch({ kind: 'run.ended.loaded', payload: run });
      const base = actual === 'fixed' && !Object.keys(payload.delays_ms ?? {}).length ? BUNDLED_DELAYS : (payload.delays_ms ?? {});
      const delays = actual === 'fixed' && deps.delays ? { ...base, ...deps.delays } : base;
      const schedule = scheduleReplay(payload.events, actual, delays, recordedCapOf(first));
      const span = totalMs(schedule);
      const startAtMs = deps.startAtStage ? stageStartMs(schedule, deps.startAtStage) : 0;
      status('open');
      inner = runSchedule(
        schedule,
        (event, atMs, index) => {
          pipeline.push(event);
          deps.onProgress?.(index + 1, schedule.length, atMs, span);
        },
        {
          pacer: deps.pacer,
          startAtMs,
          untilMs: deps.untilMs === undefined ? undefined : startAtMs + deps.untilMs,
          onEnd: () =>
            void pipeline.idle().then(() => {
              status('ended');
              if (!stopped) deps.onEnd?.();
            }),
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
      pipeline.stop();
    },
  };
}
