import { useEffect, useReducer, useState } from 'react';

import type { FileView } from '../api/client.ts';
import type { ReplayMode } from '../contracts/types.ts';
import { startLive, type DriverStatus } from './live.ts';
import { runFiles, type Dispatch } from './loader.ts';
import { initialState, reduce } from './reducer.ts';
import { startReplay } from './replay.ts';
import type { AnyEvent, RunState } from './types.ts';

export type RunSource = { kind: 'live' } | { kind: 'replay'; mode: ReplayMode; untilMs?: number; startAtStage?: string; delays?: Record<string, number> };

export interface RunHandle {
  state: RunState;
  status: DriverStatus;
  detail: string | null;
  ended: boolean;
  progress: { done: number; total: number; atMs: number; spanMs: number } | null;
}

export interface FrameBatcher {
  dispatch: Dispatch;
  flush(): void;
  cancel(): void;
}

// Applies queued events once per animation frame. A fixed replay of a real pair emits a few hundred sam.frame events a
// second and an SSE history burst thousands at once; dispatched one by one, React would render the page for each.
// All dispatches inside one flush land in one render. Order is kept, and flush() empties the queue at once (the
// drivers' end, so `ended` never shows a state that is a frame behind).
// TODO(D1): profile a fixed replay of the real sample pair (E2 lanes) for long tasks; memoize the heavy panes if a
// frame still takes over 16 ms.
export function frameBatcher(apply: (events: AnyEvent[]) => void, schedule: (fn: () => void) => () => void = nextFrame): FrameBatcher {
  let queue: AnyEvent[] = [];
  let cancelScheduled: (() => void) | null = null;
  const flush = () => {
    cancelScheduled?.();
    cancelScheduled = null;
    if (!queue.length) return;
    const batch = queue;
    queue = [];
    apply(batch);
  };
  return {
    dispatch(event) {
      queue.push(event);
      cancelScheduled ??= schedule(flush);
    },
    flush,
    cancel() {
      cancelScheduled?.();
      cancelScheduled = null;
      queue = [];
    },
  };
}

function nextFrame(fn: () => void): () => void {
  if (typeof requestAnimationFrame === 'function') {
    const id = requestAnimationFrame(fn);
    return () => cancelAnimationFrame(id);
  }
  const id = setTimeout(fn, 16);
  return () => clearTimeout(id);
}

// One reducer, two drivers. The driver restarts when the run, the source or the file view changes.
export function useRun(runId: string | null, source: RunSource, view: FileView = 'owner'): RunHandle {
  const [state, dispatch] = useReducer(reduce, undefined, initialState);
  const [status, setStatus] = useState<DriverStatus>('connecting');
  const [detail, setDetail] = useState<string | null>(null);
  const [ended, setEnded] = useState(false);
  const [progress, setProgress] = useState<RunHandle['progress']>(null);
  const key = source.kind === 'live' ? 'live' : `replay:${source.mode}:${source.untilMs ?? ''}:${source.startAtStage ?? ''}:${JSON.stringify(source.delays ?? {})}`;

  useEffect(() => {
    dispatch({ kind: 'reset', payload: {} });
    setEnded(false);
    setDetail(null);
    setProgress(null);
    if (!runId) return;
    const files = runFiles(runId, view);
    const batch = frameBatcher((events) => {
      for (const e of events) dispatch(e);
    });
    const onStatus = (s: DriverStatus, d?: string) => {
      batch.flush();
      setStatus(s);
      setDetail(d ?? null);
    };
    const onEnd = () => {
      batch.flush();
      setEnded(true);
    };
    let lastShown = -1;
    const driver =
      source.kind === 'live'
        ? startLive(runId, batch.dispatch, { files, onStatus, onEnd })
        : startReplay(runId, source.mode, batch.dispatch, {
            files,
            untilMs: source.untilMs,
            startAtStage: source.startAtStage,
            delays: source.delays,
            onStatus,
            onEnd,
            onProgress: (done, total, atMs, spanMs) => {
              // A replay emits thousands of sam.frame events; the progress readout only needs a few updates a second.
              if (done === total || atMs - lastShown >= 250) {
                lastShown = atMs;
                setProgress({ done, total, atMs, spanMs });
              }
            },
          });
    return () => {
      driver.stop();
      batch.cancel();
    };
    // source is read through key so a new object with the same meaning does not restart the driver
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId, key, view]);

  return { state, status, detail, ended, progress };
}
