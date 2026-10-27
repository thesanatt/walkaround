import { useEffect, useReducer, useRef, useState } from 'react';

import type { ReplayMode } from '../contracts/types.ts';
import { startLive, type Driver, type DriverStatus } from './live.ts';
import { initialState, reduce } from './reducer.ts';
import { startReplay } from './replay.ts';
import type { AnyEvent, RunState } from './types.ts';

export type RunSource = { kind: 'live' } | { kind: 'replay'; mode: ReplayMode; untilMs?: number };

export interface RunHandle {
  state: RunState;
  status: DriverStatus;
  detail: string | null;
  ended: boolean;
}

// One reducer, two drivers. The hook restarts the driver when the run id or the source changes.
export function useRun(runId: string | null, source: RunSource): RunHandle {
  const [state, dispatch] = useReducer(reduce, undefined, initialState);
  const [status, setStatus] = useState<DriverStatus>('connecting');
  const [detail, setDetail] = useState<string | null>(null);
  const [ended, setEnded] = useState(false);
  const driverRef = useRef<Driver | null>(null);
  const sourceKey = source.kind === 'live' ? 'live' : `replay:${source.mode}:${source.untilMs ?? ''}`;

  useEffect(() => {
    driverRef.current?.stop();
    dispatch({ kind: 'reset', payload: {} } as AnyEvent);
    setEnded(false);
    setDetail(null);
    if (!runId) return;
    const onStatus = (s: DriverStatus, d?: string) => {
      setStatus(s);
      setDetail(d ?? null);
    };
    const driver =
      source.kind === 'live'
        ? startLive(runId, dispatch, { onStatus })
        : startReplay(runId, source.mode, dispatch, { onStatus, untilMs: source.untilMs, onEnd: () => setEnded(true) });
    driverRef.current = driver;
    return () => {
      driver.stop();
      driverRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId, sourceKey]);

  return { state, status, detail, ended };
}
