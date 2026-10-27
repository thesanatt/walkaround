import { useEffect, useState } from 'react';

import type { StatusResponse } from '../contracts/types.ts';
import { api } from './client.ts';

// GET /api/status once and then every minute: the live-run counter, the pause flag, the queue position.
export function useStatus(runId: string | null = null, intervalMs = 60_000): { status: StatusResponse | null; error: string | null } {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let stopped = false;
    const load = () => {
      api
        .status(runId)
        .then((s) => {
          if (!stopped) {
            setStatus(s);
            setError(null);
          }
        })
        .catch((err: unknown) => {
          if (!stopped) setError(err instanceof Error ? err.message : String(err));
        });
    };
    load();
    const handle = setInterval(load, intervalMs);
    return () => {
      stopped = true;
      clearInterval(handle);
    };
  }, [runId, intervalMs]);

  return { status, error };
}
