import { useEffect, useState } from 'react';

import { api } from '../api/client.ts';

export const exampleUrl = (id: string) => `/run/${encodeURIComponent(id)}?replay=instant`;

// The first sample the server lists (samples/index.json order), for "See an example".
export function useExample(): string | null {
  const [id, setId] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    void api
      .samples()
      .then((r) => r.samples[0]?.run_id ?? null)
      .catch(() => api.sample().then((s) => s.run_id))
      .catch(() => null)
      .then((runId) => {
        if (live) setId(runId);
      });
    return () => {
      live = false;
    };
  }, []);
  return id;
}
