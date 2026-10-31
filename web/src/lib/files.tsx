import { createContext, useContext, useMemo, type ReactNode } from 'react';

import { routes, type FileView } from '../api/client.ts';

// The run's stored files, as URLs. The kiosk provides view "kiosk" so every crop, chip and lane it loads is requested
// as a viewer who is not the run owner.
const FileUrlContext = createContext<(rel: string) => string>((rel) => rel);

export function RunFiles({ runId, view = 'owner', children }: { runId: string; view?: FileView; children: ReactNode }) {
  const url = useMemo(() => (rel: string) => routes.file(runId, rel, view), [runId, view]);
  return <FileUrlContext.Provider value={url}>{children}</FileUrlContext.Provider>;
}

export function useFileUrl(): (rel: string) => string {
  return useContext(FileUrlContext);
}
