import { lazy, Suspense } from 'react';

import { useStatus } from './api/useStatus.ts';
import { Bar } from './components/Bar.tsx';
import { Shell } from './components/Shell.tsx';
import { Link, matchPath, useLocation, type Location } from './lib/router.tsx';
import { ClaimPage } from './pages/ClaimPage.tsx';
import { SheetPage } from './pages/SheetPage.tsx';
import { StartPage } from './pages/StartPage.tsx';
import { VerifyPage } from './pages/VerifyPage.tsx';
import type { StatusResponse } from './contracts/types.ts';

// The run and kiosk pages carry the video decoder; the start page a renter opens in the lot loads without it.
const RunPage = lazy(() => import('./pages/RunPage.tsx').then((m) => ({ default: m.RunPage })));
const KioskPage = lazy(() => import('./pages/KioskPage.tsx').then((m) => ({ default: m.KioskPage })));
const loading = <Bar />;

// The routes app.py serves as this app: /, /upload, /run/{id}, /run/{id}/sheet, /claim, /kiosk/{id}, /verify.
function page(loc: Location, status: StatusResponse | null) {
  let m: Record<string, string> | null;
  if (loc.path === '/' || loc.path === '/upload') return <StartPage status={status} />;
  if ((m = matchPath('/run/:id', loc.path)))
    return (
      <Suspense fallback={loading}>
        <RunPage key={m['id']} runId={m['id'] ?? ''} query={loc.query} />
      </Suspense>
    );
  if ((m = matchPath('/run/:id/sheet', loc.path))) return <SheetPage runId={m['id'] ?? ''} query={loc.query} />;
  if (loc.path === '/claim') return <ClaimPage query={loc.query} status={status} />;
  if (loc.path === '/verify') return <VerifyPage />;
  return (
    <div>
      <Bar />
      <h1>Nothing at this address</h1>
      <p>
        <Link to="/">Start over.</Link>
      </p>
    </div>
  );
}

export function App() {
  const loc = useLocation();
  // The kiosk has no page chrome. docs/VIDEO_PLAN.md also spells it /?replay=<id>&kiosk=1.
  const alias = loc.path === '/' && loc.query.get('kiosk') === '1' ? loc.query.get('replay') : null;
  const kiosk = matchPath('/kiosk/:id', loc.path);
  const id = alias ?? kiosk?.['id'];
  if (id)
    return (
      <Suspense fallback={null}>
        <KioskPage key={id} runId={id} query={loc.query} />
      </Suspense>
    );
  return <Site loc={loc} />;
}

function Site({ loc }: { loc: Location }) {
  const { status } = useStatus();
  return <Shell status={status}>{page(loc, status)}</Shell>;
}
