import { useStatus } from './api/useStatus.ts';
import { Shell } from './components/Shell.tsx';
import { Link, matchPath, useLocation, type Location } from './lib/router.tsx';
import { ClaimPage } from './pages/ClaimPage.tsx';
import { HomePage } from './pages/HomePage.tsx';
import { KioskPage } from './pages/KioskPage.tsx';
import { RunPage } from './pages/RunPage.tsx';
import { SheetPage } from './pages/SheetPage.tsx';
import { VerifyPage } from './pages/VerifyPage.tsx';

// The routes app.py serves as the SPA: /, /run/{id}, /run/{id}/sheet, /claim, /kiosk/{id}, /verify.
function page(loc: Location) {
  let m: Record<string, string> | null;
  if (loc.path === '/') return <HomePage />;
  if ((m = matchPath('/run/:id', loc.path))) return <RunPage runId={m['id'] ?? ''} query={loc.query} />;
  if ((m = matchPath('/run/:id/sheet', loc.path))) return <SheetPage runId={m['id'] ?? ''} query={loc.query} />;
  if (loc.path === '/claim') return <ClaimPage query={loc.query} />;
  if (loc.path === '/verify') return <VerifyPage />;
  return (
    <div>
      <h1>No page here</h1>
      <p>
        <Link to="/">Back to the upload page.</Link>
      </p>
    </div>
  );
}

export function App() {
  const loc = useLocation();
  const { status } = useStatus();

  // The kiosk has no chrome: /kiosk/{id}?scene=...; docs/VIDEO_PLAN.md also spells it /?replay={id}&kiosk=1.
  const alias = loc.path === '/' && loc.query.get('kiosk') === '1' ? loc.query.get('replay') : null;
  const kiosk = matchPath('/kiosk/:id', loc.path);
  if (alias) return <KioskPage runId={alias} query={loc.query} />;
  if (kiosk) return <KioskPage runId={kiosk['id'] ?? ''} query={loc.query} />;

  return <Shell status={status}>{page(loc)}</Shell>;
}
