import type { ReactNode } from 'react';

import type { StatusResponse } from '../contracts/types.ts';
import { Link } from '../lib/router.tsx';
import { MockBanner } from './MockBanner.tsx';

export function Shell({ children, status }: { children: ReactNode; status: StatusResponse | null }) {
  return (
    <div className="shell">
      {status?.engine === 'mock' ? <MockBanner /> : null}
      <header className="topbar">
        <Link to="/" className="wordmark">
          Walkaround
        </Link>
        <nav className="nav" aria-label="pages">
          <Link to="/">upload</Link>
          <Link to="/claim">claim photo</Link>
          <Link to="/verify">verify a sheet</Link>
        </nav>
        <span className="topbar-note small muted">evidence support, not legal advice</span>
      </header>
      <main className="page">{children}</main>
    </div>
  );
}
