import type { ReactNode } from 'react';

import type { StatusResponse } from '../contracts/types.ts';
import { Link, useLocation } from '../lib/router.tsx';
import { MockBanner } from './MockBanner.tsx';

const NAV: [string, string][] = [
  ['/upload', 'Check a rental'],
  ['/claim', 'Claim photo'],
  ['/verify', 'Verify a sheet'],
];

export function Shell({ children, status }: { children: ReactNode; status: StatusResponse | null }) {
  const { path } = useLocation();
  return (
    <div className="shell">
      {status?.engine === 'mock' ? <MockBanner /> : null}
      <header className="topbar">
        <Link to="/" className="wordmark">
          Walkaround
        </Link>
        <nav className="nav" aria-label="pages">
          {NAV.map(([to, text]) => (
            <Link key={to} to={to} current={path === to}>
              {text}
            </Link>
          ))}
        </nav>
      </header>
      <main className="page">{children}</main>
      <footer className="site-foot">Walkaround is evidence support, not legal advice.</footer>
    </div>
  );
}
