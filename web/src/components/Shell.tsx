import type { ReactNode } from 'react';

import type { StatusResponse } from '../contracts/types.ts';
import { MockBanner } from './MockBanner.tsx';

// Each page draws its own Bar, so a run can put its sealed link in it.
export function Shell({ children, status }: { children: ReactNode; status: StatusResponse | null }) {
  return (
    <div className="shell">
      {status?.engine === 'mock' ? <MockBanner /> : null}
      <main className="page">{children}</main>
      <footer className="site-foot">Walkaround is evidence support, not legal advice.</footer>
    </div>
  );
}
