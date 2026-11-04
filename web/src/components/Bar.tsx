import type { ReactNode } from 'react';

import { formatDate } from '../lib/format.ts';
import { Link, useLocation } from '../lib/router.tsx';

// The product's name as it is drawn: ALIBI. in Titillium Web Black. Read aloud as the word.
export function Wordmark() {
  return (
    <>
      <span aria-hidden="true">ALIBI.</span>
      <span className="sr-only">Alibi</span>
    </>
  );
}

// A replay says so beside the name; a phone keeps only the word.
export function ReplayNote({ date }: { date: string }) {
  return (
    <>
      <span title={`Replay of a run from ${formatDate(date)}`}>Replay</span>
    </>
  );
}

// The top row of every screen: the wordmark, a replay's note beside it, and on a run the sealed link at the right.
// The site's three places, top right on every page; the page you are on is underlined.
const NAV = [
  { to: '/how', label: 'How it works' },
  { to: '/verify', label: 'Verify' },
] as const;
const GITHUB = 'https://github.com/thesanatt/walkaround';

export function Bar({ note, children }: { note?: ReactNode; children?: ReactNode }) {
  const { path } = useLocation();
  return (
    <header className="bar">
      <span className="bar-start">
        <Link to="/" className="wordmark">
          <Wordmark />
        </Link>
        {note ? <span className="bar-note">{note}</span> : null}
      </span>
      {children ? <div className="bar-end">{children}</div> : null}
      <nav className="bar-nav" aria-label="Alibi">
        {NAV.map((n) => (
          <Link key={n.to} to={n.to} current={path === n.to}>
            {n.label}
          </Link>
        ))}
        <a href={GITHUB}>GitHub</a>
      </nav>
    </header>
  );
}
