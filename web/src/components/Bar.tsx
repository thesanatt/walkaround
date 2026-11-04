import type { ReactNode } from 'react';

import { formatDate } from '../lib/format.ts';
import { Link } from '../lib/router.tsx';

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
      Replay<span className="bar-note-more"> of a run from {formatDate(date)}</span>
    </>
  );
}

// The top row of every screen: the wordmark, a replay's note beside it, and on a run the sealed link at the right.
export function Bar({ note, children }: { note?: ReactNode; children?: ReactNode }) {
  return (
    <header className="bar">
      <span className="bar-start">
        <Link to="/" className="wordmark">
          <Wordmark />
        </Link>
        {note ? <span className="bar-note">{note}</span> : null}
      </span>
      {children ? <div className="bar-end">{children}</div> : null}
    </header>
  );
}
