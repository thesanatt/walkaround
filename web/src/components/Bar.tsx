import type { ReactNode } from 'react';

import { Link } from '../lib/router.tsx';

// The top row of every screen: the name, a replay's note beside it, and on a run the sealed link at the right.
export function Bar({ note, children }: { note?: ReactNode; children?: ReactNode }) {
  return (
    <header className="bar">
      <span className="bar-start">
        <Link to="/" className="wordmark">
          Walkaround
        </Link>
        {note ? <span className="bar-note">{note}</span> : null}
      </span>
      {children ? <div className="bar-end">{children}</div> : null}
    </header>
  );
}
