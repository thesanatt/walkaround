import { howChecked } from '../lib/checks.ts';
import { useFileUrl } from '../lib/files.tsx';
import type { RunState } from '../state/types.ts';

// One row per model, counted from this run's files: what it did, then its limit. Code is the last row because it
// writes every label.
// rows: how many rows are on screen so far (the kiosk reveals them in turn); all of them when absent.
export function HowChecked({ state, big = false, rows }: { state: RunState; big?: boolean; rows?: number }) {
  const url = useFileUrl();
  const all = howChecked(state);
  const lines = rows === undefined ? all : all.slice(0, rows);
  return (
    <div className={`how-checked ${big ? 'big' : ''}`}>
      <dl>
        {lines.map((l) => (
          <div key={l.key} className={`check check-${l.key} ${l.state}`}>
            <dt>{l.model}</dt>
            <dd>
              <span className="check-did">{l.did}</span>
              <span className="check-detail">
                {l.detail}
                {l.link ? (
                  <>
                    {' '}
                    (<a href={url(l.link.rel)}>{l.link.text}</a>)
                  </>
                ) : null}
              </span>
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
