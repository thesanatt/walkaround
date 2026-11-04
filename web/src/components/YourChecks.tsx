import type { Mark, Verdict } from '../contracts/types.ts';
import { ANSWER_WORD, cap } from '../lib/findings.ts';
import { formatStamp } from '../lib/format.ts';
import { panelName } from '../lib/panels.ts';
import type { StoredCheck } from '../lib/yourcheck.ts';

// The renter's own checks, listed apart from the sheet: on the sealed panel and the sheet page.
export function YourChecks({ checks, verdicts, marks }: { checks: Record<string, StoredCheck>; verdicts: Verdict[]; marks: Map<string, Mark> }) {
  const mine = Object.entries(checks)
    .map(([id, c]) => ({ v: verdicts.find((v) => v.verdict_id === id), c }))
    .filter((x): x is { v: Verdict; c: StoredCheck } => x.v !== undefined);
  if (!mine.length) return null;
  const words = (v: Verdict) => v.mark_ids.map((id) => marks.get(id)?.description).find(Boolean);
  return (
    <section className="checks-mine" aria-label="your checks">
      <h3>Your checks (not part of the sealed sheet)</h3>
      <ul>
        {mine.map(({ v, c }) => (
          <li key={v.verdict_id}>
            {cap(panelName(v.panel_id))}
            {words(v) ? `, ${words(v)}` : ''}: <strong>{ANSWER_WORD[c.answer]}</strong>
            {c.at ? <span className="muted"> ({formatStamp(c.at)})</span> : null}
          </li>
        ))}
      </ul>
      <p className="muted small">Kept in this browser only.</p>
    </section>
  );
}
