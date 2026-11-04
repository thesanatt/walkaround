import { useEffect, useRef } from 'react';

import { routes } from '../api/client.ts';
import type { Mark } from '../contracts/types.ts';
import { formatStamp, hashPrefix } from '../lib/format.ts';
import { Link, navigate } from '../lib/router.tsx';
import { sheetFileHash } from '../lib/runview.ts';
import type { StoredCheck } from '../lib/yourcheck.ts';
import type { RunState } from '../state/types.ts';
import { YourChecks } from './YourChecks.tsx';

function otsText(ots: string | null): string {
  if (!ots) return 'not started';
  if (ots === 'pending') return 'pending';
  if (ots.startsWith('confirmed:')) return `Bitcoin block ${ots.slice('confirmed:'.length)}`;
  if (ots.startsWith('failed:')) return 'failed, retry scheduled';
  return ots;
}

export const sealedAt = (state: RunState) => state.seal.tsr_time ?? state.run?.clocks.sealed_at ?? null;

function Hash({ sha }: { sha: string }) {
  return (
    <span className="mono hash" title={sha}>
      {hashPrefix(sha)}
      <span className="muted">{sha.slice(12)}</span>
    </span>
  );
}

// The sealed panel: when and what was sealed, the hashes, the PDF and how to check it. The renter's own checks are
// listed apart, as theirs: they are not part of the sheet.
export function SealedPanel({
  runId,
  state,
  checks,
  marks,
  closeTo,
  recordTo,
}: {
  runId: string;
  state: RunState;
  checks: Record<string, StoredCheck>;
  marks: Map<string, Mark>;
  closeTo: string;
  recordTo: string;
}) {
  const dialog = useRef<HTMLElement>(null);
  useEffect(() => dialog.current?.focus(), []);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') navigate(closeTo, { replace: true });
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [closeTo]);
  const at = sealedAt(state);
  const pdf = Boolean(sheetFileHash(state.sheet.sha256s, 'evidence.pdf'));
  const roles = (['pickup', 'return'] as const).filter((r) => state.videos[r] || state.media[r]?.upload);

  return (
    <div className="sealed-layer">
      <Link to={closeTo} className="sealed-backdrop" replace>
        <span className="sr-only">Close</span>
      </Link>
      <aside ref={dialog} tabIndex={-1} className="sealed-panel" role="dialog" aria-modal="true" aria-labelledby="sealed-title">
        <div className="sealed-top">
          <h2 id="sealed-title">Sealed</h2>
          <button type="button" className="link-button sealed-close" onClick={() => navigate(closeTo, { replace: true })}>
            Close
          </button>
        </div>
        <p className="sealed-when">
          {at ? `${roles.length > 1 ? 'Both videos were' : 'The video was'} sealed ${formatStamp(at, true)}, before any model saw ${roles.length > 1 ? 'them' : 'it'}.` : 'Not sealed yet.'}
        </p>
        <dl className="sealed-facts">
          {roles.map((role) => {
            const v = state.videos[role];
            const sha = v?.sha256_server ?? v?.sha256_browser ?? state.media[role]?.upload?.sha256 ?? null;
            return (
              <div key={role}>
                <dt>{role === 'pickup' ? 'Pickup video' : 'Return video'}</dt>
                <dd>{sha ? <Hash sha={sha} /> : 'not received'}</dd>
              </div>
            );
          })}
          {state.seal.manifest_sha256 ? (
            <div>
              <dt>Manifest</dt>
              <dd>
                <Hash sha={state.seal.manifest_sha256} />
              </dd>
            </div>
          ) : null}
          {state.seal.tsr_time ? (
            <div>
              <dt>Timestamp</dt>
              <dd>
                RFC 3161, {state.seal.tsa_issuer ?? 'freetsa'}, {formatStamp(state.seal.tsr_time, true)}
              </dd>
            </div>
          ) : null}
          <div>
            <dt>OpenTimestamps</dt>
            <dd>{otsText(state.seal.ots)}</dd>
          </div>
        </dl>
        <div className="sealed-actions">
          {pdf ? (
            <a href={routes.sheetPdf(runId)} className="button">
              Download the PDF
            </a>
          ) : (
            <Link to={`/run/${encodeURIComponent(runId)}/sheet`} className="button">
              Open the sheet
            </Link>
          )}
          <p className="sealed-links">
            {pdf ? <Link to={`/run/${encodeURIComponent(runId)}/sheet`}>Open the sheet</Link> : null}
            <Link to="/verify">Verify it yourself</Link>
            <a href={routes.bundleZip(runId)}>bundle.zip</a>
          </p>
        </div>
        <YourChecks checks={checks} verdicts={state.verdicts} marks={marks} />
        <p className="sealed-more">
          <Link to={recordTo}>Everything the check recorded</Link>
          {!state.run?.keep ? <Link to={`/claim?run=${encodeURIComponent(runId)}`}>A bill arrived? Place its photo</Link> : null}
        </p>
      </aside>
    </div>
  );
}
