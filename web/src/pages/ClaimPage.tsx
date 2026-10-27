import { useState, type FormEvent } from 'react';

import { api, ApiError } from '../api/client.ts';
import type { Claim } from '../contracts/types.ts';
import { formatClock } from '../lib/format.ts';
import { panelName } from '../lib/panels.ts';
import { ruleLabel } from '../lib/rules.ts';

function sizeLine(size: Claim['size']): string {
  if (!size) return 'size not stated';
  if (size.refused_reason) return `size not stated: ${size.refused_reason}`;
  if (size.inches === null) return 'size not stated';
  const ruler = size.ruler === 'plate' ? 'the license plate, 12 in' : size.ruler === 'wheel' ? `the wheel${size.tire_code ? `, ${size.tire_code}` : ''}` : 'no ruler';
  return `about ${size.inches.toFixed(1)} in, between ${(size.bound_lo ?? 0).toFixed(1)} and ${(size.bound_hi ?? 0).toFixed(1)} in (ruler: ${ruler})`;
}

export function ClaimResult({ claim }: { claim: Claim }) {
  const windows = claim.match?.kf_pair ? null : null;
  return (
    <div className="sealed-card">
      <h3>{claim.exit ? `exit ${claim.exit}: ${ruleLabel(claim.exit)}` : 'claim placed'}</h3>
      <dl>
        <dt>panel</dt>
        <dd>
          {claim.panel_id ? panelName(claim.panel_id) : 'not named'}
          {claim.side_source === 'both_shown' ? ' (side unknown; both sides shown)' : ''}
        </dd>
        <dt>naming</dt>
        <dd>
          Spark: {claim.naming.spark ? panelName(claim.naming.spark) : 'none'}; {claim.naming.sam_votes.length} SAM image requests {claim.naming.agreed ? 'agree' : 'do not agree'}
        </dd>
        <dt>coverage at pickup</dt>
        <dd>{claim.coverage_state ?? 'unknown'}</dd>
        <dt>placement</dt>
        <dd>{claim.placement.method === 'none' ? 'not placed' : claim.placement.method.replace('_', ' ')}</dd>
        <dt>size</dt>
        <dd>{sizeLine(claim.size)}</dd>
        {claim.scanner_style ? (
          <>
            <dt>image</dt>
            <dd>looks like a scanner image (grayscale)</dd>
          </>
        ) : null}
        {windows}
      </dl>
      {claim.paragraph ? <p>{claim.paragraph}</p> : null}
      {claim.match?.state === 'comparable' && claim.match.pairs[0] ? (
        <p className="small muted">nearest pickup mark {claim.match.pairs[0].d.toFixed(2)} units away; keyframe pair {claim.match.kf_pair?.join(' and ') ?? ''}</p>
      ) : null}
      {claim.image ? (
        <p className="small muted mono">
          image {claim.image.w}x{claim.image.h}
          {claim.image.low_res ? ' (low resolution)' : ''}, sha256 {claim.image.sha256.slice(0, 12)}
        </p>
      ) : null}
    </div>
  );
}

// Claim-first: the company's photo and text against the pickup walk of a run. The flow is the same for a staged photo.
export function ClaimPage({ query }: { query: URLSearchParams }) {
  const [runId, setRunId] = useState(query.get('run') ?? '');
  const [text, setText] = useState('');
  const [tire, setTire] = useState('');
  const [image, setImage] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [claim, setClaim] = useState<Claim | null>(null);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!image || !runId) return;
    setBusy(true);
    setError(null);
    try {
      setClaim(await api.claim({ run_id: runId, image, text, tire_code: tire || undefined }));
    } catch (err) {
      if (err instanceof ApiError && err.status === 501) setError('claim-first is not wired on this server yet');
      else setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      <h1>I have a claim photo</h1>
      <p className="muted" style={{ maxWidth: '70ch' }}>
        Paste the photo and the words from the bill. Spark names the panel, SAM image requests vote on it, and the dated pickup frame comes back with the mark boxed
        and, when a ruler is in frame, a size with its bound. Anything it cannot place says so.
      </p>
      <form className="form-grid" onSubmit={submit}>
        <label>
          run id (the URL of your run)
          <input value={runId} onChange={(e) => setRunId(e.target.value.trim())} required placeholder="01J..." />
        </label>
        <label>
          the claim photo
          <input type="file" accept="image/*" onChange={(e) => setImage(e.target.files?.[0] ?? null)} required />
        </label>
        <label>
          the words on the bill
          <textarea value={text} onChange={(e) => setText(e.target.value)} rows={3} placeholder="scratch, right rear door" />
        </label>
        <label>
          tire size from the sidewall, only for a wheel mark (for example 225/45R17)
          <input value={tire} onChange={(e) => setTire(e.target.value.trim())} placeholder="optional" />
        </label>
        <div>
          <button type="submit" className="btn primary" disabled={busy || !image || !runId}>
            {busy ? 'placing the claim' : 'place the claim against the pickup walk'}
          </button>
        </div>
      </form>
      {error ? <p className="error-line">{error}</p> : null}
      {claim ? <ClaimResult claim={claim} /> : null}
      <p className="small muted" style={{ marginTop: 16 }}>
        Video time on a placed frame is the proxy frame's pts (for example {formatClock(11)} to {formatClock(16)}); wall clocks are the camera time plus pts.
      </p>
    </div>
  );
}
