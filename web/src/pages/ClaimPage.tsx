import { useState, type FormEvent } from 'react';

import { api, ApiError } from '../api/client.ts';
import type { Claim } from '../contracts/types.ts';
import { sizeLine } from '../lib/lines.ts';
import { panelName } from '../lib/panels.ts';
import { ruleLabel } from '../lib/rules.ts';

const EXIT_TEXT: Record<string, string> = {
  E1: 'There is no car panel in this photo.',
  E2: 'Spark and the SAM vote name different panels; pick the one you mean.',
  E3: 'This panel was not filmed at pickup, so there is no dated frame to compare.',
  E4: 'This panel was only glimpsed at pickup; the sharpest frame and its seconds are below.',
  E5: 'Cannot place the claim mark precisely. Look yourself.',
  E6: 'Spark found no mark in the marked area.',
  E7: 'The marked area covers most of the panel, so it is treated as a region.',
  E8: 'Glare, water or dirt on the claim photo: at most "needs your eye".',
};

export function ClaimResult({ claim }: { claim: Claim }) {
  const size = sizeLine(claim.size);
  return (
    <div className="sealed-card claim-result">
      <h3>{claim.exit ? `${claim.exit}: ${ruleLabel(claim.exit)}` : 'placed against the pickup walk'}</h3>
      {claim.exit && EXIT_TEXT[claim.exit] ? <p>{EXIT_TEXT[claim.exit]}</p> : null}
      <dl>
        <dt>panel</dt>
        <dd>
          {claim.panel_id ? panelName(claim.panel_id) : 'not named'}
          {claim.side_source === 'both_shown' ? ' (no side in the text or the photo, so both sides are shown)' : ''}
        </dd>
        <dt>named by</dt>
        <dd>
          Muse Spark: {claim.naming.spark ? panelName(claim.naming.spark) : 'no panel'}; {claim.naming.sam_votes.length} SAM 3.1 image requests{' '}
          {claim.naming.agreed ? 'agree' : 'do not agree'}
        </dd>
        <dt>at pickup</dt>
        <dd>{claim.coverage_state ? claim.coverage_state.replace('_', ' ') : 'unknown'}</dd>
        <dt>placement</dt>
        <dd>{claim.placement.method === 'none' ? 'not placed' : claim.placement.method.replace('_', ' ')}</dd>
        <dt>size</dt>
        <dd>{size ?? 'size not stated'}</dd>
        {claim.scanner_style ? (
          <>
            <dt>photo</dt>
            <dd>grayscale, like a scanner image</dd>
          </>
        ) : null}
      </dl>
      {claim.paragraph ? <p>{claim.paragraph}</p> : null}
      <p className="small muted mono">
        photo {claim.image.w}x{claim.image.h}
        {claim.image.low_res ? ', low resolution' : ''}, sha256 {claim.image.sha256.slice(0, 12)}
      </p>
    </div>
  );
}

// Claim-first: the photo and words from a damage bill, placed against a finished run's pickup walk.
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
      if (err instanceof ApiError && err.status === 501) setError('Claim photos are not wired on this server yet.');
      else if (err instanceof ApiError && err.status === 404) setError('There is no run with that id on this server.');
      else setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="claim">
      <h1>I have a claim photo</h1>
      <p className="lead">
        Paste the photo and the words from the bill. Muse Spark names the panel, SAM 3.1 image requests check the name, and the dated pickup frame of that
        panel comes back. A size is printed only when a ruler (the plate, or a wheel with its tire size) is in frame, and always with its range.
      </p>
      <form className="form-grid" onSubmit={(e) => void submit(e)}>
        <label>
          run id (the end of your run page address)
          <input value={runId} onChange={(e) => setRunId(e.target.value.trim())} required spellCheck={false} />
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
          tire size from the sidewall, only for a wheel mark
          <input value={tire} onChange={(e) => setTire(e.target.value.trim())} placeholder="225/45R17" spellCheck={false} />
        </label>
        <div>
          <button type="submit" className="btn primary" disabled={busy || !image || !runId}>
            {busy ? 'placing the claim' : 'place the claim against the pickup walk'}
          </button>
        </div>
      </form>
      {error ? <p className="error-line">{error}</p> : null}
      {claim ? <ClaimResult claim={claim} /> : null}
    </div>
  );
}
