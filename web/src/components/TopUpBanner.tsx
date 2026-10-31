import { useEffect, useState, type ChangeEvent } from 'react';

import { api, ApiError } from '../api/client.ts';
import type { Coverage } from '../contracts/types.ts';
import { COVERAGE_LABEL, panelName } from '../lib/panels.ts';
import { hashFile, sharedHashWorker } from '../lib/upload.ts';

// What a top-up clip added: each slot filmed or glimpsed in a topup_<n> walk, with its seconds.
export function topupGains(coverage: Coverage): { panel: string; state: string; seconds: number }[] {
  const out: { panel: string; state: string; seconds: number }[] = [];
  for (const [role, per] of Object.entries(coverage.per_role)) {
    if (!role.startsWith('topup_')) continue;
    for (const [panel, cov] of Object.entries(per)) {
      if (cov && (cov.state === 'filmed' || cov.state === 'glimpsed')) out.push({ panel, state: cov.state, seconds: cov.seconds });
    }
  }
  return out;
}

export function topupErrorText(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === 409 && err.body && typeof err.body === 'object' && 'sha256_server' in err.body)
      return 'Hash mismatch: the clip that arrived is not the clip on this device. The server deleted it and ran nothing. Send it again.';
    if (err.status === 403) return 'This run is public, so it takes no top-up clip. Top up your own run from its page.';
    if (err.status === 413) return 'The clip is too large. About 10 seconds of the missing panels is enough.';
    if (err.status === 429) {
      // orchestrator.check_can_topup says which stop applied: the pause, today's card ceiling or this run's ceiling.
      const reason = err.body && typeof err.body === 'object' ? (err.body as { reason?: unknown }).reason : null;
      return typeof reason === 'string' && reason ? `The top-up did not run: ${reason}.` : 'Live runs are paused for today, so the top-up did not run.';
    }
  }
  return err instanceof Error ? err.message : String(err);
}

// The second file input: only the missing panels' nouns (plus car, plate and face) re-run on the new clip. The clip is
// hashed on this device first, like a walk, so the server checks it on receipt. capture="environment" opens the
// camera: the renter films the missing panels now, at the car.
export function TopUpBanner({ runId, coverage, onCoverage }: { runId: string; coverage: Coverage | null; onCoverage: (c: Coverage) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [running, setRunning] = useState<'hashing' | 'running' | null>(null);
  const [started, setStarted] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Coverage | null>(null);
  const [inputKey, setInputKey] = useState(0); // a new key empties the file input after a clip is sent

  useEffect(() => {
    if (running !== 'running') return;
    const handle = setInterval(() => setElapsed(Math.round((Date.now() - started) / 1000)), 1000);
    return () => clearInterval(handle);
  }, [running, started]);

  if (!coverage || (!coverage.missing.length && !result)) return null;
  const nouns = coverage.nouns_to_rerun;

  const submit = async () => {
    if (!file) return;
    setRunning('hashing');
    setError(null);
    try {
      const sha256 = await hashFile(file, sharedHashWorker());
      setStarted(Date.now());
      setElapsed(0);
      setRunning('running');
      const next = await api.topup(runId, file, sha256);
      setResult(next);
      onCoverage(next);
      setFile(null);
      setInputKey((k) => k + 1);
    } catch (err) {
      setError(topupErrorText(err));
    } finally {
      setRunning(null);
    }
  };

  const gains = result ? topupGains(result) : [];
  return (
    <div className="topup-banner">
      {coverage.missing.length ? (
        <>
          <p className="topup-ask-line">{coverage.topup_text || `missing: ${coverage.missing.map(panelName).join(', ')}`}</p>
          <p className="small muted">
            Film only what is missing, about 10 seconds. Only these lanes run on the new clip: {nouns.join(', ')} ({nouns.length} of 16).
          </p>
          <div className="topup-form">
            <input key={inputKey} type="file" accept="video/*" capture="environment" disabled={running !== null} onChange={(e: ChangeEvent<HTMLInputElement>) => setFile(e.target.files?.[0] ?? null)} />
            <button type="button" className="btn small" onClick={() => void submit()} disabled={!file || running !== null}>
              send the top-up clip
            </button>
          </div>
        </>
      ) : null}
      {running === 'hashing' ? <p className="small">hashing the clip on this device</p> : null}
      {running === 'running' ? (
        <p className="small" role="status">
          running {nouns.join(', ')} ({nouns.length} of 16 lanes), <span className="mono">{elapsed} s</span>
        </p>
      ) : null}
      {error ? <p className="error-line small">{error}</p> : null}
      {result ? (
        <div className="topup-result" role="status">
          <p>
            {gains.length
              ? gains.map((g) => `${panelName(g.panel)}: ${COVERAGE_LABEL[g.state]} in the top-up clip, ${g.seconds.toFixed(1)} s`).join('; ')
              : 'The top-up clip added no panel.'}
            {result.rerun_seconds !== null ? `. The top-up lanes ran in ${result.rerun_seconds.toFixed(0)} s.` : '.'}
          </p>
          <p className="small">{result.missing.length ? `Still missing: ${result.missing.map(panelName).join(', ')}.` : 'Nothing is missing now.'}</p>
          {result.odometer.asked ? <p className="small">odometer: {result.odometer.filmed ? 'filmed' : 'not filmed, add 3 seconds?'}</p> : null}
        </div>
      ) : null}
    </div>
  );
}
