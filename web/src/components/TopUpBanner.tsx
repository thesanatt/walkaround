import { useState, type ChangeEvent } from 'react';

import { api } from '../api/client.ts';
import type { Coverage } from '../contracts/types.ts';

// The second file input: only the missing nouns re-run on the new clip (POST /api/runs/{id}/topup).
export function TopUpBanner({ runId, coverage, onCoverage }: { runId: string; coverage: Coverage | null; onCoverage: (c: Coverage) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [seconds, setSeconds] = useState<number | null>(null);
  if (!coverage || !coverage.missing.length) return null;
  const nouns = coverage.nouns_to_rerun;

  const submit = async () => {
    if (!file) return;
    setRunning(true);
    setError(null);
    try {
      const next = await api.topup(runId, file);
      onCoverage(next);
      setSeconds(next.rerun_seconds);
      setFile(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="topup-form-wrap">
      <p className="small muted">Film only what is missing. Only those nouns re-run on the new clip.</p>
      {running ? (
        <p className="small">
          running: {nouns.join(', ')} ({nouns.length} of 14 nouns)
        </p>
      ) : null}
      {seconds !== null ? <p className="small mono">top-up lanes re-ran in {seconds.toFixed(0)} s</p> : null}
      {error ? <p className="error-line small">{error}</p> : null}
      <div className="topup-form">
        <input type="file" accept="video/*" capture="environment" disabled={running} onChange={(e: ChangeEvent<HTMLInputElement>) => setFile(e.target.files?.[0] ?? null)} />
        <button type="button" className="btn small" onClick={submit} disabled={!file || running}>
          send the top-up clip
        </button>
      </div>
    </div>
  );
}
