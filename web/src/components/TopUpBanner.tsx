import { useState, type ChangeEvent } from 'react';

import { api, ApiError } from '../api/client.ts';
import type { Coverage } from '../contracts/types.ts';
import { hashFile, sharedHashWorker } from '../lib/upload.ts';

// The second file input: only the missing panels' nouns (plus car, plate and face) re-run on the new clip. The clip is
// hashed on this device first, like a walk, so the server checks it on receipt. This one keeps capture="environment":
// the renter films the missing panels now, at the car.
export function TopUpBanner({ runId, coverage, onCoverage }: { runId: string; coverage: Coverage | null; onCoverage: (c: Coverage) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [running, setRunning] = useState<'hashing' | 'running' | null>(null);
  const [error, setError] = useState<string | null>(null);
  if (!coverage || !coverage.missing.length) return null;
  const nouns = coverage.nouns_to_rerun;

  const submit = async () => {
    if (!file) return;
    setRunning('hashing');
    setError(null);
    try {
      const sha256 = await hashFile(file, sharedHashWorker());
      setRunning('running');
      onCoverage(await api.topup(runId, file, sha256));
      setFile(null);
    } catch (err) {
      if (err instanceof ApiError && err.status === 409 && err.body && typeof err.body === 'object' && 'sha256_server' in err.body) {
        setError('Hash mismatch: the clip that arrived is not the clip on this device. The server deleted it and ran nothing. Send it again.');
      } else setError(err instanceof Error ? err.message : String(err));
    } finally {
      setRunning(null);
    }
  };

  return (
    <div className="topup-form-wrap">
      <p className="small">Film only what is missing, about 10 seconds. Only these nouns run on the new clip: {nouns.join(', ')}.</p>
      {running === 'hashing' ? <p className="small">hashing the clip on this device</p> : null}
      {running === 'running' ? (
        <p className="small">
          running: {nouns.join(', ')} ({nouns.length} of 16 lanes)
        </p>
      ) : null}
      {error ? <p className="error-line small">{error}</p> : null}
      <div className="topup-form">
        <input type="file" accept="video/*" capture="environment" disabled={running !== null} onChange={(e: ChangeEvent<HTMLInputElement>) => setFile(e.target.files?.[0] ?? null)} />
        <button type="button" className="btn small" onClick={() => void submit()} disabled={!file || running !== null}>
          send the top-up clip
        </button>
      </div>
    </div>
  );
}
