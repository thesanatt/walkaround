import type { RunManifest } from '../contracts/types.ts';
import { formatDateTime, hashPrefix } from '../lib/format.ts';
import type { SealState } from '../state/types.ts';

function otsText(ots: string | null): string {
  if (!ots) return 'OpenTimestamps not started';
  if (ots === 'pending') return 'OpenTimestamps pending';
  if (ots.startsWith('confirmed:')) return `Bitcoin block ${ots.slice('confirmed:'.length)}`;
  if (ots.startsWith('failed:')) return `OpenTimestamps failed (${ots.slice('failed:'.length)}); retry scheduled`;
  return ots;
}

function timeUtc(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return `${d.toISOString().slice(11, 19)} UTC`;
}

// The stamps and the three clocks of page 1, as they arrive. A replay loads the final run.json first, so each clock
// waits for its own event: the seal clocks for seal.manifest, the first model call for files.uploaded.
export function SealChips({ seal, run }: { seal: SealState; run: RunManifest | null }) {
  const sealed = Boolean(seal.manifest_sha256);
  const minutes = sealed ? (run?.clocks.sealed_minutes_after_camera ?? null) : null;
  const sealedAt = sealed ? (run?.clocks.sealed_at ?? null) : null;
  const firstCall = seal.files_uploaded ? (run?.clocks.first_model_call_at ?? null) : null;
  return (
    <div className="chips">
      {seal.manifest_sha256 ? <span className="chip mono">manifest {hashPrefix(seal.manifest_sha256)}</span> : null}
      {seal.ots || seal.tsr_time ? (
        <span className="chip">
          stamped: {otsText(seal.ots)}
          {seal.tsr_time ? `, ${seal.tsa_issuer ?? 'freetsa'} token ${timeUtc(seal.tsr_time)}` : ''}
        </span>
      ) : null}
      {sealedAt ? <span className="chip">sealed at {formatDateTime(sealedAt)}</span> : null}
      {firstCall ? <span className="chip">first model call at {formatDateTime(firstCall)}</span> : null}
      {minutes !== null ? <span className="chip">{typeof minutes === 'number' ? `sealed ${minutes} minutes after the camera time` : minutes}</span> : null}
    </div>
  );
}
