import type { RunManifest } from '../contracts/types.ts';
import { formatDateTime, hashPrefix } from '../lib/format.ts';
import type { SealState } from '../state/types.ts';

function otsText(ots: string | null): string {
  if (!ots) return 'OpenTimestamps not started';
  if (ots === 'pending') return 'OpenTimestamps pending';
  if (ots.startsWith('confirmed:')) return `Bitcoin block ${ots.slice('confirmed:'.length)}`;
  if (ots.startsWith('failed:')) return `OpenTimestamps failed: ${ots.slice('failed:'.length)}`;
  return ots;
}

export function SealChips({ seal, run }: { seal: SealState; run: RunManifest | null }) {
  const minutes = run?.clocks.sealed_minutes_after_camera;
  return (
    <div className="chips">
      {seal.manifest_sha256 ? <span className="chip mono">manifest {hashPrefix(seal.manifest_sha256)}</span> : null}
      {seal.ots || seal.tsr_time ? (
        <span className="chip">
          stamped: {otsText(seal.ots)}
          {seal.tsr_time ? `, ${seal.tsa_issuer ?? 'freetsa'} token ${formatDateTime(seal.tsr_time)} UTC` : ''}
        </span>
      ) : null}
      {run?.clocks.sealed_at ? <span className="chip">sealed at {formatDateTime(run.clocks.sealed_at)}</span> : null}
      {run?.clocks.first_model_call_at ? <span className="chip">first model call at {formatDateTime(run.clocks.first_model_call_at)}</span> : null}
      {minutes !== null && minutes !== undefined ? (
        <span className="chip">{typeof minutes === 'number' ? `sealed ${minutes} minutes after the camera time` : minutes}</span>
      ) : null}
    </div>
  );
}
