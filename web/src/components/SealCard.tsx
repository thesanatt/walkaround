import type { RunManifest, VideoRecord } from '../contracts/types.ts';
import { formatBytes, formatDateTime, hashPrefix } from '../lib/format.ts';
import type { SealState, UploadDone } from '../state/types.ts';
import { SealChips } from './SealChips.tsx';

export interface SealCardProps {
  role: string;
  record: VideoRecord | null;
  sha256Browser?: string | null;
  upload?: UploadDone | null;
  seal?: SealState;
  run?: RunManifest | null;
}

// The "sealed" card: name, bytes, the sha256 (prefix first), the receipt check, the camera time, the stamps.
export function SealCard({ role, record, sha256Browser = null, upload = null, seal, run = null }: SealCardProps) {
  const sha = record?.sha256_server ?? record?.sha256_browser ?? sha256Browser ?? upload?.sha256 ?? null;
  const name = record?.original_filename ?? upload?.filename ?? null;
  const bytes = record?.bytes ?? upload?.bytes ?? null;
  if (!sha && !name) return <div className="empty">{role}: not uploaded yet</div>;
  const match = record?.hash_match ?? null;
  const camera = record?.ffprobe?.creation_time ?? null;
  return (
    <div className={`sealed-card ${match === false ? 'hash-mismatch' : ''}`}>
      <h3>
        {role}: {match === false ? 'hash mismatch' : 'sealed'}
      </h3>
      <dl>
        {name ? (
          <>
            <dt>file</dt>
            <dd>
              {name}
              {bytes !== null ? `, ${formatBytes(bytes)}` : ''}
            </dd>
          </>
        ) : null}
        {sha ? (
          <>
            <dt>sha256</dt>
            <dd className="mono" title={sha}>
              <strong>{hashPrefix(sha)}</strong>
              <span className="muted">{sha.slice(12)}</span>
            </dd>
          </>
        ) : null}
        {match !== null ? (
          <>
            <dt>hash check</dt>
            <dd>hash on your phone = hash on receipt: {match ? 'match' : 'mismatch'}</dd>
          </>
        ) : null}
        {record?.received_at ? (
          <>
            <dt>received</dt>
            <dd>{formatDateTime(record.received_at)}</dd>
          </>
        ) : null}
        {camera ? (
          <>
            <dt>camera says</dt>
            <dd>{formatDateTime(camera)}</dd>
          </>
        ) : null}
        {upload?.warning ? (
          <>
            <dt>note</dt>
            <dd>{upload.warning}</dd>
          </>
        ) : null}
      </dl>
      {seal ? <SealChips seal={seal} run={run} /> : null}
    </div>
  );
}
