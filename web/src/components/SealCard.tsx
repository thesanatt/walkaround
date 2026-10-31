import type { RunManifest, VideoRecord } from '../contracts/types.ts';
import { formatBytes, formatDateTime, hashPrefix } from '../lib/format.ts';
import type { RoleMedia, SealState } from '../state/types.ts';
import { SealChips } from './SealChips.tsx';

export interface SealCardProps {
  role: string;
  video: VideoRecord | null;
  media?: RoleMedia | null;
  sha256Browser?: string | null;
  seal?: SealState | null;
  run?: RunManifest | null;
  failed?: boolean; // the run ended failed (the stage line says why)
}

function titleOf(role: string, match: boolean | null, sealed: boolean, failed: boolean): string {
  if (match === false) return `${role} walk: hash mismatch`;
  if (sealed) return `${role} walk: sealed`;
  if (failed) return `${role} walk: received; not sealed, the run failed`;
  if (match) return `${role} walk: received, hashes match; sealing now`;
  return `${role} walk: received`;
}

// File name, bytes, the SHA-256 (12-hex prefix first), the receipt check, the camera time, and the stamps once the
// seal lands. The first screen never waits on a model.
export function SealCard({ role, video, media = null, sha256Browser = null, seal = null, run = null, failed = false }: SealCardProps) {
  const sha = video?.sha256_server ?? video?.sha256_browser ?? media?.upload?.sha256 ?? sha256Browser ?? null;
  const name = video?.original_filename ?? media?.upload?.filename ?? null;
  const bytes = video?.bytes ?? media?.upload?.bytes ?? null;
  if (!sha && !name) return <div className="empty">The {role} walk has not arrived yet.</div>;
  const match = video?.hash_match ?? (media?.upload ? true : null);
  const camera = media?.ffprobe?.creation_time ?? video?.ffprobe?.creation_time ?? null;
  const sealed = Boolean(seal?.manifest_sha256);
  return (
    <div className={`sealed-card ${match === false ? 'hash-mismatch' : ''}`}>
      <h3>{titleOf(role, match, sealed, failed)}</h3>
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
            <dd className="mono hash" title={sha}>
              <strong>{hashPrefix(sha)}</strong>
              <span className="muted">{sha.slice(12)}</span>
            </dd>
          </>
        ) : null}
        {match !== null ? (
          <>
            <dt>check</dt>
            <dd>hash on your phone = hash on receipt: {match ? 'match' : 'mismatch'}</dd>
          </>
        ) : null}
        {video?.received_at ? (
          <>
            <dt>received</dt>
            <dd>{formatDateTime(video.received_at)}</dd>
          </>
        ) : null}
        {camera ? (
          <>
            <dt>camera says</dt>
            <dd>{formatDateTime(camera)}</dd>
          </>
        ) : null}
        {media?.upload?.warning ? (
          <>
            <dt>note</dt>
            <dd>{media.upload.warning}</dd>
          </>
        ) : null}
      </dl>
      {seal ? <SealChips seal={seal} run={run} /> : null}
    </div>
  );
}
