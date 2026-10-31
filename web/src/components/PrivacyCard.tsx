import type { DeletionRecord } from '../contracts/types.ts';
import { formatBytes, formatDateTime, hashPrefix } from '../lib/format.ts';
import { deleteFailed, storageLine, uploadedFileIds } from '../lib/privacy.ts';
import type { RunState } from '../state/types.ts';

const KIND_TEXT: Record<DeletionRecord['kind'], string> = {
  files_api: 'Files API upload',
  response: 'stored model response',
  quarantine: 'unblurred original',
  file: 'unblurred file',
};

function receipt(d: DeletionRecord): string {
  const r = d.api_response;
  if (r && typeof r === 'object' && 'deleted' in r) return `, receipt deleted: ${String(r['deleted'])}`;
  return '';
}

function Row({ d }: { d: DeletionRecord }) {
  if (deleteFailed(d)) {
    // The reason carries the section 14 item 25 sentence: "delete request failed at <time>; expiry at <time> applies".
    return (
      <li className="delete-failed">
        {KIND_TEXT[d.kind]} <span className="mono">{d.target}</span>: {d.reason || `delete request failed at ${formatDateTime(d.deleted_at)}`}
      </li>
    );
  }
  return (
    <li>
      {KIND_TEXT[d.kind]} <span className="mono">{d.target}</span> deleted {formatDateTime(d.deleted_at)}
      {d.bytes !== null ? `, ${formatBytes(d.bytes)}` : ''}
      {d.sha256 ? <span className="mono">, sha256 {hashPrefix(d.sha256)}</span> : null}
      {receipt(d)}
      {d.reason ? ` (${d.reason})` : ''}
    </li>
  );
}

export function PrivacyCard({ state }: { state: RunState }) {
  const { run, deletions, completed } = state;
  const uploads = uploadedFileIds(state.media, state.videos, deletions);
  return (
    <div className="privacy">
      <p>{storageLine(run, { completed, uploads, deletions })}</p>
      {run ? <p className="small">Plate: {run.keep_plate ? 'kept on this sheet because the renter asked for it at upload' : 'blurred'}. Faces: always blurred.</p> : null}
      {deletions.length ? (
        <ul className="small deletions">
          {deletions.map((d, i) => (
            <Row key={`${d.target}-${i}`} d={d} />
          ))}
        </ul>
      ) : (
        <p className="muted small">The deletion rows arrive at the end of the run, each with its time and, for the Files API, the delete receipt.</p>
      )}
    </div>
  );
}
