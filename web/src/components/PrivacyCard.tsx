import type { DeletionRecord, RunManifest } from '../contracts/types.ts';
import { formatDateTime } from '../lib/format.ts';

export const STORAGE_LINE =
  "Processed with Meta's Model API in Standard mode, which Meta's documentation says is not used to train Meta models. Uploads were set to expire after one hour and were deleted by Walkaround when the run ended. Model requests were sent with storage off. Meta's terms allow it to retain content for abuse review and legal reasons; Walkaround does not control that. SAM 3.1 saw the unblurred proxy for the length of one segmentation pass to find the plates and faces it then blurred.";

export function PrivacyCard({ deletions, run }: { deletions: DeletionRecord[]; run: RunManifest | null }) {
  return (
    <div className="privacy">
      <p>{STORAGE_LINE}</p>
      {run ? <p className="small">plate: {run.keep_plate ? 'kept on this sheet because the renter asked for it at upload' : 'blurred'}</p> : null}
      {deletions.length ? (
        <ul className="small mono">
          {deletions.map((d, i) => (
            <li key={`${d.target}-${i}`}>
              {d.kind} {d.target} deleted {formatDateTime(d.deleted_at)}
              {d.reason ? ` (${d.reason})` : ''}
            </li>
          ))}
        </ul>
      ) : (
        <p className="muted small">Deletion rows arrive at the end of the run.</p>
      )}
    </div>
  );
}
