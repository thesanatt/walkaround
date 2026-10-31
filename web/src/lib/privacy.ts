import type { DeletionRecord, RunManifest, VideoRecord } from '../contracts/types.ts';
import type { RoleMedia } from '../state/types.ts';

const TRAINING = "Processed with Meta's Model API in Standard mode, which Meta's documentation says is not used to train Meta models. ";
const TERMS = "Meta's terms allow it to retain content for abuse review and legal reasons; Walkaround does not control that. ";
const SAM_SAW = 'SAM 3.1 saw the unblurred proxy for the length of one segmentation pass to find the plates and faces it then blurred.';

// The upload page, and a run that has not ended: what will happen, in the present tense.
export const PRIVACY_BEFORE =
  TRAINING +
  'Uploads expire after one hour and Walkaround deletes them when the run ends. Model requests are sent with storage off. ' +
  TERMS +
  'SAM 3.1 sees the unblurred proxy for one segmentation pass, to find the plates and faces it then blurs.';

// A remote delete that has no receipt (privacy.delete_remote stores none once every try failed) or one saying false.
export function deleteFailed(d: DeletionRecord): boolean {
  if (d.kind !== 'files_api' && d.kind !== 'response') return false;
  return !d.api_response || d.api_response['deleted'] === false;
}

// Every Files API id the run is known to have used: files.uploaded, the video records, the deletion rows.
export function uploadedFileIds(media: Record<string, RoleMedia>, videos: Record<string, VideoRecord>, deletions: DeletionRecord[]): string[] {
  const ids = new Set<string>();
  for (const m of Object.values(media)) if (m.fileId) ids.add(m.fileId);
  for (const v of Object.values(videos)) if (v.files_api?.file_id) ids.add(v.files_api.file_id);
  for (const d of deletions) if (d.kind === 'files_api') ids.add(d.target);
  return [...ids].sort();
}

function count(n: number, one: string, many: string): string {
  return n === 1 ? one : `${n} ${many}`;
}

// privacy.privacy_line word for word, but only for what the deletion rows show: before the run ends it is the
// present-tense promise, and a delete that failed (or never ran) is said, not claimed (section 8 item 7, 14 item 25).
export function storageLine(run: RunManifest | null, opts: { completed: boolean; uploads: string[]; deletions: DeletionRecord[] }): string {
  if (!opts.completed) return PRIVACY_BEFORE;
  if (!opts.uploads.length) return "No file from this run reached Meta's Model API.";
  const rows = new Map(opts.deletions.filter((d) => d.kind === 'files_api').map((d) => [d.target, d]));
  const failed = opts.uploads.filter((id) => rows.has(id) && deleteFailed(rows.get(id)!)).length;
  const pending = opts.uploads.filter((id) => !rows.has(id)).length;
  let text = TRAINING;
  if (!failed && !pending) text += 'Uploads were set to expire after one hour and were deleted by Walkaround when the run ended. ';
  else {
    text += 'Uploads were set to expire after one hour. ';
    if (failed) text += `Walkaround's delete request failed for ${count(failed, 'one upload', 'uploads')}, so the expiry applies; the rows below have the times. `;
    if (pending) text += `The run ended before Walkaround deleted ${count(pending, 'one upload', 'uploads')}, so the one-hour expiry applies. `;
  }
  if (run?.responses_store === 'deleted_per_request') {
    const bad = opts.deletions.filter((d) => d.kind === 'response' && deleteFailed(d)).length;
    text += bad
      ? `Model requests were sent with storage off; a model refused it, and Walkaround's delete of ${count(bad, 'one stored response', 'stored responses')} failed (rows below). `
      : 'Model requests were sent with storage off and every stored model response was deleted by Walkaround when the run ended. ';
  } else {
    text += 'Model requests were sent with storage off. ';
  }
  return text + TERMS + SAM_SAW;
}
