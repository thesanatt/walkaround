import { useId, useState, type ChangeEvent, type DragEvent } from 'react';

import type { VideoRecord } from '../contracts/types.ts';
import { formatBytes, hashPrefix } from '../lib/format.ts';
import type { UploadProgress } from '../lib/upload.ts';

export interface UploadUi {
  file: File;
  progress: UploadProgress | null;
  record: VideoRecord | null;
  sha256Browser: string | null;
  error: string | null;
  mismatch: { browser: string; server: string } | null;
}

// Bytes are moving: the slot waits. An error or a mismatch frees it again, a finished upload keeps it locked.
export function uploadBusy(state: UploadUi | null): boolean {
  return state !== null && state.record === null && state.error === null && state.mismatch === null;
}

// Hashed on this device and sent, as one number: each chunk counts once for the hash and once for the send.
export function uploadPct(p: UploadProgress | null): number {
  return p ? Math.round((100 * (p.sentChunks + p.hashedChunks)) / (2 * p.totalChunks)) : 0;
}

// One video slot. accept="video/*" without capture, so a phone offers the photo library (the walk already filmed),
// the camera and files; capture="environment" would open the camera and hide the library.
export function Slot({
  title,
  disabled = false,
  state,
  onFile,
  onRetry,
}: {
  title: string;
  disabled?: boolean;
  state: UploadUi | null;
  onFile: (file: File) => void;
  onRetry?: () => void;
}) {
  const id = useId();
  const [over, setOver] = useState(false);
  const pick = (file: File | undefined) => {
    if (file && !disabled) onFile(file);
  };
  const onChange = (e: ChangeEvent<HTMLInputElement>) => {
    pick(e.target.files?.[0]);
    e.target.value = '';
  };
  const onDrop = (e: DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    setOver(false);
    pick(e.dataTransfer.files?.[0]);
  };
  const p = state?.progress ?? null;
  const busy = uploadBusy(state);
  const pct = uploadPct(p);
  const sha = state?.record?.sha256_server ?? state?.record?.sha256_browser ?? state?.sha256Browser ?? null;
  const cls = ['slot', over ? 'over' : '', disabled && !state?.record ? 'off' : '', state?.record ? 'sealed' : '', state?.error || state?.mismatch ? 'failed' : '']
    .filter(Boolean)
    .join(' ');

  return (
    <div className={cls}>
      <label
        htmlFor={id}
        className="slot-target"
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={onDrop}
      >
        <span className="slot-title">{title}</span>
        {state ? (
          <span className="slot-file">
            {state.file.name}
            <span className="muted">, {formatBytes(state.file.size)}</span>
          </span>
        ) : null}
        <span className="slot-state" aria-live="polite">
          {state?.record ? (
            <>
              Sealed <span className="mono muted">{hashPrefix(sha)}</span>
            </>
          ) : state?.mismatch ? (
            'The hashes did not match. Choose the file again.'
          ) : state?.error ? (
            state.error
          ) : busy ? (
            p ? `Sealing, ${pct}%` : 'Starting'
          ) : disabled ? null : (
            'Choose or drop a video'
          )}
        </span>
        <input id={id} type="file" accept="video/*" disabled={disabled} onChange={onChange} />
        {busy ? (
          <span className="slot-progress" role="progressbar" aria-label={`${title} upload`} aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
            <span style={{ width: `${pct}%` }} />
          </span>
        ) : null}
      </label>
      {state?.error && onRetry ? (
        <button type="button" className="link-button slot-retry" onClick={onRetry}>
          {(p?.sentChunks ?? 0) > 0 || p?.sha256 ? 'Resume' : 'Try again'}
        </button>
      ) : null}
    </div>
  );
}
