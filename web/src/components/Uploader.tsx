import { useId, useState, type ChangeEvent, type DragEvent } from 'react';

import type { VideoRecord } from '../contracts/types.ts';
import { formatBytes } from '../lib/format.ts';
import type { UploadProgress } from '../lib/upload.ts';
import { SealCard } from './SealCard.tsx';

export interface UploadUi {
  file: File;
  progress: UploadProgress | null;
  record: VideoRecord | null;
  sha256Browser: string | null;
  error: string | null;
  mismatch: { browser: string; server: string } | null;
}

export interface UploaderProps {
  role: 'pickup' | 'return';
  step?: string;
  title: string;
  hint: string;
  disabled?: boolean;
  state: UploadUi | null;
  onFile: (file: File) => void;
  onRetry?: () => void;
}

// Bytes are moving: the drop zone waits. An error or a mismatch frees it again, a finished upload keeps it locked.
export function uploadBusy(state: UploadUi | null): boolean {
  return state !== null && state.record === null && state.error === null && state.mismatch === null;
}

// One drop zone. accept="video/*" without capture, so a phone offers the photo library (the walk the renter already
// filmed), the camera and files. capture="environment" would open the camera directly and hide the library.
// TODO(Friday phone test): confirm the chooser on iOS Safari and Android Chrome with a camera-roll MOV.
export function Uploader({ role, step, title, hint, disabled = false, state, onFile, onRetry }: UploaderProps) {
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
  const pct = p ? Math.round((100 * (p.sentChunks + p.hashedChunks)) / (2 * p.totalChunks)) : 0;
  const busy = uploadBusy(state);

  return (
    <div className="uploader">
      <label
        htmlFor={id}
        className={`dropzone ${over ? 'active' : ''} ${disabled ? 'disabled' : ''} ${state?.record ? 'done' : ''}`}
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={onDrop}
      >
        {step ? <span className="dropzone-step">{step}</span> : null}
        <span className="dropzone-title">{title}</span>
        <span className="dropzone-hint">{state ? `${state.file.name}, ${formatBytes(state.file.size)}` : hint}</span>
        {busy ? <span className="dropzone-state">{p ? `Sending, ${pct}%` : 'Starting'}</span> : null}
        {state?.record ? <span className="dropzone-state done">Sealed</span> : null}
        <input id={id} type="file" accept="video/*" disabled={disabled} onChange={onChange} />
      </label>
      {busy ? (
        <div>
          <div className="progress" role="progressbar" aria-label={`${role} upload`} aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
            <span style={{ width: `${pct}%` }} />
          </div>
          <p className="small muted">
            {p
              ? `hashed ${p.hashedChunks} of ${p.totalChunks} chunks on this device, sent ${p.sentChunks} of ${p.totalChunks}${p.sha256 ? `; sha256 ${p.sha256.slice(0, 12)}, waiting for the hash on receipt` : ''}`
              : 'starting'}
          </p>
        </div>
      ) : null}
      {state?.error ? (
        <p className="error-line small">
          {state.error}{' '}
          {onRetry ? (
            <button type="button" className="btn small" onClick={onRetry}>
              {(p?.sentChunks ?? 0) > 0 || p?.sha256 ? 'resume upload' : 'try again'}
            </button>
          ) : null}
        </p>
      ) : null}
      {state?.mismatch ? (
        <div className="sealed-card hash-mismatch">
          <h3>{role} walk: hash mismatch; upload again</h3>
          <dl>
            <dt>on your phone</dt>
            <dd className="mono small hash">{state.mismatch.browser}</dd>
            <dt>on receipt</dt>
            <dd className="mono small hash">{state.mismatch.server || 'not reported'}</dd>
          </dl>
          <p className="small muted">The server deleted the bytes it received. No model call was made. Drop or choose the file again to send it fresh.</p>
        </div>
      ) : null}
      {state?.record ? (
        <details className="seal-more">
          <summary>Seal details</summary>
          <SealCard role={role} video={state.record} sha256Browser={state.sha256Browser} />
        </details>
      ) : null}
    </div>
  );
}
