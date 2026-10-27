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
  title: string;
  hint: string;
  disabled?: boolean;
  state: UploadUi | null;
  onFile: (file: File) => void;
}

// One drop zone: accept="video/*" capture="environment" so a phone opens the camera roll or the camera.
export function Uploader({ role, title, hint, disabled = false, state, onFile }: UploaderProps) {
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
  const busy = state !== null && state.record === null && state.error === null && state.mismatch === null;

  return (
    <div className="uploader">
      <label
        htmlFor={id}
        className={`dropzone ${over ? 'active' : ''} ${disabled ? 'disabled' : ''}`}
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={onDrop}
      >
        <span className="dropzone-title">{title}</span>
        <span className="dropzone-hint">{state ? `${state.file.name}, ${formatBytes(state.file.size)}` : hint}</span>
        <input id={id} type="file" accept="video/*" capture="environment" disabled={disabled} onChange={onChange} />
      </label>
      {busy && p ? (
        <div>
          <div className="progress" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
            <span style={{ width: `${pct}%` }} />
          </div>
          <p className="small muted mono">
            hashed {p.hashedChunks} of {p.totalChunks} chunks, sent {p.sentChunks} of {p.totalChunks}
            {p.sha256 ? `, sha256 ${p.sha256.slice(0, 12)}; waiting for the hash on receipt` : ''}
          </p>
        </div>
      ) : null}
      {busy && !p ? <p className="small muted">starting the upload</p> : null}
      {state?.error ? <p className="error-line small">{state.error}</p> : null}
      {state?.mismatch ? (
        <div className="sealed-card hash-mismatch">
          <h3>hash mismatch; upload again</h3>
          <dl>
            <dt>on your phone</dt>
            <dd className="mono small">{state.mismatch.browser}</dd>
            <dt>on receipt</dt>
            <dd className="mono small">{state.mismatch.server || 'not reported'}</dd>
          </dl>
          <p className="small muted">The server deleted the bytes it received. No model call was made.</p>
        </div>
      ) : null}
      {state?.record ? <SealCard role={role} record={state.record} sha256Browser={state.sha256Browser} /> : null}
    </div>
  );
}
