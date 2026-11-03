import { useEffect, useRef, useState } from 'react';

import { api, ApiError } from '../api/client.ts';
import type { RunManifest, StatusResponse, WalkRole } from '../contracts/types.ts';
import { Uploader, uploadBusy, type UploadUi } from '../components/Uploader.tsx';
import { PRIVACY_BEFORE } from '../lib/privacy.ts';
import { Link, navigate } from '../lib/router.tsx';
import { HashMismatch, limitsOf, newUploadId, sharedHashWorker, uploadFile } from '../lib/upload.ts';

type Kind = 'pickup_check' | 'pair';

function counterText(status: StatusResponse | null): string {
  if (!status) return 'Checking how many live runs are left today.';
  if (status.paused) return 'This copy only replays finished checks right now, so new uploads are off. The finished checks open with no wait.';
  if (status.pickup_left <= 0 && status.pairs_left <= 0) return 'No new checks are left today. The finished checks open with no wait.';
  return `${status.pickup_left} pickup ${status.pickup_left === 1 ? 'check' : 'checks'} and ${status.pairs_left} full ${status.pairs_left === 1 ? 'pair' : 'pairs'} left today.`;
}

// The upload screen: both drop zones, the pickup-only and plate choices, the counter and the precomputed sample.
export function UploadPage({ status }: { status: StatusResponse | null }) {
  const [kind, setKind] = useState<Kind>('pair');
  const [keepPlate, setKeepPlate] = useState(false);
  const [run, setRun] = useState<RunManifest | null>(null);
  const [uploads, setUploads] = useState<Partial<Record<WalkRole, UploadUi>>>({});
  const [error, setError] = useState<string | null>(null);
  const [sampleNote, setSampleNote] = useState<string | null>(null);
  const creating = useRef<Promise<RunManifest> | null>(null);
  const uploadIds = useRef<Partial<Record<WalkRole, string>>>({});
  const attempts = useRef<Record<WalkRole, number>>({ pickup: 0, return: 0 });

  const ensureRun = (): Promise<RunManifest> => {
    if (run) return Promise.resolve(run);
    creating.current ??= api
      .createRun({ mode: 'car', kind, keep_plate: keepPlate })
      .then((r) => {
        setRun(r);
        return r;
      })
      .catch((err: unknown) => {
        creating.current = null;
        throw err;
      });
    return creating.current;
  };

  // One attempt. A retry keeps the upload id, so the server's chunks count and finalize names only what is missing.
  const send = async (role: WalkRole, file: File, resume: boolean) => {
    setError(null);
    const attempt = ++attempts.current[role];
    // PUTs still in flight from an earlier attempt report late; only the newest attempt writes the card.
    const patch = (fn: (u: UploadUi) => UploadUi) =>
      setUploads((prev) => {
        const current = prev[role];
        return current && attempts.current[role] === attempt ? { ...prev, [role]: fn(current) } : prev;
      });
    const uploadId = uploadIds.current[role] ?? newUploadId();
    uploadIds.current[role] = uploadId;
    setUploads((prev) => ({ ...prev, [role]: { file, progress: resume ? (prev[role]?.progress ?? null) : null, record: null, sha256Browser: null, error: null, mismatch: null } }));
    try {
      const r = await ensureRun();
      const result = await uploadFile(file, {
        runId: r.run_id,
        role,
        keepPlate,
        worker: sharedHashWorker(),
        uploadId,
        resume,
        limits: limitsOf(r),
        onProgress: (progress) => patch((u) => ({ ...u, progress })),
      });
      patch((u) => ({ ...u, record: result.record, sha256Browser: result.sha256Browser }));
    } catch (err) {
      if (err instanceof HashMismatch) {
        delete uploadIds.current[role]; // the server deleted those bytes: the next file starts a fresh upload
        patch((u) => ({ ...u, mismatch: { browser: err.sha256Browser, server: err.sha256Server } }));
      } else if (err instanceof ApiError && err.status === 429) {
        const b = (err.body ?? {}) as { reason?: string; pickup_left?: number; pairs_left?: number };
        setError(`${b.reason ?? 'Live runs are paused'}. ${b.pickup_left ?? 0} pickup checks and ${b.pairs_left ?? 0} full pairs left today. The finished checks still open.`);
        delete uploadIds.current[role];
        setUploads((prev) => ({ ...prev, [role]: undefined }));
      } else {
        patch((u) => ({ ...u, error: err instanceof Error ? err.message : String(err) }));
      }
    }
  };

  // A file picked or dropped is a fresh upload; only the resume button reuses the chunks already sent.
  const onFile = (role: WalkRole, file: File) => {
    delete uploadIds.current[role];
    void send(role, file, false);
  };

  const onRetry = (role: WalkRole) => {
    const u = uploads[role];
    if (!u) return;
    const started = (u.progress?.sentChunks ?? 0) > 0 || u.progress?.sha256 != null;
    void send(role, u.file, started);
  };

  const roles: WalkRole[] = kind === 'pair' ? ['pickup', 'return'] : ['pickup'];
  const complete = run !== null && roles.every((r) => uploads[r]?.record);
  const uploading = roles.some((r) => uploadBusy(uploads[r] ?? null));

  useEffect(() => {
    if (!complete || !run) return;
    const handle = setTimeout(() => navigate(`/run/${encodeURIComponent(run.run_id)}`), 1500);
    return () => clearTimeout(handle);
  }, [complete, run]);

  // Closing or reloading the tab mid-upload loses the bytes not yet sent; the browser asks first.
  useEffect(() => {
    if (!uploading) return;
    const hold = (e: BeforeUnloadEvent) => {
      e.preventDefault();
      e.returnValue = '';
    };
    window.addEventListener('beforeunload', hold);
    return () => window.removeEventListener('beforeunload', hold);
  }, [uploading]);

  const openSample = async () => {
    setSampleNote(null);
    try {
      const s = await api.sample();
      navigate(`/run/${encodeURIComponent(s.run_id)}?replay=instant`);
    } catch (err) {
      setSampleNote(err instanceof ApiError && err.status === 404 ? 'There is no finished check on this server yet.' : err instanceof Error ? err.message : String(err));
    }
  };

  const locked = run !== null;
  // The daily counter only gates creating a run. Once this run exists it holds its own slot, and the counter (which
  // counts that reservation) must not lock its second walk or a retry.
  const paused = !run && Boolean(status && (status.paused || (kind === 'pair' ? status.pairs_left <= 0 : status.pickup_left <= 0)));
  const zoneDisabled = (u: UploadUi | undefined) => paused || uploadBusy(u ?? null) || Boolean(u?.record);

  return (
    <div className="upload">
      <h1>Add your two videos</h1>
      <p className="upload-lead">
        One slow lap of the car at pickup and one at return, about 3 seconds per panel. Your phone fingerprints each file while it uploads and the server seals
        it on arrival, before any model sees it.
      </p>

      <p className={`counter-line${paused ? ' paused' : ''}`}>{counterText(status)}</p>

      <div className="dropzones">
        <Uploader
          role="pickup"
          step="Step 1"
          title="Pickup video"
          hint="Drop the video here or tap to choose."
          disabled={zoneDisabled(uploads.pickup)}
          state={uploads.pickup ?? null}
          onFile={(f) => onFile('pickup', f)}
          onRetry={() => onRetry('pickup')}
        />
        {kind === 'pair' ? (
          <Uploader
            role="return"
            step="Step 2"
            title="Return video"
            hint="The same car at return, filmed the same way."
            disabled={zoneDisabled(uploads.return)}
            state={uploads.return ?? null}
            onFile={(f) => onFile('return', f)}
            onRetry={() => onRetry('return')}
          />
        ) : null}
      </div>

      <fieldset className="home-choice" disabled={locked}>
        <legend className="sr-only">options</legend>
        <label>
          <input type="checkbox" checked={kind === 'pickup_check'} onChange={(e) => setKind(e.target.checked ? 'pickup_check' : 'pair')} />
          I am at pickup: seal only the pickup video now
        </label>
        <label>
          <input type="checkbox" checked={keepPlate} onChange={(e) => setKeepPlate(e.target.checked)} />
          This is the car I rented: keep the plate on my sheet
        </label>
      </fieldset>
      {kind === 'pickup_check' ? (
        <p className="small muted">Good call. A seal made before you drive off shows the car as it was then. A seal made after the return only proves the file existed then.</p>
      ) : null}

      {error ? <p className="error-line">{error}</p> : null}
      {run && !complete ? (
        // Leaving now would strand the walk still to send: the run page cannot take an upload.
        <p className="small upload-next">
          {uploading ? 'Keep this tab open until the upload finishes.' : 'The check starts once every video is in.'} The link to your result appears here then.
        </p>
      ) : null}
      {run && complete ? (
        <div className="upload-next">
          <Link to={`/run/${encodeURIComponent(run.run_id)}`} className="cds--btn cds--btn--primary">
            Open the result
          </Link>
          <p className="small muted">
            Opening now. The result page is your receipt: close the tab and come back to it any time. Run <span className="mono">{run.run_id}</span>.
          </p>
        </div>
      ) : null}

      <div className="sample-row">
        <button type="button" className="cds--btn cds--btn--tertiary" onClick={() => void openSample()}>
          See a finished check
        </button>
        <p className="small muted">A recorded check, replayed. No model is called, and a badge on screen says it is a replay.</p>
        {sampleNote ? <p className="small">{sampleNote}</p> : null}
      </div>

      <div className="home-foot">
        <p className="small">
          <Link to="/claim">I have a claim photo</Link>: place the photo from a damage bill against the pickup walk of a finished check.
        </p>
        <p className="privacy-line">{PRIVACY_BEFORE}</p>
        <p className="privacy-line">
          Faces are always blurred. The plate is blurred unless you tick the box above; then it stays only in your run's sheet frames and PDF, served only at your run
          address and never on the sample or kiosk screens. Runs are deleted after 24 hours.
        </p>
      </div>
    </div>
  );
}
