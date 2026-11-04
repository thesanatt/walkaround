import { useEffect, useRef, useState } from 'react';

import { api, ApiError } from '../api/client.ts';
import { Bar } from '../components/Bar.tsx';
import { Slot, uploadBusy, type UploadUi } from '../components/Slot.tsx';
import type { RunManifest, StatusResponse, WalkRole } from '../contracts/types.ts';
import { Link, navigate } from '../lib/router.tsx';
import { HashMismatch, limitsOf, newUploadId, sharedHashWorker, uploadFile } from '../lib/upload.ts';

type Kind = 'pickup_check' | 'pair';

export const exampleUrl = (id: string) => `/run/${encodeURIComponent(id)}?replay=instant`;

// The first sample the server lists (samples/index.json order), for "See an example".
function useExample(): string | null {
  const [id, setId] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    void api
      .samples()
      .then((r) => r.samples[0]?.run_id ?? null)
      .catch(() => api.sample().then((s) => s.run_id))
      .catch(() => null)
      .then((runId) => {
        if (live) setId(runId);
      });
    return () => {
      live = false;
    };
  }, []);
  return id;
}

// Why the slots are off, in one line; null while they take files.
function offLine(status: StatusResponse | null, kind: Kind): string | null {
  if (!status) return null;
  if (status.paused) return 'New checks are off on this copy.';
  if (kind === 'pair' ? status.pairs_left <= 0 : status.pickup_left <= 0) return 'No new checks are left today.';
  return null;
}

// The start screen (/ and /upload): the two videos, one button, and the example.
export function StartPage({ status }: { status: StatusResponse | null }) {
  const [kind, setKind] = useState<Kind>('pair');
  const [keepPlate, setKeepPlate] = useState(false);
  const [run, setRun] = useState<RunManifest | null>(null);
  const [uploads, setUploads] = useState<Partial<Record<WalkRole, UploadUi>>>({});
  const [error, setError] = useState<string | null>(null);
  const example = useExample();
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
    // PUTs still in flight from an earlier attempt report late; only the newest attempt writes the slot.
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
        const b = (err.body ?? {}) as { reason?: string };
        setError(`${b.reason ?? 'Live checks are paused'}. The example still opens.`);
        delete uploadIds.current[role];
        setUploads((prev) => ({ ...prev, [role]: undefined }));
      } else {
        patch((u) => ({ ...u, error: err instanceof Error ? err.message : String(err) }));
      }
    }
  };

  // A file picked or dropped is a fresh upload; only Resume reuses the chunks already sent.
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

  // The daily counter only gates creating a run. Once this run exists it holds its own slot, and the counter (which
  // counts that reservation) must not lock its second walk or a retry.
  const off = run ? null : offLine(status, kind);
  const slotOff = (u: UploadUi | undefined) => off !== null || uploadBusy(u ?? null) || Boolean(u?.record);
  const go = () => run && navigate(`/run/${encodeURIComponent(run.run_id)}`);

  return (
    <div className="start">
      <Bar />
      <h1 className="start-q">What changed between pickup and return?</h1>
      <div className={`slots slots-${roles.length}`}>
        <Slot title="Pickup video" disabled={slotOff(uploads.pickup)} state={uploads.pickup ?? null} onFile={(f) => onFile('pickup', f)} onRetry={() => onRetry('pickup')} />
        {kind === 'pair' ? (
          <Slot title="Return video" disabled={slotOff(uploads.return)} state={uploads.return ?? null} onFile={(f) => onFile('return', f)} onRetry={() => onRetry('return')} />
        ) : null}
      </div>
      {off ? <p className="start-off">{off}</p> : null}
      {error ? <p className="error-line">{error}</p> : null}
      <div className="start-go">
        {off && example ? (
          <Link to={exampleUrl(example)} className="button">
            See an example
          </Link>
        ) : (
          <>
            <button type="button" className="button" disabled={!complete} onClick={go}>
              {kind === 'pair' ? 'Compare' : 'Open the sealed walk'}
            </button>
            {example ? <Link to={exampleUrl(example)}>See an example</Link> : null}
          </>
        )}
      </div>
      {run && !complete && uploading ? <p className="start-note">Keep this tab open until both videos are sealed.</p> : null}
      <details className="start-options">
        <summary>Options</summary>
        <fieldset disabled={run !== null}>
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
      </details>
      <p className="start-privacy">
        Faces are always blurred, and the plate unless you keep it. Runs are deleted after 24 hours.
      </p>
    </div>
  );
}
