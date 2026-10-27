import { useEffect, useRef, useState } from 'react';

import { api, ApiError } from '../api/client.ts';
import { useStatus } from '../api/useStatus.ts';
import type { RunManifest, WalkRole } from '../contracts/types.ts';
import { Uploader, type UploadUi } from '../components/Uploader.tsx';
import { STORAGE_LINE } from '../components/PrivacyCard.tsx';
import { navigate, Link } from '../lib/router.tsx';
import { createHashWorker, HashMismatch, uploadFile } from '../lib/upload.ts';

type Kind = 'pickup_check' | 'pair';

function rolesFor(kind: Kind): WalkRole[] {
  return kind === 'pair' ? ['pickup', 'return'] : ['pickup'];
}

// The first screen: the pickup drop zone, the precomputed sample, the pair option, the plate choice, the counter.
export function HomePage() {
  const { status } = useStatus();
  const [kind, setKind] = useState<Kind>('pickup_check');
  const [keepPlate, setKeepPlate] = useState(false);
  const [run, setRun] = useState<RunManifest | null>(null);
  const [uploads, setUploads] = useState<Partial<Record<WalkRole, UploadUi>>>({});
  const [error, setError] = useState<string | null>(null);
  const [sampleNote, setSampleNote] = useState<string | null>(null);
  const worker = useRef<Worker | null>(null);
  const creating = useRef<Promise<RunManifest> | null>(null);

  useEffect(() => {
    worker.current = createHashWorker();
    return () => worker.current?.terminate();
  }, []);

  const ensureRun = (): Promise<RunManifest> => {
    if (run) return Promise.resolve(run);
    if (!creating.current) {
      creating.current = api
        .createRun({ mode: 'car', kind, keep_plate: keepPlate })
        .then((r) => {
          setRun(r);
          return r;
        })
        .catch((err: unknown) => {
          creating.current = null;
          throw err;
        });
    }
    return creating.current;
  };

  const patch = (role: WalkRole, fn: (u: UploadUi) => UploadUi) =>
    setUploads((prev) => {
      const current = prev[role];
      return current ? { ...prev, [role]: fn(current) } : prev;
    });

  const onFile = async (role: WalkRole, file: File) => {
    const w = worker.current;
    if (!w) return;
    setError(null);
    setUploads((prev) => ({ ...prev, [role]: { file, progress: null, record: null, sha256Browser: null, error: null, mismatch: null } }));
    try {
      const r = await ensureRun();
      const result = await uploadFile(file, {
        runId: r.run_id,
        role,
        keepPlate,
        worker: w,
        onProgress: (progress) => patch(role, (u) => ({ ...u, progress })),
      });
      patch(role, (u) => ({ ...u, record: result.record, sha256Browser: result.sha256Browser }));
    } catch (err) {
      if (err instanceof HashMismatch) {
        patch(role, (u) => ({ ...u, mismatch: { browser: err.sha256Browser, server: err.sha256Server } }));
      } else if (err instanceof ApiError && err.status === 429) {
        const b = (err.body ?? {}) as { reason?: string; pickup_left?: number; pairs_left?: number };
        setError(`${b.reason ?? 'live runs are paused'}; ${b.pickup_left ?? 0} pickup checks and ${b.pairs_left ?? 0} full pairs left today. The sample pair is precomputed.`);
        setUploads((prev) => {
          const next = { ...prev };
          delete next[role];
          return next;
        });
      } else {
        patch(role, (u) => ({ ...u, error: err instanceof Error ? err.message : String(err) }));
      }
    }
  };

  const roles = rolesFor(kind);
  const complete = run !== null && roles.every((r) => uploads[r]?.record);
  useEffect(() => {
    if (!complete || !run) return;
    const handle = setTimeout(() => navigate(`/run/${encodeURIComponent(run.run_id)}`), 1500);
    return () => clearTimeout(handle);
  }, [complete, run]);

  const openSample = async () => {
    setSampleNote(null);
    try {
      const s = await api.sample();
      navigate(`/run/${encodeURIComponent(s.run_id)}?replay=fixed`);
    } catch (err) {
      setSampleNote(err instanceof ApiError && err.status === 404 ? 'no sample pair on this server yet' : err instanceof Error ? err.message : String(err));
    }
  };

  const locked = run !== null;
  const paused = status?.paused ?? false;

  return (
    <div>
      <h1>Two walkaround videos in, a dated panel-by-panel sheet out.</h1>
      <p className="home-lead">
        Upload the pickup walk before you leave the lot, or within the hour on wifi. A seal made after return proves only that the file existed then.
      </p>

      <div className="home-choice">
        <label>
          <input type="radio" name="kind" checked={kind === 'pickup_check'} disabled={locked} onChange={() => setKind('pickup_check')} />
          pickup check (one video)
        </label>
        <label>
          <input type="radio" name="kind" checked={kind === 'pair'} disabled={locked} onChange={() => setKind('pair')} />
          pickup and return (two videos)
        </label>
        <label>
          <input type="checkbox" checked={keepPlate} disabled={locked} onChange={(e) => setKeepPlate(e.target.checked)} />
          this is the car I rented; keep the plate on my sheet
        </label>
      </div>

      <div className="dropzones">
        <Uploader
          role="pickup"
          title="pickup walk"
          hint="drop the video here or tap to choose; it is hashed on this device while it uploads"
          disabled={paused || Boolean(uploads.pickup)}
          state={uploads.pickup ?? null}
          onFile={(f) => void onFile('pickup', f)}
        />
        {kind === 'pair' ? (
          <Uploader
            role="return"
            title="return walk"
            hint="the same car, filmed the same way, at return"
            disabled={paused || Boolean(uploads.return)}
            state={uploads.return ?? null}
            onFile={(f) => void onFile('return', f)}
          />
        ) : null}
      </div>

      {error ? <p className="error-line">{error}</p> : null}
      {run ? (
        <p className="small">
          run <span className="mono">{run.run_id}</span>: the run URL is your receipt. Close the tab and come back; the page resumes.{' '}
          <Link to={`/run/${encodeURIComponent(run.run_id)}`}>open the run page</Link>
          {complete ? ' (opening now)' : ''}
        </p>
      ) : null}

      <div className="sample-row">
        <div className="row">
          <button type="button" className="btn" onClick={() => void openSample()}>
            open the sample pair (precomputed)
          </button>
          <span className="small muted">a stored run, replayed with fixed delays and a badge that says so; no model call</span>
        </div>
        {sampleNote ? <p className="small muted">{sampleNote}</p> : null}
      </div>

      <p className="counter-line small">
        {status
          ? paused
            ? 'live runs paused; the sample pair is precomputed'
            : `${status.pickup_left} pickup checks and ${status.pairs_left} full pairs left today`
          : 'checking today's live-run count'}
      </p>
      <p className="small">
        <Link to="/claim">I have a claim photo</Link> (place a bill's photo against a finished run)
      </p>
      <p className="privacy-line">{STORAGE_LINE}</p>
      <p className="privacy-line">
        Faces are always blurred. The plate is blurred unless you tick the box above, and then it stays only in your run's sheet frames and PDF, served on your
        run URL, never on a kiosk or sample route. Runs are deleted after 24 hours.
      </p>
    </div>
  );
}
