import { useEffect, useState, type FormEvent } from 'react';

import { api, ApiError } from '../api/client.ts';
import type { Claim, Coverage, Keyframe, Mark, Panel, PanelId, VideoRecord } from '../contracts/types.ts';
import {
  claimCandidates,
  claimPhotoPath,
  claimPickupKeyframe,
  claimWorkSize,
  effectiveExit,
  EXIT_WORDS,
  exitTitle,
  pickupCoverage,
  pickupSideName,
  PICKUP_SIDE,
  placementLine,
  samVoteLine,
  sideLine,
  sparkNamedLine,
} from '../lib/claim.ts';
import { RunFiles, useFileUrl } from '../lib/files.tsx';
import { joined } from '../lib/focus.ts';
import { formatClock, formatDateTime } from '../lib/format.ts';
import { markLabel, sizeLine, wallClock } from '../lib/lines.ts';
import { colorForPanel, COVERAGE_LABEL, panelName } from '../lib/panels.ts';
import { ruleTitle } from '../lib/rules.ts';

// What the claim view reads from the run beside the claim itself, for the dated pickup frame: the pickup side is the
// pickup walk and its top-up clips (claim.PICKUP_SIDE), each with its own video record for the wall clock.
export interface ClaimContext {
  keyframes: Keyframe[];
  panels: Panel[];
  marks: Mark[];
  coverage: Coverage | null;
  videos: Record<string, VideoRecord>;
}

export function useClaimContext(runId: string | null): ClaimContext | null {
  const [ctx, setCtx] = useState<ClaimContext | null>(null);
  useEffect(() => {
    setCtx(null);
    if (!runId) return;
    let live = true;
    const soft = <T,>(p: Promise<T>) => p.catch(() => null);
    void (async () => {
      const [keyframes, panels, marks, coverage] = await Promise.all([
        soft(api.file<Keyframe[]>(runId, 'derived/keyframes.json')),
        soft(api.file<Panel[]>(runId, 'derived/panels.json')),
        soft(api.file<Mark[]>(runId, 'derived/marks.json')),
        soft(api.file<Coverage>(runId, 'derived/coverage.json')),
      ]);
      const roles = ['pickup', ...Object.keys(coverage?.per_role ?? {}).filter((r) => r !== 'pickup' && PICKUP_SIDE.test(r))];
      const records = await Promise.all(roles.map((r) => soft(api.video(runId, r))));
      const videos: Record<string, VideoRecord> = {};
      roles.forEach((r, i) => {
        if (records[i]) videos[r] = records[i]!;
      });
      if (live) setCtx({ keyframes: keyframes ?? [], panels: panels ?? [], marks: marks ?? [], coverage, videos });
    })();
    return () => {
      live = false;
    };
  }, [runId]);
  return ctx;
}

function windowsText(panel: Panel | undefined, role: string): string {
  const w = panel?.per_role[role]?.windows ?? [];
  return w.map(([a, b]) => `${formatClock(a)} to ${formatClock(b)}`).join(' and ');
}

// The pickup frame a claim lands on, as claim.py chose it: match.kf_pair[0] (the matched pickup mark's own keyframe,
// else the panel's best one), then the coverage's best keyframe over the pickup side, then the panel's best-scored
// pickup-side keyframe. The pickup mark is boxed only when code joined it to the claim mark (d under tau_eye_r);
// past that band claim.py lists the claim mark as unmatched (L5) and the nearest mark is no match.
export function claimFrame(claim: Claim, ctx: Pick<ClaimContext, 'keyframes' | 'marks' | 'coverage'>): { kf: Keyframe; mark: Mark | undefined } | null {
  if (!claim.panel_id) return null;
  const pair = claim.match?.pairs[0];
  const pickupMark = pair && joined(pair) ? ctx.marks.find((m) => m.mark_id === pair.pickup_mark) : undefined;
  const used = claim.match?.kf_pair?.[0];
  const kf =
    (used ? ctx.keyframes.find((k) => k.kf_id === used) : undefined) ??
    (pickupMark ? ctx.keyframes.find((k) => k.kf_id === pickupMark.kf_id) : undefined) ??
    claimPickupKeyframe(ctx.keyframes, claim.panel_id, pickupCoverage(ctx.coverage, claim.panel_id)?.best_kf);
  if (!kf) return null;
  return { kf, mark: pickupMark && pickupMark.kf_id === kf.kf_id ? pickupMark : undefined };
}

// The dated pickup frame of the claimed panel with the matched mark boxed, and when it was filmed. The wall clock
// comes from the keyframe's own walk; with no record of that walk loaded, no camera time is claimed either way.
export function ClaimFrame({ claim, ctx }: { claim: Claim; ctx: ClaimContext }) {
  const url = useFileUrl();
  const hit = claimFrame(claim, ctx);
  const crop = hit?.kf.crop;
  if (!claim.panel_id) return null;
  if (!hit || !crop) return <div className="empty">No pickup frame of the {panelName(claim.panel_id)} in this run.</div>;
  const { kf, mark } = hit;
  const panel = ctx.panels.find((p) => p.panel_id === claim.panel_id);
  const coverage = pickupCoverage(ctx.coverage, claim.panel_id);
  const seconds = coverage?.seconds ?? panel?.per_role[kf.role]?.seconds_visible ?? 0;
  const video = ctx.videos[kf.role];
  const when = video ? wallClock(video.ffprobe, video, kf.pts) : null;
  const windows = windowsText(panel, kf.role);
  const state = claim.coverage_state ?? coverage?.state ?? null;
  const where = pickupSideName(kf.role);
  const color = colorForPanel(claim.panel_id);
  const stroke = Math.max(3, crop.w / 360);
  const font = Math.round(crop.w / 34);
  return (
    <figure className="crop claim-frame">
      <div className="crop-frame">
        <img src={url(crop.path)} width={crop.w} height={crop.h} alt={`the ${panelName(claim.panel_id)} in ${where} at ${formatClock(kf.pts)}, plates and faces pixelated`} />
        {mark?.bbox_crop ? (
          <svg viewBox={`0 0 ${crop.w} ${crop.h}`} aria-hidden="true">
            <rect x={mark.bbox_crop[0]} y={mark.bbox_crop[1]} width={mark.bbox_crop[2] - mark.bbox_crop[0]} height={mark.bbox_crop[3] - mark.bbox_crop[1]} fill="none" stroke={color} strokeWidth={stroke} />
            <text className="pane-label" x={mark.bbox_crop[0]} y={Math.max(font, mark.bbox_crop[1] - font / 3)} style={{ fontSize: font, strokeWidth: font / 5 }}>
              {markLabel(mark)}
            </text>
          </svg>
        ) : null}
      </div>
      <figcaption>
        {state ? `${COVERAGE_LABEL[state] ?? state} at pickup` : 'pickup'}
        {state === 'glimpsed' && seconds > 0 ? ` for ${seconds.toFixed(1)} s` : ''}
        {kf.role !== 'pickup' ? `, this frame from ${where} at ${formatClock(kf.pts)}` : ''}
        {video ? (when ? `, ${formatDateTime(when)}` : ', camera time not present') : ''}
        {windows ? `, ${windows}` : ''}
      </figcaption>
    </figure>
  );
}

// The claim photo with the area the claim marks, when one was found. The box is in the working copy's pixels
// (claimWorkSize), the copy claim/<id>.blurred.jpg stores.
export function ClaimPhoto({ claim, src, staged = false, note }: { claim: Claim; src: string | null; staged?: boolean; note?: string }) {
  const roi = claim.roi?.source === 'drawn' ? claim.roi.box : null;
  const [w, h] = claimWorkSize(claim.image);
  // True whether the stand-in is a frame cut from the return walk (licensed footage) or a photo we took ourselves.
  const stagedNote = note ?? "Claim photo: staged for this demo, standing in for the company's photo. Real claims arrive as the company's photo and text; the flow is the same.";
  return (
    <figure className="crop claim-photo">
      <div className="crop-frame">
        {src ? <img src={src} width={w} height={h} alt={staged ? 'the claim photo, staged by us, plates and faces pixelated' : 'your claim photo, plates and faces pixelated'} /> : <div className="walk-placeholder">no photo to show</div>}
        {src && roi ? (
          <svg viewBox={`0 0 ${w} ${h}`} aria-hidden="true">
            <rect x={roi[0]} y={roi[1]} width={roi[2] - roi[0]} height={roi[3] - roi[1]} fill="none" className="claim-roi" strokeWidth={Math.max(2, w / 300)} />
          </svg>
        ) : null}
      </div>
      <figcaption>
        {claim.text ? <span className="claim-words">"{claim.text}"</span> : <span className="muted">no words typed</span>}
        {staged ? <span className="claim-staged">{stagedNote}</span> : null}
      </figcaption>
    </figure>
  );
}

// The facts of a claim in the order the flow found them. upTo limits how many show (the kiosk reveals them in turn).
export function ClaimSteps({ claim, upTo = 99 }: { claim: Claim; upTo?: number }) {
  const size = sizeLine(claim.size);
  const rows: [string, string][] = [
    ['Muse Spark named', sparkNamedLine(claim)],
    ['SAM 3.1 vote', samVoteLine(claim)],
    ['panel', claim.panel_id ? panelName(claim.panel_id) : 'not named'],
    ['side', sideLine(claim)],
    ['at pickup', claim.coverage_state ? COVERAGE_LABEL[claim.coverage_state] ?? claim.coverage_state : 'not looked up'],
    ['placed', placementLine(claim)],
    ['size', size ?? 'size not stated'],
  ];
  return (
    <dl className="claim-steps">
      {rows.slice(0, upTo).map(([k, v]) => (
        <div key={k}>
          <dt>{k}</dt>
          <dd>{v}</dd>
        </div>
      ))}
    </dl>
  );
}

export function ClaimExit({ claim }: { claim: Claim }) {
  const exit = effectiveExit(claim);
  if (exit)
    return (
      <p className={`claim-head exit ${exit}`}>
        <abbr className="mono" title={ruleTitle(exit)}>
          {exitTitle(exit)}
        </abbr>
        <span>{EXIT_WORDS[exit] ?? ''}</span>
      </p>
    );
  if (claim.placement.method !== 'none' || claim.match) return <p className="claim-head placed">Placed against the pickup walk.</p>;
  return <p className="claim-head exit">Not compared. The words under the photos say why.</p>;
}

// What stands beside the claim photo: the dated pickup frame, or why there is none.
function PickupSide({ claim, ctx }: { claim: Claim; ctx: ClaimContext | null }) {
  if (claim.exit === 'E1' || !claim.panel_id) return <div className="empty">No single panel named yet, so no pickup frame to show.</div>;
  if (claim.exit === 'E3') return <div className="empty">The {panelName(claim.panel_id)} was not filmed at pickup, so there is no dated frame to show.</div>;
  if (!ctx) return <p className="muted small">loading the pickup frame</p>;
  return <ClaimFrame claim={claim} ctx={ctx} />;
}

// The one-tap chooser after E2 or E2b: a button per candidate, one button when only one fits (the tap confirms it).
function PanelChooser({ claim, onPick, busy }: { claim: Claim; onPick: (panel: PanelId) => void; busy: boolean }) {
  const exit = effectiveExit(claim);
  if (exit !== 'E2' && exit !== 'E2b') return null;
  const candidates = claimCandidates(claim);
  if (!candidates.length) return <p className="small">No panel to offer from this answer. Name the panel in the words from the bill and send the photo again.</p>;
  return (
    <div className="claim-pick" role="group" aria-label="pick the panel the bill means">
      <span className="small">{candidates.length === 1 ? 'is this the panel the bill means?' : 'which panel does the bill mean?'}</span>
      {candidates.map((p) => (
        <button key={p} type="button" className="btn small" disabled={busy} onClick={() => onPick(p)}>
          {panelName(p)}
        </button>
      ))}
    </div>
  );
}

// The answer to a claim. The photo shown is the served, pixelated copy, never the file picked on this device.
export function ClaimResult({ claim, ctx, onPick, busy = false }: { claim: Claim; ctx: ClaimContext | null; onPick?: (panel: PanelId) => void; busy?: boolean }) {
  const url = useFileUrl();
  const rel = claimPhotoPath(claim);
  return (
    <div className="claim-result">
      <ClaimExit claim={claim} />
      {onPick ? <PanelChooser claim={claim} onPick={onPick} busy={busy} /> : null}
      <div className="claim-pair">
        <div>
          <h3 className="claim-side-title">your claim photo</h3>
          <ClaimPhoto claim={claim} src={rel ? url(rel) : null} />
        </div>
        <div>
          <h3 className="claim-side-title">the pickup walk</h3>
          <PickupSide claim={claim} ctx={ctx} />
        </div>
      </div>
      <div className="claim-facts">
        <ClaimSteps claim={claim} />
        {claim.paragraph ? <p className="claim-paragraph">{claim.paragraph}</p> : null}
        <p className="small muted mono">
          photo {claim.image.w}x{claim.image.h}
          {claim.image.low_res ? ', low resolution' : ''}
          {claim.scanner_style ? ', grayscale like a scanner image' : ''}, sha256 {claim.image.sha256.slice(0, 12)}
        </p>
      </div>
    </div>
  );
}

function errorText(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === 501) return 'Claim photos are not wired on this server yet.';
    if (err.status === 404) return 'There is no run with that id on this server.';
    if (err.status === 403) return 'That run is public, so it takes no claim. Place the claim against your own run.';
    if (err.status === 413) return 'The photo is too large to send.';
    if (err.status >= 500) return `The server failed while placing the claim (HTTP ${err.status}). Try again in a minute.`;
  }
  return err instanceof Error ? err.message : String(err);
}

// Claim-first: the photo and words from a damage bill, placed against a finished run's pickup walk.
export function ClaimPage({ query }: { query: URLSearchParams }) {
  const [runId, setRunId] = useState(query.get('run') ?? '');
  const [text, setText] = useState('');
  const [tire, setTire] = useState('');
  const [image, setImage] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [claim, setClaim] = useState<Claim | null>(null);
  const [placedRun, setPlacedRun] = useState<string | null>(null);
  const ctx = useClaimContext(placedRun);

  const place = async (panel?: PanelId) => {
    if (!image || !runId) return;
    setBusy(true);
    setError(null);
    try {
      const next = await api.claim({ run_id: runId, image, text, tire_code: tire || undefined, panel_id: panel });
      setClaim(next);
      setPlacedRun(runId);
    } catch (err) {
      setError(errorText(err));
    } finally {
      setBusy(false);
    }
  };

  const submit = (e: FormEvent) => {
    e.preventDefault();
    void place();
  };

  return (
    <div className="claim">
      <h1>I have a claim photo</h1>
      <p className="lead">
        Paste the photo and the words from the bill. Muse Spark names the panel, SAM 3.1 image requests check the name, and the dated pickup frame of that panel
        comes back. A size is printed only when a ruler (the plate, or a wheel with its tire size) is in frame, and always with its range.
      </p>
      <form className="form-grid" onSubmit={submit}>
        <label>
          run id (the end of your run page address)
          <input value={runId} onChange={(e) => setRunId(e.target.value.trim())} required spellCheck={false} />
        </label>
        <label>
          the claim photo
          <input type="file" accept="image/*" onChange={(e) => setImage(e.target.files?.[0] ?? null)} required />
        </label>
        <label>
          the words on the bill
          <textarea value={text} onChange={(e) => setText(e.target.value)} rows={3} placeholder="scratch, right rear door" />
        </label>
        <label>
          tire size from the sidewall, only for a wheel mark
          <input value={tire} onChange={(e) => setTire(e.target.value.trim())} placeholder="225/45R17" spellCheck={false} />
        </label>
        <div>
          <button type="submit" className="btn primary" disabled={busy || !image || !runId}>
            {busy ? 'placing the claim' : 'place the claim against the pickup walk'}
          </button>
        </div>
      </form>
      {busy ? <p className="small muted" role="status">Muse Spark is naming the panel and SAM 3.1 is checking the name. This takes up to about 30 seconds.</p> : null}
      {error ? <p className="error-line">{error}</p> : null}
      {claim && placedRun ? (
        <RunFiles runId={placedRun}>
          <ClaimResult claim={claim} ctx={ctx} onPick={(p) => void place(p)} busy={busy} />
        </RunFiles>
      ) : null}
    </div>
  );
}
