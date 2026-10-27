import { routes } from '../api/client.ts';
import type { Keyframe, Mark, Panel, PanelMatch, RunManifest, Verdict, VideoRecord } from '../contracts/types.ts';
import { daysBetween, formatClock, formatDateTime, minutesBetween } from '../lib/format.ts';
import { colorForPanel, panelName, VERDICT_LABEL } from '../lib/panels.ts';
import { ruleLabel, ruleTitle } from '../lib/rules.ts';
import { bestKeyframe } from './CropView.tsx';

const GAP = 40;

export interface MatchPaneProps {
  runId: string;
  panelId: string | null;
  match: PanelMatch | undefined;
  panel: Panel | undefined;
  keyframes: Keyframe[];
  marks: Mark[];
  verdicts: Verdict[];
  videos: Record<string, VideoRecord>;
  run: RunManifest | null;
  // The match cut: translate the return pane so the paired mark lands where the pickup mark sits.
  aligned?: boolean;
  showCards?: boolean;
  showDates?: boolean;
}

export interface WallClock {
  time: string | null;
  source: 'camera' | 'received' | null;
}

// Capture time plus pts: the camera's creation_time when ffprobe found one, else the receipt time.
export function wallClock(video: VideoRecord | undefined, pts: number): WallClock {
  const camera = video?.ffprobe?.creation_time ?? null;
  const hasCamera = camera !== null && camera !== 'not present';
  const base = hasCamera ? camera : (video?.received_at ?? null);
  if (!base) return { time: null, source: null };
  const t = Date.parse(base);
  if (!Number.isFinite(t)) return { time: null, source: null };
  return { time: new Date(t + pts * 1000).toISOString(), source: hasCamera ? 'camera' : 'received' };
}

function familyNoun(panelId: string): string {
  if (panelId.endsWith('_door')) return 'door';
  if (panelId.endsWith('_fender')) return 'fender';
  if (panelId.endsWith('_bumper')) return 'bumper';
  if (panelId.endsWith('_wheel')) return 'wheel';
  return 'panel';
}

function pairedKeyframe(keyframes: Keyframe[], role: 'pickup' | 'return', panelId: string, kfId: string | undefined): Keyframe | undefined {
  if (kfId) {
    const k = keyframes.find((x) => x.kf_id === kfId);
    if (k) return k;
  }
  return bestKeyframe(keyframes, role, panelId);
}

function MarkGlyph({ m, ox, oy, color }: { m: Mark; ox: number; oy: number; color: string }) {
  const b = m.bbox_crop;
  const p = m.point_crop;
  return (
    <g>
      {b ? <rect x={b[0] + ox} y={b[1] + oy} width={b[2] - b[0]} height={b[3] - b[1]} fill="none" stroke={color} strokeWidth={3} /> : null}
      {p ? <circle cx={p[0] + ox} cy={p[1] + oy} r={6} fill={color} /> : null}
      {b ? (
        <text className="pane-title" x={b[0] + ox} y={Math.max(26, b[1] + oy - 8)} style={{ fontSize: 20 }}>
          {`${m.type}, ${m.size_class}`}
        </text>
      ) : null}
    </g>
  );
}

export function VerdictCard({ v, marks, panel, panelId }: { v: Verdict; marks: Map<string, Mark>; panel: Panel | undefined; panelId: string }) {
  const described = v.mark_ids.map((id) => {
    const m = marks.get(id);
    return m ? `${m.role} ${m.type} ${m.size_class}${m.description ? ` (${m.description})` : ''}` : id;
  });
  const ids = [v.rule_id, ...v.trace.rule_ids.filter((r) => r !== v.rule_id)];
  const pickupWindows = panel?.per_role['pickup']?.windows ?? [];
  const windowLine =
    v.label === 'new_or_unseen' && pickupWindows.length
      ? `this panel was filmed at pickup ${pickupWindows.map(([a, b]) => `${formatClock(a)} to ${formatClock(b)}`).join(' and ')}`
      : null;
  return (
    <div className={`verdict-card ${v.label}`}>
      <p>
        <span className={`label ${v.label}`}>{VERDICT_LABEL[v.label] ?? v.label}</span>{' '}
        <span className="mono small">
          {ids.map((id) => (
            <abbr key={id} title={ruleTitle(id)}>
              {id}{' '}
            </abbr>
          ))}
        </span>
      </p>
      {v.distance_units !== null ? (
        <p>
          {v.distance_units.toFixed(2)} units apart
          {v.distance_cm_est !== null ? ` (about ${v.distance_cm_est.toFixed(0)} cm on this ${familyNoun(panelId)})` : ''}
        </p>
      ) : null}
      <p className="small muted">{described.join('; ')}</p>
      {windowLine ? <p className="small">{windowLine}</p> : null}
      {v.size?.refused_reason ? <p className="small muted">size not stated: {v.size.refused_reason}</p> : null}
    </div>
  );
}

export function DatesCard({ kfA, kfB, videos, run }: { kfA: Keyframe; kfB: Keyframe; videos: Record<string, VideoRecord>; run: RunManifest | null }) {
  const a = wallClock(videos['pickup'], kfA.pts);
  const b = wallClock(videos['return'], kfB.pts);
  const days = daysBetween(a.time, b.time);
  const minutes = minutesBetween(a.time, b.time);
  const gap = days === null ? '' : days === 0 ? ` (same day, ${minutes ?? 0} minutes later)` : ` (${days} ${days === 1 ? 'day' : 'days'} later)`;
  const sealed = run?.clocks.sealed_minutes_after_camera ?? null;
  const sourceNote = a.source === 'received' || b.source === 'received' ? ' (receipt time where the camera wrote none)' : '';
  return (
    <div className="dates-card">
      <p>
        pickup {a.time ? formatDateTime(a.time) : 'time not present'}; return {b.time ? formatDateTime(b.time) : 'time not present'}
        {gap}
        {sourceNote}
      </p>
      <p className="small muted">
        {sealed === null ? 'the seal clock is written at upload' : typeof sealed === 'number' ? `pickup sealed ${sealed} minutes after the camera time` : sealed}
      </p>
    </div>
  );
}

// Pickup keyframe left, return keyframe right, Spark boxes on an SVG overlay in crop pixels, one line per pair.
export function MatchPane(props: MatchPaneProps) {
  const { runId, panelId, match, panel, keyframes, marks, verdicts, videos, run, aligned = false, showCards = true, showDates = true } = props;
  if (!panelId) return <div className="empty">Pick a panel to see its two frames.</div>;
  const kfA = pairedKeyframe(keyframes, 'pickup', panelId, match?.kf_pair?.[0]);
  const kfB = pairedKeyframe(keyframes, 'return', panelId, match?.kf_pair?.[1]);
  if (!kfA && !kfB) return <div className="empty">No keyframe for {panelName(panelId)} in either walk yet.</div>;

  const markById = new Map(marks.map((m) => [m.mark_id, m]));
  const cropA = kfA?.crop ?? null;
  const cropB = kfB?.crop ?? null;
  const wA = cropA?.w ?? 640;
  const hA = cropA?.h ?? 480;
  const wB = cropB?.w ?? 640;
  const hB = cropB?.h ?? 480;
  const pairs = match?.state === 'comparable' ? match.pairs : [];
  const first = pairs[0];
  const pA = first ? (markById.get(first.pickup_mark)?.point_crop ?? null) : null;
  const pB = first ? (markById.get(first.return_mark)?.point_crop ?? null) : null;
  const shift: [number, number] = aligned && pA && pB ? [pA[0] - pB[0], pA[1] - pB[1]] : [0, 0];
  const ox = wA + GAP + shift[0];
  const oy = shift[1];
  const W = wA + GAP + wB;
  const H = Math.max(hA, hB);
  const marksA = kfA ? marks.filter((m) => m.kf_id === kfA.kf_id) : [];
  const marksB = kfB ? marks.filter((m) => m.kf_id === kfB.kf_id) : [];
  const color = colorForPanel(panelId);
  const clipId = `clip-${panelId}`;
  const panelVerdicts = verdicts.filter((v) => v.panel_id === panelId);

  return (
    <div className="match-pane">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${panelName(panelId)}: pickup keyframe left, return keyframe right`}>
        <defs>
          <clipPath id={clipId}>
            <rect x={wA + GAP} y={0} width={wB} height={hB} />
          </clipPath>
        </defs>
        {cropA ? <image href={routes.file(runId, cropA.path)} x={0} y={0} width={wA} height={hA} /> : <rect x={0} y={0} width={wA} height={hA} fill="#2a302e" />}
        <g clipPath={`url(#${clipId})`}>
          {cropB ? <image href={routes.file(runId, cropB.path)} x={ox} y={oy} width={wB} height={hB} /> : <rect x={wA + GAP} y={0} width={wB} height={hB} fill="#2a302e" />}
          {marksB.map((m) => (
            <MarkGlyph key={m.mark_id} m={m} ox={ox} oy={oy} color={color} />
          ))}
        </g>
        {marksA.map((m) => (
          <MarkGlyph key={m.mark_id} m={m} ox={0} oy={0} color={color} />
        ))}
        {pairs.map((pair) => {
          const a = markById.get(pair.pickup_mark)?.point_crop;
          const b = markById.get(pair.return_mark)?.point_crop;
          if (!a || !b) return null;
          return <line key={`${pair.pickup_mark}-${pair.return_mark}`} className="join-line" x1={a[0]} y1={a[1]} x2={b[0] + ox} y2={b[1] + oy} stroke={color} pathLength={1} />;
        })}
        <text className="pane-title" x={16} y={36}>
          {kfA ? `pickup, ${formatClock(kfA.pts)}` : 'pickup: no keyframe'}
        </text>
        <text className="pane-title" x={wA + GAP + 16} y={36}>
          {kfB ? `return, ${formatClock(kfB.pts)}` : 'return: no keyframe'}
        </text>
      </svg>
      {!match ? <p className="empty">The match for {panelName(panelId)} comes with match.derived; code writes every label.</p> : null}
      {match?.state === 'not_comparable' ? (
        <p className="refusal">
          Code did not compare {panelName(panelId)}: {match.reasons.map((id) => `${id} ${ruleLabel(id)}`).join('; ')}. Both best frames still show.
        </p>
      ) : null}
      {showCards && panelVerdicts.length ? (
        <div className="verdict-cards">
          {panelVerdicts.map((v) => (
            <VerdictCard key={v.verdict_id} v={v} marks={markById} panel={panel} panelId={panelId} />
          ))}
        </div>
      ) : null}
      {showDates && match?.state === 'comparable' && kfA && kfB ? <DatesCard kfA={kfA} kfB={kfB} videos={videos} run={run} /> : null}
    </div>
  );
}
