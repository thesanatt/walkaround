import type { Keyframe, LookRow, Mark, Panel, PanelMatch, RunManifest, Verdict, VideoRecord } from '../contracts/types.ts';
import { useFileUrl } from '../lib/files.tsx';
import { drawnPairs, offPaneMarks, paneKeyframes, panelFocus } from '../lib/focus.ts';
import { daysBetween, formatClock, formatDateTime, formatDay, formatTime, minutesBetween } from '../lib/format.ts';
import { keyframePairLine, markLabel, sizeLine, wallClock } from '../lib/lines.ts';
import { lookFor, loweredByLook } from '../lib/looks.ts';
import { colorForPanel, familyNoun, hexToUnit, panelName, VERDICT_LABEL } from '../lib/panels.ts';
import { ruleLabel, ruleTitle } from '../lib/rules.ts';
import type { RoleMedia } from '../state/types.ts';
import { SecondLookLine } from './SecondLookLine.tsx';

const GAP = 40;

export interface MatchPaneProps {
  panelId: string | null;
  match: PanelMatch | undefined;
  panel: Panel | undefined;
  keyframes: Keyframe[];
  marks: Mark[];
  verdicts: Verdict[];
  videos: Record<string, VideoRecord>;
  media: Record<string, RoleMedia>;
  run: RunManifest | null;
  looks?: LookRow[]; // the second looks, printed under the verdicts they touched
  aligned?: boolean; // the match cut: the return crop shifts so the focus pair's return mark sits level with its pickup mark
  showCards?: boolean;
  showDates?: boolean;
  showRefusal?: boolean;
  moment?: boolean; // the video's moment: no video times or units on the pane, the card is the label and two dates
}

interface Sizes {
  label: number;
  title: number;
  stroke: number;
  join: number;
  dot: number;
}

// Everything on the pane is in crop pixels; sizes follow the pane width so text reads the same at any display size.
function sizesFor(width: number): Sizes {
  return { label: Math.round(width / 72), title: Math.round(width / 60), stroke: width / 640, join: width / 520, dot: width / 380 };
}

function MarkGlyph({ m, ox, oy, color, z }: { m: Mark; ox: number; oy: number; color: string; z: Sizes }) {
  const b = m.bbox_crop;
  const p = m.point_crop;
  return (
    <g className="mark-glyph">
      {b ? <rect x={b[0] + ox} y={b[1] + oy} width={b[2] - b[0]} height={b[3] - b[1]} fill="none" stroke={color} strokeWidth={z.stroke} /> : null}
      {p ? <circle cx={p[0] + ox} cy={p[1] + oy} r={z.dot} fill={color} stroke="#0b0d0c" strokeWidth={z.stroke / 2} /> : null}
      {b ? (
        <text className="pane-label" x={b[0] + ox} y={Math.max(z.label, b[1] + oy - z.label / 3)} style={{ fontSize: z.label, strokeWidth: z.label / 5 }}>
          {markLabel(m)}
        </text>
      ) : null}
    </g>
  );
}

export function VerdictCard({ v, marks, panel, looks = [] }: { v: Verdict; marks: Map<string, Mark>; panel: Panel | undefined; looks?: LookRow[] }) {
  const described = v.mark_ids.map((id) => {
    const m = marks.get(id);
    return m ? `${m.role}: ${m.type}, ${m.size_class}${m.description ? `, "${m.description}"` : ''}${m.source === 'narration_seed' ? ' (seeded from your words)' : ''}` : id;
  });
  const look = lookFor(v, looks);
  const ids = [v.rule_id, ...v.trace.rule_ids.filter((r) => r !== v.rule_id)];
  const windows = panel?.per_role['pickup']?.windows ?? [];
  const windowLine =
    v.label === 'new_or_unseen' && windows.length ? `This panel was filmed at pickup ${windows.map(([a, b]) => `${formatClock(a)} to ${formatClock(b)}`).join(' and ')}.` : null;
  const size = sizeLine(v.size);
  return (
    <div className={`verdict-card ${v.label}`}>
      <p className="verdict-head">
        <span className={`label ${v.label}`}>{VERDICT_LABEL[v.label] ?? v.label}</span>
        <span className="mono small rule-ids">
          {ids.map((id) => (
            <abbr key={id} title={ruleTitle(id)}>
              {id}
            </abbr>
          ))}
        </span>
      </p>
      {v.distance_units !== null ? (
        <p>
          {v.distance_units.toFixed(2)} units apart
          {v.distance_cm_est !== null ? ` (about ${v.distance_cm_est.toFixed(0)} cm on this ${familyNoun(v.panel_id)})` : ''}
        </p>
      ) : null}
      <p className="small muted">{described.join('; ')}</p>
      {windowLine ? <p className="small">{windowLine}</p> : null}
      {size ? <p className="small">{size}</p> : null}
      {v.view_note ? <p className="small">{v.view_note}</p> : null}
      {v.type_note ? <p className="small">{v.type_note}</p> : null}
      {v.counter_note ? <p className="small">{v.counter_note}</p> : null}
      {look ? <SecondLookLine look={look} lowered={loweredByLook(v, look)} /> : null}
    </div>
  );
}

// G2's flags in words: " (glare on the return frame)". G2 is retired (10/31); runs stored before then still name it.
export function conditionsText(kfA: Keyframe | undefined, kfB: Keyframe | undefined): string {
  const flags = (kf: Keyframe | undefined) => (kf ? (['wet', 'dirty', 'glare', 'dark'] as const).filter((f) => kf.conditions[f]) : []);
  const parts = [
    ...(flags(kfA).length ? [`${flags(kfA).join(' and ')} on the pickup frame`] : []),
    ...(flags(kfB).length ? [`${flags(kfB).join(' and ')} on the return frame`] : []),
  ];
  return parts.length ? ` (${parts.join(', ')})` : '';
}

// The gap between the two camera clocks: "4 days later", or "same day, 45 minutes later" for one day. A return clock
// set before the pickup clock says "earlier"; it is printed as the cameras wrote it, never corrected.
export function gapText(a: string | null, b: string | null): string | null {
  const days = daysBetween(a, b);
  if (days === null) return null;
  if (days === 0) {
    const minutes = minutesBetween(a, b) ?? 0;
    const m = Math.abs(minutes);
    return `same day, ${m} ${m === 1 ? 'minute' : 'minutes'} ${minutes < 0 ? 'earlier' : 'later'}`;
  }
  const d = Math.abs(days);
  return `${d} ${d === 1 ? 'day' : 'days'} ${days < 0 ? 'earlier' : 'later'}`;
}

function sealedText(run: RunManifest | null): string {
  const sealed = run?.clocks.sealed_minutes_after_camera ?? null;
  if (sealed === null) return 'the seal clock appears once the seal is written';
  if (typeof sealed === 'number') return `sealed ${sealed} ${sealed === 1 ? 'minute' : 'minutes'} after the camera time`;
  return `seal: ${sealed}`;
}

// The two dates: the pickup camera time with how soon it was sealed, beside the return camera time and the gap. As
// the sheet prints them: no camera time, no wall clock and no gap (the receipt time never stands in).
export function DatesCard({ kfA, kfB, videos, media, run, stacked = false }: { kfA: Keyframe; kfB: Keyframe; videos: Record<string, VideoRecord>; media: Record<string, RoleMedia>; run: RunManifest | null; stacked?: boolean }) {
  const a = wallClock(media['pickup']?.ffprobe, videos['pickup'], kfA.pts);
  const b = wallClock(media['return']?.ffprobe, videos['return'], kfB.pts);
  const gap = a && b ? gapText(a, b) : null;
  return (
    <div className={`dates-card ${stacked ? 'stacked' : ''}`}>
      <p className="sr-only">
        pickup {a ? formatDateTime(a) : 'camera time not present'}; return {b ? formatDateTime(b) : 'camera time not present'}
        {gap ? ` (${gap})` : ''}
      </p>
      <div className="date-col" aria-hidden="true">
        <span className="date-role">pickup</span>
        {a ? (
          <>
            <span className="date-day mono">{formatDay(a)}</span>
            <span className="date-time mono">{formatTime(a)}</span>
          </>
        ) : (
          <span className="date-day">camera time not present</span>
        )}
        <span className="date-note">{sealedText(run)}</span>
      </div>
      <div className="date-gap" aria-hidden="true">
        <span className="date-rule" />
        {gap ? <span className="date-gap-text">{gap}</span> : null}
      </div>
      <div className="date-col" aria-hidden="true">
        <span className="date-role">return</span>
        {b ? (
          <>
            <span className="date-day mono">{formatDay(b)}</span>
            <span className="date-time mono">{formatTime(b)}</span>
          </>
        ) : (
          <span className="date-day">camera time not present</span>
        )}
        <span className="date-note">{a && b ? '' : 'a walk whose camera wrote no time gets no wall clock, here or on the sheet'}</span>
      </div>
    </div>
  );
}

// Pickup keyframe left, return keyframe right, on one SVG in crop pixels: the SAM panel mask tinted underneath, the
// Spark boxes on top, one line per joined pair whose marks are both on the frames shown. Code wrote the labels below;
// the pane only draws them.
export function MatchPane(props: MatchPaneProps) {
  const { panelId, match, panel, keyframes, marks, verdicts, videos, media, run, looks = [], aligned = false, showCards = true, showDates = true, showRefusal = true, moment = false } = props;
  const url = useFileUrl();
  if (!panelId) return <div className="empty">Pick a panel to see its two frames.</div>;
  const markById = new Map(marks.map((m) => [m.mark_id, m]));
  const focus = panelFocus(panelId, match, verdicts);
  const { a: kfA, b: kfB } = paneKeyframes(panelId, match, focus, markById, keyframes);
  if (!kfA && !kfB) return <div className="empty">No keyframe for the {panelName(panelId)} in either walk yet.</div>;

  const cropA = kfA?.crop ?? null;
  const cropB = kfB?.crop ?? null;
  const wA = cropA?.w ?? 1280;
  const hA = cropA?.h ?? 860;
  const wB = cropB?.w ?? 1280;
  const hB = cropB?.h ?? 860;
  const pairs = drawnPairs(match, markById, kfA, kfB);
  const lead = focus.pair && pairs.includes(focus.pair) ? focus.pair : null;
  const pA = lead ? (markById.get(lead.pickup_mark)?.point_crop ?? null) : null;
  const pB = lead ? (markById.get(lead.return_mark)?.point_crop ?? null) : null;
  const shift: [number, number] = aligned && pA && pB ? [pA[0] - pB[0], pA[1] - pB[1]] : [0, 0];
  const ox = wA + GAP + shift[0];
  const oy = shift[1];
  const W = wA + GAP + wB;
  const H = Math.max(hA, hB);
  const marksA = kfA ? marks.filter((m) => m.kf_id === kfA.kf_id) : [];
  const marksB = kfB ? marks.filter((m) => m.kf_id === kfB.kf_id) : [];
  const hidden = offPaneMarks(panelId, marks, kfA, kfB);
  const color = colorForPanel(panelId);
  const [r, g, b] = hexToUnit(color);
  const tint = `tint-${panelId}`;
  const clip = `clip-${panelId}`;
  const panelVerdicts = verdicts.filter((v) => v.panel_id === panelId);
  const z = sizesFor(W);

  return (
    <div className="match-pane">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${panelName(panelId)}: pickup frame left, return frame right`}>
        <defs>
          <clipPath id={clip}>
            <rect x={wA + GAP} y={0} width={wB} height={H} />
          </clipPath>
          <filter id={tint} colorInterpolationFilters="sRGB">
            <feColorMatrix type="matrix" values={`0 0 0 0 ${r} 0 0 0 0 ${g} 0 0 0 0 ${b} 0.333 0.333 0.333 0 0`} />
          </filter>
        </defs>
        {cropA ? <image href={url(cropA.path)} x={0} y={0} width={wA} height={hA} /> : <rect x={0} y={0} width={wA} height={hA} className="pane-empty" />}
        {kfA?.mask_png ? <image href={url(kfA.mask_png)} x={0} y={0} width={wA} height={hA} filter={`url(#${tint})`} opacity={0.2} /> : null}
        <g clipPath={`url(#${clip})`}>
          {cropB ? <image href={url(cropB.path)} x={ox} y={oy} width={wB} height={hB} /> : <rect x={wA + GAP} y={0} width={wB} height={hB} className="pane-empty" />}
          {kfB?.mask_png ? <image href={url(kfB.mask_png)} x={ox} y={oy} width={wB} height={hB} filter={`url(#${tint})`} opacity={0.2} /> : null}
          {marksB.map((m) => (
            <MarkGlyph key={m.mark_id} m={m} ox={ox} oy={oy} color={color} z={z} />
          ))}
        </g>
        {marksA.map((m) => (
          <MarkGlyph key={m.mark_id} m={m} ox={0} oy={0} color={color} z={z} />
        ))}
        {pairs.map((pair) => {
          const a = markById.get(pair.pickup_mark)?.point_crop;
          const r2 = markById.get(pair.return_mark)?.point_crop;
          if (!a || !r2) return null;
          return <line key={`${pair.pickup_mark}-${pair.return_mark}`} className="join-line" x1={a[0]} y1={a[1]} x2={r2[0] + ox} y2={r2[1] + oy} stroke={color} strokeWidth={z.join} pathLength={1} />;
        })}
        <text className="pane-title" x={z.title * 0.6} y={z.title * 1.4} style={{ fontSize: z.title, strokeWidth: z.title / 5 }}>
          {kfA ? (moment ? 'pickup' : `pickup, ${formatClock(kfA.pts)}`) : 'pickup: no keyframe'}
        </text>
        <text className="pane-title" x={wA + GAP + z.title * 0.6} y={z.title * 1.4} style={{ fontSize: z.title, strokeWidth: z.title / 5 }}>
          {kfB ? (moment ? 'return' : `return, ${formatClock(kfB.pts)}`) : 'return: no keyframe'}
        </text>
      </svg>
      {!match ? <p className="empty">The comparison for the {panelName(panelId)} comes with the match step. Code writes every label.</p> : null}
      {showRefusal && match?.state === 'not_comparable' ? (
        <p className="refusal">
          Not compared: {match.reasons.map((id) => `${id} ${ruleLabel(id)}${id === 'G2' ? conditionsText(kfA, kfB) : ''}`).join('; ')}. Both best frames are shown so you can look yourself.
          {match.reshape?.called ? ' The outline changed in place, so this panel needs your eye (L7).' : ''}
        </p>
      ) : null}
      {match?.state === 'comparable' && !match.pairs.length && !panelVerdicts.length ? <p className="small muted">Compared: no marks on either frame.</p> : null}
      {!moment && keyframePairLine(match) ? <p className="small muted">How it was checked: {keyframePairLine(match)}</p> : null}
      {!moment && hidden.length ? (
        <p className="small muted">
          {hidden.length === 1 ? 'One mark of this panel was' : `${hidden.length} marks of this panel were`} found on another frame of its walk:{' '}
          {hidden.length === 1 ? 'it is' : 'they are'} drawn on that frame's own crop in the crops section, not on these two frames.
        </p>
      ) : null}
      {moment && showCards && kfA && kfB ? (
        <div className="moment-card">
          {focus.verdict ? <span className={`label ${focus.verdict.label}`}>{VERDICT_LABEL[focus.verdict.label]}</span> : null}
          <DatesCard kfA={kfA} kfB={kfB} videos={videos} media={media} run={run} />
        </div>
      ) : null}
      {!moment && showCards && panelVerdicts.length ? (
        <div className="verdict-cards">
          {panelVerdicts.map((v) => (
            <VerdictCard key={v.verdict_id} v={v} marks={markById} panel={panel} looks={looks} />
          ))}
        </div>
      ) : null}
      {!moment && showDates && match?.state === 'comparable' && kfA && kfB ? <DatesCard kfA={kfA} kfB={kfB} videos={videos} media={media} run={run} /> : null}
    </div>
  );
}
