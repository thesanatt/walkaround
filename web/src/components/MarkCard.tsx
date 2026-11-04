import type { Keyframe, LookRow, Mark, Verdict } from '../contracts/types.ts';
import { useFileUrl } from '../lib/files.tsx';
import { bestKeyframe } from '../lib/focus.ts';
import { formatClock } from '../lib/format.ts';
import { panelName } from '../lib/panels.ts';
import { creditLine, markWords, whyLine } from '../lib/result.ts';

type Box = [number, number, number, number];

// The part of the crop a card shows: around the boxed marks, about six times the mark's size, 4:3 (or the aspect
// given, height over width), inside the crop. No box: the whole crop.
export function zoomWindow(w: number, h: number, boxes: Box[], zoom: boolean, minSide = 320, factor = 6, aspect = 3 / 4): Box {
  if (!zoom || !boxes.length) return [0, 0, w, h];
  const x0 = Math.min(...boxes.map((b) => b[0]));
  const y0 = Math.min(...boxes.map((b) => b[1]));
  const x1 = Math.max(...boxes.map((b) => b[2]));
  const y1 = Math.max(...boxes.map((b) => b[3]));
  let vw = Math.min(w, Math.max(minSide, factor * Math.max(x1 - x0, y1 - y0)));
  let vh = vw * aspect;
  if (vh > h) {
    vh = h;
    vw = Math.min(w, vh / aspect);
  }
  const cx = (x0 + x1) / 2;
  const cy = (y0 + y1) / 2;
  const left = Math.min(Math.max(0, cx - vw / 2), w - vw);
  const top = Math.min(Math.max(0, cy - vh / 2), h - vh);
  return [left, top, vw, vh];
}

// One walk's frame of the panel with this mark boxed. The other walk's frame, when it holds no mark of the verdict,
// shows whole and says so.
// tight: a thumbnail, cropped closer so the mark, not the plate block beside it, fills the frame.
export function MarkFrame({ kf, marks, role, zoom = true, tight = false }: { kf: Keyframe | undefined; marks: Mark[]; role: string; zoom?: boolean; tight?: boolean }) {
  const url = useFileUrl();
  const title = role === 'pickup' ? 'Pickup' : 'Return';
  if (!kf?.crop) return <figure className="mark-frame empty-frame">No clear view of this panel in the {role} walk</figure>;
  const boxes = marks.map((m) => m.bbox_crop).filter((b): b is Box => Array.isArray(b));
  const [x, y, vw, vh] = tight ? zoomWindow(kf.crop.w, kf.crop.h, boxes, zoom, 140, 4) : zoomWindow(kf.crop.w, kf.crop.h, boxes, zoom);
  // The box sits a little outside the mark, so the line never covers what it points at.
  const stroke = Math.max(2, vw / 220);
  const pad = stroke * 2 + Math.max(4, vw / 120);
  const what = marks.length ? '' : ', no mark here';
  return (
    <figure className="mark-frame">
      <svg viewBox={`${x} ${y} ${vw} ${vh}`} role="img" aria-label={`${panelName(kf.panel_id)} in the ${role} walk at ${formatClock(kf.pts)}${marks.length ? ', the mark boxed' : what}, plates and faces pixelated`}>
        <image href={url(kf.crop.path)} width={kf.crop.w} height={kf.crop.h} />
        {boxes.map((b, i) => {
          const r = { x: b[0] - pad, y: b[1] - pad, width: b[2] - b[0] + 2 * pad, height: b[3] - b[1] + 2 * pad };
          return (
            <g key={i}>
              <rect {...r} className="box-under" strokeWidth={stroke * 2.5} />
              <rect {...r} className="box" strokeWidth={stroke} />
            </g>
          );
        })}
      </svg>
      <figcaption>
        {title} {formatClock(kf.pts)}
        {what}
      </figcaption>
    </figure>
  );
}

// The frame a verdict's mark was found on in one walk, else that walk's best frame of the panel.
function frameFor(v: Verdict, role: string, marks: Map<string, Mark>, keyframes: Keyframe[]): { kf: Keyframe | undefined; own: Mark[] } {
  const own = v.mark_ids.map((id) => marks.get(id)).filter((m): m is Mark => m !== undefined && m.role === role);
  const kfId = own[0]?.kf_id;
  const kf = (kfId ? keyframes.find((k) => k.kf_id === kfId) : undefined) ?? bestKeyframe(keyframes, role, v.panel_id);
  return { kf, own: own.filter((m) => m.kf_id === kf?.kf_id) };
}

export const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

// The word the card leads with. Each label is a word and a colour, never colour alone.
export const CARD_WORD: Record<string, string> = {
  already_there: 'There at pickup',
  new_or_unseen: 'New',
  needs_your_eye: 'Needs your eye',
  not_seen_at_return: 'Not seen at return',
};

export interface MarkCardProps {
  v: Verdict;
  marks: Map<string, Mark>;
  keyframes: Keyframe[];
  roles: string[];
  looks?: LookRow[];
  n?: number; // the pin number on the car map
  compact?: boolean;
  onDetails?: (panelId: string) => void;
}

export function MarkCard({ v, marks, keyframes, roles, looks = [], n, compact = false, onDetails }: MarkCardProps) {
  const words = markWords(v, marks);
  return (
    <article className={`mark-card ${v.label}${compact ? ' compact' : ''}`} id={`card-${v.verdict_id}`}>
      <header className="mark-card-head">
        {n !== undefined ? (
          <span className={`pin-badge ${v.label}`} aria-hidden="true">
            {n}
          </span>
        ) : null}
        <h3>
          <span className={`label ${v.label}`}>{CARD_WORD[v.label] ?? v.label}</span> {cap(panelName(v.panel_id))}
          {words ? <span className="mark-words">: {words}</span> : null}
        </h3>
      </header>
      <div className={`mark-pair mark-pair-${roles.length}`}>
        {roles.map((role) => {
          const { kf, own } = frameFor(v, role, marks, keyframes);
          return <MarkFrame key={role} kf={kf} marks={own} role={role} />;
        })}
      </div>
      <p className="mark-why">{whyLine(v)}</p>
      <p className="mark-credit">
        {creditLine(v, marks, looks)}
        {onDetails ? (
          <>
            {' '}
            <button type="button" className="link-button" onClick={() => onDetails(v.panel_id)}>
              See how
            </button>
          </>
        ) : null}
      </p>
    </article>
  );
}
