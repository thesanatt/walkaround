import type { Keyframe, Mark, Verdict } from '../contracts/types.ts';
import { useFileUrl } from '../lib/files.tsx';
import { bestKeyframe } from '../lib/focus.ts';
import { formatClock } from '../lib/format.ts';
import { panelName, VERDICT_LABEL } from '../lib/panels.ts';
import { markWords, whyLine } from '../lib/result.ts';

type Box = [number, number, number, number];

// The part of the crop a card shows: around the boxed marks, about six times the mark's size, 4:3, inside the crop.
// No box: the whole crop.
export function zoomWindow(w: number, h: number, boxes: Box[], zoom: boolean): Box {
  if (!zoom || !boxes.length) return [0, 0, w, h];
  const x0 = Math.min(...boxes.map((b) => b[0]));
  const y0 = Math.min(...boxes.map((b) => b[1]));
  const x1 = Math.max(...boxes.map((b) => b[2]));
  const y1 = Math.max(...boxes.map((b) => b[3]));
  let vw = Math.min(w, Math.max(320, 6 * Math.max(x1 - x0, y1 - y0)));
  let vh = (vw * 3) / 4;
  if (vh > h) {
    vh = h;
    vw = Math.min(w, (vh * 4) / 3);
  }
  const cx = (x0 + x1) / 2;
  const cy = (y0 + y1) / 2;
  const left = Math.min(Math.max(0, cx - vw / 2), w - vw);
  const top = Math.min(Math.max(0, cy - vh / 2), h - vh);
  return [left, top, vw, vh];
}

// One walk's frame of the panel with this mark boxed. The other walk's frame, when it holds no mark of the verdict,
// shows whole and says so.
export function MarkFrame({ kf, marks, role, zoom = true }: { kf: Keyframe | undefined; marks: Mark[]; role: string; zoom?: boolean }) {
  const url = useFileUrl();
  const title = role === 'pickup' ? 'Pickup' : 'Return';
  if (!kf?.crop) return <figure className="mark-frame empty-frame">{title}: no frame of this panel</figure>;
  const boxes = marks.map((m) => m.bbox_crop).filter((b): b is Box => Array.isArray(b));
  const [x, y, vw, vh] = zoomWindow(kf.crop.w, kf.crop.h, boxes, zoom);
  // The box sits a little outside the mark, so the line never covers what it points at.
  const stroke = Math.max(2, vw / 220);
  const pad = stroke * 2 + Math.max(4, vw / 120);
  const what = marks.length ? (vw < kf.crop.w ? ', zoomed in' : '') : ', no matching mark found';
  return (
    <figure className="mark-frame">
      <svg viewBox={`${x} ${y} ${vw} ${vh}`} role="img" aria-label={`${panelName(kf.panel_id)} in the ${role} walk at ${formatClock(kf.pts)}${what}, plates and faces pixelated`}>
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
        {title} at {formatClock(kf.pts)}
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

export function MarkCard({ v, marks, keyframes, roles, onDetails }: { v: Verdict; marks: Map<string, Mark>; keyframes: Keyframe[]; roles: string[]; onDetails?: (panelId: string) => void }) {
  const words = markWords(v, marks);
  return (
    <article className={`mark-card ${v.label}`}>
      <header className="mark-card-head">
        <span className={`label ${v.label}`}>{VERDICT_LABEL[v.label] ?? v.label}</span>
        <h3>
          {panelName(v.panel_id)}
          {words ? <span className="mark-words">: {words}</span> : null}
        </h3>
      </header>
      <div className={`mark-pair mark-pair-${roles.length}`}>
        {roles.map((role) => {
          const { kf, own } = frameFor(v, role, marks, keyframes);
          return <MarkFrame key={role} kf={kf} marks={own} role={role} />;
        })}
      </div>
      <p className="mark-why">
        {whyLine(v)}
        {onDetails ? (
          <>
            {' '}
            <button type="button" className="link-button" onClick={() => onDetails(v.panel_id)}>
              How this was checked
            </button>
          </>
        ) : null}
      </p>
    </article>
  );
}
