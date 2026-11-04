import type { Box4, Keyframe } from '../contracts/types.ts';
import { useFileUrl } from '../lib/files.tsx';
import { formatClock } from '../lib/format.ts';
import { panelName } from '../lib/panels.ts';

export type BoxTone = 'new' | 'eye' | 'plain' | 'spot';

export interface FrameBox {
  box: Box4;
  tone: BoxTone; // spot: the other walk's mark carried into this frame, dashed
}

// The part of the crop a frame shows: around the boxed marks, about factor times the mark's size and at least minSide
// wide, at the aspect given (height over width), inside the crop. No box: the whole crop.
export function zoomWindow(w: number, h: number, boxes: Box4[], zoom: boolean, minSide = 320, factor = 6, aspect = 3 / 4): Box4 {
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

const pct = (v: number, of: number) => `${((100 * v) / of).toFixed(3)}%`;

// One walk's crop of the panel, zoomed around the boxes (at least 280 crop pixels wide), 3:2. The boxes are drawn over
// the picture in screen pixels, at least 56 px a side (18 px on a thumbnail) and padded, so a speck a few pixels wide
// still gets a box you can see. thumb: the small list image, cropped closer, 4:3, with a thin box. whole: the whole
// crop with the boxes on it, for a claim's pickup frame, where the panel matters more than the spot.
export function Frame({ kf, role, boxes, thumb = false, whole = false }: { kf: Keyframe | undefined; role: string; boxes: FrameBox[]; thumb?: boolean; whole?: boolean }) {
  const url = useFileUrl();
  const walk = role === 'pickup' ? 'Pickup' : 'Return';
  if (!kf?.crop)
    return thumb ? (
      <span className="frame-thumb empty" aria-hidden="true" />
    ) : (
      <figure className="frame empty">
        <div className="frame-view">No clear view of this panel in the {role} video</div>
      </figure>
    );
  const { w, h, path } = kf.crop;
  const zoomOn = boxes.map((b) => b.box);
  const [x, y, vw, vh] = thumb ? zoomWindow(w, h, zoomOn, true, 120, 3) : zoomWindow(w, h, zoomOn, !whole, 280, 7, 2 / 3);
  const pad = Math.max(4, vw / 90);
  const min = thumb ? 18 : 56;
  // nothing to zoom on: the whole crop, letterboxed into the 3:2 frame so both walks stand the same height
  const fit = !thumb && !whole && !zoomOn.length;
  const view = (
    <>
      <svg viewBox={`${x} ${y} ${vw} ${vh}`} preserveAspectRatio={fit ? 'xMidYMid meet' : 'xMidYMid slice'} aria-hidden="true">
        <image href={url(path)} width={w} height={h} />
      </svg>
      {boxes.map((b, i) => {
        const [x0, y0, x1, y1] = b.box;
        return (
          <span
            key={i}
            className={`frame-box ${b.tone}`}
            style={{
              left: pct((x0 + x1) / 2 - x, vw),
              top: pct((y0 + y1) / 2 - y, vh),
              width: `max(${min}px, ${pct(x1 - x0 + 2 * pad, vw)})`,
              height: `max(${min}px, ${pct(y1 - y0 + 2 * pad, vh)})`,
            }}
          />
        );
      })}
    </>
  );
  if (thumb)
    return (
      <span className="frame-thumb" style={{ aspectRatio: `${vw} / ${vh}` }} aria-hidden="true">
        {view}
      </span>
    );
  const marked = boxes.some((b) => b.tone !== 'spot');
  const spot = boxes.some((b) => b.tone === 'spot');
  return (
    <figure className="frame">
      <div
        className="frame-view"
        style={{ aspectRatio: fit ? '3 / 2' : `${vw} / ${vh}` }}
        role="img"
        aria-label={`${panelName(kf.panel_id)} in the ${role} video at ${formatClock(kf.pts)}${marked ? ', the mark boxed' : spot ? ', the same spot outlined' : ''}`}
      >
        {view}
      </div>
      <figcaption>
        {walk} <span className="mono">{formatClock(kf.pts)}</span>
      </figcaption>
    </figure>
  );
}
