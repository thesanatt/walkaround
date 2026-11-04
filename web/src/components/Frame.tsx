import type { Box4, Keyframe } from '../contracts/types.ts';
import { useFileUrl } from '../lib/files.tsx';
import { formatClock } from '../lib/format.ts';
import { panelName } from '../lib/panels.ts';
import { zoomWindow } from './MarkCard.tsx';

export type BoxTone = 'new' | 'eye' | 'plain' | 'spot';

export interface FrameBox {
  box: Box4;
  tone: BoxTone; // spot: the other walk's mark carried into this frame, dashed
}

const pct = (v: number, of: number) => `${((100 * v) / of).toFixed(3)}%`;

// One walk's crop of the panel, zoomed around the boxes, 3:2. The boxes are drawn over the picture in screen pixels,
// at least 56 px a side (18 px on a thumbnail) and padded, so a speck a few pixels wide still gets a box you can see.
// thumb: the small list image, cropped closer, 4:3, with a thin box.
export function Frame({ kf, role, boxes, thumb = false }: { kf: Keyframe | undefined; role: string; boxes: FrameBox[]; thumb?: boolean }) {
  const url = useFileUrl();
  const walk = role === 'pickup' ? 'Pickup' : 'Return';
  if (!kf?.crop)
    return thumb ? (
      <span className="frame-thumb empty" aria-hidden="true" />
    ) : (
      <figure className="frame empty">
        <div className="frame-view">No clear view of this panel in the {role} walk</div>
      </figure>
    );
  const { w, h, path } = kf.crop;
  const [x, y, vw, vh] = thumb ? zoomWindow(w, h, boxes.map((b) => b.box), true, 120, 3) : zoomWindow(w, h, boxes.map((b) => b.box), true, 360, 7, 2 / 3);
  const pad = Math.max(4, vw / 90);
  const min = thumb ? 18 : 56;
  const view = (
    <>
      <svg viewBox={`${x} ${y} ${vw} ${vh}`} preserveAspectRatio="xMidYMid slice" aria-hidden="true">
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
        style={{ aspectRatio: `${vw} / ${vh}` }}
        role="img"
        aria-label={`${panelName(kf.panel_id)} in the ${role} walk at ${formatClock(kf.pts)}${marked ? ', the mark boxed' : spot ? ', the same spot outlined' : ''}`}
      >
        {view}
      </div>
      <figcaption>
        {walk} <span className="mono">{formatClock(kf.pts)}</span>
      </figcaption>
    </figure>
  );
}
