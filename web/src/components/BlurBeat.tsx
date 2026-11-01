import type { CSSProperties } from 'react';

import type { Keyframe } from '../contracts/types.ts';
import { useFileUrl } from '../lib/files.tsx';
import { boxSrcToCrop } from '../lib/geometry.ts';
import { panelName } from '../lib/panels.ts';

// A stored keyframe whose plate region is already pixelated; the plate and face boxes outline for 400 ms and fade.
// The outlines and counts are the frame's own image checks (BlurRecord). privacy.blur_keyframe also pixelates the
// plate and face track boxes of the walk at that frame, which the record does not list, so the caption claims no
// more than the checks found.
export function BlurBeat({ kf }: { kf: Keyframe | undefined }) {
  const url = useFileUrl();
  if (!kf?.crop) return <div className="empty">No stored keyframe to show yet.</div>;
  const crop = kf.crop;
  const boxes = [...kf.blur.plate_boxes, ...kf.blur.face_boxes]
    .map((b) => boxSrcToCrop(b, crop))
    .filter((b) => b[2] > 0 && b[3] > 0 && b[0] < crop.w && b[1] < crop.h);
  const plates = kf.blur.plate_boxes.length;
  const faces = kf.blur.face_boxes.length;
  return (
    <figure className="crop blur-beat" style={{ '--crop-w': crop.w, '--crop-h': crop.h } as CSSProperties}>
      <div className="crop-frame">
        <img src={url(crop.path)} width={crop.w} height={crop.h} alt={`${panelName(kf.panel_id)} keyframe with plates and faces pixelated`} />
        <svg viewBox={`0 0 ${crop.w} ${crop.h}`} aria-hidden="true">
          {boxes.map((b, i) => (
            <rect key={i} className="plate-outline" x={b[0]} y={b[1]} width={b[2] - b[0]} height={b[3] - b[1]} />
          ))}
        </svg>
      </div>
      <figcaption>
        This crop was blurred before it was saved or sent to Muse Spark. Outlined: what the plate and face checks found on this frame, {plates} plate{' '}
        {plates === 1 ? 'region' : 'regions'} and {faces} face {faces === 1 ? 'region' : 'regions'}. Any plate or face the video tracks had at this frame was pixelated too.
      </figcaption>
    </figure>
  );
}
