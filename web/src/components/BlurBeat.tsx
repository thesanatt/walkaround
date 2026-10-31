import type { Keyframe } from '../contracts/types.ts';
import { useFileUrl } from '../lib/files.tsx';
import { boxSrcToCrop } from '../lib/geometry.ts';
import { panelName } from '../lib/panels.ts';

// A stored keyframe whose plate region is already pixelated; the plate and face boxes outline for 400 ms and fade.
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
    <figure className="crop blur-beat">
      <div className="crop-frame">
        <img src={url(crop.path)} width={crop.w} height={crop.h} alt={`${panelName(kf.panel_id)} keyframe with plates and faces pixelated`} />
        <svg viewBox={`0 0 ${crop.w} ${crop.h}`} aria-hidden="true">
          {boxes.map((b, i) => (
            <rect key={i} className="plate-outline" x={b[0]} y={b[1]} width={b[2] - b[0]} height={b[3] - b[1]} />
          ))}
        </svg>
      </div>
      <figcaption>
        Blurred before storage and before Spark: {plates} plate {plates === 1 ? 'region' : 'regions'}, {faces} face {faces === 1 ? 'region' : 'regions'}.
        {plates === 0 ? ' No plate was detected in this frame, so nothing was blurred for that reason.' : ''}
      </figcaption>
    </figure>
  );
}
