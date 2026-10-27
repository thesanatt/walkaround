import { routes } from '../api/client.ts';
import type { Box4, Keyframe } from '../contracts/types.ts';
import { panelName } from '../lib/panels.ts';

function boxToCrop(b: Box4, crop: NonNullable<Keyframe['crop']>): Box4 {
  return [(b[0] - crop.dx) * crop.scale, (b[1] - crop.dy) * crop.scale, (b[2] - crop.dx) * crop.scale, (b[3] - crop.dy) * crop.scale];
}

// A stored keyframe, plate region already pixelated; the plate box outline draws for 400 ms and fades.
export function BlurBeat({ runId, kf }: { runId: string; kf: Keyframe | undefined }) {
  if (!kf || !kf.crop) return <div className="empty">No stored keyframe to show yet.</div>;
  const crop = kf.crop;
  const boxes = [...kf.blur.plate_boxes, ...kf.blur.face_boxes].map((b) => boxToCrop(b, crop)).filter((b) => b[2] > 0 && b[3] > 0 && b[0] < crop.w && b[1] < crop.h);
  return (
    <figure className="crop blur-beat">
      <div className="crop-frame">
        <img src={routes.file(runId, crop.path)} width={crop.w} height={crop.h} alt={`${panelName(kf.panel_id)} keyframe with plates and faces pixelated`} />
        <svg viewBox={`0 0 ${crop.w} ${crop.h}`} aria-hidden="true">
          {boxes.map((b, i) => (
            <rect key={i} className="plate-outline" x={b[0]} y={b[1]} width={b[2] - b[0]} height={b[3] - b[1]} />
          ))}
        </svg>
      </div>
      <figcaption>
        blurred before storage and before Spark: {kf.blur.plate_boxes.length} plate {kf.blur.plate_boxes.length === 1 ? 'box' : 'boxes'}, {kf.blur.face_boxes.length} face{' '}
        {kf.blur.face_boxes.length === 1 ? 'box' : 'boxes'}
        {kf.blur.plate_boxes.length === 0 ? '; no plate detected in this frame, nothing was blurred for that reason' : ''}
      </figcaption>
    </figure>
  );
}
