import { useEffect, useRef, useState } from 'react';

import type { MarkTrack } from '../contracts/types.ts';
import { useFileUrl } from '../lib/files.tsx';

const TRAIL = 6;

// A Stays Put clip with the mark's own SAM 3.1 track: a 2 px outline and a fading trail of the last 6 frames.
// The clip is the blurred 1008 px window cut around the mark; the boxes are in its pixels.
export function ClipTrail({ mt, color, caption, bare = false }: { mt: MarkTrack; color: string; caption: string; bare?: boolean }) {
  const url = useFileUrl();
  const ref = useRef<HTMLVideoElement>(null);
  const [frame, setFrame] = useState(0);

  useEffect(() => {
    const v = ref.current;
    if (!v || !mt.clip || typeof v.requestVideoFrameCallback !== 'function') return;
    const frames = mt.clip.frames;
    let handle = 0;
    const onFrame: VideoFrameRequestCallback = (_now, meta) => {
      const fps = v.duration > 0 ? frames / v.duration : 30;
      setFrame(Math.min(frames - 1, Math.round(meta.mediaTime * fps)));
      handle = v.requestVideoFrameCallback(onFrame);
    };
    handle = v.requestVideoFrameCallback(onFrame);
    return () => v.cancelVideoFrameCallback(handle);
  }, [mt]);

  if (!mt.clip) return null;
  const trail = mt.track.filter((p) => p.f <= frame && p.f > frame - TRAIL).sort((a, b) => a.f - b.f);
  return (
    <figure className="clip-trail">
      <div className="clip-frame">
        <video ref={ref} src={url(mt.clip.path)} muted autoPlay loop playsInline aria-label={caption} />
        <svg viewBox={`0 0 ${mt.clip.w} ${mt.clip.h}`} aria-hidden="true">
          {trail.map((p) => (
            <rect
              key={p.f}
              x={p.box_clip[0]}
              y={p.box_clip[1]}
              width={p.box_clip[2] - p.box_clip[0]}
              height={p.box_clip[3] - p.box_clip[1]}
              fill="none"
              stroke={color}
              strokeWidth={p.f === frame ? 4 : 2}
              opacity={p.f === frame ? 1 : 0.15 + (0.6 * (TRAIL - (frame - p.f))) / TRAIL}
            />
          ))}
        </svg>
      </div>
      {bare ? null : <figcaption>{caption}</figcaption>}
    </figure>
  );
}
