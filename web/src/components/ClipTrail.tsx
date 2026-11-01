import { useEffect, useRef, useState } from 'react';

import type { MarkTrack } from '../contracts/types.ts';
import { useFileUrl } from '../lib/files.tsx';

const TRAIL = 6;
// With reduced motion the clip waits on its first frame with its controls instead of looping on its own.
const reducedMotion = typeof window !== 'undefined' && typeof window.matchMedia === 'function' && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

// A clip the server does not have shows a line instead of an empty frame: the repo's sample pair leaves most of its
// clips out to stay small.
export function ClipMissing({ caption, bare = false }: { caption: string; bare?: boolean }) {
  return (
    <figure className="clip-trail">
      <div className="walk-placeholder">This clip is not on this server. The sample pair in the repo leaves most of its clips out to stay small.</div>
      {bare ? null : <figcaption>{caption}</figcaption>}
    </figure>
  );
}

// A Stays Put clip with the mark's own SAM 3.1 track: a 2 px outline and a fading trail of the last 6 frames.
// The clip is the blurred 1008 px window cut around the mark; the boxes are in its pixels.
// atFrame pauses the clip on one frame (a frozen kiosk capture); otherwise it plays and loops.
export function ClipTrail({ mt, color, caption, bare = false, atFrame }: { mt: MarkTrack; color: string; caption: string; bare?: boolean; atFrame?: number }) {
  const url = useFileUrl();
  const ref = useRef<HTMLVideoElement>(null);
  const [frame, setFrame] = useState(atFrame ?? 0);
  const [missing, setMissing] = useState(false);
  const clipPath = mt.clip?.path;

  useEffect(() => setMissing(false), [clipPath]);

  useEffect(() => {
    const v = ref.current;
    if (!v || !mt.clip || atFrame === undefined) return;
    const frames = mt.clip.frames;
    const f = Math.max(0, Math.min(frames - 1, atFrame));
    setFrame(f);
    const seek = () => {
      v.pause();
      if (v.duration > 0) v.currentTime = ((f + 0.5) * v.duration) / frames;
    };
    if (v.readyState >= 1) seek();
    else v.addEventListener('loadedmetadata', seek, { once: true });
    return () => v.removeEventListener('loadedmetadata', seek);
  }, [mt, atFrame]);

  useEffect(() => {
    const v = ref.current;
    if (!v || !mt.clip || atFrame !== undefined || typeof v.requestVideoFrameCallback !== 'function') return;
    const frames = mt.clip.frames;
    let handle = 0;
    const onFrame: VideoFrameRequestCallback = (_now, meta) => {
      const fps = v.duration > 0 ? frames / v.duration : 30;
      setFrame(Math.min(frames - 1, Math.round(meta.mediaTime * fps)));
      handle = v.requestVideoFrameCallback(onFrame);
    };
    handle = v.requestVideoFrameCallback(onFrame);
    return () => v.cancelVideoFrameCallback(handle);
  }, [mt, atFrame]);

  if (!mt.clip) return null;
  if (missing) return <ClipMissing caption={caption} bare={bare} />;
  const trail = mt.track.filter((p) => p.f <= frame && p.f > frame - TRAIL).sort((a, b) => a.f - b.f);
  return (
    <figure className="clip-trail">
      <div className="clip-frame">
        <video ref={ref} src={url(mt.clip.path)} muted autoPlay={atFrame === undefined && !reducedMotion} controls={reducedMotion && atFrame === undefined} loop playsInline aria-label={caption} onError={() => setMissing(true)} />
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
