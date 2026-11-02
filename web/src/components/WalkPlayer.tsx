import { useMediaPlayer } from '@meta-sam/react';
import type { MediaPlayerRenderContext } from '@meta-sam/video';
import { useCallback, useEffect, useRef, useState } from 'react';

import type { LaneStore } from '../lanes/laneStore.ts';
import { formatClock } from '../lib/format.ts';
import { containFit } from '../lib/geometry.ts';
import type { SnapshotObject } from '../workers/protocol.ts';

export interface WalkMetadata {
  width: number;
  height: number;
  numFrames: number;
  fps: number;
  duration: number;
}

// The panel's objects at a frame (a merged track carries several ids, and SAM reuses an id for a later object).
export interface Spotlight {
  has: (lane: string, id: string, frame: number) => boolean;
}

export interface WalkPlayerProps {
  src: string;
  role: string;
  store: LaneStore;
  lanes: readonly string[];
  colorFor: (role: string, lane: string, id: string, frame: number) => string;
  labelFor: (role: string, lane: string, id: string, frame: number) => string;
  visible?: (role: string, lane: string, id: string, frame: number) => boolean; // the run's own filter (useTrackStyle)
  filterIds?: (lane: string, id: string, frame: number) => boolean; // only objects passing it are drawn
  spotlight?: Spotlight | null; // dims everything outside these objects' masks
  seekFrame?: number | null; // pause and show this proxy frame
  startTime?: number; // seconds to open the video at; playback continues from there
  autoPlay?: boolean;
  loop?: boolean;
  segment?: [number, number] | null; // play only these proxy frames, over and over (the result page's panel clip)
  controls?: boolean;
  bloom?: boolean; // false draws every mask at full fill at once (scripted frame capture)
  labelPx?: number; // label text size in CSS pixels, 12 by default
  clock?: () => number;
  className?: string;
  onFrame?: (frameIndex: number, time: number) => void;
  onMetadata?: (meta: WalkMetadata) => void;
}

const BLOOM_MS = 350;
const LABEL_AFTER_MS = 400;
const FILL_ALPHA = 0.45;
const DIM = 'rgba(10, 12, 12, 0.62)';

const hasWebCodecs = typeof globalThis.VideoDecoder === 'function';
const reducedMotion = typeof window !== 'undefined' && typeof window.matchMedia === 'function' && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

interface Drawn {
  o: SnapshotObject;
  lane: string;
  color: string;
  age: number;
}

// One canvas: the blurred proxy decoded by @meta-sam/video, SAM 3.1 outlines from the lane worker drawn by frame index.
export function WalkPlayer(props: WalkPlayerProps) {
  const { src, role, store, seekFrame = null, startTime, autoPlay = false, loop = true, controls = true, className } = props;
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const propsRef = useRef(props);
  propsRef.current = props;
  const dprRef = useRef(1);
  const paths = useRef(new WeakMap<SnapshotObject, Path2D>());
  const blooming = useRef(false);
  const wrapping = useRef(false);
  const [meta, setMeta] = useState<WalkMetadata | null>(null);
  const [playing, setPlaying] = useState(false);
  const [frameIndex, setFrameIndex] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const renderFrame = useCallback((context: MediaPlayerRenderContext) => {
    const { ctx, canvas, frame, frameIndex: fi, fallbackCtx, fallbackCanvas } = context;
    const p = propsRef.current;
    const dpr = dprRef.current;
    const fit = containFit(frame.width, frame.height, canvas.width, canvas.height);
    fallbackCtx.setTransform(1, 0, 0, 1, 0, 0);
    fallbackCtx.clearRect(0, 0, fallbackCanvas.width, fallbackCanvas.height);
    fallbackCtx.drawImage(frame, fit.ox, fit.oy, fit.w, fit.h);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(frame, fit.ox, fit.oy, fit.w, fit.h);

    const now = p.clock ? p.clock() : performance.now();
    const bloom = (p.bloom ?? true) && !reducedMotion;
    const drawn: Drawn[] = [];
    let anyBlooming = false;
    for (const lane of p.lanes) {
      for (const o of p.store.objectsAt(p.role, lane, fi)) {
        if (p.visible && !p.visible(p.role, lane, o.id, fi)) continue;
        if (p.filterIds && !p.filterIds(lane, o.id, fi)) continue;
        const key = `${p.role}|${lane}|${o.id}`;
        let first = p.store.firstSeen.get(key);
        if (first === undefined) {
          first = now;
          p.store.firstSeen.set(key, now);
        }
        const age = bloom ? now - first : Infinity;
        if (age < LABEL_AFTER_MS) anyBlooming = true;
        drawn.push({ o, lane, color: p.colorFor(p.role, lane, o.id, fi), age });
      }
    }
    blooming.current = anyBlooming;

    // Places a raster-pixel path on the canvas: box origin, frame scale, raster-to-box scale. The traced outline
    // runs through pixel centres, so it moves half a raster pixel to sit on the pixel edges.
    const place = (o: SnapshotObject) => {
      const sx = o.rw > 0 ? (o.box[2] - o.box[0]) / o.rw : 1;
      const sy = o.rh > 0 ? (o.box[3] - o.box[1]) / o.rh : 1;
      ctx.translate(fit.ox + o.box[0] * fit.scale, fit.oy + o.box[1] * fit.scale);
      ctx.scale(fit.scale * sx, fit.scale * sy);
      ctx.translate(0.5, 0.5);
      return Math.max(fit.scale * sx, fit.scale * sy);
    };

    if (p.spotlight) {
      ctx.fillStyle = DIM;
      ctx.fillRect(fit.ox, fit.oy, fit.w, fit.h);
      for (const target of drawn) {
        if (!target.o.path || !p.spotlight.has(target.lane, target.o.id, fi)) continue;
        ctx.save();
        place(target.o);
        ctx.clip(pathOf(paths.current, target.o), 'evenodd');
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.drawImage(frame, fit.ox, fit.oy, fit.w, fit.h);
        ctx.restore();
      }
    }

    for (const d of drawn) {
      ctx.save();
      ctx.globalAlpha = 0.9;
      ctx.strokeStyle = d.color;
      if (d.o.path) {
        const scale = place(d.o);
        const path = pathOf(paths.current, d.o);
        const fill = Math.min(FILL_ALPHA, (FILL_ALPHA * d.age) / BLOOM_MS);
        if (fill > 0) {
          ctx.globalAlpha = fill;
          ctx.fillStyle = d.color;
          ctx.fill(path, 'evenodd');
          ctx.globalAlpha = 0.9;
        }
        ctx.lineWidth = (2 * dpr) / scale;
        ctx.lineJoin = 'round';
        ctx.stroke(path);
      } else {
        // A record with a box and no mask: the box itself.
        ctx.lineWidth = 2 * dpr;
        ctx.strokeRect(fit.ox + d.o.box[0] * fit.scale, fit.oy + d.o.box[1] * fit.scale, (d.o.box[2] - d.o.box[0]) * fit.scale, (d.o.box[3] - d.o.box[1]) * fit.scale);
      }
      ctx.restore();
    }

    ctx.save();
    const px = p.labelPx ?? 12;
    ctx.font = `500 ${Math.round(px * dpr)}px "IBM Plex Sans", system-ui, sans-serif`;
    ctx.textBaseline = 'middle';
    for (const d of drawn) {
      if (d.age < LABEL_AFTER_MS) continue;
      const label = p.labelFor(p.role, d.lane, d.o.id, fi);
      const x = fit.ox + d.o.box[0] * fit.scale;
      const y = fit.oy + d.o.box[1] * fit.scale;
      const tw = ctx.measureText(label).width + ((px * 10) / 12) * dpr;
      const th = px * 1.5 * dpr;
      const ly = Math.max(fit.oy, y - th);
      ctx.fillStyle = d.color;
      ctx.fillRect(x, ly, tw, th);
      ctx.fillStyle = '#fff';
      ctx.fillText(label, x + ((px * 5) / 12) * dpr, ly + th / 2);
    }
    ctx.restore();
  }, []);

  const playerRef = useMediaPlayer({
    canvasRef,
    src,
    loop,
    muted: true,
    autoPlay,
    initialSeekTime: startTime,
    renderFrame,
    onLoadedMetadata: (m) => {
      const info = { width: m.width, height: m.height, numFrames: m.numFrames, fps: m.fps, duration: m.duration };
      setMeta(info);
      setError(null);
      propsRef.current.onMetadata?.(info);
    },
    onFrame: ({ frameIndex: fi, time }) => {
      setFrameIndex(fi);
      propsRef.current.onFrame?.(fi, time);
      const seg = propsRef.current.segment;
      const player = playerRef.current;
      if (seg && player && !wrapping.current && fi >= seg[1]) {
        wrapping.current = true;
        void player
          .seekToFrame(seg[0])
          .then(() => player.play())
          .catch(() => undefined)
          .finally(() => {
            wrapping.current = false;
          });
      }
    },
    onPlayingChange: setPlaying,
    onError: (err) => setError(err.message),
  });

  // The backing store follows the CSS size times the device pixel ratio; the retained frame is redrawn after a resize.
  useEffect(() => {
    const wrap = wrapRef.current;
    const canvas = canvasRef.current;
    if (!wrap || !canvas) return;
    const apply = () => {
      const dpr = window.devicePixelRatio || 1;
      dprRef.current = dpr;
      const w = Math.max(1, Math.round(wrap.clientWidth));
      const h = Math.max(1, Math.round(wrap.clientHeight));
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      void playerRef.current?.forceRender().catch(() => undefined);
    };
    apply();
    const observer = new ResizeObserver(apply);
    observer.observe(wrap);
    return () => observer.disconnect();
  }, [playerRef]);

  // While paused, new lane frames and a bloom in progress need repaints; during playback frames arrive anyway.
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      const player = playerRef.current;
      if (blooming.current && player?.paused) void player.forceRender().catch(() => undefined);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    const unsubscribe = store.subscribe(() => {
      const player = playerRef.current;
      if (player?.paused) void player.forceRender().catch(() => undefined);
    });
    return () => {
      cancelAnimationFrame(raf);
      unsubscribe();
    };
  }, [store, playerRef]);

  useEffect(() => {
    const player = playerRef.current;
    if (!player || !meta || seekFrame === null) return;
    player.pause();
    void player.seekToFrame(Math.max(0, Math.min(meta.numFrames - 1, seekFrame))).catch(() => undefined);
  }, [seekFrame, meta, playerRef]);

  if (!hasWebCodecs) {
    return (
      <div className={`walk ${className ?? ''}`}>
        <video className="walk-plain" src={src} controls muted playsInline loop={loop} aria-label={`${role} walk, blurred`} />
        <p className="small muted">Simple playback mode: this browser has no WebCodecs, so the SAM 3.1 outlines are not drawn here. The sheet carries the frames.</p>
      </div>
    );
  }

  const toggle = () => {
    const player = playerRef.current;
    if (!player) return;
    if (player.paused) void player.play().catch((e: Error) => setError(e.message));
    else player.pause();
  };

  const scrub = (value: number) => {
    const player = playerRef.current;
    if (!player) return;
    player.pause();
    void player.seekToFrame(value).catch(() => undefined);
  };

  const time = meta && meta.fps > 0 ? frameIndex / meta.fps : 0;

  return (
    <div className={`walk ${className ?? ''}`}>
      <div className="walk-canvas" ref={wrapRef}>
        <canvas ref={canvasRef} aria-label={`${role} walk, blurred, with SAM 3.1 outlines`} />
        {error ? (
          <div className="walk-error">
            <p>The browser could not decode this video: {error}</p>
          </div>
        ) : null}
        {!meta && !error ? <div className="walk-loading">opening the blurred video</div> : null}
      </div>
      {controls ? (
        <div className="walk-controls">
          <button type="button" className="btn small" onClick={toggle} disabled={!meta}>
            {playing ? 'pause' : 'play'}
          </button>
          <input
            type="range"
            min={0}
            max={Math.max(0, (meta?.numFrames ?? 1) - 1)}
            value={frameIndex}
            onChange={(e) => scrub(Number(e.target.value))}
            disabled={!meta}
            aria-label={`${role} walk frame`}
          />
          <span className="mono small">
            {formatClock(time)} / {formatClock(meta?.duration ?? 0)}, frame {frameIndex}
          </span>
        </div>
      ) : null}
    </div>
  );
}

function pathOf(cache: WeakMap<SnapshotObject, Path2D>, o: SnapshotObject): Path2D {
  let path = cache.get(o);
  if (!path) {
    path = new Path2D(o.path);
    cache.set(o, path);
  }
  return path;
}
