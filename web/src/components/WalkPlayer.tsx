import { useMediaPlayer } from '@meta-sam/react';
import type { IMediaPlayer, MediaPlayerRenderContext } from '@meta-sam/video';
import { useCallback, useEffect, useRef, useState } from 'react';

import type { LaneStore } from '../lanes/laneStore.ts';
import { containFit, type Fit } from '../lib/geometry.ts';
import { formatClock } from '../lib/format.ts';
import type { SnapshotObject } from '../workers/protocol.ts';

export interface WalkMetadata {
  width: number;
  height: number;
  numFrames: number;
  fps: number;
  duration: number;
}

export interface Spotlight {
  noun: string;
  id: string;
}

export interface WalkPlayerProps {
  src: string;
  role: string;
  store: LaneStore;
  nouns: readonly string[];
  colorFor: (role: string, noun: string, id: string) => string;
  labelFor: (role: string, noun: string, id: string) => string;
  // Only objects passing this filter are drawn (the MatchPane hides every other panel's ids).
  filterIds?: (noun: string, id: string) => boolean;
  spotlight?: Spotlight | null;
  seekFrame?: number | null;
  autoPlay?: boolean;
  loop?: boolean;
  controls?: boolean;
  clock?: () => number;
  className?: string;
  onFrame?: (frameIndex: number, time: number) => void;
  onMetadata?: (meta: WalkMetadata) => void;
  onFit?: (fit: Fit, cssWidth: number, cssHeight: number) => void;
}

const BLOOM_MS = 350;
const LABEL_AFTER_MS = 400;
const FILL_ALPHA = 0.45;

// One canvas: the blurred proxy decoded by @meta-sam/video, SAM masks from the worker drawn by frame index.
// Bloom per object: outline first, fill rising to 45% over 350 ms, label last.
export function WalkPlayer(props: WalkPlayerProps) {
  const {
    src,
    role,
    store,
    nouns,
    colorFor,
    labelFor,
    filterIds,
    spotlight = null,
    seekFrame = null,
    autoPlay = false,
    loop = true,
    controls = true,
    clock,
    className,
    onFrame,
    onMetadata,
    onFit,
  } = props;

  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const pathCache = useRef(new WeakMap<SnapshotObject, Path2D>());
  const propsRef = useRef(props);
  propsRef.current = props;
  const [meta, setMeta] = useState<WalkMetadata | null>(null);
  const [playing, setPlaying] = useState(false);
  const [frameIndex, setFrameIndex] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const pendingBloom = useRef(false);
  const playerHolder = useRef<IMediaPlayer | null>(null);

  const now = useCallback(() => (propsRef.current.clock ? propsRef.current.clock() : performance.now()), []);

  const renderFrame = useCallback(
    (context: MediaPlayerRenderContext) => {
      const { ctx, canvas, frame, frameIndex: fi } = context;
      const p = propsRef.current;
      const width = canvas.width;
      const height = canvas.height;
      const dpr = Math.max(1, width / Math.max(1, canvas instanceof HTMLCanvasElement ? canvas.clientWidth || width : width));
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, width, height);
      const fit = containFit(frame.width, frame.height, width, height);
      ctx.drawImage(frame, fit.ox, fit.oy, fit.w, fit.h);

      const t = now();
      let blooming = false;
      const drawn: { obj: SnapshotObject; noun: string; color: string; age: number }[] = [];

      for (const noun of p.nouns) {
        for (const obj of p.store.objectsAt(p.role, noun, fi)) {
          if (p.filterIds && !p.filterIds(noun, obj.id)) continue;
          const key = `${p.role}|${noun}|${obj.id}`;
          let first = p.store.firstSeen.get(key);
          if (first === undefined) {
            first = t;
            p.store.firstSeen.set(key, t);
          }
          const age = t - first;
          if (age < LABEL_AFTER_MS + 50) blooming = true;
          drawn.push({ obj, noun, color: p.colorFor(p.role, noun, obj.id), age });
        }
      }

      if (p.spotlight) {
        const target = drawn.find((d) => d.noun === p.spotlight!.noun && d.obj.id === p.spotlight!.id);
        ctx.save();
        ctx.fillStyle = 'rgba(10, 12, 12, 0.6)';
        ctx.fillRect(fit.ox, fit.oy, fit.w, fit.h);
        ctx.restore();
        if (target) {
          ctx.save();
          const s = fit.scale;
          ctx.translate(fit.ox + target.obj.box[0] * s, fit.oy + target.obj.box[1] * s);
          ctx.scale(s, s);
          ctx.clip(pathFor(pathCache.current, target.obj), 'evenodd');
          ctx.setTransform(1, 0, 0, 1, 0, 0);
          ctx.drawImage(frame, fit.ox, fit.oy, fit.w, fit.h);
          ctx.restore();
        }
      }

      for (const d of drawn) {
        const s = fit.scale;
        const path = pathFor(pathCache.current, d.obj);
        ctx.save();
        ctx.translate(fit.ox + d.obj.box[0] * s, fit.oy + d.obj.box[1] * s);
        ctx.scale(s, s);
        const fillAlpha = Math.min(FILL_ALPHA, (FILL_ALPHA * d.age) / BLOOM_MS);
        if (fillAlpha > 0) {
          ctx.globalAlpha = fillAlpha;
          ctx.fillStyle = d.color;
          ctx.fill(path, 'evenodd');
        }
        ctx.globalAlpha = 0.9;
        ctx.strokeStyle = d.color;
        ctx.lineWidth = (2 * dpr) / s;
        ctx.lineJoin = 'round';
        ctx.stroke(path);
        ctx.restore();

        if (d.age >= LABEL_AFTER_MS) {
          const label = p.labelFor(p.role, d.noun, d.obj.id);
          const x = fit.ox + d.obj.box[0] * s;
          const y = fit.oy + d.obj.box[1] * s;
          ctx.save();
          ctx.font = `${Math.round(12 * dpr)}px "IBM Plex Sans", system-ui, sans-serif`;
          const tw = ctx.measureText(label).width + 10 * dpr;
          const th = 18 * dpr;
          const ly = Math.max(fit.oy, y - th);
          ctx.fillStyle = d.color;
          ctx.fillRect(x, ly, tw, th);
          ctx.fillStyle = '#fff';
          ctx.textBaseline = 'middle';
          ctx.fillText(label, x + 5 * dpr, ly + th / 2);
          ctx.restore();
        }
      }
      pendingBloom.current = blooming;
    },
    [now],
  );

  const playerRef = useMediaPlayer({
    canvasRef,
    src,
    loop,
    muted: true,
    autoPlay,
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
    },
    onPlayingChange: setPlaying,
    onError: (err) => setError(err.message),
    onPlayerReady: (player) => {
      playerHolder.current = player;
    },
  });

  // Size the backing store to CSS size times DPR; re-render the retained frame after a resize.
  useEffect(() => {
    const wrap = wrapRef.current;
    const canvas = canvasRef.current;
    if (!wrap || !canvas) return;
    const apply = () => {
      const dpr = window.devicePixelRatio || 1;
      const w = Math.max(1, Math.round(wrap.clientWidth));
      const h = Math.max(1, Math.round(wrap.clientHeight));
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      const m = meta;
      if (m) propsRef.current.onFit?.(containFit(m.width, m.height, w, h), w, h);
      void playerRef.current?.forceRender().catch(() => undefined);
    };
    apply();
    const observer = new ResizeObserver(apply);
    observer.observe(wrap);
    return () => observer.disconnect();
  }, [meta, playerRef]);

  // A bloom in progress needs repaints while paused; while playing the frames arrive anyway.
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      const player = playerRef.current;
      if (pendingBloom.current && player && player.paused) void player.forceRender().catch(() => undefined);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playerRef]);

  // Store updates (new lane frames) repaint the paused frame.
  useEffect(() => {
    return store.subscribe(() => {
      const player = playerRef.current;
      if (player && player.paused) void player.forceRender().catch(() => undefined);
    });
  }, [store, playerRef]);

  useEffect(() => {
    const player = playerRef.current;
    if (!player || !meta || seekFrame === null) return;
    const target = Math.max(0, Math.min(meta.numFrames - 1, seekFrame));
    player.pause();
    void player.seekToFrame(target).catch(() => undefined);
  }, [seekFrame, meta, playerRef]);

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
        <canvas ref={canvasRef} aria-label={`${role} walk with SAM 3.1 masks`} />
        {error ? (
          <div className="walk-error">
            <p>Could not decode this video in the browser: {error}</p>
            <p className="muted">Playback needs WebCodecs (Chrome, Edge, Safari 17+ or Firefox 130+).</p>
          </div>
        ) : null}
        {!meta && !error ? <div className="walk-loading muted">opening the blurred proxy</div> : null}
      </div>
      {controls ? (
        <div className="walk-controls">
          <button type="button" onClick={toggle} disabled={!meta} aria-label={playing ? 'pause' : 'play'}>
            {playing ? 'pause' : 'play'}
          </button>
          <input
            type="range"
            min={0}
            max={Math.max(0, (meta?.numFrames ?? 1) - 1)}
            value={frameIndex}
            onChange={(e) => scrub(Number(e.target.value))}
            disabled={!meta}
            aria-label="frame"
          />
          <span className="mono small">
            {formatClock(time)} / {formatClock(meta?.duration ?? 0)} f{frameIndex}
          </span>
        </div>
      ) : null}
    </div>
  );
}

function pathFor(cache: WeakMap<SnapshotObject, Path2D>, obj: SnapshotObject): Path2D {
  let path = cache.get(obj);
  if (!path) {
    path = new Path2D(obj.path);
    cache.set(obj, path);
  }
  return path;
}
