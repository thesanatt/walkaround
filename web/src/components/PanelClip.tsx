import { routes } from '../api/client.ts';
import type { Keyframe, Mark, PanelTrack, Verdict } from '../contracts/types.ts';
import type { LaneStore } from '../lanes/laneStore.ts';
import { bestKeyframe, panelObjectKeys } from '../lib/focus.ts';
import { formatClock } from '../lib/format.ts';
import { DRAWN_LANES, panelName } from '../lib/panels.ts';
import { WalkPlayer } from './WalkPlayer.tsx';

const reducedMotion = typeof window !== 'undefined' && typeof window.matchMedia === 'function' && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

// About three seconds of the walk around the frame the card shows, inside the SAM track that holds it.
export function clipWindow(tracks: PanelTrack[], role: string, panelId: string, center: number, fps: number): [number, number] | null {
  const frags = tracks.filter((t) => t.role === role && t.panel_id === panelId).flatMap((t) => t.fragments);
  if (!frags.length) return null;
  const holding = frags.find((f) => f.first_frame <= center && center <= f.last_frame);
  const frag = holding ?? [...frags].sort((a, b) => b.last_frame - b.first_frame - (a.last_frame - a.first_frame))[0]!;
  const mid = holding ? center : Math.round((frag.first_frame + frag.last_frame) / 2);
  const half = Math.round(1.5 * fps);
  let start = Math.max(frag.first_frame, mid - half);
  let end = Math.min(frag.last_frame, mid + half);
  if (end - start < fps) {
    start = Math.max(0, mid - half);
    end = mid + half;
  }
  return [start, end];
}

export interface PanelClipProps {
  runId: string;
  v: Verdict;
  marks: Map<string, Mark>;
  keyframes: Keyframe[];
  tracks: PanelTrack[];
  blurred: Record<string, string | null>; // role -> the blurred display video
  fps: (role: string) => number;
  store: LaneStore;
  colorFor: (role: string, lane: string, id: string) => string;
  labelFor: (role: string, lane: string, id: string) => string;
}

// The first card's panel as SAM 3.1 tracked it: a short loop of the blurred walk with everything but that panel
// dimmed. The walk is the one the card's mark was found in (return for a new mark). Nothing shows when the run has no
// blurred video or no track of the panel.
export function PanelClip({ runId, v, marks, keyframes, tracks, blurred, fps, store, colorFor, labelFor }: PanelClipProps) {
  const own = v.mark_ids.map((id) => marks.get(id)).filter((m): m is Mark => m !== undefined);
  const role = v.label === 'new_or_unseen' ? 'return' : (own.find((m) => m.role === 'pickup') ?? own[0])?.role ?? 'pickup';
  const src = blurred[role];
  const keys = panelObjectKeys(tracks, role, v.panel_id);
  if (!src || !keys.size) return null;
  const mark = own.find((m) => m.role === role);
  const kf = (mark ? keyframes.find((k) => k.kf_id === mark.kf_id) : undefined) ?? bestKeyframe(keyframes, role, v.panel_id);
  const rate = fps(role) || 15;
  const win = clipWindow(tracks, role, v.panel_id, kf?.f_proxy ?? 0, rate);
  if (!win) return null;
  return (
    <figure className="panel-clip">
      <WalkPlayer
        src={routes.file(runId, src)}
        role={role}
        store={store}
        lanes={DRAWN_LANES}
        colorFor={colorFor}
        labelFor={labelFor}
        filterIds={(lane, id) => keys.has(`${lane}|${id}`)}
        spotlight={{ keys }}
        startTime={win[0] / rate}
        segment={win}
        autoPlay={!reducedMotion}
        controls={reducedMotion}
        bloom={false}
      />
      <figcaption>
        SAM 3.1 following the {panelName(v.panel_id)} through the {role} walk, {formatClock(win[0] / rate)} to {formatClock(win[1] / rate)}. Everything else is dimmed.
      </figcaption>
    </figure>
  );
}
