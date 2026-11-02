import type { PanelTrack } from '../contracts/types.ts';
import { laneId } from './panels.ts';

// Which voted track a lane object belongs to at a frame. SAM 3.1 hands an object id out again once its first object
// has left (on hero_focus4 the front door's id 1 comes back on two rear door tracks from frame 507), so the id alone
// names the wrong panel: the track whose fragment holds the frame decides. Its panel_id is the door check's result,
// the id panels.json, the match and the sheet use.
export interface TrackIndex {
  at(role: string, lane: string, id: string, frame: number): PanelTrack | undefined;
  // Whether the walk overlay draws an object: everything while the role has no panel track yet (the lanes stream
  // first), then only objects inside a kept track, never one the run's data puts off the car.
  shows(role: string, lane: string, id: string, frame: number): boolean;
}

interface Span {
  first: number;
  last: number;
  track: PanelTrack;
}

export function trackIndex(tracks: readonly PanelTrack[], offCar: readonly string[] = []): TrackIndex {
  const spans = new Map<string, Span[]>();
  const roles = new Set<string>();
  const off = new Set(offCar);
  for (const t of tracks) {
    if (!t.panel_id) continue;
    roles.add(t.role);
    const lane = laneId(t.noun);
    for (const f of t.fragments) {
      const key = `${t.role}|${lane}|${f.object_id}`;
      const list = spans.get(key) ?? [];
      list.push({ first: f.first_frame, last: f.last_frame, track: t });
      spans.set(key, list);
    }
  }
  const at = (role: string, lane: string, id: string, frame: number) =>
    spans.get(`${role}|${lane}|${id}`)?.find((s) => s.first <= frame && frame <= s.last)?.track;
  return {
    at,
    shows(role, lane, id, frame) {
      if (!roles.has(role)) return true;
      const t = at(role, lane, id, frame);
      return t !== undefined && !off.has(t.track_id);
    },
  };
}
