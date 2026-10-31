import type { PanelId } from '../contracts/types.ts';

export interface PanelInfo {
  id: PanelId;
  name: string;
  noun: string;
  family: 'body' | 'wheel' | 'light_glass';
  color: string;
}

// The 22 canonical slots of contracts.PanelId with their nouns and one fixed color each.
// Hue families by noun (doors teal, fenders amber, bumpers red, wheels slate blue); the same color follows a panel everywhere.
export const PANELS: PanelInfo[] = [
  { id: 'left_front_door', name: 'left front door', noun: 'car door', family: 'body', color: '#1f7a6d' },
  { id: 'left_rear_door', name: 'left rear door', noun: 'car door', family: 'body', color: '#3aa892' },
  { id: 'right_front_door', name: 'right front door', noun: 'car door', family: 'body', color: '#16584f' },
  { id: 'right_rear_door', name: 'right rear door', noun: 'car door', family: 'body', color: '#5fc2ad' },
  { id: 'left_front_fender', name: 'left front fender', noun: 'fender', family: 'body', color: '#b5651d' },
  { id: 'left_rear_fender', name: 'left rear fender', noun: 'fender', family: 'body', color: '#d98a3d' },
  { id: 'right_front_fender', name: 'right front fender', noun: 'fender', family: 'body', color: '#8f4e14' },
  { id: 'right_rear_fender', name: 'right rear fender', noun: 'fender', family: 'body', color: '#e9a960' },
  { id: 'hood', name: 'hood', noun: 'hood', family: 'body', color: '#c9a227' },
  { id: 'roof', name: 'roof', noun: 'roof', family: 'body', color: '#7d8a2e' },
  { id: 'trunk_lid', name: 'trunk lid', noun: 'trunk lid', family: 'body', color: '#a56b2a' },
  { id: 'front_bumper', name: 'front bumper', noun: 'bumper', family: 'body', color: '#b23a3a' },
  { id: 'rear_bumper', name: 'rear bumper', noun: 'bumper', family: 'body', color: '#dc6b6b' },
  { id: 'left_front_wheel', name: 'left front wheel', noun: 'wheel', family: 'wheel', color: '#3b5f8a' },
  { id: 'left_rear_wheel', name: 'left rear wheel', noun: 'wheel', family: 'wheel', color: '#5b84b5' },
  { id: 'right_front_wheel', name: 'right front wheel', noun: 'wheel', family: 'wheel', color: '#27456b' },
  { id: 'right_rear_wheel', name: 'right rear wheel', noun: 'wheel', family: 'wheel', color: '#86a8d4' },
  { id: 'headlights', name: 'headlights', noun: 'headlight', family: 'light_glass', color: '#9c8b4a' },
  { id: 'tail_lights', name: 'tail lights', noun: 'tail light', family: 'light_glass', color: '#9e3f5e' },
  { id: 'mirrors', name: 'mirrors', noun: 'side mirror', family: 'light_glass', color: '#5c7a5c' },
  { id: 'windshield', name: 'windshield', noun: 'windshield', family: 'light_glass', color: '#4f8a99' },
  { id: 'rear_window', name: 'rear window', noun: 'rear window', family: 'light_glass', color: '#6aa3b0' },
];

export const PANEL_BY_ID: Record<string, PanelInfo> = Object.fromEntries(PANELS.map((p) => [p.id, p]));

// contracts.LANE_NOUNS: the 16 video lanes in request order.
export const LANE_NOUNS = [
  'car',
  'license plate',
  'face',
  'car door',
  'bumper',
  'wheel',
  'hood',
  'trunk lid',
  'fender',
  'roof',
  'headlight',
  'tail light',
  'side mirror',
  'windshield',
  'rear window',
  'car window',
] as const;

// contracts.noun_slug / paths.lane_id: 'license plate' -> 'license_plate'. sam.* events carry this spelling.
export function laneId(noun: string): string {
  return noun.trim().toLowerCase().split(/[\s_]+/).filter(Boolean).join('_');
}

export function nounOfLane(lane: string): string {
  return lane.replaceAll('_', ' ');
}

// Plates and faces exist only to be blurred: their lanes are never drawn on a walk.
export const PRIVACY_LANES: ReadonlySet<string> = new Set(['license_plate', 'face']);

// The lanes drawn over a walk: the panel nouns. "car" would cover everything and "car window" only feeds the
// window subtraction, so neither is drawn.
export const DRAWN_LANES: readonly string[] = LANE_NOUNS.map(laneId).filter((l) => !PRIVACY_LANES.has(l) && l !== 'car' && l !== 'car_window');

const LANE_COLORS: Record<string, string> = {
  car: '#6b7280',
  car_window: '#8a9bb0',
  license_plate: '#a3a3a3',
  face: '#a3a3a3',
};

export function panelName(id: string | null | undefined): string {
  if (!id) return 'unassigned';
  return PANEL_BY_ID[id]?.name ?? id.replaceAll('_', ' ');
}

export function colorForPanel(id: string | null | undefined): string {
  return (id && PANEL_BY_ID[id]?.color) || '#6b7280';
}

// Before the side vote names a panel, a lane is drawn in the first color of its noun's family.
export function colorForLane(lane: string): string {
  const id = laneId(lane);
  const explicit = LANE_COLORS[id];
  if (explicit) return explicit;
  const panel = PANELS.find((p) => laneId(p.noun) === id);
  if (panel) return panel.color;
  let h = 0;
  for (const ch of id) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  return `hsl(${h % 360} 35% 42%)`;
}

export const COVERAGE_LABEL: Record<string, string> = {
  filmed: 'filmed',
  glimpsed: 'glimpsed',
  not_filmed: 'not filmed',
  not_run: 'not run',
  not_seen: 'not seen',
};

export const COVERAGE_ORDER = ['filmed', 'glimpsed', 'not_filmed', 'not_run', 'not_seen'] as const;

export const VERDICT_LABEL: Record<string, string> = {
  already_there: 'already there',
  needs_your_eye: 'needs your eye',
  new_or_unseen: 'new or unseen',
  not_seen_at_return: 'not seen at return',
  not_comparable: 'not comparable',
};

// "door" for a door slot: the noun a sentence uses for one panel ("about 3 cm on this door").
export function familyNoun(panelId: string): string {
  const noun = PANEL_BY_ID[panelId]?.noun ?? 'panel';
  return noun.replace('car ', '');
}
