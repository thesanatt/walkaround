import type { PanelId } from '../contracts/types.ts';

export interface PanelInfo {
  id: PanelId;
  name: string;
  noun: string;
  family: 'body' | 'wheel' | 'light_glass';
  color: string;
}

// The 22 canonical slots of contracts.PanelId with their nouns and one fixed color each.
// Colors: distinct, calm, no indigo or violet family; the same color follows a panel everywhere.
export const PANELS: PanelInfo[] = [
  { id: 'left_front_door', name: 'left front door', noun: 'car door', family: 'body', color: '#1f7a6d' },
  { id: 'left_rear_door', name: 'left rear door', noun: 'car door', family: 'body', color: '#2b9d8c' },
  { id: 'right_front_door', name: 'right front door', noun: 'car door', family: 'body', color: '#0f6b5c' },
  { id: 'right_rear_door', name: 'right rear door', noun: 'car door', family: 'body', color: '#3bb3a1' },
  { id: 'left_front_fender', name: 'left front fender', noun: 'fender', family: 'body', color: '#b5651d' },
  { id: 'left_rear_fender', name: 'left rear fender', noun: 'fender', family: 'body', color: '#d4873c' },
  { id: 'right_front_fender', name: 'right front fender', noun: 'fender', family: 'body', color: '#8f4e14' },
  { id: 'right_rear_fender', name: 'right rear fender', noun: 'fender', family: 'body', color: '#e6a45c' },
  { id: 'hood', name: 'hood', noun: 'hood', family: 'body', color: '#c9a227' },
  { id: 'roof', name: 'roof', noun: 'roof', family: 'body', color: '#7d8a2e' },
  { id: 'trunk_lid', name: 'trunk lid', noun: 'trunk lid', family: 'body', color: '#a56b2a' },
  { id: 'front_bumper', name: 'front bumper', noun: 'bumper', family: 'body', color: '#b23a3a' },
  { id: 'rear_bumper', name: 'rear bumper', noun: 'bumper', family: 'body', color: '#d95f5f' },
  { id: 'left_front_wheel', name: 'left front wheel', noun: 'wheel', family: 'wheel', color: '#3b5f8a' },
  { id: 'left_rear_wheel', name: 'left rear wheel', noun: 'wheel', family: 'wheel', color: '#5b84b5' },
  { id: 'right_front_wheel', name: 'right front wheel', noun: 'wheel', family: 'wheel', color: '#27456b' },
  { id: 'right_rear_wheel', name: 'right rear wheel', noun: 'wheel', family: 'wheel', color: '#7fa3d1' },
  { id: 'headlights', name: 'headlights', noun: 'headlight', family: 'light_glass', color: '#9c8b4a' },
  { id: 'tail_lights', name: 'tail lights', noun: 'tail light', family: 'light_glass', color: '#9e3f5e' },
  { id: 'mirrors', name: 'mirrors', noun: 'side mirror', family: 'light_glass', color: '#5c7a5c' },
  { id: 'windshield', name: 'windshield', noun: 'windshield', family: 'light_glass', color: '#4f8a99' },
  { id: 'rear_window', name: 'rear window', noun: 'rear window', family: 'light_glass', color: '#6aa3b0' },
];

export const PANEL_BY_ID: Record<string, PanelInfo> = Object.fromEntries(PANELS.map((p) => [p.id, p]));

// The 16 video lanes of section 5 step 5, in request order.
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

export const PRIVACY_NOUNS = new Set(['license plate', 'face']);

// Fallback colors for nouns that are not a panel (car, car window, plate, face) or a panel not yet voted.
const NOUN_COLORS: Record<string, string> = {
  car: '#6b7280',
  'car window': '#8a9bb0',
  'license plate': '#a3a3a3',
  face: '#a3a3a3',
  'ping pong ball': '#d4873c',
};

export function panelName(id: string | null | undefined): string {
  if (!id) return 'unassigned';
  return PANEL_BY_ID[id]?.name ?? id.replaceAll('_', ' ');
}

export function colorForPanel(id: string | null | undefined): string {
  return (id && PANEL_BY_ID[id]?.color) || '#6b7280';
}

export function colorForNoun(noun: string): string {
  const explicit = NOUN_COLORS[noun];
  if (explicit) return explicit;
  const panel = PANELS.find((p) => p.noun === noun);
  if (panel) return panel.color;
  let h = 0;
  for (const ch of noun) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  return `hsl(${h % 360} 35% 42%)`;
}

export function nounSlug(noun: string): string {
  return noun.trim().toLowerCase().split(/\s+/).join('_');
}

export const COVERAGE_LABEL: Record<string, string> = {
  filmed: 'filmed',
  glimpsed: 'glimpsed',
  not_filmed: 'not filmed',
  not_run: 'not run',
  not_seen: 'not seen',
};

export const VERDICT_LABEL: Record<string, string> = {
  already_there: 'already there',
  needs_your_eye: 'needs your eye',
  new_or_unseen: 'new or unseen',
  not_seen_at_return: 'not seen at return',
  not_comparable: 'not comparable',
};
