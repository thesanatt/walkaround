import type { PanelId } from '../contracts/types.ts';

// Scenes keyed to docs/VIDEO_PLAN.md beats. One keypress per step; ?scene=<name>&autostart=1 starts one by URL.
export type SceneLayout = 'upload' | 'walk' | 'fragments' | 'coverage' | 'crops' | 'moment' | 'verdict' | 'hood' | 'claim' | 'sheet' | 'topup' | 'blur' | 'guardrails';

export interface SceneStep {
  atMs: number;
  name: 'dim' | 'trail' | 'cut' | 'card';
  caption: string | null;
}

export interface Scene {
  name: string;
  beat: string;
  layout: SceneLayout;
  title: string;
  panel: PanelId | null;
  // How long the beat holds on screen; the capture script reads it, the page does not advance by itself.
  holdMs: number;
  caption?: string;
  steps?: SceneStep[];
}

export const SCENES: Scene[] = [
  { name: 'seal', beat: 'B5', layout: 'upload', title: 'two videos, hashed in the browser, then stamped', panel: null, holdMs: 7000 },
  { name: 'masks', beat: 'B5', layout: 'walk', title: 'SAM 3.1 outlines every named panel from plain nouns', panel: null, holdMs: 8000 },
  { name: 'fragments', beat: 'B6', layout: 'fragments', title: 'same door, two visits, one track', panel: 'right_rear_door', holdMs: 6000 },
  { name: 'coverage', beat: 'B7', layout: 'coverage', title: 'coverage first', panel: null, holdMs: 5000 },
  { name: 'crops', beat: 'B8', layout: 'crops', title: 'Muse Spark boxes marks in one crop; it never sees both videos together', panel: 'right_rear_door', holdMs: 5000 },
  {
    name: 'moment',
    beat: 'B9',
    layout: 'moment',
    title: '',
    panel: 'right_rear_door',
    holdMs: 12000,
    steps: [
      { atMs: 0, name: 'dim', caption: null },
      { atMs: 3000, name: 'trail', caption: 'the mark stays on the paint; the reflection slides off' },
      { atMs: 7000, name: 'cut', caption: null },
      { atMs: 10000, name: 'card', caption: null },
    ],
  },
  { name: 'tape', beat: 'B10', layout: 'verdict', title: 'new or unseen, with the pickup seconds that show the door without it', panel: 'left_front_door', holdMs: 8000 },
  { name: 'hood', beat: 'B11', layout: 'hood', title: 'models find and compare, code decides', panel: 'hood', holdMs: 8000 },
  { name: 'claim', beat: 'B12', layout: 'claim', title: 'when a bill arrives, paste its photo', panel: null, holdMs: 16000 },
  { name: 'sheet', beat: 'B13', layout: 'sheet', title: 'the sheet', panel: null, holdMs: 15000 },
  { name: 'topup', beat: 'B14', layout: 'topup', title: 'back at pickup, it names what is missing', panel: null, holdMs: 14000 },
  { name: 'blur', beat: 'B15', layout: 'blur', title: 'plates and faces are blurred before anything is stored', panel: 'right_rear_door', holdMs: 4000 },
  { name: 'guardrails', beat: 'B16', layout: 'guardrails', title: 'the ledger and the deletions', panel: null, holdMs: 8000 },
];

export function sceneByName(name: string | null): Scene {
  return SCENES.find((s) => s.name === name) ?? SCENES[1]!;
}

export function neighbourScene(name: string, direction: 1 | -1): Scene {
  const index = SCENES.findIndex((s) => s.name === name);
  const next = Math.max(0, Math.min(SCENES.length - 1, (index < 0 ? 0 : index) + direction));
  return SCENES[next]!;
}

export function stepAt(scene: Scene, elapsedMs: number): SceneStep | null {
  if (!scene.steps) return null;
  let current: SceneStep | null = null;
  for (const step of scene.steps) if (step.atMs <= elapsedMs) current = step;
  return current;
}
