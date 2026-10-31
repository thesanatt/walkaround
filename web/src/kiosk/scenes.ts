import type { PanelId } from '../contracts/types.ts';

// Kiosk scenes keyed to docs/VIDEO_PLAN.md beats. /kiosk/<run>?scene=<name>&autostart=1 starts one by URL;
// ?t=<ms> freezes it at that many ms after its start for a frame-by-frame capture.
export type SceneLayout = 'seal' | 'walk' | 'fragments' | 'coverage' | 'crops' | 'moment' | 'verdict' | 'hood' | 'claim' | 'sheet' | 'topup' | 'blur' | 'guardrails';

// How the replay feeds a scene: from the first event, from the tracking stage (everything earlier applied at once),
// or the finished run at once so the scene's own steps do the timing.
export type SceneReplay = 'from_start' | 'from_tracking' | 'final';

export interface SceneStep {
  atMs: number;
  name: 'dim' | 'trail' | 'cut' | 'card';
}

export interface Scene {
  name: string;
  beat: string;
  layout: SceneLayout;
  heading: string; // a product heading, not a caption; the edit adds the voiceover captions
  panel: PanelId | null; // the default panel; ?panel= overrides it, and a run without it falls back to its first match
  replay: SceneReplay;
  holdMs: number; // how long the beat runs; the capture script reads it, the page never advances by itself
  steps?: SceneStep[];
}

export const SCENES: Scene[] = [
  { name: 'seal', beat: 'B5', layout: 'seal', heading: 'two walks, hashed on the phone, sealed on arrival', panel: null, replay: 'from_start', holdMs: 7000 },
  // TODO(D1): outlines appear only after a lane's sam.lane_done plus its worker parse; the sam.frame boxes streamed
  // during the hold are not drawn. On the mock with the checked-in delays the pickup car_door lane_done lands 5.7 s
  // into this 8 s hold. With the real sample pair's fixed replay, measure when the pickup panel lanes' lane_done land
  // after the tracking stage starts and how long their parse takes; if outlines are not up by about 2 s, start the
  // scene later (a startAtStage offset) or give tracking a schedule that puts the pickup lane_done events first.
  { name: 'masks', beat: 'B5', layout: 'walk', heading: 'pickup walk', panel: null, replay: 'from_tracking', holdMs: 8000 },
  { name: 'fragments', beat: 'B6', layout: 'fragments', heading: 'one panel, two visits, one track', panel: 'right_rear_door', replay: 'final', holdMs: 6000 },
  { name: 'coverage', beat: 'B7', layout: 'coverage', heading: 'coverage', panel: null, replay: 'final', holdMs: 5000 },
  { name: 'crops', beat: 'B8', layout: 'crops', heading: 'one crop, one panel, one Spark call', panel: 'right_rear_door', replay: 'final', holdMs: 5000 },
  {
    name: 'moment',
    beat: 'B9',
    layout: 'moment',
    heading: '',
    panel: 'right_rear_door',
    replay: 'final',
    holdMs: 12000,
    steps: [
      { atMs: 0, name: 'dim' },
      { atMs: 3000, name: 'trail' },
      { atMs: 7000, name: 'cut' },
      { atMs: 10000, name: 'card' },
    ],
  },
  { name: 'tape', beat: 'B10', layout: 'verdict', heading: 'added between the walks', panel: 'left_front_door', replay: 'final', holdMs: 8000 },
  { name: 'hood', beat: 'B11', layout: 'hood', heading: 'sky on the hood', panel: 'hood', replay: 'final', holdMs: 8000 },
  { name: 'claim', beat: 'B12', layout: 'claim', heading: 'a claim photo against the pickup walk', panel: null, replay: 'final', holdMs: 16000 },
  { name: 'sheet', beat: 'B13', layout: 'sheet', heading: 'the sheet', panel: null, replay: 'final', holdMs: 15000 },
  { name: 'topup', beat: 'B14', layout: 'topup', heading: 'what is missing, before you drive off', panel: null, replay: 'final', holdMs: 14000 },
  { name: 'blur', beat: 'B15', layout: 'blur', heading: 'blurred before storage', panel: 'right_rear_door', replay: 'final', holdMs: 4000 },
  { name: 'guardrails', beat: 'B16', layout: 'guardrails', heading: 'what it cost and what was deleted', panel: null, replay: 'final', holdMs: 8000 },
];

export function sceneByName(name: string | null): Scene {
  return SCENES.find((s) => s.name === name) ?? SCENES.find((s) => s.name === 'masks')!;
}

export function neighbourScene(name: string, direction: 1 | -1): Scene {
  const index = SCENES.findIndex((s) => s.name === name);
  return SCENES[Math.max(0, Math.min(SCENES.length - 1, (index < 0 ? 0 : index) + direction))]!;
}

export function stepAt(scene: Scene, elapsedMs: number): SceneStep | null {
  let current: SceneStep | null = null;
  for (const step of scene.steps ?? []) if (step.atMs <= elapsedMs) current = step;
  return current;
}
