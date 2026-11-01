import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import { CropView } from '../src/components/CropView.tsx';
import { MatchPane, type MatchPaneProps } from '../src/components/MatchPane.tsx';
import { PrivacyCard } from '../src/components/PrivacyCard.tsx';
import { SealCard } from '../src/components/SealCard.tsx';
import type { DeletionRecord, Keyframe, Mark, PanelMatch, Verdict } from '../src/contracts/types.ts';
import { initialState, reduce } from '../src/state/reducer.ts';
import type { AnyEvent, RunState } from '../src/state/types.ts';
import { startLive } from '../src/state/live.ts';
import { FakeEventSource, fixtureFiles, recordedEvents, RUN_DIR } from './helpers.ts';

const derived = <T>(name: string): T => JSON.parse(readFileSync(join(RUN_DIR, 'derived', name), 'utf8')) as T;
const html = (el: Parameters<typeof renderToStaticMarkup>[0]) => renderToStaticMarkup(el);
const count = (text: string, needle: string) => text.split(needle).length - 1;

async function finalState(): Promise<RunState> {
  const seen: AnyEvent[] = [];
  let source: FakeEventSource | null = null;
  await new Promise<void>((resolve) => {
    startLive('mock-sample-pair', (e) => seen.push(e), { files: fixtureFiles(), openEvents: (url) => (source = new FakeEventSource(url)), onEnd: resolve });
    const wait = setInterval(() => {
      if (!source) return;
      clearInterval(wait);
      source.open();
      for (const e of recordedEvents()) source.emit(e);
    }, 1);
  });
  return seen.reduce(reduce, initialState());
}

function paneProps(state: RunState, panelId: string, patch: Partial<MatchPaneProps> = {}): MatchPaneProps {
  return {
    panelId,
    match: state.matches.find((m) => m.panel_id === panelId),
    panel: state.panels.find((p) => p.panel_id === panelId),
    keyframes: state.keyframes,
    marks: state.marks,
    verdicts: state.verdicts,
    videos: state.videos,
    media: state.media,
    run: state.run,
    ...patch,
  };
}

describe('the changed components render against the fixture run', () => {
  it('joins both door pairs on the run page and labels the moment with the "already there" verdict', async () => {
    const state = await finalState();
    const page = html(createElement(MatchPane, paneProps(state, 'right_rear_door')));
    expect(count(page, 'class="join-line"')).toBe(2);
    expect(page).toContain('already there');
    expect(page).toContain('needs your eye');
    // The needs-your-eye verdict first, as mutual_nn's order can put it: the moment card still says "already there".
    const verdicts: Verdict[] = [...state.verdicts.filter((v) => v.label === 'needs_your_eye'), ...state.verdicts.filter((v) => v.label !== 'needs_your_eye')];
    const moment = html(createElement(MatchPane, paneProps(state, 'right_rear_door', { verdicts, moment: true, aligned: true, showCards: true })));
    expect(moment).toMatch(/class="moment-card"><span class="label already_there">already there<\/span>/);
  });

  it('drops the join of a pair past tau_eye_r and of a mark on another keyframe, and says where that mark is', async () => {
    const state = await finalState();
    const door = state.matches.find((m) => m.panel_id === 'right_rear_door')!;
    const far: PanelMatch = { ...door, pairs: door.pairs.map((p) => (p.pickup_mark.endsWith('m2') ? { ...p, d: p.tau_eye_r * 1.5 } : p)) };
    const farPage = html(createElement(MatchPane, paneProps(state, 'right_rear_door', { match: far })));
    expect(count(farPage, 'class="join-line"')).toBe(1);

    const other: Keyframe = { ...state.keyframes.find((k) => k.kf_id === 'kf_pickup_right_rear_door')!, kf_id: 'kf_pickup_right_rear_door_b', score: { ...state.keyframes[0]!.score, total: 0.4 } };
    const marks: Mark[] = state.marks.map((m) => (m.mark_id === 'kf_pickup_right_rear_door_m2' ? { ...m, kf_id: other.kf_id } : m));
    const page = html(createElement(MatchPane, paneProps(state, 'right_rear_door', { marks, keyframes: [...state.keyframes, other] })));
    expect(count(page, 'class="join-line"')).toBe(1);
    expect(page).toContain('One mark of this panel was found on another frame of its walk');
    const crops = html(createElement(CropView, { role: 'pickup', panelId: 'right_rear_door', keyframes: [...state.keyframes, other], marks }));
    expect(count(crops, '<figure class="crop">')).toBe(2);
    expect(crops).toContain('pickup, another frame');
  });

  it('prints "camera time not present" instead of a receipt-time date', async () => {
    const state = await finalState();
    const media = { ...state.media, return: { ...state.media['return']!, ffprobe: { ...state.media['return']!.ffprobe!, creation_time: 'not present' } } };
    const videos = { ...state.videos, return: { ...state.videos['return']!, ffprobe: null } };
    const page = html(createElement(MatchPane, paneProps(state, 'right_rear_door', { media, videos })));
    expect(page).toContain('return camera time not present');
    expect(page).not.toMatch(/days? later|minutes later/);
  });

  it('prints the type note of a tight pair and how the panel was checked (11/1)', async () => {
    const state = await finalState();
    const door = state.matches.find((m) => m.panel_id === 'right_rear_door')!;
    const note = 'Muse Spark typed both marks dirt (K4), with low confidence on both marks (K12): a note here, since the mark sits in the same place in both walks and the second look compared the two crops and said same';
    const verdicts: Verdict[] = state.verdicts.map((v) => (v.panel_id === 'right_rear_door' && v.label === 'already_there' ? { ...v, type_note: note, trace: { ...v.trace, note_ids: ['K4', 'K12'] } } : v));
    const tried: PanelMatch = {
      ...door,
      kf_pairs_tried: [
        { kf_pair: ['kf_pickup_right_rear_door', 'kf_return_x'], gates: ['G3'], mask_iou: 0.81, frames_agree: true },
        { kf_pair: door.kf_pair!, gates: [], mask_iou: 0.84, frames_agree: true },
      ],
    };
    const page = html(createElement(MatchPane, paneProps(state, 'right_rear_door', { verdicts, match: tried })));
    expect(page).toContain(note);
    expect(page).toContain(`How it was checked: compared on keyframes ${door.kf_pair![0]} with ${door.kf_pair![1]}; tried first: kf_pickup_right_rear_door with kf_return_x (G3 blurry keyframe)`);
    const before = html(createElement(MatchPane, paneProps(state, 'right_rear_door')));
    expect(before).not.toContain('How it was checked');
    const moment = html(createElement(MatchPane, paneProps(state, 'right_rear_door', { match: tried, moment: true })));
    expect(moment).not.toContain('How it was checked');
  });

  it('labels a failed delete as failed on the privacy card', async () => {
    const state = await finalState();
    expect(html(createElement(PrivacyCard, { state }))).toContain('were deleted by Walkaround when the run ended');
    const rows: DeletionRecord[] = state.deletions.map((d) =>
      d.kind === 'files_api' && d.target.endsWith('return') ? { ...d, api_response: null, reason: 'delete request failed at 2026-10-27T15:11:11Z; expiry at 2026-10-27T16:02:41Z applies' } : d,
    );
    const failed = html(createElement(PrivacyCard, { state: { ...state, deletions: rows } }));
    expect(failed).not.toContain('were deleted by Walkaround');
    expect(failed).toContain('file-mock-return</span>: delete request failed at 2026-10-27T15:11:11Z');
    const running = html(createElement(PrivacyCard, { state: { ...state, completed: false } }));
    expect(running).toContain('Walkaround deletes them when the run ends');
  });

  it('says a walk was not sealed when the run failed before the seal', () => {
    const video = derived<unknown>('../videos/pickup.json') as Parameters<typeof SealCard>[0]['video'];
    const seal = { ...initialState().seal };
    expect(html(createElement(SealCard, { role: 'pickup', video, seal, failed: true }))).toContain('pickup walk: received; not sealed, the run failed');
    expect(html(createElement(SealCard, { role: 'pickup', video, seal }))).toContain('sealing now');
  });
});
