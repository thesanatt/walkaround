import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import type { DeletionRecord, Keyframe, KeyframePairTry, Mark, MarkTrack, PanelMatch, PanelTrack, RunManifest, SizeEstimate, Verdict } from '../src/contracts/types.ts';
import { drawnPairs, focusTrack, joinedPairs, offPaneMarks, paneKeyframes, panelFocus, panelObjectKeys } from '../src/lib/focus.ts';
import { keyframePairLine, sizeLine, staysPutLine, tireDiameterIn, wallClock } from '../src/lib/lines.ts';
import { deleteFailed, PRIVACY_BEFORE, storageLine, uploadedFileIds } from '../src/lib/privacy.ts';
import { RULES } from '../src/lib/rules.ts';
import { replayBadgeDate, sheetFileHash } from '../src/lib/runview.ts';
import { initialState, reduce } from '../src/state/reducer.ts';
import type { AnyEvent } from '../src/state/types.ts';
import { fixtureRun, RUN_DIR } from './helpers.ts';

const derived = <T>(name: string): T => JSON.parse(readFileSync(join(RUN_DIR, 'derived', name), 'utf8')) as T;
const DOOR = 'right_rear_door';
const P1 = 'kf_pickup_right_rear_door_m1';
const P2 = 'kf_pickup_right_rear_door_m2';
const R1 = 'kf_return_right_rear_door_m1';

function fixture() {
  const marks = derived<Mark[]>('marks.json');
  const matches = derived<PanelMatch[]>('matches.json');
  const verdicts = derived<Verdict[]>('verdicts.json');
  const keyframes = derived<Keyframe[]>('keyframes.json');
  const markTracks = derived<MarkTrack[]>('mark_tracks.json');
  const door = matches.find((m) => m.panel_id === DOOR)!;
  return { marks, matches, verdicts, keyframes, markTracks, door, byId: new Map(marks.map((m) => [m.mark_id, m])) };
}

describe('the match pane follows the verdicts code wrote', () => {
  it('draws no join for a pair at or past tau_eye_r (L5 prints two separate marks)', () => {
    const { door } = fixture();
    expect(joinedPairs(door).map((p) => p.pickup_mark)).toEqual([P1, P2]);
    const far: PanelMatch = { ...door, pairs: door.pairs.map((p) => (p.pickup_mark === P2 ? { ...p, d: p.tau_eye_r } : p)) };
    expect(joinedPairs(far).map((p) => p.pickup_mark)).toEqual([P1]);
    expect(joinedPairs({ ...door, state: 'not_comparable' })).toEqual([]);
  });

  it('leads with the "already there" verdict even when another pair verdict comes first', () => {
    const { door, verdicts } = fixture();
    const reordered = [verdicts.find((v) => v.label === 'needs_your_eye')!, ...verdicts.filter((v) => v.label !== 'needs_your_eye')];
    const focus = panelFocus(DOOR, door, reordered);
    expect(focus.verdict?.label).toBe('already_there');
    expect([focus.pair?.pickup_mark, focus.pair?.return_mark]).toEqual([P1, R1]);
    // Without an "already there", the first verdict about a joined pair, never a pair L5 split.
    const noAlready = reordered.filter((v) => v.label !== 'already_there');
    expect(panelFocus(DOOR, door, noAlready).pair?.pickup_mark).toBe(P2);
    const far: PanelMatch = { ...door, pairs: door.pairs.map((p) => ({ ...p, d: p.tau_eye_r + 0.1 })) };
    expect(panelFocus(DOOR, far, noAlready).pair).toBeNull();
  });

  it('shows the keyframe a mark was found on and joins only marks on the frames shown', () => {
    const { door, verdicts, keyframes, marks } = fixture();
    // Dedup kept the pickup scuff's medoid from a second keyframe of the walk, with its own crop.
    const other: Keyframe = { ...keyframes.find((k) => k.kf_id === 'kf_pickup_right_rear_door')!, kf_id: 'kf_pickup_right_rear_door_b', f_proxy: 52, score: { ...keyframes[0]!.score, total: 0.5 } };
    const moved = marks.map((m) => (m.mark_id === P1 ? { ...m, kf_id: other.kf_id } : m));
    const byId = new Map(moved.map((m) => [m.mark_id, m]));
    const kfs = [...keyframes, other];
    const focus = panelFocus(DOOR, door, verdicts);
    const { a, b } = paneKeyframes(DOOR, door, focus, byId, kfs);
    expect(a?.kf_id).toBe(other.kf_id); // not kf_pair[0], whose crop P1's point_crop does not belong to
    expect(b?.kf_id).toBe('kf_return_right_rear_door');
    expect(drawnPairs(door, byId, a, b).map((p) => p.pickup_mark)).toEqual([P1]);
    expect(offPaneMarks(DOOR, moved, a, b).map((m) => m.mark_id)).toEqual([P2]);
    // With no focus pair the view falls back to match.kf_pair, and the moved mark is the one left off.
    const plain = paneKeyframes(DOOR, door, { verdict: null, pair: null }, byId, kfs);
    expect(plain.a?.kf_id).toBe('kf_pickup_right_rear_door');
    expect(drawnPairs(door, byId, plain.a, plain.b).map((p) => p.pickup_mark)).toEqual([P2]);
    expect(offPaneMarks(DOOR, moved, plain.a, plain.b).map((m) => m.mark_id)).toEqual([P1]);
  });

  it("shows a one-mark verdict's mark on its own keyframe (the tape scene)", () => {
    const { matches, verdicts, keyframes, marks } = fixture();
    const lfd = matches.find((m) => m.panel_id === 'left_front_door')!;
    const other: Keyframe = { ...keyframes.find((k) => k.kf_id === 'kf_return_left_front_door')!, kf_id: 'kf_return_left_front_door_b', score: { ...keyframes[0]!.score, total: 0.3 } };
    const moved = marks.map((m) => (m.mark_id === 'kf_return_left_front_door_m1' ? { ...m, kf_id: other.kf_id } : m));
    // Only the tape's own verdict: the chip pair of this door (v5) would lead otherwise.
    const focus = panelFocus('left_front_door', lfd, verdicts.filter((v) => v.verdict_id !== 'v5'));
    expect(focus.verdict?.label).toBe('new_or_unseen');
    expect(focus.pair).toBeNull();
    const { a, b } = paneKeyframes('left_front_door', lfd, focus, new Map(moved.map((m) => [m.mark_id, m])), [...keyframes, other]);
    expect(a?.kf_id).toBe('kf_pickup_left_front_door');
    expect(b?.kf_id).toBe(other.kf_id);
    // The door's chip stayed on the first return keyframe, which this view does not show.
    expect(offPaneMarks('left_front_door', moved, a, b).map((m) => m.mark_id)).toEqual(['kf_return_left_front_door_m2']);
  });

  it("takes the moment's Stays Put trail from the focus mark only, never another mark or panel", () => {
    const { door, verdicts, markTracks, byId } = fixture();
    const focus = panelFocus(DOOR, door, verdicts);
    expect(focusTrack(focus, DOOR, markTracks, byId)?.mark_id).toBe(P1);
    const withoutP1 = markTracks.filter((m) => m.mark_id !== P1);
    expect(focusTrack(focus, DOOR, withoutP1, byId)).toBeUndefined();
    // A track of another panel's mark (not the control) is not borrowed by a panel without verdicts.
    const hoodMark = withoutP1.map((m) => ({ ...m, is_control: false }));
    expect(focusTrack({ verdict: null, pair: null }, 'left_front_door', hoodMark, byId)).toBeUndefined();
    expect(focusTrack({ verdict: null, pair: null }, 'hood', hoodMark, byId)?.mark_id).toBe('kf_pickup_hood_m1');
  });

  it('spotlights every SAM id of a panel merged from several fragments', () => {
    const tracks = derived<PanelTrack[]>('tracks.json');
    const merged = tracks.map((t) => (t.role === 'pickup' && t.panel_id === DOOR ? { ...t, sam_object_ids: ['0', '4'] } : t));
    expect([...panelObjectKeys(merged, 'pickup', DOOR)].sort()).toEqual(['car_door|0', 'car_door|4']);
    expect([...panelObjectKeys(merged, 'pickup', 'hood')]).toEqual(['hood|3']);
    expect(panelObjectKeys(merged, 'pickup', 'roof').size).toBe(0);
    expect(panelObjectKeys(merged, 'pickup', null).size).toBe(0);
  });
});

describe('the sheet line on the run page', () => {
  it('finds evidence.pdf under the run-relative keys sheet.write_sheet emits', () => {
    const payload = { sha256: { 'sheet/sheet.json': 'a'.repeat(64), 'sheet/sheet.html': 'b'.repeat(64), 'bundle/note.md': 'c'.repeat(64), 'sheet/evidence.pdf': 'd'.repeat(64) } };
    const state = reduce(initialState(), { seq: 0, ts: '', run_id: 'r', kind: 'sheet.written', payload, prev: '', hash: '' } as AnyEvent);
    expect(state.sheet.written).toBe(true);
    expect(sheetFileHash(state.sheet.sha256s, 'evidence.pdf')).toBe('d'.repeat(64));
    const { 'sheet/evidence.pdf': _pdf, ...withoutPdf } = payload.sha256;
    expect(sheetFileHash(withoutPdf, 'evidence.pdf')).toBeNull();
    expect(sheetFileHash({ 'evidence.pdf': 'e' }, 'evidence.pdf')).toBe('e');
  });

  it('shows the replay badge for a stored run whatever the URL', () => {
    const run = fixtureRun();
    const base = initialState();
    expect(replayBadgeDate({ ...base, replay: { mode: 'fixed', badgeDate: '2026-10-27' } })).toBe('2026-10-27');
    expect(replayBadgeDate({ ...base, run })).toBe(run.created_at); // live_or_replay "replay"
    const live: RunManifest = { ...run, live_or_replay: 'live', keep: false, status: 'tracking' };
    expect(replayBadgeDate({ ...base, run: live })).toBeNull();
    expect(replayBadgeDate({ ...base, run: { ...live, keep: true } })).toBeNull(); // a keep run still running is live
    expect(replayBadgeDate({ ...base, run: { ...live, keep: true, status: 'done' } })).toBe(run.created_at);
  });
});

describe('the privacy card says only what the deletion rows show', () => {
  const run = fixtureRun();
  const ok = (target: string): DeletionRecord => ({ target, kind: 'files_api', sha256: null, bytes: null, deleted_at: '2026-10-27T15:11:11Z', reason: 'run end (pickup proxy)', api_response: { id: target, object: 'file', deleted: true } });
  const failed = (target: string): DeletionRecord => ({
    ...ok(target),
    reason: 'delete request failed at 2026-10-27T15:11:11Z; expiry at 2026-10-27T16:02:41Z applies (3 tries, last error ConnectError: down)',
    api_response: null,
  });
  // privacy.privacy_line for a run with store false accepted, word for word.
  const SHEET_LINE =
    "Processed with Meta's Model API in Standard mode, which Meta's documentation says is not used to train Meta models. " +
    'Uploads were set to expire after one hour and were deleted by Walkaround when the run ended. Model requests were sent with storage off. ' +
    "Meta's terms allow it to retain content for abuse review and legal reasons; Walkaround does not control that. " +
    'SAM 3.1 saw the unblurred proxy for the length of one segmentation pass to find the plates and faces it then blurred.';

  it('promises in the present tense until the run ends', () => {
    expect(storageLine(run, { completed: false, uploads: ['f1'], deletions: [ok('f1')] })).toBe(PRIVACY_BEFORE);
  });

  it('prints the sheet line once every upload has a delete receipt', () => {
    expect(storageLine(run, { completed: true, uploads: ['f1', 'f2'], deletions: [ok('f1'), ok('f2')] })).toBe(SHEET_LINE);
  });

  it('names Baseten when the second look sent crops there, and only then', () => {
    const clause = 'two blurred crops per look were sent to Muse Spark Open on Baseten, a company separate from Meta; no video went there.';
    expect(storageLine(run, { completed: true, uploads: ['f1', 'f2'], deletions: [ok('f1'), ok('f2')], baseten: true })).toBe(`${SHEET_LINE} For the second look, ${clause}`);
    expect(storageLine(run, { completed: false, uploads: ['f1'], deletions: [], baseten: true })).toContain(clause);
    expect(storageLine(run, { completed: true, uploads: ['f1', 'f2'], deletions: [ok('f1'), ok('f2')], baseten: false })).not.toContain('Baseten');
  });

  it('says a failed delete failed, and never "deleted"', () => {
    const rows = [ok('f1'), failed('f2')];
    expect(rows.map(deleteFailed)).toEqual([false, true]);
    const line = storageLine(run, { completed: true, uploads: ['f1', 'f2'], deletions: rows });
    expect(line).not.toContain('were deleted by Walkaround');
    expect(line).toContain("Walkaround's delete request failed for one upload, so the expiry applies");
    const notRun = storageLine(run, { completed: true, uploads: ['f1', 'f2'], deletions: [ok('f1')] });
    expect(notRun).toContain('The run ended before Walkaround deleted one upload, so the one-hour expiry applies.');
    expect(storageLine(run, { completed: true, uploads: [], deletions: [] })).toBe("No file from this run reached Meta's Model API.");
    expect(deleteFailed({ ...ok('q'), kind: 'quarantine', api_response: null })).toBe(false); // a local unlink has no receipt
  });

  it('collects the upload ids from events, records and rows', () => {
    const media = { pickup: { upload: null, ffprobe: null, proxy: null, blurredPath: null, fileId: 'f1' } };
    const videos = { return: { role: 'return', files_api: { file_id: 'f2', uploaded_at: '', expires_at: null, deleted_at: null, delete_response: null } } } as never;
    expect(uploadedFileIds(media, videos, [failed('f3')])).toEqual(['f1', 'f2', 'f3']);
  });
});

describe('lines copied from sheet.py', () => {
  it('prints no wall clock without a camera time (the receipt time never stands in)', () => {
    const video = { received_at: '2026-10-30T18:00:00Z', ffprobe: null } as never;
    expect(wallClock({ creation_time: 'not present' } as never, video, 2)).toBeNull();
    expect(wallClock(null, video, 2)).toBeNull();
    expect(wallClock({ creation_time: '2026-10-27T14:55:30Z' } as never, video, 2.5)).toBe('2026-10-27T14:55:32.500Z');
  });

  it('words a size like sheet.size_line', () => {
    const size = (s: Partial<SizeEstimate>): SizeEstimate => ({ inches: null, bound_lo: null, bound_hi: null, ruler: null, ruler_source_kf: null, tire_code: null, threshold_type: null, refused_reason: null, ...s });
    expect(sizeLine(null)).toBeNull();
    expect(sizeLine(size({ refused_reason: 'the plate is not in this frame' }))).toBe('size not stated: the plate is not in this frame');
    expect(sizeLine(size({}))).toBe('size not stated: no ruler qualified');
    expect(sizeLine(size({ inches: 1.1, bound_lo: 0.9, bound_hi: 1.3, ruler: 'plate' }))).toBe('about 1.1 in, between 0.9 and 1.3 in (ruler: the license plate, 12 in)');
    expect(tireDiameterIn('225/60R16')).toBeCloseTo(26.63, 2);
    expect(sizeLine(size({ inches: 0.9, ruler: 'wheel', tire_code: '225/60R16' }))).toBe('about 0.9 in (ruler: the wheel, 26.6 in from 225/60R16, typed by you)');
    expect(sizeLine(size({ inches: 0.9, ruler: 'wheel', tire_code: 'unreadable' }))).toBe('about 0.9 in (ruler: the wheel)');
    expect(sizeLine(size({ inches: 0.9 }))).toBe('about 0.9 in (ruler: not recorded)');
    // the other ruler's note, as sheet.size_line prints it (polish lane, 10/31)
    const turned = 'the license plate is turned more than 30 degrees from the camera';
    expect(sizeLine(size({ inches: 0.9, ruler: 'wheel', tire_code: '155/80R13', ruler_notes: [turned] }))).toBe(`about 0.9 in (ruler: the wheel, 22.8 in from 155/80R13, typed by you; ${turned})`);
    expect(sizeLine(size({ refused_reason: 'tire size not given', ruler_notes: [turned] }))).toBe(`size not stated: tire size not given; ${turned}`);
  });

  it('mirrors every rule id of rules.py, K15 and the retired G2 included', () => {
    const py = readFileSync(new URL('../../walkaround/rules.py', import.meta.url), 'utf8');
    const ids = [...py.matchAll(/Rule\("([A-Za-z0-9]+)"/g)].map((m) => m[1]!).sort();
    expect(Object.keys(RULES).sort()).toEqual(ids);
    expect(RULES['K15']!.label).toBe('view flagged');
    expect(RULES['G2']!.label).toBe('crop condition (retired)');
  });

  it('words a Stays Put row like sheet.staysput_line once calibrated', () => {
    const [fixed, control] = derived<MarkTrack[]>('mark_tracks.json');
    expect(staysPutLine(fixed, true)).toBe('drift 0.03, b1 0.21, b2 0.02, not panned; fixed');
    expect(staysPutLine(control, true)).toBe('drift 0.61, b1 0.19, b2 0.03, not panned; sliding; highlight cap');
    expect(staysPutLine({ ...fixed!, drift_rms: null }, true)).toBe('drift n/a, b1 0.21, b2 0.02, not panned; fixed');
    expect(staysPutLine({ ...fixed!, verdict: 'inconclusive', panned: true }, true)).toBe('drift 0.03, b1 0.21, b2 0.02, panned; inconclusive (between fixed and sliding; sign test 0.94 on 88 pairs)');
    expect(staysPutLine({ ...fixed!, verdict: 'inconclusive', b_pass: false }, true)).toBe('inconclusive (camera did not move enough)');
    expect(staysPutLine({ ...fixed!, verdict: 'inconclusive', drift_rms: null }, true)).toBe('inconclusive (the whole panel was in view on too few frames)');
    expect(staysPutLine({ ...fixed!, verdict: 'no_track' }, true)).toBe('no track (SAM did not follow the mark through the clip)');
    expect(staysPutLine({ ...fixed!, verdict: 'not_run' }, true)).toBeNull();
  });

  it('prints only the measured numbers while Stays Put is uncalibrated (cap_enabled false)', () => {
    const [fixed, control] = derived<MarkTrack[]>('mark_tracks.json');
    // pair C's trunk chips: real paint, drift 0.21, labelled sliding by the uncalibrated rule
    expect(staysPutLine({ ...fixed!, verdict: 'sliding', drift_rms: 0.21, b1: 0.3, b2: 0.01 }, false)).toBe('drift 0.21, b1 0.30, b2 0.01, not panned');
    expect(staysPutLine(control, false)).toBe('drift 0.61, b1 0.19, b2 0.03, not panned');
    expect(staysPutLine({ ...fixed!, verdict: 'inconclusive', panned: true }, false)).toBe('drift 0.03, b1 0.21, b2 0.02, panned');
    expect(staysPutLine({ ...fixed!, verdict: 'no_track' }, false)).toBe('no track (SAM did not follow the mark through the clip)');
    expect(staysPutLine({ ...fixed!, verdict: 'not_run' }, false)).toBeNull();
  });
});

describe('keyframePairLine is match.how_checked word for word (11/1)', () => {
  const t = (p: string, r: string, gates: string[], iou: number | null = null, agree = true): KeyframePairTry => ({ kf_pair: [p, r], gates, mask_iou: iou, frames_agree: agree });
  const text = (...tries: KeyframePairTry[]) => keyframePairLine({ kf_pairs_tried: tries } as unknown as PanelMatch);

  it('words every case as tests/test_match_1101.py does', () => {
    expect(keyframePairLine(null)).toBeNull();
    expect(text()).toBeNull();
    expect(text(t('p1', 'r1', []))).toBe('compared on keyframes p1 with r1, the first pair tried');
    expect(text(t('p1', 'r1', ['G3']), t('p1', 'r2', ['G4'], 0.62), t('p2', 'r1', []))).toBe(
      'compared on keyframes p2 with r1; tried first: p1 with r1 (G3 blurry keyframe); p1 with r2 (G4 mask IoU 0.62)',
    );
    expect(text(t('p1', 'r1', ['G4', 'G8'], null, false), t('p2', 'r1', ['G8'], 0.75))).toBe(
      'not compared: none of the 2 keyframe pairs passed every gate; tried: p1 with r1 (G4 mask IoU: panel axes disagree, G8 hood IoU: panel axes disagree); p2 with r1 (G8 hood IoU 0.75)',
    );
    expect(text(t('p1', 'r1', ['G1', 'G4'], 0.5))).toBe('not compared: keyframes p1 with r1 (G1 not filmed, G4 mask IoU 0.50); G1 is a panel gate, so no other pair was tried');
    expect(text(t('p1', 'r1', ['G1', 'G6']))).toMatch(/\(G1 not filmed, G6 side uncertain\); G1 and G6 are panel gates, so no other pair was tried$/);
    expect(text(t('p1', 'r1', ['G9']))).toBe('not compared: keyframes p1 with r1 (G9 mask definition differs), and the panel has no other keyframe pair');
    expect(text(t('p1', 'r1', ['G4']))).toBe('not compared: keyframes p1 with r1 (G4 mask IoU: not measured), and the panel has no other keyframe pair');
    expect(text(t('p1', 'r1', ['G5', 'G10']))).toMatch(/G10 is a panel gate, so no other pair was tried$/);
  });
});
