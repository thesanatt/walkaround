import { useEffect, useMemo, useState, type ReactNode } from 'react';

import { api, routes } from '../api/client.ts';
import { Wordmark } from '../components/Bar.tsx';
import { BlurBeat } from '../components/BlurBeat.tsx';
import { CoverageCar, revealOrder, walkGaps } from '../components/CoverageCar.tsx';
import { bestKeyframe, CropFigure } from '../components/CropView.tsx';
import { FragmentsRow } from '../components/FragmentsRow.tsx';
import { HowChecked } from '../components/HowChecked.tsx';
import { LedgerStrip } from '../components/LedgerStrip.tsx';
import { MatchCut, type CutPhase } from '../components/MatchCut.tsx';
import { DatesCard, MatchPane } from '../components/MatchPane.tsx';
import { PanelList } from '../components/PanelList.tsx';
import { PrivacyCard } from '../components/PrivacyCard.tsx';
import { ReplayBadge } from '../components/ReplayBadge.tsx';
import { SealCard } from '../components/SealCard.tsx';
import { StageLine } from '../components/StageLine.tsx';
import { controlTrack, StaysPutPair } from '../components/StaysPutPair.tsx';
import { topupGains } from '../components/TopUpBanner.tsx';
import { WalkPlayer, type Spotlight } from '../components/WalkPlayer.tsx';
import type { Claim, CoverageState, PanelId } from '../contracts/types.ts';
import { CHECK_ROWS, COVERAGE_FILL, CROP_MARKS, neighbourScene, revealCount, sceneByName, stepAt, type Scene } from '../kiosk/scenes.ts';
import { useLanes } from '../lanes/useLanes.ts';
import { claimPhotoPath, effectiveExit } from '../lib/claim.ts';
import { RunFiles, useFileUrl } from '../lib/files.tsx';
import { focusTrack, paneKeyframes, panelFocus, panelObjectKeys } from '../lib/focus.ts';
import { formatClockTenths } from '../lib/format.ts';
import { markLabel, staysPutCalibrated } from '../lib/lines.ts';
import { colorForPanel, COVERAGE_LABEL, DRAWN_LANES, panelName, VERDICT_LABEL } from '../lib/panels.ts';
import { navigate } from '../lib/router.tsx';
import { fpsOf, rolesOf } from '../state/reducer.ts';
import type { RunState } from '../state/types.ts';
import { useRun, type RunSource } from '../state/useRun.ts';
import { ClaimExit, ClaimFrame, ClaimPhoto, ClaimSteps, type ClaimContext } from './ClaimPage.tsx';
import { defaultPanel, useTrackStyle } from './RunRecord.tsx';
import { SheetPage } from './SheetPage.tsx';

const WIDTH = 1920;
const HEIGHT = 1080;
// The verdict scene shows this many cards at most (the rest are on the sheet), so the frames keep a useful size and
// the cards fit above the credit band.
export const KIOSK_CARDS = 2;

// The verdict scene's heading is the label of the verdict the panel leads with (the one its card and the moment
// show), in the app's own words; every other scene has a fixed heading.
export function sceneHeading(scene: Scene, state: RunState, panelId: string | null): string {
  if (scene.layout !== 'verdict') return scene.heading;
  const match = state.matches.find((m) => m.panel_id === panelId);
  const v = panelFocus(panelId, match, state.verdicts).verdict;
  return v ? (VERDICT_LABEL[v.label] ?? v.label) : '';
}

function sceneUrl(runId: string, scene: Scene, query: URLSearchParams): string {
  const q = new URLSearchParams(query);
  q.set('scene', scene.name);
  q.set('autostart', '1');
  q.delete('t');
  q.delete('panel');
  return `/kiosk/${encodeURIComponent(runId)}?${q.toString()}`;
}

// A final-state scene loads the finished run at once; a paced one replays the run's own events with the scene's
// delay table laid over kiosk/delays.json.
export function sourceFor(scene: Scene, t: number | null): RunSource {
  if (scene.replay === 'final') return { kind: 'replay', mode: 'instant' };
  return { kind: 'replay', mode: 'fixed', untilMs: t ?? undefined, startAtStage: scene.replay === 'from_tracking' ? 'tracking' : undefined, delays: scene.delays };
}

// The scene's panel: ?panel=, else the scene's default when the run has a keyframe for it, else the run's first match.
function resolvePanel(scene: Scene, query: URLSearchParams, state: RunState): string | null {
  const asked = query.get('panel');
  if (asked) return asked;
  if (scene.panel && state.keyframes.some((k) => k.panel_id === scene.panel)) return scene.panel;
  return defaultPanel(state) ?? scene.panel;
}

// 1920 x 1080, no page chrome, cursor hidden, the replay badge fixed bottom right on every scene, the light theme
// unless ?theme=dark, so every take looks the same whatever the capture machine prefers.
// ?scene=<name>&autostart=1 starts a scene by URL; ?t=<ms> freezes it there; ?form=upgraded adds the Stays Put clips to
// the moment; ?panel= overrides the scene's panel; ?fit=1 scales the stage down to the window for a preview.
export function KioskPage({ runId, query }: { runId: string; query: URLSearchParams }) {
  const scene = sceneByName(query.get('scene'));
  const tRaw = query.get('t');
  const t = tRaw !== null && Number.isFinite(Number(tRaw)) ? Math.max(0, Number(tRaw)) : null;
  const [started, setStarted] = useState(query.get('autostart') === '1' || t !== null);
  const [elapsed, setElapsed] = useState(t ?? 0);
  const [scale, setScale] = useState(1);
  const fit = query.get('fit') === '1';
  const theme = query.get('theme') === 'dark' ? 'dark' : 'light';

  const source = useMemo(() => sourceFor(scene, t), [scene, t]);
  const { state, status, ended } = useRun(started ? runId : null, source, 'kiosk');
  const lanes = useLanes(started ? runId : null, state, ['pickup', 'return'], 'kiosk');
  // A final-state scene waits for the whole run to load; a paced one starts with its first event.
  const ready = started && (scene.replay === 'final' ? ended : status === 'open' || ended);

  useEffect(() => {
    document.body.classList.add('kiosk');
    const root = document.documentElement;
    const before = root.dataset['theme'];
    root.dataset['theme'] = theme;
    return () => {
      document.body.classList.remove('kiosk');
      if (before === undefined) delete root.dataset['theme'];
      else root.dataset['theme'] = before;
    };
  }, [theme]);

  useEffect(() => {
    if (!fit) return;
    // The layout viewport, not innerWidth: a phone zooms out to show the unscaled 1920 px stage, which grows innerWidth.
    const apply = () => setScale(Math.min(1, document.documentElement.clientWidth / WIDTH, document.documentElement.clientHeight / HEIGHT));
    apply();
    window.addEventListener('resize', apply);
    return () => window.removeEventListener('resize', apply);
  }, [fit]);

  // The scene clock starts when the data is ready and only moves at step boundaries, so a 60 fps capture never
  // re-renders the stage between steps. ?t= freezes it for a scripted frame capture.
  useEffect(() => {
    if (t !== null || !ready) return;
    setElapsed(0);
    const timers = scene.steps.filter((s) => s.atMs > 0).map((s) => setTimeout(() => setElapsed(s.atMs), s.atMs));
    return () => timers.forEach(clearTimeout);
  }, [scene, ready, t]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === ' ' || e.key === 'ArrowRight' || e.key === 'Enter') {
        e.preventDefault();
        if (!started) setStarted(true);
        else navigate(sceneUrl(runId, neighbourScene(scene.name, 1), query), { replace: true });
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        navigate(sceneUrl(runId, neighbourScene(scene.name, -1), query), { replace: true });
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [runId, scene.name, started, query]);

  const panelId = resolvePanel(scene, query, state);
  const badgeDate = state.replay?.badgeDate || state.run?.created_at || '';
  const [mock, setMock] = useState(false);
  useEffect(() => {
    // The kiosk has no site banner, so a capture of mock data must say so on the stage itself.
    api
      .status()
      .then((s) => setMock(s.engine === 'mock'))
      .catch(() => setMock(false));
  }, []);
  const credit = scene.footage ? (state.run?.footage_credit ?? null) : null;
  const heading = sceneHeading(scene, state, panelId);

  return (
    <RunFiles runId={runId} view="kiosk">
      <div className="kiosk-frame" style={fit ? { transform: `scale(${scale})` } : undefined}>
        <div className="kiosk-stage" style={{ width: WIDTH, height: HEIGHT }}>
          <div className="kiosk-main">
            {ready ? (
              <div className={`kiosk-scene kiosk-${scene.layout}`}>
                <header className="kiosk-head">
                  <span className="wordmark">
                    <Wordmark />
                  </span>
                  {heading ? <span className="kiosk-heading">{heading}</span> : null}
                  {panelId && scene.panel !== null ? <span className="kiosk-panel">{panelName(panelId)}</span> : null}
                </header>
                <SceneBody scene={scene} state={state} panelId={panelId} elapsed={elapsed} t={t} upgraded={query.get('form') === 'upgraded'} runId={runId} lanes={lanes.store} query={query} />
              </div>
            ) : started ? (
              status === 'error' ? <div className="kiosk-start">This run could not be loaded for the kiosk.</div> : null
            ) : (
              <div className="kiosk-start">
                <p>
                  Kiosk for run {runId}, scene {scene.name} ({scene.beat}).
                </p>
                <p className="small">Space or the right arrow starts it; the arrows move between scenes.</p>
              </div>
            )}
          </div>
          {/* The footer band: the footage credit wraps left of the replay badge, and the scene ends above it. */}
          <footer className="kiosk-foot">
            {credit && ready ? <div className="kiosk-credit">{credit}</div> : null}
            <ReplayBadge date={badgeDate} />
          </footer>
          {mock ? <div className="kiosk-mock">mock data from the dev server, not a run</div> : null}
        </div>
      </div>
    </RunFiles>
  );
}

export interface BodyProps {
  scene: Scene;
  state: RunState;
  panelId: string | null;
  elapsed: number;
  t: number | null;
  upgraded: boolean;
  runId: string;
  lanes: ReturnType<typeof useLanes>['store'];
  query: URLSearchParams;
}

interface WalkOptions {
  only?: boolean;
  spot?: boolean;
  startFrame?: number | null;
  label?: (role: string, lane: string, id: string, frame: number) => string;
}

// Exported for the tests, which render a scene's body at a given step without the page's data loading.
export function SceneBody({ scene, state, panelId, elapsed, t, upgraded, runId, lanes, query }: BodyProps) {
  const { colorFor, labelFor, visible, index } = useTrackStyle(state.tracks, state.offCar);
  const roles = rolesOf(state);
  const step = stepAt(scene, elapsed)?.name ?? scene.steps[0]?.name ?? '';
  const color = colorForPanel(panelId);
  const pickupBlurred = state.media['pickup']?.blurredPath ?? null;
  const fps = fpsOf(state, 'pickup');
  const panel = state.panels.find((p) => p.panel_id === panelId);
  const match = state.matches.find((m) => m.panel_id === panelId);
  const kf = panelId ? bestKeyframe(state.keyframes, 'pickup', panelId) : undefined;
  const panelTracks = state.tracks.filter((x) => x.role === 'pickup' && x.panel_id === panelId);
  // The panel's pickup objects at a frame: the filter, the cut-out and the chip agree on the track that holds the frame.
  const panelIds = panelObjectKeys(state.tracks, 'pickup', panelId);
  const onPanel = (lane: string, id: string, frame: number) => panelId !== null && index.at('pickup', lane, id, frame)?.panel_id === panelId;
  const spotlight: Spotlight | null = panelIds.size ? { has: onPanel } : null;
  const frozenFrame = t !== null ? Math.round((t / 1000) * fps) : null;

  const walk = (opts: WalkOptions) =>
    pickupBlurred ? (
      <WalkPlayer
        src={routes.file(runId, pickupBlurred, 'kiosk')}
        role="pickup"
        store={lanes}
        lanes={DRAWN_LANES}
        colorFor={colorFor}
        labelFor={opts.label ?? labelFor}
        visible={visible}
        filterIds={opts.only ? onPanel : undefined}
        spotlight={opts.spot ? spotlight : null}
        seekFrame={frozenFrame !== null ? (opts.startFrame ?? 0) + frozenFrame : null}
        startTime={frozenFrame === null && opts.startFrame ? opts.startFrame / fps : undefined}
        autoPlay={t === null}
        controls={false}
        bloom={t === null}
        labelPx={20}
      />
    ) : (
      <div className="walk-placeholder">The blurred pickup video is not in this run.</div>
    );

  const matchProps = {
    panelId,
    match,
    panel,
    keyframes: state.keyframes,
    marks: state.marks,
    verdicts: state.verdicts,
    videos: state.videos,
    media: state.media,
    run: state.run,
    looks: state.looks,
  };

  switch (scene.layout) {
    case 'seal':
      // A walk's card fills in when its upload.done lands in the replay, not from the final record loaded up front.
      return (
        <div className="kiosk-cols-even">
          {roles.map((role) => (
            <SealCard
              key={role}
              role={role}
              video={state.media[role]?.upload ? (state.videos[role] ?? null) : null}
              media={state.media[role] ?? null}
              seal={state.seal}
              run={state.run}
              failed={state.stage.name === 'failed'}
            />
          ))}
        </div>
      );
    case 'walk':
      return (
        <div className="kiosk-cols">
          <div className="kiosk-walk">
            {walk({})}
            <StageLine state={state} />
          </div>
          <PanelList role="pickup" panels={state.panels} coverage={state.coverage} tracks={state.tracks} lanes={state.lanes['pickup'] ?? {}} selected={null} onSelect={() => undefined} />
        </div>
      );
    case 'fragments':
      return (
        <div className="kiosk-stack">
          <div className="kiosk-walk-narrow">{walk({ only: true, startFrame: panelTracks[0]?.fragments[0]?.first_frame ?? 0 })}</div>
          <FragmentsRow
            role="pickup"
            panelId={panelId}
            tracks={state.tracks}
            fps={fps}
            durationS={state.media['pickup']?.ffprobe?.duration_s ?? null}
            color={color}
            phase={step === 'fragments' ? 'fragments' : step === 'joined' ? 'joined' : 'anchor'}
          />
        </div>
      );
    case 'coverage':
      return <CoverageScene state={state} elapsed={elapsed} step={step} />;
    case 'crops':
      return kf ? (
        <CropFigure kf={kf} marks={state.marks} title="pickup" shown={revealCount(elapsed, CROP_MARKS.fromMs, CROP_MARKS.everyMs)} />
      ) : (
        <div className="empty">No pickup keyframe for this panel.</div>
      );
    case 'moment': {
      // The walk opens 1.5 s before the pickup frame the cut will show (the focus mark's own keyframe).
      const shown = panelId ? paneKeyframes(panelId, match, panelFocus(panelId, match, state.verdicts), new Map(state.marks.map((m) => [m.mark_id, m])), state.keyframes).a : undefined;
      const startKf = shown ?? kf;
      return (
        <Moment
          state={state}
          panelId={panelId}
          step={step}
          elapsed={elapsed}
          upgraded={upgraded}
          still={t !== null}
          walk={walk}
          onPanel={onPanel}
          labelFor={labelFor}
          kfStart={startKf ? Math.max(0, startKf.f_proxy - Math.round(1.5 * fps)) : 0}
          matchProps={matchProps}
        />
      );
    }
    case 'verdict':
      return <MatchPane {...matchProps} showCards={step !== 'frames'} showDates={false} maxCards={KIOSK_CARDS} holdCards />;
    case 'hood':
      return (
        <div className="kiosk-stack">
          <MatchPane {...matchProps} showDates={false} showCards={false} showRefusal={step !== 'frames'} />
          {step === 'line' ? <p className="kiosk-line">Models find and compare; code decides.</p> : null}
        </div>
      );
    case 'claim':
      return <KioskClaim state={state} step={step} />;
    case 'sheet':
      // Section 8 item 5: a kept plate is in the owner's sheet frames only, never on a kiosk route.
      if (state.run?.keep_plate) return <div className="empty">This run kept the plate at the renter's request, so its sheet is not shown on the kiosk.</div>;
      return <SheetPage runId={runId} query={query} kiosk />;
    case 'topup':
      return <TopUpScene state={state} step={step} />;
    case 'blur':
      return <BlurBeat kf={kf} />;
    case 'checks':
      return <HowChecked state={state} big rows={revealCount(elapsed, CHECK_ROWS.fromMs, CHECK_ROWS.everyMs)} />;
    case 'guardrails':
      return (
        <div className="kiosk-stack">
          <LedgerStrip ledger={state.ledger} />
          <PrivacyCard state={state} />
        </div>
      );
  }
}

// B7: the car fills panel by panel in the order the walk reached them; then the slots it missed are named. A pickup
// check names them as the coverage step asked before any top-up (topup.requested), else as coverage.json has them. A
// pair run shows both walks, each with its own gaps, so a panel filmed in one walk only reads as missed in the other.
const SHORT_SHOWN = 6;
const CALLOUTS_SHOWN = 3;

function CoverageScene({ state, elapsed, step }: { state: RunState; elapsed: number; step: string }) {
  const walks = rolesOf(state);
  const fillTicks = Math.floor((COVERAGE_FILL.toMs - COVERAGE_FILL.fromMs) / COVERAGE_FILL.everyMs) + 1;
  const done = step === 'missing';
  const revealedFor = (role: string) => {
    const order = revealOrder(state.panels, role);
    const shown = done ? order.length : Math.ceil((revealCount(elapsed, COVERAGE_FILL.fromMs, COVERAGE_FILL.everyMs) / fillTicks) * order.length);
    return new Set(order.slice(0, shown));
  };
  if (walks.length > 1) {
    return (
      <div className="cov-scene two">
        {walks.map((role) => {
          const gaps = walkGaps(state.coverage, state.panels, role);
          return (
            <div key={role} className="cov-walk">
              <CoverageCar coverage={state.coverage} panels={state.panels} role={role} revealed={revealedFor(role)} callouts={done ? gaps.slice(0, CALLOUTS_SHOWN).map((g) => g.id) : []} title={`${role} walk`} />
              <div className={`cov-short compact ${done ? '' : 'hidden'}`}>
                {gaps.length ? <ShortList gaps={gaps} /> : <p>Every panel was filmed in this walk.</p>}
              </div>
            </div>
          );
        })}
        <p className={`cov-note small muted ${done ? '' : 'hidden'}`}>The sheet lists the same slots, so a gap is never read as a clean panel.</p>
      </div>
    );
  }
  const per = state.coverage?.per_role['pickup'] ?? {};
  const missing = (state.topup?.missing ?? state.coverage?.missing ?? []) as PanelId[];
  const short = missing.map((id) => ({ id, state: per[id]?.state ?? 'not_seen', seconds: per[id]?.seconds ?? 0 }));
  return (
    <div className="cov-scene">
      <CoverageCar coverage={state.coverage} panels={state.panels} role="pickup" revealed={revealedFor('pickup')} callouts={done ? missing.slice(0, CALLOUTS_SHOWN) : []} title="pickup walk" />
      <div className={`cov-short ${done ? '' : 'hidden'}`}>
        {short.length ? <ShortList gaps={short} /> : <p>Every panel was filmed.</p>}
        <p className="small muted">The sheet lists the same slots, so a gap is never read as a clean panel.</p>
      </div>
    </div>
  );
}

function ShortList({ gaps }: { gaps: { id: PanelId; state: CoverageState; seconds: number }[] }) {
  return (
    <>
      <ul>
        {gaps.slice(0, SHORT_SHOWN).map((x) => (
          <li key={x.id} className={x.state}>
            <span className="short-panel">{panelName(x.id)}</span>
            <span className="short-state">
              {COVERAGE_LABEL[x.state]}
              {x.state === 'glimpsed' && x.seconds > 0 ? `, ${x.seconds.toFixed(1)} s` : ''}
            </span>
          </li>
        ))}
      </ul>
      {gaps.length > SHORT_SHOWN ? <p className="small muted">and {gaps.length - SHORT_SHOWN} more</p> : null}
    </>
  );
}

interface MomentProps {
  state: RunState;
  panelId: string | null;
  step: string;
  elapsed: number;
  upgraded: boolean;
  still: boolean;
  walk: (opts: WalkOptions) => ReactNode;
  onPanel: (lane: string, id: string, frame: number) => boolean;
  labelFor: (role: string, lane: string, id: string, frame: number) => string;
  kfStart: number;
  matchProps: Parameters<typeof MatchPane>[0];
}

// B9 in 12 s, on the scene's delay table: the walk dims to the panel with the renter's own words (dim); the mark's
// own track beside a reflection (trail, upgraded form) or the boxed pickup frame (trail, guaranteed form); the match
// cut, an 8-frame dissolve to the return frame placed where the pickup panel was, then the join line (cut, return,
// join); "already there" with the two dates (card). One focus verdict picks the trail's mark, the pair the cut joins
// and the card's label, so the three agree.
const TRAIL_STEP_MS = 3000; // moment's trail step, for the frame a frozen capture shows
const CLIP_FPS = 30; // Stays Put clips are cut at the source 30 fps

function Moment({ state, panelId, step, elapsed, upgraded, still, walk, onPanel, labelFor, kfStart, matchProps }: MomentProps) {
  const markById = new Map(state.marks.map((m) => [m.mark_id, m]));
  const focus = panelFocus(panelId, matchProps.match, state.verdicts);
  const { a: kfA, b: kfB } = panelId ? paneKeyframes(panelId, matchProps.match, focus, markById, state.keyframes) : { a: undefined, b: undefined };
  const markA = focus.pair ? markById.get(focus.pair.pickup_mark) : focus.verdict?.mark_ids.map((id) => markById.get(id)).find((m) => m?.role === 'pickup');
  const markB = focus.pair ? markById.get(focus.pair.return_mark) : undefined;
  const turn = state.narration?.per_role['pickup']?.turns.find((x) => x.pinned_panel === panelId);
  const track = focusTrack(focus, panelId, state.markTracks, markById);
  // Each clip shows only under the verdict its caption claims (the mark fixed, the control sliding), and only once
  // Stays Put is calibrated; until then the trail step is the boxed pickup frame (docs/VIDEO_PLAN.md, the B9 fallback).
  const fixed = staysPutCalibrated(state.run) && track?.verdict === 'fixed' ? track : undefined;
  const control = controlTrack(state.markTracks);
  const sliding = control?.verdict === 'sliding' ? control : undefined;
  const color = colorForPanel(panelId);

  if (step === 'dim') {
    // The renter's words become the label of the panel's outline while the walk dims to it.
    const label = turn ? (role: string, lane: string, id: string, frame: number) => (onPanel(lane, id, frame) ? `"${turn.text}"` : labelFor(role, lane, id, frame)) : undefined;
    return (
      <div className="kiosk-moment">
        {walk({ only: true, spot: true, startFrame: kfStart, label })}
        {turn ? (
          <div className="kiosk-caption">
            <span>
              "{turn.text}"{' '}
              <span className="mono">
                {formatClockTenths(turn.start_ms / 1000)} to {formatClockTenths(turn.end_ms / 1000)}, Muse Voice Transcribe
              </span>
            </span>
          </div>
        ) : null}
      </div>
    );
  }
  if (step === 'trail' && upgraded && fixed) {
    const atFrame = still ? Math.round(((elapsed - TRAIL_STEP_MS) / 1000) * CLIP_FPS) : undefined;
    return (
      <div className="kiosk-moment">
        <StaysPutPair track={fixed} control={sliding} mark={markA} color={color} calibrated atFrame={atFrame} />
      </div>
    );
  }
  if (!kfA || !kfB) return <MatchPane {...matchProps} aligned moment showCards={step === 'card'} />;
  const phase: CutPhase = step === 'trail' || step === 'cut' ? 'pickup' : step === 'return' ? 'return' : 'join';
  // The box Muse Spark placed carries the renter's words when they seeded it, else its type and size.
  const seededBy = markA && turn?.seed_mark_id === markA.mark_id ? turn : undefined;
  const labelA = seededBy ? `"${seededBy.text}"` : markA ? markLabel(markA) : null;
  return (
    <div className="kiosk-moment kiosk-cut">
      <MatchCut kfA={kfA} kfB={kfB} markA={markA} markB={markB} color={color} phase={phase} labelA={labelA} still={still} />
      <div className={`moment-side ${step === 'card' ? '' : 'hidden'}`}>
        {focus.verdict ? <span className={`label ${focus.verdict.label}`}>{VERDICT_LABEL[focus.verdict.label]}</span> : null}
        <DatesCard kfA={kfA} kfB={kfB} videos={state.videos} media={state.media} run={state.run} stacked />
      </div>
    </div>
  );
}

// B14: what the coverage step asked for before the renter drove off, the nouns the top-up ran, and the car after it.
function TopUpScene({ state, step }: { state: RunState; step: string }) {
  const coverage = state.coverage;
  if (!coverage) return <div className="empty">Coverage is not in this run.</div>;
  const ask = state.topup ?? { missing: coverage.missing, topup_text: coverage.topup_text, nouns_to_rerun: coverage.nouns_to_rerun };
  const gains = topupGains(coverage);
  const after = step === 'filmed';
  return (
    <div className="topup-scene">
      <CoverageCar coverage={coverage} panels={state.panels} role="pickup" withTopups={after} callouts={after ? gains.map((g) => g.panel as PanelId) : (ask.missing as PanelId[])} title={after ? 'pickup walk with the top-up clip' : 'pickup walk'} />
      <div className="topup-ask">
        {ask.missing.length ? (
          <>
            <h2>before you drive off</h2>
            <p>{ask.topup_text || `missing: ${ask.missing.map(panelName).join(', ')}`}</p>
          </>
        ) : (
          <p>Every panel was filmed or glimpsed. Nothing to add.</p>
        )}
        {step === 'ask' && coverage.odometer.asked ? <p>odometer: add 3 seconds?</p> : null}
        {step !== 'ask' ? (
          <p className="topup-running mono">
            running: {ask.nouns_to_rerun.join(', ')} ({ask.nouns_to_rerun.length} of 16 lanes)
          </p>
        ) : null}
        {after ? (
          <>
            <p className="topup-done">
              {gains.length ? gains.map((g) => `${panelName(g.panel)}: ${COVERAGE_LABEL[g.state]}, ${g.seconds.toFixed(1)} s`).join('; ') : 'the top-up clip added no panel'}
              {coverage.rerun_seconds !== null ? `; the top-up lanes ran in ${coverage.rerun_seconds.toFixed(0)} s` : ''}
            </p>
            {coverage.odometer.asked ? <p>odometer: {coverage.odometer.filmed ? 'filmed' : 'not filmed'}</p> : null}
          </>
        ) : null}
      </div>
    </div>
  );
}

const CLAIM_ROWS: Record<string, number> = { photo: 0, spark: 1, sam: 2, frame: 5, size: 7, second: 7 };

// B12: the staged claim photo placed against the pickup walk, fact by fact on the scene's table, then a second
// photo the flow refuses to place, when the run has one. The claims are the run's own (loader.loadClaims, read when
// the run ends): the main one is the first that stopped at no exit (E2b counts as an exit), the second the next one
// that did.
function KioskClaim({ state, step }: { state: RunState; step: string }) {
  const url = useFileUrl();
  const claims = state.claims;
  if (!claims.length) return state.completed ? <div className="empty">This run has no claim photo.</div> : <p className="muted">loading the claim</p>;
  const main = claims.find((c) => !effectiveExit(c)) ?? claims[0]!;
  const second = claims.find((c) => c !== main && effectiveExit(c));
  const ctx: ClaimContext = { keyframes: state.keyframes, panels: state.panels, marks: state.marks, coverage: state.coverage, videos: state.videos };
  const photo = (c: Claim) => {
    const rel = claimPhotoPath(c);
    return rel ? url(rel) : null;
  };
  const rows = CLAIM_ROWS[step] ?? 0;
  return (
    <div className="claim-scene">
      <ClaimPhoto claim={main} src={photo(main)} staged />
      <div className="claim-scene-right">
        {rows ? <ClaimSteps claim={main} upTo={rows} /> : null}
        {rows >= 5 ? <ClaimFrame claim={main} ctx={ctx} /> : null}
      </div>
      {step === 'second' && second ? (
        <div className="claim-scene-second">
          <ClaimPhoto claim={second} src={photo(second)} staged note="Staged by us, to show a photo the flow refuses to place." />
          <ClaimExit claim={second} />
        </div>
      ) : null}
    </div>
  );
}
