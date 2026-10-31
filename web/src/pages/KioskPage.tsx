import { useEffect, useMemo, useState, type ReactNode } from 'react';

import { api, routes } from '../api/client.ts';
import { BlurBeat } from '../components/BlurBeat.tsx';
import { ClipTrail } from '../components/ClipTrail.tsx';
import { CoverageList } from '../components/CoverageList.tsx';
import { bestKeyframe, CropFigure } from '../components/CropView.tsx';
import { FragmentsRow } from '../components/FragmentsRow.tsx';
import { LedgerStrip } from '../components/LedgerStrip.tsx';
import { MatchPane } from '../components/MatchPane.tsx';
import { PanelList } from '../components/PanelList.tsx';
import { PrivacyCard } from '../components/PrivacyCard.tsx';
import { ReplayBadge } from '../components/ReplayBadge.tsx';
import { SealCard } from '../components/SealCard.tsx';
import { StageLine } from '../components/StageLine.tsx';
import { WalkPlayer, type Spotlight } from '../components/WalkPlayer.tsx';
import type { Claim } from '../contracts/types.ts';
import { neighbourScene, sceneByName, stepAt, type Scene } from '../kiosk/scenes.ts';
import { useLanes } from '../lanes/useLanes.ts';
import { RunFiles, useFileUrl } from '../lib/files.tsx';
import { focusTrack, paneKeyframes, panelFocus, panelObjectKeys } from '../lib/focus.ts';
import { formatClockTenths } from '../lib/format.ts';
import { colorForPanel, DRAWN_LANES, panelName } from '../lib/panels.ts';
import { navigate } from '../lib/router.tsx';
import { fpsOf, rolesOf } from '../state/reducer.ts';
import type { RunState } from '../state/types.ts';
import { useRun, type RunSource } from '../state/useRun.ts';
import { ClaimResult } from './ClaimPage.tsx';
import { defaultPanel, useTrackStyle } from './RunPage.tsx';
import { SheetPage } from './SheetPage.tsx';

const WIDTH = 1920;
const HEIGHT = 1080;

function sceneUrl(runId: string, scene: Scene, query: URLSearchParams): string {
  const q = new URLSearchParams(query);
  q.set('scene', scene.name);
  q.set('autostart', '1');
  q.delete('t');
  q.delete('panel');
  return `/kiosk/${encodeURIComponent(runId)}?${q.toString()}`;
}

function sourceFor(scene: Scene, t: number | null): RunSource {
  if (scene.replay === 'final') return { kind: 'replay', mode: 'instant' };
  return { kind: 'replay', mode: 'fixed', untilMs: t ?? undefined, startAtStage: scene.replay === 'from_tracking' ? 'tracking' : undefined };
}

// The scene's panel: ?panel=, else the scene's default when the run has a keyframe for it, else the run's first match.
function resolvePanel(scene: Scene, query: URLSearchParams, state: RunState): string | null {
  const asked = query.get('panel');
  if (asked) return asked;
  if (scene.panel && state.keyframes.some((k) => k.panel_id === scene.panel)) return scene.panel;
  return defaultPanel(state) ?? scene.panel;
}

// 1920 x 1080, no page chrome, cursor hidden, the replay badge fixed bottom right on every scene.
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

  const source = useMemo(() => sourceFor(scene, t), [scene, t]);
  const { state, status, ended } = useRun(started ? runId : null, source, 'kiosk');
  const lanes = useLanes(started ? runId : null, state, ['pickup', 'return'], 'kiosk');
  // A final-state scene waits for the whole run to load; a paced one starts with its first event.
  const ready = started && (scene.replay === 'final' ? ended : status === 'open' || ended);

  useEffect(() => {
    document.body.classList.add('kiosk');
    return () => document.body.classList.remove('kiosk');
  }, []);

  useEffect(() => {
    if (!fit) return;
    const apply = () => setScale(Math.min(1, window.innerWidth / WIDTH, window.innerHeight / HEIGHT));
    apply();
    window.addEventListener('resize', apply);
    return () => window.removeEventListener('resize', apply);
  }, [fit]);

  // The scene clock starts when the data is ready and only moves at step boundaries, so a 60 fps capture never re-renders
  // the stage between steps. ?t= freezes it for a scripted frame capture.
  useEffect(() => {
    if (t !== null || !ready) return;
    setElapsed(0);
    const timers = (scene.steps ?? []).filter((s) => s.atMs > 0).map((s) => setTimeout(() => setElapsed(s.atMs), s.atMs));
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

  return (
    <RunFiles runId={runId} view="kiosk">
      <div className="kiosk-frame" style={fit ? { transform: `scale(${scale})` } : undefined}>
        <div className="kiosk-stage" style={{ width: WIDTH, height: HEIGHT }}>
          {ready ? (
            <div className={`kiosk-scene kiosk-${scene.layout}`}>
              <header className="kiosk-head">
                <span className="wordmark">Walkaround</span>
                {scene.heading ? <span className="kiosk-heading">{scene.heading}</span> : null}
                {panelId && scene.panel !== null ? <span className="kiosk-panel">{panelName(panelId)}</span> : null}
              </header>
              <SceneBody scene={scene} state={state} panelId={panelId} elapsed={elapsed} t={t} upgraded={query.get('form') === 'upgraded'} runId={runId} lanes={lanes.store} query={query} />
            </div>
          ) : started ? null : (
            <div className="kiosk-start">
              <p>Kiosk for run {runId}, scene {scene.name} ({scene.beat}).</p>
              <p className="small">Space or the right arrow starts it; the arrows move between scenes.</p>
            </div>
          )}
          <ReplayBadge date={badgeDate} />
          {mock ? <div className="kiosk-mock">mock data from the dev server, not a run</div> : null}
        </div>
      </div>
    </RunFiles>
  );
}

interface BodyProps {
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

function SceneBody({ scene, state, panelId, elapsed, t, upgraded, runId, lanes, query }: BodyProps) {
  const { colorFor, labelFor } = useTrackStyle(state.tracks);
  const roles = rolesOf(state);
  const color = colorForPanel(panelId);
  const pickupBlurred = state.media['pickup']?.blurredPath ?? null;
  const fps = fpsOf(state, 'pickup');
  const panel = state.panels.find((p) => p.panel_id === panelId);
  const match = state.matches.find((m) => m.panel_id === panelId);
  const kf = panelId ? bestKeyframe(state.keyframes, 'pickup', panelId) : undefined;
  const panelTracks = state.tracks.filter((x) => x.role === 'pickup' && x.panel_id === panelId);
  // Every SAM id of the panel's pickup tracks: the filter and the cut-out agree on any fragment the frame is on.
  const panelIds = panelObjectKeys(state.tracks, 'pickup', panelId);
  const spotlight: Spotlight | null = panelIds.size ? { keys: panelIds } : null;
  const frozenFrame = t !== null ? Math.round((t / 1000) * fps) : null;

  const walk = (opts: { only?: boolean; spot?: boolean; startFrame?: number | null }) =>
    pickupBlurred ? (
      <WalkPlayer
        src={routes.file(runId, pickupBlurred, 'kiosk')}
        role="pickup"
        store={lanes}
        lanes={DRAWN_LANES}
        colorFor={colorFor}
        labelFor={labelFor}
        filterIds={opts.only ? (lane, id) => panelIds.has(`${lane}|${id}`) : undefined}
        spotlight={opts.spot ? spotlight : null}
        seekFrame={frozenFrame !== null ? (opts.startFrame ?? 0) + frozenFrame : null}
        startTime={frozenFrame === null && opts.startFrame ? opts.startFrame / fps : undefined}
        autoPlay={t === null}
        controls={false}
        bloom={t === null}
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
  };

  switch (scene.layout) {
    case 'seal':
      return (
        <div className="kiosk-cols-even">
          {roles.map((role) => (
            <SealCard key={role} role={role} video={state.videos[role] ?? null} media={state.media[role] ?? null} seal={state.seal} run={state.run} failed={state.stage.name === 'failed'} />
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
          <FragmentsRow role="pickup" panelId={panelId} tracks={state.tracks} fps={fps} durationS={state.media['pickup']?.ffprobe?.duration_s ?? null} color={color} />
        </div>
      );
    case 'coverage':
      return <CoverageList coverage={state.coverage} roles={roles} />;
    case 'crops':
      return kf ? <CropFigure kf={kf} marks={state.marks} title="pickup" /> : <div className="empty">No pickup keyframe for this panel.</div>;
    case 'moment': {
      // The walk opens 1.5 s before the pickup frame the cut will show (the focus mark's own keyframe).
      const shown = panelId ? paneKeyframes(panelId, match, panelFocus(panelId, match, state.verdicts), new Map(state.marks.map((m) => [m.mark_id, m])), state.keyframes).a : undefined;
      const startKf = shown ?? kf;
      return <Moment state={state} panelId={panelId} elapsed={elapsed} scene={scene} upgraded={upgraded} walk={walk} kfStart={startKf ? Math.max(0, startKf.f_proxy - Math.round(1.5 * fps)) : 0} matchProps={matchProps} />;
    }
    case 'verdict':
      return <MatchPane {...matchProps} />;
    case 'hood':
      return (
        <div className="kiosk-stack">
          <MatchPane {...matchProps} showDates={false} />
          <p className="kiosk-line">Models find and compare; code decides.</p>
        </div>
      );
    case 'claim':
      return <KioskClaim runId={runId} />;
    case 'sheet':
      // Section 8 item 5: a kept plate is in the owner's sheet frames only, never on a kiosk route.
      if (state.run?.keep_plate) return <div className="empty">This run kept the plate at the renter's request, so its sheet is not shown on the kiosk.</div>;
      return <SheetPage runId={runId} query={query} kiosk />;
    case 'topup':
      return (
        <div className="kiosk-cols-even">
          <CoverageList coverage={state.coverage} roles={['pickup']} showMissing={false} />
          <TopUpAsk coverage={state.coverage} />
        </div>
      );
    case 'blur':
      return <BlurBeat kf={kf} />;
    case 'guardrails':
      return (
        <div className="kiosk-stack">
          <LedgerStrip ledger={state.ledger} />
          <PrivacyCard state={state} />
        </div>
      );
  }
}

interface MomentProps {
  state: RunState;
  panelId: string | null;
  elapsed: number;
  scene: Scene;
  upgraded: boolean;
  walk: (opts: { only?: boolean; spot?: boolean; startFrame?: number | null }) => ReactNode;
  kfStart: number;
  matchProps: Parameters<typeof MatchPane>[0];
}

// B9 in 12 s: the walk dims to the panel with the renter's own words (0 s), the mark's track (3 s, upgraded form), the
// match cut with the join line (7 s), the "already there" card with both dates (10 s). One focus verdict (its first
// "already there") picks the trail's mark, the pair the cut aligns on and the card's label, so the three agree.
function Moment({ state, panelId, elapsed, scene, upgraded, walk, kfStart, matchProps }: MomentProps) {
  const step = stepAt(scene, elapsed)?.name ?? 'dim';
  const turn = state.narration?.per_role['pickup']?.turns.find((x) => x.pinned_panel === panelId);
  const markById = new Map(state.marks.map((m) => [m.mark_id, m]));
  const focus = panelFocus(panelId, matchProps.match, state.verdicts);
  const track = focusTrack(focus, panelId, state.markTracks, markById);
  // The caption states what Stays Put decided, so each clip shows only under the verdict that caption claims.
  const fixed = track?.verdict === 'fixed' ? track : undefined;
  // The control is on purpose another panel: a reflection on the hood, tracked the same way.
  const control = state.markTracks.find((m) => m.is_control && m.clip && m.verdict === 'sliding');
  const mark = fixed ? markById.get(fixed.mark_id) : undefined;
  const color = colorForPanel(panelId);

  const type = mark?.type ?? 'mark';
  if (step === 'trail' && upgraded && fixed) {
    // The upgraded form: the mark's own track with its trail, the reflection control beside it, one caption, no numbers.
    return (
      <div className="kiosk-moment">
        <div className="kiosk-trail">
          <ClipTrail mt={fixed} color={color} caption={`the ${type}, tracked through its own clip`} bare />
          {control ? <ClipTrail mt={control} color="#d9a84a" caption="a reflection on the hood, tracked the same way" bare /> : null}
        </div>
        <div className="kiosk-caption">
          <span>{control ? `the ${type} stays on the paint; the reflection slides off` : `the ${type} stays on the paint`}</span>
        </div>
      </div>
    );
  }
  if (step === 'dim' || step === 'trail') {
    return (
      <div className="kiosk-moment">
        {walk({ only: true, spot: true, startFrame: kfStart })}
        {step === 'dim' && turn ? (
          <div className="kiosk-caption">
            <span>
              "{turn.text}" <span className="mono">{formatClockTenths(turn.start_ms / 1000)} to {formatClockTenths(turn.end_ms / 1000)}, Muse Voice Transcribe</span>
            </span>
          </div>
        ) : null}
      </div>
    );
  }
  return <MatchPane {...matchProps} aligned moment showCards={step === 'card'} />;
}

// B14: what the coverage step asks for before the renter drives off, and what the top-up re-ran.
function TopUpAsk({ coverage }: { coverage: RunState['coverage'] }) {
  if (!coverage) return <div className="empty">Coverage is not in this run.</div>;
  if (!coverage.missing.length) return <div className="topup-ask"><p>Every panel was filmed or glimpsed. Nothing to add.</p></div>;
  return (
    <div className="topup-ask">
      <h2>before you drive off</h2>
      {coverage.topup_text ? <p>{coverage.topup_text}</p> : <p>missing: {coverage.missing.map(panelName).join(', ')}</p>}
      {coverage.odometer.asked ? <p>odometer: {coverage.odometer.filmed ? 'filmed' : 'not filmed, add 3 seconds?'}</p> : null}
      <p className="small muted">
        Only these lanes run on the new clip: {coverage.nouns_to_rerun.join(', ')} ({coverage.nouns_to_rerun.length} of 16).
        {coverage.rerun_seconds !== null ? ` They ran in ${coverage.rerun_seconds.toFixed(0)} s.` : ''}
      </p>
    </div>
  );
}

// B12: the stored claim (derived/claim.json) with its photo, when the run has one.
function KioskClaim({ runId }: { runId: string }) {
  const url = useFileUrl();
  const [claim, setClaim] = useState<Claim | null | 'missing'>(null);
  useEffect(() => {
    let live = true;
    api
      .file<Claim>(runId, 'derived/claim.json', 'kiosk')
      .then((c) => live && setClaim(c))
      .catch(() => live && setClaim('missing'));
    return () => {
      live = false;
    };
  }, [runId]);
  if (claim === null) return <p className="muted">loading the claim</p>;
  if (claim === 'missing') return <div className="empty">This run has no claim photo.</div>;
  return (
    <div className="kiosk-cols-even">
      <figure className="crop">
        <img src={url('claim/claim_1.jpg')} alt="the claim photo, staged: our own photo of the same mark at return" />
        <figcaption>Claim photo: staged. We took this photo of the same mark at return.</figcaption>
      </figure>
      <ClaimResult claim={claim} />
    </div>
  );
}

