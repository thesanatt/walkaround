import { useCallback, useEffect, useMemo, useState } from 'react';

import { routes } from '../api/client.ts';
import { BlurBeat } from '../components/BlurBeat.tsx';
import { CoverageList } from '../components/CoverageList.tsx';
import { bestKeyframe, CropView } from '../components/CropView.tsx';
import { FragmentsRow } from '../components/FragmentsRow.tsx';
import { LedgerStrip } from '../components/LedgerStrip.tsx';
import { MatchPane } from '../components/MatchPane.tsx';
import { PanelList } from '../components/PanelList.tsx';
import { PrivacyCard } from '../components/PrivacyCard.tsx';
import { ReplayBadge } from '../components/ReplayBadge.tsx';
import { SealCard } from '../components/SealCard.tsx';
import { StageLine } from '../components/StageLine.tsx';
import { staysPutLine } from '../components/StaysPutList.tsx';
import { WalkPlayer, type Spotlight } from '../components/WalkPlayer.tsx';
import type { PanelId } from '../contracts/types.ts';
import { neighbourScene, sceneByName, stepAt, type Scene } from '../kiosk/scenes.ts';
import { useLanes } from '../lanes/useLanes.ts';
import { colorForNoun, colorForPanel, DRAWN_NOUNS, panelName } from '../lib/panels.ts';
import { navigate } from '../lib/router.tsx';
import type { RunSource } from '../state/useRun.ts';
import { useRun } from '../state/useRun.ts';
import { ClaimPage } from './ClaimPage.tsx';
import { rolesOf, trackLookup } from './RunPage.tsx';
import { SheetPage } from './SheetPage.tsx';

function sceneUrl(runId: string, scene: Scene, extra: URLSearchParams): string {
  const q = new URLSearchParams(extra);
  q.set('scene', scene.name);
  q.set('autostart', '1');
  q.delete('t');
  return `/kiosk/${encodeURIComponent(runId)}?${q.toString()}`;
}

// 1920 x 1080, no chrome, cursor hidden, the badge fixed bottom right, one keypress per step.
// ?scene=<name>&autostart=1 starts a scene by URL; ?t=<ms> renders the replay state at a virtual time for a
// frame-by-frame capture; ?form=upgraded adds the Stays Put line to the moment; ?panel= overrides the scene's panel.
export function KioskPage({ runId, query }: { runId: string; query: URLSearchParams }) {
  const scene = sceneByName(query.get('scene'));
  const autostart = query.get('autostart') === '1';
  const tRaw = query.get('t');
  const t = tRaw !== null && Number.isFinite(Number(tRaw)) ? Number(tRaw) : null;
  const upgraded = query.get('form') === 'upgraded';
  const panelId = (query.get('panel') as PanelId | null) ?? scene.panel;
  const [started, setStarted] = useState(autostart || t !== null);
  const [elapsed, setElapsed] = useState(t ?? 0);

  const source = useMemo<RunSource>(() => ({ kind: 'replay', mode: 'fixed', untilMs: t ?? undefined }), [t]);
  const { state } = useRun(started ? runId : null, source);
  const lanes = useLanes(started ? runId : null, state);
  const tracks = useMemo(() => trackLookup(state.tracks), [state.tracks]);

  useEffect(() => {
    document.body.classList.add('kiosk');
    return () => document.body.classList.remove('kiosk');
  }, []);

  // The scene clock: wall time while recording, frozen at ?t= for a scripted frame capture.
  useEffect(() => {
    if (t !== null || !started) return;
    const start = performance.now();
    setElapsed(0);
    const handle = setInterval(() => setElapsed(performance.now() - start), 100);
    return () => clearInterval(handle);
  }, [scene.name, started, t]);

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

  const colorFor = useCallback(
    (role: string, noun: string, id: string) => {
      const tr = tracks.get(`${role}|${noun}|${id}`);
      return tr?.panel_id ? colorForPanel(tr.panel_id) : colorForNoun(noun);
    },
    [tracks],
  );
  const labelFor = useCallback(
    (role: string, noun: string, id: string) => {
      const tr = tracks.get(`${role}|${noun}|${id}`);
      return tr?.panel_id ? panelName(tr.panel_id) : `${noun} ${id}`;
    },
    [tracks],
  );

  const roles = rolesOf(state);
  const pickupProxy = state.videos['pickup']?.proxy_blurred?.path ?? null;
  const panel = state.panels.find((p) => p.panel_id === panelId);
  const match = state.matches.find((m) => m.panel_id === panelId);
  const step = stepAt(scene, elapsed);
  const spotlight: Spotlight | null = useMemo(() => {
    if (!panelId) return null;
    const tr = state.tracks.find((x) => x.role === 'pickup' && x.panel_id === panelId);
    const id = tr?.sam_object_ids[0];
    return tr && id ? { noun: tr.noun, id } : null;
  }, [state.tracks, panelId]);

  const walk = (spot: Spotlight | null) =>
    pickupProxy ? (
      <WalkPlayer
        src={routes.file(runId, pickupProxy)}
        role="pickup"
        store={lanes.store}
        nouns={DRAWN_NOUNS}
        colorFor={colorFor}
        labelFor={labelFor}
        spotlight={spot}
        autoPlay
        loop
        controls={false}
        bloom={t === null}
      />
    ) : (
      <div className="walk-placeholder">the blurred proxy is not in this run yet</div>
    );

  const matchProps = { runId, panelId, match, panel, keyframes: state.keyframes, marks: state.marks, verdicts: state.verdicts, videos: state.videos, run: state.run };

  let body;
  switch (scene.layout) {
    case 'upload':
      body = (
        <div className="kiosk-cols-even">
          {roles.map((role) => (
            <SealCard key={role} role={role} record={state.videos[role] ?? null} upload={state.uploads[role] ?? null} seal={role === 'pickup' ? state.seal : undefined} run={state.run} />
          ))}
        </div>
      );
      break;
    case 'walk':
      body = (
        <div className="kiosk-cols">
          <div>
            {walk(null)}
            <StageLine state={state} />
          </div>
          <PanelList role="pickup" panels={state.panels} coverage={state.coverage} tracks={state.tracks} lanes={state.lanes['pickup'] ?? {}} selected={panelId} onSelect={() => undefined} />
        </div>
      );
      break;
    case 'fragments':
      body = (
        <FragmentsRow role="pickup" panelId={panelId} tracks={state.tracks} fps={state.videos['pickup']?.proxy?.fps ?? null} durationS={state.videos['pickup']?.ffprobe?.duration_s ?? null} color={colorForPanel(panelId)} />
      );
      break;
    case 'coverage':
      body = <CoverageList coverage={state.coverage} roles={roles} />;
      break;
    case 'crops':
      body = (
        <div className="kiosk-cols-even">
          <CropView runId={runId} role="pickup" panelId={panelId} keyframes={state.keyframes} marks={state.marks} />
        </div>
      );
      break;
    case 'moment': {
      const turn = state.narration?.per_role['pickup']?.turns.find((x) => x.pinned_panel === panelId);
      const mt = state.markTracks.find((x) => x.role === 'pickup' && !x.is_control && x.verdict === 'fixed');
      const control = state.markTracks.find((x) => x.is_control);
      if (!step || step.name === 'dim') {
        body = (
          <div>
            {walk(spotlight)}
            {turn ? (
              <div className="kiosk-caption">
                <span>
                  {turn.text} <span className="mono small">({(turn.start_ms / 1000).toFixed(1)} s to {(turn.end_ms / 1000).toFixed(1)} s, Muse Voice Transcribe)</span>
                </span>
              </div>
            ) : null}
          </div>
        );
      } else if (step.name === 'trail') {
        body = (
          <div>
            {walk(spotlight)}
            <div className="kiosk-caption">
              <span>
                {upgraded && mt ? `${step.caption ?? ''}; ${staysPutLine(mt)}${control ? `; control ${staysPutLine(control)}` : ''}` : (step.caption ?? '')}
              </span>
            </div>
          </div>
        );
      } else {
        body = <MatchPane {...matchProps} aligned showCards={step.name === 'card'} showDates={step.name === 'card'} />;
      }
      break;
    }
    case 'verdict':
      body = <MatchPane {...matchProps} />;
      break;
    case 'hood':
      body = (
        <div>
          <MatchPane {...matchProps} showDates={false} />
          <div className="kiosk-caption">
            <span>models find and compare, code decides</span>
          </div>
        </div>
      );
      break;
    case 'claim':
      body = <ClaimPage query={new URLSearchParams({ run: runId })} />;
      break;
    case 'sheet':
      body = <SheetPage runId={runId} query={query} kiosk />;
      break;
    case 'topup':
      body = (
        <div className="kiosk-cols-even">
          <CoverageList coverage={state.coverage} roles={['pickup']} />
          <StageLine state={state} />
        </div>
      );
      break;
    case 'blur':
      body = (
        <div className="kiosk-cols-even">
          <BlurBeat runId={runId} kf={panelId ? bestKeyframe(state.keyframes, 'pickup', panelId) : undefined} />
        </div>
      );
      break;
    case 'guardrails':
      body = (
        <div>
          <LedgerStrip ledger={state.ledger} />
          <PrivacyCard deletions={state.deletions} run={state.run} />
        </div>
      );
      break;
  }

  return (
    <div className="kiosk-stage">
      {started ? (
        <div className="kiosk-scene">
          <div>
            {scene.title ? <h1>{scene.title}</h1> : null}
            <span className="small muted mono">
              {scene.beat} {scene.name}
              {panelId ? `, ${panelName(panelId)}` : ''}
            </span>
          </div>
          {body}
        </div>
      ) : (
        <div className="kiosk-start">
          <p>kiosk for run {runId}</p>
          <p className="small">press space to start the replay on scene {scene.name}; arrows move between scenes</p>
        </div>
      )}
      <ReplayBadge date={state.replay?.badgeDate ?? state.run?.created_at ?? ''} />
    </div>
  );
}
