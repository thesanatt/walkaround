import { useCallback, useMemo, useState } from 'react';

import { routes } from '../api/client.ts';
import { useStatus } from '../api/useStatus.ts';
import { CoverageList } from '../components/CoverageList.tsx';
import { CropView } from '../components/CropView.tsx';
import { FragmentsRow } from '../components/FragmentsRow.tsx';
import { LedgerStrip } from '../components/LedgerStrip.tsx';
import { MatchPane } from '../components/MatchPane.tsx';
import { NarrationChips } from '../components/NarrationChips.tsx';
import { PanelList } from '../components/PanelList.tsx';
import { PrivacyCard } from '../components/PrivacyCard.tsx';
import { QueueLine } from '../components/QueueLine.tsx';
import { ReplayBadge } from '../components/ReplayBadge.tsx';
import { SealCard } from '../components/SealCard.tsx';
import { StageLine } from '../components/StageLine.tsx';
import { StaysPutList } from '../components/StaysPutList.tsx';
import { TopUpBanner } from '../components/TopUpBanner.tsx';
import { VerdictList } from '../components/VerdictList.tsx';
import { WalkPlayer } from '../components/WalkPlayer.tsx';
import type { Coverage, PanelTrack, ReplayMode } from '../contracts/types.ts';
import { useLanes } from '../lanes/useLanes.ts';
import { colorForNoun, colorForPanel, DRAWN_NOUNS, panelName } from '../lib/panels.ts';
import { Link, navigate } from '../lib/router.tsx';
import type { RunState } from '../state/types.ts';
import { useRun, type RunSource } from '../state/useRun.ts';

export function replayModeOf(query: URLSearchParams): ReplayMode | null {
  const v = query.get('replay');
  if (v === 'fixed' || v === 'recorded' || v === 'instant') return v;
  if (v === '1' || v === 'true') return 'fixed';
  return null;
}

export function rolesOf(state: RunState): string[] {
  if (state.run?.kind === 'pair') return ['pickup', 'return'];
  if (state.run?.kind === 'pickup_check') return ['pickup'];
  const known = Object.keys(state.videos);
  return known.length ? known.filter((r) => r === 'pickup' || r === 'return') : ['pickup', 'return'];
}

// The first panel worth looking at: a comparable match, else a filmed panel, else nothing.
export function defaultPanel(state: RunState): string | null {
  const already = state.verdicts.find((v) => v.label === 'already_there');
  if (already) return already.panel_id;
  const comparable = state.matches.find((m) => m.state === 'comparable');
  if (comparable) return comparable.panel_id;
  for (const role of ['pickup', 'return']) {
    const per = state.coverage?.per_role[role];
    if (!per) continue;
    for (const [id, cov] of Object.entries(per)) if (cov?.state === 'filmed') return id;
  }
  const tracked = state.tracks.find((t) => t.panel_id);
  return tracked?.panel_id ?? null;
}

export function trackLookup(tracks: PanelTrack[]): Map<string, PanelTrack> {
  const m = new Map<string, PanelTrack>();
  for (const t of tracks) for (const id of t.sam_object_ids) m.set(`${t.role}|${t.noun}|${id}`, t);
  return m;
}

export function RunPage({ runId, query }: { runId: string; query: URLSearchParams }) {
  const mode = replayModeOf(query);
  const source = useMemo<RunSource>(() => (mode ? { kind: 'replay', mode } : { kind: 'live' }), [mode]);
  const { state, status: driver, detail } = useRun(runId, source);
  const lanes = useLanes(runId, state);
  const { status } = useStatus(mode ? null : runId);
  const [selected, setSelected] = useState<string | null>(null);
  const [roleTab, setRoleTab] = useState('pickup');
  const [seek, setSeek] = useState<Record<string, number | null>>({});
  const [coverageOverride, setCoverageOverride] = useState<Coverage | null>(null);

  const roles = rolesOf(state);
  const panelId = selected ?? defaultPanel(state);
  const tracks = useMemo(() => trackLookup(state.tracks), [state.tracks]);
  const coverage = coverageOverride ?? state.coverage;
  const panel = state.panels.find((p) => p.panel_id === panelId);
  const match = state.matches.find((m) => m.panel_id === panelId);

  const colorFor = useCallback(
    (role: string, noun: string, id: string) => {
      const t = tracks.get(`${role}|${noun}|${id}`);
      return t?.panel_id ? colorForPanel(t.panel_id) : colorForNoun(noun);
    },
    [tracks],
  );
  const labelFor = useCallback(
    (role: string, noun: string, id: string) => {
      const t = tracks.get(`${role}|${noun}|${id}`);
      return t?.panel_id ? panelName(t.panel_id) : `${noun} ${id}`;
    },
    [tracks],
  );

  const onSeek = (role: string, seconds: number) => {
    const fps = state.videos[role]?.proxy?.fps ?? state.run?.fps ?? 15;
    setSeek((s) => ({ ...s, [role]: Math.round(seconds * fps) }));
  };

  const setMode = (value: string) => {
    const base = `/run/${encodeURIComponent(runId)}`;
    navigate(value === 'live' ? base : `${base}?replay=${value}`, { replace: true });
  };

  return (
    <div className="run-page">
      <div className="run-head">
        <h1>
          run <span className="mono">{runId}</span>
        </h1>
        {state.run ? (
          <span className="small muted">
            {state.run.kind.replace('_', ' ')}, created {state.run.created_at.slice(0, 10)}
          </span>
        ) : null}
        <div className="controls">
          <label>
            timings{' '}
            <select value={mode ?? 'live'} onChange={(e) => setMode(e.target.value)}>
              <option value="live">live (as they happen)</option>
              <option value="fixed">replay, fixed delays</option>
              <option value="recorded">replay, real timings (gaps capped at 2 s)</option>
              <option value="instant">replay, instant</option>
            </select>
          </label>
          <Link to={`/run/${encodeURIComponent(runId)}/sheet`}>sheet</Link>
          <Link to={`/claim?run=${encodeURIComponent(runId)}`}>claim photo</Link>
        </div>
      </div>
      <p className="driver-line">
        {driver === 'connecting' ? 'connecting' : driver === 'reconnecting' ? 'reconnecting; the run URL is your receipt and this page resumes where it left off' : driver === 'error' ? `could not load this run: ${detail ?? 'unknown error'}` : null}
      </p>
      <StageLine state={state} />
      <LedgerStrip ledger={state.ledger} />
      <QueueLine status={status} />
      {state.partial ? <p className="small">incomplete: {state.partial.join(', ')} missing; the sheet still shows what was tracked.</p> : null}
      {state.errors.map((e, i) => (
        <p key={i} className="error-line small">
          {e}
        </p>
      ))}

      <div className="walks">
        {roles.map((role) => {
          const video = state.videos[role];
          const proxy = video?.proxy_blurred?.path ?? null;
          return (
            <div key={role} className="walk-block">
              <div className="walk-title">
                <h2>{role} walk</h2>
                <span className="small muted">{video?.ffprobe ? `${video.ffprobe.display_width}x${video.ffprobe.display_height}, ${video.ffprobe.duration_s.toFixed(0)} s` : ''}</span>
              </div>
              {proxy ? (
                <WalkPlayer
                  src={routes.file(runId, proxy)}
                  role={role}
                  store={lanes.store}
                  nouns={DRAWN_NOUNS}
                  colorFor={colorFor}
                  labelFor={labelFor}
                  seekFrame={seek[role] ?? null}
                  autoPlay={false}
                  loop
                />
              ) : (
                <div className="walk-placeholder">
                  {video ? 'the blurred proxy appears after the blur step; the browser never receives the unblurred video' : `the ${role} video has not been received yet`}
                </div>
              )}
              <SealCard role={role} record={video ?? null} upload={state.uploads[role] ?? null} seal={role === 'pickup' ? state.seal : undefined} run={state.run} />
            </div>
          );
        })}
      </div>

      <div className="run-grid">
        <div>
          <div className="section">
            <h2>track</h2>
            <div className="role-tabs">
              {roles.map((r) => (
                <button key={r} type="button" className={roleTab === r ? 'on' : ''} onClick={() => setRoleTab(r)}>
                  {r}
                </button>
              ))}
            </div>
            <FragmentsRow
              role={roleTab}
              panelId={panelId}
              tracks={state.tracks}
              fps={state.videos[roleTab]?.proxy?.fps ?? state.run?.fps ?? null}
              durationS={state.videos[roleTab]?.ffprobe?.duration_s ?? null}
              color={colorForPanel(panelId)}
            />
          </div>
          <div className="section">
            <h2>coverage</h2>
            <p className="section-note">filmed, glimpsed, not filmed, not run (the model did not answer), not seen (no track in either walk)</p>
            <CoverageList coverage={coverage} roles={roles} />
            {!mode ? <TopUpBanner runId={runId} coverage={coverage} onCoverage={setCoverageOverride} /> : null}
          </div>
        </div>
        <aside className="section">
          <h2>panels</h2>
          <div className="role-tabs">
            {roles.map((r) => (
              <button key={r} type="button" className={roleTab === r ? 'on' : ''} onClick={() => setRoleTab(r)}>
                {r}
              </button>
            ))}
          </div>
          <PanelList role={roleTab} panels={state.panels} coverage={coverage} tracks={state.tracks} lanes={state.lanes[roleTab] ?? {}} selected={panelId} onSelect={setSelected} />
        </aside>
      </div>

      <div className="section">
        <h2>crops{panelId ? `: ${panelName(panelId)}` : ''}</h2>
        <p className="section-note">Muse Spark sees one blurred crop at a time and never both videos together. Boxes carry type and size class.</p>
        <div className="crops">
          {roles.map((role) => (
            <CropView key={role} runId={runId} role={role} panelId={panelId} keyframes={state.keyframes} marks={state.marks} />
          ))}
        </div>
      </div>

      {roles.length > 1 ? (
        <div className="section">
          <h2>match{panelId ? `: ${panelName(panelId)}` : ''}</h2>
          <p className="section-note">Pickup left, return right. Code writes every label; a model answer can only lower one to needs your eye.</p>
          <MatchPane runId={runId} panelId={panelId} match={match} panel={panel} keyframes={state.keyframes} marks={state.marks} verdicts={state.verdicts} videos={state.videos} run={state.run} />
        </div>
      ) : null}

      <div className="section">
        <h2>verdicts</h2>
        <VerdictList verdicts={state.verdicts} marks={state.marks} />
      </div>

      <div className="section">
        <h2>Stays Put</h2>
        <p className="section-note">Each mark tracked through its own clip by SAM 3.1; a mark on the paint moves with the panel, a reflection slides off it.</p>
        <StaysPutList markTracks={state.markTracks} />
      </div>

      <div className="section">
        <h2>narration</h2>
        <NarrationChips runId={runId} narration={state.narration} roles={roles} onSeek={onSeek} />
      </div>

      <div className="section">
        <h2>sheet</h2>
        {state.sheet.written ? (
          <p>
            <Link to={`/run/${encodeURIComponent(runId)}/sheet`}>open the sheet</Link>, <a href={routes.sheetPdf(runId)}>evidence.pdf</a>,{' '}
            <a href={routes.bundleZip(runId)}>bundle.zip</a>
            {Object.keys(state.sheet.sha256s).length ? (
              <span className="small muted mono">
                {' '}
                {Object.entries(state.sheet.sha256s)
                  .map(([k, v]) => `${k} ${v.slice(0, 12)}`)
                  .join(', ')}
              </span>
            ) : null}
          </p>
        ) : (
          <div className="empty">The sheet is written at the end of the run. A partial run still gets one, with the missing nouns printed as not run.</div>
        )}
      </div>

      <div className="section">
        <h2>privacy</h2>
        <PrivacyCard deletions={state.deletions} run={state.run} />
      </div>

      {state.replay ? <ReplayBadge date={state.replay.badgeDate} /> : null}
    </div>
  );
}
