import { useCallback, useMemo, useState } from 'react';

import { routes } from '../api/client.ts';
import { useStatus } from '../api/useStatus.ts';
import { CompareInPlace } from '../components/CompareInPlace.tsx';
import { CoverageCar } from '../components/CoverageCar.tsx';
import { CoverageList } from '../components/CoverageList.tsx';
import { bestKeyframe, CropView } from '../components/CropView.tsx';
import { FragmentsRow } from '../components/FragmentsRow.tsx';
import { HowChecked } from '../components/HowChecked.tsx';
import { LedgerStrip } from '../components/LedgerStrip.tsx';
import { MarkCard, MarkFrame } from '../components/MarkCard.tsx';
import { MatchPane } from '../components/MatchPane.tsx';
import { NarrationChips } from '../components/NarrationChips.tsx';
import { PanelList } from '../components/PanelList.tsx';
import { PrivacyCard } from '../components/PrivacyCard.tsx';
import { QueueLine } from '../components/QueueLine.tsx';
import { ReplayBadge } from '../components/ReplayBadge.tsx';
import { SealCard } from '../components/SealCard.tsx';
import { StageLine } from '../components/StageLine.tsx';
import { StaysPutList } from '../components/StaysPutList.tsx';
import { controlTrack, StaysPutPair } from '../components/StaysPutPair.tsx';
import { TopUpBanner } from '../components/TopUpBanner.tsx';
import { VerdictList } from '../components/VerdictList.tsx';
import { WalkPlayer } from '../components/WalkPlayer.tsx';
import type { Coverage, Keyframe, Mark, PanelTrack, ReplayMode } from '../contracts/types.ts';
import { useLanes } from '../lanes/useLanes.ts';
import { RunFiles } from '../lib/files.tsx';
import { focusTrack, panelFocus } from '../lib/focus.ts';
import { formatDate } from '../lib/format.ts';
import { staysPutCalibrated } from '../lib/lines.ts';
import { answerSentences, CARD_ORDER, GROUP_TITLE, notComparableLine, tally } from '../lib/result.ts';
import { replayBadgeDate, sheetFileHash } from '../lib/runview.ts';
import { colorForLane, colorForPanel, DRAWN_LANES, laneId, nounOfLane, panelName } from '../lib/panels.ts';
import { Link, navigate } from '../lib/router.tsx';
import { fpsOf, rolesOf } from '../state/reducer.ts';
import type { RunState } from '../state/types.ts';
import { useRun, type RunSource } from '../state/useRun.ts';
import { ClaimExit, ClaimSteps } from './ClaimPage.tsx';

export function replayModeOf(query: URLSearchParams): ReplayMode | null {
  const v = query.get('replay');
  if (v === 'fixed' || v === 'recorded' || v === 'instant') return v;
  if (v === '1' || v === 'true') return 'fixed';
  return null;
}

// The panel to open on: an "already there" verdict, else a comparable panel, else a filmed one, else the first track.
export function defaultPanel(state: RunState): string | null {
  const already = state.verdicts.find((v) => v.label === 'already_there');
  if (already) return already.panel_id;
  const comparable = state.matches.find((m) => m.state === 'comparable');
  if (comparable) return comparable.panel_id;
  const filmed = Object.entries(state.coverage?.per_role['pickup'] ?? {}).find(([, c]) => c?.state === 'filmed');
  if (filmed) return filmed[0];
  return state.tracks.find((t) => t.panel_id)?.panel_id ?? null;
}

// role|lane|SAM id -> the voted track, so the walk colors each object by its canonical panel once the vote is in.
export function trackLookup(tracks: PanelTrack[]): Map<string, PanelTrack> {
  const m = new Map<string, PanelTrack>();
  for (const t of tracks) for (const id of t.sam_object_ids) m.set(`${t.role}|${laneId(t.noun)}|${id}`, t);
  return m;
}

export function useTrackStyle(tracks: PanelTrack[]) {
  const lookup = useMemo(() => trackLookup(tracks), [tracks]);
  const colorFor = useCallback(
    (role: string, lane: string, id: string) => {
      const t = lookup.get(`${role}|${lane}|${id}`);
      return t?.panel_id ? colorForPanel(t.panel_id) : colorForLane(lane);
    },
    [lookup],
  );
  const labelFor = useCallback(
    (role: string, lane: string, id: string) => {
      const t = lookup.get(`${role}|${lane}|${id}`);
      return t?.panel_id ? panelName(t.panel_id) : `${nounOfLane(lane)} ${id}`;
    },
    [lookup],
  );
  return { colorFor, labelFor };
}

// A pickup check has no second walk to match: each panel's best frame that holds marks, boxed.
export function panelsWithMarks(keyframes: Keyframe[], marks: Mark[]): Keyframe[] {
  const out: Keyframe[] = [];
  const seen = new Set<string>();
  const sorted = [...keyframes].filter((k) => k.role === 'pickup' && marks.some((m) => m.kf_id === k.kf_id)).sort((a, b) => b.score.total - a.score.total);
  for (const k of sorted) {
    if (seen.has(k.panel_id)) continue;
    seen.add(k.panel_id);
    out.push(k);
  }
  return out;
}

const MODES: [string, string][] = [
  ['live', 'live, as it happens'],
  ['fixed', 'replay with fixed delays'],
  ['recorded', 'replay with real timings (gaps over 2 s cut to 2 s)'],
  ['instant', 'replay instantly'],
];

export function RunPage({ runId, query }: { runId: string; query: URLSearchParams }) {
  const mode = replayModeOf(query);
  const source = useMemo<RunSource>(() => (mode ? { kind: 'replay', mode } : { kind: 'live' }), [mode]);
  const { state, status: driver, detail, progress } = useRun(runId, source);
  const roles = rolesOf(state);
  const lanes = useLanes(runId, state, roles);
  const { status } = useStatus(mode ? null : runId);
  const { colorFor, labelFor } = useTrackStyle(state.tracks);
  const [selected, setSelected] = useState<string | null>(null);
  const [roleTab, setRoleTab] = useState('pickup');
  const [seek, setSeek] = useState<Record<string, number | null>>({});
  const [coverageOverride, setCoverageOverride] = useState<Coverage | null>(null);
  const [open, setOpen] = useState(false);

  const badgeDate = replayBadgeDate(state);
  const panelId = selected ?? defaultPanel(state);
  const coverage = coverageOverride ?? state.coverage;
  const panel = state.panels.find((p) => p.panel_id === panelId);
  const match = state.matches.find((m) => m.panel_id === panelId);
  const tab = roles.includes(roleTab) ? roleTab : 'pickup';
  const markById = useMemo(() => new Map(state.marks.map((m) => [m.mark_id, m])), [state.marks]);
  const focus = panelFocus(panelId, match, state.verdicts);
  const focusMarkTrack = focusTrack(focus, panelId, state.markTracks, markById);
  const staysPut = {
    track: focusMarkTrack,
    control: controlTrack(state.markTracks),
    mark: focusMarkTrack ? markById.get(focusMarkTrack.mark_id) : undefined,
    calibrated: staysPutCalibrated(state.run),
  };

  const choosePanel = (id: string) => {
    setSelected(id);
    const next: Record<string, number | null> = {};
    for (const role of roles) {
      const kf = bestKeyframe(state.keyframes, role, id);
      const track = state.tracks.find((t) => t.role === role && t.panel_id === id);
      next[role] = kf?.f_proxy ?? track?.fragments[0]?.first_frame ?? null;
    }
    setSeek(next);
  };

  const onSeekSeconds = (role: string, seconds: number) => setSeek((s) => ({ ...s, [role]: Math.round(seconds * fpsOf(state, role)) }));

  const setMode = (value: string) => {
    const base = `/run/${encodeURIComponent(runId)}`;
    navigate(value === 'live' ? base : `${base}?replay=${value}`, { replace: true });
  };

  const driverLine =
    driver === 'connecting'
      ? 'connecting'
      : driver === 'reconnecting'
        ? 'Connection dropped; reconnecting. The run keeps going on the server and this page resumes where it left off.'
        : driver === 'error'
          ? `Could not load this run: ${detail ?? 'unknown error'}.`
          : null;

  const done = state.completed || state.stage.name === 'done' || state.stage.name === 'partial';
  const failed = state.stage.name === 'failed';
  const cards = CARD_ORDER.map((label) => ({ label, verdicts: state.verdicts.filter((v) => v.label === label) })).filter((g) => g.verdicts.length);
  const pickupPanels = roles.length === 1 ? panelsWithMarks(state.keyframes, state.marks) : [];
  const answer = roles.length > 1 && state.verdicts.length ? answerSentences(tally(state.verdicts)) : null;
  const ncLine = roles.length > 1 ? notComparableLine(state.verdicts) : null;
  const pickupMarks = state.marks.filter((m) => m.role === 'pickup').length;

  // A card's "How this was checked": the panel opens in every view below, and the section opens on the match.
  const openDetails = (id: string) => {
    choosePanel(id);
    setOpen(true);
    requestAnimationFrame(() => (document.getElementById('checked-match') ?? document.querySelector('details.checked'))?.scrollIntoView({ block: 'start' }));
  };

  // The two walks with SAM 3.1's outlines: the thing to watch while a check runs, then part of How it was checked.
  const walks = (
    <div className={`walks walks-${roles.length}`}>
      {roles.map((role) => {
        const media = state.media[role];
        const blurred = media?.blurredPath ?? state.videos[role]?.proxy_blurred?.path ?? null;
        return (
          <section key={role} className="walk-block" aria-label={`${role} walk`}>
            <div className="walk-title">
              <h2>{role} walk</h2>
              <span className="small muted mono">
                {media?.ffprobe ? `${media.ffprobe.display_width}x${media.ffprobe.display_height}, ${media.ffprobe.duration_s.toFixed(0)} s` : ''}
                {media?.proxy ? `, proxy ${media.proxy.width}x${media.proxy.height} at ${media.proxy.fps} fps` : ''}
              </span>
            </div>
            {blurred ? (
              <WalkPlayer
                src={routes.file(runId, blurred)}
                role={role}
                store={lanes.store}
                lanes={DRAWN_LANES}
                colorFor={colorFor}
                labelFor={labelFor}
                seekFrame={seek[role] ?? null}
              />
            ) : (
              <div className="walk-placeholder">
                {state.stage.name === 'failed'
                  ? 'The run failed before this walk was blurred, so there is no video to show. The error is above.'
                  : media?.upload || state.videos[role]
                    ? 'The video appears here after plates and faces are blurred. This page never receives the unblurred file.'
                    : `The ${role} walk has not arrived yet.`}
              </div>
            )}
            {state.run?.footage_credit ? <p className="footage-credit">{state.run.footage_credit}</p> : null}
            <SealCard role={role} video={state.videos[role] ?? null} media={media} seal={state.seal} run={state.run} failed={state.stage.name === 'failed'} />
          </section>
        );
      })}
    </div>
  );

  return (
    <RunFiles runId={runId}>
      <div className="run-page">
        <p className="result-meta">
          {state.run ? `${state.run.kind === 'pair' ? 'Pickup and return' : 'Pickup check'}, ${formatDate(state.run.created_at)}. ` : ''}
          Run <span className="mono">{runId}</span>
        </p>
        {driverLine ? <p className="driver-line">{driverLine}</p> : null}

        <section className="answer" aria-live="polite">
          {failed ? (
            <>
              <h1>This check did not finish.</h1>
              <StageLine state={state} />
            </>
          ) : done && roles.length > 1 ? (
            <h1 className="answer-line">
              {answer
                ? answer.map((sentence, i) => (
                    <span key={i}>{sentence} </span>
                  ))
                : 'No panel could be compared.'}
            </h1>
          ) : roles.length === 1 && done ? (
            <h1 className="answer-line">
              Your pickup video is sealed. Muse Spark found {pickupMarks} {pickupMarks === 1 ? 'mark' : 'marks'} on {pickupPanels.length}{' '}
              {pickupPanels.length === 1 ? 'panel' : 'panels'}.
            </h1>
          ) : (
            <>
              <h1>{mode ? 'Replaying the sample check' : 'Checking your videos'}</h1>
              <StageLine state={state} />
              <p className="answer-sub">The answer shows here when the check is done. You can close this tab and come back to this address.</p>
            </>
          )}
          {done && ncLine ? <p className="answer-sub">{ncLine}</p> : null}
          {done && state.stage.name === 'partial' ? <StageLine state={state} /> : null}
          {done && state.sheet.written ? (
            <div className="answer-actions">
              <Link to={`/run/${encodeURIComponent(runId)}/sheet`} className="cds--btn cds--btn--primary">
                Open the sealed evidence sheet
              </Link>
              <Link to={`/claim?run=${encodeURIComponent(runId)}`} className="cds--btn cds--btn--tertiary">
                Place a claim photo
              </Link>
            </div>
          ) : null}
        </section>

        {!done && !failed ? <div className="walks-live">{walks}</div> : null}

        {!mode ? <TopUpBanner runId={runId} coverage={coverage} onCoverage={setCoverageOverride} /> : null}

        {state.claim ? (
          <section className="result-claim" aria-label="claim photo">
            <h2>Your claim photo</h2>
            <ClaimExit claim={state.claim} />
          </section>
        ) : null}

        {cards.map((g) => (
          <section key={g.label} className="mark-group" aria-label={GROUP_TITLE[g.label]}>
            <h2>
              {GROUP_TITLE[g.label]} <span className="count">{g.verdicts.length}</span>
            </h2>
            <div className="mark-cards">
              {g.verdicts.map((v) => (
                <MarkCard key={v.verdict_id} v={v} marks={markById} keyframes={state.keyframes} roles={roles} onDetails={openDetails} />
              ))}
            </div>
          </section>
        ))}
        {pickupPanels.length ? (
          <section className="mark-group" aria-label="marks at pickup">
            <h2>
              Marks at pickup <span className="count">{pickupMarks}</span>
            </h2>
            <div className="mark-cards">
              {pickupPanels.map((kf) => (
                <article key={kf.kf_id} className="mark-card">
                  <header className="mark-card-head">
                    <h3>{panelName(kf.panel_id)}</h3>
                  </header>
                  <MarkFrame kf={kf} marks={state.marks.filter((m) => m.kf_id === kf.kf_id)} role="pickup" zoom={false} />
                </article>
              ))}
            </div>
          </section>
        ) : null}
        {(cards.length || pickupPanels.length) && state.run?.footage_credit ? <p className="footage-credit">{state.run.footage_credit}</p> : null}

        <details
          className="checked"
          open={open}
          onToggle={(e) => setOpen((e.currentTarget as HTMLDetailsElement).open)}
        >
          <summary>
            <span className="checked-title">How it was checked</span>
            <span className="checked-hint">The models, both walks, every panel, the rules, the seal and the hashes.</span>
          </summary>
          <div className="checked-body">
        <div className="run-head">
          <div className="controls">
            <label>
              <span className="sr-only">timing</span>
              <select value={mode ?? 'live'} onChange={(e) => setMode(e.target.value)}>
                {MODES.map(([value, text]) => (
                  <option key={value} value={value}>
                    {text}
                  </option>
                ))}
              </select>
            </label>
            <Link to={`/run/${encodeURIComponent(runId)}/sheet`}>sheet</Link>
            <Link to={`/claim?run=${encodeURIComponent(runId)}`}>claim photo</Link>
          </div>
        </div>
        {progress && mode ? (
          <p className="driver-line mono small">
            replay {Math.round(progress.atMs / 1000)} s of {Math.round(progress.spanMs / 1000)} s, event {progress.done} of {progress.total}
          </p>
        ) : null}
        <StageLine state={state} />
        <LedgerStrip ledger={state.ledger} />
        <QueueLine status={status} />
        <section className="checked-block" aria-label="how it was checked">
          <h2>The models and what each did</h2>
          <HowChecked state={state} />
        </section>
        {state.errors.map((e, i) => (
          <p key={i} className="error-line small">
            {e}
          </p>
        ))}

        {done || failed ? walks : null}

        <div className="run-grid">
          <div>
            <section className="section">
              <h2>track</h2>
              <RoleTabs roles={roles} value={tab} onChange={setRoleTab} />
              <FragmentsRow
                role={tab}
                panelId={panelId}
                tracks={state.tracks}
                fps={fpsOf(state, tab)}
                durationS={state.media[tab]?.ffprobe?.duration_s ?? null}
                color={colorForPanel(panelId)}
              />
            </section>
            <section className="section">
              <h2>coverage</h2>
              <p className="section-note">Filmed, glimpsed, not filmed, not run (no answer from the model), or not seen in either walk. A top-up clip counts for the pickup walk.</p>
              {coverage || state.panels.length ? (
                <div className={`cars cars-${roles.length}`}>
                  {roles.map((role) => (
                    <CoverageCar key={role} coverage={coverage} panels={state.panels} role={role} withTopups={role === 'pickup'} selected={panelId} onSelect={choosePanel} />
                  ))}
                </div>
              ) : null}
              <CoverageList coverage={coverage} roles={roles} />
            </section>
          </div>
          <aside className="section">
            <h2>panels</h2>
            <RoleTabs roles={roles} value={tab} onChange={setRoleTab} />
            <PanelList role={tab} panels={state.panels} coverage={coverage} tracks={state.tracks} lanes={state.lanes[tab] ?? {}} selected={panelId} onSelect={choosePanel} />
          </aside>
        </div>

        <section className="section">
          <h2>crops{panelId ? `: ${panelName(panelId)}` : ''}</h2>
          <p className="section-note">Muse Spark sees one blurred crop at a time, never both walks together, and never writes the verdict.</p>
          <div className="crops">
            {roles.map((role) => (
              <CropView key={role} role={role} panelId={panelId} keyframes={state.keyframes} marks={state.marks} />
            ))}
          </div>
        </section>

        {roles.length > 1 ? (
          <section className="section" id="checked-match">
            <h2>match{panelId ? `: ${panelName(panelId)}` : ''}</h2>
            <p className="section-note">Pickup left, return right. Code writes every label; a model answer can only lower one to needs your eye.</p>
            <MatchPane
              panelId={panelId}
              match={match}
              panel={panel}
              keyframes={state.keyframes}
              marks={state.marks}
              verdicts={state.verdicts}
              videos={state.videos}
              media={state.media}
              run={state.run}
              looks={state.looks}
            />
            <CompareInPlace panelId={panelId} match={match} verdicts={state.verdicts} marks={state.marks} keyframes={state.keyframes} />
          </section>
        ) : null}

        <section className="section">
          <h2>verdicts</h2>
          <VerdictList verdicts={state.verdicts} marks={state.marks} looks={state.looks} />
        </section>

        <section className="section">
          <h2>Stays Put</h2>
          <p className="section-note">
            SAM 3.1 tracks each mark through its own short clip. A mark on the paint moves with the panel; a reflection slides across it as the camera moves.
            {staysPut.calibrated ? '' : ' Stays Put is not calibrated yet, so in this run it records the tracks and their drift and changes no label.'}
          </p>
          {staysPut.track || staysPut.control ? (
            <StaysPutPair track={staysPut.track} control={staysPut.control} mark={staysPut.mark} color={colorForPanel(panelId)} calibrated={staysPut.calibrated} numbers />
          ) : null}
          <StaysPutList markTracks={state.markTracks} marks={state.marks} calibrated={staysPut.calibrated} missing={Boolean(state.partial?.includes('staysput'))} />
        </section>

        <section className="section">
          <h2>narration</h2>
          <NarrationChips narration={state.narration} roles={roles} onSeek={onSeekSeconds} marks={state.marks} missing={Boolean(state.partial?.includes('narration'))} />
        </section>

        {state.claim ? (
          <section className="section">
            <h2>claim</h2>
            {state.claims.length > 1 ? <p className="section-note">{state.claims.length} claim photos were placed against this run; the latest is below.</p> : null}
            <ClaimExit claim={state.claim} />
            <ClaimSteps claim={state.claim} />
            <p className="small">
              <Link to={`/claim?run=${encodeURIComponent(runId)}`}>place another claim photo</Link>
            </p>
          </section>
        ) : null}

        <section className="section">
          <h2>sheet</h2>
          {state.sheet.written ? (
            <p>
              <Link to={`/run/${encodeURIComponent(runId)}/sheet`}>open the sheet</Link>
              {sheetFileHash(state.sheet.sha256s, 'evidence.pdf') ? (
                <>
                  , <a href={routes.sheetPdf(runId)}>evidence.pdf</a>
                </>
              ) : (
                ', PDF retry pending (the HTML sheet is complete)'
              )}
              , <a href={routes.bundleZip(runId)}>bundle.zip</a>
              <span className="small muted mono">
                {' '}
                {Object.entries(state.sheet.sha256s)
                  .map(([k, v]) => `${k} ${v.slice(0, 12)}`)
                  .join(', ')}
              </span>
            </p>
          ) : (
            <div className="empty">The sheet is written at the end of the run. A partial run still gets one, with the missing nouns marked not run.</div>
          )}
        </section>

        <section className="section">
          <h2>privacy</h2>
          <PrivacyCard state={state} />
        </section>
          </div>
        </details>

        {badgeDate !== null ? <ReplayBadge date={badgeDate} /> : null}
      </div>
    </RunFiles>
  );
}

function RoleTabs({ roles, value, onChange }: { roles: string[]; value: string; onChange: (role: string) => void }) {
  if (roles.length < 2) return null;
  return (
    <div className="role-tabs" role="group" aria-label="walk">
      {roles.map((r) => (
        <button key={r} type="button" className={value === r ? 'on' : ''} aria-pressed={value === r} onClick={() => onChange(r)}>
          {r}
        </button>
      ))}
    </div>
  );
}
