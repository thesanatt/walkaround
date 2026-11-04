import { useCallback, useMemo, useState, type ReactNode } from 'react';

import { routes } from '../api/client.ts';
import { useStatus } from '../api/useStatus.ts';
import { Bar, ReplayNote } from '../components/Bar.tsx';
import { CompareInPlace } from '../components/CompareInPlace.tsx';
import { CoverageCar } from '../components/CoverageCar.tsx';
import { CoverageList } from '../components/CoverageList.tsx';
import { bestKeyframe, CropView } from '../components/CropView.tsx';
import { FragmentsRow } from '../components/FragmentsRow.tsx';
import { Frame } from '../components/Frame.tsx';
import { HowChecked } from '../components/HowChecked.tsx';
import { LedgerStrip } from '../components/LedgerStrip.tsx';
import { MatchPane } from '../components/MatchPane.tsx';
import { NarrationChips } from '../components/NarrationChips.tsx';
import { PanelList } from '../components/PanelList.tsx';
import { PrivacyCard } from '../components/PrivacyCard.tsx';
import { QueueLine } from '../components/QueueLine.tsx';
import { SealCard } from '../components/SealCard.tsx';
import { StageLine } from '../components/StageLine.tsx';
import { StaysPutList } from '../components/StaysPutList.tsx';
import { controlTrack, StaysPutPair } from '../components/StaysPutPair.tsx';
import { TopUpBanner, topUpShown } from '../components/TopUpBanner.tsx';
import { VerdictList } from '../components/VerdictList.tsx';
import { WalkPlayer } from '../components/WalkPlayer.tsx';
import type { Box4, Coverage, Keyframe, Mark, PanelTrack, ReplayMode } from '../contracts/types.ts';
import { useLanes } from '../lanes/useLanes.ts';
import { RunFiles } from '../lib/files.tsx';
import { cap } from '../lib/findings.ts';
import { focusTrack, panelFocus } from '../lib/focus.ts';
import { formatDate, formatDateTime } from '../lib/format.ts';
import { staysPutCalibrated } from '../lib/lines.ts';
import { byEyeRows, comparedPanels, headline, notComparedRows, subline } from '../lib/result.ts';
import { replayBadgeDate, sheetFileHash } from '../lib/runview.ts';
import { colorForLane, colorForPanel, DRAWN_LANES, nounOfLane, panelName } from '../lib/panels.ts';
import { Link, navigate } from '../lib/router.tsx';
import { trackIndex } from '../lib/trackIndex.ts';
import { fpsOf, rolesOf } from '../state/reducer.ts';
import type { RunState } from '../state/types.ts';
import type { RunHandle } from '../state/useRun.ts';
import { ClaimResult } from './ClaimPage.tsx';

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

// The walk overlay's colors, chips and filter, each read from the track that holds the object at that frame
// (lib/trackIndex): a chip names the panel the door check decided, and a track the run's data puts off the car
// (state.offCar) is not drawn.
const NO_IDS: readonly string[] = [];

export function useTrackStyle(tracks: PanelTrack[], offCar: readonly string[] = NO_IDS) {
  const index = useMemo(() => trackIndex(tracks, offCar), [tracks, offCar]);
  const colorFor = useCallback(
    (role: string, lane: string, id: string, frame = -1) => {
      const t = index.at(role, lane, id, frame);
      return t?.panel_id ? colorForPanel(t.panel_id) : colorForLane(lane);
    },
    [index],
  );
  const labelFor = useCallback(
    (role: string, lane: string, id: string, frame = -1) => {
      const t = index.at(role, lane, id, frame);
      return t?.panel_id ? panelName(t.panel_id) : `${nounOfLane(lane)} ${id}`;
    },
    [index],
  );
  const visible = useCallback((role: string, lane: string, id: string, frame: number) => index.shows(role, lane, id, frame), [index]);
  return { colorFor, labelFor, visible, index };
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

function Section({ id, title, note, children }: { id: string; title: string; note?: ReactNode; children?: ReactNode }) {
  return (
    <section className="rec-section" id={id} aria-labelledby={`${id}-title`}>
      <h2 id={`${id}-title`}>{title}</h2>
      {note ? <p className="rec-note">{note}</p> : null}
      {children}
    </section>
  );
}

// Everything the check recorded, and the page while a check runs: the walks with SAM 3.1's outlines, each model's part,
// coverage, panels, crops, the match, every verdict, Stays Put, narration, the claim, the sheet files and privacy. The
// run page owns the one replay or live stream and hands its state here. record: opened from the sealed panel of a
// finished run, every section open; otherwise the sections sit in one closed row under the walks.
export function RunRecord({ runId, mode, handle, record = false, bar }: { runId: string; mode: ReplayMode | null; handle: RunHandle; record?: boolean; bar?: ReactNode }) {
  const { state, status: driver, detail, progress } = handle;
  const roles = rolesOf(state);
  const lanes = useLanes(runId, state, roles);
  const { status } = useStatus(mode ? null : runId);
  const { colorFor, labelFor, visible } = useTrackStyle(state.tracks, state.offCar);
  const [selected, setSelected] = useState<string | null>(null);
  const [roleTab, setRoleTab] = useState('pickup');
  const [seek, setSeek] = useState<Record<string, number | null>>({});
  const [coverageOverride, setCoverageOverride] = useState<Coverage | null>(null);

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
    navigate(value === 'live' ? base : `${base}?replay=${value}${record ? '&record=1' : ''}`, { replace: true });
  };

  const driverLine =
    driver === 'connecting'
      ? 'Connecting.'
      : driver === 'reconnecting'
        ? 'Connection dropped; reconnecting. The run keeps going on the server and this page resumes where it left off.'
        : driver === 'error'
          ? `Could not load this run: ${detail ?? 'unknown error'}.`
          : null;

  const done = state.completed || state.stage.name === 'done' || state.stage.name === 'partial';
  const failed = state.stage.name === 'failed';
  const pair = roles.length > 1;
  const pickupPanels = roles.length === 1 ? panelsWithMarks(state.keyframes, state.marks) : [];
  const pickupMarks = state.marks.filter((m) => m.role === 'pickup').length;
  const compared = comparedPanels(state.matches, state.verdicts);
  const ncRows = notComparedRows(state.verdicts);
  const byEye = byEyeRows(state.matches);
  const pdf = Boolean(sheetFileHash(state.sheet.sha256s, 'evidence.pdf'));
  const takesClaims = !state.run?.keep;
  const credit = state.run?.footage_credit ?? null;
  const claimCtx = { keyframes: state.keyframes, panels: state.panels, marks: state.marks, coverage: state.coverage, videos: state.videos as Record<string, NonNullable<RunState['videos'][string]>> };
  const resultUrl = `/run/${encodeURIComponent(runId)}${mode ? `?replay=${mode}` : ''}`;
  const sheetUrl = `/run/${encodeURIComponent(runId)}/sheet`;

  // The two walks with SAM 3.1's outlines: the thing to watch while a check runs, then the first section of the record.
  const walks = (
    <div className={`walks walks-${roles.length}`}>
      {roles.map((role) => {
        const media = state.media[role];
        const blurred = media?.blurredPath ?? state.videos[role]?.proxy_blurred?.path ?? null;
        return (
          <section key={role} className="walk-block" aria-label={`${role} walk`}>
            <div className="walk-title">
              <h3>{role === 'pickup' ? 'Pickup walk' : 'Return walk'}</h3>
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
                visible={visible}
                seekFrame={seek[role] ?? null}
              />
            ) : (
              <div className="walk-placeholder">
                {failed
                  ? 'The run failed before this walk was blurred, so there is no video to show.'
                  : media?.upload || state.videos[role]
                    ? 'The video appears here after plates and faces are blurred. This page never receives the unblurred file.'
                    : `The ${role} walk has not arrived yet.`}
              </div>
            )}
            {credit ? <p className="footage-credit">{credit}</p> : null}
            <SealCard role={role} video={state.videos[role] ?? null} media={media} seal={state.seal} run={state.run} failed={failed} />
          </section>
        );
      })}
    </div>
  );

  const everything = (
    <>
      {done || failed ? (
        <Section id="walks" title="The two walks" note="SAM 3.1's outlines drawn over the blurred walks, and what was sealed before any model saw them.">
          {walks}
        </Section>
      ) : null}

      <Section id="models" title="What each model did">
        <HowChecked state={state} />
        <LedgerStrip ledger={state.ledger} />
        <QueueLine status={status} />
      </Section>

      <Section id="coverage" title="Coverage" note="Filmed, glimpsed, not filmed, not run (no answer from the model), or not seen in either walk. A top-up clip counts for the pickup walk.">
        {coverage || state.panels.length ? (
          <div className={`cars cars-${roles.length}`}>
            {roles.map((role) => (
              <CoverageCar key={role} coverage={coverage} panels={state.panels} role={role} withTopups={role === 'pickup'} selected={panelId} onSelect={choosePanel} />
            ))}
          </div>
        ) : null}
        <CoverageList coverage={coverage} roles={roles} />
      </Section>

      <Section id="panels" title="Panels" note="Pick a panel to see its track, its crops and its match below.">
        <RoleTabs roles={roles} value={tab} onChange={setRoleTab} />
        <div className="rec-panels">
          <PanelList role={tab} panels={state.panels} coverage={coverage} tracks={state.tracks} lanes={state.lanes[tab] ?? {}} selected={panelId} onSelect={choosePanel} />
          <div>
            <h3 className="rec-sub">{panelId ? cap(panelName(panelId)) : 'No panel'}</h3>
            <FragmentsRow
              role={tab}
              panelId={panelId}
              tracks={state.tracks}
              fps={fpsOf(state, tab)}
              durationS={state.media[tab]?.ffprobe?.duration_s ?? null}
              color={colorForPanel(panelId)}
            />
          </div>
        </div>
      </Section>

      <Section id="crops" title={`Crops${panelId ? `: ${panelName(panelId)}` : ''}`} note="Muse Spark sees one blurred crop at a time, never both walks together, and never writes the verdict.">
        <div className="crops">
          {roles.map((role) => (
            <CropView key={role} role={role} panelId={panelId} keyframes={state.keyframes} marks={state.marks} />
          ))}
        </div>
      </Section>

      {pair ? (
        <Section id="checked-match" title={`The match${panelId ? `: ${panelName(panelId)}` : ''}`} note="Pickup left, return right. Code writes every label; a model answer can only lower one to needs your eye.">
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
        </Section>
      ) : null}

      <Section id="verdicts" title="Every verdict" note="One row per mark or panel, with the rule that decided it.">
        <VerdictList verdicts={state.verdicts} marks={state.marks} looks={state.looks} />
      </Section>

      {pair && done && (ncRows.length || byEye.length) ? (
        <Section id="not-compared" title="Not compared, and left to you">
          {ncRows.length ? (
            <ul className="rec-list">
              {ncRows.map((r) => (
                <li key={r.panelId}>
                  <strong>{cap(panelName(r.panelId))}</strong> <span className="muted">not compared: {r.reason}</span>
                </li>
              ))}
            </ul>
          ) : null}
          {byEye.length ? (
            <ul className="rec-list">
              {byEye.map((r) => (
                <li key={r.key}>
                  <strong>{cap(r.title)}</strong> <span className="muted">check by eye: {r.text}</span>
                </li>
              ))}
            </ul>
          ) : null}
        </Section>
      ) : null}

      <Section
        id="stays-put"
        title="Stays Put"
        note={`SAM 3.1 tracks each mark through its own short clip. A mark on the paint moves with the panel; a reflection slides across it as the camera moves.${staysPut.calibrated ? '' : ' Stays Put is not calibrated yet, so in this run it records the tracks and their drift and changes no label.'}`}
      >
        {staysPut.track || staysPut.control ? (
          <StaysPutPair track={staysPut.track} control={staysPut.control} mark={staysPut.mark} color={colorForPanel(panelId)} calibrated={staysPut.calibrated} numbers />
        ) : null}
        {state.markTracks.length ? (
          <details className="fold">
            <summary>Every mark tracked ({state.markTracks.length})</summary>
            <StaysPutList markTracks={state.markTracks} marks={state.marks} calibrated={staysPut.calibrated} missing={Boolean(state.partial?.includes('staysput'))} />
          </details>
        ) : (
          <StaysPutList markTracks={state.markTracks} marks={state.marks} calibrated={staysPut.calibrated} missing={Boolean(state.partial?.includes('staysput'))} />
        )}
      </Section>

      <Section id="narration" title="Narration">
        <NarrationChips narration={state.narration} roles={roles} onSeek={onSeekSeconds} marks={state.marks} missing={Boolean(state.partial?.includes('narration'))} />
      </Section>

      {state.claim ? (
        <Section id="claim" title="A claim photo, placed against the pickup walk" note={state.claims.length > 1 ? `${state.claims.length} claim photos were placed against this run; the latest is below.` : undefined}>
          <ClaimResult claim={state.claim} ctx={claimCtx} sealed={state.seal.tsr_time ? formatDateTime(state.seal.tsr_time) : null} />
          {takesClaims ? (
            <p className="rec-links">
              <Link to={`/claim?run=${encodeURIComponent(runId)}`}>Place another claim photo</Link>
            </p>
          ) : null}
        </Section>
      ) : null}

      <Section id="sheet" title="The sheet">
        {state.sheet.written ? (
          <>
            <p className="rec-links">
              <Link to={sheetUrl}>Open the sheet</Link>
              {pdf ? <a href={routes.sheetPdf(runId)}>evidence.pdf</a> : <span className="muted">PDF retry pending; the HTML sheet is complete</span>}
              <a href={routes.bundleZip(runId)}>bundle.zip</a>
            </p>
            <ul className="rec-hashes">
              {Object.entries(state.sheet.sha256s).map(([k, v]) => (
                <li key={k}>
                  <span>{k}</span>
                  <span className="mono" title={v}>
                    {v.slice(0, 12)}
                    <span className="muted">{v.slice(12)}</span>
                  </span>
                </li>
              ))}
            </ul>
          </>
        ) : (
          <p className="empty">The sheet is written at the end of the run. A partial run still gets one, with the missing nouns marked not run.</p>
        )}
      </Section>

      <Section id="privacy" title="Privacy">
        <PrivacyCard state={state} />
      </Section>

      <Section id="run" title="This run">
        <p className="rec-run">
          {state.run ? `${state.run.kind === 'pair' ? 'Pickup and return' : 'Pickup check'}, ${formatDate(state.run.created_at)}. ` : ''}
          Run <span className="mono">{runId}</span>
        </p>
        <div className="rec-links">
          <label className="rec-mode">
            <span className="sr-only">timing</span>
            <select value={mode ?? 'live'} onChange={(e) => setMode(e.target.value)}>
              {MODES.map(([value, text]) => (
                <option key={value} value={value}>
                  {text}
                </option>
              ))}
            </select>
          </label>
          <Link to={sheetUrl}>Sheet</Link>
          {takesClaims ? <Link to={`/claim?run=${encodeURIComponent(runId)}`}>Claim photo</Link> : null}
        </div>
        {progress && mode ? (
          <p className="driver-line mono small">
            replay {Math.round(progress.atMs / 1000)} s of {Math.round(progress.spanMs / 1000)} s, event {progress.done} of {progress.total}
          </p>
        ) : null}
        {done ? <StageLine state={state} /> : null}
        {state.errors.map((e, i) => (
          <p key={i} className="error-line small">
            {e}
          </p>
        ))}
      </Section>
    </>
  );

  const title = failed
    ? 'This check did not finish'
    : record
      ? 'Everything the check recorded'
      : done && !pair
        ? 'Your pickup video is sealed'
        : mode
          ? 'Replaying a finished check'
          : 'Checking your videos';

  return (
    <RunFiles runId={runId}>
      <div className={`record${record ? ' full' : ''}`}>
        <Bar note={badgeDate !== null ? <ReplayNote date={badgeDate} /> : undefined}>{bar}</Bar>
        {record ? (
          <p className="record-back">
            <Link to={resultUrl}>Back to the result</Link>
          </p>
        ) : null}
        <header className="record-head" aria-live="polite">
          <h1>{title}</h1>
          {record && done && pair ? <p className="record-sum">{`${headline(state.verdicts, compared.length)} ${subline(state.verdicts, compared.length)}`}</p> : null}
          {!record && done && !pair ? (
            <p className="record-sum">
              Muse Spark found {pickupMarks} {pickupMarks === 1 ? 'mark' : 'marks'} on {pickupPanels.length} {pickupPanels.length === 1 ? 'panel' : 'panels'}.
            </p>
          ) : null}
          {(!done && !failed) || failed || (done && state.stage.name === 'partial') ? <StageLine state={state} /> : null}
          {!done && !failed ? <p className="record-sum">The answer shows here when the check is done. You can close this tab and come back to this address.</p> : null}
          {driverLine ? <p className="driver-line">{driverLine}</p> : null}
          {done && !pair && state.sheet.written ? (
            <div className="record-actions">
              {pdf ? (
                <a href={routes.sheetPdf(runId)} className="button">
                  Download the sealed PDF
                </a>
              ) : (
                <Link to={sheetUrl} className="button">
                  Open the sealed sheet
                </Link>
              )}
              {takesClaims ? <Link to={`/claim?run=${encodeURIComponent(runId)}`}>A bill arrived? Place its photo</Link> : null}
            </div>
          ) : null}
        </header>

        {!done && !failed ? <div className="walks-live">{walks}</div> : null}

        {topUpShown(mode !== null, state.run?.kind) ? <TopUpBanner runId={runId} coverage={coverage} onCoverage={setCoverageOverride} /> : null}

        {pickupPanels.length ? (
          <section className="rec-marks" aria-label="marks at pickup">
            <h2>Marks at pickup</h2>
            <div className="rec-mark-grid">
              {pickupPanels.map((kf) => (
                <div key={kf.kf_id}>
                  <h3 className="rec-sub">{cap(panelName(kf.panel_id))}</h3>
                  <Frame
                    kf={kf}
                    role="pickup"
                    boxes={state.marks
                      .filter((m) => m.kf_id === kf.kf_id)
                      .map((m) => m.bbox_crop)
                      .filter((b): b is Box4 => Array.isArray(b))
                      .map((box) => ({ box, tone: 'plain' as const }))}
                  />
                </div>
              ))}
            </div>
            {credit ? <p className="footage-credit">{credit}</p> : null}
          </section>
        ) : null}

        {record || failed ? (
          everything
        ) : (
          <details className="rec-more">
            <summary>{done ? 'How it was checked' : 'How it is being checked'}</summary>
            {everything}
          </details>
        )}
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
          {r === 'pickup' ? 'Pickup' : 'Return'}
        </button>
      ))}
    </div>
  );
}
