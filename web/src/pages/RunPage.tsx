import { useEffect, useMemo, useState, type ReactNode } from 'react';

import { Bar, ReplayNote } from '../components/Bar.tsx';
import { Frame, type BoxTone, type FrameBox } from '../components/Frame.tsx';
import { MathPanel } from '../components/MathPanel.tsx';
import { Nothing } from '../components/Nothing.tsx';
import { SealedPanel, sealedAt } from '../components/SealedPanel.tsx';
import type { Box4, Claim, Mark, Verdict } from '../contracts/types.ts';
import { claimPhotoPath, claimWorkSize, effectiveExit, EXIT_WORDS } from '../lib/claim.ts';
import { RunFiles, useFileUrl } from '../lib/files.tsx';
import {
  ANSWER_WORD,
  answerLines,
  answersFor,
  cap,
  findingsOf,
  GROUP_WORD,
  groupsOf,
  sideOf,
  spotBox,
  thumbRole,
  type Answer,
  type Finding,
  type Group,
} from '../lib/findings.ts';
import { formatStamp } from '../lib/format.ts';
import { mathFor } from '../lib/math.ts';
import { panelName } from '../lib/panels.ts';
import { comparedPanels, creditLine, whyLine } from '../lib/result.ts';
import { Link, navigate } from '../lib/router.tsx';
import { replayBadgeDate } from '../lib/runview.ts';
import { useYourChecks } from '../lib/yourcheck.ts';
import { rolesOf } from '../state/reducer.ts';
import type { RunState } from '../state/types.ts';
import { useRun, type RunSource } from '../state/useRun.ts';
import { claimFrame, ClaimSteps } from './ClaimPage.tsx';
import { replayModeOf, RunRecord } from './RunRecord.tsx';

// The run page. A finished pair opens on the answer and its findings (?f=<verdict> for one finding, &math=1 with the
// math open); ?sealed=1 lays the sealed panel over either; ?record=1 is everything the check recorded. A check that
// is still running, a pickup-only check and a failed one show the record view, which carries the walks and the stage.

const TONE: Record<Group, BoxTone> = { new: 'new', eye: 'eye', already: 'plain', gone: 'plain' };

function withQuery(runId: string, query: URLSearchParams, patch: Record<string, string | null>): string {
  const q = new URLSearchParams(query);
  for (const [k, v] of Object.entries(patch)) {
    if (v === null) q.delete(k);
    else q.set(k, v);
  }
  const s = q.toString();
  return `/run/${encodeURIComponent(runId)}${s ? `?${s}` : ''}`;
}

// The label a finding shows: the renter's answer when they gave one, else code's word.
function labelOf(f: Finding): string {
  if (f.answer) return ANSWER_WORD[f.answer];
  if (f.v.label === 'not_seen_at_return') return 'Not seen at return';
  return GROUP_WORD[f.system];
}

function boxesOf(f: Finding, kfMarks: Mark[], spot: Box4 | null): FrameBox[] {
  const tone = TONE[f.group];
  const out: FrameBox[] = kfMarks.map((m) => m.bbox_crop).filter((b): b is Box4 => Array.isArray(b)).map((box) => ({ box, tone }));
  if (!out.length && spot) out.push({ box: spot, tone: 'spot' });
  return out;
}

export function RunPage({ runId, query }: { runId: string; query: URLSearchParams }) {
  const mode = replayModeOf(query);
  const source = useMemo<RunSource>(() => (mode ? { kind: 'replay', mode } : { kind: 'live' }), [mode]);
  const handle = useRun(runId, source);
  const { state } = handle;
  const roles = rolesOf(state);
  const yours = useYourChecks(runId);
  const marks = useMemo(() => new Map(state.marks.map((m) => [m.mark_id, m])), [state.marks]);

  const done = state.completed || state.stage.name === 'done' || state.stage.name === 'partial';
  const failed = state.stage.name === 'failed';
  const pair = roles.length > 1;
  const record = query.get('record') === '1';
  const sealedOpen = query.get('sealed') === '1';
  const at = (patch: Record<string, string | null>) => withQuery(runId, query, patch);
  const stamp = sealedAt(state);
  const sealedLink = state.seal.manifest_sha256 ? (
    <Link to={at({ sealed: '1' })} className="sealed-link">
      Sealed <span className="muted">{formatStamp(stamp)}</span>
    </Link>
  ) : null;
  const badge = replayBadgeDate(state);
  // a replay says so beside the name, on every screen of it
  const note = badge !== null ? <ReplayNote date={badge} /> : null;
  const panel = sealedOpen ? (
    <SealedPanel runId={runId} state={state} checks={yours.checks} marks={marks} closeTo={at({ sealed: null })} recordTo={at({ sealed: null, f: null, math: null, record: '1' })} />
  ) : null;

  // No run at this address, or the server would not give it: one plain screen, never a half-drawn page.
  if (handle.status === 'error' && !state.run && !done) return <Nothing line={handle.detail ? `The server said: ${handle.detail}.` : 'The server could not open it.'} />;

  if (record || failed || (done && !pair) || (!done && mode !== 'instant'))
    return (
      <>
        <RunRecord runId={runId} mode={mode} handle={handle} record={record} bar={sealedLink} />
        {panel}
      </>
    );

  if (!done)
    return (
      <div className="opening">
        <Bar />
        <p className="muted" role="status">
          Opening the check
        </p>
      </div>
    );

  const list = findingsOf(state.verdicts, marks, yours.answers);
  const fid = query.get('f');
  const index = fid ? list.findIndex((f) => f.v.verdict_id === fid) : -1;

  const claimOpen = query.get('claim') === '1' && state.claim !== null;

  return (
    <RunFiles runId={runId}>
      {claimOpen && state.claim ? (
        <ClaimScreen state={state} claim={state.claim} note={note} bar={sealedLink} back={at({ claim: null })} />
      ) : index >= 0 ? (
        <FindingScreen
          key={fid}
          state={state}
          list={list}
          index={index}
          marks={marks}
          note={note}
          bar={sealedLink}
          at={at}
          mathOpen={query.get('math') === '1'}
          onAnswer={yours.set}
          panelOpen={sealedOpen}
        />
      ) : (
        <ResultScreen state={state} list={list} marks={marks} note={note} bar={sealedLink} at={at} />
      )}
      {panel}
    </RunFiles>
  );
}

function ResultScreen({
  state,
  list,
  marks,
  note,
  bar,
  at,
}: {
  state: RunState;
  list: Finding[];
  marks: Map<string, Mark>;
  note: ReactNode;
  bar: ReactNode;
  at: (p: Record<string, string | null>) => string;
}) {
  const compared = comparedPanels(state.matches, state.verdicts).length;
  const answer = answerLines(state.verdicts, compared);
  const extraNew = list.filter((f) => f.answer === 'new').length;
  const groups = groupsOf(list);
  const kinds = new Set(state.matches.flatMap((m) => (m.by_eye ?? []).map((b) => b.kind)));
  const byEye = kinds.has('wheels') && kinds.has('glass') ? 'Wheels and glass' : kinds.has('wheels') ? 'Wheels' : kinds.has('glass') ? 'Glass' : null;
  const credit = state.run?.footage_credit ?? null;

  return (
    <div className="result">
      <Bar note={note}>{bar}</Bar>
      <div className="result-grid">
        <section className="result-answer" aria-live="polite">
          <h1 className={`answer-big ${answer.tone === 'new' ? 'new' : ''}`}>{answer.big}</h1>
          {answer.where ? <p className={`answer-where ${answer.tone === 'eye' ? 'eye' : ''}`}>{answer.where}</p> : null}
          {extraNew ? <p className="answer-yours">{`and ${extraNew} more by your check`}</p> : null}
          {compared ? <p className="answer-count">{`${compared} ${compared === 1 ? 'panel' : 'panels'} compared`}</p> : null}
        </section>
        <section className="result-list" aria-label="findings">
          {state.claim ? (
            <div className="find-group claim">
              <h2>
                <span>Claim photo</span> <span className="find-count">{state.claims.length || 1}</span>
              </h2>
              <ul>
                <li>
                  <Link to={at({ claim: '1', f: null, math: null, sealed: null })} className="find-row">
                    <ClaimThumb claim={state.claim} />
                    <span className="find-text">
                      <span className="find-panel">{state.claim.panel_id ? cap(panelName(state.claim.panel_id)) : 'No panel named'}</span>
                      {state.claim.text ? <span className="find-words">{state.claim.text}</span> : null}
                    </span>
                  </Link>
                </li>
              </ul>
            </div>
          ) : null}
          {groups.map((g) => {
            const allYours = g.items.every((f) => f.answer);
            const title = g.group === 'gone' && !allYours ? 'Not seen at return' : GROUP_WORD[g.group];
            return (
              <div key={g.group} className={`find-group ${g.group}`}>
                <h2>
                  <span>{title}</span> <span className="find-count">{g.items.length}</span>
                </h2>
                <ul>
                  {g.items.map((f) => (
                    <li key={f.v.verdict_id}>
                      <Row f={f} state={state} marks={marks} to={at({ f: f.v.verdict_id, math: null, sealed: null })} />
                    </li>
                  ))}
                </ul>
              </div>
            );
          })}
          {byEye ? <p className="result-note">{byEye}: check by eye</p> : null}
          {credit ? <p className="credit">{credit}</p> : null}
        </section>
      </div>
    </div>
  );
}

function Row({ f, state, marks, to }: { f: Finding; state: RunState; marks: Map<string, Mark>; to: string }) {
  const match = state.matches.find((m) => m.panel_id === f.v.panel_id);
  const side = sideOf(f.v, thumbRole(f.v, marks), marks, state.keyframes, match);
  return (
    <Link to={to} className="find-row">
      <Frame kf={side.kf} role={side.role} boxes={boxesOf(f, side.own, null)} thumb />
      <span className="find-text">
        <span className="find-panel">{cap(panelName(f.v.panel_id))}</span>
        {f.words ? <span className="find-words">{f.words}</span> : null}
      </span>
      {f.answer ? <span className="tag">your check</span> : null}
    </Link>
  );
}

function FindingScreen({
  state,
  list,
  index,
  marks,
  note,
  bar,
  at,
  mathOpen,
  onAnswer,
  panelOpen,
}: {
  state: RunState;
  list: Finding[];
  index: number;
  marks: Map<string, Mark>;
  note: ReactNode;
  bar: ReactNode;
  at: (p: Record<string, string | null>) => string;
  mathOpen: boolean;
  onAnswer: (verdictId: string, answer: Answer | null) => void;
  panelOpen: boolean;
}) {
  const f = list[index]!;
  const v = f.v;
  const prev = index > 0 ? at({ f: list[index - 1]!.v.verdict_id, math: null }) : null;
  const next = index < list.length - 1 ? at({ f: list[index + 1]!.v.verdict_id, math: null }) : null;
  const back = at({ f: null, math: null });
  const match = state.matches.find((m) => m.panel_id === v.panel_id);
  const tone = f.answer ? (f.answer === 'new' ? 'new' : 'plain') : TONE[f.system];

  // Left and right step between findings, Escape goes back to the list; never while the sealed panel is open or a
  // key is meant for a field.
  useEffect(() => {
    if (panelOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey || e.shiftKey) return;
      const el = e.target as HTMLElement | null;
      if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT' || el.isContentEditable)) return;
      if (e.key === 'ArrowLeft' && prev) navigate(prev, { replace: true });
      else if (e.key === 'ArrowRight' && next) navigate(next, { replace: true });
      else if (e.key === 'Escape') navigate(back);
      else return;
      e.preventDefault();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [prev, next, back, panelOpen]);

  const sides = (['pickup', 'return'] as const).map((role) => sideOf(v, role, marks, state.keyframes, match));
  const credit = state.run?.footage_credit ?? null;

  return (
    <div className="finding">
      <Bar note={note}>{bar}</Bar>
      <nav className="finding-nav" aria-label="findings">
        <Link to={back}>All findings</Link>
        <span className="finding-step">
          {prev ? (
            <Link to={prev} replace>
              Previous
            </Link>
          ) : (
            <span className="off">Previous</span>
          )}
          <span className="finding-pos">
            {index + 1} of {list.length}
          </span>
          {next ? (
            <Link to={next} replace>
              Next
            </Link>
          ) : (
            <span className="off">Next</span>
          )}
        </span>
      </nav>
      <header className="finding-head">
        <p className={`finding-label ${tone}`}>
          {labelOf(f)}
          {f.answer ? <span className="tag">your check</span> : null}
        </p>
        <h1>{cap(panelName(v.panel_id))}</h1>
        {f.words ? <p className="finding-words">{cap(f.words)}</p> : null}
        <p className="finding-why">{whyLine(v)}</p>
      </header>
      <div className="frames">
        {sides.map((s) => (
          <Frame key={s.role} kf={s.kf} role={s.role} boxes={boxesOf(f, s.own, spotBox(s.spot))} />
        ))}
      </div>
      {credit ? <p className="credit">{credit}</p> : null}
      {v.label === 'needs_your_eye' ? <YourCheck v={v} marks={marks} answer={f.answer} onAnswer={onAnswer} /> : null}
      <p className="math-toggle">
        <Link to={at({ math: mathOpen ? null : '1' })} replace>
          {mathOpen ? 'Hide the math' : 'Show the math'}
        </Link>
      </p>
      {mathOpen ? <MathPanel math={mathFor(v, marks, match, state.looks)} credit={creditLine(v, marks, state.looks)} id="math" /> : null}
    </div>
  );
}

function YourCheck({ v, marks, answer, onAnswer }: { v: Verdict; marks: Map<string, Mark>; answer: Answer | null; onAnswer: (verdictId: string, answer: Answer | null) => void }) {
  const options = answersFor(v, marks);
  return (
    <section className={`yourcheck${answer ? ' done' : ''}`} aria-label="your check">
      <div className="yourcheck-row">
        <h2>Your check</h2>
        {answer ? (
          <p className="yourcheck-done">
            <strong>{ANSWER_WORD[answer]}</strong>
            <button type="button" className="link-button" onClick={() => onAnswer(v.verdict_id, null)}>
              Undo
            </button>
          </p>
        ) : (
          <div className="yourcheck-buttons">
            {options.map((a) => (
              <button key={a} type="button" className="button quiet" onClick={() => onAnswer(v.verdict_id, a)}>
                {ANSWER_WORD[a]}
              </button>
            ))}
          </div>
        )}
      </div>
      <p className="yourcheck-note">Kept in this browser. The sealed sheet does not change.</p>
    </section>
  );
}

function ClaimThumb({ claim }: { claim: Claim }) {
  const url = useFileUrl();
  const rel = claimPhotoPath(claim);
  if (!rel) return <span className="frame-thumb empty" aria-hidden="true" />;
  return (
    <span className="frame-thumb claim-thumb" aria-hidden="true">
      <img src={url(rel)} alt="" />
    </span>
  );
}

// A claim photo placed against the run: the photo from the bill beside the sealed pickup frame of that panel, the
// exit or the placement in one line, and how it was placed behind a toggle.
function ClaimScreen({ state, claim, note, bar, back }: { state: RunState; claim: Claim; note: ReactNode; bar: ReactNode; back: string }) {
  const url = useFileUrl();
  const [open, setOpen] = useState(false);
  const ctx = { keyframes: state.keyframes, marks: state.marks, coverage: state.coverage };
  const hit = claimFrame(claim, ctx);
  const exit = effectiveExit(claim);
  const rel = claimPhotoPath(claim);
  const [w, h] = claimWorkSize(claim.image);
  const roi = claim.roi?.source === 'drawn' ? claim.roi.box : null;
  const sealed = sealedAt(state);
  const pct = (v: number, of: number) => `${((100 * v) / of).toFixed(3)}%`;
  return (
    <div className="finding claim-screen">
      <Bar note={note}>{bar}</Bar>
      <nav className="finding-nav" aria-label="findings">
        <Link to={back}>All findings</Link>
      </nav>
      <header className="finding-head">
        <p className="finding-label">Claim photo</p>
        <h1>{claim.panel_id ? cap(panelName(claim.panel_id)) : 'No panel named'}</h1>
        {claim.text ? <p className="finding-words">The bill says: {claim.text}</p> : null}
        <p className="finding-why">{exit ? EXIT_WORDS[exit] : 'Placed against the sealed pickup walk, beside its frame of the same panel.'}</p>
      </header>
      <div className="frames">
        <figure className="frame">
          <div className="frame-view" style={{ aspectRatio: `${w} / ${h}` }} role="img" aria-label="the claim photo, plates and faces pixelated">
            {rel ? <img src={url(rel)} alt="" /> : null}
            {roi ? (
              <span
                className="frame-box eye"
                style={{ left: pct((roi[0] + roi[2]) / 2, w), top: pct((roi[1] + roi[3]) / 2, h), width: `max(56px, ${pct(roi[2] - roi[0], w)})`, height: `max(56px, ${pct(roi[3] - roi[1], h)})` }}
              />
            ) : null}
          </div>
          <figcaption>Claim photo</figcaption>
        </figure>
        {hit ? (
          <Frame kf={hit.kf} role={hit.kf.role} boxes={hit.mark?.bbox_crop ? [{ box: hit.mark.bbox_crop, tone: 'plain' }] : []} whole />
        ) : (
          <figure className="frame empty">
            <div className="frame-view">No pickup frame of this panel in the run.</div>
          </figure>
        )}
      </div>
      <p className="credit">
        {state.run?.keep ? 'The claim photo is a stand-in made for this example; a real one comes with the bill. ' : ''}
        {sealed ? `The pickup walk was sealed ${formatStamp(sealed, true)}. ` : ''}
        {state.run?.footage_credit ?? ''}
      </p>
      <p className="math-toggle">
        <button type="button" className="link-button" aria-expanded={open} onClick={() => setOpen(!open)}>
          {open ? 'Hide how it was placed' : 'Show how it was placed'}
        </button>
      </p>
      {open ? (
        <section className="math claim-how" aria-label="how the photo was placed">
          <ClaimSteps claim={claim} />
          {claim.paragraph ? <p className="claim-paragraph">{claim.paragraph}</p> : null}
          <p className="small muted mono">
            photo {claim.image.w}x{claim.image.h}, sha256 {claim.image.sha256.slice(0, 12)}
          </p>
        </section>
      ) : null}
    </div>
  );
}

