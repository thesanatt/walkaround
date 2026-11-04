import { useEffect, useMemo, type ReactNode } from 'react';

import { Bar } from '../components/Bar.tsx';
import { Frame, type BoxTone, type FrameBox } from '../components/Frame.tsx';
import { MathPanel } from '../components/MathPanel.tsx';
import { SealedPanel, sealedAt } from '../components/SealedPanel.tsx';
import type { Box4, Mark, Verdict } from '../contracts/types.ts';
import { RunFiles } from '../lib/files.tsx';
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
import { formatDate, formatStamp } from '../lib/format.ts';
import { mathFor } from '../lib/math.ts';
import { panelName } from '../lib/panels.ts';
import { comparedPanels, whyLine } from '../lib/result.ts';
import { Link, navigate } from '../lib/router.tsx';
import { replayBadgeDate } from '../lib/runview.ts';
import { useYourChecks } from '../lib/yourcheck.ts';
import { rolesOf } from '../state/reducer.ts';
import type { RunState } from '../state/types.ts';
import { useRun, type RunSource } from '../state/useRun.ts';
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
  const note =
    badge !== null ? (
      <>
        Replay<span className="bar-note-more"> of a run from {formatDate(badge)}</span>
      </>
    ) : null;
  const panel = sealedOpen ? (
    <SealedPanel runId={runId} state={state} checks={yours.checks} marks={marks} closeTo={at({ sealed: null })} recordTo={at({ sealed: null, f: null, math: null, record: '1' })} />
  ) : null;

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
        <p className="muted">{handle.status === 'error' ? `Could not open this check: ${handle.detail ?? 'unknown error'}.` : 'Opening the check'}</p>
      </div>
    );

  const list = findingsOf(state.verdicts, marks, yours.answers);
  const fid = query.get('f');
  const index = fid ? list.findIndex((f) => f.v.verdict_id === fid) : -1;

  return (
    <RunFiles runId={runId}>
      {index >= 0 ? (
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
          {state.claim ? (
            <p className="result-note">
              <Link to={at({ record: '1', f: null })}>A claim photo was placed on this check</Link>
            </p>
          ) : null}
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
      {mathOpen ? <MathPanel math={mathFor(v, marks, match, state.looks)} id="math" /> : null}
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

