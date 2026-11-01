import { useEffect, useState } from 'react';

import { api, routes } from '../api/client.ts';
import { cap, CARD_WORD, MarkFrame } from '../components/MarkCard.tsx';
import type { Claim, Keyframe, Mark, PanelMatch, SampleRow, StatusResponse, Verdict } from '../contracts/types.ts';
import { claimPhotoPath } from '../lib/claim.ts';
import { RunFiles } from '../lib/files.tsx';
import { bestKeyframe } from '../lib/focus.ts';
import { panelName } from '../lib/panels.ts';
import { clearVerdicts, comparedPanels, headline, markWords } from '../lib/result.ts';
import { Link, navigate } from '../lib/router.tsx';

// One finished check as the home page shows it: its first answer card (new first, then already there) with both
// frames, and its answer line. Read as a viewer who is not the run owner, like the kiosk.
interface Finished {
  row: SampleRow;
  line: string;
  card: { v: Verdict; pickup: { kf: Keyframe | undefined; marks: Mark[] }; ret: { kf: Keyframe | undefined; marks: Mark[] } } | null;
  credit: string | null;
  photo: string | null; // a claim sample's claim photo, run-relative
}

function frameOf(v: Verdict, role: string, marks: Mark[], keyframes: Keyframe[]) {
  const own = marks.filter((m) => v.mark_ids.includes(m.mark_id) && m.role === role);
  const kf = keyframes.find((k) => k.kf_id === own[0]?.kf_id) ?? bestKeyframe(keyframes, role, v.panel_id);
  return { kf, marks: own.filter((m) => m.kf_id === kf?.kf_id) };
}

async function loadFinished(row: SampleRow): Promise<Finished> {
  const id = row.run_id;
  const soft = <T,>(p: Promise<T>) => p.catch(() => null);
  const [run, verdicts, marks, keyframes, matches, claim] = await Promise.all([
    soft(api.run(id)),
    soft(api.file<Verdict[]>(id, 'derived/verdicts.json', 'kiosk')),
    soft(api.file<Mark[]>(id, 'derived/marks.json', 'kiosk')),
    soft(api.file<Keyframe[]>(id, 'derived/keyframes.json', 'kiosk')),
    soft(api.file<PanelMatch[]>(id, 'derived/matches.json', 'kiosk')),
    row.claims ? soft(api.file<Claim>(id, `claim/claim_${row.claims}.json`, 'kiosk')) : Promise.resolve(null),
  ]);
  const vs = verdicts ?? [];
  const v = clearVerdicts(vs)[0];
  const card = v && marks && keyframes ? { v, pickup: frameOf(v, 'pickup', marks, keyframes), ret: frameOf(v, 'return', marks, keyframes) } : null;
  const line =
    claim?.panel_id && !claim.exit
      ? `A claim photo placed on the ${panelName(claim.panel_id)}, next to the sealed pickup frame of that spot.`
      : vs.length
        ? headline(vs, comparedPanels(matches ?? [], vs).length)
        : 'A finished check.';
  return { row, line, card, credit: run?.footage_credit ?? null, photo: claim ? claimPhotoPath(claim) : null };
}

function useFinished(): Finished[] | null {
  const [list, setList] = useState<Finished[] | null>(null);
  useEffect(() => {
    let live = true;
    void (async () => {
      try {
        const { samples } = await api.samples();
        const loaded = await Promise.all(samples.map(loadFinished));
        if (live) setList(loaded);
      } catch {
        // an older server has only /api/sample: one row, read the same way
        try {
          const s = await api.sample();
          const one = await loadFinished({ run_id: s.run_id, title: null, kind: 'pair', recorded_on: s.recorded_on, claims: 0, first: true });
          if (live) setList([one]);
        } catch {
          if (live) setList([]);
        }
      }
    })();
    return () => {
      live = false;
    };
  }, []);
  return list;
}

const replayUrl = (id: string) => `/run/${encodeURIComponent(id)}?replay=fixed`;

function HeroCard({ f }: { f: Finished }) {
  const c = f.card!;
  const words = markWords(c.v, new Map([...c.pickup.marks, ...c.ret.marks].map((m) => [m.mark_id, m])));
  return (
    <RunFiles runId={f.row.run_id} view="kiosk">
      <figure className={`home-card mark-card ${c.v.label}`}>
        <p className="home-card-head">
          <span className={`label ${c.v.label}`}>{CARD_WORD[c.v.label]}</span> {cap(panelName(c.v.panel_id))}
          {words ? <span className="mark-words">: {words}</span> : null}
        </p>
        <div className="mark-pair mark-pair-2">
          <MarkFrame kf={c.pickup.kf} marks={c.pickup.marks} role="pickup" />
          <MarkFrame kf={c.ret.kf} marks={c.ret.marks} role="return" />
        </div>
        <figcaption>
          {f.row.title ? `${f.row.title}. ` : ''}
          <Link to={replayUrl(f.row.run_id)}>Open this check</Link>
        </figcaption>
        {f.credit ? <p className="footage-credit">{f.credit}</p> : null}
      </figure>
    </RunFiles>
  );
}

function Thumb({ f }: { f: Finished }) {
  if (f.photo)
    return (
      <span className="finished-thumb" aria-hidden="true">
        <img src={routes.file(f.row.run_id, f.photo, 'kiosk')} alt="" />
      </span>
    );
  const c = f.card;
  const side = c ? (c.v.label === 'new_or_unseen' ? c.ret : c.pickup) : null;
  if (!side?.kf) return <span className="finished-thumb empty" aria-hidden="true" />;
  return (
    <RunFiles runId={f.row.run_id} view="kiosk">
      <span className="finished-thumb" aria-hidden="true">
        <MarkFrame kf={side.kf} marks={side.marks} role={side === c!.ret ? 'return' : 'pickup'} tight />
      </span>
    </RunFiles>
  );
}

export function HomePage({ status }: { status: StatusResponse | null }) {
  const finished = useFinished();
  const hero = finished?.find((f) => f.card) ?? null;
  const first = finished?.[0] ?? null;
  const live = Boolean(status && !status.paused && status.pairs_left > 0);

  const sampleButton = first ? (
    <Link to={replayUrl(first.row.run_id)} className="cds--btn cds--btn--primary">
      See a finished check
    </Link>
  ) : null;

  return (
    <div className="home">
      <section className="home-hero">
        <div className="home-words">
          <p className="home-problem">Rental companies scan returned cars with AI. Renters get the bill.</p>
          <h1>Your pickup video, read like evidence.</h1>
          <p className="home-lede">
            Film one lap of the car at pickup and one at return. Walkaround seals both videos, tells you which panels you missed while you are still at the car,
            and puts the pickup frame of each spot next to the return frame.
          </p>
          <div className="home-actions">
            {live ? (
              <>
                <Link to="/upload" className="cds--btn cds--btn--primary">
                  Check your two videos
                </Link>
                {first ? <Link to={replayUrl(first.row.run_id)}>or see a finished check</Link> : null}
              </>
            ) : (
              <>
                {sampleButton ?? (
                  <Link to="/upload" className="cds--btn cds--btn--primary">
                    Check your two videos
                  </Link>
                )}
                {sampleButton ? <Link to="/upload">Check your own videos</Link> : null}
              </>
            )}
          </div>
        </div>
        {hero ? <HeroCard f={hero} /> : <div className="home-card-empty" aria-hidden="true" />}
      </section>

      <section className="home-moments" aria-label="when to use it">
        <div>
          <h2>Before you drive off</h2>
          <p>One slow lap, about a minute. You see which panels the video missed and film just those, while you are still in the lot.</p>
        </div>
        <div>
          <h2>When a bill arrives</h2>
          <p>
            <Link to="/claim">Drop the bill's photo.</Link> Walkaround finds the panel and puts your sealed pickup frame of that spot beside it, with the time it was
            sealed.
          </p>
        </div>
      </section>

      {finished && finished.length ? (
        <section className="home-finished" aria-label="finished checks">
          <h2>Finished checks you can open</h2>
          <ul className="finished-list">
            {finished.map((f) => (
              <li key={f.row.run_id}>
                <button type="button" className="finished-row" onClick={() => navigate(replayUrl(f.row.run_id))}>
                  <Thumb f={f} />
                  <span className="finished-words">
                    <span className="finished-title">{f.row.title ?? f.row.run_id}</span>
                    <span className="finished-line">{f.line}</span>
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      <section className="home-foot">
        <p>
          Built Oct 26 to Nov 4, 2026 on Meta's SAM 3.1, Muse Spark 1.3, Muse Voice and Muse Spark Open. Code makes every call; a model can only make it more
          careful. <Link to="/verify">Check a sheet yourself.</Link>
        </p>
      </section>
    </div>
  );
}
