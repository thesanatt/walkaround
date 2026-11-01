import { useEffect, useState } from 'react';

import { api, ApiError } from '../api/client.ts';
import { MarkFrame } from '../components/MarkCard.tsx';
import type { Keyframe, Mark, StatusResponse, Verdict } from '../contracts/types.ts';
import { RunFiles } from '../lib/files.tsx';
import { panelName } from '../lib/panels.ts';
import { Link, navigate } from '../lib/router.tsx';

interface SampleFrame {
  runId: string;
  kf: Keyframe;
  marks: Mark[];
  credit: string | null;
}

// One real frame from the sample pair: the pickup crop of its first "already there" mark, boxed. Read as a viewer who
// is not the run owner, like the kiosk. Nothing shows when the server has no sample or the files do not load.
function useSampleFrame(): SampleFrame | null {
  const [frame, setFrame] = useState<SampleFrame | null>(null);
  useEffect(() => {
    let live = true;
    void (async () => {
      try {
        const { run_id: runId } = await api.sample();
        const [run, verdicts, marks, keyframes] = await Promise.all([
          api.run(runId),
          api.file<Verdict[]>(runId, 'derived/verdicts.json', 'kiosk'),
          api.file<Mark[]>(runId, 'derived/marks.json', 'kiosk'),
          api.file<Keyframe[]>(runId, 'derived/keyframes.json', 'kiosk'),
        ]);
        const v = verdicts.find((x) => x.label === 'already_there');
        const mark = marks.find((m) => v?.mark_ids.includes(m.mark_id) && m.role === 'pickup');
        const kf = keyframes.find((k) => k.kf_id === mark?.kf_id);
        if (live && mark && kf?.crop) setFrame({ runId, kf, marks: [mark], credit: run.footage_credit ?? null });
      } catch {
        // no sample on this server: the page reads fine without the picture
      }
    })();
    return () => {
      live = false;
    };
  }, []);
  return frame;
}

export function HomePage({ status }: { status: StatusResponse | null }) {
  const frame = useSampleFrame();
  const [sampleNote, setSampleNote] = useState<string | null>(null);

  const openSample = async () => {
    setSampleNote(null);
    try {
      const s = await api.sample();
      navigate(`/run/${encodeURIComponent(s.run_id)}?replay=fixed`);
    } catch (err) {
      setSampleNote(err instanceof ApiError && err.status === 404 ? 'There is no sample pair on this server yet.' : err instanceof Error ? err.message : String(err));
    }
  };

  return (
    <div className="home">
      <section className="home-hero">
        <div>
          <p className="home-problem">Rental companies bill renters for damage they did not cause.</p>
          <h1>Film the car at pickup and at return, and Walkaround shows panel by panel what was already there, with sealed, timestamped proof.</h1>
          <div className="home-actions">
            <Link to="/upload" className="cds--btn cds--btn--primary">
              Start with your two videos
            </Link>
            <button type="button" className="cds--btn cds--btn--tertiary" onClick={() => void openSample()}>
              Open the sample pair
            </button>
          </div>
          {sampleNote ? <p className="home-note">{sampleNote}</p> : null}
          {status?.paused ? <p className="home-note">Live checks are off on this server right now. The sample pair always opens.</p> : null}
        </div>
        {frame ? (
          <RunFiles runId={frame.runId} view="kiosk">
            <figure className="home-figure">
              <MarkFrame kf={frame.kf} marks={frame.marks} role="pickup" zoom={false} />
              <figcaption>
                From the sample pair: the {panelName(frame.kf.panel_id)} at pickup. The boxed spot is in the same place at return, so it was already there.
              </figcaption>
              {frame.credit ? <p className="footage-credit">{frame.credit}</p> : null}
            </figure>
          </RunFiles>
        ) : (
          <div className="home-figure home-figure-empty" aria-hidden="true" />
        )}
      </section>

      <section className="home-more" aria-label="more">
        <div>
          <h2>Got a damage bill with a photo?</h2>
          <p>
            <Link to="/claim">Place the claim photo</Link> against your pickup video. Walkaround finds the panel and shows your dated pickup frame next to it.
          </p>
        </div>
        <div>
          <h2>Want to check a sheet yourself?</h2>
          <p>
            Every sheet carries hashes and timestamps you can check with standard tools. <Link to="/verify">Here is how.</Link>
          </p>
        </div>
      </section>
    </div>
  );
}
