import type { ReactNode } from 'react';

import { Bar } from '../components/Bar.tsx';
import { Link } from '../lib/router.tsx';
import pickupFrame from '../example/pickup.jpg';
import returnFrame from '../example/return.jpg';
import comparePickup from '../how/compare_pickup.jpg';
import compareReturn from '../how/compare_return.jpg';
import lookPickup from '../how/look_pickup.jpg';
import lookReturn from '../how/look_return.jpg';
import samFrame from '../how/sam.jpg';
import sparkCrop from '../how/spark.jpg';
import wheelCrop from '../how/wheel.jpg';

// /how: the method in seven steps, every picture taken from the sample run hero_focus5e (src/how/, made from the run's
// blurred proxies, its crops and the app's own renders). Sources of each fact are in the comments by the step.
const RESULT = '/run/hero_focus5e?replay=instant';
const MATH = `${RESULT}&f=right_rear_door_v2&math=1`;

type Shot = { src: string; alt: string; caption: string; w: number; h: number };

function Step({ n, title, children }: { n: number; title: string; children: ReactNode }) {
  return (
    <li className="how-step">
      <h2>
        <span className="how-num" aria-hidden="true">
          {n}
        </span>
        {title}
      </h2>
      {children}
    </li>
  );
}

function Pair({ shots, caption, square = false }: { shots: [Shot, Shot]; caption?: string; square?: boolean }) {
  return (
    <figure className={`how-fig${square ? ' how-square' : ''}`}>
      <div className="how-pair">
        {shots.map((s) => (
          <div key={s.caption} className="how-shot">
            <img src={s.src} alt={s.alt} width={s.w} height={s.h} loading="lazy" />
            <p className="how-shot-cap">{s.caption}</p>
          </div>
        ))}
      </div>
      {caption ? <figcaption>{caption}</figcaption> : null}
    </figure>
  );
}

export function HowPage() {
  return (
    <div className="how">
      <Bar />
      <header className="how-head">
        <h1>How it works</h1>
        <p className="how-lede">What happens between your two videos, shown on the example of a friend's Ford Focus.</p>
      </header>
      <ol className="how-steps">
        {/* camera times: videos/pickup.json and videos/return.json creation_time, 19:46:29Z and 19:50:42Z (EST) */}
        <Step n={1} title="Two videos">
          <p className="how-text">
            Film a slow lap around the car when you get it, and again when you hand it back. The example was filmed at
            2:46&nbsp;PM and 2:50&nbsp;PM on Nov&nbsp;2.
          </p>
          <Pair
            shots={[
              { src: pickupFrame, w: 960, h: 540, caption: 'Pickup 2:46 PM', alt: 'The Ford Focus from the rear right in the pickup video' },
              { src: returnFrame, w: 960, h: 540, caption: 'Return 2:50 PM', alt: 'The same car from the same side in the return video' },
            ]}
            caption="Blue tape stands in for damage, but the scrape low on the left front door is real."
          />
        </Step>

        {/* 16 car parts: README, How it works, step 2; the frame is the record view's pickup walk at proxy frame 660 */}
        <Step n={2} title="Meta SAM 3.1 finds every panel">
          <p className="how-text">
            SAM 3.1 tracks 16 car parts through each video, so each door, fender and bumper is compared with the same
            panel in the other video.
          </p>
          <figure className="how-fig">
            <img src={samFrame} width={1342} height={756} loading="lazy" alt="The pickup video at 0:44 with SAM 3.1's outlines filling the right rear door and the right front door, each labelled" />
            <figcaption>Pickup video at 0:44, with SAM 3.1's outlines on the two right doors.</figcaption>
          </figure>
        </Step>

        {/* the three boxes and their words: derived/marks.json, keyframe return_left_front_door_kf1 */}
        <Step n={3} title="Meta Muse Spark boxes the marks">
          <p className="how-text">On each panel, Muse Spark boxes every mark it sees and says what it looks like.</p>
          <figure className="how-fig how-fig-narrow">
            <img src={sparkCrop} width={772} height={752} loading="lazy" alt="The left front door in the return video with three white boxes: one around a blue tape sticker and two around a scratch low on the door" />
            <figcaption>
              Left front door, return video. Muse Spark called its three boxes a small blue tape sticker, a dark scratch
              on lower door and a small dark scratch.
            </figcaption>
          </figure>
        </Step>

        {/* 0.915 and 0.849: the verdict right_rear_door_v2's trace, d and tau_eye_r (match.py lines 1999 and 2007) */}
        <Step n={4} title="Plain code compares the same spot">
          <p className="how-text">
            Code lines the two views of a panel up and measures how far each mark is from the nearest mark in the other
            video. Code makes the call, not a model.
          </p>
          <p className="how-big">0.915 &gt; 0.849: no mark there at pickup, so it's new.</p>
          <Pair
            shots={[
              { src: comparePickup, w: 860, h: 574, caption: 'Pickup 0:40', alt: 'The right rear door in the pickup video, the spot outlined with a black and white dash and nothing in it' },
              { src: compareReturn, w: 860, h: 574, caption: 'Return 0:25', alt: 'The right rear door in the return video, a blue tape strip boxed in red' },
            ]}
            caption="Right rear door."
          />
          <p className="how-link">
            <Link to={MATH}>Show the math</Link>
          </p>
        </Step>

        {/* model and host: derived/looks.json (meta-models/muse-spark-open-1026, engine baseten); asked 3 times, same
            mark 0 of 3: the math view's second look; the two pictures are the look's own input_refs */}
        <Step n={5} title="A second look on open weights">
          <p className="how-text">
            A mark seen in one video only gets a second look from Muse Spark Open, an open-weights model on Baseten. It
            can only lower a label to needs your eye, never raise one. Asked 3 times about this strip, it never saw the
            same mark at pickup, so New stands.
          </p>
          <Pair
            shots={[
              { src: lookReturn, w: 948, h: 962, caption: 'Return, the mark outlined', alt: 'The crop of the right rear door from the return video that the second look was shown, the blue strip outlined' },
              { src: lookPickup, w: 1000, h: 979, caption: 'Pickup, the same spot outlined', alt: 'The crop of the right rear door from the pickup video that the second look was shown, the same spot outlined and empty' },
            ]}
            square
          />
        </Step>

        {/* sheet/sheet.json: originals[].name and sha256, clocks.sealed_at 23:42:53Z and first_model_call_at 23:43:16Z */}
        <Step n={6} title="Sealed before any model sees them">
          <p className="how-text">
            Both videos are hashed and timestamped as they arrive, before any model sees them. Anyone can check the
            sheet against the hashes.
          </p>
          <dl className="how-seal">
            <div>
              <dt>Pickup video</dt>
              <dd className="mono">0b1d3a373eeccd030eb8cc1eeadfa408d929a48cbabcf2f90481923bf4eca2cf</dd>
            </div>
            <div>
              <dt>Return video</dt>
              <dd className="mono">cc683f02ad13d63d148fb5568724560f5c35e3d15202edcf7a21609cc23024f3</dd>
            </div>
            <div>
              <dt>Sealed</dt>
              <dd>Nov 2, 6:42:53 PM EST</dd>
            </div>
            <div>
              <dt>First model call</dt>
              <dd>Nov 2, 6:43:16 PM EST</dd>
            </div>
          </dl>
          <p className="how-link">
            <Link to="/verify">Verify it yourself</Link>
          </p>
        </Step>

        {/* wheels and glass: the result page's own line, and README Limits */}
        <Step n={7} title="What it won't do">
          <p className="how-text">
            Wheels and glass are not compared, so check them by eye. Alibi is evidence support, not legal advice.
          </p>
          <figure className="how-fig how-fig-narrow">
            <img src={wheelCrop} width={660} height={500} loading="lazy" alt="The rear wheel and the rear windows of the Ford Focus in the pickup video" />
            <figcaption>The wheels and the windows are yours to check.</figcaption>
          </figure>
          <p className="how-link">
            <Link to={RESULT}>See the full result</Link>
          </p>
        </Step>
      </ol>
    </div>
  );
}
