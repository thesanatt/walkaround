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
    <>
      <Bar />
      <div className="how">
        <header className="how-head">
          <h1>How it works</h1>
          <p className="how-lede">What happens between your two videos, shown on the example of a friend's Ford Focus.</p>
        </header>
        <ol className="how-steps">
          {/* camera times: videos/pickup.json and videos/return.json creation_time, 19:46:29Z and 19:50:42Z (EST) */}
          <Step n={1} title="Two videos">
            <p className="how-text">
              Film a slow lap around the car when you get it, and again when you hand it back. The example was filmed
              at 2:46&nbsp;PM and 2:50&nbsp;PM on Nov&nbsp;2.
            </p>
            <Pair
              shots={[
                { src: pickupFrame, w: 960, h: 540, caption: 'Pickup 2:46 PM', alt: 'The Ford Focus from the rear right in the pickup video' },
                { src: returnFrame, w: 960, h: 540, caption: 'Return 2:50 PM', alt: 'The same car from the same side in the return video' },
              ]}
              caption="Blue tape stands in for damage, but the scrape low on the left front door is real."
            />
          </Step>

          {/* sheet/sheet.json: originals[].sha256; clocks.sealed_at 23:42:53Z, first_model_call_at 23:43:16Z and
              sealed_minutes_after_camera 236 */}
          <Step n={2} title="Sealed before any model sees them">
            <p className="how-text">
              Both videos are hashed and timestamped as they arrive, before any model sees them. Whoever kept the two
              files can hash them and compare with these. The seal covers the files from upload on, not the hours
              before it.
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
              <div>
                <dt>After filming</dt>
                <dd>3 h 56 min</dd>
              </div>
            </dl>
            <p className="how-link">
              <Link to="/verify">Verify it yourself</Link>
            </p>
          </Step>

          {/* 16 prompts and 22 panels: README, What happens in a run, step 2 (SAM 3.1 runs 16 lanes per walk, voted into
              22 panels); not compared: derived/verdicts.json, left_rear_fender, right_front_fender, right_rear_fender
              and roof are not_comparable; the frame is the record view's pickup walk at proxy frame 660 */}
          <Step n={3} title="Meta SAM 3.1 follows each panel">
            <p className="how-text">
              SAM 3.1 follows 16 prompts through each video, from car door and fender to license plate and face. Code
              turns them into 22 panels, so a door is only compared with the same door in the other video. Here 3
              fenders and the roof could not be compared, and the sheet says so.
            </p>
            <figure className="how-fig">
              <img src={samFrame} width={1342} height={756} loading="lazy" alt="The pickup video at 0:44 with SAM 3.1's outlines filling the right rear door and the right front door, each labelled" />
              <figcaption>Pickup video at 0:44, with SAM 3.1's outlines on the two right doors.</figcaption>
            </figure>
          </Step>

          {/* the three boxes and their words: derived/marks.json, keyframe return_left_front_door_kf1 */}
          <Step n={4} title="Meta Muse Spark boxes the marks">
            <p className="how-text">On each panel, Muse Spark boxes every mark it sees and says what it looks like.</p>
            <figure className="how-fig how-fig-narrow">
              <img src={sparkCrop} width={772} height={752} loading="lazy" alt="The left front door in the return video with three white boxes: one around a blue tape sticker and two around a scratch low on the door" />
              <figcaption>
                Left front door, return video. Muse Spark named its three boxes "small blue tape sticker", "dark scratch
                on lower door" and "small dark scratch".
              </figcaption>
            </figure>
          </Step>

          {/* the verdict right_rear_door_v2's trace (math view, match.py 1994 to 2007): d_pos 0.285 + type_cost 0.4 +
              size_cost 0.230 = d 0.915, against tau_eye_r 0.849; the nearest pickup mark is
              pickup_right_rear_door_kf1_s1, "tiny dark speck on paint" (derived/marks.json) */}
          <Step n={5} title="Plain code compares the same spot">
            <p className="how-text">
              Code lines the two views of a panel up and scores each mark against the nearest mark in the other video:
              the distance on the panel, plus a cost when the type or size differs. Code makes the call, not a model.
            </p>
            <p className="how-big">
              0.915 is over the 0.849 cutoff: the nearest pickup mark is a tiny dark speck, not this strip, so it's new.
            </p>
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

          {/* model and host: derived/looks.json (meta-models/muse-spark-open-1026, engine baseten); asked 3 times, damage
              yes 2 of 3 and the same mark at pickup 0 of 3 (the math view's second look); the reflection answer: README,
              Limits; the two pictures are the look's own input_refs */}
          <Step n={6} title="A second look on open weights">
            <p className="how-text">
              Close calls get a second look from Muse Spark Open, an open-weights model on Baseten: a matched pair, or a
              mark seen in one video only. It can only lower a label to Needs your eye, never raise one. Asked 3 times
              about this strip, it saw damage twice (once it called it a reflection) and never at pickup, so New stands.
            </p>
            <Pair
              shots={[
                { src: lookPickup, w: 1000, h: 979, caption: 'Pickup, the area it searched', alt: 'The crop of the right rear door from the pickup video that the second look was shown, the area it searched outlined and empty' },
                { src: lookReturn, w: 948, h: 962, caption: 'Return, the mark outlined', alt: 'The crop of the right rear door from the return video that the second look was shown, the blue strip outlined' },
              ]}
              square
            />
          </Step>

          {/* wheels and glass: the result page's own line; 17 of 53 on 49 hand-labelled crops: README, Proof (box by
              box) and docs/evidence/hero_focus5e_truth.md */}
          <Step n={7} title="What it won't do">
            <p className="how-text">
              Wheels and glass are not compared, so check them by eye. It also misses faint marks: on 49 crops of these
              two videos that I labelled by hand, Muse Spark boxed 17 of 53 real marks.
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
    </>
  );
}
