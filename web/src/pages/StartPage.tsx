import { Bar } from '../components/Bar.tsx';
import { Link } from '../lib/router.tsx';
import pickupFrame from '../example/pickup.jpg';
import returnFrame from '../example/return.jpg';

// The example the start page opens on: one frame from each of hero_focus5e's blurred videos at the rear right
// (pickup 44.07 s, return 26.87 s, the closest pair of angles on that side), the new strip on the right rear door boxed
// on the return frame and its spot outlined on the pickup frame. Box centres are fractions of the 960 x 540 frame.
const EXAMPLE_RUN = '/run/hero_focus5e?replay=instant';
// The second sample in samples/index.json: NHTSA test 15759 (public domain), filmed before and after a side impact.
const CRASH_RUN = '/run/eval_nhtsa15759_r6?replay=instant';
const SIDES = [
  {
    src: pickupFrame,
    caption: 'Pickup 2:46 PM',
    tone: 'spot',
    x: 0.6885,
    y: 0.3148,
    alt: 'The Ford Focus from the rear right at pickup, with the spot on the right rear door outlined',
  },
  {
    src: returnFrame,
    caption: 'Return 2:50 PM',
    tone: 'new',
    x: 0.7224,
    y: 0.3028,
    alt: 'The same car at return, with the new blue tape strip on the right rear door boxed in red',
  },
] as const;

// The start screen (/): the question, the example by default, and one button to check your own car.
export function StartPage() {
  return (
    <div className="start">
      <Bar />
      <h1 className="start-q">Which scratches are new since you got the car?</h1>
      <p className="start-sub">Film a slow walk around the car when you get it, and again when you hand it back.</p>
      <section className="example" aria-label="an example">
        <div className="example-pair">
          {SIDES.map((s) => (
            <figure key={s.caption} className="example-frame">
              <div className="example-view">
                <img src={s.src} alt={s.alt} width={960} height={540} />
                <span className={`frame-box ${s.tone}`} style={{ left: `${s.x * 100}%`, top: `${s.y * 100}%` }} />
              </div>
              <figcaption>{s.caption}</figcaption>
            </figure>
          ))}
        </div>
        <p className="example-line">
          Example: a friend's Ford Focus. 1 new mark since pickup, on the right rear door.{' '}
          <Link to={EXAMPLE_RUN}>See the full result</Link>
        </p>
        <p className="example-line example-more">
          Another example: a crash-test car, before and after a side impact. <Link to={CRASH_RUN}>See it</Link>
        </p>
      </section>
      <p className="start-own">
        <Link to="/upload" className="button">
          Check your own car
        </Link>
      </p>
    </div>
  );
}
