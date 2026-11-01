import { describe, expect, it } from 'vitest';

import type { FixtureDrop, PanelMatch } from '../src/contracts/types.ts';
import { fixtureWords, keyframePairLine } from '../src/lib/lines.ts';

// Factory fixtures are not damage (lead decision 4 of 11/1): match.fixture_words and how_checked word for word.
const sensor: FixtureDrop = { mark_id: 'return_rear_bumper_kf1_m1', fixture: 'sensor', source: 'spark', overlap: 0.8749 };
const handle: FixtureDrop = { mark_id: 'r_on', fixture: 'handle', source: 'sam_handle', overlap: 1 };

describe('the fixture line (lead decision 4 of 11/1)', () => {
  it('names each mark left out, what it sits on and how much of its box', () => {
    expect(fixtureWords([sensor])).toBe(
      "ignored: return_rear_bumper_kf1_m1 sits on a parking sensor (Muse Spark's landmark, 87% of its box), so it was not matched",
    );
    expect(fixtureWords([sensor, handle])).toContain("; ignored: r_on sits on the door handle (SAM's door handle, 100% of its box)");
  });

  it('follows the keyframe pair line, and stands alone on a panel with no tries', () => {
    const tries = [{ kf_pair: ['pickup_kf1', 'return_kf1'], gates: [], mask_iou: 0.9, frames_agree: true }];
    expect(keyframePairLine({ kf_pairs_tried: tries, fixtures: [handle] } as unknown as PanelMatch)).toBe(
      "compared on keyframes pickup_kf1 with return_kf1, the first pair tried; ignored: r_on sits on the door handle (SAM's door handle, 100% of its box), so it was not matched",
    );
    expect(keyframePairLine({ kf_pairs_tried: [], fixtures: [sensor] } as unknown as PanelMatch)).toBe(fixtureWords([sensor]));
    expect(keyframePairLine({ kf_pairs_tried: [] } as unknown as PanelMatch)).toBeNull();
    expect(keyframePairLine({ kf_pairs_tried: [], fixtures: [] } as unknown as PanelMatch)).toBeNull();
  });
});
