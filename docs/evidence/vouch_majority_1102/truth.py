"""The six hero marks (docs/HERO_FOCUS1_TRUTH.md) by the mark id each run gave them, read off the stored rows (a
row holding the id, whatever its partner). Other runs have no truth here."""

from __future__ import annotations

from typing import Any

RIGHT = {
    "rear bumper strip A": "already_there",
    "rear bumper strip B": "already_there",
    "front left door strip, kept": "already_there",
    "front left door strip, removed": "not_seen_at_return",
    "right rear door strip, new": "new_or_unseen",
    "front left door scrape": "already_there",
}
IDS = {
    "hero_focus2": {
        "rear bumper strip A": "pickup_rear_bumper_kf2_m1",
        "rear bumper strip B": "pickup_rear_bumper_kf1_m2",
        "front left door strip, kept": "pickup_left_front_door_kf1_m2",
        "front left door strip, removed": "pickup_left_front_door_kf1_m1",
        "right rear door strip, new": "return_right_rear_door_kf1_m1",
        "front left door scrape": "pickup_left_front_door_kf1_m3",
    },
    "hero_focus3": {
        "rear bumper strip A": "pickup_rear_bumper_kf1_m1",
        "rear bumper strip B": "pickup_rear_bumper_kf1_m2",
        "front left door strip, kept": "pickup_left_front_door_kf1_m2",
        "front left door strip, removed": "pickup_left_front_door_kf1_m1",
        "right rear door strip, new": "return_right_rear_door_kf1_m1",
        "front left door scrape": "pickup_left_front_door_kf1_m3",
    },
}


def truth_rows(run_id: str, verdicts: list[Any]) -> list[dict[str, Any]]:
    out = []
    for name, mark_id in IDS.get(run_id, {}).items():
        row = next((v for v in verdicts if mark_id in v.mark_ids), None)
        label = None if row is None else row.label.value
        out.append({"mark": name, "right": RIGHT[name], "label": label, "ok": label == RIGHT[name], "mark_ids": None if row is None else row.mark_ids, "rules": None if row is None else row.trace.rule_ids})
    return out
