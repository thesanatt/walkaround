"""hero_focus1 fixes of 11/2 (docs/evidence/hero_1102/NOTES.md), each behind a thresholds leaf that frozen runs lack:
matching.glass_by_eye (glass panels get one check-by-eye line, their marks no verdict), matching.debris_by_eye
(needs-your-eye lines of dirt on a flagged view fold into one line per panel), matching.colour_on_paint and
matching.conditions_on_panel (two marks prompt sentences; the first also reaches the two-question counter-look) and
dedup.spark_zoom_refine.prefer_kind (the zoom box of a damage type first). No network."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from walkaround import contracts as c
from walkaround import match, paths, spark_client, zoom_refine
from walkaround.contracts import Conditions, MarkKind, PanelId, VerdictLabel

from .test_match import DOOR, ROLES, kf, mark, panel, runs_dir, seed_run, th, write_crop  # noqa: F401  (fixtures)


def _with(th: dict, **leaves: Any) -> dict:
    out = copy.deepcopy(th)
    for key, value in leaves.items():
        group, leaf = key.split("__")
        if isinstance(value, dict) and isinstance(out[group].get(leaf), dict):
            out[group][leaf] = {**out[group][leaf], **value}
        else:
            out[group][leaf] = value
    return out


def _frozen(th: dict) -> dict:
    out = copy.deepcopy(th)
    for leaf in ("glass_by_eye", "debris_by_eye", "colour_on_paint", "conditions_on_panel"):
        out["matching"].pop(leaf, None)
    out["dedup"]["spark_zoom_refine"].pop("prefer_kind", None)
    return out


# Settings -----------------------------------------------------------------------------------------------------------


def test_every_leaf_is_off_when_absent_and_the_glass_default_is_the_windshield_and_rear_window(th: dict) -> None:
    frozen = _frozen(th)
    assert match.glass_by_eye(frozen) == frozenset() and match.debris_by_eye(frozen) == frozenset()
    assert not match.colour_on_paint(frozen) and not match.conditions_on_panel(frozen)
    assert not zoom_refine.prefer_kind(zoom_refine.settings(frozen))
    on = _with(frozen, matching__glass_by_eye={"enabled": True}, matching__debris_by_eye={"enabled": True})
    assert match.glass_by_eye(on) == {PanelId.WINDSHIELD, PanelId.REAR_WINDOW}
    assert match.debris_by_eye(on) == {MarkKind.DIRT}
    assert match.glass_by_eye(_with(frozen, matching__glass_by_eye={"enabled": False})) == frozenset()


def test_config_turns_every_leaf_on_for_new_runs() -> None:
    th = c.load_thresholds()
    assert match.glass_by_eye(th) == {PanelId.WINDSHIELD, PanelId.REAR_WINDOW}
    assert match.debris_by_eye(th) == {MarkKind.DIRT}
    assert match.colour_on_paint(th) and match.conditions_on_panel(th)
    assert zoom_refine.prefer_kind(zoom_refine.settings(th))


# Glass ----------------------------------------------------------------------------------------------------------------


def test_a_glass_panel_gets_one_line_and_no_verdict_and_its_marks_are_kept_on_the_line(runs_dir: Any, th: dict) -> None:  # noqa: F811
    run_id = "01TESTRUNHERO1102GLASS0000"
    seed_run(run_id)
    base = _frozen(th)
    _, before, _ = match.derive_match(run_id, th=base)
    assert any(v.panel_id == DOOR for v in before)
    # the seeded door stands in for a glass panel: the leaf's panel list decides which panels are glass
    glass_th = _with(base, matching__glass_by_eye={"enabled": True, "panels": [DOOR.value]})
    matches, verdicts, _ = match.derive_match(run_id, th=glass_th)
    door = next(pm for pm in matches if pm.panel_id == DOOR)
    assert not [v for v in verdicts if v.panel_id == DOOR]
    assert [b.kind for b in door.by_eye] == ["glass"] and door.by_eye[0].text == match.GLASS_LINE
    assert door.by_eye[0].mark_ids == sorted(["p0", "p1", "p2", "r0", "r1", "r9"])
    assert not door.pairs and not door.unmatched_pickup and not door.unmatched_return
    assert [v.label for v in verdicts if v.panel_id == PanelId.HOOD] == [VerdictLabel.NOT_COMPARABLE]  # other panels as before
    words = match.by_eye_words(door)
    assert words == [f"{match.GLASS_LINE} (6 spots Muse Spark boxed on it, not judged)"]


def test_without_the_glass_leaf_matches_and_verdicts_serialize_as_before(runs_dir: Any, th: dict) -> None:  # noqa: F811
    run_id = "01TESTRUNHERO1102GLASS0001"
    seed_run(run_id)
    match.derive_match(run_id, th=_frozen(th))
    assert b'"by_eye"' not in paths.matches_json(run_id).read_bytes()


# Debris ---------------------------------------------------------------------------------------------------------------


def _debris_world(*, dirty: bool) -> tuple[c.PanelMatch, list[c.Verdict], dict[str, c.Mark], dict[str, c.Keyframe]]:
    cond = Conditions(dirty=dirty)
    kfs = {k.kf_id: k for k in (kf("pickup_kf1", "pickup", conditions=cond), kf("return_kf1", "return", conditions=cond))}
    ms = {
        "p_dirt": mark("p_dirt", "pickup", u=0.5, v=0.5, kind=MarkKind.DIRT, caps=("K4",)),
        "r_dirt": mark("r_dirt", "return", u=-0.5, v=0.5, kind=MarkKind.DIRT, caps=("K4",)),
        "r_unsure": mark("r_unsure", "return", u=0.9, v=-0.5, kind=MarkKind.UNSURE, caps=("K4",)),
        "r_chip": mark("r_chip", "return", u=-0.9, v=-0.5, kind=MarkKind.CHIP, caps=("K12",)),
        "p_old": mark("p_old", "pickup", u=0.0, v=0.0, kind=MarkKind.DIRT),
        "r_old": mark("r_old", "return", u=0.0, v=0.0, kind=MarkKind.DIRT),
    }
    pm = c.PanelMatch(panel_id=DOOR, state=c.MatchState.COMPARABLE, kf_pair=["pickup_kf1", "return_kf1"])

    def v(n: int, label: VerdictLabel, ids: list[str]) -> c.Verdict:
        return c.Verdict(verdict_id=f"{DOOR.value}_v{n}", panel_id=DOOR, label=label, rule_id="L4", mark_ids=ids)

    vs = [
        v(1, VerdictLabel.ALREADY_THERE, ["p_old", "r_old"]),
        v(2, VerdictLabel.NEEDS_YOUR_EYE, ["p_dirt"]),
        v(3, VerdictLabel.NEEDS_YOUR_EYE, ["r_unsure"]),
        v(4, VerdictLabel.NEEDS_YOUR_EYE, ["r_dirt"]),
        v(5, VerdictLabel.NEEDS_YOUR_EYE, ["r_chip"]),
        v(6, VerdictLabel.NEEDS_YOUR_EYE, ["p_dirt", "r_chip"]),
    ]
    return pm, vs, ms, kfs


def test_dirt_lines_on_a_flagged_view_fold_into_one_line_and_nothing_else_moves(th: dict) -> None:
    pm, vs, ms, kfs = _debris_world(dirty=True)
    on = _with(th, matching__debris_by_eye={"enabled": True, "types": ["dirt"]})
    out_pm, kept = match.debris_split(pm, vs, ms, kfs, on)
    assert [b.kind for b in out_pm.by_eye] == ["debris"] and out_pm.by_eye[0].text == match.DEBRIS_LINE
    assert out_pm.by_eye[0].mark_ids == ["p_dirt", "r_dirt"]
    # already there, the unsure speck, the chip and the dirt paired with a chip stay, numbered again in order
    assert [v.mark_ids for v in kept] == [["p_old", "r_old"], ["r_unsure"], ["r_chip"], ["p_dirt", "r_chip"]]
    assert [v.verdict_id for v in kept] == [f"{DOOR.value}_v{n}" for n in range(1, 5)]
    assert match.by_eye_words(out_pm) == [f"{match.DEBRIS_LINE} (2 spots Muse Spark typed dirt on a flagged view, not listed one by one)"]


def test_dirt_on_a_clean_view_or_with_the_leaf_off_keeps_its_own_lines(th: dict) -> None:
    pm, vs, ms, kfs = _debris_world(dirty=False)
    on = _with(th, matching__debris_by_eye={"enabled": True})
    assert match.debris_split(pm, vs, ms, kfs, on) == (pm, vs)
    pm, vs, ms, kfs = _debris_world(dirty=True)
    assert match.debris_split(pm, vs, ms, kfs, _frozen(th)) == (pm, vs)
    not_compared = pm.model_copy(update={"state": c.MatchState.NOT_COMPARABLE})
    assert match.debris_split(not_compared, vs, ms, kfs, on) == (not_compared, vs)


def test_by_eye_lines_round_trip_and_are_left_out_of_the_json_when_empty() -> None:
    pm = c.PanelMatch(panel_id=DOOR, state=c.MatchState.COMPARABLE)
    assert "by_eye" not in c.canonical_json(pm)
    line = c.ByEyeLine(kind="glass", text=match.GLASS_LINE, mark_ids=["a"])
    with_line = pm.model_copy(update={"by_eye": [line]})
    assert c.PanelMatch.model_validate_json(c.canonical_json(with_line)) == with_line
    block = c.SheetPanelBlock(panel_id=DOOR, display_name="left front door")
    assert "by_eye_lines" not in c.canonical_json(block)
    assert "by_eye_lines" in c.canonical_json(block.model_copy(update={"by_eye_lines": ["x"]}))


# Prompts --------------------------------------------------------------------------------------------------------------


def test_the_colour_and_conditions_sentences_are_sent_only_with_their_flags() -> None:
    plain = spark_client.marks_prompt(PanelId.REAR_BUMPER, ["plate_corner_tl"], fixtures=True)
    assert spark_client.COLOUR_SENTENCE not in plain and "conditions describe the whole crop" in plain
    both = spark_client.marks_prompt(PanelId.REAR_BUMPER, ["plate_corner_tl"], fixtures=True, colour=True, panel_conditions=True)
    assert spark_client.COLOUR_SENTENCE in both and "type it scuff" in both
    assert "conditions describe the rear bumper itself, not the rest of the crop" in both and "whole crop" not in both
    seams = spark_client.marks_prompt(PanelId.REAR_BUMPER, [], seams=True, colour=True)
    assert spark_client.SEAMS_SENTENCE_COLOUR in seams and spark_client.SEAMS_SENTENCE not in seams
    assert plain == spark_client.marks_prompt(PanelId.REAR_BUMPER, ["plate_corner_tl"], fixtures=True, colour=False, panel_conditions=False)


def test_the_counter_look_counts_a_colour_stripe_only_with_the_flag() -> None:
    old = spark_client.counter2_prompt(PanelId.RIGHT_REAR_DOOR, "return")
    new = spark_client.counter2_prompt(PanelId.RIGHT_REAR_DOOR, "return", colour=True)
    assert "tape or a sticker" in new and "tape" not in old
    assert "is there a damage mark (a scratch, scuff, chip, dent or crack) inside the outlined box" in old


def test_find_marks_sends_the_prompt_its_flags_ask_for(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    sent: list[str] = []

    def fake_attempts(parts: list[dict[str, Any]], **kw: Any) -> list[c.SparkRequestRecord]:
        sent.append(parts[0]["text"])
        raise RuntimeError("stop after the prompt")

    monkeypatch.setattr(spark_client, "_ask_attempts", fake_attempts)
    with pytest.raises(RuntimeError):
        spark_client.find_marks(np.zeros((40, 60, 3), np.uint8), PanelId.LEFT_FRONT_DOOR, landmarks=[], colour=True, panel_conditions=True)
    assert spark_client.COLOUR_SENTENCE in sent[0] and "left front door itself" in sent[0]


# Zoom pick ------------------------------------------------------------------------------------------------------------


def _zoom_answer(items: list[tuple[str, list[int]]]) -> zoom_refine.ZoomAsk:
    marks = [{"type": t, "point": {"x": (b[0] + b[2]) // 2, "y": (b[1] + b[3]) // 2}, "bbox": {"x_min": b[0], "y_min": b[1], "x_max": b[2], "y_max": b[3]}, "size_class": "S", "confidence": "low", "note": "n"} for t, b in items]
    rec = c.SparkRequestRecord(
        request_id="resp_zoom", purpose=c.SparkPurpose.MARKS, model="m", engine="meta-model-api", schema_name="marks_v1", store=False, valid=True,
        status="completed", parsed={"conditions": {"dark": False, "dirty": False, "glare": False, "wet": False}, "landmarks": [], "marks": marks, "neighbour_car_visible": False},
    )
    return zoom_refine.ZoomAsk(kf_id="k", role="pickup", mark_id="m", panel_id=DOOR, crop_path="crops/k.jpg", crop_w=400, crop_h=300, window=[100, 100, 260, 190], out_w=1280, out_h=720, record=rec)


def _spark(kind: MarkKind) -> c.Mark:
    return c.Mark(
        mark_id="m", role="pickup", panel_id=DOOR, kf_id="k", source=c.MarkSource.SPARK_BOX, type=kind, confidence=c.Confidence.HIGH, size_class=c.SizeClass.S,
        bbox_crop=[170.0, 140.0, 190.0, 150.0], point_crop=[180.0, 145.0],
    )


CROP = c.Crop(path="crops/k.jpg", w=400, h=300, scale=1.0, dx=0.0, dy=0.0)
WIN = [100, 100, 260, 190]


def test_kind_first_takes_the_nearest_damage_box_over_a_nearer_unsure_speck() -> None:
    # grid boxes: an unsure speck at the window centre (nearest Spark's box) and a scratch off to the left
    a = _zoom_answer([("unsure", [480, 480, 520, 520]), ("scratch", [100, 400, 250, 450])])
    nearest = zoom_refine.decide(_spark(MarkKind.SCRATCH), a, WIN, CROP)
    first = zoom_refine.decide(_spark(MarkKind.SCRATCH), a, WIN, CROP, kind_first=True)
    assert nearest.zoom_type == MarkKind.UNSURE and first.zoom_type == MarkKind.SCRATCH
    assert first.moved_bbox_crop[0] < nearest.moved_bbox_crop[0]  # type: ignore[index]


def test_kind_first_falls_back_to_the_nearest_box_and_leaves_non_damage_marks_alone() -> None:
    only_unsure = _zoom_answer([("unsure", [480, 480, 520, 520]), ("dirt", [100, 400, 250, 450])])
    assert zoom_refine.decide(_spark(MarkKind.SCRATCH), only_unsure, WIN, CROP, kind_first=True) == zoom_refine.decide(_spark(MarkKind.SCRATCH), only_unsure, WIN, CROP)
    mixed = _zoom_answer([("unsure", [480, 480, 520, 520]), ("scratch", [100, 400, 250, 450])])
    assert zoom_refine.decide(_spark(MarkKind.DIRT), mixed, WIN, CROP, kind_first=True) == zoom_refine.decide(_spark(MarkKind.DIRT), mixed, WIN, CROP)


# Same walk ------------------------------------------------------------------------------------------------------------


class _SameWalkCache:
    """same_walk_step's view of a run: two keyframes of one walk with their pixels, fits made by register.align."""

    def __init__(self, px: dict[str, Any], kfs: dict[str, c.Keyframe]) -> None:
        from .test_registration_1102 import CFG

        self.cfg, self.kf_by_id, self._px, self._fits = CFG, kfs, px, {}

    def fit(self, a: str, b: str) -> Any:
        from walkaround import register

        if (a, b) not in self._fits:
            (ia, ma), (ib, mb) = self._px[a], self._px[b]
            ka, kb = self.kf_by_id[a], self.kf_by_id[b]
            self._fits[(a, b)] = register.align(a, ia, ma, b, ib, mb, self.cfg, prior=register.prior_map(ka.moments, 1, kb.moments, 1))
        return self._fits[(a, b)]


def _same_walk_world() -> tuple[dict[str, list[c.Mark]], dict[str, c.Keyframe], _SameWalkCache, Any]:
    from .test_registration_1102 import _box, _kf, _mapped, _mark, _pair

    a, ma, b, mb, hm = _pair()  # b: the same panel a little later in the walk, with one more strip in view
    k1 = _kf("pickup", ma)
    k2 = _kf("pickup", mb).model_copy(update={"kf_id": "pickup_left_front_door_kf2", "crop": c.Crop(path="crops/pickup/kf2.jpg", w=640, h=400, scale=1.0, dx=0.0, dy=0.0)})
    on_k1 = _mark("pickup_left_front_door_kf1_m1", k1, _box((250, 120))).model_copy(update={"caps": ["K2"]})
    seen_again = _mark("pickup_left_front_door_kf2_m1", k2, _box(_mapped(hm, (250, 120))))
    only_k2 = _mark("pickup_left_front_door_kf2_m2", k2, _box(_mapped(hm, (300, 290))))
    kfs = {k1.kf_id: k1, k2.kf_id: k2}
    cache = _SameWalkCache({k1.kf_id: (a, ma), k2.kf_id: (b, mb)}, kfs)
    return {"pickup": [on_k1, seen_again, only_k2], "return": []}, kfs, cache, hm


def test_a_copy_registered_onto_a_mark_of_the_compared_keyframe_merges_and_lifts_k2(th: dict) -> None:
    panel_marks, kfs, cache, _ = _same_walk_world()
    out, changed, notes = match.same_walk_step(panel_marks, {"pickup": kfs["pickup_left_front_door_kf1"], "return": None}, cache, th)  # type: ignore[dict-item]
    ids = [m.mark_id for m in out["pickup"]]
    assert "pickup_left_front_door_kf2_m1" not in ids  # the same strip seen again leaves matching
    merged = next(n for n in notes if n.mark_id == "pickup_left_front_door_kf2_m1")
    assert merged.merged_into == "pickup_left_front_door_kf1_m1" and merged.from_kf == "pickup_left_front_door_kf2"
    kept = next(m for m in out["pickup"] if m.mark_id == "pickup_left_front_door_kf1_m1")
    assert "K2" not in kept.caps and kept in changed


def test_a_mark_only_the_other_keyframe_shows_is_carried_and_whitened_in_the_compared_keyframe(th: dict) -> None:
    panel_marks, kfs, cache, _ = _same_walk_world()
    k1 = kfs["pickup_left_front_door_kf1"]
    out, changed, notes = match.same_walk_step(panel_marks, {"pickup": k1, "return": None}, cache, th)  # type: ignore[dict-item]
    carried = next(m for m in out["pickup"] if m.mark_id == "pickup_left_front_door_kf2_m2")
    assert carried.kf_id == k1.kf_id and carried in changed
    cx, cy = carried.point_crop  # type: ignore[misc]
    assert abs(cx - 300) < 4 and abs(cy - 290) < 4  # back where the strip is drawn in kf1's pixels
    from walkaround import geometry

    u, v = geometry.whiten((cx, cy), k1.moments, 1)  # type: ignore[arg-type]
    assert carried.w == pytest.approx([u, v], abs=1e-6)
    note = next(n for n in notes if n.mark_id == carried.mark_id)
    assert note.merged_into is None and note.to_kf == k1.kf_id
    assert "carried from pickup_left_front_door_kf2" in (match.same_walk_words(c.PanelMatch(panel_id=DOOR, state=c.MatchState.COMPARABLE, same_walk=notes)) or "")


def test_a_refused_fit_leaves_the_marks_as_they_were(th: dict) -> None:
    panel_marks, kfs, cache, _ = _same_walk_world()
    cache.fit = lambda a, b: None  # type: ignore[method-assign]
    out, changed, notes = match.same_walk_step(panel_marks, {"pickup": kfs["pickup_left_front_door_kf1"], "return": None}, cache, th)  # type: ignore[dict-item]
    assert out["pickup"] == panel_marks["pickup"] and not changed and not notes


def test_same_walk_is_off_when_absent(th: dict) -> None:
    assert match.same_walk_config(_frozen_all(th)) is None
    assert match.same_walk_config(c.load_thresholds()) is not None


def _frozen_all(th: dict) -> dict:
    out = _frozen(th)
    for leaf in ("same_walk_register", "inside_other_box"):
        out["matching"].pop(leaf, None)
    return out


# K19 ------------------------------------------------------------------------------------------------------------------


def _covered_world() -> tuple[c.PanelMatch, dict[str, c.Mark], dict[str, c.Keyframe]]:
    kfs = {k.kf_id: k for k in (kf("pickup_kf1", "pickup"), kf("return_kf1", "return"))}

    def boxed(mid: str, role: str, box: list[float], kind: MarkKind = MarkKind.SCUFF) -> c.Mark:
        return mark(mid, role, u=0.0, v=0.0, kind=kind).model_copy(update={"bbox_crop": box, "point_crop": [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]})

    ms = {
        "p_tape": boxed("p_tape", "pickup", [190.0, 140.0, 210.0, 150.0]),  # a thin stripe under the return's big box
        "p_far": boxed("p_far", "pickup", [30.0, 30.0, 50.0, 40.0]),
        "r_crush": boxed("r_crush", "return", [100.0, 100.0, 300.0, 220.0], MarkKind.DENT),
    }
    pm = c.PanelMatch(panel_id=DOOR, state=c.MatchState.COMPARABLE, kf_pair=["pickup_kf1", "return_kf1"], unmatched_pickup=["p_tape", "p_far"], unmatched_return=["r_crush"])
    return pm, ms, kfs


def test_k19_caps_a_mark_the_other_walks_box_covers_and_not_the_big_mark_over_it() -> None:
    pm, ms, kfs = _covered_world()
    assert match.covered_by_other_walk(pm, ms, kfs) == {"p_tape": "r_crush"}  # the crush is not covered by the stripe


def test_k19_lowers_only_with_the_leaf(th: dict) -> None:
    pm, ms, kfs = _covered_world()
    cal = match.calibrate(sigma_frame=None, sigma_spark=None, sigma_view=None, pseudo_pair_d=None, th=th)
    on = _with(_frozen_all(th), matching__inside_other_box={"enabled": True})
    by = {tuple(v.mark_ids): v for v in match.verdicts_for(pm, ms, {}, {}, kfs, cal, c.Trace(), on)}
    assert by[("p_tape",)].label == VerdictLabel.NEEDS_YOUR_EYE and "K19" in by[("p_tape",)].trace.rule_ids
    assert by[("r_crush",)].label == VerdictLabel.NEW_OR_UNSEEN and by[("p_far",)].label == VerdictLabel.NOT_SEEN_AT_RETURN
    off = {tuple(v.mark_ids): v for v in match.verdicts_for(pm, ms, {}, {}, kfs, cal, c.Trace(), _frozen_all(th))}
    assert off[("p_tape",)].label == VerdictLabel.NOT_SEEN_AT_RETURN
