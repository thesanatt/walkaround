"""The lead's match decisions of 11/1, offline, every branch:

Rule 1 (thresholds matching.tight_pair_second_look): a pair inside tau_pre_r with K4 or K12 goes to the second look;
once it answers (same, different, unsure) K4 and K12 are notes (trace.note_ids, Verdict.type_note) and only the look
(K7) or another cap can lower the pair; with no usable answer (asked and nothing came back, or never asked) they cap as
before. Marks seen in one walk only and pairs in the band keep them as caps.

Rule 2 (thresholds matching.next_keyframe_pair): a panel whose chosen keyframe pair fails a view gate (G3, G4, G5, G8,
G9) is compared on the first pair in keyframe_pair_order that passes every gate; each pair tried is recorded with its
gates (PanelMatch.kf_pairs_tried) and printed on the sheet (how_checked). G1, G6, G7 and G10 stay panel-level.

Both leaves absent (a run frozen before 11/1): the old behaviour and the old file bytes. No model call anywhere."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from walkaround import geometry, match, paths, rules, second_look, sheet
from walkaround.contracts import (
    CAPPED_MARK_KINDS,
    Calibration,
    Conditions,
    Confidence,
    CoverageState,
    FrameUsed,
    Keyframe,
    KeyframePairTry,
    LaneRecord,
    LaneStatus,
    LookVerdict,
    Mark,
    MarkKind,
    MarksStatus,
    MatchCandidate,
    MatchState,
    Noun,
    PanelId,
    PanelMatch,
    RunManifest,
    Trace,
    Verdict,
    VerdictLabel,
    config_hash,
    read_json_list,
    write_json,
)

from .test_match import CROP_H, CROP_W, DOOR, RECT_MASK, cal, kf, mark, panel, runs_dir, th, wm  # noqa: F401  (cal, runs_dir, th are fixtures)

RULES_1101 = ("tight_pair_second_look", "next_keyframe_pair")


def frozen(th: dict, *leaves: str) -> dict:
    """The thresholds of a run frozen before 11/1: the leaves absent."""
    leaves = leaves or RULES_1101
    return {**th, "matching": {k: v for k, v in th["matching"].items() if k not in leaves}}


def spark_mark(mark_id: str, role: str, u: float, *, kind: MarkKind = MarkKind.DIRT, low: bool = True, extra: tuple[str, ...] = (), kf_id: str | None = None, v: float = 0.0) -> Mark:
    """A mark with the caps marks_from_answer sets: K4 for dirt, reflection or unsure, K12 for low confidence."""
    caps = tuple(c for c, on in (("K4", kind in CAPPED_MARK_KINDS), ("K12", low)) if on) + extra
    return mark(mark_id, role, u=u, v=v, kind=kind, confidence=Confidence.LOW if low else Confidence.HIGH, caps=caps, kf_id=kf_id)


KFS = {k: kf(k, r) for k, r in (("pickup_kf1", "pickup"), ("return_kf1", "return"))}


def judge(th: dict, cal: Calibration, a: Mark, b: Mark, looks: dict | None = None) -> list[Verdict]:
    """verdicts_for on one pair of a comparable door, with the looks given (a key with None: asked, no usable answer)."""
    pair = match.cost(a, b, wm(a), wm(b), cal, th)
    pm = PanelMatch(panel_id=DOOR, state=MatchState.COMPARABLE, kf_pair=["pickup_kf1", "return_kf1"], pairs=[pair])
    return match.verdicts_for(pm, {a.mark_id: a, b.mark_id: b}, {}, {} if looks is None else looks, KFS, cal, Trace(), th)


CHIPS = "Muse Spark typed both marks dirt (K4), with low confidence on both marks (K12)"


# Rule 1: a tight pair with K4 and K12 ------------------------------------------------------------------------------


def test_a_same_answer_keeps_a_tight_dirt_pair_already_there_with_k4_and_k12_as_notes(th: dict, cal: Calibration) -> None:
    a, b = spark_mark("p0", "pickup", 0.10), spark_mark("r0", "return", 0.12)
    (v,) = judge(th, cal, a, b, {("p0", "r0"): LookVerdict.SAME})
    assert v.label == VerdictLabel.ALREADY_THERE and v.rule_id == "L3" and v.trace.rule_ids == ["L3"] and v.reasons == []
    assert v.trace.note_ids == ["K4", "K12"]
    assert v.type_note == f"{CHIPS}: a note here, since the mark sits in the same place in both walks and the second look compared the two crops and said same"
    data = json.loads(v.to_json())
    assert data["type_note"] == v.type_note and data["trace"]["note_ids"] == ["K4", "K12"]


@pytest.mark.parametrize("look, said", [(LookVerdict.DIFFERENT, "said different"), (LookVerdict.UNSURE, "was unsure")])
def test_a_different_or_unsure_answer_lowers_the_pair_through_k7_and_k4_k12_stay_notes(look: LookVerdict, said: str, th: dict, cal: Calibration) -> None:
    (v,) = judge(th, cal, spark_mark("p0", "pickup", 0.10), spark_mark("r0", "return", 0.12), {("p0", "r0"): look})
    assert v.label == VerdictLabel.NEEDS_YOUR_EYE and v.trace.rule_ids == ["L4", "L3", "K7"] and v.reasons == rules.describe(["K7"])
    assert v.trace.note_ids == ["K4", "K12"]
    assert v.type_note == f"{CHIPS}: a note here; the second look compared the two crops and {said} (K7)"


def test_with_no_usable_answer_k4_and_k12_cap_as_before(th: dict, cal: Calibration) -> None:
    a, b = spark_mark("p0", "pickup", 0.10), spark_mark("r0", "return", 0.12)
    (asked,) = judge(th, cal, a, b, {("p0", "r0"): None})  # put to a model, every engine failed
    (never,) = judge(th, cal, a, b, {})  # not asked: spend ceiling, deadline, look budget or the look off
    for v in (asked, never):
        assert v.label == VerdictLabel.NEEDS_YOUR_EYE and v.trace.rule_ids == ["L4", "L3", "K12", "K4"] and v.trace.note_ids == []
        assert v.reasons == rules.describe(["K12", "K4"])
        assert "note_ids" not in json.loads(v.to_json())["trace"]
    assert asked.type_note == f"{CHIPS}: the second look gave no usable answer, so K4 and K12 cap the pair as before"
    assert never.type_note == f"{CHIPS}: the second look was not asked, so K4 and K12 cap the pair as before"


def test_another_cap_still_lowers_a_tight_pair_and_the_note_says_why_no_look_was_asked(th: dict, cal: Calibration) -> None:
    a, b = spark_mark("p0", "pickup", 0.10, extra=("K3",)), spark_mark("r0", "return", 0.12)
    (v,) = judge(th, cal, a, b)
    assert v.trace.rule_ids == ["L4", "L3", "K12", "K3", "K4"] and v.trace.note_ids == []
    assert v.type_note == f"{CHIPS}: another cap already lowers this pair, so no second look was asked and K4 and K12 cap it too"
    (v,) = judge(th, cal, a, b, {("p0", "r0"): LookVerdict.SAME})  # an answer makes K4 and K12 notes; K3 still lowers it
    assert v.label == VerdictLabel.NEEDS_YOUR_EYE and v.trace.rule_ids == ["L4", "L3", "K3"] and v.trace.note_ids == ["K4", "K12"]


def test_the_note_names_what_muse_spark_said(th: dict, cal: Calibration) -> None:
    def note(a: Mark, b: Mark) -> str:
        (v,) = judge(th, cal, a, b)
        assert v.type_note is not None
        return v.type_note.split(": ")[0]

    assert note(spark_mark("p0", "pickup", 0.1, low=False), spark_mark("r0", "return", 0.12, low=False)) == "Muse Spark typed both marks dirt (K4)"
    assert note(spark_mark("p0", "pickup", 0.1, kind=MarkKind.CHIP), spark_mark("r0", "return", 0.12, kind=MarkKind.CHIP)) == "Muse Spark's confidence was low on both marks (K12)"
    assert note(spark_mark("p0", "pickup", 0.1, low=False), spark_mark("r0", "return", 0.12)) == "Muse Spark typed both marks dirt (K4), with low confidence on the return mark (K12)"
    (v,) = judge(th, cal, spark_mark("p0", "pickup", 0.1, low=False), spark_mark("r0", "return", 0.12, low=False))
    assert v.type_note.endswith("the second look was not asked, so K4 caps the pair as before")  # one id: caps
    # two types inside tau_pre_r (a hand-made pair, as on a lower bumper's wider band): both named
    a, b = spark_mark("p0", "pickup", 0.1), spark_mark("r0", "return", 0.12, kind=MarkKind.REFLECTION, low=False)
    pair = MatchCandidate(pickup_mark="p0", return_mark="r0", d=0.1, d_pos=0.02, type_cost=0.0, size_cost=0.0, r_mean=0.1, tau_pre_r=0.4, tau_eye_r=0.8)
    flags = match.flags_for(pair, a, b, {}, None, {}, return_mark=False)
    _, notes, text = match.tight_pair_flags(pair, a, b, flags, LookVerdict.SAME, asked=True)
    assert notes == ["K4", "K12"] and text.startswith("Muse Spark typed the pickup mark dirt and the return mark reflection (K4), with low confidence on the pickup mark (K12): a note here")
    clean = mark("p1", "pickup", u=0.1, v=0.0)
    assert match.tight_pair_flags(pair, clean, clean, match.flags_for(pair, clean, clean, {}, None, {}, return_mark=False), None, asked=False)[1:] == ([], None)


def test_a_pair_in_the_band_and_marks_in_one_walk_keep_k4_and_k12_as_caps(th: dict, cal: Calibration) -> None:
    a, b = spark_mark("p0", "pickup", 0.10), spark_mark("r0", "return", 0.40)  # d 0.3: the band
    (v,) = judge(th, cal, a, b, {("p0", "r0"): LookVerdict.SAME})
    assert v.trace.rule_ids == ["L4", "K1", "K12", "K4"] and v.trace.note_ids == [] and v.type_note is None
    verdicts = judge(th, cal, a, spark_mark("r0", "return", 1.5))  # beyond tau_eye_r: L5 on both marks
    assert [v.trace.rule_ids for v in verdicts] == [["L4", "L5", "K12", "K4"]] * 2 and all(v.type_note is None and not v.trace.note_ids for v in verdicts)
    pm = PanelMatch(panel_id=DOOR, state=MatchState.COMPARABLE, kf_pair=["pickup_kf1", "return_kf1"], unmatched_pickup=["p0"], unmatched_return=["r0"])
    verdicts = match.verdicts_for(pm, {"p0": a, "r0": b}, {}, {("p0", "r0"): LookVerdict.SAME}, KFS, cal, Trace(), th)
    assert [v.trace.rule_ids for v in verdicts] == [["L4", "L1", "K12", "K4"], ["L4", "L2", "K12", "K4"]] and all(v.type_note is None for v in verdicts)


def test_rule_1_off_keeps_the_old_caps_and_the_old_bytes(th: dict, cal: Calibration) -> None:
    old = frozen(th)
    a, b = spark_mark("p0", "pickup", 0.10), spark_mark("r0", "return", 0.12)
    for looks in ({}, {("p0", "r0"): LookVerdict.SAME}, {("p0", "r0"): None}):
        (v,) = judge(old, cal, a, b, looks)
        assert v.label == VerdictLabel.NEEDS_YOUR_EYE and v.trace.rule_ids == ["L4", "L3", "K12", "K4"]
        data = json.loads(v.to_json())
        assert v.type_note is None and "type_note" not in data and "note_ids" not in data["trace"]
    assert not match.tight_pair_look(old) and match.tight_pair_look(th) and not match.tight_pair_look(None)


# Rule 1 through derive_match: who is asked, and the rebuild -----------------------------------------------------------


def write_kf(run_id: str, k: Keyframe, mask: np.ndarray | None = RECT_MASK) -> Keyframe:
    """The keyframe's blurred crop and its mask PNG (moments from that mask); mask None leaves it without one."""
    crop_file = paths.run_dir(run_id) / k.crop.path
    crop_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(crop_file), np.full((CROP_H, CROP_W, 3), 120, dtype=np.uint8))
    if mask is None:
        return k.model_copy(update={"mask_png": None})
    mask_file = paths.crop_mask_png_path(run_id, k.role, k.kf_id)
    cv2.imwrite(str(mask_file), mask.astype(np.uint8) * 255)
    return k.model_copy(update={"mask_png": paths.rel(run_id, mask_file), "moments": geometry.moments_from_mask(mask)})


def tight_run(run_id: str, th: dict) -> None:
    """A door filmed in both walks: the trunk-chip case (dirt, low confidence, the same spot) and a clean scratch pair."""
    paths.ensure_run_layout(run_id)
    write_json(paths.run_json(run_id), RunManifest(run_id=run_id, created_at="2026-11-01T05:00:00Z", thresholds=th, config_hash=config_hash(th)))
    kfs = [write_kf(run_id, kf(f"{role}_kf1", role)) for role in ("pickup", "return")]
    write_json(paths.panels_json(run_id), [panel()])
    write_json(paths.keyframes_json(run_id), kfs)
    marks = [spark_mark("p0", "pickup", 0.10), mark("p1", "pickup", u=1.0, v=0.5), spark_mark("r0", "return", 0.12), mark("r1", "return", u=1.0, v=0.5)]
    write_json(paths.marks_json(run_id), marks)


def by_marks(verdicts: list[Verdict]) -> dict[tuple[str, ...], Verdict]:
    return {tuple(v.mark_ids): v for v in verdicts}


def test_derive_match_sends_the_tight_dirt_pair_to_the_second_look(runs_dir: Path, th: dict) -> None:
    run_id = "01TESTRUNTIGHT000000000000"
    tight_run(run_id, th)
    asked: list[tuple[str, str]] = []

    def look(pair: MatchCandidate) -> LookVerdict:
        asked.append((pair.pickup_mark, pair.return_mark))
        return LookVerdict.SAME

    _, verdicts, _ = match.derive_match(run_id, th=th, second_look=look)
    assert asked == [("p0", "r0"), ("p1", "r1")]
    v = by_marks(verdicts)[("p0", "r0")]
    assert v.label == VerdictLabel.ALREADY_THERE and v.trace.rule_ids == ["L3"] and v.trace.note_ids == ["K4", "K12"]
    assert by_marks(verdicts)[("p1", "r1")].trace.rule_ids == ["L3"] and by_marks(verdicts)[("p1", "r1")].type_note is None
    asked.clear()
    _, verdicts, _ = match.derive_match(run_id, th=frozen(th), second_look=look)  # before 11/1: K4 and K12 kept it from the look
    assert asked == [("p1", "r1")] and by_marks(verdicts)[("p0", "r0")].trace.rule_ids == ["L4", "L3", "K12", "K4"]


def test_derive_match_with_no_usable_answer_caps_and_says_why(runs_dir: Path, th: dict) -> None:
    run_id = "01TESTRUNTIGHT000000000001"
    tight_run(run_id, th)
    cases = (
        (lambda pair: None, th, "the second look gave no usable answer"),  # every engine failed
        (None, th, "the second look was not asked"),  # past the ceiling or the deadline: the stage passes no look
        (lambda pair: LookVerdict.SAME, {**th, "spark": {**th["spark"], "second_looks_per_pair_max": 0}}, "the second look was not asked"),  # no budget left
    )
    for look, values, why in cases:
        _, verdicts, _ = match.derive_match(run_id, th=values, second_look=look)
        v = by_marks(verdicts)[("p0", "r0")]
        assert v.label == VerdictLabel.NEEDS_YOUR_EYE and v.trace.rule_ids == ["L4", "L3", "K12", "K4"] and v.trace.note_ids == []
        assert v.type_note == f"{CHIPS}: {why}, so K4 and K12 cap the pair as before"


def test_the_replay_from_looks_json_rebuilds_the_tight_pair_byte_identical(runs_dir: Path, th: dict) -> None:
    run_id = "01TESTRUNTIGHT000000000002"
    tight_run(run_id, th)
    for answers in ({("p0", "r0"): LookVerdict.SAME, ("p1", "r1"): LookVerdict.DIFFERENT}, {("p0", "r0"): None, ("p1", "r1"): LookVerdict.SAME}):
        match.derive_match(run_id, th=th, second_look=lambda pair, answers=answers: answers[(pair.pickup_mark, pair.return_mark)])
        live = (paths.matches_json(run_id).read_bytes(), paths.verdicts_json(run_id).read_bytes())
        rows = [second_look.LookRecord(pickup_mark=p, return_mark=r, panel_id=DOOR, verdict=verdict, valid=verdict is not None, reason="same mark" if verdict else "not run: no engine answered") for (p, r), verdict in sorted(answers.items())]
        write_json(paths.looks_json(run_id), rows)  # what SecondLook writes on the live run
        match.derive_match(run_id, th=th, second_look=second_look.replay_second_look(run_id, th=th))
        assert (paths.matches_json(run_id).read_bytes(), paths.verdicts_json(run_id).read_bytes()) == live
    paths.looks_json(run_id).unlink()  # a run that asked no look at all: the replay is None and nothing is asked
    assert second_look.replay_second_look(run_id, th=th) is None
    _, verdicts, _ = match.derive_match(run_id, th=th, second_look=None)
    assert by_marks(verdicts)[("p0", "r0")].type_note.endswith("the second look was not asked, so K4 and K12 cap the pair as before")


# Rule 2: the next keyframe pair on a view gate ----------------------------------------------------------------------

NOTCH = RECT_MASK.copy()
NOTCH[20:110, 20:110] = False  # whitened IoU 0.785 against the rectangle: passes G4 (0.7), fails G8 (0.8)
TRIANGLE = np.zeros_like(RECT_MASK, dtype=np.uint8)
cv2.fillPoly(TRIANGLE, [np.array([[20, 280], [380, 280], [200, 20]])], 255)
TRIANGLE = TRIANGLE > 0  # whitened IoU 0.649: fails G4


def pair_run(run_id: str, th: dict, keyframes: list[tuple[Keyframe, np.ndarray | None]], *, p: Any = None, marks: list[Mark] | None = None) -> None:
    paths.ensure_run_layout(run_id)
    write_json(paths.run_json(run_id), RunManifest(run_id=run_id, created_at="2026-11-01T05:00:00Z", thresholds=th, config_hash=config_hash(th)))
    kfs = [write_kf(run_id, k, m) for k, m in keyframes]
    pid = kfs[0].panel_id
    write_json(paths.panels_json(run_id), [p or panel(pid)])
    write_json(paths.keyframes_json(run_id), kfs)
    if marks is None:
        pick = next(k for k in kfs if k.role == "pickup").kf_id
        ret = next(k for k in kfs if k.role == "return").kf_id
        marks = [mark("p0", "pickup", u=0.1, v=0.0, kf_id=pick, panel_id=pid), mark("r0", "return", u=0.1, v=0.0, kf_id=ret, panel_id=pid)]
    write_json(paths.marks_json(run_id), marks)


def the_panel(run_id: str, th: dict) -> PanelMatch:
    (pm,) = match.derive_match(run_id, th=th)[0]
    return pm


def tries(pm: PanelMatch) -> list[tuple[str, str, list[str]]]:
    return [(t.kf_pair[0], t.kf_pair[1], t.gates) for t in pm.kf_pairs_tried]


VIEW_CASES = {
    # gate: (panel, the first return keyframe and its mask) -- the pickup keyframe and return_kf2 are clean
    "G3": (DOOR, lambda: (kf("return_kf1", "return", sharpness=0.3), RECT_MASK)),
    "G4": (DOOR, lambda: (kf("return_kf1", "return"), TRIANGLE)),
    "G5": (DOOR, lambda: (kf("return_kf1", "return", handle=[300.0, 150.0]), RECT_MASK)),
    "G8": (PanelId.HOOD, lambda: (kf("return_kf1", "return", PanelId.HOOD), NOTCH)),
    "G9": (DOOR, lambda: (kf("return_kf1", "return", window_subtracted=True), RECT_MASK)),
}


@pytest.mark.parametrize("gate", sorted(VIEW_CASES, key=lambda g: int(g[1:])))
def test_each_view_gate_sends_the_panel_to_its_next_keyframe_pair(gate: str, runs_dir: Path, th: dict) -> None:
    pid, first = VIEW_CASES[gate]
    r1, mask = first()
    pickup = kf("pickup_kf1", "pickup", pid, handle=[200.0, 150.0])
    r2 = kf("return_kf2", "return", pid, aspect=1.5, handle=[200.0, 150.0])  # a worse aspect match: tried second
    run_id = f"01TESTRUNNEXT{gate}".ljust(26, "0")
    pair_run(run_id, th, [(pickup, RECT_MASK), (r1.model_copy(update={"panel_id": pid}), mask), (r2, RECT_MASK)], p=panel(pid))
    pm = the_panel(run_id, th)
    assert tries(pm) == [("pickup_kf1", "return_kf1", [gate]), ("pickup_kf1", "return_kf2", [])]
    assert pm.state == MatchState.COMPARABLE and pm.reasons == [] and pm.kf_pair == ["pickup_kf1", "return_kf2"] and pm.mask_iou == pytest.approx(1.0)
    assert [p.pickup_mark for p in pm.pairs] == ["p0"]
    if gate in ("G4", "G8"):
        assert pm.kf_pairs_tried[0].mask_iou == pytest.approx({"G4": 0.649, "G8": 0.785}[gate], abs=0.005) and pm.kf_pairs_tried[0].frames_agree
    old = the_panel(run_id, frozen(th))  # before 11/1: one pair per panel, and no tries in the file
    assert old.state == MatchState.NOT_COMPARABLE and old.reasons == [gate] and old.kf_pair == ["pickup_kf1", "return_kf1"] and old.kf_pairs_tried == []
    assert "kf_pairs_tried" not in json.loads(paths.matches_json(run_id).read_text())[0]


def lane_failed(run_id: str, role: str) -> None:
    write_json(paths.lane_summary_path(run_id, role, Noun.CAR_DOOR), LaneRecord(lane_id="car_door", role=role, noun=Noun.CAR_DOOR, prompt_text="car door", media_size=[1920, 1080], status=LaneStatus.FAILED))


@pytest.mark.parametrize("gate", ["G1", "G6", "G7", "G10"])
def test_a_panel_gate_on_the_chosen_pair_tries_no_other_pair(gate: str, runs_dir: Path, th: dict) -> None:
    run_id = f"01TESTRUNPANEL{gate}".ljust(26, "0")
    status = MarksStatus.TRUNCATED if gate == "G10" else MarksStatus.COMPLETE
    p = panel(states=(CoverageState.FILMED, CoverageState.GLIMPSED)) if gate == "G1" else panel(side_uncertain=gate == "G6")
    keyframes = [(kf("pickup_kf1", "pickup"), RECT_MASK), (kf("return_kf1", "return", sharpness=0.3, status=status), RECT_MASK), (kf("return_kf2", "return", aspect=1.5, status=status), RECT_MASK)]
    pair_run(run_id, th, keyframes, p=p)
    if gate == "G7":
        lane_failed(run_id, "return")
    pm = the_panel(run_id, th)
    assert pm.state == MatchState.NOT_COMPARABLE and gate in pm.reasons and "G3" in pm.reasons
    assert len(pm.kf_pairs_tried) == 1 and pm.kf_pair == ["pickup_kf1", "return_kf1"] and pm.kf_pairs_tried[0].gates == pm.reasons
    assert match.how_checked(pm).endswith(f"{gate} is a panel gate, so no other pair was tried")


def test_when_no_pair_passes_the_panel_keeps_the_chosen_pair_and_lists_every_try(runs_dir: Path, th: dict) -> None:
    run_id = "01TESTRUNNEXTNONE000000000"
    keyframes = [(kf("pickup_kf1", "pickup"), RECT_MASK), (kf("return_kf1", "return", sharpness=0.3), RECT_MASK), (kf("return_kf2", "return", aspect=1.5), TRIANGLE)]
    pair_run(run_id, th, keyframes)
    pm = the_panel(run_id, th)
    assert tries(pm) == [("pickup_kf1", "return_kf1", ["G3"]), ("pickup_kf1", "return_kf2", ["G4"])]
    assert pm.state == MatchState.NOT_COMPARABLE and pm.reasons == ["G3"] and pm.kf_pair == ["pickup_kf1", "return_kf1"] and pm.mask_iou == pytest.approx(1.0)
    assert match.how_checked(pm) == (
        "not compared: none of the 2 keyframe pairs passed every gate; tried: pickup_kf1 with return_kf1 (G3 blurry keyframe); "
        "pickup_kf1 with return_kf2 (G4 mask IoU 0.65)"
    )
    (v,) = read_json_list(paths.verdicts_json(run_id), Verdict)
    assert v.label == VerdictLabel.NOT_COMPARABLE and v.trace.rule_ids == ["L6", "G3"]


def test_a_candidate_failing_a_panel_level_gate_of_its_own_is_skipped(runs_dir: Path, th: dict) -> None:
    """return_kf2 has no front_sign (G6, and no IoU for G4): it fails, so the next pair is tried; G6 on a candidate
    does not stop the search, only on the chosen pair."""
    run_id = "01TESTRUNNEXTSKIP000000000"
    keyframes = [
        (kf("pickup_kf1", "pickup"), RECT_MASK),
        (kf("return_kf1", "return", sharpness=0.3), RECT_MASK),
        (kf("return_kf2", "return", aspect=1.5, front_sign=None), RECT_MASK),
        (kf("return_kf3", "return", aspect=1.6, sharpness=0.9), RECT_MASK),
    ]
    pair_run(run_id, th, keyframes)
    pm = the_panel(run_id, th)
    assert tries(pm) == [("pickup_kf1", "return_kf1", ["G3"]), ("pickup_kf1", "return_kf2", ["G4", "G6"]), ("pickup_kf1", "return_kf3", [])]
    assert pm.state == MatchState.COMPARABLE and pm.kf_pair == ["pickup_kf1", "return_kf3"]
    assert match.how_checked(pm) == (
        "compared on keyframes pickup_kf1 with return_kf3; tried first: pickup_kf1 with return_kf1 (G3 blurry keyframe); "
        "pickup_kf1 with return_kf2 (G4 mask IoU: not measured, G6 side uncertain)"
    )


def test_a_keyframe_turned_too_far_fails_every_pair_and_the_try_says_the_axes_disagree(runs_dir: Path, th: dict) -> None:
    """return_kf2 holds a compared mark and took axis 1 40 degrees away: every pair compares that mark, so every pair
    fails G4 with no IoU, and the sheet says why instead of printing a number."""
    run_id = "01TESTRUNNEXTAXES000000000"
    ys, xs = np.nonzero(RECT_MASK)
    a = math.radians(40.0)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    turned = np.zeros_like(RECT_MASK, dtype=np.uint8)
    pts = ((np.stack([xs, ys], axis=1) - [200.0, 150.0]) @ rot.T * 0.7 + [200.0, 150.0]).astype(int)
    inside = (pts[:, 0] >= 0) & (pts[:, 0] < CROP_W) & (pts[:, 1] >= 0) & (pts[:, 1] < CROP_H)
    turned[pts[inside, 1], pts[inside, 0]] = 255
    turned = cv2.morphologyEx(turned, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    keyframes = [(kf("pickup_kf1", "pickup"), RECT_MASK), (kf("return_kf1", "return"), RECT_MASK), (kf("return_kf2", "return", aspect=2.0), turned)]
    marks = [mark("p0", "pickup", u=0.1, v=0.0, kf_id="pickup_kf1"), mark("r0", "return", u=0.1, v=0.0, kf_id="return_kf1"), mark("r7", "return", u=0.5, v=0.5, kf_id="return_kf2")]
    pair_run(run_id, th, keyframes, marks=marks)
    pm = the_panel(run_id, th)
    assert tries(pm) == [("pickup_kf1", "return_kf1", ["G4"]), ("pickup_kf1", "return_kf2", ["G4"])]
    assert all(t.mask_iou is None and not t.frames_agree for t in pm.kf_pairs_tried) and pm.mask_iou is None
    assert "pickup_kf1 with return_kf2 (G4 mask IoU: panel axes disagree)" in match.how_checked(pm)


def test_a_wheel_takes_its_next_pair_too(runs_dir: Path, th: dict) -> None:
    wheel = PanelId.LEFT_FRONT_WHEEL
    run_id = "01TESTRUNNEXTWHEEL00000000"
    keyframes = [(kf("pickup_kf1", "pickup", wheel), RECT_MASK), (kf("return_kf1", "return", wheel, sharpness=0.3), RECT_MASK), (kf("return_kf2", "return", wheel, aspect=1.5), RECT_MASK)]
    pair_run(run_id, th, keyframes, p=panel(wheel))
    pm = the_panel(run_id, th)
    assert tries(pm) == [("pickup_kf1", "return_kf1", ["G3"]), ("pickup_kf1", "return_kf2", [])]
    assert pm.state == MatchState.COMPARABLE and pm.frame_used == FrameUsed.WHEEL and all(t.frames_agree for t in pm.kf_pairs_tried)


def test_every_panel_with_both_walks_records_its_tries_and_a_one_walk_panel_none(runs_dir: Path, th: dict) -> None:
    run_id = "01TESTRUNNEXTPASS000000000"
    pair_run(run_id, th, [(kf("pickup_kf1", "pickup"), RECT_MASK), (kf("return_kf1", "return"), RECT_MASK)])
    pm = the_panel(run_id, th)
    assert tries(pm) == [("pickup_kf1", "return_kf1", [])] and pm.state == MatchState.COMPARABLE
    assert match.how_checked(pm) == "compared on keyframes pickup_kf1 with return_kf1, the first pair tried"
    run_id = "01TESTRUNNEXTONEWALK000000"
    pair_run(run_id, th, [(kf("pickup_kf1", "pickup"), RECT_MASK)], p=panel(states=(CoverageState.FILMED, CoverageState.NOT_FILMED)), marks=[])
    pm = the_panel(run_id, th)
    assert pm.kf_pair is None and pm.kf_pairs_tried == [] and match.how_checked(pm) is None


# The order, determinism, the rebuild check ----------------------------------------------------------------------------


def scored(kf_id: str, role: str, total: float, **kw: Any) -> Keyframe:
    k = kf(kf_id, role, **kw)
    return k.model_copy(update={"score": k.score.model_copy(update={"total": total})})


def test_the_order_is_the_chosen_pair_then_clean_views_then_the_score_product_then_the_keyframe_number(th: dict) -> None:
    pickup = [scored("pickup_kf1", "pickup", 0.5, aspect=1.2, conditions=Conditions(dirty=True)), scored("pickup_kf2", "pickup", 0.2, aspect=1.0)]
    ret = [scored("return_kf1", "return", 0.4, aspect=1.0), scored("return_kf2", "return", 0.6, aspect=1.5), scored("return_kf3", "return", 0.4, aspect=2.0)]
    order = [(a.kf_id[-3:], b.kf_id[-3:]) for a, b in match.keyframe_pair_order(pickup, ret, th=th)]
    # chosen: the clean pickup kf2 with the aspect match r1; then pairs with both views clean by score product
    # (0.12, 0.08); then the flagged pickup kf1 (0.30, then 0.20 for r1 and r3: a tie, r1 first by number)
    assert order == [("kf2", "kf1"), ("kf2", "kf2"), ("kf2", "kf3"), ("kf1", "kf2"), ("kf1", "kf1"), ("kf1", "kf3")]
    rng = random.Random(7)
    for _ in range(5):
        p, r = pickup[:], ret[:]
        rng.shuffle(p)
        rng.shuffle(r)
        assert [(a.kf_id[-3:], b.kf_id[-3:]) for a, b in match.keyframe_pair_order(p, r, th=th)] == order


def test_the_tries_rebuild_byte_identical_whatever_order_keyframes_json_lists(runs_dir: Path, th: dict) -> None:
    run_id = "01TESTRUNNEXTBYTES0000000"
    keyframes = [(kf("pickup_kf1", "pickup"), RECT_MASK), (kf("return_kf1", "return", sharpness=0.3), RECT_MASK), (kf("return_kf2", "return", aspect=1.5, sharpness=0.5), TRIANGLE), (kf("return_kf3", "return", aspect=1.7), RECT_MASK)]
    pair_run(run_id, th, keyframes)
    match.derive_match(run_id, th=th)
    first = (paths.matches_json(run_id).read_bytes(), paths.verdicts_json(run_id).read_bytes())
    match.derive_match(run_id, th=th)
    assert (paths.matches_json(run_id).read_bytes(), paths.verdicts_json(run_id).read_bytes()) == first
    stored = read_json_list(paths.keyframes_json(run_id), Keyframe)
    write_json(paths.keyframes_json(run_id), [stored[0], stored[3], stored[2], stored[1]])
    match.derive_match(run_id, th=th)
    assert (paths.matches_json(run_id).read_bytes(), paths.verdicts_json(run_id).read_bytes()) == first
    assert b'"kf_pairs_tried"' in first[0]


def test_why_rebuild_differs_reads_the_chosen_pair_from_the_first_try(runs_dir: Path, th: dict) -> None:
    run_id = "01TESTRUNNEXTWHY000000000"
    pair_run(run_id, th, [(kf("pickup_kf1", "pickup"), RECT_MASK), (kf("return_kf1", "return", sharpness=0.3), RECT_MASK), (kf("return_kf2", "return", aspect=1.5), RECT_MASK)])
    pm = the_panel(run_id, th)
    assert pm.kf_pair == ["pickup_kf1", "return_kf2"]  # not the pair pair_keyframes chooses, which the first try holds
    assert match.why_rebuild_differs(run_id, "derived/matches.json") is None
    rows = json.loads(paths.matches_json(run_id).read_text())
    rows[0]["kf_pairs_tried"][0]["kf_pair"] = ["pickup_kf1", "return_kf2"]  # a first try today's preference would not pick
    paths.matches_json(run_id).write_text(json.dumps(rows))
    assert match.why_rebuild_differs(run_id, "derived/matches.json") == match.RULE_CHANGE_1031


# The sheet's words --------------------------------------------------------------------------------------------------


def t(p: str, r: str, gates: list[str], iou: float | None = None, agree: bool = True) -> KeyframePairTry:
    return KeyframePairTry(kf_pair=[p, r], gates=gates, mask_iou=iou, frames_agree=agree)


def test_how_checked_words_every_case() -> None:
    def text(*ts: KeyframePairTry) -> str | None:
        return match.how_checked(PanelMatch(panel_id=PanelId.HOOD, state=MatchState.COMPARABLE, kf_pairs_tried=list(ts)))

    assert text() is None
    assert text(t("p1", "r1", [])) == "compared on keyframes p1 with r1, the first pair tried"
    assert text(t("p1", "r1", ["G3"]), t("p1", "r2", ["G4"], 0.62), t("p2", "r1", [])) == "compared on keyframes p2 with r1; tried first: p1 with r1 (G3 blurry keyframe); p1 with r2 (G4 mask IoU 0.62)"
    assert text(t("p1", "r1", ["G4", "G8"], None, False), t("p2", "r1", ["G8"], 0.75)) == (
        "not compared: none of the 2 keyframe pairs passed every gate; tried: p1 with r1 (G4 mask IoU: panel axes disagree, G8 hood IoU: panel axes disagree); p2 with r1 (G8 hood IoU 0.75)"
    )
    assert text(t("p1", "r1", ["G1", "G4"], 0.5)) == "not compared: keyframes p1 with r1 (G1 not filmed, G4 mask IoU 0.50); G1 is a panel gate, so no other pair was tried"
    assert text(t("p1", "r1", ["G1", "G6"])).endswith("(G1 not filmed, G6 side uncertain); G1 and G6 are panel gates, so no other pair was tried")
    assert text(t("p1", "r1", ["G9"])) == "not compared: keyframes p1 with r1 (G9 mask definition differs), and the panel has no other keyframe pair"
    assert text(t("p1", "r1", ["G4"])) == "not compared: keyframes p1 with r1 (G4 mask IoU: not measured), and the panel has no other keyframe pair"
    assert text(t("p1", "r1", ["G5", "G10"])).endswith("G10 is a panel gate, so no other pair was tried")
    for words in (text(t("p1", "r1", ["G4", "G8"], None, False), t("p2", "r1", [])), text(t("p1", "r1", ["G1", "G6"]))):
        assert words.isascii() and "--" not in words


def test_the_sheet_prints_how_it_was_checked_the_type_note_and_the_glossary_lines(runs_dir: Path, th: dict) -> None:
    run_id = "sheet1101"
    paths.ensure_run_layout(run_id)
    write_json(paths.run_json(run_id), RunManifest(run_id=run_id, created_at="2026-11-01T05:00:00Z", thresholds=th, config_hash=config_hash(th)))
    note = f"{CHIPS}: the second look was not asked, so K4 and K12 cap the pair as before"
    write_json(paths.matches_json(run_id), [
        PanelMatch(panel_id=DOOR, state=MatchState.COMPARABLE, kf_pair=["pickup_kf1", "return_kf2"], kf_pairs_tried=[t("pickup_kf1", "return_kf1", ["G3"]), t("pickup_kf1", "return_kf2", [])]),
        PanelMatch(panel_id=PanelId.HOOD, state=MatchState.NOT_COMPARABLE, reasons=["G4"], kf_pair=["pickup_hood_kf1", "return_hood_kf1"]),
    ])
    write_json(paths.verdicts_json(run_id), [
        Verdict(verdict_id="left_front_door_v1", panel_id=DOOR, label=VerdictLabel.NEEDS_YOUR_EYE, rule_id="L4", mark_ids=["p0", "r0"], trace=Trace(rule_ids=["L4", "L3", "K12", "K4"]), reasons=rules.describe(["K12", "K4"]), type_note=note),
        Verdict(verdict_id="left_front_door_v2", panel_id=DOOR, label=VerdictLabel.ALREADY_THERE, rule_id="L3", mark_ids=["p1", "r1"], trace=Trace(rule_ids=["L3"])),
        Verdict(verdict_id="hood_v1", panel_id=PanelId.HOOD, label=VerdictLabel.NOT_COMPARABLE, rule_id="L6", trace=Trace(rule_ids=["L6", "G4"])),
    ])
    model = sheet.build_sheet(run_id, th=th)
    door = next(b for b in model.panels if b.panel_id == DOOR)
    hood = next(b for b in model.panels if b.panel_id == PanelId.HOOD)
    assert door.how_checked == "compared on keyframes pickup_kf1 with return_kf2; tried first: pickup_kf1 with return_kf1 (G3 blurry keyframe)"
    assert hood.how_checked is None and "how_checked" not in json.loads(hood.to_json())  # a pair matched before 11/1
    assert [line.type_line for line in door.verdicts] == [note, None] and "type_line" not in json.loads(door.verdicts[1].to_json())
    html = sheet.render_html(model)
    assert f"how it was checked: {door.how_checked}" in html and f"<div>{note}</div>" in html and html.isascii()
    glossary = model.appendices["B glossary"]
    assert sheet.GLOSSARY_TIGHT_PAIR in glossary and sheet.GLOSSARY_KEYFRAME_PAIR in glossary and glossary.isascii() and "--" not in glossary
    old = sheet.glossary_md(frozen(th))
    assert sheet.GLOSSARY_TIGHT_PAIR not in old and sheet.GLOSSARY_KEYFRAME_PAIR not in old and old == sheet.glossary_md()
    assert "K1 to K15" in old and sheet.glossary_md(frozen(th, "next_keyframe_pair")) == old.replace("units:", sheet.GLOSSARY_TIGHT_PAIR + "units:", 1)
