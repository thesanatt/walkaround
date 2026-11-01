"""The lead's second set of decisions of 11/1, offline, every branch:

K5 on tight pairs (thresholds matching.tight_pair_k5, with tight_pair_second_look): like K4 and K12, K5 (the two marks
typed as incompatible kinds) no longer caps a pair inside tau_pre_r by itself. It is a note once the second look
answered and the look may lower the pair (K7); with no usable answer K5 caps as before. Walks and claims alike.

Reshaped panel (thresholds matching.reshaped_panel): when G4 is the only failing gate on every keyframe pair tried and
the best whitened mask IoU is in [0.5, 0.7), the panel is needs your eye (L7, outline changed in place) when the two
keyframes' box aspect ratios agree within 2% and the largest blob of the opened mask difference holds at least half of
it. It never says already there or new.

Leaves absent (runs frozen before, pairC_core3 and the eval runs of 11/1 among them): the old behaviour and bytes."""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from tests.test_claim import TH, FakeSpark, answers, derive, install, make_run, marks_answer, photo, runs, upload  # noqa: F401
from walkaround import claim, geometry, match, paths, rules, sheet
from walkaround import contracts as c
from walkaround.contracts import Calibration, LookVerdict, MarkKind, MatchState, PanelMatch, ReshapeCheck, VerdictLabel

from .test_match import RECT_MASK, cal, kf, mark, runs_dir, th  # noqa: F401  (cal, runs_dir, th are fixtures)
from .test_match_1101 import TRIANGLE, frozen, judge, pair_run, spark_mark, the_panel

NEW_LEAVES = ("tight_pair_k5", "reshaped_panel")


def bite(w: int, h: int) -> np.ndarray:
    """The door rectangle with a bite out of the middle of its top edge: a reshaped panel, the difference in one place."""
    out = RECT_MASK.copy()
    out[20 : 20 + h, 200 - w // 2 : 200 + w // 2] = False
    return out


CRUSHED = bite(160, 140)  # whitened IoU 0.621 against the rectangle, the whole opened difference in one blob
CRUSHED_DEEP = bite(240, 160)  # IoU 0.463: under reshape_iou_min


def test_the_leaves_switch_the_rules_and_a_run_frozen_without_them_keeps_the_old_ones(th: dict) -> None:
    assert match.tight_pair_deferrable(th) == ("K4", "K5", "K12")
    assert match.tight_pair_deferrable(frozen(th, "tight_pair_k5")) == ("K4", "K12")
    assert match.tight_pair_deferrable(frozen(th, "tight_pair_second_look")) == ()
    assert match.reshaped_panel(th) and not match.reshaped_panel(frozen(th, "reshaped_panel"))
    assert rules.rule("L7").label == "outline changed in place"


# K5 on tight pairs, walks ------------------------------------------------------------------------------------------


def k5_pair() -> tuple[c.Mark, c.Mark]:
    """Pickup dirt (K4, low confidence K12) and a sure return chip at the same place, far enough out (r about 2) that
    the incompatible type cost (0.4) still leaves the pair inside tau_pre_r."""
    return spark_mark("p0", "pickup", 2.0), spark_mark("r0", "return", 2.02, kind=MarkKind.CHIP, low=False)


def test_a_same_answer_keeps_a_tight_pair_whose_types_disagree_already_there_with_k5_a_note(th: dict, cal: Calibration) -> None:
    a, b = k5_pair()
    (v,) = judge(th, cal, a, b, {("p0", "r0"): LookVerdict.SAME})
    assert (v.label, v.trace.rule_ids, v.trace.note_ids) == (VerdictLabel.ALREADY_THERE, ["L3"], ["K4", "K5", "K12"])
    assert v.type_note == (
        "Muse Spark typed the pickup mark dirt and the return mark chip (K4), two kinds that disagree (K5), with low confidence on the pickup mark (K12): "
        "a note here, since the mark sits in the same place in both walks and the second look compared the two crops and said same"
    )


@pytest.mark.parametrize("look", [LookVerdict.DIFFERENT, LookVerdict.UNSURE])
def test_a_different_or_unsure_answer_lowers_it_through_k7_alone(look: LookVerdict, th: dict, cal: Calibration) -> None:
    a, b = k5_pair()
    (v,) = judge(th, cal, a, b, {("p0", "r0"): look})
    assert (v.label, v.trace.rule_ids, v.trace.note_ids) == (VerdictLabel.NEEDS_YOUR_EYE, ["L4", "L3", "K7"], ["K4", "K5", "K12"])


def test_with_no_usable_answer_k5_caps_as_before(th: dict, cal: Calibration) -> None:
    a, b = k5_pair()
    (asked,) = judge(th, cal, a, b, {("p0", "r0"): None})
    assert asked.label == VerdictLabel.NEEDS_YOUR_EYE and {"K4", "K5", "K12"} <= set(asked.trace.rule_ids) and not asked.trace.note_ids
    assert asked.type_note.endswith("the second look gave no usable answer, so K4, K5 and K12 cap the pair as before")
    (unasked,) = judge(th, cal, a, b)
    assert unasked.type_note.endswith("the second look was not asked, so K4, K5 and K12 cap the pair as before")


def test_k5_alone_reads_as_two_kinds_that_disagree(th: dict, cal: Calibration) -> None:
    a = spark_mark("p0", "pickup", 2.0, kind=MarkKind.DENT, low=False)
    b = spark_mark("r0", "return", 2.02, kind=MarkKind.SCRATCH, low=False)
    (v,) = judge(th, cal, a, b, {("p0", "r0"): LookVerdict.SAME})
    assert v.trace.note_ids == ["K5"]
    assert v.type_note.startswith("Muse Spark typed the pickup mark dent and the return mark scratch, two kinds that disagree (K5): a note here")


def k5_run(run_id: str, th: dict) -> None:
    keyframes = [(kf("pickup_kf1", "pickup"), RECT_MASK), (kf("return_kf1", "return"), RECT_MASK)]
    a, b = k5_pair()
    pair_run(run_id, th, keyframes, marks=[a.model_copy(update={"kf_id": "pickup_kf1"}), b.model_copy(update={"kf_id": "return_kf1"})])


def test_derive_match_sends_the_k5_pair_to_the_look_and_a_run_frozen_without_the_leaf_does_not(runs_dir: Path, th: dict) -> None:
    asked: list[str] = []

    def look(pair: c.MatchCandidate) -> LookVerdict:
        asked.append(pair.pickup_mark)
        return LookVerdict.SAME

    k5_run("01TESTRUNK5ON0000000000000", th)
    (v,) = match.derive_match("01TESTRUNK5ON0000000000000", th=th, second_look=look)[1]
    assert asked == ["p0"] and v.label == VerdictLabel.ALREADY_THERE and v.trace.note_ids == ["K4", "K5", "K12"]
    old = frozen(th, "tight_pair_k5")
    k5_run("01TESTRUNK5OFF000000000000", old)
    (w,) = match.derive_match("01TESTRUNK5OFF000000000000", th=old, second_look=look)[1]
    assert asked == ["p0"] and w.label == VerdictLabel.NEEDS_YOUR_EYE and "K5" in w.trace.rule_ids
    assert w.type_note.endswith("another cap already lowers this pair, so no second look was asked and K4 and K12 cap it too")


# K5 on tight pairs, claims -----------------------------------------------------------------------------------------

CHIP_SURE = marks_answer(kind="chip", conf="high")


def dirt_run(run_id: str) -> None:
    """A sure pickup dirt mark (K4) 0.05 units from where the claim mark lands."""
    make_run(run_id, offsets=((0.05, 0.0),), kind=c.MarkKind.DIRT)
    marks = c.read_json_list(paths.marks_json(run_id), c.Mark)
    c.write_json(paths.marks_json(run_id), [m.model_copy(update={"caps": ["K4"], "confidence": c.Confidence.HIGH}) for m in marks])


def cheap_types(*drop: str) -> dict:
    """TH with a small incompatible-type cost, so the claim's dirt-chip pair sits inside tau_pre_r as claim_4 did on
    pairC_core3_claim (d 0.684 inside 0.754); drop removes leaves."""
    th = copy.deepcopy(TH)
    th["matching"]["type_cost_incompatible"] = 0.05
    for leaf in drop:
        th["matching"].pop(leaf)
    return th


def claim_with(th: dict) -> c.Claim:
    return asyncio.run(claim.derive_claim("r", image=upload("r", photo()), text="", tire_code=None, th=th, panel_id=None))


def test_a_claim_chip_on_a_pickup_dirt_mark_goes_to_the_look_and_same_keeps_it_already_there(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dirt_run("r")
    _, spark = install(monkeypatch, spark=FakeSpark(answers(marks_v1=CHIP_SURE)))
    out = claim_with(cheap_types())
    assert "look_v1" in spark.names()
    assert out.match.reasons == ["L3"]
    assert (
        "Muse Spark typed the pickup mark dirt and the claim mark chip (K4), two kinds that disagree (K5): a note here, since the claim mark sits where the"
        " pickup mark is and the second look compared the two crops and said same." in out.paragraph
    )
    assert claim.rebuild_claim("r", out.claim_id, th=cheap_types(), write=False).model_dump(mode="json") == out.model_dump(mode="json")


def test_without_the_leaf_k5_keeps_the_claims_look_away_as_on_pairC_core3_claim_4(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dirt_run("r")
    _, spark = install(monkeypatch, spark=FakeSpark(answers(marks_v1=CHIP_SURE)))
    out = claim_with(cheap_types("tight_pair_k5"))
    assert "look_v1" not in spark.names()
    assert out.match.reasons[:2] == ["L4", "L3"] and {"K4", "K5"} <= set(out.match.reasons)
    assert "Muse Spark typed the pickup mark dirt and the claim mark chip (K4): another cap already lowers this pair, so no second look was asked and K4 caps it too." in out.paragraph


# Reshaped panel ----------------------------------------------------------------------------------------------------


def reshape_run(run_id: str, th: dict, returns: list[tuple[c.Keyframe, np.ndarray]]) -> PanelMatch:
    pair_run(run_id, th, [(kf("pickup_kf1", "pickup"), RECT_MASK), *returns])
    return the_panel(run_id, th)


def verdicts(run_id: str) -> list[c.Verdict]:
    return c.read_json_list(paths.verdicts_json(run_id), c.Verdict)


def test_a_panel_crushed_in_place_is_needs_your_eye_with_L7_and_never_already_there_or_new(runs_dir: Path, th: dict) -> None:
    pm = reshape_run("01TESTRUNRESHAPED000000000", th, [(kf("return_kf1", "return", aspect=1.21), CRUSHED)])
    assert pm.state == MatchState.NOT_COMPARABLE and pm.reasons == ["G4"] and not pm.pairs
    rc = pm.reshape
    assert rc is not None and rc.called and rc.kf_pair == ["pickup_kf1", "return_kf1"]
    assert 0.5 <= rc.mask_iou < 0.7 and rc.aspect_off < 0.02 and rc.blob_share == pytest.approx(1.0)
    (v,) = verdicts("01TESTRUNRESHAPED000000000")
    assert (v.label, v.rule_id, v.trace.rule_ids, v.mark_ids) == (VerdictLabel.NEEDS_YOUR_EYE, "L7", ["L7", "G4"], ["p0", "r0"])
    assert v.reasons[0].startswith("L7 outline changed in place: G4 alone failed") and v.size is None
    assert match.how_checked(pm).startswith("not compared: keyframes pickup_kf1 with return_kf1 (G4 mask IoU 0.62), and the panel has no other keyframe pair; outline changed in place (L7): keyframes pickup_kf1 with return_kf1, IoU 0.62, box aspect ratios 1.200 and 1.210 (0.8% apart), largest blob 100% of the mask difference")


@pytest.mark.parametrize(
    "case, aspect, mask",
    [("a turned camera changes the box aspect", 1.5, CRUSHED), ("the difference spreads along the outline", 1.2, TRIANGLE)],
)
def test_each_failed_test_keeps_the_panel_not_comparable_and_the_record_says_why(case: str, aspect: float, mask: np.ndarray, runs_dir: Path, th: dict) -> None:
    pm = reshape_run("01TESTRUNVIEWPOINT00000000", th, [(kf("return_kf1", "return", aspect=aspect), mask)])
    assert pm.reshape is not None and not pm.reshape.called
    (v,) = verdicts("01TESTRUNVIEWPOINT00000000")
    assert (v.label, v.rule_id) == (VerdictLabel.NOT_COMPARABLE, "L6")
    assert "not called reshaped: keyframes pickup_kf1 with return_kf1" in match.how_checked(pm)


def test_the_precondition_iou_band_every_pair_g4_only_and_the_leaf(runs_dir: Path, th: dict) -> None:
    assert reshape_run("01TESTRUNDEEPBITE000000000", th, [(kf("return_kf1", "return"), CRUSHED_DEEP)]).reshape is None  # IoU under 0.5
    other = reshape_run("01TESTRUNG9TOO00000000000", th, [(kf("return_kf1", "return", window_subtracted=True), CRUSHED)])
    assert other.reasons == ["G4", "G9"] and other.reshape is None
    # a second pair failing G3 as well as G4: G4 is not the only failing gate on every pair tried
    two = reshape_run("01TESTRUNTWOPAIRS000000000", th, [(kf("return_kf1", "return"), CRUSHED), (kf("return_kf2", "return", aspect=1.25, sharpness=0.3), CRUSHED)])
    assert len(two.kf_pairs_tried) == 2 and two.reshape is None
    old = frozen(th, "reshaped_panel")
    pm = reshape_run("01TESTRUNRESHAPEOFF0000000", old, [(kf("return_kf1", "return", aspect=1.21), CRUSHED)])
    assert pm.reshape is None and "reshape" not in json.loads(paths.matches_json("01TESTRUNRESHAPEOFF0000000").read_text())[0]
    assert verdicts("01TESTRUNRESHAPEOFF0000000")[0].rule_id == "L6"


def test_the_best_iou_pair_is_tested_and_a_tie_keeps_the_first_tried(th: dict) -> None:
    def t(p: str, r: str, iou: float | None) -> c.KeyframePairTry:
        return c.KeyframePairTry(kf_pair=[p, r], gates=["G4"], mask_iou=iou, frames_agree=iou is not None)

    kfs = {k: kf(k, k.split("_")[0], aspect=1.2) for k in ("pickup_kf1", "pickup_kf2", "return_kf1")}
    kfs = {k: v.model_copy(update={"moments": geometry.moments_from_mask(RECT_MASK)}) for k, v in kfs.items()}
    masks = {k: RECT_MASK for k in kfs}
    rc = match.reshape_check("unused", [t("pickup_kf1", "return_kf1", None), t("pickup_kf2", "return_kf1", 0.66), t("pickup_kf1", "return_kf1", 0.66)], kfs, th, masks)
    assert rc is not None and rc.kf_pair == ["pickup_kf2", "return_kf1"] and rc.blob_share is None and not rc.called  # identical masks: nothing left
    assert match.reshape_check("unused", [t("pickup_kf1", "return_kf1", 0.7)], kfs, th, masks) is None
    assert match.reshape_check("unused", [t("pickup_kf1", "return_kf1", None)], kfs, th, masks) is None


def test_the_sheet_lists_the_reshaped_panel_with_its_verdict_and_the_glossary_names_both_rules(th: dict) -> None:
    text = sheet.glossary_md(th)
    assert sheet.GLOSSARY_TIGHT_PAIR_K5 in text and sheet.GLOSSARY_RESHAPED in text and sheet.GLOSSARY_TIGHT_PAIR not in text
    old = sheet.glossary_md(frozen(th, *NEW_LEAVES))
    assert sheet.GLOSSARY_TIGHT_PAIR in old and sheet.GLOSSARY_RESHAPED not in old and "K5" not in old.split("tight pair:")[1].split("\n")[0]
    rc = ReshapeCheck(kf_pair=["a", "b"], mask_iou=0.699, aspect_ratios=[0.988, 0.984], aspect_off=0.004049, blob_share=0.64, called=True)
    assert match.reshape_words(rc) == "outline changed in place (L7): keyframes a with b, IoU 0.70, box aspect ratios 0.988 and 0.984 (0.4% apart), largest blob 64% of the mask difference"
