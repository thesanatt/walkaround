"""R24 (Nov 2): the registration vouches for the views when G4 alone fails (matching.registration_vouches), and the
pair look and the counter-look are asked up to three times with the label following the majority
(matching.look_majority). Both leaves are absent from every run frozen before them."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from walkaround import match, open_weights, paths, register, second_look, sheet, spark_client
from walkaround import contracts as c
from walkaround.contracts import LookVerdict, MatchState, VerdictLabel, load_thresholds

from .test_counter_confirm import _record
from .test_counter_look import RUN, lone, lone_run
from .test_match import RECT_MASK, kf, runs_dir, th  # noqa: F401  (runs_dir, th are fixtures)
from .test_match_1101 import TRIANGLE, pair_run, the_panel, tries

S, D, U = LookVerdict.SAME, LookVerdict.DIFFERENT, LookVerdict.UNSURE
A = match.CounterAnswer
LEAVES = load_thresholds()["matching"]


def with_leaves(th: dict, *names: str) -> dict:
    return {**th, "matching": {**th["matching"], **{n: LEAVES[n] for n in names}}}


# The majority, pure ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("answers", "want"),
    [
        ([S, S], (S, 2, 2)),
        ([S, D, S], (S, 2, 3)),
        ([D, S, D], (D, 2, 3)),
        ([D, U, S], (U, 2, 3)),  # lowered; different and unsure tie, so unsure
        ([S, D], (U, 1, 2)),  # a split did not confirm the pair
        ([S, None], (S, 1, 1)),
        ([None], (None, 0, 0)),
        ([U, U], (U, 2, 2)),
    ],
)
def test_the_pair_look_majority_keeps_same_only_when_same_outnumbers_the_rest(answers: list, want: tuple) -> None:
    assert match.majority_pair_look(answers) == want


def test_the_counter_majority_takes_each_part_on_its_own() -> None:
    m = match.majority_counter([A("no", D), A("yes", D), A("yes", D)])
    assert (m.mark, m.spot, m.answers, m.mark_votes, m.spot_votes) == ("yes", D, 3, 2, 3)
    m = match.majority_counter([A("yes", D), A("no", D), A("no", S)])
    assert (m.mark, m.spot, m.mark_votes, m.spot_votes) == ("no", D, 2, 2)
    m = match.majority_counter([A("yes", D), A("unsure", S)])  # both parts split: unsure on both
    assert (m.mark, m.spot) == ("unsure", U)
    m = match.majority_counter([A("no", D), A("unsure", D), A("unsure", D)])
    assert m.mark == "unsure" and m.mark_votes == 3
    assert match.majority_counter([None, None]) is None


def test_asking_stops_once_two_agree_on_every_part_and_after_a_failed_ask() -> None:
    cfg = {"asks_max": 3, "agree": 2}

    def run(seq: list) -> list:
        it = iter(seq)
        return match.ask_majority(lambda: next(it), cfg)

    assert run([S, S, D]) == [S, S]
    assert run([S, D, D]) == [S, D, D]
    assert run([D, U, S]) == [D, U]  # different and unsure both lower: two agree
    assert run([None, S, S]) == [None]
    assert run([S, None, S]) == [S, None]
    assert run([A("yes", D), A("yes", S), A("yes", D)]) == [A("yes", D), A("yes", S), A("yes", D)]  # part 2 split
    assert run([A("no", D), A("no", D), A("yes", D)]) == [A("no", D), A("no", D)]
    assert match.ask_majority(lambda: S, None) == [S]


def test_a_one_question_counter_answer_stands_alone() -> None:
    assert match.counter_result([D]) == D
    assert match.counter_result([A("no", D)]) == A("no", D)
    assert match.counter_result([A("no", D), A("yes", D), A("yes", D)]).mark == "yes"


def test_the_leaves_are_off_when_absent(th: dict) -> None:
    assert match.look_majority(th) is None and match.registration_vouches(th) is None
    on = with_leaves(th, "look_majority", "registration_vouches")
    assert match.look_majority(on) == {"enabled": True, "asks_max": 3, "agree": 2}
    assert match.registration_vouches(on)["min_inliers"] == 40
    assert "majority look" not in sheet.glossary_md(th) and "majority look" in sheet.glossary_md(on)
    assert "registration vouches" not in sheet.glossary_md(th) and "registration vouches" in sheet.glossary_md(on)


# The majority in derive_match --------------------------------------------------------------------------------------


def counter_seq(seq: list) -> tuple[list, Any]:
    asked: list = []

    def counter(ask: match.CounterAsk) -> Any:
        asked.append(ask.mark_id)
        return seq[len(asked) - 1]

    return asked, counter


def test_the_counter_label_follows_two_of_three_answers(runs_dir: Path, th: dict) -> None:
    on = with_leaves(th, "look_majority")
    lone_run(RUN, on)
    asked, counter = counter_seq([A("no", D), A("yes", D), A("yes", D)])
    (v,) = match.derive_match(RUN, th=on, counter_look=counter)[1]
    assert asked == ["r0"] * 3 and v.label == VerdictLabel.NEW_OR_UNSEEN
    assert v.counter_note == "the second look saw damage in the box (2 of 3 answers) and not at the same spot in the pickup walk (3 of 3 answers), so the label stands"
    asked, counter = counter_seq([A("yes", D), A("no", D), A("no", D)])
    (v,) = match.derive_match(RUN, th=on, counter_look=counter)[1]
    assert v.trace.rule_ids == ["L4", "L2", "K17"] and "saw no damage in the box (2 of 3 answers) (K17)" in v.counter_note
    asked, counter = counter_seq([A("no", D), A("no", D)])
    match.derive_match(RUN, th=on, counter_look=counter)
    assert len(asked) == 2  # two agree on both parts


def test_without_the_leaf_the_counter_look_is_asked_once(runs_dir: Path, th: dict) -> None:
    lone_run(RUN, th)
    asked, counter = counter_seq([A("no", D), A("yes", D), A("yes", D)])
    (v,) = match.derive_match(RUN, th=th, counter_look=counter)[1]
    assert asked == ["r0"] and v.trace.rule_ids == ["L4", "L2", "K17"] and "answers" not in v.counter_note


def look_seq(seq: list) -> tuple[list, Any]:
    asked: list = []

    def look(pair: c.MatchCandidate) -> Any:
        asked.append((pair.pickup_mark, pair.return_mark))
        return seq[len(asked) - 1]

    return asked, look


def test_the_pair_look_label_follows_the_majority(runs_dir: Path, th: dict) -> None:
    on = with_leaves(th, "look_majority")
    run_id = "01TESTRUNMAJORITYPAIR00000"
    pair_run(run_id, on, [(kf("pickup_kf1", "pickup"), RECT_MASK), (kf("return_kf1", "return"), RECT_MASK)])
    asked, look = look_seq([D, S, S])
    (v,) = match.derive_match(run_id, th=on, second_look=look)[1]
    assert asked == [("p0", "r0")] * 3 and v.label == VerdictLabel.ALREADY_THERE
    asked, look = look_seq([S, D, U])
    (v,) = match.derive_match(run_id, th=on, second_look=look)[1]
    assert len(asked) == 3 and v.label == VerdictLabel.NEEDS_YOUR_EYE and "K7" in v.trace.rule_ids
    asked, look = look_seq([S, S, D])
    (v,) = match.derive_match(run_id, th=on, second_look=look)[1]
    assert len(asked) == 2 and v.label == VerdictLabel.ALREADY_THERE
    asked, look = look_seq([D, S, S])
    (v,) = match.derive_match(run_id, th=th, second_look=look)[1]  # leaf off: the one answer stands
    assert len(asked) == 1 and "K7" in v.trace.rule_ids


def test_the_tight_pair_note_names_the_tally(th: dict) -> None:
    from .test_match_1101 import spark_mark, wm

    cal = match.calibrate(sigma_frame=None, sigma_spark=None, sigma_view=None, pseudo_pair_d=None, th=th)
    a, b = spark_mark("p0", "pickup", 0.10), spark_mark("r0", "return", 0.12)
    pair = match.cost(a, b, wm(a), wm(b), cal, th)
    flags = match.flags_for(pair, a, b, {}, S, {}, return_mark=False)
    _, _, note = match.tight_pair_flags(pair, a, b, flags, S, asked=True, tally=(2, 3))
    assert note.endswith("said same (2 of 3 answers)")
    _, _, note = match.tight_pair_flags(pair, a, b, flags, S, asked=True, tally=(1, 1))
    assert note.endswith("said same")


# The majority, live rows and the replay ---------------------------------------------------------------------------


@pytest.fixture
def seq_engine(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    calls: dict[str, Any] = {"open": [], "answers": []}

    def fake_open(image_a: np.ndarray, image_b: np.ndarray, **kw: Any) -> spark_client.LookResult:
        calls["open"].append(kw)
        rec = _record(calls["answers"][len(calls["open"]) - 1], schema=kw.get("schema_name", "look_v1"))
        mark_seen, verdict, reason = open_weights.counter2_answer(rec)
        return spark_client.LookResult(verdict=verdict, reason=reason, record=rec, attempts=[rec], mark_seen=mark_seen)

    def fake_hosted(image_a: np.ndarray, image_b: np.ndarray, **kw: Any) -> spark_client.LookResult:
        raise RuntimeError("no hosted key in tests")

    monkeypatch.setattr(open_weights, "second_look", fake_open)
    monkeypatch.setattr(spark_client, "second_look", fake_hosted)
    return calls


def answer(mark: str, spot: str) -> dict[str, str]:
    return {"mark_in_box": mark, "same_mark_at_spot": spot, "reason": "r"}


def test_every_answer_is_a_row_and_the_rebuild_replays_them_byte_identical(runs_dir: Path, th: dict, seq_engine: dict) -> None:
    on = with_leaves({**th, "spark": {**th["spark"], "second_look_engine": "spark_open"}}, "look_majority")
    seq_engine["answers"] = [answer("no", "no"), answer("yes", "no"), answer("yes", "no")]
    lone_run(RUN, on)
    (v,) = match.derive_match(RUN, th=on, second_look=second_look.make_second_look(RUN, th=on))[1]
    assert v.label == VerdictLabel.NEW_OR_UNSEEN and len(seq_engine["open"]) == 3
    rows = json.loads(paths.looks_json(RUN).read_text())
    assert [(r.get("ask"), r["mark_seen"]) for r in rows] == [(None, "no"), (1, "yes"), (2, "yes")]
    live = (paths.matches_json(RUN).read_bytes(), paths.verdicts_json(RUN).read_bytes())
    match.derive_match(RUN, th=on, second_look=second_look.replay_second_look(RUN, th=on))
    assert (paths.matches_json(RUN).read_bytes(), paths.verdicts_json(RUN).read_bytes()) == live and len(seq_engine["open"]) == 3
    # a resumed run reuses every stored answer by its ask
    match.derive_match(RUN, th=on, second_look=second_look.make_second_look(RUN, th=on))
    assert len(seq_engine["open"]) == 3 and json.loads(paths.looks_json(RUN).read_text()) == rows


def test_a_stored_single_answer_replays_alone_under_the_leaf(runs_dir: Path, th: dict) -> None:
    on = with_leaves({**th, "spark": {**th["spark"], "second_look_engine": "spark_open"}}, "look_majority")
    lone_run(RUN, on)
    row = second_look.LookRecord(pickup_mark="pickup_kf1", return_mark="r0", panel_id=lone().panel_id, verdict=D, kind="counter2", mark_id="r0", mark_seen="no", valid=True)
    c.write_json(paths.looks_json(RUN), [row])
    assert "ask" not in json.loads(paths.looks_json(RUN).read_text())[0]
    (v,) = match.derive_match(RUN, th=on, second_look=second_look.replay_second_look(RUN, th=on))[1]
    assert v.trace.rule_ids == ["L4", "L2", "K17"] and "answers" not in v.counter_note


# The registration vouch --------------------------------------------------------------------------------------------


def aligned(*, ok: bool = True, model: str = "features", n_inliers: int = 110, spread: float = 0.665, reproj: float = 1.32, iou: float = 0.858) -> register.Aligned:
    return register.Aligned(
        kf_a="pickup_kf1", kf_b="return_kf1", model=model, H=np.eye(3), n_matches=179, n_inliers=n_inliers, spread=spread, mask_iou=iou, reproj_px=reproj, ok=ok, why=None if ok else "mask IoU after the warp 0.86"
    )


VOUCH = {**match.VOUCH_DEFAULTS, "enabled": True}


@pytest.mark.parametrize(
    ("kw", "why"),
    [
        ({}, None),
        ({"ok": False, "model": "outline"}, "the fit was refused (mask IoU after the warp 0.86)"),
        ({"model": "outline"}, "fitted on the outline only"),
        ({"n_inliers": 39}, "39 matched points agree, under 40"),
        ({"spread": 0.2}, "they span 0.20 of the panel, under 0.30"),
        ({"reproj": 2.5}, "reprojection 2.5 px, over 2.0"),
        ({"reproj": float("inf")}, "reprojection inf px, over 2.0"),
        ({"iou": 0.84}, "mask IoU after the warp 0.84, under 0.85"),
    ],
)
def test_the_vouch_needs_a_strong_fit_on_matched_edges(kw: dict, why: str | None) -> None:
    v = match.vouch_check(aligned(**kw), VOUCH)
    assert v.vouched is (why is None) and v.why == why


def g4_run(run_id: str, th: dict, *, extra: bool = False) -> None:
    keyframes = [(kf("pickup_kf1", "pickup", handle=[200.0, 150.0]), RECT_MASK), (kf("return_kf1", "return"), TRIANGLE)]  # no return handle: no G5
    if extra:
        keyframes.append((kf("return_kf2", "return", aspect=1.5, handle=[200.0, 150.0]), RECT_MASK))
    pair_run(run_id, th, keyframes)


@pytest.fixture
def fits(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {"fit": aligned(), "asked": []}

    def fake_fit(self: match.RegistrationCache, kf_p: str, kf_r: str) -> Any:
        state["asked"].append((kf_p, kf_r))
        return state["fit"]

    monkeypatch.setattr(match.RegistrationCache, "fit", fake_fit)
    return state


def test_a_strong_fit_passes_g4_and_the_panel_is_compared(runs_dir: Path, th: dict, fits: dict) -> None:
    on = with_leaves(th, "registration_vouches")
    run_id = "01TESTRUNVOUCH000000000000"
    g4_run(run_id, on)
    pm = the_panel(run_id, on)
    assert pm.state == MatchState.COMPARABLE and pm.kf_pair == ["pickup_kf1", "return_kf1"] and [p.pickup_mark for p in pm.pairs] == ["p0"]
    assert tries(pm) == [("pickup_kf1", "return_kf1", [])] and pm.kf_pairs_tried[0].vouch.vouched
    assert match.how_checked(pm) == (
        "compared on keyframes pickup_kf1 with return_kf1 (G4 mask IoU 0.65; G4 passed on the registration instead: 110 of 179 matched points agree "
        "over 0.67 of the panel, reprojection 1.3 px), the first pair tried"
    )
    first = (paths.matches_json(run_id).read_bytes(), paths.verdicts_json(run_id).read_bytes())
    the_panel(run_id, on)
    assert (paths.matches_json(run_id).read_bytes(), paths.verdicts_json(run_id).read_bytes()) == first


def test_a_weak_fit_leaves_g4_and_is_recorded(runs_dir: Path, th: dict, fits: dict) -> None:
    on = with_leaves(th, "registration_vouches")
    fits["fit"] = aligned(model="outline")
    run_id = "01TESTRUNVOUCHWEAK00000000"
    g4_run(run_id, on)
    pm = the_panel(run_id, on)
    assert pm.state == MatchState.NOT_COMPARABLE and pm.reasons == ["G4"] and not pm.kf_pairs_tried[0].vouch.vouched
    assert match.how_checked(pm).startswith(
        "not compared: keyframes pickup_kf1 with return_kf1 (G4 mask IoU 0.65, the registration did not vouch: fitted on the outline only), and the panel has no other keyframe pair; "
    )
    (v,) = c.read_json_list(paths.verdicts_json(run_id), c.Verdict)
    assert v.label == VerdictLabel.NOT_COMPARABLE and v.trace.rule_ids == ["L6", "G4"]


def test_a_native_pass_is_preferred_and_no_fit_is_asked(runs_dir: Path, th: dict, fits: dict) -> None:
    on = with_leaves(th, "registration_vouches")
    run_id = "01TESTRUNVOUCHNATIVE000000"
    g4_run(run_id, on, extra=True)
    pm = the_panel(run_id, on)
    assert pm.kf_pair == ["pickup_kf1", "return_kf2"] and fits["asked"] == [] and all(t.vouch is None for t in pm.kf_pairs_tried)
    assert "vouch" not in paths.matches_json(run_id).read_text()


def test_frozen_runs_never_vouch_and_keep_their_bytes(runs_dir: Path, th: dict, fits: dict) -> None:
    run_id = "01TESTRUNVOUCHFROZEN000000"
    g4_run(run_id, th)
    pm = the_panel(run_id, th)
    assert pm.state == MatchState.NOT_COMPARABLE and fits["asked"] == [] and "vouch" not in paths.matches_json(run_id).read_text()


def test_g4_with_another_gate_or_disagreeing_frames_is_not_vouched(runs_dir: Path, th: dict, fits: dict) -> None:
    on = with_leaves(th, "registration_vouches")
    run_id = "01TESTRUNVOUCHG3G4000000000"[:26]
    pair_run(run_id, on, [(kf("pickup_kf1", "pickup"), RECT_MASK), (kf("return_kf1", "return", sharpness=0.3), TRIANGLE)])
    pm = the_panel(run_id, on)
    assert pm.reasons == ["G3", "G4"] and fits["asked"] == []


def test_pairs_tried_after_the_vouched_one_are_named_as_also_tried(runs_dir: Path, th: dict, fits: dict) -> None:
    on = with_leaves(th, "registration_vouches")
    run_id = "01TESTRUNVOUCHALSO00000000"
    keyframes = [(kf("pickup_kf1", "pickup"), RECT_MASK), (kf("return_kf1", "return"), TRIANGLE), (kf("return_kf2", "return", aspect=1.5), TRIANGLE)]
    pair_run(run_id, on, keyframes)
    pm = the_panel(run_id, on)
    assert tries(pm) == [("pickup_kf1", "return_kf1", []), ("pickup_kf1", "return_kf2", ["G4"])] and fits["asked"] == [("pickup_kf1", "return_kf1")]
    assert match.how_checked(pm).startswith(
        "compared on keyframes pickup_kf1 with return_kf1 (G4 mask IoU 0.65; G4 passed on the registration instead: 110 of 179 matched points agree "
        "over 0.67 of the panel, reprojection 1.3 px), the first pair tried; also tried: pickup_kf1 with return_kf2 (G4 mask IoU 0.65)"
    )
