"""The one-walk counter-look (lead decision 3 of 11/1, rule K16), offline, every branch.

A mark seen in one walk only whose label came out new_or_unseen (L2, or a return mark of an L5 pair) or
not_seen_at_return (L1) with no cap goes to the second look with two outlined images: the mark on its own walk's crop,
and the other walk's crop of the pair compared on with the spot where it would sit outlined (carried through the
whitened panel frame, the handle residual undone). same or unsure caps it at needs your eye (K16, "the other walk may
show it"); different keeps the label; no usable answer keeps the label. At most matching.counter_looks_max per run,
surest first. Runs frozen without matching.one_walk_counter_look rebuild unchanged; a rebuild replays the stored
answers with no call."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

SCRIPTS = str(Path(__file__).resolve().parents[1] / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

import eval_run  # noqa: E402
from walkaround import geometry, match, open_weights, paths, rules, second_look, sheet, spark_client  # noqa: E402
from walkaround import contracts as c  # noqa: E402
from walkaround.contracts import Calibration, Confidence, FrameUsed, LookVerdict, MatchState, PanelMatch, Trace, VerdictLabel  # noqa: E402

from .test_match import CROP_H, CROP_W, DOOR, RECT_MASK, RECT_MOMENTS, cal, kf, mark, runs_dir, th, wm  # noqa: F401  (cal, runs_dir, th are fixtures)
from .test_match_1101 import pair_run

KFS = {k: kf(k, r) for k, r in (("pickup_kf1", "pickup"), ("return_kf1", "return"))}
RUN = "01TESTRUNCOUNTER0000000000"


def frozen(th: dict) -> dict:
    """The thresholds of a run frozen before the counter-look: both leaves absent."""
    return {**th, "matching": {k: v for k, v in th["matching"].items() if k not in ("one_walk_counter_look", "counter_looks_max")}}


def lone(mark_id: str = "r0", role: str = "return", u: float = 0.3, v: float = 0.1, **kw: Any) -> c.Mark:
    """A sure scratch seen in one walk only, with a box in its crop."""
    m = mark(mark_id, role, u=u, v=v, kf_id=f"{role}_kf1", **kw)
    return m.model_copy(update={"bbox_crop": [190.0, 140.0, 210.0, 160.0], "point_crop": [200.0, 150.0]})


def one_walk_panel(*marks: c.Mark, residual: list[float] | None = None) -> PanelMatch:
    return PanelMatch(
        panel_id=DOOR,
        state=MatchState.COMPARABLE,
        kf_pair=["pickup_kf1", "return_kf1"],
        handle_residual=residual,
        unmatched_pickup=[m.mark_id for m in marks if m.role == "pickup"],
        unmatched_return=[m.mark_id for m in marks if m.role == "return"],
    )


def judge(th: dict, cal: Calibration, *marks: c.Mark, counters: dict | None = None, skipped: dict | None = None) -> list[c.Verdict]:
    pm = one_walk_panel(*marks)
    return match.verdicts_for(pm, {m.mark_id: m for m in marks}, {}, {}, KFS, cal, Trace(), th, counters, skipped)


# Leaves and the rule ------------------------------------------------------------------------------------------------


def test_the_leaves_turn_it_on_and_a_run_frozen_without_them_keeps_it_off(th: dict) -> None:
    assert match.one_walk_counter_look(th) and match.counter_looks_max(th) == 8
    assert not match.one_walk_counter_look(frozen(th)) and match.counter_looks_max(frozen(th)) == 0
    assert not match.one_walk_counter_look(None)
    assert rules.rule("K16").label == "the other walk may show it"
    assert "K16" in rules.cap_ids()


# Where the spot is ------------------------------------------------------------------------------------------------


def test_a_return_marks_spot_is_its_place_with_the_residual_undone_and_the_box_covers_tau_eye_r() -> None:
    m = lone(u=0.3, v=0.1)
    (u, v), box = match.counter_spot(m, KFS["return_kf1"], KFS["pickup_kf1"], [0.05, -0.02], 0.5)
    assert (u, v) == pytest.approx((0.25, 0.12))
    cx, cy = geometry.unwhiten((0.25, 0.12), RECT_MOMENTS, 1)
    assert box[0] < cx < box[2] and box[1] < cy < box[3]
    for t in np.linspace(0, 2 * np.pi, 7):
        px, py = geometry.unwhiten((0.25 + 0.5 * np.cos(t), 0.12 + 0.5 * np.sin(t)), RECT_MOMENTS, 1)
        assert box[0] - 0.01 <= px <= box[2] + 0.01 and box[1] - 0.01 <= py <= box[3] + 0.01
    # a pickup mark goes the other way: w + residual
    (pu, pv), _ = match.counter_spot(lone("p0", "pickup", u=0.3, v=0.1), KFS["pickup_kf1"], KFS["return_kf1"], [0.05, -0.02], 0.5)
    assert (pu, pv) == pytest.approx((0.35, 0.08))


def test_the_box_takes_in_a_large_marks_own_box_and_is_clipped_to_the_crop() -> None:
    big = lone(u=0.0, v=0.0).model_copy(update={"bbox_crop": [30.0, 30.0, 370.0, 270.0]})
    _, box = match.counter_spot(big, KFS["return_kf1"], KFS["pickup_kf1"], None, 0.05)
    assert box[0] <= 30.0 and box[1] <= 30.0 and box[2] >= 370.0 and box[3] >= 270.0
    _, wide = match.counter_spot(lone(u=0.0, v=0.0), KFS["return_kf1"], KFS["pickup_kf1"], None, 5.0)
    assert wide == (0.0, 0.0, float(CROP_W), float(CROP_H))


def test_no_spot_without_a_frame_or_a_crop_or_when_it_falls_outside_the_crop() -> None:
    other = KFS["pickup_kf1"]
    assert match.counter_spot(lone().model_copy(update={"w": None}), None, other, None, 0.5) is None
    assert match.counter_spot(lone(), None, other.model_copy(update={"moments": None}), None, 0.5) is None
    assert match.counter_spot(lone(), None, other.model_copy(update={"front_sign": None}), None, 0.5) is None
    assert match.counter_spot(lone(), None, other.model_copy(update={"crop": None}), None, 0.5) is None
    assert match.counter_spot(lone(u=9.0, v=0.0), None, other, None, 0.5) is None


def test_asks_only_uncapped_one_walk_marks_of_a_comparable_moment_panel(th: dict, cal: Calibration) -> None:
    sure, capped = lone("r0"), lone("r1", u=-0.4, caps=("K12",), confidence=Confidence.LOW)
    marks = {m.mark_id: m for m in (sure, capped)}
    pm = one_walk_panel(sure, capped)
    verdicts = match.verdicts_for(pm, marks, {}, {}, KFS, cal, Trace(), th)
    asks = match.counter_asks(pm, verdicts, marks, KFS, cal, th)
    assert [a.mark_id for a in asks] == ["r0"]
    (ask,) = asks
    assert ask.other_kf == "pickup_kf1" and ask.other_role == "pickup" and ask.key == ("pickup_kf1", "r0")
    assert ask.radius_units == pytest.approx(match.tau_r(cal.tau_eye, sure.r, th), abs=1e-6)
    assert match.counter_asks(pm.model_copy(update={"frame_used": FrameUsed.WHEEL}), verdicts, marks, KFS, cal, th) == []
    assert match.counter_asks(pm.model_copy(update={"state": MatchState.NOT_COMPARABLE}), verdicts, marks, KFS, cal, th) == []
    assert match.counter_asks(pm, verdicts, marks, {"return_kf1": KFS["return_kf1"]}, cal, th) == []  # no other keyframe
    pickup = lone("p0", "pickup")
    pm_p = one_walk_panel(pickup)
    vp = match.verdicts_for(pm_p, {"p0": pickup}, {}, {}, KFS, cal, Trace(), th)
    assert vp[0].label == VerdictLabel.NOT_SEEN_AT_RETURN
    (ap,) = match.counter_asks(pm_p, vp, {"p0": pickup}, KFS, cal, th)
    assert ap.other_kf == "return_kf1" and ap.key == ("p0", "return_kf1")


def test_both_marks_of_an_l5_pair_count_as_seen_in_one_walk(th: dict, cal: Calibration) -> None:
    a, b = lone("p0", "pickup", u=-1.2, v=0.0), lone("r0", "return", u=1.2, v=0.0)
    pair = match.cost(a, b, wm(a), wm(b), cal, th)
    assert pair.d >= pair.tau_eye_r
    pm = PanelMatch(panel_id=DOOR, state=MatchState.COMPARABLE, kf_pair=["pickup_kf1", "return_kf1"], pairs=[pair])
    marks = {"p0": a, "r0": b}
    verdicts = match.verdicts_for(pm, marks, {}, {}, KFS, cal, Trace(), th)
    assert [v.rule_id for v in verdicts] == ["L5", "L5"]
    assert sorted(x.mark_id for x in match.counter_asks(pm, verdicts, marks, KFS, cal, th)) == ["p0", "r0"]
    (vr,) = [v for v in match.verdicts_for(pm, marks, {}, {}, KFS, cal, Trace(), th, {"r0": LookVerdict.SAME}) if v.mark_ids == ["r0"]]
    assert vr.label == VerdictLabel.NEEDS_YOUR_EYE and vr.trace.rule_ids == ["L4", "L5", "K16"]


def test_surest_first_then_panel_order_then_id() -> None:
    def ask(mark_id: str) -> match.CounterAsk:
        return match.CounterAsk(mark_id=mark_id, role="return", panel_id=DOOR, other_kf="pickup_kf1", spot=(0, 0, 1, 1), w=(0.0, 0.0), radius_units=0.5)

    marks = {
        "a": lone("a", confidence=Confidence.LOW),
        "b": lone("b", confidence=Confidence.HIGH),
        "c": lone("c", confidence=Confidence.MEDIUM),
        "d": lone("d", confidence=Confidence.HIGH),
    }
    assert [x.mark_id for x in match.rank_counter_asks([ask("a"), ask("d"), ask("c"), ask("b")], marks)] == ["d", "b", "c", "a"]


# What each answer does ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("look, said", [(LookVerdict.SAME, "said same"), (LookVerdict.UNSURE, "was unsure")])
def test_same_or_unsure_caps_it_at_needs_your_eye_with_k16(look: LookVerdict, said: str, th: dict, cal: Calibration) -> None:
    (v,) = judge(th, cal, lone(), counters={"r0": look})
    assert v.label == VerdictLabel.NEEDS_YOUR_EYE and v.rule_id == "L4" and v.trace.rule_ids == ["L4", "L2", "K16"]
    assert v.reasons == rules.describe(["K16"])
    assert v.counter_note == f"the second look compared this mark with the same spot in the pickup walk and {said}: the pickup walk may show it (K16)"


def test_different_keeps_the_label_with_a_note(th: dict, cal: Calibration) -> None:
    (v,) = judge(th, cal, lone("p0", "pickup"), counters={"p0": LookVerdict.DIFFERENT})
    assert v.label == VerdictLabel.NOT_SEEN_AT_RETURN and v.trace.rule_ids == ["L1"]
    assert v.counter_note == "the second look compared this mark with the same spot in the return walk and said different, so the label stands"


def test_no_usable_answer_keeps_the_label(th: dict, cal: Calibration) -> None:
    (v,) = judge(th, cal, lone(), counters={"r0": None})
    assert v.label == VerdictLabel.NEW_OR_UNSEEN and v.trace.rule_ids == ["L2"]
    assert v.counter_note == "the second look at the same spot in the pickup walk gave no usable answer, so the label stands"


def test_a_mark_the_cap_left_out_keeps_its_label_and_says_why(th: dict, cal: Calibration) -> None:
    (v,) = judge(th, cal, lone(), skipped={"r0": 1})
    assert v.label == VerdictLabel.NEW_OR_UNSEEN
    assert v.counter_note == "the second look at the same spot in the pickup walk was not asked: the run's 1 counter-look went to surer marks first, so the label stands"
    (w,) = judge(th, cal, lone(), skipped={"r0": 8})
    assert "the run's 8 counter-looks went" in w.counter_note


def test_without_counters_the_verdict_has_no_note_and_keeps_its_bytes(th: dict, cal: Calibration) -> None:
    (v,) = judge(th, cal, lone())
    assert v.counter_note is None and "counter_note" not in json.loads(v.to_json())
    assert v.label == VerdictLabel.NEW_OR_UNSEEN


# derive_match ------------------------------------------------------------------------------------------------------


def lone_run(run_id: str, th: dict, marks: list[c.Mark] | None = None) -> None:
    keyframes = [(kf("pickup_kf1", "pickup"), RECT_MASK), (kf("return_kf1", "return"), RECT_MASK)]
    pair_run(run_id, th, keyframes, marks=marks if marks is not None else [lone()])


def test_derive_match_asks_the_lone_mark_and_same_caps_it(runs_dir: Path, th: dict) -> None:
    asked: list[match.CounterAsk] = []

    def counter(ask: match.CounterAsk) -> LookVerdict:
        asked.append(ask)
        return LookVerdict.SAME

    lone_run(RUN, th)
    (v,) = match.derive_match(RUN, th=th, counter_look=counter)[1]
    assert [a.mark_id for a in asked] == ["r0"] and asked[0].other_kf == "pickup_kf1"
    assert v.label == VerdictLabel.NEEDS_YOUR_EYE and "K16" in v.trace.rule_ids


def test_a_run_frozen_without_the_leaf_asks_nothing_and_keeps_its_bytes(runs_dir: Path, th: dict) -> None:
    old = frozen(th)
    lone_run(RUN, old)
    match.derive_match(RUN, th=old)
    before = paths.verdicts_json(RUN).read_bytes()
    match.derive_match(RUN, th=old, counter_look=lambda ask: pytest.fail("asked on a frozen run"))
    assert paths.verdicts_json(RUN).read_bytes() == before
    (v,) = json.loads(before)
    assert v["label"] == "new_or_unseen" and "counter_note" not in v


def test_with_no_look_at_all_nothing_is_asked_and_no_note_is_written(runs_dir: Path, th: dict) -> None:
    lone_run(RUN, th)
    (v,) = match.derive_match(RUN, th=th)[1]
    assert v.label == VerdictLabel.NEW_OR_UNSEEN and v.counter_note is None


def test_the_per_run_cap_goes_to_the_surest_marks(runs_dir: Path, th: dict) -> None:
    tight = {**th, "matching": {**th["matching"], "counter_looks_max": 2}}
    marks = [
        lone("r0", u=-0.9, v=0.0, confidence=Confidence.LOW),
        lone("r1", u=0.0, v=-0.6, confidence=Confidence.HIGH),
        lone("r2", u=0.9, v=0.0, confidence=Confidence.MEDIUM),
    ]
    lone_run(RUN, tight, marks)
    asked: list[str] = []

    def counter(ask: match.CounterAsk) -> LookVerdict:
        asked.append(ask.mark_id)
        return LookVerdict.DIFFERENT

    verdicts = match.derive_match(RUN, th=tight, counter_look=counter)[1]
    assert asked == ["r1", "r2"]
    by = {v.mark_ids[0]: v for v in verdicts}
    assert by["r0"].label == VerdictLabel.NEW_OR_UNSEEN and "the run's 2 counter-looks went to surer marks first" in by["r0"].counter_note
    assert all(by[m].counter_note.endswith("said different, so the label stands") for m in ("r1", "r2"))


def test_the_second_looks_own_counter_method_is_used_and_a_replay_asks_nothing(runs_dir: Path, th: dict) -> None:
    lone_run(RUN, th)
    row = second_look.LookRecord(pickup_mark="pickup_kf1", return_mark="r0", panel_id=DOOR, verdict=LookVerdict.UNSURE, kind="counter", mark_id="r0", valid=True)
    c.write_json(paths.looks_json(RUN), [row])
    replay = second_look.replay_second_look(RUN, th=th)
    assert replay is not None and replay.counters == {"r0": LookVerdict.UNSURE} and replay.pairs == {}
    (v,) = match.derive_match(RUN, th=th, second_look=replay)[1]
    assert v.label == VerdictLabel.NEEDS_YOUR_EYE and v.trace.rule_ids == ["L4", "L2", "K16"]
    assert "was unsure: the pickup walk may show it (K16)" in v.counter_note


# The live look, faked at the engine ----------------------------------------------------------------------------------


def _record(verdict: str, request_id: str = "chatcmpl-test") -> c.SparkRequestRecord:
    return c.SparkRequestRecord(
        request_id=request_id,
        purpose=c.SparkPurpose.SECOND_LOOK,
        model="meta-models/muse-spark-open-1026",
        engine=c.SparkEngine.BASETEN,
        schema_name="look_v1",
        parsed={"verdict": verdict, "reason": "a dot shows in the box"},
        valid=True,
    )


@pytest.fixture
def engines(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    calls: dict[str, Any] = {"open": [], "hosted": [], "answer": "same", "open_fails": False}

    def fake_open(image_a: np.ndarray, image_b: np.ndarray, **kw: Any) -> spark_client.LookResult:
        calls["open"].append(kw)
        if calls["open_fails"]:
            raise open_weights.OpenWeightsError("no answer", attempts=[])
        rec = _record(calls["answer"])
        return spark_client.LookResult(verdict=LookVerdict(calls["answer"]), reason="", record=rec, attempts=[rec])

    def fake_hosted(image_a: np.ndarray, image_b: np.ndarray, **kw: Any) -> spark_client.LookResult:
        calls["hosted"].append(kw)
        raise RuntimeError("no hosted key in tests")

    monkeypatch.setattr(open_weights, "second_look", fake_open)
    monkeypatch.setattr(spark_client, "second_look", fake_hosted)
    return calls


def look_th(th: dict) -> dict:
    return {**th, "spark": {**th["spark"], "second_look_engine": "spark_open"}}


def test_a_live_counter_look_writes_a_counter_row_and_rebuild_replays_it_byte_for_byte(runs_dir: Path, th: dict, engines: dict) -> None:
    th = look_th(th)
    lone_run(RUN, th)
    look = second_look.make_second_look(RUN, th=th)
    (v,) = match.derive_match(RUN, th=th, second_look=look)[1]
    assert v.label == VerdictLabel.NEEDS_YOUR_EYE and "K16" in v.trace.rule_ids
    (kw,) = engines["open"]
    assert kw["kind"] == "counter" and kw["prompt"] == spark_client.counter_prompt(DOOR, "return") and kw["pair"] == ("pickup_kf1", "r0")
    (row,) = json.loads(paths.looks_json(RUN).read_text())
    assert row["kind"] == "counter" and row["mark_id"] == "r0" and (row["pickup_mark"], row["return_mark"]) == ("pickup_kf1", "r0")
    assert row["verdict"] == "same" and row["model"] == "meta-models/muse-spark-open-1026"
    assert [Path(r["path"]).parent.name for r in row["input_refs"]] == ["return", "pickup"]
    assert all((paths.run_dir(RUN) / r["path"]).is_file() for r in row["input_refs"])
    live = paths.verdicts_json(RUN).read_bytes()
    replay = second_look.replay_second_look(RUN, th=th)
    match.derive_match(RUN, th=th, second_look=replay)
    assert paths.verdicts_json(RUN).read_bytes() == live and len(engines["open"]) == 1


def test_a_resumed_run_reuses_the_answered_counter_question(runs_dir: Path, th: dict, engines: dict) -> None:
    th = look_th(th)
    lone_run(RUN, th)
    match.derive_match(RUN, th=th, second_look=second_look.make_second_look(RUN, th=th))
    match.derive_match(RUN, th=th, second_look=second_look.make_second_look(RUN, th=th))
    assert len(engines["open"]) == 1


def test_different_from_the_engine_keeps_new(runs_dir: Path, th: dict, engines: dict) -> None:
    th = look_th(th)
    engines["answer"] = "different"
    lone_run(RUN, th)
    (v,) = match.derive_match(RUN, th=th, second_look=second_look.make_second_look(RUN, th=th))[1]
    assert v.label == VerdictLabel.NEW_OR_UNSEEN and v.counter_note.endswith("said different, so the label stands")


def test_when_every_engine_fails_the_row_says_not_run_and_the_label_stands(runs_dir: Path, th: dict, engines: dict) -> None:
    th = look_th(th)
    engines["open_fails"] = True
    lone_run(RUN, th)
    (v,) = match.derive_match(RUN, th=th, second_look=second_look.make_second_look(RUN, th=th))[1]
    assert v.label == VerdictLabel.NEW_OR_UNSEEN and "gave no usable answer" in v.counter_note
    (row,) = json.loads(paths.looks_json(RUN).read_text())
    assert row["verdict"] is None and row["reason"].startswith("not run:") and row["kind"] == "counter"
    replay = second_look.replay_second_look(RUN, th=th)
    assert replay.counter(match.CounterAsk("r0", "return", DOOR, "pickup_kf1", (0, 0, 1, 1), (0.0, 0.0), 0.5)) is None


def test_a_mark_without_a_box_or_point_is_not_run(runs_dir: Path, th: dict, engines: dict) -> None:
    th = look_th(th)
    lone_run(RUN, th, [lone().model_copy(update={"bbox_crop": None, "point_crop": None})])
    (v,) = match.derive_match(RUN, th=th, second_look=second_look.make_second_look(RUN, th=th))[1]
    assert v.label == VerdictLabel.NEW_OR_UNSEEN and engines["open"] == []
    (row,) = json.loads(paths.looks_json(RUN).read_text())
    assert row["reason"] == "not run: no box or point for r0"


def test_a_pair_row_keeps_its_bytes_and_the_provenance_line_names_the_kind(runs_dir: Path) -> None:
    pair = second_look.LookRecord(pickup_mark="p0", return_mark="r0", verdict=LookVerdict.SAME)
    assert "kind" not in json.loads(pair.to_json()) and "mark_id" not in json.loads(pair.to_json())
    paths.ensure_run_layout(RUN)
    open_weights._log(RUN, {"pickup_mark": "pickup_kf1", "return_mark": "r0", "panel_id": "left_front_door", "kind": "counter"}, open_weights.KEY_SLOTS[0])
    open_weights._log(RUN, {"pickup_mark": "p0", "return_mark": "r0", "panel_id": "left_front_door"}, open_weights.KEY_SLOTS[0])
    lines = [json.loads(x) for x in paths.open_weights_log(RUN).read_text().splitlines()]
    assert lines[0]["kind"] == "counter" and "kind" not in lines[1]


def test_the_counter_prompt_names_both_walks_and_the_three_answers() -> None:
    text = spark_client.counter_prompt(DOOR, "return")
    assert "at return, with one mark outlined" in text and "same door at pickup" in text.replace("left front door", "door")
    assert all(word in text for word in ("Answer same", "Answer different", "Answer unsure"))
    assert text.isascii() and "--" not in text


# Sheet, eval --------------------------------------------------------------------------------------------------------


def test_the_glossary_names_the_counter_look_only_when_the_leaf_is_on(th: dict) -> None:
    assert sheet.GLOSSARY_COUNTER_LOOK in sheet.glossary_md(th)
    assert sheet.GLOSSARY_COUNTER_LOOK not in sheet.glossary_md(frozen(th))
    assert sheet.GLOSSARY_COUNTER_LOOK.isascii()
    line = c.SheetVerdictLine(verdict_id="v1", label=VerdictLabel.NEW_OR_UNSEEN, rule_ids=["L2"], mark_ids=["r0"])
    assert "counter_line" not in json.loads(line.to_json())


def test_eval_takes_the_newest_eval_run_of_a_pair() -> None:
    class Run:
        def __init__(self, created: str) -> None:
            self.run = type("R", (), {"created_at": created})()

    lab = {"pair_id": "px_two_takes", "run_id": None}
    loaded = {"eval_px_two_takes": Run("2026-11-01T06:42:00Z"), "eval_px_two_takes_r6": Run("2026-11-01T08:00:00Z"), "other": Run("2026-11-02T00:00:00Z")}
    assert eval_run.choose_primary(lab, loaded, None) == "eval_px_two_takes_r6"
    assert eval_run.choose_primary(lab, {"eval_px_two_takes": loaded["eval_px_two_takes"]}, None) == "eval_px_two_takes"
    assert eval_run.choose_primary({**lab, "run_id": "eval_px_two_takes"}, loaded, None) == "eval_px_two_takes"
    assert eval_run.choose_primary(lab, {"other": loaded["other"]}, None) == "other"
    assert eval_run.choose_primary(lab, loaded, "eval_px_two_takes") == "eval_px_two_takes"
