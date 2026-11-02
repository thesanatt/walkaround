"""Lead decision 5 of 11/1: new needs a confirmed mark (thresholds matching.counter_confirms_mark), offline, every branch.

The counter-look asks two questions in one look on the same two outlined images: (1) is there a damage mark inside
the outlined box on the mark's own crop, (2) does the other walk's crop show the same mark at the outlined spot, each
yes, no or unsure. Only (yes, no) keeps new_or_unseen or not_seen_at_return; part 1 not yes caps it with K17 ("the
second look saw no damage in the box"), part 2 not no caps it with K16. No usable answer keeps the label. A run frozen
without the leaf asks the one-question look; a stored "counter" row keeps its kind on replay."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from walkaround import match, open_weights, paths, rules, second_look, sheet, spark_client
from walkaround import contracts as c
from walkaround.contracts import Calibration, LookVerdict, VerdictLabel

from .test_counter_look import KFS, RUN, judge, lone, lone_run, one_walk_panel
from .test_match import DOOR, cal, runs_dir, th, wm  # noqa: F401  (cal, runs_dir, th are fixtures)

S, D, U = LookVerdict.SAME, LookVerdict.DIFFERENT, LookVerdict.UNSURE
A = match.CounterAnswer


def one_question(th: dict) -> dict:
    """A run frozen after decision 3 and before decision 5: the counter-look on, the new leaf absent."""
    return {**th, "matching": {k: v for k, v in th["matching"].items() if k != "counter_confirms_mark"}}


# The leaf, the rule, the answer ---------------------------------------------------------------------------------------


def test_the_leaf_needs_the_counter_look_and_is_off_when_absent(th: dict) -> None:
    assert match.counter_confirms_mark(th)
    assert not match.counter_confirms_mark(one_question(th)) and not match.counter_confirms_mark(None)
    off = {**th, "matching": {**th["matching"], "one_walk_counter_look": False}}
    assert not match.counter_confirms_mark(off)
    assert rules.rule("K17").label == "the second look saw no damage in the box"
    assert "K17" in rules.cap_ids() and "K17" in c.VerdictFlags.model_fields


def test_a_counter_answer_takes_only_yes_no_or_unsure() -> None:
    assert A("yes", D).mark == "yes"
    with pytest.raises(ValueError):
        A("maybe", D)


def test_asks_carry_the_switch(th: dict, cal: Calibration) -> None:
    m = lone()
    pm = one_walk_panel(m)
    for thx, want in ((th, True), (one_question(th), False)):
        verdicts = match.verdicts_for(pm, {"r0": m}, {}, {}, KFS, cal, c.Trace(), thx)
        (ask,) = match.counter_asks(pm, verdicts, {"r0": m}, KFS, cal, thx)
        assert ask.confirm_mark is want


# Every combination -----------------------------------------------------------------------------------------------------

K16_SAME = "said the same mark shows at the same spot in the pickup walk: the pickup walk may show it (K16)"
K16_UNSURE = "was unsure about the same spot in the pickup walk: the pickup walk may show it (K16)"
K17_NO = "saw no damage in the box (K17)"
K17_UNSURE = "was unsure there is damage in the box (K17)"

COMBOS = [
    # (part 1, part 2, rule ids, note)
    ("yes", D, ["L2"], "the second look saw damage in the box and not at the same spot in the pickup walk, so the label stands"),
    ("yes", S, ["L4", "L2", "K16"], f"the second look {K16_SAME}"),
    ("yes", U, ["L4", "L2", "K16"], f"the second look {K16_UNSURE}"),
    ("no", D, ["L4", "L2", "K17"], f"the second look {K17_NO}"),
    ("unsure", D, ["L4", "L2", "K17"], f"the second look {K17_UNSURE}"),
    ("no", S, ["L4", "L2", "K16", "K17"], f"the second look {K17_NO} and {K16_SAME}"),
    ("no", U, ["L4", "L2", "K16", "K17"], f"the second look {K17_NO} and {K16_UNSURE}"),
    ("unsure", S, ["L4", "L2", "K16", "K17"], f"the second look {K17_UNSURE} and {K16_SAME}"),
    ("unsure", U, ["L4", "L2", "K16", "K17"], f"the second look {K17_UNSURE} and {K16_UNSURE}"),
]


@pytest.mark.parametrize("part1, part2, ids, note", COMBOS)
def test_only_yes_and_no_keeps_new(part1: str, part2: LookVerdict, ids: list[str], note: str, th: dict, cal: Calibration) -> None:
    (v,) = judge(th, cal, lone(), counters={"r0": A(part1, part2)})
    assert v.trace.rule_ids == ids and v.rule_id == ids[0]
    assert v.label == (VerdictLabel.NEW_OR_UNSEEN if ids == ["L2"] else VerdictLabel.NEEDS_YOUR_EYE)
    assert v.reasons == rules.describe([i for i in ids if i.startswith("K")])
    assert v.counter_note == note


def test_a_pickup_mark_keeps_not_seen_at_return_only_on_yes_and_no(th: dict, cal: Calibration) -> None:
    (kept,) = judge(th, cal, lone("p0", "pickup"), counters={"p0": A("yes", D)})
    assert kept.label == VerdictLabel.NOT_SEEN_AT_RETURN and kept.trace.rule_ids == ["L1"]
    assert kept.counter_note == "the second look saw damage in the box and not at the same spot in the return walk, so the label stands"
    (capped,) = judge(th, cal, lone("p0", "pickup"), counters={"p0": A("no", D)})
    assert capped.label == VerdictLabel.NEEDS_YOUR_EYE and capped.trace.rule_ids == ["L4", "L1", "K17"]


def test_no_usable_answer_or_not_asked_keeps_the_label(th: dict, cal: Calibration) -> None:
    (v,) = judge(th, cal, lone(), counters={"r0": None})
    assert v.label == VerdictLabel.NEW_OR_UNSEEN and v.trace.rule_ids == ["L2"]
    assert v.counter_note == "the second look at the mark and the same spot in the pickup walk gave no usable answer, so the label stands"
    (w,) = judge(th, cal, lone(), skipped={"r0": 8})
    assert w.label == VerdictLabel.NEW_OR_UNSEEN
    assert w.counter_note == "the second look at the mark and the same spot in the pickup walk was not asked: the run's 8 counter-looks went to surer marks first, so the label stands"


def test_a_stored_one_question_answer_keeps_its_meaning_with_the_leaf_on(th: dict, cal: Calibration) -> None:
    (same,) = judge(th, cal, lone(), counters={"r0": S})
    assert same.trace.rule_ids == ["L4", "L2", "K16"] and "K17" not in same.trace.rule_ids
    (diff,) = judge(th, cal, lone(), counters={"r0": D})
    assert diff.label == VerdictLabel.NEW_OR_UNSEEN and diff.counter_note.endswith("said different, so the label stands")


def test_the_l5_pair_marks_get_the_two_question_caps(th: dict, cal: Calibration) -> None:
    a, b = lone("p0", "pickup", u=-1.2, v=0.0), lone("r0", "return", u=1.2, v=0.0)
    pair = match.cost(a, b, wm(a), wm(b), cal, th)
    assert pair.d >= pair.tau_eye_r
    pm = c.PanelMatch(panel_id=DOOR, state=c.MatchState.COMPARABLE, kf_pair=["pickup_kf1", "return_kf1"], pairs=[pair])
    out = match.verdicts_for(pm, {"p0": a, "r0": b}, {}, {}, KFS, cal, c.Trace(), th, {"p0": A("yes", D), "r0": A("no", D)})
    by = {v.mark_ids[0]: v for v in out}
    assert by["p0"].label == VerdictLabel.NOT_SEEN_AT_RETURN and by["p0"].trace.rule_ids == ["L5"]
    assert by["r0"].label == VerdictLabel.NEEDS_YOUR_EYE and by["r0"].trace.rule_ids == ["L4", "L5", "K17"]


# derive_match, replay ---------------------------------------------------------------------------------------------------


def test_derive_match_asks_two_questions_and_yes_no_keeps_new(runs_dir: Path, th: dict) -> None:
    asked: list[match.CounterAsk] = []

    def counter(ask: match.CounterAsk) -> match.CounterAnswer:
        asked.append(ask)
        return A("yes", D)

    lone_run(RUN, th)
    (v,) = match.derive_match(RUN, th=th, counter_look=counter)[1]
    assert [a.confirm_mark for a in asked] == [True]
    assert v.label == VerdictLabel.NEW_OR_UNSEEN


def test_derive_match_caps_a_blank_box(runs_dir: Path, th: dict) -> None:
    lone_run(RUN, th)
    (v,) = match.derive_match(RUN, th=th, counter_look=lambda ask: A("no", D))[1]
    assert v.label == VerdictLabel.NEEDS_YOUR_EYE and v.trace.rule_ids == ["L4", "L2", "K17"]


def test_replay_reads_each_row_by_its_own_kind(runs_dir: Path, th: dict) -> None:
    lone_run(RUN, th, [lone("r0"), lone("r1", u=-0.6, v=0.0)])
    rows = [
        second_look.LookRecord(pickup_mark="pickup_kf1", return_mark="r0", panel_id=DOOR, verdict=D, kind="counter2", mark_id="r0", mark_seen="no", valid=True),
        second_look.LookRecord(pickup_mark="pickup_kf1", return_mark="r1", panel_id=DOOR, verdict=D, kind="counter", mark_id="r1", valid=True),
    ]
    c.write_json(paths.looks_json(RUN), rows)
    replay = second_look.replay_second_look(RUN, th=th)
    assert replay.counters == {"r0": [A("no", D)], "r1": [D]}
    by = {v.mark_ids[0]: v for v in match.derive_match(RUN, th=th, second_look=replay)[1]}
    assert by["r0"].trace.rule_ids == ["L4", "L2", "K17"]
    assert by["r1"].label == VerdictLabel.NEW_OR_UNSEEN and by["r1"].counter_note.endswith("said different, so the label stands")


def test_a_row_that_did_not_run_replays_as_no_answer() -> None:
    row = second_look.LookRecord(pickup_mark="pickup_kf1", return_mark="r0", kind="counter2", mark_id="r0", reason="not run: x")
    assert second_look.counter_answer_of(row) is None
    assert second_look.counter_answer_of(row.model_copy(update={"verdict": S})) == A("unsure", S)  # part 1 lost reads unsure


# The live look, faked at the engine --------------------------------------------------------------------------------


def _record(parsed: dict[str, Any] | None, *, valid: bool = True, schema: str = "counter2_v1") -> c.SparkRequestRecord:
    return c.SparkRequestRecord(
        request_id="chatcmpl-test",
        purpose=c.SparkPurpose.SECOND_LOOK,
        model="meta-models/muse-spark-open-1026",
        engine=c.SparkEngine.BASETEN,
        schema_name=schema,
        parsed=parsed,
        valid=valid,
    )


@pytest.fixture
def engines(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    calls: dict[str, Any] = {"open": [], "hosted": [], "answer": {"mark_in_box": "no", "same_mark_at_spot": "no", "reason": "clean paint in the box"}, "open_fails": False, "valid": True}

    def fake_open(image_a: np.ndarray, image_b: np.ndarray, **kw: Any) -> spark_client.LookResult:
        calls["open"].append(kw)
        if calls["open_fails"]:
            raise open_weights.OpenWeightsError("no answer", attempts=[])
        schema = kw.get("schema_name", "look_v1")
        rec = _record(calls["answer"], valid=calls["valid"], schema=schema)
        if schema == "counter2_v1":
            mark_seen, verdict, reason = open_weights.counter2_answer(rec)
            return spark_client.LookResult(verdict=verdict, reason=reason, record=rec, attempts=[rec], mark_seen=mark_seen)
        return spark_client.LookResult(verdict=LookVerdict.SAME, reason="", record=rec, attempts=[rec])

    def fake_hosted(image_a: np.ndarray, image_b: np.ndarray, **kw: Any) -> spark_client.LookResult:
        calls["hosted"].append(kw)
        raise RuntimeError("no hosted key in tests")

    monkeypatch.setattr(open_weights, "second_look", fake_open)
    monkeypatch.setattr(spark_client, "second_look", fake_hosted)
    return calls


def look_th(th: dict) -> dict:
    return {**th, "spark": {**th["spark"], "second_look_engine": "spark_open"}}


def test_a_live_two_question_look_writes_a_counter2_row_and_rebuild_replays_it(runs_dir: Path, th: dict, engines: dict) -> None:
    th = look_th(th)
    lone_run(RUN, th)
    (v,) = match.derive_match(RUN, th=th, second_look=second_look.make_second_look(RUN, th=th))[1]
    assert v.label == VerdictLabel.NEEDS_YOUR_EYE and v.trace.rule_ids == ["L4", "L2", "K17"]
    (kw,) = engines["open"]
    assert kw["kind"] == "counter2" and kw["schema_name"] == "counter2_v1"
    assert kw["prompt"] == spark_client.counter2_prompt(DOOR, "return") and kw["pair"] == ("pickup_kf1", "r0")
    (row,) = json.loads(paths.looks_json(RUN).read_text())
    assert row["kind"] == "counter2" and row["mark_id"] == "r0" and row["mark_seen"] == "no" and row["verdict"] == "different"
    assert [Path(r["path"]).parent.name for r in row["input_refs"]] == ["return", "pickup"]
    live = paths.verdicts_json(RUN).read_bytes()
    match.derive_match(RUN, th=th, second_look=second_look.replay_second_look(RUN, th=th))
    assert paths.verdicts_json(RUN).read_bytes() == live and len(engines["open"]) == 1


def test_yes_and_no_from_the_engine_keeps_new(runs_dir: Path, th: dict, engines: dict) -> None:
    th = look_th(th)
    engines["answer"] = {"mark_in_box": "yes", "same_mark_at_spot": "no", "reason": "a scratch, clean at pickup"}
    lone_run(RUN, th)
    (v,) = match.derive_match(RUN, th=th, second_look=second_look.make_second_look(RUN, th=th))[1]
    assert v.label == VerdictLabel.NEW_OR_UNSEEN and v.counter_note.endswith("so the label stands")


def test_an_answer_that_fails_the_schema_is_unsure_on_both_parts(runs_dir: Path, th: dict, engines: dict) -> None:
    th = look_th(th)
    engines["answer"], engines["valid"] = {"verdict": "same"}, False
    lone_run(RUN, th)
    (v,) = match.derive_match(RUN, th=th, second_look=second_look.make_second_look(RUN, th=th))[1]
    assert v.trace.rule_ids == ["L4", "L2", "K16", "K17"]
    (row,) = json.loads(paths.looks_json(RUN).read_text())
    assert row["mark_seen"] == "unsure" and row["verdict"] == "unsure" and row["reason"].startswith("second look unusable")


def test_a_resumed_run_reuses_the_two_question_answer_but_not_a_one_question_row(runs_dir: Path, th: dict, engines: dict) -> None:
    th = look_th(th)
    lone_run(RUN, th)
    match.derive_match(RUN, th=th, second_look=second_look.make_second_look(RUN, th=th))
    match.derive_match(RUN, th=th, second_look=second_look.make_second_look(RUN, th=th))
    assert len(engines["open"]) == 1
    # a stored one-question row for the same mark is a different question: asked again, as counter2
    old = one_question(th)
    match.derive_match(RUN, th=old, second_look=second_look.make_second_look(RUN, th=old))
    assert engines["open"][-1]["kind"] == "counter" and "schema_name" not in engines["open"][-1]
    match.derive_match(RUN, th=th, second_look=second_look.make_second_look(RUN, th=th))
    assert engines["open"][-1]["kind"] == "counter2" and len(engines["open"]) == 3


def test_when_every_engine_fails_the_counter2_row_says_not_run_and_the_label_stands(runs_dir: Path, th: dict, engines: dict) -> None:
    th = look_th(th)
    engines["open_fails"] = True
    lone_run(RUN, th)
    (v,) = match.derive_match(RUN, th=th, second_look=second_look.make_second_look(RUN, th=th))[1]
    assert v.label == VerdictLabel.NEW_OR_UNSEEN and "gave no usable answer" in v.counter_note
    (row,) = json.loads(paths.looks_json(RUN).read_text())
    assert row["verdict"] is None and row["kind"] == "counter2" and "mark_seen" not in row
    (hosted,) = engines["hosted"]
    assert hosted["schema_name"] == "counter2_v1" and hosted["prompt"] == spark_client.counter2_prompt(DOOR, "return")


def test_a_run_frozen_without_the_leaf_asks_the_one_question_look(runs_dir: Path, th: dict, engines: dict) -> None:
    old = look_th(one_question(th))
    lone_run(RUN, old)
    match.derive_match(RUN, th=old, second_look=second_look.make_second_look(RUN, th=old))
    (kw,) = engines["open"]
    assert kw["kind"] == "counter" and kw["prompt"] == spark_client.counter_prompt(DOOR, "return") and "schema_name" not in kw
    (row,) = json.loads(paths.looks_json(RUN).read_text())
    assert row["kind"] == "counter" and "mark_seen" not in row


# The engines' own parsing ----------------------------------------------------------------------------------------------


def test_the_engines_read_both_parts(monkeypatch: pytest.MonkeyPatch) -> None:
    ok = _record({"mark_in_box": "yes", "same_mark_at_spot": "unsure", "reason": "r"})
    assert open_weights.counter2_answer(ok) == ("yes", U, "r")
    bad = _record(None, valid=False)
    part1, part2, why = open_weights.counter2_answer(bad)
    assert (part1, part2) == ("unsure", U) and why.startswith("second look unusable")
    seen: dict[str, Any] = {}

    def fake_attempts(parts: list, **kw: Any) -> list[c.SparkRequestRecord]:
        seen.update(kw, text=parts[0]["text"])
        return [ok]

    monkeypatch.setattr(spark_client, "_ask_attempts", fake_attempts)
    monkeypatch.setattr(spark_client, "prepare_image", lambda img, *a, **k: ("data:image/jpeg;base64,", 1, 1, 1.0))
    img = np.zeros((8, 8, 3), np.uint8)
    out = spark_client.second_look(img, img, panel_id=DOOR, prompt="p", schema_name="counter2_v1")
    assert seen["schema_name"] == "counter2_v1" and (out.mark_seen, out.verdict) == ("yes", U)
    monkeypatch.setattr(spark_client, "_ask_attempts", lambda parts, **kw: [bad])
    out = spark_client.second_look(img, img, panel_id=DOOR, prompt="p", schema_name="counter2_v1")
    assert (out.mark_seen, out.verdict) == ("unsure", U)


def test_the_schema_and_prompt() -> None:
    schema = spark_client.SCHEMAS["counter2_v1"]
    assert schema["required"] == ["mark_in_box", "same_mark_at_spot", "reason"] and schema["additionalProperties"] is False
    assert spark_client.schema_problems(schema, {"mark_in_box": "yes", "same_mark_at_spot": "no", "reason": "r"}) == []
    assert spark_client.schema_problems(schema, {"mark_in_box": "same", "same_mark_at_spot": "no", "reason": "r"})
    text = spark_client.counter2_prompt(DOOR, "return")
    assert "at return, with one candidate mark outlined" in text and "at pickup" in text
    assert "mark_in_box" in text and "same_mark_at_spot" in text and "scratch, scuff, chip, dent or crack" in text
    assert text.count("Answer yes") == 2 and text.count("Answer no") == 2 and text.count("Answer unsure") == 2
    assert text.isascii() and "--" not in text


def test_rows_keep_their_bytes() -> None:
    pair = second_look.LookRecord(pickup_mark="p0", return_mark="r0", verdict=S)
    old = second_look.LookRecord(pickup_mark="pickup_kf1", return_mark="r0", verdict=S, kind="counter", mark_id="r0")
    for row in (pair, old):
        assert "mark_seen" not in json.loads(row.to_json())


# Sheet --------------------------------------------------------------------------------------------------------------


def test_the_glossary_prints_the_two_question_line_only_with_the_leaf(th: dict) -> None:
    now, before = sheet.glossary_md(th), sheet.glossary_md(one_question(th))
    assert sheet.GLOSSARY_COUNTER_CONFIRM in now and sheet.GLOSSARY_COUNTER_LOOK not in now
    assert sheet.GLOSSARY_COUNTER_LOOK in before and sheet.GLOSSARY_COUNTER_CONFIRM not in before
    assert sheet.GLOSSARY_COUNTER_CONFIRM.isascii() and "K17" in sheet.GLOSSARY_COUNTER_CONFIRM
