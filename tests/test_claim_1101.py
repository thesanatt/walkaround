"""Claims follow rule 1 of 11/1 (integration, round 4): on a claim mark within tau_pre_r of a pickup mark, K4 (type dirt,
reflection or unsure) and K12 (Spark confidence low) wait for the claim's second look, as on a walk pair. Same keeps
the claim already there with a note, different or unsure lowers it (K7), and with no look (the pickup crop is gone)
they cap as before. The rule reads the run's frozen thresholds (matching.tight_pair_second_look), so a run frozen
before 11/1 keeps the caps and its claims rebuild unchanged. Every photo is the synthetic one of tests/test_claim.py."""

from __future__ import annotations

import asyncio
import copy
from pathlib import Path

import pytest

from tests.test_claim import KF, PANEL, TH, FakeSam, FakeSpark, answers, default_dets, derive, install, make_run, marks_answer, photo, rect_mask, runs, upload  # noqa: F401
from walkaround import claim, paths, spark_client
from walkaround import contracts as c

DIRT_LOW = marks_answer(kind="dirt", conf="low")
NEAR = "Already there at pickup: the claim mark sits 0.05 units"


def tight_run(run_id: str, *, kind: c.MarkKind = c.MarkKind.DIRT, caps: tuple[str, ...] = ("K4", "K12")) -> None:
    """A pickup mark 0.05 units from where the claim mark lands (inside tau_pre_r), typed by Spark with the given caps."""
    make_run(run_id, offsets=((0.05, 0.0),), kind=kind)
    marks = c.read_json_list(paths.marks_json(run_id), c.Mark)
    low = c.Confidence.LOW if "K12" in caps else c.Confidence.HIGH
    c.write_json(paths.marks_json(run_id), [m.model_copy(update={"caps": list(caps), "confidence": low}) for m in marks])


def rule_off() -> dict:
    th = copy.deepcopy(TH)
    th["matching"].pop("tight_pair_second_look")
    return th


def test_a_tight_claim_pair_typed_dirt_goes_to_the_look_and_same_keeps_it_already_there(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tight_run("r")
    _, spark = install(monkeypatch, spark=FakeSpark(answers(marks_v1=DIRT_LOW)))
    out = derive("r", photo())
    assert "look_v1" in spark.names()
    assert out.match.reasons == ["L3"]
    assert NEAR in out.paragraph
    assert (
        "Muse Spark typed both marks dirt (K4), with low confidence on both marks (K12): a note here, since the claim mark sits where the pickup mark is"
        " and the second look compared the two crops and said same." in out.paragraph
    )


def test_a_tight_claim_pair_the_look_calls_different_is_lowered_by_k7_alone(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tight_run("r")
    install(monkeypatch, spark=FakeSpark(answers(marks_v1=DIRT_LOW, look_v1={"verdict": "different", "reason": "not the same line"})))
    out = derive("r", photo())
    assert out.match.reasons == ["L4", "L3", "K7"]
    assert "second look doubts the pairing" in out.paragraph
    assert "with low confidence on both marks (K12): a note here; the second look compared the two crops and said different (K7)." in out.paragraph


def test_one_capped_mark_is_named_by_its_role(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tight_run("r", kind=c.MarkKind.SCRATCH, caps=("K12",))  # a low-confidence pickup scratch; the claim mark is a sure scratch
    install(monkeypatch)
    out = derive("r", photo())
    assert out.match.reasons == ["L3"]
    assert "Muse Spark's confidence was low on the pickup mark (K12): a note here, since the claim mark sits where the pickup mark is" in out.paragraph


def test_with_the_rule_off_k4_and_k12_cap_the_claim_and_no_look_is_asked(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tight_run("r")
    _, spark = install(monkeypatch, spark=FakeSpark(answers(marks_v1=DIRT_LOW)))
    image = upload("r", photo())
    out = asyncio.run(claim.derive_claim("r", image=image, text="", tire_code=None, th=rule_off(), panel_id=None))
    assert "look_v1" not in spark.names()
    assert out.match.reasons == ["L4", "L3", "K12", "K4"]
    assert "(K4)" not in out.paragraph and "(K12)" not in out.paragraph


def test_another_cap_keeps_the_look_unasked_and_the_note_says_so(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tight_run("r", caps=("K3", "K4", "K12"))  # the pickup mark is also under the small-mark floor: K3 still caps
    _, spark = install(monkeypatch, spark=FakeSpark(answers(marks_v1=DIRT_LOW)))
    out = derive("r", photo())
    assert "look_v1" not in spark.names()
    assert out.match.reasons[:2] == ["L4", "L3"] and {"K3", "K4", "K12"} <= set(out.match.reasons)
    assert "another cap already lowers this pair, so no second look was asked and K4 and K12 cap it too." in out.paragraph


def test_a_tight_claim_pair_whose_pickup_crop_is_gone_keeps_the_caps(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tight_run("r")
    paths.crop_path("r", "pickup", KF).unlink()
    _, spark = install(monkeypatch, spark=FakeSpark(answers(marks_v1=DIRT_LOW)))
    out = derive("r", photo())
    assert "look_v1" not in spark.names()
    assert out.match.reasons == ["L4", "L3", "K12", "K4"]
    assert "the second look was not asked, so K4 and K12 cap the pair as before." in out.paragraph


def test_a_tight_claim_rebuilds_from_its_answers_with_no_call(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tight_run("r")
    install(monkeypatch, spark=FakeSpark(answers(marks_v1=DIRT_LOW)))
    out = derive("r", photo())
    spark_client.set_client(None)  # any model call would now fail
    again = claim.rebuild_claim("r", out.claim_id, th=TH, write=False)
    assert again.model_dump(mode="json") == out.model_dump(mode="json")
    assert claim.rebuild_claim("r", out.claim_id, th=rule_off(), write=False).match.reasons == ["L4", "L3", "K12", "K4"]  # thresholds without the leaf: the old caps


def test_a_claim_decides_on_its_stored_answers_so_it_rebuilds_byte_identical(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # integration 11/1: a SAM confidence kept at full precision put a vote score across a 6-place rounding step, so the
    # live claim and its rebuild (which reads the stored answers) differed by 1e-6 (pairC_core3_claim claim_4)
    make_run("r", offsets=((0.05, 0.0),))
    dets = default_dets()
    dets["trunk lid"] = [(rect_mask(PANEL), 0.600000633)]  # the lid covers half the photo: 0.5 x c sits on a rounding step
    assert round(0.5 * 0.600000633, 6) != round(0.5 * round(0.600000633, 6), 6)
    install(monkeypatch, sam=FakeSam(dets))
    out = derive("r", photo())
    stored = paths.claim_record_path("r", out.claim_id).read_bytes()
    assert c.json_bytes(claim.rebuild_claim("r", out.claim_id, th=TH, write=False)) == stored
