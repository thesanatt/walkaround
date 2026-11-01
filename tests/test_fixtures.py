"""Factory fixtures are not damage (lead decision 4 of 11/1, thresholds matching.fixtures_not_damage), offline.

Two parts behind the one leaf. (a) A mark whose box sits on a fixture the run already has on its own crop (a fixture
landmark Muse Spark pointed at, or the SAM door-handle point) by at least matching.fixture_overlap_min of its box is
left out of matching, gets no verdict, and is named in how_checked. A real mark next to a fixture, or a large mark that
only contains one, stays in. (b) The marks_v1 prompt carries one sentence asking Spark not to box factory fixtures;
the schema is unchanged. With the leaf absent (every run frozen before it) nothing changes, byte for byte, and the
rebuild reads spark.json with no call."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from walkaround import match, paths, sheet, spark_client
from walkaround import contracts as c
from walkaround.contracts import FixtureDrop, GridBox, InputRef, MarkSource, PanelId, SparkPurpose, SparkRequestRecord, write_json

from .test_match import RECT_MASK, kf, mark, runs_dir, th  # noqa: F401  (runs_dir, th are fixtures)
from .test_match_1101 import pair_run
from .test_spark_client import GOOD_MARKS, completed, crop, fake, ledger_path, requests_of  # noqa: F401  (fake, ledger_path are fixtures)

SHA = "0" * 64
OLD_PROMPT_SHA = "6f9b110e7a088efc6c5baf019b5a30f1ed1d7a84edc0f766f57505a0ac0bc57c"  # marks_prompt(rear bumper, ["sensor"]) at 06b61be


def frozen(th: dict) -> dict:
    """The thresholds of a run frozen before lead decision 4: the three fixture leaves absent."""
    return {**th, "matching": {k: v for k, v in th["matching"].items() if not k.startswith("fixture")}}


def boxed(m: c.Mark, x1: int, y1: int, x2: int, y2: int) -> c.Mark:
    return m.model_copy(update={"grid_bbox": GridBox(x_min=x1, y_min=y1, x_max=x2, y_max=y2)})


def marks_record(crop_path: str, landmarks: list[dict[str, Any]], rid: str = "req_return") -> SparkRequestRecord:
    parsed = {"marks": [], "landmarks": landmarks, "conditions": {"wet": False, "dirty": False, "glare": False, "dark": False}, "neighbour_car_visible": False}
    return SparkRequestRecord(
        request_id=rid, purpose=SparkPurpose.MARKS, model="muse-spark-1.3", schema_name="marks_v1",
        input_refs=[InputRef(path=crop_path, sha256=SHA)], raw_text=json.dumps(parsed), parsed=parsed, valid=True,
    )


HANDLE_AT_CENTRE = [{"name": "handle", "point": {"x": 500, "y": 500}}]  # the door handle zone: x 410 to 590, y 470 to 530


# The leaf and the prompt ---------------------------------------------------------------------------------------------


def test_the_leaf_turns_it_on_and_a_run_frozen_without_it_keeps_it_off(th: dict) -> None:
    assert match.fixtures_not_damage(th) and th["matching"]["fixture_overlap_min"] == 0.5
    assert match.fixture_halves(th)["sensor"] == (30.0, 30.0) and match.fixture_halves(th)["handle"] == (90.0, 30.0)
    assert not match.fixtures_not_damage(frozen(th)) and not match.fixtures_not_damage(None)
    assert match.fixture_drops([boxed(mark("r1", "return", u=0.0, v=0.0), 450, 480, 550, 520)], [kf("return_kf1", "return")], [marks_record("crops/return/return_kf1.jpg", HANDLE_AT_CENTRE)], frozen(th)) == {}


def test_the_prompt_says_it_in_one_sentence_only_with_the_leaf_and_the_old_prompt_keeps_its_bytes() -> None:
    old = spark_client.marks_prompt(PanelId.REAR_BUMPER, ["sensor"])
    assert hashlib.sha256(old.encode()).hexdigest() == OLD_PROMPT_SHA and "fixture" not in old
    assert spark_client.marks_prompt(PanelId.REAR_BUMPER, ["sensor"], fixtures=False) == old
    new = spark_client.marks_prompt(PanelId.REAR_BUMPER, ["sensor"], fixtures=True)
    assert new.count(spark_client.FIXTURES_SENTENCE) == 1 and new.replace(spark_client.FIXTURES_SENTENCE, "") == old
    for words in ("parking sensors", "badges and emblems", "door locks and handles", "washer nozzles", "fuel door edges", "panel gaps", "trim seams"):
        assert words in spark_client.FIXTURES_SENTENCE
    assert spark_client.FIXTURES_SENTENCE.count(". ") == 1 and spark_client.FIXTURES_SENTENCE.isascii()


def test_find_marks_sends_the_sentence_when_asked_with_the_same_schema(fake: Any, ledger_path: Path) -> None:
    fake.responses.replies.extend([completed(GOOD_MARKS), completed(GOOD_MARKS, rid="resp_2")])
    ref = InputRef(path="crops/return/kf1.jpg", sha256=SHA)
    plain = spark_client.find_marks(crop(), PanelId.REAR_BUMPER, landmarks=["plate_corner_tl"], input_ref=ref)
    told = spark_client.find_marks(crop(), PanelId.REAR_BUMPER, landmarks=["plate_corner_tl"], input_ref=ref, fixtures=True)
    (r1, r2) = requests_of(fake)
    text = [r["input"][1]["content"][0]["text"] for r in (r1, r2)]
    assert spark_client.FIXTURES_SENTENCE not in text[0] and spark_client.FIXTURES_SENTENCE in text[1]
    assert r1["text"]["format"] == r2["text"]["format"]  # marks_v1, the same strict schema
    assert plain.answer == told.answer and told.record.schema_name == "marks_v1"


# The overlap and who is dropped --------------------------------------------------------------------------------------


def test_the_overlap_is_the_share_of_the_marks_box_on_the_fixture() -> None:
    zone = match.FixtureZone("sensor", "spark", (470.0, 470.0, 530.0, 530.0))
    assert match.fixture_overlap((480, 480, 520, 520), zone) == 1.0
    assert match.fixture_overlap((500, 480, 560, 520), zone) == pytest.approx(0.5)
    assert match.fixture_overlap((540, 480, 580, 520), zone) == 0.0
    assert match.fixture_overlap((500, 500, 500, 500), zone) == 1.0  # no area: its centre point
    assert match.fixture_overlap((600, 600, 600, 610), zone) == 0.0


def test_a_mark_on_a_spark_landmark_is_dropped_and_a_real_mark_next_to_it_or_around_it_stays(th: dict) -> None:
    k = kf("return_kf1", "return")
    on = boxed(mark("r_on", "return", u=0.0, v=0.0), 440, 480, 560, 520)  # the handle itself, boxed as a scratch
    beside = boxed(mark("r_beside", "return", u=0.4, v=0.0), 560, 480, 640, 520)  # 30 of its 80 grid units on the zone
    around = boxed(mark("r_dent", "return", u=0.2, v=0.1), 200, 300, 800, 800)  # a large dent that contains the handle
    seed = boxed(mark("r_said", "return", u=0.0, v=0.0, source=MarkSource.NARRATION_SEED), 440, 480, 560, 520)
    other_crop = boxed(mark("r_other", "return", u=0.0, v=0.0, kf_id="return_kf2"), 440, 480, 560, 520)
    drops = match.fixture_drops([on, beside, around, seed, other_crop], [k, kf("return_kf2", "return")], [marks_record(k.crop.path, HANDLE_AT_CENTRE)], th)
    assert drops == {"r_on": FixtureDrop(mark_id="r_on", fixture="handle", source="spark", overlap=1.0)}
    assert match.fixture_overlap((560, 480, 640, 520), match.fixture_zones(k, [marks_record(k.crop.path, HANDLE_AT_CENTRE)], th)[0]) == pytest.approx(0.375)


def test_only_fixture_names_of_the_panels_own_list_count(th: dict) -> None:
    k = kf("return_kf1", "return")
    rec = marks_record(k.crop.path, [{"name": "mirror_base", "point": {"x": 500, "y": 500}}, {"name": "sensor", "point": {"x": 500, "y": 500}}])
    assert match.fixture_zones(k, [rec], th) == []  # mirror_base is not a fixture; sensor is not on a door's list
    bumper = kf("return_rb_kf1", "return", PanelId.REAR_BUMPER)
    corner = marks_record(bumper.crop.path, [{"name": "plate_corner_tl", "point": {"x": 300, "y": 400}}])
    (zone,) = match.fixture_zones(bumper, [corner], th)
    assert (zone.name, zone.source, zone.box) == ("plate_corner_tl", "spark", (270.0, 370.0, 330.0, 430.0))
    a_look = corner.model_copy(update={"purpose": SparkPurpose.SECOND_LOOK})
    assert match.fixture_zones(bumper, [a_look], th) == []  # only marks answers on this crop
    not_valid = corner.model_copy(update={"valid": False})
    assert match.fixture_zones(bumper, [not_valid], th) == []


def test_the_sam_door_handle_point_is_a_fixture_too(th: dict) -> None:
    k = kf("return_kf1", "return", handle=[200.0, 150.0])  # crop 400 x 300 at scale 1: grid (500, 500)
    (zone,) = match.fixture_zones(k, [], th)
    assert (zone.name, zone.source, zone.box) == ("handle", "sam_handle", (410.0, 470.0, 590.0, 530.0))
    on = boxed(mark("r_on", "return", u=0.0, v=0.0), 470, 480, 530, 520)
    assert match.fixture_drops([on], [k], [], th)["r_on"].source == "sam_handle"
    assert match.fixture_words([match.fixture_drops([on], [k], [], th)["r_on"]]) == (
        "ignored: r_on sits on the door handle (SAM's door handle, 100% of its box), so it was not matched"
    )


# derive_match, how_checked, the sheet, the rebuild -------------------------------------------------------------------


def fixture_run(run_id: str, th: dict) -> None:
    pickup, ret = kf("pickup_kf1", "pickup"), kf("return_kf1", "return")
    ms = [
        boxed(mark("p0", "pickup", u=-0.8, v=0.0), 100, 100, 140, 140),
        boxed(mark("r0", "return", u=-0.8, v=0.0), 100, 100, 140, 140),
        boxed(mark("r_on", "return", u=0.0, v=0.0), 440, 480, 560, 520),
        boxed(mark("r_beside", "return", u=0.6, v=0.3), 560, 480, 640, 520),
    ]
    pair_run(run_id, th, [(pickup, RECT_MASK), (ret, RECT_MASK)], marks=ms)
    write_json(paths.spark_json(run_id), [marks_record("crops/return/return_kf1.jpg", HANDLE_AT_CENTRE)])


def test_derive_match_leaves_the_fixture_out_names_it_and_keeps_the_mark_beside_it(runs_dir: Path, th: dict) -> None:
    run_id = "01TESTRUNFIXTURE0000000000"
    fixture_run(run_id, th)
    (pm,), verdicts, _ = match.derive_match(run_id, th=th)
    assert "r_on" not in pm.unmatched_return and all("r_on" not in (p.pickup_mark, p.return_mark) for p in pm.pairs)
    assert pm.fixtures == [FixtureDrop(mark_id="r_on", fixture="handle", source="spark", overlap=1.0)]
    assert not any("r_on" in v.mark_ids for v in verdicts)
    assert any("r_beside" in v.mark_ids for v in verdicts) and any("r0" in v.mark_ids for v in verdicts)
    line = match.how_checked(pm)
    assert line is not None and line.endswith("ignored: r_on sits on the door handle (Muse Spark's landmark, 100% of its box), so it was not matched")
    stored = json.loads(paths.matches_json(run_id).read_text())[0]
    assert stored["fixtures"] == [{"mark_id": "r_on", "fixture": "handle", "source": "spark", "overlap": 1.0}]


def test_a_run_frozen_without_the_leaf_matches_the_fixture_mark_and_writes_no_fixtures_key(runs_dir: Path, th: dict) -> None:
    run_id = "01TESTRUNFIXTUREOLD0000000"
    fixture_run(run_id, frozen(th))
    (pm,), verdicts, _ = match.derive_match(run_id, th=frozen(th))
    assert pm.fixtures == [] and any("r_on" in v.mark_ids for v in verdicts)
    assert "fixtures" not in json.loads(paths.matches_json(run_id).read_text())[0]
    assert match.how_checked(pm) is None or "ignored" not in match.how_checked(pm)


def test_the_rebuild_reads_spark_json_with_no_call_and_gives_the_same_bytes(runs_dir: Path, th: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    run_id = "01TESTRUNFIXTUREREBUILD000"
    fixture_run(run_id, th)
    match.derive_match(run_id, th=th)
    first = (paths.matches_json(run_id).read_bytes(), paths.verdicts_json(run_id).read_bytes())

    def no_call(*_: Any, **__: Any) -> Any:
        raise AssertionError("the rebuild made a model call")

    monkeypatch.setattr(spark_client, "_send", no_call)
    match.derive_match(run_id, th=th)
    assert (paths.matches_json(run_id).read_bytes(), paths.verdicts_json(run_id).read_bytes()) == first


def test_how_checked_prints_fixtures_on_a_panel_with_no_tries_and_old_panels_print_as_before() -> None:
    drop = FixtureDrop(mark_id="return_rear_bumper_kf1_m1", fixture="sensor", source="spark", overlap=0.8749)
    pm = c.PanelMatch(panel_id=PanelId.REAR_BUMPER, state=c.MatchState.NOT_COMPARABLE, reasons=["G1"], fixtures=[drop])
    assert match.how_checked(pm) == (
        "ignored: return_rear_bumper_kf1_m1 sits on a parking sensor (Muse Spark's landmark, 87% of its box), so it was not matched"
    )
    assert match.how_checked(pm.model_copy(update={"fixtures": []})) is None
    assert "fixtures" not in pm.model_copy(update={"fixtures": []}).model_dump(mode="json")


def test_the_glossary_line_prints_only_with_the_leaf(th: dict) -> None:
    assert sheet.GLOSSARY_FIXTURES in sheet.glossary_md(th)
    assert sheet.GLOSSARY_FIXTURES not in sheet.glossary_md(frozen(th))
    assert sheet.GLOSSARY_FIXTURES.isascii() and "--" not in sheet.GLOSSARY_FIXTURES
