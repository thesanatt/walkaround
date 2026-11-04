"""Halves merge of 11/2 (thresholds dedup.spark_halves_pass merge_patches, off when absent and the leaf not in
config/thresholds.json): once the zoom move and the SAM step have run, a halves box close to a full-crop box gives that
mark its box and is not kept as a second mark; a lone halves box stays only uncapped with lone_uncapped; the stored
study replays through the product code. No network."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from walkaround import halves_pass
from walkaround import contracts as c
from tests.test_halves_1102 import _kf, _mark, _th

REPO = Path(__file__).resolve().parent.parent
EVIDENCE = REPO / "docs" / "evidence" / "halves_merge_1102"
MERGE = {"merge_patches": 1.5, "patch_sent_px": 28, "lone_uncapped": True}  # the crop is sent at 1280 wide: a patch is 28 crop px


def _ids(rows: list[c.Mark]) -> list[str]:
    return [m.mark_id for m in rows]


def test_overlapping_halves_box_gives_the_full_mark_its_box() -> None:
    full = _mark([200.0, 200.0, 240.0, 220.0], "pickup_hood_kf1_m1").model_copy(update={"caps": ["K12"], "confidence": c.Confidence.LOW})
    on_it = _mark([215.0, 212.0, 251.0, 232.0], "pickup_hood_kf1_h1", halves=True).model_copy(update={"type": c.MarkKind.DIRT, "spark_request_id": "rh"})
    out = halves_pass.merge_after_refine([full, on_it], _kf(), th=_th(**MERGE))
    assert _ids(out) == ["pickup_hood_kf1_m1"]
    m = out[0]
    assert m.bbox_crop == on_it.bbox_crop and m.point_crop == on_it.point_crop and m.bbox_src == on_it.bbox_src
    assert m.type == c.MarkKind.SCRATCH and m.confidence == c.Confidence.LOW and m.caps == ["K12"] and m.halves is None
    r = m.halves_merge
    assert r is not None and r.halves_mark_id == "pickup_hood_kf1_h1" and r.overlap and r.full_bbox_crop == [200.0, 200.0, 240.0, 220.0] and r.spark_request_id == "rh"
    assert m.model_dump(mode="json")["halves_merge"]["half"] == 0
    assert "halves_merge" not in full.model_dump(mode="json")


def test_reach_in_patches() -> None:
    full = _mark([200.0, 200.0, 240.0, 220.0], "pickup_hood_kf1_m1")
    near = _mark([255.0, 200.0, 275.0, 220.0], "pickup_hood_kf1_h1", halves=True)  # centres 45 px apart: 1.6 patches, no overlap
    rows = [full, near]
    assert _ids(halves_pass.merge_after_refine(rows, _kf(), th=_th(**{**MERGE, "merge_patches": 1.5}))) == ["pickup_hood_kf1_m1", "pickup_hood_kf1_h1"]
    got = halves_pass.merge_after_refine(rows, _kf(), th=_th(**{**MERGE, "merge_patches": 2.0}))
    assert _ids(got) == ["pickup_hood_kf1_m1"] and got[0].halves_merge is not None and abs(got[0].halves_merge.gap_patches - 45 / 28) < 0.01


def test_lone_box_kept_only_uncapped_with_lone_uncapped() -> None:
    full = _mark([200.0, 200.0, 240.0, 220.0], "pickup_hood_kf1_m1")
    lone_ok = _mark([700.0, 300.0, 740.0, 320.0], "pickup_hood_kf1_h1", halves=True)
    lone_low = _mark([900.0, 100.0, 940.0, 120.0], "pickup_hood_kf1_h2", halves=True).model_copy(update={"caps": ["K12"]})
    rows = [full, lone_ok, lone_low]
    assert _ids(halves_pass.merge_after_refine(rows, _kf(), th=_th(**MERGE))) == ["pickup_hood_kf1_m1", "pickup_hood_kf1_h1"]
    keep = {k: v for k, v in MERGE.items() if k != "lone_uncapped"}
    assert _ids(halves_pass.merge_after_refine(rows, _kf(), th=_th(**keep))) == ["pickup_hood_kf1_m1", "pickup_hood_kf1_h1", "pickup_hood_kf1_h2"]


def test_closest_pair_wins_and_a_second_close_halves_box_is_dropped() -> None:
    full = _mark([200.0, 200.0, 240.0, 220.0], "pickup_hood_kf1_m1")
    far = _mark([230.0, 205.0, 260.0, 225.0], "pickup_hood_kf1_h1", halves=True)
    close = _mark([205.0, 202.0, 241.0, 222.0], "pickup_hood_kf1_h2", halves=True)
    out = halves_pass.merge_after_refine([full, far, close], _kf(), th=_th(**MERGE))
    assert _ids(out) == ["pickup_hood_kf1_m1"] and out[0].halves_merge.halves_mark_id == "pickup_hood_kf1_h2"


def test_k3_follows_the_halves_box_and_other_rows_pass_through() -> None:
    full = _mark([200.0, 200.0, 240.0, 220.0], "pickup_hood_kf1_m1").model_copy(update={"caps": ["K3", "K4"]})
    big = _mark([198.0, 198.0, 250.0, 226.0], "pickup_hood_kf1_h1", halves=True).model_copy(update={"size_px_src": 500.0})
    other = _mark([1.0, 1.0, 5.0, 5.0], "return_hood_kf1_m1").model_copy(update={"kf_id": "return_hood_kf1"})
    out = halves_pass.merge_after_refine([other, full, big], _kf(), th=_th(**MERGE))
    assert _ids(out) == ["return_hood_kf1_m1", "pickup_hood_kf1_m1"] and out[1].caps == ["K4"]


def test_without_merge_key_it_is_the_after_refine_drop() -> None:
    full = _mark([200.0, 200.0, 240.0, 220.0], "pickup_hood_kf1_m1")
    on_it = _mark([210.0, 205.0, 236.0, 219.0], "pickup_hood_kf1_h1", halves=True)
    rows = [full, on_it]
    assert halves_pass.merge_after_refine(rows, _kf(), th=_th()) == rows
    assert _ids(halves_pass.merge_after_refine(rows, _kf(), th=_th(after_refine=True))) == ["pickup_hood_kf1_m1"]
    assert halves_pass.merge_after_refine(rows, _kf(), th=c.load_thresholds()) == rows  # leaf absent


@pytest.mark.skipif(not (REPO / "runs" / "friend_pair1" / "run.json").is_file() or not (EVIDENCE / "study.json").is_file(), reason="friend_pair1 or the study is not on this machine")
def test_stored_study_replays_through_the_product_code() -> None:
    spec = importlib.util.spec_from_file_location("halves_merge_1102_study", EVIDENCE / "study.py")
    study = importlib.util.module_from_spec(spec)
    sys.modules["halves_merge_1102_study"] = study
    spec.loader.exec_module(study)  # type: ignore[union-attr]
    got = study.score("friend_pair1", "merge15")
    saved = json.loads((EVIDENCE / "study.json").read_text())["friend_pair1"]["merge15"]
    assert (got["found"], got["missed"], got["false"], got["merged"]) == (saved["found"], saved["missed"], saved["false"], saved["merged"]) == (22, 10, 13, 6)
