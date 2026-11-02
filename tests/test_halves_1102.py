"""Halves pass of 11/2 (thresholds dedup.spark_halves_pass, off when absent and not in config/thresholds.json): the two
halves of a crop, each half's answer mapped back to crop pixels, the same-mark rule against boxes already kept, the
after_refine drop, the leaf off leaves the marks as they were, and the stored study answers replay through the product
code. No network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from walkaround import geometry, halves_pass, marks
from walkaround import contracts as c

TH = c.load_thresholds()
LEAF = {"enabled": True, "span_frac": 0.55, "scale": 2.0, "dup_iou": 0.3}
CROP = c.Crop(path="crops/pickup/pickup_hood_kf1.jpg", sha256="b" * 64, w=1280, h=486, scale=0.5, dx=100.0, dy=50.0)
REPO = Path(__file__).resolve().parent.parent
EVIDENCE = REPO / "docs" / "evidence" / "halves_1102"


def _th(**extra: object) -> dict:
    th = json.loads(json.dumps(TH))
    th["dedup"]["spark_halves_pass"] = {**LEAF, **extra}
    return th


def _kf() -> c.Keyframe:
    return c.Keyframe(
        kf_id="pickup_hood_kf1", role="pickup", panel_id=c.PanelId.HOOD, track_id="pickup_hood_00", f_proxy=3, pts=0.2,
        score=c.KeyframeScore(area=1, completeness=1, sharpness=1, glare=0, frontality=1, total=1), aspect_ratio=2.6,
        box_src=[100, 50, 2660, 1022], crop=CROP, moments=None, front_sign=1, marks_status=c.MarksStatus.COMPLETE,
    )


def _mark(box: list[float], mid: str, halves: bool = False) -> c.Mark:
    point = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
    return c.Mark(
        mark_id=mid, role="pickup", panel_id=c.PanelId.HOOD, kf_id="pickup_hood_kf1", source=c.MarkSource.SPARK_BOX,
        type=c.MarkKind.SCRATCH, confidence=c.Confidence.MEDIUM, size_class=c.SizeClass.S, bbox_crop=box, point_crop=point,
        bbox_src=geometry.box_crop_to_src(box, CROP), point_src=geometry.crop_to_src(point, CROP), size_px_src=40.0,
        halves=c.HalvesFind(half=0, window_crop=[0, 0, 704, 486], boxes=1) if halves else None,
    )


def _answer(boxes: list[tuple[list[float], str, str]], win: list[int], half: int, rid: str) -> halves_pass.HalvesAsk:
    """A stored half answer whose boxes are given in crop pixels (turned into the 0..1000 grid of the enlarged half)."""
    ww, wh = win[2] - win[0], win[3] - win[1]
    items = []
    for b, kind, conf in boxes:
        g = [round((b[0] - win[0]) * 1000 / ww), round((b[1] - win[1]) * 1000 / wh), round((b[2] - win[0]) * 1000 / ww), round((b[3] - win[1]) * 1000 / wh)]
        items.append({"type": kind, "point": {"x": (g[0] + g[2]) // 2, "y": (g[1] + g[3]) // 2}, "bbox": {"x_min": g[0], "y_min": g[1], "x_max": g[2], "y_max": g[3]}, "size_class": "S", "confidence": conf, "note": "n"})
    text = json.dumps({"marks": items, "conditions": {"wet": False, "dirty": False, "glare": False, "dark": False}, "landmarks": [], "neighbour_car_visible": False})
    rec = c.SparkRequestRecord(request_id=rid, purpose=c.SparkPurpose.MARKS, model="muse-spark-1.3", schema_name="marks_v1", raw_text=text, parsed=json.loads(text), valid=True)
    ow, oh = halves_pass.out_size(win, LEAF)
    return halves_pass.HalvesAsk(kf_id="pickup_hood_kf1", role="pickup", panel_id=c.PanelId.HOOD, half=half, crop_path=CROP.path, crop_sha256=CROP.sha256, crop_w=CROP.w, crop_h=CROP.h, window=win, out_w=ow, out_h=oh, record=rec)


def test_off_when_absent_or_disabled_and_not_in_config() -> None:
    assert halves_pass.settings(TH) is None
    assert halves_pass.settings({"dedup": {"spark_halves_pass": {**LEAF, "enabled": False}}}) is None
    assert halves_pass.settings(_th()) is not None


def test_windows_split_the_long_side_with_a_shared_band() -> None:
    assert halves_pass.windows(CROP, LEAF) == [[0, 0, 704, 486], [576, 0, 1280, 486]]
    tall = CROP.model_copy(update={"w": 400, "h": 1000})
    assert halves_pass.windows(tall, LEAF) == [[0, 0, 400, 550], [0, 450, 400, 1000]]
    assert halves_pass.out_size([576, 0, 1280, 486], LEAF) == (1408, 972)


def test_same_mark_rule() -> None:
    kept = [([100.0, 100.0, 140.0, 120.0], None)]
    same = lambda box: halves_pass.same_mark(box, None, kept, dup_iou=0.3, radius=0.2, dist=marks._dist)  # noqa: E731
    assert same([110, 105, 130, 115])  # centre inside a kept box
    assert same([90, 95, 160, 130])  # a kept centre inside it
    assert not same([300, 300, 330, 320])
    assert halves_pass.same_mark([300, 300, 330, 320], [0.0, 0.0], [([0.0, 0.0, 1.0, 1.0], [0.1, 0.1])], dup_iou=0.3, radius=0.2, dist=marks._dist)


def test_halves_marks_map_back_and_drop_what_the_full_crop_already_boxed() -> None:
    th = _th()
    kf = _kf()
    w0, w1 = halves_pass.windows(CROP, LEAF)
    base = [_mark([200.0, 200.0, 240.0, 220.0], "pickup_hood_kf1_m1")]
    asks = [
        _answer([([205.0, 202.0, 238.0, 218.0], "scratch", "medium"), ([500.0, 100.0, 540.0, 130.0], "dirt", "low")], w0, 0, "r0"),
        _answer([([600.0, 300.0, 640.0, 330.0], "chip", "high"), ([505.0, 102.0, 541.0, 131.0], "dirt", "low")], w1, 1, "r1"),  # the second is in the shared band
    ]
    out = halves_pass.halves_marks(base, kf, halves_pass.Index(asks), th=th)
    assert [m.mark_id for m in out] == ["pickup_hood_kf1_h1", "pickup_hood_kf1_h2"]
    a, b = out
    assert a.type == c.MarkKind.DIRT and set(a.caps) >= {"K4", "K12"} and a.halves.half == 0 and a.halves.boxes == 2
    assert abs(a.bbox_crop[0] - 500) <= 2 and abs(a.bbox_crop[3] - 130) <= 2
    assert b.type == c.MarkKind.CHIP and b.halves.half == 1 and abs(b.bbox_crop[0] - 600) <= 2
    assert b.grid_bbox is not None and abs(b.grid_bbox.x_min - round(b.bbox_crop[0] * 1000 / CROP.w)) <= 1
    assert halves_pass.halves_marks(base, kf, halves_pass.Index(asks), th=TH) == []  # leaf off


def test_drop_after_refine() -> None:
    full = _mark([200.0, 200.0, 240.0, 220.0], "pickup_hood_kf1_m1")
    on_it = _mark([210.0, 205.0, 236.0, 219.0], "pickup_hood_kf1_h1", halves=True)
    apart = _mark([700.0, 300.0, 730.0, 320.0], "pickup_hood_kf1_h2", halves=True)
    rows = [full, on_it, apart]
    assert halves_pass.drop_after_refine(rows, _kf(), th=_th()) == rows
    assert [m.mark_id for m in halves_pass.drop_after_refine(rows, _kf(), th=_th(after_refine=True))] == ["pickup_hood_kf1_m1", "pickup_hood_kf1_h2"]


def test_mark_json_has_no_halves_key_when_absent() -> None:
    assert "halves" not in _mark([1.0, 1.0, 5.0, 5.0], "m").model_dump(mode="json")
    assert _mark([1.0, 1.0, 5.0, 5.0], "m", halves=True).model_dump(mode="json")["halves"]["half"] == 0


@pytest.mark.skipif(not (REPO / "runs" / "lexi_pair1" / "run.json").is_file() or not (EVIDENCE / "halves_answers.json").is_file(), reason="lexi_pair1 or the study answers are not on this machine")
def test_stored_study_answers_replay_through_the_product_code() -> None:
    import sys

    sys.path.insert(0, str(EVIDENCE))
    import study  # type: ignore[import-not-found]

    run = "lexi_pair1"
    th = study.th_for(run)
    res, kfs = study.results(run, th), study.keyframes(run)
    kid = "return_left_rear_fender_kf2"
    got = study.new_marks(run, th, res, kfs, [kid])
    missed = [m["box"] for m in study.truth(run)["missed"] if m["kf_id"] == kid]
    assert sum(any(study.hits(list(m.bbox_crop), g) for m in got) for g in missed) >= 2  # raw halves boxes, before the zoom move
    saved = json.loads((EVIDENCE / "study.json").read_text())[run]["main"]
    assert saved["found_full"] == 22 and saved["finds_missed"] == 3 and saved["real"] == 32
