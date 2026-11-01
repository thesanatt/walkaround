"""Box fix of 11/1: the transforms a mark box goes through (Spark's grid to crop pixels per axis, crop to source and
back through the pixels cut_crop writes, SAM rasters scaled to their box) and the SAM box refinement (decide, apply,
plan, the stored answers, derive and rebuild with no call). No network."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from meta_sam_parser import SegmentationMask
from meta_sam_parser._mask_codec import _encode_raster

from walkaround import box_refine, config, geometry, ledger, marks, paths, privacy, sam_client, spark_client, video
from walkaround import contracts as c
from walkaround.claim import SamAnswer, SamDetection

# Spark's grid to crop pixels ---------------------------------------------------------------------------------------


def _result(sent: tuple[int, int], orig: tuple[int, int]) -> spark_client.MarksResult:
    return spark_client.MarksResult(answer=None, truncated=False, record=None, sent_width=sent[0], sent_height=sent[1], scale=sent[0] / orig[0], orig_width=orig[0], orig_height=orig[1])  # type: ignore[arg-type]


@pytest.mark.parametrize("orig,sent", [((1280, 441), (1280, 441)), ((938, 1280), (938, 1280)), ((1490, 660), (1280, 567)), ((2000, 600), (1280, 384))])
def test_grid_maps_each_axis_by_its_own_side(orig: tuple[int, int], sent: tuple[int, int]) -> None:
    """x scales by the width and y by the height, so a non-square crop maps back where Spark pointed (scaling both by
    one side would put y at 2.9x on a 2.9:1 crop)."""
    crop = c.Crop(path="", w=orig[0], h=orig[1], scale=1.0, dx=0.0, dy=0.0)
    res = _result(sent, orig)
    for gx, gy in [(0, 0), (180, 220), (500, 500), (820, 780), (999, 999)]:
        sm = c.SparkMark(type=c.MarkKind.CHIP, point=c.GridPoint(x=gx, y=gy), bbox=c.GridBox(x_min=gx, y_min=gy, x_max=min(gx + 10, 1000), y_max=min(gy + 10, 1000)), size_class=c.SizeClass.S, confidence=c.Confidence.HIGH)
        box, point, _, _ = marks.map_spark_mark(sm, res, crop)
        px_w, px_h = orig[0] / sent[0], orig[1] / sent[1]  # one sent pixel in crop pixels
        assert abs(point[0] - gx * orig[0] / 1000) <= px_w and abs(point[1] - gy * orig[1] / 1000) <= px_h
        assert abs(box[0] - gx * orig[0] / 1000) <= px_w and abs(box[1] - gy * orig[1] / 1000) <= px_h


@pytest.mark.parametrize("w,h", [(1490, 660), (2000, 600), (700, 1500)])
def test_the_spark_helpers_undo_the_resize_on_each_axis(w: int, h: int) -> None:
    """prepare_image rounds each side on its own; the far corner of the grid maps back to the crop's own far corner
    (with one scale from the width, the bottom of a 700 x 1500 crop came back at 1501)."""
    _url, sw, sh, scale = spark_client.prepare_image(np.zeros((h, w, 3), np.uint8))
    assert max(sw, sh) <= spark_client.IMAGE_LONG_EDGE
    res = spark_client.MarksResult(answer=None, truncated=False, record=None, sent_width=sw, sent_height=sh, scale=scale, orig_width=w, orig_height=h)  # type: ignore[arg-type]
    assert spark_client.box_to_crop_pixels(c.GridBox(x_min=0, y_min=0, x_max=1000, y_max=1000), res) == (0, 0, w, h)
    px, py = spark_client.point_to_crop_pixels(c.GridPoint(x=500, y=500), res)
    assert abs(px - w / 2) <= w / sw and abs(py - h / 2) <= h / sh


# Crop to source, through the pixels cut_crop writes ----------------------------------------------------------------


@pytest.mark.parametrize("box,pad,long_edge", [([2000, 900, 3100, 1500], 0.1, 1280), ([2001, 903, 2400, 1201], 0.07, 1280), ([100, 1700, 3700, 2150], 0.05, 1280)])
def test_a_dot_in_the_source_lands_where_crop_to_src_says(box: list[int], pad: float, long_edge: int) -> None:
    """A bright square at a known source position, cut with cut_crop: its centroid in the crop maps back through
    crop_to_src within half a crop pixel, and src_to_crop puts the source centre on the crop centroid."""
    W, H = 3840, 2160
    frame = np.zeros((H, W, 3), np.uint8)
    cx, cy = (box[0] + box[2]) / 2 + 13.0, (box[1] + box[3]) / 2 - 7.0
    frame[int(cy) - 6 : int(cy) + 6, int(cx) - 6 : int(cx) + 6] = 255  # centre (int(cx), int(cy)) in pixel-edge units
    true = [int(cx), int(cy)]
    crop = geometry.crop_geometry(box, (W, H), pad_frac=pad, long_edge=long_edge)
    out = video.cut_crop(frame, crop)
    assert out.shape[:2] == (crop.h, crop.w)
    lum = out[..., 0].astype(float)
    ys, xs = np.mgrid[0 : crop.h, 0 : crop.w]
    centroid = [float((lum * (xs + 0.5)).sum() / lum.sum()), float((lum * (ys + 0.5)).sum() / lum.sum())]  # pixel centres
    back = geometry.crop_to_src(centroid, crop)
    assert abs(back[0] - true[0]) <= 0.5 / crop.scale and abs(back[1] - true[1]) <= 0.5 / crop.scale
    fwd = geometry.src_to_crop(true, crop)
    assert abs(fwd[0] - centroid[0]) <= 0.5 and abs(fwd[1] - centroid[1]) <= 0.5


# SAM rasters --------------------------------------------------------------------------------------------------------


def _det(box: list[float], raster: np.ndarray, object_id: str = "1", c_: float | None = 0.8) -> SamDetection:
    h, w = raster.shape
    payload = _encode_raster(bytes(raster.astype(np.uint8).flatten().tolist()), w, h)
    return SamDetection(object_id=object_id, box=box, c=c_, mask_encoding="one_bit", mask_payload=payload, mask_w=w, mask_h=h)


def _sam_det(d: SamDetection) -> sam_client.Detection:
    seg = SegmentationMask(encoding=d.mask_encoding, payload=d.mask_payload, width=d.mask_w, height=d.mask_h)  # type: ignore[arg-type]
    return sam_client.Detection(object_id=d.object_id, frame_index=None, box=sam_client.Box(*d.box), mask=seg, confidence=d.c)


def test_a_raster_sam_scaled_down_is_stretched_back_over_its_box() -> None:
    """SAM sends an object wider than 1024 px as a 1024-wide raster (box 1101 x 154, raster 1024 x 143 on pair C's
    claims); pasted as it came the mask stopped 77 px short of the box's right edge."""
    raster = np.ones((143, 1024), bool)
    mask = privacy.detection_mask(_sam_det(_det([299.0, 746.0, 1400.0, 900.0], raster)), (1600, 1000))
    ys, xs = np.nonzero(mask)
    assert (xs.min(), xs.max() + 1, ys.min(), ys.max() + 1) == (299, 1400, 746, 900)
    small = np.zeros((20, 30), bool)
    small[5:15, 10:20] = True  # a raster the size of its box is pasted pixel for pixel
    m2 = privacy.detection_mask(_sam_det(_det([100.0, 50.0, 130.0, 70.0], small)), (400, 300))
    ys, xs = np.nonzero(m2)
    assert (xs.min(), xs.max() + 1, ys.min(), ys.max() + 1) == (110, 120, 55, 65)


# The refinement -----------------------------------------------------------------------------------------------------

TH = c.load_thresholds()
S = box_refine.settings(TH)
CROP = c.Crop(path="crops/pickup/pickup_hood_kf1.jpg", sha256="a" * 64, w=400, h=300, scale=0.5, dx=1000.0, dy=500.0)


def _mark(box: list[float], mid: str = "pickup_hood_kf1_m1", caps: list[str] | None = None) -> c.Mark:
    point = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
    return c.Mark(
        mark_id=mid, role="pickup", panel_id=c.PanelId.HOOD, kf_id="pickup_hood_kf1", source=c.MarkSource.SPARK_BOX,
        type=c.MarkKind.SCRATCH, confidence=c.Confidence.HIGH, size_class=c.SizeClass.S, bbox_crop=box, point_crop=point,
        bbox_src=geometry.box_crop_to_src(box, CROP), point_src=geometry.crop_to_src(point, CROP),
        size_px_src=max(box[2] - box[0], box[3] - box[1]) / CROP.scale, caps=list(caps or []),
    )


def _rect_det(box: list[int], object_id: str = "1") -> SamDetection:
    x1, y1, x2, y2 = box
    return _det([float(x1), float(y1), float(x2), float(y2)], np.ones((y2 - y1, x2 - x1), bool), object_id)


def _answer(dets: list[SamDetection], status: str = "completed") -> SamAnswer:
    return SamAnswer(concept="x", status=status, request_id="resp_sam_1", detections=dets)


def test_settings_are_off_when_the_leaf_is_absent() -> None:
    assert S is not None and S["max_objects"] == 16
    old = json.loads(json.dumps(TH))
    del old["dedup"]["sam_box_refine"]
    assert box_refine.settings(old) is None
    m = _mark([100.0, 100.0, 140.0, 110.0])
    assert box_refine.refine_marks([m], _kf(), box_refine.Index([]), th=old) == [m]


@pytest.mark.parametrize(
    "answer,det,used,note",
    [
        (None, None, "spark", "no SAM answer is stored for this box"),
        (_answer([], "failed"), None, "spark", "the SAM check did not complete"),
        (_answer([]), None, "spark", "SAM found nothing in the box"),
        (_answer([]), _rect_det([0, 0, 400, 300]), "spark", "covers 100.0% of the crop"),
        (_answer([]), _rect_det([95, 95, 150, 120]), "spark", "times Spark's box"),
        (_answer([]), _rect_det([150, 100, 170, 112]), "spark", "sits on Spark's box"),
        (_answer([]), _rect_det([99, 99, 141, 111]), "spark", "fills Spark's box"),
        (_answer([]), _rect_det([104, 102, 130, 107]), "sam", ""),
    ],
)
def test_decide_takes_sams_box_only_when_small_and_on_sparks_box(answer: Any, det: Any, used: str, note: str) -> None:
    m = _mark([100.0, 100.0, 140.0, 110.0])
    r = box_refine.decide(m, answer, det, CROP, S)  # type: ignore[arg-type]
    assert r.used == used and note in r.note and r.spark_bbox_crop == [100.0, 100.0, 140.0, 110.0]
    if used == "sam":
        assert r.sam_bbox_crop == [104.0, 102.0, 130.0, 107.0] and r.mask_px == 26 * 5 and r.inside == 1.0


def test_apply_moves_box_point_source_and_size_and_k3_with_the_sam_box() -> None:
    m = _mark([100.0, 100.0, 140.0, 110.0])
    r = box_refine.decide(m, _answer([]), _rect_det([110, 102, 114, 106]), CROP, S)  # type: ignore[arg-type]
    out = box_refine.apply(m, r, CROP, TH)
    assert out.bbox_crop == [110.0, 102.0, 114.0, 106.0] and out.point_crop == [112.0, 104.0]
    assert out.bbox_src == [1220.0, 704.0, 1228.0, 712.0] and out.point_src == [1224.0, 708.0] and out.size_px_src == 8.0
    assert "K3" in out.caps  # 8 source px is under small_mark_px
    assert out.box_refine is not None and out.box_refine.spark_bbox_crop == [100.0, 100.0, 140.0, 110.0] and out.box_refine.spark_point_crop == [120.0, 105.0]
    tiny = _mark([100.0, 100.0, 104.0, 104.0], caps=["K3", "K12"])
    r2 = box_refine.decide(tiny, _answer([]), _rect_det([99, 100, 106, 105]), CROP, S)  # type: ignore[arg-type]
    assert r2.used == "sam" and box_refine.apply(tiny, r2, CROP, TH).caps == ["K12"]  # 14 source px: K3 lifted
    kept = box_refine.apply(m, box_refine.decide(m, None, None, CROP, S), CROP, TH)  # type: ignore[arg-type]
    assert kept.bbox_crop == m.bbox_crop and kept.point_crop == m.point_crop and kept.box_refine is not None and kept.box_refine.used == "spark"


def test_a_mark_without_box_refine_keeps_its_bytes() -> None:
    m = _mark([100.0, 100.0, 140.0, 110.0])
    assert "box_refine" not in json.loads(c.canonical_json(m))
    back = c.Mark.model_validate_json(c.canonical_json(m.model_copy(update={"box_refine": box_refine.decide(m, None, None, CROP, S)})))  # type: ignore[arg-type]
    assert back.box_refine is not None and back.box_refine.used == "spark"


def _kf(role: str = "pickup", crop: c.Crop = CROP) -> c.Keyframe:
    return c.Keyframe(
        kf_id=f"{role}_hood_kf1", role=role, panel_id=c.PanelId.HOOD, track_id=f"{role}_hood_00", f_proxy=3, pts=0.2,
        score=c.KeyframeScore(area=1, completeness=1, sharpness=1, glare=0, frontality=1, total=1), aspect_ratio=1.5,
        box_src=[1000, 500, 1800, 1100], crop=crop, moments=None, front_sign=1, marks_status=c.MarksStatus.COMPLETE,
    )


def test_plan_puts_a_crops_boxes_in_requests_of_at_most_max_objects() -> None:
    ms = [_mark([10.0 + 20 * i, 10.0, 20.0 + 20 * i, 20.0], mid=f"pickup_hood_kf1_m{i + 1}") for i in range(18)]
    chunks = box_refine.plan(ms, _kf(), S)  # type: ignore[arg-type]
    assert [len(ch) for ch in chunks] == [16, 2] and [o.object_id for o in chunks[1]] == [1, 2]
    assert chunks[0][0].prompt == [8, 8, 22, 22]  # pad_px 2 a side
    text = box_refine.concept(chunks[1], CROP)
    assert text.startswith("<0f>1<|box;") and ",2<|box;" in text and "w=400;h=300" in text


# Derive and rebuild ------------------------------------------------------------------------------------------------


@pytest.fixture
def run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setattr(paths, "RUNS_DIR", root)
    monkeypatch.setattr(config, "RUNS_DIR", root)
    monkeypatch.setattr(ledger, "LEDGER_PATH", root / "ledger.jsonl")
    run_id = "bx1"
    paths.ensure_run_layout(run_id)
    crop_rgb = np.full((300, 400, 3), 90, np.uint8)
    path = paths.crop_path(run_id, "pickup", "pickup_hood_kf1")
    digest = video.write_jpeg(crop_rgb, path, quality=90)
    crop = c.Crop(path=paths.rel(run_id, path), sha256=digest, w=400, h=300, scale=0.5, dx=1000.0, dy=500.0)
    kf = _kf(crop=crop)
    c.write_json(paths.keyframes_json(run_id), [kf])
    text = json.dumps({"marks": [{"type": "scratch", "point": {"x": 300, "y": 350}, "bbox": {"x_min": 250, "y_min": 333, "x_max": 350, "y_max": 367}, "size_class": "M", "confidence": "high", "note": "line"}],
                       "conditions": {"wet": False, "dirty": False, "glare": False, "dark": False}, "landmarks": [], "neighbour_car_visible": False})
    record = c.SparkRequestRecord(request_id="resp_1", purpose=c.SparkPurpose.MARKS, model="muse-spark-1.3", schema_name="marks_v1", raw_text=text,
                                  parsed=json.loads(text), valid=True, input_refs=[c.InputRef(path=crop.path, sha256=digest)])
    c.write_json(paths.spark_json(run_id), [record])
    calls: list[str] = []

    async def fake_segment_image(image: Any, concept: str, **kw: Any) -> sam_client.SegmentationRun:
        calls.append(concept)
        run = sam_client.SegmentationRun(concept=concept, media="image", media_width=400, media_height=300, outcome="completed", response_id=f"resp_sam_{len(calls)}")
        run.frames[0] = [_sam_det(_rect_det([108, 102, 132, 108], "1"))]
        return run

    monkeypatch.setattr(sam_client, "segment_image", fake_segment_image)
    return {"run_id": run_id, "kf": kf, "record": record, "calls": calls, "th": json.loads(json.dumps(TH))}


def test_derive_asks_sam_once_and_rebuild_replays_with_no_call(run: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    run_id, kf, th = run["run_id"], run["kf"], run["th"]
    record = run["record"]
    monkeypatch.setattr(spark_client, "find_marks", lambda *a, **k: spark_client.marks_result_from_record(record, orig_width=400, orig_height=300, landmarks=[]))
    out, _kfs, _recs = asyncio.run(marks.derive_marks(run_id, "pickup", [kf], th=th))
    assert len(run["calls"]) == 1
    m = out[0]
    assert m.box_refine is not None and m.box_refine.used == "sam" and m.bbox_crop == [108.0, 102.0, 132.0, 108.0]
    assert m.box_refine.spark_bbox_crop == [100.0, 100.0, 141.0, 111.0]  # grid 250..350 x 333..367 on 400 x 300, far edge +1
    stored = box_refine.read_asks(run_id)
    assert len(stored) == 1 and stored[0].objects[0].mark_id == m.mark_id
    before = paths.marks_json(run_id).read_bytes()

    async def no_call(*a: Any, **k: Any) -> Any:
        raise AssertionError("rebuild must not call SAM")

    monkeypatch.setattr(sam_client, "segment_image", no_call)
    marks.rebuild_marks(run_id, "pickup", th=th)
    assert paths.marks_json(run_id).read_bytes() == before
    asyncio.run(marks.derive_marks(run_id, "pickup", [kf], th=th))  # a resumed run reuses the stored answer
    assert len(box_refine.read_asks(run_id)) == 1


def test_rebuild_without_the_leaf_gives_spark_boxes_and_no_new_key(run: dict[str, Any]) -> None:
    run_id, th = run["run_id"], run["th"]
    del th["dedup"]["sam_box_refine"]
    out = marks.rebuild_marks(run_id, "pickup", th=th)
    assert out[0].box_refine is None and "box_refine" not in paths.marks_json(run_id).read_text()
    assert run["calls"] == []
