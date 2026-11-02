"""Zoom refine of 11/1 (thresholds dedup.spark_zoom_refine): the window around a Spark box, the zoomed answer mapped back
to crop pixels, the move (Spark's box size, the nearest zoom box's centre, kept inside the crop), derive asks once and
SAM starts from the moved box, rebuild replays with no call, and the product code on the stored study answers gives
the study's numbers. No network."""

from __future__ import annotations

import asyncio
import json
import statistics as st
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from walkaround import box_refine, config, geometry, ledger, marks, paths, sam_client, spark_client, video, zoom_refine
from walkaround import contracts as c

TH = c.load_thresholds()
S = zoom_refine.settings(TH)
# the 11/1 study ran with the nearest zoom box; prefer_kind (11/2) is scored in docs/evidence/hero_1102/zoom_kind.py
STUDY_TH = {**TH, "dedup": {**TH["dedup"], "spark_zoom_refine": {k: v for k, v in TH["dedup"]["spark_zoom_refine"].items() if k != "prefer_kind"}}}
CROP = c.Crop(path="crops/pickup/pickup_hood_kf1.jpg", sha256="a" * 64, w=400, h=300, scale=0.5, dx=1000.0, dy=500.0)
EVIDENCE = Path(__file__).resolve().parent.parent / "docs" / "evidence" / "zoom_refine_1101"


def _mark(box: list[float], mid: str = "pickup_hood_kf1_m1", caps: list[str] | None = None, crop: c.Crop = CROP, kf_id: str = "pickup_hood_kf1") -> c.Mark:
    point = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
    return c.Mark(
        mark_id=mid, role="pickup", panel_id=c.PanelId.HOOD, kf_id=kf_id, source=c.MarkSource.SPARK_BOX,
        type=c.MarkKind.SCRATCH, confidence=c.Confidence.HIGH, size_class=c.SizeClass.S, bbox_crop=box, point_crop=point,
        bbox_src=geometry.box_crop_to_src(box, crop), point_src=geometry.crop_to_src(point, crop),
        size_px_src=max(box[2] - box[0], box[3] - box[1]) / crop.scale, caps=list(caps or []),
    )


def _kf(crop: c.Crop = CROP) -> c.Keyframe:
    return c.Keyframe(
        kf_id="pickup_hood_kf1", role="pickup", panel_id=c.PanelId.HOOD, track_id="pickup_hood_00", f_proxy=3, pts=0.2,
        score=c.KeyframeScore(area=1, completeness=1, sharpness=1, glare=0, frontality=1, total=1), aspect_ratio=1.5,
        box_src=[1000, 500, 1800, 1100], crop=crop, moments=None, front_sign=1, marks_status=c.MarksStatus.COMPLETE,
    )


def _marks_text(boxes: list[tuple[list[int], str]]) -> str:
    items = [{"type": t, "point": {"x": (b[0] + b[2]) // 2, "y": (b[1] + b[3]) // 2}, "bbox": {"x_min": b[0], "y_min": b[1], "x_max": b[2], "y_max": b[3]}, "size_class": "S", "confidence": "medium", "note": ""} for b, t in boxes]
    return json.dumps({"marks": items, "conditions": {"wet": False, "dirty": False, "glare": False, "dark": False}, "landmarks": [], "neighbour_car_visible": False})


def _record(text: str, rid: str = "resp_zoom_1", valid: bool = True, truncated: bool = False) -> c.SparkRequestRecord:
    return c.SparkRequestRecord(request_id=rid, purpose=c.SparkPurpose.MARKS, model="muse-spark-1.3", schema_name="marks_v1", raw_text=text, parsed=json.loads(text) if valid else None, valid=valid, truncated=truncated)


def _grid_of(box: list[float], win: list[int]) -> list[int]:
    """A crop-pixel box as the 0..1000 grid of the enlarged window (what Spark would answer for it)."""
    ww, wh = win[2] - win[0], win[3] - win[1]
    return [round((box[0] - win[0]) * 1000 / ww), round((box[1] - win[1]) * 1000 / wh), round((box[2] - win[0]) * 1000 / ww), round((box[3] - win[1]) * 1000 / wh)]


def _ask(m: c.Mark, win: list[int], record: c.SparkRequestRecord, crop: c.Crop = CROP) -> zoom_refine.ZoomAsk:
    return zoom_refine.ZoomAsk(kf_id=m.kf_id, role="pickup", mark_id=m.mark_id, panel_id=c.PanelId.HOOD, crop_path=crop.path, crop_sha256=crop.sha256, crop_w=crop.w, crop_h=crop.h, window=win, out_w=1280, out_h=720, record=record)


# Settings and the window ---------------------------------------------------------------------------------------------


def test_settings_are_off_when_the_leaf_is_absent() -> None:
    assert S is not None and S["min_px"] == 96 and (S["out_w"], S["out_h"]) == (1280, 720)
    old = json.loads(json.dumps(TH))
    del old["dedup"]["spark_zoom_refine"]
    assert zoom_refine.settings(old) is None
    m = _mark([100.0, 100.0, 141.0, 111.0])
    assert zoom_refine.refine_marks([m], _kf(), zoom_refine.Index([]), th=old) == [m]


def test_the_window_is_16_by_9_at_least_96_px_around_the_box_and_inside_the_crop() -> None:
    win = zoom_refine.window([100.0, 100.0, 141.0, 111.0], CROP, S)  # type: ignore[arg-type]
    assert win == [35, 58, 206, 154]  # short side 96, long side 96 x 16/9, centred on the box
    big = zoom_refine.window([150.0, 100.0, 300.0, 200.0], CROP, S)  # type: ignore[arg-type]
    assert big[3] - big[1] >= 100 + 2 * 28 - 1 and big[0] <= 150 and big[2] >= 300  # the box plus one patch a side
    edge = zoom_refine.window([0.0, 280.0, 20.0, 300.0], CROP, S)  # type: ignore[arg-type]
    assert edge[0] == 0 and edge[3] == 300 and edge[2] - edge[0] == 171  # shifted inside, size kept
    small = c.Crop(path="crops/pickup/small.jpg", w=126, h=168, scale=1.0, dx=0.0, dy=0.0)
    assert zoom_refine.window([40.0, 60.0, 70.0, 90.0], small, S) == [0, 27, 126, 123]  # type: ignore[arg-type]
    wide = c.Crop(path="crops/pickup/wide.jpg", w=2560, h=1440, scale=1.0, dx=0.0, dy=0.0)  # sent at 1280: a patch is 56 crop px
    w2 = zoom_refine.window([1000.0, 700.0, 1010.0, 710.0], wide, S)  # type: ignore[arg-type]
    assert w2[3] - w2[1] == 168  # 3 patches of 56


def test_the_zoom_image_is_the_window_enlarged_to_1280x720() -> None:
    rgb = np.zeros((300, 400, 3), np.uint8)
    rgb[58:154, 35:206] = 200
    out = zoom_refine.zoom_image(rgb, [35, 58, 206, 154], S)  # type: ignore[arg-type]
    assert out.shape == (720, 1280, 3) and int(out.min()) >= 190


# decide and apply ------------------------------------------------------------------------------------------------------


def test_decide_moves_sparks_box_to_the_nearest_zoom_box_centre_and_keeps_its_size() -> None:
    m = _mark([100.0, 100.0, 141.0, 111.0])
    win = zoom_refine.window(list(m.bbox_crop), CROP, S)  # type: ignore[arg-type]
    near, far = [140.0, 84.0, 150.0, 92.0], [40.0, 140.0, 50.0, 150.0]
    a = _ask(m, win, _record(_marks_text([(_grid_of(far, win), "dirt"), (_grid_of(near, win), "chip")])))
    r = zoom_refine.decide(m, a, win, CROP)
    assert r.used == "zoom" and r.zoom_boxes == 2 and r.zoom_type == c.MarkKind.CHIP and r.spark_zoom_request_id == "resp_zoom_1"
    assert r.zoom_bbox_crop is not None and max(abs(x - y) for x, y in zip(r.zoom_bbox_crop, near)) < 0.5
    mv = r.moved_bbox_crop
    assert mv is not None and abs((mv[2] - mv[0]) - 41) < 1e-6 and abs((mv[3] - mv[1]) - 11) < 1e-6
    assert abs((mv[0] + mv[2]) / 2 - 145) < 0.5 and abs((mv[1] + mv[3]) / 2 - 88) < 0.5
    out = zoom_refine.apply(m, r, CROP, TH)
    assert out.bbox_crop == mv and out.point_crop == [(mv[0] + mv[2]) / 2, (mv[1] + mv[3]) / 2] and out.type == c.MarkKind.SCRATCH
    assert out.bbox_src == geometry.box_crop_to_src(mv, CROP) and out.point_src == geometry.crop_to_src(out.point_crop, CROP)  # type: ignore[arg-type]
    assert out.zoom_refine is not None and out.zoom_refine.spark_bbox_crop == [100.0, 100.0, 141.0, 111.0] and out.zoom_refine.window_crop == win


def test_the_moved_box_is_shifted_back_inside_the_crop() -> None:
    m = _mark([10.0, 280.0, 50.0, 298.0])
    win = zoom_refine.window(list(m.bbox_crop), CROP, S)  # type: ignore[arg-type]
    a = _ask(m, win, _record(_marks_text([(_grid_of([0.0, 290.0, 6.0, 300.0], win), "chip")])))
    mv = zoom_refine.decide(m, a, win, CROP).moved_bbox_crop
    assert mv is not None and mv[0] == 0.0 and mv[3] == 300.0 and abs((mv[2] - mv[0]) - 40) < 1e-6 and abs((mv[3] - mv[1]) - 18) < 1e-6


def test_decide_keeps_sparks_box_and_says_why() -> None:
    m = _mark([100.0, 100.0, 141.0, 111.0])
    win = zoom_refine.window(list(m.bbox_crop), CROP, S)  # type: ignore[arg-type]
    none = zoom_refine.decide(m, None, win, CROP)
    assert none.used == "spark" and none.note == "no zoom answer is stored for this box"
    empty = zoom_refine.decide(m, _ask(m, win, _record(_marks_text([]))), win, CROP)
    assert empty.used == "spark" and empty.zoom_boxes == 0 and empty.note == "the zoomed look found no mark near the box"
    cut = zoom_refine.decide(m, _ask(m, win, _record('{"marks": [', valid=False, truncated=True)), win, CROP)
    assert cut.used == "spark" and cut.note == "the zoomed look was cut short or not valid"
    kept = zoom_refine.apply(m, empty, CROP, TH)
    assert kept.bbox_crop == m.bbox_crop and kept.point_crop == m.point_crop and kept.zoom_refine == empty


def test_apply_sets_k3_by_the_moved_boxs_source_size() -> None:
    small = c.Crop(path="crops/pickup/pickup_hood_kf1.jpg", sha256="a" * 64, w=400, h=300, scale=1.0, dx=0.0, dy=0.0)
    m = _mark([100.0, 100.0, 120.0, 108.0], caps=["K12"], crop=small)  # 20 source px: no K3
    win = zoom_refine.window(list(m.bbox_crop), small, S)  # type: ignore[arg-type]
    a = _ask(m, win, _record(_marks_text([(_grid_of([0.0, 100.0, 4.0, 104.0], win), "chip")])), crop=small)
    out = zoom_refine.apply(m, zoom_refine.decide(m, a, win, small), small, TH)
    assert out.size_px_src == 20.0 and out.caps == ["K12"]


def test_a_mark_without_zoom_refine_keeps_its_bytes() -> None:
    m = _mark([100.0, 100.0, 141.0, 111.0])
    assert "zoom_refine" not in json.loads(c.canonical_json(m))
    r = zoom_refine.decide(m, None, [35, 58, 206, 154], CROP)
    back = c.Mark.model_validate_json(c.canonical_json(m.model_copy(update={"zoom_refine": r})))
    assert back.zoom_refine == r


# Derive and rebuild ------------------------------------------------------------------------------------------------


@pytest.fixture
def run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setattr(paths, "RUNS_DIR", root)
    monkeypatch.setattr(config, "RUNS_DIR", root)
    monkeypatch.setattr(ledger, "LEDGER_PATH", root / "ledger.jsonl")
    run_id = "zm1"
    paths.ensure_run_layout(run_id)
    path = paths.crop_path(run_id, "pickup", "pickup_hood_kf1")
    digest = video.write_jpeg(np.full((300, 400, 3), 90, np.uint8), path, quality=90)
    crop = c.Crop(path=paths.rel(run_id, path), sha256=digest, w=400, h=300, scale=0.5, dx=1000.0, dy=500.0)
    kf = _kf(crop)
    c.write_json(paths.keyframes_json(run_id), [kf])
    main = _record(_marks_text([([250, 333, 350, 367], "scratch")]), rid="resp_main")  # crop box [100, 100, 141, 111]
    main = main.model_copy(update={"input_refs": [c.InputRef(path=crop.path, sha256=digest)]})
    win = zoom_refine.window([100.0, 100.0, 141.0, 111.0], crop, S)  # type: ignore[arg-type]
    zoom = _record(_marks_text([(_grid_of([140.0, 84.0, 150.0, 92.0], win), "chip")]))
    spark_calls: list[tuple[int, int]] = []

    def fake_find_marks(crop_rgb: np.ndarray, panel_id: c.PanelId, **kw: Any) -> spark_client.MarksResult:
        h, w = crop_rgb.shape[:2]
        spark_calls.append((w, h))
        rec = zoom if (w, h) == (1280, 720) else main
        return spark_client.marks_result_from_record(rec, orig_width=w, orig_height=h, landmarks=kw.get("landmarks") or [])

    sam_prompts: list[str] = []

    async def fake_segment_image(image: Any, concept: str, **kw: Any) -> sam_client.SegmentationRun:
        sam_prompts.append(concept)  # completed, no mask: SAM keeps the box it was given
        return sam_client.SegmentationRun(concept=concept, media="image", media_width=400, media_height=300, outcome="completed", response_id="resp_sam")

    monkeypatch.setattr(spark_client, "find_marks", fake_find_marks)
    monkeypatch.setattr(sam_client, "segment_image", fake_segment_image)
    return {"run_id": run_id, "kf": kf, "win": win, "spark_calls": spark_calls, "sam_prompts": sam_prompts}


def test_derive_asks_one_zoom_per_box_then_sam_on_the_moved_box_and_rebuild_replays(run: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    run_id, kf = run["run_id"], run["kf"]
    out, _kfs, _recs = asyncio.run(marks.derive_marks(run_id, "pickup", [kf], th=TH))
    assert run["spark_calls"] == [(400, 300), (1280, 720)] and len(run["sam_prompts"]) == 1
    m = out[0]
    assert m.zoom_refine is not None and m.zoom_refine.used == "zoom" and m.zoom_refine.spark_bbox_crop == [100.0, 100.0, 141.0, 111.0]
    moved = m.zoom_refine.moved_bbox_crop
    assert moved is not None and m.bbox_crop == moved and abs((moved[0] + moved[2]) / 2 - 145) < 0.5
    assert m.box_refine is not None and m.box_refine.used == "spark" and m.box_refine.spark_bbox_crop == moved  # SAM saw the moved box
    stored = zoom_refine.read_asks(run_id)
    assert len(stored) == 1 and stored[0].window == run["win"] and stored[0].mark_id == m.mark_id
    sam = box_refine.read_asks(run_id)
    assert sam[0].objects[0].prompt == box_refine.prompt_of(m, kf.crop, box_refine.settings(TH))  # type: ignore[arg-type]
    before = paths.marks_json(run_id).read_bytes()

    def no_spark(*a: Any, **k: Any) -> Any:
        raise AssertionError("rebuild must not call Spark")

    async def no_sam(*a: Any, **k: Any) -> Any:
        raise AssertionError("rebuild must not call SAM")

    monkeypatch.setattr(spark_client, "find_marks", no_spark)
    monkeypatch.setattr(sam_client, "segment_image", no_sam)
    marks.rebuild_marks(run_id, "pickup", th=TH)
    assert paths.marks_json(run_id).read_bytes() == before
    asyncio.run(marks.derive_marks(run_id, "pickup", [kf], th=TH))  # a resumed run reuses every stored answer
    assert len(zoom_refine.read_asks(run_id)) == 1 and paths.marks_json(run_id).read_bytes() == before


def test_without_the_leaf_derive_makes_no_zoom_call_and_writes_no_key(run: dict[str, Any]) -> None:
    th = json.loads(json.dumps(TH))
    del th["dedup"]["spark_zoom_refine"]
    out, _kfs, _recs = asyncio.run(marks.derive_marks(run["run_id"], "pickup", [run["kf"]], th=th))
    assert run["spark_calls"] == [(400, 300)] and out[0].zoom_refine is None
    assert not paths.zoom_marks_json(run["run_id"]).exists() and "zoom_refine" not in paths.marks_json(run["run_id"]).read_text()


# The study's numbers through the product code ----------------------------------------------------------------------


def _iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / union if union > 0 else 0.0


def test_the_product_code_gives_the_studys_numbers_on_the_stored_answers() -> None:
    """Sanat's 48 moved boxes of pairC_core3 and the 53 stored small-window answers (docs/evidence/zoom_refine_1101)."""
    study = json.loads((EVIDENCE / "study.json").read_text())
    answers = json.loads((EVIDENCE / "answers.json").read_text())
    ious, on, offsets, base = [], 0, [], []
    for row in study["rows"]:
        kf_id = row["mark_id"].rsplit("_m", 1)[0]
        crop = c.Crop(path=f"crops/{kf_id}.jpg", w=row["crop_wh"][0], h=row["crop_wh"][1], scale=1.0, dx=0.0, dy=0.0)
        m = _mark(row["spark"], mid=row["mark_id"], crop=crop, kf_id=kf_id)
        e = answers[f"{row['mark_id']}#w1"]
        win = zoom_refine.window(row["spark"], crop, S)  # type: ignore[arg-type]
        assert win == e["window"]
        a = zoom_refine.ZoomAsk(kf_id=kf_id, role="pickup", mark_id=m.mark_id, panel_id=c.PanelId(e["panel_id"]), crop_path=crop.path, crop_w=crop.w, crop_h=crop.h, window=win, out_w=1280, out_h=720, record=c.SparkRequestRecord.model_validate(e["record"]))
        kf = _kf(crop).model_copy(update={"kf_id": kf_id})
        out = zoom_refine.refine_marks([m], kf, zoom_refine.Index([a]), th=STUDY_TH)[0]
        if row["status"] != "moved":
            continue
        b, h = list(out.bbox_crop), row["hand"]  # type: ignore[arg-type]
        ious.append(_iou(b, h))
        base.append(_iou(row["spark"], h))
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        on += h[0] <= cx <= h[2] and h[1] <= cy <= h[3]
        offsets.append(((cx - (h[0] + h[2]) / 2) ** 2 + (cy - (h[1] + h[3]) / 2) ** 2) ** 0.5)
    assert len(ious) == 48
    assert round(st.median(base), 3) == 0.149 and round(st.median(ious), 3) == 0.353
    assert on == 34 and sum(i >= 0.5 for i in ious) == 12 and round(st.median(offsets), 1) == 11.0
