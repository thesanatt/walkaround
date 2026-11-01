"""ruler (M11, 5.12): tire codes, the plate and wheel rulers with tilt, mark extents from masks and Stays Put tracks,
the size arithmetic and its interval, every refusal, the Hertz threshold type and comparison, and the run-level
helpers that fill Verdict.size."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from walkaround import config
from walkaround import contracts as c
from walkaround import ledger, paths, privacy, ruler, sheet

TH = c.load_thresholds()
W = 1920  # the source frame width the camera distance is computed for


@pytest.fixture
def runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "runs"
    monkeypatch.setattr(paths, "RUNS_DIR", root)
    monkeypatch.setattr(config, "RUNS_DIR", root)
    monkeypatch.setattr(ledger, "LEDGER_PATH", root / "ledger.jsonl")
    return root


def blur(long_px: float | None = 240.0, hw: float | None = 0.5, c_: float | None = 0.9, box: list[float] | None = None) -> c.BlurRecord:
    box = box or [900.0, 700.0, 900.0 + (long_px or 0.0), 700.0 + (long_px or 0.0) * (hw or 0.5)]
    return c.BlurRecord(plate_boxes=[box] if long_px else [], plate_ruler_px=long_px, plate_tilt_hw=hw, plate_ruler_c=c_)


def ellipse_mask(major: int, minor: int, *, size: tuple[int, int] = (800, 600), center: tuple[int, int] = (400, 300), angle: float = 90.0) -> np.ndarray:
    mask = np.zeros((size[1], size[0]), np.uint8)
    cv2.ellipse(mask, center, (major // 2, minor // 2), angle, 0, 360, 1, -1)
    return mask.astype(bool)


def line_mask(length: int, angle_deg: float, *, thickness: int = 3, size: tuple[int, int] = (400, 400)) -> np.ndarray:
    mask = np.zeros((size[1], size[0]), np.uint8)
    cx, cy = size[0] / 2, size[1] / 2
    dx, dy = math.cos(math.radians(angle_deg)) * (length - 1) / 2, math.sin(math.radians(angle_deg)) * (length - 1) / 2
    cv2.line(mask, (round(cx - dx), round(cy - dy)), (round(cx + dx), round(cy + dy)), 1, thickness)
    return mask.astype(bool)


def extent(long_px: float, *, axis: float = 0.0, center: tuple[float, float] = (1000.0, 760.0), **kw) -> ruler.MarkExtent:
    return ruler.MarkExtent(long_px=long_px, axis_deg=axis, center=center, **kw)


# Tire codes -----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "code, norm",
    [
        ("225/60R16", "225/60R16"),
        ("P225/60 R16 98H", "225/60R16"),
        ("lt265/70r17", "265/70R17"),
        ("205/55ZR16", "205/55R16"),
        (" 195/65 R 15 ", "195/65R15"),
        ("245/40R18XL", "245/40R18"),
        ("LT245/75R16 120/116S", "245/75R16"),
    ],
)
def test_tire_codes_normalise_to_the_spelling_the_sheet_reads(code: str, norm: str) -> None:
    assert ruler.normalize_tire_code(code) == norm
    assert ruler.tire_diameter_in(code) == pytest.approx(sheet.tire_diameter_in(norm))


def test_tire_diameter_formula_and_unknown_codes() -> None:
    assert ruler.tire_diameter_in("225/60R16") == pytest.approx(225 * 60 / 1270 + 16)  # 26.6 in, the size line's example
    assert ruler.tire_diameter_in("195/55R16") == pytest.approx(24.44, abs=0.01)  # the research's worked example
    for bad in (None, "", "R16", "225-60-16", "wheel", "225/60R165"):
        assert ruler.tire_diameter_in(bad) is None


# Plate ruler ----------------------------------------------------------------------------------------------------------


def test_tilt_from_aspect_matches_privacy_plate_extent() -> None:
    for deg in (0.0, 15.0, 25.0, 29.0):
        cos = math.cos(math.radians(deg))
        mask = np.zeros((400, 600), bool)
        mask[100 : 100 + 120, 50 : 50 + round(240 * cos)] = True
        long_px, hw, tilt = privacy.plate_extent(mask)
        assert ruler.tilt_from_aspect(hw) == pytest.approx(tilt)
        assert ruler.tilt_from_aspect(hw) == pytest.approx(deg, abs=0.8)
    assert ruler.tilt_from_aspect(0.4) == 0.0 and ruler.tilt_from_aspect(None) == 0.0


def test_plate_ruler_ppi_frontal_and_turned() -> None:
    frontal = ruler.plate_ruler(blur(240.0, 0.5), th=TH, source_kf="kf1")
    assert isinstance(frontal, ruler.Ruler)
    assert frontal.ppi == pytest.approx(20.0) and frontal.tilt_deg == 0.0 and frontal.source_kf == "kf1"
    hw = 0.5 / math.cos(math.radians(25))
    turned = ruler.plate_ruler(blur(240.0 * math.cos(math.radians(25)), hw), th=TH)
    assert isinstance(turned, ruler.Ruler)
    assert turned.tilt_deg == pytest.approx(25.0) and turned.ppi == pytest.approx(20.0)  # the scale across the view
    assert turned.tilt_axis_deg == 0.0


@pytest.mark.parametrize(
    "record, words",
    [
        (c.BlurRecord(), "no license plate"),
        (blur(240.0, 0.5, 0.3), "under c 0.5"),
        (blur(50.0, 0.5), "under 60 px"),
        (blur(240.0, 0.5 / math.cos(math.radians(35))), "more than 30 degrees"),
    ],
)
def test_plate_ruler_refusals(record: c.BlurRecord, words: str) -> None:
    out = ruler.plate_ruler(record, th=TH)
    assert isinstance(out, str) and words in out


# Wheel ruler ----------------------------------------------------------------------------------------------------------


def test_wheel_ruler_from_mask_with_a_typed_code() -> None:
    mask = ellipse_mask(300, 300)
    w = ruler.wheel_ruler(mask, 0.9, touches_border=False, tire_code="225/60R16", th=TH, source_kf="kf_w")
    assert isinstance(w, ruler.Ruler)
    diameter = ruler.tire_diameter_in("225/60R16")
    assert w.ppi == pytest.approx(300 / diameter, rel=0.01)
    assert w.tilt_deg < 6 and w.spread == 0.0 and w.tire_code == "225/60R16"


def test_wheel_ruler_tilt_and_its_axis() -> None:
    minor = round(300 * math.cos(math.radians(20)))
    w = ruler.wheel_ruler(ellipse_mask(300, minor, angle=90.0), 0.9, touches_border=False, tire_code="225/60R16", th=TH)
    assert isinstance(w, ruler.Ruler)
    assert w.tilt_deg == pytest.approx(20.0, abs=2.0)
    assert min(abs(w.tilt_axis_deg - 0.0), abs(w.tilt_axis_deg - 180.0)) < 2.0  # an upright wheel narrows sideways


def test_wheel_ruler_refuses_without_a_readable_code() -> None:
    for code, reason in ((None, ruler.TIRE_NOT_GIVEN), ("  ", ruler.TIRE_NOT_GIVEN), ("not a code", ruler.TIRE_UNREADABLE)):
        assert ruler.wheel_ruler(ellipse_mask(300, 300), 0.9, touches_border=False, tire_code=code, th=TH) == reason
    assert ruler.wheel_ruler_from_box([100.0, 100.0, 400.0, 400.0], 0.9, touches_border=False, tire_code=None, th=TH) == ruler.TIRE_NOT_GIVEN
    # a code would not fix a wheel at the frame edge, so that reason comes first
    assert ruler.wheel_ruler(ellipse_mask(300, 300), 0.9, touches_border=True, tire_code=None, th=TH) == "the wheel touches the frame edge"
    assert sheet.size_line(c.SizeEstimate(refused_reason=ruler.TIRE_NOT_GIVEN)) == "size not stated: tire size not given"  # section 14 item 21


@pytest.mark.parametrize(
    "kw, words",
    [
        (dict(mask=ellipse_mask(300, 300), c=0.9, touches_border=True), "frame edge"),
        (dict(mask=ellipse_mask(300, 300), c=0.3, touches_border=False), "under c 0.5"),
        (dict(mask=ellipse_mask(100, 100), c=0.9, touches_border=False), "under 120 px"),
        (dict(mask=ellipse_mask(300, 200), c=0.9, touches_border=False), "more than 30 degrees"),
    ],
)
def test_wheel_ruler_refusals(kw: dict, words: str) -> None:
    out = ruler.wheel_ruler(kw["mask"], kw["c"], touches_border=kw["touches_border"], tire_code="225/60R16", th=TH)
    assert isinstance(out, str) and words in out


def test_wheel_ruler_from_a_track_box() -> None:
    cos = math.cos(math.radians(20))
    w = ruler.wheel_ruler_from_box([100.0, 100.0, 100.0 + 400 * cos, 500.0], 0.8, touches_border=False, tire_code="225/60R16", th=TH, quantization_px=2.0)
    assert isinstance(w, ruler.Ruler)
    assert w.ppi == pytest.approx(400 / ruler.tire_diameter_in("225/60R16"))
    assert w.tilt_deg == pytest.approx(20.0, abs=0.01) and w.tilt_axis_deg == 0.0
    assert w.spread == pytest.approx(2 * 2.0 / 400)


# Mark extents ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("angle", [0.0, 30.0, 90.0, 135.0])
def test_extent_from_mask_measures_along_the_principal_axis(angle: float) -> None:
    ex = ruler.extent_from_mask(line_mask(120, angle, thickness=1))
    assert ex is not None and ex.source == "mask"
    assert ex.long_px == pytest.approx(120, abs=2.5)
    gap = abs(ex.axis_deg - angle) % 180
    assert min(gap, 180 - gap) < 2.0


def test_the_field_of_view_comes_from_thresholds_when_they_carry_it() -> None:
    import copy

    t = math.radians(20)
    plate = ruler.plate_ruler(blur(240.0 * math.cos(t), 0.5 / math.cos(t), box=[900.0, 700.0, 900.0 + 240 * math.cos(t), 820.0]), th=TH)
    far = (plate.center[0] + 10 * plate.ppi, plate.center[1])
    th = copy.deepcopy(TH)
    th["ruler"]["camera_hfov_deg"] = 50.0
    a = ruler.size_estimate(extent(40.0, axis=90.0, center=far), plate, mark_type=c.MarkKind.DENT, panel_id=c.PanelId.REAR_BUMPER, frame_w=W, th=TH)
    b = ruler.size_estimate(extent(40.0, axis=90.0, center=far), plate, mark_type=c.MarkKind.DENT, panel_id=c.PanelId.REAR_BUMPER, frame_w=W, th=th)
    depth = 10 * math.sin(t) / ruler.camera_distance_in(plate.ppi, W, 50.0) - 10 * math.sin(t) / ruler.camera_distance_in(plate.ppi, W)
    assert (b.bound_hi - a.bound_hi) / a.inches == pytest.approx(depth, rel=1e-6)


def test_extent_from_mask_maps_crop_pixels_to_source() -> None:
    ex = ruler.extent_from_mask(line_mask(100, 0.0, thickness=1), scale=0.5, offset=(1000.0, 500.0))
    assert ex.long_px == pytest.approx(200, abs=3)
    assert ex.center == pytest.approx((1000 + 200 / 0.5, 500 + 200 / 0.5), abs=2)
    assert ruler.extent_from_mask(np.zeros((10, 10), bool)) is None


def track(boxes: list[list[float]], *, window: list[float] | None = None, verdict=c.StaysPutVerdict.FIXED) -> c.MarkTrack:
    window = window or [500.0, 400.0, 1508.0 * 1.0, 1408.0]  # 1008 px window: one source px per clip px
    clip = c.ClipRecord(path="clips/return/m.mp4", f0_src=0, frames=len(boxes), window_px=1008, windows_src=[list(window)] * len(boxes))
    return c.MarkTrack(mark_id="m", role="return", clip=clip, track=[c.TrackPoint(f=i, box_clip=b) for i, b in enumerate(boxes)], verdict=verdict)


def test_extent_from_track_takes_the_median_box_in_source_pixels() -> None:
    boxes = [[500.0, 500.0, 530.0 + d, 510.0] for d in (0, 1, 2, -1, -2, 0, 1)]
    ex = ruler.extent_from_track(track(boxes))
    assert ex is not None and ex.source == "track"
    assert ex.long_px == pytest.approx(30.0) and ex.long_px_hi == pytest.approx(math.hypot(30, 10))
    assert ex.axis_deg == 0.0 and ex.center == pytest.approx((500 + 515, 400 + 505))
    half = ruler.extent_from_track(track(boxes, window=[0.0, 0.0, 2016.0, 2016.0]))  # 2 source px per clip px
    assert half.long_px == pytest.approx(60.0)


def test_extent_from_track_refuses_short_or_failed_tracks() -> None:
    assert ruler.extent_from_track(None) is None
    assert ruler.extent_from_track(track([[0.0, 0.0, 30.0, 10.0]] * 3)) is None
    assert ruler.extent_from_track(track([[0.0, 0.0, 30.0, 10.0]] * 9, verdict=c.StaysPutVerdict.NO_TRACK)) is None


# The size and its interval --------------------------------------------------------------------------------------------


def frontal_plate() -> ruler.Ruler:
    out = ruler.plate_ruler(blur(240.0, 0.5, box=[900.0, 700.0, 1140.0, 820.0]), th=TH, source_kf="kf1")
    assert isinstance(out, ruler.Ruler)
    return out


def test_size_from_a_frontal_plate() -> None:
    s = ruler.size_estimate(extent(30.0), frontal_plate(), mark_type=c.MarkKind.SCRATCH, panel_id=c.PanelId.REAR_BUMPER, frame_w=W, th=TH)
    rel = 2 * 2 / 30  # 2 px per side over a 30 px mark; no tilt, so no foreshortening or depth term
    assert s.inches == pytest.approx(1.5)
    assert s.bound_lo == pytest.approx(1.5 * (1 - rel)) and s.bound_hi == pytest.approx(1.5 * (1 + rel))
    assert s.ruler == c.RulerKind.PLATE and s.ruler_source_kf == "kf1" and s.threshold_type == c.ThresholdType.BODY_SCRATCH
    assert sheet.size_line(s) == "about 1.5 in, between 1.3 and 1.7 in (ruler: the license plate, 12 in)"


def test_size_corrects_a_mark_along_the_turned_axis_only() -> None:
    t = math.radians(25)
    plate = ruler.plate_ruler(blur(240.0 * math.cos(t), 0.5 / math.cos(t), box=[900.0, 700.0, 900.0 + 240 * math.cos(t), 820.0]), th=TH)
    along = ruler.size_estimate(extent(30.0 * math.cos(t), axis=10.0, center=plate.center), plate, mark_type=c.MarkKind.SCRATCH, panel_id=c.PanelId.TRUNK_LID, frame_w=W, th=TH)
    across = ruler.size_estimate(extent(30.0, axis=90.0, center=plate.center), plate, mark_type=c.MarkKind.SCRATCH, panel_id=c.PanelId.TRUNK_LID, frame_w=W, th=TH)
    assert along.inches == pytest.approx(1.5)  # 30 px of a 1.5 in mark shrink by cos(25); the correction restores it
    assert across.inches == pytest.approx(1.5)  # a vertical mark is not foreshortened by a yaw
    rel_along = (1 - math.cos(t)) + 2 * 2 / (30.0 * math.cos(t))
    assert along.bound_hi == pytest.approx(1.5 * (1 + rel_along))


def test_depth_term_grows_with_the_distance_from_the_ruler() -> None:
    t = math.radians(20)
    plate = ruler.plate_ruler(blur(240.0 * math.cos(t), 0.5 / math.cos(t), box=[900.0, 700.0, 900.0 + 240 * math.cos(t), 820.0]), th=TH)
    near = ruler.size_estimate(extent(40.0, axis=90.0, center=plate.center), plate, mark_type=c.MarkKind.DENT, panel_id=c.PanelId.REAR_BUMPER, frame_w=W, th=TH)
    far_center = (plate.center[0] + 10 * plate.ppi, plate.center[1])  # 10 in away
    far = ruler.size_estimate(extent(40.0, axis=90.0, center=far_center), plate, mark_type=c.MarkKind.DENT, panel_id=c.PanelId.REAR_BUMPER, frame_w=W, th=TH)
    depth = 10 * math.sin(t) / ruler.camera_distance_in(plate.ppi, W)
    assert (far.bound_hi - near.bound_hi) / near.inches == pytest.approx(depth, rel=1e-6)
    assert ruler.camera_distance_in(20.0, W) == pytest.approx((W / 2) / math.tan(math.radians(35)) / 20.0)


def test_a_track_box_widens_the_upper_end_to_its_diagonal() -> None:
    s = ruler.size_estimate(extent(30.0, long_px_hi=40.0, source="track"), frontal_plate(), mark_type=c.MarkKind.SCRATCH, panel_id=c.PanelId.REAR_BUMPER, frame_w=W, th=TH)
    assert s.inches == pytest.approx(1.5) and s.bound_hi == pytest.approx(2.0 * (1 + 4 / 30))


def test_a_typed_code_sizes_with_no_assumed_spread() -> None:
    mask = ellipse_mask(300, 300, size=(2000, 1200), center=(1000, 760))
    typed = ruler.wheel_ruler(mask, 0.9, touches_border=False, tire_code="235/65R17", th=TH)
    assert isinstance(typed, ruler.Ruler) and typed.spread == 0.0 and typed.tire_code == "235/65R17"
    a = ruler.size_estimate(extent(25.0, center=(1000.0, 700.0)), typed, mark_type=c.MarkKind.SCUFF, panel_id=c.PanelId.LEFT_REAR_WHEEL, frame_w=W, th=TH)
    assert a.tire_code == "235/65R17" and a.ruler == c.RulerKind.WHEEL
    assert sheet.size_line(a).endswith(f"(ruler: the wheel, {ruler.tire_diameter_in('235/65R17'):.1f} in from 235/65R17, typed by you)")


@pytest.mark.parametrize(
    "kw, reason",
    [
        (dict(crop_flag=True), ruler.CROP_FLAG),
        (dict(drawn_box_overlap=True), ruler.DRAWN_BOX),
        (dict(ex=None), ruler.NO_MARK_MASK),
        (dict(ex=extent(30.0, source="spark_box")), ruler.NO_MARK_MASK),
        (dict(ex=extent(30.0, proxy=True)), ruler.PROXY_FRAME),
        (dict(ex=extent(6.0)), "the mark is under 8 px along its length"),
        (dict(rl=None), "no ruler within 0.6 m of this mark"),
        (dict(rl="the license plate is under 60 px"), "the license plate is under 60 px"),
        (dict(panel=c.PanelId.LEFT_FRONT_DOOR), "no ruler within 0.6 m of this mark"),
        (dict(ex=extent(30.0, center=(1000.0 + 30 * 20.0, 760.0))), "no ruler within 0.6 m of this mark"),
        (dict(frame_w=None), ruler.NO_CAMERA_DISTANCE),
    ],
)
def test_every_refusal_of_5_12(kw: dict, reason: str) -> None:
    ex = kw.get("ex", extent(30.0))
    rl = kw.get("rl", frontal_plate())
    s = ruler.size_estimate(ex, rl, mark_type=c.MarkKind.SCRATCH, panel_id=kw.get("panel", c.PanelId.REAR_BUMPER), frame_w=kw.get("frame_w", W), th=TH, crop_flag=kw.get("crop_flag", False), drawn_box_overlap=kw.get("drawn_box_overlap", False))
    assert s.inches is None and s.refused_reason == reason
    assert sheet.size_line(s) == f"size not stated: {reason}"


def test_the_no_ruler_reason_is_the_sheets_line() -> None:
    assert f"size not stated: {ruler.no_ruler_reason(TH)}" == sheet.NO_RULER_LINE


def test_best_size_takes_the_first_ruler_that_sizes_and_keeps_a_telling_refusal() -> None:
    tilted = ruler.plate_ruler(blur(240.0, 0.5 / math.cos(math.radians(40))), th=TH)
    wheel = ruler.wheel_ruler(ellipse_mask(300, 300, size=(2000, 1200), center=(1000, 760)), 0.9, touches_border=False, tire_code="225/60R16", th=TH)
    s = ruler.best_size(extent(30.0, center=(1000.0, 700.0)), [tilted, wheel], mark_type=c.MarkKind.SCRATCH, panel_id=c.PanelId.LEFT_REAR_WHEEL, frame_w=W, th=TH)
    assert s.ruler == c.RulerKind.WHEEL and s.inches is not None
    r = ruler.best_size(extent(30.0), [tilted], mark_type=c.MarkKind.SCRATCH, panel_id=c.PanelId.REAR_BUMPER, frame_w=W, th=TH)
    assert "more than 30 degrees" in r.refused_reason
    assert ruler.best_size(extent(30.0), [], mark_type=None, panel_id=c.PanelId.HOOD, frame_w=W, th=TH).refused_reason == ruler.no_ruler_reason(TH)


@pytest.mark.parametrize(
    "ex, rulers, panel, kw, reason",
    [
        # no ruler qualifies: the mark's extent is not consulted, so a mask never asked for is never blamed
        (None, [ruler.NO_PLATE], c.PanelId.TRUNK_LID, {}, "no ruler within 0.6 m of this mark"),
        (None, ["the license plate is under 60 px"], c.PanelId.REAR_BUMPER, {}, "the license plate is under 60 px"),
        (None, [ruler.NO_PLATE, ruler.TIRE_NOT_GIVEN], c.PanelId.LEFT_REAR_FENDER, {}, ruler.TIRE_NOT_GIVEN),
        (None, ["the wheel touches the frame edge", ruler.TIRE_UNREADABLE], c.PanelId.FRONT_BUMPER, {}, ruler.TIRE_UNREADABLE),
        # a plate never sizes a door, whatever the plate's own state
        (extent(30.0), ["frontal plate"], c.PanelId.LEFT_FRONT_DOOR, {}, "no ruler within 0.6 m of this mark"),
        (extent(30.0), [ruler.NO_PLATE], c.PanelId.LEFT_FRONT_DOOR, {}, "no ruler within 0.6 m of this mark"),
        # with a ruler that qualifies, the mark side speaks
        (None, ["frontal plate"], c.PanelId.REAR_BUMPER, {}, ruler.NO_MARK_MASK),
        (extent(30.0), [ruler.NO_PLATE], c.PanelId.REAR_BUMPER, {"crop_flag": True}, ruler.CROP_FLAG),
        (extent(30.0), ["frontal plate"], c.PanelId.REAR_BUMPER, {"drawn_box_overlap": True}, ruler.DRAWN_BOX),
    ],
)
def test_best_size_names_the_reason_that_applies(ex, rulers, panel, kw, reason) -> None:
    rulers = [frontal_plate() if r == "frontal plate" else r for r in rulers]
    s = ruler.best_size(ex, rulers, mark_type=c.MarkKind.SCRATCH, panel_id=panel, frame_w=W, th=TH, **kw)
    assert s.inches is None and s.refused_reason == reason


# Hertz ----------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mark, panel, expected",
    [
        (c.MarkKind.SCRATCH, c.PanelId.LEFT_FRONT_DOOR, c.ThresholdType.BODY_SCRATCH),
        (c.MarkKind.DENT, c.PanelId.HOOD, c.ThresholdType.DENT),
        (c.MarkKind.SCUFF, c.PanelId.REAR_BUMPER, c.ThresholdType.BODY_SCUFF),
        (c.MarkKind.SCUFF, c.PanelId.LEFT_FRONT_WHEEL, c.ThresholdType.WHEEL_MARK),
        (c.MarkKind.CHIP, c.PanelId.RIGHT_REAR_WHEEL, None),  # Hertz names scratches and scuffs on a wheel
        (c.MarkKind.DENT, c.PanelId.RIGHT_REAR_WHEEL, None),
        (c.MarkKind.CHIP, c.PanelId.WINDSHIELD, None),
        (c.MarkKind.CHIP, c.PanelId.TRUNK_LID, None),
        (None, c.PanelId.HOOD, None),
    ],
)
def test_threshold_type_per_mark_and_panel(mark, panel, expected) -> None:
    assert ruler.threshold_type_for(mark, panel) == expected


def sized(inches: float, lo: float, hi: float, tt: c.ThresholdType, **kw) -> c.SizeEstimate:
    return c.SizeEstimate(inches=inches, bound_lo=lo, bound_hi=hi, ruler=c.RulerKind.PLATE, threshold_type=tt, **kw)


def test_compare_hertz_agrees_with_the_sheet_sentence() -> None:
    cases = [
        (sized(0.5, 0.4, 0.6, c.ThresholdType.DENT), "under", "under Hertz's example threshold of 1 in for a dent (REV 7.2025, illustrative only)"),
        (sized(4.4, 3.8, 5.0, c.ThresholdType.BODY_SCRATCH), "over", "over Hertz's example threshold of 3.5 in for a body scratch along its long axis (REV 7.2025, illustrative only)"),
        (
            sized(0.9, 0.6, 1.3, c.ThresholdType.WHEEL_MARK),
            "straddles",
            "threshold not compared: the size bound straddles 1 in, Hertz's example threshold of 1 in for a wheel scratch or scuff (REV 7.2025, illustrative only)",
        ),
    ]
    for size, verdict, line in cases:
        assert ruler.compare_hertz(size, TH) == verdict
        assert sheet.threshold_line(size, TH) == line
    scuff = sized(2.0, 1.5, 2.5, c.ThresholdType.BODY_SCUFF)
    assert ruler.compare_hertz(scuff, TH) is None and sheet.threshold_line(scuff, TH) is None
    assert sheet.ordinary_wear_line(c.MarkKind.SCUFF, c.PanelId.REAR_BUMPER) == (
        "Hertz's page lists scuffs that do not break paint as ordinary wear (REV 7.2025, illustrative only; page saved 2026-10-26)."
    )
    assert ruler.compare_hertz(c.SizeEstimate(refused_reason="x"), TH) is None and ruler.compare_hertz(None, TH) is None


def test_a_measured_wheel_scuff_straddling_one_inch_is_not_compared() -> None:
    mask = ellipse_mask(300, 300, size=(2000, 1200), center=(1000, 760))
    wheel = ruler.wheel_ruler(mask, 0.9, touches_border=False, tire_code="225/60R16", th=TH)
    ppi = wheel.ppi
    s = ruler.size_estimate(extent(0.95 * ppi, center=(1000.0, 700.0)), wheel, mark_type=c.MarkKind.SCUFF, panel_id=c.PanelId.LEFT_REAR_WHEEL, frame_w=W, th=TH)
    assert s.bound_lo < 1.0 < s.bound_hi and ruler.compare_hertz(s, TH) == "straddles"
    assert sheet.threshold_line(s, TH).startswith("threshold not compared")


# A run's marks and verdicts ------------------------------------------------------------------------------------------


def write_run(run_id: str) -> None:
    c.write_json(paths.run_json(run_id), c.RunManifest(run_id=run_id, created_at="2026-10-27T15:00:00Z"))
    probe = c.FfprobeInfo(codec="h264", container="mp4", width=W, height=1080, display_width=W, display_height=1080, fps_avg=30.0, duration_s=20.0)
    proxy = c.ProxyRecord(path="media/return.proxy.mp4", fps=15.0, width=960, height=540, frames=300, pts_map_path="media/return.pts.json")
    c.write_json(paths.video_json(run_id, "return"), c.VideoRecord(role="return", original_filename="r.mp4", bytes=1, ffprobe=probe, proxy=proxy))
    score = c.KeyframeScore(area=1, completeness=1, sharpness=1, glare=0, frontality=1, total=1)
    kfs = [
        c.Keyframe(kf_id="return_rear_bumper_kf0", role="return", panel_id=c.PanelId.REAR_BUMPER, track_id="t1", f_proxy=40, pts=13.3, score=score, aspect_ratio=2.0, box_src=[600.0, 600.0, 1600.0, 1000.0], blur=blur(240.0, 0.5, box=[900.0, 700.0, 1140.0, 820.0])),
        c.Keyframe(kf_id="return_left_rear_wheel_kf0", role="return", panel_id=c.PanelId.LEFT_REAR_WHEEL, track_id="t2", f_proxy=60, pts=8.0, score=score, aspect_ratio=1.0, box_src=[500.0, 500.0, 900.0, 900.0]),
    ]
    c.write_json(paths.keyframes_json(run_id), kfs)
    marks = [
        c.Mark(mark_id="return_rear_bumper_kf0_m1", role="return", panel_id=c.PanelId.REAR_BUMPER, kf_id="return_rear_bumper_kf0", source=c.MarkSource.SPARK_BOX, type=c.MarkKind.SCRATCH, confidence=c.Confidence.HIGH, size_class=c.SizeClass.S, size_px_src=55.0),
        c.Mark(mark_id="return_rear_bumper_kf0_m2", role="return", panel_id=c.PanelId.REAR_BUMPER, kf_id="return_rear_bumper_kf0", source=c.MarkSource.SPARK_BOX, type=c.MarkKind.CHIP, confidence=c.Confidence.HIGH, size_class=c.SizeClass.S),
        c.Mark(mark_id="return_left_rear_wheel_kf0_m1", role="return", panel_id=c.PanelId.LEFT_REAR_WHEEL, kf_id="return_left_rear_wheel_kf0", source=c.MarkSource.SPARK_BOX, type=c.MarkKind.SCUFF, confidence=c.Confidence.HIGH, size_class=c.SizeClass.S),
        c.Mark(mark_id="pickup_m9", role="pickup", panel_id=c.PanelId.REAR_BUMPER, kf_id="missing_kf", source=c.MarkSource.SPARK_BOX, type=c.MarkKind.SCRATCH, confidence=c.Confidence.HIGH, size_class=c.SizeClass.S),
    ]
    c.write_json(paths.marks_json(run_id), marks)
    mt = track([[480.0, 350.0, 510.0, 356.0]] * 7, window=[500.0, 400.0, 1508.0, 1408.0])  # source centre (995, 753)
    mt = mt.model_copy(update={"mark_id": "return_rear_bumper_kf0_m1"})
    wheel_mt = track([[450.0, 250.0, 470.0, 256.0]] * 7, window=[200.0, 200.0, 1208.0, 1208.0])
    wheel_mt = wheel_mt.model_copy(update={"mark_id": "return_left_rear_wheel_kf0_m1"})
    c.write_json(paths.mark_tracks_json(run_id), [mt, wheel_mt])
    wheel_track = c.PanelTrack(
        track_id="w1", role="return", noun=c.Noun.WHEEL, sam_object_ids=["0"], fragments=[c.TrackFragment(object_id="0", first_frame=60, last_frame=60)],
        per_frame=[c.FrameObs(frame=60, pts=8.0, box=[300.0, 200.0, 450.0, 350.0], box_area_frac=0.04, in_car_box=True, c=0.9)],
    )
    c.write_json(paths.tracks_json(run_id), [wheel_track])


def test_size_marks_reads_the_keyframe_plate_and_the_stays_put_track(runs: Path) -> None:
    write_run("r1")
    sizes = ruler.size_marks("r1", th=TH, tire_code="225/60R16")
    s = sizes["return_rear_bumper_kf0_m1"]
    assert s.ruler == c.RulerKind.PLATE and s.inches == pytest.approx(30.0 / 20.0) and s.ruler_source_kf == "return_rear_bumper_kf0"
    assert sizes["return_rear_bumper_kf0_m2"].refused_reason == ruler.NO_MARK_MASK  # no track: never the Spark box
    assert sizes["pickup_m9"].refused_reason == "the mark's keyframe is not recorded"
    w = sizes["return_left_rear_wheel_kf0_m1"]
    assert w.ruler == c.RulerKind.WHEEL and w.tire_code == "225/60R16"  # the wheel track box on the same proxy frame
    assert w.inches == pytest.approx(20.0 / (300.0 / ruler.tire_diameter_in("225/60R16")))
    assert ruler.size_marks("r1", th=TH)["return_left_rear_wheel_kf0_m1"].refused_reason == ruler.TIRE_NOT_GIVEN  # no run-level code


def test_a_door_mark_never_takes_the_plates_reason(runs: Path) -> None:
    write_run("r3")
    score = c.KeyframeScore(area=1, completeness=1, sharpness=1, glare=0, frontality=1, total=1)
    kfs = c.read_json_list(paths.keyframes_json("r3"), c.Keyframe)
    door = c.Keyframe(kf_id="return_left_front_door_kf0", role="return", panel_id=c.PanelId.LEFT_FRONT_DOOR, track_id="t3", f_proxy=20, pts=4.0, score=score, aspect_ratio=1.5, box_src=[0.0, 0.0, 900.0, 600.0], blur=blur(240.0, 0.5, box=[900.0, 700.0, 1140.0, 820.0]))
    c.write_json(paths.keyframes_json("r3"), [*kfs, door])
    marks = c.read_json_list(paths.marks_json("r3"), c.Mark)
    mark = c.Mark(mark_id="return_left_front_door_kf0_m1", role="return", panel_id=c.PanelId.LEFT_FRONT_DOOR, kf_id=door.kf_id, source=c.MarkSource.SPARK_BOX, type=c.MarkKind.SCRATCH, confidence=c.Confidence.HIGH, size_class=c.SizeClass.S)
    c.write_json(paths.marks_json("r3"), [*marks, mark])
    assert ruler.size_marks("r3", th=TH)[mark.mark_id].refused_reason == ruler.no_ruler_reason(TH)


def test_size_verdicts_writes_a_size_on_each_mark_verdict(runs: Path) -> None:
    write_run("r2")
    verdicts = [
        c.Verdict(verdict_id="rear_bumper_v1", panel_id=c.PanelId.REAR_BUMPER, label=c.VerdictLabel.ALREADY_THERE, rule_id="L3", mark_ids=["pickup_m9", "return_rear_bumper_kf0_m1"]),
        c.Verdict(verdict_id="rear_bumper_v2", panel_id=c.PanelId.REAR_BUMPER, label=c.VerdictLabel.NEW_OR_UNSEEN, rule_id="L2", mark_ids=["return_rear_bumper_kf0_m2"]),
        c.Verdict(verdict_id="hood_v1", panel_id=c.PanelId.HOOD, label=c.VerdictLabel.NOT_COMPARABLE, rule_id="L6", mark_ids=[]),
    ]
    c.write_json(paths.verdicts_json("r2"), verdicts)
    out = ruler.size_verdicts("r2", th=TH)
    assert out[0].size is not None and out[0].size.ruler == c.RulerKind.PLATE  # the return mark of the pair
    assert out[1].size.refused_reason == ruler.NO_MARK_MASK
    assert out[2].size is None
    again = c.read_json_list(paths.verdicts_json("r2"), c.Verdict)
    assert c.canonical_json(again) == c.canonical_json(out)
    first = paths.verdicts_json("r2").read_bytes()
    c.write_json(paths.verdicts_json("r2"), verdicts)
    ruler.size_verdicts("r2", th=TH)
    assert paths.verdicts_json("r2").read_bytes() == first  # the same bytes on a rebuild


# The most frontal view, the run's tire code and the notes on the other ruler (polish lane, Oct 31) ----------------------


def test_a_strip_plate_is_not_the_12_in_ruler() -> None:
    strip = ruler.plate_ruler(blur(337.0, 0.12), th=TH)  # the pair C front plate on pairC_core1: 337 px, h / w 0.12
    assert strip == ruler.NOT_A_US_PLATE
    seen_from_above = ruler.plate_ruler(blur(240.0, 0.5 * math.cos(math.radians(40))), th=TH)  # a US plate from 40 degrees up
    assert isinstance(seen_from_above, ruler.Ruler) and seen_from_above.tilt_deg == 0.0
    old = {**TH, "ruler": {k: v for k, v in TH["ruler"].items() if k != "plate_hw_min"}}  # a run frozen before the key
    assert ruler.plate_ruler(blur(337.0, 0.12), th=old) == ruler.NOT_A_US_PLATE  # the module's 0.35 stands in
    assert TH["ruler"]["plate_hw_min"] == ruler.PLATE_HW_MIN


def turned_plate(deg: float, x: float = 900.0) -> c.BlurRecord:
    cos = math.cos(math.radians(deg))
    return blur(240.0 * cos, 0.5 / cos, box=[x, 700.0, x + 240.0 * cos, 820.0])


def write_clip_run(run_id: str, *, with_plan: bool = True) -> str:
    """A return walk: the trunk-lid mark's keyframe shows the plate turned 40 degrees, the rear bumper keyframe half a
    second later inside the mark's clip shows it at 10 degrees, and a rear window keyframe outside the clip shows it
    frontal. The mark's own track: 30 source px wide and level on every clip frame."""
    write_run(run_id)
    score = c.KeyframeScore(area=1, completeness=1, sharpness=1, glare=0, frontality=1, total=1)
    kf = lambda kf_id, pid, pts, b: c.Keyframe(kf_id=kf_id, role="return", panel_id=pid, track_id="t", f_proxy=round(pts * 15), pts=pts, score=score, aspect_ratio=1.5, box_src=[600.0, 500.0, 1600.0, 1000.0], blur=b)  # noqa: E731
    kfs = [
        kf("return_trunk_lid_kf1", c.PanelId.TRUNK_LID, 5.0, turned_plate(40)),
        kf("return_rear_bumper_kf1", c.PanelId.REAR_BUMPER, 5.5, turned_plate(10)),
        kf("return_rear_window_kf1", c.PanelId.REAR_WINDOW, 9.0, turned_plate(0)),
    ]
    c.write_json(paths.keyframes_json(run_id), kfs)
    mark_id = "return_trunk_lid_kf1_m1"
    c.write_json(paths.marks_json(run_id), [c.Mark(mark_id=mark_id, role="return", panel_id=c.PanelId.TRUNK_LID, kf_id="return_trunk_lid_kf1", source=c.MarkSource.SPARK_BOX, type=c.MarkKind.SCRATCH, confidence=c.Confidence.HIGH, size_class=c.SizeClass.S)])
    window = [500.0, 400.0, 1508.0, 1408.0]
    mt = track([[480.0, 350.0, 510.0, 356.0]] * 91, window=window).model_copy(update={"mark_id": mark_id})
    mt = mt.model_copy(update={"clip": mt.clip.model_copy(update={"path": "clips/return/return_trunk_lid_kf1_sp01.mp4", "f0_src": 105})})
    c.write_json(paths.mark_tracks_json(run_id), [mt])
    if with_plan:
        from walkaround import staysput

        obj = staysput.PlanObject(object_id="1", mark_id=mark_id, type=c.MarkKind.SCRATCH, box_src=[980.0, 750.0, 1010.0, 756.0], box_clip=[480.0, 350.0, 510.0, 356.0])
        plan = staysput.ClipPlan(
            clip_id="return_trunk_lid_kf1_sp01", role="return", kf_id="return_trunk_lid_kf1", panel_id=c.PanelId.TRUNK_LID, lane_id="sp_return_trunk_lid_kf1_sp01",
            src_size=[W, 1080], proxy_size=[960, 540], size=1008, prompt_frame=45, f0_src=105, pts_src=[3.5 + i / 30 for i in range(91)],
            window_px=1008.0, windows_src=[window] * 91, panel_src=[None] * 91, objects=[obj], status="answered",
        )
        c.write_json(paths.run_dir(run_id) / "clips" / "return" / staysput.PLAN_NAME, staysput.RolePlans(role="return", complete=True, clips=[plan]))
    return mark_id


def test_the_most_frontal_plate_in_the_marks_clip_sizes_it_on_that_frame(runs: Path) -> None:
    mark_id = write_clip_run("fr1")
    s = ruler.size_marks("fr1", th=TH)[mark_id]
    assert s.ruler == c.RulerKind.PLATE and s.ruler_source_kf == "return_rear_bumper_kf1"  # 10 degrees, inside the clip
    assert s.inches == pytest.approx(30.0 / 20.0 / math.cos(math.radians(10)))  # a level mark along the turned axis
    assert s.ruler_notes == []  # a trunk-lid mark has no wheel next to it


def test_without_the_clip_plan_only_the_marks_own_keyframe_counts(runs: Path) -> None:
    mark_id = write_clip_run("fr2", with_plan=False)
    s = ruler.size_marks("fr2", th=TH)[mark_id]
    assert s.inches is None and s.refused_reason == "the license plate is turned more than 30 degrees from the camera"


def test_a_refusal_names_the_most_frontal_plate_the_clip_saw(runs: Path) -> None:
    mark_id = write_clip_run("fr3")
    kfs = c.read_json_list(paths.keyframes_json("fr3"), c.Keyframe)
    kfs[1] = kfs[1].model_copy(update={"blur": blur(50.0, 0.5)})  # the 10 degree view is now too small to measure
    c.write_json(paths.keyframes_json("fr3"), kfs)
    s = ruler.size_marks("fr3", th=TH)[mark_id]
    assert s.refused_reason == "the license plate is under 60 px"  # 0 degrees beats 40: the frontal view says why


def test_the_run_tire_code_on_the_manifest_sizes_wheel_marks(runs: Path) -> None:
    write_run("tc1")
    assert ruler.run_tire_code("tc1") is None
    assert ruler.size_marks("tc1", th=TH)["return_left_rear_wheel_kf0_m1"].refused_reason == ruler.TIRE_NOT_GIVEN
    run = c.read_json(paths.run_json("tc1"), c.RunManifest).model_copy(update={"tire_code": "225/60R16"})
    c.write_json(paths.run_json("tc1"), run)
    assert ruler.run_tire_code("tc1") == "225/60R16"
    w = ruler.size_marks("tc1", th=TH)["return_left_rear_wheel_kf0_m1"]
    assert w.ruler == c.RulerKind.WHEEL and w.tire_code == "225/60R16"
    assert ruler.size_marks("tc1", th=TH, tire_code="")["return_left_rear_wheel_kf0_m1"].refused_reason == ruler.TIRE_NOT_GIVEN  # an explicit "none"
    c.write_json(paths.verdicts_json("tc1"), [c.Verdict(verdict_id="w1", panel_id=c.PanelId.LEFT_REAR_WHEEL, label=c.VerdictLabel.NEW_OR_UNSEEN, rule_id="L2", mark_ids=["return_left_rear_wheel_kf0_m1"])])
    (v,) = ruler.size_verdicts("tc1", th=TH)
    assert v.size.ruler == c.RulerKind.WHEEL and sheet.size_line(v.size).endswith("(ruler: the wheel, 26.6 in from 225/60R16, typed by you)")


def test_the_size_says_why_the_other_ruler_was_not_used(runs: Path) -> None:
    write_run("nt1")
    plate_sized = ruler.size_marks("nt1", th=TH)["return_rear_bumper_kf0_m1"]
    assert plate_sized.ruler == c.RulerKind.PLATE and plate_sized.ruler_notes == [ruler.NO_WHEEL]  # no wheel box on that frame
    tracks = c.read_json_list(paths.tracks_json("nt1"), c.PanelTrack)
    near = c.FrameObs(frame=40, pts=13.3, box=[480.0, 360.0, 560.0, 440.0], box_area_frac=0.04, in_car_box=True, c=0.9)  # a wheel on the bumper frame
    tracks[0] = tracks[0].model_copy(update={"per_frame": [*tracks[0].per_frame, near]})
    c.write_json(paths.tracks_json("nt1"), tracks)
    assert ruler.size_marks("nt1", th=TH)["return_rear_bumper_kf0_m1"].ruler_notes == ["the wheel: tire size not given"]
    kfs = c.read_json_list(paths.keyframes_json("nt1"), c.Keyframe)
    kfs[0] = kfs[0].model_copy(update={"blur": turned_plate(40)})
    c.write_json(paths.keyframes_json("nt1"), kfs)
    refused = ruler.size_marks("nt1", th=TH)["return_rear_bumper_kf0_m1"]
    assert refused.refused_reason == ruler.TIRE_NOT_GIVEN  # the reason the renter can fix comes first
    assert refused.ruler_notes == ["the license plate is turned more than 30 degrees from the camera"]
    assert sheet.size_line(refused) == "size not stated: tire size not given; the license plate is turned more than 30 degrees from the camera"


def test_the_sheet_line_names_the_ruler_used_then_why_another_was_not() -> None:
    s = c.SizeEstimate(inches=1.2, bound_lo=0.9, bound_hi=1.6, ruler=c.RulerKind.WHEEL, tire_code="155/80R13", ruler_notes=["the license plate is flatter than a 12 by 6 in plate, so its size is not known"])
    assert sheet.size_line(s) == "about 1.2 in, between 0.9 and 1.6 in (ruler: the wheel, 22.8 in from 155/80R13, typed by you; the license plate is flatter than a 12 by 6 in plate, so its size is not known)"
    assert sheet.size_line(c.SizeEstimate(inches=1.5, bound_lo=1.3, bound_hi=1.7, ruler=c.RulerKind.PLATE)) == "about 1.5 in, between 1.3 and 1.7 in (ruler: the license plate, 12 in)"


# walkaround run --tire-code ------------------------------------------------------------------------------------------


def test_the_cli_reads_the_tire_code_before_spending(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from walkaround import __main__ as cli

    walk = tmp_path / "walk.mp4"
    walk.write_bytes(b"x")
    monkeypatch.delenv("LIVE", raising=False)
    parse = cli.build_parser().parse_args
    assert cli.cmd_run(parse(["run", str(walk), str(walk), "--tire-code", "bogus"]), orch=None) == 2  # type: ignore[arg-type]
    assert "the tire code typed could not be read: 'bogus'" in capsys.readouterr().err
    args = parse(["run", str(walk), str(walk), "--tire-code", "P155/80 R13 77S"])
    assert cli.cmd_run(args, orch=None) == 2 and args.tire_code == "155/80R13"  # type: ignore[arg-type]
    assert "tire code 155/80R13 (22.8 in); set LIVE=1" in capsys.readouterr().err


def test_a_run_the_cli_creates_carries_the_tire_code(runs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--tire-code travels through pipeline.run_pair_async to Orchestrator.create_run (integrator 10/31: no wrapper)."""
    import asyncio

    from walkaround import __main__ as cli
    from walkaround import pipeline

    made: dict[str, object] = {}

    class Orch:
        def run_exists(self, run_id: str) -> bool:
            return False

        async def create_run(self, **kwargs: object) -> c.RunManifest:
            made.update(kwargs)
            return c.RunManifest(run_id=str(kwargs["run_id"]), created_at="2026-10-31T23:00:00Z", tire_code=kwargs.get("tire_code"))  # type: ignore[arg-type]

        def load_run(self, run_id: str) -> c.RunManifest:
            return c.RunManifest(run_id=run_id, created_at="2026-10-31T23:00:00Z", tire_code=made.get("tire_code"))  # type: ignore[arg-type]

        async def run_pipeline(self, run_id: str) -> c.RunManifest:
            return self.load_run(run_id)

    async def staged(*_args: object, **_kw: object) -> None:
        return None

    monkeypatch.setattr(pipeline, "stage_local_file", staged)
    walk = tmp_path / "walk.mp4"
    walk.write_bytes(b"not a video")
    args = cli.build_parser().parse_args(["run", str(walk), str(walk), "--run-id", "tc_cli", "--tire-code", "155/80R13", "--quiet"])
    asyncio.run(cli._run_local(Orch(), args))  # type: ignore[arg-type]
    assert made["tire_code"] == "155/80R13" and made["run_id"] == "tc_cli"


def test_create_run_stores_the_tire_code_on_run_json(runs: Path) -> None:
    import asyncio

    from walkaround.orchestrator import FileEventLog, Orchestrator

    orch = Orchestrator(th=c.load_thresholds(), engine=c.SamEngine.META_MODEL_API, max_concurrent_runs=1, media_workers=1, decode_workers=1, events=FileEventLog(), delays_path=runs / "delays.json")
    run = asyncio.run(orch.create_run(mode=c.RunMode.CAR, kind=c.RunKind.PAIR, keep_plate=False, run_id="tc_orch", tire_code="155/80R13"))
    assert run.tire_code == "155/80R13" and ruler.run_tire_code("tc_orch") == "155/80R13"
    plain = asyncio.run(orch.create_run(mode=c.RunMode.CAR, kind=c.RunKind.PAIR, keep_plate=False, run_id="tc_none"))
    assert plain.tire_code is None and "tire_code" not in paths.run_json("tc_none").read_text()


def test_the_new_fields_round_trip_and_stay_out_of_the_bytes_when_empty() -> None:
    plain = c.SizeEstimate(refused_reason=ruler.TIRE_NOT_GIVEN)
    noted = plain.model_copy(update={"ruler_notes": ["the license plate is turned more than 30 degrees from the camera"]})
    assert "ruler_notes" not in plain.to_json() and c.SizeEstimate.from_json(noted.to_json()) == noted
    run = c.RunManifest(run_id="r", created_at="2026-10-31T23:00:00Z")
    typed = run.model_copy(update={"tire_code": "155/80R13"})
    assert "tire_code" not in run.to_json() and c.RunManifest.from_json(typed.to_json()).tire_code == "155/80R13"


# A size on a flagged view (lead 11/1): printed with a note in K15's words; darkness and the geometry still refuse -------


FROZEN = {**TH, "ruler": {k: v for k, v in TH["ruler"].items() if k != "view_flag_sizes"}}  # a run frozen before 11/1


def flag(run_id: str, kf_id: str, **flags: bool) -> None:
    kfs = c.read_json_list(paths.keyframes_json(run_id), c.Keyframe)
    c.write_json(paths.keyframes_json(run_id), [k.model_copy(update={"conditions": c.Conditions(**flags)}) if k.kf_id == kf_id else k for k in kfs])


def test_the_view_note_speaks_like_k15() -> None:
    assert ruler.view_note("return", ["dirty"]) == "return view flagged dirty: the size may include dirt"
    assert ruler.view_note("pickup", ["dirty", "glare"]) == "pickup view flagged dirty and glare: the size may include dirt or a reflection"
    assert ruler.view_note("pickup", []) is None
    assert TH["ruler"]["view_flag_sizes"] is True and ruler.sizes_on_view_flags(TH) and not ruler.sizes_on_view_flags(FROZEN)


def test_a_flagged_crop_prints_its_size_with_a_view_note(runs: Path) -> None:
    mark_id = write_clip_run("vf1")
    clean = ruler.size_marks("vf1", th=TH)[mark_id]
    flag("vf1", "return_trunk_lid_kf1", dirty=True)  # the mark's own keyframe
    s = ruler.size_marks("vf1", th=TH)[mark_id]
    assert s.inches == clean.inches and s.ruler_source_kf == "return_rear_bumper_kf1"
    assert s.ruler_notes == ["return view flagged dirty: the size may include dirt"]
    assert sheet.size_line(s).endswith("in (ruler: the license plate, 12 in; return view flagged dirty: the size may include dirt)")
    flag("vf1", "return_rear_bumper_kf1", wet=True, glare=True)  # and the keyframe the size was measured on
    s = ruler.size_marks("vf1", th=TH)[mark_id]
    assert s.inches == clean.inches and s.ruler_notes == ["return view flagged wet, dirty and glare: the size may include water, dirt or a reflection"]
    flag("vf1", "return_rear_window_kf1", glare=True)  # outside the mark's clip: not a view of this size
    assert ruler.size_marks("vf1", th=TH)[mark_id].ruler_notes == s.ruler_notes
    assert ruler.size_marks("vf1", th=FROZEN)[mark_id].refused_reason == ruler.CROP_FLAG  # a run frozen before 11/1 rebuilds as it was


def test_a_dark_crop_and_the_geometric_refusals_still_refuse(runs: Path) -> None:
    mark_id = write_clip_run("vf2")
    flag("vf2", "return_trunk_lid_kf1", dark=True, dirty=True)
    assert ruler.size_marks("vf2", th=TH)[mark_id].refused_reason == ruler.DARK_FRAME
    mark_id = write_clip_run("vf3", with_plan=False)  # only the mark's own keyframe counts, and its plate is turned 40 degrees
    flag("vf3", "return_trunk_lid_kf1", glare=True)
    s = ruler.size_marks("vf3", th=TH)[mark_id]
    assert s.refused_reason == "the license plate is turned more than 30 degrees from the camera" and s.ruler_notes == []  # no view note without a size
    assert ruler.size_marks("vf3", th=FROZEN)[mark_id].refused_reason == ruler.CROP_FLAG
    write_run("vf4")  # a wheel mark with no tire code: still the reason the renter can fix
    flag("vf4", "return_left_rear_wheel_kf0", dirty=True)
    assert ruler.size_marks("vf4", th=TH)["return_left_rear_wheel_kf0_m1"].refused_reason == ruler.TIRE_NOT_GIVEN
    sized = ruler.size_marks("vf4", th=TH, tire_code="225/60R16")["return_left_rear_wheel_kf0_m1"]
    assert sized.ruler == c.RulerKind.WHEEL and sized.ruler_notes == ["return view flagged dirty: the size may include dirt"]
