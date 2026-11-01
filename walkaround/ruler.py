"""Size from a ruler (docs/ARCHITECTURE.md 5.12; docs/CONTRACTS.md M11): the license plate or the wheel.

Both rulers are SAM masks measured before any pixel was pixelated: the plate from the BlurRecord the image checks wrote
on the unblurred frame (privacy.image_checks), the wheel from its own mask or its SAM track box on the same frame. The
mark is measured from its own SAM mask or SAM track, never from a Spark box (D1 E6: Spark's positions jitter by 6.5%
of the crop width). Every size carries its interval and the ruler it came from; anything that breaks a rule of 5.12
comes back as a refusal with its reason, and the Hertz comparison is refused, not the size, when the interval straddles
the threshold (compare_hertz; sheet.threshold_line prints it). The wheel needs the tire code the renter typed: 5.12
refuses an unknown code ("tire size not given", section 14 item 21); a walk takes it from the run manifest
(RunManifest.tire_code, walkaround run --tire-code). Everything here is arithmetic except size_marks and size_verdicts,
which read a run's frozen files.

A walk's mark is sized on the frame where its ruler faces the camera best, not only on the keyframe it was found on:
every keyframe of the walk inside the mark's Stays Put clip is a candidate, the plate (or wheel) measured on that
keyframe and the mark's extent measured by its own track around the same frame, so ruler and mark stay on one frame as
5.12 asks. The most frontal ruler that sizes the mark wins (the plate before the wheel), and the size carries
ruler_notes: why each other ruler next to the mark did not size it, which the sheet prints after the ruler used. A plate
flatter than a 12 by 6 in plate seen from 45 degrees above (ruler.plate_hw_min) is not this ruler: a European or Indian
strip plate would read as frontal and size every mark too small.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from . import paths
from .contracts import (
    BODY_PANELS,
    LOWER_BUMPERS,
    WHEEL_PANELS,
    BlurRecord,
    Keyframe,
    Mark,
    MarkKind,
    MarkTrack,
    Noun,
    PanelId,
    PanelTrack,
    RulerKind,
    SizeEstimate,
    StaysPutVerdict,
    ThresholdType,
    Verdict,
    VideoRecord,
    read_json,
    read_json_list,
    write_json,
)

Box4 = list[float]

PLATE_FRONTAL_HW = 0.5  # a US plate is 12 in by 6 in, so a frontal plate's apparent h / w is 0.5
PLATE_HW_MIN = 0.35  # thresholds ruler.plate_hw_min (estimate): 0.5 x cos 45, a 12 by 6 in plate seen from 45 degrees above
NEAR_FRAMES = 3  # a mark's extent on a frame is the median of its track boxes within this many clip frames of it
NEAR_POINTS_MIN = 3  # track boxes that extent needs
# The camera distance enters the depth term of the bound. With a ruler in frame it follows from the ruler's pixels per
# inch and a focal length; phone video does not carry one, so a main camera's horizontal field of view is assumed
# (ruler.camera_hfov_deg overrides it once thresholds.json carries it).
CAMERA_HFOV_DEG = 70.0  # estimate
INCH_M = 0.0254
NO_PLATE = "no license plate measured on that frame"
NO_WHEEL = "no wheel measured on that frame"
NOT_A_US_PLATE = "the license plate is flatter than a 12 by 6 in plate, so its size is not known"
# 5.12: an unknown code is refused, since the US market spans 28.4 in +/- 12% (claim_first_and_metrology.md section 6).
TIRE_NOT_GIVEN = "tire size not given"
TIRE_UNREADABLE = "the tire code typed could not be read"
NO_MARK_MASK = "no SAM mask of this mark; a Spark box is never used for a size"
CROP_FLAG = "glare, water, dirt or darkness on that frame"
DRAWN_BOX = "a drawn box overlaps the mark"
PROXY_FRAME = "the mark was measured on a proxy frame"
NO_CAMERA_DISTANCE = "the frame size is not recorded, so the camera distance is unknown"

# Marks a ruler may size: the plate's neighbours, and for the wheel the wheel, fender and lower-bumper marks of 5.12.
FENDERS: tuple[PanelId, ...] = tuple(p for p in BODY_PANELS if p.value.endswith("_fender"))
PLATE_PANELS: frozenset[PanelId] = frozenset(
    {PanelId.FRONT_BUMPER, PanelId.REAR_BUMPER, PanelId.HOOD, PanelId.TRUNK_LID, PanelId.HEADLIGHTS, PanelId.TAIL_LIGHTS, *FENDERS}
)
WHEEL_RULER_PANELS: frozenset[PanelId] = frozenset({*WHEEL_PANELS, *FENDERS, *LOWER_BUMPERS})

# A load index, speed rating or XL may follow the rim size, with or without a space ('245/40R18XL', '225/60R16 98H').
_TIRE_CODE = re.compile(r"^\s*(?:P|LT|ST|T)?\s*(\d{3})\s*/\s*(\d{2})\s*[A-Z]{0,2}\s*(\d{2}(?:\.\d)?)(?![\d.])(?:(?:\s+|(?=[A-Z]))\S.*?)?\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class Ruler:
    """An object of known size on the mark's frame: source pixels per inch across the view at the ruler's depth."""

    kind: RulerKind
    ppi: float
    tilt_deg: float  # how far the ruler is turned from facing the camera
    tilt_axis_deg: float  # image angle of the foreshortened direction, degrees in [0, 180)
    center: tuple[float, float]  # source pixels
    spread: float = 0.0  # relative uncertainty of the ruler's own length (a proxy track box's rounding)
    tire_code: str | None = None
    source_kf: str | None = None


@dataclass(frozen=True)
class MarkExtent:
    """A mark's length along its principal axis in source pixels, from a SAM mask or a SAM track."""

    long_px: float
    axis_deg: float  # image angle of the long axis, degrees in [0, 180)
    center: tuple[float, float]
    long_px_hi: float | None = None  # a box gives only a range: its long side up to its diagonal
    source: Literal["mask", "track", "spark_box"] = "mask"
    proxy: bool = False


# Tire codes ------------------------------------------------------------------------------------------------------------


def normalize_tire_code(code: str | None) -> str | None:
    """'P225/60 R16 98H' or '245/40R18XL' -> '225/60R16' or '245/40R18', the spelling sheet.tire_diameter_in reads;
    None for anything else."""
    m = _TIRE_CODE.match(code or "")
    if not m:
        return None
    return f"{m.group(1)}/{m.group(2)}R{m.group(3)}"


def tire_diameter_in(code: str | None) -> float | None:
    """Overall diameter D = 2 x (SW x AR / 2540 + RD / 2) inches from a sidewall code; None for an unknown code."""
    norm = normalize_tire_code(code)
    if norm is None:
        return None
    sw, rest = norm.split("/")
    ar, rd = rest.split("R")
    return 2.0 * (float(sw) * float(ar) / 2540.0 + float(rd) / 2.0)


# Rulers ----------------------------------------------------------------------------------------------------------------


def tilt_from_aspect(apparent_hw: float | None, frontal_hw: float = PLATE_FRONTAL_HW) -> float:
    """Degrees a plate is turned, from apparent h / w = 0.5 / cos(theta); 0 when it looks no taller than frontal."""
    if apparent_hw is None or apparent_hw <= frontal_hw:
        return 0.0
    return math.degrees(math.acos(frontal_hw / apparent_hw))


def _plate_center(blur: BlurRecord) -> tuple[float, float] | None:
    """Centre of the plate box the ruler was measured on: the box whose long side is nearest plate_ruler_px."""
    if not blur.plate_boxes or blur.plate_ruler_px is None:
        return None
    box = min(blur.plate_boxes, key=lambda b: (abs(max(b[2] - b[0], b[3] - b[1]) - blur.plate_ruler_px), b[0], b[1]))
    return (0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3]))


def plate_ruler(blur: BlurRecord, *, th: dict[str, Any], source_kf: str | None = None) -> Ruler | str:
    """The plate as a ruler, or the reason it is not one. The long side shrinks by cos(theta) when the plate is turned,
    so ppi = long side / (12 in x cos(theta)): the scale across the view, which the mark's axis correction expects."""
    r = th["ruler"]
    if blur.plate_ruler_px is None:
        return NO_PLATE
    if blur.plate_ruler_c is not None and blur.plate_ruler_c < float(r["plate_c_min"]):
        return f"the license plate mask is under c {float(r['plate_c_min']):g}"
    if blur.plate_ruler_px < float(r["plate_min_px"]):
        return f"the license plate is under {float(r['plate_min_px']):g} px"
    if blur.plate_tilt_hw is not None and blur.plate_tilt_hw < float(r.get("plate_hw_min", PLATE_HW_MIN)):
        return NOT_A_US_PLATE  # a strip plate reads as frontal; its long side is not 12 in
    tilt = tilt_from_aspect(blur.plate_tilt_hw)
    if tilt > float(r["plate_tilt_max_deg"]):
        return f"the license plate is turned more than {float(r['plate_tilt_max_deg']):g} degrees from the camera"
    center = _plate_center(blur)
    if center is None:
        return "the license plate position was not recorded"
    ppi = float(blur.plate_ruler_px) / (float(r["plate_long_side_in"]) * math.cos(math.radians(tilt)))
    return Ruler(kind=RulerKind.PLATE, ppi=ppi, tilt_deg=tilt, tilt_axis_deg=0.0, center=center, source_kf=source_kf)


def ellipse_axes(mask: np.ndarray) -> tuple[float, float, float, tuple[float, float]] | None:
    """(major px, minor px, major axis angle in degrees, centroid) of a mask's moment ellipse. A filled ellipse with
    semi-axes a and b has variances a^2 / 4 and b^2 / 4 along its axes, so each full axis is 4 sqrt(variance)."""
    ys, xs = np.nonzero(np.asarray(mask))
    if xs.size < 5:
        return None
    cx, cy = float(xs.mean()), float(ys.mean())
    cov = np.cov(np.vstack([xs - cx, ys - cy]).astype(np.float64), bias=True)
    evals, evecs = np.linalg.eigh(cov)
    major, minor = 4.0 * math.sqrt(max(float(evals[1]), 0.0)), 4.0 * math.sqrt(max(float(evals[0]), 0.0))
    angle = math.degrees(math.atan2(float(evecs[1, 1]), float(evecs[0, 1]))) % 180.0
    return major, minor, angle, (cx, cy)


def wheel_scale(tire_code: str | None) -> tuple[float, str] | str:
    """(overall diameter in inches, the code as stored) from the code the renter typed, or the refusal: no code, or
    one that does not read as a sidewall size."""
    if not (tire_code or "").strip():
        return TIRE_NOT_GIVEN
    norm = normalize_tire_code(tire_code)
    diameter = tire_diameter_in(norm)
    if norm is None or diameter is None:
        return TIRE_UNREADABLE
    return diameter, norm


def _wheel(major_px: float, minor_px: float, axis_deg: float, center: tuple[float, float], c: float | None, *, touches_border: bool, tire_code: str | None, th: dict[str, Any], extra_spread: float, source_kf: str | None) -> Ruler | str:
    r = th["ruler"]
    if touches_border:
        return "the wheel touches the frame edge"
    if c is not None and c < float(r["wheel_c_min"]):
        return f"the wheel mask is under c {float(r['wheel_c_min']):g}"
    if major_px < float(r["wheel_min_px"]):
        return f"the wheel is under {float(r['wheel_min_px']):g} px across"
    tilt = math.degrees(math.acos(min(1.0, max(minor_px, 0.0) / major_px)))
    if tilt > float(r["plate_tilt_max_deg"]):  # the same 30 degree gauge as the plate (5.12 names one)
        return f"the wheel is turned more than {float(r['plate_tilt_max_deg']):g} degrees from the camera"
    scale = wheel_scale(tire_code)  # after the checks of the wheel itself: typing a code fixes only this one
    if isinstance(scale, str):
        return scale
    diameter, code = scale
    return Ruler(
        kind=RulerKind.WHEEL,
        ppi=major_px / diameter,
        tilt_deg=tilt,
        tilt_axis_deg=(axis_deg + 90.0) % 180.0,  # a turned wheel shrinks across its major axis
        center=center,
        spread=extra_spread,
        tire_code=code,
        source_kf=source_kf,
    )


def wheel_ruler(mask: np.ndarray, c: float | None, *, touches_border: bool, tire_code: str | None, th: dict[str, Any], scale: float = 1.0, offset: tuple[float, float] = (0.0, 0.0), source_kf: str | None = None) -> Ruler | str:
    """The wheel as a ruler from its mask: the major axis of the moment ellipse is the tire's overall diameter, the
    minor over the major is cos(tilt). Mask pixels map to source as offset + p / scale (the Crop convention)."""
    # TODO(D1 E11.ppi): that SAM 3.1's "wheel" mask spans the tire's overall diameter (not the rim) is unmeasured; E11.ppi
    # (plate and wheel ppi within 15% on one frame) needs a typed code on the friend's car.
    axes = ellipse_axes(mask)
    if axes is None:
        return "the wheel mask is empty"
    major, minor, angle, (cx, cy) = axes
    center = (offset[0] + cx / scale, offset[1] + cy / scale)
    return _wheel(major / scale, minor / scale, angle, center, c, touches_border=touches_border, tire_code=tire_code, th=th, extra_spread=0.0, source_kf=source_kf)


def wheel_ruler_from_box(box_src: Box4, c: float | None, *, touches_border: bool, tire_code: str | None, th: dict[str, Any], quantization_px: float = 0.0, source_kf: str | None = None) -> Ruler | str:
    """The wheel from its SAM track box on the mark's frame: a wheel turned about the vertical keeps its height and
    narrows, so the box's long side is the diameter and the short side over it is cos(tilt). quantization_px is the
    box's own rounding in source pixels (a proxy box scaled up), added to the ruler's spread for both ends."""
    w, h = float(box_src[2]) - float(box_src[0]), float(box_src[3]) - float(box_src[1])
    if w <= 0 or h <= 0:
        return "the wheel box is empty"
    major, minor = max(w, h), min(w, h)
    axis = 90.0 if h >= w else 0.0
    center = (0.5 * (float(box_src[0]) + float(box_src[2])), 0.5 * (float(box_src[1]) + float(box_src[3])))
    extra = 2.0 * quantization_px / major
    return _wheel(major, minor, axis, center, c, touches_border=touches_border, tire_code=tire_code, th=th, extra_spread=extra, source_kf=source_kf)


# Mark extents ---------------------------------------------------------------------------------------------------------


def extent_from_mask(mask: np.ndarray, *, scale: float = 1.0, offset: tuple[float, float] = (0.0, 0.0), proxy: bool = False) -> MarkExtent | None:
    """The mask's pixel extent along its principal axis (both end pixels counted), in source pixels."""
    ys, xs = np.nonzero(np.asarray(mask))
    if xs.size == 0:
        return None
    cx, cy = float(xs.mean()), float(ys.mean())
    if xs.size >= 3:
        cov = np.cov(np.vstack([xs - cx, ys - cy]).astype(np.float64), bias=True)
        evecs = np.linalg.eigh(cov)[1]
        ux, uy = float(evecs[0, 1]), float(evecs[1, 1])
    else:
        ux, uy = 1.0, 0.0
    proj = (xs - cx) * ux + (ys - cy) * uy
    long_px = float(proj.max() - proj.min()) + 1.0
    angle = math.degrees(math.atan2(uy, ux)) % 180.0
    return MarkExtent(long_px=long_px / scale, axis_deg=angle, center=(offset[0] + cx / scale, offset[1] + cy / scale), proxy=proxy)


def extent_from_track(mt: MarkTrack | None, *, min_points: int = 5, frames: tuple[int, int] | None = None) -> MarkExtent | None:
    """The median extent of a Stays Put track in source pixels (5.12: the mark's own SAM track in the clip frames cut
    from the original), over the whole clip or only clip frames lo to hi (frames). A track box gives a range: the long
    side when the mark lies along an image axis, the diagonal when it lies across it."""
    if mt is None or mt.clip is None or mt.verdict in (StaysPutVerdict.NO_TRACK, StaysPutVerdict.NOT_RUN):
        return None
    clip = mt.clip
    longs, diags, xs, ys, wide = [], [], [], [], 0
    for p in mt.track:
        if not p.inside_panel or p.f < 0 or p.f >= len(clip.windows_src):
            continue
        if frames is not None and not frames[0] <= p.f <= frames[1]:
            continue
        win = clip.windows_src[p.f]
        s = (float(win[2]) - float(win[0])) / float(clip.w)  # source pixels per clip pixel
        bw, bh = float(p.box_clip[2]) - float(p.box_clip[0]), float(p.box_clip[3]) - float(p.box_clip[1])
        if bw <= 0 or bh <= 0 or s <= 0:
            continue
        longs.append(max(bw, bh) * s)
        diags.append(math.hypot(bw, bh) * s)
        xs.append(float(win[0]) + 0.5 * (float(p.box_clip[0]) + float(p.box_clip[2])) * s)
        ys.append(float(win[1]) + 0.5 * (float(p.box_clip[1]) + float(p.box_clip[3])) * s)
        wide += 1 if bw >= bh else 0
    if len(longs) < min_points:
        return None
    axis = 0.0 if 2 * wide >= len(longs) else 90.0
    return MarkExtent(
        long_px=float(statistics.median(longs)),
        axis_deg=axis,
        center=(float(statistics.median(xs)), float(statistics.median(ys))),
        long_px_hi=float(statistics.median(diags)),
        source="track",
    )


# The size and the Hertz comparison -----------------------------------------------------------------------------------


def threshold_type_for(mark_type: MarkKind | None, panel_id: PanelId | None) -> ThresholdType | None:
    """Which Hertz example threshold (REV 7.2025, illustrative only) a mark is compared against: wheel scratches and
    scuffs against 1 in (the page names those two; a chip is missing material, which it lists under replacement);
    body scratches against the 3.5 in card, dents against 1 in; a body scuff gets the ordinary wear sentence instead of
    a comparison. Glass and lights have no size threshold on that page."""
    if mark_type is None or panel_id is None:
        return None
    if panel_id in WHEEL_PANELS:
        return ThresholdType.WHEEL_MARK if mark_type in (MarkKind.SCRATCH, MarkKind.SCUFF) else None
    if panel_id in BODY_PANELS:
        return {MarkKind.SCRATCH: ThresholdType.BODY_SCRATCH, MarkKind.DENT: ThresholdType.DENT, MarkKind.SCUFF: ThresholdType.BODY_SCUFF}.get(mark_type)
    return None


def threshold_in(threshold_type: ThresholdType | None, th: dict[str, Any]) -> float | None:
    r = th["ruler"]
    return {
        ThresholdType.BODY_SCRATCH: r["threshold_body_scratch_in"],
        ThresholdType.DENT: r["threshold_dent_in"],
        ThresholdType.WHEEL_MARK: r["threshold_wheel_in"],
    }.get(threshold_type)  # type: ignore[arg-type]


def compare_hertz(size: SizeEstimate | None, th: dict[str, Any]) -> Literal["under", "over", "straddles"] | None:
    """The comparison sheet.threshold_line prints: under or over only when the whole interval is on one side,
    straddles (the sentence is refused, not the size) otherwise; None with no size or no threshold."""
    if size is None or size.inches is None:
        return None
    limit = threshold_in(size.threshold_type, th)
    if limit is None:
        return None
    lo = size.bound_lo if size.bound_lo is not None else size.inches
    hi = size.bound_hi if size.bound_hi is not None else size.inches
    if hi < float(limit):
        return "under"
    if lo > float(limit):
        return "over"
    return "straddles"


def no_ruler_reason(th: dict[str, Any]) -> str:
    """The reason sheet.NO_RULER_LINE prints, with the distance from thresholds."""
    return f"no ruler within {float(th['ruler']['ruler_max_distance_m']):g} m of this mark"


def _axis_gap(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def camera_distance_in(ppi: float, frame_w: int, hfov_deg: float = CAMERA_HFOV_DEG) -> float:
    """Inches from the camera to the ruler: ppi = f / D with f = (w / 2) / tan(hfov / 2) in pixels."""
    f_px = (float(frame_w) / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return f_px / ppi


def size_estimate(
    extent: MarkExtent | None,
    ruler: Ruler | str | None,
    *,
    mark_type: MarkKind | None,
    panel_id: PanelId,
    frame_w: int | None,
    th: dict[str, Any],
    crop_flag: bool = False,
    drawn_box_overlap: bool = False,
) -> SizeEstimate:
    """5.12 for one mark and one ruler on the same frame. size = mark px / ppi, divided by cos(tilt) only when the
    mark's axis lies within 30 degrees of the ruler's foreshortened axis; the interval is size x (1 -/+ the sum of
    foreshortening 1 - cos(tilt), depth L sin(tilt) / D, 2 px per side over the mark's length and the ruler's own
    spread). Every refusal of 5.12 returns refused_reason instead."""
    r = th["ruler"]

    def refuse(reason: str) -> SizeEstimate:
        return SizeEstimate(refused_reason=reason)

    if crop_flag:
        return refuse(CROP_FLAG)
    if drawn_box_overlap:
        return refuse(DRAWN_BOX)
    if extent is None or extent.source == "spark_box":
        return refuse(NO_MARK_MASK)
    if extent.proxy:
        return refuse(PROXY_FRAME)
    if extent.long_px < float(r["mark_min_px"]):
        return refuse(f"the mark is under {float(r['mark_min_px']):g} px along its length")
    if ruler is None:
        return refuse(no_ruler_reason(th))
    if isinstance(ruler, str):
        return refuse(ruler)
    if panel_id not in _adjacent_panels(ruler.kind):
        return refuse(no_ruler_reason(th))
    if not frame_w:
        return refuse(NO_CAMERA_DISTANCE)
    dist_in = math.dist(extent.center, ruler.center) / ruler.ppi
    if dist_in * INCH_M > float(r["ruler_max_distance_m"]):
        return refuse(no_ruler_reason(th))
    tilt = math.radians(ruler.tilt_deg)
    size = extent.long_px / ruler.ppi
    size_hi = (extent.long_px_hi or extent.long_px) / ruler.ppi
    if ruler.tilt_deg > 0 and _axis_gap(extent.axis_deg, ruler.tilt_axis_deg) <= float(r["axis_tilt_correction_max_deg"]):
        size, size_hi = size / math.cos(tilt), size_hi / math.cos(tilt)
    hfov = float(r.get("camera_hfov_deg", CAMERA_HFOV_DEG))
    rel = (1.0 - math.cos(tilt)) + dist_in * math.sin(tilt) / camera_distance_in(ruler.ppi, frame_w, hfov)
    rel += 2.0 * float(r["bound_px_per_side"]) / extent.long_px + ruler.spread
    return SizeEstimate(
        inches=size,
        bound_lo=max(0.0, size * (1.0 - rel)),
        bound_hi=size_hi * (1.0 + rel),
        ruler=ruler.kind,
        ruler_source_kf=ruler.source_kf,
        tire_code=ruler.tire_code,
        threshold_type=threshold_type_for(mark_type, panel_id),
    )


def best_size(
    extent: MarkExtent | None,
    rulers: list[Ruler | str],
    *,
    mark_type: MarkKind | None,
    panel_id: PanelId,
    frame_w: int | None,
    th: dict[str, Any],
    crop_flag: bool = False,
    drawn_box_overlap: bool = False,
) -> SizeEstimate:
    """The first ruler that sizes the mark, in the order given (the plate first, it is the better ruler); callers pass
    only the rulers adjacent to the mark's panel. A crop flag or a drawn box refuses first. When no adjacent ruler
    qualifies, the reason is a ruler's own (a missing or unreadable tire code first, the one the renter can fix), or no
    ruler within 0.6 m when none is on the frame; the extent is not consulted then, so a mask that was never asked for
    is never blamed. Otherwise a mark-side reason, else the first ruler's."""
    if crop_flag:
        return SizeEstimate(refused_reason=CROP_FLAG)
    if drawn_box_overlap:
        return SizeEstimate(refused_reason=DRAWN_BOX)
    usable = [rl for rl in rulers if isinstance(rl, Ruler) and panel_id in _adjacent_panels(rl.kind)]
    if not usable:
        failed = [rl for rl in rulers if isinstance(rl, str) and rl != NO_PLATE]
        fixable = [rl for rl in failed if rl in (TIRE_NOT_GIVEN, TIRE_UNREADABLE)]
        return SizeEstimate(refused_reason=(fixable or failed or [no_ruler_reason(th)])[0])
    tries = [size_estimate(extent, rl, mark_type=mark_type, panel_id=panel_id, frame_w=frame_w, th=th) for rl in usable]
    for s in tries:
        if s.inches is not None:
            return s
    reasons = [s.refused_reason for s in tries if s.refused_reason]
    near = [x for x in reasons if x != no_ruler_reason(th)]
    return SizeEstimate(refused_reason=(near or reasons or [no_ruler_reason(th)])[0])


def _adjacent_panels(kind: RulerKind) -> frozenset[PanelId]:
    return PLATE_PANELS if kind == RulerKind.PLATE else WHEEL_RULER_PANELS


# A run's marks and verdicts -------------------------------------------------------------------------------------------


def _read_list(path: Any, model: Any) -> list[Any]:
    return read_json_list(path, model) if path.is_file() and path.stat().st_size > 0 else []


def _video(run_id: str, role: str) -> VideoRecord | None:
    path = paths.videos_dir(run_id) / f"{role}.json"
    return read_json(path, VideoRecord) if path.is_file() else None


def _wheel_rulers(kf: Keyframe, tracks: list[PanelTrack], video: VideoRecord | None, *, tire_code: str | None, th: dict[str, Any]) -> list[Ruler | str]:
    """Every wheel track box on the keyframe's proxy frame, scaled to source pixels."""
    if video is None or video.proxy is None or video.ffprobe is None:
        return []
    sx = video.ffprobe.display_width / video.proxy.width
    sy = video.ffprobe.display_height / video.proxy.height
    out: list[Ruler | str] = []
    for t in tracks:
        if t.role != kf.role or t.noun != Noun.WHEEL:
            continue
        row = next((r for r in t.per_frame if r.frame == kf.f_proxy), None)
        if row is None:
            continue
        box = [row.box[0] * sx, row.box[1] * sy, row.box[2] * sx, row.box[3] * sy]
        out.append(wheel_ruler_from_box(box, row.c, touches_border=row.border_frac > 0, tire_code=tire_code, th=th, quantization_px=max(sx, sy), source_kf=kf.kf_id))
    return out


def run_tire_code(run_id: str) -> str | None:
    """The tire code typed for the run (RunManifest.tire_code, from walkaround run --tire-code), or None."""
    path = paths.run_json(run_id)
    try:
        code = json.loads(path.read_text(encoding="utf-8")).get("tire_code") if path.is_file() else None
    except (OSError, ValueError, AttributeError):
        return None
    return code if isinstance(code, str) and code.strip() else None


def _clip_times(run_id: str) -> dict[str, list[float]]:
    """mark id -> the source times of its Stays Put clip's frames (clips/<role>/staysput.json); empty for a walk
    without plans (a sample copy keeps none)."""
    from . import staysput  # here, not at import: staysput pulls in the SAM and video clients

    out: dict[str, list[float]] = {}
    for path in sorted(paths.run_dir(run_id).glob(f"clips/*/{staysput.PLAN_NAME}")):
        try:
            plans = read_json(path, staysput.RolePlans)
        except (OSError, ValueError):
            continue
        for plan in plans.clips:
            for obj in plan.objects:
                out[obj.mark_id] = [float(t) for t in plan.pts_src]
    return out


def _frame_at(times: list[float], t: float) -> int | None:
    """The clip frame shown at source time t, or None when t is outside the clip by more than half a frame."""
    if not times:
        return None
    half = 0.5 * statistics.median([b - a for a, b in zip(times, times[1:])]) if len(times) > 1 else 0.02
    j = min(range(len(times)), key=lambda i: (abs(times[i] - t), i))
    return j if abs(times[j] - t) <= half + 1e-6 else None


@dataclass(frozen=True)
class _View:
    """A frame a ruler may size a mark on: a keyframe of the mark's walk and the mark's extent on that frame."""

    kf: Keyframe
    extent: MarkExtent | None
    dt: float  # seconds from the mark's own keyframe


def _views(kf: Keyframe, mt: MarkTrack | None, times: list[float] | None, walk: list[Keyframe]) -> list[_View]:
    """Every keyframe of the walk inside the mark's Stays Put clip, with the mark's extent from its own track around
    that frame (ruler and mark on one frame). Without the clip's frame times: the mark's keyframe and the track's
    median extent over the clip."""
    if mt is None or mt.clip is None or not times:
        return [_View(kf, extent_from_track(mt), 0.0)]
    views = []
    for k in walk:
        j = _frame_at(times, k.pts)
        if j is not None:
            views.append(_View(k, extent_from_track(mt, min_points=NEAR_POINTS_MIN, frames=(j - NEAR_FRAMES, j + NEAR_FRAMES)), abs(k.pts - kf.pts)))
    if not any(v.kf.kf_id == kf.kf_id for v in views):
        views.insert(0, _View(kf, extent_from_track(mt), 0.0))
    return views


def _plate_tilt(blur: BlurRecord) -> float:
    return tilt_from_aspect(blur.plate_tilt_hw) if blur.plate_ruler_px is not None else math.inf


def _best_view(m: Mark, pairs: list[tuple[_View, Ruler | str]], kind: RulerKind, *, th: dict[str, Any], frame_w: int | None) -> tuple[SizeEstimate | None, str, bool]:
    """(size, "", True) from the most frontal ruler of one kind that sizes the mark (ties: the frame nearest the mark's
    keyframe); else (None, why, whether any ruler of that kind qualified on its own)."""
    usable = sorted(((r.tilt_deg, v.dt, v.kf.kf_id, i) for i, (v, r) in enumerate(pairs) if isinstance(r, Ruler)))
    refusals: list[str] = []
    for *_, i in usable:
        v, r = pairs[i]
        s = size_estimate(v.extent, r, mark_type=m.type, panel_id=m.panel_id, frame_w=frame_w, th=th)
        if s.inches is not None:
            return s, "", True
        refusals.append(str(s.refused_reason))
    if refusals:
        near = [x for x in refusals if x != no_ruler_reason(th)]
        return None, (near or refusals)[0], True
    failed = [(v, r) for v, r in pairs if isinstance(r, str)]
    if kind == RulerKind.PLATE:  # the most frontal plate seen says why
        seen = sorted(((_plate_tilt(v.kf.blur), v.dt, v.kf.kf_id, r) for v, r in failed if r != NO_PLATE))
        return None, seen[0][-1] if seen else NO_PLATE, False
    reasons = [r for _, _, r in sorted((v.dt, v.kf.kf_id, r) for v, r in failed)]
    fixable = [r for r in reasons if r in (TIRE_NOT_GIVEN, TIRE_UNREADABLE)]
    return None, (fixable or reasons or [NO_WHEEL])[0], False


def _note(kind: RulerKind, reason: str, th: dict[str, Any]) -> str:
    """Why a ruler next to the mark did not size it, as the sheet prints it after the ruler used."""
    name = "the license plate" if kind == RulerKind.PLATE else "the wheel"
    if reason == no_ruler_reason(th):
        return f"{name} is more than {float(th['ruler']['ruler_max_distance_m']):g} m from this mark"
    if reason in (NO_PLATE, NO_WHEEL) or reason.startswith(name):
        return reason
    return f"{name}: {reason}"


def _size_mark(m: Mark, kf: Keyframe, views: list[_View], tracks: list[PanelTrack], video: VideoRecord | None, *, tire_code: str | None, th: dict[str, Any], frame_w: int | None) -> SizeEstimate:
    """5.12 for one walk mark over its candidate frames: a crop flag on the mark's own keyframe refuses first; then the
    plate (plate-adjacent panels) and the wheel (wheel, fender and lower-bumper marks) each on their most frontal view,
    the plate first. The refusal follows best_size: with a ruler that qualified, the mark side speaks; else a tire
    code the renter can fix, the plate's reason, the wheel's, or no ruler near."""
    if kf.conditions.any_flag():
        return SizeEstimate(refused_reason=CROP_FLAG)
    kinds = [k for k, ok in ((RulerKind.PLATE, m.panel_id in PLATE_PANELS), (RulerKind.WHEEL, m.panel_id in WHEEL_RULER_PANELS)) if ok]
    if not kinds:
        return SizeEstimate(refused_reason=no_ruler_reason(th))
    sized: dict[RulerKind, SizeEstimate] = {}
    why: dict[RulerKind, tuple[str, bool]] = {}
    for kind in kinds:
        pairs: list[tuple[_View, Ruler | str]] = []
        for v in views:
            found = [plate_ruler(v.kf.blur, th=th, source_kf=v.kf.kf_id)] if kind == RulerKind.PLATE else _wheel_rulers(v.kf, tracks, video, tire_code=tire_code, th=th)
            pairs += [(v, r) for r in found]
        size, reason, qualified = _best_view(m, pairs, kind, th=th, frame_w=frame_w)
        if size is not None:
            sized[kind] = size
        else:
            why[kind] = (reason, qualified)
    used = next((k for k in kinds if k in sized), None)
    if used is not None:
        return sized[used].model_copy(update={"ruler_notes": [_note(k, why[k][0], th) for k in kinds if k in why]})
    qualified = [why[k][0] for k in kinds if why[k][1]]
    if qualified:
        near = [x for x in qualified if x != no_ruler_reason(th)]
        main = (near or qualified)[0]
    else:
        fixable = [why[k][0] for k in kinds if why[k][0] in (TIRE_NOT_GIVEN, TIRE_UNREADABLE)]
        failed = [why[k][0] for k in kinds if why[k][0] not in (NO_PLATE, NO_WHEEL)]
        main = (fixable or failed or [no_ruler_reason(th)])[0]
    notes = [_note(k, why[k][0], th) for k in kinds if why[k][0] != main]
    return SizeEstimate(refused_reason=main, ruler_notes=notes)


def size_marks(run_id: str, *, th: dict[str, Any], tire_code: str | None = None) -> dict[str, SizeEstimate]:
    """One SizeEstimate per mark of marks.json: the plate (plate-adjacent panels) and the wheels (wheel, fender and
    lower-bumper marks; they need the tire code: tire_code, else the run's own) on the most frontal of the walk's
    keyframes inside the mark's Stays Put clip, the mark's extent from its own track on that frame. A mark with no
    track is refused, never sized from its Spark box."""
    code = run_tire_code(run_id) if tire_code is None else tire_code
    marks: list[Mark] = _read_list(paths.marks_json(run_id), Mark)
    keyframes = {k.kf_id: k for k in _read_list(paths.keyframes_json(run_id), Keyframe)}
    tracks: list[PanelTrack] = _read_list(paths.tracks_json(run_id), PanelTrack)
    mark_tracks = {mt.mark_id: mt for mt in _read_list(paths.mark_tracks_json(run_id), MarkTrack)}
    times = _clip_times(run_id)
    walks: dict[str, list[Keyframe]] = {}
    for k in sorted(keyframes.values(), key=lambda k: (k.pts, k.kf_id)):
        walks.setdefault(k.role, []).append(k)
    videos: dict[str, VideoRecord | None] = {}
    out: dict[str, SizeEstimate] = {}
    for m in marks:
        kf = keyframes.get(m.kf_id)
        if kf is None:
            out[m.mark_id] = SizeEstimate(refused_reason="the mark's keyframe is not recorded")
            continue
        if kf.role not in videos:
            videos[kf.role] = _video(run_id, kf.role)
        video = videos[kf.role]
        frame_w = video.ffprobe.display_width if video is not None and video.ffprobe is not None else None
        views = _views(kf, mark_tracks.get(m.mark_id), times.get(m.mark_id), walks.get(kf.role, []))
        out[m.mark_id] = _size_mark(m, kf, views, tracks, video, tire_code=code, th=th, frame_w=frame_w)
    return out


def attach_sizes(verdicts: list[Verdict], sizes: dict[str, SizeEstimate], marks: dict[str, Mark]) -> list[Verdict]:
    """Verdict.size from the return-side mark of a verdict (the mark as it is now), else its first mark. A
    not-comparable verdict names every mark of the panel and gets no size."""
    out: list[Verdict] = []
    for v in verdicts:
        size = None
        if len(v.mark_ids) <= 2 and v.mark_ids and v.rule_id != "L6":
            ordered = sorted(v.mark_ids, key=lambda i: 0 if (marks.get(i) is not None and marks[i].role == "return") else 1)
            size = sizes.get(ordered[0])
        out.append(v.model_copy(update={"size": size}))
    return out


def size_verdicts(run_id: str, *, th: dict[str, Any], tire_code: str | None = None) -> list[Verdict]:
    """verdicts.json with a size on every mark verdict, written back; the same on a rebuild since every input is a
    frozen file (the tire code too: tire_code, else the run manifest's)."""
    verdicts: list[Verdict] = _read_list(paths.verdicts_json(run_id), Verdict)
    if not verdicts:
        return []
    marks = {m.mark_id: m for m in _read_list(paths.marks_json(run_id), Mark)}
    sized = attach_sizes(verdicts, size_marks(run_id, th=th, tire_code=tire_code), marks)
    write_json(paths.verdicts_json(run_id), sized)
    return sized


__all__ = [
    "CAMERA_HFOV_DEG",
    "MarkExtent",
    "NOT_A_US_PLATE",
    "NO_MARK_MASK",
    "NO_PLATE",
    "NO_WHEEL",
    "PLATE_HW_MIN",
    "PLATE_PANELS",
    "Ruler",
    "TIRE_NOT_GIVEN",
    "TIRE_UNREADABLE",
    "WHEEL_RULER_PANELS",
    "attach_sizes",
    "best_size",
    "camera_distance_in",
    "compare_hertz",
    "ellipse_axes",
    "extent_from_mask",
    "extent_from_track",
    "no_ruler_reason",
    "normalize_tire_code",
    "plate_ruler",
    "run_tire_code",
    "size_estimate",
    "size_marks",
    "size_verdicts",
    "threshold_in",
    "threshold_type_for",
    "tilt_from_aspect",
    "tire_diameter_in",
    "wheel_ruler",
    "wheel_ruler_from_box",
    "wheel_scale",
]
