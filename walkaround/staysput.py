"""Stays Put (docs/ARCHITECTURE.md 5.9, docs/CONTRACTS.md staysput): paint stays on its panel while a reflection
slides with the camera.

Per walk, every mark and one reflection control get a square window that follows the mark's panel box through a 3 s
clip (91 frames at 30 fps, 73 at 24) cut from the original around the mark's keyframe; marks whose windows overlap
share one clip, up to 16. A walk whose plate or face lane did not complete gets no clip. Each clip's prompt frame is
checked for plates and faces on the unblurred pixels first, and a find the walk's own plate and face tracks do not
cover refuses the clip; then every frame is pixelated and encoded. SAM 3.1 tracks each mark box as its own object
(propagation both) and code measures how far the track moves in the panel's whitened frame: fixed, sliding,
inconclusive or no_track. The curvature and highlight caps come from the keyframe crop alone, for every mark, clip or
not. Nothing here raises a verdict. match.py may turn these rows into K6, and only while thresholds
staysput.cap_enabled is on; that check lives in match, not here.

What a walk asked is frozen in clips/<role>/staysput.json (windows, panel boxes per clip frame, prompts, the image
check, the clip record, the lane id, the marks that got no clip). derived/mark_tracks.json is recomputed from those
plans, the stored clip lanes and the keyframe crops by one function, live and in rebuild_staysput, so a rebuild is
byte-identical and needs neither the original nor the network.
"""

from __future__ import annotations

import asyncio
import bisect
import hashlib
import json
import logging
import math
import os
import shutil
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import cv2
import numpy as np
from PIL import Image
from pydantic import Field

from . import contracts, geometry, paths, privacy, sam_client, video
from .config import redact
from .contracts import (
    Box4,
    ClipRecord,
    Confidence,
    Contract,
    Crop,
    Diagnostic,
    IndexFrame,
    Keyframe,
    LaneRecord,
    LaneStatus,
    Mark,
    MarkKind,
    MarkTrack,
    Moments,
    NeighbourTest,
    Noun,
    PanelId,
    PanelTrack,
    Point2,
    Role,
    Sha256,
    Size2,
    StaysPutNeighbour,
    StaysPutVerdict,
    TrackPoint,
    VideoRecord,
)

log = logging.getLogger(__name__)

BoxF = list[float]  # x1, y1, x2, y2 in pixels, right and bottom half-open

REUSED_CODE = "reused_identical_request"  # the orchestrator's diagnostic for an answer reused within a run
ANSWERED = (LaneStatus.COMPLETED, LaneStatus.ZERO_MATCHES)
KEPT = ("answered", "refused")  # a resumed walk never asks these again
CURVED_PANELS = frozenset(
    {
        PanelId.FRONT_BUMPER,
        PanelId.REAR_BUMPER,
        PanelId.LEFT_FRONT_FENDER,
        PanelId.LEFT_REAR_FENDER,
        PanelId.RIGHT_FRONT_FENDER,
        PanelId.RIGHT_REAR_FENDER,
        PanelId.HOOD,
    }
)
CONFIDENCE_RANK = {Confidence.HIGH: 0, Confidence.MEDIUM: 1, Confidence.LOW: 2}
SRC_REFERENCE_EDGE = 3840  # the "at 4K" of min_bbox_px_at_4k and the privacy pixel sizes
PLAN_NAME = "staysput.json"
# Leaves not yet in config/thresholds.json (staysput section); th wins once the integrator adds them. Estimates.
# TODO(D1 E5): recalibrate on dark paint with a real reflection (lead note 2 on docs/D1_MEMO.md) before these, tau_fix
# or the b floors are trusted; until then thresholds staysput.cap_enabled false keeps K6 off in match.
DEFAULTS: dict[str, float] = {
    "min_drift_frames": 10,  # whitened frames a drift needs before it is a number
    "max_clips_per_walk": 16,  # section 5 step 9 expects about 12 clips on a pickup walk, 10 on a return, plus a control
    "window_mark_margin": 1.5,  # a window side is at least this many times the longest side of its marks
    "control_blob_min_px": 9,  # crop pixels
    "control_blob_max_frac": 0.15,  # of the hood mask: a larger glare region is paint in sun, not a highlight
    "check_box_covered_min": 0.9,  # share of a prompt-frame check box the walk's grown plate and face track boxes cover
    "clip_frames_max": 181,  # 3 s at 60 fps; a faster walk gets a shorter clip
    # Stays Put by neighbours (docs/STAYS_PUT_ANCHORS.md); thresholds.json carries the same leaves with their sources.
    "neighbours_max": 12,
    "neighbours_target": 6,
    "neighbours_min": 4,
    "neighbour_track_share_min": 0.8,
    "neighbour_fixed_below": 1.5,
    "neighbour_sliding_above": 2.5,
    "neighbour_near_half_window_s": 0.5,
    "neighbour_window_margin": 1.15,
    "neighbour_window_max_panel_frac": 1.5,
    "neighbour_window_max_growth": 1000.0,  # no growth cap unless thresholds.json sets one
    "neighbour_mark_min_clip_px": 16,
    "neighbour_part_max_window_frac": 0.5,
    "neighbour_residual_floor_px": 0.5,
    "neighbour_dedupe_iou": 0.5,
    "neighbour_handle_half_w_frac": 0.08,
    "neighbour_handle_half_h_frac": 0.02,
}
NEIGHBOUR_RULE = 1  # the neighbour test as built 10/31; a plan without it was made before the test existed
RIGID_MARK_KINDS = frozenset({MarkKind.SCRATCH, MarkKind.SCUFF, MarkKind.CHIP, MarkKind.DENT, MarkKind.CRACK, MarkKind.DIRT})
HANDLE_KIND = "door handle"  # from the keyframe's handle check (a point), never a lane
NEIGHBOUR_NOUNS: tuple[Noun, ...] = (
    Noun.SIDE_MIRROR,
    Noun.WHEEL,
    Noun.HEADLIGHT,
    Noun.TAIL_LIGHT,
    Noun.LICENSE_PLATE,
    Noun.CAR_WINDOW,
    Noun.WINDSHIELD,
    Noun.REAR_WINDOW,
)
NEIGHBOUR_KINDS: tuple[str, ...] = ("mark", HANDLE_KIND, *(n.value for n in NEIGHBOUR_NOUNS))  # the tie-break order


def _sp(th: dict[str, Any], key: str) -> float:
    return float(th["staysput"].get(key, DEFAULTS[key]))


# ----------------------------------------------------------------------------------------------------------------
# The plan file: clips/<role>/staysput.json
# ----------------------------------------------------------------------------------------------------------------


class PlanObject(Contract):
    """One mark (or the reflection control) of a clip, as sent: object_id is the SAM object id."""

    object_id: str
    mark_id: str
    type: MarkKind
    is_control: bool = False
    box_src: Box4  # at the prompt frame, source pixels
    box_clip: Box4  # the same box in clip pixels, before the prompt padding
    point_crop: Point2 | None = None  # keyframe crop pixels, for the curvature cap
    bbox_crop: Box4 | None = None  # keyframe crop pixels, for the highlight cap


class PlanNeighbour(Contract):
    """A rigid neighbour prompted in the clip after the marks (docs/STAYS_PUT_ANCHORS.md): a lane part at the keyframe,
    the keyframe's door handle, or a mark that is tested in another clip. object_id is the SAM object id."""

    object_id: str
    kind: str  # "mark", "door handle" or a lane noun
    ref: str  # the mark id, the track id or "<kf_id>_handle"
    box_src: Box4  # at the prompt frame, source pixels
    box_clip: Box4  # the same box in clip pixels, before the prompt padding


class ClipCheck(Contract):
    """The plate and face image checks on the unblurred prompt frame, boxes in clip pixels."""

    window_sha256: Sha256
    plate_boxes: list[Box4] = Field(default_factory=list)
    face_boxes: list[Box4] = Field(default_factory=list)
    request_ids: list[str] = Field(default_factory=list)
    reused_from: str | None = None  # "<role>/<clip_id>" when the same pixels were checked earlier in the run


class ClipPlan(Contract):
    """One clip: what was cut, what was asked, and how it ended. refused: the check found a plate or face the walk's
    tracks miss, so nothing was cut; a resumed walk does not ask again."""

    clip_id: str
    role: Role
    kf_id: str
    panel_id: PanelId
    lane_id: str
    src_size: Size2
    proxy_size: Size2
    size: int  # clip side in pixels
    prompt_frame: int
    f0_src: int
    pts_src: list[float]  # source pts of each clip frame
    window_px: float
    windows_src: list[Box4]
    panel_src: list[Optional[Box4]]  # panel box per clip frame in source pixels; None where unknown
    objects: list[PlanObject]
    neighbours: list[PlanNeighbour] = Field(default_factory=list)  # prompted after the objects, ids continuing theirs
    neighbour_rule: int | None = None  # NEIGHBOUR_RULE when the neighbour test was planned; None on older plans
    prompt_text: str = ""
    check: ClipCheck | None = None
    clip: ClipRecord | None = None
    status: Literal["planned", "answered", "failed", "refused"] = "planned"
    error: str | None = None
    lane_reused_from: str | None = None  # "<role>/<lane_id>" when the same clip bytes and prompt were asked earlier


class SkippedMark(Contract):
    """A mark that got no clip, and what its not_run row needs for the keyframe caps."""

    mark_id: str
    kf_id: str
    panel_id: PanelId
    reason: str
    is_control: bool = False  # a mark Spark typed reflection
    point_crop: Point2 | None = None
    bbox_crop: Box4 | None = None


class RolePlans(Contract):
    role: Role
    complete: bool = False
    clips: list[ClipPlan] = Field(default_factory=list)
    unclipped: list[SkippedMark] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)  # walk-level notes: no proxy, a control that got no clip
    earlier_request_ids: list[str] = Field(default_factory=list)  # checks and lanes of plans a later derive replaced


def _role_key(role: str) -> tuple[int, int, str]:
    if role in ("pickup", "return"):
        return (0 if role == "pickup" else 1, 0, role)
    kind, _, num = role.partition("_")
    return (2 if kind == "topup" else 3, int(num) if num.isdigit() else 0, role)


def _plan_file(run_id: str, role: str) -> Path:
    return paths.run_dir(run_id) / "clips" / role / PLAN_NAME


def _read_plans(run_id: str, role: str) -> RolePlans | None:
    path = _plan_file(run_id, role)
    return contracts.read_json(path, RolePlans) if path.is_file() else None


def _write_atomic(path: Path, value: Any) -> None:
    """contracts.write_json's bytes through a temp file and a rename, so a crash never leaves half a file."""
    data = contracts.json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _write_plans(run_id: str, plans: RolePlans) -> None:
    _write_atomic(paths.clips_dir(run_id, plans.role) / PLAN_NAME, plans)


def planned_roles(run_id: str) -> list[str]:
    """Roles with a stored plan file, pickup first."""
    root = paths.run_dir(run_id) / "clips"
    roles = [p.parent.name for p in root.glob(f"*/{PLAN_NAME}")] if root.is_dir() else []
    return sorted(roles, key=_role_key)


def role_done(run_id: str, role: str) -> bool:
    """True once derive_staysput finished this walk (the orchestrator's done check, one walk at a time)."""
    plans = _read_plans(run_id, role)
    return plans is not None and plans.complete


# ----------------------------------------------------------------------------------------------------------------
# Small geometry
# ----------------------------------------------------------------------------------------------------------------


def _us(t: float) -> int:
    return int(round(float(t) * 1_000_000))


def _on_screen(times_us: list[int], t: float) -> int:
    """The source frame on screen at t: the last frame whose pts is at or before t (frame 0 before the first)."""
    return max(0, bisect.bisect_right(times_us, _us(t)) - 1)


def _fps(times: list[float]) -> float | None:
    """Mean frame rate over the times; None for fewer than two distinct times."""
    if len(times) < 2 or times[-1] <= times[0]:
        return None
    return (len(times) - 1) / (times[-1] - times[0])


def _center(b: Any) -> tuple[float, float]:
    return ((float(b[0]) + float(b[2])) / 2.0, (float(b[1]) + float(b[3])) / 2.0)


def _square(c: tuple[float, float], side: float) -> BoxF:
    half = side / 2.0
    return [c[0] - half, c[1] - half, c[0] + half, c[1] + half]


def _at_least(box: Any, min_px: float) -> BoxF:
    """The box grown about its centre to at least min_px on each side."""
    cx, cy = _center(box)
    w = max(float(box[2]) - float(box[0]), min_px)
    h = max(float(box[3]) - float(box[1]), min_px)
    return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]


def _overlap(a: BoxF, b: BoxF) -> float:
    """Intersection over the smaller box's area."""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return iw * ih / small if small > 0 else 0.0


def _carry_point(p: tuple[float, float], box_from: Any, box_to: Any) -> tuple[float, float] | None:
    """p moved by the box-to-box affine (translation and per-axis scale) that maps box_from onto box_to."""
    fw, fh = box_from[2] - box_from[0], box_from[3] - box_from[1]
    if fw <= 0 or fh <= 0:
        return None
    sx, sy = (box_to[2] - box_to[0]) / fw, (box_to[3] - box_to[1]) / fh
    return (box_to[0] + (p[0] - box_from[0]) * sx, box_to[1] + (p[1] - box_from[1]) * sy)


def _in_frame(b: Any, size: Any, tol: float) -> bool:
    """The box keeps one proxy pixel off every frame edge (a box cut by the edge is not the panel's extent)."""
    return b[0] >= tol and b[1] >= tol and b[2] <= size[0] - tol and b[3] <= size[1] - tol


def _aspect(b: Any) -> float:
    h = b[3] - b[1]
    return (b[2] - b[0]) / h if h > 0 else 0.0


def _sign(v: float) -> int:
    return 1 if v > 0 else -1 if v < 0 else 0


def _min_box_px(th: dict[str, Any], src_size: Any) -> float:
    return float(th["staysput"]["min_bbox_px_at_4k"]) * max(src_size) / SRC_REFERENCE_EDGE


def _cut_window(frame: np.ndarray, window: BoxF, size: int) -> np.ndarray:
    """The source window resampled onto size x size the way video.cut_clip does; off-frame pixels are black."""
    x1, y1, x2, _ = (float(v) for v in window)
    s = (x2 - x1) / size
    matrix = np.array([[s, 0.0, x1 + 0.5 * s - 0.5], [0.0, s, y1 + 0.5 * s - 0.5]], dtype=np.float64)
    interp = cv2.INTER_LANCZOS4 if s < 1.0 else cv2.INTER_LINEAR
    return cv2.warpAffine(frame, matrix, (size, size), flags=interp | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def _to_clip(box: Any, window: Any, size: int) -> BoxF:
    s = size / (window[2] - window[0])
    return [(box[0] - window[0]) * s, (box[1] - window[1]) * s, (box[2] - window[0]) * s, (box[3] - window[1]) * s]


def _to_src(box: Any, window: Any, size: int) -> BoxF:
    s = (window[2] - window[0]) / size
    return [window[0] + box[0] * s, window[1] + box[1] * s, window[0] + box[2] * s, window[1] + box[3] * s]


def _pixel_span(box: Any, size: int) -> tuple[slice, slice] | None:
    """The whole clip pixels a box touches, clamped to the clip; None when it misses the clip."""
    x1, y1 = max(0, int(math.floor(box[0]))), max(0, int(math.floor(box[1])))
    x2, y2 = min(size, int(math.ceil(box[2]))), min(size, int(math.ceil(box[3])))
    return (slice(y1, y2), slice(x1, x2)) if x2 > x1 and y2 > y1 else None


# ----------------------------------------------------------------------------------------------------------------
# One walk's inputs
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class _Walk:
    role: str
    src: Path
    src_size: tuple[int, int]
    proxy_size: tuple[int, int]
    proxy_fps: float
    proxy_pts: list[float]
    tracks: dict[str, PanelTrack]
    privacy_tracks: list[PanelTrack]
    keyframes: dict[str, Keyframe]
    marks: list[Mark]
    times: list[float] = field(default_factory=list)  # source frame times, read only when a clip is planned


@dataclass(frozen=True)
class _Item:
    """A mark or the control before grouping."""

    mark_id: str
    kf_id: str
    panel_id: PanelId
    type: MarkKind
    is_control: bool
    box_src: tuple[float, float, float, float]
    point_crop: tuple[float, float] | None
    bbox_crop: tuple[float, float, float, float] | None
    rank: int  # 0 first under the clip cap; -1 is the glare control, which is no mark


def _read_list(path: Path, model: Any) -> list[Any]:
    return contracts.read_json_list(path, model) if path.is_file() else []


def _load_walk(run_id: str, role: str) -> _Walk | None:
    vpath = paths.video_json(run_id, role)
    if not vpath.is_file():
        return None
    record = contracts.read_json(vpath, VideoRecord)
    if record.proxy is None or record.ffprobe is None:
        return None
    pts = [float(v) for v in json.loads((paths.run_dir(run_id) / record.proxy.pts_map_path).read_text(encoding="utf-8"))]
    tracks = [t for t in _read_list(paths.tracks_json(run_id), PanelTrack) if t.role == role]
    return _Walk(
        role=role,
        src=paths.run_dir(run_id) / "quarantine" / f"{role}.orig",  # paths.quarantine_original without its mkdir
        src_size=(int(record.ffprobe.display_width), int(record.ffprobe.display_height)),
        proxy_size=(int(record.proxy.width), int(record.proxy.height)),
        proxy_fps=float(record.proxy.fps),
        proxy_pts=pts,
        tracks={t.track_id: t for t in tracks},
        privacy_tracks=[t for t in tracks if t.noun in contracts.PRIVACY_NOUNS],
        keyframes={k.kf_id: k for k in _read_list(paths.keyframes_json(run_id), Keyframe) if k.role == role},
        marks=[m for m in _read_list(paths.marks_json(run_id), Mark) if m.role == role],
    )


def _privacy_gap(run_id: str, role: str) -> str | None:
    """Why no clip of this walk can be blurred: its plate or face lane is missing or did not complete, so the boxes to
    blur are unknown (the orchestrator's rule for the display proxy). None when both lanes answered."""
    for noun in contracts.PRIVACY_NOUNS:
        path = paths.run_dir(run_id) / "sam" / role / f"{paths.lane_id(noun)}.summary.json"
        try:
            status = contracts.read_json(path, LaneRecord).status if path.is_file() else None
        except (OSError, ValueError):
            status = None
        if status not in ANSWERED:
            said = status.value if status is not None else "missing"
            return f"its {noun.value} lane is {said}, so a clip cannot be blurred"
    return None


def _why_not(kf: Keyframe | None, walk: _Walk) -> str | None:
    if kf is None:
        return "its keyframe is missing"
    if kf.crop is None or kf.moments is None:
        return "its keyframe has no crop or no moments"
    if kf.track_id not in walk.tracks:
        return "its panel track is missing"
    return None


def _raw_box(mark: Mark) -> tuple[BoxF, float | None] | None:
    """The mark's source box as found, with its long side (None for a point mark of unknown size)."""
    if mark.bbox_src is not None:
        b = [float(v) for v in mark.bbox_src]
        return b, max(b[2] - b[0], b[3] - b[1])
    if mark.point_src is not None:
        size = None if mark.size_px_src is None else float(mark.size_px_src)
        half = (size or 0.0) / 2.0
        x, y = (float(v) for v in mark.point_src)
        return [x - half, y - half, x + half, y + half], size
    return None


def _mark_box(mark: Mark, min_px: float) -> tuple[BoxF | None, str | None]:
    """(the box a clip prompts with, grown to min_px a side, None), or (None, why the mark gets no clip): no source
    position, or a long side under min_px (section 14 item 17: the mark keeps its deterministic label)."""
    raw = _raw_box(mark)
    if raw is None:
        return None, "the mark has no source position"
    box, long_side = raw
    if long_side is not None and long_side < min_px:
        return None, f"too small for a clip: {long_side:.1f} px on its long side, under {min_px:.1f} px (section 14 item 17)"
    return _at_least(box, min_px), None


def _crop_refs(mark: Mark) -> tuple[list[float] | None, list[float] | None]:
    """(point, box) in keyframe crop pixels; the point falls back to the box centre."""
    point = mark.point_crop if mark.point_crop is not None else (list(_center(mark.bbox_crop)) if mark.bbox_crop is not None else None)
    return (None if point is None else [float(point[0]), float(point[1])]), (None if mark.bbox_crop is None else [float(v) for v in mark.bbox_crop])


def _item(mark: Mark, box: BoxF) -> _Item:
    point, bbox = _crop_refs(mark)
    return _Item(
        mark_id=mark.mark_id,
        kf_id=mark.kf_id,
        panel_id=mark.panel_id,
        type=mark.type,
        is_control=mark.type is MarkKind.REFLECTION,
        box_src=(box[0], box[1], box[2], box[3]),
        point_crop=None if point is None else (point[0], point[1]),
        bbox_crop=None if bbox is None else (bbox[0], bbox[1], bbox[2], bbox[3]),
        rank=CONFIDENCE_RANK.get(mark.confidence, 2),
    )


def _unclipped(mark: Mark, reason: str) -> SkippedMark:
    point, bbox = _crop_refs(mark)
    return SkippedMark(
        mark_id=mark.mark_id, kf_id=mark.kf_id, panel_id=mark.panel_id, reason=reason, is_control=mark.type is MarkKind.REFLECTION, point_crop=point, bbox_crop=bbox
    )


def _skip(it: _Item, reason: str) -> SkippedMark:
    return SkippedMark(
        mark_id=it.mark_id,
        kf_id=it.kf_id,
        panel_id=it.panel_id,
        reason=reason,
        is_control=it.is_control,
        point_crop=None if it.point_crop is None else list(it.point_crop),
        bbox_crop=None if it.bbox_crop is None else list(it.bbox_crop),
    )


def _items(run_id: str, walk: _Walk, th: dict[str, Any]) -> tuple[list[_Item], list[SkippedMark]]:
    """Every mark of the walk that can get a clip, plus the reflection control when no mark is a reflection; and the
    marks that cannot, with the reason."""
    min_px = _min_box_px(th, walk.src_size)
    items: list[_Item] = []
    unclipped: list[SkippedMark] = []
    for mark in sorted(walk.marks, key=lambda m: m.mark_id):
        reason = _why_not(walk.keyframes.get(mark.kf_id), walk)
        box = None
        if reason is None:
            box, reason = _mark_box(mark, min_px)
        if box is None:
            unclipped.append(_unclipped(mark, reason or "no clip"))
            continue
        items.append(_item(mark, box))
    if not any(it.is_control for it in items) and th["staysput"].get("reflection_control_from_hood_glare", True):
        control = _glare_control(run_id, walk, th, min_px)
        if control is not None:
            items.append(control)
    return items, unclipped


# ----------------------------------------------------------------------------------------------------------------
# The reflection control and the photometric reads (keyframe crop and mask)
# ----------------------------------------------------------------------------------------------------------------


def _kf_pixels(run_id: str, kf: Keyframe, cache: dict[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None]:
    """(panel mask, HSV crop) of a keyframe, both in crop pixels; None for a file that is missing or does not fit."""
    if kf.kf_id not in cache:
        root = paths.run_dir(run_id)
        mask = hsv = None
        if kf.crop is not None and (root / kf.crop.path).is_file():
            rgb = np.asarray(Image.open(root / kf.crop.path).convert("RGB"))
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        if kf.mask_png and (root / kf.mask_png).is_file():
            mask = np.asarray(Image.open(root / kf.mask_png).convert("L")) > 0
        if mask is not None and hsv is not None and mask.shape != hsv.shape[:2]:
            mask = None
        cache[kf.kf_id] = (mask, hsv)
    return cache[kf.kf_id]


def _region(bbox_crop: Any, point_crop: Any, shape: tuple[int, ...]) -> tuple[slice, slice] | None:
    if bbox_crop is not None:
        b = bbox_crop
    elif point_crop is not None:
        b = [point_crop[0] - 2, point_crop[1] - 2, point_crop[0] + 2, point_crop[1] + 2]
    else:
        return None
    h, w = shape[:2]
    x1, y1 = max(0, int(math.floor(b[0]))), max(0, int(math.floor(b[1])))
    x2, y2 = min(w, int(math.ceil(b[2]))), min(h, int(math.ceil(b[3])))
    return (slice(y1, y2), slice(x1, x2)) if x2 > x1 and y2 > y1 else None


def _highlight(mask: np.ndarray, hsv: np.ndarray, bbox_crop: Any, point_crop: Any, th: dict[str, Any]) -> bool:
    """5.9 item 6's highlight photometry: the region's mean V above the panel's highlight_v_percentile and its mean S
    below the panel's highlight_s_percentile."""
    region = _region(bbox_crop, point_crop, mask.shape)
    if region is None:
        return False
    sp = th["staysput"]
    v, s = hsv[..., 2], hsv[..., 1]
    v_top = float(np.percentile(v[mask], float(sp["highlight_v_percentile"])))
    s_low = float(np.percentile(s[mask], float(sp["highlight_s_percentile"])))
    return float(v[region].mean()) > v_top and float(s[region].mean()) < s_low


def glare_blob(mask: np.ndarray | None, hsv: np.ndarray | None, th: dict[str, Any]) -> tuple[BoxF, list[float]] | None:
    """The reflection control's blob, as (box, centroid) in crop pixels: of the connected blobs of the 5.3 glare rule
    inside the panel mask (V > glare_v_min, S < glare_s_max) that are not too big to be a highlight, the largest whose
    box also passes 5.9 item 6's highlight photometry. On white paint the glare rule alone picks plain paint, which
    that test refuses (D1 note 2), and then there is no control."""
    if mask is None or hsv is None or not mask.any():
        return None
    p = th["panels"]
    glare = (hsv[..., 2] > p["glare_v_min"]) & (hsv[..., 1] < p["glare_s_max"]) & mask
    if not glare.any():
        return None
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(glare.astype(np.uint8), connectivity=8)
    area_min, area_max = _sp(th, "control_blob_min_px"), _sp(th, "control_blob_max_frac") * float(mask.sum())
    blobs = [i for i in range(1, count) if area_min <= int(stats[i, cv2.CC_STAT_AREA]) <= area_max]
    for i in sorted(blobs, key=lambda i: (-int(stats[i, cv2.CC_STAT_AREA]), i)):
        x, y, w, h = (int(v) for v in stats[i, :4])
        box = [float(x), float(y), float(x + w), float(y + h)]
        if _highlight(mask, hsv, box, None, th):
            cx, cy = (float(v) for v in centroids[i])
            return box, [cx, cy]
    return None


def _glare_control(run_id: str, walk: _Walk, th: dict[str, Any], min_px: float) -> _Item | None:
    """5.9 item 4: the control blob of the walk's best-scored hood crop that has one, flagged as the control."""
    hoods = [kf for kf in walk.keyframes.values() if kf.panel_id is PanelId.HOOD and kf.mask_png and _why_not(kf, walk) is None]
    cache: dict[str, Any] = {}
    for kf in sorted(hoods, key=lambda k: (-k.score.total, k.kf_id)):
        blob = glare_blob(*_kf_pixels(run_id, kf, cache), th)
        if blob is None or kf.crop is None:
            continue
        bbox_crop, point_crop = blob
        box = _at_least(geometry.box_crop_to_src(bbox_crop, kf.crop), min_px)
        return _Item(
            mark_id=f"{kf.kf_id}_ctrl",
            kf_id=kf.kf_id,
            panel_id=kf.panel_id,
            type=MarkKind.REFLECTION,
            is_control=True,
            box_src=(box[0], box[1], box[2], box[3]),
            point_crop=(point_crop[0], point_crop[1]),
            bbox_crop=(bbox_crop[0], bbox_crop[1], bbox_crop[2], bbox_crop[3]),
            rank=-1,
        )
    return None


def _caps(run_id: str, panel_id: PanelId, point_crop: Any, bbox_crop: Any, kf: Keyframe | None, th: dict[str, Any], cache: dict[str, Any]) -> tuple[bool, bool]:
    """5.9 item 6 on the keyframe crop and mask alone, so no model failure can drop a cap: (curvature, highlight)."""
    if kf is None or kf.crop is None:
        return False, False
    mask, hsv = _kf_pixels(run_id, kf, cache)
    if mask is None or not mask.any():
        return False, False
    curvature = False
    if panel_id in CURVED_PANELS and point_crop is not None and kf.moments is not None:
        sign = kf.front_sign if kf.front_sign in (1, -1) else 1
        try:
            curvature = geometry.boundary_distance_units(list(point_crop), mask, kf.moments, sign) < float(th["staysput"]["curvature_boundary_units"])
        except ValueError:
            curvature = False
    highlight = hsv is not None and _highlight(mask, hsv, bbox_crop, point_crop, th)
    return bool(curvature), bool(highlight)


def caps_for(run_id: str, plan: ClipPlan, obj: PlanObject, kf: Keyframe, th: dict[str, Any], cache: dict[str, Any]) -> tuple[bool, bool]:
    """5.9 item 6 on the keyframe crop: (curvature cap, highlight cap). Curvature: a bumper, fender or hood mark
    within curvature_boundary_units of its mask boundary. Highlight: the mark's mean V above the panel's
    highlight_v_percentile and its mean S below the panel's highlight_s_percentile."""
    return _caps(run_id, plan.panel_id, obj.point_crop, obj.bbox_crop, kf, th, cache)


# ----------------------------------------------------------------------------------------------------------------
# Windows, spans and prompts (5.9 items 1 to 3)
# ----------------------------------------------------------------------------------------------------------------


def clip_frames(times: list[float], th: dict[str, Any]) -> int:
    """Source frames per clip: 2 x round(clip_half_window_s x fps) + 1 with fps from the original's frame times (91 at
    30 fps, 89 at 29.6, 73 at 24), at most clip_frames_max; clip_frames when the rate is unknown."""
    fps = _fps(times)
    if fps is None:
        return int(th["media"]["clip_frames"])
    n = 2 * int(round(float(th["media"]["clip_half_window_s"]) * fps)) + 1
    return max(3, min(n, int(_sp(th, "clip_frames_max"))))


def _span(times_us: list[int], t: float, frames: int) -> tuple[int, int, int] | None:
    """(f0, n, prompt frame) for n source frames around the frame on screen at t, slid inside the walk at its ends."""
    total = len(times_us)
    n = min(int(frames), total)
    if n < 2:
        return None
    centre = _on_screen(times_us, t)
    f0 = min(max(centre - n // 2, 0), total - n)
    return f0, n, centre - f0


def _panel_rows(track: PanelTrack, times_us: list[int], proxy_size: Any, src_size: Any) -> dict[int, BoxF]:
    """Each proxy row's box in source pixels, keyed by the source frame that proxy frame shows (first row wins)."""
    rows: dict[int, BoxF] = {}
    for row in sorted(track.per_frame, key=lambda r: r.frame):
        rows.setdefault(_on_screen(times_us, row.pts), geometry.proxy_to_src(row.box, proxy_size, src_size))
    return rows


def _interp(rows: dict[int, BoxF], keys: list[int], s: int, times_us: list[int], gap_us: int) -> BoxF | None:
    """The panel box on source frame s, linear in time between the two nearest proxy rows; None past a gap."""
    i = bisect.bisect_left(keys, s)
    if i < len(keys) and keys[i] == s:
        return list(rows[s])
    if i == 0 or i == len(keys):
        return None
    j0, j1 = keys[i - 1], keys[i]
    t0, t1 = times_us[j0], times_us[j1]
    if t1 <= t0 or t1 - t0 > gap_us:
        return None
    w = (times_us[s] - t0) / (t1 - t0)
    a, b = rows[j0], rows[j1]
    return [a[q] + (b[q] - a[q]) * w for q in range(4)]


def panel_boxes(track: PanelTrack, kf: Keyframe, times: list[float], f0: int, n: int, *, src_size: Any, proxy_size: Any, th: dict[str, Any]) -> list[BoxF | None]:
    """The panel box per clip frame in source pixels (5.9 item 1); the keyframe's own box on the frame it shows."""
    times_us = [_us(t) for t in times]
    rows = _panel_rows(track, times_us, proxy_size, src_size)
    keys = sorted(rows)
    gap_us = _us(th["panels"]["fragment_gap_split_s"])
    out = [_interp(rows, keys, f0 + i, times_us, gap_us) for i in range(n)]
    k = _on_screen(times_us, kf.pts) - f0
    if 0 <= k < n:
        out[k] = [float(v) for v in kf.box_src]
    return out


def _window_of(members: list[_Item], base_side: float, margin: float) -> tuple[tuple[float, float], float]:
    """A group's window at the keyframe: centred on the mean mark centre, big enough to hold every box with margin."""
    cx = sum((it.box_src[0] + it.box_src[2]) / 2.0 for it in members) / len(members)
    cy = sum((it.box_src[1] + it.box_src[3]) / 2.0 for it in members) / len(members)
    reach = max(max(abs(b[0] - cx), abs(b[2] - cx), abs(b[1] - cy), abs(b[3] - cy)) for b in (it.box_src for it in members))
    return (cx, cy), max(base_side, 2.0 * margin * reach)


def _group(items: list[_Item], base_side: float, th: dict[str, Any]) -> list[tuple[list[_Item], tuple[float, float], float]]:
    """Greedy in the given order: an item joins the first group whose window it overlaps by window_share_overlap_min
    (intersection over the smaller window) while the group holds fewer than max_objects_per_clip."""
    sp = th["staysput"]
    overlap_min, cap, margin = float(sp["window_share_overlap_min"]), int(sp["max_objects_per_clip"]), _sp(th, "window_mark_margin")
    groups: list[list[_Item]] = []
    for it in items:
        own = _square(*_window_of([it], base_side, margin))
        for members in groups:
            if len(members) < cap and _overlap(own, _square(*_window_of(members, base_side, margin))) >= overlap_min:
                members.append(it)
                break
        else:
            groups.append([it])
    return [(m, *_window_of(m, base_side, margin)) for m in groups]


def _window_centres(centre: tuple[float, float], kf_box: Any, panels: list[BoxF | None], k: int) -> list[tuple[float, float]]:
    """The window centre per clip frame: carried with the panel box; held at the nearest known frame elsewhere."""
    known: dict[int, tuple[float, float]] = {}
    for i, box in enumerate(panels):
        if box is not None and (p := _carry_point(centre, kf_box, box)) is not None:
            known[i] = p
    known[k] = centre
    idx = sorted(known)
    out = []
    for i in range(len(panels)):
        if i not in known:
            j = bisect.bisect_left(idx, i)
            near = [idx[q] for q in (j - 1, j) if 0 <= q < len(idx)]
            i_near = min(near, key=lambda q: (abs(q - i), q))
            out.append(known[i_near])
        else:
            out.append(known[i])
    return out


def _base_side(kf: Keyframe, th: dict[str, Any]) -> float:
    sp = th["staysput"]
    return max(float(sp["window_frac_of_panel_width"]) * (kf.box_src[2] - kf.box_src[0]), float(sp["window_min_px"]))


def windows_for(
    marks: list[Mark],
    kf: Keyframe,
    track: PanelTrack,
    pts: list[float],
    *,
    th: dict[str, Any],
    src_size: tuple[int, int],
    proxy_size: tuple[int, int],
) -> list[tuple[list[str], list[BoxF]]]:
    """(mark ids, one square window per clip frame) per clip around kf, 5.9 item 1. pts are the original's frame
    times (video.frame_times); the windows follow the panel box interpolated between the proxy frames. Marks too small
    for a clip are left out, as in derive_staysput."""
    times_us = [_us(t) for t in pts]
    span = _span(times_us, kf.pts, clip_frames(pts, th))
    if span is None:
        return []
    f0, n, k = span
    panels = panel_boxes(track, kf, pts, f0, n, src_size=src_size, proxy_size=proxy_size, th=th)
    min_px = _min_box_px(th, src_size)
    items = [_item(m, box) for m in sorted(marks, key=lambda m: m.mark_id) if m.kf_id == kf.kf_id and (box := _mark_box(m, min_px)[0]) is not None]
    out = []
    for members, centre, side in _group(items, _base_side(kf, th), th):
        out.append(([it.mark_id for it in members], [_square(c, side) for c in _window_centres(centre, kf.box_src, panels, k)]))
    return out


def _prompt_text(boxes_clip: list[Any], size: int, frame: int, th: dict[str, Any]) -> str:
    pad = int(th["panels"]["prompt_box_pad_px"])
    objects = [
        sam_client.PromptObject(id=i, prompts=[sam_client.FramePrompt(frame=frame, box=sam_client.prompt_box(b, (size, size), pad_px=pad))])
        for i, b in enumerate(boxes_clip, start=1)
    ]
    return sam_client.object_prompt(objects, (size, size))


def prompt_for(
    marks: list[Mark],
    window_frame45: BoxF,
    clip_size: int,
    th: dict[str, Any],
    *,
    prompt_frame: int | None = None,
    src_size: tuple[int, int] | None = None,
) -> str:
    """5.9 item 3: one PromptObject per mark (ids 1..n in order), a FramePrompt on the prompt frame, each box grown to
    the live path's minimum side when src_size is given, mapped into clip pixels through that frame's window, then
    padded and clamped by sam_client.prompt_box."""
    frame = int(th["media"]["clip_prompt_frame"]) if prompt_frame is None else int(prompt_frame)
    min_px = _min_box_px(th, src_size) if src_size is not None else 1.0
    boxes = []
    for m in marks:
        raw = _raw_box(m)
        if raw is None:
            raise ValueError(f"{m.mark_id} has no source position")
        boxes.append(_to_clip(_at_least(raw[0], min_px), window_frame45, clip_size))
    return _prompt_text(boxes, clip_size, frame, th)


# ----------------------------------------------------------------------------------------------------------------
# Neighbours (docs/STAYS_PUT_ANCHORS.md): rigid things tracked in the same clip request as the marks
# ----------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Cand:
    """A neighbour the clip could prompt: a rigid mark on the keyframe, a lane part at the keyframe, or its handle."""

    kind: str  # one of NEIGHBOUR_KINDS
    ref: str
    box: tuple[float, float, float, float]  # source pixels at the prompt frame
    long_side: float  # source pixels


def _neighbours_on(th: dict[str, Any]) -> bool:
    return bool(th["staysput"].get("neighbours_enabled", True))


def _long(b: Any) -> float:
    return max(float(b[2]) - float(b[0]), float(b[3]) - float(b[1]))


def _iou(a: Any, b: Any) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - iw * ih
    return iw * ih / union if union > 0 else 0.0


def _row_at(track: PanelTrack, frame: int) -> Any:
    return next((r for r in track.per_frame if r.frame == frame), None)


def neighbour_candidates(walk_marks: list[Mark], tracks: list[PanelTrack], kf: Keyframe, *, th: dict[str, Any], src_size: tuple[int, int], proxy_size: tuple[int, int]) -> list[_Cand]:
    """What a clip on kf may track as neighbours, in source pixels at the keyframe: every other mark on the same crop
    whose type sits on the paint (not a reflection, not unsure) and that is big enough for a clip; the lane parts SAM
    saw on the keyframe's own proxy frame (side mirror, wheel, lights, plate, windows; never a face, the car or a
    panel) whose box the frame edge does not cut; and the keyframe's door handle, boxed from its point."""
    min_px = _min_box_px(th, src_size)
    out: list[_Cand] = []
    for m in sorted(walk_marks, key=lambda m: m.mark_id):
        if m.kf_id != kf.kf_id or m.type not in RIGID_MARK_KINDS:
            continue
        box, _why = _mark_box(m, min_px)
        if box is not None:
            out.append(_Cand("mark", m.mark_id, (box[0], box[1], box[2], box[3]), _long(box)))
    for t in sorted(tracks, key=lambda t: t.track_id):
        if t.noun not in NEIGHBOUR_NOUNS or t.track_id == kf.track_id:
            continue
        row = _row_at(t, kf.f_proxy)
        if row is None or row.border_frac > 0:
            continue
        b = geometry.proxy_to_src(row.box, proxy_size, src_size)
        if b[2] > b[0] and b[3] > b[1]:
            out.append(_Cand(t.noun.value, t.track_id, (b[0], b[1], b[2], b[3]), _long(b)))
    if kf.handle.point_src is not None:
        x, y = (float(v) for v in kf.handle.point_src)
        width = float(kf.box_src[2]) - float(kf.box_src[0])
        hw, hh = _sp(th, "neighbour_handle_half_w_frac") * width, _sp(th, "neighbour_handle_half_h_frac") * width
        if hw > 0 and hh > 0:
            out.append(_Cand(HANDLE_KIND, f"{kf.kf_id}_handle", (x - hw, y - hh, x + hw, y + hh), 2.0 * max(hw, hh)))
    return out


def pick_neighbours(
    member_boxes: list[Any],
    member_ids: list[str],
    centre: tuple[float, float],
    side0: float,
    panel_box: Any,
    cands: list[_Cand],
    th: dict[str, Any],
    *,
    member_rigid: int = 0,
) -> tuple[float, list[_Cand]]:
    """(window side, neighbours) for one clip. Candidates go nearest first by the window side each needs (twice its
    farthest box edge from the clip centre, times neighbour_window_margin). Every candidate the base window already
    holds is taken; the window then grows until it holds neighbours_target of them. It never grows past
    neighbour_window_max_panel_frac of the panel's long side, nor so far that a mark in the clip (its own or a
    neighbour mark) falls under neighbour_mark_min_clip_px on its long side; a part longer than
    neighbour_part_max_window_frac of the window, or one overlapping a part already taken (IoU at least
    neighbour_dedupe_iou), is skipped. At most neighbours_max, and marks plus neighbours stay within
    max_objects_per_clip. The clip's own marks that sit on the paint (member_rigid of them) are neighbours of each
    other too, so they count toward the target. A window that grew but still holds fewer than neighbours_min goes back
    to the base side and keeps what that holds, so the marks stay sharp for a test that cannot run. The growth is also
    capped at neighbour_window_max_growth x the base side."""
    sp = th["staysput"]
    size = float(th["media"]["clip_size_px"])
    target, most, least = int(_sp(th, "neighbours_target")), int(_sp(th, "neighbours_max")), int(_sp(th, "neighbours_min"))
    room = min(most, int(sp["max_objects_per_clip"]) - len(member_boxes))
    if room <= 0:
        return side0, []
    floor, margin = _sp(th, "neighbour_mark_min_clip_px"), _sp(th, "neighbour_window_margin")
    by_marks = size * min(_long(b) for b in member_boxes) / floor if member_boxes else math.inf
    limit = max(side0, min(_sp(th, "neighbour_window_max_panel_frac") * _long(panel_box), by_marks, _sp(th, "neighbour_window_max_growth") * side0))
    part_frac, dedupe = _sp(th, "neighbour_part_max_window_frac"), _sp(th, "neighbour_dedupe_iou")
    cx, cy = centre
    taken_ids = set(member_ids)
    scored = []
    for c in cands:
        if c.kind == "mark" and c.ref in taken_ids:
            continue
        reach = max(abs(c.box[0] - cx), abs(c.box[2] - cx), abs(c.box[1] - cy), abs(c.box[3] - cy))
        scored.append((2.0 * reach * margin, NEIGHBOUR_KINDS.index(c.kind), c.ref, c))
    scored.sort(key=lambda s: s[:3])
    have = max(0, int(member_rigid) - 1)  # the fewest own marks any mark of the clip sees as neighbours
    side, chosen, needs = side0, [], []
    for need, _rank, _ref, c in scored:
        if len(chosen) >= room or (need > side and len(chosen) + have >= target):
            break
        new_side = max(side, need)
        new_limit = min(limit, max(side0, size * c.long_side / floor)) if c.kind == "mark" else limit
        if new_side > new_limit:
            continue
        if c.kind != "mark" and (c.long_side > part_frac * new_side or any(o.kind != "mark" and _iou(c.box, o.box) >= dedupe for o in chosen)):
            continue
        side, limit = new_side, new_limit
        chosen.append(c)
        needs.append(need)
    if len(chosen) + have < least and side > side0:
        chosen = [c for c, need in zip(chosen, needs) if need <= side0]
        side = side0
    return side, chosen


def _neighbour_layout(walk: _Walk, kf: Keyframe, groups: list[tuple[list[_Item], tuple[float, float], float]], th: dict[str, Any]) -> list[tuple[int, list[_Item], tuple[float, float], float, list[_Cand]]]:
    """(group number, members, centre, side, neighbours) per clip of one keyframe. A group whose marks all ride as
    neighbours in an earlier clip of the keyframe gets no clip of its own: its marks are tested in that clip instead,
    so every mark is a tested object in exactly one clip."""
    cands = neighbour_candidates(walk.marks, list(walk.tracks.values()), kf, th=th, src_size=walk.src_size, proxy_size=walk.proxy_size)
    out: list[list[Any]] = []
    covered: dict[str, int] = {}
    for g, (members, centre, side) in enumerate(groups, start=1):
        if all(it.mark_id in covered for it in members):
            for it in members:
                slot = out[covered[it.mark_id]]
                slot[4] = [c for c in slot[4] if not (c.kind == "mark" and c.ref == it.mark_id)]
                slot[1] = [*slot[1], it]
            continue
        boxes = [it.box_src for it in members]
        rigid = sum(1 for it in members if not it.is_control and it.type in RIGID_MARK_KINDS)
        grown, chosen = pick_neighbours(boxes, [it.mark_id for it in members], centre, side, kf.box_src, cands, th, member_rigid=rigid)
        for c in chosen:
            if c.kind == "mark":
                covered.setdefault(c.ref, len(out))
        out.append([g, list(members), centre, grown, chosen])
    return [(g, members, centre, side, chosen) for g, members, centre, side, chosen in out]


def _plan(
    walk: _Walk,
    kf: Keyframe,
    members: list[_Item],
    centre: tuple[float, float],
    side: float,
    span: tuple[int, int, int],
    panels: list[BoxF | None],
    g: int,
    th: dict[str, Any],
    neighbours: list[_Cand] | None = None,
) -> ClipPlan:
    """One clip: the marks as objects 1..m, then the neighbours (None: the neighbour test is off for this walk)."""
    f0, n, k = span
    size = int(th["media"]["clip_size_px"])
    windows = [_square(c, side) for c in _window_centres(centre, kf.box_src, panels, k)]
    objects = []
    for i, it in enumerate(members, start=1):
        objects.append(
            PlanObject(
                object_id=str(i),
                mark_id=it.mark_id,
                type=it.type,
                is_control=it.is_control,
                box_src=list(it.box_src),
                box_clip=_to_clip(it.box_src, windows[k], size),
                point_crop=None if it.point_crop is None else list(it.point_crop),
                bbox_crop=None if it.bbox_crop is None else list(it.bbox_crop),
            )
        )
    near = [
        PlanNeighbour(object_id=str(len(objects) + j), kind=c.kind, ref=c.ref, box_src=list(c.box), box_clip=_to_clip(c.box, windows[k], size))
        for j, c in enumerate(neighbours or [], start=1)
    ]
    clip_id = f"{kf.kf_id}_sp{g:02d}"
    plan = ClipPlan(
        clip_id=clip_id,
        role=walk.role,
        kf_id=kf.kf_id,
        panel_id=kf.panel_id,
        lane_id=f"sp_{clip_id}",
        src_size=list(walk.src_size),
        proxy_size=list(walk.proxy_size),
        size=size,
        prompt_frame=k,
        f0_src=f0,
        pts_src=walk.times[f0 : f0 + n],
        window_px=side,
        windows_src=windows,
        panel_src=panels,
        objects=objects,
        neighbours=near,
        neighbour_rule=None if neighbours is None else NEIGHBOUR_RULE,
    )
    try:
        return plan.model_copy(update={"prompt_text": _prompt_text([o.box_clip for o in objects] + [nb.box_clip for nb in near], size, k, th)})
    except sam_client.ObjectPromptError as exc:
        return plan.model_copy(update={"status": "failed", "error": f"object prompt refused: {getattr(exc, 'code', type(exc).__name__)}"})


def _plans(walk: _Walk, items: list[_Item], th: dict[str, Any]) -> tuple[list[ClipPlan], list[tuple[_Item, str]]]:
    """Clips per keyframe (controls group last), each with its neighbours when the test is on, then the per-walk clip
    cap: controls, then by mark confidence."""
    times_us = [_us(t) for t in walk.times]
    frames = clip_frames(walk.times, th)
    by_kf: dict[str, list[_Item]] = {}
    for it in items:
        by_kf.setdefault(it.kf_id, []).append(it)
    ranked: list[tuple[tuple[int, int, str], ClipPlan, list[_Item]]] = []
    dropped: list[tuple[_Item, str]] = []
    for kf_id in sorted(by_kf):
        kf = walk.keyframes[kf_id]
        span = _span(times_us, kf.pts, frames)
        if span is None:
            dropped += [(it, "the walk is too short for a clip") for it in by_kf[kf_id]]
            continue
        panels = panel_boxes(walk.tracks[kf.track_id], kf, walk.times, span[0], span[1], src_size=walk.src_size, proxy_size=walk.proxy_size, th=th)
        members_in_order = sorted(by_kf[kf_id], key=lambda it: (it.is_control and it.rank < 0, it.mark_id))
        groups = _group(members_in_order, _base_side(kf, th), th)
        if _neighbours_on(th):
            layout = _neighbour_layout(walk, kf, groups, th)
        else:
            layout = [(g, members, centre, side, None) for g, (members, centre, side) in enumerate(groups, start=1)]
        for g, members, centre, side, near in layout:
            plan = _plan(walk, kf, members, centre, side, span, panels, g, th, near)
            ranked.append(((min(it.rank for it in members), 0, plan.clip_id), plan, members))
    cap = int(_sp(th, "max_clips_per_walk"))
    ranked.sort(key=lambda r: r[0])
    for _, _plan_over, members in ranked[cap:]:
        dropped += [(it, f"over the {cap}-clip cap for one walk") for it in members]
    return sorted((p for _, p, _ in ranked[:cap]), key=lambda p: p.clip_id), dropped


_KEY_FIELDS = (
    "clip_id", "kf_id", "panel_id", "lane_id", "src_size", "proxy_size", "size", "prompt_frame", "f0_src", "pts_src", "window_px", "windows_src",
    "panel_src", "objects", "neighbours", "neighbour_rule", "prompt_text",
)


def _plan_key(plan: ClipPlan) -> str:
    return contracts.sha256_json({k: getattr(plan, k) for k in _KEY_FIELDS})


# ----------------------------------------------------------------------------------------------------------------
# Check, cut, ask (live)
# ----------------------------------------------------------------------------------------------------------------


def _error(prefix: str, exc: BaseException) -> str:
    return redact(f"{prefix}: {type(exc).__name__}: {exc}")[:500]


def _grow(b: Any, frac: float, px: float) -> BoxF:
    dx, dy = (b[2] - b[0]) * frac + px, (b[3] - b[1]) * frac + px
    return [b[0] - dx, b[1] - dy, b[2] + dx, b[3] + dy]


def _dilation(th: dict[str, Any], src_size: Any) -> tuple[float, float]:
    """(clip_blur_dilation_frac per side, the privacy dilation in source pixels)."""
    return float(th["staysput"]["clip_blur_dilation_frac"]), float(th["privacy"]["track_box_dilation_px"]) * max(src_size) / SRC_REFERENCE_EDGE


def track_blur_boxes(walk_privacy: list[PanelTrack], plan: ClipPlan, *, proxy_pts: list[float], proxy_fps: float, th: dict[str, Any]) -> list[list[BoxF]]:
    """Per clip frame, in source pixels: the plate and face track boxes on the proxy frames either side of the frame
    (privacy.track_boxes_at, its gap bridge included), each grown by clip_blur_dilation_frac per side plus the privacy
    dilation (over-blurring is the safe side)."""
    src_size, proxy_size = tuple(plan.src_size), tuple(plan.proxy_size)
    frac, px = _dilation(th, src_size)
    out: list[list[BoxF]] = []
    for t in plan.pts_src:
        boxes: list[Any] = []
        p = max(0, bisect.bisect_right(proxy_pts, t) - 1)
        for q in sorted({p, min(p + 1, len(proxy_pts) - 1)}):
            boxes += privacy.track_boxes_at(q, walk_privacy, fps=proxy_fps, media_size_proxy=proxy_size, media_size_src=src_size, th=th)
        out.append([_grow(b, frac, px) for b in boxes])
    return out


def _with_checks(tracked: list[list[BoxF]], plan: ClipPlan, check: ClipCheck | None, th: dict[str, Any]) -> list[list[BoxF]]:
    frac, px = _dilation(th, tuple(plan.src_size))
    found = [] if check is None else [*check.plate_boxes, *check.face_boxes]
    return [boxes + [_grow(_to_src(b, win, plan.size), frac, px) for b in found] for boxes, win in zip(tracked, plan.windows_src)]


def blur_boxes(walk_privacy: list[PanelTrack], plan: ClipPlan, check: ClipCheck | None, *, proxy_pts: list[float], proxy_fps: float, th: dict[str, Any]) -> list[list[BoxF]]:
    """Per clip frame, in source pixels: track_blur_boxes plus the prompt frame's image-check boxes carried with the
    window, grown the same way. A check box the tracks do not cover never gets this far: uncovered_checks refuses
    that clip first, because a box carried with the window cannot follow its object."""
    tracked = track_blur_boxes(walk_privacy, plan, proxy_pts=proxy_pts, proxy_fps=proxy_fps, th=th)
    return _with_checks(tracked, plan, check, th)


def uncovered_checks(plan: ClipPlan, check: ClipCheck, tracked_prompt: list[BoxF], th: dict[str, Any]) -> list[str]:
    """What the prompt-frame check found that the walk's own plate and face tracks miss: one 'license plate' or 'face'
    per check box that the frame's grown track boxes cover by less than check_box_covered_min of its area in the clip.
    Only a track can keep an object blurred on the other clip frames."""
    size, win = plan.size, plan.windows_src[plan.prompt_frame]
    covered = np.zeros((size, size), dtype=bool)
    for b in tracked_prompt:
        span = _pixel_span(_to_clip(b, win, size), size)
        if span is not None:
            covered[span] = True
    need = _sp(th, "check_box_covered_min")
    missed: list[str] = []
    for kind, boxes in (("license plate", check.plate_boxes), ("face", check.face_boxes)):
        for b in boxes:
            span = _pixel_span(b, size)
            if span is not None and float(covered[span].mean()) < need:
                missed.append(kind)
    return missed


def _block_px(th: dict[str, Any], src_size: Any) -> int:
    return max(2, int(round(float(th["privacy"]["pixel_block_px_at_4k"]) * max(src_size) / SRC_REFERENCE_EDGE)))


def _metadata() -> dict[str, str]:
    # what the live E5 gate sent (D1, 10/31): the API answers object prompts with one_bit masks anyway
    return sam_client.sam_metadata(mask_encoding=None, propagation_direction="both")


def _copy_lane(run_id: str, src_role: str, src_lane: str, role: str, lane: str) -> LaneRecord:
    """Another walk's answer to the same clip bytes and prompt, copied under this walk's lane id: no request, no
    ledger row, no dollars (the orchestrator's rule for byte-identical walks)."""
    record = contracts.read_json(paths.lane_summary_path(run_id, src_role, src_lane), LaneRecord)
    note = Diagnostic(
        severity="warning",
        code=REUSED_CODE,
        message=f"the {src_role} clip {src_lane} has the same bytes and prompt, so its answer stands for this clip; no request was sent",
    )
    update: dict[str, Any] = {
        "lane_id": paths.lane_id(lane),
        "role": role,
        "dollars": 0.0,
        "retries": 0,
        "upload_s": None,
        "first_delta_s": None,
        "completed_s": None,
        "diagnostics": [*record.diagnostics, note],
    }
    if record.lane_path:
        gz = paths.lane_events_gz_path(run_id, role, lane)
        shutil.copyfile(paths.run_dir(run_id) / record.lane_path, gz)
        update["lane_path"] = paths.rel(run_id, gz)
    if record.index_path:
        index = paths.lane_index_path(run_id, role, lane)
        shutil.copyfile(paths.run_dir(run_id) / record.index_path, index)
        update["index_path"] = paths.rel(run_id, index)
    copied = record.model_copy(update=update)
    contracts.write_json(paths.lane_summary_path(run_id, role, lane), copied)
    return copied


def _lane_record(run_id: str, plan: ClipPlan) -> LaneRecord | None:
    path = paths.run_dir(run_id) / "sam" / plan.role / f"{plan.lane_id}.summary.json"
    return contracts.read_json(path, LaneRecord) if path.is_file() else None


def _own_lane(run_id: str, plan: ClipPlan) -> LaneRecord | None:
    """The lane record this plan's request wrote: none before its clip was cut, or when the record on disk was asked
    with another prompt (left by a plan this one replaced)."""
    record = _lane_record(run_id, plan) if plan.clip is not None else None
    return record if record is not None and record.prompt_text == plan.prompt_text else None


def _lane_ok(run_id: str, plan: ClipPlan) -> bool:
    record = _own_lane(run_id, plan)
    if record is None or record.status not in ANSWERED or plan.clip is None:
        return False
    root = paths.run_dir(run_id)
    return (root / plan.clip.path).is_file() and (record.index_path is None or (root / record.index_path).is_file())


def _own_request_ids(run_id: str, plan: ClipPlan) -> list[str]:
    """The response ids this plan asked itself: its image checks, then its lane (never answers reused from a walk)."""
    ids = list(plan.check.request_ids) if plan.check is not None else []
    record = _own_lane(run_id, plan)
    if record is not None and record.request_id and plan.lane_reused_from is None:
        ids.append(record.request_id)
    return ids


def _known_answers(run_id: str, role: str) -> tuple[dict[str, tuple[str, ClipCheck]], dict[tuple[str, str], tuple[str, str]]]:
    """Image checks by window hash and answered lanes by (clip sha256, prompt) from the run's other walks."""
    checks: dict[str, tuple[str, ClipCheck]] = {}
    lanes: dict[tuple[str, str], tuple[str, str]] = {}
    for other in planned_roles(run_id):
        stored = _read_plans(run_id, other) if other != role else None
        for plan in stored.clips if stored else []:
            if plan.check is not None:
                checks.setdefault(plan.check.window_sha256, (plan.check.reused_from or f"{other}/{plan.clip_id}", plan.check))
            if plan.status == "answered" and plan.clip is not None and plan.clip.sha256 and _lane_ok(run_id, plan):
                lanes.setdefault((plan.clip.sha256, plan.prompt_text), (other, plan.lane_id))
    return checks, lanes


async def _check(run_id: str, plan: ClipPlan, window_rgb: np.ndarray, *, th: dict[str, Any], sem: asyncio.Semaphore, known: dict[str, tuple[str, ClipCheck]]) -> ClipCheck:
    """Plate and face checks on the unblurred prompt-frame window, or the earlier answer for the same pixels."""
    digest = hashlib.sha256(repr(window_rgb.shape).encode("ascii") + window_rgb.tobytes()).hexdigest()
    if digest in known:
        ref, prior = known[digest]
        return ClipCheck(window_sha256=digest, plate_boxes=prior.plate_boxes, face_boxes=prior.face_boxes, reused_from=ref)
    async with sem:
        _plates, _faces, record = await privacy.image_checks(window_rgb, run_id=run_id, role=plan.role, tag=f"{plan.clip_id}.f{plan.prompt_frame:02d}", th=th)
    return ClipCheck(window_sha256=digest, plate_boxes=record.plate_boxes, face_boxes=record.face_boxes, request_ids=record.request_ids)


def _prompt_window(run_id: str, role: str, plan: ClipPlan) -> np.ndarray:
    """The unblurred prompt-frame window, decoded in memory for the image checks; never written."""
    k = plan.prompt_frame
    return _cut_window(privacy.read_original_frame(run_id, role, plan.pts_src[k]), plan.windows_src[k], plan.size)


async def _run_clip(
    run_id: str,
    walk: _Walk,
    plan: ClipPlan,
    *,
    th: dict[str, Any],
    sems: dict[str, asyncio.Semaphore],
    checks: dict[str, tuple[str, ClipCheck]],
    lanes: dict[tuple[str, str], tuple[str, str]],
    stop: Callable[[], bool] | None = None,
) -> ClipPlan:
    """Image checks on the unblurred prompt frame FIRST, the refusal when they found what the walk's tracks miss, then
    the pixelated clip, then the SAM request (or the answer another walk got for the same bytes). A step that fails
    ends the clip as failed with its reason. When stop() says so as the clip's turn comes, nothing starts and the plan
    stays planned."""
    k, n = plan.prompt_frame, len(plan.windows_src)
    try:
        async with sems["media"]:
            if stop is not None and stop():
                return plan
            window_rgb = await asyncio.to_thread(_prompt_window, run_id, walk.role, plan)
        check = await _check(run_id, plan, window_rgb, th=th, sem=sems["image"], known=checks)
        del window_rgb
    except Exception as exc:  # no frame is written from a check that did not happen (section 8 item 4)
        return plan.model_copy(update={"status": "failed", "error": _error("image check did not complete", exc)})
    plan = plan.model_copy(update={"check": check})
    try:
        tracked = track_blur_boxes(walk.privacy_tracks, plan, proxy_pts=walk.proxy_pts, proxy_fps=walk.proxy_fps, th=th)
        missed = uncovered_checks(plan, check, tracked[k], th)
    except Exception as exc:  # unknown blur boxes cut nothing
        return plan.model_copy(update={"status": "failed", "error": _error("blur boxes not computed", exc)})
    if missed:
        what = " and a ".join(sorted(set(missed)))
        why = f"the prompt-frame check found a {what} that the walk's plate and face tracks miss, so no clip is cut: only a track keeps it blurred on the other frames"
        return plan.model_copy(update={"status": "refused", "error": why})
    try:
        blur = _with_checks(tracked, plan, check, th)
        dst = paths.clip_path(run_id, walk.role, plan.clip_id)
        async with sems["media"]:
            clip = await asyncio.to_thread(
                video.cut_clip,
                walk.src,
                dst,
                pts_center=walk.times[plan.f0_src + n // 2],
                frames=n,
                windows_src=[list(w) for w in plan.windows_src],
                size=plan.size,
                crf=int(th["media"]["clip_crf"]),
                blur_boxes=blur,
                block_px=_block_px(th, walk.src_size),
            )
        if clip.f0_src != plan.f0_src:
            raise video.VideoError(f"clip starts at source frame {clip.f0_src}, planned {plan.f0_src}")
    except Exception as exc:
        return plan.model_copy(update={"status": "failed", "error": _error("clip not cut", exc)})
    plan = plan.model_copy(update={"clip": clip})
    reused = None
    try:
        hit = lanes.get((clip.sha256 or "", plan.prompt_text))
        if hit is not None:
            record = _copy_lane(run_id, hit[0], hit[1], walk.role, plan.lane_id)
            reused = f"{hit[0]}/{hit[1]}"
        else:
            async with sems["sam"]:
                record, _ = await sam_client.segment_clip(
                    paths.run_dir(run_id) / clip.path,
                    plan.prompt_text,
                    run_id=run_id,
                    role=walk.role,
                    lane=plan.lane_id,
                    media_size=(plan.size, plan.size),
                    metadata=_metadata(),
                    frames_sent=n,
                )
    except Exception as exc:
        return plan.model_copy(update={"status": "failed", "error": _error("SAM request not made", exc)})
    if record.status in ANSWERED:
        return plan.model_copy(update={"status": "answered", "error": None, "lane_reused_from": reused})
    reason = record.error or f"the clip lane ended {record.status.value}"
    return plan.model_copy(update={"status": "failed", "error": redact(reason)[:500], "lane_reused_from": reused})


# ----------------------------------------------------------------------------------------------------------------
# Analysis (5.9 items 5 to 7), pure on stored facts
# ----------------------------------------------------------------------------------------------------------------


def tau_fix(th: dict[str, Any]) -> float:
    """max(tau_fix_floor, tau_fix_sigma_factor x staysput.sigma_frame); the floor alone while that leaf is absent. It
    does not read matching.sigma_frame: D1's 0.1689 there would make tau_fix 0.507, which the lead ruled out until E5
    is recalibrated (docs/D1_MEMO.md note 2)."""
    sp = th["staysput"]
    floor = float(sp["tau_fix_floor"])
    sigma_frame = sp.get("sigma_frame")
    return floor if sigma_frame is None else max(floor, float(sp["tau_fix_sigma_factor"]) * float(sigma_frame))


def _object_boxes(frames: list[IndexFrame], object_id: str) -> dict[int, tuple[BoxF, float | None]]:
    out: dict[int, tuple[BoxF, float | None]] = {}
    for fr in frames:
        if fr.frame in out:
            continue
        for obj in fr.objects:
            if obj.id == object_id:
                out[fr.frame] = ([float(v) for v in obj.box], obj.c)
                break
    return out


def _whitened(p_src: tuple[float, float], panel: Any, moments: Moments, from_crop: BoxF, crop: Crop, sign: int) -> list[float] | None:
    try:
        carried = geometry.carry_moments(moments, from_crop, geometry.box_src_to_crop(panel, crop))
        u, v = geometry.whiten(tuple(geometry.src_to_crop(p_src, crop)), carried, sign)  # type: ignore[arg-type]
    except ValueError:
        return None
    return [u, v]


def _baselines(panels: list[Any], src_size: Any, tol: float) -> tuple[float | None, float | None]:
    """b1 (panel centre displacement first to last usable frame over the frame width) and b2 (|ln| of the aspect
    ratio change), over the clip frames whose panel box is known and inside the frame."""
    usable = [b for b in panels if b is not None and _in_frame(b, src_size, tol)]
    if len(usable) < 2:
        return None, None
    first, last = usable[0], usable[-1]
    b1 = math.dist(_center(first), _center(last)) / float(src_size[0])
    ar0, ar1 = _aspect(first), _aspect(last)
    return b1, (abs(math.log(ar1 / ar0)) if ar0 > 0 and ar1 > 0 else None)


def _drift(ws: list[list[float]], min_frames: int) -> float | None:
    if len(ws) < max(2, min_frames):
        return None
    a = np.asarray(ws, dtype=np.float64)
    return float(math.sqrt(float(np.mean(np.sum((a - np.median(a, axis=0)) ** 2, axis=1)))))


def _sign_test(points: list[TrackPoint], panels: list[Any], direction: int | None, th: dict[str, Any], rate: float = 1.0) -> tuple[float | None, int]:
    """Pairs of frames sign_pair_gap_frames apart that do not overlap, so sign_pairs_min counts separate stretches of
    the clip; the gap and the per-frame panel motion floor are set at 30 fps and scale by rate (clip fps / 30). The
    camera's sign is -sign(panel dx) when the panel moved more than that floor, else minus the walk direction;
    agreement is the share of pairs whose sign(du) equals it. A reflection drifts with the camera (agreement near 1);
    paint has no direction of its own."""
    # TODO(D1 E5): on real orbits du on paint also follows the perspective trend of the panel box (the pair C trunk
    # chips: 0.31 on 74 overlapping pairs in sp_smoke_1031b, 0.33 on 15 here in sp_smoke_1031c), so this can confirm
    # sliding on paint. Detrending u would not help: a reflection's slide is itself the trend. The neighbour-ratio test
    # of docs/STAYS_PUT_ANCHORS.md is the candidate replacement; a live run on dark paint with a real reflection must
    # pick before cap_enabled goes on.
    sp = th["staysput"]
    gap = max(1, int(round(float(sp["sign_pair_gap_frames"]) * rate)))
    dx_floor = float(sp["sign_panel_dx_min_px_per_frame"]) / rate * gap
    by_f = {p.f: p for p in points if p.u_rel is not None}
    pairs = agree = 0
    free = None  # the first frame a new pair may start on
    for f in sorted(by_f):
        if free is not None and f < free:
            continue
        g = by_f.get(f + gap)
        if g is None:
            continue
        dx = _center(panels[f + gap])[0] - _center(panels[f])[0]
        if abs(dx) > dx_floor:
            s_cam = -_sign(dx)
        elif direction in (1, -1):
            s_cam = -int(direction)
        else:
            continue
        du = float(g.u_rel) - float(by_f[f].u_rel)  # type: ignore[arg-type]
        pairs += 1
        agree += int(du != 0 and _sign(du) == s_cam)
        free = f + gap
    return (agree / pairs if pairs else None), pairs


def _label(box_share: float, b_pass: bool, drift: float | None, agreement: float | None, pairs: int, th: dict[str, Any]) -> StaysPutVerdict:
    sp = th["staysput"]
    if box_share < float(sp["no_track_box_share_below"]):
        return StaysPutVerdict.NO_TRACK
    if not b_pass or drift is None:
        return StaysPutVerdict.INCONCLUSIVE
    tau = tau_fix(th)
    if drift < tau:
        return StaysPutVerdict.FIXED
    confirmed = agreement is not None and agreement > float(sp["sign_agreement_min"]) and pairs >= int(sp["sign_pairs_min"])
    if drift > float(sp["sliding_factor"]) * tau and confirmed:
        return StaysPutVerdict.SLIDING
    return StaysPutVerdict.INCONCLUSIVE


def analyse(
    track_frames: list[IndexFrame],
    mark_id: str,
    panel_boxes_src: list[Any],
    kf_moments: Moments,
    front_sign: int | None,
    direction: int | None,
    *,
    th: dict[str, Any],
    object_id: str,
    windows_src: list[Any],
    clip_size: int,
    kf_box_src: Any,
    crop: Crop,
    src_size: tuple[int, int],
    proxy_size: tuple[int, int],
    role: str,
    fps: float | None = None,
    prompt_frame: int | None = None,
) -> MarkTrack:
    """One object's track against its panel (5.9 items 5 and 7), on the clip frames whose panel box is known and inside
    the frame and that hold the box centre inside the panel box: w_t is that centre whitened in the frame's keyframe
    moments carried by the panel box, u_t its x offset over the panel width. Then b1, b2, panned, the drift, the sign
    test (scaled to fps; 30 when None) and the label; the box share leaves the prompt frame out when it is given (E5
    counts the propagated frames). Caps, the clip and the request id are the caller's. front_sign None whitens with
    +1 (the drift is a distance, so the sign does not change it)."""
    n = len(windows_src)
    sign = front_sign if front_sign in (1, -1) else 1
    tol = src_size[0] / proxy_size[0]
    from_crop = geometry.box_src_to_crop(kf_box_src, crop)
    boxes = _object_boxes(track_frames, object_id)
    points: list[TrackPoint] = []
    for f in sorted(b for b in boxes if 0 <= b < n):
        box, c = boxes[f]
        bx, by = _center(box)
        cx, cy = _center(_to_src([bx, by, bx, by], windows_src[f], clip_size))
        panel = panel_boxes_src[f]
        inside = panel is not None and panel[0] <= cx < panel[2] and panel[1] <= cy < panel[3]
        measured = inside and _in_frame(panel, src_size, tol)
        u_rel = (cx - (panel[0] + panel[2]) / 2.0) / (panel[2] - panel[0]) if measured else None
        w = _whitened((cx, cy), panel, kf_moments, from_crop, crop, sign) if measured else None
        points.append(TrackPoint(f=f, box_clip=box, c=c, inside_panel=inside, w=w, u_rel=u_rel))
    sp = th["staysput"]
    b1, b2 = _baselines(panel_boxes_src, src_size, tol)
    b_pass = (b1 is not None and b1 >= float(sp["b1_floor"])) or (b2 is not None and b2 >= float(sp["b2_floor"]))
    panned = b1 is not None and b2 is not None and b1 < float(sp["b1_floor"]) and b2 >= float(sp["b2_floor"])
    drift = _drift([p.w for p in points if p.w is not None], int(_sp(th, "min_drift_frames")))
    rate = float(fps) / float(th["media"]["source_fps_assumed"]) if fps else 1.0
    agreement, pairs = _sign_test(points, panel_boxes_src, direction, th, rate)
    if prompt_frame is not None and n > 1:
        share = sum(1 for p in points if p.f != prompt_frame) / (n - 1)
    else:
        share = len(points) / n if n else 0.0
    return MarkTrack(
        mark_id=mark_id,
        role=role,
        track=points,
        b1=b1,
        b2=b2,
        b_pass=b_pass,
        panned=panned,
        drift_rms=drift,
        sign_agreement=agreement,
        pairs_n=pairs,
        verdict=_label(share, b_pass, drift, agreement, pairs, th),
    )


def _centres(frames: list[IndexFrame], n: int) -> dict[str, dict[int, tuple[float, float]]]:
    """Box centre per object id and clip frame (the first record of a frame wins), clip pixels."""
    out: dict[str, dict[int, tuple[float, float]]] = {}
    for fr in frames:
        if not 0 <= fr.frame < n:
            continue
        for obj in fr.objects:
            out.setdefault(obj.id, {}).setdefault(fr.frame, _center(obj.box))
    return out


def _affine_residuals(anchors: dict[str, dict[int, tuple[float, float]]], target: dict[int, tuple[float, float]], frames: range, k: int) -> list[float]:
    """Per frame f of frames (not k) where target and at least 3 anchors have a box on both k and f: how far target's
    centre on f is from where the least-squares affine taking the anchors' centres on k to theirs on f puts its centre
    on k. Clip pixels."""
    if k not in target:
        return []
    out: list[float] = []
    for f in frames:
        if f == k or f not in target:
            continue
        ids = [a for a, pts in anchors.items() if k in pts and f in pts]
        if len(ids) < 3:
            continue
        src = np.array([[anchors[a][k][0], anchors[a][k][1], 1.0] for a in ids], dtype=np.float64)
        dst = np.array([anchors[a][f] for a in ids], dtype=np.float64)
        matrix, _res, rank, _sv = np.linalg.lstsq(src, dst, rcond=None)
        if rank < 3:  # anchors on one line (or one point) fix no affine
            continue
        pred = np.array([target[k][0], target[k][1], 1.0], dtype=np.float64) @ matrix
        out.append(float(math.hypot(target[f][0] - pred[0], target[f][1] - pred[1])))
    return out


def _median(values: list[float]) -> float | None:
    return float(np.median(np.asarray(values, dtype=np.float64))) if values else None


def neighbour_test(
    track_frames: list[IndexFrame],
    object_id: str,
    neighbours: list[tuple[str, str, str]],
    *,
    n_frames: int,
    prompt_frame: int,
    fps: float | None,
    th: dict[str, Any],
) -> NeighbourTest:
    """Stays Put by neighbours (docs/STAYS_PUT_ANCHORS.md) for one object of a clip lane. neighbours are (object id,
    kind, ref) in prompt order. A neighbour is in the fit when SAM returned it on its prompt frame and on at least
    neighbour_track_share_min of the clip. Per window (the whole clip, and round(neighbour_near_half_window_s x fps)
    frames either side of the prompt frame): the mark's median residual against the affine fit to the neighbours, and
    the median over the neighbours of each one's median leave-one-out residual (floored at
    neighbour_residual_floor_px); the ratio is the first over the second. fixed when both ratios are under
    neighbour_fixed_below, sliding when both are over neighbour_sliding_above, else inconclusive. no_track when the
    mark itself has a box on fewer than no_track_box_share_below of the clip; inconclusive with fewer than
    neighbours_min neighbours in the fit. Pure: the same lane gives the same numbers, live and in rebuild."""
    sp = th["staysput"]
    near_s = _sp(th, "neighbour_near_half_window_s")
    rate = float(fps) if fps else float(th["media"]["source_fps_assumed"])
    near = max(1, int(round(near_s * rate)))
    k, n = int(prompt_frame), int(n_frames)
    pts = _centres(track_frames, n)
    share_min, least = _sp(th, "neighbour_track_share_min"), int(_sp(th, "neighbours_min"))
    used = {oid: pts[oid] for oid, _kind, _ref in neighbours if oid in pts and k in pts[oid] and n > 0 and len(pts[oid]) / n >= share_min}
    mark = pts.get(object_id, {})
    windows = {"full": range(0, n), "near": range(max(0, k - near), min(n, k + near + 1))}
    loo: dict[str, dict[str, float | None]] = {name: {} for name in windows}
    numbers: dict[str, float | None] = {}
    verdict, reason = StaysPutVerdict.INCONCLUSIVE, None
    if n <= 0 or k not in mark or len(mark) / n < float(sp["no_track_box_share_below"]):
        verdict, reason = StaysPutVerdict.NO_TRACK, "SAM did not follow the mark through enough of the clip"
    elif len(used) < least:
        reason = f"{len(used)} neighbours tracked through the clip, {least} needed"
    else:
        floor = _sp(th, "neighbour_residual_floor_px")
        ratios: dict[str, float | None] = {}
        for name, span in windows.items():
            own = _median(_affine_residuals(used, mark, span, k))
            for oid in used:
                loo[name][oid] = _median(_affine_residuals({a: p for a, p in used.items() if a != oid}, used[oid], span, k))
            per = [v for v in loo[name].values() if v is not None]
            group = _median(per)
            numbers[f"mark_{name}_px"], numbers[f"neighbours_{name}_px"] = own, group
            ratios[name] = None if own is None or group is None else own / max(group, floor)
        numbers["ratio_full"], numbers["ratio_near"] = ratios["full"], ratios["near"]
        full, close = ratios["full"], ratios["near"]
        fixed_below, sliding_above = _sp(th, "neighbour_fixed_below"), _sp(th, "neighbour_sliding_above")
        if full is None or close is None:
            reason = "too few frames with three neighbours in view"
        elif full < fixed_below and close < fixed_below:
            verdict = StaysPutVerdict.FIXED
        elif full > sliding_above and close > sliding_above:
            verdict = StaysPutVerdict.SLIDING
        else:
            reason = f"not under {fixed_below:g}x in both windows nor over {sliding_above:g}x in both"
    rows = [
        StaysPutNeighbour(
            object_id=oid,
            kind=kind,
            ref=ref,
            frames=len(pts.get(oid, {})),
            used=oid in used,
            residual_full_px=loo["full"].get(oid),
            residual_near_px=loo["near"].get(oid),
            centres=[[float(f), x, y] for f, (x, y) in sorted(pts.get(oid, {}).items())],
        )
        for oid, kind, ref in neighbours
    ]
    return NeighbourTest(verdict=verdict, near_s=near_s, near_frames=near, used=len(used), reason=reason, neighbours=rows, **numbers)


def _neighbour_refs(plan: ClipPlan, obj: PlanObject) -> list[tuple[str, str, str]]:
    """obj's neighbours in its clip: the plan's neighbours, then the clip's other marks that sit on the paint."""
    refs = [(nb.object_id, nb.kind, nb.ref) for nb in plan.neighbours]
    refs += [(o.object_id, "mark", o.mark_id) for o in plan.objects if o.object_id != obj.object_id and not o.is_control and o.type in RIGID_MARK_KINDS]
    return refs


def _neighbour_row(frames: list[IndexFrame] | None, plan: ClipPlan, obj: PlanObject, th: dict[str, Any]) -> NeighbourTest | None:
    """The row's neighbour test: None for a plan made before the test existed, not_run (with the planned neighbours)
    when the clip lane gave no answer."""
    if plan.neighbour_rule is None:
        return None
    refs = _neighbour_refs(plan, obj)
    if frames is None:
        near_s = _sp(th, "neighbour_near_half_window_s")
        return NeighbourTest(near_s=near_s, neighbours=[StaysPutNeighbour(object_id=oid, kind=kind, ref=ref) for oid, kind, ref in refs])
    return neighbour_test(frames, obj.object_id, refs, n_frames=len(plan.windows_src), prompt_frame=plan.prompt_frame, fps=_fps(plan.pts_src), th=th)


def _lane_frames(run_id: str, record: LaneRecord) -> list[IndexFrame]:
    if record.index_path is None:
        return []
    path = paths.run_dir(run_id) / record.index_path
    return contracts.read_json_list(path, IndexFrame) if path.is_file() else []


def _plan_tracks(run_id: str, plan: ClipPlan, kf: Keyframe | None, direction: int | None, th: dict[str, Any], cache: dict[str, Any]) -> list[MarkTrack]:
    record = _own_lane(run_id, plan)
    ran = plan.status == "answered" and record is not None and record.status in ANSWERED
    frames = _lane_frames(run_id, record) if ran and record is not None else []
    rows: list[MarkTrack] = []
    for obj in plan.objects:
        curvature, highlight = _caps(run_id, plan.panel_id, obj.point_crop, obj.bbox_crop, kf, th, cache)
        extra: dict[str, Any] = {
            "clip": plan.clip,
            "request_id": record.request_id if record is not None else None,
            "prompt_text": plan.prompt_text,
            "is_control": obj.is_control,
            "curvature_cap": curvature,
            "highlight_cap": highlight,
        }
        extra["neighbour_test"] = _neighbour_row(frames if ran else None, plan, obj, th)
        if not ran or kf is None or kf.moments is None or kf.crop is None:
            rows.append(MarkTrack(mark_id=obj.mark_id, role=plan.role, verdict=StaysPutVerdict.NOT_RUN, **extra))
            continue
        mt = analyse(
            frames,
            obj.mark_id,
            list(plan.panel_src),
            kf.moments,
            kf.front_sign,
            direction,
            th=th,
            object_id=obj.object_id,
            windows_src=list(plan.windows_src),
            clip_size=plan.size,
            kf_box_src=kf.box_src,
            crop=kf.crop,
            src_size=(int(plan.src_size[0]), int(plan.src_size[1])),
            proxy_size=(int(plan.proxy_size[0]), int(plan.proxy_size[1])),
            role=plan.role,
            fps=_fps(plan.pts_src),
            prompt_frame=plan.prompt_frame,
        )
        rows.append(mt.model_copy(update=extra))
    return rows


def _role_tracks(run_id: str, role: str, th: dict[str, Any]) -> list[MarkTrack]:
    """A walk's rows: every clip object in plan order, then a not_run row for every mark that got no clip; each row
    carries the keyframe caps."""
    from .panels import walk_direction

    stored = _read_plans(run_id, role)
    if stored is None or not (stored.clips or stored.unclipped):
        return []
    kfs = {k.kf_id: k for k in _read_list(paths.keyframes_json(run_id), Keyframe) if k.role == role}
    direction = walk_direction([t for t in _read_list(paths.tracks_json(run_id), PanelTrack) if t.role == role], th)
    cache: dict[str, Any] = {}
    rows = [row for plan in stored.clips for row in _plan_tracks(run_id, plan, kfs.get(plan.kf_id), direction, th, cache)]
    for s in stored.unclipped:
        curvature, highlight = _caps(run_id, s.panel_id, s.point_crop, s.bbox_crop, kfs.get(s.kf_id), th, cache)
        rows.append(MarkTrack(mark_id=s.mark_id, role=role, is_control=s.is_control, curvature_cap=curvature, highlight_cap=highlight))
    return rows


def _write_tracks(run_id: str, th: dict[str, Any]) -> list[MarkTrack]:
    """derived/mark_tracks.json from every stored plan and lane, walks in role order: the one writer, live and rebuild."""
    rows = [row for role in planned_roles(run_id) for row in _role_tracks(run_id, role, th)]
    _write_atomic(paths.mark_tracks_json(run_id), rows)
    return rows


# ----------------------------------------------------------------------------------------------------------------
# Entry points
# ----------------------------------------------------------------------------------------------------------------


def _resume(run_id: str, stored: RolePlans | None, plans: list[ClipPlan]) -> tuple[list[ClipPlan], list[str]]:
    """Each plan as stored when an earlier derive answered or refused it under the same key; plus the request ids of
    the stored plans that are replaced, so the run-end deletion list keeps them."""
    if stored is None:
        return plans, []
    done = {p.clip_id: p for p in stored.clips if p.status in KEPT}
    out = []
    for p in plans:
        prior = done.get(p.clip_id)
        same = prior is not None and _plan_key(prior) == _plan_key(p)
        out.append(prior if prior is not None and same and (prior.status == "refused" or _lane_ok(run_id, prior)) else p)
    kept = {id(p) for p in out}
    earlier = list(stored.earlier_request_ids)
    for p in stored.clips:
        if id(p) not in kept:
            earlier += _own_request_ids(run_id, p)
    return out, list(dict.fromkeys(earlier))


def _drop_orphan_clips(run_id: str, role: str, plans: list[ClipPlan]) -> None:
    """Clip files of this walk that no current plan holds (left by a plan a later derive replaced or refused) are
    removed, so what the app serves under clips/<role>/ is what the plan file vouches for."""
    folder = paths.run_dir(run_id) / "clips" / role
    keep = {p.clip.path for p in plans if p.clip is not None}
    for f in sorted(folder.glob("*.mp4")) if folder.is_dir() else []:
        if paths.rel(run_id, f) not in keep:
            f.unlink(missing_ok=True)


async def _rows(run_id: str, role: str, th: dict[str, Any]) -> list[MarkTrack]:
    rows = await asyncio.to_thread(_write_tracks, run_id, th)
    return [r for r in rows if r.role == role]


async def derive_staysput(run_id: str, role: str, *, th: dict, stop: Callable[[], bool] | None = None) -> list[contracts.MarkTrack]:
    """LIVE. One walk: plan its clips (none when its plate or face lane did not complete), then check, cut and ask
    every new clip (a resumed walk asks nothing twice; a clip with the same bytes and prompt as another walk's reuses
    that answer), store the plan, and write derived/mark_tracks.json for every walk planned so far. Returns this
    walk's rows. stop (optional, e.g. the orchestrator's past_deadline) is asked before each clip starts; a clip it
    holds back stays planned, the walk is not complete, and a resume asks it."""
    walk = await asyncio.to_thread(_load_walk, run_id, role)
    stored = _read_plans(run_id, role)
    if walk is None:
        if stored is None:
            _write_plans(run_id, RolePlans(role=role, complete=True, skipped=["the walk has no video record or proxy"]))
        return await _rows(run_id, role, th)
    items, unclipped = await asyncio.to_thread(_items, run_id, walk, th)
    gap = _privacy_gap(run_id, role) if items else None
    plans: list[ClipPlan] = []
    dropped: list[tuple[_Item, str]] = []
    if gap is not None:
        dropped = [(it, gap) for it in items]
    elif items and not walk.src.is_file():
        if stored is not None:  # asked before the original went: keep what was asked; nothing more can be
            if not stored.complete:
                _write_plans(run_id, stored.model_copy(update={"complete": True}))
            return await _rows(run_id, role, th)
        dropped = [(it, "the original is gone, so no clip can be cut") for it in items]
    elif items:
        walk.times = await asyncio.to_thread(video.frame_times, walk.src)
        plans, dropped = _plans(walk, items, th)
    notes = [f"{it.mark_id}, the reflection control: {why}" for it, why in dropped if it.rank < 0]
    unclipped = sorted(unclipped + [_skip(it, why) for it, why in dropped if it.rank >= 0], key=lambda s: s.mark_id)
    plans, earlier = _resume(run_id, stored, plans)
    base = RolePlans(role=role, clips=plans, unclipped=unclipped, skipped=notes, earlier_request_ids=earlier)
    _write_plans(run_id, base)
    todo = [i for i, p in enumerate(plans) if p.status == "planned"]
    if todo:
        # TODO(E13): time a pair on the Railway image. On an M4 Max a 50 Mbps 4K HEVC walk costs 0.3 to 0.5 s per clip
        # for the check frame and about 2 s for the cut (2.8 s a clip two at a time), so a 16-clip walk is about 45 s.
        o = th["orchestrator"]
        sems = {
            "media": asyncio.Semaphore(max(1, int(o["media_workers"]))),
            "image": asyncio.Semaphore(max(1, int(o["sam_image_semaphore"]))),
            "sam": asyncio.Semaphore(max(1, int(o["sam_video_semaphore_per_run"]))),
        }
        checks, lanes = _known_answers(run_id, role)
        lock = asyncio.Lock()

        async def one(i: int) -> None:
            plans[i] = await _run_clip(run_id, walk, plans[i], th=th, sems=sems, checks=checks, lanes=lanes, stop=stop)
            if plans[i].status in ("failed", "refused"):
                log.warning("staysput %s %s %s: %s", role, plans[i].clip_id, plans[i].status, plans[i].error)
            async with lock:
                _write_plans(run_id, base.model_copy(update={"clips": list(plans)}))

        await asyncio.gather(*(one(i) for i in todo))
    await asyncio.to_thread(_drop_orphan_clips, run_id, role, plans)
    complete = all(p.status != "planned" for p in plans)
    _write_plans(run_id, base.model_copy(update={"clips": list(plans), "complete": complete}))
    return await _rows(run_id, role, th)


def request_ids(run_id: str) -> list[str]:
    """Every response id Stays Put asked in the run: each clip's plate and face checks, then its lane, and those of
    plans a later derive replaced. The run-end deletion list needs the check ids, which live only here, when the API
    refused store false."""
    ids: list[str] = []
    for role in planned_roles(run_id):
        stored = _read_plans(run_id, role)
        if stored is None:
            continue
        ids += stored.earlier_request_ids
        for plan in stored.clips:
            ids += _own_request_ids(run_id, plan)
    return list(dict.fromkeys(ids))


def rebuild_staysput(run_id: str, *, th: dict) -> list[contracts.MarkTrack]:
    """derived/mark_tracks.json recomputed from the stored plans, clip lanes and keyframe crops with no network call
    and no original; byte-identical to what derive_staysput wrote under the same thresholds. A run that never planned
    a clip and has no mark_tracks.json is left as it is."""
    if not planned_roles(run_id) and not (paths.run_dir(run_id) / "derived" / "mark_tracks.json").is_file():
        return []
    return _write_tracks(run_id, th)
