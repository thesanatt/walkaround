"""Blur, quarantine and deletion (docs/ARCHITECTURE.md section 8; docs/CONTRACTS.md privacy M12).

Order of operations for every unblurred frame: the SAM image checks run on the unblurred pixels FIRST and the plate
extent is recorded for the ruler; only then is the frame pixelated. Nothing unblurred is written outside
paths.quarantine_dir. Faces are always pixelated, and so is the plate in every frame this module hands back for
storage or for Spark. keep_plate only reaches the run owner's sheet frames, which the sheet writer makes with
blur_frame(keep_plate_region=True); served_file_allowed keeps those owner-only. Every SAM call here goes through
sam_client, which writes the ledger row; nothing in this module prices a call itself.

Every pixelated region gets a block that grows with the largest box or mask in it (box_blocks_across), so a close
plate is never left readable by a small block. Plate and face tracks reach past their sightings: each sighting also
covers the frames within track_box_union_s, and for up to track_carry_s before a first sighting and after a last one
a carried box keeps covering the object, its margin growing with the time carried. On the display video the carried
box follows the object with optical flow on the unblurred proxy (flow_carry) and also covers its path back to the
sighting it left, within a reach (track_flow_cover_path, review 11/1); every other consumer extrapolates it.
The display video stays a derived file: rederive_display makes it again from the renter's original and the stored
lanes, byte for byte for a run made with this code on the same ffmpeg and OpenCV builds.
"""

from __future__ import annotations

import asyncio
import bisect
import hashlib
import json
import math
import re
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Awaitable, Callable, Iterable

import cv2
import numpy as np
from meta_sam_parser import decode_mask_to_raster

from . import paths, sam_client
from .config import redact
from .contracts import (
    PRIVACY_NOUNS,
    BlurRecord,
    BlurredProxyRecord,
    DeletionKind,
    DeletionRecord,
    Deletions,
    IndexFrame,
    Noun,
    PanelTrack,
    ResponsesStore,
    RunManifest,
    VideoRecord,
    load_thresholds,
    read_json,
    read_json_list,
    write_json,
)

Box4 = list[float]

SRC_REFERENCE_EDGE = 3840  # the "at 4K" the privacy thresholds are written for
_RETRY_SLEEP_S = 0.5  # between Files API delete retries; tests set it to 0

# Keys added on Oct 31 (config/thresholds.json carries the same values; tests/test_privacy.py compares them). A run
# sealed before then has none of them in its thresholds snapshot and gets these, so a display video re-derived for an
# old run is never blurred more weakly than a new one.
PRIVACY_DEFAULTS: dict[str, float | bool] = {
    "box_blocks_across": 4.0,
    "track_box_union_s": 0.1,
    "track_carry_s": 3.0,  # 2.0 until the review of 11/1 (pairC_core2 showed a plate edge-on 34 frames early)
    "track_carry_fit_s": 0.4,
    "track_carry_grow_per_s": 0.5,
    "track_carry_speed_max_frac_per_s": 1.0,
    "track_flow_grow_per_s": 0.2,
    "track_flow_cover_path": True,  # review 11/1: a flow-carried box also covers its path back to the sighting it left
    "track_flow_cover_max_frac": 0.1,  # review 11/1: that path cover reaches at most this much of the long edge past the box
}

# Optical flow for the display video's carry (flow_carry): grayscale frames at most FLOW_MAX_EDGE px on the long edge,
# a FLOW_GRID x FLOW_GRID grid of points over the box (or over FLOW_MIN_SPAN_FRAC of the frame's long edge when the box
# is smaller, so an edge-on plate still has texture around it), pyramidal Lucas-Kanade both ways, and a step kept only
# when FLOW_MIN_POINTS points come back within FLOW_FB_MAX_PX of where they started.
FLOW_MAX_EDGE = 480
FLOW_GRID = 10
FLOW_MIN_SPAN_FRAC = 0.08
FLOW_MIN_POINTS = 8
FLOW_FB_MAX_PX = 1.0
_LK = dict(winSize=(15, 15), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def privacy_params(th: dict[str, Any]) -> dict[str, Any]:
    """th["privacy"] over PRIVACY_DEFAULTS."""
    return {**PRIVACY_DEFAULTS, **th.get("privacy", {})}


# Mask blur (R27, 11/2). On hero_focus4 SAM 3.1's "face" lane tracked the Focus's windshield for 264 frames (a 380 x
# 150 px box on the 960 px proxy, its mask the glass), and the box rule pixelated that box, carried it 3 s past the
# track growing by half its size a second, in 4 blocks across: the pickup mirror, roof and hood crops were mostly
# mosaic and the blocks spilled onto the right front door and fender. Under privacy.mask_blur a face or plate is
# blurred inside its SAM mask (the lane's mask on the video, the image check's on a keyframe) with a small dilation,
# none for a face whose mask lies on the car's own glass or paint (a reflection), by a smooth blur whose strength
# follows the object's own size (samples_across samples on its long side), so it looks the same in both walks. Faces
# are never carried; a plate is still carried before and after its sightings (edge-on plates), its box growing at most
# plate_carry_grow_max. Runs frozen before it have no mask_blur leaf and blur as they did.
MASK_BLUR_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "dilate_frac": 0.03,
    "dilate_px_at_4k": 12,
    "samples_across": 4.0,
    "on_car_min": 0.6,
    "on_car_frames": 5,
    "plate_carry_grow_max": 0.5,
}


def mask_blur_settings(th: dict[str, Any] | None) -> dict[str, Any] | None:
    """thresholds privacy.mask_blur with its defaults filled, or None when absent or off (every run frozen before R27)."""
    raw = (th or {}).get("privacy", {}).get("mask_blur")
    if not isinstance(raw, dict) or not raw.get("enabled", False):
        return None
    return {**MASK_BLUR_DEFAULTS, **raw}


def carry_params(th: dict[str, Any]) -> dict[str, Any]:
    """privacy_params plus, under mask_blur, the plate carry's growth cap and no carry for faces."""
    p = privacy_params(th)
    mb = mask_blur_settings(th)
    if mb is not None:
        p = {**p, "track_carry_grow_max": float(mb["plate_carry_grow_max"]), "carry_nouns": (Noun.LICENSE_PLATE,)}
    return p


def _carries(noun: Noun, p: dict[str, Any]) -> bool:
    nouns = p.get("carry_nouns")
    return nouns is None or noun in nouns


def _capped(grow: float, p: dict[str, Any]) -> float:
    cap = p.get("track_carry_grow_max")
    return grow if cap is None else min(grow, float(cap))


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(text: str) -> datetime:
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _sha256_file(path: Path) -> str:
    with open(path, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


# Pixelation ----------------------------------------------------------------------------------------------------


def _scale_for(shape: tuple[int, ...]) -> float:
    h, w = shape[:2]
    return max(h, w) / SRC_REFERENCE_EDGE


def _block_for(th: dict[str, Any], shape: tuple[int, ...]) -> int:
    return max(2, int(round(th["privacy"]["pixel_block_px_at_4k"] * _scale_for(shape))))


def _dilate_px_for(th: dict[str, Any], shape: tuple[int, ...]) -> int:
    return int(round(th["privacy"]["track_box_dilation_px"] * _scale_for(shape)))


def _int_region(box: Iterable[float], shape: tuple[int, ...], *, dilate_frac: float, dilate_px: float) -> tuple[int, int, int, int] | None:
    """Dilate a half-open box by dilate_frac of its long side plus dilate_px on every side (the rule _dilate_mask
    applies to a mask, so a plate seen edge-on, a thin box, still gets a margin across its short side) and round it
    outward to whole pixels inside the frame."""
    h, w = shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    if x2 <= x1 or y2 <= y1:
        return None
    px = py = max(x2 - x1, y2 - y1) * dilate_frac + dilate_px
    ix1, iy1 = max(0, math.floor(x1 - px)), max(0, math.floor(y1 - py))
    ix2, iy2 = min(w, math.ceil(x2 + px)), min(h, math.ceil(y2 + py))
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return ix1, iy1, ix2, iy2


def _block_mean(sub: np.ndarray, block: int) -> np.ndarray:
    """Replace every block x block tile by its mean colour (edge tiles use edge padding)."""
    squeeze = sub.ndim == 2
    data = sub[..., None] if squeeze else sub
    h, w = data.shape[:2]
    padded = np.pad(data, ((0, (-h) % block), (0, (-w) % block), (0, 0)), mode="edge").astype(np.float64)
    hb, wb = padded.shape[0] // block, padded.shape[1] // block
    means = padded.reshape(hb, block, wb, block, -1).mean(axis=(1, 3))
    out = np.repeat(np.repeat(means, block, axis=0), block, axis=1)[:h, :w]
    out = np.clip(np.rint(out), 0, 255).astype(sub.dtype)
    return out[..., 0] if squeeze else out


def block_for_box(box: Iterable[float], *, block_px: int, blocks_across: float | None) -> int:
    """The pixel block for one box: at least block_px, and big enough that the box's long side spans at most
    blocks_across blocks, so a plate or face is never left readable by a block that is small next to it however close
    it is (pairC_core1's return walk at 4.4 s: 5 px blocks on a 127 px plate box left the plate readable)."""
    x1, y1, x2, y2 = (float(v) for v in box)
    block = int(block_px)
    long_side = max(x2 - x1, y2 - y1)
    if blocks_across and long_side > 0:
        block = max(block, int(math.ceil(long_side / float(blocks_across) - 1e-9)))
    return block


def _pixelate_where(out: np.ndarray, where: np.ndarray, block: int, seeds: Iterable[tuple[int, int, int]] = ()) -> None:
    """In place: block means over each connected region's bounding box, composited on that region's own pixels. A
    region's block is the largest of `block` and of every seed (x, y, block) inside it, so boxes that overlap share
    the coarsest block. One box per region keeps a face in one corner and a plate in the other from averaging the
    whole 4K frame (about 800 MB of float64 in one go)."""
    if not where.any():
        return
    count, labels, stats, _ = cv2.connectedComponentsWithStats(where.astype(np.uint8), connectivity=8)
    blocks = [int(block)] * count
    h_img, w_img = where.shape[:2]
    for x, y, b in seeds:
        if 0 <= y < h_img and 0 <= x < w_img and labels[y, x] > 0:
            blocks[labels[y, x]] = max(blocks[labels[y, x]], int(b))
    for i in range(1, count):
        x, y, w, h = (int(v) for v in stats[i, :4])
        sub = out[y : y + h, x : x + w]
        m = labels[y : y + h, x : x + w] == i
        pix = _block_mean(sub, blocks[i])
        sub[m] = pix[m]


def pixelate(
    rgb: np.ndarray,
    regions: list[Box4],
    *,
    block_px: int,
    dilate_frac: float = 0.0,
    dilate_px: int = 0,
    blocks_across: float | None = PRIVACY_DEFAULTS["box_blocks_across"],
) -> np.ndarray:
    """Pure: each region dilated then replaced by block means; pixels outside every region are returned unchanged.
    The block is block_px, grown per region so that its largest box spans at most blocks_across blocks (None keeps
    block_px everywhere)."""
    if block_px < 1:
        raise ValueError("block_px must be at least 1")
    out = np.array(rgb, copy=True)
    where = np.zeros(out.shape[:2], dtype=bool)
    seeds: list[tuple[int, int, int]] = []
    for box in regions:
        region = _int_region(box, out.shape, dilate_frac=dilate_frac, dilate_px=dilate_px)
        if region is None:
            continue
        x1, y1, x2, y2 = region
        where[y1:y2, x1:x2] = True
        seeds.append((x1, y1, block_for_box(box, block_px=int(block_px), blocks_across=blocks_across)))
    _pixelate_where(out, where, int(block_px), seeds)
    return out


def _dilate_mask(mask: np.ndarray, *, dilate_frac: float, dilate_px: float) -> np.ndarray:
    """Grow a boolean mask by dilate_frac of its bounding box plus dilate_px, inside its own neighbourhood. A distance
    transform does it in one pass whatever the radius; it covers every pixel an elliptical kernel of that radius
    would."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return mask
    bw, bh = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
    r = int(math.ceil(max(bw, bh) * dilate_frac + dilate_px))
    if r <= 0:
        return mask
    h, w = mask.shape
    y1, y2 = max(0, int(ys.min()) - r), min(h, int(ys.max()) + 1 + r)
    x1, x2 = max(0, int(xs.min()) - r), min(w, int(xs.max()) + 1 + r)
    outside = (~mask[y1:y2, x1:x2]).astype(np.uint8)
    dist = cv2.distanceTransform(outside, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    out = mask.copy()
    out[y1:y2, x1:x2] |= dist <= r + 0.5
    return out


def detection_mask(det: sam_client.Detection, media_size: tuple[int, int]) -> np.ndarray:
    """Boolean (H, W) frame mask of one image detection: its box-local mask pasted at the box, or the box itself."""
    w, h = media_size
    out = np.zeros((h, w), dtype=bool)
    x1, y1 = int(math.floor(det.box.left)), int(math.floor(det.box.top))
    x2, y2 = int(math.ceil(det.box.right)), int(math.ceil(det.box.bottom))
    if det.mask is None:
        out[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)] = True
        return out
    raw = decode_mask_to_raster(det.mask)
    local = np.frombuffer(raw, dtype=np.uint8).reshape(det.mask.height, det.mask.width).astype(bool)
    # SAM 3.1 sends the raster of an object wider or taller than 1024 px scaled down to 1024 on its long side (box
    # 1101 x 154 came with a 1024 x 143 raster, box fix 11/1); it is stretched back over its box, or the mask would sit
    # short of the box's right and bottom edges. A raster within a pixel of the box is pasted as it came.
    bw, bh = int(round(det.box.right - det.box.left)), int(round(det.box.bottom - det.box.top))
    if bw > 0 and bh > 0 and (abs(local.shape[1] - bw) > 1 or abs(local.shape[0] - bh) > 1):
        import cv2

        local = cv2.resize(local.astype(np.uint8), (bw, bh), interpolation=cv2.INTER_NEAREST).astype(bool)
    ty1, tx1 = max(0, y1), max(0, x1)
    ty2, tx2 = min(h, y1 + local.shape[0]), min(w, x1 + local.shape[1])
    if ty2 > ty1 and tx2 > tx1:
        out[ty1:ty2, tx1:tx2] = local[ty1 - y1 : ty2 - y1, tx1 - x1 : tx2 - x1]
    return out


def plate_extent(mask: np.ndarray) -> tuple[float, float, float] | None:
    """(long side of the min-area rectangle in px, apparent h / w, tilt degrees) of an unblurred plate mask.

    The rectangle is fitted to pixel centres, so one pixel is added to each side to get the pixel extent. Tilt
    follows 5.12: a frontal US plate has h / w = 0.5, so apparent h / w = 0.5 / cos(theta). None for an empty mask.
    """
    ys, xs = np.nonzero(mask)
    if ys.size < 4:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float32)
    (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
    long_px, short_px = float(max(rw, rh)) + 1.0, float(min(rw, rh)) + 1.0
    if long_px <= 0:
        return None
    hw = short_px / long_px
    tilt = math.degrees(math.acos(min(1.0, 0.5 / hw))) if hw > 0.5 else 0.0
    return long_px, hw, tilt


# SAM image checks (LIVE through sam_client) ----------------------------------------------------------------------


def _detections(run: sam_client.SegmentationRun) -> list[sam_client.Detection]:
    return [d for frame in sorted(run.frames) for d in run.frames[frame]]


def _box4(det: sam_client.Detection) -> Box4:
    return [float(det.box.left), float(det.box.top), float(det.box.right), float(det.box.bottom)]


def _request_id(run: sam_client.SegmentationRun) -> str | None:
    for name in ("request_id", "response_id"):
        value = getattr(run, name, None)
        if isinstance(value, str) and value:
            return value
    events_path = getattr(run, "events_path", None)
    if events_path is not None:
        stem = Path(events_path).name.removesuffix(".ndjson").removesuffix(".gz")
        return stem.rsplit(".", 1)[-1] if "." in stem else stem
    return None


def _touches_border(det: sam_client.Detection, media_size: tuple[int, int]) -> bool:
    w, h = media_size
    return det.box.left <= 0 or det.box.top <= 0 or det.box.right >= w or det.box.bottom >= h


def _best_plate(plates: list[sam_client.Detection], media_size: tuple[int, int], th: dict[str, Any]) -> sam_client.Detection | None:
    """The plate the ruler may use: c at or above plate_c_min, not touching the border, largest long side."""
    c_min = th["ruler"]["plate_c_min"]
    good = [d for d in plates if (d.confidence is None or d.confidence >= c_min) and not _touches_border(d, media_size)]
    if not good:
        return None
    return max(good, key=lambda d: max(d.box.width, d.box.height))


async def image_checks(
    rgb_unblurred: np.ndarray, *, run_id: str, role: str, tag: str, th: dict[str, Any]
) -> tuple[list[sam_client.Detection], list[sam_client.Detection], BlurRecord]:
    """LIVE: "license plate" then "face" on the in-memory unblurred frame; the plate extent is recorded here, before
    any pixel is touched. Two sam_image ledger rows, written by sam_client.segment_image."""
    media_size = (int(rgb_unblurred.shape[1]), int(rgb_unblurred.shape[0]))
    plate_run, face_run = await asyncio.gather(
        sam_client.segment_image(rgb_unblurred, Noun.LICENSE_PLATE.value, run_id=run_id, role=role, tag=f"{tag}.license_plate"),
        sam_client.segment_image(rgb_unblurred, Noun.FACE.value, run_id=run_id, role=role, tag=f"{tag}.face"),
    )
    plates, faces = _detections(plate_run), _detections(face_run)
    record = BlurRecord(
        plate_boxes=[_box4(d) for d in plates],
        face_boxes=[_box4(d) for d in faces],
        request_ids=[rid for rid in (_request_id(plate_run), _request_id(face_run)) if rid],
    )
    best = _best_plate(plates, media_size, th)
    if best is not None:
        extent = plate_extent(detection_mask(best, media_size))
        if extent is not None:
            record.plate_ruler_px, record.plate_tilt_hw = extent[0], extent[1]
            record.plate_ruler_c = best.confidence
    return plates, faces, record


# Track boxes and frame blur ---------------------------------------------------------------------------------------


def _frames(seconds: float, fps: float) -> int:
    return int(math.ceil(float(seconds) * fps - 1e-9))


def _run_edges(frames: list[int], n_frames: int | None, carry: int) -> list[tuple[int, int, int]]:
    """(row index, side, frames to carry) for every run of consecutive sightings in a sorted frame list: side -1 before
    the run's first sighting, +1 after its last, at most carry frames and never onto another sighting, before frame 0
    or past the last frame (n_frames None: no end known)."""
    edges: list[tuple[int, int, int]] = []
    n = len(frames)
    for i, f in enumerate(frames):
        if i == 0 or frames[i - 1] < f - 1:
            room = f if i == 0 else f - frames[i - 1] - 1
            if min(room, carry) > 0:
                edges.append((i, -1, min(room, carry)))
        if i == n - 1 or frames[i + 1] > f + 1:
            room = (carry if n_frames is None else n_frames - 1 - f) if i == n - 1 else frames[i + 1] - f - 1
            if min(room, carry) > 0:
                edges.append((i, 1, min(room, carry)))
    return edges


def _carry_seed(rows: list[Any], i: int, side: int, *, fps: float, frame_long: float, p: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    """(cx, cy, w, h, vx, vy) of a carry from rows[i], side -1 backward from a run's first sighting or +1 forward from
    its last: the box centre; a size no smaller than that box or than the median of the sightings within
    track_carry_fit_s of it; and a velocity in proxy px per frame, the median step over those sightings with its speed
    capped at track_carry_speed_max_frac_per_s of the frame's long edge per second."""
    fit = max(1, _frames(p["track_carry_fit_s"], fps))
    f0 = rows[i].frame
    window = []
    j = i
    while 0 <= j < len(rows) and abs(rows[j].frame - f0) <= fit:
        window.append(rows[j])
        j -= side  # into the run: forward in time before a first sighting, backward after a last one
    window.sort(key=lambda r: r.frame)
    x1, y1, x2, y2 = (float(v) for v in rows[i].box)
    w = max(x2 - x1, float(np.median([r.box[2] - r.box[0] for r in window])))
    h = max(y2 - y1, float(np.median([r.box[3] - r.box[1] for r in window])))
    steps = [
        (((b.box[0] + b.box[2]) - (a.box[0] + a.box[2])) / 2 / (b.frame - a.frame), ((b.box[1] + b.box[3]) - (a.box[1] + a.box[3])) / 2 / (b.frame - a.frame))
        for a, b in zip(window, window[1:])
        if b.frame > a.frame
    ]
    vx = float(np.median([v[0] for v in steps])) if steps else 0.0
    vy = float(np.median([v[1] for v in steps])) if steps else 0.0
    vmax = float(p["track_carry_speed_max_frac_per_s"]) * frame_long / fps
    speed = math.hypot(vx, vy)
    if speed > vmax > 0:
        vx, vy = vx * vmax / speed, vy * vmax / speed
    return (x1 + x2) / 2, (y1 + y2) / 2, w, h, vx, vy


def _box_around(cx: float, cy: float, w: float, h: float, grow: float) -> Box4:
    k = 1.0 + grow
    return [round(cx - w * k / 2, 3), round(cy - h * k / 2, 3), round(cx + w * k / 2, 3), round(cy + h * k / 2, 3)]


def _extrapolated(rows: list[Any], i: int, side: int, d: int, *, fps: float, frame_long: float, p: dict[str, Any]) -> Box4:
    """The carried box d frames from rows[i] when no flow is known: moved at the seed velocity, its margin grown by
    track_carry_grow_per_s of its size per second carried."""
    cx, cy, w, h, vx, vy = _carry_seed(rows, i, side, fps=fps, frame_long=frame_long, p=p)
    return _box_around(cx + side * vx * d, cy + side * vy * d, w, h, _capped(float(p["track_carry_grow_per_s"]) * d / fps, p))


Carried = dict[tuple[str, int, int], Box4]


def track_boxes_at(
    f_proxy: int,
    tracks: list[PanelTrack],
    *,
    fps: float,
    media_size_proxy: tuple[int, int],
    media_size_src: tuple[int, int],
    th: dict[str, Any],
    nouns: tuple[Noun, ...] = PRIVACY_NOUNS,
    carried: Carried | None = None,
    sightings: bool = True,
) -> list[Box4]:
    """Plate and face track boxes at a proxy frame, scaled to source pixels: every sighting within track_box_union_s
    of the frame (so a one-frame box glitch never uncovers the object); after a sighting, its box held for
    track_gap_bridge_s; and before each run of sightings and after it, for up to track_carry_s, a carried box: moved
    on from the run's edge and grown by track_carry_grow_per_s of its size per second carried (extrapolated), or the
    box flow_carry followed on the unblurred proxy, keyed (track_id, frame, side) in carried. That covers an edge-on
    plate before SAM's lane picks it up and one that leaves after the lane lets go. Over-blurring is the safe side.
    Under privacy.mask_blur (R27) a face track is never carried and a plate's carried box grows at most
    plate_carry_grow_max; sightings=False leaves out the sighting and bridge boxes (their masks are blurred instead)."""
    p = carry_params(th)
    union = _frames(p["track_box_union_s"], fps)
    bridge = int(math.ceil(p["track_gap_bridge_s"] * fps))
    frame_long = float(max(media_size_proxy))
    sx = media_size_src[0] / media_size_proxy[0]
    sy = media_size_src[1] / media_size_proxy[1]
    out: list[Box4] = []
    seen: set[tuple[float, ...]] = set()
    f = int(f_proxy)
    for track in tracks:
        if track.noun not in nouns or not track.per_frame:
            continue
        carry = _frames(p["track_carry_s"], fps) if _carries(track.noun, p) else 0
        rows = sorted(track.per_frame, key=lambda r: r.frame)
        frames = [r.frame for r in rows]
        boxes: list[Any] = [rows[j].box for j in range(bisect.bisect_left(frames, f - union), bisect.bisect_right(frames, f + union))] if sightings else []
        k = bisect.bisect_left(frames, f)
        if k == len(frames) or frames[k] != f:
            for i, side in ((k - 1, 1), (k, -1)):  # the last sighting before the frame, the first one after it
                if not 0 <= i < len(frames):
                    continue
                d = abs(f - frames[i])
                if sightings and side > 0 and d <= bridge:
                    boxes.append(rows[i].box)
                if 0 < d <= carry:
                    known = carried.get((track.track_id, f, side)) if carried else None
                    boxes.append(known if known is not None else _extrapolated(rows, i, side, d, fps=fps, frame_long=frame_long, p=p))
        for b in boxes:
            scaled = [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]
            key = tuple(scaled)
            if key not in seen:
                seen.add(key)
                out.append(scaled)
    return out


def _flow_step(a: np.ndarray | None, b: np.ndarray | None, cx: float, cy: float, w: float, h: float, *, scale: float, frame_long: float, vmax: float) -> tuple[float, float] | None:
    """The median optical flow, in proxy px, of a grid of points around a box from grayscale frame a to frame b; None
    when a frame is missing, fewer than FLOW_MIN_POINTS points survive the forward and backward check, or the step is
    faster than vmax."""
    if a is None or b is None:
        return None
    gh, gw = a.shape[:2]
    span_w = max(w, FLOW_MIN_SPAN_FRAC * frame_long) * scale
    span_h = max(h, FLOW_MIN_SPAN_FRAC * frame_long) * scale
    offsets = np.linspace(-0.45, 0.45, FLOW_GRID)
    pts = [(cx * scale + ox * span_w, cy * scale + oy * span_h) for oy in offsets for ox in offsets]
    pts = [(x, y) for x, y in pts if 1.0 <= x <= gw - 2.0 and 1.0 <= y <= gh - 2.0]
    if len(pts) < FLOW_MIN_POINTS:
        return None
    p0 = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
    p1, st1, _ = cv2.calcOpticalFlowPyrLK(a, b, p0, None, **_LK)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(b, a, p1, None, **_LK)
    fb = np.linalg.norm((back - p0).reshape(-1, 2), axis=1)
    ok = (st1.reshape(-1) == 1) & (st2.reshape(-1) == 1) & (fb < FLOW_FB_MAX_PX)
    if int(ok.sum()) < FLOW_MIN_POINTS:
        return None
    moved = (p1 - p0).reshape(-1, 2)[ok]
    dx, dy = float(np.median(moved[:, 0])) / scale, float(np.median(moved[:, 1])) / scale
    if math.hypot(dx, dy) > vmax:
        return None
    return dx, dy


def carry_frames_needed(tracks: list[PanelTrack], *, fps: float, n_frames: int, th: dict[str, Any], nouns: tuple[Noun, ...] = PRIVACY_NOUNS) -> set[int]:
    """The proxy frames flow_carry reads: every run edge and the frames it carries into."""
    p = carry_params(th)
    carry = _frames(p["track_carry_s"], fps)
    needed: set[int] = set()
    for track in tracks:
        if track.noun not in nouns or not track.per_frame or not _carries(track.noun, p):
            continue
        frames = sorted({r.frame for r in track.per_frame})
        for i, side, span in _run_edges(frames, n_frames, carry):
            needed.update(frames[i] + side * d for d in range(span + 1))
    return needed


def flow_carry(
    grays: dict[int, np.ndarray],
    tracks: list[PanelTrack],
    *,
    fps: float,
    media_size_proxy: tuple[int, int],
    n_frames: int,
    th: dict[str, Any],
    nouns: tuple[Noun, ...] = PRIVACY_NOUNS,
) -> Carried:
    """The display video's carried boxes, keyed (track_id, proxy frame, side) for track_boxes_at: from each run edge the
    box moves frame by frame with the median optical flow around it on the unblurred proxy (grays: grayscale frames
    by proxy frame, all at one size), never smaller than the edge box, its margin grown by track_flow_grow_per_s of
    its size per second. Where the flow loses the object (a missing frame counts as lost) the box goes on at its last
    step, capped at track_carry_speed_max_frac_per_s, and grows by track_carry_grow_per_s from there. With
    track_flow_cover_path each carried box also covers every box before it back to the sighting it left: on a small box
    the flow can follow the ground instead of the object (pairC_core1's return front plate, frames 185 to 189: the box
    drifted 80 px off the edge-on plate while the plate stayed near its first sighting), so the path is covered, out
    to track_flow_cover_max_frac of the frame's long edge past the box (a whole pan's path would cover a car's side).
    Pure and deterministic for the same frames, so a re-render gives the same boxes."""
    p = carry_params(th)
    carry = _frames(p["track_carry_s"], fps)
    frame_long = float(max(media_size_proxy))
    vmax = float(p["track_carry_speed_max_frac_per_s"]) * frame_long / fps
    g_flow, g_kin = float(p["track_flow_grow_per_s"]), float(p["track_carry_grow_per_s"])
    cover_path = bool(p.get("track_flow_cover_path", False))
    reach = float(p.get("track_flow_cover_max_frac", 0.0)) * frame_long
    sample = next(iter(grays.values()), None)
    scale = float(sample.shape[1]) / float(media_size_proxy[0]) if sample is not None else 1.0
    out: Carried = {}
    for track in tracks:
        if track.noun not in nouns or not track.per_frame or not _carries(track.noun, p):
            continue
        rows = sorted(track.per_frame, key=lambda r: r.frame)
        frames = [r.frame for r in rows]
        for i, side, span in _run_edges(frames, n_frames, carry):
            cx, cy, w, h, vx, vy = _carry_seed(rows, i, side, fps=fps, frame_long=frame_long, p=p)
            step_x, step_y = side * vx, side * vy
            followed: int | None = None  # frames the flow held for, once it lets go
            path = _box_around(cx, cy, w, h, 0.0)  # the boxes so far, from the sighting this carry leaves
            for d in range(1, span + 1):
                g = frames[i] + side * d
                moved = None
                if followed is None:
                    moved = _flow_step(grays.get(g - side), grays.get(g), cx, cy, w, h, scale=scale, frame_long=frame_long, vmax=vmax)
                    if moved is None:
                        followed = d - 1
                if moved is not None:
                    step_x, step_y = moved
                cx, cy = cx + step_x, cy + step_y
                held = d if followed is None else followed
                box = _box_around(cx, cy, w, h, _capped((g_flow * held + g_kin * (d - held)) / fps, p))
                if cover_path:
                    path = [min(path[0], box[0]), min(path[1], box[1]), max(path[2], box[2]), max(path[3], box[3])]
                    box = [max(path[0], box[0] - reach), max(path[1], box[1] - reach), min(path[2], box[2] + reach), min(path[3], box[3] + reach)]
                out[(track.track_id, g, side)] = box
    return out


def blur_frame(
    rgb_unblurred: np.ndarray,
    *,
    track_boxes_src: list[Box4],
    detections: list[sam_client.Detection],
    keep_plate_region: bool,
    th: dict[str, Any],
    plate_track_boxes_src: list[Box4] = (),
    plate_detections: list[sam_client.Detection] = (),
) -> np.ndarray:
    """Pixelate the union of track boxes and image masks. `track_boxes_src` and `detections` are always pixelated
    (faces, and plates when the caller merged them in); `plate_track_boxes_src` and `plate_detections` are skipped
    only when keep_plate_region is true. Dilation is track_box_dilation_frac plus track_box_dilation_px and the block
    is pixel_block_px_at_4k, both scaled by the frame's long edge over 3840; each region's block then grows so its
    largest box or mask spans at most box_blocks_across blocks."""
    out = np.array(rgb_unblurred, copy=True)
    media_size = (int(out.shape[1]), int(out.shape[0]))
    block = _block_for(th, out.shape)
    across = privacy_params(th)["box_blocks_across"]
    dfrac = th["privacy"]["track_box_dilation_frac"]
    dpx = _dilate_px_for(th, out.shape)
    boxes = list(track_boxes_src) + ([] if keep_plate_region else list(plate_track_boxes_src))
    dets = list(detections) + ([] if keep_plate_region else list(plate_detections))
    where = np.zeros(out.shape[:2], dtype=bool)
    seeds: list[tuple[int, int, int]] = []
    for box in boxes:
        region = _int_region(box, out.shape, dilate_frac=dfrac, dilate_px=dpx)
        if region is not None:
            x1, y1, x2, y2 = region
            where[y1:y2, x1:x2] = True
            seeds.append((x1, y1, block_for_box(box, block_px=block, blocks_across=across)))
    for det in dets:
        grown = _dilate_mask(detection_mask(det, media_size), dilate_frac=dfrac, dilate_px=dpx)
        if grown.any():
            y, x = divmod(int(np.argmax(grown)), grown.shape[1])
            seeds.append((x, y, block_for_box(_box4(det), block_px=block, blocks_across=across)))
            where |= grown
    _pixelate_where(out, where, block, seeds)
    return out


async def blur_keyframe(
    run_id: str,
    role: str,
    frame_rgb: np.ndarray,
    *,
    f_proxy: int,
    tracks: list[PanelTrack],
    fps: float,
    sizes: tuple[tuple[int, int], tuple[int, int]],
    keep_plate: bool,
    th: dict[str, Any],
    checks: Callable[..., Awaitable[tuple[list[sam_client.Detection], list[sam_client.Detection], BlurRecord]]] | None = None,
) -> tuple[np.ndarray, BlurRecord]:
    """Step 7 order: image checks FIRST on the unblurred frame (plate extent recorded), THEN pixelation. Writes
    nothing; sizes is (proxy size, source size). Every frame returned here becomes a stored crop, a keyframe frame or
    a Spark input, so the plate is pixelated whatever keep_plate says (section 8 items 5 and 6). The run owner's
    plate-kept variant is a separate blur_frame(keep_plate_region=True) call made only by the sheet writer;
    keep_plate stays in the signature because the contract names it. checks replaces image_checks (the orchestrator
    passes one that asks once per distinct frame of a run)."""
    del keep_plate
    proxy_size, src_size = sizes
    if (int(frame_rgb.shape[1]), int(frame_rgb.shape[0])) != tuple(src_size):
        raise ValueError(f"frame is {frame_rgb.shape[1]}x{frame_rgb.shape[0]} but sizes says source {src_size[0]}x{src_size[1]}")
    plates, faces, record = await (checks or image_checks)(frame_rgb, run_id=run_id, role=role, tag=f"f{f_proxy:06d}", th=th)
    if mask_blur_settings(th) is not None:
        masks = _walk_masks(run_id, role, tracks, proxy_size)
        blurred, where = await asyncio.to_thread(
            blur_keyframe_masks, frame_rgb, f_proxy=f_proxy, tracks=tracks, fps=fps, proxy_size=proxy_size, masks=masks, plates=plates, faces=faces, th=th
        )
        record.set_blurred(where)
        return blurred, record
    common = dict(fps=fps, media_size_proxy=proxy_size, media_size_src=src_size, th=th)
    face_boxes = track_boxes_at(f_proxy, tracks, nouns=(Noun.FACE,), **common)
    plate_boxes = track_boxes_at(f_proxy, tracks, nouns=(Noun.LICENSE_PLATE,), **common)
    blurred = await asyncio.to_thread(  # a few hundred ms of numpy on a 4K frame; keep it off the event loop
        blur_frame,
        frame_rgb,
        track_boxes_src=face_boxes,
        detections=faces,
        keep_plate_region=False,
        th=th,
        plate_track_boxes_src=plate_boxes,
        plate_detections=plates,
    )
    return blurred, record


# Mask blur (R27): masks from the stored lanes, the smooth blur, keyframes and the display video ------------------------


class TrackMasks:
    """Proxy-size masks of a walk's plate, face and car objects, decoded from its stored lanes (sam/<role>/*.index.json
    and the gzipped events) on first use and kept. Thread safe: keyframe blurs run in worker threads."""

    def __init__(self, run_id: str, role: str, media_size: tuple[int, int]) -> None:
        self.run_id, self.role = run_id, role
        self.media_size = (int(media_size[0]), int(media_size[1]))
        self._lanes: dict[Noun, tuple[Path | None, dict[tuple[int, str], Any], dict[int, str]]] = {}
        self._masks: dict[tuple[Noun, int, str], np.ndarray | None] = {}
        self._on_car: dict[str, bool] = {}
        self._decoder: Any = None
        self._lock = threading.RLock()

    def _lane(self, noun: Noun) -> tuple[Path | None, dict[tuple[int, str], Any], dict[int, str]]:
        if noun not in self._lanes:
            index = paths.sam_dir(self.run_id, self.role) / f"{paths.lane_id(noun)}.index.json"
            frames = read_json_list(index, IndexFrame) if index.is_file() else []
            lookup = {(fr.frame, o.id): o for fr in frames for o in fr.objects}
            largest = {fr.frame: max(fr.objects, key=lambda o: (o.box[2] - o.box[0]) * (o.box[3] - o.box[1])).id for fr in frames if fr.objects}
            lane = None
            for name in (f"{paths.lane_id(noun)}.ndjson.gz", f"{paths.lane_id(noun)}.ndjson"):
                candidate = paths.sam_dir(self.run_id, self.role) / name
                if candidate.is_file():
                    lane = candidate
                    break
            self._lanes[noun] = (lane, lookup, largest)
        return self._lanes[noun]

    def object_mask(self, noun: Noun, frame: int, oid: str) -> np.ndarray | None:
        """The object's mask pasted on the proxy frame; its box filled when the lane carries no mask for it; None when
        the lane has no such object at that frame."""
        with self._lock:
            key = (noun, int(frame), str(oid))
            if key in self._masks:
                return self._masks[key]
            from . import fastmask, sam_wire

            lane, lookup, _ = self._lane(noun)
            obj = lookup.get((int(frame), str(oid)))
            out: np.ndarray | None = None
            if obj is not None:
                w, h = self.media_size
                if obj.mask is not None and lane is not None:
                    if self._decoder is None:
                        self._decoder = fastmask.LaneMasks()
                    raster = self._decoder.decode(lane, [obj.mask])[0]
                    out = sam_wire.paste_mask(np.asarray(raster, dtype=bool), list(obj.box), obj.mask, self.media_size)
                else:
                    out = np.zeros((h, w), dtype=bool)
                    x1, y1, x2, y2 = (int(round(v)) for v in obj.box)
                    out[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)] = True
            self._masks[key] = out
            return out

    def track_mask(self, track: PanelTrack, frame: int) -> np.ndarray | None:
        oid = next((fr.object_id for fr in track.fragments if fr.first_frame <= frame <= fr.last_frame), None)
        return None if oid is None else self.object_mask(track.noun, frame, oid)

    def car_mask(self, frame: int) -> np.ndarray | None:
        """The largest car object's mask at the proxy frame (the object the keyframe car test reads)."""
        _, _, largest = self._lane(Noun.CAR)
        oid = largest.get(int(frame))
        return None if oid is None else self.object_mask(Noun.CAR, frame, oid)

    def on_car(self, track: PanelTrack, settings: dict[str, Any]) -> bool:
        """True when the track's mask lies on the car's own glass or paint: the median share of its pixels inside the car
        mask, over up to on_car_frames frames spread over the track, at or above on_car_min (a reflection, or SAM
        calling the glass a face). A person standing in front of the car is not in the car's mask."""
        with self._lock:
            if track.track_id in self._on_car:
                return self._on_car[track.track_id]
            rows = sorted(r.frame for r in track.per_frame)
            n = max(1, int(settings["on_car_frames"]))
            picks = sorted({rows[int(round(i * (len(rows) - 1) / max(1, n - 1)))] for i in range(n)}) if rows else []
            shares = []
            for f in picks:
                m, car = self.track_mask(track, f), self.car_mask(f)
                if m is not None and car is not None and m.any():
                    shares.append(float((m & car).sum()) / float(m.sum()))
            got = bool(shares) and float(np.median(shares)) >= float(settings["on_car_min"])
            self._on_car[track.track_id] = got
            return got


_TRACK_MASKS: dict[tuple[str, str, str], TrackMasks] = {}


def track_masks_for(run_id: str, role: str, media_size: tuple[int, int]) -> TrackMasks:
    """One TrackMasks per walk while its lanes stay the same files (a re-run under the same id gets a new one)."""
    index = paths.sam_dir(run_id, role) / f"{paths.lane_id(Noun.FACE)}.index.json"
    stamp = f"{paths.run_dir(run_id)}:{index.stat().st_mtime_ns if index.is_file() else 0}"
    key = (run_id, role, stamp)
    got = _TRACK_MASKS.get(key)
    if got is None or got.media_size != (int(media_size[0]), int(media_size[1])):
        if len(_TRACK_MASKS) > 8:
            _TRACK_MASKS.clear()
        got = _TRACK_MASKS[key] = TrackMasks(run_id, role, media_size)
    return got


def _fit_bool(mask: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    h, w = int(shape[0]), int(shape[1])
    if mask.shape[:2] == (h, w):
        return mask
    return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


def _long_side(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return 0.0
    return float(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))


def smooth_fill(out: np.ndarray, where: np.ndarray, cell: float) -> None:
    """In place: the pixels under where replaced by a smooth blur of their neighbourhood that keeps about one sample
    per cell px (area means on a cell grid, spread back with linear interpolation and a Gaussian of cell / 2), the same
    information a pixelation of that block keeps, without the blocks."""
    ys, xs = np.nonzero(where)
    if ys.size == 0:
        return
    h_img, w_img = where.shape[:2]
    pad = int(math.ceil(cell))
    y1, y2 = max(0, int(ys.min()) - pad), min(h_img, int(ys.max()) + 1 + pad)
    x1, x2 = max(0, int(xs.min()) - pad), min(w_img, int(xs.max()) + 1 + pad)
    sub = out[y1:y2, x1:x2]
    h, w = sub.shape[:2]
    sw, sh = max(1, int(round(w / cell))), max(1, int(round(h / cell)))
    small = cv2.resize(sub, (sw, sh), interpolation=cv2.INTER_AREA)
    big = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    big = cv2.GaussianBlur(big, (0, 0), sigmaX=max(0.5, cell / 2.0))
    m = where[y1:y2, x1:x2]
    sub[m] = big[m]


Region = tuple[np.ndarray, float, bool]  # (mask on the frame, long side of the object in frame px, dilate it)


def blur_regions(rgb: np.ndarray, regions: list[Region], *, th: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """(blurred copy, the mask of every pixel blurred): each region dilated by mask_blur's dilate_frac of its long side
    plus dilate_px_at_4k (scaled to the frame) unless it is a reflection, then smooth_fill with a cell of the object's
    long side over samples_across, at least pixel_block_px_at_4k scaled to the frame. Largest objects first, so a
    plate inside a blurred windshield keeps its own, finer, cell on top."""
    mb = mask_blur_settings(th) or MASK_BLUR_DEFAULTS
    out = np.array(rgb, copy=True)
    where_all = np.zeros(out.shape[:2], dtype=bool)
    block = _block_for(th, out.shape)
    dpx = float(mb["dilate_px_at_4k"]) * _scale_for(out.shape)
    for mask, long_side, dilate in sorted(regions, key=lambda r: -r[1]):
        if not mask.any():
            continue
        grown = _dilate_mask(mask, dilate_frac=float(mb["dilate_frac"]), dilate_px=dpx) if dilate else mask
        cell = max(float(block), long_side / float(mb["samples_across"]))
        smooth_fill(out, grown, cell)
        where_all |= grown
    return out, where_all


def _box_region(box: Iterable[float], shape: tuple[int, ...]) -> Region | None:
    h, w = shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    ix1, iy1, ix2, iy2 = max(0, math.floor(x1)), max(0, math.floor(y1)), min(w, math.ceil(x2)), min(h, math.ceil(y2))
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    m = np.zeros((h, w), dtype=bool)
    m[iy1:iy2, ix1:ix2] = True
    return m, float(max(x2 - x1, y2 - y1)), True


def track_mask_regions(
    f_proxy: int, tracks: list[PanelTrack], masks: TrackMasks, *, fps: float, shape: tuple[int, ...], th: dict[str, Any], nouns: tuple[Noun, ...] = PRIVACY_NOUNS
) -> list[Region]:
    """The plate and face track masks at a proxy frame, on a frame of shape: the masks of every sighting within
    track_box_union_s, else the last sighting's mask held for track_gap_bridge_s (the box rule's bridge). A face on the
    car's own glass or paint is not dilated. Carried plate boxes come from track_boxes_at(sightings=False)."""
    p = privacy_params(th)
    mb = mask_blur_settings(th) or MASK_BLUR_DEFAULTS
    union = _frames(p["track_box_union_s"], fps)
    bridge = int(math.ceil(p["track_gap_bridge_s"] * fps))
    sx = shape[1] / float(masks.media_size[0])
    out: list[Region] = []
    f = int(f_proxy)
    for track in tracks:
        if track.noun not in nouns or not track.per_frame:
            continue
        frames = sorted({r.frame for r in track.per_frame})
        near = frames[bisect.bisect_left(frames, f - union) : bisect.bisect_right(frames, f + union)]
        if not near:
            k = bisect.bisect_left(frames, f) - 1
            near = [frames[k]] if k >= 0 and f - frames[k] <= bridge else []
        if not near:
            continue
        got = [m for m in (masks.track_mask(track, g) for g in near) if m is not None and m.any()]
        if not got:
            continue
        union_mask = np.logical_or.reduce(got)
        reflection = track.noun == Noun.FACE and masks.on_car(track, mb)
        out.append((_fit_bool(union_mask, shape), _long_side(union_mask) * sx, not reflection))
    return out


def detection_regions(dets: list[sam_client.Detection], shape: tuple[int, ...], *, car_mask: np.ndarray | None, th: dict[str, Any], faces: bool) -> list[Region]:
    """Image-check detections as regions; a face whose mask lies on the car (car_mask, any size) is not dilated."""
    mb = mask_blur_settings(th) or MASK_BLUR_DEFAULTS
    size = (int(shape[1]), int(shape[0]))
    car = _fit_bool(car_mask, shape) if car_mask is not None else None
    out: list[Region] = []
    for det in dets:
        m = detection_mask(det, size)
        if not m.any():
            continue
        reflection = faces and car is not None and float((m & car).sum()) / float(m.sum()) >= float(mb["on_car_min"])
        out.append((m, _long_side(m), not reflection))
    return out


def blur_keyframe_masks(
    frame_rgb: np.ndarray, *, f_proxy: int, tracks: list[PanelTrack], fps: float, proxy_size: tuple[int, int], masks: TrackMasks | None,
    plates: list[sam_client.Detection], faces: list[sam_client.Detection], th: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """A keyframe under mask_blur: track masks (when the walk's lanes are on disk), carried plate boxes and the image
    checks' masks, through blur_regions. Returns (blurred, blurred-pixel mask)."""
    shape = frame_rgb.shape
    src_size = (int(shape[1]), int(shape[0]))
    regions: list[Region] = []
    car = None
    if masks is not None:
        regions += track_mask_regions(f_proxy, tracks, masks, fps=fps, shape=shape, th=th)
        car = masks.car_mask(f_proxy)
        carried = track_boxes_at(f_proxy, tracks, fps=fps, media_size_proxy=proxy_size, media_size_src=src_size, th=th, nouns=(Noun.LICENSE_PLATE,), sightings=False)
    else:  # no lanes on disk (a stand-in run): the track boxes, as the box rule gives them, minus the face carry
        carried = track_boxes_at(f_proxy, tracks, fps=fps, media_size_proxy=proxy_size, media_size_src=src_size, th=th)
    regions += [r for r in (_box_region(b, shape) for b in carried) if r is not None]
    regions += detection_regions(plates, shape, car_mask=None, th=th, faces=False)
    regions += detection_regions(faces, shape, car_mask=car, th=th, faces=True)
    return blur_regions(frame_rgb, regions, th=th)


def _walk_masks(run_id: str, role: str, tracks: list[PanelTrack], proxy_size: tuple[int, int]) -> TrackMasks | None:
    """The walk's TrackMasks when every privacy noun it has a track for has its lane index on disk; else None, and the
    caller falls back to boxes (never to no blur)."""
    nouns = {t.noun for t in tracks if t.noun in PRIVACY_NOUNS and t.per_frame}
    if any(not (paths.sam_dir(run_id, role) / f"{paths.lane_id(n)}.index.json").is_file() for n in nouns):
        return None
    return track_masks_for(run_id, role, proxy_size)


def display_frame_fn(tracks: list[PanelTrack], masks: TrackMasks, carried: Carried, *, fps: float, size: tuple[int, int], th: dict[str, Any]) -> Callable[[int, np.ndarray], np.ndarray]:
    """The display video's per-frame blur under mask_blur: track masks plus the flow-carried plate boxes."""

    def fn(index: int, rgb: np.ndarray) -> np.ndarray:
        regions = track_mask_regions(index, tracks, masks, fps=fps, shape=rgb.shape, th=th)
        boxes = track_boxes_at(index, tracks, fps=fps, media_size_proxy=size, media_size_src=size, th=th, nouns=(Noun.LICENSE_PLATE,), carried=carried, sightings=False)
        regions += [r for r in (_box_region(b, rgb.shape) for b in boxes) if r is not None]
        return blur_regions(rgb, regions, th=th)[0] if regions else rgb

    return fn


# Proxy blur, quarantine, deletions --------------------------------------------------------------------------------


def _delete_file(run_id: str, path: Path, *, kind: DeletionKind, reason: str, now: str) -> DeletionRecord:
    """Hash and size a file, remove it, and return its row."""
    sha, size = _sha256_file(path), path.stat().st_size
    path.unlink()
    return DeletionRecord(target=paths.rel(run_id, path), kind=kind, sha256=sha, bytes=size, deleted_at=now, reason=reason)


def display_boxes(proxy: Path, tracks: list[PanelTrack], *, fps: float, frames: int, th: dict[str, Any]) -> dict[int, list[Box4]]:
    """Every proxy frame's plate and face boxes for the display video, in proxy pixels: track_boxes_at with its carried
    boxes followed by optical flow on this unblurred proxy (flow_carry; only the frames it needs are kept, in gray at
    FLOW_MAX_EDGE). A function of the proxy's pixels, the tracks and th only."""
    from . import video  # the media lane

    info = video.probe(proxy)
    size = (int(info.width), int(info.height))
    carried = display_carried(proxy, tracks, fps=fps, frames=frames, th=th)
    out: dict[int, list[Box4]] = {}
    for f in range(frames):
        boxes = track_boxes_at(f, tracks, fps=fps, media_size_proxy=size, media_size_src=size, th=th, carried=carried)
        if boxes:
            out[f] = boxes
    return out


def display_carried(proxy: Path, tracks: list[PanelTrack], *, fps: float, frames: int, th: dict[str, Any]) -> Carried:
    """flow_carry on the unblurred proxy's gray frames, only the frames it needs."""
    from . import video  # the media lane

    info = video.probe(proxy)
    size = (int(info.width), int(info.height))
    needed = carry_frames_needed(tracks, fps=fps, n_frames=frames, th=th)
    scale = min(1.0, FLOW_MAX_EDGE / float(max(size)))
    gray_size = (max(1, int(round(size[0] * scale))), max(1, int(round(size[1] * scale))))
    grays: dict[int, np.ndarray] = {}
    if needed:
        for index, _t, rgb in video.iter_frames(proxy):
            if index in needed:
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                grays[index] = cv2.resize(gray, gray_size, interpolation=cv2.INTER_AREA) if scale < 1.0 else gray
    return flow_carry(grays, tracks, fps=fps, media_size_proxy=size, n_frames=frames, th=th)


def render_display(proxy: Path, dst: Path, tracks: list[PanelTrack], *, fps: float, frames: int, th: dict[str, Any], masks: TrackMasks | None = None) -> Any:
    """The blurred display video from an unblurred proxy: display_boxes pixelated through video.render_blurred_proxy
    with the privacy dilation and block scaled to the proxy, each box's block grown by box_blocks_across. Writes dst
    only; the blur stage and rederive_display both call it, so the same proxy, tracks and th give the same bytes."""
    from . import video  # the media lane

    p = privacy_params(th)
    info = video.probe(proxy)
    scale = max(int(info.width), int(info.height)) / SRC_REFERENCE_EDGE
    if mask_blur_settings(th) is not None and masks is not None:
        size = (int(info.width), int(info.height))
        carried = display_carried(proxy, tracks, fps=fps, frames=frames, th=th)
        fn = display_frame_fn(tracks, masks, carried, fps=fps, size=size, th=th)
        return video.render_blurred_proxy(proxy, dst, {}, block_px=1, dilate_frac=0.0, dilate_px=0, frame_fn=fn)
    boxes = display_boxes(proxy, tracks, fps=fps, frames=frames, th=th)
    return video.render_blurred_proxy(
        proxy,
        dst,
        boxes,
        block_px=max(2, int(round(p["pixel_block_px_at_4k"] * scale))),
        dilate_frac=p["track_box_dilation_frac"],
        dilate_px=int(round(p["track_box_dilation_px"] * scale)),
        blocks_across=p["box_blocks_across"],
    )


async def render_blurred_proxy(
    run_id: str, role: str, tracks: list[PanelTrack], *, fps: float, th: dict[str, Any]
) -> tuple[BlurredProxyRecord, DeletionRecord]:
    """render_display from the plate and face tracks, then the unblurred proxy is deleted (blur.done)."""
    record = read_json(paths.video_json(run_id, role), VideoRecord)
    if record.proxy is None:
        raise FileNotFoundError(f"videos/{role}.json has no proxy record")
    proxy = paths.run_dir(run_id) / record.proxy.path
    dst = paths.proxy_blurred_path(run_id, role)
    masks = _walk_masks(run_id, role, tracks, (record.proxy.width, record.proxy.height)) if mask_blur_settings(th) is not None else None
    await asyncio.to_thread(render_display, proxy, dst, tracks, fps=fps, frames=record.proxy.frames, th=th, masks=masks)
    blurred = BlurredProxyRecord(path=paths.rel(run_id, dst), sha256=_sha256_file(dst))
    deletion = _delete_file(run_id, proxy, kind=DeletionKind.FILE, reason="unblurred proxy deleted at blur.done", now=_utc_now())
    return blurred, deletion


def walk_privacy_tracks(run_id: str, role: str, *, th: dict[str, Any]) -> list[PanelTrack]:
    """The walk's plate and face tracks rebuilt from its stored lanes by panels.privacy_tracks, as the blur stage builds
    them (orchestrator._privacy_tracks); no call."""
    from . import panels  # imported here: panels never needs this module, so no cycle at import time

    record = read_json(paths.video_json(run_id, role), VideoRecord)
    if record.proxy is None:
        raise FileNotFoundError(f"videos/{role}.json has no proxy record")
    pts = [float(v) for v in json.loads((paths.run_dir(run_id) / record.proxy.pts_map_path).read_text(encoding="utf-8"))]

    def lane(noun: Noun) -> list[IndexFrame]:
        index = paths.lane_index_path(run_id, role, noun)
        return read_json_list(index, IndexFrame) if index.is_file() else []

    car_boxes = panels.car_boxes_by_frame(lane(Noun.CAR))
    size = (record.proxy.width, record.proxy.height)
    return [t for noun in PRIVACY_NOUNS for t in panels.privacy_tracks(role, noun, lane(noun), pts, car_boxes, size, fps=record.proxy.fps, th=th)]


def rederive_display(run_id: str, role: str, original: Path, out_dir: Path, *, th: dict[str, Any] | None = None) -> dict[str, Any]:
    """Make one walk's blurred display video again, outside the run, from the renter's original: the original must
    carry the sealed sha256_server; the proxy is cut again with the run's media settings and compared with the sealed
    proxy hash; then render_display pixelates it from the walk's stored plate and face lanes with the run's thresholds
    snapshot (th replaces it; keys the snapshot lacks come from PRIVACY_DEFAULTS). The unblurred proxy lives only in
    out_dir and is deleted before this returns. Writes out_dir/<role>.proxy_blurred.mp4 and returns the hashes, with
    matches_sealed true when it is byte for byte the run's own display video (a run made with this blur code)."""
    from . import video  # the media lane

    run = read_json(paths.run_json(run_id), RunManifest)
    th = th or run.thresholds or load_thresholds()
    record = read_json(paths.video_json(run_id, role), VideoRecord)
    if record.proxy is None:
        raise FileNotFoundError(f"videos/{role}.json has no proxy record")
    out_dir = Path(out_dir).resolve()
    run_root = paths.run_dir(run_id).resolve()
    if out_dir == run_root or run_root in out_dir.parents:
        raise ValueError("rederive_display never writes inside the run; pick an out_dir outside it")
    original_sha = _sha256_file(Path(original))
    sealed = record.sha256_server or record.sha256_browser
    if original_sha != sealed:
        raise ValueError(f"{Path(original).name} is not the sealed {role} original (sha256 {original_sha[:12]}, sealed {str(sealed)[:12]})")
    out_dir.mkdir(parents=True, exist_ok=True)
    media = th["media"]
    proxy = out_dir / f".{role}.unblurred_proxy.mp4"
    dst = out_dir / f"{role}.proxy_blurred.mp4"
    try:
        video.make_proxy(Path(original), proxy, fps=float(media["proxy_fps"]), max_edge=int(media["proxy_max_edge"]), preset=media["proxy_preset"], crf=int(media["proxy_crf"]))
        proxy_sha = _sha256_file(proxy)
        tracks = walk_privacy_tracks(run_id, role, th=th)
        masks = _walk_masks(run_id, role, tracks, (record.proxy.width, record.proxy.height)) if mask_blur_settings(th) is not None else None
        render_display(proxy, dst, tracks, fps=record.proxy.fps, frames=record.proxy.frames, th=th, masks=masks)
    finally:
        proxy.unlink(missing_ok=True)
    blurred_sha = _sha256_file(dst)
    sealed_blurred = record.proxy_blurred.sha256 if record.proxy_blurred else None
    return {
        "role": role,
        "original_sha256": original_sha,
        "proxy_sha256": proxy_sha,
        "proxy_matches_sealed": proxy_sha == record.proxy.sha256,
        "blurred_path": str(dst),
        "blurred_sha256": blurred_sha,
        "sealed_blurred_sha256": sealed_blurred,
        "matches_sealed": blurred_sha == sealed_blurred,
        "privacy": {k: privacy_params(th)[k] for k in sorted(privacy_params(th)) if k not in ("image_check_nouns",)},
    }


REDERIVED_DISPLAY_NAME = "rederived_display.json"  # at the run root: never served, read by walkaround verify


def install_rederived_display(run_id: str, got: dict[str, Any], *, reason: str, now: str | None = None) -> dict[str, Any]:
    """Put a display video rederive_display made (its returned dict) in place of the run's media/<role>.proxy_blurred.mp4
    and append a row to rederived_display.json: the sealed and the new sha256, when and why. For a run whose sealed
    display video left a plate readable (pairC_core1, 5 px blocks): the sealed hashes in videos/<role>.json and
    bundle/manifest_run.json stay as sealed, and walkaround verify names this row on the media lines it reports."""
    if got.get("matches_sealed"):
        raise ValueError("the re-derived display video is the sealed one byte for byte; nothing to install")
    src = Path(got["blurred_path"])
    if _sha256_file(src) != got["blurred_sha256"]:
        raise ValueError(f"{src.name} changed since rederive_display made it")
    dst = paths.proxy_blurred_path(run_id, got["role"])
    before = _sha256_file(dst) if dst.is_file() else None
    shutil.copyfile(src, dst)
    record_path = paths.run_dir(run_id) / REDERIVED_DISPLAY_NAME
    rows = json.loads(record_path.read_text(encoding="utf-8")) if record_path.is_file() else []
    row = {
        "path": paths.rel(run_id, dst),
        "sealed_sha256": got["sealed_blurred_sha256"],
        "replaced_sha256": before,
        "sha256": got["blurred_sha256"],
        "bytes": dst.stat().st_size,
        "made_at": now or _utc_now(),
        "reason": reason,
        "privacy": got.get("privacy", {}),
    }
    rows = [r for r in rows if r.get("path") != row["path"]] + [row]
    record_path.write_text(json.dumps(rows, indent=1, sort_keys=True) + "\n", encoding="ascii")
    return row


def rederived_display_note(run_id: str, rel_path: str) -> str | None:
    """Why a display video differs from its sealed hash, when install_rederived_display put it there; else None."""
    record_path = paths.run_dir(run_id) / REDERIVED_DISPLAY_NAME
    if not record_path.is_file():
        return None
    try:
        rows = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict) and row.get("path") == rel_path and paths.run_dir(run_id).joinpath(rel_path).is_file():
            if _sha256_file(paths.run_dir(run_id) / rel_path) != row.get("sha256"):
                return None  # changed again since: no known reason
            sealed = str(row.get("sealed_sha256") or "none")[:12]
            return f"display video re-derived on {str(row.get('made_at', ''))[:10]} from the sealed original (sealed sha256 {sealed}): {row.get('reason', '')} ({REDERIVED_DISPLAY_NAME})"
    return None


def quarantine_put(run_id: str, role: str, src: Path) -> Path:
    """Move the uploaded original to quarantine/<role>.orig (never served by any route)."""
    dst = paths.quarantine_original(run_id, role)
    shutil.move(str(src), str(dst))
    return dst


def read_original_frame(run_id: str, role: str, pts_s: float) -> np.ndarray:
    """One unblurred RGB frame decoded in memory from quarantine/<role>.orig at pts_s. The caller runs
    blur_keyframe on it before any write; nothing here touches the disk."""
    from . import video  # the media lane

    original = paths.quarantine_original(run_id, role)
    if not original.is_file():
        raise FileNotFoundError(f"quarantine/{role}.orig is gone (deleted at run end?)")
    return video.read_frame_at(original, float(pts_s))


def delete_quarantine(run_id: str, *, now: str) -> list[DeletionRecord]:
    """Every file under quarantine, hashed before removal; empty subdirectories are removed too."""
    root = paths.quarantine_dir(run_id)
    rows = [
        _delete_file(run_id, p, kind=DeletionKind.QUARANTINE, reason="run end", now=now)
        for p in sorted(p for p in root.rglob("*") if p.is_file())
    ]
    for d in sorted((d for d in root.rglob("*") if d.is_dir()), reverse=True):
        d.rmdir()
    return rows


async def delete_remote(
    run_id: str, videos: list[VideoRecord], *, response_ids: list[str] | None, now: str, retries: int | None = None
) -> list[DeletionRecord]:
    """LIVE: DELETE /v1/files per upload (files_delete_retries tries; a failure is recorded with the section 14
    item 25 sentence as its reason) and DELETE /v1/responses per id only when response_ids is given (store false was
    refused). The receipt body is stored verbatim on the row (deleted true on 10/26)."""
    if retries is None:
        from .contracts import load_thresholds

        retries = int(load_thresholds()["privacy"]["files_delete_retries"])
    retries = max(1, int(retries))
    rows: list[DeletionRecord] = []
    for video in videos:
        if video.files_api is None or video.files_api.deleted_at is not None:
            continue
        sha = video.proxy.sha256 if video.proxy else None
        error = ""
        for attempt in range(1, retries + 1):
            try:
                body = await sam_client.delete_file(video.files_api.file_id)
                rows.append(DeletionRecord(target=video.files_api.file_id, kind=DeletionKind.FILES_API, sha256=sha, deleted_at=now, reason=f"run end ({video.role} proxy)", api_response=body))
                break
            except Exception as exc:  # noqa: BLE001 - the reason is recorded, never raised
                error = redact(f"{type(exc).__name__}: {exc}")[:300]
                if attempt < retries:
                    await asyncio.sleep(_RETRY_SLEEP_S * attempt)
        else:
            expires_at = video.files_api.expires_at
            expiry = f"expiry at {expires_at} applies" if expires_at else "the one-hour upload expiry applies"
            reason = f"delete request failed at {now}; {expiry} ({retries} tries, last error {error})"
            rows.append(DeletionRecord(target=video.files_api.file_id, kind=DeletionKind.FILES_API, sha256=sha, deleted_at=now, reason=reason))
    for rid in response_ids or []:
        try:
            body = await sam_client.delete_response(rid)
            rows.append(DeletionRecord(target=rid, kind=DeletionKind.RESPONSE, deleted_at=now, reason="store false refused; response deleted at run end", api_response=body))
        except Exception as exc:  # noqa: BLE001
            rows.append(DeletionRecord(target=rid, kind=DeletionKind.RESPONSE, deleted_at=now, reason=redact(f"delete request failed ({type(exc).__name__}: {exc})")[:300]))
    return rows


def close_deletions(run_id: str, rows: list[DeletionRecord], *, keep_plate: bool, chosen_by: str | None) -> Deletions:
    """Write deletions.json."""
    deletions = Deletions(run_id=run_id, rows=list(rows), keep_plate=keep_plate, keep_plate_chosen_by=chosen_by)
    write_json(paths.deletions_json(run_id), deletions)
    return deletions


def _dir_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def sweep(*, now: str, retention_hours: int, temp_minutes: int, keep_run_ids: set[str]) -> list[DeletionRecord]:
    """Delete runs past retention (except keep runs) and any quarantine temp file older than temp_minutes."""
    now_dt = _parse_iso(now)
    rows: list[DeletionRecord] = []
    for run_root in sorted(p for p in paths.runs_root().iterdir() if p.is_dir()):
        run_file = run_root / "run.json"
        if not run_file.is_file():
            continue
        try:
            run = read_json(run_file, RunManifest)
        except Exception:  # noqa: BLE001 - a half-written run.json is not this sweeper's problem
            continue
        keep = run.keep or run.run_id in keep_run_ids or run_root.name in keep_run_ids
        age_h = (now_dt - _parse_iso(run.created_at)).total_seconds() / 3600
        if not keep and age_h > retention_hours:
            size = _dir_bytes(run_root)
            shutil.rmtree(run_root)
            rows.append(DeletionRecord(target=f"{run_root.name}/", kind=DeletionKind.FILE, bytes=size, deleted_at=now, reason=f"retention {retention_hours} h"))
            continue
        tmp = run_root / "quarantine" / "tmp"
        if not tmp.is_dir():
            continue
        for p in sorted(p for p in tmp.rglob("*") if p.is_file()):
            age_min = (now_dt.timestamp() - p.stat().st_mtime) / 60
            if age_min > temp_minutes:
                sha, size = _sha256_file(p), p.stat().st_size
                p.unlink()
                rows.append(DeletionRecord(target=f"{run_root.name}/{p.relative_to(run_root).as_posix()}", kind=DeletionKind.QUARANTINE, sha256=sha, bytes=size, deleted_at=now, reason=f"unblurred temp older than {temp_minutes} min"))
    return rows


# Serving rule and the privacy line ------------------------------------------------------------------------------


# What a browser may fetch from a run dir (section 8 items 5 and 6; the app's file route): per top directory, the
# depths a file may sit at and the endings allowed there. Everything else is refused: quarantine, upload chunks, the
# unblurred proxy, the WAV, raw image-check events, spark.json, voice.json, and any temp or dot file.
_SERVED: dict[str, tuple[frozenset[int], tuple[str, ...]]] = {
    "videos": (frozenset({2}), (".json",)),
    "media": (frozenset({2}), (".proxy_blurred.mp4", ".pts.json")),
    "sam": (frozenset({3}), (".ndjson.gz", ".index.json", ".summary.json")),
    "derived": (frozenset({2}), (".json",)),
    "crops": (frozenset({3}), (".jpg", ".png")),
    "keyframes": (frozenset({3}), (".jpg",)),
    "clips": (frozenset({3}), (".mp4",)),
    "chips": (frozenset({3}), (".m4a",)),
    "claim": (frozenset({2}), (".blurred.jpg", ".json", ".open_weights.ndjson")),  # narrowed by _CLAIM_SERVED: never the upload or the answers
    "sheet": (frozenset({2, 3}), (".html", ".json", ".pdf", ".jpg", ".zip")),
    "bundle": (frozenset({2}), (".json", ".ots", ".tsq", ".tsr", ".md", ".pem", ".crt")),
}
_SERVED_RUN_FILES = ("deletions.json", "open_weights.ndjson")  # the Baseten provenance log holds no pixels or keys
# Under claim/ only the pixelated photo, the Claim itself and its Baseten provenance lines (no pixels, no keys, like the
# run's open_weights.ndjson): claim_<n>.jpg (the upload) and claim_<n>.answers.json (model answers, mask payloads) are
# never served.
_CLAIM_SERVED = re.compile(r"claim_[0-9]+\.(blurred\.jpg|json|open_weights\.ndjson)")
_NEVER_IN_NAME = (".part", ".tmp", ".orig", ".bak")
# Cut from 4K frames (or the renter's own claim photo): blurred by design, still owner-only under keep_plate, so an
# unblurred plate can never reach a browser other than the owner's whatever an upstream lane passed.
OWNER_ONLY_UNDER_KEEP_PLATE = ("sheet", "crops", "keyframes", "clips", "claim")


def served_file_allowed(run: RunManifest, rel_path: str, *, run_owner: bool) -> bool:
    """Section 8 items 5 and 6 as an allowlist. Parts are compared casefolded because APFS (the Mac that runs the
    demos) resolves QUARANTINE/ to quarantine/. With keep_plate the sheet, crops, keyframes, clips and claim photos
    are for the run owner only; an owner frame (sheet/frames/*.owner.jpg) is owner-only on any run."""
    p = PurePosixPath(rel_path)
    if not p.parts or p.is_absolute():
        return False
    parts = tuple(part.casefold() for part in p.parts)
    if any(part in ("", ".", "..") or part.startswith(".") or "\\" in part for part in parts):
        return False
    name = parts[-1]
    if any(bad in name for bad in _NEVER_IN_NAME):
        return False
    if len(parts) == 1:
        return name in _SERVED_RUN_FILES
    rule = _SERVED.get(parts[0])
    if rule is None or len(parts) not in rule[0] or not name.endswith(rule[1]):
        return False
    if parts[0] == "sheet" and len(parts) == 3 and parts[1] != "frames":
        return False
    if parts[0] == "claim" and not _CLAIM_SERVED.fullmatch(name):
        return False
    if ".owner." in name and not run_owner:
        return False
    if run.keep_plate and not run_owner and parts[0] in OWNER_ONLY_UNDER_KEEP_PLATE:
        return False
    return True


BASETEN_CLAUSE = "For the second look, two blurred crops per look were sent to Muse Spark Open on Baseten, a company separate from Meta; no video went there."
# A claim's look goes to Baseten after the run's sheet is sealed, so the claim's own paragraph says it (review 11/1).
CLAIM_BASETEN_CLAUSE = "For the second look, two blurred crops, one from the pickup walk and one from the claim photo, were sent to Muse Spark Open on Baseten, a company separate from Meta."


def privacy_line(run: RunManifest, plate_found: dict[str, bool], deletions: list[DeletionRecord] | None = None, *, baseten: bool = False) -> str:
    """The sentence of section 8 item 9 word for word, with the storage variant and the no-plate clause per walk.
    Given the deletion rows, a Files API delete that failed says so (section 14 item 25) instead of claiming one.
    baseten: the run sent second looks to Muse Spark Open on Baseten, a company that is not Meta, so it says so
    (the same clause as web/src/lib/privacy.ts)."""
    failed = [d for d in deletions or [] if d.kind == DeletionKind.FILES_API and d.api_response is None]
    if failed:
        uploads = "Uploads were set to expire after one hour; Walkaround's delete request failed, so that expiry applies. "
    else:
        uploads = "Uploads were set to expire after one hour and were deleted by Walkaround when the run ended. "
    text = (
        "Processed with Meta's Model API in Standard mode, which Meta's documentation says is not used to train Meta "
        "models. " + uploads
    )
    if run.responses_store == ResponsesStore.DELETED_PER_REQUEST:
        text += "Model requests were sent with storage off and every stored model response was deleted by Walkaround when the run ended. "
    else:
        text += "Model requests were sent with storage off. "
    text += (
        "Meta's terms allow it to retain content for abuse review and legal reasons; Walkaround does not control that. "
        "SAM 3.1 saw the unblurred proxy for the length of one segmentation pass to find the plates and faces it then blurred."
    )
    if baseten:
        text += " " + BASETEN_CLAUSE
    for role in sorted(plate_found):
        if not plate_found[role]:
            text += f" {role.replace('_', ' ').capitalize()} walk: no plate detected; nothing was blurred for that reason."
    return text
