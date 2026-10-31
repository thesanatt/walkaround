"""Blur, quarantine and deletion (docs/ARCHITECTURE.md section 8; docs/CONTRACTS.md privacy M12).

Order of operations for every unblurred frame: the SAM image checks run on the unblurred pixels FIRST and the plate
extent is recorded for the ruler; only then is the frame pixelated. Nothing unblurred is written outside
paths.quarantine_dir. Faces are always pixelated, and so is the plate in every frame this module hands back for
storage or for Spark. keep_plate only reaches the run owner's sheet frames, which the sheet writer makes with
blur_frame(keep_plate_region=True); served_file_allowed keeps those owner-only. Every SAM call here goes through
sam_client, which writes the ledger row; nothing in this module prices a call itself.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
import shutil
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
    Noun,
    PanelTrack,
    ResponsesStore,
    RunManifest,
    VideoRecord,
    read_json,
    write_json,
)

Box4 = list[float]

SRC_REFERENCE_EDGE = 3840  # the "at 4K" the privacy thresholds are written for
_RETRY_SLEEP_S = 0.5  # between Files API delete retries; tests set it to 0


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
    """Dilate a half-open box and round it outward to whole pixels inside the frame."""
    h, w = shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    if x2 <= x1 or y2 <= y1:
        return None
    px = (x2 - x1) * dilate_frac + dilate_px
    py = (y2 - y1) * dilate_frac + dilate_px
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


def _pixelate_where(out: np.ndarray, where: np.ndarray, block: int) -> None:
    """In place: block means over each connected region's bounding box, composited only where `where` is true.
    One box per region keeps a face in one corner and a plate in the other from averaging the whole 4K frame
    (about 800 MB of float64 in one go)."""
    if not where.any():
        return
    count, _labels, stats, _ = cv2.connectedComponentsWithStats(where.astype(np.uint8), connectivity=8)
    for i in range(1, count):
        x, y, w, h = (int(v) for v in stats[i, :4])
        sub = out[y : y + h, x : x + w]
        m = where[y : y + h, x : x + w]
        pix = _block_mean(sub, block)
        sub[m] = pix[m]


def pixelate(rgb: np.ndarray, regions: list[Box4], *, block_px: int, dilate_frac: float = 0.0, dilate_px: int = 0) -> np.ndarray:
    """Pure: each region dilated then replaced by block means; pixels outside every region are returned unchanged."""
    if block_px < 1:
        raise ValueError("block_px must be at least 1")
    out = np.array(rgb, copy=True)
    where = np.zeros(out.shape[:2], dtype=bool)
    for box in regions:
        region = _int_region(box, out.shape, dilate_frac=dilate_frac, dilate_px=dilate_px)
        if region is None:
            continue
        x1, y1, x2, y2 = region
        where[y1:y2, x1:x2] = True
    _pixelate_where(out, where, int(block_px))
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


def track_boxes_at(
    f_proxy: int,
    tracks: list[PanelTrack],
    *,
    fps: float,
    media_size_proxy: tuple[int, int],
    media_size_src: tuple[int, int],
    th: dict[str, Any],
    nouns: tuple[Noun, ...] = PRIVACY_NOUNS,
) -> list[Box4]:
    """Plate and face track boxes at a proxy frame, scaled to source pixels. A gap after the last sighting shorter
    than track_gap_bridge_s is bridged by carrying that last box (over-blurring is the safe side)."""
    bridge = int(math.ceil(th["privacy"]["track_gap_bridge_s"] * fps))
    sx = media_size_src[0] / media_size_proxy[0]
    sy = media_size_src[1] / media_size_proxy[1]
    out: list[Box4] = []
    for track in tracks:
        if track.noun not in nouns:
            continue
        chosen = None
        for row in sorted(track.per_frame, key=lambda r: r.frame):
            if row.frame > f_proxy:
                break
            if row.frame == f_proxy or f_proxy - row.frame <= bridge:
                chosen = row
        if chosen is not None:
            b = chosen.box
            out.append([b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy])
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
    is pixel_block_px_at_4k, both scaled by the frame's long edge over 3840."""
    out = np.array(rgb_unblurred, copy=True)
    media_size = (int(out.shape[1]), int(out.shape[0]))
    block = _block_for(th, out.shape)
    dfrac = th["privacy"]["track_box_dilation_frac"]
    dpx = _dilate_px_for(th, out.shape)
    boxes = list(track_boxes_src) + ([] if keep_plate_region else list(plate_track_boxes_src))
    dets = list(detections) + ([] if keep_plate_region else list(plate_detections))
    where = np.zeros(out.shape[:2], dtype=bool)
    for box in boxes:
        region = _int_region(box, out.shape, dilate_frac=dfrac, dilate_px=dpx)
        if region is not None:
            x1, y1, x2, y2 = region
            where[y1:y2, x1:x2] = True
    for det in dets:
        where |= _dilate_mask(detection_mask(det, media_size), dilate_frac=dfrac, dilate_px=dpx)
    _pixelate_where(out, where, block)
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


# Proxy blur, quarantine, deletions --------------------------------------------------------------------------------


def _delete_file(run_id: str, path: Path, *, kind: DeletionKind, reason: str, now: str) -> DeletionRecord:
    """Hash and size a file, remove it, and return its row."""
    sha, size = _sha256_file(path), path.stat().st_size
    path.unlink()
    return DeletionRecord(target=paths.rel(run_id, path), kind=kind, sha256=sha, bytes=size, deleted_at=now, reason=reason)


async def render_blurred_proxy(
    run_id: str, role: str, tracks: list[PanelTrack], *, fps: float, th: dict[str, Any]
) -> tuple[BlurredProxyRecord, DeletionRecord]:
    """video.render_blurred_proxy from the plate and face tracks, then the unblurred proxy is deleted (blur.done)."""
    from . import video  # the media lane; imported here so a stub never breaks this module's import

    record = read_json(paths.video_json(run_id, role), VideoRecord)
    if record.proxy is None:
        raise FileNotFoundError(f"videos/{role}.json has no proxy record")
    proxy = paths.run_dir(run_id) / record.proxy.path
    dst = paths.proxy_blurred_path(run_id, role)
    size = (record.proxy.width, record.proxy.height)
    frames = record.proxy.frames
    scale = max(size) / SRC_REFERENCE_EDGE

    def render() -> None:
        boxes_by_frame: dict[int, list[Box4]] = {}
        for f in range(frames):
            boxes = track_boxes_at(f, tracks, fps=fps, media_size_proxy=size, media_size_src=size, th=th)
            if boxes:
                boxes_by_frame[f] = boxes
        video.render_blurred_proxy(
            proxy,
            dst,
            boxes_by_frame,
            block_px=max(2, int(round(th["privacy"]["pixel_block_px_at_4k"] * scale))),
            dilate_frac=th["privacy"]["track_box_dilation_frac"],
            dilate_px=int(round(th["privacy"]["track_box_dilation_px"] * scale)),
        )

    await asyncio.to_thread(render)
    blurred = BlurredProxyRecord(path=paths.rel(run_id, dst), sha256=_sha256_file(dst))
    deletion = _delete_file(run_id, proxy, kind=DeletionKind.FILE, reason="unblurred proxy deleted at blur.done", now=_utc_now())
    return blurred, deletion


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
    "claim": (frozenset({2}), (".blurred.jpg", ".json")),  # narrowed by _CLAIM_SERVED: never the upload or the answers
    "sheet": (frozenset({2, 3}), (".html", ".json", ".pdf", ".jpg", ".zip")),
    "bundle": (frozenset({2}), (".json", ".ots", ".tsq", ".tsr", ".md", ".pem", ".crt")),
}
_SERVED_RUN_FILES = ("deletions.json", "open_weights.ndjson")  # the Baseten provenance log holds no pixels or keys
# Under claim/ only the pixelated photo and the Claim itself: claim_<n>.jpg (the upload) and claim_<n>.answers.json
# (model answers, mask payloads) are never served.
_CLAIM_SERVED = re.compile(r"claim_[0-9]+\.(blurred\.jpg|json)")
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
