"""D1 gate rows on SAM 3.1 video lanes: E2 (panel sweep), E3 (c semantics), E4 (anchor bridge), E8 (sigma_frame,
the door keyframe image requests, the cross-take and cross-walk residuals), E9 (blur lanes), E12 (store echo, the
Files API expires_at, DELETE bodies), E15 (lane sizes) and the E17 timeline probe on the blurred proxy.

    LIVE=1 python scripts/gate_sam.py --pickup P.mp4 [--return R.mp4] [--take2 T.mp4] [--fps 15 10] [--fps-fallback 30] \
        [--handle] [--bridge] [--plate-checks] [--timeline] [--delete-response-probe] [--budget 25] --out DIR
    python scripts/gate_sam.py --fixture tests/fixtures/sam/pingpong_text.events.ndjson \
        [--bridge-fixture tests/fixtures/sam/pingpong_box.events.ndjson] [--clip data/test/pingpong.mp4] --out DIR

Without LIVE=1 and without --fixture the script prints what it would send and exits 2. Every live call goes through
walkaround.sam_client (and spark_client for the timeline probe), which writes the ledger rows; the lane names carry
"D1 E<n> <what>" and the run ids the D1 prefix, so the rows read as gate rows. A sweep that would pass --budget is
skipped before it starts. Every number printed also lands in <out>/gate_sam.json, with per-noun keyframe
suggestions run_d1 cuts its crops from; the live path writes it with partial true after the lanes and after every
stage (a stage that raises records its error and the others still run), and once more at the end. Plate and face lane boxes are pixelated before the overlay MP4 is written
and before the timeline proxy is uploaded. The lane modules other lanes are writing (sam_wire, panels, geometry,
privacy) are used when built; small local fallbacks cover the gaps, and the JSON says which path ran.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import importlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
for entry in (str(REPO), str(SCRIPTS)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from meta_sam_parser import (  # noqa: E402
    FramePrompt,
    PromptBox,
    PromptObject,
    PromptPoint,
    SegmentationMask,
    build_object_prompt,
    decode_mask_to_raster,
    parse_responses_stream,
    video_segmentation_format,
)

from walkaround import contracts, ledger, paths  # noqa: E402
from walkaround.config import SAM_MODEL  # noqa: E402
from walkaround.contracts import IndexFrame, IndexObject, LaneRecord, LaneStatus, Noun  # noqa: E402

Box4 = list[float]
FIXTURE_TEXT = REPO / "tests" / "fixtures" / "sam" / "pingpong_text.events.ndjson"
FIXTURE_BOX = REPO / "tests" / "fixtures" / "sam" / "pingpong_box.events.ndjson"
TEST_CLIP = REPO / "data" / "test" / "pingpong.mp4"
DISPLAY_NOUNS = tuple(n for n in contracts.LANE_NOUNS if n not in contracts.PRIVACY_NOUNS)
DOOR_AND_BUMPER = (Noun.CAR_DOOR, Noun.BUMPER)
E3_AGREE_MIN = 11 / 12  # section 11 E3: states agree on >= 11 of 12 panels (estimate)
SUGGEST_NOUNS = (Noun.CAR_DOOR, Noun.BUMPER, Noun.HOOD, Noun.LICENSE_PLATE, Noun.WHEEL)  # frames run_d1 cuts crops from
PROP_BOTH_LANE = "D1 E2 car door propboth"  # the deliberate probe; a 400 on it is one of the answers E2 wants
# E9: a gap this long between plate detections reads as the plate out of view (walking along a side), so coverage is
# measured inside the windows it splits. An estimate. TODO(live D1): check every split gap on the overlay; a long miss
# with the plate in view would be excluded here.
PLATE_WINDOW_GAP_S = 2.0
# Tests set this to let a faked client through without LIVE=1.
FORCE_LIVE = False


# ------------------------------------------------------------------------------------------------------------------
# Shared helpers (gate_stays_put.py and run_d1.py import these)
# ------------------------------------------------------------------------------------------------------------------


def optional_module(name: str) -> Any:
    """walkaround.<name> when it imports, else None (other lanes are still writing these)."""
    try:
        return importlib.import_module(f"walkaround.{name}")
    except Exception:  # noqa: BLE001 - a half-written module must not stop the gate
        return None


def implemented(module: Any, name: str) -> bool:
    """True when module.name exists and is not the stub that raises NotImplementedError on every call."""
    fn = getattr(module, name, None)
    if fn is None:
        return False
    code = getattr(fn, "__code__", None)
    if code is None:
        return True
    return "NotImplementedError" not in (code.co_names or ())


def live_allowed() -> bool:
    return FORCE_LIVE or os.environ.get("LIVE") == "1"


def sam_request_error() -> type[Exception]:
    """sam_client.SamRequestError (an image check that did not complete); RuntimeError until it exists."""
    return getattr(optional_module("sam_client"), "SamRequestError", RuntimeError)


def error_text(exc: BaseException, limit: int = 300) -> str:
    return f"{type(exc).__name__}: {str(exc)[:limit]}"


def billed_on_error(exc: BaseException) -> float:
    """Dollars of an image check that raised after the API answered (its ledger row is already written)."""
    run = getattr(exc, "run", None)
    return float(getattr(run, "usd", 0.0) or 0.0)


def utc_ts() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    with Path(path).open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def read_events(path: Path) -> list[dict[str, Any]]:
    """One JSON event per line from .ndjson or .ndjson.gz."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:  # type: ignore[operator]
        return [json.loads(line) for line in fh if line.strip()]


def write_json(path: Path, value: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


def lane_text(events: Iterable[dict[str, Any]]) -> str:
    """The deltas concatenated; the finalizer text when there were no deltas; '' for a zero-match lane."""
    deltas: list[str] = []
    final = ""
    for event in events:
        kind = event.get("type")
        if kind == "response.output_text.delta":
            deltas.append(event.get("delta") or "")
        elif kind == "response.output_text.done" and not final:
            final = event.get("text") or ""
        elif kind == "response.content_part.done" and not final:
            part = event.get("part") or {}
            final = part.get("text") or "" if isinstance(part, dict) else ""
    return "".join(deltas) if deltas else final


def response_fields(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Response id, store echo, usage and timing read from response.created and response.completed by dict access."""
    out: dict[str, Any] = {
        "response_id": None,
        "store_echo": None,
        "usage": None,
        "video_frames_processed": None,
        "created_at": None,
        "completed_at": None,
        "status": None,
        "metadata_echo": None,
    }
    for event in events:
        kind = event.get("type")
        response = event.get("response") if isinstance(event.get("response"), dict) else None
        if kind == "response.created" and response:
            out["response_id"] = response.get("id")
            out["store_echo"] = response.get("store")
            out["created_at"] = response.get("created_at")
            out["metadata_echo"] = response.get("metadata")
        if kind in ("response.completed", "response.incomplete", "response.failed") and response:
            out["status"] = response.get("status") or kind.split(".")[-1]
            out["completed_at"] = response.get("completed_at")
            usage = response.get("usage")
            if isinstance(usage, dict):
                out["usage"] = usage
                out["video_frames_processed"] = usage.get("video_frames_processed")
            if out["store_echo"] is None:
                out["store_echo"] = response.get("store")
    return out


@dataclass
class Lane:
    """One SAM request: the contract record, the thin index and what a decode needs."""

    record: LaneRecord
    frames: list[IndexFrame]
    text: str = ""
    masks: dict[tuple[int, str], SegmentationMask] = field(default_factory=dict)  # oracle path only
    reader: str = "sam_wire"  # or "meta_sam_parser"
    events_path: Path | None = None
    first_delta_wall_s: float | None = None
    name: str | None = None  # the gate's lane name ("D1 E2 car door"); sam_client stores a slug as record.lane_id

    @property
    def lane_id(self) -> str:
        return self.name or self.record.lane_id

    @property
    def noun(self) -> Noun | None:
        return self.record.noun

    def objects_at(self, frame: int) -> list[IndexObject]:
        for fr in self.frames:
            if fr.frame == frame:
                return fr.objects
        return []


async def _oracle_result(events: list[dict[str, Any]]) -> Any:
    async def gen():
        for event in events:
            yield event

    parsed = parse_responses_stream(gen(), video_segmentation_format())
    async with parsed:
        async for _ in parsed:
            pass
        return await parsed.final_result()


def run_sync(coro: Any) -> Any:
    """asyncio.run, or the same on a worker thread when called inside a running loop (the live sweep analyses its
    lanes inside asyncio.run)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def oracle_index(events: list[dict[str, Any]]) -> tuple[list[IndexFrame], dict[tuple[int, str], SegmentationMask], Any]:
    """The index through meta_sam_parser (tests' oracle; the fallback reader until sam_wire lands)."""
    result = run_sync(_oracle_result(events))
    boxes: dict[tuple[int, str], Any] = {}
    masks: dict[tuple[int, str], SegmentationMask] = {}
    for rec in result.records:
        frame = rec.frame.frame_index if rec.frame is not None else 0
        if rec.kind == "box":
            boxes[(frame, rec.object_id)] = rec
        elif rec.kind == "mask":
            masks[(frame, rec.object_id)] = rec.mask
    frames: dict[int, list[IndexObject]] = {}
    for (frame, obj_id), rec in sorted(boxes.items(), key=lambda kv: (kv[0][0], (len(kv[0][1]), kv[0][1]))):
        mask = masks.get((frame, obj_id))
        ref = None
        if mask is not None:
            ref = contracts.MaskRef(
                x=0,
                y=0,
                h=mask.height,
                w=mask.width,
                encoding=contracts.MaskEncoding(mask.encoding),
                payload_offset=0,
                payload_len=0,
            )
        frames.setdefault(frame, []).append(
            IndexObject(id=obj_id, box=[int(rec.left), int(rec.top), int(rec.right), int(rec.bottom)], c=rec.confidence, mask=ref)
        )
    index = [IndexFrame(frame=f, objects=objs) for f, objs in sorted(frames.items())]
    return index, masks, result


def index_from_events(events: list[dict[str, Any]]) -> tuple[list[IndexFrame], dict[str, Any], dict[tuple[int, str], SegmentationMask], str]:
    """(frames, outcome, side masks, reader) through sam_wire when present, else the parser oracle."""
    sam_wire = optional_module("sam_wire")
    fields = response_fields(events)
    if sam_wire is not None and hasattr(sam_wire, "index_from_events"):
        t0 = time.process_time()
        frames, outcome = sam_wire.index_from_events(events, media="video")
        cpu = time.process_time() - t0
        info = {
            "status": getattr(outcome, "status", LaneStatus.INCOMPLETE),
            "response_id": getattr(outcome, "response_id", None) or fields["response_id"],
            "store_echo": getattr(outcome, "store_echo", None) if getattr(outcome, "store_echo", None) is not None else fields["store_echo"],
            "video_frames_processed": getattr(outcome, "video_frames_processed", None) or fields["video_frames_processed"],
            "diagnostics": [d.model_dump() if hasattr(d, "model_dump") else str(d) for d in (getattr(outcome, "diagnostics", None) or [])],
            "parse_cpu_s": cpu,
            **{k: fields[k] for k in ("created_at", "completed_at", "usage", "metadata_echo")},
        }
        return frames, info, {}, "sam_wire"
    t0 = time.process_time()
    frames, masks, result = oracle_index(events)
    cpu = time.process_time() - t0
    completed = getattr(result.outcome, "status", "") == "completed"
    if completed and frames:
        status = LaneStatus.COMPLETED
    elif completed:
        status = LaneStatus.ZERO_MATCHES
    else:
        status = LaneStatus.INCOMPLETE
    errors = [d for d in result.diagnostics if getattr(d, "severity", "") == "error"]
    if errors:
        status = LaneStatus.INCOMPLETE
    info = {
        "status": status,
        "response_id": fields["response_id"],
        "store_echo": fields["store_echo"],
        "video_frames_processed": fields["video_frames_processed"],
        "diagnostics": [{"severity": d.severity, "code": d.code, "message": d.message, "line": d.line} for d in result.diagnostics],
        "parse_cpu_s": cpu,
        **{k: fields[k] for k in ("created_at", "completed_at", "usage", "metadata_echo")},
    }
    return frames, info, masks, "meta_sam_parser"


def lane_from_events(
    events: list[dict[str, Any]],
    *,
    lane_id: str,
    role: str,
    noun: Noun | None,
    prompt_text: str,
    kind: str = "text",
    events_path: Path | None = None,
    frames_sent: int | None = None,
) -> Lane:
    """A Lane from saved events (the fixture path and the replay of a live lane's raw file)."""
    frames, info, masks, reader = index_from_events(events)
    media_size = media_size_from_frames(frames) or [0, 0]
    vfp = info.get("video_frames_processed")
    created, completed = info.get("created_at"), info.get("completed_at")
    completed_s = float(completed) - float(created) if created and completed else None
    metadata = info.get("metadata_echo") if isinstance(info.get("metadata_echo"), dict) else {}
    record = LaneRecord(
        lane_id=lane_id,
        role=role,
        kind=kind,  # type: ignore[arg-type]
        media="video",
        noun=noun,
        prompt_text=prompt_text,
        media_size=media_size,
        params={k: str(v) for k, v in metadata.items()},
        store=False,
        store_echo=info.get("store_echo"),
        request_id=info.get("response_id"),
        status=info["status"],
        lane_path=str(events_path) if events_path else None,
        completed_s=completed_s,
        frames_sent=int(frames_sent or vfp or 0),
        frames_returned=len([f for f in frames if f.objects]),
        video_frames_processed=int(vfp) if vfp is not None else None,
        ids=sorted({o.id for f in frames for o in f.objects}, key=lambda s: (len(s), s)),
        zero_matches=info["status"] == LaneStatus.ZERO_MATCHES,
        parse_cpu_s=round(info["parse_cpu_s"], 6),
        dollars=0.0,
        diagnostics=[contracts.Diagnostic(**d) for d in info["diagnostics"] if isinstance(d, dict) and "severity" in d],
    )
    return Lane(record=record, frames=frames, text=lane_text(events), masks=masks, reader=reader, events_path=events_path, name=lane_id)


def lane_file(run_id: str, record: LaneRecord) -> Path | None:
    """The lane's events on disk; sam_client stores lane_path relative to the run directory."""
    if not record.lane_path:
        return None
    path = Path(record.lane_path)
    return path if path.is_absolute() else paths.run_dir(run_id) / path


def live_lane(run_id: str, name: str, record: LaneRecord, frames: list[IndexFrame], first_delta_wall_s: float | None = None) -> Lane:
    """A Lane from what sam_client returned, with the lane text read back from the gz file it wrote."""
    events_path = lane_file(run_id, record)
    text = ""
    if events_path is not None and events_path.exists():
        text = lane_text(read_events(events_path))
    return Lane(record=record, frames=frames, text=text, reader="sam_wire", events_path=events_path, first_delta_wall_s=first_delta_wall_s, name=name)


def media_size_from_frames(frames: list[IndexFrame]) -> list[int] | None:
    """Not on the wire of the index; read from the first raw record when the lane text is available."""
    return None


def media_size_from_text(text: str) -> list[int] | None:
    pos = text.find(";w=")
    if pos < 0:
        return None
    try:
        tail = text[pos + 3 : pos + 40]
        w = int(tail.split(";")[0])
        h = int(tail.split(";h=")[1].split(";")[0].split("|")[0])
        return [w, h]
    except (IndexError, ValueError):
        return None


def decode_mask(lane: Lane, frame: int, obj_id: str) -> np.ndarray | None:
    """Box-local bool raster for one object in one frame, through sam_wire when the lane was read by it."""
    obj = next((o for o in lane.objects_at(frame) if o.id == obj_id), None)
    if obj is None or obj.mask is None:
        return None
    if lane.reader == "sam_wire":
        sam_wire = optional_module("sam_wire")
        payload = sam_wire.mask_payload(lane.text, obj.mask)
        return np.asarray(sam_wire.decode_mask(obj.mask, payload), dtype=bool)
    mask = lane.masks.get((frame, obj_id))
    if mask is None:
        return None
    raw = decode_mask_to_raster(mask)
    return np.frombuffer(bytes(raw), dtype=np.uint8).reshape(mask.height, mask.width).astype(bool)


def paste_mask(raster: np.ndarray, box: list[int], ref: contracts.MaskRef | None, media_size: tuple[int, int]) -> np.ndarray:
    """The frame-sized bool mask with the box-local raster at (x1 + ref.x, y1 + ref.y), clipped to the frame."""
    w, h = media_size
    full = np.zeros((h, w), dtype=bool)
    x0 = int(box[0]) + (ref.x if ref else 0)
    y0 = int(box[1]) + (ref.y if ref else 0)
    rh, rw = raster.shape
    xs, ys = max(0, x0), max(0, y0)
    xe, ye = min(w, x0 + rw), min(h, y0 + rh)
    if xe > xs and ye > ys:
        full[ys:ye, xs:xe] = raster[ys - y0 : ye - y0, xs - x0 : xe - x0]
    return full


def full_mask(lane: Lane, frame: int, obj_id: str, media_size: tuple[int, int]) -> np.ndarray | None:
    raster = decode_mask(lane, frame, obj_id)
    if raster is None:
        return None
    obj = next(o for o in lane.objects_at(frame) if o.id == obj_id)
    return paste_mask(raster, obj.box, obj.mask, media_size)


# Geometry fallbacks (walkaround.geometry when it exists) -------------------------------------------------------


def box_iou(a: Box4, b: Box4) -> float:
    geometry = optional_module("geometry")
    if geometry is not None and implemented(geometry, "box_iou"):
        return float(geometry.box_iou(list(a), list(b)))
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def containment(inner: Box4, outer: Box4) -> float:
    geometry = optional_module("geometry")
    if geometry is not None and implemented(geometry, "containment"):
        return float(geometry.containment(list(inner), list(outer)))
    ix = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return (ix * iy) / area if area > 0 else 0.0


def moments_from_mask(mask: np.ndarray) -> contracts.Moments:
    geometry = optional_module("geometry")
    if geometry is not None and implemented(geometry, "moments_from_mask"):
        return geometry.moments_from_mask(mask)
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("empty mask")
    cx, cy = float(xs.mean()), float(ys.mean())
    dx, dy = xs - cx, ys - cy
    mu20, mu02, mu11 = float((dx * dx).mean()), float((dy * dy).mean()), float((dx * dy).mean())
    return _moments(cx, cy, mu20, mu02, mu11)


def _moments(cx: float, cy: float, mu20: float, mu02: float, mu11: float) -> contracts.Moments:
    sigma = np.array([[mu20, mu11], [mu11, mu02]], dtype=float)
    vals, vecs = np.linalg.eigh(sigma)
    vals = np.clip(vals, 1e-12, None)
    inv_half = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
    theta = 0.5 * math.atan2(2.0 * mu11, mu20 - mu02)
    return contracts.Moments(cx=cx, cy=cy, mu20=mu20, mu02=mu02, mu11=mu11, theta=theta, sigma_inv_half=inv_half.tolist())


def rect_moments(box: Box4) -> contracts.Moments:
    """Moments of a filled rectangle (a panel stand-in when no mask exists): variance w^2 / 12."""
    w, h = box[2] - box[0], box[3] - box[1]
    return _moments((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0, w * w / 12.0, h * h / 12.0, 0.0)


def carry_moments(moments: contracts.Moments, box_from: Box4, box_to: Box4) -> contracts.Moments:
    geometry = optional_module("geometry")
    if geometry is not None and implemented(geometry, "carry_moments"):
        return geometry.carry_moments(moments, list(box_from), list(box_to))
    sx = (box_to[2] - box_to[0]) / max(box_from[2] - box_from[0], 1e-9)
    sy = (box_to[3] - box_to[1]) / max(box_from[3] - box_from[1], 1e-9)
    cx = box_to[0] + (moments.cx - box_from[0]) * sx
    cy = box_to[1] + (moments.cy - box_from[1]) * sy
    return _moments(cx, cy, moments.mu20 * sx * sx, moments.mu02 * sy * sy, moments.mu11 * sx * sy)


def whiten(point: tuple[float, float], moments: contracts.Moments, front_sign: int = 1) -> tuple[float, float]:
    geometry = optional_module("geometry")
    if geometry is not None and implemented(geometry, "whiten"):
        return tuple(geometry.whiten((float(point[0]), float(point[1])), moments, front_sign))  # type: ignore[return-value]
    if front_sign not in (1, -1):
        raise ValueError("front_sign must be 1 or -1")
    p = np.array([point[0] - moments.cx, point[1] - moments.cy], dtype=float)
    w = np.asarray(moments.sigma_inv_half, dtype=float) @ p
    c, s = math.cos(moments.theta), math.sin(moments.theta)
    rot = np.array([[c, s], [-s, c]])  # axis 1 along the principal axis
    w = rot @ w
    return float(front_sign * w[0]), float(w[1])


# Panel-rule fallbacks (walkaround.panels when it exists) -------------------------------------------------------


def largest_car_boxes(car_lane: Lane | None) -> dict[int, list[int]]:
    if car_lane is None:
        return {}
    panels = optional_module("panels")
    if panels is not None and implemented(panels, "car_boxes_by_frame"):
        return panels.car_boxes_by_frame(car_lane.frames)
    out: dict[int, list[int]] = {}
    for fr in car_lane.frames:
        if fr.objects:
            best = max(fr.objects, key=lambda o: (o.box[2] - o.box[0]) * (o.box[3] - o.box[1]))
            out[fr.frame] = list(best.box)
    return out


def filter_objects(frames: list[IndexFrame], noun: Noun, car_boxes: dict[int, list[int]], media_size: tuple[int, int], th: dict) -> list[IndexFrame]:
    panels = optional_module("panels")
    if panels is not None and implemented(panels, "filter_objects"):
        return panels.filter_objects(frames, noun, car_boxes, media_size, th)
    p = th["panels"]
    floor = p["area_floor_small"] if noun.value in p["small_nouns"] else p["area_floor_default"]
    frame_area = float(media_size[0] * media_size[1]) or 1.0
    out: list[IndexFrame] = []
    for fr in frames:
        kept = []
        car = car_boxes.get(fr.frame)
        for obj in fr.objects:
            area = (obj.box[2] - obj.box[0]) * (obj.box[3] - obj.box[1]) / frame_area
            if area < floor:
                continue
            if car is not None and containment(obj.box, car) < p["car_box_containment_min"]:
                continue
            kept.append(obj)
        if kept:
            out.append(IndexFrame(frame=fr.frame, objects=kept))
    return out


def build_fragments(frames: list[IndexFrame], *, fps: float, th: dict) -> list[contracts.TrackFragment]:
    panels = optional_module("panels")
    if panels is not None and implemented(panels, "build_fragments"):
        return panels.build_fragments(frames, fps=fps, th=th)
    p = th["panels"]
    gap_max = max(p["fragment_gap_split_s"] * fps, p["fragment_gap_split_frames"])
    min_frames = max(p["fragment_min_frames"], round(fps * p["fragment_min_s"]))
    by_id: dict[str, list[int]] = {}
    for fr in frames:
        for obj in fr.objects:
            by_id.setdefault(obj.id, []).append(fr.frame)
    out: list[contracts.TrackFragment] = []
    for obj_id, fr_list in by_id.items():
        fr_list = sorted(set(fr_list))
        start = prev = fr_list[0]
        count = 1
        for f in fr_list[1:]:
            if f - prev > gap_max:
                if count >= min_frames:
                    out.append(contracts.TrackFragment(object_id=obj_id, first_frame=start, last_frame=prev))
                start, count = f, 0
            prev = f
            count += 1
        if count >= min_frames:
            out.append(contracts.TrackFragment(object_id=obj_id, first_frame=start, last_frame=prev))
    out.sort(key=lambda t: (t.first_frame, t.last_frame, t.object_id))
    return out


def _box_of(frames: list[IndexFrame], frame: int, obj_id: str) -> list[int] | None:
    for fr in frames:
        if fr.frame == frame:
            for obj in fr.objects:
                if obj.id == obj_id:
                    return list(obj.box)
    return None


def _centroid(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def merge_fragments(
    fragments: list[contracts.TrackFragment], frames: list[IndexFrame], *, fps: float, th: dict
) -> list[tuple[list[contracts.TrackFragment], list[contracts.MergeRecord]]]:
    """Greedy merge in time order per 5.1 item 4; one group per merged track."""
    panels = optional_module("panels")
    if panels is not None and implemented(panels, "merge_fragments"):
        return panels.merge_fragments(fragments, frames, fps=fps, th=th)
    p = th["panels"]
    groups: list[tuple[list[contracts.TrackFragment], list[contracts.MergeRecord]]] = []
    for frag in sorted(fragments, key=lambda t: (t.first_frame, t.last_frame)):
        merged = False
        for frags, records in groups:
            last = frags[-1]
            gap_frames = frag.first_frame - last.last_frame
            if gap_frames <= 0 or gap_frames / fps > p["merge_gap_max_s"]:
                continue
            box_a = _box_of(frames, last.last_frame, last.object_id)
            box_b = _box_of(frames, frag.first_frame, frag.object_id)
            if box_a is None or box_b is None:
                continue
            velocity = _median_velocity(frames, last)
            ca, cb = _centroid(box_a), _centroid(box_b)
            pred = (ca[0] + velocity[0] * gap_frames, ca[1] + velocity[1] * gap_frames)
            width_a = max(box_a[2] - box_a[0], 1.0)
            pred_dist = math.dist(pred, cb) / width_a
            area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
            area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
            area_ratio = area_b / area_a if area_a > 0 else 0.0
            aspect_a = (box_a[2] - box_a[0]) / max(box_a[3] - box_a[1], 1.0)
            aspect_b = (box_b[2] - box_b[0]) / max(box_b[3] - box_b[1], 1.0)
            aspect_ok = abs(aspect_b - aspect_a) <= p["merge_aspect_tolerance"] * aspect_a
            lo, hi = p["merge_area_ratio"]
            if pred_dist <= p["merge_pred_dist_box_widths"] and lo <= area_ratio <= hi and aspect_ok:
                frags.append(frag)
                records.append(
                    contracts.MergeRecord(rule=contracts.MergeRule.VELOCITY_GAP, gap_s=gap_frames / fps, pred_dist=pred_dist, area_ratio=area_ratio)
                )
                merged = True
                break
        if not merged:
            groups.append(([frag], []))
    return groups


def _median_velocity(frames: list[IndexFrame], frag: contracts.TrackFragment) -> tuple[float, float]:
    centers = []
    for fr in frames:
        if frag.first_frame <= fr.frame <= frag.last_frame:
            for obj in fr.objects:
                if obj.id == frag.object_id:
                    centers.append((fr.frame, _centroid(obj.box)))
    if len(centers) < 2:
        return (0.0, 0.0)
    vx = [(b[1][0] - a[1][0]) / (b[0] - a[0]) for a, b in zip(centers, centers[1:]) if b[0] > a[0]]
    vy = [(b[1][1] - a[1][1]) / (b[0] - a[0]) for a, b in zip(centers, centers[1:]) if b[0] > a[0]]
    return (statistics.median(vx), statistics.median(vy)) if vx else (0.0, 0.0)


@dataclass
class TrackLite:
    """A merged track of one noun: the fragments and the frames they cover."""

    track_id: str
    noun: Noun | None
    fragments: list[contracts.TrackFragment]
    merges: list[contracts.MergeRecord]
    frames: list[int]

    @property
    def first_frame(self) -> int:
        return min(self.frames)

    @property
    def last_frame(self) -> int:
        return max(self.frames)


def tracks_for(frames: list[IndexFrame], noun: Noun | None, *, fps: float, th: dict) -> list[TrackLite]:
    fragments = build_fragments(frames, fps=fps, th=th)
    groups = merge_fragments(fragments, frames, fps=fps, th=th)
    tracks: list[TrackLite] = []
    for n, (frags, merges) in enumerate(groups):
        covered: set[int] = set()
        for frag in frags:
            for fr in frames:
                if frag.first_frame <= fr.frame <= frag.last_frame and any(o.id == frag.object_id for o in fr.objects):
                    covered.add(fr.frame)
        tracks.append(TrackLite(track_id=f"t{n}", noun=noun, fragments=frags, merges=merges, frames=sorted(covered)))
    tracks.sort(key=lambda t: -len(t.frames))
    return tracks


def fragment_report(tracks: list[Any]) -> dict[str, Any]:
    """Tracks, fragments and the longest track (any objects with .fragments and .frames or per_frame)."""
    if not tracks:
        return {"tracks": 0, "fragments_total": 0, "longest_frames": 0, "longest_fragments": 0}

    def n_frames(t: Any) -> int:
        if hasattr(t, "frames"):
            return len(t.frames)
        return len(getattr(t, "per_frame", []) or [])

    longest = max(tracks, key=n_frames)
    return {
        "tracks": len(tracks),
        "fragments_total": sum(len(t.fragments) for t in tracks),
        "longest_frames": n_frames(longest),
        "longest_fragments": len(longest.fragments),
        "longest_span": [longest.fragments[0].first_frame, longest.fragments[-1].last_frame] if longest.fragments else None,
    }


# Prompt helpers -------------------------------------------------------------------------------------------------


def prompt_box(box: Box4, media_size: tuple[int, int], *, pad_px: int = 2) -> PromptBox:
    """Step 6's rule: integer edges, at least pad_px per side, clamped, left < right <= w, top < bottom <= h."""
    sam_client = optional_module("sam_client")
    if sam_client is not None and implemented(sam_client, "prompt_box"):
        return sam_client.prompt_box(list(box), media_size, pad_px=pad_px)
    w, h = media_size
    x1, y1 = int(math.floor(box[0])) - pad_px, int(math.floor(box[1])) - pad_px
    x2, y2 = int(math.ceil(box[2])) + pad_px, int(math.ceil(box[3])) + pad_px
    x1, y1 = max(0, min(x1, w - 2)), max(0, min(y1, h - 2))
    x2, y2 = max(x1 + 1, min(x2, w)), max(y1 + 1, min(y2, h))
    return PromptBox(left=x1, top=y1, right=x2, bottom=y2)


def object_prompt(objects: list[PromptObject], media_size: tuple[int, int]) -> str:
    sam_client = optional_module("sam_client")
    if sam_client is not None and implemented(sam_client, "object_prompt"):
        return sam_client.object_prompt(objects, media_size)
    return build_object_prompt(size=(int(media_size[0]), int(media_size[1])), objects=objects)


def sam_metadata(*, include_confidence: bool = True, mask_encoding: str | None = "one_bit", propagation_direction: str | None = None) -> dict[str, str]:
    sam_client = optional_module("sam_client")
    if sam_client is not None and implemented(sam_client, "sam_metadata"):
        return sam_client.sam_metadata(include_confidence=include_confidence, mask_encoding=mask_encoding, propagation_direction=propagation_direction)
    meta = {"include_confidence": "true" if include_confidence else "false"}
    if mask_encoding:
        meta["mask_encoding"] = mask_encoding
    if propagation_direction:
        meta["propagation_direction"] = propagation_direction
    return meta


# Media helpers (walkaround.video when implemented, else ffmpeg directly) ---------------------------------------


def _ffprobe_json(path: Path) -> dict[str, Any]:
    cmd = ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]
    return json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)


def probe_media(path: Path) -> dict[str, Any]:
    """width, height (after rotation), fps, frames, duration_s, has_audio, codec, creation_time."""
    video = optional_module("video")
    if video is not None and implemented(video, "probe"):
        info = video.probe(Path(path))
        return {
            "width": info.width,
            "height": info.height,
            "fps": info.fps,
            "frames": info.nb_frames,
            "duration_s": info.duration_s,
            "has_audio": info.has_audio,
            "codec": info.codec,
            "rotation": info.rotation,
            "creation_time": info.creation_time.isoformat() if info.creation_time else "not present",
        }
    data = _ffprobe_json(Path(path))
    streams = data.get("streams", [])
    vs = next(s for s in streams if s.get("codec_type") == "video")
    rotation = 0
    for sd in vs.get("side_data_list", []) or []:
        if "rotation" in sd:
            rotation = int(round(float(sd["rotation"]))) % 360
    tag_rot = (vs.get("tags") or {}).get("rotate")
    if tag_rot:
        rotation = int(tag_rot) % 360
    w, h = int(vs["width"]), int(vs["height"])
    if rotation in (90, 270):
        w, h = h, w
    num, den = (vs.get("avg_frame_rate") or vs.get("r_frame_rate") or "0/1").split("/")
    fps = float(num) / float(den) if float(den) else 0.0
    duration = float(data.get("format", {}).get("duration") or vs.get("duration") or 0.0)
    frames = int(vs.get("nb_frames") or 0) or int(round(duration * fps))
    tags = data.get("format", {}).get("tags", {}) or {}
    return {
        "width": w,
        "height": h,
        "fps": fps,
        "frames": frames,
        "duration_s": duration,
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
        "codec": vs.get("codec_name"),
        "rotation": rotation,
        "creation_time": tags.get("creation_time", "not present"),
    }


def make_proxy(src: Path, dst: Path, *, fps: float, max_edge: int = 960, preset: str = "medium", crf: int = 20) -> dict[str, Any]:
    """The SAM proxy (section 3 M1) and its probe; walkaround.video.make_proxy when implemented."""
    video = optional_module("video")
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    if video is not None and implemented(video, "make_proxy"):
        try:
            video.make_proxy(Path(src), dst, fps=fps, max_edge=max_edge, preset=preset, crf=crf)
        except TypeError:  # the pre-contract stub without preset and crf
            video.make_proxy(Path(src), dst, fps=fps, max_edge=max_edge)
    else:
        info = probe_media(src)
        scale = min(1.0, max_edge / max(info["width"], info["height"]))
        w = max(2, int(round(info["width"] * scale / 2)) * 2)
        h = max(2, int(round(info["height"] * scale / 2)) * 2)
        cmd = [
            "ffmpeg", "-y", "-v", "error", "-i", str(src),
            "-vf", f"scale={w}:{h}:flags=lanczos,format=yuv420p", "-r", str(fps), "-fps_mode", "cfr",
            "-an", "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-movflags", "+faststart", str(dst),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    out = probe_media(dst)
    out["path"] = str(dst)
    out["encode_s"] = round(time.monotonic() - t0, 3)
    out["preset"] = preset
    out["sha256"] = sha256_file(dst)
    return out


def frame_at(path: Path, t_seconds: float) -> np.ndarray:
    """One RGB frame of the original at t (walkaround.video.read_frame_at when implemented)."""
    video = optional_module("video")
    if video is not None and implemented(video, "read_frame_at"):
        return np.asarray(video.read_frame_at(Path(path), float(t_seconds)))
    info = probe_media(path)
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{t_seconds:.3f}", "-i", str(path), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(info["height"], info["width"], 3).copy()


def cut_subclip(proxy: Path, dst: Path, *, first_frame: int, frames: int, fps: float) -> dict[str, Any]:
    """frames proxy frames from first_frame, re-encoded so indices are exact (the anchor clip of step 6)."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-v", "error", "-ss", f"{first_frame / fps:.4f}", "-i", str(proxy), "-frames:v", str(frames),
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    out = probe_media(dst)
    out["path"] = str(dst)
    out["first_frame"] = first_frame
    return out


def read_clip_frames(path: Path, wanted: set[int] | None = None) -> dict[int, np.ndarray]:
    """BGR frames by index through cv2 (all frames, or only the wanted ones)."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames: dict[int, np.ndarray] = {}
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if wanted is None or i in wanted:
            frames[i] = frame
        i += 1
    cap.release()
    return frames


# ------------------------------------------------------------------------------------------------------------------
# Measurements
# ------------------------------------------------------------------------------------------------------------------


def c_stats(frames: list[IndexFrame]) -> dict[str, Any]:
    records = [(fr.frame, o) for fr in frames for o in fr.objects]
    present = [o.c for _, o in records if o.c is not None]
    per_id: dict[str, list[tuple[int, float]]] = {}
    for frame, obj in records:
        if obj.c is not None:
            per_id.setdefault(obj.id, []).append((frame, obj.c))
    phases: dict[str, dict[str, float | None]] = {}
    distinct: dict[str, int] = {}
    for obj_id, rows in per_id.items():
        rows.sort()
        values = [c for _, c in rows]
        distinct[obj_id] = len({round(c, 6) for c in values})
        n = len(values)
        phases[obj_id] = {
            "first": values[0],
            "mid": values[n // 2],
            "last": values[-1],
            "first_quarter_mean": float(np.mean(values[: max(1, n // 4)])),
            "rest_mean": float(np.mean(values[max(1, n // 4) :])) if n > 1 else None,
        }
    quantiles = {f"p{q}": round(float(np.percentile(present, q)), 6) for q in (10, 25, 50, 75, 90)} if present else {}
    return {
        "records": len(records),
        "c_present": len(present),
        "c_present_share": round(len(present) / len(records), 4) if records else None,
        "c_min": min(present) if present else None,
        "c_median": statistics.median(present) if present else None,
        "c_max": max(present) if present else None,
        "c_quantiles": quantiles,
        "c_distinct_values": len({round(c, 6) for c in present}),
        "distinct_per_id": distinct,
        "c_varies_per_frame": any(v > 1 for v in distinct.values()),
        "phases_per_id": phases,
    }


def lane_metrics(lane: Lane, *, fps: float, th: dict, car_boxes: dict[int, list[int]] | None = None, media_size: tuple[int, int] | None = None) -> dict[str, Any]:
    """Everything E2 and E3 read from one lane: status, timings, ids, c, fragments, the longest track share."""
    rec = lane.record
    frames = lane.frames
    visible = sorted({fr.frame for fr in frames if fr.objects})
    tracks = tracks_for(frames, rec.noun, fps=fps, th=th)
    report = fragment_report(tracks)
    out: dict[str, Any] = {
        "lane_id": lane.lane_id,
        "record_lane_id": rec.lane_id,
        "index_path": rec.index_path,  # relative to the run directory (sam_client)
        "lane_file": str(lane.events_path) if lane.events_path is not None else None,
        "media_size": list(rec.media_size),
        "noun": rec.noun.value if rec.noun else None,
        "prompt_text": rec.prompt_text,
        "kind": rec.kind,
        "reader": lane.reader,
        "status": rec.status.value,
        "zero_matches": rec.zero_matches,
        "store_echo": rec.store_echo,
        "request_id": rec.request_id,
        "params": rec.params,
        "upload_s": rec.upload_s,
        "first_delta_s": rec.first_delta_s,
        "completed_s": rec.completed_s,
        "frames_sent": rec.frames_sent,
        "frames_returned": len(visible),
        "frame_range": [visible[0], visible[-1]] if visible else None,
        "video_frames_processed": rec.video_frames_processed,
        "vfp_equals_frames_sent": (rec.video_frames_processed == rec.frames_sent) if rec.video_frames_processed is not None and rec.frames_sent else None,
        "vfp_minus_frames_sent": (rec.video_frames_processed - rec.frames_sent) if rec.video_frames_processed is not None and rec.frames_sent else None,
        "ids": rec.ids,
        "n_ids": len(rec.ids),
        "retries_429": rec.retries,  # sam_client counts 429 backoffs plus the one retry of a broken stream or 5xx
        "error": rec.error,
        "parse_cpu_s": rec.parse_cpu_s,
        "diagnostics": len(rec.diagnostics),
        "dollars": rec.dollars,
        "c": c_stats(frames),
        "fragments_raw": len(build_fragments(frames, fps=fps, th=th)),
        "merged_tracks": report["tracks"],
        "longest_track_frames": report["longest_frames"],
        "longest_track_fragments": report["longest_fragments"],
        "longest_track_share": round(report["longest_frames"] / len(visible), 4) if visible else None,
        "longest_track_span": report.get("longest_span"),
        "windows_s": [[round(t.first_frame / fps, 3), round((t.last_frame + 1) / fps, 3)] for t in tracks if t.frames],  # E10 pins
    }
    if car_boxes is not None and rec.noun is not None and media_size is not None:
        filtered = filter_objects(frames, rec.noun, car_boxes, media_size, th)
        f_visible = sorted({fr.frame for fr in filtered if fr.objects})
        f_tracks = tracks_for(filtered, rec.noun, fps=fps, th=th)
        f_report = fragment_report(f_tracks)
        out["after_car_filter"] = {
            "frames_returned": len(f_visible),
            "merged_tracks": f_report["tracks"],
            "longest_track_frames": f_report["longest_frames"],
            "longest_track_fragments": f_report["longest_fragments"],
            "longest_track_share": round(f_report["longest_frames"] / len(f_visible), 4) if f_visible else None,
        }
    return out


def c_semantics(records: list[LaneRecord], indexes: dict[str, list[IndexFrame]]) -> dict[str, Any]:
    """E3: c on first frames vs mid-track vs propagated frames per noun, c_present_share, whether c varies."""
    out: dict[str, Any] = {}
    for rec in records:
        frames = indexes.get(rec.lane_id, [])
        out[rec.lane_id] = {"noun": rec.noun.value if rec.noun else None, "kind": rec.kind, **c_stats(frames)}
    return out


def variance_report(lanes: list[Lane], *, fps: float, th: dict) -> dict[str, Any]:
    """Run-to-run variance of repeated requests: ids, frames, best-box IoU per shared frame, fragments."""
    if len(lanes) < 2:
        return {"runs": len(lanes), "note": "fewer than two runs"}
    per_run = []
    for lane in lanes:
        tracks = tracks_for(lane.frames, lane.noun, fps=fps, th=th)
        report = fragment_report(tracks)
        per_run.append(
            {
                "lane_id": lane.lane_id,
                "n_ids": len(lane.record.ids),
                "frames_returned": lane.record.frames_returned,
                "merged_tracks": report["tracks"],
                "longest_fragments": report["longest_fragments"],
                "longest_frames": report["longest_frames"],
                "completed_s": lane.record.completed_s,
            }
        )
    pairs = []
    for i in range(len(lanes)):
        for j in range(i + 1, len(lanes)):
            a, b = lanes[i], lanes[j]
            shared = sorted({fr.frame for fr in a.frames if fr.objects} & {fr.frame for fr in b.frames if fr.objects})
            ious = []
            for f in shared:
                for obj in a.objects_at(f):
                    best = max((box_iou(obj.box, o.box) for o in b.objects_at(f)), default=0.0)
                    ious.append(best)
            pairs.append(
                {
                    "runs": [a.lane_id, b.lane_id],
                    "shared_frames": len(shared),
                    "mean_best_iou": round(float(np.mean(ious)), 4) if ious else None,
                    "frames_only_in_first": len({fr.frame for fr in a.frames if fr.objects} - set(shared)),
                    "frames_only_in_second": len({fr.frame for fr in b.frames if fr.objects} - set(shared)),
                }
            )
    return {"runs": len(lanes), "per_run": per_run, "pairs": pairs}


def _track_states(lane: Lane, *, fps: float, th: dict, media_size: tuple[int, int]) -> list[dict[str, Any]]:
    cov = th["coverage"]
    frame_area = float(media_size[0] * media_size[1]) or 1.0
    all_c = [o.c for fr in lane.frames for o in fr.objects if o.c is not None]
    p25 = float(np.percentile(all_c, cov["glimpsed_c_percentile"])) if all_c else None
    rows = []
    for track in tracks_for(lane.frames, lane.noun, fps=fps, th=th):
        ids = {f.object_id for f in track.fragments}
        cs, areas, n_obs = [], [], 0
        for fr in lane.frames:
            if fr.frame in track.frames:
                for obj in fr.objects:
                    if obj.id in ids:
                        n_obs += 1
                        areas.append((obj.box[2] - obj.box[0]) * (obj.box[3] - obj.box[1]) / frame_area)
                        if obj.c is not None:
                            cs.append(obj.c)
        share = len(cs) / n_obs if n_obs else 0.0
        median_c = statistics.median(cs) if cs else None
        if share >= cov["c_present_floor"] and p25 is not None and median_c is not None:
            state_c = "glimpsed" if median_c >= p25 else "below_p25"
            basis = "c"
        else:
            state_c = "glimpsed" if any(a >= cov["area_only_box_area_frac_min"] for a in areas) else "below_area"
            basis = "area_only"
        state_area = "glimpsed" if any(a >= cov["area_only_box_area_frac_min"] for a in areas) else "below_area"
        rows.append(
            {
                "track_id": track.track_id,
                "first": track.first_frame,
                "last": track.last_frame,
                "frames": len(track.frames),
                "c_present_share": round(share, 4),
                "median_c": median_c,
                "state_c": state_c,
                "state_basis": basis,
                "state_area": state_area,
            }
        )
    return rows


def states_agreement(lanes: list[Lane], *, fps: float, th: dict, media_size: tuple[int, int]) -> dict[str, Any]:
    """E3: the three-state agreement across repeated runs, once on the c basis and once area-only."""
    if len(lanes) < 2:
        return {"runs": len(lanes), "note": "fewer than two runs"}
    per_run = [_track_states(lane, fps=fps, th=th, media_size=media_size) for lane in lanes]
    base = per_run[0]
    matched = 0
    agree_c = agree_area = 0
    details = []
    for track in base:
        states_c, states_area = [track["state_c"]], [track["state_area"]]
        complete = True
        for other in per_run[1:]:
            best, best_ov = None, 0.0
            for cand in other:
                lo, hi = max(track["first"], cand["first"]), min(track["last"], cand["last"])
                span = max(track["last"], cand["last"]) - min(track["first"], cand["first"]) + 1
                ov = (hi - lo + 1) / span if hi >= lo and span > 0 else 0.0
                if ov > best_ov:
                    best, best_ov = cand, ov
            if best is None or best_ov < 0.3:
                complete = False
                break
            states_c.append(best["state_c"])
            states_area.append(best["state_area"])
        if not complete:
            continue
        matched += 1
        agree_c += len(set(states_c)) == 1
        agree_area += len(set(states_area)) == 1
        details.append({"base_track": track["track_id"], "states_c": states_c, "states_area": states_area})
    share_c = agree_c / matched if matched else None
    return {
        "runs": len(lanes),
        "tracks_matched_across_runs": matched,
        "agree_c_basis": agree_c,
        "agree_area_basis": agree_area,
        "agree_c_share": round(share_c, 4) if share_c is not None else None,
        "agree_area_share": round(agree_area / matched, 4) if matched else None,
        "pass_share_min": round(E3_AGREE_MIN, 4),
        "pass": (share_c >= E3_AGREE_MIN) if share_c is not None else None,
        "per_run_tracks": per_run,
        "details": details,
    }


def decode_budget(lanes: list[Lane], *, media_size: tuple[int, int], k: int = 3, car_lane: Lane | None = None) -> dict[str, Any]:
    """Decode the keyframe set (top k frames by box area per display noun plus the car companion); CPU seconds."""
    t_cpu, t_wall = time.process_time(), time.monotonic()
    decoded = 0
    sizes = []
    for lane in lanes:
        if lane.noun is None or lane.noun in contracts.PRIVACY_NOUNS or lane.noun == Noun.CAR:
            continue
        scored = []
        for fr in lane.frames:
            for obj in fr.objects:
                if obj.mask is not None:
                    scored.append(((obj.box[2] - obj.box[0]) * (obj.box[3] - obj.box[1]), fr.frame, obj.id))
        scored.sort(reverse=True)
        for _, frame, obj_id in scored[:k]:
            raster = decode_mask(lane, frame, obj_id)
            if raster is not None:
                decoded += 1
                sizes.append(list(raster.shape))
            if car_lane is not None:
                car = next((o for o in car_lane.objects_at(frame) if o.mask is not None), None)
                if car is not None and decode_mask(car_lane, frame, car.id) is not None:
                    decoded += 1
    return {
        "masks_decoded": decoded,
        "decode_cpu_s": round(time.process_time() - t_cpu, 4),
        "decode_wall_s": round(time.monotonic() - t_wall, 4),
        "per_mask_ms": round(1000 * (time.process_time() - t_cpu) / decoded, 2) if decoded else None,
        "largest_raster": max(sizes, key=lambda s: s[0] * s[1]) if sizes else None,
        "k_per_noun": k,
    }


def oracle_check(lane: Lane, events: list[dict[str, Any]]) -> dict[str, Any]:
    """Once per sweep: the same lane through meta_sam_parser; boxes, ids, c and frames must agree; CPU compared."""
    t0 = time.process_time()
    frames, _, _ = oracle_index(events)
    cpu = time.process_time() - t0
    mine = {(fr.frame, o.id): (tuple(o.box), None if o.c is None else round(o.c, 6)) for fr in lane.frames for o in fr.objects}
    theirs = {(fr.frame, o.id): (tuple(o.box), None if o.c is None else round(o.c, 6)) for fr in frames for o in fr.objects}
    return {
        "lane_id": lane.lane_id,
        "reader": lane.reader,
        "reader_cpu_s": lane.record.parse_cpu_s,
        "parser_cpu_s": round(cpu, 4),
        "records_reader": len(mine),
        "records_parser": len(theirs),
        "agree": mine == theirs,
        "mismatches": sorted(k for k in set(mine) ^ set(theirs) | {k for k in set(mine) & set(theirs) if mine[k] != theirs[k]})[:10],
    }


def privacy_boxes_by_frame(lanes: list[Lane], *, fps: float, th: dict, total: int | None = None) -> dict[int, list[Box4]]:
    """Plate and face boxes per proxy frame from the plain privacy lanes; a gap under track_gap_bridge_s carries the
    object's last box (over-blurring is the safe side, as in privacy.track_boxes_at). total, the proxy's frame count,
    stops the carry at the last frame (video.render_blurred_proxy refuses boxes past it)."""
    bridge = int(math.ceil(th["privacy"]["track_gap_bridge_s"] * fps))
    out: dict[int, list[Box4]] = {}
    for lane in lanes:
        if lane.noun not in contracts.PRIVACY_NOUNS or lane.lane_id != f"D1 E2 {lane.noun.value}":
            continue
        last: dict[str, tuple[int, list[int]]] = {}
        frames = sorted(lane.frames, key=lambda fr: fr.frame)
        end = frames[-1].frame + bridge + 1 if frames else 0
        if total is not None:
            end = min(end, int(total))
        by_frame = {fr.frame: fr for fr in frames}
        for f in range(frames[0].frame if frames else 0, end):
            for obj in by_frame[f].objects if f in by_frame else []:
                last[obj.id] = (f, list(obj.box))
            for seen, box in last.values():
                if f - seen <= bridge:
                    out.setdefault(f, []).append([float(v) for v in box])
    return out


def pixelate_boxes(rgb: np.ndarray, boxes: list[Box4], th: dict) -> np.ndarray:
    """privacy.pixelate with the privacy dilation and block scaled to the frame's long edge."""
    if not boxes:
        return rgb
    scale = max(rgb.shape[:2]) / 3840.0
    block = max(2, int(round(th["privacy"]["pixel_block_px_at_4k"] * scale)))
    dilate_px = int(round(th["privacy"]["track_box_dilation_px"] * scale))
    privacy = optional_module("privacy")
    if privacy is not None and implemented(privacy, "pixelate"):
        return privacy.pixelate(rgb, boxes, block_px=block, dilate_frac=th["privacy"]["track_box_dilation_frac"], dilate_px=dilate_px)
    import cv2

    out = rgb.copy()
    h, w = out.shape[:2]
    for b in boxes:
        grow_x = th["privacy"]["track_box_dilation_frac"] * (b[2] - b[0]) / 2 + dilate_px
        grow_y = th["privacy"]["track_box_dilation_frac"] * (b[3] - b[1]) / 2 + dilate_px
        x1, y1 = max(0, int(b[0] - grow_x)), max(0, int(b[1] - grow_y))
        x2, y2 = min(w, int(math.ceil(b[2] + grow_x))), min(h, int(math.ceil(b[3] + grow_y)))
        if x2 > x1 and y2 > y1:
            small = cv2.resize(out[y1:y2, x1:x2], (max(1, (x2 - x1) // block), max(1, (y2 - y1) // block)), interpolation=cv2.INTER_AREA)
            out[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    return out


def render_overlay(lane: Lane, clip: Path, out_path: Path, *, media_size: tuple[int, int], check_frames: list[int], max_mask_frames: int = 30, blur_boxes: dict[int, list[Box4]] | None = None, th: dict | None = None) -> dict[str, Any]:
    """Boxes on every frame and pasted masks on the check frames (and up to max_mask_frames more), as an MP4. With
    blur_boxes (proxy pixels per frame) the plate and face regions are pixelated before anything is drawn or written."""
    import cv2

    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        return {"error": f"cannot open {clip}"}
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    frames_with_masks = [fr.frame for fr in lane.frames if any(o.mask is not None for o in fr.objects)]
    step = max(1, len(frames_with_masks) // max_mask_frames) if frames_with_masks else 1
    mask_frames = set(check_frames) | set(frames_with_masks[::step])
    sx, sy = w / media_size[0], h / media_size[1]
    i = written = masks_drawn = ok = mismatch = 0
    colors = [(60, 220, 60), (255, 160, 0), (0, 160, 255), (220, 60, 220)]
    blurred_frames = 0
    while True:
        got, frame = cap.read()
        if not got:
            break
        if blur_boxes and th is not None and blur_boxes.get(i):
            scaled = [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in blur_boxes[i]]
            frame = np.ascontiguousarray(pixelate_boxes(frame, scaled, th))
            blurred_frames += 1
        for n, obj in enumerate(lane.objects_at(i)):
            color = colors[n % len(colors)]
            x1, y1, x2, y2 = obj.box
            if i in mask_frames and obj.mask is not None:
                raster = decode_mask(lane, i, obj.id)
                if raster is not None:
                    masks_drawn += 1
                    if raster.shape == (y2 - y1, x2 - x1):
                        ok += 1
                    else:
                        mismatch += 1
                    full = paste_mask(raster, obj.box, obj.mask, media_size)
                    if (sx, sy) != (1.0, 1.0):
                        full = cv2.resize(full.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                    tint = frame.copy()
                    tint[full] = (0.45 * tint[full] + 0.55 * np.array(color)).astype(np.uint8)
                    frame = tint
            cv2.rectangle(frame, (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy)), color, 2)
            label = f"{obj.id}" + (f" c={obj.c:.2f}" if obj.c is not None else "")
            cv2.putText(frame, label, (int(x1 * sx), max(12, int(y1 * sy) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{lane.record.prompt_text}  f{i}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
        written += 1
        i += 1
    cap.release()
    writer.release()
    return {
        "path": str(out_path),
        "frames_written": written,
        "masks_drawn": masks_drawn,
        "placement_ok": ok,
        "placement_mismatch": mismatch,
        "check_frames": check_frames,
        "frames_pixelated": blurred_frames,
        "bytes": out_path.stat().st_size if out_path.exists() else 0,
    }


def plate_windows(frames: list[int], split_frames: int) -> list[tuple[int, int]]:
    """Runs of plate detections; a gap over split_frames starts a new window (the plate out of view in between)."""
    if not frames:
        return []
    windows = []
    start = prev = frames[0]
    for f in frames[1:]:
        if f - prev > split_frames:
            windows.append((start, prev))
            start = f
        prev = f
    windows.append((start, prev))
    return windows


def blur_coverage(plate_lane: Lane | None, *, fps: float, th: dict) -> dict[str, Any]:
    """E9: plate-track coverage over the frames where the plate is visible, raw and with the 0.5 s carry-over. The
    visible frames are the windows between the first and last detection of each run, split at gaps over
    PLATE_WINDOW_GAP_S, so a plate seen at both ends of the loop (or a front and a rear plate) is not charged for the
    sides of the car in between. The old first-to-last span is kept beside it."""
    if plate_lane is None:
        return {"note": "no license plate lane"}
    frames = sorted({fr.frame for fr in plate_lane.frames if fr.objects})
    if not frames:
        return {"status": plate_lane.record.status.value, "plate_frames": 0, "note": "no plate detected in this walk"}
    bridge = th["privacy"]["track_gap_bridge_s"] * fps
    split = max(int(math.ceil(bridge)) + 1, int(round(PLATE_WINDOW_GAP_S * fps)))
    windows = plate_windows(frames, split)
    visible = sum(b - a + 1 for a, b in windows)
    covered = set(frames)
    for a, b in zip(frames, frames[1:]):
        if 1 < b - a <= bridge:
            covered.update(range(a + 1, b))
    first_to_last = frames[-1] - frames[0] + 1
    return {
        "status": plate_lane.record.status.value,
        "plate_frames": len(frames),
        "windows": [[a, b] for a, b in windows],
        "windows_s": [[round(a / fps, 3), round((b + 1) / fps, 3)] for a, b in windows],
        "window_split_gap_s": PLATE_WINDOW_GAP_S,
        "gaps_split": len(windows) - 1,
        "span_frames": visible,
        "coverage_raw": round(len(frames) / visible, 4),
        "coverage_bridged": round(len(covered) / visible, 4),
        "bridge_frames": bridge,
        "gaps_over_bridge": sum(1 for a, b in zip(frames, frames[1:]) if bridge < b - a <= split),
        "first_to_last_span_frames": first_to_last,
        "coverage_raw_first_to_last": round(len(frames) / first_to_last, 4),
        "note": f"windows split at gaps over {PLATE_WINDOW_GAP_S:g} s (an estimate of the plate leaving view); check each split gap on the overlay",
    }


def lane_sizes(out: Path, lanes: list[Lane] | None = None) -> dict[str, Any]:
    """E15: gzipped bytes per lane (the gz files sam_client wrote, or the fixture gzipped into out). The display
    total counts the 14 display nouns once each (the plain 'D1 E2 <noun>' lanes, no repeats)."""
    sizes: dict[str, int] = {}
    display: dict[str, int] = {}
    if lanes:
        for lane in lanes:
            src = Path(lane.events_path) if lane.events_path is not None else None
            if src is None or not src.exists():
                continue
            if src.suffix == ".gz":
                gz = src
            else:
                gz = Path(out) / "lane_gz" / (contracts.noun_slug(lane.lane_id) + ".ndjson.gz")
                gz.parent.mkdir(parents=True, exist_ok=True)
                with src.open("rb") as fh, gz.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as dst:
                    shutil.copyfileobj(fh, dst)
            sizes[lane.lane_id] = gz.stat().st_size
            if lane.noun in DISPLAY_NOUNS and lane.lane_id == f"D1 E2 {lane.noun.value}":
                display[lane.noun.value] = sizes[lane.lane_id]
    else:
        for p in Path(out).rglob("*.ndjson.gz"):
            sizes[p.stem.replace(".ndjson", "")] = p.stat().st_size
    return {
        "per_lane_bytes": sizes,
        "total_bytes": sum(sizes.values()),
        "display_nouns_bytes": sum(display.values()),
        "display_nouns_counted": sorted(display),
        "display_nouns_pair_estimate_mb": round(2 * sum(display.values()) / 1e6, 3) if display else None,
    }


def first_delta_trend(lanes: list[Lane]) -> dict[str, Any]:
    """Whether first-delta time grows with request index (least-squares slope in s per index)."""
    pts = [(lane.record.request_index, lane.record.first_delta_s) for lane in lanes if lane.record.first_delta_s is not None]
    if len(pts) < 3:
        return {"n": len(pts), "slope_s_per_index": None}
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)
    slope = float(np.polyfit(xs, ys, 1)[0]) if np.ptp(xs) > 0 else 0.0
    return {"n": len(pts), "slope_s_per_index": round(slope, 4), "first_delta_min_s": float(ys.min()), "first_delta_max_s": float(ys.max()), "grows": slope > 0.5}


def absorb_report(pass1: Lane, bridge: Lane, *, bridge_object: str, frame_offset: int, th: dict, fps: float, fragments_to_join: int | None = None) -> dict[str, Any]:
    """Fragments after absorb: pass-1 fragments the bridged object overlaps with box IoU >= the absorb floor. joined
    is None, not False, when there was nothing to join: fewer than 2 fragments on the bridged track (fragments_to_join,
    from the caller), or in the bridge span when the caller does not say."""
    iou_min = th["panels"]["anchor_absorb_iou_min"]
    bridge_frames = {fr.frame + frame_offset: next((o for o in fr.objects if o.id == bridge_object), None) for fr in bridge.frames}
    bridge_frames = {f: o for f, o in bridge_frames.items() if o is not None}
    fragments = build_fragments(pass1.frames, fps=fps, th=th)
    if fragments_to_join is None:
        lo, hi = (min(bridge_frames), max(bridge_frames)) if bridge_frames else (0, -1)
        fragments_to_join = sum(1 for frag in fragments if frag.first_frame <= hi and frag.last_frame >= lo)
    absorbed, ious = [], []
    for frag in fragments:
        shared = [f for f in range(frag.first_frame, frag.last_frame + 1) if f in bridge_frames]
        if not shared:
            continue
        vals = []
        for f in shared:
            box = _box_of(pass1.frames, f, frag.object_id)
            if box is not None:
                vals.append(box_iou(box, bridge_frames[f].box))
        if vals and float(np.mean(vals)) >= iou_min:
            absorbed.append({"object_id": frag.object_id, "first": frag.first_frame, "last": frag.last_frame, "mean_iou": round(float(np.mean(vals)), 4)})
            ious.extend(vals)
    span = [min(bridge_frames), max(bridge_frames)] if bridge_frames else None
    out = {
        "bridge_object": bridge_object,
        "bridge_frames": len(bridge_frames),
        "bridge_span": span,
        "pass1_fragments": len(fragments),
        "fragments_to_join": fragments_to_join,
        "absorbed": absorbed,
        "fragments_after_absorb": len(fragments) - len(absorbed) + (1 if bridge_frames else 0),
        "mean_iou_absorbed": round(float(np.mean(ious)), 4) if ious else None,
        "joined": (len(absorbed) >= 2) if fragments_to_join >= 2 else None,
    }
    if fragments_to_join < 2:
        out["note"] = f"nothing to join: {fragments_to_join} fragment(s) on the bridged track"
    return out


def glass_overlap(bridge: Lane, window_lane: Lane | None, *, bridge_object: str, frame_offset: int, n_frames: int = 3) -> dict[str, Any]:
    """Box IoU of the bridged object with the car window boxes (mask IoU on n decoded frames when both exist)."""
    if window_lane is None:
        return {"note": "no car window lane"}
    box_vals, mask_vals = [], []
    frames = [fr for fr in bridge.frames if any(o.id == bridge_object for o in fr.objects)]
    step = max(1, len(frames) // n_frames) if frames else 1
    for fr in frames:
        obj = next(o for o in fr.objects if o.id == bridge_object)
        src_frame = fr.frame + frame_offset
        windows = window_lane.objects_at(src_frame)
        if not windows:
            continue
        box_vals.append(max(box_iou(obj.box, w.box) for w in windows))
    for fr in frames[::step][:n_frames]:
        obj = next(o for o in fr.objects if o.id == bridge_object)
        src_frame = fr.frame + frame_offset
        windows = [w for w in window_lane.objects_at(src_frame) if w.mask is not None]
        if not windows or obj.mask is None:
            continue
        size = (bridge.record.media_size[0], bridge.record.media_size[1])
        a = full_mask(bridge, fr.frame, obj.id, size)
        if a is None:
            continue
        for w in windows:
            b = full_mask(window_lane, src_frame, w.id, (window_lane.record.media_size[0], window_lane.record.media_size[1]))
            if b is None or b.shape != a.shape:
                continue
            inter = np.logical_and(a, b).sum()
            mask_vals.append(float(inter / a.sum()) if a.sum() else 0.0)
    return {
        "frames_compared": len(box_vals),
        "mean_box_iou_with_window": round(float(np.mean(box_vals)), 4) if box_vals else None,
        "mean_mask_share_in_window": round(float(np.mean(mask_vals)), 4) if mask_vals else None,
    }


def handle_sigma(door_lane: Lane, handle_lane: Lane, *, media_size: tuple[int, int], th: dict, n: int = 10) -> dict[str, Any]:
    """E8: the handle centroid whitened in the door moments over n consecutive frames; sigma_frame = RMS spread."""
    pairs = []
    for fr in handle_lane.frames:
        doors = door_lane.objects_at(fr.frame)
        for handle in fr.objects:
            door = next((d for d in doors if d.mask is not None and containment(handle.box, d.box) >= th["panels"]["car_box_containment_min"]), None)
            if door is not None:
                pairs.append((fr.frame, handle, door))
                break
    run: list[tuple[int, IndexObject, IndexObject]] = []
    best: list[tuple[int, IndexObject, IndexObject]] = []
    for row in pairs:
        if run and (row[0] != run[-1][0] + 1 or row[2].id != run[-1][2].id):
            run = []
        run.append(row)
        if len(run) > len(best):
            best = list(run)
        if len(best) >= n:
            break
    best = best[:n]
    if len(best) < 3:
        return {"frames_used": len(best), "sigma_frame": None, "note": "fewer than 3 consecutive handle-in-door frames"}
    ws = []
    for frame, handle, door in best:
        mask = full_mask(door_lane, frame, door.id, media_size)
        if mask is None or not mask.any():
            continue
        moments = moments_from_mask(mask)
        ws.append(whiten(_centroid(handle.box), moments, 1))
    arr = np.array(ws)
    mean = arr.mean(axis=0)
    sigma = float(np.sqrt(((arr - mean) ** 2).sum(axis=1).mean())) if len(arr) else None
    return {
        "frames_used": len(ws),
        "frame_range": [best[0][0], best[-1][0]],
        "door_object": best[0][2].id,
        "sigma_frame": round(sigma, 5) if sigma is not None else None,
        "whitened_handle": [[round(float(a), 5), round(float(b), 5)] for a, b in ws],
    }


def handle_position(door_lane: Lane, handle_lane: Lane, *, media_size: tuple[int, int], th: dict) -> dict[str, Any] | None:
    """The whitened handle centroid on the frame with the largest door that contains a handle."""
    best = None
    for fr in handle_lane.frames:
        for handle in fr.objects:
            for door in door_lane.objects_at(fr.frame):
                if door.mask is None or containment(handle.box, door.box) < th["panels"]["car_box_containment_min"]:
                    continue
                area = (door.box[2] - door.box[0]) * (door.box[3] - door.box[1])
                if best is None or area > best[0]:
                    best = (area, fr.frame, handle, door)
    if best is None:
        return None
    _, frame, handle, door = best
    mask = full_mask(door_lane, frame, door.id, media_size)
    if mask is None or not mask.any():
        return None
    w = whiten(_centroid(handle.box), moments_from_mask(mask), 1)
    return {"frame": frame, "door_object": door.id, "w": [round(w[0], 5), round(w[1], 5)]}


def border_frac(box: Box4, media_size: tuple[int, int]) -> float:
    """Share of the box perimeter on the frame edge (walkaround.panels.border_frac when implemented)."""
    panels = optional_module("panels")
    if panels is not None and implemented(panels, "border_frac"):
        return float(panels.border_frac([float(v) for v in box], media_size))
    w, h = media_size
    bw, bh = max(box[2] - box[0], 0.0), max(box[3] - box[1], 0.0)
    perim = 2.0 * (bw + bh)
    if perim <= 0:
        return 0.0
    edge = (bh if box[0] <= 0 else 0.0) + (bw if box[1] <= 0 else 0.0) + (bh if box[2] >= w else 0.0) + (bw if box[3] >= h else 0.0)
    return edge / perim


def top_frames(frames: list[IndexFrame], *, n: int, min_gap_frames: int, media_size: tuple[int, int], th: dict) -> list[tuple[int, IndexObject]]:
    """The n frames with the largest box off the frame edge (border share under the keyframe limit), min_gap_frames apart."""
    limit = th["panels"]["border_frac_keyframe_max"]
    scored = []
    for fr in frames:
        for obj in fr.objects:
            if border_frac(obj.box, media_size) <= limit:
                scored.append(((obj.box[2] - obj.box[0]) * (obj.box[3] - obj.box[1]), fr.frame, obj))
    scored.sort(key=lambda s: (-s[0], s[1]))
    picks: list[tuple[int, IndexObject]] = []
    for _, frame, obj in scored:
        if all(abs(frame - f) >= min_gap_frames for f, _ in picks):
            picks.append((frame, obj))
        if len(picks) >= n:
            break
    return picks


def suggestions(lanes: list[Lane], *, fps: float, media_size: tuple[int, int], src_size: tuple[int, int] | None, car_boxes: dict[int, list[int]] | None, th: dict, n: int = 3) -> dict[str, list[dict[str, Any]]]:
    """Per noun, the n best frames after the car filter: proxy frame, seconds, box in proxy and source pixels.
    run_d1 cuts the E6 crops, the E5 windows and the E11 ruler frame from these when its plan names none."""
    gap = max(1, int(round(th["panels"]["keyframe_min_gap_s"] * fps)))
    sx, sy = (src_size[0] / media_size[0], src_size[1] / media_size[1]) if src_size else (1.0, 1.0)
    out: dict[str, list[dict[str, Any]]] = {}
    for noun in SUGGEST_NOUNS:
        lane = next((l for l in lanes if l.noun == noun and l.lane_id == f"D1 E2 {noun.value}"), None)
        if lane is None:
            continue
        frames = filter_objects(lane.frames, noun, car_boxes, media_size, th) if car_boxes else lane.frames
        rows = []
        for frame, obj in top_frames(frames, n=n, min_gap_frames=gap, media_size=media_size, th=th):
            b = obj.box
            rows.append({"frame": frame, "t": round(frame / fps, 4), "object_id": obj.id, "c": obj.c, "box_proxy": list(b), "box_src": [round(b[0] * sx, 1), round(b[1] * sy, 1), round(b[2] * sx, 1), round(b[3] * sy, 1)]})
        out[noun.value] = rows
    return out


def _det_box(det: Any) -> Box4:
    return [float(det.box.left), float(det.box.top), float(det.box.right), float(det.box.bottom)]


def _dets(run: Any) -> list[Any]:
    return [d for dets in run.frames.values() for d in dets]


def side_agnostic_residual(a: list[float], b: list[float]) -> dict[str, Any]:
    """Distance between two whitened points, also with b mirrored on axis 1 (a door seen from the other side flips
    the front direction when front_sign is not known); the smaller one is the residual."""
    direct = math.dist(a, b)
    mirrored = math.dist(a, [-b[0], b[1]])
    return {"residual": round(min(direct, mirrored), 5), "direct": round(direct, 5), "mirrored": round(mirrored, 5), "used": "direct" if direct <= mirrored else "mirrored"}


async def handle_keyframes(src: Path, door_lane: Lane, *, fps: float, media_size: tuple[int, int], run_id: str, role: str, th: dict, car_boxes: dict[int, list[int]] | None = None, n: int = 3) -> dict[str, Any]:
    """E8: 'car door', 'door handle' and 'side mirror' image requests on the n largest door frames of the 4K source;
    the handle centre and the mirror base whitened in the door mask's moments (front_sign 1)."""
    sam_client = optional_module("sam_client")
    if sam_client is None or not implemented(sam_client, "segment_image"):
        return {"note": "sam_client.segment_image not implemented"}
    info = probe_media(src)
    sx, sy = info["width"] / media_size[0], info["height"] / media_size[1]
    gap = max(1, int(round(th["panels"]["keyframe_min_gap_s"] * fps)))
    frames = filter_objects(door_lane.frames, Noun.CAR_DOOR, car_boxes, media_size, th) if car_boxes else door_lane.frames
    rows = []
    for frame, obj in top_frames(frames, n=n, min_gap_frames=gap, media_size=media_size, th=th):
        try:
            rgb = frame_at(src, frame / fps)
        except Exception as exc:  # noqa: BLE001 - one unreadable frame must not stop the row
            rows.append({"frame": frame, "dollars": 0.0, "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
            continue
        size = (rgb.shape[1], rgb.shape[0])
        door_src = [obj.box[0] * sx, obj.box[1] * sy, obj.box[2] * sx, obj.box[3] * sy]
        runs: dict[str, Any] = {}
        errors: dict[str, str] = {}
        usd = 0.0
        for concept in (Noun.CAR_DOOR.value, Noun.DOOR_HANDLE.value, Noun.SIDE_MIRROR.value):
            try:
                runs[concept] = await sam_client.segment_image(rgb, concept, run_id=run_id, role=role, tag=f"D1 E8 {concept} f{frame}")
                usd += runs[concept].usd
            except sam_request_error() as exc:  # one check that did not complete is recorded, the row goes on
                errors[concept] = error_text(exc)
                usd += billed_on_error(exc)
        row: dict[str, Any] = {"frame": frame, "t": round(frame / fps, 4), "door_box_src": [round(v, 1) for v in door_src], "dollars": round(usd, 4)}
        if errors:
            row["errors"] = errors
        if Noun.CAR_DOOR.value not in runs:
            rows.append({**row, "note": "the car door image check did not complete"})
            continue
        doors = _dets(runs[Noun.CAR_DOOR.value])
        if not doors:
            rows.append({**row, "note": "no car door on the 4K frame"})
            continue
        door = max(doors, key=lambda d: box_iou(_det_box(d), door_src))
        dbox = _det_box(door)
        mask = sam_client.full_mask(door, size) if door.mask is not None else None
        moments = moments_from_mask(mask) if mask is not None and int(mask.sum()) >= 3 else rect_moments(dbox)
        row.update({"door_c": door.confidence, "door_basis": "mask" if mask is not None else "box"})
        handle_dets = _dets(runs[Noun.DOOR_HANDLE.value]) if Noun.DOOR_HANDLE.value in runs else []
        handles = [d for d in handle_dets if dbox[0] <= _centroid(_det_box(d))[0] < dbox[2] and dbox[1] <= _centroid(_det_box(d))[1] < dbox[3]]
        handle = max(handles, key=lambda d: d.confidence or 0.0, default=None)
        if handle is not None:
            row.update({"handle_c": handle.confidence, "handle_w": [round(v, 5) for v in whiten(_centroid(_det_box(handle)), moments, 1)]})
        grow = 0.15 * (dbox[2] - dbox[0])
        near = [dbox[0] - grow, dbox[1] - grow, dbox[2] + grow, dbox[3] + grow]
        mirror_dets = _dets(runs[Noun.SIDE_MIRROR.value]) if Noun.SIDE_MIRROR.value in runs else []
        mirrors = [d for d in mirror_dets if containment(_det_box(d), near) > 0.5]
        mirror = max(mirrors, key=lambda d: d.confidence or 0.0, default=None)
        if mirror is not None:
            mb = _det_box(mirror)
            row.update({"mirror_c": mirror.confidence, "mirror_base_w": [round(v, 5) for v in whiten(((mb[0] + mb[2]) / 2.0, mb[3]), moments, 1)]})
        rows.append(row)
    out: dict[str, Any] = {"keyframes": rows, "dollars": round(sum(r["dollars"] for r in rows), 4)}
    for key in ("handle_w", "mirror_base_w"):
        pts = np.array([r[key] for r in rows if r.get(key)], dtype=float).reshape(-1, 2)
        if len(pts):
            mean = pts.mean(axis=0)
            out[f"{key}_mean"] = [round(float(v), 5) for v in mean]
            out[f"{key}_spread_units"] = round(float(np.sqrt(((pts - mean) ** 2).sum(axis=1).mean())), 5) if len(pts) >= 2 else None
            out[f"{key}_n"] = int(len(pts))
    return out


async def files_probe(file_id: str) -> dict[str, Any]:
    """E12: GET /v1/files/{id} through sam_client's client: whether expires_after came back as expires_at (no model call)."""
    sam_client = optional_module("sam_client")
    try:
        async with sam_client.client() as api:
            info = await api.files.retrieve(file_id)
        data = info.model_dump() if hasattr(info, "model_dump") else dict(info)
    except Exception as exc:  # noqa: BLE001 - the answer (404, 405, ...) is the finding
        return {"file_id": file_id, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    return {"file_id": file_id, "expires_at": data.get("expires_at"), "status": data.get("status"), "bytes": data.get("bytes"), "purpose": data.get("purpose"), "expires_after_accepted": data.get("expires_at") is not None}


def timeline_agreement(parsed: dict[str, Any] | None, lanes_by_noun: dict[Noun, Lane], *, fps: float, th: dict) -> dict[str, Any]:
    """E17: share of Spark timeline panels whose window lies within 1 s of a SAM track window of that noun."""
    if not parsed or not isinstance(parsed.get("panels"), list):
        return {"panels": 0, "within_1s": 0, "share": None, "note": "no panels in the timeline answer"}
    within = 0
    rows = []
    for entry in parsed["panels"]:
        name = str(entry.get("panel", "")).strip().lower()
        noun = None
        try:
            noun = contracts.PANEL_NOUN[contracts.PanelId(name)]
        except ValueError:
            noun = next((n for n in Noun if n.value == name or n.value.replace(" ", "_") == name), None)
        lane = lanes_by_noun.get(noun) if noun else None
        windows = []
        if lane is not None:
            for track in tracks_for(lane.frames, noun, fps=fps, th=th):
                windows.append((track.first_frame / fps, (track.last_frame + 1) / fps))
        start, end = float(entry.get("start_s", -99)), float(entry.get("end_s", -99))
        hit = any(abs(start - a) <= 1.0 and abs(end - b) <= 1.0 for a, b in windows)
        within += hit
        rows.append({"panel": name, "noun": noun.value if noun else None, "start_s": start, "end_s": end, "sam_windows": windows, "within_1s": hit})
    return {"panels": len(rows), "within_1s": within, "share": round(within / len(rows), 4) if rows else None, "rows": rows}


# ------------------------------------------------------------------------------------------------------------------
# E2 evaluation (run_d1 reuses it)
# ------------------------------------------------------------------------------------------------------------------


def completion_limit_s(frames: int) -> float:
    """p50 completion limit: 120 s at 400 frames, 180 s at 600, linear between, scaled beyond."""
    if frames <= 400:
        return 120.0
    if frames <= 600:
        return 120.0 + (frames - 400) * 0.3
    return 180.0 * frames / 600.0


def lost_to_429(m: dict[str, Any]) -> bool:
    """A lane that failed because the API kept answering 429 (sam_client writes '429 after N tries: ...')."""
    error = str(m.get("error") or "")
    return m.get("status") == LaneStatus.FAILED.value and (error.startswith("429") or "RateLimitError" in error)


def evaluate_sweep(sweep: dict[str, Any], th: dict) -> dict[str, Any]:
    """Pass or fail per E2 criterion for one fps sweep (both roles), from the measured metrics. The propagation
    direction probe lane is reported under prop_both_lane only: a 400 on it is an answer, not a lost request."""
    roles = {role: list((r.get("lanes") or {}).values()) for role, r in sweep["roles"].items()}
    lanes = [m for rows in roles.values() for m in rows]
    concurrent = {role: [m for m in rows if m.get("lane_id") != PROP_BOTH_LANE] for role, rows in roles.items()}
    batch = [m for rows in concurrent.values() for m in rows]
    dnb = [m for m in batch if m.get("noun") in {n.value for n in DOOR_AND_BUMPER} and m.get("kind") == "text" and not str(m.get("lane_id")).endswith(("rep2", "rep3"))]
    tracks_ok = []
    for m in dnb:
        stats = m.get("after_car_filter") or m
        share = stats.get("longest_track_share")
        frags = stats.get("longest_track_fragments")
        tracks_ok.append(share is not None and share >= th["panels"]["anchor_pass_coverage_below"] and frags is not None and frags <= 3)
    completions = [m["completed_s"] for m in lanes if m.get("completed_s") is not None and m.get("status") != LaneStatus.FAILED.value]  # a refused lane's 0.3 s is no completion
    frames_sent = max((m.get("frames_sent") or 0 for m in lanes), default=0)
    p50 = statistics.median(completions) if completions else None
    parse = [m["parse_cpu_s"] for m in lanes if m.get("parse_cpu_s") is not None]
    decode = sweep.get("decode_budget") or {}
    failed_batch = [m["lane_id"] for m in batch if m.get("status") == LaneStatus.FAILED.value]
    lost = [m["lane_id"] for m in lanes if lost_to_429(m)]
    failed_other = [m["lane_id"] for m in lanes if m.get("status") == LaneStatus.FAILED.value and not lost_to_429(m)]
    retries = sum(m.get("retries_429") or 0 for m in lanes)
    first_delta = sweep.get("first_delta_trend") or {}
    echoes = [m.get("store_echo") for m in lanes if m.get("store_echo") is not None]
    no_echo = [m["lane_id"] for m in lanes if m.get("store_echo") is None and m.get("status") != LaneStatus.FAILED.value]
    started_ok = bool(concurrent) and all(len(rows) >= 16 for rows in concurrent.values())
    return {
        "fps": sweep["fps"],
        "tracks": {"pass": bool(tracks_ok) and all(tracks_ok), "door_bumper_lanes": len(dnb), "lanes_ok": sum(tracks_ok)},
        "completion": {"pass": (p50 is not None and p50 <= completion_limit_s(frames_sent)) if p50 is not None else None, "p50_s": p50, "limit_s": completion_limit_s(frames_sent), "frames_sent": frames_sent},
        "wire_parse": {"pass": (max(parse) <= 5.0) if parse else None, "max_parse_cpu_s": max(parse) if parse else None},
        "decode": {"pass": (decode.get("decode_cpu_s", 0) <= th["panels"]["keyframe_decode_budget_s_per_video"]) if decode else None, **decode},
        "no_429_loss": {"pass": not lost if lanes else None, "failed_lanes": lost, "failed_other": failed_other, "retries_total": retries},
        "concurrency": {"pass": (not failed_batch and started_ok) if batch else None, "lanes_started": len(batch), "failed_lanes": failed_batch, "first_delta_grows": first_delta.get("grows")},
        "vfp_equals_sent": {"all_equal": all(m.get("vfp_equals_frames_sent") in (True, None) for m in lanes), "differences": {m["lane_id"]: m["vfp_minus_frames_sent"] for m in lanes if m.get("vfp_minus_frames_sent")}},
        "store_echo_false_everywhere": all(e is False for e in echoes) if echoes else None,  # a failed lane echoes nothing
        "store_echo_missing": no_echo,
        "zero_match_lanes": [m["lane_id"] for m in lanes if m.get("status") == LaneStatus.ZERO_MATCHES.value],
        "incomplete_lanes": [m["lane_id"] for m in lanes if m.get("status") == LaneStatus.INCOMPLETE.value],
    }


# ------------------------------------------------------------------------------------------------------------------
# Live path
# ------------------------------------------------------------------------------------------------------------------


def lane_plan(nouns: list[Noun], *, repeats: int, handle: bool, prop_both: bool) -> list[tuple[str, str, str]]:
    """(lane, prompt_text, tag) in the fixed order of section 5 step 5 plus the D1 extras."""
    plan = [(f"D1 E2 {n.value}", n.value, "noun") for n in nouns]
    for k in range(2, repeats + 1):
        plan.append((f"D1 E2 car door rep{k}", Noun.CAR_DOOR.value, "repeat"))
    if handle:
        plan.append(("D1 E8 door handle", Noun.DOOR_HANDLE.value, "handle"))
    if prop_both:
        plan.append(("D1 E2 car door propboth", Noun.CAR_DOOR.value, "propboth"))
    return plan


def planned_dollars(frames: int, n_lanes: int, th: dict) -> float:
    per = max(frames, th["cost"]["sam_video_min_frames"]) * th["cost"]["sam_video_usd_per_frame"]
    return round(per * n_lanes, 4)


def spend_for_prefix(prefix: str) -> float:
    """Dollars in runs/ledger.jsonl whose run_id starts with prefix."""
    return round(sum(float(r.get("usd") or 0.0) for r in ledger.rows(ledger.LEDGER_PATH) if str(r.get("run_id") or "").startswith(prefix)), 6)


def _noun_for(prompt_text: str) -> Noun | None:
    return next((n for n in Noun if n.value == prompt_text), None)


async def sweep(file_id: str, nouns: list[Noun], *, run_id: str, role: str, frames_sent: int, media_size: tuple[int, int], out: Path, repeats: int = 3, handle: bool = False, prop_both: bool = True) -> list[LaneRecord]:
    """The contract entry: 16 lanes at once (store false) plus the repeats, the handle lane and one propagation_direction lane."""
    lanes = await sweep_lanes(file_id, nouns, run_id=run_id, role=role, frames_sent=frames_sent, media_size=media_size, out=out, repeats=repeats, handle=handle, prop_both=prop_both)
    return [lane.record for lane in lanes]


async def sweep_lanes(file_id: str, nouns: list[Noun], *, run_id: str, role: str, frames_sent: int, media_size: tuple[int, int], out: Path, repeats: int, handle: bool, prop_both: bool) -> list[Lane]:
    sam_client = optional_module("sam_client")
    if sam_client is None or not implemented(sam_client, "segment_video_many"):
        raise RuntimeError("walkaround.sam_client.segment_video_many is not implemented yet; the live sweep needs it")
    plan = lane_plan(nouns, repeats=repeats, handle=handle, prop_both=prop_both)
    first_delta_wall: dict[str, float] = {}
    started = time.monotonic()
    errors: dict[str, int] = {}

    def on_event(lane: str, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "response.output_text.delta" and lane not in first_delta_wall:
            first_delta_wall[lane] = time.monotonic() - started
        if kind == "error":
            errors[lane] = errors.get(lane, 0) + 1

    prompts = [(lane, text) for lane, text, tag in plan if tag != "propboth"]
    results = await sam_client.segment_video_many(
        file_id, prompts, run_id=run_id, role=role, frames_sent=frames_sent, media_size=media_size, metadata=sam_metadata(), concurrency=16, on_event=on_event
    )
    if prop_both:
        lane_name = next(lane for lane, _, tag in plan if tag == "propboth")
        record, frames = await sam_client.segment_video(
            file_id, Noun.CAR_DOOR.value, run_id=run_id, role=role, lane=lane_name, frames_sent=frames_sent, media_size=media_size,
            metadata=sam_metadata(propagation_direction="both"), kind="text", noun=Noun.CAR_DOOR, request_index=len(prompts),
            on_event=lambda event: on_event(lane_name, event),
        )
        results[lane_name] = (record, frames)
    lanes: list[Lane] = []
    for lane_name, text, _ in plan:
        if lane_name not in results:
            continue
        record, frames = results[lane_name]
        if record.noun is None and record.kind == "text":
            record.noun = _noun_for(text)
        lanes.append(live_lane(run_id, lane_name, record, frames, first_delta_wall.get(lane_name)))
    write_json(out / f"first_delta_wall_{role}_{run_id}.json", {"first_delta_wall_s": first_delta_wall, "error_events": errors, "sweep_wall_s": round(time.monotonic() - started, 3)})
    return lanes


def bridge_window(f_a: int, f_b: int, *, half: int, max_frames: int, total: int) -> tuple[int, int] | None:
    """(first frame, frame count) of the anchor clip holding both prompt frames with up to half frames of margin, at
    most max_frames long and inside the proxy; None when the two frames are farther apart than one clip holds."""
    if f_b - f_a + 1 > max_frames:
        return None
    margin = min(half, (max_frames - (f_b - f_a + 1)) // 2)
    f0 = max(0, f_a - margin)
    end = min(total - 1, f_b + margin)
    return f0, end - f0 + 1


async def anchor_bridge(door_lane: Lane, window_lane: Lane | None, *, proxy: Path, fps: float, media_size: tuple[int, int], run_id: str, role: str, out: Path, th: dict) -> dict[str, Any]:
    """E4: the worst-fragmented door, one PromptObject with two FramePrompts, with and without the negative glass point."""
    sam_client = optional_module("sam_client")
    if sam_client is None or not implemented(sam_client, "segment_clip"):
        return {"note": "sam_client.segment_clip not implemented"}
    tracks = tracks_for(door_lane.frames, Noun.CAR_DOOR, fps=fps, th=th)
    fragmented = [t for t in tracks if len(t.fragments) >= 2]
    nothing_to_join = None
    if not fragmented:
        single = [t for t in tracks if len(t.frames) >= 2]
        if not single:
            return {"note": "no door track to bridge", "joined": None}
        track = single[0]
        frag_a = frag_b = track.fragments[0]
        # nothing to join: two prompts 2 s apart on the one track still measure the glass overlap and the negative point
        nothing_to_join = "nothing to join: no door track had 2 or more fragments, so the bridge ran on one track (two prompts 2 s apart) for the glass overlap and the negative point only"
        f_a = track.first_frame
        f_b = max((f for f in track.frames if f <= min(frag_a.last_frame, f_a + int(round(2 * fps)))), default=f_a)
    else:
        track = max(fragmented, key=lambda t: len(t.fragments))
        frag_a, frag_b = track.fragments[0], track.fragments[1]
        f_a, f_b = frag_a.last_frame, frag_b.first_frame
    box_a = _box_of(door_lane.frames, f_a, frag_a.object_id)
    box_b = _box_of(door_lane.frames, f_b, frag_b.object_id)
    if box_a is None or box_b is None:
        return {"note": "bridge boxes missing"}
    window = bridge_window(f_a, f_b, half=int(round(th["media"]["anchor_clip_half_window_s"] * fps)), max_frames=int(th["media"]["anchor_clip_frames"]), total=int(door_lane.record.frames_sent or f_b + 1))
    if window is None:
        return {"note": f"the fragments are {f_b - f_a} frames apart, more than one {th['media']['anchor_clip_frames']}-frame clip holds", "frames": [f_a, f_b]}
    f0, n = window
    clip = cut_subclip(proxy, out / "clips" / f"{role}_bridge.mp4", first_frame=f0, frames=n, fps=fps)
    size = (clip["width"], clip["height"])
    results: dict[str, Any] = {"track": {"fragments": len(track.fragments), "frames": len(track.frames)}, "frames": [f_a, f_b], "clip": clip, "variants": {}}
    if nothing_to_join:
        results["note"] = nothing_to_join
    negative = None
    if window_lane is not None:
        inside = [w for w in window_lane.objects_at(f_a) if containment(w.box, box_a) >= 0.5]
        if inside:
            negative = _centroid(inside[0].box)
    for variant in ("box_only", "box_plus_negative_point"):
        if variant == "box_plus_negative_point" and negative is None:
            results["variants"][variant] = {"note": "no car window inside the door box at the first frame; variant skipped"}
            continue
        prompts = [FramePrompt(frame=f_a - f0, box=prompt_box(box_a, size)), FramePrompt(frame=f_b - f0, box=prompt_box(box_b, size))]
        if variant == "box_plus_negative_point" and negative is not None:
            nx, ny = int(min(max(negative[0], 0), size[0] - 1)), int(min(max(negative[1], 0), size[1] - 1))
            prompts[0] = FramePrompt(frame=f_a - f0, box=prompts[0].box, points=[PromptPoint(x=nx, y=ny, label="negative")])
        try:
            text = object_prompt([PromptObject(id=1, prompts=prompts)], size)
        except Exception as exc:  # noqa: BLE001 - the builder's code is the finding (section 14 item 35)
            results["variants"][variant] = {"error": f"{type(exc).__name__}: {getattr(exc, 'code', exc)}"}
            continue
        try:
            record, frames = await sam_client.segment_clip(
                Path(clip["path"]), text, run_id=run_id, role=role, lane=f"D1 E4 bridge {variant}", media_size=size,
                metadata=sam_metadata(mask_encoding=None, propagation_direction="both"), frames_sent=clip["frames"],
            )
        except Exception as exc:  # noqa: BLE001 - a refused clip is this variant's answer, not the gate's end
            results["variants"][variant] = {"prompt": text[:200], "error": error_text(exc)}
            continue
        row: dict[str, Any] = {"prompt": text[:200], "status": record.status.value, "ids": record.ids, "frames_returned": record.frames_returned, "dollars": record.dollars}
        if record.status not in (LaneStatus.COMPLETED, LaneStatus.ZERO_MATCHES):
            # a failed or broken request measured nothing; reading it as "did not join" would send E2 to fal
            results["variants"][variant] = {**row, "error": record.error or f"lane {record.status.value}"}
            continue
        bridge = live_lane(run_id, f"D1 E4 bridge {variant}", record, frames)
        row["absorb"] = absorb_report(door_lane, bridge, bridge_object="1", frame_offset=f0, th=th, fps=fps, fragments_to_join=len(track.fragments))
        row["glass"] = glass_overlap(bridge, window_lane, bridge_object="1", frame_offset=f0)
        results["variants"][variant] = row
    a = results["variants"].get("box_only", {}).get("glass", {}).get("mean_box_iou_with_window")
    b = results["variants"].get("box_plus_negative_point", {}).get("glass", {}).get("mean_box_iou_with_window")
    results["joined"] = results["variants"].get("box_only", {}).get("absorb", {}).get("joined")  # None: nothing joined or not measured
    results["negative_point_lowers_glass_overlap"] = (b < a) if a is not None and b is not None else None
    return results


async def plate_image_checks(src: Path, plate_lane: Lane | None, *, proxy_fps: float, run_id: str, role: str, n: int = 3) -> dict[str, Any]:
    """E9: 'license plate' image requests on n unblurred 4K frames; the plate extent recorded before any blur."""
    sam_client = optional_module("sam_client")
    privacy = optional_module("privacy")
    if plate_lane is None or sam_client is None or not implemented(sam_client, "segment_image"):
        return {"note": "no plate lane or sam_client.segment_image not implemented"}
    scored = sorted(((o.box[2] - o.box[0]) * (o.box[3] - o.box[1]), fr.frame) for fr in plate_lane.frames for o in fr.objects)
    picks = list(dict.fromkeys(f for _, f in scored[::-1]))[:n]  # one request per frame, even with two plates in it
    rows = []
    for frame in picks:
        try:
            rgb = frame_at(src, frame / proxy_fps)
        except Exception as exc:  # noqa: BLE001 - one unreadable frame must not stop the sweep
            rows.append({"proxy_frame": frame, "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
            continue
        try:
            run = await sam_client.segment_image(rgb, Noun.LICENSE_PLATE.value, run_id=run_id, role=role, tag=f"D1 E9 plate f{frame}")
        except sam_request_error() as exc:  # recorded per frame; the other checks still run
            rows.append({"proxy_frame": frame, "error": error_text(exc), "dollars": billed_on_error(exc)})
            continue
        dets = [d for dets in run.frames.values() for d in dets]
        best = max(dets, key=lambda d: d.confidence or 0.0, default=None)
        extent = None
        if best is not None:
            try:
                mask = sam_client.full_mask(best, (rgb.shape[1], rgb.shape[0]))
                if privacy is not None and implemented(privacy, "plate_extent"):
                    extent = privacy.plate_extent(mask)
                else:
                    extent = _plate_extent(mask)
            except Exception as exc:  # noqa: BLE001
                extent = f"error: {exc}"
        rows.append({"proxy_frame": frame, "matches": len(dets), "best_c": best.confidence if best else None, "plate_extent_px_hw_tilt": extent, "dollars": run.usd})
    return {"frames": rows}


def _plate_extent(mask: np.ndarray) -> tuple[float, float, float] | None:
    import cv2

    pts = cv2.findNonZero(mask.astype(np.uint8))
    if pts is None:
        return None
    (_, _), (w, h), angle = cv2.minAreaRect(pts)
    long_side, short_side = max(w, h), min(w, h)
    return (float(long_side), float(short_side / long_side) if long_side else 0.0, float(angle))


async def stage(holder: dict[str, Any], key: str, work: Any, save: Callable[[], None]) -> Any:
    """Await one stage into holder[key]. An exception lands there as {'error': ...} instead of ending the gate, and
    the JSON is saved after it, so a late failure keeps every result already paid for."""
    try:
        holder[key] = await work
    except Exception as exc:  # noqa: BLE001 - the error is this stage's finding
        holder[key] = {"error": error_text(exc)}
    save()
    return holder[key]


def lane_status_row(lane: Lane) -> dict[str, Any]:
    """What a lane cost and how it ended, written before the analysis so a crash there still leaves the paid lanes."""
    rec = lane.record
    return {
        "lane_id": lane.lane_id,
        "record_lane_id": rec.lane_id,
        "noun": rec.noun.value if rec.noun else None,
        "kind": rec.kind,
        "status": rec.status.value,
        "error": rec.error,
        "request_id": rec.request_id,
        "lane_file": str(lane.events_path) if lane.events_path is not None else None,
        "media_size": list(rec.media_size),
        "store_echo": rec.store_echo,
        "frames_sent": rec.frames_sent,
        "completed_s": rec.completed_s,
        "parse_cpu_s": rec.parse_cpu_s,
        "retries_429": rec.retries,
        "dollars": rec.dollars,
    }


async def run_live(args: argparse.Namespace, th: dict, save: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    sam_client = optional_module("sam_client")
    if sam_client is None or not implemented(sam_client, "upload_file"):
        raise RuntimeError("walkaround.sam_client.upload_file is not implemented yet")
    out = Path(args.out)
    prefix = args.run_prefix or f"d1_{utc_ts()}"
    videos = [("pickup", Path(args.pickup))] + ([("return", Path(args.__dict__["return"]))] if args.__dict__.get("return") else [])
    nouns = [Noun(n) for n in args.nouns] if args.nouns else list(contracts.LANE_NOUNS)
    results: dict[str, Any] = {"mode": "live", "run_prefix": prefix, "videos": {}, "sweeps": [], "spend_usd": 0.0}

    def checkpoint() -> None:
        if save is not None:
            results["spend_usd"] = spend_for_prefix(prefix)
            save(results)

    for role, src in videos:
        info = probe_media(src)
        results["videos"][role] = {"path": str(src), "sha256": sha256_file(src), **info}
    fps_list = [float(f) for f in args.fps]
    fallback = float(args.fps_fallback) if args.fps_fallback else None
    queue = list(fps_list)
    n_lanes = len(lane_plan(nouns, repeats=args.repeats, handle=args.handle, prop_both=not args.no_prop_both))
    while queue:
        fps = queue.pop(0)
        spend = spend_for_prefix(prefix)
        planned = sum(planned_dollars(int(round(results["videos"][r]["duration_s"] * fps)), n_lanes, th) for r, _ in videos)
        if args.budget is not None and spend + planned > args.budget:
            results["sweeps"].append({"fps": fps, "skipped": f"budget ${args.budget:.2f}: spent ${spend:.4f}, this sweep plans ${planned:.2f}"})
            continue
        sweep_out: dict[str, Any] = {"fps": fps, "roles": {}, "run_id": f"{prefix}_E2_{int(fps)}fps", "planned_usd": planned}
        results["sweeps"].append(sweep_out)  # in the checkpoints from the first lane on; evaluated once both walks ran
        run_id = sweep_out["run_id"]
        first = fps == fps_list[0]
        for role, src in videos:
            role_out: dict[str, Any] = {}
            sweep_out["roles"][role] = role_out
            await run_role(args, th, sam_client, role_out, role=role, src=src, fps=fps, first=first, run_id=run_id, out=out, nouns=nouns, src_info=results["videos"][role], save=checkpoint)
        budgets = [r["decode_budget"] for r in sweep_out["roles"].values() if r.get("decode_budget")]
        sweep_out["decode_budget"] = max(budgets, key=lambda d: d.get("decode_cpu_s", 0)) if budgets else None
        trends = [r["first_delta_trend"] for r in sweep_out["roles"].values() if r.get("first_delta_trend")]
        sweep_out["first_delta_trend"] = trends[0] if trends else None
        if first and args.take2 and args.handle:
            await stage(sweep_out, "take2", take2_handles(Path(args.take2), sam_client, fps=fps, run_id=f"{prefix}_E8_take2", out=out, th=th), checkpoint)
        sweep_out["evaluation"] = evaluate_sweep(sweep_out, th)
        sweep_out["spend_usd_so_far"] = spend_for_prefix(prefix)
        checkpoint()
        if fallback is not None and first and not sweep_out["evaluation"]["tracks"]["pass"] and fallback not in queue:
            queue.append(fallback)
    if args.handle:
        results["handle_cross_walk"] = cross_walk_residual(results, th)
        results["handle_cross_take"] = cross_take_residual(results)
    results["spend_usd"] = spend_for_prefix(prefix)
    return results


async def run_role(args: argparse.Namespace, th: dict, sam_client: Any, role_out: dict[str, Any], *, role: str, src: Path, fps: float, first: bool, run_id: str, out: Path, nouns: list[Noun], src_info: dict[str, Any], save: Callable[[], None]) -> None:
    """One walk at one fps into role_out: proxy, upload, the lanes, the analysis and the extra rows that ride on them.
    The JSON is saved after the lanes and after every stage; a stage that raises records its error and the rest run.
    The upload is deleted however the role ends (unless --keep-upload)."""
    m = th["media"]
    try:
        proxy = make_proxy(src, out / "proxies" / f"{role}_{int(fps)}fps.mp4", fps=fps, max_edge=m["proxy_max_edge"], preset=m["proxy_preset"], crf=m["proxy_crf"])
        t0 = time.monotonic()
        upload = await sam_client.upload_file(Path(proxy["path"]), run_id=run_id, expires_after_s=m["files_expires_after_s"])
    except Exception as exc:  # noqa: BLE001 - nothing paid yet beyond the upload; the other walk still runs
        role_out["error"] = f"proxy or upload: {error_text(exc)}"
        save()
        return
    size = (proxy["width"], proxy["height"])
    upload_s = round(time.monotonic() - t0, 3)
    role_out["proxy"] = proxy
    role_out["upload"] = {"file_id": upload.file_id, "upload_s": upload_s, "uploaded_at": upload.uploaded_at, "expires_at": upload.expires_at}
    try:
        t_sweep = time.monotonic()
        try:
            lanes = await sweep_lanes(upload.file_id, nouns, run_id=run_id, role=role, frames_sent=proxy["frames"], media_size=size, out=out, repeats=args.repeats, handle=args.handle, prop_both=not args.no_prop_both)
        except Exception as exc:  # noqa: BLE001
            role_out["error"] = f"sweep: {error_text(exc)}"
            save()
            return
        role_out["sweep_wall_s"] = round(time.monotonic() - t_sweep, 3)
        role_out["lanes"] = {lane.lane_id: lane_status_row(lane) for lane in lanes}
        save()
        blur = privacy_boxes_by_frame(lanes, fps=fps, th=th, total=proxy["frames"])
        try:
            role_out.update(analyse_lanes(lanes, fps=fps, media_size=size, th=th, out=out / f"{role}_{int(fps)}fps", clip=Path(proxy["path"]), blur_boxes=blur, src_size=(src_info["width"], src_info["height"])))
        except Exception as exc:  # noqa: BLE001 - the status rows above stay
            role_out["analysis_error"] = error_text(exc)
        for lane_metrics_row in role_out["lanes"].values():
            lane_metrics_row["upload_s"] = upload_s
        save()
        await stage(role_out, "files_probe", files_probe(upload.file_id), save)
        by_noun = {lane.noun: lane for lane in lanes if lane.noun is not None and lane.lane_id == f"D1 E2 {lane.noun.value}"}
        try:
            car_boxes = largest_car_boxes(by_noun.get(Noun.CAR))
        except Exception:  # noqa: BLE001 - a half-built panels module; the keyframes then go without the car filter
            car_boxes = None
        if args.bridge:
            if Noun.CAR_DOOR in by_noun:
                await stage(role_out, "anchor_bridge", anchor_bridge(by_noun[Noun.CAR_DOOR], by_noun.get(Noun.CAR_WINDOW), proxy=Path(proxy["path"]), fps=fps, media_size=size, run_id=run_id, role=role, out=out, th=th), save)
            else:
                role_out["anchor_bridge"] = {"note": "no car door lane", "joined": None}
        if args.plate_checks:
            await stage(role_out, "plate_image_checks", plate_image_checks(src, by_noun.get(Noun.LICENSE_PLATE), proxy_fps=fps, run_id=run_id, role=role), save)
        if args.handle and first and Noun.CAR_DOOR in by_noun:
            await stage(role_out, "handle_keyframes", handle_keyframes(src, by_noun[Noun.CAR_DOOR], fps=fps, media_size=size, run_id=run_id, role=role, th=th, car_boxes=car_boxes), save)
        if args.timeline and role == "pickup" and first:
            await stage(role_out, "timeline", timeline_probe(Path(proxy["path"]), blur, by_noun, sam_client, fps=fps, run_id=run_id, out=out, th=th), save)
        if args.delete_response_probe and first and role == "pickup" and lanes and lanes[0].record.request_id and implemented(sam_client, "delete_response"):
            try:
                role_out["delete_response_probe"] = {"request_id": lanes[0].record.request_id, "body": await sam_client.delete_response(lanes[0].record.request_id)}
            except Exception as exc:  # noqa: BLE001 - the answer (404, 405, ...) is the finding
                role_out["delete_response_probe"] = {"request_id": lanes[0].record.request_id, "error": str(exc)[:300]}
            save()
    finally:
        if not args.keep_upload and implemented(sam_client, "delete_file"):
            try:
                role_out["delete_file"] = await sam_client.delete_file(upload.file_id)
            except Exception as exc:  # noqa: BLE001
                role_out["delete_file"] = {"error": str(exc)[:300]}
            save()


async def take2_handles(src: Path, sam_client: Any, *, fps: float, run_id: str, out: Path, th: dict) -> dict[str, Any]:
    """E8 between takes: 'car' and 'car door' lanes on take 2 of the pickup walk, then the same keyframe image requests."""
    m = th["media"]
    proxy = make_proxy(src, out / "proxies" / f"take2_{int(fps)}fps.mp4", fps=fps, max_edge=m["proxy_max_edge"], preset=m["proxy_preset"], crf=m["proxy_crf"])
    size = (proxy["width"], proxy["height"])
    upload = await sam_client.upload_file(Path(proxy["path"]), run_id=run_id, expires_after_s=m["files_expires_after_s"])
    out_row: dict[str, Any] = {"source": str(src), "proxy": proxy}
    try:
        prompts = [(f"D1 E2 {Noun.CAR.value}", Noun.CAR.value), (f"D1 E2 {Noun.CAR_DOOR.value}", Noun.CAR_DOOR.value)]
        results = await sam_client.segment_video_many(upload.file_id, prompts, run_id=run_id, role="pickup", frames_sent=proxy["frames"], media_size=size, metadata=sam_metadata(), concurrency=2)
        lanes = {name: live_lane(run_id, name, *results[name]) for name, _ in prompts if name in results}
        out_row["lanes"] = {name: {"status": lane.record.status.value, "frames_returned": lane.record.frames_returned, "dollars": lane.record.dollars, "error": lane.record.error} for name, lane in lanes.items()}
        door = lanes.get(f"D1 E2 {Noun.CAR_DOOR.value}")
        if door is not None and door.frames:
            out_row["handle_keyframes"] = await handle_keyframes(src, door, fps=fps, media_size=size, run_id=run_id, role="pickup", th=th, car_boxes=largest_car_boxes(lanes.get(f"D1 E2 {Noun.CAR.value}")))
    finally:
        if implemented(sam_client, "delete_file"):
            try:
                out_row["delete_file"] = await sam_client.delete_file(upload.file_id)
            except Exception as exc:  # noqa: BLE001
                out_row["delete_file"] = {"error": str(exc)[:300]}
    return out_row


async def timeline_probe(proxy: Path, blur: dict[int, list[Box4]], by_noun: dict[Noun, Lane], sam_client: Any, *, fps: float, run_id: str, out: Path, th: dict) -> dict[str, Any]:
    """E17: timeline_v1 over the blurred proxy (plate and face lane boxes pixelated first), uploaded on its own."""
    spark_client = optional_module("spark_client")
    video = optional_module("video")
    if spark_client is None or not implemented(spark_client, "timeline"):
        return {"note": "spark_client.timeline not implemented"}
    if video is None or not implemented(video, "render_blurred_proxy"):
        return {"note": "video.render_blurred_proxy not implemented; the timeline probe never sees an unblurred proxy"}
    p = th["privacy"]
    info = probe_media(proxy)
    scale = max(info["width"], info["height"]) / 3840.0
    blur = {k: v for k, v in blur.items() if k < int(info["frames"])}  # carry-over boxes past the last frame blur nothing
    blurred = out / "proxies" / f"{Path(proxy).stem}.blurred.mp4"
    video.render_blurred_proxy(Path(proxy), blurred, blur, block_px=max(2, int(round(p["pixel_block_px_at_4k"] * scale))), dilate_frac=p["track_box_dilation_frac"], dilate_px=int(round(p["track_box_dilation_px"] * scale)))
    upload = await sam_client.upload_file(blurred, run_id=run_id, expires_after_s=th["media"]["files_expires_after_s"])
    row: dict[str, Any] = {"proxy": str(blurred), "frames_pixelated": len(blur)}
    try:
        t0 = time.monotonic()
        record = spark_client.timeline(upload.file_id, run_id=run_id.split("_E2_")[0] + "_E17_timeline")  # its own ledger run id
        row.update({
            "seconds": round(time.monotonic() - t0, 3),
            "status": record.status.value,
            "valid": record.valid,
            "usage": record.usage.model_dump(),
            "dollars": record.dollars,
            "request_id": record.request_id,
        })
        try:
            row["agreement"] = timeline_agreement(record.parsed, by_noun, fps=fps, th=th)
        except Exception as exc:  # noqa: BLE001 - the call is paid; keep its tokens beside the error
            row["agreement"] = {"error": error_text(exc), "panels": 0, "within_1s": 0, "share": None}
    finally:
        try:
            row["delete_file"] = await sam_client.delete_file(upload.file_id)
        except Exception as exc:  # noqa: BLE001
            row["delete_file"] = {"error": str(exc)[:300]}
    return row


def _first_roles(results: dict[str, Any]) -> dict[str, Any]:
    return next((s.get("roles", {}) for s in results.get("sweeps", []) if s.get("roles")), {})


def cross_walk_residual(results: dict[str, Any], th: dict) -> dict[str, Any]:
    """E8 across walks: the mean whitened handle of each walk's door keyframes (4K image requests), else the lane
    position on the proxy; side-agnostic because the door is not matched by side here."""
    roles = _first_roles(results)
    kf = {r: (o.get("handle_keyframes") or {}).get("handle_w_mean") for r, o in roles.items()}
    if kf.get("pickup") and kf.get("return"):
        return {**side_agnostic_residual(kf["pickup"], kf["return"]), "basis": "4K door keyframes, mean of up to 3 per walk", "pickup": kf["pickup"], "return": kf["return"]}
    pos = {r: ((o.get("handle") or {}).get("position") or {}).get("w") for r, o in roles.items()}
    if pos.get("pickup") and pos.get("return"):
        return {**side_agnostic_residual(pos["pickup"], pos["return"]), "basis": "the largest proxy door holding a handle in each walk", "pickup": pos["pickup"], "return": pos["return"]}
    return {"residual": None, "note": "handle not found in both walks"}


def cross_take_residual(results: dict[str, Any]) -> dict[str, Any]:
    """E8 between takes: pickup take 1 against take 2, the same keyframe basis."""
    sweep = next((s for s in results.get("sweeps", []) if s.get("take2")), None)
    if sweep is None:
        return {"residual": None, "note": "no take 2 given (--take2)"}
    a = ((sweep["roles"].get("pickup") or {}).get("handle_keyframes") or {}).get("handle_w_mean")
    b = ((sweep["take2"].get("handle_keyframes") or {})).get("handle_w_mean")
    if not a or not b:
        return {"residual": None, "note": "handle not found in both takes"}
    return {**side_agnostic_residual(a, b), "basis": "4K door keyframes, mean of up to 3 per take", "take1": a, "take2": b}


# ------------------------------------------------------------------------------------------------------------------
# Analysis of a set of lanes (live or fixture)
# ------------------------------------------------------------------------------------------------------------------


def analyse_lanes(lanes: list[Lane], *, fps: float, media_size: tuple[int, int], th: dict, out: Path, clip: Path | None, oracle_events: list[dict[str, Any]] | None = None, blur_boxes: dict[int, list[Box4]] | None = None, src_size: tuple[int, int] | None = None) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    by_id = {lane.lane_id: lane for lane in lanes}
    car = next((lane for lane in lanes if lane.noun == Noun.CAR and lane.lane_id == f"D1 E2 {Noun.CAR.value}"), None)
    car_boxes = largest_car_boxes(car) if car is not None else None
    metrics = {lane.lane_id: lane_metrics(lane, fps=fps, th=th, car_boxes=car_boxes, media_size=media_size) for lane in lanes}
    repeats = [lane for lane in lanes if lane.lane_id in ("D1 E2 car door", "D1 E2 car door rep2", "D1 E2 car door rep3")]
    result: dict[str, Any] = {
        "fps": fps,
        "media_size": list(media_size),
        "lanes": metrics,
        "c_semantics": c_semantics([lane.record for lane in lanes], {lane.record.lane_id: lane.frames for lane in lanes}),
        "suggestions": suggestions(lanes, fps=fps, media_size=media_size, src_size=src_size, car_boxes=car_boxes, th=th),
        "variance_car_door": variance_report(repeats, fps=fps, th=th),
        "states_agreement": states_agreement(repeats, fps=fps, th=th, media_size=media_size),
        "first_delta_trend": first_delta_trend(lanes),
        "blur_coverage": blur_coverage(next((lane for lane in lanes if lane.noun == Noun.LICENSE_PLATE), None), fps=fps, th=th),
        "lane_sizes": lane_sizes(out, lanes),
        "decode_budget": decode_budget(lanes, media_size=media_size, k=th["panels"]["keyframe_top_k"], car_lane=car),
        "prop_both_lane": {
            "present": "D1 E2 car door propboth" in by_id,
            "status": metrics.get("D1 E2 car door propboth", {}).get("status"),
            "metadata_echo": metrics.get("D1 E2 car door propboth", {}).get("params"),
            "frames_returned": metrics.get("D1 E2 car door propboth", {}).get("frames_returned"),
        },
    }
    check_lane = by_id.get("D1 E2 car door") or (lanes[0] if lanes else None)
    if check_lane is not None:
        events = oracle_events
        if events is None and check_lane.events_path is not None and Path(check_lane.events_path).exists():
            events = read_events(check_lane.events_path)
        if events is not None:
            result["oracle_check"] = oracle_check(check_lane, events)
    overlay_lane = by_id.get("D1 E2 car door") or (lanes[0] if lanes else None)
    if clip is not None and Path(clip).exists() and overlay_lane is not None and overlay_lane.frames:
        visible = [fr.frame for fr in overlay_lane.frames if fr.objects]
        checks = sorted({visible[0], visible[len(visible) // 2], visible[-1]})
        result["overlay"] = render_overlay(overlay_lane, Path(clip), out / f"overlay_{contracts.noun_slug(overlay_lane.lane_id)}.mp4", media_size=media_size, check_frames=checks, blur_boxes=blur_boxes, th=th)
    elif clip is not None:
        result["overlay"] = {"note": f"clip not present: {clip}"}
    handle_lane = next((lane for lane in lanes if lane.noun == Noun.DOOR_HANDLE), None)
    door_lane = by_id.get("D1 E2 car door")
    if handle_lane is not None and door_lane is not None:
        result["handle"] = {"sigma": handle_sigma(door_lane, handle_lane, media_size=media_size, th=th), "position": handle_position(door_lane, handle_lane, media_size=media_size, th=th)}
    return result


# ------------------------------------------------------------------------------------------------------------------
# Fixture path
# ------------------------------------------------------------------------------------------------------------------


def run_fixture(args: argparse.Namespace, th: dict) -> dict[str, Any]:
    """Replay saved events as the lanes of one sweep; no call. The text fixture stands in for every noun asked."""
    out = Path(args.out)
    fixture = Path(args.fixture)
    events = read_events(fixture)
    nouns = [Noun(n) for n in args.nouns] if args.nouns else [Noun.CAR, Noun.CAR_DOOR, Noun.BUMPER, Noun.LICENSE_PLATE]
    media_size_list = media_size_from_text(lane_text(events)) or [1280, 720]
    media_size = (media_size_list[0], media_size_list[1])
    fps = float(args.fps[0]) if args.fps else 24.0
    lanes: list[Lane] = []
    for lane_name, text, tag in lane_plan(nouns, repeats=args.repeats, handle=False, prop_both=not args.no_prop_both):
        lane = lane_from_events(events, lane_id=lane_name, role="pickup", noun=_noun_for(text), prompt_text=text, kind="text", events_path=fixture)
        lane.record.media_size = list(media_size)
        lane.record.request_index = len(lanes)
        lanes.append(lane)
    clip = Path(args.clip) if args.clip else TEST_CLIP
    role_out = analyse_lanes(lanes, fps=fps, media_size=media_size, th=th, out=out / "pickup_fixture", clip=clip, oracle_events=events, src_size=media_size)
    role_out["suggestions"] = suggestions(lanes, fps=fps, media_size=media_size, src_size=media_size, car_boxes=None, th=th)
    role_out["suggestions_note"] = "fixture: car filter off, the ping pong ball is under the area floor"
    role_out["source"] = {"fixture": str(fixture), "sha256": sha256_file(fixture), "note": "one saved stream replayed as every lane; timings are the stream's own created_at to completed_at"}
    if args.bridge_fixture:
        bridge_events = read_events(Path(args.bridge_fixture))
        bridge = lane_from_events(bridge_events, lane_id="D1 E4 bridge fixture", role="pickup", noun=None, prompt_text="object prompt", kind="object", events_path=Path(args.bridge_fixture))
        bridge.record.media_size = list(media_size)
        door = next(lane for lane in lanes if lane.lane_id == f"D1 E2 {Noun.CAR_DOOR.value}") if any(l.noun == Noun.CAR_DOOR for l in lanes) else lanes[0]
        obj = bridge.record.ids[0] if bridge.record.ids else "1"
        absorb = absorb_report(door, bridge, bridge_object=obj, frame_offset=0, th=th, fps=fps)
        role_out["anchor_bridge"] = {
            "fixture": str(args.bridge_fixture),
            "variants": {
                "box_only": {
                    "status": bridge.record.status.value,
                    "ids": bridge.record.ids,
                    "c": c_stats(bridge.frames),
                    "absorb": absorb,
                    "glass": {"note": "no car window lane in the fixture"},
                },
                "box_plus_negative_point": {"note": "needs a live request"},
            },
            "joined": absorb["joined"],
            "negative_point_lowers_glass_overlap": None,
        }
        if absorb.get("note"):
            role_out["anchor_bridge"]["note"] = absorb["note"]
        role_out["lanes"][bridge.lane_id] = lane_metrics(bridge, fps=fps, th=th)
    sweep_out = {"fps": fps, "run_id": "fixture", "roles": {"pickup": role_out}, "decode_budget": role_out["decode_budget"], "first_delta_trend": role_out["first_delta_trend"]}
    sweep_out["evaluation"] = evaluate_sweep(sweep_out, th)
    return {"mode": "fixture", "run_prefix": None, "videos": {}, "sweeps": [sweep_out], "spend_usd": 0.0}


# ------------------------------------------------------------------------------------------------------------------
# Summary and CLI
# ------------------------------------------------------------------------------------------------------------------


def _fmt(value: Any, places: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{places}f}"
    return str(value)


def summary_markdown(results: dict[str, Any]) -> str:
    lines = [f"# SAM gate ({results['mode']})", ""]
    for video, info in results.get("videos", {}).items():
        lines.append(f"- {video}: {info['path']} {info['width']}x{info['height']} {_fmt(info['fps'], 2)} fps, {info['frames']} frames, sha256 {info['sha256'][:12]}")
    for sweep_out in results["sweeps"]:
        if "skipped" in sweep_out:
            lines += ["", f"## {sweep_out['fps']} fps: skipped ({sweep_out['skipped']})"]
            continue
        if "evaluation" not in sweep_out:
            lines += ["", f"## {sweep_out['fps']:g} fps: not evaluated (the gate stopped inside this sweep; gate_sam.json holds what finished)"]
            continue
        ev = sweep_out["evaluation"]
        lines += [
            "",
            f"## {sweep_out['fps']:g} fps (run {sweep_out['run_id']})",
            "",
            f"- tracks (door and bumper >= 70% of visible frames, <= 3 fragments): {_fmt(ev['tracks']['pass'])} ({ev['tracks']['lanes_ok']} of {ev['tracks']['door_bumper_lanes']} lanes)",
            f"- p50 completion: {_fmt(ev['completion']['p50_s'], 1)} s against {_fmt(ev['completion']['limit_s'], 0)} s at {ev['completion']['frames_sent']} frames: {_fmt(ev['completion']['pass'])}",
            f"- wire parse max per lane: {_fmt(ev['wire_parse']['max_parse_cpu_s'])} s: {_fmt(ev['wire_parse']['pass'])}",
            f"- keyframe decodes: {ev['decode'].get('masks_decoded')} masks in {_fmt(ev['decode'].get('decode_cpu_s'))} s CPU: {_fmt(ev['decode']['pass'])}",
            f"- no request lost to 429: {_fmt(ev['no_429_loss']['pass'])} (retries {ev['no_429_loss']['retries_total']}, lost to 429 {ev['no_429_loss']['failed_lanes']}, failed otherwise {ev['no_429_loss'].get('failed_other')})",
            f"- 16 concurrent accepted: {_fmt(ev['concurrency']['pass'])} ({ev['concurrency']['lanes_started']} lanes, failed {ev['concurrency'].get('failed_lanes')}; first delta grows with index: {_fmt(ev['concurrency']['first_delta_grows'])})",
            f"- video_frames_processed equals frames_sent everywhere: {_fmt(ev['vfp_equals_sent']['all_equal'])} {ev['vfp_equals_sent']['differences'] or ''}",
            f"- store false echoed everywhere: {_fmt(ev['store_echo_false_everywhere'])}",
            f"- zero-match lanes: {ev['zero_match_lanes'] or 'none'}; incomplete lanes: {ev['incomplete_lanes'] or 'none'}",
            "",
            "| lane | status | ids | frames | share | frags | c varies | c present | first delta s | done s | parse s | retries | $ |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for role, role_out in sweep_out["roles"].items():
            for m in (role_out.get("lanes") or {}).values():
                stats = m.get("after_car_filter") or m
                c = m.get("c") or {}
                lines.append(
                    f"| {role} {m['lane_id']} | {m['status']} | {m.get('n_ids', '-')} | {m.get('frames_returned', '-')}/{m.get('frames_sent')} | {_fmt(stats.get('longest_track_share'), 2)} | "
                    f"{_fmt(stats.get('longest_track_fragments'))} | {_fmt(c.get('c_varies_per_frame'))} | {_fmt(c.get('c_present_share'), 2)} | {_fmt(m.get('first_delta_s'), 1)} | "
                    f"{_fmt(m.get('completed_s'), 1)} | {_fmt(m.get('parse_cpu_s'))} | {m.get('retries_429')} | {_fmt(m.get('dollars'), 4)} |"
                )
            extra = [f"{key}: {role_out[key]}" for key in ("error", "analysis_error") if role_out.get(key)]
            sa = role_out.get("states_agreement", {})
            if sa.get("tracks_matched_across_runs"):
                extra.append(f"E3 states agree on the c basis {sa['agree_c_basis']} of {sa['tracks_matched_across_runs']} tracks, area-only {sa['agree_area_basis']} of {sa['tracks_matched_across_runs']}")
            bc = role_out.get("blur_coverage", {})
            if "coverage_bridged" in bc:
                extra.append(f"E9 plate-track coverage raw {bc['coverage_raw']:.2f}, with the 0.5 s carry-over {bc['coverage_bridged']:.2f}, over {len(bc.get('windows') or [])} visible window(s) {bc.get('windows_s')} (first to last detection: {bc.get('coverage_raw_first_to_last')})")
            ls = role_out.get("lane_sizes", {})
            if ls.get("total_bytes"):
                extra.append(f"E15 gz lane bytes total {ls['total_bytes']}, display nouns pair estimate {ls.get('display_nouns_pair_estimate_mb')} MB")
            oc = role_out.get("oracle_check")
            if oc:
                extra.append(f"oracle check on {oc['lane_id']}: agree {_fmt(oc['agree'])}, reader {oc['reader']} {_fmt(oc['reader_cpu_s'])} s vs parser {_fmt(oc['parser_cpu_s'])} s")
            ov = role_out.get("overlay")
            if ov and "path" in ov:
                extra.append(f"overlay {ov['path']}: {ov['masks_drawn']} masks drawn, placement ok {ov['placement_ok']}, mismatch {ov['placement_mismatch']}")
            ab = role_out.get("anchor_bridge")
            if ab and "variants" in ab:
                for name, var in ab["variants"].items():
                    if "absorb" in var:
                        extra.append(f"E4 {name}: fragments {var['absorb']['pass1_fragments']} -> {var['absorb']['fragments_after_absorb']} after absorb, joined {_fmt(var['absorb']['joined'])}{' (' + var['absorb']['note'] + ')' if var['absorb'].get('note') else ''}, glass box IoU {_fmt(var['glass'].get('mean_box_iou_with_window'))}")
                    else:
                        extra.append(f"E4 {name}: {var.get('note') or var.get('error')}")
            hd = (role_out.get("handle") or {}).get("sigma")
            if hd:
                extra.append(f"E8 sigma_frame {_fmt(hd.get('sigma_frame'), 4)} over {hd.get('frames_used')} frames")
            hk = role_out.get("handle_keyframes")
            if hk and "keyframes" in hk:
                extra.append(f"E8 door keyframes: handle found on {hk.get('handle_w_n', 0)} of {len(hk['keyframes'])}, mean w {hk.get('handle_w_mean')}, spread {_fmt(hk.get('handle_w_spread_units'), 4)} units; mirror base on {hk.get('mirror_base_w_n', 0)}")
            fp = role_out.get("files_probe")
            if fp:
                extra.append(f"E12 files probe: expires_at {fp.get('expires_at')} (expires_after accepted {_fmt(fp.get('expires_after_accepted'))}){' error ' + fp['error'] if fp.get('error') else ''}")
            if role_out.get("sweep_wall_s") is not None:
                extra.append(f"16-lane sweep wall clock {_fmt(role_out['sweep_wall_s'], 1)} s after a {_fmt(role_out.get('upload', {}).get('upload_s'), 1)} s Files upload")
            tl = role_out.get("timeline")
            if tl and "agreement" in tl:
                extra.append(f"E17 timeline: {tl['agreement']['within_1s']} of {tl['agreement']['panels']} panels within 1 s, {tl['usage']} tokens, ${tl['dollars']}")
            for item in ("delete_file", "delete_response_probe"):
                if item in role_out:
                    extra.append(f"E12 {item}: {json.dumps(role_out[item], default=str)[:200]}")
            if extra:
                lines += ["", f"### {role}", ""] + [f"- {e}" for e in extra]
    for key, label in (("handle_cross_walk", "cross-walk"), ("handle_cross_take", "cross-take")):
        row = results.get(key)
        if row:
            lines += ["", f"E8 {label} handle residual: {_fmt(row.get('residual'), 4)} units ({row.get('basis') or row.get('note')}; {row.get('used', '-')})"]
    spent = "dry-run ledger, nothing spent" if results.get("mode") == "fixture" else "ledger rows with the run prefix"
    lines += ["", f"Spend so far under this run prefix ({spent}): ${results.get('spend_usd', 0.0):.4f}", ""]
    return "\n".join(lines)


def print_plan(args: argparse.Namespace, th: dict) -> None:
    videos = [p for p in (args.pickup, args.__dict__.get("return")) if p]
    nouns = [Noun(n) for n in args.nouns] if args.nouns else list(contracts.LANE_NOUNS)
    plan = lane_plan(nouns, repeats=args.repeats, handle=args.handle, prop_both=not args.no_prop_both)
    print("gate_sam would send (set LIVE=1 to run):")
    for src in videos:
        try:
            info = probe_media(Path(src))
            seconds = info["duration_s"]
        except Exception:  # noqa: BLE001
            seconds = 40.0
        for fps in args.fps:
            frames = int(round(seconds * float(fps)))
            print(f"  {src}: proxy {th['media']['proxy_max_edge']} px at {fps} fps, about {frames} frames; {len(plan)} lanes x ${planned_dollars(frames, 1, th):.4f} = ${planned_dollars(frames, len(plan), th):.2f}")
    print("  lanes: " + ", ".join(lane for lane, _, _ in plan))
    extras = []
    if args.bridge:
        extras.append("E4 bridge clips (2 per walk, about $0.036 each)")
    if args.plate_checks:
        extras.append("E9 plate image checks (3 per walk x $0.0025)")
    if args.handle:
        extras.append("E8 door keyframes (3 per walk x 3 image requests x $0.0025)" + (" and take 2 ('car' and 'car door' lanes plus 9 image requests)" if args.take2 else ""))
    if args.timeline:
        extras.append("E17 timeline_v1 on the blurred proxy (about $0.01)")
    if extras:
        print("  plus " + "; ".join(extras))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pickup", help="the pickup walk (4K source)")
    parser.add_argument("--return", dest="return", help="the return walk")
    parser.add_argument("--fps", nargs="+", default=[15, 10], help="proxy fps sweep order (default 15 10)")
    parser.add_argument("--fps-fallback", default=30, help="run this fps only when the first fails tracks (default 30; 0 disables)")
    parser.add_argument("--nouns", nargs="+", help="override the noun list (default the 16 lanes of section 5)")
    parser.add_argument("--repeats", type=int, default=3, help="'car door' runs for variance (default 3)")
    parser.add_argument("--handle", action="store_true", help="E8: the 'door handle' lane for sigma_frame plus image requests on 3 door keyframes per walk")
    parser.add_argument("--take2", help="E8: take 2 of the pickup walk, for the cross-take handle residual ('car' and 'car door' lanes only)")
    parser.add_argument("--no-prop-both", action="store_true", help="skip the text lane sent with propagation_direction both")
    parser.add_argument("--bridge", action="store_true", help="E4 anchor bridge on the worst-fragmented door")
    parser.add_argument("--plate-checks", action="store_true", help="E9 image checks on 3 plate frames of the 4K source")
    parser.add_argument("--timeline", action="store_true", help="E17: one timeline_v1 Spark call on the first proxy")
    parser.add_argument("--delete-response-probe", action="store_true", help="E12: DELETE /v1/responses/{id} once")
    parser.add_argument("--keep-upload", action="store_true", help="do not DELETE the Files API upload at the end")
    parser.add_argument("--budget", type=float, default=None, help="stop starting sweeps past this many dollars on the run prefix")
    parser.add_argument("--run-prefix", default=None, help="ledger run_id prefix (default d1_<timestamp>)")
    parser.add_argument("--fixture", help="saved events ndjson to replay as every lane (no call)")
    parser.add_argument("--bridge-fixture", help="saved object-prompt events used as the E4 bridge result in fixture mode")
    parser.add_argument("--clip", help="the clip the fixture was made from, for the overlay MP4")
    parser.add_argument("--out", required=True, help="output directory")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    """The gate as a function: results dict, also written to <out>/gate_sam.json and gate_summary.md. The live path
    writes the JSON with partial true after every paid stage; the last write sets partial false."""
    th = contracts.load_thresholds()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.fps_fallback in (0, "0", None):
        args.fps_fallback = None
    def save(partial: dict[str, Any]) -> None:
        write_json(out / "gate_sam.json", {**partial, "partial": True, "thresholds_config_hash": contracts.config_hash(th), "written_at": utc_ts()})

    if args.fixture:
        results = run_fixture(args, th)
    else:
        results = asyncio.run(run_live(args, th, save))
    results["partial"] = False
    results["thresholds_config_hash"] = contracts.config_hash(th)
    results["written_at"] = utc_ts()
    write_json(out / "gate_sam.json", results)
    (out / "gate_summary.md").write_text(summary_markdown(results), encoding="utf-8")
    return results


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    th = contracts.load_thresholds()
    if not args.fixture:
        if not args.pickup:
            print("gate_sam: --pickup VIDEO or --fixture EVENTS is required", file=sys.stderr)
            return 2
        if not live_allowed():
            print_plan(args, th)
            return 2
    results = run(args)
    print((Path(args.out) / "gate_summary.md").read_text(encoding="utf-8"))
    print(f"wrote {Path(args.out) / 'gate_sam.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
