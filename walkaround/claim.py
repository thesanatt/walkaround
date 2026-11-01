"""Claim-first (docs/ARCHITECTURE.md 5.11; docs/CONTRACTS.md M11): the photo from a damage bill placed against the
pickup walk.

derive_claim hashes the photo on receipt, runs the plate and face checks on it before any other model sees it and keeps
only the pixelated copy. Muse Spark then names the panel while 14 SAM 3.1 image requests vote on it, the panel is looked
up in the pickup coverage, and when it was filmed Spark grounds the mark inside the claimed area, code places it in the
panel's whitened frame and compares it with the pickup marks there through match.verdict, the one place a label is
made. A model answer can only lower that label (Spark's second look, K7). The pickup side counts only when its marks
were read on a clean, sharp frame and every one of them is placed; otherwise nothing is compared, so marks never read
are never printed as "no pickup mark". When the photo's own plate or a wheel of typed size qualifies as a ruler, one
SAM 3.1 box prompt on the mark gives the mask its size is measured from (5.12).

Every model answer is kept in claim/<claim_id>.answers.json, and the decision is a pure function of those answers, the
pixelated photo and the run's derived files, so rebuild_claim reproduces claim/<claim_id>.json with no network call.
Claim calls are billed in the ledger under '<run_id>.<claim_id>', so a claim that arrives after the run closed leaves
the run's own sheet, ledger totals and sealed files as they were.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import math
import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
from meta_sam_parser import FramePrompt, PromptObject, SegmentationMask
from PIL import Image, ImageOps
from pydantic import Field

from . import geometry, match, panels, paths, privacy, ruler, sam_client, second_look, sheet, spark_client, video
from . import marks as marks_mod
from .config import redact
from .contracts import (
    PANEL_NOUN,
    PANEL_NOUNS,
    WHEEL_PANELS,
    BlurRecord,
    Box4,
    Calibration,
    Claim,
    ClaimExit,
    ClaimImage,
    ClaimNaming,
    ClaimPlacement,
    ClaimRoi,
    Conditions,
    Contract,
    Coverage,
    CoverageState,
    Crop,
    DeletionKind,
    DeletionRecord,
    FrameUsed,
    InputRef,
    Keyframe,
    KeyframeScore,
    Landmark,
    LookVerdict,
    Mark,
    MarkKind,
    MarksStatus,
    MarkTrack,
    MatchCandidate,
    MatchState,
    Noun,
    Panel,
    PanelCoverage,
    PanelId,
    PanelMatch,
    PlacementMethod,
    Position,
    RoiSource,
    RunManifest,
    Sha256,
    Side,
    SizeEstimate,
    SparkEngine,
    SparkRequestRecord,
    Timestamp,
    VerdictFlags,
    VerdictLabel,
    VideoRecord,
    WhitenedMark,
    read_json,
    read_json_list,
    write_json,
)

# The long edge every check and model call sees; SAM and Spark resize below this anyway (estimate). Claim.roi.box is in
# this working copy's pixels and web/src/lib/claim.ts (claimWorkSize) mirrors the number, so it stays a constant here
# until the working size is a field of the Claim.
WORK_EDGE = 2048
# Defaults until thresholds.json carries claim.dark_v_max and claim.on_panel_frac (then they freeze into run.json).
DARK_V = 0.15  # pixels darker than this are left out of the saturation tests: JPEG chroma noise makes S jump there
# How far off the panel mask a claim mark's point may sit, over the crop's long side: Spark's points on real marks
# jitter by 2.0 to 2.4% of the crop width (D1 re-run on the trunk chips, docs/BUILD_STATUS.md section 0).
ON_PANEL_FRAC = 0.025
VOTE_NOUNS: tuple[Noun, ...] = PANEL_NOUNS  # car plus the 13 panel nouns: one SAM image request each (5 step 13)
NO_PANEL_NOUNS = (Noun.CAR, Noun.CAR_WINDOW)  # nouns with no canonical panel slot
SIDED_NOUNS = (Noun.CAR_DOOR, Noun.FENDER, Noun.WHEEL)
# The same shape seen from the other end. The 10/31 smoke: SAM 3.1 image requests found no "trunk lid" on the pair C
# Ambassador's rear deck and called it "hood". The vote stays strict (E2), but once the panel is settled its twin's
# mask at the claimed spot can stand in for the placement.
NOUN_TWINS = {
    Noun.HOOD: Noun.TRUNK_LID,
    Noun.TRUNK_LID: Noun.HOOD,
    Noun.HEADLIGHT: Noun.TAIL_LIGHT,
    Noun.TAIL_LIGHT: Noun.HEADLIGHT,
    Noun.WINDSHIELD: Noun.REAR_WINDOW,
    Noun.REAR_WINDOW: Noun.WINDSHIELD,
}
PICKUP_SIDE = re.compile(r"^(pickup|topup_[0-9]+)$")
_STATE_RANK = {CoverageState.FILMED: 4, CoverageState.GLIMPSED: 3, CoverageState.NOT_FILMED: 2, CoverageState.NOT_SEEN: 1, CoverageState.NOT_RUN: 0}
_SIDE_WORDS = re.compile(r"\b(left|right|lt|rt|lh|rh|passengers?|drivers?|lf|rf|lr|rr)\b", re.IGNORECASE)
_POSITION_WORDS = re.compile(r"\b(front|frt|rear|back|lf|rf|lr|rr)\b", re.IGNORECASE)
_CORNERS = ("lf", "rf", "lr", "rr")  # left front, right front, left rear, right rear
_CAP_TEXT = {
    "K2": "the pickup mark showed on one keyframe only",
    "K3": "the mark is tiny",
    "K4": "the mark may be dirt or a reflection",
    "K5": "the two marks are of different kinds",
    "K6": "the pickup mark did not pass Stays Put",
    "K7": "Muse Spark's second look doubts the pairing",
    "K8": "the pickup mark came from count mode only",
    "K9": "the panel looks like droplets or road spray",
    "K10": "one mark on a wheel cannot be placed by its radius alone",
    "K11": "glare, water or dirt on the claim photo",
    "K12": "Muse Spark's confidence in a mark is low",
    "K13": "the mark search on a pickup frame was cut short",
    "K14": "Muse Spark boxed the mark because the renter named it",
    "K15": "every usable pickup frame of this panel is flagged for glare, water or dirt, which could have hidden a mark there",
}


class ClaimRefused(ValueError):
    """The run or the photo cannot take a claim (the app answers 400 or 409 with the message)."""


class ClaimRebuildError(RuntimeError):
    """A rebuild found an answer missing or a file changed since the claim was derived."""


# The record of one claim (claim/<claim_id>.answers.json) ------------------------------------------------------------


class SamDetection(Contract):
    object_id: str
    box: Box4
    c: float | None = None
    mask_encoding: Literal["lossless", "one_bit"] | None = None
    mask_payload: str | None = None
    mask_w: int | None = None
    mask_h: int | None = None


class SamAnswer(Contract):
    """One SAM 3.1 image request on the pixelated photo, as parsed (masks kept as their wire payload)."""

    concept: str
    status: str = "completed"  # completed | incomplete | failed
    request_id: str | None = None
    detections: list[SamDetection] = Field(default_factory=list)
    error: str | None = None
    usd: float = 0.0


class ClaimAnswers(Contract):
    """The inputs and every model answer one claim used. rebuild_claim replays these and calls nothing."""

    claim_id: str
    run_id: str
    image_sha256: Sha256
    received_at: Timestamp
    w: int  # the photo as received, after its EXIF rotation
    h: int
    work_w: int
    work_h: int
    blurred_sha256: Sha256
    text: str = ""
    tire_code: str | None = None
    panel_id: PanelId | None = None  # the renter's one tap after E2 or E2b
    blur: BlurRecord = Field(default_factory=BlurRecord)
    privacy_request_ids: list[str] = Field(default_factory=list)
    sam: dict[str, SamAnswer] = Field(default_factory=dict)
    spark: dict[str, SparkRequestRecord] = Field(default_factory=dict)
    reused_from: str | None = None  # an earlier claim of the same photo whose answers were reused
    deletions: list[DeletionRecord] = Field(default_factory=list)


def answers_path(run_id: str, claim_id: str) -> Path:
    """claim/<claim_id>.answers.json; never served (the claim folder serves only images)."""
    return paths.claim_dir(run_id) / f"{claim_id}.answers.json"


# Words on the bill ---------------------------------------------------------------------------------------------------


def side_from_text(text: str, th: dict[str, Any]) -> Side | None:
    """The side the claim text names, or None when it names none or both. 'passenger' is th claim.passenger_side_is
    (right in the US) and 'driver' the other. LF, RF, LR and RR are the usual damage-report corners, but an explicit
    side word (left, right, LT, RT, LH, RH, passenger, driver) beats them: 'LT RR DOOR' is the left rear door, with RR
    read as rear."""
    passenger = Side(th["claim"]["passenger_side_is"])
    driver = Side.LEFT if passenger is Side.RIGHT else Side.RIGHT
    explicit: set[Side] = set()
    corner: set[Side] = set()
    for word in (w.lower() for w in _SIDE_WORDS.findall(text or "")):
        if word in _CORNERS:
            corner.add(Side.LEFT if word[0] == "l" else Side.RIGHT)
        elif word in ("left", "lt", "lh"):
            explicit.add(Side.LEFT)
        elif word in ("right", "rt", "rh"):
            explicit.add(Side.RIGHT)
        else:
            explicit.add(passenger if word.startswith("passenger") else driver)
    found = explicit or corner
    return found.pop() if len(found) == 1 else None


def position_from_text(text: str) -> Position | None:
    """front or rear from the claim text ('back' is rear, FRT front, a corner by its second letter), or None when it
    names none or both."""
    found: set[Position] = set()
    for word in (w.lower() for w in _POSITION_WORDS.findall(text or "")):
        found.add(Position.FRONT if word in ("front", "frt", "lf", "rf") else Position.REAR)
    return found.pop() if len(found) == 1 else None


# Pixel tests on the pixelated photo ---------------------------------------------------------------------------------


def _hsv_sv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(rgb, dtype=np.float32)[..., :3] / 255.0
    v = a.max(axis=2)
    s = np.where(v > 0, (v - a.min(axis=2)) / np.maximum(v, 1e-6), 0.0)
    return s, v


def _dark_v(th: dict[str, Any]) -> float:
    return float(th["claim"].get("dark_v_max", DARK_V))


def mean_saturation(rgb: np.ndarray, dark_v: float = DARK_V) -> float:
    """Mean HSV saturation over the pixels brighter than dark_v (0 when none are)."""
    s, v = _hsv_sv(rgb)
    keep = v >= dark_v
    return float(s[keep].mean()) if keep.any() else 0.0


def _ring_share(sat: np.ndarray, box: tuple[int, int, int, int], inset: int) -> float:
    x1, y1, x2, y2 = box[0] + inset, box[1] + inset, box[2] - 1 - inset, box[3] - 1 - inset
    if x2 <= x1 or y2 <= y1:
        return 0.0
    ring = np.concatenate([sat[y1, x1 : x2 + 1], sat[y2, x1 : x2 + 1], sat[y1 + 1 : y2, x1], sat[y1 + 1 : y2, x2]])
    return float(ring.mean()) if ring.size else 0.0


def verify_drawn_box(rgb: np.ndarray, box: tuple[int, int, int, int], th: dict[str, Any]) -> tuple[Box4 | None, dict[str, float]]:
    """Code's check of the box Spark says was drawn (5.11): the strongly saturated pixels near Spark's box give the
    drawn rectangle, which passes when one ring along its edge is at least 30% saturated (S > 0.5) and its inside at
    most 10%. Spark only says where to look; the returned ROI is measured from the pixels. D1 E11 showed why: Spark's
    box sat a few pixels off the drawn line, so testing its own perimeter missed the line every time."""
    c = th["claim"]
    s, v = _hsv_sv(rgb)
    sat = (s > float(c["drawn_box_saturation_min"])) & (v >= _dark_v(th))
    h, w = sat.shape
    x1, y1, x2, y2 = box
    margin = max(8, int(round(0.1 * max(x2 - x1, y2 - y1))))
    wx1, wy1, wx2, wy2 = max(0, x1 - margin), max(0, y1 - margin), min(w, x2 + margin), min(h, y2 + margin)
    ys, xs = np.nonzero(sat[wy1:wy2, wx1:wx2])
    stats = {"saturated_px": float(xs.size), "perimeter_saturated_share": 0.0, "inside_saturated_share": 1.0}
    if xs.size < 20:
        return None, stats
    rect = (wx1 + int(xs.min()), wy1 + int(ys.min()), wx1 + int(xs.max()) + 1, wy1 + int(ys.max()) + 1)
    band = max(3, int(round(0.02 * max(rect[2] - rect[0], rect[3] - rect[1]))))
    perimeter = max(_ring_share(sat, rect, d) for d in range(band + 1))
    inner = sat[rect[1] + 2 * band : rect[3] - 2 * band, rect[0] + 2 * band : rect[2] - 2 * band]
    inside = float(inner.mean()) if inner.size else 1.0
    stats.update({"perimeter_saturated_share": round(perimeter, 6), "inside_saturated_share": round(inside, 6)})
    ok = perimeter >= float(c["drawn_box_perimeter_saturated_min"]) and inside <= float(c["drawn_box_inside_saturated_max"])
    return ([float(v) for v in rect] if ok else None), stats


def border_share(mask: np.ndarray) -> float:
    """Share of the mask's boundary pixels that lie on the image edge (5.11: under 15% means the whole panel shows)."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return 1.0
    padded = np.pad(m, 1, constant_values=False)
    inner = padded[1:-1, 1:-1] & padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    boundary = m & ~inner
    edge = np.zeros_like(m)
    edge[0, :] = edge[-1, :] = True
    edge[:, 0] = edge[:, -1] = True
    return float((boundary & edge).sum()) / float(max(boundary.sum(), 1))


# SAM answers ----------------------------------------------------------------------------------------------------------


def sam_answer_from_run(run: sam_client.SegmentationRun) -> SamAnswer:
    dets = []
    for frame in sorted(run.frames):
        for d in run.frames[frame]:
            m = d.mask
            dets.append(
                SamDetection(
                    object_id=str(d.object_id),
                    box=[float(d.box.left), float(d.box.top), float(d.box.right), float(d.box.bottom)],
                    c=d.confidence,
                    mask_encoding=m.encoding if m is not None else None,
                    mask_payload=m.payload if m is not None else None,
                    mask_w=m.width if m is not None else None,
                    mask_h=m.height if m is not None else None,
                )
            )
    return SamAnswer(concept=run.concept, status=run.outcome, request_id=run.response_id, detections=dets, usd=float(run.usd))


def _detection(d: SamDetection) -> sam_client.Detection:
    mask = None
    if d.mask_payload is not None and d.mask_encoding is not None and d.mask_w and d.mask_h:
        mask = SegmentationMask(encoding=d.mask_encoding, payload=d.mask_payload, width=int(d.mask_w), height=int(d.mask_h))
    return sam_client.Detection(object_id=d.object_id, frame_index=None, box=sam_client.Box(*d.box), mask=mask, confidence=d.c)


class _Masks:
    """Frame-sized masks of the photo's detections, decoded once per derive (a decode costs up to 0.4 s)."""

    def __init__(self, size: tuple[int, int]) -> None:
        self.size = size
        self._cache: dict[tuple[str, str, int], np.ndarray] = {}

    def get(self, concept: str, index: int, d: SamDetection) -> np.ndarray:
        key = (concept, d.object_id, index)
        if key not in self._cache:
            self._cache[key] = privacy.detection_mask(_detection(d), self.size)
        return self._cache[key]


def _contains(d: SamDetection, center: tuple[float, float]) -> bool:
    return d.box[0] <= center[0] < d.box[2] and d.box[1] <= center[1] < d.box[3]


def _scored(concept: str, answer: SamAnswer | None, center: tuple[float, float], masks: _Masks, median_c: float) -> tuple[float, int | None]:
    """(best score, index of that detection): mask contains the centre x area share x c (c absent counts as the
    median c of the vote). Only detections whose box holds the centre are decoded."""
    if answer is None or answer.status != "completed":
        return 0.0, None
    w, h = masks.size
    cx, cy = int(min(max(center[0], 0), w - 1)), int(min(max(center[1], 0), h - 1))
    best: tuple[float, int | None] = (0.0, None)
    for i, d in enumerate(answer.detections):
        if not _contains(d, center):
            continue
        mask = masks.get(concept, i, d)
        if not mask[cy, cx]:
            continue
        score = float(mask.sum()) / float(w * h) * (d.c if d.c is not None else median_c)
        if score > best[0]:
            best = (score, i)
    return best


def sam_vote(sam: dict[str, SamAnswer], center: tuple[float, float], masks: _Masks) -> tuple[Noun | None, list[dict[str, Any]]]:
    """The SAM vote of 5.11 over the 13 panel nouns (car is asked for E1 and never wins): the winning noun or None,
    and one row per request in request order."""
    cs = [d.c for noun, a in sam.items() if noun != Noun.CAR.value and a.status == "completed" for d in a.detections if d.c is not None]
    median_c = statistics.median(cs) if cs else 0.5
    rows: list[dict[str, Any]] = []
    winner: tuple[float, Noun | None] = (0.0, None)
    for noun in VOTE_NOUNS:
        a = sam.get(noun.value)
        status = a.status if a is not None else "not_asked"
        score, _ = (0.0, None) if noun is Noun.CAR else _scored(noun.value, a, center, masks, median_c)
        rows.append({"noun": noun.value, "status": status, "matches": len(a.detections) if a is not None else 0, "score": round(score, 6)})
        if noun is not Noun.CAR and score > winner[0]:
            winner = (score, noun)
    for row in rows:
        row["winner"] = winner[1] is not None and row["noun"] == winner[1].value
    return winner[1], rows


def panel_detection(sam: dict[str, SamAnswer], noun: Noun, center: tuple[float, float], masks: _Masks) -> np.ndarray | None:
    """The mask of the noun's detection that holds the ROI centre (largest score), or None."""
    cs = [d.c for a in sam.values() if a.status == "completed" for d in a.detections if d.c is not None]
    _, index = _scored(noun.value, sam.get(noun.value), center, masks, statistics.median(cs) if cs else 0.5)
    if index is None:
        return None
    return masks.get(noun.value, index, sam[noun.value].detections[index])


def _boxes(sam: dict[str, SamAnswer], noun: Noun) -> list[Box4]:
    answer = sam.get(noun.value)
    return [d.box for d in answer.detections] if answer is not None and answer.status == "completed" else []


def photo_twin_check(
    sam: dict[str, SamAnswer], noun: Noun | None, center: tuple[float, float], plates: list[Box4], masks: _Masks, th: dict[str, Any]
) -> tuple[Noun | None, dict[str, Any]] | None:
    """panels' twin check on the photo when the SAM vote names a hood or a trunk lid: the lid is the winning detection
    (with the twin lane's detection on the same lid, box IoU >= panels.anchor_absorb_iou_min), the cues are the plate
    boxes of the privacy check and the vote's own mirror, light and glass detections, one frame of evidence each.
    (the lid the photo shows, or None when its evidence is split, and the record kept on the vote row); None when
    the vote is no lid or the run's thresholds predate the check."""
    cfg = panels.twin_settings(th)
    if cfg is None or noun not in panels.LID_NOUNS:
        return None
    answer = sam.get(noun.value)
    cs = [d.c for a in sam.values() if a.status == "completed" for d in a.detections if d.c is not None]
    _, index = _scored(noun.value, answer, center, masks, statistics.median(cs) if cs else 0.5)
    if answer is None or index is None:
        return None
    lid = answer.detections[index].box
    twin = NOUN_TWINS[noun]
    iou_min = float(th["panels"]["anchor_absorb_iou_min"])
    nouns = [noun] + ([twin] if any(panels.box_iou(box, lid) >= iou_min for box in _boxes(sam, twin)) else [])
    seen = {Noun.LICENSE_PLATE: {0: list(plates)}}
    seen.update({n: {0: _boxes(sam, n)} for n in (Noun.SIDE_MIRROR, Noun.HEADLIGHT, Noun.TAIL_LIGHT, Noun.WINDSHIELD, Noun.REAR_WINDOW)})
    votes, numbers = panels.twin_cues({0: lid}, seen, cfg, min_frames=1)
    # a photo keeps margin 2 (photo_margin_min) after the walks moved to 3: Spark's own panel name must still agree
    end, score = panels.twin_decision(votes, nouns, {**cfg, "margin_min": cfg.get("photo_margin_min", cfg["margin_min"])})
    found = panels.END_LID[end] if end is not None else None
    record = {
        "lid_nouns": [n.value for n in nouns],
        "votes": votes,
        "numbers": numbers,
        "score": score,
        "decided": end is not None,
        "panel_id": found.value if found is not None else None,
    }
    return (PANEL_NOUN[found] if found is not None else None), record


def cut_windows(windows: SamAnswer, mask: np.ndarray, masks: _Masks, th: dict[str, Any]) -> tuple[np.ndarray, bool]:
    """panels.score_keyframe's rule for a door: every car window over panels' area floor that overlaps the door mask
    is cut out of it, so the claim's moments come from the same mask definition as the pickup keyframes' (G9).
    (mask, window_subtracted); only windows whose box meets the mask's box are decoded. Panels' car-box containment
    is not applied: SAM often finds no whole car on a close-up."""
    ys, xs = np.nonzero(mask)
    if not xs.size:
        return mask, False
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    floor = panels.area_floor(Noun.CAR_WINDOW, th) * masks.size[0] * masks.size[1]
    out, cut = mask, False
    for i, d in enumerate(windows.detections):
        if (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]) < floor:
            continue
        if d.box[2] <= x1 or d.box[0] >= x2 or d.box[3] <= y1 or d.box[1] >= y2:
            continue
        window = masks.get(windows.concept, i, d)
        if (out & window).any():
            out, cut = out & ~window, True
    return out, cut


def _on_mask(mask: np.ndarray, point: list[float] | None, radius: float) -> bool:
    """True when a mask pixel centre lies within radius of the point."""
    if point is None:
        return False
    h, w = mask.shape[:2]
    x, y = float(point[0]), float(point[1])
    r = int(math.ceil(radius))
    x0, x1 = max(0, int(math.floor(x)) - r), min(w, int(math.floor(x)) + r + 1)
    y0, y1 = max(0, int(math.floor(y)) - r), min(h, int(math.floor(y)) + r + 1)
    if x0 >= x1 or y0 >= y1:
        return False
    ys, xs = np.nonzero(mask[y0:y1, x0:x1])
    return bool(((xs + x0 + 0.5 - x) ** 2 + (ys + y0 + 0.5 - y) ** 2 <= radius * radius).any())


# Spark answers read back ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Naming:
    """claim_name_v1 read back from its record exactly as spark_client.claim_name reads a fresh one."""

    valid: bool
    noun: Noun | None
    side: Side
    position: Position
    front_direction: str | None
    context: str
    landmarks: tuple[Landmark, ...]


def naming_from_record(record: SparkRequestRecord | None) -> Naming:
    if record is None or not record.valid or not isinstance(record.parsed, dict):
        return Naming(False, None, Side.NONE, Position.NONE, None, "", ())
    p = record.parsed
    panel = p.get("panel")
    noun = Noun(panel) if panel in spark_client.PANEL_NOUN_NAMES else None
    side = Side(p["side"]) if p.get("side") in {s.value for s in Side} else Side.NONE
    position = Position(p["position"]) if p.get("position") in {q.value for q in Position} else Position.NONE
    front = p.get("front_direction") if p.get("front_direction") in ("left", "right") else None
    landmarks = []
    for item in p.get("landmarks") or []:
        if not isinstance(item, dict) or not item.get("name") or not isinstance(item.get("point"), dict):
            continue
        x, y = spark_client.clamp_grid(item["point"].get("x")), spark_client.clamp_grid(item["point"].get("y"))
        if x is not None and y is not None and len(landmarks) < spark_client.LANDMARKS_MAX:
            landmarks.append(Landmark(name=str(item["name"]), point={"x": x, "y": y}))
    return Naming(True, noun, side, position, front, str(p.get("context") or ""), tuple(landmarks))


def claim_box_px(record: SparkRequestRecord | None, size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """claim_box_v1's box in photo pixels (grid -> the image Spark received -> the photo), or None."""
    if record is None or not record.valid or not isinstance(record.parsed, dict) or not record.parsed.get("found"):
        return None
    raw = record.parsed.get("bbox")
    if not isinstance(raw, dict):
        return None
    vals = [spark_client.clamp_grid(raw.get(k)) for k in ("x_min", "y_min", "x_max", "y_max")]
    if any(v is None for v in vals):
        return None
    x_min, y_min, x_max, y_max = (int(v) for v in vals)  # type: ignore[arg-type]
    box = spark_client.GridBox(x_min=min(x_min, x_max), y_min=min(y_min, y_max), x_max=max(x_min, x_max), y_max=max(y_min, y_max))
    w, h = size
    sw, sh = spark_client.sent_size(w, h)
    x1, y1, x2, y2 = spark_client.grid_box_to_pixels(box, sw, sh)
    sx, sy = w / sw, h / sh
    return int(math.floor(x1 * sx)), int(math.floor(y1 * sy)), min(w, int(math.ceil(x2 * sx))), min(h, int(math.ceil(y2 * sy)))


def look_from_record(record: SparkRequestRecord | None) -> LookVerdict:
    """look_v1 read back: an unusable answer is unsure, a downgrade (spark_client.second_look's rule)."""
    if record is None or not record.valid or not isinstance(record.parsed, dict):
        return LookVerdict.UNSURE
    try:
        return LookVerdict(record.parsed.get("verdict"))
    except ValueError:
        return LookVerdict.UNSURE


# The run the claim is placed against --------------------------------------------------------------------------------


def _read_opt(path: Path, model: Any) -> Any:
    return read_json(path, model) if path.is_file() and path.stat().st_size > 0 else None


def _read_list(path: Path, model: Any) -> list[Any]:
    return read_json_list(path, model) if path.is_file() and path.stat().st_size > 0 else []


@dataclass
class RunView:
    """The pickup facts a claim reads: frozen files of the run, read once."""

    run_id: str
    run: RunManifest
    coverage: Coverage | None
    panels: dict[PanelId, Panel]
    keyframes: dict[str, Keyframe]
    marks: list[Mark]
    mark_tracks: dict[str, MarkTrack]
    calibration: Calibration
    videos: dict[str, VideoRecord]  # videos/<role>.json of the pickup walk and its top-up clips

    @property
    def pickup_video(self) -> VideoRecord | None:
        return self.videos.get("pickup")


def _pickup_videos(run_id: str) -> dict[str, VideoRecord]:
    out: dict[str, VideoRecord] = {}
    for path in sorted(paths.videos_dir(run_id).glob("*.json")):
        record = _read_opt(path, VideoRecord) if PICKUP_SIDE.match(path.stem) else None
        if record is not None:
            out[path.stem] = record
    return out


def load_run_view(run_id: str, th: dict[str, Any]) -> RunView:
    run = read_json(paths.run_json(run_id), RunManifest)
    calibration = _read_opt(paths.calibration_json(run_id), Calibration)
    if calibration is None:
        sigma_frame, sigma_spark = match.measured_sigmas(th)
        calibration = match.calibrate(sigma_frame=sigma_frame, sigma_spark=sigma_spark, sigma_view=None, pseudo_pair_d=None, th=th)
    return RunView(
        run_id=run_id,
        run=run,
        coverage=_read_opt(paths.coverage_json(run_id), Coverage),
        panels={p.panel_id: p for p in _read_list(paths.panels_json(run_id), Panel)},
        keyframes={k.kf_id: k for k in _read_list(paths.keyframes_json(run_id), Keyframe)},
        marks=_read_list(paths.marks_json(run_id), Mark),
        mark_tracks={m.mark_id: m for m in _read_list(paths.mark_tracks_json(run_id), MarkTrack)},
        calibration=calibration,
        videos=_pickup_videos(run_id),
    )


def pickup_coverage(coverage: Coverage, panel_id: PanelId) -> PanelCoverage:
    """The panel's best state over the pickup side (the pickup walk and its top-ups); not_seen when no walk lists it."""
    best: PanelCoverage | None = None
    for role in sorted(coverage.per_role):
        pc = coverage.per_role[role].get(panel_id)
        if pc is None or not PICKUP_SIDE.match(role):
            continue
        if best is None or (_STATE_RANK[pc.state], pc.seconds) > (_STATE_RANK[best.state], best.seconds):
            best = pc
    return best or PanelCoverage(state=CoverageState.NOT_SEEN)


def _pickup_keyframe(view: RunView, panel_id: PanelId, best_kf: str | None) -> Keyframe | None:
    if best_kf and best_kf in view.keyframes:
        return view.keyframes[best_kf]
    kfs = [k for k in view.keyframes.values() if k.panel_id == panel_id and PICKUP_SIDE.match(k.role)]
    return max(kfs, key=lambda k: (k.score.total, k.kf_id), default=None)


def panel_candidates(noun: Noun | None, side: Side | None, position: Position | None) -> list[PanelId]:
    """The canonical slots a noun can be with what is known of its side and position (1 when both pin it)."""
    if noun is None or noun in NO_PANEL_NOUNS:
        return []
    sides = [side] if side not in (None, Side.NONE) else [Side.LEFT, Side.RIGHT]
    positions = [position] if position not in (None, Position.NONE) else [Position.FRONT, Position.REAR]
    out: list[PanelId] = []
    for s in sides:
        for p in positions:
            pid = spark_client.panel_from_naming(noun, s, p)
            if pid is not None and pid not in out:
                out.append(pid)
    return out


# The decision -----------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Ask:
    """One model answer the decision needs and the answers file does not hold yet."""

    kind: Literal["spark", "sam"]
    key: str
    purpose: str = ""  # claim_name | claim_box | marks | look | the SAM noun
    panel: PanelId | None = None
    box: tuple[int, int, int, int] | None = None  # the crop on the photo (marks, look)
    pickup_mark: str | None = None  # look
    claim_bbox: Box4 | None = None  # look: the claim mark in crop pixels


class NeedAnswers(Exception):
    def __init__(self, asks: list[Ask]) -> None:
        super().__init__(", ".join(a.key for a in asks))
        self.asks = asks


def _require(answers: ClaimAnswers, asks: list[Ask]) -> None:
    missing = [a for a in asks if (a.key not in answers.spark if a.kind == "spark" else a.key not in answers.sam)]
    if missing:
        raise NeedAnswers(missing)


def _crop_key(img: np.ndarray, box: tuple[int, int, int, int]) -> str:
    x1, y1, x2, y2 = box
    return hashlib.sha256(np.ascontiguousarray(img[y1:y2, x1:x2]).tobytes()).hexdigest()[:16]


def _pad_box(box: Box4, frac: float, size: tuple[int, int]) -> tuple[int, int, int, int]:
    w, h = size
    px, py = frac * (box[2] - box[0]), frac * (box[3] - box[1])
    return max(0, int(math.floor(box[0] - px))), max(0, int(math.floor(box[1] - py))), min(w, int(math.ceil(box[2] + px))), min(h, int(math.ceil(box[3] + py)))


def _name(pid: PanelId) -> str:
    return spark_client.panel_display_name(pid)


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _clock(seconds: float) -> str:
    s = max(0, int(round(seconds)))
    return f"{s // 60}:{s % 60:02d}"


def _clip_name(role: str) -> str:
    """'the pickup video' or 'top-up clip 2': the file a pickup-side frame was cut from."""
    return f"top-up clip {role.split('_', 1)[1]}" if role.startswith("topup_") else f"the {role} video"


def _wall(view: RunView, kf: Keyframe) -> str:
    """The frame's camera time from its own clip's creation time (a top-up clip has its own)."""
    record = view.videos.get(kf.role)
    camera = record.ffprobe.creation_time if record is not None and record.ffprobe is not None else "not present"
    if camera == "not present":
        return " (the video carries no camera time)"
    start = datetime.fromisoformat(str(camera).replace("Z", "+00:00"))
    at = start + timedelta(seconds=max(0, int(round(kf.pts))))  # the same whole second _clock prints
    return f" (camera time {at.astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC)"


def _frame_text(view: RunView, kf: Keyframe | None) -> str:
    return f", pickup frame at {_clock(kf.pts)} of {_clip_name(kf.role)}{_wall(view, kf)}" if kf is not None else ""


def _where(view: RunView, kf: Keyframe | None) -> str:
    if kf is None:
        return ""
    clip = "" if kf.role == "pickup" else f" of {_clip_name(kf.role)}"
    return f" on the pickup frame at {_clock(kf.pts)}{clip}{_wall(view, kf)}"


def _filmed_text(view: RunView, pid: PanelId, pc: PanelCoverage, kf: Keyframe | None) -> str:
    return f"{_cap(_name(pid))}. Filmed at pickup for {pc.seconds:.1f} s{_frame_text(view, kf)}."


def _tail(view: RunView, answers: ClaimAnswers, *kfs: Keyframe | None) -> str:
    """The run, the sha256 of the pickup video and of each top-up clip a printed frame came from, and the photo's."""
    videos = ""
    for role in ["pickup", *sorted({k.role for k in kfs if k is not None and k.role != "pickup"})]:
        record = view.videos.get(role)
        sha = (record.sha256_server or record.sha256_browser) if record is not None else None
        if sha:
            videos += f" {_cap(_clip_name(role).removeprefix('the '))} sha256 {sha}."
    return f" Run {view.run_id}.{videos} Claim photo sha256 {answers.image_sha256}. Source files available on request."


def _a_kind(kind: MarkKind) -> str:
    """'a scratch', 'a dirt mark', 'a mark of unsure type'."""
    if kind is MarkKind.UNSURE:
        return "a mark of unsure type"
    return "a dirt mark" if kind is MarkKind.DIRT else f"a {kind.value}"


def _caps_text(ids: list[str]) -> str:
    caps = [_CAP_TEXT[i] for i in ids if i in _CAP_TEXT]
    return "; ".join(caps)


def _size_text(size: SizeEstimate | None, mark_type: MarkKind | None, pid: PanelId, th: dict[str, Any]) -> str:
    if size is None:
        return ""
    line = sheet.size_line(size)
    extra = sheet.threshold_line(size, th)
    wear = sheet.ordinary_wear_line(mark_type, pid)
    text = _cap(line) if size.inches is None else "Size: " + line
    return " " + text + (f"; {extra}" if extra else "") + "." + (f" {wear}" if wear else "")


@dataclass
class _Ctx:
    view: RunView
    answers: ClaimAnswers
    img: np.ndarray
    th: dict[str, Any]
    masks: _Masks


def decide(view: RunView, answers: ClaimAnswers, img: np.ndarray, th: dict[str, Any], masks: _Masks | None = None) -> Claim:
    """The claim from the recorded answers, the pixelated photo and the run's files. Raises NeedAnswers naming every
    answer the next step needs; derive_claim fetches those and calls again, rebuild_claim treats it as an error."""
    if th["claim"].get("align_landmark_fit"):
        # the landmark fit of 5.11 is not built, so the flag would silently change nothing (a partial panel stays E5)
        raise RuntimeError("claim.align_landmark_fit is true, but this build has no landmark fit (5.11); set it to false")
    claim = _decide(view, answers, img, th, masks)
    if answers.panel_id is not None:
        claim.paragraph = f"Panel picked by you: the {_name(answers.panel_id)}. " + claim.paragraph
    return claim


def _decide(view: RunView, answers: ClaimAnswers, img: np.ndarray, th: dict[str, Any], masks: _Masks | None) -> Claim:
    h, w = img.shape[:2]
    ctx = _Ctx(view, answers, img, th, masks or _Masks((w, h)))
    c = th["claim"]
    claim = Claim(
        claim_id=answers.claim_id,
        image=ClaimImage(sha256=answers.image_sha256, received_at=answers.received_at, w=answers.w, h=answers.h, low_res=max(answers.w, answers.h) < int(c["low_res_px"])),
        text=answers.text,
        scanner_style=mean_saturation(img, _dark_v(th)) < float(c["scanner_style_saturation_max"]),
    )
    stage1 = [Ask("spark", "claim_name", "claim_name")]
    if claim.scanner_style:
        stage1.append(Ask("spark", "claim_box", "claim_box"))
    stage1 += [Ask("sam", noun.value, noun.value) for noun in VOTE_NOUNS]
    _require(answers, stage1)

    # working-copy pixels (long edge WORK_EDGE), the frame of every check and model call; the web draws Claim.roi in
    # that frame (claimWorkSize in web/src/lib/claim.ts), not in Claim.image.w and h
    roi: Box4 = [0.0, 0.0, float(w), float(h)]
    claim.roi = ClaimRoi(box=roi, source=RoiSource.WHOLE)
    if claim.scanner_style:
        spark_box = claim_box_px(answers.spark["claim_box"], (w, h))
        drawn = verify_drawn_box(img, spark_box, th)[0] if spark_box is not None else None
        if drawn is not None:
            roi = drawn
            claim.roi = ClaimRoi(box=roi, source=RoiSource.DRAWN)
    center = (0.5 * (roi[0] + roi[2]), 0.5 * (roi[1] + roi[3]))

    naming = naming_from_record(answers.spark["claim_name"])
    vote_noun, rows = sam_vote(answers.sam, center, ctx.masks)
    spark_noun = naming.noun if naming.noun not in NO_PANEL_NOUNS else None
    sam_noun = vote_noun if vote_noun not in NO_PANEL_NOUNS else None
    # The hood and the trunk lid can look alike (the 10/31 claim on pair C: the vote named the hood on the rear deck),
    # so a lid vote goes through the walk's twin check on what the photo itself shows; split evidence names no lid.
    twin = photo_twin_check(answers.sam, sam_noun, center, list(answers.blur.plate_boxes), ctx.masks, th)
    twin_split = False
    if twin is not None:
        lid, record = twin
        next(row for row in rows if row["winner"])["twin_check"] = record
        twin_split = lid is None
        sam_noun = lid if lid is not None else sam_noun
    text_side, text_pos = side_from_text(answers.text, th), position_from_text(answers.text)
    spark_side = naming.side if naming.side is not Side.NONE else None
    spark_pos = naming.position if naming.position is not Position.NONE else None
    side, side_clash = _settle(text_side, spark_side, "the {} side")
    position, pos_clash = _settle(text_pos, spark_pos, "{}")
    for row in rows:
        cands = panel_candidates(Noun(row["noun"]), side, position) if row["noun"] != Noun.CAR.value else []
        row["panel_id"] = cands[0].value if len(cands) == 1 else None
    spark_cands = panel_candidates(spark_noun, spark_side or text_side, spark_pos or text_pos)  # what Spark itself named
    vote_complete = all(row["status"] == "completed" for row in rows if row["noun"] != Noun.CAR.value)  # an unfinished one might have won
    agreed = spark_noun is not None and spark_noun == sam_noun and vote_complete and not twin_split
    claim.naming = ClaimNaming(spark=spark_cands[0] if len(spark_cands) == 1 else None, sam_votes=rows, agreed=agreed)

    if answers.panel_id is not None:
        cands = [answers.panel_id]
        noun = PANEL_NOUN[answers.panel_id]
    elif spark_noun is None and sam_noun is None:
        claim.exit = ClaimExit.E1
        car = answers.sam.get(Noun.CAR.value)
        no_car = " SAM 3.1 found no car in it." if car is not None and car.status == "completed" and not car.detections else ""
        claim.paragraph = "This photo does not show a car panel Walkaround can name, so nothing was compared." + no_car + _tail(view, answers)
        return claim
    elif not agreed:
        claim.exit = ClaimExit.E2
        a = _noun_text(spark_noun, side, position)
        b = _noun_text(sam_noun, side, position)
        unfinished = sum(1 for row in rows if row["status"] != "completed" and row["noun"] != Noun.CAR.value)
        note = f" {unfinished} of the SAM 3.1 requests did not finish." if unfinished else ""
        spark_part = f"Muse Spark named {a}" if naming.valid else "Muse Spark gave no usable answer"
        if twin_split:
            vote_part = "the SAM 3.1 vote found the hood or the trunk lid, and the photo does not show which end of the car it is"
        else:
            same_shape = " (the same shape from the other end)" if sam_noun is not None and NOUN_TWINS.get(sam_noun) == spark_noun else ""
            vote_part = f"the SAM 3.1 vote named {b}{same_shape}"
        claim.paragraph = f"{spark_part}; {vote_part}.{note} Pick the panel the bill means; your pick is recorded as your input." + _tail(view, answers)
        return claim
    else:
        noun = spark_noun
        cands = panel_candidates(noun, side, position)
        if noun in SIDED_NOUNS and len(cands) == 1:
            claim.side_source = "text" if text_side is not None else "spark"
    if len(cands) != 1:
        claim.side_source = "both_shown"
        clash = "; ".join(x for x in (side_clash, pos_clash) if x)
        claim.paragraph = _both_shown_text(view, noun, cands, clash) + _tail(view, answers)
        return claim
    pid = cands[0]
    claim.panel_id = pid
    if view.coverage is None:
        raise ClaimRefused("the pickup run has no coverage yet")
    pc = pickup_coverage(view.coverage, pid)
    claim.coverage_state = pc.state
    kf = _pickup_keyframe(view, pid, pc.best_kf)
    if pc.state in (CoverageState.NOT_FILMED, CoverageState.NOT_SEEN):
        claim.exit = ClaimExit.E3
        claim.paragraph = f"{_cap(_name(pid))}. Not filmed at pickup, so there is no dated pickup frame to compare." + _tail(view, answers)
        return claim
    if pc.state is CoverageState.GLIMPSED:
        claim.exit = ClaimExit.E4
        claim.paragraph = (
            f"{_cap(_name(pid))}. Only glimpsed at pickup ({pc.seconds:.1f} s){_frame_text(view, kf)}. Walkaround does not compare marks on a "
            "glimpsed panel; look at that frame yourself." + _tail(view, answers, kf)
        )
        return claim
    if pc.state is CoverageState.NOT_RUN:
        claim.paragraph = f"{_cap(_name(pid))}. The pickup run did not finish checking this panel, so nothing was compared." + _tail(view, answers)
        return claim
    return _layer2(ctx, claim, pid, noun, roi, center, pc, kf)


def _settle(text: Any, spark: Any, form: str) -> tuple[Any, str]:
    """The bill's word, else Spark's, and the clash sentence. 5.11 puts the text first, but when the text and Spark
    both name one and they differ, both slots are shown for the renter's tap (E2b) instead of trusting either: the
    wrong side mirrors the whitened frame."""
    if text is not None and spark is not None and text != spark:
        return None, f"the words on the bill say {form.format(text.value)} and Muse Spark sees {form.format(spark.value)}"
    return (text if text is not None else spark), ""


def _noun_text(noun: Noun | None, side: Side | None, position: Position | None) -> str:
    if noun is None:
        return "no panel"
    cands = panel_candidates(noun, side, position)
    return f"the {_name(cands[0])}" if len(cands) == 1 else f"a {noun.value}"


def _both_shown_text(view: RunView, noun: Noun, cands: list[PanelId], clash: str = "") -> str:
    why = clash or "neither the words on the bill nor the photo say which one"
    parts = [f"The photo shows a {noun.value}, but {why}."]
    for pid in cands:
        if view.coverage is None:
            parts.append(f"{_cap(_name(pid))}: no pickup coverage yet.")
            continue
        pc = pickup_coverage(view.coverage, pid)
        state = {CoverageState.FILMED: f"filmed at pickup for {pc.seconds:.1f} s", CoverageState.GLIMPSED: f"only glimpsed at pickup ({pc.seconds:.1f} s)", CoverageState.NOT_RUN: "not checked at pickup"}.get(pc.state, "not filmed at pickup")
        parts.append(f"{_cap(_name(pid))}: {state}.")
    parts.append("Pick one.")
    return " ".join(parts)


def _front_sign(view: RunView, pid: PanelId, kf: Keyframe | None) -> int | None:
    panel = view.panels.get(pid)
    if panel is not None and panel.front_sign is not None:
        return int(panel.front_sign)
    return int(kf.front_sign) if kf is not None and kf.front_sign is not None else None


def _flagged(kf: Keyframe) -> bool:
    """G2 of section 6: wet, dirty or glare on the crop (dark is not a gate)."""
    return kf.conditions.wet or kf.conditions.dirty or kf.conditions.glare


def _view_flags_cap(th: dict[str, Any]) -> bool:
    """thresholds claim.view_flag_caps (lead decision 10/31, G2 retired): a wet, dirty or glare pickup frame caps the
    claim verdict (K15) instead of stopping the comparison. Absent, as in every run frozen before it, the old refusal
    stands, so those claims rebuild from the answers they recorded."""
    return bool(th.get("claim", {}).get("view_flag_caps", False))


def _photo_flags_one_walk_only(th: dict[str, Any]) -> bool:
    """thresholds claim.photo_flag_caps_one_walk_only (review 11/1, the G2 decision for the claim photo): a wet, dirty or
    glare claim photo caps the claim mark (K11, exit E8) only when no pickup mark lies within tau_eye_r of it; a pickup
    mark that near keeps its label and the paragraph names the flag. Absent, as in every run frozen before it, K11
    caps every E8 claim mark."""
    return bool(th.get("claim", {}).get("photo_flag_caps_one_walk_only", False))


def _sent_to_baseten(view: RunView, answers: ClaimAnswers) -> bool:
    """The claim's look reached Baseten: a look answer from Muse Spark Open, or a provenance line in
    claim/<claim_id>.open_weights.ndjson (a failed attempt sent the two crops too), for this claim or the earlier claim of
    the same photo whose answers it reuses. Both files are kept, so a rebuild reads the same answer."""
    if any(key.startswith("look:") and record.engine is SparkEngine.BASETEN for key, record in answers.spark.items()):
        return True
    claim_id, reused = answers.claim_id, answers.reused_from
    seen: set[str] = set()
    while claim_id and claim_id not in seen and len(seen) < 8:
        seen.add(claim_id)
        log = paths.claim_open_weights_log(view.run_id, claim_id)
        if log.is_file() and log.stat().st_size > 0:
            return True
        claim_id = reused
        if claim_id:
            try:
                reused = read_json(answers_path(view.run_id, claim_id), ClaimAnswers).reused_from
            except (OSError, ValueError):
                reused = None
    return False


def _pickup_view_flags(view: RunView, pid: PanelId, th: dict[str, Any]) -> list[str]:
    """The wet, dirty and glare flags of the panel's pickup view when none of its complete, sharp pickup keyframes is
    clean (match.VIEW_FLAGS order); empty when one is clean, since then the pickup walk's silence at a spot means
    something. K15 caps a claim mark the pickup walk shows nothing at when this is not empty."""
    sharp_min = float(th["matching"]["sharpness_rel_min"])
    kfs = [k for k in view.keyframes.values() if k.panel_id == pid and PICKUP_SIDE.match(k.role) and k.marks_status is MarksStatus.COMPLETE and k.score.sharpness >= sharp_min]
    if not kfs or any(not match.view_flags(k) for k in kfs):
        return []
    found = {f for k in kfs for f in match.view_flags(k)}
    return [f for f in match.VIEW_FLAGS if f in found]


def _pickup_unread(view: RunView, pid: PanelId, th: dict[str, Any]) -> str | None:
    """Why the panel's pickup marks cannot be compared with, or None. The pickup side's own gates of section 6 (G6 side
    uncertain, G3 blurry, G10 every search cut short), marks never read (a claim sent before the marks stage, or after
    it failed) and a pickup mark with no place in the panel's frame all mean an absence of pickup marks proves nothing.
    One complete and sharp pickup-side keyframe is enough. A wet, dirty or glare flag on it caps the verdict (K15,
    thresholds claim.view_flag_caps); a run frozen without that key keeps the old rule (G2: one clean frame needed)."""
    panel = view.panels.get(pid)
    if panel is not None and panel.side_uncertain:
        return "The pickup walk could not tell which side this panel is on, so nothing was compared."
    kfs = [k for k in view.keyframes.values() if k.panel_id == pid and PICKUP_SIDE.match(k.role)]
    sharp_min = float(th["matching"]["sharpness_rel_min"])
    read = [k for k in kfs if k.marks_status is MarksStatus.COMPLETE]
    caps_flags = _view_flags_cap(th)  # since 10/31 a flagged pickup frame caps (K15) instead of stopping the comparison
    if not any((caps_flags or not _flagged(k)) and k.score.sharpness >= sharp_min for k in read):
        if not kfs:
            return "The pickup walk kept no frame of this panel to search for marks, so nothing was compared."
        if not caps_flags and any(_flagged(k) for k in read):
            return "The pickup frame shows glare, water or dirt, so Walkaround does not compare marks on it. Look at that frame yourself."
        if read:
            return "The pickup frames of this panel are too blurry to compare marks on. Look at that frame yourself."
        if any(k.marks_status is MarksStatus.TRUNCATED for k in kfs):
            return "The mark search on the pickup frames of this panel was cut short, so nothing was compared."
        return "The pickup marks of this panel were not read, so nothing was compared."
    unplaced = sum(1 for m in view.marks if m.panel_id == pid and PICKUP_SIDE.match(m.role) and (m.w is None or m.r is None))
    if unplaced:
        # TODO(contracts): a K id for "a pickup mark could not be placed" would let this cap instead (match.py notes the
        # same gap for walk marks)
        marks = "mark" if unplaced == 1 else "marks"
        return f"{unplaced} pickup {marks} of this panel could not be placed in its frame, so Walkaround cannot say what was at that spot. Look yourself."
    return None


def _compared_keyframes(view: RunView, pid: PanelId, kf: Keyframe | None) -> list[Keyframe]:
    """The reference pickup keyframe and every pickup-side keyframe a placed mark of the panel was whitened in."""
    ids = {m.kf_id for m in view.marks if m.panel_id == pid and PICKUP_SIDE.match(m.role) and m.w is not None}
    if kf is not None:
        ids.add(kf.kf_id)
    return [view.keyframes[i] for i in sorted(ids) if i in view.keyframes]


def _layer2(ctx: _Ctx, claim: Claim, pid: PanelId, noun: Noun, roi: Box4, center: tuple[float, float], pc: PanelCoverage, kf: Keyframe | None) -> Claim:
    """The filmed panel: the pickup side's own gates, E7, the marks call in the ROI, E6 (none, or none on the panel),
    E8, placement (E5), the comparison and the size."""
    view, answers, img, th = ctx.view, ctx.answers, ctx.img, ctx.th
    c = th["claim"]
    h, w = img.shape[:2]
    head = _filmed_text(view, pid, pc, kf)
    tail = _tail(view, answers, kf)
    unread = _pickup_unread(view, pid, th)
    if unread is not None:
        claim.paragraph = f"{head} {unread}{tail}"
        return claim
    mask = panel_detection(answers.sam, noun, center, ctx.masks)
    if mask is None and noun in NOUN_TWINS:
        mask = panel_detection(answers.sam, NOUN_TWINS[noun], center, ctx.masks)
    window_cut: bool | None = False  # None: a door whose car window request did not finish
    if mask is not None and noun is Noun.CAR_DOOR:
        windows = answers.sam.get(Noun.CAR_WINDOW.value)
        if windows is None or windows.status != "completed":
            window_cut = None
        else:
            mask, window_cut = cut_windows(windows, mask, ctx.masks, th)
    if claim.roi is not None and claim.roi.source is RoiSource.DRAWN and mask is not None:
        x1, y1, x2, y2 = (int(round(v)) for v in roi)
        covered = float(mask[y1:y2, x1:x2].sum()) / float(max(mask.sum(), 1))
        if covered > float(c["roi_region_max_panel_frac"]):
            claim.exit = ClaimExit.E7
            claim.paragraph = head + " The marked area covers most of the panel, so it is treated as a region, not one mark. Look yourself." + tail
            return claim

    box = _pad_box(roi, float(c["roi_pad_frac"]), (w, h))
    crop_key = _crop_key(img, box)
    key = f"marks:{pid.value}:{crop_key}"
    _require(answers, [Ask("spark", key, "marks", panel=pid, box=box)])
    cw, ch = box[2] - box[0], box[3] - box[1]
    landmarks = marks_mod.landmarks_for(pid, th)
    result = spark_client.marks_result_from_record(answers.spark[key], orig_width=cw, orig_height=ch, landmarks=landmarks)
    if result.answer is None:
        why = "was cut short" if result.truncated else "did not return a usable answer"
        claim.paragraph = head + f" The mark search on the photo {why}, so nothing was compared." + tail
        return claim
    cond = result.answer.conditions
    glare = cond.wet or cond.dirty or cond.glare
    if not result.answer.marks:
        claim.exit = ClaimExit.E6
        claim.paragraph = head + " Muse Spark found no mark it can name inside the claimed area." + tail
        return claim

    claim_id = answers.claim_id
    roi_crop = Crop(path=paths.rel(view.run_id, paths.claim_blurred_path(view.run_id, claim_id)), w=cw, h=ch, scale=1.0, dx=float(box[0]), dy=float(box[1]))
    pseudo = Keyframe(
        kf_id=claim_id, role=claim_id, panel_id=pid, track_id=claim_id, f_proxy=0, pts=0.0,
        score=KeyframeScore(area=0.0, completeness=0.0, sharpness=0.0, glare=0.0, frontality=0.0, total=0.0),
        aspect_ratio=w / h, box_src=[0.0, 0.0, float(w), float(h)], crop=roi_crop,
    )
    found = marks_mod.marks_from_answer(result.answer, result, pseudo, roi_crop, th=th)
    if mask is not None:
        # a mark on the grass or a neighbouring panel would be whitened far outside this panel and judged
        radius = max(3.0, float(c.get("on_panel_frac", ON_PANEL_FRAC)) * max(cw, ch))
        found = [m for m in found if _on_mask(mask, m.point_src, radius)]
        if not found:
            claim.exit = ClaimExit.E6
            claim.paragraph = head + " Muse Spark found marks in the claimed area, but none on the panel itself." + tail
            return claim
    one_walk_only = _photo_flags_one_walk_only(th)
    if glare and not one_walk_only:
        claim.exit = ClaimExit.E8
    chosen = min(found, key=lambda m: (math.dist(m.point_src or center, center), m.mark_id))
    if glare and not one_walk_only:
        chosen = chosen.model_copy(update={"caps": [*chosen.caps, "K11"]})
    photo_flags = [f for f in match.VIEW_FLAGS if getattr(cond, f)] if glare and one_walk_only else []

    front = _front_sign(view, pid, kf)
    wheel = pid in WHEEL_PANELS
    if mask is None or front is None:
        claim.exit = ClaimExit.E5
        why = "SAM 3.1 did not find the panel around the claimed spot" if mask is None else "the pickup walk could not orient this panel"
        claim.paragraph = head + f" The claim mark cannot be placed on the pickup frame: {why}. Look yourself." + tail
        return claim
    if window_cut is None:
        claim.exit = ClaimExit.E5
        claim.paragraph = head + " The claim mark cannot be placed on the pickup frame: the window check on the photo did not finish, so the door's outline is not known. Look yourself." + tail
        return claim
    full_panel = border_share(mask) < float(c["full_panel_border_max"]) and float(mask.mean()) >= float(c["full_panel_area_min"])
    if not full_panel:
        # 5.11's landmark fit stays behind claim.align_landmark_fit (false until Sanat answers; decide refuses true).
        # TODO(live): report the E5 share on the staged claim photos; a close-up may rarely show the whole panel.
        claim.exit = ClaimExit.E5
        claim.paragraph = head + " The photo shows only part of the panel, so the claim mark cannot be placed on the pickup frame. Look yourself." + tail
        return claim
    if any(k.window_subtracted != window_cut for k in _compared_keyframes(view, pid, kf)):
        # G9 of section 6: a door's moments with and without its window are two different frames
        claim.exit = ClaimExit.E5
        claim.paragraph = head + " The claim mark cannot be placed on the pickup frame: the door's outline has its window cut out on one of the two frames and not on the other. Look yourself." + tail
        return claim
    full_crop = Crop(path=roi_crop.path, w=w, h=h, scale=1.0, dx=0.0, dy=0.0)
    kf_full = pseudo.model_copy(update={"crop": full_crop, "moments": geometry.moments_from_mask(mask), "front_sign": front})
    in_full = chosen.model_copy(update={"point_crop": chosen.point_src, "bbox_crop": chosen.bbox_src})
    placed, whitened = marks_mod.whiten_marks([in_full], kf_full, th=th, mask=mask if wheel else None)
    if chosen.mark_id not in whitened:
        claim.exit = ClaimExit.E5
        claim.paragraph = head + " The claim mark cannot be placed in the panel's frame. Look yourself." + tail
        return claim
    claim_mark, wc = placed[0], whitened[chosen.mark_id]
    r_claim = float(c["r_claim_full_panel"])
    claim.placement = ClaimPlacement(method=PlacementMethod.FULL_PANEL, w_claim=[wc.u, wc.v], r_claim=r_claim)
    cmp = _compare(ctx, pid, claim_mark, wc, r_claim, box, kf, photo_flags=photo_flags)
    claim.match = cmp.match
    if photo_flags and not cmp.near:
        claim.exit = ClaimExit.E8  # K11 capped the mark, as E8 says; a near pair was compared and keeps its label
    partner_kf = view.keyframes.get(cmp.partner.kf_id) if cmp.partner is not None else None
    size = _claim_size_text(ctx, claim, claim_mark, pid, roi, cond)
    baseten = " " + privacy.CLAIM_BASETEN_CLAUSE if _sent_to_baseten(view, answers) else ""
    claim.paragraph = head + " " + _verdict_text(view, cmp, th) + size + baseten + _tail(view, answers, kf, partner_kf)
    return claim


@dataclass(frozen=True)
class _Compared:
    match: PanelMatch
    label: VerdictLabel
    ids: list[str]
    nearest: MatchCandidate | None  # the placed pickup mark nearest by cost (the pair, when one was made)
    partner: Mark | None  # that pickup mark
    ambiguous: bool  # the ratio test failed inside the band
    pair_view_flags: list[str] = field(default_factory=list)  # a near pair on a flagged pickup view: the note's flags
    near: bool = False  # a pickup mark within tau_eye_r of the claim mark
    photo_view_flags: list[str] = field(default_factory=list)  # a near pair on a flagged claim photo: the note's flags
    type_note: str | None = None  # rule 1 of 11/1: K4 and K12 on a near pair inside tau_pre_r and what the look made of them


def _compare(
    ctx: _Ctx, pid: PanelId, claim_mark: Mark, wc: WhitenedMark, r_claim: float, box: tuple[int, int, int, int], kf: Keyframe | None, *, photo_flags: list[str] | tuple[str, ...] = ()
) -> _Compared:
    """5.8 against the panel's pickup marks with the bands widened by r_claim; one verdict through match.verdict. As
    in match.verdicts_for, a pair beyond tau_eye_r passes no partner, so a far pickup mark's caps never lower it.
    photo_flags (claim.photo_flag_caps_one_walk_only): the claim photo's wet, dirty or glare flags, which cap the claim
    mark with K11 only when no pickup mark is near; a near pair keeps its label and the paragraph names them."""
    view, answers, th = ctx.view, ctx.answers, ctx.th
    cal = view.calibration
    wheel = pid in WHEEL_PANELS
    pickup = [m for m in view.marks if m.panel_id == pid and PICKUP_SIDE.match(m.role) and m.w is not None and m.r is not None]
    by_id = {m.mark_id: m for m in pickup}
    frame = FrameUsed.WHEEL if wheel else FrameUsed.MOMENT
    w_claim = wc.model_copy(update={"v": 0.0}) if wheel else wc  # angles do not compare across a turning wheel
    cands: list[MatchCandidate] = []
    for p in pickup:
        wp = WhitenedMark(mark_id=p.mark_id, role=p.role, panel_id=pid, kf_id=p.kf_id, u=p.w[0], v=0.0 if wheel else p.w[1], r=p.r, frame_used=frame)  # type: ignore[index]
        cand = match.cost(p, claim_mark, wp, w_claim, cal, th)
        cands.append(cand.model_copy(update={"tau_pre_r": cand.tau_pre_r + r_claim, "tau_eye_r": cand.tau_eye_r + r_claim}))
    pairs = match.mutual_nn(cands, th, tau_pre=cal.tau_pre)
    pair = pairs[0] if pairs else None
    nearest = min(cands, key=lambda x: (x.d, x.pickup_mark)) if cands else None
    ambiguous = False
    if pair is None and nearest is not None and nearest.d < nearest.tau_eye_r:
        pair, ambiguous = nearest, True  # two pickup marks about as close: never more than needs your eye
    partner = by_id.get(nearest.pickup_mark) if nearest is not None else None
    near = pair is not None and pair.d < pair.tau_eye_r
    kf_status = {k.kf_id: k.marks_status for k in view.keyframes.values() if k.panel_id == pid and PICKUP_SIDE.match(k.role)}
    tau_pre_r = pair.tau_pre_r if pair is not None else match.tau_r(cal.tau_pre, wc.r, th) + r_claim
    tau_eye_r = pair.tau_eye_r if pair is not None else match.tau_r(cal.tau_eye, wc.r, th) + r_claim
    # K15 (lead 10/31): with every usable pickup frame flagged, a claim mark the pickup walk shows nothing near may have
    # been hidden there; a pickup mark near it is no less there for the flag, so a near pair only gets the note
    view_flags = _pickup_view_flags(view, pid, th) if _view_flags_cap(th) else []
    view_flagged = bool(view_flags) and not near
    # Rule 1 of 11/1 (match.tight_pair_look, read from the run's frozen thresholds): on a near pair inside tau_pre_r, K4
    # and K12 wait for the second look as on a walk pair, and cap as before when it gives no answer. Runs frozen before
    # lack the leaf, so their claims rebuild unchanged.
    # K5 joins them under matching.tight_pair_k5 (lead 11/1, second): the claim mark typed another kind waits for the
    # look too.
    tight = match.tight_pair_look(th) and near and not ambiguous and partner is not None and pair is not None and pair.d < pair.tau_pre_r
    deferrable = match.tight_pair_deferrable(th)

    def flags_of(look: LookVerdict | None) -> VerdictFlags:
        flags = match.flags_for(pair, claim_mark, partner if near else None, match.staysput_caps(view.mark_tracks, th), look, kf_status, return_mark=True, view_flagged=view_flagged)
        update = {}
        if ambiguous:
            update["K1"] = True
        if wheel:
            update["K10"] = True
        if photo_flags and not near:
            update["K11"] = True
        return flags.model_copy(update=update) if update else flags

    def judge(look: LookVerdict | None, *, defer: bool = False) -> tuple[VerdictLabel, list[str]]:
        flags = flags_of(look)
        if defer:
            flags = flags.model_copy(update={k: False for k in deferrable})
        return match.verdict(pair.d if pair is not None else None, tau_pre_r, tau_eye_r, flags)

    label, ids = judge(None)
    look: LookVerdict | None = None
    if judge(None, defer=tight)[0] is VerdictLabel.ALREADY_THERE and near and partner is not None:
        look = _second_look(ctx, pid, partner, claim_mark, box)
        if look is not None:
            label, ids = judge(look, defer=tight)
    type_note = None
    if tight and partner is not None and pair is not None:
        _, _, type_note = match.tight_pair_flags(pair, partner, claim_mark, flags_of(look), look, asked=look is not None, roles=("pickup", "claim"), deferrable=deferrable)
    kf_pickup = view.keyframes.get(partner.kf_id) if near and partner is not None else kf
    pm = PanelMatch(
        panel_id=pid,
        state=MatchState.COMPARABLE,
        reasons=ids,
        kf_pair=[kf_pickup.kf_id, answers.claim_id] if kf_pickup is not None else None,
        frame_used=frame,
        pairs=[pair] if pair is not None else [],
        unmatched_pickup=[],
        unmatched_return=[] if near else [claim_mark.mark_id],
    )
    return _Compared(pm, label, ids, nearest, partner, ambiguous, view_flags if near else [], near=near, photo_view_flags=list(photo_flags) if near else [], type_note=type_note)


def _second_look(ctx: _Ctx, pid: PanelId, partner: Mark, claim_mark: Mark, box: tuple[int, int, int, int]) -> LookVerdict | None:
    """look_v1 on the pickup crop and the claim crop with both marks outlined; None when the pickup crop is gone."""
    view, answers = ctx.view, ctx.answers
    key = f"look:{partner.mark_id}:{_crop_key(ctx.img, box)}"
    if key not in answers.spark:
        kf = view.keyframes.get(partner.kf_id)
        if kf is None or kf.crop is None or not kf.crop.path or not (paths.run_dir(view.run_id) / kf.crop.path).is_file() or partner.bbox_crop is None:
            return None
        bbox = [claim_mark.bbox_src[0] - box[0], claim_mark.bbox_src[1] - box[1], claim_mark.bbox_src[2] - box[0], claim_mark.bbox_src[3] - box[1]] if claim_mark.bbox_src else None
        _require(answers, [Ask("spark", key, "look", panel=pid, box=box, pickup_mark=partner.mark_id, claim_bbox=bbox)])
    return look_from_record(answers.spark[key])


def _verdict_text(view: RunView, cmp: _Compared, th: dict[str, Any]) -> str:
    cm = float(th["matching"]["cm_per_unit_for_print"])
    label, ids, near, partner = cmp.label, cmp.ids, cmp.nearest, cmp.partner
    where = _where(view, view.keyframes.get(partner.kf_id) if partner is not None else None)
    dist = f"{near.d_pos:.2f} units (about {near.d_pos * cm:.0f} cm)" if near is not None else ""
    kind = _a_kind(partner.type) if partner is not None else "a mark"
    caps = _caps_text([i for i in ids if i.startswith("K") and i != "K1"])
    original = ids[1] if ids[0] == "L4" and len(ids) > 1 and ids[1] in ("L2", "L3", "L5") else ids[0]
    lowered = label is VerdictLabel.NEEDS_YOUR_EYE and original != "L4"
    flagged = " and ".join(cmp.pair_view_flags)
    note = f" The pickup frame was flagged {flagged}, but a mark shows at this spot on it, so something was there at pickup." if flagged else ""
    if cmp.photo_view_flags:
        photo = " and ".join(cmp.photo_view_flags)
        if original == "L3":
            note += f" The claim photo was flagged {photo}, but a pickup mark sits at this spot, so something was there at pickup."
        else:
            note += f" The claim photo was flagged {photo}; a pickup mark shows near this spot."
    if cmp.ambiguous:
        tail = f" Also: {caps}." if caps else ""
        return f"Needs your eye: two pickup marks lie about as close to the claim mark; the nearest, {kind}{where}, is {dist} away.{tail}{note}"
    if original == "L3":
        typed = f" {_cap(cmp.type_note)}." if cmp.type_note else ""
        if lowered:
            return f"Needs your eye: {kind}{where} sits {dist} from the claim mark, but {caps or 'a check lowered the answer'}.{note}{typed}"
        return f"Already there at pickup: the claim mark sits {dist} from {kind}{where}.{note}{typed}"
    if original == "L4":
        band = f"Needs your eye: the nearest pickup mark, {kind}{where}, is {dist} from the claim mark, inside the band where Walkaround does not decide."
        return band + (f" Also: {caps}." if caps else "") + note
    nearest = f" The nearest pickup mark, {kind}, is {dist} away." if near is not None else " No pickup mark was found on this panel."
    if lowered:
        return f"Needs your eye: no pickup mark at that spot, but {caps}.{nearest}"
    return f"No pickup mark at that spot.{nearest} Look yourself."


def _claim_size_text(ctx: _Ctx, claim: Claim, claim_mark: Mark, pid: PanelId, roi: Box4, cond: Conditions) -> str:
    """5.12 on the claim photo: its own plate (measured before the pixelation) for a plate-adjacent panel, the vote's
    wheels for wheel, fender and lower-bumper marks (they need the typed tire code), and the mark's length from its
    own SAM mask. That box prompt on Spark's box is asked only when such a ruler qualifies and no crop flag or drawn
    box already refuses the size; Spark's box itself is never a size."""
    answers, th = ctx.answers, ctx.th
    h, w = ctx.img.shape[:2]
    rulers: list[ruler.Ruler | str] = []
    if pid in ruler.PLATE_PANELS:
        rulers.append(ruler.plate_ruler(answers.blur, th=th, source_kf=answers.claim_id))
    wheel = answers.sam.get(Noun.WHEEL.value)
    dets = wheel.detections if pid in ruler.WHEEL_RULER_PANELS and wheel is not None and wheel.status == "completed" else []
    scale = ruler.wheel_scale(answers.tire_code)
    if dets and isinstance(scale, str):
        rulers.append(scale)  # without a readable code no wheel qualifies, so no wheel mask is decoded
    else:
        for i, d in enumerate(dets):
            touches = d.box[0] <= 0 or d.box[1] <= 0 or d.box[2] >= w or d.box[3] >= h
            rulers.append(ruler.wheel_ruler(ctx.masks.get(Noun.WHEEL.value, i, d), d.c, touches_border=touches, tire_code=answers.tire_code, th=th, source_kf=answers.claim_id))
    crop_flag = cond.any_flag()  # 5.12: any crop flag refuses the size, dark included
    drawn = claim.roi is not None and claim.roi.source is RoiSource.DRAWN and _touches_line(claim_mark.bbox_src, roi)
    extent = None
    if any(isinstance(r, ruler.Ruler) for r in rulers) and claim_mark.bbox_src is not None and not crop_flag and not drawn:
        extent = _mark_extent(ctx, claim_mark.bbox_src)
    claim.size = ruler.best_size(extent, rulers, mark_type=claim_mark.type, panel_id=pid, frame_w=w, th=th, crop_flag=crop_flag, drawn_box_overlap=drawn)
    return _size_text(claim.size, claim_mark.type, pid, th)


def _mark_box(bbox: Box4, size: tuple[int, int]) -> tuple[int, int, int, int]:
    p = sam_client.prompt_box(bbox, size)
    return int(p.left), int(p.top), int(p.right), int(p.bottom)


def _mark_extent(ctx: _Ctx, bbox: Box4) -> ruler.MarkExtent | None:
    """The claim mark's own SAM 3.1 mask from one image box prompt on Spark's box (the smoke of 10/31 showed image
    box prompts return a mask); its length along its principal axis is what 5.12 measures, never the box itself."""
    h, w = ctx.img.shape[:2]
    box = _mark_box(bbox, (w, h))
    key = f"box {box[0]} {box[1]} {box[2]} {box[3]}"
    _require(ctx.answers, [Ask("sam", key, "box", box=box)])
    answer = ctx.answers.sam[key]
    if answer.status != "completed" or not answer.detections:
        return None
    i, d = max(enumerate(answer.detections), key=lambda item: (item[1].c if item[1].c is not None else 0.0, -item[0]))
    if d.mask_payload is None:
        return None
    return ruler.extent_from_mask(ctx.masks.get(key, i, d))


def _touches_line(bbox: Box4 | None, roi: Box4, band: float = 4.0) -> bool:
    """True when the mark's box reaches the drawn rectangle's line."""
    if bbox is None:
        return False
    inside = bbox[0] >= roi[0] + band and bbox[1] >= roi[1] + band and bbox[2] <= roi[2] - band and bbox[3] <= roi[3] - band
    return not inside


# Fetching the answers a decision asked for (LIVE) ----------------------------------------------------------------------


def ledger_id(run_id: str, claim_id: str) -> str:
    """The run id claim calls are billed under."""
    return f"{run_id}.{claim_id}"


async def _sam_answer(img: np.ndarray, concept: str, *, run_id: str, claim_id: str, tag: str) -> SamAnswer:
    try:
        run = await sam_client.segment_image(img, concept, run_id=ledger_id(run_id, claim_id), role=None, tag=f"{claim_id}.{tag}")
    except sam_client.SamRequestError as exc:
        status = exc.run.outcome if exc.run is not None else "failed"
        return SamAnswer(concept=concept, status=status if status != "completed" else "failed", error=redact(str(exc))[:300])
    return sam_answer_from_run(run)


def _spark_call(ask: Ask, img: np.ndarray, view: RunView, answers: ClaimAnswers, th: dict[str, Any]) -> SparkRequestRecord:
    effort = str(th["spark"]["effort"])
    rid = ledger_id(answers.run_id, answers.claim_id)
    ref = InputRef(path=paths.rel(answers.run_id, paths.claim_blurred_path(answers.run_id, answers.claim_id)), sha256=answers.blurred_sha256)
    if ask.purpose == "claim_name":
        return spark_client.claim_name(img, run_id=rid, input_ref=ref, effort=effort).record
    if ask.purpose == "claim_box":
        return spark_client.claim_box(img, run_id=rid, input_ref=ref, effort=effort).record
    x1, y1, x2, y2 = ask.box or (0, 0, img.shape[1], img.shape[0])
    crop = np.ascontiguousarray(img[y1:y2, x1:x2])
    if ask.purpose == "marks":
        return spark_client.find_marks(crop, ask.panel, landmarks=marks_mod.landmarks_for(ask.panel, th), effort=effort, run_id=rid, input_ref=ref, fixtures=match.fixtures_not_damage(th)).record  # type: ignore[arg-type]
    partner = next(m for m in view.marks if m.mark_id == ask.pickup_mark)
    kf = view.keyframes[partner.kf_id]
    pickup = _read_rgb(paths.run_dir(view.run_id) / kf.crop.path)  # type: ignore[union-attr]
    a = marks_mod.draw_outline(pickup, partner.bbox_crop)  # type: ignore[arg-type]
    b = marks_mod.draw_outline(crop, ask.claim_bbox) if ask.claim_bbox is not None else crop
    refs = [InputRef(path=kf.crop.path, sha256=kf.crop.sha256), ref] if kf.crop is not None and kf.crop.sha256 else [ref]
    return _look_record(a, b, panel_id=ask.panel, run_id=answers.run_id, ledger_run_id=rid, effort=effort, input_refs=refs, pair=(partner.mark_id, answers.claim_id), th=th)  # type: ignore[arg-type]


def _look_record(
    image_a: np.ndarray,
    image_b: np.ndarray,
    *,
    panel_id: PanelId,
    run_id: str,
    ledger_run_id: str,
    effort: str,
    input_refs: list[InputRef],
    pair: tuple[str, str],
    th: dict[str, Any],
) -> SparkRequestRecord:
    """The claim's look (K7), asked in second_look's engine order: Muse Spark Open on Baseten first (the hackathon key,
    then the account key), hosted Muse Spark 1.3 when Baseten gives no answer, hosted only under engine "spark". The
    calls bill under the claim's ledger id; Baseten's provenance lines go to claim/<claim_id>.open_weights.ndjson (the
    run's own open_weights.ndjson is sealed at run end, so a claim never appends to it). When the run asks no look in
    its walks, the claim keeps its hosted Muse Spark look. The record that stands is kept; with none, the last one (it
    reads as unsure, a downgrade); a look nothing answered raises, as a failed hosted call always did."""
    if second_look.engine_of(th) != "off":
        claim_id = pair[1]
        standing, records, notes = second_look.look_images(
            image_a, image_b, panel_id=panel_id, run_id=run_id, ledger_run_id=ledger_run_id, input_refs=input_refs, th=th, pair=pair,
            provenance=paths.claim_open_weights_log(run_id, claim_id),
        )
        if standing is not None:
            return standing
        if records:
            return records[-1]
        raise RuntimeError("the second look got no answer: " + ("; ".join(notes) or "no engine answered"))
    return spark_client.second_look(image_a, image_b, panel_id=panel_id, run_id=ledger_run_id, effort=effort, input_refs=input_refs).record


async def fetch(asks: list[Ask], answers: ClaimAnswers, img: np.ndarray, view: RunView, th: dict[str, Any]) -> ClaimAnswers:
    """LIVE: every asked answer at once (SAM requests under the image semaphore, Spark calls in threads)."""
    sem = asyncio.Semaphore(int(th["orchestrator"]["sam_image_semaphore"]))

    async def sam_one(ask: Ask) -> tuple[Ask, Any]:
        concept = ask.purpose
        if ask.purpose == "box" and ask.box is not None:
            size = (int(img.shape[1]), int(img.shape[0]))
            concept = sam_client.object_prompt([PromptObject(id=1, prompts=[FramePrompt(frame=0, box=sam_client.prompt_box(list(ask.box), size, pad_px=0))])], size)
        async with sem:
            return ask, await _sam_answer(img, concept, run_id=answers.run_id, claim_id=answers.claim_id, tag=ask.key.replace(" ", "_"))

    async def spark_one(ask: Ask) -> tuple[Ask, Any]:
        return ask, await asyncio.to_thread(_spark_call, ask, img, view, answers, th)

    results = await asyncio.gather(*(sam_one(a) if a.kind == "sam" else spark_one(a) for a in asks))
    sam, spark = dict(answers.sam), dict(answers.spark)
    for ask, value in results:
        (sam if ask.kind == "sam" else spark)[ask.key] = value
    return answers.model_copy(update={"sam": sam, "spark": spark})


# The photo -----------------------------------------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def claim_id_for(image: Path) -> str:
    """claim/claim_3.jpg -> claim_3 (anything after the first dot of the name is dropped)."""
    name = Path(image).name.split(".", 1)[0]
    if not re.fullmatch(r"claim_[0-9]+", name):
        raise ClaimRefused(f"a claim photo is named claim_<n>, not {Path(image).name}")
    return name


def decode_photo(raw: bytes) -> np.ndarray:
    """RGB uint8 of the uploaded bytes with the EXIF rotation applied; ClaimRefused when it is not an image."""
    try:
        with Image.open(io.BytesIO(raw)) as im:
            return np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    except Exception as exc:  # PIL raises several types for a non-image
        raise ClaimRefused(f"the claim photo could not be read as an image ({type(exc).__name__})") from exc


def _plate_extent(d: sam_client.Detection, size: tuple[int, int]) -> tuple[float, float, float] | None:
    return privacy.plate_extent(privacy.detection_mask(d, size))


def working_copy(rgb: np.ndarray, edge: int = WORK_EDGE) -> np.ndarray:
    """The photo with its long edge at most `edge` px (never upscaled)."""
    h, w = rgb.shape[:2]
    scale = min(1.0, edge / max(w, h))
    if scale >= 1.0:
        return np.ascontiguousarray(rgb)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return np.asarray(Image.fromarray(rgb).resize(size, Image.Resampling.LANCZOS))


async def _privacy(work: np.ndarray, *, run_id: str, claim_id: str, th: dict[str, Any]) -> tuple[np.ndarray, BlurRecord, list[str]]:
    """The plate and face checks on the unblurred working copy, the plate extent recorded first (privacy.image_checks'
    order and rules), then the pixelation. A check that does not complete refuses the claim (section 8 item 4)."""
    size = (int(work.shape[1]), int(work.shape[0]))
    rid = ledger_id(run_id, claim_id)
    try:
        plate_run, face_run = await asyncio.gather(
            sam_client.segment_image(work, Noun.LICENSE_PLATE.value, run_id=rid, role=None, tag=f"{claim_id}.license_plate"),
            sam_client.segment_image(work, Noun.FACE.value, run_id=rid, role=None, tag=f"{claim_id}.face"),
        )
    except sam_client.SamRequestError as exc:
        raise ClaimRefused(f"the plate and face checks on the photo did not complete, so it was not kept: {redact(str(exc))[:200]}") from exc
    plates = [d for f in sorted(plate_run.frames) for d in plate_run.frames[f]]
    faces = [d for f in sorted(face_run.frames) for d in face_run.frames[f]]
    record = BlurRecord(
        plate_boxes=[[float(d.box.left), float(d.box.top), float(d.box.right), float(d.box.bottom)] for d in plates],
        face_boxes=[[float(d.box.left), float(d.box.top), float(d.box.right), float(d.box.bottom)] for d in faces],
        request_ids=[r for r in (plate_run.response_id, face_run.response_id) if r],
    )
    c_min = float(th["ruler"]["plate_c_min"])
    good = [d for d in plates if (d.confidence is None or d.confidence >= c_min) and not (d.box.left <= 0 or d.box.top <= 0 or d.box.right >= size[0] or d.box.bottom >= size[1])]
    if good:
        best = max(good, key=lambda d: max(d.box.width, d.box.height))
        extent = await asyncio.to_thread(_plate_extent, best, size)
        if extent is not None:
            record.plate_ruler_px, record.plate_tilt_hw, record.plate_ruler_c = extent[0], extent[1], best.confidence
    blurred = await asyncio.to_thread(privacy.blur_frame, work, track_boxes_src=[], detections=faces, keep_plate_region=False, th=th, plate_detections=plates)
    return blurred, record, list(record.request_ids)


def _prior_answers(run_id: str, sha: str, claim_id: str) -> ClaimAnswers | None:
    """The newest earlier claim of the same photo bytes, whose answers a re-sent photo (the renter's tap) reuses."""
    best: ClaimAnswers | None = None
    for path in sorted(paths.claim_dir(run_id).glob("claim_*.answers.json")):
        try:
            prior = read_json(path, ClaimAnswers)
        except Exception:  # noqa: BLE001 - a broken file is simply not reused
            continue
        if prior.claim_id == claim_id or prior.image_sha256 != sha:
            continue
        if not paths.claim_blurred_path(run_id, prior.claim_id).is_file():
            continue
        if best is None or (prior.received_at, prior.claim_id) > (best.received_at, best.claim_id):
            best = prior
    return best


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def _is_closed(run_id: str) -> bool:
    return (paths.runs_root() / run_id / "bundle" / "manifest_run.json").is_file()


def _write(view: RunView, answers: ClaimAnswers, claim: Claim) -> None:
    write_json(answers_path(view.run_id, answers.claim_id), answers)
    write_json(paths.claim_record_path(view.run_id, answers.claim_id), claim)
    if not _is_closed(view.run_id):
        # derived/claim.json feeds the sheet's claim appendix; a closed run's sheet is sealed, so it is left alone
        write_json(paths.claim_json(view.run_id), claim)


def _delete_upload(run_id: str, image: Path, sha: str, size: int, now: str) -> list[DeletionRecord]:
    """Remove the photo as uploaded when it sits in the run (the app writes it to claim/); only the pixelated copy
    is kept or served. A file outside the run (a script's input) is left where it is."""
    path = Path(image).resolve()
    root = paths.run_dir(run_id).resolve()
    if root not in path.parents or not path.is_file():
        return []
    target = paths.rel(run_id, path)
    path.unlink()
    return [DeletionRecord(target=target, kind=DeletionKind.FILE, sha256=sha, bytes=size, deleted_at=now, reason="the claim photo as uploaded; only the pixelated copy is kept")]


async def derive_claim(
    run_id: str,
    *,
    image: Path,
    text: str,
    tire_code: str | None,
    th: dict[str, Any],
    panel_id: str | None = None,
    claim_id: str | None = None,
) -> Claim:
    """LIVE: one claim photo against the run's pickup walk; writes claim/<claim_id>.json, its answers and the
    pixelated photo, mirrors the claim into derived/claim.json while the run is open, and returns the Claim.
    panel_id is the renter's one tap after E2 or E2b; the same photo sent again reuses every earlier answer. Every
    refusal after the photo is read still deletes the upload, and the pixel work and decide run off the event loop."""
    image = Path(image)
    if not paths.run_json(run_id).is_file():
        raise ClaimRefused(f"there is no run {run_id}")
    raw = image.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    received_at = _now()
    try:
        claim_id = claim_id or claim_id_for(image)
        chosen = _panel_choice(panel_id)
        blurred_path = paths.claim_blurred_path(run_id, claim_id)
        view = await asyncio.to_thread(load_run_view, run_id, th)
        if view.coverage is None:
            raise ClaimRefused("the pickup run has no coverage yet; send the claim once the run has finished")
        rgb = await asyncio.to_thread(decode_photo, raw)
        prior = await asyncio.to_thread(_prior_answers, run_id, sha, claim_id)
        if prior is not None:
            blurred_path.write_bytes(paths.claim_blurred_path(run_id, prior.claim_id).read_bytes())
            record, request_ids = prior.blur, list(prior.privacy_request_ids)
        else:
            work = await asyncio.to_thread(working_copy, rgb)
            blurred, record, request_ids = await _privacy(work, run_id=run_id, claim_id=claim_id, th=th)
            await asyncio.to_thread(video.write_jpeg, blurred, blurred_path)
    finally:
        deletions = _delete_upload(run_id, image, sha, len(raw), received_at)
    img = await asyncio.to_thread(_read_rgb, blurred_path)  # every test and model call reads the stored bytes, as a rebuild does
    answers = ClaimAnswers(
        claim_id=claim_id, run_id=run_id, image_sha256=sha, received_at=received_at, w=int(rgb.shape[1]), h=int(rgb.shape[0]),
        work_w=int(img.shape[1]), work_h=int(img.shape[0]), blurred_sha256=hashlib.sha256(blurred_path.read_bytes()).hexdigest(),
        text=text or "", tire_code=ruler.normalize_tire_code(tire_code) or ((tire_code or "").strip() or None), panel_id=chosen,
        blur=record, privacy_request_ids=request_ids, deletions=deletions,
    )
    if prior is not None:
        keep_sam = {k: v for k, v in prior.sam.items() if v.status == "completed"}
        keep_spark = {k: v for k, v in prior.spark.items() if v.valid or v.truncated}
        answers = answers.model_copy(update={"sam": keep_sam, "spark": keep_spark, "reused_from": prior.claim_id})
    masks = _Masks((int(img.shape[1]), int(img.shape[0])))
    for _ in range(6):
        try:
            claim = await asyncio.to_thread(decide, view, answers, img, th, masks)  # mask decodes take seconds
            break
        except NeedAnswers as need:
            answers = await fetch(need.asks, answers, img, view, th)
            await asyncio.to_thread(write_json, answers_path(run_id, claim_id), answers)  # a crash after a paid call keeps the answer
            # decide on the answers as stored (floats at 6 places), as a rebuild does: a SAM confidence such as 49/81 kept
            # at full precision put a vote score on the other side of a rounding step (pairC_core3_claim claim_4, 11/1)
            answers = await asyncio.to_thread(read_json, answers_path(run_id, claim_id), ClaimAnswers)
    else:
        raise RuntimeError("claim decision kept asking for answers")
    await asyncio.to_thread(_write, view, answers, claim)
    return claim


def _panel_choice(panel_id: str | None) -> PanelId | None:
    if not panel_id:
        return None
    try:
        return PanelId(panel_id)
    except ValueError as exc:
        raise ClaimRefused(f"{panel_id!r} is not a panel id") from exc


def rebuild_claim(run_id: str, claim_id: str, *, th: dict[str, Any], write: bool = True) -> Claim:
    """The claim from its recorded answers, the pixelated photo and the run's files, with no network call."""
    answers = read_json(answers_path(run_id, claim_id), ClaimAnswers)
    blurred_path = paths.claim_blurred_path(run_id, claim_id)
    if not blurred_path.is_file() or hashlib.sha256(blurred_path.read_bytes()).hexdigest() != answers.blurred_sha256:
        raise ClaimRebuildError(f"{paths.rel(run_id, blurred_path)} is missing or changed since the claim was derived")
    img = _read_rgb(blurred_path)
    view = load_run_view(run_id, th)
    try:
        claim = decide(view, answers, img, th)
    except NeedAnswers as need:
        raise ClaimRebuildError(f"claim {claim_id} has no recorded answer for {need}") from need
    if write:
        write_json(paths.claim_record_path(run_id, claim_id), claim)
    return claim


def rebuild_claims(run_id: str, *, th: dict[str, Any]) -> dict[str, str]:
    """Every claim of the run rebuilt in place ({file: sha256 or 'failed: ...'}). derived/claim.json, where it exists,
    is rewritten from the claim it already holds: a claim sent after the run closed never entered it (the sheet was
    sealed), so a rebuild must not put it there either, and a rebuild never adds a file the run did not have."""
    out: dict[str, str] = {}
    rebuilt: dict[str, Claim] = {}
    for path in sorted(paths.claim_dir(run_id).glob("claim_*.answers.json")):
        claim_id = path.name.split(".", 1)[0]
        try:
            rebuilt[claim_id] = rebuild_claim(run_id, claim_id, th=th)
        except Exception as exc:  # noqa: BLE001 - one bad claim must not hide the others
            out[f"claim/{claim_id}.json"] = f"failed: {redact(str(exc))[:200]}"
            continue
        out[f"claim/{claim_id}.json"] = hashlib.sha256(paths.claim_record_path(run_id, claim_id).read_bytes()).hexdigest()
    current = paths.claim_json(run_id)
    if current.is_file():
        try:
            held = read_json(current, Claim).claim_id
        except Exception as exc:  # noqa: BLE001 - reported, and the recorded file stands
            out["claim.json"] = f"failed: derived/claim.json could not be read ({type(exc).__name__}); the recorded file stands"
            return out
        if held in rebuilt:
            out["claim.json"] = write_json(current, rebuilt[held])
        else:
            out["claim.json"] = f"failed: {held or 'its claim'} did not rebuild; the recorded file stands"
    return out


__all__ = [
    "Ask",
    "ClaimAnswers",
    "ClaimRebuildError",
    "ClaimRefused",
    "NeedAnswers",
    "Naming",
    "RunView",
    "SamAnswer",
    "SamDetection",
    "answers_path",
    "border_share",
    "claim_box_px",
    "claim_id_for",
    "cut_windows",
    "decide",
    "decode_photo",
    "derive_claim",
    "fetch",
    "ledger_id",
    "load_run_view",
    "look_from_record",
    "mean_saturation",
    "naming_from_record",
    "panel_candidates",
    "panel_detection",
    "pickup_coverage",
    "position_from_text",
    "rebuild_claim",
    "rebuild_claims",
    "sam_answer_from_run",
    "sam_vote",
    "side_from_text",
    "verify_drawn_box",
    "working_copy",
]
