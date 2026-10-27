"""SAM 3.1 through the Meta Model API: upload, one concept per request, streamed parsing, replay from disk.

Request shape (docs/API_NOTES.md, cookbook 07_sam/01_api_basics/python/sam_api.py):
  POST /v1/responses, model "sam-3.1", stream=True,
  input=[{"type":"message","role":"user","content":[{"type":"input_text","text":concept}, media]}]
  media = {"type":"input_video","file_id":...} | {"type":"input_image","image_url":data_url}
  metadata={"include_confidence":"true","mask_encoding":"lossless"|"one_bit","propagation_direction":"both"}
The live API emits response.content_part.done, not response.output_text.done; synthesize the latter before handing
events to meta_sam_parser.parse_responses_stream (see the cookbook's stream_events). Save every raw event to
<run_dir>/<concept>.events.ndjson so a run can be replayed without a call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from meta_sam_parser import SegmentationMask


@dataclass(frozen=True)
class Box:
    left: float
    top: float
    right: float  # half-open
    bottom: float  # half-open

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def area(self) -> float:
        return max(self.width, 0.0) * max(self.height, 0.0)


@dataclass(frozen=True)
class Detection:
    """One object in one frame. mask is box-local (mask.width x mask.height == box size) at (left, top)."""

    object_id: str
    frame_index: int | None  # None for images
    box: Box
    mask: SegmentationMask | None
    confidence: float | None


@dataclass
class SegmentationRun:
    concept: str
    media: str  # "video" | "image"
    media_width: int
    media_height: int
    frames: dict[int, list[Detection]] = field(default_factory=dict)  # frame_index -> detections (images use 0)
    frames_billed: int = 0  # frames in the proxy (what SAM charged for), >= 50 for video
    usd: float = 0.0
    events_path: Path | None = None
    diagnostics: list[str] = field(default_factory=list)

    @property
    def object_ids(self) -> list[str]:
        ids = {d.object_id for dets in self.frames.values() for d in dets}
        return sorted(ids, key=lambda s: (len(s), s))

    def track(self, object_id: str) -> list[Detection]:
        """Detections of one object ordered by frame."""
        return [d for i in sorted(self.frames) for d in self.frames[i] if d.object_id == object_id]


def upload_file(path: Path) -> str:
    """POST /v1/files purpose=user_data. Returns the file id (starts with 'file-'). Caches by sha256 in runs/uploads.json."""
    raise NotImplementedError


def segment_video(
    file_id: str,
    concept: str,
    *,
    run_dir: Path,
    frames_in_proxy: int,
    media_size: tuple[int, int],
    include_confidence: bool = True,
    mask_encoding: str = "lossless",
    on_progress: Callable[[int, int], None] | None = None,
) -> SegmentationRun:
    """One streamed SAM request for one concept. Writes raw events to run_dir, records a ledger row
    (kind sam_video, frames=frames_in_proxy), returns the parsed run. on_progress(revision, n_records)."""
    raise NotImplementedError


def segment_video_many(
    file_id: str,
    concepts: list[str],
    *,
    run_dir: Path,
    frames_in_proxy: int,
    media_size: tuple[int, int],
    concurrency: int = 4,
) -> dict[str, SegmentationRun]:
    """segment_video for each concept, concurrently (asyncio). Concepts that fail raise after the others finish."""
    raise NotImplementedError


def segment_image(image: Path | np.ndarray, concept: str, *, run_dir: Path | None = None) -> SegmentationRun:
    """One SAM image request (data URL input). Records a ledger row (kind sam_image, images=1)."""
    raise NotImplementedError


def replay(events_path: Path, *, media: str, media_size: tuple[int, int], concept: str) -> SegmentationRun:
    """Rebuild a SegmentationRun from saved events without any API call."""
    raise NotImplementedError


def mask_array(det: Detection) -> np.ndarray:
    """Box-local boolean array (mask.height, mask.width) decoded from the lossless or one_bit payload."""
    raise NotImplementedError


def full_mask(det: Detection, media_size: tuple[int, int]) -> np.ndarray:
    """Boolean array (H, W) of the whole frame with the box-local mask pasted at (left, top)."""
    raise NotImplementedError
