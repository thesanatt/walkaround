"""ffmpeg and ffprobe helpers: proxies for SAM, full-resolution frames for Spark, sharpness, hashes.

Design (docs/API_NOTES.md decision 2): SAM is billed per frame and tracks every frame it is given, so it gets a
small proxy (H.264 MP4, longest edge <= max_edge, fixed fps, rotation baked in, no audio). Spark crops come from
the original file at full resolution at the same timestamps. proxy frame i is at t = i / proxy_fps.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int  # displayed width after rotation
    height: int  # displayed height after rotation
    fps: float
    duration_s: float
    nb_frames: int
    rotation: int  # degrees from the display matrix or rotate tag, 0 if none
    codec: str
    has_audio: bool
    creation_time: datetime | None  # from container tags, if present


def probe(path: Path) -> VideoInfo:
    """ffprobe the file. width/height are the displayed size (swap for 90/270 rotation)."""
    raise NotImplementedError


def make_proxy(src: Path, dst: Path, *, fps: float = 10.0, max_edge: int = 960) -> VideoInfo:
    """Write an H.264 MP4 proxy with rotation baked in, no audio, yuv420p, even dimensions, constant fps.
    Returns probe(dst). Idempotent: skips the encode when dst exists and is newer than src."""
    raise NotImplementedError


def read_frame_at(path: Path, t_seconds: float) -> np.ndarray:
    """One RGB uint8 frame (H, W, 3) from the ORIGINAL file at time t, rotation applied, full resolution."""
    raise NotImplementedError


def iter_frames(path: Path, *, fps: float | None = None, max_edge: int | None = None) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (index, t_seconds, rgb_frame) decoded through ffmpeg. fps=None keeps the source rate."""
    raise NotImplementedError


def sharpness(rgb_or_gray: np.ndarray) -> float:
    """Variance of the Laplacian on the grayscale image. Higher is sharper."""
    raise NotImplementedError


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of the file bytes, streamed."""
    raise NotImplementedError


def extract_wav(src: Path, dst: Path, *, rate: int = 24000) -> Path:
    """Mono 16-bit PCM WAV for Muse Voice: ffmpeg -vn -ac 1 -ar rate -c:a pcm_s16le -map_metadata -1.
    Returns dst. Raises if the source has no audio stream."""
    raise NotImplementedError
