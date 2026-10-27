"""ffmpeg and ffprobe helpers: proxies for SAM, full-resolution frames for Spark, clips for Stays Put, WAV for Voice.

Design (docs/ARCHITECTURE.md section 3 M1, docs/API_NOTES.md decision 2): SAM is billed per frame and tracks every
frame it is given, so it gets a small proxy (H.264 MP4, longest edge <= max_edge, constant fps, rotation baked in,
no audio, no metadata). Spark crops and Stays Put clips come from the original at full resolution at the same
timestamps.

Time convention: every timestamp here is seconds since the file start, the convention ffmpeg uses for -ss and for
the pts it writes. iPhone footage is variable frame rate, so a proxy frame maps to a source time through the pts
map (pts_map, proxy_frame_to_pts, pts_to_proxy_frame), never through a frame count.

Measured on 10/26 on synthetic CFR, jittered VFR and late-starting clips (tests/test_video.py keeps the
measurement): with the fps filter at round=up and start_time=0, proxy frame i shows the last source frame at or
before i / fps (a late first frame is repeated into the empty slots), so the pts map is the proxy's own timestamps
and read_frame_at, which returns the source frame on screen at t by the packet pts, gives back the very picture
SAM saw for a proxy frame's pts. cut_clip centres on that same frame. Raw frame pipes use -fps_mode passthrough so
a VFR source never has frames duplicated or dropped on the way out.

No function here calls a model API or writes a ledger row.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import cv2
import numpy as np
from PIL import Image

from . import contracts, paths

FFMPEG = os.environ.get("WALKAROUND_FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("WALKAROUND_FFPROBE", "ffprobe")

_QUIET = ["-nostdin", "-hide_banner", "-loglevel", "error"]
_RAW_OUT = ["-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
_HDR_TRANSFERS = ("smpte2084", "arib-std-b67")
_PROBE_CACHE: dict[tuple[str, int, int], VideoInfo] = {}
_FRAME_TIMES_CACHE: dict[tuple[str, int, int], list[float]] = {}
_CACHE_MAX = 256

Box4 = list[float]


class VideoError(RuntimeError):
    """ffmpeg or ffprobe failed, or the file is not a usable video."""


class NoAudioError(VideoError):
    """extract_wav on a file with no audio stream."""


class ClipOutOfRange(VideoError):
    """cut_clip asked for frames before the first or after the last source frame."""


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int  # displayed width after rotation
    height: int  # displayed height after rotation
    fps: float
    duration_s: float
    nb_frames: int
    rotation: int  # clockwise display rotation (0, 90, 180, 270), the legacy rotate-tag convention
    codec: str
    has_audio: bool
    creation_time: datetime | None  # from container tags, if present
    vfr: bool = False  # r_frame_rate and avg_frame_rate disagree; coarse, mild iPhone jitter is not flagged
    hdr: bool = False  # PQ or HLG transfer
    device: str | None = None  # com.apple.quicktime.model when present
    start_time: float = 0.0  # container start time; pts maps and seeks count from it


# ----------------------------------------------------------------------------------------------------------------
# Subprocess plumbing
# ----------------------------------------------------------------------------------------------------------------


def _tail(data: bytes, n: int = 2000) -> str:
    return data.decode("utf-8", "replace")[-n:].strip()


def _run(cmd: list[str]) -> bytes:
    """Run ffmpeg or ffprobe to completion and return stdout; VideoError carries the stderr tail."""
    try:
        done = subprocess.run(cmd, capture_output=True, check=False)
    except FileNotFoundError as exc:
        raise VideoError(f"{cmd[0]} is not on PATH") from exc
    if done.returncode != 0:
        raise VideoError(f"{cmd[0]} failed ({done.returncode}): {_tail(done.stderr)}")
    return done.stdout


def _ffprobe_json(path: Path, args: list[str]) -> dict[str, Any]:
    out = _run([FFPROBE, "-v", "error", "-print_format", "json", *args, str(path)])
    return json.loads(out or b"{}")


class _Decode:
    """ffmpeg decoding to a rawvideo pipe, read one frame at a time so a 4K clip never sits in memory whole."""

    def __init__(self, cmd: list[str]) -> None:
        self.err = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=self.err)

    def read(self, n: int) -> bytes:
        assert self.proc.stdout is not None
        return self.proc.stdout.read(n)

    def close(self, *, check: bool) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()
        assert self.proc.stdout is not None
        self.proc.stdout.close()
        self.err.seek(0)
        err = self.err.read()
        self.err.close()
        if check and self.proc.returncode != 0:
            raise VideoError(f"ffmpeg decode failed ({self.proc.returncode}): {_tail(err)}")


class _Encode:
    """ffmpeg encoding from a rawvideo pipe on stdin."""

    def __init__(self, cmd: list[str]) -> None:
        self.err = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.err)

    def _error(self) -> str:
        self.err.seek(0)
        text = _tail(self.err.read())
        self.err.close()
        return text

    def write(self, data: bytes) -> None:
        assert self.proc.stdin is not None
        try:
            self.proc.stdin.write(data)
        except BrokenPipeError as exc:
            self.proc.wait()
            raise VideoError(f"ffmpeg encode stopped early: {self._error()}") from exc

    def finish(self) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        self.proc.wait()
        if self.proc.returncode != 0:
            raise VideoError(f"ffmpeg encode failed ({self.proc.returncode}): {self._error()}")
        self.err.close()

    def abort(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()
        if not self.err.closed:
            self.err.close()


def _encoder_cmd(width: int, height: int, fps: float, dst: Path, *, preset: str, crf: int) -> list[str]:
    """libx264 from an rgb24 rawvideo pipe: yuv420p, constant fps, faststart, no metadata."""
    return [
        FFMPEG, "-y", *_QUIET,
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-framerate", f"{fps:.6f}", "-i", "-",
        "-an", "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-fps_mode", "cfr", "-movflags", "+faststart", "-map_metadata", "-1", str(dst),
    ]


def _tmp_for(dst: Path) -> Path:
    """A sibling the encoder writes; os.replace moves it onto dst so a half-written dst never exists."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    return dst.with_name(f".{dst.name}.part{dst.suffix}")


def _raw_frames(
    path: Path, width: int, height: int, *, filters: Sequence[str], ss: float | None = None, frames: int | None = None
) -> Iterator[tuple[int, np.ndarray]]:
    """(index, rgb) frames through an rgb24 pipe with -fps_mode passthrough: one output per decoded frame."""
    cmd = [FFMPEG, *_QUIET]
    if ss is not None:
        cmd += ["-ss", f"{ss:.6f}"]
    cmd += ["-i", str(path), "-map", "0:v:0"]
    if frames is not None:
        cmd += ["-frames:v", str(frames)]
    if filters:
        cmd += ["-vf", ",".join(filters)]
    cmd += _RAW_OUT
    dec = _Decode(cmd)
    n = width * height * 3
    index = 0
    try:
        while True:
            raw = dec.read(n)
            if len(raw) < n:
                if raw:
                    raise VideoError(f"{path}: short frame from ffmpeg ({len(raw)} of {n} bytes)")
                break
            yield index, np.frombuffer(raw, np.uint8).reshape(height, width, 3).copy()
            index += 1
    except BaseException:
        dec.close(check=False)
        raise
    dec.close(check=True)


# ----------------------------------------------------------------------------------------------------------------
# ffprobe
# ----------------------------------------------------------------------------------------------------------------


def _ratio(text: Any) -> float:
    """'24000/1001' -> 23.976; 0.0 when absent or zero."""
    if not text:
        return 0.0
    try:
        if "/" in str(text):
            num, den = str(text).split("/", 1)
            return float(num) / float(den) if float(den) else 0.0
        return float(text)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _float(text: Any) -> float | None:
    try:
        return float(text) if text not in (None, "", "N/A") else None
    except (TypeError, ValueError):
        return None


def _int(text: Any) -> int | None:
    try:
        return int(text) if text not in (None, "", "N/A") else None
    except (TypeError, ValueError):
        return None


def _parse_time(text: Any) -> datetime | None:
    """ffprobe creation_time ('2026-10-26T12:00:00.000000Z' or with an offset) -> aware datetime, else None."""
    if not text:
        return None
    value = str(text).strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def iso_z(moment: datetime) -> str:
    """contracts.Timestamp spelling: ISO 8601 UTC to the second with a Z."""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rotation(stream: dict[str, Any]) -> int:
    """Clockwise display rotation in degrees. The display matrix angle ffprobe reports is counterclockwise."""
    degrees: float | None = None
    for side in stream.get("side_data_list") or []:
        if side.get("rotation") is not None:
            degrees = -float(side["rotation"])
            break
    if degrees is None:
        tag = (stream.get("tags") or {}).get("rotate")
        degrees = _float(tag)
    if degrees is None:
        return 0
    return int(round(degrees / 90.0)) * 90 % 360


def probe(path: Path) -> VideoInfo:
    """ffprobe the file. width/height are the displayed size (swapped for 90/270 rotation)."""
    path = Path(path)
    if not path.is_file():
        raise VideoError(f"{path}: no such file")
    data = _ffprobe_json(path, ["-show_format", "-show_streams"])
    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise VideoError(f"{path}: no video stream")
    fmt = data.get("format") or {}
    rotation = _rotation(video)
    width, height = int(video["width"]), int(video["height"])
    if rotation in (90, 270):
        width, height = height, width
    r_fps, avg_fps = _ratio(video.get("r_frame_rate")), _ratio(video.get("avg_frame_rate"))
    fps = avg_fps or r_fps
    duration = _float(fmt.get("duration")) or _float(video.get("duration")) or 0.0
    nb_frames = _int(video.get("nb_frames")) or (int(round(duration * fps)) if fps else 0)
    tags = {**(video.get("tags") or {}), **(fmt.get("tags") or {})}
    return VideoInfo(
        path=path,
        width=width,
        height=height,
        fps=fps,
        duration_s=duration,
        nb_frames=nb_frames,
        rotation=rotation,
        codec=str(video.get("codec_name") or ""),
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
        creation_time=_parse_time(tags.get("creation_time")),
        vfr=bool(r_fps and avg_fps and abs(r_fps - avg_fps) > 0.001 * r_fps),
        hdr=str(video.get("color_transfer") or "") in _HDR_TRANSFERS,
        device=tags.get("com.apple.quicktime.model"),
        start_time=_float(fmt.get("start_time")) or 0.0,
    )


def _cache_key(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return (str(path.resolve()), stat.st_mtime_ns, stat.st_size)


def _probe_cached(path: Path) -> VideoInfo:
    """probe() keyed by path, mtime and size, for the many seeks into one original."""
    path = Path(path)
    key = _cache_key(path)
    info = _PROBE_CACHE.get(key)
    if info is None:
        if len(_PROBE_CACHE) > _CACHE_MAX:
            _PROBE_CACHE.clear()
        info = _PROBE_CACHE[key] = probe(path)
    return info


def ffprobe_info(
    info: VideoInfo, *, container: str, device: str | None = None, hdr: bool = False, vfr: bool = False
) -> contracts.FfprobeInfo:
    """The videos/<role>.json ffprobe block: stored and displayed sizes, creation_time or 'not present'."""
    stored_w, stored_h = (info.height, info.width) if info.rotation in (90, 270) else (info.width, info.height)
    return contracts.FfprobeInfo(
        codec=info.codec,
        container=container,
        width=stored_w,
        height=stored_h,
        rotation=info.rotation,
        display_width=info.width,
        display_height=info.height,
        fps_avg=info.fps,
        vfr=vfr or info.vfr,
        duration_s=info.duration_s,
        creation_time=iso_z(info.creation_time) if info.creation_time else "not present",
        device=device or info.device,
        hdr=hdr or info.hdr,
        has_audio=info.has_audio,
    )


# ----------------------------------------------------------------------------------------------------------------
# Proxy and pts map
# ----------------------------------------------------------------------------------------------------------------


def _even(value: float) -> int:
    return max(2, int(round(value)) // 2 * 2)


def _fit(width: int, height: int, max_edge: int | None) -> tuple[int, int]:
    """Even output size with the long edge capped at max_edge; never upscales."""
    scale = 1.0 if not max_edge else min(1.0, max_edge / max(width, height))
    return _even(width * scale), _even(height * scale)


def _fps_filter(fps: float) -> str:
    """Slot i of the output shows the last source frame at or before i / fps; slots before the first frame repeat it."""
    return f"fps={fps:g}:round=up:start_time=0"


def _proxy_current(src: Path, dst: Path, *, fps: float, size: tuple[int, int]) -> bool:
    """True when dst is newer than src and already has the requested fps and size."""
    if not dst.is_file() or dst.stat().st_mtime <= src.stat().st_mtime:
        return False
    try:
        info = probe(dst)
    except VideoError:
        return False
    return abs(info.fps - fps) < 0.01 and (info.width, info.height) == size and not info.has_audio


def make_proxy(
    src: Path, dst: Path, *, fps: float = 15.0, max_edge: int = 960, preset: str = "medium", crf: int = 20
) -> VideoInfo:
    """Write an H.264 MP4 proxy with rotation baked in, no audio, yuv420p, even dimensions, constant fps.
    Returns probe(dst). Idempotent: skips the encode when dst exists, is newer than src and matches fps and size."""
    src, dst = Path(src), Path(dst)
    info = probe(src)
    width, height = _fit(info.width, info.height, max_edge)
    if _proxy_current(src, dst, fps=fps, size=(width, height)):
        return probe(dst)
    tmp = _tmp_for(dst)
    filters = f"{_fps_filter(fps)},scale={width}:{height}:flags=lanczos,format=yuv420p"
    _run([
        FFMPEG, "-y", *_QUIET, "-i", str(src), "-map", "0:v:0", "-an", "-sn", "-dn",
        "-vf", filters, "-fps_mode", "cfr", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-map_metadata", "-1", str(tmp),
    ])
    os.replace(tmp, dst)
    return probe(dst)


def _times(path: Path, *, section: str) -> list[float]:
    """pts_time of the video stream in presentation order, seconds since the file start."""
    data = _ffprobe_json(path, ["-select_streams", "v:0", "-show_entries", f"{section}=pts_time:format=start_time"])
    start = _float((data.get("format") or {}).get("start_time")) or 0.0
    out: list[float] = []
    for row in data.get(section + "s") or []:
        t = _float(row.get("pts_time"))
        if t is None:
            t = _float(row.get("best_effort_timestamp_time"))
        if t is not None and t - start >= -1e-6:  # frames before the file start never come out of a decode
            out.append(round(t - start, 6))
    if section == "packet":
        out.sort()
    return out


def pts_map(proxy: Path) -> list[float]:
    """ffprobe -show_frames pts_time per proxy frame; index = frame."""
    return _times(Path(proxy), section="frame")


def _frame_times(path: Path) -> list[float]:
    """Source frame times from packet pts (sorted = presentation order), no decode; equal to the frame pts."""
    return _times(Path(path), section="packet")


def _frame_times_cached(path: Path) -> list[float]:
    """_frame_times keyed like the probe cache: one demux per original, however many seeks follow."""
    path = Path(path)
    key = _cache_key(path)
    times = _FRAME_TIMES_CACHE.get(key)
    if times is None:
        if len(_FRAME_TIMES_CACHE) > _CACHE_MAX:
            _FRAME_TIMES_CACHE.clear()
        times = _FRAME_TIMES_CACHE[key] = _frame_times(path)
    return times


def _frame_interval(times: Sequence[float], fps: float) -> float:
    """Typical seconds between frames: the median gap, or the nominal interval for a one-frame file."""
    if len(times) >= 2:
        return float(np.median(np.diff(np.asarray(times, dtype=np.float64))))
    return 1.0 / fps if fps else 0.1


def _seek_start(times: Sequence[float], k: int) -> float | None:
    """The -ss that makes source frame k the first frame out: halfway from the frame before it (None for frame 0).
    ffmpeg's accurate seek discards every frame before -ss, so the midpoint lands on k whatever the jitter."""
    return (times[k - 1] + times[k]) / 2 if k > 0 else None


def _on_screen_index(times: Sequence[float], t: float, *, eps: float = 1e-5) -> int:
    """The source frame on screen at t: the last frame whose pts is at or before t (frame 0 before the first).
    The same rule the proxy's fps filter applies, so a proxy pts names the frame the proxy showed."""
    i = bisect.bisect_right(times, t + eps)
    return max(0, i - 1)


def pts_to_proxy_frame(pts: list[float], t: float) -> int:
    """The proxy frame whose pts is nearest to t (clamped to the first and last frame)."""
    if not pts:
        raise ValueError("empty pts map")
    i = bisect.bisect_left(pts, t)
    if i <= 0:
        return 0
    if i >= len(pts):
        return len(pts) - 1
    return i if (pts[i] - t) < (t - pts[i - 1]) else i - 1


def proxy_frame_to_pts(pts: list[float], f: int) -> float:
    """The source time (seconds since the file start) of proxy frame f."""
    if not pts:
        raise ValueError("empty pts map")
    if f < 0 or f >= len(pts):
        raise IndexError(f"proxy frame {f} outside 0..{len(pts) - 1}")
    return pts[f]


def _rel(run_id: str, path: Path) -> str:
    try:
        return paths.rel(run_id, path)
    except ValueError:
        return str(path)


def _record_path(path: Path) -> str:
    """Run-relative when the file sits under runs/<run_id>/ (what records store), else the path as given."""
    try:
        rel = Path(path).resolve().relative_to(Path(paths.RUNS_DIR).resolve())
    except ValueError:
        return str(path)
    return str(Path(*rel.parts[1:])) if len(rel.parts) > 1 else str(path)


def proxy_record(run_id: str, role: str, proxy: Path) -> contracts.ProxyRecord:
    """Probe the proxy, write paths.pts_map_path, hash it and fill the record with run-relative paths."""
    proxy = Path(proxy)
    info = probe(proxy)
    pts = pts_map(proxy)
    pts_path = paths.pts_map_path(run_id, role)
    contracts.write_json(pts_path, pts)
    return contracts.ProxyRecord(
        path=_rel(run_id, proxy),
        fps=info.fps,
        width=info.width,
        height=info.height,
        frames=len(pts),
        pts_map_path=_rel(run_id, pts_path),
        sha256=sha256_file(proxy),
    )


# ----------------------------------------------------------------------------------------------------------------
# Frames
# ----------------------------------------------------------------------------------------------------------------


def read_frame_at(path: Path, t_seconds: float) -> np.ndarray:
    """One RGB uint8 frame (H, W, 3) from the ORIGINAL file: the source frame on screen at t (the last frame at or
    before t by its own pts, never a frame count), at full resolution with the rotation applied. For a proxy pts
    this is the picture the proxy frame showed. Raises VideoError more than one frame past either end."""
    path = Path(path)
    info = _probe_cached(path)
    times = _frame_times_cached(path)
    if not times:
        raise VideoError(f"{path}: no video frames")
    slack = _frame_interval(times, info.fps)
    if t_seconds < -slack or t_seconds > times[-1] + slack:
        raise VideoError(f"{path}: no frame at {t_seconds:.3f} s (frames run {times[0]:.3f} to {times[-1]:.3f} s)")
    k = _on_screen_index(times, t_seconds)
    for _, rgb in _raw_frames(path, info.width, info.height, filters=(), ss=_seek_start(times, k), frames=1):
        return rgb
    raise VideoError(f"{path}: ffmpeg gave no frame for source frame {k} at {times[k]:.3f} s")


def iter_frames(
    path: Path, *, fps: float | None = None, max_edge: int | None = None
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (index, t_seconds, rgb_frame) decoded through ffmpeg. fps=None keeps the source rate and reads each
    frame's own pts; with fps the frames are resampled exactly as make_proxy does and t = index / fps."""
    path = Path(path)
    info = probe(path)
    width, height = _fit(info.width, info.height, max_edge)
    filters: list[str] = []
    if fps:
        filters.append(_fps_filter(fps))
    if (width, height) != (info.width, info.height):
        filters.append(f"scale={width}:{height}:flags=lanczos")
    times = None if fps else _frame_times_cached(path)
    for index, rgb in _raw_frames(path, width, height, filters=filters):
        if fps:
            t = index / fps
        elif times is not None and index < len(times):
            t = times[index]
        else:
            t = index / info.fps if info.fps else float(index)
        yield index, t, rgb


def sharpness(rgb_or_gray: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Variance of the Laplacian on the grayscale image (inside mask when given). Higher is sharper."""
    gray = rgb_or_gray if rgb_or_gray.ndim == 2 else cv2.cvtColor(rgb_or_gray, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    values = lap[mask.astype(bool)] if mask is not None else lap
    return float(values.var()) if values.size > 1 else 0.0


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of the file bytes, streamed."""
    with open(path, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def extract_wav(src: Path, dst: Path, *, rate: int = 24000) -> Path:
    """Mono 16-bit PCM WAV for Muse Voice: ffmpeg -vn -ac 1 -ar rate -c:a pcm_s16le -map_metadata -1, plus
    -bitexact so no LIST/INFO chunk is written. Returns dst. Raises NoAudioError without an audio stream."""
    src, dst = Path(src), Path(dst)
    if not probe(src).has_audio:
        raise NoAudioError(f"{src}: no audio stream")
    tmp = _tmp_for(dst)
    _run([
        FFMPEG, "-y", *_QUIET, "-i", str(src), "-vn", "-sn", "-dn", "-map", "0:a:0",
        "-ac", "1", "-ar", str(rate), "-c:a", "pcm_s16le", "-map_metadata", "-1", "-bitexact", "-f", "wav", str(tmp),
    ])
    os.replace(tmp, dst)
    return dst


# ----------------------------------------------------------------------------------------------------------------
# Pixelation (privacy.pixelate when that lane has landed, the same block-mean rule here until then)
# ----------------------------------------------------------------------------------------------------------------


def _pixelate(
    rgb: np.ndarray, regions: Sequence[Sequence[float]], *, block_px: int, dilate_frac: float = 0.0, dilate_px: int = 0
) -> np.ndarray:
    """Each region, grown by dilate_frac of its size plus dilate_px per side, replaced by block means."""
    out = rgb.copy()
    height, width = out.shape[:2]
    block = max(1, int(block_px))
    for x1, y1, x2, y2 in regions:
        grow_x = (x2 - x1) * dilate_frac / 2 + dilate_px
        grow_y = (y2 - y1) * dilate_frac / 2 + dilate_px
        xa, ya = max(0, int(math.floor(x1 - grow_x))), max(0, int(math.floor(y1 - grow_y)))
        xb, yb = min(width, int(math.ceil(x2 + grow_x))), min(height, int(math.ceil(y2 + grow_y)))
        if xb <= xa or yb <= ya:
            continue
        region = out[ya:yb, xa:xb]
        rh, rw = region.shape[:2]
        pad_h, pad_w = -rh % block, -rw % block
        padded = np.pad(region, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge").astype(np.float32)
        means = padded.reshape((rh + pad_h) // block, block, (rw + pad_w) // block, block, -1).mean(axis=(1, 3))
        filled = np.repeat(np.repeat(means, block, axis=0), block, axis=1)[:rh, :rw]
        out[ya:yb, xa:xb] = np.clip(np.round(filled), 0, 255).astype(np.uint8)
    return out


def _pixelate_fn() -> Callable[..., np.ndarray]:
    try:
        from . import privacy  # noqa: PLC0415
    except ImportError:
        return _pixelate
    return getattr(privacy, "pixelate", _pixelate)


# ----------------------------------------------------------------------------------------------------------------
# Stays Put clips and the display proxy
# ----------------------------------------------------------------------------------------------------------------


def _cut_window(frame: np.ndarray, window: Sequence[float], size: int) -> np.ndarray:
    """Resample the source window onto size x size: clip (u, v) -> source (x1 + u * sw / size, y1 + v * sh / size).
    Pixels outside the frame come back black, so the geometry never shifts near an edge."""
    x1, y1, x2, y2 = window
    sw, sh = x2 - x1, y2 - y1
    if sw <= 0 or sh <= 0:
        raise ValueError(f"empty window {list(window)}")
    sx, sy = sw / size, sh / size
    matrix = np.array([[sx, 0.0, x1 + 0.5 * sx - 0.5], [0.0, sy, y1 + 0.5 * sy - 0.5]], dtype=np.float64)
    interp = cv2.INTER_LANCZOS4 if sx < 1.0 else cv2.INTER_LINEAR
    return cv2.warpAffine(
        frame, matrix, (size, size), flags=interp | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )


def _box_to_window(box: Sequence[float], window: Sequence[float], size: int) -> list[float]:
    x1, y1, x2, y2 = window
    fx, fy = size / (x2 - x1), size / (y2 - y1)
    return [(box[0] - x1) * fx, (box[1] - y1) * fy, (box[2] - x1) * fx, (box[3] - y1) * fy]


def clip_span(src: Path, *, pts_center: float, frames: int) -> tuple[int, list[float]]:
    """The source frames a Stays Put clip covers: (f0_src, the pts of each of the `frames` frames centred on the
    frame on screen at pts_center, the frame read_frame_at returns there). The caller builds one window per listed
    pts, then cut_clip cuts exactly these frames. Raises ClipOutOfRange when the span would run past either end,
    since a shifted clip would misalign every window."""
    src = Path(src)
    if frames < 1:
        raise ValueError("a clip needs at least one frame")
    times = _frame_times_cached(src)
    if not times:
        raise VideoError(f"{src}: no video frames")
    f0 = _on_screen_index(times, pts_center) - frames // 2
    if f0 < 0 or f0 + frames > len(times):
        raise ClipOutOfRange(
            f"{src.name}: {frames} frames around {pts_center:.3f} s would use source frames {f0} to "
            f"{f0 + frames - 1} of 0..{len(times) - 1}"
        )
    return f0, list(times[f0 : f0 + frames])


def cut_clip(
    src: Path,
    dst: Path,
    *,
    pts_center: float,
    frames: int,
    windows_src: list[Box4],
    size: int = 1008,
    crf: int = 18,
    blur_boxes: list[list[Box4]] | None = None,
    block_px: int = 20,
) -> contracts.ClipRecord:
    """The panel-locked Stays Put clip: `frames` source frames centred on pts_center (the span of clip_span), one
    square source window per frame cut and resampled to size x size, optional per-frame pixelation (boxes in source
    pixels, block_px at source scale), libx264 at the source frame rate; the encoded file is probed and must be
    size x size with `frames` frames. Raises ClipOutOfRange when the span would run past either end."""
    src, dst = Path(src), Path(dst)
    if frames < 1 or len(windows_src) != frames:
        raise ValueError(f"cut_clip needs one window per frame: {len(windows_src)} windows for {frames} frames")
    if blur_boxes is not None and len(blur_boxes) != frames:
        raise ValueError(f"cut_clip needs one blur list per frame: {len(blur_boxes)} for {frames} frames")
    info = _probe_cached(src)
    f0, _ = clip_span(src, pts_center=pts_center, frames=frames)
    ss = _seek_start(_frame_times_cached(src), f0)
    pixelate = _pixelate_fn() if blur_boxes else None
    tmp = _tmp_for(dst)
    enc = _Encode(_encoder_cmd(size, size, info.fps or 30.0, tmp, preset="medium", crf=crf))
    written = 0
    try:
        for i, frame in _raw_frames(src, info.width, info.height, filters=(), ss=ss, frames=frames):
            window = [float(v) for v in windows_src[i]]
            clip = _cut_window(frame, window, size)
            if pixelate is not None and blur_boxes is not None and blur_boxes[i]:
                regions = [_box_to_window(b, window, size) for b in blur_boxes[i]]
                clip = pixelate(clip, regions, block_px=max(1, round(block_px * size / (window[2] - window[0]))))
            enc.write(np.ascontiguousarray(clip).tobytes())
            written += 1
        if written != frames:
            raise VideoError(f"{src}: decoded {written} of {frames} frames from source frame {f0}")
        enc.finish()
    except BaseException:
        enc.abort()
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, dst)
    out = probe(dst)
    if (out.width, out.height) != (size, size) or out.nb_frames != frames:
        raise VideoError(f"{dst}: encoded {out.width}x{out.height} x {out.nb_frames}, wanted {size}x{size} x {frames}")
    return contracts.ClipRecord(
        path=_record_path(dst),
        sha256=sha256_file(dst),
        f0_src=f0,
        frames=frames,
        window_px=int(round(windows_src[0][2] - windows_src[0][0])),
        windows_src=[[float(v) for v in w] for w in windows_src],
        w=size,
        h=size,
    )


def render_blurred_proxy(
    proxy: Path,
    dst: Path,
    boxes_by_frame: dict[int, list[Box4]],
    *,
    block_px: int,
    dilate_frac: float,
    dilate_px: int,
    preset: str = "medium",
    crf: int = 20,
) -> VideoInfo:
    """Decode the proxy, pixelate the boxes of each frame (proxy pixels), re-encode: the display proxy."""
    proxy, dst = Path(proxy), Path(dst)
    info = probe(proxy)
    pixelate = _pixelate_fn()
    tmp = _tmp_for(dst)
    enc = _Encode(_encoder_cmd(info.width, info.height, info.fps or 15.0, tmp, preset=preset, crf=crf))
    try:
        for index, rgb in _raw_frames(proxy, info.width, info.height, filters=()):
            boxes = boxes_by_frame.get(index)
            if boxes:
                rgb = pixelate(rgb, boxes, block_px=block_px, dilate_frac=dilate_frac, dilate_px=dilate_px)
            enc.write(np.ascontiguousarray(rgb).tobytes())
        enc.finish()
    except BaseException:
        enc.abort()
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, dst)
    return probe(dst)


# ----------------------------------------------------------------------------------------------------------------
# Stills and crops
# ----------------------------------------------------------------------------------------------------------------


def write_jpeg(rgb: np.ndarray, dst: Path, *, quality: int = 90) -> str:
    """Write an RGB uint8 array as a JPEG with no metadata; returns its sha256."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(rgb), "RGB").save(dst, "JPEG", quality=quality)
    return sha256_file(dst)


def write_png_mask(mask: np.ndarray, dst: Path) -> str:
    """Write a boolean (H, W) mask as a one-bit PNG; returns its sha256."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(mask).astype(bool)).save(dst, "PNG")
    return sha256_file(dst)


def cut_crop(rgb: np.ndarray, crop: contracts.Crop) -> np.ndarray:
    """Cut the source box [dx, dy, dx + w / scale, dy + h / scale] and resample it to (w, h): crop pixel (u, v)
    reads source (dx + u / scale, dy + v / scale). Outside the frame is black. A reduction (scale < 1) goes
    through an area average first so thin marks are not aliased away; an enlargement is bicubic."""
    src = np.ascontiguousarray(rgb)
    sx = sy = 1.0 / crop.scale  # source pixels per crop pixel on each axis
    if crop.scale < 1.0:
        h, w = src.shape[:2]
        small_w, small_h = max(1, int(round(w * crop.scale))), max(1, int(round(h * crop.scale)))
        src = cv2.resize(src, (small_w, small_h), interpolation=cv2.INTER_AREA)
        sx, sy = (small_w / w) / crop.scale, (small_h / h) / crop.scale  # reduced pixels per crop pixel
        dx, dy = crop.dx * small_w / w, crop.dy * small_h / h
        interp = cv2.INTER_LINEAR
    else:
        dx, dy = crop.dx, crop.dy
        interp = cv2.INTER_CUBIC if crop.scale > 1.0 else cv2.INTER_LINEAR
    matrix = np.array([[sx, 0.0, dx + 0.5 * sx - 0.5], [0.0, sy, dy + 0.5 * sy - 0.5]], dtype=np.float64)
    return cv2.warpAffine(
        src, matrix, (crop.w, crop.h), flags=interp | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )
