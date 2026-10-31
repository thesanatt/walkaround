"""ffmpeg and ffprobe helpers for the media lane (ARCHITECTURE section 3 M1); no model calls, no ledger rows."""

# Times are seconds since the file start (ffmpeg's -ss convention), compared as whole microseconds. Phone footage is
# variable frame rate, so a proxy frame maps to a source time through its pts, never a frame count. Proxy slot i shows
# the last source frame at or before i / fps, and read_frame_at(t) returns the last source frame at or before t, both
# on the 6-decimal times that pts maps and frame indexes print, so a proxy pts gives back the very picture SAM saw.
# Source frame times come from packet pts without a full decode, minus edit-list discards and the leading frames of an
# open-GOP start that the decoder never outputs.
# x264 output depends on its thread count, so proxy and clip sha256s describe the bytes made on this machine; nothing
# re-encodes to check them (rebuild's frozen inputs are the lanes, ARCHITECTURE section 7 item 3).
# TODO(D1): HLG and Dolby Vision (the iPhone default) are decoded to rgb24 with no tone map, so Spark gets BT.2020 HLG
# values read as sRGB. D1 must run one real iPhone HDR take of a marked panel through panels and Spark next to an SDR
# take of the same panel; if recall drops, tone-map before write_jpeg (this ffmpeg 9 build's scale filter has
# out_transfer, out_primaries and intent; there is no zscale). The shot list asks for HDR off (section 11).

from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
import subprocess
import tempfile
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import cv2
import numpy as np
from PIL import Image

from . import contracts, paths

FFMPEG = os.environ.get("WALKAROUND_FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("WALKAROUND_FFPROBE", "ffprobe")
VIDEO_STREAM = "V:0"  # the first video stream that is not cover art, for ffmpeg and ffprobe alike

Box4 = list[float]

_QUIET = ["-nostdin", "-hide_banner", "-loglevel", "error"]
_HDR_TRANSFERS = ("smpte2084", "arib-std-b67")
# Capture start first: on an iPhone, creation_time is written when the take ends.
_CAPTURE_TAGS = ("com.apple.quicktime.creationdate", "creation_time")
_DEVICE_TAGS = ("com.apple.quicktime.model", "com.android.model")
_EARLIEST_CAPTURE_YEAR = 2000  # an unset QuickTime clock reads 1904 or 1970
_STANDARD_RATES = tuple(Fraction(n, d) for n, d in ((24000, 1001), (30000, 1001), (60000, 1001))) + tuple(
    Fraction(n) for n in (5, 10, 12, 15, 24, 25, 30, 50, 60, 120)
)
_BT709 = ("bt709", "bt709", "bt709", "bt709")  # RGB-to-YUV matrix, primaries, transfer, colour space
_CROP_MARGIN_PX = 8  # Lanczos reads 4 px past a window edge; the decode crop keeps twice that
_SEEK_SAVES_FRAMES = 12  # estimate: decoding this many frames costs about one more ffmpeg start and seek
_CACHE_MAX = 256
_US = 1_000_000
_MICRO = Fraction(1, _US)


class VideoError(RuntimeError):
    """ffmpeg or ffprobe failed, or the file is not a usable video."""


class NoAudioError(VideoError):
    """extract_wav on a file with no audio stream."""


class ClipOutOfRange(VideoError):
    """A clip span runs before the first or past the last source frame."""


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int  # displayed width after rotation
    height: int  # displayed height after rotation
    fps: float  # avg_frame_rate
    duration_s: float
    nb_frames: int  # the container's count; frame_times is exact when an edit list hides frames
    rotation: int  # clockwise display rotation: 0, 90, 180 or 270
    codec: str
    has_audio: bool
    creation_time: datetime | None  # capture time from the container tags, if present
    vfr: bool = False  # r_frame_rate and avg_frame_rate disagree; coarse, mild jitter is not flagged
    hdr: bool = False  # PQ, HLG or Dolby Vision; nothing here tone-maps it
    device: str | None = None
    start_time: float = 0.0  # container start; every time here counts from it
    color_space: str = ""  # ffprobe's color_space, color_transfer and color_primaries, "" when untagged
    color_transfer: str = ""
    color_primaries: str = ""


@dataclass(frozen=True)
class _FrameIndex:
    times: list[float]  # presentation order, seconds since the file start (whole microseconds)
    us: list[int]  # the same times as integers; every comparison uses these
    key_at: list[int]  # indices into times of the keyframes
    end_us: int  # where the last frame stops showing (pts plus duration); the fps filter fills slots up to it


@dataclass(frozen=True)
class _Clock:
    """Stream ticks to whole microseconds since the file start, rounded where ffmpeg rounds."""

    time_base: Fraction
    offset: int  # ticks; ffmpeg shifts every stream by minus the container start, rounded to the stream's ticks

    def us(self, ticks: int) -> int:
        return _rescale(ticks + self.offset, self.time_base, _MICRO)

    def span_us(self, ticks: int) -> int:
        return _rescale(ticks, self.time_base, _MICRO)


_PROBE_CACHE: dict[tuple[str, int, int], VideoInfo] = {}
_INDEX_CACHE: dict[tuple[str, int, int], _FrameIndex] = {}


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
        raise VideoError(f"{Path(cmd[0]).name} failed ({done.returncode}): {_tail(done.stderr)}")
    return done.stdout


def _ffprobe_json(path: Path, args: list[str]) -> dict[str, Any]:
    out = _run([FFPROBE, "-v", "error", "-print_format", "json", *args, str(path)])
    return json.loads(out.decode("utf-8", "replace") or "{}")


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
        if self.proc.stdin is not None and not self.proc.stdin.closed:
            try:
                self.proc.stdin.close()
            except BrokenPipeError:
                pass
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()
        if not self.err.closed:
            self.err.close()


def _rate_text(fps: float) -> str:
    """A frame rate as ffmpeg reads it: the exact fraction for a standard rate (30000/1001), else 6 decimals."""
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"bad frame rate {fps}")
    for rate in _STANDARD_RATES:
        if abs(fps - float(rate)) <= 1e-4 * float(rate):
            return f"{rate.numerator}/{rate.denominator}"
    return f"{fps:.6f}"


def _color_args(like: VideoInfo | None) -> tuple[str, str, str, str]:
    """(matrix, primaries, transfer, colour space) of an encode from RGB: an HDR source keeps its own, the rest BT.709."""
    if like is not None and like.color_transfer in _HDR_TRANSFERS and like.color_space.startswith("bt2020"):
        return "bt2020", like.color_primaries or "bt2020", like.color_transfer, like.color_space
    return _BT709


def _encoder_cmd(
    width: int, height: int, fps: float, dst: Path, *, preset: str, crf: int, like: VideoInfo | None = None
) -> list[str]:
    """libx264 from an rgb24 pipe: yuv420p in the colours of `like`, constant fps, faststart, no audio, no metadata."""
    matrix, primaries, transfer, space = _color_args(like)
    # the encoder takes its colour tags from the frames (ffmpeg 7+ ignores -color_trc there), so setparams sets them
    colors = f"scale=out_color_matrix={matrix}:out_range=tv,format=yuv420p,setparams=range=tv:color_primaries={primaries}"
    return [
        FFMPEG, "-y", *_QUIET,
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-framerate", _rate_text(fps), "-i", "-",
        "-an", "-vf", f"{colors}:color_trc={transfer}:colorspace={space}", "-c:v", "libx264",
        "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-fps_mode", "cfr", "-movflags", "+faststart", "-map_metadata", "-1", str(dst),
    ]


def _tmp_for(dst: Path) -> Path:
    """A sibling the encoder writes; os.replace moves it onto dst so a half-written dst never exists."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    return dst.with_name(f".{dst.name}.part{dst.suffix}")


def _raw_frames(
    path: Path, width: int, height: int, *, filters: Sequence[str] = (), ss: float | None = None, frames: int | None = None
) -> Iterator[tuple[int, np.ndarray]]:
    """(index, rgb) per decoded frame through an rgb24 pipe; -fps_mode passthrough never duplicates or drops one."""
    cmd = [FFMPEG, *_QUIET]
    if ss is not None:
        cmd += ["-ss", f"{ss:.6f}"]
    cmd += ["-i", str(path), "-map", f"0:{VIDEO_STREAM}"]
    if frames is not None:
        cmd += ["-frames:v", str(frames)]
    if filters:
        cmd += ["-vf", ",".join(filters)]
    cmd += ["-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
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
    """A container date ('...Z', '+02:00', '-0400', or two joined by ';') -> aware datetime; None if absent or unset."""
    for part in str(text or "").split(";"):
        value = part.strip()
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            continue
        parsed = parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        if parsed.year >= _EARLIEST_CAPTURE_YEAR:
            return parsed
    return None


def iso_z(moment: datetime) -> str:
    """contracts.Timestamp spelling: ISO 8601 UTC to the second with a Z (a naive datetime is taken as UTC)."""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _video_stream(streams: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The stream VIDEO_STREAM selects: the first video stream that is not an attached picture."""
    for stream in streams:
        if stream.get("codec_type") == "video" and not _int((stream.get("disposition") or {}).get("attached_pic")):
            return stream
    return None


def _rotation(stream: dict[str, Any]) -> int:
    """Clockwise display rotation in degrees; the display matrix angle ffprobe prints is counterclockwise."""
    degrees: float | None = None
    for side in stream.get("side_data_list") or []:
        if side.get("rotation") is not None:
            degrees = -float(side["rotation"])
            break
    if degrees is None:
        degrees = _float((stream.get("tags") or {}).get("rotate"))
    if degrees is None:
        return 0
    return int(round(degrees / 90.0)) * 90 % 360


def _hdr(stream: dict[str, Any]) -> bool:
    if str(stream.get("color_transfer") or "") in _HDR_TRANSFERS:
        return True
    return any("dovi" in str(side.get("side_data_type") or "").lower() for side in stream.get("side_data_list") or [])


def _first_tag(tags: dict[str, Any], keys: Sequence[str]) -> Any:
    return next((tags[k] for k in keys if tags.get(k)), None)


def _capture_time(tags: dict[str, Any]) -> datetime | None:
    for key in _CAPTURE_TAGS:
        parsed = _parse_time(tags.get(key))
        if parsed is not None:
            return parsed
    return None


def probe(path: Path) -> VideoInfo:
    """ffprobe the file; width and height are the displayed size after rotation."""
    path = Path(path)
    if not path.is_file():
        raise VideoError(f"{path}: no such file")
    data = _ffprobe_json(path, ["-show_format", "-show_streams"])
    streams = data.get("streams") or []
    video = _video_stream(streams)
    if video is None or not _int(video.get("width")) or not _int(video.get("height")):
        raise VideoError(f"{path}: no video stream")
    fmt = data.get("format") or {}
    rotation = _rotation(video)
    width, height = int(video["width"]), int(video["height"])
    if rotation in (90, 270):
        width, height = height, width
    r_fps, avg_fps = _ratio(video.get("r_frame_rate")), _ratio(video.get("avg_frame_rate"))
    fps = avg_fps or r_fps
    duration = _float(fmt.get("duration")) or _float(video.get("duration")) or 0.0
    tags = {str(k).lower(): v for k, v in {**(video.get("tags") or {}), **(fmt.get("tags") or {})}.items()}
    return VideoInfo(
        path=path,
        width=width,
        height=height,
        fps=fps,
        duration_s=duration,
        nb_frames=_int(video.get("nb_frames")) or (int(round(duration * fps)) if fps else 0),
        rotation=rotation,
        codec=str(video.get("codec_name") or ""),
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
        creation_time=_capture_time(tags),
        vfr=bool(r_fps and avg_fps and abs(r_fps - avg_fps) > 0.001 * r_fps),
        hdr=_hdr(video),
        device=_first_tag(tags, _DEVICE_TAGS),
        start_time=_float(fmt.get("start_time")) or 0.0,
        color_space=str(video.get("color_space") or ""),
        color_transfer=str(video.get("color_transfer") or ""),
        color_primaries=str(video.get("color_primaries") or ""),
    )


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


def _rescale(ticks: int, src: Fraction, dst: Fraction) -> int:
    """av_rescale_q: ticks of src seconds as ticks of dst seconds, nearest with halves away from zero."""
    exact = ticks * src / dst
    rounded = math.floor(abs(exact) + Fraction(1, 2))
    return rounded if exact >= 0 else -rounded


def _clock(path: Path, data: dict[str, Any]) -> _Clock:
    """The _Clock of the stream ffprobe printed with stream=time_base:format=start_time."""
    streams = data.get("streams") or [{}]
    try:
        time_base = Fraction(str(streams[0].get("time_base")))
    except (ValueError, ZeroDivisionError) as exc:
        raise VideoError(f"{path}: no usable time base") from exc
    if time_base <= 0:
        raise VideoError(f"{path}: time base {time_base}")
    try:  # ffprobe prints the container start (whole microseconds inside ffmpeg) to 6 places, so this is exact
        start = int((Decimal(str((data.get("format") or {}).get("start_time"))) * _US).to_integral_value())
    except InvalidOperation:
        start = 0  # N/A: ffmpeg applies no offset either
    return _Clock(time_base, _rescale(-start, _MICRO, time_base))


def _ticks(entry: dict[str, Any], *keys: str) -> int | None:
    return next((v for v in (_int(entry.get(k)) for k in keys) if v is not None), None)


def _frame_index(path: Path) -> _FrameIndex:
    """Source frames from packets in presentation order and whole microseconds, as the proxy's fps filter sees them."""
    entries = "packet=pts,dts,duration,flags:stream=time_base:format=start_time"
    data = _ffprobe_json(path, ["-select_streams", VIDEO_STREAM, "-show_entries", entries])
    clock = _clock(path, data)
    rows: list[tuple[int, int, bool, int]] = []  # (us, decode order, keyframe, duration us) per frame a decode shows
    first_key: tuple[int, int] | None = None  # (decode order, us) of the first keyframe
    for order, packet in enumerate(data.get("packets") or []):
        ticks = _ticks(packet, "pts", "dts")
        if ticks is None:
            continue
        flags = str(packet.get("flags") or "")
        if "K" in flags and first_key is None:
            first_key = (order, clock.us(ticks))
        if "D" not in flags:  # an edit-list discard is decoded but never shown
            rows.append((clock.us(ticks), order, "K" in flags, clock.span_us(_int(packet.get("duration")) or 0)))
    rows = sorted(_shown_leaders(path, rows, first_key, clock))
    us = [row[0] for row in rows]
    end = max((row[0] + row[3] for row in rows), default=0)
    return _FrameIndex([u / _US for u in us], us, [i for i, row in enumerate(rows) if row[2]], end)


def _shown_leaders(
    path: Path, rows: list[tuple[int, int, bool, int]], first_key: tuple[int, int] | None, clock: _Clock
) -> list[tuple[int, int, bool, int]]:
    """rows without the frames ahead of the first keyframe that a decode from the file start never outputs."""
    # A stream copy cut from mid-GOP can start on an open-GOP keyframe (HEVC CRA) whose leading frames reference the GOP
    # that was cut away, and the decoder drops them. Which ones survive only a decode can say; one GOP is enough.
    if first_key is None:
        return rows
    leaders = [row for row in rows if row[1] < first_key[0] or row[0] < first_key[1]]
    if not leaders:
        return rows
    count = max(row[1] for row in leaders) + 1  # packets in decode order; -read_intervals counts the selected stream only
    args = ["-select_streams", VIDEO_STREAM, "-read_intervals", f"%+#{count}", "-show_entries", "frame=pts,best_effort_timestamp"]
    frames = _ffprobe_json(path, args).get("frames") or []
    shown = {clock.us(t) for t in (_ticks(f, "pts", "best_effort_timestamp") for f in frames) if t is not None}
    dropped = {row[1] for row in leaders if row[0] not in shown}
    return [row for row in rows if row[1] not in dropped]


def _cache_key(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return (str(path.resolve()), stat.st_mtime_ns, stat.st_size)


def _cached(cache: dict[tuple[str, int, int], Any], path: Path, build: Callable[[Path], Any]) -> Any:
    """build(path) memoised on path, mtime and size, for the many seeks into one original."""
    key = _cache_key(path)
    value = cache.get(key)
    if value is None:
        if len(cache) >= _CACHE_MAX:
            cache.clear()
        value = cache[key] = build(path)
    return value


def _probe_cached(path: Path) -> VideoInfo:
    return _cached(_PROBE_CACHE, Path(path), probe)


def _index_cached(path: Path) -> _FrameIndex:
    return _cached(_INDEX_CACHE, Path(path), _frame_index)


def frame_times(path: Path) -> list[float]:
    """Source frame times in presentation order from packet pts, without a full decode; pts_map decodes to this list."""
    return list(_index_cached(Path(path)).times)


# ----------------------------------------------------------------------------------------------------------------
# Proxy and pts map
# ----------------------------------------------------------------------------------------------------------------


def _even(value: float) -> int:
    return max(2, int(round(value)) // 2 * 2)


def _proxy_size(width: int, height: int, max_edge: int | None) -> tuple[int, int]:
    """Even size with the long edge at most max_edge; never upscales."""
    scale = min(1.0, max_edge / max(width, height)) if max_edge else 1.0
    return _even(width * scale), _even(height * scale)


def _fps_filter(fps: float) -> str:
    """Slot i shows the last source frame at or before i / fps in whole microseconds; slots before the first repeat it."""
    # The fps filter compares exact times, but 6 decimals cannot tell a frame at 66667 us (1 MHz timescale) from the slot
    # at 66666.67 us. Rounding each pts to whole microseconds and taking off half of one makes "frame <= slot" exactly
    # "frame us <= slot us rounded", the rule _frame_at applies to the printed times.
    return f"settb=1/{_US},settb=1/{2 * _US},setpts=PTS-1,fps={_rate_text(fps)}:round=up:start_time=0"


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
    """The SAM proxy: H.264 MP4, rotation baked in, no audio or metadata, yuv420p, even size, constant fps; idempotent."""
    src, dst = Path(src), Path(dst)
    info = probe(src)
    width, height = _proxy_size(info.width, info.height, max_edge)
    if _proxy_current(src, dst, fps=fps, size=(width, height)):
        return probe(dst)
    tmp = _tmp_for(dst)
    filters = f"{_fps_filter(fps)},scale={width}:{height}:flags=lanczos,format=yuv420p"
    try:
        _run([
            FFMPEG, "-y", *_QUIET, "-i", str(src), "-map", f"0:{VIDEO_STREAM}", "-an", "-sn", "-dn",
            "-vf", filters, "-fps_mode", "cfr", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-map_metadata", "-1", "-map_chapters", "-1", str(tmp),
        ])
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, dst)
    return probe(dst)


def pts_map(proxy: Path) -> list[float]:
    """ffprobe -show_frames pts per decoded frame in seconds since the file start (whole microseconds); index = frame."""
    proxy = Path(proxy)
    entries = "frame=pts,best_effort_timestamp:stream=time_base:format=start_time"
    data = _ffprobe_json(proxy, ["-select_streams", VIDEO_STREAM, "-show_entries", entries])
    clock = _clock(proxy, data)
    out: list[float] = []
    for frame in data.get("frames") or []:
        ticks = _ticks(frame, "pts", "best_effort_timestamp")
        if ticks is None:
            raise VideoError(f"{proxy}: frame {len(out)} has no timestamp")
        out.append(clock.us(ticks) / _US)
    return out


def pts_to_proxy_frame(pts: list[float], t: float) -> int:
    """The proxy frame whose pts is nearest to t, clamped to the ends; at most half a proxy frame (33 ms at 15 fps) off."""
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


def _frame_interval_us(us: Sequence[int], fps: float) -> int:
    """Typical microseconds between frames: the median gap, or the nominal interval for a one-frame file."""
    if len(us) >= 2:
        return int(np.median(np.diff(np.asarray(us, dtype=np.int64))))
    return int(_US / fps) if fps else _US // 10


def _frame_at(index: _FrameIndex, t: float, fps: float, path: Path) -> int:
    """The source frame on screen at t: the last frame whose pts is at or before t (frame 0 before the first)."""
    us = index.us
    if not us:
        raise VideoError(f"{path}: no video frames")
    slack = _frame_interval_us(us, fps)
    at = round(t * _US) if math.isfinite(t) else None
    # proxy slots run to the end of the last frame, which can outlast the usual gap
    if at is None or at < min(0, us[0]) - slack or at > max(index.end_us, us[-1] + slack):
        raise VideoError(f"{path}: no frame at {t:.3f} s (frames run {us[0] / _US:.3f} to {us[-1] / _US:.3f} s)")
    return max(0, bisect.bisect_right(us, at) - 1)


def _seek_start(times: Sequence[float], k: int) -> float | None:
    """The -ss that makes frame k the first out: halfway from the frame before, since accurate seek drops earlier pts."""
    return (times[k - 1] + times[k]) / 2 if k > 0 else None


def _decode_start(key_at: Sequence[int], k: int) -> int:
    """The last keyframe at or before frame k in presentation order (0 when none is flagged)."""
    # On open-GOP files the frames shown just before a keyframe are decoded after it; an -ss between them lands on
    # that keyframe and its decoder drops them. Decoding from the keyframe before k and counting frames is always right.
    j = bisect.bisect_right(key_at, k) - 1
    return key_at[j] if j >= 0 else 0


def _seek_groups(ks: Sequence[int], key_at: Sequence[int]) -> list[list[int]]:
    """Sorted frame indices in runs one decode covers; a keyframe skipping over _SEEK_SAVES_FRAMES starts a new run."""
    groups: list[list[int]] = []
    for k in ks:
        if groups and _decode_start(key_at, k) - groups[-1][-1] <= _SEEK_SAVES_FRAMES:
            groups[-1].append(k)
        else:
            groups.append([k])
    return groups


def iter_frames_at(path: Path, times: Sequence[float]) -> Iterator[tuple[float, np.ndarray]]:
    """read_frame_at for many times: (t, rgb) per distinct t in ascending order, one decode per run of nearby frames."""
    path = Path(path)
    info = _probe_cached(path)
    index = _index_cached(path)
    wanted: dict[int, list[float]] = {}
    for t in sorted({float(t) for t in times}):
        wanted.setdefault(_frame_at(index, t, info.fps, path), []).append(t)
    for group in _seek_groups(sorted(wanted), index.key_at):
        start = _decode_start(index.key_at, group[0])
        picks = [k - start for k in group]
        filters = [] if picks == [0] else ["select='" + "+".join(f"eq(n,{n})" for n in picks) + "'"]
        got = 0
        ss = _seek_start(index.times, start)
        with closing(_raw_frames(path, info.width, info.height, filters=filters, ss=ss, frames=len(group))) as frames:
            for i, rgb in frames:
                for t in wanted[group[i]]:
                    yield t, rgb
                got += 1
        if got != len(group):
            raise VideoError(f"{path}: ffmpeg gave {got} of {len(group)} frames from source frame {group[0]}")


def read_frame_at(path: Path, t_seconds: float) -> np.ndarray:
    """One RGB uint8 (H, W, 3) frame from the original: the source frame on screen at t, full size, rotation applied."""
    for _, rgb in iter_frames_at(path, [t_seconds]):
        return rgb
    raise VideoError(f"{path}: no frame at {t_seconds:.3f} s")


def iter_frames(
    path: Path, *, fps: float | None = None, max_edge: int | None = None
) -> Iterator[tuple[int, float, np.ndarray]]:
    """(index, t_seconds, rgb) through ffmpeg: each frame at its own pts, or resampled as make_proxy does (t = i / fps)."""
    path = Path(path)
    info = _probe_cached(path)
    native = (info.width, info.height)
    width, height = _proxy_size(*native, max_edge) if max_edge and max(native) > max_edge else native
    filters: list[str] = [_fps_filter(fps)] if fps else []
    if (width, height) != native:
        filters.append(f"scale={width}:{height}:flags=lanczos")
    times = None if fps else _index_cached(path).times
    for index, rgb in _raw_frames(path, width, height, filters=filters):
        if fps:
            t = index / fps
        elif times is not None and index < len(times):
            t = times[index]
        else:
            t = index / info.fps if info.fps else float(index)
        yield index, t, rgb


def sharpness(rgb_or_gray: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Variance of the Laplacian of the grayscale image (inside mask when given); higher is sharper."""
    image = np.asarray(rgb_or_gray)
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    values = lap[np.asarray(mask, dtype=bool)] if mask is not None else lap
    return float(values.var()) if values.size > 1 else 0.0


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of the file bytes, streamed."""
    with open(path, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def extract_wav(src: Path, dst: Path, *, rate: int = 24000) -> Path:
    """Mono 16-bit PCM WAV for Voice from the file start (silence before late audio); NoAudioError without audio."""
    src, dst = Path(src), Path(dst)
    if not probe(src).has_audio:
        raise NoAudioError(f"{src}: no audio stream")
    tmp = _tmp_for(dst)
    try:
        # first_pts=0 pads a late first sample with silence (and async fills gaps over 0.1 s), so Voice times are
        # file times like every other time here
        _run([
            FFMPEG, "-y", *_QUIET, "-i", str(src), "-map", "0:a:0", "-vn", "-sn", "-dn",
            "-af", "aresample=async=1:first_pts=0", "-ac", "1", "-ar", str(rate), "-c:a", "pcm_s16le", "-map_metadata", "-1",
            "-fflags", "+bitexact", "-flags:a", "+bitexact", "-f", "wav", str(tmp),
        ])
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, dst)
    return dst


# ----------------------------------------------------------------------------------------------------------------
# Stays Put clips and the display proxy
# ----------------------------------------------------------------------------------------------------------------


def _pixelate() -> Callable[..., np.ndarray]:
    """privacy.pixelate, imported at call time; if that module cannot load, nothing gets written (fails closed)."""
    from . import privacy  # noqa: PLC0415

    return privacy.pixelate


def _square_windows(windows: Sequence[Sequence[float]], frames: int) -> list[list[float]]:
    """Validated float windows, one per frame; a window that is empty or not square raises ValueError."""
    if frames < 1 or len(windows) != frames:
        raise ValueError(f"cut_clip needs one window per frame: {len(windows)} windows for {frames} frames")
    out: list[list[float]] = []
    for window in windows:
        x1, y1, x2, y2 = (float(v) for v in window)
        side_w, side_h = x2 - x1, y2 - y1
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or side_w <= 0 or side_h <= 0:
            raise ValueError(f"empty window {list(window)}")
        if abs(side_w - side_h) > max(1.0, 0.01 * side_w):
            raise ValueError(f"window {list(window)} is not square")
        out.append([x1, y1, x2, y2])
    return out


def _union_crop(windows: Sequence[Sequence[float]], width: int, height: int) -> tuple[int, int, int, int] | None:
    """Even-aligned (x, y, w, h) holding the in-frame part of every window plus a margin; None when not worth it."""
    if width % 2 or height % 2:
        return None
    x1 = max(0, int(math.floor(min(w[0] for w in windows))) - _CROP_MARGIN_PX) // 2 * 2
    y1 = max(0, int(math.floor(min(w[1] for w in windows))) - _CROP_MARGIN_PX) // 2 * 2
    x2 = min(width, int(math.ceil(max(w[2] for w in windows))) + _CROP_MARGIN_PX)
    y2 = min(height, int(math.ceil(max(w[3] for w in windows))) + _CROP_MARGIN_PX)
    x2, y2 = x2 + x2 % 2, y2 + y2 % 2
    if x2 <= x1 or y2 <= y1 or (x2 - x1) * (y2 - y1) > 0.8 * width * height:
        return None
    return x1, y1, x2 - x1, y2 - y1


def _cut_window(frame: np.ndarray, window: Sequence[float], size: int) -> np.ndarray:
    """Resample a source window onto size x size (clip u reads x1 + (u + 0.5) * sw / size); off-frame pixels are black."""
    x1, y1, x2, y2 = (float(v) for v in window)
    sx, sy = (x2 - x1) / size, (y2 - y1) / size
    if sx <= 0 or sy <= 0:
        raise ValueError(f"empty window {list(window)}")
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
    """(f0_src, the pts of each clip frame) for `frames` source frames centred on the frame on screen at pts_center."""
    src = Path(src)
    if frames < 1:
        raise ValueError("a clip needs at least one frame")
    index = _index_cached(src)
    times = index.times
    try:
        centre = _frame_at(index, float(pts_center), _probe_cached(src).fps, src)
    except VideoError as exc:
        raise ClipOutOfRange(str(exc)) from exc
    f0 = centre - frames // 2
    if f0 < 0 or f0 + frames > len(times):
        raise ClipOutOfRange(
            f"{src.name}: {frames} frames around {pts_center:.3f} s would use source frames {f0} to "
            f"{f0 + frames - 1} of 0..{len(times) - 1}"
        )
    return f0, list(times[f0 : f0 + frames])


def _clip_frames(
    src: Path, f0: int, windows: list[list[float]], size: int, blur_boxes: list[list[Box4]] | None, block_px: int
) -> Iterator[np.ndarray]:
    """The size x size clip frames before encoding: source frame f0 + i cut to window i, then pixelated."""
    info = _probe_cached(src)
    index = _index_cached(src)
    start = _decode_start(index.key_at, f0)
    crop = _union_crop(windows, info.width, info.height)
    ox, oy, cw, ch = crop or (0, 0, info.width, info.height)
    filters = [f"select='gte(n,{f0 - start})'"] if f0 > start else []
    if crop:
        filters.append(f"crop={cw}:{ch}:{ox}:{oy}")
    pixelate = _pixelate() if blur_boxes else None
    ss = _seek_start(index.times, start)
    with closing(_raw_frames(src, cw, ch, filters=filters, ss=ss, frames=len(windows))) as frames:
        for i, frame in frames:
            x1, y1, x2, y2 = windows[i]
            clip = _cut_window(frame, (x1 - ox, y1 - oy, x2 - ox, y2 - oy), size)
            if pixelate is not None and blur_boxes is not None and blur_boxes[i]:
                regions = [_box_to_window(box, windows[i], size) for box in blur_boxes[i]]
                clip = pixelate(clip, regions, block_px=max(1, round(block_px * size / (x2 - x1))))
            yield clip


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
    """The panel-locked Stays Put clip: clip_span's frames, each cut to its square window, pixelated, H.264 CFR."""
    src, dst = Path(src), Path(dst)
    windows = _square_windows(windows_src, frames)
    if blur_boxes is not None and len(blur_boxes) != frames:
        raise ValueError(f"cut_clip needs one blur list per frame: {len(blur_boxes)} for {frames} frames")
    f0, _ = clip_span(src, pts_center=pts_center, frames=frames)
    info = _probe_cached(src)
    tmp = _tmp_for(dst)
    enc = _Encode(_encoder_cmd(size, size, info.fps or 30.0, tmp, preset="medium", crf=crf, like=info))
    written = 0
    try:
        with closing(_clip_frames(src, f0, windows, size, blur_boxes, block_px)) as clips:
            for clip in clips:
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
        window_px=int(round(windows[0][2] - windows[0][0])),
        windows_src=windows,
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
    """Decode the proxy, pixelate each frame's boxes (proxy pixels) through privacy.pixelate, re-encode it CFR."""
    proxy, dst = Path(proxy), Path(dst)
    # a key that is not a frame index (str after a JSON round trip) would blur nothing; fail closed instead. Keys past
    # the last frame are allowed: gate_sam carries a track's last box a bridge past the lane's end on purpose
    bad = [k for k in boxes_by_frame if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 0]
    if bad:
        raise ValueError(f"boxes_by_frame keys must be 0-based proxy frame ints, got {bad[:5]}")
    info = probe(proxy)
    pixelate = _pixelate()
    tmp = _tmp_for(dst)
    enc = _Encode(_encoder_cmd(info.width, info.height, info.fps or 15.0, tmp, preset=preset, crf=crf, like=info))
    try:
        with closing(_raw_frames(proxy, info.width, info.height)) as frames:
            for index, rgb in frames:
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
    """Write an RGB uint8 (H, W, 3) array as a JPEG with no metadata; returns its sha256."""
    image = np.ascontiguousarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"write_jpeg wants (H, W, 3), got {image.shape}")
    if image.dtype != np.uint8:
        if not np.issubdtype(image.dtype, np.integer) or image.min() < 0 or image.max() > 255:
            raise ValueError(f"write_jpeg wants uint8 RGB, got {image.dtype}")
        image = image.astype(np.uint8)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(dst, "JPEG", quality=quality)
    return sha256_file(dst)


def write_png_mask(mask: np.ndarray, dst: Path) -> str:
    """Write a truthy (H, W) raster as a one-bit PNG; returns its sha256."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(mask).astype(bool)).save(dst, "PNG")
    return sha256_file(dst)


def cut_crop(rgb: np.ndarray, crop: contracts.Crop) -> np.ndarray:
    """The source box [dx, dy, dx + w / scale, dy + h / scale] at (w, h); area-averaged when reducing, off-frame black."""
    src = np.ascontiguousarray(rgb)
    sx = sy = 1.0 / crop.scale  # source pixels per crop pixel
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
