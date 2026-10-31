"""Media lane (M1) tests on clips ffmpeg builds here with lavfi, plus data/test/pingpong.mp4 (24000/1001 fps)."""

# The clips are built once per session; no fixture files live in the repo. Every identity picture has luma 20 + 2 N,
# so a decoded frame names its source index.
# - ident: 320x240, 30 fps CFR, 3 s, keyframe every 10 frames, a 440 Hz AAC track, a creation_time tag.
# - vfr: the same picture with jittered and gapped timestamps, no audio.
# - late: 1 s with audio from 0 s and video from 0.05 s (the iPhone edit-list shape).
# - cut: ident stream-copied from 0.5 s, so the edit list discards frames 10 to 14 of the first GOP.
# - portrait: stored 1920x1080 with a 90 degree display rotation, the top half of the stored picture white.
# - square: 640x480, a white 60x50 box moving one pixel per frame, for the clip round trip and blur.
# - iphone: HEVC lossless MOV with B-frames and open GOPs, 180 degree rotation, AAC and the Apple tags a phone writes.
# - odd: 161x121 yuv444p; cover: ident plus PNG cover art; hlg: a tiny clip tagged HLG.
# The three contract tests of docs/CONTRACTS.md are test_vfr_proxy_maps_within_one_frame,
# test_cut_clip_round_trips_a_known_box and test_portrait_source_reports_540x960.

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import shutil
import subprocess
import wave
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from PIL import Image

from walkaround import contracts, paths, video

REPO = Path(__file__).resolve().parent.parent
PINGPONG = REPO / "data" / "test" / "pingpong.mp4"

FFMPEG = ["ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error"]
IDENT_SRC = "color=c=gray:s=320x240:r=30:d=3,geq=lum=20+2*N:cb=128:cr=128"
SQUARE_SRC = "color=c=black:s=640x480:r=30:d=2,geq=lum='255*between(X\\,200+N\\,259+N)*between(Y\\,150\\,199)':cb=128:cr=128"
LOSSLESS = ["-c:v", "libx264", "-preset", "ultrafast", "-qp", "0", "-pix_fmt", "yuv420p"]
APPLE_TAGS = [
    "-movflags", "use_metadata_tags",
    "-metadata", "creation_time=2026-10-27T15:05:00Z",
    "-metadata", "com.apple.quicktime.creationdate=2026-10-27T11:04:20-0400",
    "-metadata", "com.apple.quicktime.model=iPhone 15 Pro",
    "-metadata", "com.apple.quicktime.location.ISO6709=+42.2780-083.7382+250.000/",
]


def square_box(n: int) -> list[float]:
    """The white box of square.mp4 at source frame n, half-open source pixels."""
    return [200.0 + n, 150.0, 260.0 + n, 200.0]


def ffmpeg(*args: str) -> None:
    subprocess.run([*FFMPEG, *args], check=True, capture_output=True)


def ffprobe_value(path: Path, entry: str) -> str:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "V:0", "-show_entries", entry, "-of", "default=nw=1:nk=1", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    return out.strip()


def all_tags(path: Path) -> dict[str, str]:
    """Format and stream tags of a file, lower-cased keys."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format_tags:stream_tags", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    data = json.loads(out)
    tags: dict[str, str] = {}
    for section in [data.get("format") or {}, *(data.get("streams") or [])]:
        tags.update({k.lower(): str(v) for k, v in (section.get("tags") or {}).items()})
    return tags


def packet_flags(path: Path) -> list[str]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "V:0", "-show_entries", "packet=flags", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    return out.split()


def rotate_tag(src: Path, dst: Path, clockwise: int) -> None:
    """Stream-copy src with a display rotation; ffmpeg 7+ spells it -display_rotation, older ones the rotate tag."""
    try:
        ffmpeg("-display_rotation", str(-clockwise), "-i", str(src), "-c", "copy", str(dst))
    except subprocess.CalledProcessError:
        ffmpeg("-i", str(src), "-c", "copy", "-metadata:s:v:0", f"rotate={clockwise % 360}", str(dst))
    if video.probe(dst).rotation != clockwise % 360:
        pytest.skip("this ffmpeg cannot write a display rotation")


def has_encoder(name: str) -> bool:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    return any(line.split()[1:2] == [name] for line in out.splitlines() if line.strip())


def source_index(rgb: np.ndarray, means: np.ndarray) -> int:
    """Which identity-clip frame this is: the one whose mean is nearest."""
    return int(np.abs(means - rgb[..., 0].mean()).argmin())


def blob(rgb: np.ndarray) -> list[int] | None:
    """Half-open bounding box of the bright pixels, or None."""
    ys, xs = np.nonzero(rgb[..., 0] > 128)
    if not xs.size:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def coarse_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Mean difference at 20 x 20 after area averaging, so 4:2:0 chroma loss averages out but a wrong matrix does not."""
    # measured on the HLG card: 1.0 when matrix and tags agree, 2.6 to 3.0 when they disagree
    shrink = lambda x: cv2.resize(x, (20, 20), interpolation=cv2.INTER_AREA).astype(np.float64)  # noqa: E731
    return float(np.abs(shrink(a) - shrink(b)).mean())


def frames_by_index(path: Path, wanted: set[int]) -> dict[int, np.ndarray]:
    """A sequential decode of the frames at the given indices only (pingpong whole would be 390 MB)."""
    return {i: rgb for i, _, rgb in video.iter_frames(path) if i in wanted}


@pytest.fixture(scope="session")
def clips(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("clips")
    names = ("ident", "vfr", "late", "cut", "stored", "portrait", "square", "odd", "cover", "hlg")
    c = {n: d / f"{n}.mp4" for n in names}
    ffmpeg(
        "-f", "lavfi", "-i", IDENT_SRC, "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=3",
        "-map", "0:v", "-map", "1:a", *LOSSLESS, "-g", "10", "-c:a", "aac", "-b:a", "64k",
        "-metadata", "creation_time=2026-10-26T12:00:00Z", "-shortest", str(c["ident"]),
    )
    ffmpeg(
        "-f", "lavfi", "-i", IDENT_SRC, "-vf", "settb=1/90000,setpts=3000*(N+0.4*sin(1.7*N)+0.6*floor(N/10))",
        "-fps_mode", "passthrough", "-enc_time_base", "1/90000", "-video_track_timescale", "90000", *LOSSLESS, str(c["vfr"]),
    )
    ffmpeg(
        "-f", "lavfi", "-i", IDENT_SRC.replace("d=3", "d=1"), "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1.2",
        "-filter_complex", "[0:v]settb=1/90000,setpts=3000*N+4500[v]", "-map", "[v]", "-map", "1:a",
        "-fps_mode", "passthrough", "-enc_time_base", "1/90000", "-video_track_timescale", "90000", *LOSSLESS,
        "-c:a", "aac", "-b:a", "64k", str(c["late"]),
    )
    ffmpeg("-ss", "0.5", "-i", str(c["ident"]), "-map", "0:v", "-c", "copy", str(c["cut"]))
    ffmpeg(
        "-f", "lavfi", "-i", "color=c=black:s=1920x1080:r=30:d=0.2,drawbox=x=0:y=0:w=1920:h=540:color=white:t=fill",
        *LOSSLESS, str(c["stored"]),
    )
    rotate_tag(c["stored"], c["portrait"], 90)
    ffmpeg("-f", "lavfi", "-i", SQUARE_SRC, "-c:v", "libx264", "-preset", "ultrafast", "-crf", "10", "-pix_fmt", "yuv420p", str(c["square"]))
    ffmpeg(
        "-f", "lavfi", "-i", "color=c=gray:s=161x121:r=30:d=1,geq=lum='16+X+N':cb=128:cr=128", "-c:v", "libx264",
        "-preset", "ultrafast", "-qp", "0", "-pix_fmt", "yuv444p", str(c["odd"]),
    )
    cover_png = d / "cover.png"
    ffmpeg("-f", "lavfi", "-i", "color=c=red:s=64x64:d=1", "-frames:v", "1", "-update", "1", str(cover_png))
    ffmpeg(
        "-i", str(cover_png), "-i", str(c["ident"]), "-map", "0", "-map", "1:v", "-c:v:0", "png", "-c:v:1", "copy",
        "-disposition:v:0", "attached_pic", str(c["cover"]),
    )
    ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=s=128x96:r=30:d=0.5,setparams=color_trc=arib-std-b67:color_primaries=bt2020:colorspace=bt2020nc",
        "-c:v", "libx264", "-preset", "ultrafast", "-qp", "0", str(c["hlg"]),
    )
    return c


@pytest.fixture(scope="session")
def iphone(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not has_encoder("libx265"):
        pytest.skip("ffmpeg has no libx265")
    d = tmp_path_factory.mktemp("iphone")
    raw, out = d / "hevc.mp4", d / "IMG_0001.MOV"
    ffmpeg(
        "-f", "lavfi", "-i", IDENT_SRC + ",drawbox=x=0:y=0:w=40:h=30:color=white:t=fill", "-c:v", "libx265", "-preset", "fast",
        "-x265-params", "lossless=1:bframes=3:keyint=12:min-keyint=12:log-level=error", "-pix_fmt", "yuv420p", "-tag:v", "hvc1", str(raw),
    )
    turned = d / "turned.mp4"
    rotate_tag(raw, turned, 180)
    ffmpeg(
        "-i", str(turned), "-f", "lavfi", "-i", "sine=frequency=300:sample_rate=48000:duration=3",
        "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "64k", "-shortest", *APPLE_TAGS, str(out),
    )
    return out


@pytest.fixture(scope="session")
def open_gop_cut(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Open-GOP HEVC stream-copied from mid-GOP: it starts on a CRA whose kept leading frames no decoder outputs."""
    if not has_encoder("libx265"):
        pytest.skip("ffmpeg has no libx265")
    d = tmp_path_factory.mktemp("opengop")
    src = d / "open.mp4"
    params = "lossless=1:bframes=4:b-adapt=0:keyint=15:min-keyint=15:log-level=error"
    ffmpeg(
        "-f", "lavfi", "-i", IDENT_SRC, "-c:v", "libx265", "-preset", "fast", "-x265-params", params,
        "-pix_fmt", "yuv420p", "-tag:v", "hvc1", str(src),
    )
    for ss in ("0.40", "0.42", "0.44"):  # the copy starts on the CRA at 0.5 s, its leading frames after the cut stay
        cut = d / f"cut_{ss}.mov"
        ffmpeg("-ss", ss, "-i", str(src), "-map", "0:v", "-c", "copy", str(cut))
        if sum("D" not in f for f in packet_flags(cut)) > len(video.pts_map(cut)):
            return cut
    pytest.skip("this ffmpeg never cut onto the open-GOP keyframe")


@pytest.fixture(scope="session")
def fine_clips(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """The identity picture on timescales whose times 6 decimals cannot hold, and in MPEG-TS (start 1.4 s)."""
    d = tmp_path_factory.mktemp("fine")
    c = {n: d / f"{n}.mp4" for n in ("us_cfr", "us_jitter", "tenth_us")} | {"ts": d / "ts.ts"}
    # us_cfr: frame 2 at 66667 us against the 15 fps slot at 66666.67 us; the others jitter around each 1/30 s
    ffmpeg("-f", "lavfi", "-i", IDENT_SRC, *LOSSLESS, "-video_track_timescale", "1000000", str(c["us_cfr"]))
    for name, ticks, spread in (("us_jitter", 1_000_000, 5), ("tenth_us", 10_000_000, 7)):  # +-2 us; +-0.3 us
        jitter = f"settb=1/{ticks},setpts='if(N,round(N*{ticks}/30)+mod(N,{spread})-{spread // 2},0)'"
        ffmpeg(
            "-f", "lavfi", "-i", IDENT_SRC, "-vf", jitter, "-fps_mode", "passthrough", "-enc_time_base", f"1/{ticks}",
            "-video_track_timescale", str(ticks), *LOSSLESS, str(c[name]),
        )
    ffmpeg("-f", "lavfi", "-i", IDENT_SRC, *LOSSLESS, "-f", "mpegts", str(c["ts"]))
    return c


@pytest.fixture(scope="session")
def long_last(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """30 identity frames at 30 fps, then frame 30 (luma 80) shown for 125 ms."""
    dst = tmp_path_factory.mktemp("tail") / "tail.mp4"
    graph = (
        "color=c=gray:s=320x240:r=30:d=0.9667,geq=lum=20+2*N:cb=128:cr=128[a];"
        "color=c=gray:s=320x240:r=8:d=0.125,geq=lum=80:cb=128:cr=128[b];[a][b]concat=n=2:v=1:a=0"
    )
    ffmpeg("-filter_complex", graph, "-fps_mode", "passthrough", *LOSSLESS, "-video_track_timescale", "120", str(dst))
    return dst


@pytest.fixture(scope="session")
def ident_means(clips: dict[str, Path]) -> np.ndarray:
    return np.array([rgb[..., 0].mean() for _, _, rgb in video.iter_frames(clips["ident"])])


@pytest.fixture(scope="session")
def vfr_frames(clips: dict[str, Path]) -> tuple[list[float], np.ndarray]:
    """(source pts, mean luma per frame) of the VFR clip."""
    frames = list(video.iter_frames(clips["vfr"]))
    return [t for _, t, _ in frames], np.array([rgb[..., 0].mean() for _, _, rgb in frames])


@pytest.fixture(scope="session")
def ident_proxy(clips: dict[str, Path], tmp_path_factory: pytest.TempPathFactory) -> Path:
    dst = tmp_path_factory.mktemp("proxy") / "ident.proxy.mp4"
    video.make_proxy(clips["ident"], dst, fps=15.0, max_edge=160, preset="veryfast")
    return dst


# Module hygiene and defaults -----------------------------------------------------------------------------------------


def imported_names(source: str) -> set[str]:
    """Every module name an import statement mentions, with relative imports spelled from walkaround."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = "walkaround" if node.level else ""
            module = ".".join(p for p in (base, node.module or "") if p)
            names.add(module)
            names.update(f"{module}.{alias.name}" for alias in node.names)
    return names


def test_imported_names_sees_relative_imports() -> None:
    names = imported_names("from . import ledger as l\nfrom .x import y\nimport openai.types\n")
    assert {"walkaround.ledger", "walkaround.x.y", "openai.types"} <= names


def test_module_is_ascii_and_offline() -> None:
    source = Path(video.__file__).read_bytes()
    assert source.isascii()
    assert Path(__file__).read_bytes().isascii()
    for name in imported_names(source.decode()):
        root = name.split(".")[0]
        assert root not in ("openai", "httpx", "meta_sam_parser"), name
        assert name != "walkaround.ledger" and not name.startswith("walkaround.ledger."), name


def test_defaults_follow_thresholds() -> None:
    th = contracts.load_thresholds()
    media = th["media"]
    proxy = inspect.signature(video.make_proxy).parameters
    assert proxy["fps"].default == media["proxy_fps"]
    assert proxy["max_edge"].default == media["proxy_max_edge"]
    assert proxy["preset"].default == media["proxy_preset"]
    assert proxy["crf"].default == media["proxy_crf"]
    assert media["proxy_codec"] == "libx264"
    assert inspect.signature(video.extract_wav).parameters["rate"].default == media["wav_rate"]
    clip = inspect.signature(video.cut_clip).parameters
    assert clip["size"].default == media["clip_size_px"]
    assert clip["crf"].default == media["clip_crf"]
    assert clip["block_px"].default == th["privacy"]["pixel_block_px_at_4k"]
    # 91 frames at the assumed 30 fps are the +/- 1.5 s of 5.9, and the prompt frame is the centre cut_clip uses
    assert media["clip_frames"] == 2 * round(media["clip_half_window_s"] * media["source_fps_assumed"]) + 1
    assert media["clip_prompt_frame"] == media["clip_frames"] // 2


def test_rate_text() -> None:
    assert video._rate_text(15.0) == "15/1"
    assert video._rate_text(30000 / 1001) == "30000/1001"
    assert video._rate_text(29.97003) == "30000/1001"
    assert video._rate_text(24000 / 1001) == "24000/1001"
    assert video._rate_text(29.95) == "29.950000"  # a VFR average is kept as measured
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            video._rate_text(bad)


def test_rescale_and_clock_round_like_ffmpeg(tmp_path: Path) -> None:
    micro = Fraction(1, 1_000_000)
    assert video._rescale(2, Fraction(1, 30), micro) == 66667 and video._rescale(1, Fraction(1, 30), micro) == 33333
    assert video._rescale(1, Fraction(1, 2_000_000), micro) == 1  # halves go away from zero (av_rescale_q)
    assert video._rescale(-1, Fraction(1, 2_000_000), micro) == -1
    assert video._rescale(-3, Fraction(1, 10_000_000), micro) == 0
    ts = video._clock(tmp_path, {"streams": [{"time_base": "1/90000"}], "format": {"start_time": "1.400000"}})
    assert ts.offset == -126000 and ts.us(126000 + 3000) == 33333 and ts.span_us(3000) == 33333
    assert video._clock(tmp_path, {"streams": [{"time_base": "1/600"}], "format": {"start_time": "N/A"}}).offset == 0
    with pytest.raises(video.VideoError):
        video._clock(tmp_path, {"streams": [{"time_base": "0/1"}], "format": {}})


# probe and ffprobe_info -------------------------------------------------------------------------------------------


def test_probe_ident(clips: dict[str, Path]) -> None:
    info = video.probe(clips["ident"])
    assert (info.width, info.height) == (320, 240)
    assert info.fps == pytest.approx(30.0)
    assert info.nb_frames == 90
    assert info.duration_s == pytest.approx(3.0, abs=0.05)
    assert info.rotation == 0 and info.codec == "h264"
    assert info.has_audio and not info.vfr and not info.hdr
    assert info.creation_time == datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc)
    assert info.device is None and info.start_time == 0.0


def test_probe_vfr_flags_the_jitter(clips: dict[str, Path]) -> None:
    info = video.probe(clips["vfr"])
    assert info.vfr and not info.has_audio and info.creation_time is None
    assert info.nb_frames == 90
    assert 25.0 < info.fps < 30.0  # the average rate over the gapped timeline


def test_probe_pingpong() -> None:
    info = video.probe(PINGPONG)
    assert (info.width, info.height) == (1280, 720)
    assert info.fps == pytest.approx(24000 / 1001, abs=1e-4)
    assert info.nb_frames == 141
    assert info.duration_s == pytest.approx(5.88, abs=0.01)
    assert not info.has_audio and info.creation_time is None and info.rotation == 0


def test_probe_errors(tmp_path: Path) -> None:
    with pytest.raises(video.VideoError):
        video.probe(tmp_path / "missing.mp4")
    text = tmp_path / "notes.txt"
    text.write_text("not a video\n")
    with pytest.raises(video.VideoError):
        video.probe(text)
    wav = tmp_path / "tone.wav"
    ffmpeg("-f", "lavfi", "-i", "sine=duration=0.2", str(wav))
    with pytest.raises(video.VideoError, match="no video stream"):
        video.probe(wav)


def test_rotation_sign_convention() -> None:
    assert video._rotation({"side_data_list": [{"rotation": -90}]}) == 90  # iPhone portrait: ffprobe says -90
    assert video._rotation({"side_data_list": [{"rotation": 90}]}) == 270
    assert video._rotation({"side_data_list": [{"rotation": 180}]}) == 180
    assert video._rotation({"side_data_list": [{"rotation": -180}]}) == 180
    assert video._rotation({"tags": {"rotate": "90"}}) == 90
    assert video._rotation({"side_data_list": [{"rotation": -90}], "tags": {"rotate": "270"}}) == 90
    assert video._rotation({}) == 0


def test_parse_time_and_iso_z() -> None:
    assert video.iso_z(video._parse_time("2026-10-26T12:00:00.000000Z")) == "2026-10-26T12:00:00Z"
    assert video.iso_z(video._parse_time("2026-10-26T14:00:00+02:00")) == "2026-10-26T12:00:00Z"
    assert video.iso_z(video._parse_time("2026-10-27T11:04:20-0400")) == "2026-10-27T15:04:20Z"  # the Apple spelling
    assert video.iso_z(video._parse_time("2026-10-27T15:05:00.000000Z;2026-10-27T15:05:00Z")) == "2026-10-27T15:05:00Z"
    assert video.iso_z(datetime(2026, 1, 2, 3, 4, 5)) == "2026-01-02T03:04:05Z"  # naive means UTC
    for unset in ("1904-01-01T00:00:00.000000Z", "1970-01-01T00:00:00Z", "garbage", "", None):
        assert video._parse_time(unset) is None


def test_video_stream_skips_cover_art(clips: dict[str, Path]) -> None:
    cover = {"codec_type": "video", "codec_name": "png", "disposition": {"attached_pic": 1}}
    real = {"codec_type": "video", "codec_name": "h264", "disposition": {"attached_pic": 0}}
    audio = {"codec_type": "audio", "codec_name": "aac"}
    assert video._video_stream([audio, cover, real]) is real
    assert video._video_stream([cover, audio]) is None
    src = clips["cover"]
    assert "png" in subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(src)], capture_output=True, text=True).stdout
    info = video.probe(src)
    assert (info.codec, info.width, info.height, info.nb_frames) == ("h264", 320, 240, 90)
    assert len(video.frame_times(src)) == 90
    assert video.read_frame_at(src, 1.0).shape == (240, 320, 3)


def test_hdr_flag(clips: dict[str, Path]) -> None:
    info = video.probe(clips["hlg"])
    assert info.hdr and video.ffprobe_info(info, container="mp4").hdr
    assert (info.color_space, info.color_transfer, info.color_primaries) == ("bt2020nc", "arib-std-b67", "bt2020")
    sdr = video.probe(clips["ident"])
    assert not sdr.hdr and video._color_args(sdr) == video._BT709 == ("bt709",) * 4
    assert video._hdr({"side_data_list": [{"side_data_type": "DOVI configuration record"}]})


def test_hdr_sources_keep_their_colour_tags(clips: dict[str, Path], tmp_path: Path) -> None:
    """An iPhone records HLG by default; nothing here tone-maps it, so re-encodes keep its tags instead of BT.709."""
    src = clips["hlg"]
    hlg = ("bt2020nc", "arib-std-b67", "bt2020")
    tags = lambda p: tuple(ffprobe_value(p, f"stream={k}") for k in ("color_space", "color_transfer", "color_primaries"))  # noqa: E731
    proxy = tmp_path / "hlg.proxy.mp4"
    video.make_proxy(src, proxy, preset="veryfast")
    assert tags(proxy) == hlg
    shown = tmp_path / "hlg.proxy_blurred.mp4"
    video.render_blurred_proxy(proxy, shown, {}, block_px=8, dilate_frac=0.0, dilate_px=0)
    assert tags(shown) == hlg
    for (_, _, a), (_, _, b) in zip(video.iter_frames(proxy), video.iter_frames(shown)):
        assert coarse_diff(a, b) < 1.8  # the colourful test card survives the RGB round trip
    clip = tmp_path / "hlg.clip.mp4"
    window = [[20.0, 10.0, 100.0, 90.0]] * 5
    times = video.frame_times(src)
    rec = video.cut_clip(src, clip, pts_center=times[7], frames=5, windows_src=window, size=160)
    assert tags(clip) == hlg
    decoded = [rgb for _, _, rgb in video.iter_frames(clip)]
    for i in range(5):
        want = video._cut_window(video.read_frame_at(src, times[rec.f0_src + i]), window[i], 160)
        assert coarse_diff(decoded[i], want) < 1.8


def test_ffprobe_info_ident(clips: dict[str, Path]) -> None:
    info = video.ffprobe_info(video.probe(clips["ident"]), container="mp4")
    assert isinstance(info, contracts.FfprobeInfo)
    assert (info.width, info.height, info.display_width, info.display_height) == (320, 240, 320, 240)
    assert info.creation_time == "2026-10-26T12:00:00Z"
    assert info.container == "mp4" and info.codec == "h264" and info.has_audio
    assert not info.vfr and not info.hdr and info.device is None
    assert info.fps_avg == pytest.approx(30.0)
    forced = video.ffprobe_info(video.probe(clips["ident"]), container="mov", device="iPhone", hdr=True, vfr=True)
    assert forced.device == "iPhone" and forced.hdr and forced.vfr


def test_ffprobe_info_without_creation_time(clips: dict[str, Path]) -> None:
    info = video.ffprobe_info(video.probe(clips["vfr"]), container="mp4")
    assert info.creation_time == "not present" and info.vfr and not info.has_audio
    assert video.ffprobe_info(video.probe(PINGPONG), container="mp4").creation_time == "not present"


def test_portrait_source_reports_540x960(clips: dict[str, Path], tmp_path: Path) -> None:
    src = clips["portrait"]
    info = video.probe(src)
    assert (info.width, info.height) == (1080, 1920) and info.rotation == 90
    rec = video.ffprobe_info(info, container="mov")
    assert (rec.width, rec.height) == (1920, 1080)  # stored
    assert (rec.display_width, rec.display_height) == (1080, 1920) and rec.rotation == 90
    rgb = video.read_frame_at(src, 0.0)
    assert rgb.shape == (1920, 1080, 3)
    assert rgb[:, 540:].mean() > 200 and rgb[:, :540].mean() < 50  # the stored top edge turned clockwise to the right
    proxy = tmp_path / "portrait.proxy.mp4"
    out = video.make_proxy(src, proxy, fps=15.0, max_edge=960, preset="veryfast")
    assert (out.width, out.height) == (540, 960)
    assert out.rotation == 0  # baked in: nothing downstream depends on a tag
    assert out.nb_frames == 3
    assert ffprobe_value(proxy, "stream_side_data=rotation") == ""
    _, _, first = next(video.iter_frames(proxy))
    assert first.shape == (960, 540, 3)
    assert first[:, 270:].mean() > 200 and first[:, :270].mean() < 50


def test_iphone_like_hevc_mov(iphone: Path, tmp_path: Path) -> None:
    info = video.probe(iphone)
    assert info.codec == "hevc" and info.rotation == 180 and (info.width, info.height) == (320, 240)
    assert info.has_audio and info.device == "iPhone 15 Pro" and not info.hdr
    # the capture start (11:04:20 at -04:00), not creation_time, which the phone writes when the take ends
    assert info.creation_time == datetime(2026, 10, 27, 15, 4, 20, tzinfo=timezone.utc)
    rec = video.ffprobe_info(info, container="mov")
    assert rec.creation_time == "2026-10-27T15:04:20Z" and rec.device == "iPhone 15 Pro"
    times = video.frame_times(iphone)
    assert times == pytest.approx([k / 30 for k in range(90)], abs=1e-6)
    assert times == video.pts_map(iphone)  # B-frames sorted into presentation order
    frames = [rgb for _, _, rgb in video.iter_frames(iphone)]
    assert len(frames) == 90
    assert frames[0][-30:, -40:].mean() > 200 and frames[0][:30, :40].mean() < 100  # stored top-left, shown bottom-right
    for k in (0, 1, 11, 12, 13, 40, 89):  # around the open-GOP keyframe at 12 and mid-GOP
        assert np.array_equal(video.read_frame_at(iphone, times[k]), frames[k])
    # 21 to 23 are shown before the keyframe at 24 but decoded after it; an -ss between them would drop them
    picked = list(video.iter_frames_at(iphone, [times[k] for k in (21, 22, 23, 24, 50)]))
    for (t, rgb), k in zip(picked, (21, 22, 23, 24, 50)):
        assert t == times[k] and np.array_equal(rgb, frames[k])
    window = [[60.0, 40.0, 260.0, 240.0]] * 5
    for f0 in (9, 21):  # a clip that starts on such a frame is not shifted onto the keyframe
        got = list(video._clip_frames(iphone, f0, window, 200, None, 20))
        want = [video._cut_window(frames[f0 + i], window[i], 200) for i in range(5)]
        assert all(np.abs(g.astype(np.int16) - w.astype(np.int16)).max() <= 2 for g, w in zip(got, want))
    proxy = tmp_path / "iphone.proxy.mp4"
    out = video.make_proxy(iphone, proxy, preset="veryfast")
    assert (out.width, out.height, out.rotation, out.codec) == (320, 240, 0, "h264")
    assert out.nb_frames == 45 and not out.has_audio
    assert out.creation_time is None and out.device is None
    assert next(video.iter_frames(proxy))[2][-30:, -40:].mean() > 200


def test_metadata_never_reaches_derived_files(iphone: Path, tmp_path: Path) -> None:
    assert any("location" in key for key in all_tags(iphone))  # the source carries GPS
    proxy = tmp_path / "p.mp4"
    video.make_proxy(iphone, proxy, preset="veryfast")
    wav = video.extract_wav(iphone, tmp_path / "a.wav")
    clip = tmp_path / "c.mp4"
    video.cut_clip(iphone, clip, pts_center=1.5, frames=9, windows_src=[[40.0, 30.0, 200.0, 190.0]] * 9, size=160)
    for derived in (proxy, clip):
        keys = " ".join(all_tags(derived))
        for leak in ("location", "creationdate", "model", "creation_time", "make"):
            assert leak not in keys, (derived.name, leak)
    data = wav.read_bytes()
    assert data[36:40] == b"data" and b"LIST" not in data[:64] and b"iPhone" not in data and b"Lavf" not in data


def test_orig_extension_works(clips: dict[str, Path], tmp_path: Path) -> None:
    """The orchestrator hands over quarantine/<role>.orig; ffmpeg reads the container from the bytes."""
    orig = tmp_path / "quarantine" / "pickup.orig"
    orig.parent.mkdir()
    shutil.copyfile(clips["ident"], orig)
    info = video.probe(orig)
    assert (info.width, info.height, info.nb_frames) == (320, 240, 90) and info.has_audio
    out = video.make_proxy(orig, tmp_path / "media" / "pickup.proxy.mp4", max_edge=160, preset="veryfast")
    assert out.nb_frames == 45
    assert video.read_frame_at(orig, 1.0).shape == (240, 320, 3)
    assert video.extract_wav(orig, tmp_path / "media" / "pickup.wav").is_file()
    assert len(video.frame_times(orig)) == 90


# Proxy and pts map ------------------------------------------------------------------------------------------------


def test_make_proxy_ident(clips: dict[str, Path], ident_means: np.ndarray, ident_proxy: Path) -> None:
    info = video.probe(ident_proxy)
    assert (info.width, info.height) == (160, 120)
    assert info.fps == pytest.approx(15.0) and info.nb_frames == 45
    assert not info.has_audio and info.rotation == 0 and info.creation_time is None  # no audio, no metadata
    assert ffprobe_value(ident_proxy, "stream=pix_fmt") == "yuv420p"
    assert ffprobe_value(ident_proxy, "stream=codec_name") == "h264"
    data = ident_proxy.read_bytes()
    assert data.index(b"moov") < data.index(b"mdat")  # faststart
    pts = video.pts_map(ident_proxy)
    for i, t, rgb in video.iter_frames(ident_proxy):
        assert source_index(rgb, ident_means) == 2 * i  # 30 fps source, 15 fps proxy: every other frame
        assert t == pytest.approx(i / 15, abs=1e-6)
        if i % 11 == 0:  # the full-resolution read at a proxy pts is the picture the proxy showed
            assert source_index(video.read_frame_at(clips["ident"], pts[i]), ident_means) == 2 * i


def test_make_proxy_is_idempotent_until_the_request_changes(clips: dict[str, Path], tmp_path: Path) -> None:
    proxy = tmp_path / "p.mp4"
    video.make_proxy(clips["ident"], proxy, fps=15.0, max_edge=160, preset="veryfast")
    first = proxy.stat().st_mtime_ns
    video.make_proxy(clips["ident"], proxy, fps=15.0, max_edge=160, preset="veryfast")
    assert proxy.stat().st_mtime_ns == first
    again = video.make_proxy(clips["ident"], proxy, fps=10.0, max_edge=160, preset="veryfast")
    assert again.fps == pytest.approx(10.0) and again.nb_frames == 30
    assert proxy.stat().st_mtime_ns != first
    assert not list(tmp_path.glob(".*.part*"))


def test_make_proxy_even_dims_and_no_upscale(clips: dict[str, Path], tmp_path: Path) -> None:
    odd = video.make_proxy(clips["ident"], tmp_path / "odd.mp4", fps=15.0, max_edge=150, preset="veryfast")
    assert (odd.width, odd.height) == (150, 112)
    big = video.make_proxy(clips["ident"], tmp_path / "big.mp4", fps=15.0, max_edge=4000, preset="veryfast")
    assert (big.width, big.height) == (320, 240)
    from_odd = video.make_proxy(clips["odd"], tmp_path / "from_odd.mp4", fps=15.0, preset="veryfast")
    assert (from_odd.width, from_odd.height) == (160, 120) and ffprobe_value(tmp_path / "from_odd.mp4", "stream=pix_fmt") == "yuv420p"


def test_make_proxy_failure_leaves_nothing(tmp_path: Path) -> None:
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)
    with pytest.raises(video.VideoError):
        video.make_proxy(bad, tmp_path / "out.mp4")
    assert not list(tmp_path.glob("*out*")) and not list(tmp_path.glob(".*"))


def test_pts_map_and_round_trip(ident_proxy: Path) -> None:
    pts = video.pts_map(ident_proxy)
    assert len(pts) == 45
    assert all(b > a for a, b in zip(pts, pts[1:]))
    for i, t in enumerate(pts):
        assert t == pytest.approx(i / 15, abs=1e-6)
        assert video.pts_to_proxy_frame(pts, t) == i
        assert video.proxy_frame_to_pts(pts, i) == t
    assert video.pts_to_proxy_frame(pts, pts[7] + 0.4 / 15) == 7
    assert video.pts_to_proxy_frame(pts, pts[7] + 0.6 / 15) == 8
    assert video.pts_to_proxy_frame(pts, pts[7] + 0.5 / 15) in (7, 8)
    assert video.pts_to_proxy_frame(pts, -1.0) == 0
    assert video.pts_to_proxy_frame(pts, 100.0) == 44
    with pytest.raises(IndexError):
        video.proxy_frame_to_pts(pts, 45)
    with pytest.raises(IndexError):
        video.proxy_frame_to_pts(pts, -1)
    with pytest.raises(ValueError):
        video.pts_to_proxy_frame([], 0.0)
    with pytest.raises(ValueError):
        video.proxy_frame_to_pts([], 0)


@pytest.mark.parametrize("fps", [15.0, 10.0])
def test_vfr_proxy_maps_within_one_frame(
    clips: dict[str, Path], vfr_frames: tuple[list[float], np.ndarray], tmp_path: Path, fps: float
) -> None:
    """The contract test: on a jittered, gapped source each proxy frame maps to the source frame on screen at its pts."""
    # k started at or before pts[i] and k + 1 had not, so the map is within one source frame; the full-resolution read
    # at pts[i] is k; pts_to_proxy_frame is never more than half a proxy frame off (the 33 ms of 5.9 at 15 fps)
    src_pts, src_means = vfr_frames
    assert video.probe(clips["vfr"]).vfr
    assert float(np.diff(src_pts).max()) > 1.0 / 15  # gaps longer than a 15 fps proxy frame: some frames show twice
    proxy = tmp_path / f"vfr{int(fps)}.mp4"
    info = video.make_proxy(clips["vfr"], proxy, fps=fps, max_edge=160, preset="veryfast")
    pts = video.pts_map(proxy)
    assert len(pts) == info.nb_frames
    assert pts == pytest.approx([i / fps for i in range(len(pts))], abs=1e-6)
    for i, _, rgb in video.iter_frames(proxy):
        k = source_index(rgb, src_means)
        assert src_pts[k] <= pts[i] + 1e-6
        if k + 1 < len(src_pts):
            assert pts[i] < src_pts[k + 1] + 1e-6
        if i % 5 == 0:
            assert source_index(video.read_frame_at(clips["vfr"], pts[i]), src_means) == k
    for s in src_pts:
        f = video.pts_to_proxy_frame(pts, s)
        assert abs(pts[f] - s) == pytest.approx(min(abs(p - s) for p in pts), abs=1e-9)
        if pts[0] <= s <= pts[-1]:
            assert abs(pts[f] - s) <= 0.5 / fps + 1e-6


@pytest.mark.parametrize("name", ["us_cfr", "us_jitter", "tenth_us", "ts"])
def test_proxy_pts_reads_back_the_shown_frame_on_fine_timescales(
    fine_clips: dict[str, Path], ident_means: np.ndarray, tmp_path: Path, name: str
) -> None:
    """Where 6 decimals cannot tell a frame from a slot, every proxy pts still reads back the frame that slot shows."""
    src = fine_clips[name]
    times = video.frame_times(src)
    assert len(times) == 90 and times == video.pts_map(src) and times[0] == 0.0
    for fps in (15.0, 10.0):
        proxy = tmp_path / f"{name}.{int(fps)}.mp4"
        video.make_proxy(src, proxy, fps=fps, max_edge=160, preset="veryfast")
        pts = video.pts_map(proxy)
        shown = [source_index(rgb, ident_means) for _, _, rgb in video.iter_frames(proxy)]
        read = [source_index(rgb, ident_means) for _, rgb in video.iter_frames_at(src, pts)]
        assert read == shown
        assert [source_index(rgb, ident_means) for _, _, rgb in video.iter_frames(src, fps=fps)] == shown


def test_proxy_record_writes_run_relative_paths(clips: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path / "runs")
    run_id = "01TESTRUNVIDEO"
    proxy = paths.proxy_path(run_id, "pickup")
    video.make_proxy(clips["ident"], proxy, fps=15.0, max_edge=160, preset="veryfast")
    rec = video.proxy_record(run_id, "pickup", proxy)
    assert isinstance(rec, contracts.ProxyRecord)
    assert rec.path == "media/pickup.proxy.mp4" and rec.pts_map_path == "media/pickup.pts.json"
    assert (rec.width, rec.height, rec.fps, rec.frames) == (160, 120, 15.0, 45)
    assert rec.sha256 == video.sha256_file(proxy) == hashlib.sha256(proxy.read_bytes()).hexdigest()
    written = json.loads((paths.run_dir(run_id) / rec.pts_map_path).read_text(encoding="utf-8"))
    assert written == video.pts_map(proxy) and len(written) == rec.frames
    assert contracts.ProxyRecord.model_validate_json(rec.to_json()) == rec


# Frames -------------------------------------------------------------------------------------------------------------


def test_read_frame_at_cfr(clips: dict[str, Path], ident_means: np.ndarray) -> None:
    src = clips["ident"]
    for k in (0, 1, 9, 10, 11, 44, 89):
        rgb = video.read_frame_at(src, k / 30)
        assert rgb.shape == (240, 320, 3) and rgb.dtype == np.uint8
        assert source_index(rgb, ident_means) == k
    assert source_index(video.read_frame_at(src, 44 / 30 + 0.010), ident_means) == 44  # frame 44 is on screen
    assert source_index(video.read_frame_at(src, 44 / 30 + 0.030), ident_means) == 44  # still, until frame 45's pts
    assert source_index(video.read_frame_at(src, 45 / 30 - 0.001), ident_means) == 44
    assert source_index(video.read_frame_at(src, 45 / 30), ident_means) == 45
    assert source_index(video.read_frame_at(src, 2.99), ident_means) == 89  # the last frame, within one interval
    assert source_index(video.read_frame_at(src, -0.01), ident_means) == 0  # just before the first frame
    for bad in (3.5, -0.5, float("nan")):
        with pytest.raises(video.VideoError):
            video.read_frame_at(src, bad)


def test_long_last_frame_keeps_its_proxy_slots(long_last: Path, ident_means: np.ndarray, tmp_path: Path) -> None:
    """The fps filter fills slots to the end of a long last frame, and read_frame_at answers at those pts too."""
    times = video.frame_times(long_last)
    assert len(times) == 31 and times[-1] == 1.0
    proxy = tmp_path / "tail.proxy.mp4"
    video.make_proxy(long_last, proxy, fps=15.0, max_edge=160, preset="veryfast")
    pts = video.pts_map(proxy)
    assert pts[-1] > times[-1] + (times[-1] - times[-2])  # beyond one usual frame gap
    shown = [source_index(rgb, ident_means) for _, _, rgb in video.iter_frames(proxy)]
    assert [source_index(rgb, ident_means) for _, rgb in video.iter_frames_at(long_last, pts)] == shown
    assert shown[-1] == 30 and source_index(video.read_frame_at(long_last, pts[-1]), ident_means) == 30
    with pytest.raises(video.VideoError):
        video.read_frame_at(long_last, 1.0 + 0.125 + 0.05)  # past the end of the last frame


def test_read_frame_at_is_exact_on_vfr(clips: dict[str, Path], vfr_frames: tuple[list[float], np.ndarray]) -> None:
    src_pts, src_means = vfr_frames
    for k in range(0, 90, 5):
        assert source_index(video.read_frame_at(clips["vfr"], src_pts[k]), src_means) == k
        if k + 1 < 90:
            gap = src_pts[k + 1] - src_pts[k]
            assert source_index(video.read_frame_at(clips["vfr"], src_pts[k] + 0.9 * gap), src_means) == k  # on screen
        if k > 0:
            gap = src_pts[k] - src_pts[k - 1]
            assert source_index(video.read_frame_at(clips["vfr"], src_pts[k] - 0.1 * gap), src_means) == k - 1


def test_read_frame_at_matches_a_sequential_decode() -> None:
    """On a real x264 file with B-frames the seek gives the very frame a front-to-back decode gives, bit for bit."""
    times = video.frame_times(PINGPONG)
    assert times == video.pts_map(PINGPONG)
    wanted = {0, 1, 2, 47, 48, 99, 139, 140}
    frames = frames_by_index(PINGPONG, wanted)
    for k in sorted(wanted):
        assert np.array_equal(video.read_frame_at(PINGPONG, times[k]), frames[k])


def test_late_video_start_keeps_proxy_and_source_aligned(clips: dict[str, Path], ident_means: np.ndarray, tmp_path: Path) -> None:
    """Audio from 0 s, video from 0.05 s: the proxy repeats the first frame into the empty slot and still reads back."""
    src = clips["late"]
    info = video.probe(src)
    assert info.has_audio and info.start_time == 0.0 and info.nb_frames == 30
    times = video.frame_times(src)
    assert times[0] == pytest.approx(0.05) and len(times) == 30
    proxy = tmp_path / "late.proxy.mp4"
    out = video.make_proxy(src, proxy, fps=15.0, max_edge=160, preset="veryfast")
    pts = video.pts_map(proxy)
    assert out.nb_frames == len(pts) == 16 and pts[0] == 0.0
    shown = [source_index(rgb, ident_means) for _, _, rgb in video.iter_frames(proxy)]
    assert shown[:4] == [0, 0, 2, 4]
    for i in range(0, 16, 3):
        assert source_index(video.read_frame_at(src, pts[i]), ident_means) == shown[i]
    assert [t for _, t, _ in video.iter_frames(src)] == times


def test_edit_list_discards_are_not_frames(clips: dict[str, Path], ident_means: np.ndarray) -> None:
    """A stream copy from 0.5 s keeps the GOP from frame 10 but its edit list starts the picture at frame 15."""
    src = clips["cut"]
    flags = packet_flags(src)
    assert sum("D" in f for f in flags) == 5 and len(flags) == 80
    times = video.frame_times(src)
    assert len(times) == 75 and times[0] == 0.0
    assert times == video.pts_map(src)
    frames = list(video.iter_frames(src))
    assert len(frames) == 75 and source_index(frames[0][2], ident_means) == 15
    assert [t for _, t, _ in frames] == times
    assert source_index(video.read_frame_at(src, 0.0), ident_means) == 15
    assert source_index(video.read_frame_at(src, 1.0), ident_means) == 45
    assert video.clip_span(src, pts_center=1.0, frames=5)[0] == 28


def test_open_gop_start_leaves_out_the_frames_no_decode_outputs(open_gop_cut: Path, ident_means: np.ndarray, tmp_path: Path) -> None:
    """A copy that starts on an HEVC CRA: its kept leading frames are not frames, so every label and seek still holds."""
    src = open_gop_cut
    flags = packet_flags(src)
    times = video.frame_times(src)
    assert "K" in flags[0] and sum("D" not in f for f in flags) > len(times)  # kept by the edit list, dropped by the decoder
    assert times == video.pts_map(src) and times[0] > 0.0 and video._frame_index(src).key_at[0] == 0
    frames = [(t, rgb) for _, t, rgb in video.iter_frames(src)]
    assert [t for t, _ in frames] == times
    first = source_index(frames[0][1], ident_means)  # the CRA, then every later frame in order
    assert [source_index(rgb, ident_means) for _, rgb in frames[:30]] == list(range(first, first + 30))
    for k in (0, 1, 2, 3, 4, 14, 15, 16, 40):
        assert np.array_equal(video.read_frame_at(src, times[k]), frames[k][1])
    ks = (0, 1, 2, 31)
    assert all(np.array_equal(rgb, frames[k][1]) for (_, rgb), k in zip(video.iter_frames_at(src, [times[k] for k in ks]), ks))
    window = [[40.0, 0.0, 280.0, 240.0]] * 5
    for f0 in (0, 1, 2):
        got = list(video._clip_frames(src, f0, window, 240, None, 20))
        assert all(np.array_equal(g, video._cut_window(frames[f0 + i][1], window[i], 240)) for i, g in enumerate(got))
    assert video.clip_span(src, pts_center=times[2], frames=5) == (0, times[:5])
    proxy = tmp_path / "cut.proxy.mp4"
    video.make_proxy(src, proxy, fps=15.0, max_edge=160, preset="veryfast")
    pts = video.pts_map(proxy)
    shown = [source_index(rgb, ident_means) for _, _, rgb in video.iter_frames(proxy)]
    assert shown[0] == first  # the slots before the CRA repeat it
    assert [source_index(rgb, ident_means) for _, rgb in video.iter_frames_at(src, pts[:10])] == shown[:10]


def test_iter_frames(clips: dict[str, Path], ident_means: np.ndarray) -> None:
    src = clips["ident"]
    frames = list(video.iter_frames(src))
    assert len(frames) == 90
    assert [t for _, t, _ in frames] == video.pts_map(src)  # ffprobe frames as the oracle
    assert [i for i, _, _ in frames] == list(range(90))
    assert all(source_index(rgb, ident_means) == i for i, _, rgb in frames[::9])
    resampled = list(video.iter_frames(src, fps=10.0))
    assert len(resampled) == 30
    assert all(t == pytest.approx(i / 10) and source_index(rgb, ident_means) == 3 * i for i, t, rgb in resampled)
    small = list(video.iter_frames(src, max_edge=160))
    assert len(small) == 90 and small[0][2].shape == (120, 160, 3)
    both = list(video.iter_frames(src, fps=5.0, max_edge=100))
    assert len(both) == 15 and both[0][2].shape == (74, 100, 3)  # 75 rows rounded to an even count
    odd = list(video.iter_frames(clips["odd"]))
    assert len(odd) == 30 and odd[0][2].shape == (121, 161, 3)  # native size, nothing rounded without max_edge
    assert video.read_frame_at(clips["odd"], 0.5).shape == (121, 161, 3)


def test_iter_frames_pingpong_matches_ffprobe_frames() -> None:
    times = []
    for i, t, rgb in video.iter_frames(PINGPONG):
        assert rgb.shape == (720, 1280, 3)
        times.append(t)
    assert len(times) == 141
    assert times == video.pts_map(PINGPONG)  # B-frames in presentation order


def test_seek_groups_and_decode_start() -> None:
    keys = list(range(0, 100, 10))
    assert [video._decode_start(keys, k) for k in (0, 9, 10, 11, 99)] == [0, 0, 10, 10, 90]
    assert video._decode_start([], 57) == 0 and video._decode_start([5], 3) == 0
    assert video._seek_groups([3, 5, 47, 48, 85], keys) == [[3, 5], [47, 48], [85]]
    assert video._seek_groups([12, 21], keys) == [[12, 21]]  # the keyframe at 20 would save only 8 frames
    assert video._seek_groups([12, 41], keys) == [[12], [41]]
    assert video._seek_groups([0, 99], []) == [[0, 99]]  # no keyframe flags: one decode
    assert video._seek_groups([], keys) == []


def test_frame_index_keyframes(clips: dict[str, Path]) -> None:
    assert video._frame_index(clips["ident"]).key_at == list(range(0, 90, 10))
    cut = video._frame_index(clips["cut"])  # the GOP's keyframe is discarded; the next one is shown frame 5
    assert cut.key_at[:2] == [5, 15] and len(cut.times) == 75


def test_iter_frames_at(clips: dict[str, Path], ident_means: np.ndarray, monkeypatch: pytest.MonkeyPatch) -> None:
    src = clips["ident"]  # a keyframe every 10 frames
    times = video.frame_times(src)
    asked = [times[85], times[3], times[47] + 0.01, times[5], times[3], times[48], times[47] + 0.01, times[48] + 0.02]
    starts: list[Any] = []
    real = video._raw_frames

    def spy(*args: Any, **kwargs: Any) -> Any:
        starts.append(kwargs.get("ss"))
        return real(*args, **kwargs)

    monkeypatch.setattr(video, "_raw_frames", spy)
    out = list(video.iter_frames_at(src, asked))
    monkeypatch.undo()
    assert [t for t, _ in out] == sorted(set(asked))
    assert [source_index(rgb, ident_means) for _, rgb in out] == [3, 5, 47, 48, 48, 85]
    assert len(starts) == 3  # frames 3 and 5, 47 and 48, then 85
    assert out[3][1] is out[4][1]  # two times on one frame share the decode
    for t, rgb in out:
        assert np.array_equal(rgb, video.read_frame_at(src, t))
    with pytest.raises(video.VideoError):
        list(video.iter_frames_at(src, [1.0, 9.0]))
    assert list(video.iter_frames_at(src, [])) == []


def test_iter_frames_at_closes_ffmpeg_when_the_caller_stops(clips: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    procs: list[subprocess.Popen[bytes]] = []
    real = video._Decode.__init__

    def keep(self: Any, cmd: list[str]) -> None:
        real(self, cmd)
        procs.append(self.proc)

    monkeypatch.setattr(video._Decode, "__init__", keep)
    gen = video.iter_frames_at(clips["ident"], [0.1, 0.5, 1.0, 2.0])
    next(gen)
    gen.close()
    assert procs and all(p.poll() is not None for p in procs)


def test_sharpness() -> None:
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    flat = np.full((64, 64, 3), 100, dtype=np.uint8)
    blurred = cv2.GaussianBlur(noise, (9, 9), 3)
    assert video.sharpness(flat) == 0.0
    assert video.sharpness(noise) > video.sharpness(blurred) > 0.0
    assert video.sharpness(cv2.cvtColor(noise, cv2.COLOR_RGB2GRAY)) == pytest.approx(video.sharpness(noise))
    half = np.concatenate([flat[:, :32], noise[:, :32]], axis=1)
    left, right = np.zeros((64, 64), bool), np.zeros((64, 64), bool)
    left[:, :24], right[:, 40:] = True, True
    assert video.sharpness(half, mask=left) == 0.0
    assert video.sharpness(half, mask=right) > 0.0
    assert video.sharpness(half, mask=np.zeros((64, 64), bool)) == 0.0


def test_sha256_file(tmp_path: Path) -> None:
    data = np.random.default_rng(1).integers(0, 256, 3 * 1024 * 1024, dtype=np.uint8).tobytes()
    big = tmp_path / "big.bin"
    big.write_bytes(data)
    assert video.sha256_file(big) == hashlib.sha256(data).hexdigest()
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    assert video.sha256_file(empty) == hashlib.sha256(b"").hexdigest()


# WAV --------------------------------------------------------------------------------------------------------------


def test_extract_wav(clips: dict[str, Path], tmp_path: Path) -> None:
    dst = tmp_path / "nested" / "pickup.wav"
    assert video.extract_wav(clips["ident"], dst) == dst
    with wave.open(str(dst), "rb") as w:
        assert (w.getnchannels(), w.getframerate(), w.getsampwidth(), w.getcomptype()) == (1, 24000, 2, "NONE")
        assert w.getnframes() / w.getframerate() == pytest.approx(3.0, abs=0.05)
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64)
    head = dst.read_bytes()[:64]
    assert head[:4] == b"RIFF" and head[8:12] == b"WAVE" and head[36:40] == b"data"  # no LIST/INFO chunk
    assert b"LIST" not in head
    spectrum = np.abs(np.fft.rfft(samples))
    assert np.fft.rfftfreq(samples.size, 1 / 24000)[spectrum.argmax()] == pytest.approx(440.0, abs=2.0)
    other = video.extract_wav(clips["ident"], tmp_path / "16k.wav", rate=16000)
    with wave.open(str(other), "rb") as w:
        assert w.getframerate() == 16000
    assert not list(tmp_path.rglob(".*.part*"))


def test_extract_wav_starts_at_the_file_start(tmp_path: Path) -> None:
    """Audio that starts 0.5 s after the video is padded with silence, so a Voice time is a file time."""
    src = tmp_path / "late_audio.mp4"
    ffmpeg(
        "-f", "lavfi", "-i", IDENT_SRC, "-itsoffset", "0.5", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=2.5",
        "-map", "0:v", "-map", "1:a", *LOSSLESS, "-c:a", "aac", "-b:a", "64k", str(src),
    )
    start = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=start_time", "-of", "csv=p=0", str(src)],
        check=True, capture_output=True, text=True,
    ).stdout
    assert float(start) > 0.4 and video.probe(src).start_time == 0.0  # the audio really starts late
    wav = video.extract_wav(src, tmp_path / "late.wav")
    with wave.open(str(wav), "rb") as w:
        rate = w.getframerate()
        samples = np.abs(np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64))
    assert int(np.argmax(samples > 0.1 * samples.max())) / rate == pytest.approx(0.5, abs=0.01)
    assert samples.size / rate == pytest.approx(3.0, abs=0.05)
    assert wav.read_bytes()[36:40] == b"data"


def test_extract_wav_without_audio_raises(clips: dict[str, Path], tmp_path: Path) -> None:
    with pytest.raises(video.NoAudioError):
        video.extract_wav(clips["vfr"], tmp_path / "a.wav")
    with pytest.raises(video.VideoError):  # NoAudioError is a VideoError
        video.extract_wav(PINGPONG, tmp_path / "b.wav")
    assert not (tmp_path / "a.wav").exists() and not (tmp_path / "b.wav").exists()


# Stays Put clips ----------------------------------------------------------------------------------------------------


def test_clip_span(clips: dict[str, Path]) -> None:
    src = clips["square"]
    times = video.pts_map(src)
    f0, span = video.clip_span(src, pts_center=1.0, frames=9)
    assert f0 == 26 and span == times[26:35]
    f0, span = video.clip_span(src, pts_center=1.0 + 0.020, frames=9)  # frame 30 is on screen until 31/30
    assert f0 == 26
    f0, span = video.clip_span(src, pts_center=1.0 + 0.034, frames=9)  # frame 31 is on screen
    assert f0 == 27 and span == times[27:36]
    for centre in (0.05, 1.95, 5.0):
        with pytest.raises(video.ClipOutOfRange):
            video.clip_span(src, pts_center=centre, frames=9)
    with pytest.raises(ValueError):
        video.clip_span(src, pts_center=1.0, frames=0)
    f0, span = video.clip_span(PINGPONG, pts_center=2.5, frames=11)  # frame 59 (2.461 s) is on screen at 2.5 s
    assert f0 == 54 and span == video.pts_map(PINGPONG)[54:65]


def test_union_crop() -> None:
    assert video._union_crop([[100.3, 50.7, 300.2, 250.9]], 3840, 2160) == (92, 42, 218, 218)
    assert video._union_crop([[100, 100, 300, 300], [500, 120, 700, 320]], 3840, 2160) == (92, 92, 616, 236)
    assert video._union_crop([[-50, -20, 150, 180]], 3840, 2160) == (0, 0, 158, 188)
    assert video._union_crop([[3700, 2000, 3900, 2200]], 3840, 2160) == (3692, 1992, 148, 168)
    assert video._union_crop([[4000, 10, 4200, 210]], 3840, 2160) is None  # wholly outside: black, no crop
    assert video._union_crop([[0, 0, 3840, 2160]], 3840, 2160) is None  # nothing to save
    assert video._union_crop([[10, 10, 50, 50]], 161, 121) is None  # odd frame: no chroma-aligned crop


@pytest.mark.parametrize(
    "name, size, windows",
    [
        ("square", 252, [[-60.0, 50.0, 192.0, 302.0], [-40.0, 60.0, 212.0, 312.0], [-20.0, 70.0, 232.0, 322.0]]),
        ("portrait", 300, [[-50.0, -40.0, 250.0, 260.0], [0.0, 10.0, 300.0, 310.0], [40.0, 60.0, 340.0, 360.0]]),
        ("portrait", 256, [[900.0, 1700.0, 1200.0, 2000.0], [880.0, 1690.0, 1180.0, 1990.0], [860.0, 1680.0, 1160.0, 1980.0]]),
    ],
)
def test_decode_crop_matches_the_full_frame(clips: dict[str, Path], name: str, size: int, windows: list[list[float]]) -> None:
    """The decode crop is only a speed-up: clip frames equal windows cut from full, rotated frames, off-frame parts too."""
    src = clips[name]
    info = video.probe(src)
    assert video._union_crop(windows, info.width, info.height) is not None  # the crop path runs
    times = video.frame_times(src)
    f0, _ = video.clip_span(src, pts_center=times[len(times) // 2], frames=3)
    got = list(video._clip_frames(src, f0, windows, size, None, 20))
    full = {i: rgb for i, _, rgb in video.iter_frames(src) if f0 <= i < f0 + 3}
    assert len(got) == 3
    for i in range(3):
        want = video._cut_window(full[f0 + i], windows[i], size)
        assert np.abs(got[i].astype(np.int16) - want.astype(np.int16)).max() <= 2


def test_cut_clip_round_trips_a_known_box(clips: dict[str, Path], tmp_path: Path) -> None:
    """The contract test: a known box lands where the window says in every frame, and clip frame i is source f0 + i."""
    src, dst = clips["square"], tmp_path / "clip.mp4"
    frames, size, window_px = 9, 1008, 252  # 252 source px onto 1008: scale 4 exactly
    f0, span = video.clip_span(src, pts_center=1.0, frames=frames)
    windows: list[list[float]] = []
    for i in range(frames):
        cx, cy = 230.0 + (f0 + i) + 2 * i, 175.0 - i  # the box centre, plus a drift so every window differs
        windows.append([cx - window_px / 2, cy - window_px / 2, cx + window_px / 2, cy + window_px / 2])
    rec = video.cut_clip(src, dst, pts_center=1.0, frames=frames, windows_src=windows)
    assert isinstance(rec, contracts.ClipRecord)
    assert rec.f0_src == f0 == 26 and rec.frames == frames
    assert (rec.w, rec.h, rec.window_px) == (size, size, window_px)
    assert rec.windows_src == windows and rec.path == str(dst)
    assert rec.sha256 == video.sha256_file(dst)
    out = video.probe(dst)
    assert (out.width, out.height, out.nb_frames) == (size, size, frames)
    assert out.fps == pytest.approx(30.0) and not out.has_audio and out.rotation == 0
    decoded = list(video.iter_frames(dst))
    assert len(decoded) == frames
    for i, _, rgb in decoded:
        expected = video._box_to_window(square_box(f0 + i), windows[i], size)
        found = blob(rgb)
        assert found is not None
        assert found == pytest.approx(expected, abs=4.0)  # one source pixel at scale 4
    assert ffprobe_value(dst, "stream=pix_fmt") == "yuv420p"
    assert [ffprobe_value(dst, f"stream={k}") for k in ("color_space", "color_transfer", "color_primaries", "color_range")] == ["bt709", "bt709", "bt709", "tv"]


def test_cut_clip_record_path_is_run_relative(clips: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path / "runs")
    dst = paths.clip_path("01TESTCLIP", "pickup", "pickup_hood_kf1_m1")
    rec = video.cut_clip(clips["square"], dst, pts_center=1.0, frames=3, windows_src=[[104.0, 49.0, 356.0, 301.0]] * 3, size=252)
    assert rec.path == "clips/pickup/pickup_hood_kf1_m1.mp4"
    assert contracts.ClipRecord.model_validate_json(rec.to_json()) == rec


def test_cut_clip_rejects_bad_requests(clips: dict[str, Path], tmp_path: Path) -> None:
    src = clips["square"]
    window = [100.0, 50.0, 352.0, 302.0]
    with pytest.raises(video.ClipOutOfRange):
        video.cut_clip(src, tmp_path / "early.mp4", pts_center=0.05, frames=9, windows_src=[window] * 9, size=252)
    with pytest.raises(ValueError):
        video.cut_clip(src, tmp_path / "n.mp4", pts_center=1.0, frames=9, windows_src=[window] * 8, size=252)
    with pytest.raises(ValueError):
        video.cut_clip(src, tmp_path / "b.mp4", pts_center=1.0, frames=3, windows_src=[window] * 3, blur_boxes=[[]] * 2, size=252)
    with pytest.raises(ValueError):
        video.cut_clip(src, tmp_path / "z.mp4", pts_center=1.0, frames=0, windows_src=[], size=252)
    with pytest.raises(ValueError, match="not square"):
        video.cut_clip(src, tmp_path / "r.mp4", pts_center=1.0, frames=1, windows_src=[[100.0, 50.0, 352.0, 200.0]], size=252)
    with pytest.raises(ValueError, match="empty"):
        video.cut_clip(src, tmp_path / "e.mp4", pts_center=1.0, frames=1, windows_src=[[100.0, 50.0, 100.0, 50.0]], size=252)
    assert not list(tmp_path.glob("*.mp4")) and not list(tmp_path.glob(".*.part*"))


def test_cut_clip_pixelates_only_the_given_frames(clips: dict[str, Path], tmp_path: Path) -> None:
    src = clips["square"]
    frames, size = 3, 252
    f0, _ = video.clip_span(src, pts_center=1.0, frames=frames)
    window = [104.0, 49.0, 356.0, 301.0]  # 252 px at scale 1 around the box
    plain = video.cut_clip(src, tmp_path / "plain.mp4", pts_center=1.0, frames=frames, windows_src=[window] * frames, size=size)
    box = square_box(f0 + 1)
    blur_box = [box[0] - 20, box[1] - 20, box[0] + 20, box[3] + 20]  # straddles the box's left edge
    blurred = video.cut_clip(
        src, tmp_path / "blurred.mp4", pts_center=1.0, frames=frames, windows_src=[window] * frames, size=size,
        blur_boxes=[[], [blur_box], []], block_px=20,
    )
    assert plain.f0_src == blurred.f0_src and plain.sha256 != blurred.sha256
    plain_frames = [rgb for _, _, rgb in video.iter_frames(tmp_path / "plain.mp4")]
    blurred_frames = [rgb for _, _, rgb in video.iter_frames(tmp_path / "blurred.mp4")]
    for i in (0, 2):
        assert mean_abs_diff(plain_frames[i], blurred_frames[i]) < 3.0
    x1, y1, x2, y2 = (int(round(v)) for v in video._box_to_window(blur_box, window, size))
    inside = (slice(y1, y2), slice(x1, x2))
    assert mean_abs_diff(plain_frames[1][inside], blurred_frames[1][inside]) > 10.0
    outside = np.ones(plain_frames[1].shape[:2], bool)
    outside[y1 - 2 : y2 + 2, x1 - 2 : x2 + 2] = False
    assert mean_abs_diff(plain_frames[1][outside], blurred_frames[1][outside]) < 3.0


def test_clip_blur_is_privacy_pixelate(clips: dict[str, Path]) -> None:
    """The clip's blur is privacy.pixelate in clip pixels, with the source-scale block grown by the window's zoom."""
    from walkaround import privacy

    src = clips["square"]
    f0, _ = video.clip_span(src, pts_center=1.0, frames=3)
    window = [104.0, 49.0, 356.0, 301.0]  # 252 source px onto 504: zoom 2, so 20 px blocks become 40
    boxes = [[[180.0, 130.0, 240.0, 210.0]], [], [[0.0, 0.0, 140.0, 90.0], [300.0, 250.0, 380.0, 330.0]]]
    plain = list(video._clip_frames(src, f0, [window] * 3, 504, None, 20))
    blurred = list(video._clip_frames(src, f0, [window] * 3, 504, boxes, 20))
    for i in range(3):
        regions = [video._box_to_window(b, window, 504) for b in boxes[i]]
        want = privacy.pixelate(plain[i], regions, block_px=40) if regions else plain[i]
        assert np.array_equal(blurred[i], want)
    assert not np.array_equal(blurred[0], plain[0]) and np.array_equal(blurred[1], plain[1])


def test_cut_clip_on_pingpong(tmp_path: Path) -> None:
    """A mark-centred square window on real footage: 31 frames around 2.5 s, 400 source px onto 1008 px."""
    f0, span = video.clip_span(PINGPONG, pts_center=2.5, frames=31)
    side, cx, cy = 400.0, 640.0, 360.0
    windows = [[cx - side / 2 + 2 * i, cy - side / 2, cx + side / 2 + 2 * i, cy + side / 2] for i in range(31)]
    dst = tmp_path / "pp.mp4"
    rec = video.cut_clip(PINGPONG, dst, pts_center=2.5, frames=31, windows_src=windows)
    assert (rec.w, rec.h, rec.frames, rec.f0_src, rec.window_px) == (1008, 1008, 31, f0, 400)
    out = video.probe(dst)
    assert out.fps == pytest.approx(24000 / 1001, abs=1e-3) and out.nb_frames == 31 and not out.has_audio
    assert dst.stat().st_size < contracts.load_thresholds()["media"]["inline_data_url_max_bytes"]
    wanted = {0, 15, 30}
    decoded = frames_by_index(dst, wanted)
    for i in sorted(wanted):
        want = video._cut_window(video.read_frame_at(PINGPONG, span[i]), windows[i], 1008)
        assert mean_abs_diff(decoded[i], want) < 3.0


def test_render_blurred_proxy(clips: dict[str, Path], tmp_path: Path) -> None:
    proxy = tmp_path / "square.proxy.mp4"
    video.make_proxy(clips["square"], proxy, fps=15.0, max_edge=320, preset="veryfast")
    boxes = {0: [[90.0, 65.0, 140.0, 110.0]], 5: [[95.0, 65.0, 145.0, 110.0], [10.0, 10.0, 30.0, 30.0]]}  # proxy px
    dst = tmp_path / "square.proxy_blurred.mp4"
    info = video.render_blurred_proxy(proxy, dst, boxes, block_px=8, dilate_frac=0.1, dilate_px=2)
    assert (info.width, info.height, info.nb_frames) == (320, 240, 30)
    assert info.fps == pytest.approx(15.0) and not info.has_audio
    assert video.pts_map(dst) == video.pts_map(proxy)  # same frames at the same times: lanes still line up
    assert ffprobe_value(dst, "stream=color_space") == "bt709"
    before = [rgb for _, _, rgb in video.iter_frames(proxy)]
    after = [rgb for _, _, rgb in video.iter_frames(dst)]
    assert len(before) == len(after) == 30
    for f in (1, 2, 29):
        assert mean_abs_diff(before[f], after[f]) < 3.0
    for f, (x1, y1, x2, y2) in ((0, (90, 65, 140, 110)), (5, (95, 65, 145, 110))):
        assert mean_abs_diff(before[f][y1:y2, x1:x2], after[f][y1:y2, x1:x2]) > 10.0
        outside = np.ones((240, 320), bool)
        outside[y1 - 12 : y2 + 12, x1 - 12 : x2 + 12] = False
        outside[0:40, 0:40] = False
        assert mean_abs_diff(before[f][outside], after[f][outside]) < 3.0


def test_render_blurred_proxy_fails_closed_on_keys_that_name_no_frame(ident_proxy: Path, tmp_path: Path) -> None:
    box = [[10.0, 10.0, 40.0, 40.0]]
    dst = tmp_path / "blurred.mp4"
    for bad in ({"0": box}, {"44": box, 3: box}, {True: box}, {-1: box}):  # a JSON round trip turns keys into str
        with pytest.raises(ValueError):
            video.render_blurred_proxy(ident_proxy, dst, bad, block_px=8, dilate_frac=0.0, dilate_px=0)  # type: ignore[arg-type]
        assert not dst.exists() and not list(tmp_path.glob(".*"))
    # numpy ints are frame indices; keys past frame 44 blur frames that do not exist (gate_sam bridges past a lane's end)
    keys = {np.int64(3): box, 44: box, 45: box, 1000: []}
    video.render_blurred_proxy(ident_proxy, dst, keys, block_px=8, dilate_frac=0.0, dilate_px=0)  # type: ignore[arg-type]
    assert video.probe(dst).nb_frames == 45


# Stills and crops ---------------------------------------------------------------------------------------------------


def test_write_jpeg(tmp_path: Path) -> None:
    ys, xs = np.mgrid[0:48, 0:64]
    rgb = np.stack([(xs * 4) % 256, (ys * 5) % 256, np.full_like(xs, 128)], axis=-1).astype(np.uint8)
    dst = tmp_path / "a" / "b.jpg"
    sha = video.write_jpeg(rgb, dst, quality=95)
    assert dst.is_file() and sha == video.sha256_file(dst)
    with Image.open(dst) as img:
        assert img.format == "JPEG" and img.size == (64, 48) and img.mode == "RGB"
        assert img.info.get("exif") is None
        back = np.asarray(img.convert("RGB"))
    assert mean_abs_diff(rgb, back) < 4.0
    assert video.write_jpeg(rgb.astype(np.int64), tmp_path / "c.jpg", quality=95) == sha  # same bytes from ints
    with pytest.raises(ValueError):
        video.write_jpeg(rgb.astype(np.float32) / 255, tmp_path / "f.jpg")
    with pytest.raises(ValueError):
        video.write_jpeg(rgb[..., 0], tmp_path / "g.jpg")


def test_write_png_mask(tmp_path: Path) -> None:
    ys, xs = np.mgrid[0:40, 0:70]
    mask = ((xs // 7 + ys // 5) % 2 == 0)
    dst = tmp_path / "m.png"
    sha = video.write_png_mask(mask, dst)
    assert sha == video.sha256_file(dst)
    with Image.open(dst) as img:
        assert img.format == "PNG" and img.mode == "1" and img.size == (70, 40)
        back = np.asarray(img)
    assert back.dtype == bool and np.array_equal(back, mask)
    assert video.write_png_mask(mask.astype(np.uint8) * 255, tmp_path / "m2.png") == sha  # any truthy raster


def test_cut_crop_maps_crop_pixels_to_source() -> None:
    ys, xs = np.mgrid[0:120, 0:200]
    rgb = np.stack([xs, ys, np.zeros_like(xs)], axis=-1).astype(np.uint8)  # R = x, G = y
    up = contracts.Crop(path="", w=100, h=50, scale=2.0, dx=30.0, dy=20.0)
    out = video.cut_crop(rgb, up)
    assert out.shape == (50, 100, 3)
    u, v = np.arange(100), np.arange(50)
    assert np.abs(out[25, :, 0].astype(float) - (30 + u / 2)).max() <= 1.0
    assert np.abs(out[:, 50, 1].astype(float) - (20 + v / 2)).max() <= 1.0
    down = contracts.Crop(path="", w=80, h=40, scale=0.5, dx=20.0, dy=10.0)
    out = video.cut_crop(rgb, down)
    assert out.shape == (40, 80, 3)
    u, v = np.arange(80), np.arange(40)
    assert np.abs(out[20, :, 0].astype(float) - (20 + u / 0.5)).max() <= 1.5
    assert np.abs(out[:, 40, 1].astype(float) - (10 + v / 0.5)).max() <= 1.5
    edge = contracts.Crop(path="", w=100, h=50, scale=2.0, dx=180.0, dy=20.0)
    out = video.cut_crop(rgb, edge)
    assert np.abs(out[25, :36, 0].astype(float) - (180 + u[:36] / 2)).max() <= 1.0
    assert not out[:, 42:].any()  # past the right edge of the frame is black


# The real clip --------------------------------------------------------------------------------------------------


def test_pingpong_proxy_path(tmp_path: Path) -> None:
    proxy = tmp_path / "pingpong.proxy.mp4"
    info = video.make_proxy(PINGPONG, proxy)  # the defaults of config/thresholds.json media: 15 fps, 960 px, medium, crf 20
    assert (info.width, info.height) == (960, 540)
    assert info.fps == pytest.approx(15.0)
    assert info.nb_frames in (88, 89)
    pts = video.pts_map(proxy)
    assert len(pts) == info.nb_frames
    assert video.pts_to_proxy_frame(pts, 2.0) == 30 and video.proxy_frame_to_pts(pts, 30) == pytest.approx(2.0)
    proxy_frames = frames_by_index(proxy, {0, 30, 70})
    for i, small in proxy_frames.items():
        frame = video.read_frame_at(PINGPONG, video.proxy_frame_to_pts(pts, i))
        assert frame.shape == (720, 1280, 3)
        assert mean_abs_diff(cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA), small) < 4.0
    batch = dict(video.iter_frames_at(PINGPONG, [video.proxy_frame_to_pts(pts, i) for i in proxy_frames]))
    for i in proxy_frames:
        assert np.array_equal(batch[pts[i]], video.read_frame_at(PINGPONG, pts[i]))
    assert video.sharpness(proxy_frames[30]) > 0.0
    assert len(video.sha256_file(proxy)) == 64
