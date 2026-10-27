"""Media lane (M1) tests: every walkaround.video function on clips ffmpeg builds here, plus data/test/pingpong.mp4.

Clips (built once per session with lavfi; no fixture files in the repo):
- ident: 320x240, 30 fps CFR, 3 s, luma 20 + 2 N so a decoded frame names its source index; a 440 Hz AAC sine
  track; a creation_time tag.
- vfr: the same picture with jittered and gapped timestamps (settb, setpts, -enc_time_base), no audio.
- late: the identity picture for 1 s with audio from 0 s and video from 0.05 s (the iPhone edit-list shape).
- portrait: stored 1920x1080 with a 90 degree display rotation tag, the top half of the stored picture white.
- square: 640x480, 30 fps, a white 60x50 box that moves one pixel per frame, for the clip round trip and blur.

The three contract tests of docs/CONTRACTS.md (proxy frame i maps to source pts within one frame on a synthetic VFR
clip; cut_clip round-trips a known box; a portrait source reports 540x960) are test_vfr_proxy_maps_within_one_frame,
test_cut_clip_round_trips_a_known_box and test_portrait_source_reports_540x960.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import wave
from datetime import datetime, timezone
from pathlib import Path

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


def square_box(n: int) -> list[float]:
    """The white box of square.mp4 at source frame n, half-open source pixels."""
    return [200.0 + n, 150.0, 260.0 + n, 200.0]


def ffmpeg(*args: str) -> None:
    subprocess.run([*FFMPEG, *args], check=True, capture_output=True)


def ffprobe_value(path: Path, entry: str) -> str:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", entry, "-of", "default=nw=1:nk=1", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    return out.strip()


def source_index(rgb: np.ndarray, means: np.ndarray) -> int:
    """Which identity-clip frame this is: the one whose mean luma is nearest."""
    return int(np.abs(means - rgb[..., 0].mean()).argmin())


def blob(rgb: np.ndarray) -> list[int] | None:
    """Half-open bounding box of the bright pixels, or None."""
    ys, xs = np.nonzero(rgb[..., 0] > 128)
    if not xs.size:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


@pytest.fixture(scope="session")
def clips(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("clips")
    names = ("ident.mp4", "vfr.mp4", "late.mp4", "stored.mp4", "portrait.mp4", "square.mp4")
    ident, vfr, late, stored, portrait, square = (d / n for n in names)
    ffmpeg(
        "-f", "lavfi", "-i", IDENT_SRC, "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=3",
        "-map", "0:v", "-map", "1:a", *LOSSLESS, "-c:a", "aac", "-b:a", "64k",
        "-metadata", "creation_time=2026-10-26T12:00:00Z", "-shortest", str(ident),
    )
    ffmpeg(
        "-f", "lavfi", "-i", IDENT_SRC, "-vf", "settb=1/90000,setpts=3000*(N+0.4*sin(1.7*N)+0.6*floor(N/10))",
        "-fps_mode", "passthrough", "-enc_time_base", "1/90000", "-video_track_timescale", "90000", *LOSSLESS, str(vfr),
    )
    ffmpeg(
        "-f", "lavfi", "-i", IDENT_SRC.replace("d=3", "d=1"), "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1.2",
        "-filter_complex", "[0:v]settb=1/90000,setpts=3000*N+4500[v]", "-map", "[v]", "-map", "1:a",
        "-fps_mode", "passthrough", "-enc_time_base", "1/90000", "-video_track_timescale", "90000", *LOSSLESS,
        "-c:a", "aac", "-b:a", "64k", str(late),
    )
    ffmpeg(
        "-f", "lavfi", "-i", "color=c=black:s=1920x1080:r=30:d=0.2,drawbox=x=0:y=0:w=1920:h=540:color=white:t=fill",
        *LOSSLESS, str(stored),
    )
    ffmpeg("-display_rotation", "-90", "-i", str(stored), "-c", "copy", str(portrait))
    ffmpeg("-f", "lavfi", "-i", SQUARE_SRC, "-c:v", "libx264", "-preset", "ultrafast", "-crf", "10", "-pix_fmt", "yuv420p", str(square))
    return {"ident": ident, "vfr": vfr, "late": late, "portrait": portrait, "square": square}


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


# Module hygiene ---------------------------------------------------------------------------------------------------


def test_module_is_ascii_and_offline() -> None:
    source = Path(video.__file__).read_bytes()
    assert source.isascii()
    assert Path(__file__).read_bytes().isascii()
    imports = re.findall(r"^(?:from|import)\s+(\S+)", source.decode(), flags=re.M)
    for banned in ("openai", "httpx", "ledger", ".ledger", "walkaround.ledger"):
        assert banned not in imports


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


def test_rotation_sign_convention() -> None:
    assert video._rotation({"side_data_list": [{"rotation": -90}]}) == 90  # iPhone portrait: ffprobe says -90
    assert video._rotation({"side_data_list": [{"rotation": 90}]}) == 270
    assert video._rotation({"side_data_list": [{"rotation": 180}]}) == 180
    assert video._rotation({"tags": {"rotate": "90"}}) == 90
    assert video._rotation({"side_data_list": [{"rotation": -90}], "tags": {"rotate": "270"}}) == 90
    assert video._rotation({}) == 0


def test_parse_time_and_iso_z() -> None:
    assert video.iso_z(video._parse_time("2026-10-26T12:00:00.000000Z")) == "2026-10-26T12:00:00Z"
    assert video.iso_z(video._parse_time("2026-10-26T14:00:00+02:00")) == "2026-10-26T12:00:00Z"
    assert video.iso_z(datetime(2026, 1, 2, 3, 4, 5)) == "2026-01-02T03:04:05Z"  # naive means UTC
    assert video._parse_time("garbage") is None and video._parse_time(None) is None


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


# Proxy and pts map ------------------------------------------------------------------------------------------------


def test_make_proxy_ident(clips: dict[str, Path], ident_means: np.ndarray, ident_proxy: Path) -> None:
    info = video.probe(ident_proxy)
    assert (info.width, info.height) == (160, 120)
    assert info.fps == pytest.approx(15.0) and info.nb_frames == 45
    assert not info.has_audio and info.rotation == 0 and info.creation_time is None  # no audio, no metadata
    assert ffprobe_value(ident_proxy, "stream=pix_fmt") == "yuv420p"
    data = ident_proxy.read_bytes()
    assert data.index(b"moov") < data.index(b"mdat")  # faststart
    pts = video.pts_map(ident_proxy)
    for i, t, rgb in video.iter_frames(ident_proxy):
        assert source_index(rgb, ident_means) == 2 * i  # 30 fps source, 15 fps proxy: every other frame
        assert t == pytest.approx(i / 15, abs=1e-6)
        if i % 11 == 0:  # the 4K read at a proxy pts is the picture the proxy showed
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
    """The contract test: on a jittered, gapped source every proxy frame shows the source frame on screen at its
    pts (started at or before it, the next one not yet), so the pts error is never positive, within one proxy
    frame except where the source itself has a gap longer than that, and bounded by the source's largest gap; that
    pts maps back to the frame, and the full-resolution frame read at the proxy pts is the picture shown."""
    src_pts, src_means = vfr_frames
    assert video.probe(clips["vfr"]).vfr
    proxy = tmp_path / f"vfr{int(fps)}.mp4"
    info = video.make_proxy(clips["vfr"], proxy, fps=fps, max_edge=160, preset="veryfast")
    pts = video.pts_map(proxy)
    assert len(pts) == info.nb_frames
    errors, back = [], []
    for i, _, rgb in video.iter_frames(proxy):
        k = source_index(rgb, src_means)
        assert src_pts[k] <= pts[i] + 1e-6
        if k + 1 < len(src_pts):
            assert pts[i] < src_pts[k + 1] + 1e-6
        errors.append(src_pts[k] - pts[i])
        back.append(video.pts_to_proxy_frame(pts, src_pts[k]) - i)
        if i % 5 == 0:
            assert source_index(video.read_frame_at(clips["vfr"], pts[i]), src_means) == k
    largest_gap = float(np.diff(src_pts).max())
    assert max(abs(e) for e in errors) <= max(largest_gap, 1.0 / fps) + 1e-6
    assert np.mean(np.abs(errors) <= 1.0 / fps + 1e-6) >= 0.9
    assert all(abs(b) <= 1 for b in back)
    assert np.mean(np.array(back) == 0) >= 0.9


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
    for k in (0, 1, 44, 89):
        rgb = video.read_frame_at(src, k / 30)
        assert rgb.shape == (240, 320, 3) and rgb.dtype == np.uint8
        assert source_index(rgb, ident_means) == k
    assert source_index(video.read_frame_at(src, 44 / 30 + 0.010), ident_means) == 44  # frame 44 is on screen
    assert source_index(video.read_frame_at(src, 44 / 30 + 0.030), ident_means) == 44  # still, until frame 45's pts
    assert source_index(video.read_frame_at(src, 45 / 30 - 0.001), ident_means) == 44
    assert source_index(video.read_frame_at(src, 45 / 30), ident_means) == 45
    assert source_index(video.read_frame_at(src, 2.99), ident_means) == 89  # the last frame, within one interval
    assert source_index(video.read_frame_at(src, -0.01), ident_means) == 0  # just before the first frame
    with pytest.raises(video.VideoError):
        video.read_frame_at(src, 3.5)
    with pytest.raises(video.VideoError):
        video.read_frame_at(src, -0.5)


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


def test_late_video_start_keeps_proxy_and_source_aligned(clips: dict[str, Path], ident_means: np.ndarray, tmp_path: Path) -> None:
    """Audio from 0 s, video from 0.05 s: the proxy pads the empty slot with the first frame, the pts map is the
    proxy's own timestamps, and the frame on screen at each proxy pts is the proxy's picture."""
    src = clips["late"]
    info = video.probe(src)
    assert info.has_audio and info.start_time == 0.0 and info.nb_frames == 30
    times = video._frame_times(src)
    assert times[0] == pytest.approx(0.05) and len(times) == 30
    proxy = tmp_path / "late.proxy.mp4"
    out = video.make_proxy(src, proxy, fps=15.0, max_edge=160, preset="veryfast")
    pts = video.pts_map(proxy)
    assert out.nb_frames == len(pts) == 16 and pts[0] == 0.0
    shown = [source_index(rgb, ident_means) for _, _, rgb in video.iter_frames(proxy)]
    assert shown[:4] == [0, 0, 2, 4]
    for i in range(0, 16, 3):
        assert source_index(video.read_frame_at(src, pts[i]), ident_means) == shown[i]
    assert [t for _, t, _ in video.iter_frames(src)] == pytest.approx(times, abs=1e-6)


def test_iter_frames(clips: dict[str, Path], ident_means: np.ndarray) -> None:
    src = clips["ident"]
    frames = list(video.iter_frames(src))
    assert len(frames) == 90
    assert [t for _, t, _ in frames] == pytest.approx(video.pts_map(src), abs=1e-6)  # ffprobe frames as the oracle
    assert [i for i, _, _ in frames] == list(range(90))
    assert all(source_index(rgb, ident_means) == i for i, _, rgb in frames[::9])
    resampled = list(video.iter_frames(src, fps=10.0))
    assert len(resampled) == 30
    assert all(t == pytest.approx(i / 10) and source_index(rgb, ident_means) == 3 * i for i, t, rgb in resampled)
    small = list(video.iter_frames(src, max_edge=160))
    assert len(small) == 90 and small[0][2].shape == (120, 160, 3)
    both = list(video.iter_frames(src, fps=5.0, max_edge=100))
    assert len(both) == 15 and both[0][2].shape == (74, 100, 3)  # 75 rows rounded to an even count


def test_iter_frames_pingpong_matches_ffprobe_frames() -> None:
    frames = list(video.iter_frames(PINGPONG))
    assert len(frames) == 141 and frames[0][2].shape == (720, 1280, 3)
    assert [t for _, t, _ in frames] == pytest.approx(video.pts_map(PINGPONG), abs=1e-6)  # B-frames in presentation order


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
        assert (w.getnchannels(), w.getframerate(), w.getsampwidth()) == (1, 24000, 2)
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
    with pytest.raises(video.ClipOutOfRange):
        video.clip_span(src, pts_center=0.05, frames=9)
    with pytest.raises(video.ClipOutOfRange):
        video.clip_span(src, pts_center=1.95, frames=9)
    with pytest.raises(ValueError):
        video.clip_span(src, pts_center=1.0, frames=0)
    f0, span = video.clip_span(PINGPONG, pts_center=2.5, frames=11)  # frame 59 (2.461 s) is on screen at 2.5 s
    assert f0 == 54 and span == video.pts_map(PINGPONG)[54:65]


def test_cut_clip_round_trips_a_known_box(clips: dict[str, Path], tmp_path: Path) -> None:
    """The contract test: a known source box lands where the window geometry says, in every clip frame, and
    clip frame i is source frame f0 + i (the box moves one source pixel per frame)."""
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


def test_render_blurred_proxy(clips: dict[str, Path], tmp_path: Path) -> None:
    proxy = tmp_path / "square.proxy.mp4"
    video.make_proxy(clips["square"], proxy, fps=15.0, max_edge=320, preset="veryfast")
    boxes = {0: [[90.0, 65.0, 140.0, 110.0]], 5: [[95.0, 65.0, 145.0, 110.0], [10.0, 10.0, 30.0, 30.0]]}  # proxy px
    dst = tmp_path / "square.proxy_blurred.mp4"
    info = video.render_blurred_proxy(proxy, dst, boxes, block_px=8, dilate_frac=0.1, dilate_px=2)
    assert (info.width, info.height, info.nb_frames) == (320, 240, 30)
    assert info.fps == pytest.approx(15.0) and not info.has_audio
    before = [rgb for _, _, rgb in video.iter_frames(proxy)]
    after = [rgb for _, _, rgb in video.iter_frames(dst)]
    assert len(before) == len(after) == 30
    for f in (1, 2, 29):
        assert mean_abs_diff(before[f], after[f]) < 3.0
    for f, (x1, y1, x2, y2) in ((0, (90, 65, 140, 110)), (5, (95, 65, 145, 110))):
        assert mean_abs_diff(before[f][y1:y2, x1:x2], after[f][y1:y2, x1:x2]) > 10.0
        outside = np.ones((240, 320), bool)
        outside[y1 - 8 : y2 + 8, x1 - 8 : x2 + 8] = False
        outside[0:40, 0:40] = False
        assert mean_abs_diff(before[f][outside], after[f][outside]) < 3.0


# Stills and crops ---------------------------------------------------------------------------------------------------


def test_write_jpeg(tmp_path: Path) -> None:
    ys, xs = np.mgrid[0:48, 0:64]
    rgb = np.stack([(xs * 4) % 256, (ys * 5) % 256, np.full_like(xs, 128)], axis=-1).astype(np.uint8)
    dst = tmp_path / "a" / "b.jpg"
    sha = video.write_jpeg(rgb, dst, quality=95)
    assert dst.is_file() and sha == video.sha256_file(dst)
    with Image.open(dst) as img:
        assert img.format == "JPEG" and img.size == (64, 48)
        assert img.info.get("exif") is None
        back = np.asarray(img.convert("RGB"))
    assert mean_abs_diff(rgb, back) < 4.0


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
    proxy_frames = {i: rgb for i, _, rgb in video.iter_frames(proxy) if i in (0, 30, 70)}
    for i, small in proxy_frames.items():
        frame = video.read_frame_at(PINGPONG, video.proxy_frame_to_pts(pts, i))
        assert frame.shape == (720, 1280, 3)
        assert mean_abs_diff(cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA), small) < 4.0
    assert video.sharpness(proxy_frames[30]) > 0.0
    assert len(video.sha256_file(proxy)) == 64
