"""privacy (M12): pixelation, the plate extent, the image checks through a fake sam_client, the step-7 order, the
keep_plate rule, track boxes, deletions with their receipts, the quarantine, the sweeper, the serving rule and the
privacy line. LIVE=1 adds one real pair of image checks on the ping pong clip (two sam_image rows, about $0.005)."""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest
from meta_sam_parser import SegmentationMask
from meta_sam_parser._mask_codec import _encode_raster

from walkaround import config
from walkaround import contracts as c
from walkaround import ledger, paths, privacy, sam_client

REPO = Path(__file__).resolve().parent.parent
TH = c.load_thresholds()
NOW = "2026-10-27T12:00:00Z"


# Fixtures and helpers --------------------------------------------------------------------------------------------------


@pytest.fixture
def runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "runs"
    monkeypatch.setattr(paths, "RUNS_DIR", root)
    monkeypatch.setattr(config, "RUNS_DIR", root)
    monkeypatch.setattr(ledger, "LEDGER_PATH", root / "ledger.jsonl")
    return root


def _noise(h: int, w: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _rect_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    x1, y1, x2, y2 = box
    mask[y1:y2, x1:x2] = True
    return mask


def _rotated_rect_mask(shape: tuple[int, int], center: tuple[int, int], size: tuple[int, int], angle: float) -> np.ndarray:
    pts = cv2.boxPoints((center, size, angle)).astype(np.int32)
    canvas = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(canvas, [pts], 1)
    return canvas.astype(bool)


def _bbox(mask: np.ndarray) -> list[float]:
    ys, xs = np.nonzero(mask)
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def _detection(mask: np.ndarray, object_id: str = "0", confidence: float | None = 0.9) -> sam_client.Detection:
    """A Detection whose one_bit mask is the real wire encoding of the box-local raster."""
    x1, y1, x2, y2 = (int(v) for v in _bbox(mask))
    local = mask[y1:y2, x1:x2].astype(np.uint8)
    payload = _encode_raster(bytes(local.flatten().tolist()), x2 - x1, y2 - y1)
    seg = SegmentationMask(encoding="one_bit", payload=payload, width=x2 - x1, height=y2 - y1)
    return sam_client.Detection(object_id=object_id, frame_index=None, box=sam_client.Box(x1, y1, x2, y2), mask=seg, confidence=confidence)


def _sha(rgb: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()


class FakeSam:
    """Stands in for sam_client.segment_image: records what it saw and answers with the given detections."""

    def __init__(self, plates: list[sam_client.Detection], faces: list[sam_client.Detection]) -> None:
        self.plates, self.faces = plates, faces
        self.calls: list[tuple[str, str, str | None, str | None, str]] = []

    async def segment_image(self, image, concept: str, *, run_id, role=None, tag="", include_confidence=True):
        self.calls.append((concept, _sha(image), run_id, role, tag))
        dets = self.plates if concept == c.Noun.LICENSE_PLATE.value else self.faces
        run = sam_client.SegmentationRun(concept=concept, media="image", media_width=int(image.shape[1]), media_height=int(image.shape[0]))
        if dets:
            run.frames[0] = list(dets)
        run.response_id = "resp_plate" if concept == c.Noun.LICENSE_PLATE.value else "resp_face"
        return run


def _changed(a: np.ndarray, b: np.ndarray) -> float:
    return float((a != b).any(axis=2).mean())


def _make_clip(path: Path, *, w: int = 320, h: int = 180, fps: int = 15, seconds: float = 0.5) -> None:
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size={w}x{h}:rate={fps}", "-t", str(seconds),
           "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "ultrafast", str(path)]
    subprocess.run(cmd, check=True)


def _plate_track(frames: list[int], box: list[float], *, role: str = "pickup", noun: c.Noun = c.Noun.LICENSE_PLATE) -> c.PanelTrack:
    return c.PanelTrack(
        track_id=f"{role}_{c.noun_slug(noun)}_0", role=role, noun=noun, sam_object_ids=["0"],
        fragments=[c.TrackFragment(object_id="0", first_frame=min(frames), last_frame=max(frames))],
        per_frame=[c.FrameObs(frame=f, pts=f / 15.0, box=list(box), box_area_frac=0.002, in_car_box=True, c=0.7) for f in frames],
    )


# Pixelation ---------------------------------------------------------------------------------------------------------------


def test_pixelate_region_unrecognisable_rest_untouched() -> None:
    rgb = _noise(240, 320, seed=1)
    out = privacy.pixelate(rgb, [[80.0, 60.0, 200.0, 150.0]], block_px=10)
    assert out.shape == rgb.shape and out.dtype == np.uint8
    assert not np.shares_memory(out, rgb)
    region = _rect_mask((240, 320), (80, 60, 200, 150))
    assert np.array_equal(out[~region], rgb[~region])
    inside_out, inside_in = out[60:150, 80:200], rgb[60:150, 80:200]
    blocks = inside_out.reshape(9, 10, 12, 10, 3)
    assert (blocks == blocks[:, :1, :, :1, :]).all()  # one colour per 10 x 10 block
    assert _changed(inside_out, inside_in) > 0.95
    assert len(np.unique(inside_in.reshape(-1, 3), axis=0)) > 5000
    assert len(np.unique(inside_out.reshape(-1, 3), axis=0)) <= 9 * 12


def test_pixelate_text_is_gone() -> None:
    frame = np.full((200, 400, 3), 230, dtype=np.uint8)
    cv2.putText(frame, "ABC 1234", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (20, 20, 20), 8, cv2.LINE_AA)
    out = privacy.pixelate(frame, [[0.0, 0.0, 400.0, 200.0]], block_px=24)
    gray_in = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray_out = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
    assert (gray_in < 100).sum() > 2000  # the strokes
    assert gray_out.min() > 20  # no pixel keeps the stroke colour: every one is a block mean
    blocks = gray_out[:192, :384].reshape(8, 24, 16, 24)
    assert (blocks == blocks[:, :1, :, :1]).all()  # one value per block: no stroke shape survives


def test_pixelate_dilation_grows_the_region_and_keeps_the_rest() -> None:
    rgb = _noise(240, 320, seed=2)
    out = privacy.pixelate(rgb, [[100.0, 100.0, 120.0, 120.0]], block_px=4, dilate_frac=0.5, dilate_px=6)
    grown = _rect_mask((240, 320), (84, 84, 136, 136))  # 20 px box x 0.5 + 6 px per side
    assert _changed(out[84:136, 84:136], rgb[84:136, 84:136]) > 0.9
    assert np.array_equal(out[~grown], rgb[~grown])
    assert np.array_equal(privacy.pixelate(rgb, [], block_px=4), rgb)
    assert np.array_equal(privacy.pixelate(rgb, [[50.0, 50.0, 50.0, 90.0]], block_px=4), rgb)  # an empty box
    with pytest.raises(ValueError):
        privacy.pixelate(rgb, [[0.0, 0.0, 10.0, 10.0]], block_px=0)


# Plate extent -----------------------------------------------------------------------------------------------------------


def test_plate_extent_measures_long_side_and_tilt() -> None:
    frontal = _rotated_rect_mask((400, 600), (300, 200), (240, 120), 25)
    long_px, hw, tilt = privacy.plate_extent(frontal)
    assert abs(long_px - 240) < 240 * 0.03
    assert abs(hw - 0.5) < 0.03
    assert tilt < 12  # acos is steep near h / w = 0.5, so a 1% extent error reads as several degrees
    yawed = _rotated_rect_mask((400, 600), (300, 200), (144, 120), 0)  # h / w 0.833: cos theta = 0.6
    _, hw2, tilt2 = privacy.plate_extent(yawed)
    assert abs(hw2 - 0.833) < 0.03
    assert abs(tilt2 - math.degrees(math.acos(0.6))) < 3
    assert privacy.plate_extent(np.zeros((10, 10), dtype=bool)) is None


def test_plate_extent_read_from_a_pixelated_region_overstates_by_over_20_percent() -> None:
    frame = _noise(720, 1280, seed=3)
    plate = _rotated_rect_mask((720, 1280), (640, 400), (200, 100), 0)
    frame[plate] = (240, 240, 240)
    true_long = privacy.plate_extent(plate)[0]
    blurred = privacy.pixelate(
        frame, [_bbox(plate)], block_px=20,
        dilate_frac=TH["privacy"]["track_box_dilation_frac"], dilate_px=TH["privacy"]["track_box_dilation_px"],
    )
    region = (blurred != frame).any(axis=2)  # what a reader of the blurred frame would call the plate
    assert privacy.plate_extent(region)[0] / true_long > 1.2


# Image checks and the blur order -------------------------------------------------------------------------------------------


async def test_image_checks_record_the_plate_extent_from_the_unblurred_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _noise(720, 1280, seed=4)
    plate = _detection(_rect_mask((720, 1280), (500, 350, 700, 450)), confidence=0.91)
    face = _detection(_rect_mask((720, 1280), (100, 100, 160, 180)), confidence=0.8)
    fake = FakeSam([plate], [face])
    monkeypatch.setattr(sam_client, "segment_image", fake.segment_image)
    plates, faces, record = await privacy.image_checks(frame, run_id="r1", role="pickup", tag="f000003", th=TH)
    assert [call[0] for call in fake.calls] == ["license plate", "face"]
    assert all(call[1] == _sha(frame) for call in fake.calls)  # the unblurred pixels went to SAM
    assert [call[2:] for call in fake.calls] == [("r1", "pickup", "f000003.license_plate"), ("r1", "pickup", "f000003.face")]
    assert plates == [plate] and faces == [face]
    assert record.plate_boxes == [[500.0, 350.0, 700.0, 450.0]]
    assert record.face_boxes == [[100.0, 100.0, 160.0, 180.0]]
    assert record.request_ids == ["resp_plate", "resp_face"]
    assert record.plate_ruler_px is not None and abs(record.plate_ruler_px - 200) < 6
    assert record.plate_ruler_c == 0.91
    assert abs(record.plate_tilt_hw - 0.5) < 0.03


async def test_image_checks_ruler_refuses_low_c_and_border_plates(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _noise(720, 1280, seed=5)
    low_c = _detection(_rect_mask((720, 1280), (500, 350, 700, 450)), confidence=0.3)
    on_border = _detection(_rect_mask((720, 1280), (0, 350, 200, 450)), confidence=0.95)
    fake = FakeSam([low_c, on_border], [])
    monkeypatch.setattr(sam_client, "segment_image", fake.segment_image)
    plates, faces, record = await privacy.image_checks(frame, run_id="r1", role="pickup", tag="t", th=TH)
    assert len(plates) == 2 and faces == []
    assert len(record.plate_boxes) == 2 and record.face_boxes == []
    assert record.plate_ruler_px is None and record.plate_ruler_c is None and record.plate_tilt_hw is None


async def test_blur_keyframe_checks_first_then_pixelates_and_honours_keep_plate(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _noise(720, 1280, seed=6)
    plate_mask = _rect_mask((720, 1280), (500, 350, 700, 450))
    face_mask = _rect_mask((720, 1280), (100, 100, 160, 180))
    fake = FakeSam([_detection(plate_mask, confidence=0.9)], [_detection(face_mask, confidence=0.8)])
    monkeypatch.setattr(sam_client, "segment_image", fake.segment_image)
    sizes = ((640, 360), (1280, 720))
    before = frame.copy()
    blurred, record = await privacy.blur_keyframe("r1", "pickup", frame, f_proxy=3, tracks=[], fps=15.0, sizes=sizes, keep_plate=False, th=TH)
    assert np.array_equal(frame, before)  # the caller's frame is untouched
    assert all(call[1] == _sha(before) for call in fake.calls)  # SAM saw the frame before any pixel changed
    assert record.plate_ruler_px is not None and abs(record.plate_ruler_px - 200) < 6
    assert _changed(blurred[350:450, 500:700], before[350:450, 500:700]) > 0.9  # the plate mask is pixelated
    assert _changed(blurred[100:180, 100:160], before[100:180, 100:160]) > 0.9  # so is the face
    assert np.array_equal(blurred[600:720, 900:1280], before[600:720, 900:1280])  # far from both masks
    kept, record2 = await privacy.blur_keyframe("r1", "pickup", frame, f_proxy=3, tracks=[], fps=15.0, sizes=sizes, keep_plate=True, th=TH)
    assert np.array_equal(kept[350:450, 500:700], before[350:450, 500:700])  # the plate stays for the owner's sheet
    assert _changed(kept[100:180, 100:160], before[100:180, 100:160]) > 0.9  # faces are always pixelated
    assert record2.plate_ruler_px == record.plate_ruler_px
    with pytest.raises(ValueError):
        await privacy.blur_keyframe("r1", "pickup", frame, f_proxy=3, tracks=[], fps=15.0, sizes=((640, 360), (1920, 1080)), keep_plate=False, th=TH)


def test_blur_frame_pixelates_track_boxes_scaled_from_the_proxy() -> None:
    frame = _noise(360, 640, seed=7)
    track = _plate_track([0, 1, 2], [10.0, 20.0, 30.0, 40.0])
    boxes = privacy.track_boxes_at(1, [track], fps=15.0, media_size_proxy=(160, 90), media_size_src=(640, 360), th=TH)
    assert boxes == [[40.0, 80.0, 120.0, 160.0]]
    out = privacy.blur_frame(frame, track_boxes_src=boxes, detections=[], keep_plate_region=False, th=TH)
    assert _changed(out[80:160, 40:120], frame[80:160, 40:120]) > 0.9
    assert np.array_equal(out[250:360, 300:640], frame[250:360, 300:640])


def test_track_boxes_at_bridges_short_gaps_and_ignores_other_nouns() -> None:
    plate = _plate_track([0, 1, 2, 3, 10], [10.0, 20.0, 30.0, 40.0])
    car = _plate_track([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [0.0, 0.0, 100.0, 100.0], noun=c.Noun.CAR)
    kw = dict(fps=15.0, media_size_proxy=(100, 100), media_size_src=(400, 200), th=TH)
    assert privacy.track_boxes_at(3, [plate, car], **kw) == [[40.0, 40.0, 120.0, 80.0]]
    assert privacy.track_boxes_at(5, [plate, car], **kw) == [[40.0, 40.0, 120.0, 80.0]]  # 2 frames after the last sighting
    assert privacy.track_boxes_at(11, [plate, car], **kw) == [[40.0, 40.0, 120.0, 80.0]]  # bridged from frame 10
    assert privacy.track_boxes_at(19, [plate, car], **kw) == []  # 9 frames = 0.6 s, over track_gap_bridge_s
    assert privacy.track_boxes_at(5, [car], **kw) == []


# Quarantine, deletions, the sweeper ----------------------------------------------------------------------------------------


def test_quarantine_put_and_read_original_frame(runs: Path, tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")
    run_id = "r1"
    paths.ensure_run_layout(run_id)
    src = tmp_path / "upload.mp4"
    _make_clip(src)
    dst = privacy.quarantine_put(run_id, "pickup", src)
    assert dst == paths.quarantine_original(run_id, "pickup") and dst.is_file() and not src.exists()
    frame = privacy.read_original_frame(run_id, "pickup", 0.2)
    assert frame.shape == (180, 320, 3) and frame.dtype == np.uint8
    with pytest.raises(FileNotFoundError):
        privacy.read_original_frame(run_id, "return", 0.0)


def test_delete_quarantine_logs_hash_bytes_time_and_removes_everything(runs: Path) -> None:
    run_id = "r1"
    paths.ensure_run_layout(run_id)
    data = bytes(range(256)) * 4
    paths.quarantine_original(run_id, "pickup").write_bytes(data)
    paths.quarantine_temp(run_id, "f1.jpg").write_bytes(b"y" * 10)
    rows = privacy.delete_quarantine(run_id, now=NOW)
    assert {r.target for r in rows} == {"quarantine/pickup.orig", "quarantine/tmp/f1.jpg"}
    row = next(r for r in rows if r.target == "quarantine/pickup.orig")
    assert row.kind == c.DeletionKind.QUARANTINE
    assert row.sha256 == hashlib.sha256(data).hexdigest() and row.bytes == len(data) and row.deleted_at == NOW
    assert not [p for p in (runs / run_id / "quarantine").rglob("*") if p.is_file()]
    deletions = privacy.close_deletions(run_id, rows, keep_plate=True, chosen_by="renter at upload")
    back = c.read_json(paths.deletions_json(run_id), c.Deletions)
    assert back == deletions and back.rows == rows and back.keep_plate and back.keep_plate_chosen_by == "renter at upload"
    assert paths.deletions_json(run_id).read_bytes() == c.json_bytes(deletions)
    assert privacy.delete_quarantine(run_id, now=NOW) == []


async def test_delete_remote_records_receipts_and_failures(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(privacy, "_RETRY_SLEEP_S", 0.0)
    calls: list[str] = []

    async def fake_delete_file(file_id: str) -> dict:
        calls.append(file_id)
        if file_id == "file_bad":
            raise RuntimeError("boom")
        return {"id": file_id, "object": "file", "deleted": True}

    async def fake_delete_response(response_id: str) -> dict:
        calls.append(response_id)
        return {"id": response_id, "status_code": 200, "deleted": True, "body": None}

    monkeypatch.setattr(sam_client, "delete_file", fake_delete_file)
    monkeypatch.setattr(sam_client, "delete_response", fake_delete_response)

    def video(role: str, file_id: str, deleted_at: str | None = None) -> c.VideoRecord:
        proxy = c.ProxyRecord(path=f"media/{role}.proxy.mp4", fps=15.0, width=960, height=540, frames=600, pts_map_path=f"media/{role}.pts.json", sha256="ab" * 32)
        files = c.FilesApiRecord(file_id=file_id, uploaded_at="2026-10-27T11:00:00Z", expires_at="2026-10-27T12:00:00Z", deleted_at=deleted_at)
        return c.VideoRecord(role=role, original_filename=f"{role}.mp4", bytes=1, proxy=proxy, files_api=files)

    videos = [video("pickup", "file_ok"), video("return", "file_bad"), video("topup_1", "file_done", deleted_at="2026-10-27T11:30:00Z")]
    rows = await privacy.delete_remote("r1", videos, response_ids=["resp_1"], now=NOW, retries=3)
    by_target = {r.target: r for r in rows}
    ok = by_target["file_ok"]
    assert ok.kind == c.DeletionKind.FILES_API and ok.api_response == {"id": "file_ok", "object": "file", "deleted": True}
    assert ok.sha256 == "ab" * 32 and ok.deleted_at == NOW
    bad = by_target["file_bad"]
    assert bad.api_response is None and "failed after 3 tries" in bad.reason and "expiry at 2026-10-27T12:00:00Z" in bad.reason
    assert calls.count("file_bad") == 3 and "file_done" not in calls
    resp = by_target["resp_1"]
    assert resp.kind == c.DeletionKind.RESPONSE and resp.api_response["deleted"] is True
    calls.clear()
    rows = await privacy.delete_remote("r1", [video("pickup", "file_ok")], response_ids=None, now=NOW, retries=1)
    assert [r.target for r in rows] == ["file_ok"] and calls == ["file_ok"]


async def test_render_blurred_proxy_pixelates_the_track_and_deletes_the_unblurred_proxy(runs: Path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")
    from walkaround import video as video_mod

    run_id = "r1"
    paths.ensure_run_layout(run_id)
    rng = np.random.default_rng(8)
    frame = np.full((540, 960, 3), 128, dtype=np.uint8)
    frame[200:320, 300:600] = rng.integers(0, 256, (120, 300, 3), dtype=np.uint8)  # inside the plate track
    frame[380:500, 650:950] = rng.integers(0, 256, (120, 300, 3), dtype=np.uint8)  # an untracked noisy patch
    proxy = paths.proxy_path(run_id, "pickup")
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "960x540", "-r", "15", "-i", "-",
         "-c:v", "libx264", "-preset", "ultrafast", "-qp", "0", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(proxy)],
        stdin=subprocess.PIPE,
    )
    assert enc.stdin is not None
    for _ in range(15):
        enc.stdin.write(frame.tobytes())
    enc.stdin.close()
    assert enc.wait() == 0
    record = video_mod.proxy_record(run_id, "pickup", proxy)
    c.write_json(paths.video_json(run_id, "pickup"), c.VideoRecord(role="pickup", original_filename="a.mp4", bytes=1, proxy=record))
    track = _plate_track(list(range(15)), [300.0, 200.0, 600.0, 320.0])
    blurred, deletion = await privacy.render_blurred_proxy(run_id, "pickup", [track], fps=15.0, th=TH)
    out_path = paths.run_dir(run_id) / blurred.path
    assert out_path == paths.proxy_blurred_path(run_id, "pickup") and out_path.is_file()
    assert blurred.sha256 == hashlib.sha256(out_path.read_bytes()).hexdigest()
    assert not proxy.exists()
    assert deletion.kind == c.DeletionKind.FILE and deletion.target == "media/pickup.proxy.mp4"
    assert deletion.sha256 == record.sha256 and deletion.bytes and deletion.bytes > 0
    got = video_mod.read_frame_at(out_path, 0.3)
    gray = cv2.cvtColor(got, cv2.COLOR_RGB2GRAY).astype(np.float64)
    assert gray[200:320, 300:600].var() < 0.3 * gray[380:500, 650:950].var()  # the tracked noise became blocks
    assert abs(gray[0:150, 0:250].mean() - 128) < 6  # the flat background is untouched


def test_sweep_deletes_old_runs_and_stale_temps_but_keeps_keep_runs(runs: Path) -> None:
    def make_run(run_id: str, created_at: str, *, keep: bool = False) -> None:
        paths.ensure_run_layout(run_id)
        c.write_json(paths.run_json(run_id), c.RunManifest(run_id=run_id, created_at=created_at, keep=keep))
        paths.quarantine_original(run_id, "pickup").write_bytes(b"z" * 100)

    make_run("old", "2026-10-25T12:00:00Z")
    make_run("kept", "2026-10-25T12:00:00Z", keep=True)
    make_run("listed", "2026-10-25T12:00:00Z")
    make_run("young", "2026-10-27T11:00:00Z")
    stale = paths.quarantine_temp("young", "a.jpg")
    stale.write_bytes(b"s" * 20)
    fresh = paths.quarantine_temp("young", "b.jpg")
    fresh.write_bytes(b"f" * 20)
    now_ts = datetime(2026, 10, 27, 12, 0, 0, tzinfo=timezone.utc).timestamp()
    os.utime(stale, (now_ts - 7200, now_ts - 7200))
    os.utime(fresh, (now_ts - 60, now_ts - 60))
    rows = privacy.sweep(now=NOW, retention_hours=24, temp_minutes=30, keep_run_ids={"listed"})
    assert not (runs / "old").exists()
    assert (runs / "kept").exists() and (runs / "listed").exists() and (runs / "young").exists()
    assert not stale.exists() and fresh.exists()
    targets = {r.target: r for r in rows}
    assert targets["old/"].kind == c.DeletionKind.FILE and targets["old/"].bytes and targets["old/"].bytes > 100
    temp = targets["young/quarantine/tmp/a.jpg"]
    assert temp.kind == c.DeletionKind.QUARANTINE and temp.sha256 == hashlib.sha256(b"s" * 20).hexdigest()
    assert len(rows) == 2


# Serving rule and the privacy line ------------------------------------------------------------------------------------------


def test_served_file_allowed() -> None:
    run = c.RunManifest(run_id="r1", created_at=NOW)
    kept = c.RunManifest(run_id="r2", created_at=NOW, keep_plate=True)
    for rel in ("quarantine/pickup.orig", "quarantine/tmp/x.jpg", "uploads/u1/chunks/000000", "media/pickup.proxy.mp4", "../other/run.json", "/etc/passwd", ""):
        assert privacy.served_file_allowed(run, rel, run_owner=True) is False, rel
    for rel in ("media/pickup.proxy_blurred.mp4", "sam/pickup/car.ndjson.gz", "crops/pickup/k.jpg", "derived/verdicts.json", "sheet/frames/k.jpg"):
        assert privacy.served_file_allowed(run, rel, run_owner=False) is True, rel
    for rel in ("sheet/frames/k.jpg", "sheet/evidence.pdf", "crops/pickup/k.jpg", "keyframes/pickup/k.frame.jpg", "clips/return/m1.mp4"):
        assert privacy.served_file_allowed(kept, rel, run_owner=True) is True, rel
        assert privacy.served_file_allowed(kept, rel, run_owner=False) is False, rel
    assert privacy.served_file_allowed(kept, "media/pickup.proxy_blurred.mp4", run_owner=False) is True
    assert privacy.served_file_allowed(kept, "derived/verdicts.json", run_owner=False) is True
    assert privacy.served_file_allowed(kept, "quarantine/pickup.orig", run_owner=True) is False


def test_privacy_line_word_for_word() -> None:
    run = c.RunManifest(run_id="r1", created_at=NOW, responses_store=c.ResponsesStore.FALSE_ACCEPTED)
    line = privacy.privacy_line(run, {"pickup": True, "return": True})
    assert line.startswith(
        "Processed with Meta's Model API in Standard mode, which Meta's documentation says is not used to train Meta "
        "models. Uploads were set to expire after one hour and were deleted by Walkaround when the run ended. "
        "Model requests were sent with storage off. Meta's terms allow it to retain content for abuse review and legal "
        "reasons; Walkaround does not control that. SAM 3.1 saw the unblurred proxy for the length of one segmentation "
        "pass to find the plates and faces it then blurred."
    )
    assert "no plate detected" not in line
    for banned in ("zero data retention", "Meta deleted", "Contributor"):
        assert banned not in line
    refused = c.RunManifest(run_id="r1", created_at=NOW, responses_store=c.ResponsesStore.DELETED_PER_REQUEST)
    line2 = privacy.privacy_line(refused, {"pickup": True, "return": False})
    assert "Model requests were sent with storage off and every stored model response was deleted by Walkaround when the run ended." in line2
    assert line2.endswith("Return walk: no plate detected; nothing was blurred for that reason.")
    assert line.isascii() and line2.isascii()


# LIVE -----------------------------------------------------------------------------------------------------------------------


@pytest.mark.skipif(os.environ.get("LIVE") != "1", reason="LIVE=1 only: two sam_image calls on one frame, about $0.005")
async def test_live_image_checks_on_the_pingpong_clip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The real sam_client on a real frame; the ledger rows land in runs/ledger.jsonl because the call costs money,
    the run files go to a temporary runs dir."""
    from walkaround import video as video_mod

    root = tmp_path / "runs"
    monkeypatch.setattr(paths, "RUNS_DIR", root)
    monkeypatch.setattr(config, "RUNS_DIR", root)
    clip = REPO / "data" / "test" / "pingpong.mp4"
    frame = video_mod.read_frame_at(clip, 1.0)
    run_id = "lane_privacy_smoke"
    plates, faces, record = await privacy.image_checks(frame, run_id=run_id, role="pickup", tag="live", th=TH)
    assert len(record.request_ids) == 2
    assert len(record.plate_boxes) == len(plates) and len(record.face_boxes) == len(faces)
    rows = [r for r in ledger.rows(ledger.LEDGER_PATH) if r.get("run_id") == run_id]
    assert len(rows) >= 2 and all(r["kind"] == "sam_image" and r["usd"] > 0 for r in rows)
    saved = sorted(paths.image_requests_dir(run_id, "pickup").glob("live.*.ndjson"))
    assert len(saved) == 2
    print(f"LIVE image checks: {len(plates)} plates, {len(faces)} faces, ruler px {record.plate_ruler_px}, requests {record.request_ids}")
