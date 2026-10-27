"""seal (M4): the manifest round trip, verify catching a flipped byte, the camera time from ffprobe, the three
clocks, the typed stamp hooks, manifest_run.json and the event chain check."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from walkaround import config
from walkaround import contracts as c
from walkaround import ledger, paths, seal

NOW = "2026-10-27T15:00:00Z"
CAMERA = "2026-10-27T14:37:00Z"


@pytest.fixture
def runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "runs"
    monkeypatch.setattr(paths, "RUNS_DIR", root)
    monkeypatch.setattr(config, "RUNS_DIR", root)
    monkeypatch.setattr(ledger, "LEDGER_PATH", root / "ledger.jsonl")
    return root


def _originals(run_id: str = "r1", *, run_json: bool = True) -> list[Path]:
    paths.ensure_run_layout(run_id)
    pickup = paths.quarantine_original(run_id, "pickup")
    pickup.write_bytes(bytes(range(256)) * 40)
    ret = paths.quarantine_original(run_id, "return")
    ret.write_bytes(b"return bytes " * 500)
    if run_json:
        c.write_json(paths.run_json(run_id), c.RunManifest(run_id=run_id, created_at=NOW, config_hash="ab" * 32, app_commit="abc1234"))
    return [pickup, ret]


def _event(seq: int, prev: str, kind: c.EventKind, payload: dict, run_id: str = "r1") -> c.Event:
    event = c.Event(seq=seq, ts="2026-10-27T15:00:01Z", run_id=run_id, kind=kind, payload=payload, prev=prev, hash="0" * 64)
    event.hash = c.chain_hash(prev, event.line_without_hash())
    return event


def _write_chain(run_id: str, events: list[c.Event]) -> None:
    paths.events_path(run_id).write_text("".join(e.to_json() + "\n" for e in events), encoding="utf-8")


def _make_clip(path: Path, *, creation_time: str | None) -> None:
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=160x96:rate=15", "-t", "0.4",
           "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "ultrafast"]
    if creation_time:
        cmd += ["-metadata", f"creation_time={creation_time}"]
    subprocess.run(cmd + [str(path)], check=True)


# Manifest and verify -----------------------------------------------------------------------------------------------------


def test_build_manifest_round_trip(runs: Path) -> None:
    originals = _originals()
    manifest = seal.build_manifest("r1", originals)
    path = paths.manifest_json("r1")
    assert path.read_bytes() == c.json_bytes(manifest)
    assert c.read_json(path, c.SealManifest) == manifest
    assert [e.path for e in manifest.files] == ["quarantine/pickup.orig", "quarantine/return.orig"]
    for entry, original in zip(manifest.files, originals):
        assert entry.sha256 == hashlib.sha256(original.read_bytes()).hexdigest()
        assert entry.bytes == original.stat().st_size
    assert manifest.config_hash == "ab" * 32 and manifest.app_commit == "abc1234"
    assert manifest.stamps == c.Stamps() and manifest.deletions == [] and manifest.pdf_sha256 is None
    assert seal.sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_verify_catches_a_flipped_byte(runs: Path) -> None:
    originals = _originals()
    seal.build_manifest("r1", originals)
    assert seal.verify_run("r1") == [] and seal.verify("r1") == []
    data = bytearray(originals[0].read_bytes())
    data[1000] ^= 0x01
    originals[0].write_bytes(bytes(data))
    assert seal.verify_run("r1") == ["hash differs: quarantine/pickup.orig"]
    originals[1].write_bytes(originals[1].read_bytes()[:-1])
    problems = seal.verify_run("r1")
    assert "size differs: quarantine/return.orig" in problems and "hash differs: quarantine/return.orig" in problems
    originals[1].unlink()
    assert "missing: quarantine/return.orig" in seal.verify_run("r1")
    paths.manifest_json("r1").unlink()
    assert seal.verify_run("r1") == ["bundle/manifest.json missing"]


def test_verify_accepts_a_provided_original_after_the_quarantine_is_gone(runs: Path, tmp_path: Path) -> None:
    originals = _originals()
    manifest = seal.build_manifest("r1", originals)
    copy = tmp_path / "IMG_0001.MOV"
    shutil.copy(originals[0], copy)
    deletions = [c.DeletionRecord(target=e.path, kind=c.DeletionKind.QUARANTINE, sha256=e.sha256, bytes=e.bytes, deleted_at=NOW, reason="run end") for e in manifest.files]
    seal.write_manifest_run("r1", deletions=deletions, events_head_hash="", config_hash="ab" * 32, app_commit="abc1234", pdf_sha256=None)
    for original in originals:
        original.unlink()
    assert seal.verify_run("r1") == []  # deleted at run end, as manifest_run.json says
    assert seal.verify_run("r1", originals=[copy]) == ["no provided file matches quarantine/return.orig (sha256 " + manifest.files[1].sha256[:12] + ")"]
    copy.write_bytes(copy.read_bytes() + b"x")
    assert "no provided file matches quarantine/pickup.orig" in seal.verify_run("r1", originals=[copy])[0]


# Times and clocks ------------------------------------------------------------------------------------------------------------


def test_minutes_after_camera() -> None:
    assert seal.minutes_after_camera(NOW, CAMERA) == 23
    assert seal.minutes_after_camera(NOW, "2026-10-27T14:37:00.000000Z") == 23
    assert seal.minutes_after_camera(NOW, "2026-10-27T16:37:00+02:00") == 23
    assert seal.minutes_after_camera(NOW, NOW) == 0
    assert seal.minutes_after_camera(NOW, "not present") == c.CAMERA_TIME_NOT_PRESENT
    assert seal.minutes_after_camera(NOW, c.CAMERA_TIME_NOT_PRESENT) == "camera time not present"
    assert seal.minutes_after_camera(NOW, "") == c.CAMERA_TIME_NOT_PRESENT


def test_normalize_time() -> None:
    assert seal.normalize_time("2026-10-27T14:37:00.000000Z") == CAMERA
    assert seal.normalize_time("2026-10-27T16:37:00+02:00") == CAMERA
    assert seal.normalize_time("2026-10-27 14:37:00") == CAMERA  # naive times count as UTC
    for absent in (None, "", "not present", c.CAMERA_TIME_NOT_PRESENT, "yesterday"):
        assert seal.normalize_time(absent) == seal.NOT_PRESENT


def test_camera_time_of_reads_ffprobe(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg not on PATH")
    stamped, bare = tmp_path / "stamped.mp4", tmp_path / "bare.mp4"
    _make_clip(stamped, creation_time=CAMERA)
    _make_clip(bare, creation_time=None)
    assert seal.camera_time_of(stamped) == CAMERA
    assert seal.camera_time_of(bare) == seal.NOT_PRESENT
    assert seal.camera_time_of(tmp_path / "missing.mp4") == seal.NOT_PRESENT


def test_seal_run_clocks_and_unbuilt_hooks(runs: Path) -> None:
    originals = _originals()
    manifest, clocks = seal.seal_run("r1", originals, now=NOW, camera_time=CAMERA)
    assert clocks == c.ThreeClocks(sealed_at=NOW, first_model_call_at=None, sealed_minutes_after_camera=23, camera_time=CAMERA)
    assert manifest.stamps == c.Stamps()  # hooks not built: nothing stamped, nothing failed
    assert paths.manifest_json("r1").read_bytes() == c.json_bytes(manifest.model_copy(update={"stamps": c.Stamps()}))
    _, clocks2 = seal.seal_run("r1", originals, now=NOW, camera_time=c.CAMERA_TIME_NOT_PRESENT)
    assert clocks2.sealed_minutes_after_camera == "camera time not present" and clocks2.camera_time is None
    _, clocks3 = seal.seal_run("r1", originals, now=NOW, camera_time="2026-10-27T14:37:00.000000Z", stamp=False)
    assert clocks3.sealed_minutes_after_camera == 23 and clocks3.camera_time == CAMERA
    for hook, args in (
        (seal.stamp_ots, (Path("x"),)),
        (seal.ots_status, (Path("x.ots"),)),
        (seal.upgrade_ots, (Path("x.ots"),)),
        (seal.stamp_tsa, (Path("x"),)),
    ):
        with pytest.raises(NotImplementedError):
            hook(*args)
    with pytest.raises(NotImplementedError):
        seal.verify_tsr(Path("x"), Path("x.tsr"), cacert=Path("ca.pem"), tsa_crt=Path("tsa.crt"))


def test_seal_run_records_a_built_hook_that_fails_or_stamps(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    originals = _originals()

    def broken(path: Path) -> Path:
        raise RuntimeError("calendars unreachable")

    monkeypatch.setattr(seal, "stamp_ots", broken)
    manifest, _ = seal.seal_run("r1", originals, now=NOW, camera_time=CAMERA)
    assert manifest.stamps.manifest_ots.startswith("failed:RuntimeError: calendars unreachable")
    assert manifest.stamps.pickup_ots.startswith("failed:") and manifest.stamps.return_ots.startswith("failed:")

    monkeypatch.setattr(seal, "stamp_ots", lambda path: Path(str(path) + ".ots"))
    monkeypatch.setattr(seal, "ots_status", lambda ots_path: "pending")
    monkeypatch.setattr(seal, "stamp_tsa", lambda path, *, tsa_url=seal.FREETSA_URL, timeout_s=20.0: (Path("m.tsq"), Path("m.tsr"), "2026-10-27T15:00:02Z", "0x1f"))
    manifest, _ = seal.seal_run("r1", originals, now=NOW, camera_time=CAMERA)
    assert manifest.stamps == c.Stamps(manifest_ots="pending", pickup_ots="pending", return_ots="pending", tsr_time="2026-10-27T15:00:02Z", tsa_serial="0x1f", tsa_issuer="freetsa")
    assert c.read_json(paths.manifest_json("r1"), c.SealManifest).stamps == c.Stamps()  # the stamped bytes never change


# manifest_run.json and the chain -------------------------------------------------------------------------------------------


def test_write_manifest_run_and_verify_the_chain(runs: Path, tmp_path: Path) -> None:
    run_id = "r1"
    originals = _originals(run_id)
    manifest = seal.build_manifest(run_id, originals)
    manifest_sha = seal.sha256_file(paths.manifest_json(run_id))
    events = [_event(0, manifest_sha, c.EventKind.RUN_CREATED, {"mode": "car"})]
    events.append(_event(1, events[-1].hash, c.EventKind.SEAL_MANIFEST, {"sha256": manifest_sha}))
    events.append(_event(2, events[-1].hash, c.EventKind.SEAL_OTS, {"status": "pending"}))
    events.append(_event(3, events[-1].hash, c.EventKind.SEAL_TSR, {"tsr_time": "2026-10-27T15:00:02Z", "serial": "0x1f", "issuer": "freetsa"}))
    events.append(_event(4, events[-1].hash, c.EventKind.SEAL_OTS, {"status": "confirmed:921000"}))
    _write_chain(run_id, events)
    marks = [c.Mark(mark_id="m1", role="pickup", panel_id=c.PanelId.HOOD, kf_id="k1", source=c.MarkSource.SPARK_BOX, type=c.MarkKind.CHIP, confidence=c.Confidence.HIGH, size_class=c.SizeClass.S)]
    c.write_json(paths.marks_json(run_id), marks)
    c.write_json(paths.tracks_json(run_id), [])
    c.write_json(paths.sheet_json(run_id), c.SheetModel(run_id=run_id))
    paths.upload_chunk_path(run_id, "u1", 0).write_bytes(b"chunk")
    deletion = c.DeletionRecord(target="file_abc", kind=c.DeletionKind.FILES_API, deleted_at=NOW, reason="run end", api_response={"deleted": True})

    manifest_run = seal.write_manifest_run(run_id, deletions=[deletion], events_head_hash=events[-1].hash, config_hash="ab" * 32, app_commit="abc1234", pdf_sha256=None)
    listed = [e.path for e in manifest_run.files]
    assert "bundle/manifest.json" in listed and "derived/marks.json" in listed and "sheet/sheet.json" in listed
    for excluded in ("run.json", "events.ndjson", "bundle/manifest_run.json", "quarantine/pickup.orig", "uploads/u1/chunks/000000"):
        assert excluded not in listed, excluded
    assert manifest_run.deletions == [deletion] and manifest_run.events_head_hash == events[-1].hash
    assert manifest_run.stamps == c.Stamps(manifest_ots="confirmed:921000", tsr_time="2026-10-27T15:00:02Z", tsa_serial="0x1f", tsa_issuer="freetsa")
    assert c.read_json(paths.manifest_run_json(run_id), c.SealManifest) == manifest_run
    assert seal.verify_run(run_id) == []

    # Events appended after the run manifest keep the head inside the chain: still fine.
    events.append(_event(5, events[-1].hash, c.EventKind.RUN_COMPLETED, {}))
    _write_chain(run_id, events)
    assert seal.verify_run(run_id) == []

    # A rebuild that matches passes; one derived file that differs is named; frozen files are not compared.
    rebuild = tmp_path / "rebuild"
    (rebuild / "derived").mkdir(parents=True)
    (rebuild / "sheet").mkdir()
    shutil.copy(paths.marks_json(run_id), rebuild / "derived" / "marks.json")
    shutil.copy(paths.sheet_json(run_id), rebuild / "sheet" / "sheet.json")
    (rebuild / "derived" / "tracks.json").write_bytes(b"[1]\n")
    assert seal.verify_run(run_id, rebuild_dir=rebuild) == []
    assert seal.status_text(seal.verify_run(run_id, rebuild_dir=rebuild)) == "rebuild identical"
    c.write_json(rebuild / "derived" / "marks.json", [])
    assert seal.verify_run(run_id, rebuild_dir=rebuild) == ["rebuild differs: derived/marks.json"]

    # One edited event breaks the chain at that seq.
    edited = events[2].model_copy(update={"payload": {"status": "confirmed:1"}})
    _write_chain(run_id, events[:2] + [edited] + events[3:])
    problems = seal.verify_run(run_id)
    assert problems[0] == "event chain broken at seq 2" and len(problems) == 2  # and the head is no longer reachable
    _write_chain(run_id, events)

    # A head hash that is not in the chain is a problem of the run manifest.
    bogus = manifest_run.model_copy(update={"events_head_hash": "cd" * 32})
    c.write_json(paths.manifest_run_json(run_id), bogus)
    assert seal.verify_run(run_id) == ["run manifest: events head hash is not in the event chain"]

    # A derived file changed after the run manifest was written is named too.
    c.write_json(paths.manifest_run_json(run_id), manifest_run)
    c.write_json(paths.marks_json(run_id), [])
    assert seal.verify_run(run_id) == ["run manifest: size differs: derived/marks.json", "run manifest: hash differs: derived/marks.json"]


def test_chain_prev_of_seq_zero_is_the_manifest_hash(runs: Path) -> None:
    originals = _originals()
    seal.build_manifest("r1", originals)
    _write_chain("r1", [_event(0, "ff" * 32, c.EventKind.RUN_CREATED, {})])
    assert seal.verify_run("r1") == ["event chain broken at seq 0"]
    paths.events_path("r1").write_text("not json\n", encoding="utf-8")
    assert seal.verify_run("r1") == ["event chain broken at line 0: not an event"]
