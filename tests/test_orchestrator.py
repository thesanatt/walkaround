"""Orchestrator (M14) and CLI tests: the state machine with injected fakes, no model call, no network.

FileEventLog (the orchestrator's stand-in for walkaround.eventlog, same chain rule) records the events, and Fakes
stands in for every other lane's entry point, writing exactly the files each stage's completion check reads.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from walkaround import __main__ as cli
from walkaround import config, contracts as c, ledger, paths
from walkaround.orchestrator import (
    PENDING_EVENTS_NAME,
    STAGES,
    CapsExceeded,
    FileEventLog,
    HashMismatch,
    MissingChunks,
    Orchestrator,
    et_day_of,
    new_run_id,
    parse_ts,
    utc_now,
)

FUTURE = "2099-01-01T00:00:00Z"
BOX_RE = re.compile(r"<(\d+)f>(\d+)<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=\d+;h=\d+(?:;c=([0-9.]+))?\|>")
HOOD_COVERAGE = {c.PanelId.HOOD: c.PanelCoverage(state=c.CoverageState.FILMED, seconds=3.0)}


# Stand-ins ---------------------------------------------------------------------------------------------------------


class Crash(BaseException):
    """A simulated process death: not an Exception, so no stage retry catches it."""


class Fakes:
    """Every other lane's contract entry point, writing the files the stage completion checks read."""

    def __init__(self, *, ledger_frames: int = 0, crash_at: tuple[str, str] | None = None, clock_step: float = 0.0, clock: list[float] | None = None) -> None:
        self.calls: Counter[str] = Counter()
        self.lane_calls: list[tuple[str, list[str]]] = []
        self.ledger_frames = ledger_frames
        self.crash_at = crash_at
        self.clock_step = clock_step
        self.clock = clock
        self.missing_panels: list[c.PanelId] = []

    def overrides(self) -> dict[str, Any]:
        return {
            "video.probe": self.probe,
            "video.ffprobe_info": self.ffprobe_info,
            "video.make_proxy": self.make_proxy,
            "video.proxy_record": self.proxy_record,
            "video.extract_wav": self.extract_wav,
            "sam_client.upload_file": self.upload_file,
            "sam_client.sam_metadata": self.sam_metadata,
            "sam_client.segment_video_many": self.segment_video_many,
            "sam_wire.parse_lane_text": self.parse_lane_text,
            "sam_wire.DecodePool": lambda workers: SimpleNamespace(workers=workers, close=lambda: None),
            "panels.derive": self.derive_panels,
            "panels.coverage_for_role": self.coverage_for_role,
            "panels.plan_topup": self.plan_topup,
            "privacy.blur_keyframe": self.blur_keyframe,
            "privacy.render_blurred_proxy": self.render_blurred_proxy,
            "privacy.delete_remote": self.delete_remote,
            "privacy.delete_quarantine": self.delete_quarantine,
            "privacy.close_deletions": self.close_deletions,
            "privacy.sweep": self.sweep,
            "privacy.served_file_allowed": self.served_file_allowed,
            "marks.derive_marks": self.derive_marks,
            "staysput.derive_staysput": self.derive_staysput,
            "voice_client.transcribe_wav": self.transcribe_wav,
            "voice_client.voice_record": self.voice_record,
            "narration.derive_narration": self.derive_narration,
            "match.calibrate": self.calibrate,
            "match.derive_match": self.derive_match,
            "sheet.write_sheet": self.write_sheet,
            "seal.seal_run": self.seal_run,
            "seal.write_manifest_run": self.write_manifest_run,
        }

    # video
    def probe(self, path: Path) -> Any:
        self.calls["probe"] += 1
        return SimpleNamespace(path=path, width=1280, height=720, fps=24.0, duration_s=5.875, nb_frames=141, rotation=0, codec="h264", has_audio=True, creation_time=None)

    def ffprobe_info(self, info: Any, *, container: str, device: str | None = None, hdr: bool = False, vfr: bool = False) -> c.FfprobeInfo:
        return c.FfprobeInfo(codec="h264", container=container, width=1280, height=720, display_width=1280, display_height=720, fps_avg=24.0, duration_s=5.875, has_audio=True, creation_time="2026-10-27T14:00:00Z")

    def make_proxy(self, src: Path, dst: Path, *, fps: float, max_edge: int, preset: str, crf: int) -> Any:
        self.calls["make_proxy"] += 1
        dst.write_bytes(b"proxy bytes")
        return self.probe(dst)

    def proxy_record(self, run_id: str, role: str, proxy: Path) -> c.ProxyRecord:
        pts = paths.pts_map_path(run_id, role)
        pts.write_text(json.dumps([i / 15.0 for i in range(88)]), encoding="utf-8")
        return c.ProxyRecord(path=paths.rel(run_id, proxy), fps=15.0, width=960, height=540, frames=88, pts_map_path=paths.rel(run_id, pts), sha256=hashlib.sha256(proxy.read_bytes()).hexdigest())

    def extract_wav(self, src: Path, dst: Path, *, rate: int) -> Path:
        dst.write_bytes(b"RIFF")
        return dst

    # sam_client
    async def upload_file(self, path: Path, *, run_id: str | None = None, expires_after_s: int = 3600) -> c.FilesApiRecord:
        self.calls["upload_file"] += 1
        return c.FilesApiRecord(file_id=f"file-{path.stem}", uploaded_at="2026-10-27T15:00:00Z", expires_at=FUTURE)

    def sam_metadata(self, **_: Any) -> dict[str, str]:
        return {"include_confidence": "true", "mask_encoding": "one_bit"}

    async def segment_video_many(self, file_id: str, prompts: list[tuple[str, str]], *, run_id: str, role: str, frames_sent: int, media_size: tuple[int, int], metadata: dict[str, str], concurrency: int = 16, on_event: Any = None) -> dict[str, Any]:
        self.calls["segment_video_many"] += 1
        self.lane_calls.append((role, [lane for lane, _ in prompts]))
        if self.clock is not None:
            self.clock[0] += self.clock_step
        results = {}
        for lane, text in prompts:
            if self.crash_at == (role, lane):
                raise Crash()
            if on_event is not None:
                on_event(lane, {"type": "response.created", "response": {"id": f"resp_{role}_{lane}", "metadata": metadata, "store": False}})
                on_event(lane, {"type": "response.output_text.delta", "delta": f"<0f>0<|box;x1=1;y1=2;x2=30;y2=40;w={media_size[0]};h={media_size[1]};c=0.7|>"})
            record = c.LaneRecord(lane_id=lane, role=role, noun=c.Noun(text), prompt_text=text, media_size=list(media_size), params=metadata, request_id=f"resp_{role}_{lane}", frames_sent=frames_sent, frames_returned=1, video_frames_processed=frames_sent, ids=["0"], dollars=0.0176, lane_path=f"sam/{role}/{lane}.ndjson.gz", lane_sha256="a" * 64)
            frames = [c.IndexFrame(frame=0, objects=[c.IndexObject(id="0", box=[1, 2, 30, 40], c=0.7)])]
            c.write_json(paths.lane_summary_path(run_id, role, lane), record)
            c.write_json(paths.lane_index_path(run_id, role, lane), frames)
            if self.ledger_frames:
                ledger.record("sam_video", "sam-3.1", note=lane, run_id=run_id, frames=self.ledger_frames)
            results[lane] = (record, frames)
        return results

    def parse_lane_text(self, text: str, *, media: str) -> tuple[list[c.IndexFrame], list[c.Diagnostic]]:
        frames = []
        for m in BOX_RE.finditer(text):
            frames.append(c.IndexFrame(frame=int(m.group(1)), objects=[c.IndexObject(id=m.group(2), box=[int(m.group(i)) for i in range(3, 7)], c=float(m.group(7)) if m.group(7) else None)]))
        return frames, []

    # panels and privacy
    async def derive_panels(self, run_id: str, role: str, *, lanes: dict[str, Any], pts: list[float], media_size: tuple[int, int], fps: float, decode: Any, blur_keyframe: Any, th: dict) -> tuple[list, list, list, dict]:
        self.calls["derive_panels"] += 1
        self.calls[f"derive_panels_lanes_{role}"] = len(lanes)
        derived = paths.derived_dir(run_id)
        for name in ("tracks.json", "panels.json", "keyframes.json"):
            c.write_json(derived / name, [])
        return [], [], [], dict(HOOD_COVERAGE)

    def coverage_for_role(self, panels: list, tracks: list, keyframes: list, *, role: str, fps: float, lane_status: dict, th: dict) -> dict:
        self.calls["coverage_for_role"] += 1
        return dict(HOOD_COVERAGE)

    def plan_topup(self, coverage: c.Coverage, th: dict) -> c.Coverage:
        coverage.missing = list(self.missing_panels)
        coverage.topup_text = "film the " + ", ".join(p.value.replace("_", " ") for p in coverage.missing) if coverage.missing else ""
        return coverage

    async def blur_keyframe(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the fake panels.derive never decodes a frame")

    async def render_blurred_proxy(self, run_id: str, role: str, tracks: list, *, fps: float, th: dict) -> tuple[c.BlurredProxyRecord, c.DeletionRecord]:
        self.calls["render_blurred_proxy"] += 1
        blurred = paths.proxy_blurred_path(run_id, role)
        blurred.write_bytes(b"blurred")
        proxy = paths.proxy_path(run_id, role)
        row = c.DeletionRecord(target=paths.rel(run_id, proxy), kind=c.DeletionKind.FILE, bytes=proxy.stat().st_size if proxy.is_file() else 0, deleted_at=utc_now(), reason="blur.done")
        proxy.unlink(missing_ok=True)
        return c.BlurredProxyRecord(path=paths.rel(run_id, blurred), sha256=hashlib.sha256(b"blurred").hexdigest()), row

    async def delete_remote(self, run_id: str, videos: list[c.VideoRecord], *, response_ids: list[str] | None, now: str) -> list[c.DeletionRecord]:
        return [c.DeletionRecord(target=v.files_api.file_id, kind=c.DeletionKind.FILES_API, deleted_at=now, reason="run end", api_response={"deleted": True}) for v in videos if v.files_api]

    def delete_quarantine(self, run_id: str, *, now: str) -> list[c.DeletionRecord]:
        rows = []
        for path in sorted(paths.quarantine_dir(run_id).rglob("*")):
            if path.is_file():
                rows.append(c.DeletionRecord(target=paths.rel(run_id, path), kind=c.DeletionKind.QUARANTINE, bytes=path.stat().st_size, deleted_at=now, reason="run end"))
                path.unlink()
        return rows

    def close_deletions(self, run_id: str, rows: list[c.DeletionRecord], *, keep_plate: bool, chosen_by: str | None) -> c.Deletions:
        deletions = c.Deletions(run_id=run_id, rows=rows, keep_plate=keep_plate, keep_plate_chosen_by=chosen_by)
        c.write_json(paths.deletions_json(run_id), deletions)
        return deletions

    def sweep(self, *, now: str, retention_hours: int, temp_minutes: int, keep_run_ids: set[str]) -> list[c.DeletionRecord]:
        self.calls["sweep"] += 1
        return []

    def served_file_allowed(self, run: c.RunManifest, rel_path: str, *, run_owner: bool) -> bool:
        return not rel_path.startswith("quarantine/") and not rel_path.endswith(".proxy.mp4")

    # marks, staysput, voice, narration
    async def derive_marks(self, run_id: str, role: str, keyframes: list, *, th: dict, count_mode: bool = False) -> tuple[list, list, list]:
        self.calls["derive_marks"] += 1
        c.write_json(paths.marks_json(run_id), [])
        return [], keyframes, []

    async def derive_staysput(self, run_id: str, role: str, *, th: dict) -> list:
        self.calls["derive_staysput"] += 1
        c.write_json(paths.mark_tracks_json(run_id), [])
        return []

    def transcribe_wav(self, wav_path: Path, *, mode: str, keywords: list[str], run_id: str | None = None) -> Any:
        self.calls["transcribe_wav"] += 1
        assert mode == "ENDPOINTING" and len(keywords) == 21
        return SimpleNamespace(mode=mode, transcript="small chip on the hood", mode_applied=True)

    def voice_record(self, transcript: Any, *, role: str, wav_sha256: str) -> c.VoiceRecord:
        request = c.VoiceRequest(mode=c.VoiceMode.ENDPOINTING, model="muse-voice-transcribe-1.0")
        response = c.VoiceResponse(transcript=transcript.transcript, audioDurationMs=5875, turns=[c.VoiceTurn(startMs=500, endMs=3000, transcript=transcript.transcript)])
        return c.VoiceRecord(role=role, request=request, response=response, wav_sha256=wav_sha256, mode_applied=True)

    async def derive_narration(self, run_id: str, *, th: dict) -> c.Narration:
        self.calls["derive_narration"] += 1
        narration = c.Narration(per_role={}, mode_applied=True)
        c.write_json(paths.narration_json(run_id), narration)
        return narration

    # match, sheet, seal
    def calibrate(self, *, sigma_frame: float | None, sigma_spark: float | None, sigma_view: float | None, pseudo_pair_d: Any, th: dict) -> c.Calibration:
        return c.Calibration(tau_pre=0.25, tau_eye=0.5, tau_fix=0.08, calibrated=False)

    def derive_match(self, run_id: str, *, th: dict, second_look: Any = None) -> tuple[list, list, c.Calibration]:
        self.calls["derive_match"] += 1
        c.write_json(paths.matches_json(run_id), [])
        c.write_json(paths.verdicts_json(run_id), [])
        return [], [], c.read_json(paths.calibration_json(run_id), c.Calibration)

    def write_sheet(self, run_id: str, *, th: dict) -> tuple[c.SheetModel, dict[str, str]]:
        self.calls["write_sheet"] += 1
        model = c.SheetModel(run_id=run_id, config_hash=c.config_hash(th))
        digest = c.write_json(paths.sheet_json(run_id), model)
        paths.sheet_html(run_id).write_text("<html><body>sheet</body></html>", encoding="utf-8")
        return model, {"sheet.json": digest}

    def seal_run(self, run_id: str, originals: list[Path], *, now: str, camera_time: str) -> tuple[c.SealManifest, c.ThreeClocks]:
        self.calls["seal_run"] += 1
        files = [c.ManifestEntry(path=paths.rel(run_id, p), sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size) for p in originals]
        manifest = c.SealManifest(run_id=run_id, files=files, stamps=c.Stamps(manifest_ots="pending", tsr_time=now, tsa_serial="1", tsa_issuer="freetsa"))
        c.write_json(paths.manifest_json(run_id), manifest)
        minutes: Any = "camera time not present" if camera_time == "camera time not present" else int((parse_ts(now) - parse_ts(camera_time)).total_seconds() // 60)
        return manifest, c.ThreeClocks(sealed_at=now, sealed_minutes_after_camera=minutes, camera_time=None if camera_time == "camera time not present" else camera_time)

    def write_manifest_run(self, run_id: str, *, deletions: list, events_head_hash: str, config_hash: str, app_commit: str, pdf_sha256: str | None) -> c.SealManifest:
        manifest = c.SealManifest(run_id=run_id, files=[], deletions=deletions, events_head_hash=events_head_hash or None, config_hash=config_hash or None, app_commit=app_commit or None, pdf_sha256=pdf_sha256)
        c.write_json(paths.manifest_run_json(run_id), manifest)
        return manifest


# Fixtures ----------------------------------------------------------------------------------------------------------


@pytest.fixture
def runs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setattr(paths, "RUNS_DIR", root)
    monkeypatch.setattr(config, "RUNS_DIR", root)
    monkeypatch.setattr(ledger, "LEDGER_PATH", root / "ledger.jsonl")
    monkeypatch.setenv("APP_COMMIT", "abc1234")
    monkeypatch.delenv("WALKAROUND_SAMPLE_RUN", raising=False)
    return root


@pytest.fixture
def th() -> dict[str, Any]:
    return c.load_thresholds()


def frozen(th: dict[str, Any]) -> dict[str, Any]:
    """The thresholds as run.json stores them: the canonical JSON round trip (floats at 6 places)."""
    return json.loads(c.canonical_json(th))


def make_orch(th: dict[str, Any], fakes: Fakes | None = None, *, delays_path: Path | None = None, **kwargs: Any) -> Orchestrator:
    fakes = fakes or Fakes()
    return Orchestrator(th=th, engine=c.SamEngine.META_MODEL_API, max_concurrent_runs=2, media_workers=2, decode_workers=1, events=FileEventLog(), overrides=fakes.overrides(), delays_path=delays_path or Path("/nonexistent/delays.json"), **kwargs)


def stage_chunks(orch: Orchestrator, upload_id: str, data: bytes, run_id: str | None = None) -> str:
    size = int(orch.th["media"]["chunk_bytes"])
    for index, start in enumerate(range(0, max(len(data), 1), size)):
        orch.chunk_path(upload_id, index, run_id).write_bytes(data[start : start + size])
    return hashlib.sha256(data).hexdigest()


async def upload_pair(orch: Orchestrator, run: c.RunManifest, *, roles: tuple[str, ...] = ("pickup", "return")) -> None:
    for role in roles:
        data = f"{role} walk bytes ".encode() * 1000
        digest = stage_chunks(orch, f"up-{role}", data)
        await orch.finalize_upload(run.run_id, upload_id=f"up-{role}", role=role, sha256_browser=digest, bytes=len(data), filename=f"{role}.mp4")


def kinds(orch: Orchestrator, run_id: str) -> list[str]:
    return [e.kind.value for e in orch.read_events(run_id)]


def pending_kinds(orch: Orchestrator, run_id: str) -> list[str]:
    return [e.kind.value for e in orch.pending_events(run_id)]


def stage_names(orch: Orchestrator, run_id: str) -> list[str]:
    return [e.payload["name"] for e in orch.read_events(run_id) if e.kind == c.EventKind.STAGE]


def write_ledger_row(ts: str, run_id: str | None, usd: float) -> None:
    with open(ledger.LEDGER_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": ts, "kind": "sam_video", "model": "sam-3.1", "run_id": run_id, "note": "t", "usd": usd}) + "\n")


def write_run(run_id: str, *, created_at: str, kind: c.RunKind, status: c.RunStatus = c.RunStatus.TRACKING, live: bool = True) -> None:
    """A run on disk with a manifest, so its chain has a genesis."""
    paths.ensure_run_layout(run_id)
    run = c.RunManifest(run_id=run_id, created_at=created_at, kind=kind, status=status, live_or_replay=c.LiveOrReplay.LIVE if live else c.LiveOrReplay.REPLAY)
    c.write_json(paths.run_json(run_id), run)
    c.write_json(paths.manifest_json(run_id), c.SealManifest(run_id=run_id, files=[]))


# Helpers under test --------------------------------------------------------------------------------------------------


def test_run_id_is_a_sortable_ulid() -> None:
    first = new_run_id(ms=1_700_000_000_000)
    second = new_run_id(ms=1_700_000_000_001)
    assert len(first) == 26 and re.fullmatch(r"[0-9A-HJKMNP-TV-Z]{26}", first)
    assert first[:10] == new_run_id(ms=1_700_000_000_000)[:10]
    assert first < second


def test_et_day_of_handles_ledger_and_utc_stamps() -> None:
    assert et_day_of("2026-10-26T21:30:00-0400") == "2026-10-26"
    assert et_day_of("2026-10-27T03:30:00Z") == "2026-10-26"  # 11:30 PM ET the day before
    assert et_day_of("2026-10-27T04:00:00.000Z") == "2026-10-27"
    assert utc_now().endswith("Z") and parse_ts(utc_now()).tzinfo is not None


def test_stage_table_covers_the_section_5_order() -> None:
    assert [s.name for s in STAGES] == ["seal", "media", "files", "lanes", "panels", "blur", "marks", "staysput", "narration", "match", "sheet", "close"]
    assert [s for s in STAGES if s.fatal] == [STAGES[1]]
    assert [s.name for s in STAGES if s.pair_only] == ["match"]


# The event log stand-in ----------------------------------------------------------------------------------------------


def test_file_event_log_chains_from_the_manifest_and_refuses_without_one(runs_dir: Path) -> None:
    log = FileEventLog()
    paths.ensure_run_layout("RUNLOG1")
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        log.append("RUNLOG1", c.EventKind.RUN_CREATED, {})
    c.write_json(paths.manifest_json("RUNLOG1"), c.SealManifest(run_id="RUNLOG1", files=[]))
    genesis = hashlib.sha256(paths.manifest_json("RUNLOG1").read_bytes()).hexdigest()
    first = log.append("RUNLOG1", c.EventKind.RUN_CREATED, {"kind": "pair"}, ts="2026-10-27T14:00:00.000Z")
    second = log.append("RUNLOG1", c.EventKind.STAGE, {"name": "sealing", "pct": 5})
    assert (first.seq, first.prev, second.seq, second.prev) == (0, genesis, 1, first.hash)
    assert first.check() and second.check() and log.head_hash("RUNLOG1") == second.hash
    assert [e.seq for e in log.read("RUNLOG1", since_seq=0)] == [1]
    assert log.verify_chain("RUNLOG1", manifest_sha256=genesis)
    lines = paths.events_path("RUNLOG1").read_text(encoding="utf-8").splitlines()
    lines[1] = lines[1].replace('"sealing"', '"sheet"')
    paths.events_path("RUNLOG1").write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert not FileEventLog().verify_chain("RUNLOG1", manifest_sha256=genesis)


async def test_file_event_log_subscribe_gives_history_then_live_and_stops_at_the_end(runs_dir: Path) -> None:
    log = FileEventLog()
    write_run("RUNLOG2", created_at="2026-10-27T14:00:00Z", kind=c.RunKind.PAIR)
    log.append("RUNLOG2", c.EventKind.RUN_CREATED, {})
    log.append("RUNLOG2", c.EventKind.STAGE, {"name": "sealing", "pct": 5})

    async def consume() -> list[int]:
        return [e.seq async for e in log.subscribe("RUNLOG2", since_seq=0)]

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.01)
    log.append("RUNLOG2", c.EventKind.STAGE, {"name": "done", "pct": 100})
    log.append("RUNLOG2", c.EventKind.RUN_COMPLETED, {"status": "done"})
    assert await asyncio.wait_for(task, 2) == [1, 2, 3]
    assert [e.seq async for e in log.subscribe("RUNLOG2", since_seq=2)] == [3]  # an ended run never waits
    assert [e.seq async for e in log.subscribe("RUNLOG2", since_seq=3)] == []


# Create and upload -------------------------------------------------------------------------------------------------


async def test_create_run_freezes_thresholds_and_queues_run_created_until_the_seal(runs_dir: Path, th: dict) -> None:
    orch = make_orch(th)
    run = await orch.create_run(mode=c.RunMode.CAR, kind=c.RunKind.PAIR, keep_plate=True)
    on_disk = orch.load_run(run.run_id)
    assert on_disk.config_hash == c.config_hash(th) == c.config_hash(on_disk.thresholds)
    assert on_disk.thresholds == frozen(th) and orch.run_th(on_disk)["matching"]["tau_pre0_floor"] == th["matching"]["tau_pre0_floor"]
    assert on_disk.keep_plate is True and on_disk.app_commit == "abc1234"
    assert on_disk.spend_ceiling_cents == th["orchestrator"]["run_spend_ceiling_cents"]
    assert kinds(orch, run.run_id) == [] and pending_kinds(orch, run.run_id) == ["run.created"]
    assert (runs_dir / run.run_id / PENDING_EVENTS_NAME).is_file() and not (runs_dir / run.run_id / "events.ndjson").exists()
    assert (runs_dir / run.run_id / "quarantine").is_dir()
    assert orch.list_runs()[0].run_id == run.run_id


async def test_finalize_upload_assembles_50mb_of_chunks_and_checks_the_hash(runs_dir: Path, th: dict) -> None:
    orch = make_orch(th)
    run = await orch.create_run(mode=c.RunMode.CAR, kind=c.RunKind.PICKUP_CHECK, keep_plate=False)
    data = os.urandom(50 * 1024 * 1024)  # seven chunks, the last one short
    digest = stage_chunks(orch, "u1", data)
    record = await orch.finalize_upload(run.run_id, upload_id="u1", role="pickup", sha256_browser=digest.upper(), bytes=len(data), filename="IMG_1.MOV")
    assert record.hash_match is True and record.sha256_server == digest and record.chunks_received == 7
    target = paths.quarantine_original(run.run_id, "pickup")
    assert target.stat().st_size == len(data) and hashlib.sha256(target.read_bytes()).hexdigest() == digest
    assert not (runs_dir / "_uploads" / "u1").exists()
    assert pending_kinds(orch, run.run_id) == ["run.created", "upload.started", "upload.done"]
    assert orch.load_video(run.run_id, "pickup").quarantine_path == "quarantine/pickup.orig"


async def test_finalize_upload_mismatch_and_missing_chunks(runs_dir: Path, th: dict) -> None:
    orch = make_orch(th)
    run = await orch.create_run(mode=c.RunMode.CAR, kind=c.RunKind.PICKUP_CHECK, keep_plate=False)
    data = b"x" * 1000
    stage_chunks(orch, "u2", data)
    with pytest.raises(HashMismatch) as info:
        await orch.finalize_upload(run.run_id, upload_id="u2", role="pickup", sha256_browser="b" * 64, bytes=len(data), filename="a.mp4")
    assert info.value.sha256_browser == "b" * 64 and info.value.sha256_server == hashlib.sha256(data).hexdigest()
    assert not paths.quarantine_original(run.run_id, "pickup").exists()
    orch.chunk_path("u3", 0).write_bytes(b"first chunk only")
    with pytest.raises(MissingChunks) as missing:
        await orch.finalize_upload(run.run_id, upload_id="u3", role="return", sha256_browser="a" * 64, bytes=3 * 8 * 1024 * 1024, filename="b.mp4")
    assert missing.value.missing == [1, 2]


# The pipeline ------------------------------------------------------------------------------------------------------


async def test_pipeline_runs_every_stage_in_order_with_fakes(runs_dir: Path, th: dict) -> None:
    fakes = Fakes()
    orch = make_orch(th, fakes)
    run = await orch.create_run(mode=c.RunMode.CAR, kind=c.RunKind.PAIR, keep_plate=False)
    await upload_pair(orch, run)
    final = await orch.run_pipeline(run.run_id)
    assert final.status == c.RunStatus.DONE and final.errors == []
    assert final.clocks.sealed_at and final.clocks.first_model_call_at == "2026-10-27T15:00:00Z"
    assert final.clocks.sealed_minutes_after_camera is not None
    assert final.events_head_hash == orch.events.head_hash(run.run_id)
    events = orch.read_events(run.run_id)
    seen = [e.kind.value for e in events]
    assert seen[:5] == ["run.created", "upload.started", "upload.done", "upload.started", "upload.done"]
    assert events[0].seq == 0 and events[0].prev == hashlib.sha256(paths.manifest_json(run.run_id).read_bytes()).hexdigest()
    assert [e.seq for e in events] == list(range(len(events))) and not (runs_dir / run.run_id / PENDING_EVENTS_NAME).exists()
    assert 5 == seen.index("stage") < seen.index("seal.manifest")  # the sealing stage event waited with run.created
    assert stage_names(orch, run.run_id) == ["sealing", "sealed", "converting", "converting", "tracking", "tracking", "blurring", "marking", "staysput", "staysput", "matching", "sheet", "sheet", "done"]
    for kind in ("seal.manifest", "seal.tsr", "media.proxy_done", "files.uploaded", "sam.request", "sam.first_delta", "sam.frame", "sam.lane_done", "panels.derived", "coverage.updated", "blur.done", "marks.derived", "staysput.derived", "voice.done", "narration.derived", "match.derived", "sheet.written", "files.deleted", "quarantine.deleted", "ledger.row", "run.completed"):
        assert kind in seen, kind
    assert seen[-1] == "run.completed" and seen.count("sam.lane_done") == 32
    assert fakes.lane_calls == [("pickup", [paths.lane_id(n) for n in c.LANE_NOUNS]), ("return", [paths.lane_id(n) for n in c.LANE_NOUNS])]
    assert fakes.calls["derive_panels_lanes_pickup"] == 16
    frame_events = [e for e in events if e.kind == c.EventKind.SAM_FRAME]
    assert frame_events[0].payload["objects"] == [{"id": "0", "box": [1, 2, 30, 40], "c": 0.7}] and "mask" not in frame_events[0].payload
    assert not list(paths.quarantine_dir(run.run_id).rglob("*.orig"))
    deletions = c.read_json(paths.deletions_json(run.run_id), c.Deletions)
    assert {r.kind for r in deletions.rows} == {c.DeletionKind.FILE, c.DeletionKind.FILES_API, c.DeletionKind.QUARANTINE}
    assert paths.manifest_run_json(run.run_id).is_file() and paths.sheet_html(run.run_id).is_file()
    assert orch.events.verify_chain(run.run_id, manifest_sha256=hashlib.sha256(paths.manifest_json(run.run_id).read_bytes()).hexdigest())
    coverage = c.read_json(paths.coverage_json(run.run_id), c.Coverage)
    assert set(coverage.per_role) == {"pickup", "return"}
    for stage in orch.stages_for(final):
        assert orch.stage_done(final, stage), stage.name


async def test_pre_seal_events_keep_their_timestamps_and_order(runs_dir: Path, th: dict) -> None:
    counter = itertools.count()

    def now() -> str:
        n = next(counter)
        return f"2026-10-27T14:{n // 60:02d}:{n % 60:02d}.000Z"

    orch = make_orch(th, now=now)
    run = await orch.create_run(mode=c.RunMode.CAR, kind=c.RunKind.PICKUP_CHECK, keep_plate=False)
    await upload_pair(orch, run, roles=("pickup",))
    pending = orch.pending_events(run.run_id)
    assert [e.ts for e in pending] == ["2026-10-27T14:00:01.000Z", "2026-10-27T14:00:02.000Z", "2026-10-27T14:00:04.000Z"]  # now() also stamps created_at and received_at
    await orch.run_pipeline(run.run_id)
    chained = orch.read_events(run.run_id)[:3]
    assert [(e.kind, e.ts, e.payload) for e in chained] == [(e.kind, e.ts, e.payload) for e in pending]
    assert orch.pending_events(run.run_id) == []


async def test_pickup_check_skips_the_match_stage(runs_dir: Path, th: dict) -> None:
    fakes = Fakes()
    orch = make_orch(th, fakes)
    run = await orch.create_run(mode=c.RunMode.CAR, kind=c.RunKind.PICKUP_CHECK, keep_plate=False)
    await upload_pair(orch, run, roles=("pickup",))
    final = await orch.run_pipeline(run.run_id)
    assert final.status == c.RunStatus.DONE and fakes.calls["derive_match"] == 0
    assert fakes.lane_calls == [("pickup", [paths.lane_id(n) for n in c.LANE_NOUNS])]
    assert "matching" not in stage_names(orch, run.run_id)


async def test_resume_after_a_simulated_crash_reruns_only_lanes_without_lane_done(runs_dir: Path, th: dict) -> None:
    crash_lane = paths.lane_id(c.Noun.HOOD)  # the seventh lane of the fixed order
    fakes = Fakes(crash_at=("pickup", crash_lane))
    orch = make_orch(th, fakes)
    run = await orch.create_run(mode=c.RunMode.CAR, kind=c.RunKind.PAIR, keep_plate=False)
    await upload_pair(orch, run)
    with pytest.raises(Crash):
        await orch.run_pipeline(run.run_id)
    assert orch.load_run(run.run_id).status == c.RunStatus.TRACKING
    done_before = [n for n in c.LANE_NOUNS if n not in orch.missing_lanes(run, "pickup")]
    assert done_before == list(c.LANE_NOUNS[:6])

    fakes2 = Fakes()
    orch2 = make_orch(th, fakes2)
    resumed = await orch2.resume_all()
    assert resumed == [run.run_id]
    await asyncio.gather(*orch2._tasks.values())
    final = orch2.load_run(run.run_id)
    assert final.status == c.RunStatus.DONE
    assert fakes2.lane_calls == [("pickup", [paths.lane_id(n) for n in c.LANE_NOUNS[6:]]), ("return", [paths.lane_id(n) for n in c.LANE_NOUNS])]
    assert fakes2.calls["seal_run"] == 0 and fakes2.calls["make_proxy"] == 0 and fakes2.calls["upload_file"] == 0
    names = stage_names(orch2, run.run_id)
    assert "interrupted" in names and names[-1] == "done"
    assert orch2.events.verify_chain(run.run_id, manifest_sha256=hashlib.sha256(paths.manifest_json(run.run_id).read_bytes()).hexdigest())
    assert await orch2.resume_all() == []  # a finished run stays finished


async def test_resume_starts_a_created_run_whose_uploads_are_complete(runs_dir: Path, th: dict) -> None:
    orch = make_orch(th)
    run = await orch.create_run(mode=c.RunMode.CAR, kind=c.RunKind.PICKUP_CHECK, keep_plate=False)
    await upload_pair(orch, run, roles=("pickup",))
    waiting = await orch.create_run(mode=c.RunMode.CAR, kind=c.RunKind.PAIR, keep_plate=False)  # no uploads yet
    orch2 = make_orch(th)
    assert await orch2.resume_all() == [run.run_id]
    await asyncio.gather(*orch2._tasks.values())
    assert orch2.load_run(run.run_id).status == c.RunStatus.DONE and orch2.load_run(waiting.run_id).status == c.RunStatus.CREATED


async def test_spend_ceiling_marks_remaining_lanes_not_run_and_still_writes_the_sheet(runs_dir: Path, th: dict) -> None:
    fakes = Fakes(ledger_frames=3000)  # 16 pickup lanes x $0.60 = $9.60, over the $8 ceiling
    orch = make_orch(th, fakes)
    run = await orch.create_run(mode=c.RunMode.CAR, kind=c.RunKind.PAIR, keep_plate=False)
    await upload_pair(orch, run)
    final = await orch.run_pipeline(run.run_id)
    assert final.status == c.RunStatus.PARTIAL
    assert final.api_spend_cents == 960 and orch.spend_cents(run.run_id) == 960
    assert fakes.lane_calls == [("pickup", [paths.lane_id(n) for n in c.LANE_NOUNS])]
    for noun in c.LANE_NOUNS:
        record = c.read_json(paths.lane_summary_path(run.run_id, "return", noun), c.LaneRecord)
        assert record.status == c.LaneStatus.FAILED and "spend ceiling" in (record.error or "")
    events = orch.read_events(run.run_id)
    assert sum(1 for e in events if e.kind == c.EventKind.SAM_FAILED and e.payload["role"] == "return") == 16
    partial = next(e for e in events if e.kind == c.EventKind.RUN_PARTIAL)
    assert partial.payload["missing"][:2] == ["lanes", "panels"] and "marks" in partial.payload["missing"]
    assert partial.payload["text"] == "incomplete: 0 of 16 nouns tracked" and partial.payload["tracked"] == {"pickup": 16, "return": 0}
    assert fakes.calls["derive_marks"] == 0 and fakes.calls["write_sheet"] == 1
    assert [e.stage for e in final.errors][:2] == ["lanes", "panels"]
    assert paths.sheet_html(run.run_id).is_file() and paths.manifest_run_json(run.run_id).is_file()


async def test_deadline_stops_model_stages_after_it_passes(runs_dir: Path, th: dict) -> None:
    clock = [0.0]
    fakes = Fakes(clock=clock, clock_step=800.0)  # the pickup lanes alone take longer than the 720 s deadline
    orch = make_orch(th, fakes, clock=lambda: clock[0])
    run = await orch.create_run(mode=c.RunMode.CAR, kind=c.RunKind.PAIR, keep_plate=False)
    await upload_pair(orch, run)
    final = await orch.run_pipeline(run.run_id)
    assert final.status == c.RunStatus.PARTIAL
    assert fakes.lane_calls == [("pickup", [paths.lane_id(n) for n in c.LANE_NOUNS])]
    record = c.read_json(paths.lane_summary_path(run.run_id, "return", c.Noun.CAR), c.LaneRecord)
    assert record.status == c.LaneStatus.FAILED and "deadline" in (record.error or "")
    assert any("deadline" in e.text for e in final.errors if e.stage == "marks")
    assert fakes.calls["write_sheet"] == 1 and fakes.calls["render_blurred_proxy"] == 2


async def test_a_lane_that_is_not_wired_is_a_recorded_gap_not_a_retry(runs_dir: Path, th: dict) -> None:
    fakes = Fakes()
    overrides = fakes.overrides()
    del overrides["staysput.derive_staysput"]
    orch = Orchestrator(th=th, engine=c.SamEngine.META_MODEL_API, max_concurrent_runs=1, media_workers=1, decode_workers=1, events=FileEventLog(), overrides=overrides, delays_path=Path("/nonexistent"))
    run = await orch.create_run(mode=c.RunMode.CAR, kind=c.RunKind.PICKUP_CHECK, keep_plate=False)
    await upload_pair(orch, run, roles=("pickup",))
    final = await orch.run_pipeline(run.run_id)
    assert final.status == c.RunStatus.PARTIAL
    gaps = [e for e in final.errors if e.stage == "staysput"]
    assert len(gaps) == 1 and "not wired" in gaps[0].text
    assert fakes.calls["derive_narration"] == 1 and fakes.calls["write_sheet"] == 1


async def test_a_failing_stage_is_retried_once_then_recorded(runs_dir: Path, th: dict) -> None:
    fakes = Fakes()
    attempts = []

    async def flaky_marks(run_id: str, role: str, keyframes: list, *, th: dict, count_mode: bool = False) -> tuple[list, list, list]:
        attempts.append(role)
        raise RuntimeError("spark 504")

    orch = make_orch(th, fakes)
    orch.overrides["marks.derive_marks"] = flaky_marks
    run = await orch.create_run(mode=c.RunMode.CAR, kind=c.RunKind.PICKUP_CHECK, keep_plate=False)
    await upload_pair(orch, run, roles=("pickup",))
    final = await orch.run_pipeline(run.run_id)
    assert attempts == ["pickup", "pickup"]
    assert final.status == c.RunStatus.PARTIAL and [e.stage for e in final.errors] == ["marks"] and "spark 504" in final.errors[0].text


async def test_media_failure_is_fatal(runs_dir: Path, th: dict) -> None:
    fakes = Fakes()
    orch = make_orch(th, fakes)

    def broken(src: Path, dst: Path, **kwargs: Any) -> Any:
        raise RuntimeError("could not read this video")

    orch.overrides["video.make_proxy"] = broken
    run = await orch.create_run(mode=c.RunMode.CAR, kind=c.RunKind.PICKUP_CHECK, keep_plate=False)
    await upload_pair(orch, run, roles=("pickup",))
    final = await orch.run_pipeline(run.run_id)
    assert final.status == c.RunStatus.FAILED and final.errors[-1].stage == "media"
    assert kinds(orch, run.run_id)[-1] == "run.failed" and fakes.calls["write_sheet"] == 0


async def test_stages_compute_with_the_frozen_thresholds(runs_dir: Path, th: dict) -> None:
    seen: list[dict] = []
    fakes = Fakes()
    orch = make_orch(th, fakes)
    original = fakes.write_sheet

    def spy(run_id: str, *, th: dict) -> Any:
        seen.append(th)
        return original(run_id, th=th)

    orch.overrides["sheet.write_sheet"] = spy
    run = await orch.create_run(mode=c.RunMode.CAR, kind=c.RunKind.PICKUP_CHECK, keep_plate=False)
    await upload_pair(orch, run, roles=("pickup",))
    await orch.run_pipeline(run.run_id)
    assert seen == [frozen(th)] and seen[0] is not th


# Caps and ceiling from the ledger --------------------------------------------------------------------------------------


def test_caps_remaining_counts_runs_and_dollars_per_et_day(runs_dir: Path, th: dict) -> None:
    orch = make_orch(th)
    write_run("RUNPICKUP1", created_at="2026-10-27T14:00:00Z", kind=c.RunKind.PICKUP_CHECK)
    write_run("RUNPICKUP2", created_at="2026-10-27T15:00:00Z", kind=c.RunKind.PICKUP_CHECK)
    write_run("RUNPAIR1", created_at="2026-10-27T16:00:00Z", kind=c.RunKind.PAIR)
    write_run("RUNREPLAY", created_at="2026-10-27T16:30:00Z", kind=c.RunKind.PAIR, live=False)
    write_run("RUNCREATED", created_at="2026-10-27T17:00:00Z", kind=c.RunKind.PICKUP_CHECK, status=c.RunStatus.CREATED)
    write_run("RUNYESTERDAY", created_at="2026-10-26T20:00:00Z", kind=c.RunKind.PICKUP_CHECK)
    write_ledger_row("2026-10-27T10:00:00-0400", "RUNPICKUP1", 2.6)
    write_ledger_row("2026-10-27T11:00:00-0400", "RUNPAIR1", 6.2)
    write_ledger_row("2026-10-27T23:30:00-0400", "gate_E2", 1.0)  # a D1 script row: dollars count, no run
    write_ledger_row("2026-10-26T22:00:00-0400", "RUNYESTERDAY", 2.6)
    left = orch.caps_remaining(et_day="2026-10-27")
    assert left == {"pickup": th["orchestrator"]["pickup_runs_per_day"] - 2, "pair": th["orchestrator"]["pair_runs_per_day"] - 1}
    assert orch.caps_remaining(et_day="2026-10-26") == {"pickup": th["orchestrator"]["pickup_runs_per_day"] - 1, "pair": th["orchestrator"]["pair_runs_per_day"]}
    assert orch.spend_cents("RUNPAIR1") == 620


def test_daily_card_ceiling_zeroes_both_caps(runs_dir: Path, th: dict) -> None:
    orch = make_orch(th)
    for hour in range(8):
        write_ledger_row(f"2026-10-27T{10 + hour:02d}:00:00-0400", None, 2.6)  # $20.80 in one ET day
    assert orch.caps_remaining(et_day="2026-10-27") == {"pickup": 0, "pair": 0}


def test_check_can_start_raises_on_pause_and_on_caps(runs_dir: Path, th: dict) -> None:
    orch = make_orch(th, now=lambda: "2026-10-27T18:00:00.000Z")
    assert orch.check_can_start(c.RunKind.PAIR)["pair"] == th["orchestrator"]["pair_runs_per_day"]
    write_run("RUNPAIRX", created_at="2026-10-27T16:00:00Z", kind=c.RunKind.PAIR)
    with pytest.raises(CapsExceeded) as info:
        orch.check_can_start(c.RunKind.PAIR)
    assert info.value.pairs_left == 0 and info.value.pickup_left == th["orchestrator"]["pickup_runs_per_day"]
    orch.check_can_start(c.RunKind.PICKUP_CHECK)
    orch.paused = True
    with pytest.raises(CapsExceeded, match="paused"):
        orch.check_can_start(c.RunKind.PICKUP_CHECK)


async def test_health_probe_pauses_after_three_failures_and_clears_on_a_pass(runs_dir: Path, th: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_API_KEY", "LLM_test_key")
    orch = make_orch(th)
    statuses = iter([500, 503, 500, 200])

    async def fake_get(url: str, headers: dict[str, str]) -> int:
        assert url.endswith("/v1/models") and headers["Authorization"].startswith("Bearer ")
        return next(statuses)

    orch.http_get = fake_get
    assert [await orch.health_probe() for _ in range(2)] == [False, False] and orch.paused is False
    assert await orch.health_probe() is False and orch.paused is True
    assert await orch.health_probe() is True and orch.paused is False and orch.probe_ok is True


# Replay driver -------------------------------------------------------------------------------------------------------


def seed_replay_run(orch: Orchestrator, run_id: str, stamps: list[str]) -> list[c.Event]:
    write_run(run_id, created_at="2026-10-27T14:00:00Z", kind=c.RunKind.PAIR, status=c.RunStatus.DONE)
    plan = [
        (c.EventKind.RUN_CREATED, {}),
        (c.EventKind.STAGE, {"name": "sealing", "pct": 5}),
        (c.EventKind.SEAL_MANIFEST, {"sha256": "a" * 64}),
        (c.EventKind.STAGE, {"name": "tracking", "pct": 45}),
        (c.EventKind.SAM_LANE_DONE, {"noun": "car"}),
        (c.EventKind.SAM_LANE_DONE, {"noun": "wheel"}),
        (c.EventKind.SAM_LANE_DONE, {"noun": "hood"}),
        (c.EventKind.STAGE, {"name": "done", "pct": 100}),
        (c.EventKind.RUN_COMPLETED, {"status": "done"}),
    ]
    return [orch.events.append(run_id, kind, payload, ts=ts) for (kind, payload), ts in zip(plan, stamps)]


STAMPS = ["2026-10-27T14:00:00.000Z", "2026-10-27T14:00:00.500Z", "2026-10-27T14:00:01.000Z", "2026-10-27T14:00:06.000Z", "2026-10-27T14:00:06.200Z", "2026-10-27T14:00:06.400Z", "2026-10-27T14:00:06.600Z", "2026-10-27T14:00:09.600Z", "2026-10-27T14:00:09.700Z"]


async def test_fixed_replay_spreads_each_stage_total_over_its_events(runs_dir: Path, th: dict, tmp_path: Path) -> None:
    delays = tmp_path / "delays.json"
    delays.write_text(json.dumps({"created": 100, "sealing": 200, "tracking": 400, "done": 300}), encoding="utf-8")
    orch = make_orch(th, delays_path=delays)
    events = seed_replay_run(orch, "RUNREPLAY1", STAMPS)
    waits = orch.replay_delays(events, mode=c.ReplayMode.FIXED)
    assert [round(w * 1000) for w in waits] == [100, 100, 100, 100, 100, 100, 100, 150, 150]
    slept: list[float] = []

    async def record_sleep(seconds: float) -> None:
        slept.append(seconds)

    orch.sleep = record_sleep
    replayed = [e async for e in orch.replay_events("RUNREPLAY1", mode=c.ReplayMode.FIXED)]
    assert replayed == events and slept == waits
    payload = orch.replay_payload("RUNREPLAY1", mode=c.ReplayMode.FIXED)
    assert payload.recorded_on == "2026-10-27" and payload.delays_ms == {"created": 100, "sealing": 200, "tracking": 400, "done": 300}
    assert payload.events == events and payload.mode == c.ReplayMode.FIXED


async def test_fixed_replay_wall_clock_matches_the_stage_totals(runs_dir: Path, th: dict, tmp_path: Path) -> None:
    delays = tmp_path / "delays.json"
    delays.write_text(json.dumps({"created": 10, "sealing": 20, "tracking": 40, "done": 30}), encoding="utf-8")
    orch = make_orch(th, delays_path=delays)
    seed_replay_run(orch, "RUNREPLAY2", STAMPS)
    t0 = time.perf_counter()
    count = len([e async for e in orch.replay_events("RUNREPLAY2", mode=c.ReplayMode.FIXED)])
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert count == 9 and 95 <= elapsed_ms < 600  # 100 ms of configured delay; the upper bound only guards a stall


async def test_recorded_replay_uses_capped_ts_deltas_and_instant_uses_none(runs_dir: Path, th: dict) -> None:
    orch = make_orch(th)
    events = seed_replay_run(orch, "RUNREPLAY3", STAMPS)
    waits = orch.replay_delays(events, mode=c.ReplayMode.RECORDED)
    assert [round(w, 3) for w in waits] == [0.0, 0.5, 0.5, 2.0, 0.2, 0.2, 0.2, 2.0, 0.1]  # the 5 s and 3 s gaps cap at 2 s
    assert orch.replay_delays(events, mode=c.ReplayMode.INSTANT) == [0.0] * 9
    assert orch.replay_payload("RUNREPLAY3", mode=c.ReplayMode.RECORDED).delays_ms == {}


def test_the_checked_in_delays_json_names_every_stage_the_pipeline_emits(th: dict) -> None:
    delays = Orchestrator(th=th, engine=c.SamEngine.META_MODEL_API, max_concurrent_runs=1, media_workers=1, decode_workers=1, events=FileEventLog()).load_delays()
    emitted = {"created", c.RunStatus.SEALED.value, c.RunStatus.DONE.value} | {s.status.value for s in STAGES}
    assert emitted <= set(delays), sorted(emitted - set(delays))
    assert all(isinstance(v, int) and v >= 0 for v in delays.values())


# Rebuild and verify --------------------------------------------------------------------------------------------------


async def test_rebuild_into_out_dir_and_verify_reports_a_tampered_file(runs_dir: Path, th: dict, tmp_path: Path) -> None:
    fakes = Fakes()
    orch = make_orch(th, fakes)
    run = await orch.create_run(mode=c.RunMode.CAR, kind=c.RunKind.PAIR, keep_plate=False)
    await upload_pair(orch, run)
    await orch.run_pipeline(run.run_id)
    out = tmp_path / "out"
    result = orch.rebuild(run.run_id, out=out)
    assert (out / run.run_id / "derived" / "matches.json").is_file() and not (out / run.run_id / "quarantine").exists()
    assert (out / "ledger.jsonl").is_file()
    assert result["matches.json"] == hashlib.sha256(paths.matches_json(run.run_id).read_bytes()).hexdigest()
    assert result["coverage.json"] == hashlib.sha256(paths.coverage_json(run.run_id).read_bytes()).hexdigest()
    assert result["sheet.json"] == hashlib.sha256(paths.sheet_json(run.run_id).read_bytes()).hexdigest()
    assert result["marks.json"].startswith("not rebuilt")
    assert paths.RUNS_DIR == runs_dir  # the root swap is undone
    assert orch.verify(run.run_id, out=tmp_path / "verify") == []
    paths.matches_json(run.run_id).write_text('[{"tampered": true}]\n', encoding="utf-8")
    assert orch.verify(run.run_id, out=tmp_path / "verify2") == ["rebuild differs: derived/matches.json"]
    events_file = paths.events_path(run.run_id)
    lines = events_file.read_text(encoding="utf-8").splitlines()
    lines[3] = lines[3].replace("upload.started", "upload.done")
    events_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    problems = orch.verify(run.run_id, out=tmp_path / "verify3")
    assert problems[0] == "rebuild differs: derived/matches.json" and sum(p.startswith("event chain broken") for p in problems) == 1


async def test_rebuild_never_touches_the_frozen_model_era_files(runs_dir: Path, th: dict) -> None:
    fakes = Fakes()
    orch = make_orch(th, fakes)
    run = await orch.create_run(mode=c.RunMode.CAR, kind=c.RunKind.PAIR, keep_plate=False)
    await upload_pair(orch, run)
    await orch.run_pipeline(run.run_id)
    c.write_json(paths.spark_json(run.run_id), [])
    c.write_json(paths.voice_json(run.run_id), [])
    frozen_files = [paths.events_path(run.run_id), paths.spark_json(run.run_id), paths.voice_json(run.run_id), paths.tracks_json(run.run_id), paths.keyframes_json(run.run_id)]
    before = [(p.read_bytes(), p.stat().st_mtime_ns) for p in frozen_files]
    orch.rebuild(run.run_id)
    assert [(p.read_bytes(), p.stat().st_mtime_ns) for p in frozen_files] == before
    assert fakes.calls["segment_video_many"] == 2 and fakes.calls["derive_marks"] == 2 and fakes.calls["derive_match"] == 2


# Top-up ---------------------------------------------------------------------------------------------------------------


async def test_run_topup_reruns_only_nouns_to_rerun_and_measures_the_time(runs_dir: Path, th: dict) -> None:
    fakes = Fakes()
    fakes.missing_panels = [c.PanelId.LEFT_REAR_WHEEL]
    clock = [0.0]
    orch = make_orch(th, fakes, clock=lambda: clock[0])
    run = await orch.create_run(mode=c.RunMode.CAR, kind=c.RunKind.PICKUP_CHECK, keep_plate=False)
    await upload_pair(orch, run, roles=("pickup",))
    await orch.run_pipeline(run.run_id)
    assert "topup.requested" in kinds(orch, run.run_id)
    coverage = c.read_json(paths.coverage_json(run.run_id), c.Coverage)
    coverage.nouns_to_rerun = [c.Noun.WHEEL, c.Noun.CAR, c.Noun.LICENSE_PLATE, c.Noun.FACE]
    c.write_json(paths.coverage_json(run.run_id), coverage)
    data = b"topup clip " * 500
    digest = stage_chunks(orch, "t1", data)
    await orch.finalize_upload(run.run_id, upload_id="t1", role="topup_1", sha256_browser=digest, bytes=len(data), filename="topup.mp4")
    fakes.clock_step = 41.2
    fakes.clock = clock
    result = await orch.run_topup(run.run_id, "topup_1")
    assert fakes.lane_calls[-1] == ("topup_1", ["wheel", "car", "license_plate", "face"])
    assert result.rerun_seconds == 41.2 and set(result.per_role) == {"pickup", "topup_1"}
    assert kinds(orch, run.run_id)[-1] == "coverage.updated"
    assert c.read_json(paths.coverage_json(run.run_id), c.Coverage).rerun_seconds == 41.2


# CLI -----------------------------------------------------------------------------------------------------------------


def test_cli_parses_every_subcommand() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["run", "pickup.mp4", "return.mov", "--keep-plate"])
    assert (args.command, args.pickup, args.return_, args.keep_plate, args.replay) == ("run", Path("pickup.mp4"), Path("return.mov"), True, None)
    args = parser.parse_args(["run", "--replay", "01ARZ3NDEKTSV4RRFFQ69G5FAV", "--mode", "instant"])
    assert args.pickup is None and args.replay == "01ARZ3NDEKTSV4RRFFQ69G5FAV" and args.mode == "instant"
    args = parser.parse_args(["rebuild", "RUN1", "--out", "/tmp/x"])
    assert args.command == "rebuild" and args.out == Path("/tmp/x") and args.thresholds is None
    assert cli.build_parser().parse_args(["verify", "RUN1"]).command == "verify"
    args = parser.parse_args(["serve", "--port", "9000"])
    assert args.command == "serve" and args.port == 9000
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--mode", "fast"])
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_cli_run_refuses_to_spend_without_live(runs_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.delenv("LIVE", raising=False)
    clip = tmp_path / "walk.mp4"
    clip.write_bytes(b"not really a video")
    assert cli.main(["run", str(clip)]) == 2
    assert "set LIVE=1" in capsys.readouterr().err
    assert cli.main(["run", str(tmp_path / "missing.mp4")]) == 2
    assert cli.main(["run"]) == 2


def test_cli_run_with_live_drives_the_whole_pipeline_on_local_files(runs_dir: Path, th: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    fakes = Fakes()
    orch = make_orch(th, fakes)
    monkeypatch.setattr(cli, "make_orchestrator", lambda th=None: orch)
    monkeypatch.setenv("LIVE", "1")
    pickup, ret = tmp_path / "pickup.mp4", tmp_path / "return.mp4"
    pickup.write_bytes(b"pickup bytes" * 100)
    ret.write_bytes(b"return bytes" * 100)
    assert cli.main(["run", str(pickup), str(ret), "--quiet"]) == 0
    final = c.RunManifest.model_validate_json(capsys.readouterr().out.strip().splitlines()[-1])
    assert final.status == c.RunStatus.DONE and final.kind == c.RunKind.PAIR
    assert fakes.calls["derive_match"] == 1 and paths.sheet_html(final.run_id).is_file()
    video = orch.load_video(final.run_id, "pickup")
    assert video is not None and video.hash_match is True and video.original_filename == "pickup.mp4"


def test_cli_replay_prints_events_as_json_lines(runs_dir: Path, th: dict, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    orch = make_orch(th)
    events = seed_replay_run(orch, "RUNCLI1", STAMPS)
    monkeypatch.setattr(cli, "make_orchestrator", lambda th=None: orch)
    assert cli.main(["run", "--replay", "RUNCLI1", "--mode", "instant"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert [c.Event.model_validate_json(line) for line in lines] == events
    assert cli.main(["run", "--replay", "NOPE"]) == 1
    assert cli.main(["verify", "NOPE"]) == 1 and cli.main(["rebuild", "NOPE"]) == 1


def test_cli_rebuild_and_verify_print_results(runs_dir: Path, th: dict, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    fakes = Fakes()
    orch = make_orch(th, fakes)

    async def build() -> str:
        run = await orch.create_run(mode=c.RunMode.CAR, kind=c.RunKind.PAIR, keep_plate=False)
        await upload_pair(orch, run)
        await orch.run_pipeline(run.run_id)
        return run.run_id

    run_id = asyncio.run(build())
    monkeypatch.setattr(cli, "make_orchestrator", lambda th=None: orch)
    assert cli.main(["verify", run_id]) == 0
    assert capsys.readouterr().out.strip() == "rebuild identical"
    assert cli.main(["rebuild", run_id]) == 0
    out = capsys.readouterr().out
    assert "matches.json: " in out and "sheet.json: " in out and "coverage.json: " in out
