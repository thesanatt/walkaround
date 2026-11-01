"""Orchestrator (M14) and CLI tests: the state machine with injected fakes, no model call, no network.

FileEventLog (the orchestrator's stand-in for walkaround.eventlog, same chain rule) records the events, and Fakes
stands in for every other lane's entry point, writing the files each stage's completion check reads the way the real
lanes write them (panels.derive keeps the other roles' entries, marks.derive_marks stamps marks_status, a lane that
raised inside the client writes no summary).
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import os
import re
import sys
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
    RebuildRefused,
    UploadRefused,
    VideoUnreadable,
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


def keyframe(role: str, n: int = 1) -> c.Keyframe:
    kf_id = f"{role}_hood_kf{n}"
    score = c.KeyframeScore(area=0.2, completeness=1.0, sharpness=1.0, glare=0.0, frontality=1.0, total=1.0)
    crop = c.Crop(path=f"crops/{role}/{kf_id}.jpg", w=64, h=48, scale=1.0, dx=0.0, dy=0.0)
    return c.Keyframe(kf_id=kf_id, role=role, panel_id=c.PanelId.HOOD, track_id=f"{role}_hood_t1", f_proxy=4, pts=0.25, score=score, aspect_ratio=1.3, box_src=[0, 0, 64, 48], crop=crop)


def spark_record(kf: c.Keyframe) -> c.SparkRequestRecord:
    return c.SparkRequestRecord(request_id=f"resp_spark_{kf.kf_id}", purpose=c.SparkPurpose.MARKS, model="muse-spark-1.3", schema_name="marks_v1", raw_text='{"marks":[]}', valid=True)


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
        self.fail_lanes: set[tuple[str, str]] = set()  # (role, lane) that raise inside the client: FAILED, no summary
        self.store_echo: bool | None = False
        self.proxy_frames = 88
        self.crash_derive: str | None = None
        self.voice_error: dict[str, Exception] = {}
        self.blur_keep_plate: list[bool] = []
        self.marks_roles: list[str] = []
        self.derive_roles: list[str] = []

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
            "staysput.role_done": lambda run_id, role: paths.mark_tracks_json(run_id).is_file(),
            "staysput.rebuild_staysput": lambda run_id, *, th: [],
            "staysput.request_ids": lambda run_id: [],
            "voice_client.transcribe_wav": self.transcribe_wav,
            "voice_client.voice_record": self.voice_record,
            "narration.derive_narration": self.derive_narration,
            "narration.rebuild_narration": self.rebuild_narration,
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
        pts.write_text(json.dumps([i / 15.0 for i in range(self.proxy_frames)]), encoding="utf-8")
        return c.ProxyRecord(path=paths.rel(run_id, proxy), fps=15.0, width=960, height=540, frames=self.proxy_frames, pts_map_path=paths.rel(run_id, pts), sha256=hashlib.sha256(proxy.read_bytes()).hexdigest())

    def extract_wav(self, src: Path, dst: Path, *, rate: int) -> Path:
        import wave

        with wave.open(str(dst), "wb") as fh:  # a real WAV: the orchestrator reads its length from the file
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(rate)
            fh.writeframes(b"\x00\x00" * rate)
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
        for index, (lane, text) in enumerate(prompts):
            if self.crash_at == (role, lane):
                raise Crash()
            if (role, lane) in self.fail_lanes:
                results[lane] = (c.LaneRecord(lane_id=lane, role=role, noun=c.Noun(text), prompt_text=text, media_size=list(media_size), request_index=index, status=c.LaneStatus.FAILED, error="RuntimeError: boom"), [])
                continue
            if on_event is not None:
                on_event(lane, {"type": "response.created", "response": {"id": f"resp_{role}_{lane}", "metadata": metadata, "store": self.store_echo}})
                for frame in range(6):
                    on_event(lane, {"type": "response.output_text.delta", "delta": f"<{frame}f>0<|box;x1=1;y1=2;x2=30;y2=40;w={media_size[0]};h={media_size[1]};c=0.7|>\n"})
            record = c.LaneRecord(lane_id=lane, role=role, noun=c.Noun(text), prompt_text=text, media_size=list(media_size), params=metadata, request_id=f"resp_{role}_{lane}", store_echo=self.store_echo, request_index=index, frames_sent=frames_sent, frames_returned=1, video_frames_processed=frames_sent, ids=["0"], dollars=0.0176, lane_path=f"sam/{role}/{lane}.ndjson.gz", lane_sha256="a" * 64)
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
        self.derive_roles.append(role)
        if self.crash_derive == role:
            raise Crash()
        await blur_keyframe(run_id, role, None, f_proxy=4, tracks=[], fps=fps, sizes=(media_size, media_size), keep_plate=True, th=th)
        kfs = [k for k in _read(paths.keyframes_json(run_id), c.Keyframe) if k.role != role] + [keyframe(role)]
        c.write_json(paths.tracks_json(run_id), [])
        c.write_json(paths.panels_json(run_id), [])
        c.write_json(paths.keyframes_json(run_id), kfs)
        prior = c.read_json(paths.coverage_json(run_id), c.Coverage) if paths.coverage_json(run_id).is_file() else c.Coverage()
        prior.per_role[role] = dict(HOOD_COVERAGE)
        c.write_json(paths.coverage_json(run_id), self.plan_topup(prior, th))
        return [], [], kfs, dict(HOOD_COVERAGE)

    def coverage_for_role(self, panels: list, tracks: list, keyframes: list, *, role: str, fps: float, lane_status: dict, th: dict) -> dict:
        self.calls["coverage_for_role"] += 1
        return dict(HOOD_COVERAGE)

    def plan_topup(self, coverage: c.Coverage, th: dict) -> c.Coverage:
        coverage.missing = list(self.missing_panels)
        coverage.topup_text = "film the " + ", ".join(p.value.replace("_", " ") for p in coverage.missing) if coverage.missing else ""
        return coverage

    async def blur_keyframe(self, run_id: str, role: str, frame: Any, *, f_proxy: int, tracks: list, fps: float, sizes: Any, keep_plate: bool, th: dict) -> Any:
        self.blur_keep_plate.append(keep_plate)
        return frame, c.BlurRecord()

    async def render_blurred_proxy(self, run_id: str, role: str, tracks: list, *, fps: float, th: dict) -> tuple[c.BlurredProxyRecord, c.DeletionRecord]:
        self.calls["render_blurred_proxy"] += 1
        blurred = paths.proxy_blurred_path(run_id, role)
        blurred.write_bytes(b"blurred")
        proxy = paths.proxy_path(run_id, role)
        row = c.DeletionRecord(target=paths.rel(run_id, proxy), kind=c.DeletionKind.FILE, bytes=proxy.stat().st_size if proxy.is_file() else 0, deleted_at=utc_now(), reason="blur.done")
        proxy.unlink(missing_ok=True)
        return c.BlurredProxyRecord(path=paths.rel(run_id, blurred), sha256=hashlib.sha256(b"blurred").hexdigest()), row

    async def delete_remote(self, run_id: str, videos: list[c.VideoRecord], *, response_ids: list[str] | None, now: str) -> list[c.DeletionRecord]:
        self.calls["delete_remote"] += 1
        self.response_ids = response_ids
        rows = [c.DeletionRecord(target=v.files_api.file_id, kind=c.DeletionKind.FILES_API, deleted_at=now, reason="run end", api_response={"deleted": True}) for v in videos if v.files_api and v.files_api.deleted_at is None]
        rows += [c.DeletionRecord(target=rid, kind=c.DeletionKind.RESPONSE, deleted_at=now, reason="store false refused", api_response={"deleted": True}) for rid in response_ids or []]
        return rows

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
        self.sweep_args = {"retention_hours": retention_hours, "temp_minutes": temp_minutes, "keep": set(keep_run_ids)}
        return []

    def served_file_allowed(self, run: c.RunManifest, rel_path: str, *, run_owner: bool) -> bool:
        if run.keep_plate and not run_owner and rel_path.startswith(("sheet/", "crops/")):
            return False
        return not rel_path.startswith("quarantine/") and not rel_path.endswith(".proxy.mp4")

    # marks, staysput, voice, narration
    async def derive_marks(self, run_id: str, role: str, keyframes: list, *, th: dict, count_mode: bool = False) -> tuple[list, list, list]:
        self.calls["derive_marks"] += 1
        self.marks_roles.append(role)
        mine = [k for k in keyframes if k.role == role]
        stamped = {k.kf_id: k.model_copy(update={"marks_status": c.MarksStatus.COMPLETE}) for k in mine}
        c.write_json(paths.keyframes_json(run_id), [stamped.get(k.kf_id, k) for k in _read(paths.keyframes_json(run_id), c.Keyframe)])
        c.write_json(paths.marks_json(run_id), [m for m in _read(paths.marks_json(run_id), c.Mark) if m.role != role])
        records = [spark_record(k) for k in mine]
        c.write_json(paths.spark_json(run_id), _read(paths.spark_json(run_id), c.SparkRequestRecord) + records)
        return [], list(stamped.values()), records

    async def derive_staysput(self, run_id: str, role: str, *, th: dict, stop: Any = None) -> list:
        self.calls["derive_staysput"] += 1
        c.write_json(paths.mark_tracks_json(run_id), [])
        return []

    def transcribe_wav(self, wav_path: Path, *, mode: str, keywords: list[str], run_id: str | None = None) -> Any:
        self.calls["transcribe_wav"] += 1
        assert mode == "ENDPOINTING" and len(keywords) == 21
        role = wav_path.stem
        if role in self.voice_error:
            raise self.voice_error[role]
        return SimpleNamespace(mode=mode, transcript="small chip on the hood", mode_applied=True)

    def voice_record(self, transcript: Any, *, role: str, wav_sha256: str) -> c.VoiceRecord:
        request = c.VoiceRequest(mode=c.VoiceMode.ENDPOINTING, model="muse-voice-transcribe-1.0")
        response = c.VoiceResponse(transcript=transcript.transcript, audioDurationMs=5875, turns=[c.VoiceTurn(startMs=500, endMs=3000, transcript=transcript.transcript)])
        return c.VoiceRecord(role=role, request=request, response=response, wav_sha256=wav_sha256, mode_applied=True)

    async def derive_narration(self, run_id: str, *, th: dict) -> c.Narration:
        self.calls["derive_narration"] += 1
        return self.rebuild_narration(run_id, th=th)

    def rebuild_narration(self, run_id: str, *, th: dict) -> c.Narration:
        narration = c.Narration(per_role={}, mode_applied=True)
        c.write_json(paths.narration_json(run_id), narration)
        return narration

    # match, sheet, seal
    def derive_match(self, run_id: str, *, th: dict, second_look: Any = None) -> tuple[list, list, c.Calibration]:
        self.calls["derive_match"] += 1
        c.write_json(paths.matches_json(run_id), [])
        c.write_json(paths.verdicts_json(run_id), [])
        return [], [], c.Calibration(tau_pre=0.25, tau_eye=0.5, tau_fix=0.08)

    def write_sheet(self, run_id: str, *, th: dict) -> tuple[c.SheetModel, dict[str, str]]:
        self.calls["write_sheet"] += 1
        run = c.read_json(paths.run_json(run_id), c.RunManifest)
        model = c.SheetModel(run_id=run_id, config_hash=c.config_hash(th), events_head_hash=run.events_head_hash)
        digest = c.write_json(paths.sheet_json(run_id), model)
        paths.sheet_html(run_id).write_text("<html><body>sheet</body></html>", encoding="utf-8")
        return model, {"sheet/sheet.json": digest}

    def seal_run(self, run_id: str, originals: list[Path], *, now: str, camera_time: str) -> tuple[c.SealManifest, c.ThreeClocks]:
        self.calls["seal_run"] += 1
        files = [c.ManifestEntry(path=paths.rel(run_id, p), sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size) for p in originals]
        manifest = c.SealManifest(run_id=run_id, files=files, stamps=c.Stamps(manifest_ots="pending", tsr_time=now, tsa_serial="1", tsa_issuer="freetsa"))
        c.write_json(paths.manifest_json(run_id), manifest)
        minutes: Any = "camera time not present" if camera_time == "camera time not present" else int((parse_ts(now) - parse_ts(camera_time)).total_seconds() // 60)
        return manifest, c.ThreeClocks(sealed_at=now, sealed_minutes_after_camera=minutes, camera_time=None if camera_time == "camera time not present" else camera_time)

    def write_manifest_run(self, run_id: str, *, deletions: list, events_head_hash: str, config_hash: str, app_commit: str, pdf_sha256: str | None) -> c.SealManifest:
        self.calls["write_manifest_run"] += 1
        manifest = c.SealManifest(run_id=run_id, files=[], deletions=deletions, events_head_hash=events_head_hash or None, config_hash=config_hash or None, app_commit=app_commit or None, pdf_sha256=pdf_sha256)
        c.write_json(paths.manifest_run_json(run_id), manifest)
        return manifest


def _read(path: Path, model: Any) -> list[Any]:
    return c.read_json_list(path, model) if path.is_file() else []


# Fixtures ----------------------------------------------------------------------------------------------------------


@pytest.fixture
def runs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setattr(paths, "RUNS_DIR", root)
    monkeypatch.setattr(config, "RUNS_DIR", root)
    monkeypatch.setattr(ledger, "LEDGER_PATH", root / "ledger.jsonl")
    monkeypatch.setenv("APP_COMMIT", "abc1234")
    for name in ("WALKAROUND_SAMPLE_RUN", "PICKUP_RUNS_PER_DAY", "PAIR_RUNS_PER_DAY", "RUN_SPEND_CEILING_CENTS", "RUN_RETENTION_HOURS", "SAM_ENGINE"):
        monkeypatch.delenv(name, raising=False)
    import walkaround.orchestrator as orchestrator_module

    monkeypatch.setattr(orchestrator_module, "REPO_DIR", tmp_path / "repo")  # no samples/: the repo's own sample pair stays out
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


async def new_run(orch: Orchestrator, kind: c.RunKind = c.RunKind.PAIR, *, keep_plate: bool = False) -> c.RunManifest:
    run = await orch.create_run(mode=c.RunMode.CAR, kind=kind, keep_plate=keep_plate)
    await upload_pair(orch, run, roles=orch.expected_roles(run))
    return run


def kinds(orch: Orchestrator, run_id: str) -> list[str]:
    return [e.kind.value for e in orch.read_events(run_id)]


def pending_kinds(orch: Orchestrator, run_id: str) -> list[str]:
    return [e.kind.value for e in orch.pending_events(run_id)]


def stage_names(orch: Orchestrator, run_id: str) -> list[str]:
    return [e.payload["name"] for e in orch.read_events(run_id) if e.kind == c.EventKind.STAGE]


def genesis(run_id: str) -> str:
    return hashlib.sha256(paths.manifest_json(run_id).read_bytes()).hexdigest()


def write_ledger_row(ts: str, run_id: str | None, usd: float) -> None:
    with open(ledger.LEDGER_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": ts, "kind": "sam_video", "model": "sam-3.1", "run_id": run_id, "note": "t", "usd": usd}) + "\n")


def write_run(run_id: str, *, created_at: str, kind: c.RunKind, status: c.RunStatus = c.RunStatus.TRACKING, live: bool = True, keep: bool = False, keep_plate: bool = False) -> None:
    """A run on disk with a manifest, so its chain has a genesis."""
    paths.ensure_run_layout(run_id)
    run = c.RunManifest(run_id=run_id, created_at=created_at, kind=kind, status=status, live_or_replay=c.LiveOrReplay.LIVE if live else c.LiveOrReplay.REPLAY, keep=keep, keep_plate=keep_plate)
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
    assert [s.name for s in STAGES] == ["seal", "media", "files", "lanes", "panels", "blur", "marks", "narration", "staysput", "match", "release", "sheet", "close"]
    assert [s.name for s in STAGES if s.fatal] == ["seal", "media"]
    assert [s.name for s in STAGES if s.pair_only] == ["match"]
    assert [s.name for s in STAGES if s.model] == ["lanes", "panels", "marks", "narration", "staysput"]


# The event log stand-in ----------------------------------------------------------------------------------------------


def test_file_event_log_chains_from_the_manifest_and_refuses_without_one(runs_dir: Path) -> None:
    log = FileEventLog()
    paths.ensure_run_layout("RUNLOG1")
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        log.append("RUNLOG1", c.EventKind.RUN_CREATED, {})
    c.write_json(paths.manifest_json("RUNLOG1"), c.SealManifest(run_id="RUNLOG1", files=[]))
    first = log.append("RUNLOG1", c.EventKind.RUN_CREATED, {"kind": "pair"}, ts="2026-10-27T14:00:00.000Z")
    second = log.append("RUNLOG1", c.EventKind.STAGE, {"name": "sealing", "pct": 5})
    assert (first.seq, first.prev, second.seq, second.prev) == (0, genesis("RUNLOG1"), 1, first.hash)
    assert first.check() and second.check() and log.head_hash("RUNLOG1") == second.hash
    assert [e.seq for e in log.read("RUNLOG1", since_seq=0)] == [1]
    assert log.verify_chain("RUNLOG1", manifest_sha256=genesis("RUNLOG1"))
    lines = paths.events_path("RUNLOG1").read_text(encoding="utf-8").splitlines()
    lines[1] = lines[1].replace('"sealing"', '"sheet"')
    paths.events_path("RUNLOG1").write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert not FileEventLog().verify_chain("RUNLOG1", manifest_sha256=genesis("RUNLOG1"))


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
    orch = make_orch(th, now=lambda: "2026-10-27T14:00:00.000Z")
    run = await orch.create_run(mode=c.RunMode.CAR, kind=c.RunKind.PAIR, keep_plate=True)
    on_disk = orch.load_run(run.run_id)
    assert on_disk.config_hash == c.config_hash(th) == c.config_hash(on_disk.thresholds)
    assert on_disk.thresholds == frozen(th) and orch.run_th(on_disk)["matching"]["tau_pre0_floor"] == th["matching"]["tau_pre0_floor"]
    assert on_disk.keep_plate is True and on_disk.app_commit == "abc1234"
    assert on_disk.spend_ceiling_cents == th["orchestrator"]["run_spend_ceiling_cents"]
    assert on_disk.retention_deadline == "2026-10-28T14:00:00Z"  # 24 h retention
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
    run = await orch.create_run(mode=c.RunMode.CAR, kind=c.RunKind.PAIR, keep_plate=False)
    data = b"x" * 1000
    stage_chunks(orch, "u2", data)
    with pytest.raises(HashMismatch) as info:
        await orch.finalize_upload(run.run_id, upload_id="u2", role="pickup", sha256_browser="b" * 64, bytes=len(data), filename="a.mp4")
    assert info.value.sha256_browser == "b" * 64 and info.value.sha256_server == hashlib.sha256(data).hexdigest()
    assert not paths.quarantine_original(run.run_id, "pickup").exists()
    assert orch.load_video(run.run_id, "pickup").hash_match is False
    orch.chunk_path("u3", 0, run.run_id).write_bytes(b"first chunk only")
    with pytest.raises(MissingChunks) as missing:
        await orch.finalize_upload(run.run_id, upload_id="u3", role="return", sha256_browser="a" * 64, bytes=3 * 8 * 1024 * 1024, filename="b.mp4")
    assert missing.value.missing == [1, 2]


async def test_finalize_refuses_a_sealed_walk_and_repeats_idempotently(runs_dir: Path, th: dict) -> None:
    orch = make_orch(th)
    run = await new_run(orch, c.RunKind.PICKUP_CHECK)
    with pytest.raises(UploadRefused, match="not a walk"):
        await orch.finalize_upload(run.run_id, upload_id="x", role="return", sha256_browser="a" * 64, bytes=10, filename="r.mp4")
    first = orch.load_video(run.run_id, "pickup")
    assert first is not None
    again = await orch.finalize_upload(run.run_id, upload_id="lost-response", role="pickup", sha256_browser=first.sha256_server, bytes=first.bytes, filename="pickup.mp4")
    assert again == first  # a repeated finalize of the same bytes answers the stored record
    await orch.run_pipeline(run.run_id)
    other = b"a different file"
    stage_chunks(orch, "late", other, run.run_id)
    with pytest.raises(UploadRefused, match="already sealed"):
        await orch.finalize_upload(run.run_id, upload_id="late", role="pickup", sha256_browser=hashlib.sha256(other).hexdigest(), bytes=len(other), filename="p2.mp4")


# The pipeline ------------------------------------------------------------------------------------------------------


async def test_pipeline_runs_every_stage_in_order_with_fakes(runs_dir: Path, th: dict) -> None:
    fakes = Fakes()
    orch = make_orch(th, fakes)
    run = await new_run(orch)
    final = await orch.run_pipeline(run.run_id)
    assert final.status == c.RunStatus.DONE and final.errors == []
    assert final.clocks.sealed_at and final.clocks.first_model_call_at == "2026-10-27T15:00:00Z"
    assert final.clocks.sealed_minutes_after_camera is not None
    assert final.responses_store == c.ResponsesStore.FALSE_ACCEPTED and final.counts.marks == 0
    events = orch.read_events(run.run_id)
    seen = [e.kind.value for e in events]
    assert seen[:5] == ["run.created", "upload.started", "upload.done", "upload.started", "upload.done"]
    assert events[0].seq == 0 and events[0].prev == genesis(run.run_id)
    assert [e.seq for e in events] == list(range(len(events))) and not (runs_dir / run.run_id / PENDING_EVENTS_NAME).exists()
    assert 5 == seen.index("stage") < seen.index("seal.manifest")  # the sealing stage event waited with run.created
    assert stage_names(orch, run.run_id) == ["sealing", "sealed", "converting", "converting", "tracking", "tracking", "blurring", "marking", "staysput", "staysput", "matching", "sheet", "sheet", "sheet", "done"]
    for kind in ("seal.manifest", "seal.tsr", "seal.ots", "media.probed", "media.proxy_done", "files.uploaded", "sam.request", "sam.first_delta", "sam.frame", "sam.lane_done", "panels.derived", "coverage.updated", "blur.done", "spark.request", "spark.response", "marks.derived", "staysput.derived", "voice.done", "narration.derived", "match.derived", "sheet.written", "files.deleted", "quarantine.deleted", "ledger.row", "run.completed"):
        assert kind in seen, kind
    assert seen[-1] == "run.completed" and seen.count("sam.lane_done") == 32
    assert fakes.lane_calls == [("pickup", [paths.lane_id(n) for n in c.LANE_NOUNS]), ("return", [paths.lane_id(n) for n in c.LANE_NOUNS])]
    assert fakes.calls["derive_panels_lanes_pickup"] == 16
    frames = [e for e in events if e.kind == c.EventKind.SAM_FRAME]
    assert frames[0].payload["objects"] == [{"id": "0", "box": [1, 2, 30, 40], "c": 0.7}] and "mask" not in frames[0].payload
    assert sorted({e.payload["frame"] for e in frames}) == [0, 3]  # 15 fps proxy: one frame event every 3 frames per lane
    spark = next(e for e in events if e.kind == c.EventKind.SPARK_RESPONSE)
    assert spark.payload["sha256"] == spark_record(keyframe("pickup")).digest()
    assert [e.payload["role"] for e in events if e.kind == c.EventKind.MARKS_DERIVED] == ["pickup", "return"]
    assert [e.payload["role"] for e in events if e.kind == c.EventKind.COVERAGE_UPDATED] == ["pickup", "return"]
    sheet_seq = next(e.seq for e in events if e.kind == c.EventKind.SHEET_WRITTEN)
    assert final.events_head_hash == events[sheet_seq - 1].hash  # the head the sheet printed, frozen before it
    assert c.read_json(paths.sheet_json(run.run_id), c.SheetModel).events_head_hash == final.events_head_hash
    assert c.read_json(paths.manifest_run_json(run.run_id), c.SealManifest).events_head_hash == events[-3].hash
    assert not list(paths.quarantine_dir(run.run_id).rglob("*.orig"))
    deletions = c.read_json(paths.deletions_json(run.run_id), c.Deletions)
    assert {r.kind for r in deletions.rows} == {c.DeletionKind.FILE, c.DeletionKind.FILES_API, c.DeletionKind.QUARANTINE}
    video = orch.load_video(run.run_id, "pickup")
    assert video.files_api.deleted_at and video.files_api.delete_response == {"deleted": True} and video.quarantine_deleted_at
    assert paths.manifest_run_json(run.run_id).is_file() and paths.sheet_html(run.run_id).is_file()
    assert orch.events.verify_chain(run.run_id, manifest_sha256=genesis(run.run_id))
    assert set(c.read_json(paths.coverage_json(run.run_id), c.Coverage).per_role) == {"pickup", "return"}
    assert fakes.blur_keep_plate == [False, False]
    for stage in orch.stages_for(final):
        assert orch.stage_done(final, stage), stage.name


async def test_run_end_deletions_land_before_the_sheet_so_it_prints_their_clocks(runs_dir: Path, th: dict) -> None:
    fakes = Fakes()
    orch = make_orch(th, fakes)
    seen_at_sheet: list[bool] = []
    real = fakes.write_sheet

    def spy(run_id: str, *, th: dict) -> Any:
        seen_at_sheet.append(paths.quarantine_original(run_id, "pickup").exists())
        return real(run_id, th=th)

    orch.overrides["sheet.write_sheet"] = spy
    run = await new_run(orch)
    await orch.run_pipeline(run.run_id)
    seen = kinds(orch, run.run_id)
    assert seen.index("files.deleted") < seen.index("quarantine.deleted") < seen.index("sheet.written")
    assert seen_at_sheet == [False] and fakes.calls["write_manifest_run"] == 2  # a first one before the sheet, the final one after


async def test_keep_plate_keeps_the_original_until_the_sheet_is_written(runs_dir: Path, th: dict) -> None:
    fakes = Fakes()
    orch = make_orch(th, fakes)
    seen_at_sheet: list[bool] = []
    real = fakes.write_sheet

    def spy(run_id: str, *, th: dict) -> Any:
        seen_at_sheet.append(paths.quarantine_original(run_id, "pickup").exists())
        return real(run_id, th=th)

    orch.overrides["sheet.write_sheet"] = spy
    run = await new_run(orch, c.RunKind.PICKUP_CHECK, keep_plate=True)
    final = await orch.run_pipeline(run.run_id)
    seen = kinds(orch, run.run_id)
    assert seen.index("files.deleted") < seen.index("sheet.written") < seen.index("quarantine.deleted")
    assert seen_at_sheet == [True] and not paths.quarantine_original(run.run_id, "pickup").exists()
    assert c.read_json(paths.deletions_json(run.run_id), c.Deletions).keep_plate_chosen_by == "renter at upload"
    assert all(orch.stage_done(final, s) for s in orch.stages_for(final))


async def test_a_crash_in_the_sheet_stage_resumes_without_repeating_the_deletions(runs_dir: Path, th: dict) -> None:
    fakes = Fakes()
    orch = make_orch(th, fakes)

    def crash(run_id: str, *, th: dict) -> Any:
        raise Crash()

    orch.overrides["sheet.write_sheet"] = crash
    run = await new_run(orch, c.RunKind.PICKUP_CHECK)
    with pytest.raises(Crash):
        await orch.run_pipeline(run.run_id)
    assert fakes.calls["delete_remote"] == 1 and not paths.quarantine_original(run.run_id, "pickup").exists()
    fakes2 = Fakes()
    orch2 = make_orch(th, fakes2)
    final = await orch2.run_pipeline(run.run_id)
    assert final.status == c.RunStatus.DONE and fakes2.calls["write_sheet"] == 1
    assert fakes2.calls["delete_remote"] == 0 and fakes2.calls["write_manifest_run"] == 1  # only the final manifest
    assert orch2.stage_done(final, orch2.stages_for(final)[-1])


async def test_pre_seal_events_keep_their_timestamps_and_order(runs_dir: Path, th: dict) -> None:
    counter = itertools.count()

    def now() -> str:
        n = next(counter)
        return f"2026-10-27T14:{n // 60:02d}:{n % 60:02d}.000Z"

    orch = make_orch(th, now=now)
    run = await new_run(orch, c.RunKind.PICKUP_CHECK)
    pending = orch.pending_events(run.run_id)
    assert [e.ts for e in pending] == ["2026-10-27T14:00:01.000Z", "2026-10-27T14:00:02.000Z", "2026-10-27T14:00:04.000Z"]  # now() also stamps created_at and received_at
    await orch.run_pipeline(run.run_id)
    chained = orch.read_events(run.run_id)[:3]
    assert [(e.kind, e.ts, e.payload) for e in chained] == [(e.kind, e.ts, e.payload) for e in pending]
    assert orch.pending_events(run.run_id) == []


async def test_pickup_check_skips_the_match_stage(runs_dir: Path, th: dict) -> None:
    fakes = Fakes()
    orch = make_orch(th, fakes)
    run = await new_run(orch, c.RunKind.PICKUP_CHECK)
    final = await orch.run_pipeline(run.run_id)
    assert final.status == c.RunStatus.DONE and fakes.calls["derive_match"] == 0
    assert fakes.lane_calls == [("pickup", [paths.lane_id(n) for n in c.LANE_NOUNS])]
    assert "matching" not in stage_names(orch, run.run_id)


async def test_resume_after_a_simulated_crash_reruns_only_lanes_without_lane_done(runs_dir: Path, th: dict) -> None:
    crash_lane = paths.lane_id(c.Noun.HOOD)  # the seventh lane of the fixed order
    fakes = Fakes(crash_at=("pickup", crash_lane))
    orch = make_orch(th, fakes)
    run = await new_run(orch)
    with pytest.raises(Crash):
        await orch.run_pipeline(run.run_id)
    assert orch.load_run(run.run_id).status == c.RunStatus.TRACKING
    assert [n for n in c.LANE_NOUNS if n not in orch.missing_lanes(run, "pickup")] == list(c.LANE_NOUNS[:6])

    fakes2 = Fakes()
    orch2 = make_orch(th, fakes2)
    assert await orch2.resume_all() == [run.run_id]
    await asyncio.gather(*orch2._tasks.values())
    final = orch2.load_run(run.run_id)
    assert final.status == c.RunStatus.DONE
    assert fakes2.lane_calls == [("pickup", [paths.lane_id(n) for n in c.LANE_NOUNS[6:]]), ("return", [paths.lane_id(n) for n in c.LANE_NOUNS])]
    assert fakes2.calls["seal_run"] == 0 and fakes2.calls["make_proxy"] == 0 and fakes2.calls["upload_file"] == 0
    names = stage_names(orch2, run.run_id)
    assert "interrupted" in names and names[-1] == "done"
    assert orch2.events.verify_chain(run.run_id, manifest_sha256=genesis(run.run_id))
    assert await orch2.resume_all() == []  # a finished run stays finished


async def test_resume_after_a_crash_between_walks_redoes_only_the_missing_walk(runs_dir: Path, th: dict) -> None:
    fakes = Fakes()
    fakes.crash_derive = "return"
    orch = make_orch(th, fakes)
    run = await new_run(orch)
    with pytest.raises(Crash):
        await orch.run_pipeline(run.run_id)
    assert set(c.read_json(paths.coverage_json(run.run_id), c.Coverage).per_role) == {"pickup"}
    assert not orch.stage_done(orch.load_run(run.run_id), orch.stages_for(run)[4])  # panels: the return walk is missing

    fakes2 = Fakes()
    orch2 = make_orch(th, fakes2)
    final = await orch2.run_pipeline(run.run_id)
    assert final.status == c.RunStatus.DONE
    assert fakes2.lane_calls == [] and fakes2.derive_roles == ["return"]  # the pickup derive stands
    assert fakes2.marks_roles == ["pickup", "return"]  # downstream of a stage that ran again, every walk runs again
    assert {k.role for k in c.read_json_list(paths.keyframes_json(run.run_id), c.Keyframe)} == {"pickup", "return"}


async def test_marks_resume_keeps_the_walk_spark_already_answered(runs_dir: Path, th: dict) -> None:
    fakes = Fakes()
    orch = make_orch(th, fakes)
    run = await new_run(orch)
    real = fakes.derive_marks

    async def crash_on_return(run_id: str, role: str, keyframes: list, *, th: dict, count_mode: bool = False) -> Any:
        if role == "return":
            raise Crash()
        return await real(run_id, role, keyframes, th=th)

    orch.overrides["marks.derive_marks"] = crash_on_return
    with pytest.raises(Crash):
        await orch.run_pipeline(run.run_id)
    fakes2 = Fakes()
    orch2 = make_orch(th, fakes2)
    assert (await orch2.run_pipeline(run.run_id)).status == c.RunStatus.DONE
    assert fakes2.marks_roles == ["return"] and fakes2.derive_roles == []


async def test_failed_lanes_get_summaries_and_are_resent_while_the_proxy_exists(runs_dir: Path, th: dict) -> None:
    fakes = Fakes()
    fakes.fail_lanes = {("pickup", "wheel"), ("pickup", "hood")}
    orch = make_orch(th, fakes)
    run = await new_run(orch, c.RunKind.PICKUP_CHECK)
    blocked = {"blur": False}

    async def no_blur(run_id: str, role: str, tracks: list, *, fps: float, th: dict) -> Any:
        blocked["blur"] = True
        raise Crash()  # stop right after the lanes and panels, while the unblurred proxy still exists

    orch.overrides["privacy.render_blurred_proxy"] = no_blur
    with pytest.raises(Crash):
        await orch.run_pipeline(run.run_id)
    record = c.read_json(paths.lane_summary_path(run.run_id, "pickup", c.Noun.WHEEL), c.LaneRecord)
    assert record.status == c.LaneStatus.FAILED and record.error == "RuntimeError: boom"
    assert paths.lane_index_path(run.run_id, "pickup", c.Noun.WHEEL).is_file()
    assert c.read_json(paths.run_json(run.run_id), c.RunManifest).errors[0].text.endswith("2 lanes not run (wheel, hood)")
    assert orch.missing_lanes(orch.load_run(run.run_id), "pickup") == [c.Noun.WHEEL, c.Noun.HOOD]

    fakes2 = Fakes()
    orch2 = make_orch(th, fakes2)
    final = await orch2.run_pipeline(run.run_id)
    assert fakes2.lane_calls == [("pickup", ["wheel", "hood"])] and final.status == c.RunStatus.DONE
    assert orch2.missing_lanes(final, "pickup") == []


async def test_failed_lanes_are_not_resent_once_the_proxy_is_gone(runs_dir: Path, th: dict) -> None:
    fakes = Fakes()
    fakes.fail_lanes = {("pickup", "roof")}
    orch = make_orch(th, fakes)
    run = await new_run(orch, c.RunKind.PICKUP_CHECK)
    final = await orch.run_pipeline(run.run_id)
    assert final.status == c.RunStatus.PARTIAL and "lanes" in [e.stage for e in final.errors]
    assert orch.missing_lanes(final, "pickup") == []  # blurred: the roof lane stays not_run
    assert orch.tracked_counts(final) == {"pickup": 15}


async def test_resume_starts_a_created_run_whose_uploads_are_complete(runs_dir: Path, th: dict) -> None:
    orch = make_orch(th)
    run = await new_run(orch, c.RunKind.PICKUP_CHECK)
    waiting = await orch.create_run(mode=c.RunMode.CAR, kind=c.RunKind.PAIR, keep_plate=False)  # no uploads yet
    orch2 = make_orch(th)
    assert await orch2.resume_all() == [run.run_id]
    await asyncio.gather(*orch2._tasks.values())
    assert orch2.load_run(run.run_id).status == c.RunStatus.DONE and orch2.load_run(waiting.run_id).status == c.RunStatus.CREATED


async def test_spend_ceiling_marks_remaining_lanes_not_run_and_still_writes_the_sheet(runs_dir: Path, th: dict) -> None:
    fakes = Fakes(ledger_frames=3000)  # 16 pickup lanes x $0.60 = $9.60, over the $8 ceiling
    orch = make_orch(th, fakes)
    run = await new_run(orch)
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


async def test_lane_budget_cuts_the_lanes_that_would_pass_the_ceiling_before_any_is_sent(runs_dir: Path, th: dict) -> None:
    fakes = Fakes()
    fakes.proxy_frames = 4000  # $0.80 a lane; 800 cents less 66 kept for the image checks fits 9 lanes
    orch = make_orch(th, fakes)
    run = await new_run(orch)
    assert orch.lane_cost_cents(run, 4000) == pytest.approx(80.0) and orch.lane_cost_cents(run, 10) == pytest.approx(1.0)  # 50-frame minimum
    final = await orch.run_pipeline(run.run_id)
    sent = [paths.lane_id(n) for n in c.LANE_NOUNS[:9]]
    assert fakes.lane_calls[0] == ("pickup", sent)  # car, license plate and face are never the ones cut
    cut = c.read_json(paths.lane_summary_path(run.run_id, "pickup", c.Noun.CAR_WINDOW), c.LaneRecord)
    assert cut.status == c.LaneStatus.FAILED and cut.error == f"spend ceiling 800 cents: 7 lanes would pass it"
    assert final.status == c.RunStatus.PARTIAL and final.errors[0].stage == "lanes"


async def test_deadline_stops_model_stages_after_it_passes(runs_dir: Path, th: dict) -> None:
    clock = [0.0]
    fakes = Fakes(clock=clock, clock_step=800.0)  # the pickup lanes alone take longer than the 720 s deadline
    orch = make_orch(th, fakes, clock=lambda: clock[0])
    run = await new_run(orch)
    final = await orch.run_pipeline(run.run_id)
    assert final.status == c.RunStatus.PARTIAL
    assert fakes.lane_calls == [("pickup", [paths.lane_id(n) for n in c.LANE_NOUNS])]
    record = c.read_json(paths.lane_summary_path(run.run_id, "return", c.Noun.CAR), c.LaneRecord)
    assert record.status == c.LaneStatus.FAILED and "deadline" in (record.error or "")
    assert any("deadline" in e.text for e in final.errors if e.stage == "marks")
    # the pickup walk is blurred from its own plate and face lanes; the return walk's never ran, so it gets no display video
    assert fakes.calls["write_sheet"] == 1 and fakes.calls["render_blurred_proxy"] == 1
    assert orch.load_video(run.run_id, "pickup").proxy_blurred is not None and orch.load_video(run.run_id, "return").proxy_blurred is None
    blur = [e.text for e in final.errors if e.stage == "blur"]
    assert blur == ["return walk: no display video, because its license plate lane is failed; the unblurred proxy is deleted at run end"]
    assert not paths.proxy_path(run.run_id, "return").exists()
    rows = c.read_json(paths.deletions_json(run.run_id), c.Deletions).rows
    assert any(r.target == "media/return.proxy.mp4" and r.kind == c.DeletionKind.FILE and "no blurred display proxy" in r.reason for r in rows)


async def test_deadline_cuts_lanes_still_running_and_keeps_the_ones_that_finished(runs_dir: Path, th: dict) -> None:
    quick = json.loads(json.dumps(th))
    quick["orchestrator"]["run_deadline_s"] = 1
    fakes = Fakes()
    orch = make_orch(quick, fakes)
    real = fakes.segment_video_many

    async def stalls_after_three(file_id: str, prompts: list, **kwargs: Any) -> Any:
        await real(file_id, prompts[:3], **kwargs)
        await asyncio.sleep(30)

    orch.overrides["sam_client.segment_video_many"] = stalls_after_three
    run = await new_run(orch, c.RunKind.PICKUP_CHECK)
    t0 = time.monotonic()
    final = await orch.run_pipeline(run.run_id)
    assert time.monotonic() - t0 < 10
    statuses = orch.lane_status(run.run_id, "pickup")
    assert [statuses[n] for n in c.LANE_NOUNS[:3]] == [c.LaneStatus.COMPLETED] * 3
    assert all(statuses[n] == c.LaneStatus.FAILED for n in c.LANE_NOUNS[3:])
    done = [e.payload["noun"] for e in orch.read_events(run.run_id) if e.kind == c.EventKind.SAM_LANE_DONE]
    assert done == ["car", "license_plate", "face"]
    assert final.status == c.RunStatus.PARTIAL and "deadline" in final.errors[0].text


async def test_blur_rebuilds_plate_and_face_tracks_from_the_lanes_when_panels_is_skipped(runs_dir: Path, th: dict) -> None:
    clock = [0.0]
    fakes = Fakes(clock=clock, clock_step=10_000.0)  # the deadline passes during the lanes, so panels.derive never runs
    seen: list[list[c.PanelTrack]] = []
    real = fakes.render_blurred_proxy

    async def spy(run_id: str, role: str, tracks: list, *, fps: float, th: dict) -> Any:
        seen.append(list(tracks))
        return await real(run_id, role, tracks, fps=fps, th=th)

    orch = make_orch(th, fakes, clock=lambda: clock[0])
    orch.overrides["privacy.render_blurred_proxy"] = spy
    run = await new_run(orch, c.RunKind.PICKUP_CHECK)
    final = await orch.run_pipeline(run.run_id)
    assert fakes.calls["derive_panels"] == 0 and len(seen) == 1
    assert {t.noun for t in seen[0]} == set(c.PRIVACY_NOUNS) and all(t.role == "pickup" for t in seen[0])
    assert all(t.per_frame[0].box == [1.0, 2.0, 30.0, 40.0] for t in seen[0])  # the lane index boxes, not an empty list
    assert orch.load_video(run.run_id, "pickup").proxy_blurred is not None and "panels" in [e.stage for e in final.errors]


async def test_blur_is_refused_when_a_privacy_lane_did_not_complete(runs_dir: Path, th: dict) -> None:
    fakes = Fakes()
    fakes.fail_lanes = {("pickup", "face")}
    orch = make_orch(th, fakes)
    run = await new_run(orch, c.RunKind.PICKUP_CHECK)
    final = await orch.run_pipeline(run.run_id)
    assert fakes.calls["render_blurred_proxy"] == 0 and orch.load_video(run.run_id, "pickup").proxy_blurred is None
    assert "blur.done" not in kinds(orch, run.run_id)
    blur = [e.text for e in final.errors if e.stage == "blur"]
    assert final.status == c.RunStatus.PARTIAL and blur == ["pickup walk: no display video, because its face lane is failed; the unblurred proxy is deleted at run end"]
    assert not paths.proxy_path(run.run_id, "pickup").exists()  # deleted at run end with its row, never served meanwhile
    assert "media/pickup.proxy.mp4" in [r.target for r in c.read_json(paths.deletions_json(run.run_id), c.Deletions).rows]
    assert all(orch.stage_done(final, s) for s in orch.stages_for(final))  # the refusal settles the stage: a resume redoes nothing


async def test_a_lane_that_is_not_wired_is_a_recorded_gap_not_a_retry(runs_dir: Path, th: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    fakes = Fakes()
    overrides = {k: v for k, v in fakes.overrides().items() if not k.startswith("staysput.")}
    monkeypatch.setitem(sys.modules, "walkaround.staysput", None)  # the module exists now: hide it to test an unwired lane
    orch = Orchestrator(th=th, engine=c.SamEngine.META_MODEL_API, max_concurrent_runs=1, media_workers=1, decode_workers=1, events=FileEventLog(), overrides=overrides, delays_path=Path("/nonexistent"))
    run = await new_run(orch, c.RunKind.PICKUP_CHECK)
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
    run = await new_run(orch, c.RunKind.PICKUP_CHECK)
    final = await orch.run_pipeline(run.run_id)
    assert attempts == ["pickup", "pickup"]
    assert final.status == c.RunStatus.PARTIAL and [e.stage for e in final.errors] == ["marks"] and "spark 504" in final.errors[0].text


async def test_media_failure_is_fatal(runs_dir: Path, th: dict) -> None:
    fakes = Fakes()
    orch = make_orch(th, fakes)

    def broken(src: Path, dst: Path, **kwargs: Any) -> Any:
        raise RuntimeError("could not read this video")

    orch.overrides["video.make_proxy"] = broken
    run = await new_run(orch, c.RunKind.PICKUP_CHECK)
    final = await orch.run_pipeline(run.run_id)
    text = "could not read this video; on iPhone set Most Compatible and re-record (pickup: RuntimeError: could not read this video)"
    assert final.status == c.RunStatus.FAILED and final.errors[-1].stage == "media" and final.errors[-1].text == text
    events = orch.read_events(run.run_id)
    assert events[-1].kind == c.EventKind.RUN_FAILED and events[-1].payload == {"stage": "media", "text": text}
    assert fakes.calls["write_sheet"] == 0
    # run-end deletion still ran: no original left, deletions.json says what went
    assert not paths.quarantine_original(run.run_id, "pickup").exists()
    assert [r.target for r in c.read_json(paths.deletions_json(run.run_id), c.Deletions).rows] == ["quarantine/pickup.orig"]
    assert kinds(orch, run.run_id)[-2] == "quarantine.deleted"


async def test_a_fatal_failure_still_deletes_every_unblurred_file_and_upload(runs_dir: Path, th: dict) -> None:
    fakes = Fakes()
    orch = make_orch(th, fakes)
    real = fakes.make_proxy

    def proxy_then_fail(src: Path, dst: Path, **kwargs: Any) -> Any:
        if src.name.startswith("return"):
            raise RuntimeError("ffmpeg: could not decode")
        return real(src, dst, **kwargs)

    orch.overrides["video.make_proxy"] = proxy_then_fail
    run = await new_run(orch, c.RunKind.PAIR)
    final = await orch.run_pipeline(run.run_id)
    assert final.status == c.RunStatus.FAILED
    left = [p for p in (runs_dir / run.run_id).rglob("*") if p.is_file() and ("quarantine" in p.parts or "uploads" in p.parts or p.name.endswith(".proxy.mp4"))]
    assert left == []
    targets = sorted(r.target for r in c.read_json(paths.deletions_json(run.run_id), c.Deletions).rows)
    assert targets == ["media/pickup.proxy.mp4", "quarantine/pickup.orig", "quarantine/return.orig"]
    assert orch.load_video(run.run_id, "pickup").quarantine_deleted_at is not None


async def test_a_failure_caught_by_the_last_net_deletes_the_files_api_upload_too(runs_dir: Path, th: dict) -> None:
    fakes = Fakes()
    orch = make_orch(th, fakes)
    real = orch._note_spend

    def note_then_break(run_id: str) -> None:
        if orch.load_run(run_id).status == c.RunStatus.MARKING:
            raise RuntimeError("ledger unreadable")
        real(run_id)

    orch._note_spend = note_then_break  # type: ignore[method-assign]
    run = await new_run(orch, c.RunKind.PICKUP_CHECK)
    await orch.start(run.run_id)
    final = orch.load_run(run.run_id)
    assert final.status == c.RunStatus.FAILED and final.errors[-1] == c.RunError(stage="pipeline", text="RuntimeError: ledger unreadable")
    video = orch.load_video(run.run_id, "pickup")
    assert video.files_api.deleted_at is not None and video.quarantine_deleted_at is not None
    assert {r.kind for r in c.read_json(paths.deletions_json(run.run_id), c.Deletions).rows} == {c.DeletionKind.FILE, c.DeletionKind.FILES_API, c.DeletionKind.QUARANTINE}
    assert kinds(orch, run.run_id)[-1] == "run.failed"


async def test_seal_failure_is_fatal_and_no_model_is_called(runs_dir: Path, th: dict) -> None:
    fakes = Fakes()
    orch = make_orch(th, fakes)

    def broken(run_id: str, originals: list, **kwargs: Any) -> Any:
        raise OSError("disk full")

    orch.overrides["seal.seal_run"] = broken
    run = await new_run(orch, c.RunKind.PICKUP_CHECK)
    final = await orch.run_pipeline(run.run_id)
    assert final.status == c.RunStatus.FAILED and final.errors[-1].stage == "seal"
    assert fakes.calls["upload_file"] == 0 and fakes.lane_calls == []
    assert kinds(orch, run.run_id) == [] and pending_kinds(orch, run.run_id)[-1] == "run.failed"  # no genesis, no chain


async def test_stages_compute_with_the_frozen_thresholds(runs_dir: Path, th: dict) -> None:
    seen: list[dict] = []
    fakes = Fakes()
    orch = make_orch(th, fakes)
    original = fakes.write_sheet

    def spy(run_id: str, *, th: dict) -> Any:
        seen.append(th)
        return original(run_id, th=th)

    orch.overrides["sheet.write_sheet"] = spy
    run = await new_run(orch, c.RunKind.PICKUP_CHECK)
    await orch.run_pipeline(run.run_id)
    assert seen == [frozen(th)] and seen[0] is not th


async def test_narration_says_no_narration_found_per_walk_and_keeps_going(runs_dir: Path, th: dict) -> None:
    fakes = Fakes()
    fakes.voice_error = {"pickup": ValueError("audio over 10 minutes")}
    orch = make_orch(th, fakes)
    run = await new_run(orch)
    final = await orch.run_pipeline(run.run_id)
    assert final.status == c.RunStatus.DONE and fakes.calls["derive_narration"] == 1
    voice = [e.payload for e in orch.read_events(run.run_id) if e.kind == c.EventKind.VOICE_DONE]
    assert voice[0] == {"role": "pickup", "mode_applied": None, "text": "no narration found: ValueError: audio over 10 minutes"}
    assert voice[1]["role"] == "return" and voice[1]["mode_applied"] is True
    assert [r.role for r in c.read_json_list(paths.voice_json(run.run_id), c.VoiceRecord)] == ["return"]


async def test_store_echo_true_switches_on_the_per_request_deletion(runs_dir: Path, th: dict) -> None:
    fakes = Fakes()
    fakes.store_echo = True
    orch = make_orch(th, fakes)
    run = await new_run(orch, c.RunKind.PICKUP_CHECK)
    final = await orch.run_pipeline(run.run_id)
    assert final.responses_store == c.ResponsesStore.DELETED_PER_REQUEST
    assert len(fakes.response_ids) == 17 and "resp_pickup_car" in fakes.response_ids and "resp_spark_pickup_hood_kf1" in fakes.response_ids
    assert sum(1 for k in kinds(orch, run.run_id) if k == "responses.deleted") == 17


# Caps and ceiling from the ledger --------------------------------------------------------------------------------------


def test_caps_remaining_counts_runs_and_dollars_per_et_day(runs_dir: Path, th: dict) -> None:
    orch = make_orch(th, now=lambda: "2026-10-27T17:30:00.000Z")
    write_run("RUNPICKUP1", created_at="2026-10-27T14:00:00Z", kind=c.RunKind.PICKUP_CHECK)
    write_run("RUNPICKUP2", created_at="2026-10-27T15:00:00Z", kind=c.RunKind.PICKUP_CHECK)
    write_run("RUNPAIR1", created_at="2026-10-27T16:00:00Z", kind=c.RunKind.PAIR)
    write_run("RUNREPLAY", created_at="2026-10-27T16:30:00Z", kind=c.RunKind.PAIR, live=False)
    write_run("RUNRESERVED", created_at="2026-10-27T17:00:00Z", kind=c.RunKind.PICKUP_CHECK, status=c.RunStatus.CREATED)  # uploads coming in
    write_run("RUNABANDONED", created_at="2026-10-27T15:10:00Z", kind=c.RunKind.PICKUP_CHECK, status=c.RunStatus.CREATED)  # reservation lapsed
    write_run("RUNYESTERDAY", created_at="2026-10-26T20:00:00Z", kind=c.RunKind.PICKUP_CHECK)
    write_ledger_row("2026-10-27T10:00:00-0400", "RUNPICKUP1", 2.6)
    write_ledger_row("2026-10-27T11:00:00-0400", "RUNPAIR1", 6.2)
    write_ledger_row("2026-10-27T23:30:00-0400", "gate_E2", 1.0)  # a D1 script row: dollars count, no run
    write_ledger_row("2026-10-26T22:00:00-0400", "RUNYESTERDAY", 2.6)
    left = orch.caps_remaining(et_day="2026-10-27")
    assert left == {"pickup": th["orchestrator"]["pickup_runs_per_day"] - 3, "pair": th["orchestrator"]["pair_runs_per_day"] - 1}
    assert orch.caps_remaining(et_day="2026-10-26") == {"pickup": th["orchestrator"]["pickup_runs_per_day"] - 1, "pair": th["orchestrator"]["pair_runs_per_day"]}
    assert orch.spend_cents("RUNPAIR1") == 620


def test_daily_card_ceiling_zeroes_both_caps(runs_dir: Path, th: dict) -> None:
    orch = make_orch(th)
    ceiling = th["orchestrator"]["daily_card_ceiling_cents"] / 100
    rows = int(ceiling // 2.6) + 1  # just past the ceiling in one ET day
    for i in range(rows):
        write_ledger_row(f"2026-10-27T{10 + i // 6:02d}:{(i % 6) * 10:02d}:00-0400", None, 2.6)
    assert orch.caps_remaining(et_day="2026-10-27") == {"pickup": 0, "pair": 0}


async def test_check_can_start_raises_on_pause_and_on_caps_and_a_new_run_reserves_its_slot(runs_dir: Path, th: dict) -> None:
    orch = make_orch(th, now=lambda: "2026-10-27T18:00:00.000Z")
    assert orch.check_can_start(c.RunKind.PAIR)["pair"] == th["orchestrator"]["pair_runs_per_day"]
    await orch.create_run(mode=c.RunMode.CAR, kind=c.RunKind.PAIR, keep_plate=False)  # created, nothing uploaded yet
    with pytest.raises(CapsExceeded) as info:
        orch.check_can_start(c.RunKind.PAIR)
    assert info.value.pairs_left == 0 and info.value.pickup_left == th["orchestrator"]["pickup_runs_per_day"]
    orch.check_can_start(c.RunKind.PICKUP_CHECK)
    orch.paused = True
    with pytest.raises(CapsExceeded, match="paused"):
        orch.check_can_start(c.RunKind.PICKUP_CHECK)


async def test_env_limits_change_caps_and_ceiling_but_not_the_config_hash(runs_dir: Path, th: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PICKUP_RUNS_PER_DAY", "3")
    monkeypatch.setenv("PAIR_RUNS_PER_DAY", "0")
    monkeypatch.setenv("RUN_SPEND_CEILING_CENTS", "500")
    monkeypatch.setenv("RUN_RETENTION_HOURS", "48")
    monkeypatch.setenv("MAX_CONCURRENT_RUNS", "1")
    orch = Orchestrator.from_env(th, events=FileEventLog(), overrides=Fakes().overrides(), now=lambda: "2026-10-27T14:00:00.000Z")
    assert orch.max_concurrent_runs == 1 and orch.engine == c.SamEngine.META_MODEL_API
    assert orch.caps_remaining(et_day="2026-10-27") == {"pickup": 3, "pair": 0}
    run = await orch.create_run(mode=c.RunMode.CAR, kind=c.RunKind.PICKUP_CHECK, keep_plate=False)
    assert run.spend_ceiling_cents == 500 and run.retention_deadline == "2026-10-29T14:00:00Z"
    assert run.config_hash == c.config_hash(th)


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


async def test_health_probe_pauses_at_once_without_a_key(runs_dir: Path, th: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    import walkaround.config as config_module

    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.setattr(config_module, "load_env", lambda: None)  # no .env and no ~/meta2026/.keys, like a fresh clone
    orch = make_orch(th)

    async def no_call(url: str, headers: dict[str, str]) -> int:
        raise AssertionError("no request without a key")

    orch.http_get = no_call
    assert await orch.health_probe() is False and orch.paused is True and orch.probe_ok is False
    with pytest.raises(CapsExceeded, match="paused"):
        orch.check_can_start(c.RunKind.PAIR)


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
    now = [0.0]
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)
        now[0] += seconds

    orch = make_orch(th, delays_path=delays, clock=lambda: now[0], sleep=fake_sleep)
    events = seed_replay_run(orch, "RUNREPLAY1", STAMPS)
    waits = orch.replay_delays(events, mode=c.ReplayMode.FIXED)
    assert [round(w * 1000) for w in waits] == [100, 100, 100, 100, 100, 100, 100, 150, 150]
    replayed = [e async for e in orch.replay_events("RUNREPLAY1", mode=c.ReplayMode.FIXED)]
    assert replayed == events and [round(s, 6) for s in slept] == [round(w, 6) for w in waits]
    payload = orch.replay_payload("RUNREPLAY1", mode=c.ReplayMode.FIXED)
    assert payload.recorded_on == "2026-10-27" and payload.delays_ms == {"created": 100, "sealing": 200, "tracking": 400, "done": 300}
    assert payload.events == events and payload.mode == c.ReplayMode.FIXED


async def test_fixed_replay_wall_clock_matches_the_stage_totals_within_20_ms(runs_dir: Path, th: dict, tmp_path: Path) -> None:
    delays = tmp_path / "delays.json"
    delays.write_text(json.dumps({"created": 10, "sealing": 20, "tracking": 40, "done": 30}), encoding="utf-8")
    orch = make_orch(th, delays_path=delays)
    seed_replay_run(orch, "RUNREPLAY2", STAMPS)
    t0 = time.perf_counter()
    arrivals = [(time.perf_counter() - t0) * 1000 async for _ in orch.replay_events("RUNREPLAY2", mode=c.ReplayMode.FIXED)]
    due = list(itertools.accumulate(w * 1000 for w in orch.replay_delays(orch.read_events("RUNREPLAY2"), mode=c.ReplayMode.FIXED)))
    assert len(arrivals) == 9 and due[-1] == pytest.approx(100.0)
    assert all(0 <= a - d < 20 for a, d in zip(arrivals, due)), list(zip(arrivals, due))


async def test_recorded_replay_uses_capped_ts_deltas_and_instant_uses_none(runs_dir: Path, th: dict) -> None:
    orch = make_orch(th)
    events = seed_replay_run(orch, "RUNREPLAY3", STAMPS)
    waits = orch.replay_delays(events, mode=c.ReplayMode.RECORDED)
    assert [round(w, 3) for w in waits] == [0.0, 0.5, 0.5, 2.0, 0.2, 0.2, 0.2, 2.0, 0.1]  # the 5 s and 3 s gaps cap at 2 s
    assert orch.replay_delays(events, mode=c.ReplayMode.INSTANT) == [0.0] * 9
    assert orch.replay_payload("RUNREPLAY3", mode=c.ReplayMode.RECORDED).delays_ms == {}


def test_recorded_on_is_the_et_day(runs_dir: Path, th: dict) -> None:
    orch = make_orch(th)
    write_run("RUNLATE", created_at="2026-10-29T01:30:00Z", kind=c.RunKind.PAIR, status=c.RunStatus.DONE)  # 9:30 PM ET on 10/28
    assert orch.recorded_on(orch.load_run("RUNLATE")) == "2026-10-28"


def test_the_checked_in_delays_json_names_every_stage_the_pipeline_emits(th: dict) -> None:
    delays = Orchestrator(th=th, engine=c.SamEngine.META_MODEL_API, max_concurrent_runs=1, media_workers=1, decode_workers=1, events=FileEventLog()).load_delays()
    emitted = {"created", c.RunStatus.SEALED.value, c.RunStatus.DONE.value} | {s.status.value for s in STAGES}
    assert emitted <= set(delays), sorted(emitted - set(delays))
    assert all(isinstance(v, int) and v >= 0 for v in delays.values())


# Rebuild and verify --------------------------------------------------------------------------------------------------


async def test_rebuild_into_out_dir_and_verify_reports_a_tampered_file(runs_dir: Path, th: dict, tmp_path: Path) -> None:
    fakes = Fakes()
    orch = make_orch(th, fakes)
    run = await new_run(orch)
    await orch.run_pipeline(run.run_id)
    out = tmp_path / "out"
    result = orch.rebuild(run.run_id, out=out)
    assert (out / run.run_id / "derived" / "matches.json").is_file() and not (out / run.run_id / "quarantine").exists()
    assert (out / "ledger.jsonl").is_file()
    assert result["matches.json"] == hashlib.sha256(paths.matches_json(run.run_id).read_bytes()).hexdigest()
    assert result["coverage.json"] == hashlib.sha256(paths.coverage_json(run.run_id).read_bytes()).hexdigest()
    assert result["sheet.json"] == hashlib.sha256(paths.sheet_json(run.run_id).read_bytes()).hexdigest()  # the frozen head
    assert result["marks.json"] == hashlib.sha256(paths.marks_json(run.run_id).read_bytes()).hexdigest()  # from spark.json
    assert result["mark_tracks.json"] == hashlib.sha256(paths.mark_tracks_json(run.run_id).read_bytes()).hexdigest()  # staysput.rebuild_staysput
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


async def test_the_match_stage_asks_the_second_look_then_sizes_and_rebuild_replays_both(runs_dir: Path, th: dict, tmp_path: Path) -> None:
    fakes = Fakes()
    given: list[Any] = []
    seen: dict[str, list[str]] = {"live": [], "replay": [], "sizes": [], "claims": []}

    def live_look(pair: Any) -> Any:
        return c.LookVerdict.SAME

    def replay_look(pair: Any) -> Any:
        return None

    def derive_match(run_id: str, *, th: dict, second_look: Any = None) -> Any:
        given.append(second_look)
        return fakes.derive_match(run_id, th=th, second_look=second_look)

    def rebuild_claims(run_id: str, *, th: dict) -> dict[str, str]:
        seen["claims"].append(run_id)
        return {"claim/claim_1.json": "0" * 64}

    overrides = {
        **fakes.overrides(),
        "match.derive_match": derive_match,
        "second_look.make_second_look": lambda run_id, *, th: seen["live"].append(run_id) or live_look,
        "second_look.replay_second_look": lambda run_id, *, th=None: seen["replay"].append(run_id) or replay_look,
        "second_look.stored_looks": lambda run_id: [],
        "ruler.size_verdicts": lambda run_id, *, th: seen["sizes"].append(run_id) or [],
        "claim.rebuild_claims": rebuild_claims,
    }
    orch = Orchestrator(th=th, engine=c.SamEngine.META_MODEL_API, max_concurrent_runs=1, media_workers=1, decode_workers=1, events=FileEventLog(), overrides=overrides, delays_path=Path("/nonexistent"))
    run = await new_run(orch)
    assert (await orch.run_pipeline(run.run_id)).status == c.RunStatus.DONE
    assert given == [live_look] and seen["live"] == [run.run_id] and seen["sizes"] == [run.run_id]
    result = orch.rebuild(run.run_id, out=tmp_path / "out")
    assert given == [live_look, replay_look] and seen["replay"] == [run.run_id] and seen["sizes"] == [run.run_id] * 2
    assert seen["claims"] == [] and not any(k.startswith("claim") for k in result)  # no claim on the run: nothing to rebuild
    answers = paths.runs_root() / run.run_id / "claim" / "claim_1.answers.json"
    answers.parent.mkdir(parents=True, exist_ok=True)
    answers.write_text("{}", encoding="utf-8")
    result = orch.rebuild(run.run_id, out=tmp_path / "out")
    assert seen["claims"] == [run.run_id] and result["claim/claim_1.json"] == "0" * 64


async def test_no_second_look_is_asked_past_the_deadline_but_the_free_match_still_runs(runs_dir: Path, th: dict) -> None:
    fakes = Fakes()
    given: list[Any] = []
    asked: list[str] = []

    def derive_match(run_id: str, *, th: dict, second_look: Any = None) -> Any:
        given.append(second_look)
        return fakes.derive_match(run_id, th=th, second_look=second_look)

    overrides = {**fakes.overrides(), "match.derive_match": derive_match, "second_look.make_second_look": lambda run_id, *, th: asked.append(run_id) or (lambda pair: c.LookVerdict.SAME)}
    orch = Orchestrator(th=th, engine=c.SamEngine.META_MODEL_API, max_concurrent_runs=1, media_workers=1, decode_workers=1, events=FileEventLog(), overrides=overrides, delays_path=Path("/nonexistent"))
    run = await new_run(orch)
    orch.past_deadline = lambda run_id: True  # type: ignore[method-assign]
    await orch.run_pipeline(run.run_id)
    assert given == [None] and asked == []


def _snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    return {str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns) for p in sorted(root.rglob("*")) if p.is_file()}


async def test_rebuild_never_touches_the_frozen_model_era_files(runs_dir: Path, th: dict, tmp_path: Path) -> None:
    fakes = Fakes()
    orch = make_orch(th, fakes)
    run = await new_run(orch)
    final = await orch.run_pipeline(run.run_id)
    assert final.status == c.RunStatus.DONE
    c.write_json(paths.spark_json(run.run_id), [])
    c.write_json(paths.voice_json(run.run_id), [])
    frozen_files = [paths.events_path(run.run_id), paths.spark_json(run.run_id), paths.voice_json(run.run_id), paths.tracks_json(run.run_id), paths.keyframes_json(run.run_id)]
    before = [(p.read_bytes(), p.stat().st_mtime_ns) for p in frozen_files]
    orch.rebuild(run.run_id, out=tmp_path / "out")
    assert [(p.read_bytes(), p.stat().st_mtime_ns) for p in frozen_files] == before
    assert fakes.calls["segment_video_many"] == 2 and fakes.calls["derive_marks"] == 2 and fakes.calls["derive_match"] == 2


async def test_a_closed_run_is_never_rebuilt_in_place(runs_dir: Path, th: dict) -> None:
    # its manifest_run.json seals the sheet and the PDF, and a PDF never renders to the same bytes twice
    orch = make_orch(th)
    run = await new_run(orch, c.RunKind.PICKUP_CHECK)
    await orch.run_pipeline(run.run_id)
    before = _snapshot(runs_dir / run.run_id)
    with pytest.raises(RebuildRefused, match="closed"):
        orch.rebuild(run.run_id)
    assert _snapshot(runs_dir / run.run_id) == before


async def test_rebuild_refuses_an_out_dir_that_would_replace_the_run_or_the_ledger(runs_dir: Path, th: dict) -> None:
    orch = make_orch(th)
    run = await new_run(orch, c.RunKind.PICKUP_CHECK)
    await orch.run_pipeline(run.run_id)
    ledger.record("sam_video", "sam-3.1", note="another run", run_id="OTHER", frames=100)
    before, ledger_before = _snapshot(runs_dir / run.run_id), ledger.LEDGER_PATH.read_bytes()
    for out in (runs_dir, runs_dir / run.run_id / "derived"):
        with pytest.raises(RebuildRefused):
            orch.rebuild(run.run_id, out=out)
    assert _snapshot(runs_dir / run.run_id) == before and ledger.LEDGER_PATH.read_bytes() == ledger_before


async def test_rebuild_never_copies_an_unblurred_file(runs_dir: Path, th: dict, tmp_path: Path) -> None:
    orch = make_orch(th)
    run = await new_run(orch, c.RunKind.PICKUP_CHECK)
    await orch.run_pipeline(run.run_id)
    paths.proxy_path(run.run_id, "pickup").write_bytes(b"an unblurred proxy left behind")
    paths.quarantine_original(run.run_id, "pickup").write_bytes(b"an original left behind")
    orch.rebuild(run.run_id, out=tmp_path / "out")
    copied = [p.name for p in (tmp_path / "out" / run.run_id).rglob("*") if p.is_file()]
    assert "pickup.proxy.mp4" not in copied and "pickup.orig" not in copied and "pickup.proxy_blurred.mp4" in copied


async def test_verify_reports_a_deleted_event_log_and_cleans_its_scratch(runs_dir: Path, th: dict) -> None:
    orch = make_orch(th)
    run = await new_run(orch, c.RunKind.PICKUP_CHECK)
    await orch.run_pipeline(run.run_id)
    assert orch.verify(run.run_id) == []
    assert not (runs_dir / "_verify" / run.run_id).exists()  # copies of the run's artifacts never outlive the check
    paths.events_path(run.run_id).unlink()
    assert make_orch(th).verify(run.run_id) == ["event log missing: the run is sealed, so events.ndjson must start at seq 0"]


async def test_verify_reports_a_sheet_head_the_event_log_does_not_hold(runs_dir: Path, th: dict) -> None:
    orch = make_orch(th)
    run = await new_run(orch, c.RunKind.PICKUP_CHECK)
    await orch.run_pipeline(run.run_id)
    lines = paths.events_path(run.run_id).read_text(encoding="utf-8").splitlines()
    head = orch.load_run(run.run_id).events_head_hash
    kept = [line for line in lines if json.loads(line)["hash"] != head]
    paths.events_path(run.run_id).write_text("\n".join(kept) + "\n", encoding="utf-8")
    problems = make_orch(th).verify(run.run_id)
    assert "events head hash printed on the sheet is not in the event log" in problems and "event chain broken" in problems


# Top-up ---------------------------------------------------------------------------------------------------------------


async def test_run_topup_reruns_only_nouns_to_rerun_and_measures_the_time(runs_dir: Path, th: dict) -> None:
    fakes = Fakes()
    fakes.missing_panels = [c.PanelId.LEFT_REAR_WHEEL]
    clock = [0.0]
    orch = make_orch(th, fakes, clock=lambda: clock[0])
    run = await new_run(orch, c.RunKind.PICKUP_CHECK)
    await orch.run_pipeline(run.run_id)
    assert "topup.requested" in kinds(orch, run.run_id)
    coverage = c.read_json(paths.coverage_json(run.run_id), c.Coverage)
    coverage.nouns_to_rerun = [c.Noun.WHEEL, c.Noun.CAR, c.Noun.LICENSE_PLATE, c.Noun.FACE]
    c.write_json(paths.coverage_json(run.run_id), coverage)
    role = orch.reserve_role(run.run_id)
    assert role == "topup_1" and orch.reserve_role(run.run_id) == "topup_2"  # two top-ups at once never share a role
    orch.release_role(run.run_id, "topup_2")
    data = b"topup clip " * 500
    digest = stage_chunks(orch, "t1", data)
    await orch.finalize_upload(run.run_id, upload_id="t1", role=role, sha256_browser=digest, bytes=len(data), filename="topup.mp4")
    orch.release_role(run.run_id, role)
    fakes.clock_step = 41.2
    fakes.clock = clock
    writes_before = fakes.calls["write_sheet"]
    result = await orch.run_topup(run.run_id, role)
    assert fakes.lane_calls[-1] == ("topup_1", ["wheel", "car", "license_plate", "face"])
    assert result.rerun_seconds == 41.2 and set(result.per_role) == {"pickup", "topup_1"}
    assert c.read_json(paths.coverage_json(run.run_id), c.Coverage).rerun_seconds == 41.2
    video = orch.load_video(run.run_id, "topup_1")
    assert video.proxy_blurred is not None and video.wav is None  # its own display proxy; no narration on a top-up
    assert video.files_api.deleted_at and video.quarantine_deleted_at  # the run had ended: deleted at once
    assert not paths.quarantine_original(run.run_id, "topup_1").exists()
    assert fakes.calls["write_sheet"] == writes_before + 1 and fakes.calls["write_manifest_run"] == 4  # release and close, then again around the new sheet
    assert kinds(orch, run.run_id)[-5:] == ["coverage.updated", "blur.done", "files.deleted", "quarantine.deleted", "sheet.written"]
    assert c.read_json(paths.manifest_run_json(run.run_id), c.SealManifest).events_head_hash == orch.events.head_hash(run.run_id)


async def test_finalize_takes_a_topup_role_only_through_the_routes_reservation(runs_dir: Path, th: dict) -> None:
    orch = make_orch(th)
    run = await new_run(orch, c.RunKind.PICKUP_CHECK)
    data = b"a stranger's clip " * 100
    digest = stage_chunks(orch, "t9", data)
    with pytest.raises(UploadRefused, match="topup"):
        await orch.finalize_upload(run.run_id, upload_id="t9", role="topup_1", sha256_browser=digest, bytes=len(data), filename="x.mp4")
    assert not paths.quarantine_original(run.run_id, "topup_1").exists() and orch.load_video(run.run_id, "topup_1") is None


async def test_a_failed_topup_after_run_end_deletes_its_original_and_changes_no_sheet(runs_dir: Path, th: dict) -> None:
    fakes = Fakes()
    orch = make_orch(th, fakes)
    run = await new_run(orch, c.RunKind.PICKUP_CHECK)
    await orch.run_pipeline(run.run_id)
    role = orch.reserve_role(run.run_id)
    data = b"a clip ffmpeg cannot cut " * 200
    digest = stage_chunks(orch, "t2", data)
    await orch.finalize_upload(run.run_id, upload_id="t2", role=role, sha256_browser=digest, bytes=len(data), filename="topup.mov")
    orch.release_role(run.run_id, role)

    def broken(src: Path, dst: Path, **kwargs: Any) -> Any:
        raise RuntimeError("moov atom not found")

    orch.overrides["video.make_proxy"] = broken
    with pytest.raises(VideoUnreadable, match="could not read this video"):
        await orch.run_topup(run.run_id, role)
    assert not paths.quarantine_original(run.run_id, role).exists()
    assert f"quarantine/{role}.orig" in [r.target for r in c.read_json(paths.deletions_json(run.run_id), c.Deletions).rows]
    assert fakes.calls["write_sheet"] == 1


async def test_a_lapsed_reservation_is_admitted_again_at_start(runs_dir: Path, th: dict) -> None:
    now = ["2026-10-31T13:00:00.000Z"]  # 9 AM ET
    orch = make_orch(th, now=lambda: now[0])
    cap = th["orchestrator"]["pickup_runs_per_day"]
    early = [await orch.create_run(mode=c.RunMode.CAR, kind=c.RunKind.PICKUP_CHECK, keep_plate=False) for _ in range(cap)]
    with pytest.raises(CapsExceeded):
        orch.check_can_start(c.RunKind.PICKUP_CHECK)
    now[0] = "2026-10-31T14:01:00.000Z"  # an hour on with nothing uploaded: those reservations lapsed
    late = [await orch.create_run(mode=c.RunMode.CAR, kind=c.RunKind.PICKUP_CHECK, keep_plate=False) for _ in range(cap)]
    admitted = []
    for run in early + late:
        try:
            orch.admit(orch.load_run(run.run_id))
        except CapsExceeded:
            continue
        orch.update_run(run.run_id, lambda r: setattr(r, "status", c.RunStatus.SEALING))  # what start() leads to
        admitted.append(run.run_id)
    assert admitted == [r.run_id for r in late]  # the day still starts only `cap` runs
    orch.paused = True
    with pytest.raises(CapsExceeded, match="paused"):
        orch.admit(orch.load_run(early[0].run_id))


async def test_run_end_hashing_and_stamping_run_off_the_event_loop(runs_dir: Path, th: dict) -> None:
    import threading

    fakes = Fakes()
    orch = make_orch(th, fakes)
    on_loop: list[tuple[str, bool]] = []

    def watch(name: str, fn: Any) -> Any:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            on_loop.append((name, threading.current_thread() is threading.main_thread()))
            return fn(*args, **kwargs)

        return wrapped

    orch.overrides["seal.write_manifest_run"] = watch("manifest_run", fakes.write_manifest_run)
    orch.overrides["privacy.delete_quarantine"] = watch("quarantine", fakes.delete_quarantine)
    run = await new_run(orch, c.RunKind.PICKUP_CHECK)
    await orch.run_pipeline(run.run_id)
    assert {name for name, _ in on_loop} == {"manifest_run", "quarantine"} and not any(main for _, main in on_loop)


# Sweep and samples -----------------------------------------------------------------------------------------------------


async def test_sweep_removes_idle_upload_chunks_and_keeps_live_ones(runs_dir: Path, th: dict) -> None:
    fakes = Fakes()
    orch = make_orch(th, fakes)
    run = await orch.create_run(mode=c.RunMode.CAR, kind=c.RunKind.PAIR, keep_plate=False)
    idle = orch.chunk_path("idle", 0, run.run_id)
    idle.write_bytes(b"abandoned bytes")
    staged = orch.chunk_path("staged", 0)
    staged.write_bytes(b"staged bytes")
    fresh = orch.chunk_path("fresh", 0, run.run_id)
    fresh.write_bytes(b"arriving now")
    old = time.time() - 31 * 60
    for path in (idle, staged):
        os.utime(path, (old, old))
    rows = await orch.sweep()
    assert sorted(r.target for r in rows) == ["_uploads/staged/chunks/000000", "uploads/idle/chunks/000000"]
    assert not idle.exists() and not staged.exists() and fresh.exists()
    assert [r.target for r in c.read_json(paths.deletions_json(run.run_id), c.Deletions).rows] == ["uploads/idle/chunks/000000"]
    assert fakes.sweep_args == {"retention_hours": 24, "temp_minutes": 30, "keep": set()}


def test_seed_samples_copies_the_demo_run_once(runs_dir: Path, th: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import walkaround.orchestrator as orchestrator_module

    repo = tmp_path / "repo"
    sample = repo / "samples" / "RUNSAMPLE"
    sample.mkdir(parents=True)
    c.write_json(sample / "run.json", c.RunManifest(run_id="RUNSAMPLE", created_at="2026-10-28T14:00:00Z", keep=True, live_or_replay=c.LiveOrReplay.REPLAY))
    monkeypatch.setattr(orchestrator_module, "REPO_DIR", repo)
    orch = make_orch(th)
    assert orch.seed_samples() == ["RUNSAMPLE"] and orch.seed_samples() == []
    assert orch.run_exists("RUNSAMPLE") and orch.sample_run_id() == "RUNSAMPLE" and orch.is_public(orch.load_run("RUNSAMPLE"))


def test_seed_samples_unpacks_a_gzipped_event_log(runs_dir: Path, th: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import gzip

    import walkaround.orchestrator as orchestrator_module

    repo = tmp_path / "repo"
    sample = repo / "samples" / "RUNSAMPLE"
    sample.mkdir(parents=True)
    c.write_json(sample / "run.json", c.RunManifest(run_id="RUNSAMPLE", created_at="2026-10-28T14:00:00Z", keep=True, live_or_replay=c.LiveOrReplay.REPLAY))
    log = b'{"seq": 0}\n{"seq": 1}\n'
    (sample / "events.ndjson.gz").write_bytes(gzip.compress(log))
    monkeypatch.setattr(orchestrator_module, "REPO_DIR", repo)
    assert make_orch(th).seed_samples() == ["RUNSAMPLE"]
    assert (runs_dir / "RUNSAMPLE" / "events.ndjson").read_bytes() == log
    assert not (runs_dir / "RUNSAMPLE" / "events.ndjson.gz").exists()


def test_error_text_and_event_payloads_never_carry_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from walkaround.orchestrator import _error_text, _scrubbed

    monkeypatch.setenv("MODEL_API_KEY", "LLM_1788601_notarealsecretvalue")
    text = _error_text(RuntimeError("401: bad key LLM_1788601_notarealsecretvalue (Authorization: Bearer abc.def)"))
    assert "notarealsecret" not in text and "abc.def" not in text and "[redacted]" in text
    payload = _scrubbed({"error": ["sent LLM_42_shapeonlysecret"], "n": 3, "ok": None})
    assert payload == {"error": ["sent [redacted]"], "n": 3, "ok": None}


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
    assert cli.main(["run", str(pickup), str(ret)]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    final = c.RunManifest.model_validate_json(lines[-1])
    assert final.status == c.RunStatus.DONE and final.kind == c.RunKind.PAIR
    printed = [c.Event.model_validate_json(line) for line in lines[:-1]]
    assert printed[0].kind == c.EventKind.RUN_CREATED and printed[-1].kind == c.EventKind.RUN_COMPLETED  # nothing cut off
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
        run = await new_run(orch)
        await orch.run_pipeline(run.run_id)
        return run.run_id

    run_id = asyncio.run(build())
    monkeypatch.setattr(cli, "make_orchestrator", lambda th=None: orch)
    assert cli.main(["verify", run_id]) == 0
    assert capsys.readouterr().out.strip() == "rebuild identical"
    before = _snapshot(runs_dir / run_id)
    assert cli.main(["rebuild", run_id]) == 0  # closed: rebuilt into runs/_rebuild, the sealed files stay
    captured = capsys.readouterr()
    assert "matches.json: " in captured.out and "sheet.json: " in captured.out and "coverage.json: " in captured.out
    assert "rebuilding into" in captured.err and (runs_dir / "_rebuild" / run_id / "derived" / "matches.json").is_file()
    assert _snapshot(runs_dir / run_id) == before
    assert cli.main(["rebuild", run_id, "--out", str(runs_dir)]) == 2
    assert "would overwrite" in capsys.readouterr().err
