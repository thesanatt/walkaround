"""The sample pair in samples/ (scripts/make_sample.py): only files the app serves to anyone, under the repo's 60 MB
budget, a keep run marked as a replay, and an event log that unpacks into a chain starting at its own manifest. A
clean clone serves this folder with no key, so a file the allowlist refuses here would be published."""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

from walkaround import __main__ as cli
from walkaround import config, ledger, paths
from walkaround import contracts as c

REPO = Path(__file__).resolve().parent.parent
SAMPLES = REPO / "samples"
FOLDERS = sorted(p for p in SAMPLES.iterdir() if p.is_dir() and not p.name.startswith(".")) if SAMPLES.is_dir() else []


def make_sample():
    spec = importlib.util.spec_from_file_location("make_sample", REPO / "scripts" / "make_sample.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_sample"] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not FOLDERS, reason="no sample folder in this checkout")
@pytest.mark.parametrize("folder", FOLDERS, ids=lambda p: p.name)
def test_every_sample_file_is_public_and_the_folder_fits(folder: Path) -> None:
    assert make_sample().problems(folder) == []


@pytest.mark.skipif(not FOLDERS, reason="no sample folder in this checkout")
@pytest.mark.parametrize("folder", FOLDERS, ids=lambda p: p.name)
def test_sample_run_is_a_finished_public_replay_with_a_credit(folder: Path) -> None:
    run = c.RunManifest.model_validate_json((folder / "run.json").read_text(encoding="utf-8"))
    assert run.keep and not run.keep_plate and run.live_or_replay == c.LiveOrReplay.REPLAY
    assert run.status in (c.RunStatus.DONE, c.RunStatus.PARTIAL) and run.footage_credit


@pytest.mark.skipif(not FOLDERS, reason="no sample folder in this checkout")
@pytest.mark.parametrize("folder", FOLDERS, ids=lambda p: p.name)
def test_sample_event_log_is_an_unbroken_chain_from_its_manifest(folder: Path) -> None:
    prev = hashlib.sha256((folder / "bundle" / "manifest.json").read_bytes()).hexdigest()
    lines = gzip.decompress((folder / "events.ndjson.gz").read_bytes()).decode("utf-8").splitlines()
    assert lines
    for i, line in enumerate(lines):
        event = c.Event.model_validate_json(line)
        assert event.seq == i and event.prev == prev and event.check(), f"event {i} breaks the chain"
        prev = event.hash
    assert c.Event.model_validate_json(lines[-1]).kind == c.EventKind.RUN_COMPLETED


def test_problems_names_a_refused_file_and_a_folder_over_budget(tmp_path: Path) -> None:
    folder = tmp_path / "RUNSAMPLE"
    c.write_json(folder / "run.json", c.RunManifest(run_id="RUNSAMPLE", created_at="2026-10-28T14:00:00Z", keep=True, live_or_replay=c.LiveOrReplay.REPLAY))
    (folder / "events.ndjson.gz").write_bytes(gzip.compress(b'{"seq": 0}\n', mtime=0))
    (folder / "derived").mkdir()
    (folder / "derived" / "coverage.json").write_text("{}\n")
    module = make_sample()
    assert module.problems(folder) == []
    (folder / "spark.json").write_text("{}\n")
    (folder / "media").mkdir()
    (folder / "media" / "pickup.proxy.mp4").write_bytes(b"unblurred")
    (folder / "quarantine").mkdir()
    (folder / "quarantine" / "pickup.orig").write_bytes(b"original")
    found = module.problems(folder, max_mb=0.000001)
    assert any("spark.json" in line for line in found)
    assert any("media/pickup.proxy.mp4" in line for line in found)
    assert any("quarantine/pickup.orig" in line for line in found)
    assert any("over" in line for line in found)


def test_dropped_keeps_named_clips_and_leaves_out_what_no_page_reads() -> None:
    module = make_sample()
    clips = {"clips/pickup/a.mp4"}
    assert module.dropped("clips/pickup/a.mp4", clips) is None
    assert module.dropped("clips/pickup/b.mp4", clips) == "Stays Put clip not named"
    assert module.dropped("sam/pickup/sp_pickup_hood_kf1_sp01.ndjson.gz", clips) == "Stays Put lane"
    assert module.dropped("sam/pickup/hood.index.json", clips) == "lane index"
    assert module.dropped("keyframes/pickup/x.frame.jpg", clips) == "full keyframe"
    assert module.dropped("sam/pickup/hood.ndjson.gz", clips) is None


def test_verify_and_rebuild_say_a_sample_copy_cannot_be_rebuilt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "runs"
    monkeypatch.setattr(paths, "RUNS_DIR", root)
    monkeypatch.setattr(config, "RUNS_DIR", root)
    monkeypatch.setattr(ledger, "LEDGER_PATH", root / "ledger.jsonl")
    c.write_json(root / "RUNSAMPLE" / "run.json", c.RunManifest(run_id="RUNSAMPLE", created_at="2026-10-28T14:00:00Z", keep=True, live_or_replay=c.LiveOrReplay.REPLAY, status=c.RunStatus.DONE))
    assert cli.main(["verify", "RUNSAMPLE"]) == 2
    assert "replay copy" in capsys.readouterr().err
    assert cli.main(["rebuild", "RUNSAMPLE"]) == 2
    assert not (root / "_rebuild").exists()
