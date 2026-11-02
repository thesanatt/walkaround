"""The samples in samples/ (scripts/make_sample.py): only files the app serves to anyone, under the repo's 60 MB
budget per sample, a keep run marked as a replay, and an event log that unpacks into a chain starting at its own
manifest; samples/index.json names only those folders or empty slots. A clean clone serves these folders with no key,
so a file the allowlist refuses here would be published."""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

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
    kinds = [c.Event.model_validate_json(line).kind for line in lines]
    # the run ends with run.completed; after it only the hourly ots upgrade may append (seal.ots, confirmed)
    done = kinds.index(c.EventKind.RUN_COMPLETED)
    assert all(k == c.EventKind.SEAL_OTS for k in kinds[done + 1 :])


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


@pytest.mark.skipif(not (SAMPLES / "index.json").is_file(), reason="no samples/index.json in this checkout")
def test_index_names_every_sample_folder_once_and_nothing_else() -> None:
    module = make_sample()
    assert module.index_problems(SAMPLES) == []
    entries = json.loads((SAMPLES / "index.json").read_text(encoding="utf-8"))["samples"]
    listed = [e["run_id"] for e in entries if e.get("run_id")]
    assert sorted(listed) == [p.name for p in FOLDERS]
    assert all(e.get("title") for e in entries)


def test_index_problems_names_a_missing_folder_and_a_double_entry(tmp_path: Path) -> None:
    module = make_sample()
    c.write_json(tmp_path / "RUNA" / "run.json", c.RunManifest(run_id="RUNA", created_at="2026-10-28T14:00:00Z", keep=True))
    index = {"samples": [{"slot": "friend_car", "run_id": None, "title": "t"}, {"run_id": "RUNA", "title": "a"}]}
    (tmp_path / "index.json").write_text(json.dumps(index))
    assert module.index_problems(tmp_path) == []
    index["samples"] += [{"run_id": "RUNA", "title": "again"}, {"run_id": "RUNGONE", "title": "g"}, {"title": "neither"}]
    (tmp_path / "index.json").write_text(json.dumps(index))
    found = module.index_problems(tmp_path)
    assert any("RUNA listed twice" in line for line in found)
    assert any("RUNGONE has no folder" in line for line in found)
    assert any("neither a run_id nor a slot" in line for line in found)


def test_place_fills_a_slot_in_its_place_appends_a_new_run_and_moves_first() -> None:
    module = make_sample()
    index = {"_about": "x", "samples": [{"slot": "friend_car", "run_id": None, "title": "Friend"}, {"run_id": "RUNA", "title": "A"}]}
    assert module.place(index, "RUNB", title=None, slot=None, first=False) == 2
    assert index["samples"][2] == {"run_id": "RUNB", "title": None}
    assert module.place(index, "RUNF", title="  A friend's   car ", slot="friend_car", first=False) == 0
    assert index["samples"][0] == {"slot": "friend_car", "run_id": "RUNF", "title": "A friend's car"}
    assert module.place(index, "RUNA", title="A again", slot=None, first=False) == 1
    assert [e["run_id"] for e in index["samples"]] == ["RUNF", "RUNA", "RUNB"] and index["samples"][1]["title"] == "A again"
    assert module.place(index, "RUNB", title=None, slot=None, first=True) == 0
    assert [e["run_id"] for e in index["samples"]] == ["RUNB", "RUNF", "RUNA"]
    # a run moved into a slot leaves its old entry; a slot it leaves stays, empty
    assert module.place(index, "RUNA", title=None, slot="friend_car", first=False) == 1
    assert [e.get("run_id") for e in index["samples"]] == ["RUNB", "RUNA"]
    assert module.place(index, "RUNA", title=None, slot="other", first=False) == 2
    assert index["samples"][1] == {"slot": "friend_car", "run_id": None, "title": "A friend's car"}
    assert index["samples"][2] == {"slot": "other", "run_id": "RUNA", "title": None}  # a new slot takes only --title


def test_write_copies_a_run_as_a_kept_replay_and_fills_its_slot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "runs"
    monkeypatch.setattr(paths, "RUNS_DIR", root)
    monkeypatch.setattr(config, "RUNS_DIR", root)
    module = make_sample()
    samples = tmp_path / "samples"
    monkeypatch.setattr(module, "SAMPLES", samples)
    samples.mkdir()
    (samples / "index.json").write_text(json.dumps({"_about": "x", "samples": [{"slot": "friend_car", "run_id": None, "title": "Friend"}, {"run_id": "RUNOLD", "title": "Old"}]}))
    c.write_json(samples / "RUNOLD" / "run.json", c.RunManifest(run_id="RUNOLD", created_at="2026-10-28T14:00:00Z", keep=True))
    run_dir = root / "RUNCLAIMCOPY"
    c.write_json(run_dir / "run.json", c.RunManifest(run_id="RUNCLAIMCOPY", created_at="2026-10-28T14:00:00Z", keep=False, status=c.RunStatus.DONE))
    (run_dir / "events.ndjson").write_bytes(b'{"seq": 0}\n')
    (run_dir / "derived").mkdir()
    (run_dir / "derived" / "coverage.json").write_text("{}\n")
    (run_dir / "claim").mkdir()
    for name in ("claim_1.json", "claim_1.blurred.jpg", "claim_1.open_weights.ndjson", "claim_1.answers.json", "claim_1.jpg"):
        (run_dir / "claim" / name).write_bytes(b"x")
    before = (samples / "index.json").read_text()
    assert module.main(["RUNCLAIMCOPY", "--slot", "friend_car", "--title", "A claim copy", "--dry-run"]) == 0
    assert not (samples / "RUNCLAIMCOPY").exists() and (samples / "index.json").read_text() == before
    assert module.main(["RUNCLAIMCOPY", "--slot", "friend_car", "--title", "A claim copy"]) == 0
    copy = c.RunManifest.model_validate_json((samples / "RUNCLAIMCOPY" / "run.json").read_text(encoding="utf-8"))
    assert copy.keep and copy.live_or_replay == c.LiveOrReplay.REPLAY
    assert sorted(p.name for p in (samples / "RUNCLAIMCOPY" / "claim").iterdir()) == ["claim_1.blurred.jpg", "claim_1.json", "claim_1.open_weights.ndjson"]
    entries = json.loads((samples / "index.json").read_text())["samples"]
    assert entries == [{"slot": "friend_car", "run_id": "RUNCLAIMCOPY", "title": "A claim copy"}, {"run_id": "RUNOLD", "title": "Old"}]
    # a run with no slot is appended, and --first moves it to the top
    c.write_json(root / "RUNNEW" / "run.json", c.RunManifest(run_id="RUNNEW", created_at="2026-10-29T14:00:00Z", keep=True, status=c.RunStatus.DONE))
    (root / "RUNNEW" / "events.ndjson").write_bytes(b'{"seq": 0}\n')
    assert module.main(["RUNNEW", "--first"]) == 0
    entries = json.loads((samples / "index.json").read_text())["samples"]
    assert [e["run_id"] for e in entries] == ["RUNNEW", "RUNCLAIMCOPY", "RUNOLD"] and entries[0]["title"] is None
    assert module.index_problems(samples) == []


def test_repixelate_re_encodes_at_the_crf_asked_and_the_runs_own_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    module = make_sample()
    seen: dict[str, Any] = {}

    def fake(src: Path, dst: Path, boxes: dict, **kw: Any) -> None:
        seen.update(kw)
        Path(dst).write_bytes(b"v")

    monkeypatch.setattr(module.video, "probe", lambda p: SimpleNamespace(width=960, height=540))
    monkeypatch.setattr(module.video, "render_blurred_proxy", fake)
    th = c.load_thresholds()
    f = tmp_path / "pickup.proxy_blurred.mp4"
    f.write_bytes(b"x")
    module.repixelate(f, {}, th, block_px=None, crf=28)
    assert seen["crf"] == 28 and f.read_bytes() == b"v"
    module.repixelate(f, {}, th, block_px=None)
    assert seen["crf"] == th["media"]["proxy_crf"]
