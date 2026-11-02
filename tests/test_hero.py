"""scripts/hero_check.py (estimate, credit, default panel and clips for the hero sample), make_sample.py --samples-dir,
tools/data_hunt/intake_check.py and scripts/hero.sh's argument checks. No model call."""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[1]


def load(name: str, rel: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


hc = load("hero_check", "scripts/hero_check.py")


def write_run(root: Path, run_id: str, secs: tuple[float, float], *, blur_s: float, end_s: float) -> None:
    d = root / run_id
    (d / "videos").mkdir(parents=True)
    for role, s in zip(("pickup", "return"), secs):
        (d / "videos" / f"{role}.json").write_text(json.dumps({"ffprobe": {"duration_s": s}}))
    t0 = dt.datetime(2026, 11, 2, 10, 0, tzinfo=dt.timezone.utc)
    events = [("run.created", 0), ("blur.done", blur_s), ("run.completed", end_s), ("seal.ots", end_s + 999)]
    (d / "events.ndjson").write_text("".join(json.dumps({"kind": k, "ts": (t0 + dt.timedelta(seconds=s)).isoformat().replace("+00:00", "Z")}) + "\n" for k, s in events))


def test_ref_stats_splits_lanes_from_clips_and_stops_at_run_completed(tmp_path: Path) -> None:
    write_run(tmp_path, "ref", (30.0, 40.0), blur_s=100, end_s=300)
    rows = [
        {"run_id": "ref", "kind": "sam_video", "note": "car", "usd": 3.0},
        {"run_id": "ref", "kind": "sam_video", "note": "sp_pickup_hood_kf1_sp01", "usd": 0.5},
        {"run_id": "ref", "kind": "spark", "usd": 1.0},
        {"run_id": "other", "kind": "spark", "usd": 9.0},
    ]
    r = hc.ref_stats(tmp_path, rows, "ref")
    assert r == {"secs": 70.0, "usd": 4.5, "lanes_usd": 3.0, "wall_s": 300.0, "scaled_s": 100.0}


def test_estimate_scales_lanes_and_early_stages_by_walk_seconds() -> None:
    refs = {"a": {"secs": 60.0, "usd": 5.0, "lanes_usd": 3.0, "wall_s": 400.0, "scaled_s": 200.0}}
    e = hc.estimate(120.0, refs)
    assert e["usd"] == pytest.approx(3.0 * 2 + 2.0)
    assert e["usd_upper"] == pytest.approx(10.0)
    assert e["wall_s"] == pytest.approx(200 * 2 + 200)
    assert e["wall_upper_s"] == pytest.approx(800.0)
    short = hc.estimate(30.0, refs)  # shorter than the reference: the upper bound never sits under the central one
    assert short["usd_upper"] >= short["usd"] and short["wall_upper_s"] >= short["wall_s"]


def test_meta_spend_leaves_out_baseten_and_counts_the_et_day() -> None:
    rows = [
        {"ts": "2026-11-03T08:00:00-0500", "kind": "sam_video", "engine": "meta-model-api", "usd": 2.0},
        {"ts": "2026-11-03T04:30:00Z", "kind": "spark", "engine": "meta-model-api", "usd": 1.0},  # 11:30 PM ET on 11/2
        {"ts": "2026-11-03T09:00:00-0500", "kind": "spark_open", "engine": "baseten", "usd": 5.0},
    ]
    assert hc.meta_spend(rows, today=dt.date(2026, 11, 3)) == (3.0, 2.0)


def test_config_caps_hold_a_padded_sixty_second_pair() -> None:
    """The lead's Nov 2 decision: the hero run (60 s + 60 s at 4K) must not stop on a cap. The reference numbers are
    lexi_pair2 and synth_pair2_r9 as measured (scripts/hero_check.py on runs/ledger.jsonl)."""
    refs = {
        "lexi_pair2": {"secs": 70.61, "usd": 4.8863, "lanes_usd": 3.392, "wall_s": 367.4, "scaled_s": 171.0},
        "synth_pair2_r9": {"secs": 78.4, "usd": 5.3229, "lanes_usd": 3.881, "wall_s": 382.5, "scaled_s": 194.0},
    }
    e = hc.estimate(120.0, refs)
    o = json.loads((REPO / "config" / "thresholds.json").read_text())["orchestrator"]
    assert e["usd_upper"] * hc.SAFETY <= o["run_spend_ceiling_cents"]["value"] / 100
    assert e["wall_upper_s"] * hc.SAFETY <= o["run_deadline_s"]["value"]
    assert o["daily_card_ceiling_cents"]["value"] == 5000
    assert "hero" in o["run_spend_ceiling_cents"]["source"] and "hero" in o["run_deadline_s"]["source"]


def test_default_panel_and_focus_follow_the_run_page() -> None:
    verdicts = [
        {"panel_id": "hood", "label": "needs_your_eye", "mark_ids": ["p1", "r1"]},
        {"panel_id": "left_front_door", "label": "new_or_unseen", "mark_ids": ["r2"]},
        {"panel_id": "left_front_door", "label": "already_there", "mark_ids": ["p3", "r3"]},
    ]
    matches = [{"panel_id": "hood", "state": "comparable", "pairs": []}]
    assert hc.default_panel(verdicts, matches, {}, []) == "left_front_door"
    assert hc.default_panel(verdicts[:2], matches, {}, []) == "hood"
    assert hc.default_panel([], [], {"per_role": {"pickup": {"roof": {"state": "filmed"}}}}, []) == "roof"
    match = {"panel_id": "left_front_door", "state": "comparable", "pairs": [{"pickup_mark": "p3", "return_mark": "r3", "d": 0.1, "tau_eye_r": 0.8}]}
    focus, pair = hc.panel_focus("left_front_door", match, verdicts)
    assert focus is verdicts[2] and pair["return_mark"] == "r3"


def test_pick_clips_names_the_focus_marks_the_control_then_the_panel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = tmp_path / "hero_x" / "derived"
    d.mkdir(parents=True)
    (tmp_path / "hero_x" / "run.json").write_text(json.dumps({"status": "done"}))

    def clip(path: str) -> dict[str, str]:
        return {"path": path}

    files = {
        "verdicts.json": [{"panel_id": "left_front_door", "label": "already_there", "mark_ids": ["p1", "r1"]}],
        "matches.json": [{"panel_id": "left_front_door", "state": "comparable", "pairs": [{"pickup_mark": "p1", "return_mark": "r1", "d": 0.1, "tau_eye_r": 0.8}]}],
        "marks.json": [
            {"mark_id": "p1", "role": "pickup", "panel_id": "left_front_door"},
            {"mark_id": "r1", "role": "return", "panel_id": "left_front_door"},
            {"mark_id": "p2", "role": "pickup", "panel_id": "left_front_door"},
            {"mark_id": "p9", "role": "pickup", "panel_id": "hood"},
        ],
        "mark_tracks.json": [
            {"mark_id": "p9", "role": "pickup", "is_control": False, "clip": clip("clips/pickup/hood.mp4")},
            {"mark_id": "p2", "role": "pickup", "is_control": False, "clip": clip("clips/pickup/lfd_2.mp4")},
            {"mark_id": "p1", "role": "pickup", "is_control": False, "clip": clip("clips/pickup/lfd_1.mp4")},
            {"mark_id": "r1", "role": "return", "is_control": False, "clip": clip("clips/return/lfd_1.mp4")},
            {"mark_id": "c1", "role": "pickup", "is_control": True, "verdict": "fixed", "clip": clip("clips/pickup/ctl_a.mp4")},
            {"mark_id": "c2", "role": "pickup", "is_control": True, "verdict": "sliding", "clip": clip("clips/pickup/ctl_b.mp4")},
        ],
    }
    for name, value in files.items():
        (d / name).write_text(json.dumps(value))
    monkeypatch.setenv("WALKAROUND_RUNS", str(tmp_path))
    run = hc.Run("hero_x")
    assert run.panel == "left_front_door"
    assert hc.pick_clips(run, minimal=True) == ["clips/pickup/lfd_1.mp4", "clips/return/lfd_1.mp4", "clips/pickup/ctl_b.mp4"]
    assert hc.pick_clips(run) == ["clips/pickup/lfd_1.mp4", "clips/return/lfd_1.mp4", "clips/pickup/ctl_b.mp4", "clips/pickup/lfd_2.mp4"]


def test_make_sample_samples_dir_goes_to_write_and_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    ms = load("make_sample_hero", "scripts/make_sample.py")
    seen: dict[str, Any] = {}
    monkeypatch.setattr(ms, "write", lambda run_id, **kw: seen.update(kw, run_id=run_id) or 0)
    assert ms.main(["hero_x", "--samples-dir", str(tmp_path), "--slot", "friend_car", "--dry-run"]) == 0
    assert seen["samples"] == tmp_path and seen["run_id"] == "hero_x" and seen["slot"] == "friend_car"
    assert ms.main(["--check", "--samples-dir", str(tmp_path)]) == 0
    assert "0 sample folder(s)" in capsys.readouterr().out


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_intake_check_passes_a_landscape_1080p_clip_and_refuses_portrait(tmp_path: Path) -> None:
    def clip(name: str, size: str) -> Path:
        out = tmp_path / name
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"testsrc=size={size}:rate=30:duration=1", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)], check=True)
        return out

    tool = REPO / "tools" / "data_hunt" / "intake_check.py"
    wide = subprocess.run([sys.executable, "-I", str(tool), str(clip("wide.mp4", "1920x1080")), "--frames", "0"], capture_output=True, text=True)
    assert wide.returncode == 0, wide.stdout
    d = json.loads(wide.stdout)
    assert d["hard_pass"] and "duration_in_range failed" in d["warnings"] and "audio_present failed" in d["warnings"]
    tall = subprocess.run([sys.executable, "-I", str(tool), str(clip("tall.mp4", "1080x1920")), "--frames", "0"], capture_output=True, text=True)
    assert tall.returncode == 1 and not json.loads(tall.stdout)["checks"]["landscape"]


def test_hero_sh_refuses_a_bad_run_id_and_a_missing_file(tmp_path: Path) -> None:
    script = REPO / "scripts" / "hero.sh"
    a, b = tmp_path / "a.mov", tmp_path / "b.mov"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    bad_id = subprocess.run(["bash", str(script), "--dry-run", "--dest", str(tmp_path / "own"), str(a), str(b), "bad id"], capture_output=True, text=True)
    assert bad_id.returncode == 1 and "letters, digits" in bad_id.stderr
    missing = subprocess.run(["bash", str(script), "--dry-run", "--dest", str(tmp_path / "own"), str(a), str(tmp_path / "nope.mov"), "hero_t"], capture_output=True, text=True)
    assert missing.returncode == 1 and "not a file" in missing.stderr
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0
