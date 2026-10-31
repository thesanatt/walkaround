"""Eval harness (lane M17): scripts/eval_run.py on synthetic runs and labels, scripts/pseudo_pair.py's noise floor,
cut and pair lookup, the starting labels in eval/labels/, and web/public/eval/label.html (the routes it reads, its
labeling core run under node and checked by the eval's own schema, and the page in Chromium against a local stand-in
server when Playwright has its browser). No model or network call; the real ledger is never written."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
for entry in (str(REPO), str(SCRIPTS)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import eval_run  # noqa: E402
import pseudo_pair  # noqa: E402
from walkaround import config, contracts, geometry, ledger, paths, privacy  # noqa: E402
from walkaround.contracts import (  # noqa: E402
    Calibration,
    Confidence,
    Coverage,
    CoverageState,
    Crop,
    FfprobeInfo,
    InputRef,
    Keyframe,
    KeyframeScore,
    LaneRecord,
    Mark,
    MarkKind,
    MarkSource,
    MarksStatus,
    MarkTrack,
    MatchCandidate,
    MatchState,
    Noun,
    PanelCoverage,
    PanelId,
    PanelMatch,
    PanelTrack,
    RunManifest,
    RunStatus,
    SizeClass,
    SparkPurpose,
    SparkRequestRecord,
    StaysPutVerdict,
    TrackFragment,
    Verdict,
    VerdictLabel,
    VideoRecord,
)

TH = contracts.load_thresholds()
LABEL_HTML = REPO / "web" / "public" / "eval" / "label.html"
LABELS = REPO / "eval" / "labels"
SHA_P, SHA_R = "a" * 64, "b" * 64
W, H = 400, 300
TRUNK, DOOR, HOOD = "trunk_lid", "left_front_door", "hood"
P1, P2 = "pickup_trunk_lid_kf1_m1", "pickup_trunk_lid_kf1_m2"
R1, R2, R3 = "return_trunk_lid_kf1_m1", "return_trunk_lid_kf1_m2", "return_trunk_lid_kf1_m3"
R4, R5 = "return_left_front_door_kf1_m1", "return_left_front_door_kf1_m2"
PTS = {"pickup": 2.0, "return": 3.0}


def _moments() -> contracts.Moments:
    mask = np.zeros((H, W), dtype=bool)
    mask[50:250, 50:350] = True
    return geometry.moments_from_mask(mask)


MOM = _moments()


# Synthetic runs ------------------------------------------------------------------------------------------------------


@pytest.fixture
def env(tmp_path, monkeypatch):
    """runs/, the ledger and eval/snapshots/ inside tmp_path; the real ledger's bytes are checked untouched."""
    real = Path(ledger.LEDGER_PATH)
    before = real.read_bytes() if real.exists() else None
    root = tmp_path / "runs"
    monkeypatch.setattr(paths, "RUNS_DIR", root)
    monkeypatch.setattr(config, "RUNS_DIR", root)
    monkeypatch.setattr(ledger, "LEDGER_PATH", root / "ledger.jsonl")
    monkeypatch.setattr(eval_run, "SNAPSHOTS", tmp_path / "snapshots")
    (tmp_path / "labels").mkdir()
    yield tmp_path
    after = real.read_bytes() if real.exists() else b""
    assert after.startswith(before or b""), "a test rewrote the real runs/ledger.jsonl"


def kf_id(role: str, panel: str) -> str:
    return f"{role}_{panel}_kf1"


def keyframe(role: str, panel: str) -> Keyframe:
    k = kf_id(role, panel)
    return Keyframe(
        kf_id=k,
        role=role,
        panel_id=PanelId(panel),
        track_id=f"{role}_{panel}_01",
        f_proxy=int(PTS[role] * 15),
        pts=PTS[role],
        score=KeyframeScore(area=0.1, completeness=1.0, sharpness=1.0, glare=0.0, frontality=1.0, total=0.1),
        aspect_ratio=1.5,
        box_src=[100.0, 200.0, 500.0, 500.0],
        crop=Crop(path=f"crops/{role}/{k}.jpg", w=W, h=H, scale=1.0, dx=100.0, dy=200.0),
        mask_png=f"crops/{role}/{k}.mask.png",
        moments=MOM,
        front_sign=1,
        marks_status=MarksStatus.COMPLETE,
    )


def mark(mark_id: str, point: tuple[float, float], kind: str, size: str, shift: float = 0.0) -> Mark:
    role, panel = mark_id.split("_", 1)[0], TRUNK if "trunk" in mark_id else DOOR
    u, v = geometry.whiten(point, MOM, 1)
    return Mark(
        mark_id=mark_id,
        role=role,
        panel_id=PanelId(panel),
        kf_id=kf_id(role, panel),
        source=MarkSource.SPARK_BOX,
        type=MarkKind(kind),
        confidence=Confidence.HIGH,
        size_class=SizeClass(size),
        point_crop=list(point),
        bbox_crop=[point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5],
        w=[u + shift, v],
        r=math.hypot(u + shift, v),
    )


def cand(p: str, r: str, d: float, mutual: bool = True) -> MatchCandidate:
    return MatchCandidate(pickup_mark=p, return_mark=r, d=d, d_pos=d, type_cost=0.0, size_cost=0.0, r_mean=0.0, tau_pre_r=0.25, tau_eye_r=0.5, mutual=mutual)


def verdict(vid: str, panel: str, label: VerdictLabel, rule: str, ids: list[str]) -> Verdict:
    return Verdict(verdict_id=vid, panel_id=PanelId(panel), label=label, rule_id=rule, mark_ids=ids)


def pcov(state: str, kf: str | None = None) -> PanelCoverage:
    return PanelCoverage(state=CoverageState(state), seconds=2.0 if state in ("filmed", "glimpsed") else 0.0, best_kf=kf)


def write_run(root: Path, run_id: str, *, created: str = "2026-10-31T12:00:00Z", r2_shift: float = 0.0, hood_return: str = "not_filmed", return_ids: tuple[str, ...] = ("0",)) -> Path:
    """One synthetic pair run: trunk lid unchanged (two pairs and a dirt mark called new), the left front door struck
    (a new dent and a capped scratch), the hood not comparable (G1), and a left and right rear door side swap."""
    run_dir = root / run_id
    w = lambda rel, value: contracts.write_json(run_dir / rel, value)  # noqa: E731
    w("run.json", RunManifest(run_id=run_id, created_at=created, status=RunStatus.DONE, fps=15.0, thresholds=TH, config_hash=contracts.config_hash(TH)))
    for role, sha in (("pickup", SHA_P), ("return", SHA_R)):
        info = FfprobeInfo(codec="h264", container="mp4", width=1920, height=1080, display_width=1920, display_height=1080, fps_avg=30.0, duration_s=20.0)
        w(f"videos/{role}.json", VideoRecord(role=role, original_filename=f"{role}.mp4", bytes=1, sha256_server=sha, ffprobe=info))
    kfs = [keyframe("pickup", TRUNK), keyframe("return", TRUNK), keyframe("pickup", DOOR), keyframe("return", DOOR), keyframe("pickup", HOOD)]
    w("derived/keyframes.json", kfs)
    for kf in kfs:
        (run_dir / kf.crop.path).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (W, H), (120, 130, 140)).save(run_dir / kf.crop.path, quality=80)
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[50:250, 50:350] = 255
        Image.fromarray(mask).save(run_dir / kf.mask_png)
    marks = [
        mark(P1, (120, 100), "chip", "S"),
        mark(P2, (300, 200), "scratch", "M"),
        mark(R1, (121, 101), "chip", "S"),
        mark(R2, (300, 200), "scratch", "M", shift=r2_shift),
        mark(R3, (200, 120), "dirt", "S"),
        mark(R4, (150, 150), "dent", "L"),
        mark(R5, (250, 100), "scratch", "S"),
    ]
    w("derived/marks.json", marks)
    w("derived/matches.json", [
        PanelMatch(panel_id=PanelId(TRUNK), state=MatchState.COMPARABLE, kf_pair=[kf_id("pickup", TRUNK), kf_id("return", TRUNK)], pairs=[cand(P1, R1, 0.02), cand(P2, R2, 0.01)], unmatched_return=[R3]),
        PanelMatch(panel_id=PanelId(DOOR), state=MatchState.COMPARABLE, kf_pair=[kf_id("pickup", DOOR), kf_id("return", DOOR)], unmatched_return=[R4, R5]),
        PanelMatch(panel_id=PanelId(HOOD), state=MatchState.NOT_COMPARABLE, reasons=["G1"]),
    ])
    w("derived/verdicts.json", [
        verdict("trunk_lid_v1", TRUNK, VerdictLabel.ALREADY_THERE, "L3", [P1, R1]),
        verdict("trunk_lid_v2", TRUNK, VerdictLabel.ALREADY_THERE, "L3", [P2, R2]),
        verdict("trunk_lid_v3", TRUNK, VerdictLabel.NEW_OR_UNSEEN, "L2", [R3]),
        verdict("left_front_door_v1", DOOR, VerdictLabel.NEW_OR_UNSEEN, "L2", [R4]),
        verdict("left_front_door_v2", DOOR, VerdictLabel.NEEDS_YOUR_EYE, "L4", [R5]),
        verdict("hood_v1", HOOD, VerdictLabel.NOT_COMPARABLE, "L6", []),
    ])
    w("derived/coverage.json", Coverage(per_role={
        "pickup": {PanelId(TRUNK): pcov("filmed", kf_id("pickup", TRUNK)), PanelId(DOOR): pcov("filmed", kf_id("pickup", DOOR)), PanelId(HOOD): pcov("filmed", kf_id("pickup", HOOD)),
                   PanelId.RIGHT_FRONT_DOOR: pcov("not_seen"), PanelId.LEFT_REAR_DOOR: pcov("not_filmed"), PanelId.RIGHT_REAR_DOOR: pcov("filmed")},
        "return": {PanelId(TRUNK): pcov("filmed", kf_id("return", TRUNK)), PanelId(DOOR): pcov("filmed", kf_id("return", DOOR)), PanelId(HOOD): pcov(hood_return), PanelId.RIGHT_FRONT_DOOR: pcov("not_run")},
    }))
    w("derived/calibration.json", Calibration(tau_pre=0.25, tau_eye=0.5, tau_fix=0.08, calibrated=True))
    w("derived/mark_tracks.json", [
        MarkTrack(mark_id=R1, role="return", verdict=StaysPutVerdict.FIXED),
        MarkTrack(mark_id=R3, role="return", verdict=StaysPutVerdict.SLIDING),
        MarkTrack(mark_id=R4, role="return", verdict=StaysPutVerdict.INCONCLUSIVE),
        MarkTrack(mark_id="control_1", role="return", verdict=StaysPutVerdict.SLIDING, is_control=True),
    ])
    frag = [TrackFragment(object_id="0", first_frame=0, last_frame=10)]
    w("derived/tracks.json", [
        PanelTrack(track_id="p_t", role="pickup", noun=Noun.TRUNK_LID, sam_object_ids=["0"], fragments=frag, per_frame=[], c_present_share=0.8),
        PanelTrack(track_id="r_t", role="return", noun=Noun.TRUNK_LID, sam_object_ids=["0"], fragments=frag, per_frame=[], c_present_share=0.6),
        PanelTrack(track_id="r_d", role="return", noun=Noun.CAR_DOOR, sam_object_ids=["0"], fragments=frag, per_frame=[], c_present_share=1.0),
    ])
    w("spark.json", [SparkRequestRecord(request_id="resp_spark_p", purpose=SparkPurpose.MARKS, model="muse-spark-1.3", schema_name="marks_v1", input_refs=[InputRef(path=f"crops/pickup/{kf_id('pickup', TRUNK)}.jpg", sha256="c" * 64)])])
    for role, ids in (("pickup", ("0", "1")), ("return", return_ids)):
        w(f"sam/{role}/car_door.summary.json", LaneRecord(lane_id="car_door", role=role, noun=Noun.CAR_DOOR, prompt_text="car door", media_size=[960, 540], ids=list(ids)))
    events = [
        ("2026-10-31T12:00:00.000Z", "run.created", {}),
        ("2026-10-31T12:00:01.000Z", "upload.done", {"role": "pickup"}),
        ("2026-10-31T12:00:02.000Z", "upload.done", {"role": "return"}),
        ("2026-10-31T12:00:02.500Z", "stage", {"name": "sealing", "pct": 5}),
        ("2026-10-31T12:00:05.000Z", "sam.frame", {"frame": 1}),
        ("2026-10-31T12:00:10.000Z", "stage", {"name": "tracking", "pct": 45}),
        ("2026-10-31T12:00:40.000Z", "spark.response", {"role": "return", "request_id": "resp_spark_r"}),
        ("2026-10-31T12:01:00.000Z", "stage", {"name": "matching", "pct": 88}),
        ("2026-10-31T12:01:30.000Z", "run.completed", {"status": "done"}),
    ]
    lines = [json.dumps({"seq": i, "ts": ts, "run_id": run_id, "kind": kind, "payload": payload, "prev": "0" * 64, "hash": "0" * 64}) for i, (ts, kind, payload) in enumerate(events)]
    (run_dir / "events.ndjson").write_text("\n".join(lines) + "\n", encoding="utf-8")
    rows = [
        {"kind": "sam_video", "run_id": run_id, "note": "car_door", "usd": 0.01, "role": "pickup", "latency_ms": {"upload": None, "first_delta": 2000, "done": 5000}, "parse_cpu_ms": 10},
        {"kind": "sam_video", "run_id": run_id, "note": "car_door", "usd": 0.02, "role": "return", "latency_ms": {"upload": None, "first_delta": 3000, "done": 6000}, "parse_cpu_ms": 20},
        {"kind": "sam_image", "run_id": run_id, "note": "license plate", "usd": 0.0025, "role": "pickup"},
        {"kind": "spark", "run_id": run_id, "note": "marks", "usd": 0.003, "request_id": "resp_spark_p", "input_tokens": 1000, "output_tokens": 200, "reasoning_tokens": 150},
        {"kind": "spark", "run_id": run_id, "note": "marks", "usd": 0.004, "request_id": "resp_spark_r", "input_tokens": 1200, "output_tokens": 300, "reasoning_tokens": 250},
        {"kind": "voice", "run_id": run_id, "note": "ENDPOINTING", "usd": 0.0009, "request_id": "voyager/x"},
        {"kind": "sam_video", "run_id": "someone_else", "note": "car", "usd": 9.0, "role": "pickup"},
    ]
    with (root / "ledger.jsonl").open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps({"ts": "2026-10-31T12:00:00-0400", "model": "m", **row}) + "\n")
    return run_dir


def labels_doc(*, pair: str = "synth", run_id: str | None = "eval_synth", nothing_changed: bool | None = False, with_marks: bool = True) -> dict:
    ref = {"ref_id": "debris", "panel_id": TRUNK, "role": "return", "t": PTS["return"], "point_src": [300.0, 321.0], "type": "dirt", "size_class": "S", "real": False, "not_real_reason": "dirt", "group": "debris", "note": "a leaf"}
    doc = {
        "schema": "walkaround_labels_v1",
        "pair_id": pair,
        "status": "labeled by a test, not yet reviewed",
        "kind": "cross_walk",
        "table": "B",
        "split": "report",
        "nothing_changed": nothing_changed,
        "footage": {"pickup": {"path": "data/derived/p.mp4", "sha256": SHA_P}, "return": {"path": "data/derived/r.mp4", "sha256": SHA_R}},
        "run_id": run_id,
        "panels": {
            TRUNK: {"changed": False, "coverage": {"pickup": "filmed", "return": "filmed"}, "marks_checked": {"pickup": True, "return": True}},
            DOOR: {"changed": True, "coverage": {"pickup": "filmed", "return": "glimpsed"}, "marks_checked": {"pickup": True, "return": True}},
            HOOD: {"changed": False, "coverage": {"pickup": "filmed", "return": "not_filmed"}},
            "right_front_door": {"changed": None, "coverage": {"pickup": "not_filmed", "return": "not_filmed"}},
            "left_rear_door": {"coverage": {"pickup": "filmed"}},
            "right_rear_door": {"coverage": {"pickup": "not_filmed"}},
        },
        "marks": [],
        "reference_marks": [ref],
    }
    if nothing_changed:
        doc["panels"][DOOR]["changed"] = False
    if not with_marks:  # panel labels only: nobody looked for marks
        doc["reference_marks"] = []
        for panel in doc["panels"].values():
            panel.pop("marks_checked", None)
    if with_marks:
        system = {P1: True, P2: True, R1: True, R2: True, R3: False, R4: True, R5: True}
        for mid, real in system.items():
            role, panel = mid.split("_", 1)[0], TRUNK if "trunk" in mid else DOOR
            entry = {"mark_id": mid, "source": "system", "role": role, "panel_id": panel, "kf_id": kf_id(role, panel), "real": real, "new": None}
            if mid == R3:
                entry["not_real_reason"] = "dirt"
            if mid == R4:
                entry["new"] = True
            if mid == R5:
                entry["new"] = False
            doc["marks"].append(entry)
        doc["marks"].append({"mark_id": "hand_1", "source": "hand", "role": "pickup", "panel_id": DOOR, "kf_id": kf_id("pickup", DOOR), "point_crop": [250.0, 100.0], "type": "chip", "size_class": "S", "real": True, "new": None})
    return doc


def write_labels(env: Path, doc: dict) -> Path:
    path = env / "labels" / f"{doc['pair_id']}.json"
    path.write_text(json.dumps(doc, indent=1), encoding="ascii")
    return path


def evaluate(env: Path, **kw):
    results, errors = eval_run.evaluate(sorted((env / "labels").glob("*.json")), **kw)
    assert not errors, errors
    return results


# Counting and the labels schema ------------------------------------------------------------------------------------


def test_count_prints_a_percentage_only_from_20():
    assert eval_run.count(41, 53)["text"] == "41 of 53 (77%)"
    assert eval_run.count(7, 9)["text"] == "7 of 9"
    assert eval_run.count(19, 19)["text"] == "19 of 19"
    assert eval_run.count(0, 0) == {"n": 0, "m": 0, "text": "0 of 0"}


def test_labels_schema_names_every_problem():
    good = labels_doc()
    assert eval_run.label_problems(good) == []
    bad = json.loads(json.dumps(good))
    bad["panels"]["bonnet"] = {"changed": False}
    bad["panels"][TRUNK]["coverage"]["pickup"] = "seen"
    bad["marks"][-1]["real"] = None
    bad["status"] = " "
    bad["surprise"] = 1
    problems = " | ".join(eval_run.label_problems(bad))
    for needle in ("unknown panel id 'bonnet'", "panels.trunk_lid.coverage", "a hand mark is one the labeler saw", "status must say", "unknown key 'surprise'"):
        assert needle in problems
    no_run = labels_doc(run_id=None)
    assert eval_run.label_problems(no_run) == ["marks are tied to a run, so run_id must be set"]


def test_load_labels_refuses_a_broken_file(tmp_path):
    path = tmp_path / "x.json"
    path.write_text("{not json", encoding="ascii")
    with pytest.raises(eval_run.LabelError, match="not JSON"):
        eval_run.load_labels(path)


# Metrics on the labeled run ------------------------------------------------------------------------------------------


def test_metrics_on_the_labeled_run(env):
    write_run(env / "runs", "eval_synth")
    write_labels(env, labels_doc())
    pair = evaluate(env)["pairs"]["synth"]
    assert pair["primary_run"] == "eval_synth" and pair["runs"] == ["eval_synth"]
    marks = pair["marks"]
    assert marks["precision"]["text"] == "6 of 7"
    assert marks["false_mark_rate"]["text"] == "1 of 7"
    assert marks["recall"]["text"] == "6 of 7"  # the hand mark on the pickup door was missed
    assert {k: v["text"] for k, v in marks["recall_by_size"].items()} == {"L": "1 of 1", "M": "2 of 2", "S": "3 of 4"}
    assert marks["per_panel"][TRUNK] == {"precision": eval_run.count(4, 5), "recall": eval_run.count(4, 4)}
    assert marks["per_panel"][DOOR] == {"precision": eval_run.count(2, 2), "recall": eval_run.count(2, 3)}
    v = pair["verdicts"]
    assert v["panel_calls_right"]["text"] == "1 of 2"  # the trunk lid's dirt mark made it look changed
    assert v["changed_panels_called_changed"]["text"] == "1 of 1"
    assert (v["panels_not_comparable"], v["not_comparable_gates"], v["panels_not_compared"]) == (1, {"G1": 1}, 0)
    assert v["needs_your_eye_share"]["text"] == "1 of 5"
    assert v["re_shoot_match_unchanged_panels"]["text"] == "2 of 2"
    assert v["false_new"]["text"] == "0 of 3"
    assert v["false_pre_existing"]["text"] == "0 of 1"
    assert v["mark_verdicts_right"]["text"] == "3 of 3"
    assert v["needs_your_eye_resolved_already_there"]["text"] == "1 of 1"
    assert v["re_shoot_match_real_marks"]["text"] == "2 of 2"
    assert "false_new_nothing_changed" not in v


def test_coverage_accuracy_lane_not_run_and_side_swap(env):
    write_run(env / "runs", "eval_synth")
    write_labels(env, labels_doc())
    cov = evaluate(env)["pairs"]["synth"]["coverage"]
    assert cov["right"]["text"] == "6 of 9"
    assert cov["lane_not_run"] == 1  # right front door on the return walk
    assert cov["side_swaps"] == 1  # left and right rear doors traded states on the pickup walk
    assert {"role": "return", "panel_id": DOOR, "label": "glimpsed", "system": "filmed"} in cov["wrong"]
    assert cov["confusion"]["not_filmed -> not_filmed"] == 2  # not_seen reads as not_filmed


def test_staysput_agreement_and_controls(env):
    write_run(env / "runs", "eval_synth")
    write_labels(env, labels_doc())
    sp = evaluate(env)["pairs"]["synth"]["staysput"]
    assert sp["agreement"]["text"] == "1 of 2"  # R1 real and fixed; R3 is dirt, so it should have stayed fixed
    assert sp["inconclusive_or_no_track"]["text"] == "1 of 3"
    assert sp["controls_sliding"]["text"] == "1 of 1"


def test_staysput_not_run_when_mark_tracks_are_empty(env):
    run_dir = write_run(env / "runs", "eval_synth")
    contracts.write_json(run_dir / "derived" / "mark_tracks.json", [])
    write_labels(env, labels_doc())
    assert evaluate(env)["pairs"]["synth"]["staysput"]["status"] == "not run"


def test_false_new_where_nothing_changed_from_panel_labels_alone(env):
    write_run(env / "runs", "eval_synth")
    write_labels(env, labels_doc(run_id=None, nothing_changed=True, with_marks=False))
    pair = evaluate(env)["pairs"]["synth"]
    v = pair["verdicts"]
    assert v["false_new_nothing_changed"]["text"] == "2 of 5"
    assert v["panels_with_a_new_call"]["text"] == "2 of 2"
    assert v["panel_calls_right"]["text"] == "0 of 2"
    assert pair["marks"]["status"] == "no mark labels for this run"
    missing = {m["what"]: m["commands"] for m in pair["missing"]}
    assert missing["mark labels"] == [
        "RUN_RETENTION_HOURS=720 .venv/bin/walkaround serve",  # the sweeper would delete a keep=false run a day old
        "open 'http://127.0.0.1:8000/eval/label.html?run=eval_synth&pair=synth'",
        'mv "$(ls -t ~/Downloads/synth*.json | head -1)" eval/labels/synth.json',  # the newest, '(1)' included
    ]
    assert "second run (run-to-run stability)" not in missing  # asked for pair C only


def test_cost_and_wall_clock_per_walk(env):
    write_run(env / "runs", "eval_synth")
    write_labels(env, labels_doc())
    cost = evaluate(env)["pairs"]["synth"]["cost"]["eval_synth"]
    assert cost["rows"] == 6  # the other run's row is not counted
    assert cost["walks"]["pickup"] == {"usd": 0.0155, "video_s": 20.0, "usd_per_video_s": 0.000775}
    assert cost["walks"]["return"] == {"usd": 0.024, "video_s": 20.0, "usd_per_video_s": 0.0012}  # Spark row by its event
    assert cost["usd_unattributed"] == 0.0009 and cost["usd_total"] == 0.0404
    assert cost["first_delta_p50_s"] == {"car_door": 2.5} and cost["parse_cpu_ms_by_lane"] == {"car_door": 30}
    assert cost["spark_tokens_p50"]["input_tokens"] == 1100
    wc = cost["wall_clock"]
    assert (wc["first_event_to_end_s"], wc["upload_to_end_s"], wc["ended_with"]) == (90.0, 88.0, "run.completed")
    assert wc["stage_s"] == [{"name": "sealing", "s": 7.5}, {"name": "tracking", "s": 50.0}, {"name": "matching", "s": 30.0}]


def test_calibration_and_c_present_share(env):
    write_run(env / "runs", "eval_synth")
    write_labels(env, labels_doc())
    pair = evaluate(env)["pairs"]["synth"]
    assert pair["calibration"]["tau_pre"] == 0.25 and pair["calibration"]["calibrated"] is True
    assert pair["c_present_share"] == {"car door": {"median": 1.0, "tracks": 1}, "trunk lid": {"median": 0.7, "tracks": 2}}
    assert pair["billable"] is None  # no verdict carries a ruler size


# Two runs: transfer and stability --------------------------------------------------------------------------------------


def test_run_to_run_stability(env):
    write_run(env / "runs", "eval_synth")
    write_run(env / "runs", "eval_synth_2", created="2026-10-31T13:00:00Z", r2_shift=0.5, hood_return="glimpsed", return_ids=("0", "1", "2"))
    write_labels(env, labels_doc())
    pair = evaluate(env)["pairs"]["synth"]
    assert pair["runs"] == ["eval_synth", "eval_synth_2"] and pair["primary_run"] == "eval_synth"
    (s,) = pair["stability"]
    assert s["other_run"] == "eval_synth_2"
    assert s["marks_found_again"]["text"] == "6 of 7"  # one return mark moved half a unit
    assert s["coverage_same"]["text"] == "8 of 9"
    assert s["best_keyframe_same_moment"]["text"] == "5 of 5"
    assert s["lane_object_ids_same"]["text"] == "1 of 2"
    assert s["panel_verdicts_same"]["text"] == "3 of 3"
    assert not any(m["what"].startswith("second run") for m in pair["missing"])


def test_labels_transfer_to_another_run_of_the_same_walks(env):
    write_run(env / "runs", "eval_synth")
    write_run(env / "runs", "eval_synth_2", created="2026-10-31T13:00:00Z", r2_shift=0.5)
    write_labels(env, labels_doc())
    marks = evaluate(env, overrides={"synth": "eval_synth_2"})["pairs"]["synth"]["marks"]
    assert marks["precision"]["text"] == "5 of 7"  # the moved mark has nothing real near it on a checked panel
    assert marks["recall"]["text"] == "5 of 7"


def test_reference_marks_land_on_a_keyframe_cut_at_the_same_slot(env):
    run_dir = write_run(env / "runs", "eval_synth")
    run = eval_run.load_run("eval_synth", run_dir, False)
    ref = labels_doc()["reference_marks"][0]
    w = eval_run.place_reference(ref, run)
    expected = geometry.whiten((200.0, 121.0), MOM, 1)
    assert w is not None and math.dist(w, expected) < 1e-9
    assert eval_run.place_reference({**ref, "t": PTS["return"] + 1 / 15}, run) is None  # the next 15 fps slot
    assert eval_run.place_reference({**ref, "t": PTS["return"] + 0.000002}, run) is None
    assert eval_run.same_slot(16.4, 16.400001) and eval_run.same_slot(16.466667, 16.4666667)
    assert not eval_run.same_slot(16.4, 16.43002)  # a source frame between two slots is never cut from


def test_reference_marks_count_once_per_walk_and_group(env):
    run_dir = write_run(env / "runs", "eval_synth")
    run = eval_run.load_run("eval_synth", run_dir, False)
    doc = labels_doc(run_id=None, with_marks=False)
    ref = {"ref_id": "chip_a", "panel_id": TRUNK, "role": "return", "t": PTS["return"], "point_src": [300.0, 321.0], "type": "chip", "size_class": "S", "real": True, "new": False, "group": "chip"}
    doc["reference_marks"] = [ref, {**ref, "ref_id": "chip_b", "point_src": [302.0, 322.0]}, {**ref, "ref_id": "chip_p", "role": "pickup", "t": PTS["pickup"]}]
    truths, notes = eval_run.truths_for(doc, run, None, 0.25)
    assert [t.key for t in truths] == ["chip_a", "chip_p"]
    assert notes == {"reference marks repeating a placed group": 1}


def test_hand_marks_on_a_wheel_use_wheel_coordinates(env):
    run_dir = write_run(env / "runs", "eval_synth")
    run = eval_run.load_run("eval_synth", run_dir, False)
    kf = keyframe("pickup", "left_front_wheel")
    (run_dir / kf.mask_png).parent.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[0:H, 0:W]
    Image.fromarray((((xx - 200) ** 2 + (yy - 150) ** 2) <= 100**2).astype(np.uint8) * 255).save(run_dir / kf.mask_png)
    radius, _angle = eval_run.whiten_point(run, kf, [200.0, 150.0])
    assert radius == pytest.approx(0.0, abs=0.02)  # the centre of the wheel
    assert eval_run.whiten_point(run, kf.model_copy(update={"front_sign": None}), [200.0, 150.0]) is None


# Missing runs and labels, determinism, snapshots, the command line -----------------------------------------------------


def test_a_missing_run_prints_the_command_that_makes_it(env):
    doc = labels_doc(pair="ghost", run_id=None, with_marks=False)
    doc["footage"]["pickup"]["sha256"] = "d" * 64
    write_labels(env, doc)
    results = evaluate(env)
    (miss,) = results["pairs"]["ghost"]["missing"]
    assert miss["what"] == "run"
    assert miss["commands"] == ["LIVE=1 .venv/bin/walkaround run data/derived/p.mp4 data/derived/r.mp4 --run-id eval_ghost"]
    for pair in pseudo_pair.PAIRS:  # every pseudo pair without a labels file says how to make one
        assert results["pairs"][pair]["missing"][0]["what"] == "labels"
        assert any(c.startswith("LIVE=1 .venv/bin/walkaround run ") for c in results["pairs"][pair]["missing"][0]["commands"])


def test_an_unreadable_run_is_reported_and_the_rest_still_scores(env):
    write_run(env / "runs", "eval_synth")
    broken = write_run(env / "runs", "eval_synth_2", created="2026-10-31T13:00:00Z")
    (broken / "derived" / "marks.json").write_text('[{"mark_id": "x"}]', encoding="ascii")
    write_labels(env, labels_doc())
    pair = evaluate(env)["pairs"]["synth"]
    assert pair["runs"] == ["eval_synth"] and pair["marks"]["precision"]["text"] == "6 of 7"
    assert any(m["what"] == "readable run" and m["note"].startswith("eval_synth_2 does not read") for m in pair["missing"])


def test_run_override_and_labeled_run_missing(env):
    write_run(env / "runs", "eval_synth")
    write_run(env / "runs", "other_name")  # same walks, found by the hashes of both originals
    doc = labels_doc(run_id="gone_run")
    write_labels(env, doc)
    pair = evaluate(env, overrides={"synth": "other_name"})["pairs"]["synth"]
    assert pair["primary_run"] == "other_name" and pair["runs"] == ["eval_synth", "other_name"]
    assert any(m["what"] == "labeled run" for m in pair["missing"])
    assert pair["marks"]["notes"]["mark labels made on a run that is not here"] == 8
    nowhere = evaluate(env, overrides={"synth": "not_there"})["pairs"]["synth"]
    assert nowhere["primary_run"] is None and nowhere["missing"][-1]["what"] == "run"


def test_a_reference_mark_on_an_already_labeled_mark_counts_once(env):
    run_dir = write_run(env / "runs", "eval_synth")
    doc = labels_doc()
    doc["reference_marks"] = [{"ref_id": "chip_again", "panel_id": TRUNK, "role": "pickup", "t": PTS["pickup"], "point_src": [220.0, 300.0], "type": "chip", "size_class": "S", "real": True, "group": "chip"}]
    run = eval_run.load_run("eval_synth", run_dir, False)
    truths, notes = eval_run.truths_for(doc, run, run, 0.25)
    assert notes == {"reference marks already labeled on the run": 1}
    assert [t.key for t in truths if t.source == "reference"] == []


def test_results_are_byte_identical_and_hold_no_clock(env):
    write_run(env / "runs", "eval_synth")
    write_labels(env, labels_doc())
    a, b = contracts.json_bytes(evaluate(env)), contracts.json_bytes(evaluate(env))
    assert a == b
    text = a.decode("ascii")
    assert "2026-10-31T12:" not in text.split('"inputs"')[0].replace('"ts"', "")  # event times feed durations only
    inputs = json.loads(text)["inputs"]
    assert inputs["eval_synth/derived/marks.json"] == hashlib.sha256((env / "runs" / "eval_synth" / "derived" / "marks.json").read_bytes()).hexdigest()


def test_snapshots_rebuild_the_same_results(env):
    write_run(env / "runs", "eval_synth")
    write_run(env / "runs", "eval_synth_2", created="2026-10-31T13:00:00Z", r2_shift=0.5)
    write_labels(env, labels_doc())
    live = contracts.json_bytes(evaluate(env))
    for run_id in ("eval_synth", "eval_synth_2"):
        dst = eval_run.snapshot_run(eval_run.load_run(run_id, env / "runs" / run_id, False))
        assert not list(dst.rglob("*.jpg")) and not (dst / "voice.json").exists()
        assert (dst / "ledger.jsonl").read_text(encoding="ascii").count("\n") == 6
    shutil.rmtree(env / "runs")
    assert contracts.json_bytes(evaluate(env, from_snapshots=True)) == live


def test_main_writes_results_and_prints_the_tables(env, capsys):
    write_run(env / "runs", "eval_synth")
    write_labels(env, labels_doc())
    out = env / "results.json"
    md = env / "eval.md"
    assert eval_run.main(["--labels", str(env / "labels"), "--out", str(out), "--markdown", str(md)]) == 0
    printed = capsys.readouterr().out
    assert out.read_bytes() == contracts.json_bytes(json.loads(out.read_text(encoding="ascii")))
    for needle in ("## synth", "| Panel calls right | 1 of 2", "| Mark precision | 6 of 7", "| Coverage state right | 6 of 9", "| synth | eval_synth | pickup | 20.0 | 0.0155 | 0.0008 |"):
        assert needle in printed
    assert md.read_text(encoding="ascii") == printed
    assert printed.isascii()


def test_main_snapshot_scores_from_the_copies(env, capsys):
    write_run(env / "runs", "eval_synth")
    write_labels(env, labels_doc())
    out = env / "results.json"
    assert eval_run.main(["--labels", str(env / "labels"), "--out", str(out), "--snapshot"]) == 0
    assert (env / "snapshots" / "eval_synth" / "derived" / "marks.json").is_file()
    assert "snapshot: " in capsys.readouterr().err
    shutil.rmtree(env / "runs")
    again = env / "again.json"
    assert eval_run.main(["--labels", str(env / "labels"), "--out", str(again), "--from-snapshots"]) == 0
    assert again.read_bytes() == out.read_bytes()


def test_main_refuses_a_bad_labels_file_and_writes_nothing(env, capsys):
    write_run(env / "runs", "eval_synth")
    bad = labels_doc()
    bad["kind"] = "vibes"
    write_labels(env, bad)
    out = env / "r.json"
    assert eval_run.main(["--labels", str(env / "labels"), "--out", str(out)]) == 1
    captured = capsys.readouterr()
    assert "labels file refused" in captured.err and "was not written" in captured.err
    assert not out.exists() and "Refused labels files (results.json was not written):" in captured.out


def test_a_labels_file_must_be_named_for_its_pair(env):
    write_run(env / "runs", "eval_synth")
    doc = labels_doc()
    (env / "labels" / "synth (1).json").write_text(json.dumps(doc), encoding="ascii")  # a second browser download
    write_labels(env, doc)
    results, errors = eval_run.evaluate(sorted((env / "labels").glob("*.json")))
    assert len(errors) == 1 and "differs from the file name; save it as eval/labels/synth.json" in errors[0]
    assert results["pairs"]["synth"]["labels"]["path"].endswith("labels/synth.json")
    doc = labels_doc(pair="px4228872_cut")
    (env / "labels" / "px4228872_cut.json").write_text(json.dumps({**doc, "table": "Z"}), encoding="ascii")
    results, errors = eval_run.evaluate(sorted((env / "labels").glob("*.json")))
    (miss,) = results["pairs"]["px4228872_cut"]["missing"]  # a refused pseudo-pair file is not reported as missing
    assert miss["commands"] == [] and "was refused" in miss["note"]


def test_footage_must_be_a_licensed_repo_path():
    for path, needle in (("data/eval_internal/yt_01.mp4", "no eval may use"), ("", "repo-relative"), ("/Users/x/walk.mp4", "repo-relative"), ("data/../../walk.mp4", "repo-relative")):
        doc = labels_doc()
        doc["footage"]["return"]["path"] = path
        assert any(needle in p for p in eval_run.label_problems(doc)), path
    doc = labels_doc()
    doc["footage"]["pickup"]["source"] = "data/EVAL_INTERNAL/x.mp4"
    assert any("no eval may use" in p for p in eval_run.label_problems(doc))
    doc = labels_doc()
    doc["marks"][0]["size_class"] = "XL"
    assert any("size class the labeler set" in p for p in eval_run.label_problems(doc))


def test_an_uncalled_mark_on_a_checked_panel_is_unjudged_on_the_labeled_run_only(env):
    write_run(env / "runs", "eval_synth")
    write_run(env / "runs", "eval_synth_2", created="2026-10-31T13:00:00Z")
    doc = labels_doc()
    doc["marks"] = [m for m in doc["marks"] if m["mark_id"] != R3]  # the dirt mark left without a call
    doc["reference_marks"] = []  # the leaf reference sits on R3 and would judge it
    write_labels(env, doc)
    marks = evaluate(env)["pairs"]["synth"]["marks"]
    assert marks["precision"]["text"] == "6 of 6" and marks["unjudged_on_checked_panels"] == 1
    other = evaluate(env, overrides={"synth": "eval_synth_2"})["pairs"]["synth"]["marks"]
    assert other["precision"]["text"] == "6 of 7" and other["unjudged_on_checked_panels"] == 0  # nothing labeled near it


def test_hand_marks_on_a_real_system_mark_count_once_and_sizes_follow_the_labeler(env):
    write_run(env / "runs", "eval_synth")
    doc = labels_doc()
    doc["marks"].append({"mark_id": "hand_2", "source": "hand", "role": "pickup", "panel_id": TRUNK, "kf_id": kf_id("pickup", TRUNK), "point_crop": [121.0, 100.0], "type": "chip", "size_class": "S", "real": True, "new": None})
    next(m for m in doc["marks"] if m["mark_id"] == R4)["size_class"] = "M"  # the system said L
    write_labels(env, doc)
    marks = evaluate(env)["pairs"]["synth"]["marks"]
    assert marks["notes"]["hand marks on a system mark called real"] == 1
    assert marks["recall"]["text"] == "6 of 7"  # the blind-pass hand mark on P1 is P1 found again, not a second mark
    assert {k: v["text"] for k, v in marks["recall_by_size"].items()} == {"L": "0 of 0", "M": "3 of 3", "S": "3 of 4"}
    assert marks["labeler_overrides"]["text"] == "3 of 9"  # R3 called not real, two marks added by hand


def test_billable_counts_only_new_or_unseen_and_needs_your_eye_resolution_needs_a_return_mark(env):
    run_dir = write_run(env / "runs", "eval_synth")
    verdicts = json.loads((run_dir / "derived" / "verdicts.json").read_text(encoding="ascii"))
    for v in verdicts:
        if v["rule_id"] != "L6":
            v["size"] = {"inches": 30 / 25.4}
    verdicts.append({"verdict_id": "trunk_lid_v9", "panel_id": TRUNK, "label": "needs_your_eye", "rule_id": "L4", "mark_ids": [P2]})
    (run_dir / "derived" / "verdicts.json").write_text(json.dumps(verdicts), encoding="ascii")
    write_labels(env, labels_doc())
    pair = evaluate(env)["pairs"]["synth"]
    assert {k: v["text"] for k, v in pair["billable"]["new_or_unseen"].items()} == {"10 mm": "2 of 2", "20 mm": "2 of 2", "25.4 mm": "2 of 2"}
    assert pair["billable"]["needs_your_eye"]["25.4 mm"]["text"] == "1 of 1"  # the pickup-only one has no size
    v = pair["verdicts"]
    assert v["needs_your_eye_resolved_already_there"]["text"] == "1 of 1" and v["needs_your_eye_pickup_only"] == 1


def test_stability_compares_derived_files_only_under_the_same_thresholds(env):
    write_run(env / "runs", "eval_synth")
    other = write_run(env / "runs", "eval_synth_2", created="2026-10-31T13:00:00Z")
    changed = json.loads(json.dumps(TH))
    changed["eval"]["recall_radius_units"] = 0.3
    run = contracts.read_json(other / "run.json", RunManifest)
    contracts.write_json(other / "run.json", run.model_copy(update={"thresholds": changed, "config_hash": contracts.config_hash(changed)}))
    write_labels(env, labels_doc())
    results = evaluate(env)
    (s,) = results["pairs"]["synth"]["stability"]
    assert s["same_config"] is False and "coverage_same" not in s and s["lane_object_ids_same"]["text"] == "2 of 2"
    assert "thresholds differ" in eval_run.markdown(results)


def test_second_run_is_asked_for_pair_c_only(env):
    write_run(env / "runs", "eval_uco3d_pairC")
    write_labels(env, labels_doc(pair="uco3d_pairC", run_id=None, with_marks=False))
    missing = {m["what"]: m for m in evaluate(env)["pairs"]["uco3d_pairC"]["missing"]}
    second = missing["second run (run-to-run stability)"]
    assert second["commands"][-1].endswith("--run-id eval_uco3d_pairC_2") and second["note"].startswith("optional")


def test_handle_residual_chance_match_coverage_per_panel_and_method_lines(env):
    run_dir = write_run(env / "runs", "eval_synth")
    matches = json.loads((run_dir / "derived" / "matches.json").read_text(encoding="ascii"))
    matches[0]["handle_residual"] = [0.03, 0.04]
    (run_dir / "derived" / "matches.json").write_text(json.dumps(matches), encoding="ascii")
    write_labels(env, labels_doc())
    results = evaluate(env)
    pair = results["pairs"]["synth"]
    assert pair["handle_residual_units"] == {"n": 1, "median": 0.05, "max": 0.05}
    assert pair["chance_match"] == {"tau_pre": 0.25, "per_pickup_mark": 0.016362, "pickup_marks": 2, "comparable_panels": 2, "per_false_mark": 0.016362}
    assert pair["coverage"]["per_panel"][TRUNK]["text"] == "2 of 2" and pair["coverage"]["per_panel"][DOOR]["text"] == "1 of 2"
    assert results["method_lines"] == [
        "synth: false new on marks labeled pre-existing, 0 of 3; labels not yet reviewed",
        "synth: re-shoot match, 2 of 2 pickup marks found again as already there; labels not yet reviewed",
        "synth: needs your eye, 1 of 5 verdicts; the labeler resolved 1 of 1 as already there; labels not yet reviewed",
    ]
    assert all(line.isascii() for line in results["method_lines"])


def test_snapshots_carry_no_spoken_words_file_name_camera_time_or_model_text(env):
    run_dir = write_run(env / "runs", "eval_synth")
    marks = json.loads((run_dir / "derived" / "marks.json").read_text(encoding="ascii"))
    marks[5].update({"source": "narration_seed", "description": "my own words about the door"})
    contracts.write_json(run_dir / "derived" / "marks.json", marks)
    video = json.loads((run_dir / "videos" / "pickup.json").read_text(encoding="ascii"))
    video["original_filename"] = "IMG_0042 Jane.MOV"
    video["ffprobe"].update({"device": "iPhone 15", "creation_time": "2026-10-30T10:00:00Z"})
    contracts.write_json(run_dir / "videos" / "pickup.json", video)
    spark = json.loads((run_dir / "spark.json").read_text(encoding="ascii"))
    spark[0]["raw_text"] = '{"marks": [{"note": "the renter said a scratch"}]}'
    contracts.write_json(run_dir / "spark.json", spark)
    run = json.loads((run_dir / "run.json").read_text(encoding="ascii"))
    run["clocks"] = {"camera_time": "2026-10-30T10:00:00Z", "sealed_minutes_after_camera": 12}
    contracts.write_json(run_dir / "run.json", run)
    write_labels(env, labels_doc())
    live = contracts.json_bytes(evaluate(env))
    dst = eval_run.snapshot_run(eval_run.load_run("eval_synth", run_dir, False))
    text = "".join(p.read_text(encoding="ascii") for p in dst.rglob("*.json"))
    for secret in ("my own words", "Jane", "iPhone", "2026-10-30T10:00", "the renter said"):
        assert secret not in text, secret
    assert json.loads((dst / "videos" / "pickup.json").read_text(encoding="ascii"))["original_filename"] == "pickup.mov"
    note = json.loads((dst / "snapshot.json").read_text(encoding="ascii"))
    assert note["contracts_sha256"] == hashlib.sha256((REPO / "walkaround" / "contracts.py").read_bytes()).hexdigest()
    shutil.rmtree(env / "runs")
    assert contracts.json_bytes(evaluate(env, from_snapshots=True)) == live  # the same redaction on both sides


# Pseudo pairs ----------------------------------------------------------------------------------------------------------


def _pm(state: MatchState, pairs: list[MatchCandidate], unmatched_p: list[str] = ()) -> PanelMatch:
    return PanelMatch(panel_id=PanelId.HOOD, state=state, pairs=pairs, unmatched_pickup=list(unmatched_p))


def test_noise_floor_raises_tau_pre_on_a_wide_spread():
    ds = [0.05, 0.1, 0.12, 0.2, 0.3, 0.32, 0.35, 0.4, 0.45]
    pairs = [cand(f"p{i}", f"r{i}", d) for i, d in enumerate(ds)]
    matches = [_pm(MatchState.COMPARABLE, pairs + [cand("px", "rx", 0.01, mutual=False)], ["p_alone"]), _pm(MatchState.NOT_COMPARABLE, [cand("q", "s", 0.01)])]
    floor = pseudo_pair.noise_floor(matches, Calibration(tau_pre=0.25, tau_eye=0.5, tau_fix=0.08), TH)
    assert floor["n_rows"] == 9 and floor["under_tau_pre"] == 4
    assert [row["n"] for row in floor["histogram"]] == [0, 1, 2, 0, 1, 0, 2, 1, 1, 1, 0]
    assert floor["tau_pre_implied"] == 0.4 and floor["calibrated_implied"] is False and floor["raised"] is True
    assert floor["pickup_marks_matched"] == [4, 11]  # 4 under tau_pre_r of 11 pickup marks on comparable panels
    assert pseudo_pair.floor_line(floor, "time_split") == "4 of 11 pickup marks matched, same car, one clip split in two (noise floor)"
    assert pseudo_pair.floor_line(floor, "two_takes") == "4 of 11 pickup marks matched, same car, two takes from one shoot (noise floor)"
    assert pseudo_pair.floor_line(floor) == "4 of 11 pickup marks matched, same car, two walks where nothing changed (noise floor)"


def test_noise_floor_keeps_tau_pre_when_the_pairs_sit_tight():
    pairs = [cand(f"p{i}", f"r{i}", 0.02 * (i + 1)) for i in range(10)]
    floor = pseudo_pair.noise_floor([_pm(MatchState.COMPARABLE, pairs)], Calibration(tau_pre=0.25, tau_eye=0.5, tau_fix=0.08), TH)
    assert floor["raised"] is False and floor["tau_pre_implied"] == 0.25 and floor["calibrated_implied"] is True


def test_calibrate_run_writes_the_floor(env):
    write_run(env / "runs", "eval_synth")
    result, out = pseudo_pair.calibrate_run("eval_synth", out_dir=env / "pseudo")
    assert out == env / "pseudo" / "eval_synth.json"
    assert result["n_rows"] == 2 and result["line"].startswith("2 of 2 pickup marks matched")
    assert json.loads(out.read_text(encoding="ascii"))["schema"] == "walkaround_pseudo_v1"
    with pytest.raises(FileNotFoundError):
        pseudo_pair.calibrate_run("no_such_run", out_dir=env / "pseudo")


def test_calibrate_finds_the_pair_of_a_run_and_refuses_a_crash_pair(env, monkeypatch, capsys):
    monkeypatch.setattr(pseudo_pair, "PSEUDO_OUT", env / "pseudo")
    assert pseudo_pair.pair_for_run("eval_px4228872_cut_2").pair_id == "px4228872_cut"
    assert pseudo_pair.pair_for_run("eval_nhtsa15759").pair_id == "nhtsa15759"
    assert pseudo_pair.pair_for_run("eval_px4228872") is None and pseudo_pair.pair_for_run("x", {"pickup": None, "return": None}) is None
    footage = eval_run.load_labels(LABELS / "nhtsa15759.json")["footage"]
    found = pseudo_pair.pair_for_run("renamed_run", {role: footage[role]["sha256"] for role in ("pickup", "return")})
    assert found is not None and found.pair_id == "nhtsa15759"
    write_run(env / "runs", "eval_nhtsa15759")
    assert pseudo_pair.main(["calibrate", "eval_nhtsa15759"]) == 2  # no --pair needed to refuse it
    assert "before_after pair" in capsys.readouterr().err
    write_run(env / "runs", "eval_px4228872_front")
    assert pseudo_pair.main(["calibrate", "eval_px4228872_front", "--pair", "px_two_takes"]) == 2
    assert pseudo_pair.main(["calibrate", "eval_px4228872_front"]) == 0
    assert "same car, one clip split in two (noise floor)" in capsys.readouterr().out
    assert json.loads((env / "pseudo" / "eval_px4228872_front.json").read_text(encoding="ascii"))["kind"] == "time_split"


def test_pairs_list_only_licensed_footage_and_cut_cleanly():
    licensed = ("data/candidates/pexels_westarmoney_s550", "data/raw/pexels", "data/candidates/nhtsa_crash_test_db/video", "data/raw/nhtsa/video")
    for pair in pseudo_pair.PAIRS.values():
        for cut in (pair.pickup, pair.return_):
            assert set(cut.source.folders) <= set(licensed) and re.fullmatch(r"[0-9a-f]{64}", cut.source.sha256)
            assert cut.out is None or cut.out.startswith("data/derived/")
        if pair.kind == "time_split":
            assert pair.pickup.t1 == pair.return_.t0 and pair.pickup.source == pair.return_.source
    cmd = pseudo_pair.cut_command(Path("src.mp4"), pseudo_pair.PAIRS["nhtsa15759"].pickup, Path("out.mp4"))
    assert cmd[cmd.index("-ss") + 1] == "32.000000" and cmd[cmd.index("-t") + 1] == "20.000000"
    for flag in ("-map_metadata", "+bitexact", "-threads", "-an"):
        assert flag in cmd
    assert pseudo_pair.run_command(pseudo_pair.PAIRS["nhtsa15759"]) == "LIVE=1 .venv/bin/walkaround run data/derived/t15759_pre.mp4 data/derived/t15759_post.mp4 --run-id eval_nhtsa15759"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_cut_clip_cuts_once_and_checks_the_source(tmp_path, monkeypatch):
    clip = REPO / "data" / "test" / "pingpong.mp4"  # tests/conftest.py makes a stand-in when it is missing
    (tmp_path / "data" / "test").mkdir(parents=True)
    shutil.copyfile(clip, tmp_path / "data" / "test" / "pingpong.mp4")
    monkeypatch.setattr(pseudo_pair, "REPO", tmp_path)
    source = pseudo_pair.Source("pingpong.mp4", ("data/test",), pseudo_pair.sha256_file(clip), "test clip")
    cut = pseudo_pair.Cut(source, 1.0, 2.0, "data/derived/pp_1_2.mp4")
    manifest = pseudo_pair.cut_clip(cut)
    out = tmp_path / "data" / "derived" / "pp_1_2.mp4"
    assert manifest["out_sha256"] == pseudo_pair.sha256_file(out) and 23 <= manifest["frames"] <= 25
    stamp = out.stat().st_mtime_ns
    assert pseudo_pair.cut_clip(cut) == manifest and out.stat().st_mtime_ns == stamp
    wrong = pseudo_pair.Cut(pseudo_pair.Source("pingpong.mp4", ("data/test",), "0" * 64, "x"), 0.0, 1.0, "data/derived/pp_0_1.mp4")
    with pytest.raises(ValueError, match="hashes to"):
        pseudo_pair.cut_clip(wrong)


# The starting labels -----------------------------------------------------------------------------------------------------


def _label_files() -> list[Path]:
    return sorted(LABELS.glob("*.json"))


def test_starting_labels_follow_the_schema_and_say_who_labeled_them():
    files = _label_files()
    assert {p.stem for p in files} >= {"uco3d_pairC", "nhtsa15759", "px4228872_cut", "px4228872_front", "px_two_takes"}
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert text.isascii(), path
        doc = eval_run.load_labels(path)
        assert doc["pair_id"] == path.stem
        assert doc["status"] == eval_run.DEVELOPER_STATUS
        for role in ("pickup", "return"):
            assert "eval_internal" not in doc["footage"][role]["path"] and "eval_internal" not in str(doc["footage"][role].get("source"))


def test_starting_labels_say_what_changed():
    docs = {p.stem: eval_run.load_labels(p) for p in _label_files()}
    nhtsa = docs["nhtsa15759"]
    assert nhtsa["nothing_changed"] is False
    assert {pid for pid, p in nhtsa["panels"].items() if p.get("changed") is True} == {"left_front_door", "left_rear_door"}
    for name in ("uco3d_pairC", "px4228872_cut", "px4228872_front", "px_two_takes"):
        assert docs[name]["nothing_changed"] is True
        assert not any(p.get("changed") is True for p in docs[name]["panels"].values())
    for name in ("px4228872_cut", "px4228872_front", "px_two_takes"):
        assert docs[name]["kind"] == "pseudo" and len(docs[name]["panels"]) == 22
    refs = docs["uco3d_pairC"]["reference_marks"]
    assert {r["group"] for r in refs if r["real"]} == {"trunk_chips_upper", "trunk_chips_left", "rfd_chip"}
    assert all(r["new"] is False for r in refs if r["real"])
    for group in ("trunk_chips_upper", "trunk_chips_left", "rfd_chip"):  # each real mark is placed in both walks
        assert {r["role"] for r in refs if r["group"] == group} == {"pickup", "return"}


def test_starting_labels_point_at_the_files_they_were_made_from():
    """The footage hashes match the files on disk, where they are (data/ is not in the repo)."""
    checked = 0
    for path in _label_files():
        doc = eval_run.load_labels(path)
        for role in ("pickup", "return"):
            f = doc["footage"][role]
            local = REPO / eval_run.resolve_walk(f["path"])
            if local.is_file() and f.get("sha256"):
                assert pseudo_pair.sha256_file(local) == f["sha256"], f"{path.name} {role}"
                checked += 1
    if not checked:
        pytest.skip("no labeled footage on this machine")


# The labeling page -------------------------------------------------------------------------------------------------------


def _html() -> str:
    return LABEL_HTML.read_text(encoding="utf-8")


def test_label_page_is_plain_ascii_and_reads_only_served_files():
    html = _html()
    assert html.isascii()
    assert "innerHTML" not in html and "eval(" not in html
    urls = set(re.findall(r"https?://[^\s'\"<>)]+", html))
    assert urls <= {"http://www.w3.org/2000/svg"}, urls
    rels = set(re.findall(r"fileUrl\('([^']+)'\)", html))
    assert rels == {"derived/keyframes.json", "derived/marks.json", "derived/coverage.json", "derived/matches.json", "videos/pickup.json", "videos/return.json"}
    run = RunManifest(run_id="r", created_at="2026-10-31T12:00:00Z")
    assert "'media/' + role + '.proxy_blurred.mp4'" in html
    for rel in rels | {"crops/pickup/pickup_trunk_lid_kf1.jpg", "keyframes/return/return_hood_kf2.frame.jpg", "media/return.proxy_blurred.mp4"}:
        assert privacy.served_file_allowed(run, rel, run_owner=True), rel
    app = (REPO / "walkaround" / "app.py").read_text(encoding="utf-8")
    for route in ('"/api/runs"', '"/api/runs/{run_id}"', '"/api/runs/{run_id}/files/{rel_path:path}"'):
        assert route in app


def _label_core() -> str:
    match = re.search(r'<script id="label-core">(.*?)</script>', _html(), re.S)
    assert match is not None
    return match.group(1)


NODE_DRIVER = r"""
const fs = require('fs');
const vm = require('vm');
const ctx = {};
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), ctx);
const C = ctx.LabelCore;
const input = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const s = C.newState(input.runId, input.pairId, input.labeler);
Object.assign(s, input.state || {});
s.labeledOn = '2026-10-31';
const notes = input.file ? C.mergeLabels(s, input.file) : [];
Object.assign(s.judgments, input.judgments || {});
(input.hand || []).forEach(function (h) { s.hand.push(h); });
// the labeler's ticks, as the page allows them (the box is disabled until canCheck), then marks put back to not judged
const ticked = (input.ticks || []).map(function (t) {
  const ok = C.canCheck(s, t[0], t[1]);
  if (ok) s.panelLabels[t[0]].marks_checked[t[1]] = true;
  return ok;
});
(input.unjudge || []).forEach(function (id) { s.judgments[id] = Object.assign({}, s.judgments[id], { real: null }); });
C.fixChecks(s);
const labels = C.buildLabels(s);
process.stdout.write(JSON.stringify({ text: C.toJSON(labels), notes: notes, problems: C.problems(labels), foreign: s.foreign, ticked: ticked,
  cycle: [C.cycleReal({ real: null }).real, C.cycleReal({ real: true }).real, C.cycleReal({ real: false }).real],
  point: C.cropPoint(150, 75, { left: 100, top: 50, width: 200, height: 100 }, 400, 200) }));
"""


def _run_core(tmp_path: Path, payload: dict) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("needs node")
    (tmp_path / "core.js").write_text(_label_core(), encoding="utf-8")
    (tmp_path / "driver.js").write_text(NODE_DRIVER, encoding="utf-8")
    (tmp_path / "input.json").write_text(json.dumps(payload), encoding="utf-8")
    proc = subprocess.run([node, str(tmp_path / "driver.js"), str(tmp_path / "core.js"), str(tmp_path / "input.json")], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_label_page_core_keeps_a_starting_file_and_passes_the_eval_schema(tmp_path):
    start = json.loads((LABELS / "uco3d_pairC.json").read_text(encoding="utf-8"))
    got = _run_core(tmp_path, {"runId": "eval_uco3d_pairC", "pairId": "", "labeler": "Sanat", "file": start})
    assert got["problems"] == [] and got["cycle"] == [True, False, None] and got["point"] == [100.0, 50.0]
    assert got["text"].isascii()
    labels = json.loads(got["text"])
    assert eval_run.label_problems(labels) == []
    assert labels["pair_id"] == "uco3d_pairC" and labels["run_id"] == "eval_uco3d_pairC"
    assert labels["panels"] == start["panels"] and labels["reference_marks"] == start["reference_marks"] and labels["footage"] == start["footage"]
    assert (labels["kind"], labels["table"], labels["split"], labels["nothing_changed"]) == ("cross_walk", "B", "report", True)
    assert labels["status"].startswith("labeled by Sanat on the labeling page, not yet reviewed (started from a file labeled by the developer")


def _run_state(run_dir: Path) -> dict:
    state = {name: json.loads((run_dir / "derived" / f"{name}.json").read_text(encoding="ascii")) for name in ("marks", "keyframes")}
    state["videos"] = {role: json.loads((run_dir / "videos" / f"{role}.json").read_text(encoding="ascii")) for role in ("pickup", "return")}
    return state


def test_label_page_core_output_scores_like_the_hand_written_labels(env):
    run_dir = write_run(env / "runs", "eval_synth")
    reference = labels_doc()
    got = _run_core(env, {"runId": "eval_synth", "pairId": "synth", "labeler": "", "file": reference, "state": _run_state(run_dir)})
    labels = json.loads(got["text"])
    assert eval_run.label_problems(labels) == [] and got["foreign"] is None
    checked = {pid: p["marks_checked"] for pid, p in labels["panels"].items() if "marks_checked" in p}
    assert checked == {TRUNK: {"pickup": True, "return": True}, DOOR: {"pickup": True, "return": True}}  # every mark has a call
    assert {m["mark_id"]: m["new"] for m in labels["marks"] if m["role"] == "return"} == {R1: None, R2: None, R3: None, R4: True, R5: False}
    (env / "labels" / "synth.json").write_text(got["text"], encoding="ascii")
    pair = evaluate(env)["pairs"]["synth"]
    assert pair["marks"]["precision"]["text"] == "6 of 7" and pair["marks"]["recall"]["text"] == "6 of 7"
    assert pair["verdicts"]["false_new"]["text"] == "0 of 3"


def test_label_page_core_ticks_a_walk_only_when_every_mark_has_a_call(env):
    run_dir = write_run(env / "runs", "eval_synth")
    judged = {"real": True, "new": None, "not_real_reason": None, "note": ""}
    payload = {"runId": "eval_synth", "pairId": "synth", "labeler": "", "state": _run_state(run_dir), "judgments": {P1: judged}}
    got = _run_core(env, {**payload, "ticks": [[TRUNK, "pickup"], [TRUNK, "return"], [HOOD, "return"]]})
    assert got["ticked"] == [False, False, False]  # P2 has no call; no return mark has one; no return crop of the hood
    assert TRUNK not in json.loads(got["text"])["panels"]
    both = {**payload, "judgments": {P1: judged, P2: judged}}
    got = _run_core(env, {**both, "ticks": [[TRUNK, "pickup"], [HOOD, "pickup"]]})
    assert got["ticked"] == [True, True]  # nothing auto-ticks: only these two were asked for
    assert json.loads(got["text"])["panels"][TRUNK]["marks_checked"] == {"pickup": True, "return": False}
    got = _run_core(env, {**both, "ticks": [[TRUNK, "pickup"]], "unjudge": [P1]})
    assert TRUNK not in json.loads(got["text"])["panels"]  # a mark back to not judged takes the tick away


def test_label_page_core_writes_three_state_new_and_the_labelers_size(env):
    run_dir = write_run(env / "runs", "eval_synth")
    judgments = {
        R1: {"real": True, "new": False, "not_real_reason": None, "note": "", "size_class": "M"},  # the system said S
        R2: {"real": True, "new": True, "not_real_reason": None, "note": "", "size_class": "M"},  # the system said M too
        R3: {"real": False, "new": True, "not_real_reason": "dirt", "note": ""},
    }
    hand = [{"mark_id": "hand_1", "role": "return", "panel_id": TRUNK, "kf_id": kf_id("return", TRUNK), "point_crop": [10.0, 20.0], "type": "chip", "size_class": "S", "new": False}]
    got = _run_core(env, {"runId": "eval_synth", "pairId": "synth", "labeler": "", "state": _run_state(run_dir), "judgments": judgments, "hand": hand})
    marks = {m["mark_id"]: m for m in json.loads(got["text"])["marks"]}
    assert (marks[R1]["new"], marks[R1]["size_class"]) == (False, "M")
    assert marks[R2]["new"] is True and "size_class" not in marks[R2]
    assert marks[R3]["new"] is None and marks[R3]["not_real_reason"] == "dirt" and "size_class" not in marks[R3]
    assert marks["hand_1"]["new"] is False
    labels = json.loads(got["text"])
    assert eval_run.label_problems(labels) == [] and labels["footage"]["pickup"]["path"] == "pickup.mp4"
    bare = _run_core(env, {"runId": "eval_synth", "pairId": "synth", "labeler": "", "state": {**_run_state(run_dir), "videos": {}}})
    assert bare["problems"] == ["footage for the pickup walk is missing (load the pair's labels file)", "footage for the return walk is missing (load the pair's labels file)"]


def test_label_page_core_leaves_out_marks_from_another_run_and_says_so(tmp_path):
    other = labels_doc()
    got = _run_core(tmp_path, {"runId": "a_different_run", "pairId": "synth", "labeler": "", "file": other})
    assert got["notes"] == ["8 mark labels were made on run eval_synth; they are not loaded, and a download for this run leaves them out, so keep that file"]
    assert got["foreign"] == {"run_id": "eval_synth", "count": 8}  # the download button warns once before saving
    labels = json.loads(got["text"])
    assert labels["marks"] == [] and labels["panels"][TRUNK]["changed"] is False
    assert "marks_checked" not in labels["panels"][TRUNK]  # no crops of this run loaded, so nothing stays checked


def test_label_page_in_a_browser_ticks_refocuses_and_warns_before_dropping_marks(env):
    """The page in Chromium against the synthetic run, served by a stand-in for the three GET routes it reads."""
    sync_api = pytest.importorskip("playwright.sync_api")
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import unquote, urlparse

    run_dir = write_run(env / "runs", "eval_synth")
    html = LABEL_HTML.read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):  # noqa: N802
            path = unquote(urlparse(self.path).path)
            prefix = "/api/runs/eval_synth/files/"
            body, ctype = b"{}", "application/json"
            if path == "/eval/label.html":
                body, ctype = html, "text/html"
            elif path == "/api/runs":
                body = b"[]"
            elif path == "/api/runs/eval_synth":
                body = (run_dir / "run.json").read_bytes()
            elif path.startswith(prefix) and (run_dir / path[len(prefix):]).is_file():
                body = (run_dir / path[len(prefix):]).read_bytes()
                ctype = {".jpg": "image/jpeg", ".png": "image/png"}.get(Path(path).suffix, "application/json")
            else:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_api.sync_playwright() as pw:
            try:
                browser = pw.chromium.launch()
            except Exception as exc:  # playwright without its browser
                pytest.skip(f"Chromium not available: {exc}")
            page = browser.new_page(accept_downloads=True)
            page.goto(f"http://127.0.0.1:{server.server_port}/eval/label.html?run=eval_synth&pair=synth")
            page.wait_for_selector("#panel-trunk_lid")
            box = page.locator("#panel-trunk_lid input[data-focus='checked:trunk_lid:pickup']")
            assert box.is_disabled() and not box.is_checked()
            page.locator(f"button[data-focus='real:{P1}']").focus()
            page.keyboard.press("Enter")
            assert page.evaluate("document.activeElement.dataset.focus") == f"real:{P1}"  # focus survives the rebuild
            page.locator(f"button[data-focus='real:{P2}']").click()
            assert box.is_enabled() and not box.is_checked()  # every mark has a call, but only the labeler ticks
            box.check()
            for _ in range(2):  # real -> not real -> not judged
                page.locator(f"#panel-trunk_lid rect[data-mark='{P1}']").click()
            assert box.is_disabled() and not box.is_checked()
            stage = page.locator(f"[data-focus='stage:{kf_id('pickup', TRUNK)}']")
            stage.focus()
            page.keyboard.press("Enter")
            dot = page.locator("#panel-trunk_lid circle.hand")
            x0 = float(dot.get_attribute("cx"))
            page.keyboard.press("ArrowRight")
            assert float(dot.get_attribute("cx")) == pytest.approx(x0 + 4.0)  # 1% of the 400 px crop
            other = env / "synth.json"
            other.write_text(json.dumps(labels_doc(run_id="another_run")), encoding="ascii")
            page.set_input_files("#labels-file", str(other))
            page.wait_for_function("document.getElementById('status').textContent.indexOf('Loaded synth.json') === 0")
            page.click("#download")
            assert page.text_content("#status").startswith("Not saved yet: this download is for run eval_synth and leaves out the 8 mark labels")
            with page.expect_download() as download:
                page.click("#download")
            saved = json.loads(Path(download.value.path()).read_text(encoding="ascii"))
            assert saved["run_id"] == "eval_synth" and [m["mark_id"] for m in saved["marks"]] == [P2, "hand_1"]
            assert eval_run.label_problems(saved) == []
            browser.close()
    finally:
        server.shutdown()
