"""The D1 gate scripts (lane M19): run_d1's dry run on the probe fixtures, the memo with every section 11 criterion,
the budget ceiling, and the live paths of the gates against fakes that write lanes the way sam_client does (a slug
lane id and a gz path relative to the run directory). No test makes a model or network call."""

from __future__ import annotations

import asyncio
import gzip
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
for entry in (str(REPO), str(SCRIPTS)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import gate_sam  # noqa: E402
import gate_spark  # noqa: E402
import gate_stays_put  # noqa: E402
import gate_voice  # noqa: E402
import run_d1  # noqa: E402
from walkaround import config, contracts, ledger, paths, sam_client, sam_wire, spark_client, voice_client  # noqa: E402
from walkaround.contracts import IndexFrame, IndexObject, LaneRecord, LaneStatus, Noun  # noqa: E402

FIX_TEXT = REPO / "tests" / "fixtures" / "sam" / "pingpong_text.events.ndjson"
FIX_BOX = REPO / "tests" / "fixtures" / "sam" / "pingpong_box.events.ndjson"
CLIP = REPO / "data" / "test" / "pingpong.mp4"
SPARK_FIXTURE = REPO / "tests" / "fixtures" / "d1" / "spark_fixture.json"
TH = contracts.load_thresholds()


# ------------------------------------------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------------------------------------------


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """runs/ and the ledger inside tmp_path; the real ledger's bytes are checked untouched at the end."""
    real = Path(ledger.LEDGER_PATH)
    before = real.read_bytes() if real.exists() else None
    root = tmp_path / "runs"
    monkeypatch.setattr(paths, "RUNS_DIR", root)
    monkeypatch.setattr(config, "RUNS_DIR", root)
    monkeypatch.setattr(ledger, "LEDGER_PATH", root / "ledger.jsonl")
    monkeypatch.delenv("LIVE", raising=False)
    yield root
    after = real.read_bytes() if real.exists() else None
    assert after == before, "a test wrote to the real runs/ledger.jsonl"


@pytest.fixture(scope="module")
def dry_run(tmp_path_factory):
    """One full dry run shared by the tests that read it (about 10 s)."""
    real = Path(ledger.LEDGER_PATH)
    before = real.read_bytes() if real.exists() else None
    out = tmp_path_factory.mktemp("d1") / "dry"
    code = run_d1.main(["--dry-run", "--out", str(out)])
    after = real.read_bytes() if real.exists() else None
    assert after == before, "the dry run wrote to the real ledger"
    return code, out


# ------------------------------------------------------------------------------------------------------------------
# run_d1: the dry run and the memo
# ------------------------------------------------------------------------------------------------------------------


def test_dry_run_completes_every_step(dry_run):
    code, out = dry_run
    assert code == 0
    results = json.loads((out / "results.json").read_text())
    assert results["mode"] == "dry_run"
    assert [k for k in run_d1.STEP_KEYS if results["steps"][k]["status"] != "done"] == []
    for step, name in run_d1.GATE_JSON.items():
        assert (out / step / name).exists(), step
    assert results["recommendation"]["headline"].startswith("DRY RUN, not a decision.")


def test_memo_renders_every_criterion(dry_run):
    _, out = dry_run
    memo = (out / "D1_MEMO.md").read_text(encoding="ascii")
    results = json.loads((out / "results.json").read_text())
    assert [c["id"] for c in results["criteria"]] == list(run_d1.CRITERION_IDS)
    missing = [cid for cid in run_d1.CRITERION_IDS if f"| {cid} |" not in memo]
    assert missing == []
    for row in run_d1.ROWS:
        assert f"### {row} " in memo, row
    assert {c["status"] for c in results["criteria"]} <= {"pass", "fail", "not run", "info"}
    assert all(c["status"] == "not run" and c["note"] for c in results["criteria"] if c["row"] in ("E14", "E16", "E19"))
    assert memo.startswith("# D1 decision memo (DRY RUN)\n\n**DRY RUN.**")
    for word in ("Recommendation", "Decisions", "Every criterion, by row", "Steps and spend"):
        assert f"## {word}" in memo
    for item in ("fps", "noun list", "concurrency policy", "SAM engine", "decode path", "the ruler", "pickup-check wording", "S1, S2 and S2b for Mon 11/2"):
        assert f"| {item} |" in memo, item


def test_memo_is_plain_ascii_without_dashes(dry_run):
    _, out = dry_run
    raw = (out / "D1_MEMO.md").read_bytes()
    raw.decode("ascii")
    template = run_d1.TEMPLATE.read_bytes()
    template.decode("ascii")
    assert b"--" not in template.replace(b"|---", b"").replace(b"---|", b"")


def test_dry_run_reads_the_fixtures(dry_run):
    _, out = dry_run
    results = json.loads((out / "results.json").read_text())
    crit = {c["id"]: c for c in results["criteria"]}
    assert crit["E2.oracle"]["status"] == "pass"  # the wire reader agrees with meta_sam_parser on the fixture
    assert crit["E2.overlay"]["status"] == "pass"
    assert crit["E3.c_semantics"]["value"].startswith("c varies per frame on 0 of 16")  # 10/26: constant on text lanes
    assert crit["E6.truncation_signal"]["status"] == "pass"
    assert crit["E12.store_false_sam"]["status"] == "fail"  # the probe sent no store field; its echo was true
    assert crit["E13.pair_dollars"]["status"] in ("pass", "fail")
    assert results["auto"]["crops"], "the dry run exercises the crop picks from gate_sam's suggestions"
    gate = json.loads((out / "spark" / "gate_spark.json").read_text())
    assert any(c["name"].startswith("auto_") for c in gate["crops"])
    assert results["deleted_unblurred"], "the E0 proxies are deleted at the end"
    assert not list((out / "E0" / "proxies").glob("*.mp4"))
    auto_plan = json.loads((out / "staysput" / "auto_plan.json").read_text())
    assert auto_plan and auto_plan[0]["label"] == "auto_mark"  # built from gate_spark's repeated marks, not run in a dry run


def test_dry_run_dollars_stay_in_the_dry_ledgers(dry_run):
    _, out = dry_run
    results = json.loads((out / "results.json").read_text())
    rows = [json.loads(line) for p in out.rglob("ledger*.jsonl") for line in p.read_text().splitlines() if line.strip()]
    assert rows and all(str(r["run_id"]).startswith(results["run_prefix"]) for r in rows)
    assert results["spend_usd"] == pytest.approx(sum(r["usd"] for r in rows), abs=1e-6)


def test_render_only_rebuilds_the_memo(dry_run, tmp_path):
    _, out = dry_run
    copy = tmp_path / "copy"
    import shutil

    shutil.copytree(out, copy)
    (copy / "D1_MEMO.md").unlink()
    data = json.loads((copy / "results.json").read_text())
    data["out"] = str(copy)
    (copy / "results.json").write_text(json.dumps(data))
    assert run_d1.main(["--render-only", str(copy)]) == 0
    assert (copy / "D1_MEMO.md").read_text() == (out / "D1_MEMO.md").read_text().replace(str(out), str(copy))


def test_budget_zero_skips_every_paid_step(tmp_path):
    out = tmp_path / "zero"
    assert run_d1.main(["--dry-run", "--budget", "0", "--out", str(out), "--steps", "E0", "sam", "spark", "staysput", "voice", "E13"]) == 0
    results = json.loads((out / "results.json").read_text())
    for step in ("sam", "spark", "staysput", "voice"):
        assert results["steps"][step]["status"] == "skipped", step
        assert results["steps"][step]["reason"].startswith("budget:")
    crit = {c["id"]: c for c in results["criteria"]}
    assert crit["E2.tracks"]["status"] == "not run" and "budget" in crit["E2.tracks"]["note"]
    assert "No engine decision yet" in results["recommendation"]["headline"]
    memo = (out / "D1_MEMO.md").read_text()
    assert all(f"| {cid} |" in memo for cid in run_d1.CRITERION_IDS)


def test_budget_counts_rows_with_the_prefix_only(tmp_path):
    out = tmp_path / "b"
    out.mkdir()
    ledger_file = out / "dryrun_runs" / "ledger.jsonl"
    ledger_file.parent.mkdir()
    ledger_file.write_text(json.dumps({"run_id": "d1dry_x_E2_15fps", "usd": 1.5}) + "\n" + json.dumps({"run_id": "other", "usd": 9.0}) + "\n")
    ctx = SimpleNamespace(dry=True, out=out, prefix="d1dry_x")
    assert run_d1.spend(ctx) == pytest.approx(1.5)


def test_without_live_every_script_prints_its_plan_and_exits_2(isolated, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(gate_sam, "FORCE_LIVE", False)
    assert run_d1.main(["--pickup", str(CLIP), "--return", str(CLIP)]) == 2
    assert "run_d1 would run" in capsys.readouterr().out
    assert gate_sam.main(["--pickup", str(CLIP), "--out", str(tmp_path / "s")]) == 2
    assert gate_spark.main([str(CLIP), "--t", "1.0", "--out", str(tmp_path / "p")]) == 2
    assert gate_stays_put.main(["--src", str(CLIP), "--pts", "2.0", "--mark", "10", "10", "50", "50", "--out", str(tmp_path / "t")]) == 2
    assert gate_voice.main([str(CLIP), "--out", str(tmp_path / "v")]) == 2
    printed = capsys.readouterr().out
    for name in ("gate_sam would send", "gate_spark would send", "gate_stays_put would send", "gate_voice would send"):
        assert name in printed
    assert not ledger.rows(ledger.LEDGER_PATH)


def test_run_d1_takes_any_readable_walk_and_names_a_bad_one(isolated, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(gate_sam, "FORCE_LIVE", False)
    elsewhere = tmp_path / "phone export" / "Pickup Walk.MOV"
    elsewhere.parent.mkdir()
    elsewhere.write_bytes(CLIP.read_bytes())
    assert run_d1.main(["--pickup", str(elsewhere), "--return", str(CLIP)]) == 2  # the plan prints; no LIVE
    assert "Pickup Walk.MOV" in capsys.readouterr().out
    assert run_d1.main(["--pickup", str(tmp_path / "missing.mp4")]) == 2
    assert "pickup: no such file" in capsys.readouterr().err
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    assert run_d1.main(["--pickup", str(CLIP), "--return", str(broken)]) == 2
    assert "return: ffprobe cannot read" in capsys.readouterr().err
    assert not ledger.rows(ledger.LEDGER_PATH)


def test_plan_template_prints_and_loads(tmp_path, capsys):
    assert run_d1.main(["--plan-template"]) == 0
    plan = json.loads(capsys.readouterr().out)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    loaded = run_d1.load_plan(str(path))
    assert loaded["pickup"] == str(tmp_path / "pickup_take1.mov")
    assert loaded["crops"][0]["source"] == str(tmp_path / "pickup_take1.mov")
    assert loaded["staysput"][2]["control"] is True
    plan["staysput"][0]["role"] = "take2"
    path.write_text(json.dumps(plan))
    with pytest.raises(SystemExit):
        run_d1.load_plan(str(path))


def test_out_folder_resumes_with_its_prefix(tmp_path):
    out = tmp_path / "resume"
    assert run_d1.main(["--dry-run", "--out", str(out), "--steps", "E0"]) == 0
    first = json.loads((out / "results.json").read_text())
    assert run_d1.main(["--dry-run", "--out", str(out), "--steps", "E1"]) == 0
    second = json.loads((out / "results.json").read_text())
    assert second["run_prefix"] == first["run_prefix"] and {"E0", "E1"} <= set(second["steps"])
    with pytest.raises(SystemExit):
        run_d1.make_ctx(run_d1.build_parser().parse_args(["--out", str(out)]), TH)  # a dry folder never takes a live run


# ------------------------------------------------------------------------------------------------------------------
# run_d1: the live arguments it hands each gate, parsed by that gate's own parser
# ------------------------------------------------------------------------------------------------------------------


def live_ctx(tmp_path: Path, plan: dict | None = None) -> run_d1.Ctx:
    args = run_d1.build_parser().parse_args(["--pickup", str(CLIP)])
    return run_d1.Ctx(args=args, th=TH, out=tmp_path, prefix="d1_test", dry=False, budget=25.0, plan=plan or {}, inputs={"pickup": CLIP, "return": CLIP, "take2": CLIP, "narration": None}, results={"steps": {}})


def synthetic_sam(tmp_path: Path) -> dict:
    def lane(noun: str) -> dict:
        return {"noun": noun, "lane_file": str(tmp_path / f"{noun}.ndjson.gz"), "media_size": [960, 540]}

    def role(door_t: float) -> dict:
        return {
            "lanes": {f"D1 E2 {n}": lane(n) for n in ("car door", "bumper", "hood")},
            "suggestions": {
                "car door": [{"frame": 30, "t": door_t, "object_id": "3", "c": 0.9, "box_proxy": [200, 100, 700, 450], "box_src": [266.7, 133.3, 933.3, 600.0]}, {"frame": 60, "t": 2.5, "object_id": "5", "c": 0.9, "box_proxy": [100, 100, 400, 450], "box_src": [133.3, 133.3, 533.3, 600.0]}],
                "bumper": [{"frame": 12, "t": 0.5, "object_id": "1", "c": 0.8, "box_proxy": [50, 300, 900, 500], "box_src": [66.7, 400.0, 1200.0, 666.7]}],
                "hood": [{"frame": 40, "t": 1.7, "object_id": "2", "c": 0.8, "box_proxy": [0, 0, 960, 540], "box_src": [0.0, 0.0, 1280.0, 720.0]}],
                "license plate": [{"frame": 20, "t": 0.8, "object_id": "4", "c": 0.9, "box_proxy": [400, 380, 520, 420], "box_src": [533.3, 506.7, 693.3, 560.0]}],
            },
            "handle": {"sigma": {"sigma_frame": 0.021}},
        }

    return {"sweeps": [{"fps": 15.0, "roles": {"pickup": role(1.25), "return": role(1.5)}, "evaluation": {"tracks": {"pass": True}}}]}


def test_live_arguments_parse_with_each_gate(tmp_path):
    ctx = live_ctx(tmp_path)
    sam_argv = run_d1.sam_args(ctx)
    parsed = gate_sam.build_parser().parse_args(sam_argv + ["--out", str(tmp_path / "s")])
    assert parsed.fps == ["15", "10"] and parsed.fps_fallback == "30" and parsed.handle and parsed.bridge and parsed.timeline and parsed.take2 == str(CLIP)
    assert parsed.run_prefix == "d1_test" and parsed.budget == 25.0
    sam = synthetic_sam(tmp_path)
    spark_argv, picked = run_d1.spark_args(ctx, sam)
    sp = gate_spark.build_parser().parse_args(spark_argv + ["--out", str(tmp_path / "p")])
    assert sp.efforts == ["low", "medium"] and sp.n == 5 and sp.count and sp.truncate == 600 and sp.auto_looks and sp.claims_from_crops
    assert sp.ruler_frame == str(CLIP) and sp.ruler_t == 0.8
    crops = json.loads(Path(sp.crops).read_text())
    assert [c["name"] for c in crops] == ["auto_door", "auto_bumper", "auto_door_return"]
    assert crops[0]["crop"][0] < 266 and crops[0]["crop"][2] > 933 and crops[0]["role"] == "pickup" and crops[0]["object_id"] == "3"
    spark = {"crop_specs": crops, "crops": [{"name": "auto_door", "crop_size": [800, 560], "efforts": {"low": {"runs": [
        {"marks": [{"px": [300, 200], "bbox_px": [290, 190, 310, 210], "type": "chip", "confidence": "high", "capped": False}, {"px": [600, 400], "bbox_px": [590, 395, 610, 405], "type": "dirt", "confidence": "low", "capped": True}]},
        {"marks": [{"px": [304, 203], "bbox_px": [292, 192, 312, 212], "type": "chip", "confidence": "high", "capped": False}]},
        {"marks": [{"px": [299, 198], "bbox_px": [289, 189, 309, 209], "type": "chip", "confidence": "medium", "capped": False}]},
    ]}}}]}
    sp_argv, picked = run_d1.staysput_args(ctx, sam, spark)
    st = gate_stays_put.build_parser().parse_args(sp_argv + ["--out", str(tmp_path / "t")])
    entries = json.loads(Path(st.plan).read_text())
    mark = entries[0]
    ox, oy = crops[0]["crop"][0], crops[0]["crop"][1]
    assert mark["label"] == "auto_mark" and mark["mark"] == [290 + ox, 190 + oy, 310 + ox, 210 + oy]
    assert mark["panel_object_id"] == "3" and mark["sigma_frame"] == 0.021 and mark["scuff_text"] is True
    assert all(e["role"] in ("pickup", "return") for e in entries)
    voice_argv = run_d1.voice_args(ctx)
    v = gate_voice.build_parser().parse_args(voice_argv + ["--out", str(tmp_path / "v")])
    assert v.source == str(CLIP) and v.modes == ["ENDPOINTING", "DIARIZATION"]


def test_plan_crops_and_pairs_go_through_untouched(tmp_path):
    plan = {"crops": [{"name": "mark_door", "source": str(CLIP), "t": 1.0, "crop": [0, 0, 100, 100], "panel": "left_front_door", "clean": False, "targets": [[10, 10]]}],
            "second_look": ["a.jpg", "b.jpg"], "second_look_control": ["a.jpg", "c.jpg"], "claims": ["p1.jpg"], "tire_code": "225/60R16", "door_pair": ["w.jpg", "c.jpg"], "ruler_frame": {"source": str(CLIP), "t": 2.0}}
    argv, picked = run_d1.spark_args(live_ctx(tmp_path, plan), None)
    sp = gate_spark.build_parser().parse_args(argv + ["--out", str(tmp_path / "p")])
    assert sp.second_look == ["a.jpg", "b.jpg"] and sp.second_look_control == ["a.jpg", "c.jpg"] and not sp.auto_looks
    assert sp.claim == ["p1.jpg"] and not sp.claims_from_crops and sp.tire_code == "225/60R16" and sp.door_pair == ["w.jpg", "c.jpg"]
    assert json.loads(Path(sp.crops).read_text()) == plan["crops"] and "crops" not in picked


# ------------------------------------------------------------------------------------------------------------------
# run_d1: the decision logic on synthetic results
# ------------------------------------------------------------------------------------------------------------------


def sweep(fps: float, tracks: bool) -> dict:
    return {"fps": fps, "roles": {}, "evaluation": {"tracks": {"pass": tracks}}}


def test_chosen_sweep_follows_section_16_item_2():
    assert run_d1.chosen_sweep({"sweeps": [sweep(15, True), sweep(10, True)]}, TH)[1] == 15.0
    assert run_d1.chosen_sweep({"sweeps": [sweep(15, False), sweep(10, False), sweep(30, True)]}, TH)[1] == 30.0
    assert run_d1.chosen_sweep({"sweeps": [sweep(15, False), sweep(10, True)]}, TH)[1] == 10.0
    chosen, fps = run_d1.chosen_sweep({"sweeps": [sweep(15, False), sweep(10, False), {"fps": 30, "skipped": "budget"}]}, TH)
    assert fps is None and chosen["fps"] == 15
    assert run_d1.chosen_sweep(None, TH) == (None, None)


def criteria(**statuses: str) -> list[dict]:
    rows = []
    for cid, row, rule, on_fail in run_d1.CRITERIA:
        rows.append({"id": cid, "row": row, "rule": rule, "on_fail": on_fail, "status": statuses.get(cid.replace(".", "_"), "pass"), "value": "", "note": ""})
    return rows


def test_recommend_hosted_when_tracks_pass():
    rec = run_d1.recommend(SimpleNamespace(fps=15.0, dry=False), criteria())
    assert rec["headline"] == "Use hosted SAM 3.1 (meta-model-api) at 15 fps"
    assert any("section 14 item 33" in line for line in rec["lines"])


def test_recommend_anchor_pass_then_fallback():
    rec = run_d1.recommend(SimpleNamespace(fps=None, dry=False), criteria(E2_tracks="fail"))
    assert rec["headline"].startswith("Use hosted SAM 3.1 with the anchor pass on every panel")
    rec = run_d1.recommend(SimpleNamespace(fps=None, dry=False), criteria(E2_tracks="fail", E4_joins="fail"))
    assert rec["headline"].startswith("Fall back: SAM_ENGINE=fal")
    assert any("open-weight SAM 3 on a GPU" in line for line in rec["lines"])


def test_recommend_modifiers_and_no_decision():
    rec = run_d1.recommend(SimpleNamespace(fps=15.0, dry=False), criteria(E2_concurrency="fail", E2_decode="fail", E12_store_false_spark="fail"))
    text = " ".join(rec["lines"])
    assert "priority waves of 6" in text and "Node sidecar" in text and "per-request deletion path" in text
    rows = criteria(E2_tracks="not run")
    rows[[r["id"] for r in rows].index("E2.tracks")]["note"] = "step skipped: budget"
    rec = run_d1.recommend(SimpleNamespace(fps=None, dry=True), rows)
    assert rec["headline"] == "DRY RUN, not a decision. No engine decision yet: E2 did not run"


def test_row_status():
    assert run_d1.row_status([{"status": "pass"}, {"status": "info"}]) == "pass"
    assert run_d1.row_status([{"status": "pass"}, {"status": "fail"}]) == "fail"
    assert run_d1.row_status([{"status": "info"}, {"status": "not run"}]) == "partial"
    assert run_d1.row_status([{"status": "info"}]) == "measured"
    assert run_d1.row_status([{"status": "not run"}]) == "not run"


def test_cells_are_ascii_and_pipe_free():
    assert run_d1.cell("a | b\nc " + chr(0x2014) + " d " + chr(0x201C) + "q" + chr(0x201D)) == 'a / b c - d "q"'
    assert run_d1.ascii_text("x" + chr(0xE9)) == "x?"


def test_glare_box_finds_the_bright_blob():
    rgb = np.full((400, 600, 3), 90, dtype=np.uint8)
    rgb[100:130, 300:360] = 255
    box = run_d1.glare_box(rgb, [200, 50, 500, 300], TH)
    assert box is not None and box[0] <= 300 and box[2] >= 359 and box[1] <= 100 and box[3] >= 129
    assert run_d1.glare_box(np.full((400, 600, 3), 90, dtype=np.uint8), [200, 50, 500, 300], TH) is None


def test_mark_clusters_take_one_mark_per_call():
    runs = [[{"px": [100, 100]}, {"px": [400, 400]}], [{"px": [103, 98]}], [{"px": [99, 104]}, {"px": [101, 101]}]]
    clusters = run_d1.mark_clusters(runs, radius=20)
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [1, 1, 3]
    assert max(clusters, key=len)[0]["px"] == [100, 100]


# ------------------------------------------------------------------------------------------------------------------
# gate_sam helpers
# ------------------------------------------------------------------------------------------------------------------


def test_bridge_window_holds_both_prompts():
    assert gate_sam.bridge_window(100, 130, half=90, max_frames=180, total=600) == (26, 179)  # margin (180 - 31) // 2
    f0, n = gate_sam.bridge_window(10, 40, half=90, max_frames=180, total=600)
    assert f0 == 0 and f0 + n - 1 >= 40 and n <= 180
    f0, n = gate_sam.bridge_window(560, 590, half=90, max_frames=180, total=600)
    assert f0 + n == 600 and f0 <= 560
    assert gate_sam.bridge_window(0, 400, half=90, max_frames=180, total=600) is None


def record(lane_id: str, noun: Noun | None) -> LaneRecord:
    return LaneRecord(lane_id=lane_id, role="pickup", noun=noun, prompt_text=noun.value if noun else "x", media_size=[960, 540])


def test_privacy_boxes_carry_over_short_gaps_only():
    frames = [IndexFrame(frame=f, objects=[IndexObject(id="0", box=[10, 10, 50, 30])]) for f in (0, 1, 2, 20)]
    lane = gate_sam.Lane(record=record("d1_e2_license_plate", Noun.LICENSE_PLATE), frames=frames, name="D1 E2 license plate")
    boxes = gate_sam.privacy_boxes_by_frame([lane], fps=15.0, th=TH)
    bridge = math.ceil(TH["privacy"]["track_gap_bridge_s"] * 15.0)
    assert all(f in boxes for f in range(0, 3 + bridge))
    assert 3 + bridge not in boxes and 19 not in boxes and 20 in boxes


def test_side_agnostic_residual_takes_the_mirror():
    row = gate_sam.side_agnostic_residual([0.3, 0.1], [-0.3, 0.1])
    assert row["residual"] == 0.0 and row["used"] == "mirrored"
    assert gate_sam.side_agnostic_residual([0.3, 0.1], [0.31, 0.1])["used"] == "direct"


def test_cross_walk_and_cross_take_residuals():
    roles = {"pickup": {"handle_keyframes": {"handle_w_mean": [0.30, 0.10]}}, "return": {"handle_keyframes": {"handle_w_mean": [-0.28, 0.12]}}}
    results = {"sweeps": [{"fps": 15, "roles": roles, "take2": {"handle_keyframes": {"handle_w_mean": [0.33, 0.06]}}}]}
    walk = gate_sam.cross_walk_residual(results, TH)
    assert walk["used"] == "mirrored" and walk["residual"] == pytest.approx(math.dist([0.30, 0.10], [0.28, 0.12]), abs=1e-5)
    take = gate_sam.cross_take_residual(results)
    assert take["used"] == "direct" and take["residual"] == pytest.approx(0.05, abs=1e-5)
    lane_only = {"sweeps": [{"roles": {"pickup": {"handle": {"position": {"w": [0.1, 0.2]}}}, "return": {"handle": {"position": {"w": [0.1, 0.25]}}}}}]}
    assert gate_sam.cross_walk_residual(lane_only, TH)["residual"] == pytest.approx(0.05, abs=1e-5)
    assert gate_sam.cross_take_residual({"sweeps": []})["residual"] is None


def test_top_frames_skip_the_frame_edge_and_keep_a_gap():
    frames = [
        IndexFrame(frame=0, objects=[IndexObject(id="0", box=[0, 0, 500, 500])]),  # touches the edge
        IndexFrame(frame=1, objects=[IndexObject(id="0", box=[100, 100, 400, 400])]),
        IndexFrame(frame=2, objects=[IndexObject(id="0", box=[100, 100, 410, 410])]),
        IndexFrame(frame=9, objects=[IndexObject(id="0", box=[100, 100, 300, 300])]),
    ]
    picks = gate_sam.top_frames(frames, n=3, min_gap_frames=5, media_size=(960, 540), th=TH)
    assert [f for f, _ in picks] == [2, 9]


# ------------------------------------------------------------------------------------------------------------------
# Live paths against fakes (sam_client patched where it is defined, so privacy's calls are caught too)
# ------------------------------------------------------------------------------------------------------------------


TEXT_EVENTS = gate_sam.read_events(FIX_TEXT)
BOX_EVENTS = gate_sam.read_events(FIX_BOX)
IMAGE_BOXES = {
    "car door": [(0.10, 0.20, 0.70, 0.90, 0.91)],
    "door handle": [(0.40, 0.45, 0.48, 0.50, 0.82)],
    "side mirror": [(0.62, 0.10, 0.72, 0.22, 0.75)],
    "license plate": [(0.45, 0.62, 0.58, 0.69, 0.90)],
    "wheel": [(0.10, 0.55, 0.30, 0.95, 0.95)],
    "bumper": [(0.00, 0.60, 0.45, 0.80, 0.70)],
}


class FakeSam:
    """Replays the 10/26 streams for every lane and writes them like sam_client: slug lane id, gz relative path."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def _record(self, events, *, run_id, role, lane, prompt_text, frames_sent, media_size, metadata, kind, request_index):
        slug = paths.lane_id(lane)
        gz = paths.lane_events_gz_path(run_id, role, slug)
        with gzip.open(gz, "wt", encoding="utf-8") as fh:
            fh.writelines(json.dumps(e) + "\n" for e in events)
        frames, outcome = sam_wire.index_from_events(events, media="video")
        noun = None if kind == "object" else next((n for n in Noun if n.value == prompt_text), None)
        rec = LaneRecord(
            lane_id=slug, role=role, kind=kind, noun=noun, prompt_text=prompt_text, media_size=list(media_size), params=dict(metadata),
            store_echo=False, request_id=outcome.response_id, request_index=request_index, status=outcome.status, lane_path=paths.rel(run_id, gz),
            first_delta_s=1.0, completed_s=5.0, frames_sent=frames_sent, frames_returned=len(frames), video_frames_processed=frames_sent,
            ids=sorted({o.id for f in frames for o in f.objects}), dollars=0.03,
        )
        self.calls.append(f"{kind} {lane}")
        return rec, frames

    async def upload_file(self, path, *, run_id=None, expires_after_s=3600):
        self.calls.append("upload")
        return contracts.FilesApiRecord(file_id="file-fake0001", uploaded_at="2026-10-31T00:00:00Z", expires_at="2026-10-31T01:00:00Z")

    async def segment_video(self, file_id, prompt_text, *, run_id, role, lane, frames_sent, media_size, metadata, kind="text", noun=None, request_index=0, on_event=None, max_tries=6):
        for event in TEXT_EVENTS:
            if on_event:
                on_event(event)
        return self._record(TEXT_EVENTS, run_id=run_id, role=role, lane=lane, prompt_text=prompt_text, frames_sent=frames_sent, media_size=media_size, metadata=metadata, kind="text", request_index=request_index)

    async def segment_video_many(self, file_id, prompts, *, run_id, role, frames_sent, media_size, metadata, concurrency=16, on_event=None):
        out = {}
        for i, (lane, text) in enumerate(prompts):
            callback = (lambda e, lane=lane: on_event(lane, e)) if on_event else None
            out[lane] = await self.segment_video(file_id, text, run_id=run_id, role=role, lane=lane, frames_sent=frames_sent, media_size=media_size, metadata=metadata, request_index=i, on_event=callback)
        return out

    async def segment_clip(self, clip, prompt_text, *, run_id, role, lane, media_size, metadata, frames_sent, on_event=None):
        assert Path(clip).exists()
        return self._record(BOX_EVENTS, run_id=run_id, role=role, lane=lane, prompt_text=prompt_text, frames_sent=frames_sent, media_size=media_size, metadata=metadata, kind="object", request_index=0)

    async def segment_image(self, image, concept, *, run_id, role=None, tag="", include_confidence=True):
        arr = np.asarray(image)
        h, w = arr.shape[:2]
        dets = [sam_client.Detection(object_id=str(i), frame_index=None, box=sam_client.Box(x1 * w, y1 * h, x2 * w, y2 * h), mask=None, confidence=c) for i, (x1, y1, x2, y2, c) in enumerate(IMAGE_BOXES.get(concept, []))]
        self.calls.append(f"image {concept}")
        return sam_client.SegmentationRun(concept=concept, media="image", media_width=w, media_height=h, frames={0: dets}, usd=0.0025, response_id=f"fake-{concept}")

    async def delete_file(self, file_id):
        return {"id": file_id, "object": "file", "deleted": True}

    async def delete_response(self, response_id):
        return {"id": response_id, "status_code": 404, "deleted": False}

    def client(self):
        class Files:
            async def retrieve(self, file_id):
                return SimpleNamespace(model_dump=lambda: {"id": file_id, "expires_at": 1793500000, "status": "processed", "bytes": 1, "purpose": "user_data"})

        class Client:
            files = Files()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

        return Client()


@pytest.fixture
def fake_sam(isolated, monkeypatch):
    fake = FakeSam()
    for name in ("upload_file", "segment_video", "segment_video_many", "segment_clip", "segment_image", "delete_file", "delete_response", "client"):
        monkeypatch.setattr(sam_client, name, getattr(fake, name))
    monkeypatch.setattr(gate_sam, "FORCE_LIVE", True)
    return fake


@pytest.fixture
def fake_spark(isolated):
    fixture = json.loads(SPARK_FIXTURE.read_text())
    fixture["timeline_v1"] = {"panels": [{"panel": "car door", "side": "left", "start_s": 0.0, "end_s": 5.6}, {"panel": "bumper", "side": "none", "start_s": 0.0, "end_s": 5.6}]}
    client = gate_spark.FixtureClient(fixture)
    spark_client.set_client(client)
    yield client
    spark_client.set_client(None)


def test_gate_sam_live_path_reads_what_sam_client_writes(fake_sam, fake_spark, tmp_path):
    out = tmp_path / "sam"
    # the source fps, so the replayed 141-frame lanes line up with the proxy's frames
    args = gate_sam.build_parser().parse_args(["--pickup", str(CLIP), "--return", str(CLIP), "--fps", "23.976", "--fps-fallback", "0", "--handle", "--bridge", "--plate-checks", "--timeline", "--delete-response-probe", "--run-prefix", "d1test", "--out", str(out)])
    results = gate_sam.run(args)
    role = results["sweeps"][0]["roles"]["pickup"]
    door = role["lanes"]["D1 E2 car door"]
    assert door["record_lane_id"] == "d1_e2_car_door" and door["lane_file"].endswith(".ndjson.gz")
    assert role["decode_budget"]["masks_decoded"] > 0  # the lane text was found through the relative lane_path
    assert role["overlay"]["masks_drawn"] > 0 and role["overlay"]["placement_mismatch"] == 0
    assert role["overlay"]["frames_pixelated"] > 0  # the plate lane's boxes are pixelated before the overlay is written
    assert role["variance_car_door"]["runs"] == 3 and role["prop_both_lane"]["present"] is True
    assert "after_car_filter" in door  # the 'car' lane was found by name
    assert len(role["lane_sizes"]["display_nouns_counted"]) == 14
    assert role["files_probe"]["expires_after_accepted"] is True
    assert role["delete_file"]["deleted"] is True and role["delete_response_probe"]["body"]["status_code"] == 404
    assert "variants" in role["anchor_bridge"] and role["anchor_bridge"]["variants"]["box_only"]["status"] == "completed"
    assert "keyframes" in role["handle_keyframes"]  # empty here: the car filter drops the ping pong ball
    assert role["handle"]["sigma"]["frames_used"] >= 3  # the 'door handle' lane replays the same ball inside the 'door'
    assert role["timeline"]["valid"] is True and role["timeline"]["proxy"].endswith(".blurred.mp4")
    assert "handle_cross_walk" in results and "handle_cross_take" in results
    assert results["sweeps"][0]["evaluation"]["store_echo_false_everywhere"] is True
    assert {"car door", "bumper"} <= set(role["suggestions"])
    assert (out / "gate_summary.md").exists()
    spark_rows = ledger.rows(ledger.LEDGER_PATH)
    assert spark_rows and all(r["run_id"].startswith("d1test") for r in spark_rows)  # the timeline call's row


def test_handle_keyframes_whiten_the_handle_and_mirror(fake_sam):
    frames = [IndexFrame(frame=f, objects=[IndexObject(id="0", box=[100 + f, 120, 600 + f, 480])]) for f in (6, 18, 30, 42)]
    door = gate_sam.Lane(record=record("d1_e2_car_door", Noun.CAR_DOOR), frames=frames, name="D1 E2 car door")
    out = gate_sam.run_sync(gate_sam.handle_keyframes(CLIP, door, fps=23.976, media_size=(960, 540), run_id="d1test_E2", role="pickup", th=TH))
    assert len(out["keyframes"]) == 3 and out["handle_w_n"] == 3 and out["mirror_base_w_n"] == 3
    assert out["handle_w_spread_units"] == pytest.approx(0.0, abs=1e-9)  # the fake answers the same boxes every frame
    assert sum(1 for c in fake_sam.calls if c.startswith("image ")) == 9


def test_gate_sam_budget_skips_a_sweep_before_it_starts(fake_sam, tmp_path):
    args = gate_sam.build_parser().parse_args(["--pickup", str(CLIP), "--fps", "15", "--fps-fallback", "0", "--budget", "0.01", "--run-prefix", "d1test", "--out", str(tmp_path / "sam")])
    results = gate_sam.run(args)
    assert results["sweeps"][0]["skipped"].startswith("budget $0.01")
    assert "upload" not in fake_sam.calls


CLIP_FPS = 23.976023976023978


def gz_lane(path: Path, events: list[dict]) -> Path:
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.writelines(json.dumps(e) + "\n" for e in events)
    return path


def moving_entry(tmp_path: Path, **extra) -> dict:
    """An E5 entry whose window follows hand-measured boxes, with the ball lane standing in for the plate lane."""
    plate = gz_lane(tmp_path / "plate.ndjson.gz", TEXT_EVENTS)
    entry = {"src": str(CLIP), "pts": 2.9, "mark": [600, 300, 640, 340], "panel": [400, 150, 900, 600], "panel_start": [340, 150, 840, 600], "panel_end": [460, 150, 960, 600],
             "label": "tmark", "role": "pickup", "scuff_text": True,
             "privacy_lanes": [{"noun": "license plate", "path": str(plate), "status": "completed", "fps": CLIP_FPS, "size": [1280, 720]}, {"noun": "face", "path": str(plate), "status": "zero_matches", "fps": CLIP_FPS, "size": [1280, 720]}]}
    entry.update(extra)
    return entry


def run_stays_put(tmp_path: Path, entries: list[dict]) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(entries))
    args = gate_stays_put.build_parser().parse_args(["--plan", str(plan), "--run-prefix", "d1test", "--out", str(tmp_path / "sp")])
    return gate_stays_put.run(args)


def test_gate_stays_put_live_entry(fake_sam, tmp_path, monkeypatch):
    video = gate_sam.optional_module("video")
    cut = video.cut_clip
    seen: list[list[list[float]]] = []

    def spy(*args, **kwargs):
        seen.append(kwargs.get("blur_boxes"))
        return cut(*args, **kwargs)

    monkeypatch.setattr(video, "cut_clip", spy)
    results = run_stays_put(tmp_path, [moving_entry(tmp_path)])
    clips = results["clips"]
    assert {c["clip_type"] for c in clips} == {"mark", "panel"}
    assert all(c["clip"]["w"] == 1008 and c["clip"]["h"] == 1008 for c in clips)
    assert all(c["blur"]["plates"] == 1 for c in clips)  # image checks on the prompt frame before the clip is cut
    assert all(c["blur"]["track_based"] and c["blur"]["frames_pixelated"] == 91 for c in clips)
    assert all(Path(c["clip_file"]).exists() for c in clips)
    assert results["entries"][0]["scuff_text"]["status"] == "completed"
    assert "image license plate" in fake_sam.calls
    blur = seen[0]
    assert len(blur) == 91 and blur[0] != blur[90]  # per frame: the image box rides the panel and the lane boxes move
    assert results["partial"] is False and json.loads((tmp_path / "sp" / "gate_stays_put.json").read_text())["partial"] is False


def test_stays_put_refuses_a_static_window_before_any_request(fake_sam, tmp_path):
    entry = moving_entry(tmp_path)
    for key in ("panel_start", "panel_end"):
        entry.pop(key)
    results = run_stays_put(tmp_path, [entry])
    assert results["clips"] == [] and results["entries"][0]["skipped"].startswith("static window")
    assert not [c for c in fake_sam.calls if c.startswith(("image", "object"))]
    assert results["evaluation"]["kill"] is None and results["evaluation"]["demo_pass_any"] is None


def failing_images(monkeypatch, concept: str) -> None:
    real = sam_client.segment_image

    async def segment_image(image, wanted, **kwargs):
        if wanted == concept:
            raise sam_client.SamRequestError(f"The {wanted!r} image check ended incomplete: test")
        return await real(image, wanted, **kwargs)

    monkeypatch.setattr(sam_client, "segment_image", segment_image)


def test_stays_put_image_checks_that_fail(fake_sam, tmp_path, monkeypatch):
    failing_images(monkeypatch, "license plate")
    no_lanes = run_stays_put(tmp_path / "a", [moving_entry(tmp_path, privacy_lanes=[])])
    assert no_lanes["clips"] == [] and "no clip is cut" in no_lanes["entries"][0]["skipped"]
    with_lanes = run_stays_put(tmp_path / "b", [moving_entry(tmp_path)])
    assert {c["clip_type"] for c in with_lanes["clips"]} == {"mark", "panel"}
    assert all("error" in c["blur"] and c["blur"]["track_based"] for c in with_lanes["clips"])


def test_lane_blur_boxes_scale_dilate_and_carry():
    frames = [IndexFrame(frame=f, objects=[IndexObject(id="0", box=[100 + 10 * f, 100, 140 + 10 * f, 120])]) for f in range(0, 4)]
    times = [i / 30.0 for i in range(60)]  # source at 30 fps, lane at 15 fps, source twice the lane size
    boxes = gate_stays_put.lane_blur_boxes(times, list(range(0, 30)), [(frames, 15.0, (960, 540))], src_size=(1920, 1080), bridge_s=0.5, dilate_frac=0.3)
    assert boxes[0] == [[200 - 12, 200 - 6, 280 + 12, 240 + 6]]  # scaled x2, then 30% wider
    assert len(boxes[1]) == 2  # between lane frames 0 and 1: both boxes
    bridge = math.ceil(0.5 * 15)
    last_covered = 2 * (3 + bridge) + 1  # clip frame 23 sits between lane frames 11 and 12: frame 3 carried to 11
    assert boxes[last_covered] and not boxes[last_covered + 1]  # the last box carried for 0.5 s, then nothing


def test_stays_put_evaluate_counts_measured_clips_only():
    def clip(label, clip_type, ok, status="completed", error=None):
        row = {"label": label, "clip_type": clip_type, "is_control": False, "panned_pass": False, "backtrack": False, "pass": ok, "metrics": {"lane_status": status, "verdict": "inconclusive", "b1": None, "b2": None}}
        if error:
            row["error"] = error
        return row

    ev = gate_stays_put.evaluate([clip("a", "mark", False), clip("a", "panel", False, status="failed")], TH)
    assert ev["kill"] is None and ev["demo_pass_any"] is False and ev["demo_clips_not_measured"] == 1
    ev = gate_stays_put.evaluate([clip("a", "mark", False), clip("a", "panel", None, error="ClipOutOfRange: x")], TH)
    assert ev["kill"] is None
    ev = gate_stays_put.evaluate([clip("a", "mark", False), clip("a", "panel", False, status="zero_matches")], TH)
    assert ev["kill"] is True
    ev = gate_stays_put.evaluate([clip("a", "mark", True), clip("b", "panel", None, status="failed")], TH)
    assert ev["kill"] is False and ev["demo_pass_any"] is True
    assert gate_stays_put.evaluate([clip("a", "mark", None, status="failed")], TH)["demo_pass_any"] is None


def test_gate_spark_live_blurs_and_records_the_store_echo(fake_sam, fake_spark, tmp_path):
    crops = [{"name": "door", "source": str(CLIP), "t": 2.0, "crop": [300, 150, 1000, 650], "panel": "car door", "clean": False, "targets": []},
             {"name": "other", "source": str(CLIP), "t": 2.0, "crop": [0, 0, 640, 360], "panel": "bumper", "clean": True, "targets": []}]
    path = tmp_path / "crops.json"
    path.write_text(json.dumps(crops))
    args = gate_spark.build_parser().parse_args(["--crops", str(path), "--efforts", "low", "--n", "2", "--count", "--truncate", "600", "--auto-looks", "--claims-from-crops", "--ruler-frame", str(CLIP), "--ruler-t", "2.0", "--tire-code", "225/60R16", "--note-prefill-synthetic", "--run-prefix", "d1test", "--out", str(tmp_path / "sp")])
    results = gate_spark.run(args)
    assert results["crop_blur"]["door"]["plates"] == 1  # one blur per source frame, before Spark sees the crop
    assert results["store_echo"]["answers"] > 0 and results["store_echo"]["all_false"] is None  # canned answers carry no store
    assert set(results["second_look"]) == {"pair", "control"}
    assert results["second_look"]["pair"]["source"].startswith("auto:")
    assert len(results["claims"]) == 2 and results["ruler"]["plate"]["qualifies"] is True
    rows = ledger.rows(ledger.LEDGER_PATH)
    assert rows and {r["kind"] for r in rows} == {"spark"} and all(r["run_id"].startswith("d1test") for r in rows)


def test_truncation_probe_drops_to_the_fallback_budget(isolated):
    class Responses:
        def __init__(self) -> None:
            self.budgets: list[int] = []

        def create(self, **request):
            budget = request["max_output_tokens"]
            self.budgets.append(budget)
            usage = {"input_tokens": 1000, "output_tokens": min(budget, 260), "output_tokens_details": {"reasoning_tokens": 200}, "input_tokens_details": {"cached_tokens": 0}}
            if budget < 200:  # the 10/31 smoke: a markless crop finished inside 600 tokens
                return {"id": "r2", "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}, "usage": usage, "output": []}
            text = json.dumps({"marks": [], "conditions": {"wet": False, "dirty": False, "glare": False, "dark": False}, "landmarks": [], "neighbour_car_visible": False})
            return {"id": "r1", "status": "completed", "usage": usage, "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}

    client = SimpleNamespace(responses=Responses())
    spark_client.set_client(client)
    try:
        crop = gate_spark.synthetic_crop("t", [])
        out = gate_spark.truncation_probe(crop, contracts.PanelId.LEFT_FRONT_DOOR, max_output_tokens=600, run_id="d1test_E6_truncate", th=TH)
    finally:
        spark_client.set_client(None)
    assert client.responses.budgets == [600, 150]
    assert out["signal_seen"] is True and out["incomplete_reason"] == "max_output_tokens" and len(out["attempts"]) == 2
    assert len(ledger.rows(ledger.LEDGER_PATH)) == 2


def test_auto_looks_skip_without_a_mark_to_outline():
    specs = [{"name": "a", "clean": False}, {"name": "b", "clean": False}]
    crops = {"a": np.zeros((50, 80, 3), np.uint8), "b": np.zeros((50, 80, 3), np.uint8)}
    results = {"crops": [{"name": "a", "efforts": {"low": {"runs": [{"marks": [{"bbox_px": [1, 1, 5, 5], "capped": True}]}]}}}]}
    looks, notes = gate_spark.auto_looks(results, specs, crops, have=set())
    assert looks == {} and "no uncapped mark" in notes["skipped"]
    results["crops"][0]["efforts"]["low"]["runs"][0]["marks"].append({"bbox_px": [10, 10, 20, 20], "capped": False})
    looks, notes = gate_spark.auto_looks(results, specs, crops, have=set())
    assert set(looks) == {"pair", "control"} and notes["control"].endswith("(another panel, weak)")


def test_note_prefill_on_one_given_pdf(isolated, fake_spark, tmp_path):
    fields = gate_spark.SYNTHETIC_AGREEMENTS[0]
    pdf = gate_spark.render_agreement_pdf(fields, tmp_path / "agreement.pdf")
    out = gate_spark.note_prefill_probe(tmp_path, run_id="d1test", th=TH, pdf=pdf, claim_text="They charged me for the door.", expected=fields)
    assert len(out["agreements"]) == 1 and out["agreements"][0]["agreed_fields"] == 5 and out["all_fields_on_all"] is True
    assert gate_spark.note_prefill_probe(tmp_path, run_id="d1test", th=TH, pdf=pdf)["all_fields_on_all"] is None  # nothing typed to compare


def test_echo_client_reads_store_from_plain_and_streamed_answers():
    class Inner:
        class responses:  # noqa: N801 - the SDK's shape
            @staticmethod
            def create(**request):
                if request.get("stream"):
                    return iter([{"type": "response.output_text.delta", "delta": "{}"}, {"type": "response.completed", "response": {"store": False}}])
                return {"store": False, "status": "completed"}

    echo = gate_spark.EchoClient(Inner())
    echo.responses.create(model="m")
    list(echo.responses.create(model="m", stream=True))
    assert echo.summary() == {"answers": 2, "counts": {"False": 2}, "all_false": True}


def test_gate_voice_live_pins_against_gate_sam_windows(isolated, tmp_path, monkeypatch):
    fixture = json.loads((REPO / "tests" / "fixtures" / "d1" / "voice_fixture.json").read_text())

    def handler(request: httpx.Request) -> httpx.Response:
        mode = "DIARIZATION" if b'"DIARIZATION"' in request.read() else "ENDPOINTING"
        return httpx.Response(200, json=fixture["modes"][mode])

    monkeypatch.setattr(voice_client, "TRANSPORT", httpx.MockTransport(handler))
    monkeypatch.setattr(voice_client, "api_key", lambda: "test-key")
    monkeypatch.setattr(gate_sam, "FORCE_LIVE", True)
    wav = gate_voice.synthetic_wav(tmp_path / "narration.wav", seconds=2.0)
    pins = tmp_path / "gate_sam.json"
    pins.write_text(json.dumps({"sweeps": [{"fps": 15, "roles": {"pickup": {"lanes": {"D1 E2 bumper": {"noun": "bumper", "windows_s": [[0.0, 2.0], [4.5, 6.0]]}, "D1 E2 car door": {"noun": "car door", "windows_s": [[1.4, 3.8]]}}}}}]}))
    args = gate_voice.build_parser().parse_args([str(wav), "--modes", "ENDPOINTING", "DIARIZATION", "--pins", str(pins), "--run-prefix", "d1test", "--out", str(tmp_path / "v")])
    results = gate_voice.run(args)
    assert results["modes"]["DIARIZATION"]["pin_share"]["landed"] == 3
    assert results["evaluation"]["mode_applied"] is True
    assert results["modes"]["ENDPOINTING"]["split"]["turns"] >= 1 and "pin_share_split" in results["evaluation"]
    rows = ledger.rows(ledger.LEDGER_PATH)
    assert [r["note"] for r in rows] == ["ENDPOINTING", "DIARIZATION"] and all(r["run_id"].startswith("d1test") for r in rows)


# ------------------------------------------------------------------------------------------------------------------
# Review fixes: E4 nothing to join, E2's probe lane, E9 windows, paid results kept, Spark retries, E5 lanes
# ------------------------------------------------------------------------------------------------------------------


def test_absorb_report_reads_nothing_to_join_as_none():
    door = gate_sam.lane_from_events(TEXT_EVENTS, lane_id="D1 E2 car door", role="pickup", noun=Noun.CAR_DOOR, prompt_text="car door")
    bridge = gate_sam.lane_from_events(BOX_EVENTS, lane_id="bridge", role="pickup", noun=None, prompt_text="object prompt", kind="object")
    row = gate_sam.absorb_report(door, bridge, bridge_object=bridge.record.ids[0], frame_offset=0, th=TH, fps=CLIP_FPS)
    assert row["joined"] is None and row["note"].startswith("nothing to join")
    row = gate_sam.absorb_report(door, bridge, bridge_object=bridge.record.ids[0], frame_offset=0, th=TH, fps=CLIP_FPS, fragments_to_join=2)
    assert row["joined"] is False  # two fragments given, one absorbed: a measured failure to join


def test_recommend_takes_the_anchor_pass_when_e4_measured_no_join():
    for status in ("info", "not run"):
        rec = run_d1.recommend(SimpleNamespace(fps=None, dry=False), criteria(E2_tracks="fail", E4_joins=status))
        assert rec["headline"].startswith("Use hosted SAM 3.1 with the anchor pass"), status
        assert "measured no join" in rec["lines"][0] and "SAM_ENGINE=fal" in rec["lines"][1]


def test_eval_e4_reads_nothing_to_join_as_measured():
    bridge = {"note": "nothing to join: no door track had 2 or more fragments", "joined": None, "variants": {"box_only": {"absorb": {"joined": None, "note": "nothing to join: 1 fragment(s) on the bridged track"}}}}
    res = SimpleNamespace(roles=lambda: {"pickup": {"anchor_bridge": bridge}}, missing=lambda key: run_d1.Verdict("not run", "", key))
    assert run_d1.eval_e4(res)["E4.joins"].status == "info"
    failed = {"variants": {"box_only": {"status": "failed", "error": "429 after 6 tries: x"}}, "joined": None}
    res = SimpleNamespace(roles=lambda: {"pickup": {"anchor_bridge": failed}}, missing=lambda key: run_d1.Verdict("not run", "", key))
    joins = run_d1.eval_e4(res)["E4.joins"]
    assert joins.status == "not run" and "429" in joins.note  # an API failure is never a failed join


def sweep_lane(name: str, noun: str, status: str = "completed", error: str | None = None, echo: bool | None = False) -> dict:
    return {"lane_id": name, "noun": noun, "kind": "text", "status": status, "error": error, "store_echo": echo, "frames_sent": 600, "completed_s": 50.0, "parse_cpu_s": 0.1, "retries_429": 0, "longest_track_share": 0.9, "longest_track_fragments": 1}


def test_evaluate_sweep_reads_the_probe_lane_apart():
    lanes = {f"D1 E2 {n.value}": sweep_lane(f"D1 E2 {n.value}", n.value) for n in contracts.LANE_NOUNS}
    lanes[gate_sam.PROP_BOTH_LANE] = sweep_lane(gate_sam.PROP_BOTH_LANE, "car door", status="failed", error="400: propagation_direction is only valid for object prompts", echo=None)
    sweep_out = {"fps": 15.0, "roles": {"pickup": {"lanes": lanes}}}
    ev = gate_sam.evaluate_sweep(sweep_out, TH)
    assert ev["no_429_loss"]["pass"] is True and ev["concurrency"]["pass"] is True and ev["store_echo_false_everywhere"] is True
    assert ev["no_429_loss"]["failed_other"] == [gate_sam.PROP_BOTH_LANE] and ev["concurrency"]["lanes_started"] == 16
    lanes["D1 E2 hood"] = sweep_lane("D1 E2 hood", "hood", status="failed", error="429 after 6 tries: rate limited", echo=None)
    ev = gate_sam.evaluate_sweep(sweep_out, TH)
    assert ev["no_429_loss"]["pass"] is False and ev["no_429_loss"]["failed_lanes"] == ["D1 E2 hood"] and ev["concurrency"]["pass"] is False


def plate_lane(frames: list[int]) -> gate_sam.Lane:
    rows = [IndexFrame(frame=f, objects=[IndexObject(id="0", box=[10, 10, 50, 30])]) for f in frames]
    return gate_sam.Lane(record=record("d1_e2_license_plate", Noun.LICENSE_PLATE), frames=rows, name="D1 E2 license plate")


def test_blur_coverage_splits_the_sides_of_the_car_out():
    seen = list(range(0, 60)) + list(range(615, 675))  # 45 s at 15 fps: the plate at the start and the end of the loop
    cov = gate_sam.blur_coverage(plate_lane(seen), fps=15.0, th=TH)
    assert cov["coverage_raw"] == 1.0 and cov["gaps_split"] == 1 and cov["coverage_raw_first_to_last"] < 0.2
    holes = [f for f in seen if not 20 <= f < 35]  # a 1 s miss with the plate in view still counts against it
    cov = gate_sam.blur_coverage(plate_lane(holes), fps=15.0, th=TH)
    assert cov["gaps_split"] == 1 and cov["gaps_over_bridge"] == 1 and cov["coverage_bridged"] == pytest.approx(105 / 120, abs=1e-4)


def test_gate_sam_keeps_paid_results_when_a_stage_fails(fake_sam, tmp_path, monkeypatch):
    failing_images(monkeypatch, "license plate")

    async def broken_timeline(*args, **kwargs):
        raise RuntimeError("timeline broke")

    monkeypatch.setattr(gate_sam, "timeline_probe", broken_timeline)
    nouns = ["car", "car door", "bumper", "license plate", "face"]
    argv = ["--pickup", str(CLIP), "--fps", "23.976", "--fps-fallback", "0", "--nouns", *nouns, "--bridge", "--plate-checks", "--timeline", "--run-prefix", "d1test"]
    out = tmp_path / "sam"
    results = gate_sam.run(gate_sam.build_parser().parse_args(argv + ["--out", str(out)]))
    role = results["sweeps"][0]["roles"]["pickup"]
    rows = role["plate_image_checks"]["frames"]
    assert rows and all("SamRequestError" in r["error"] for r in rows)
    assert role["timeline"] == {"error": "RuntimeError: timeline broke"} and role["delete_file"]["deleted"] is True
    assert json.loads((out / "gate_sam.json").read_text())["partial"] is False and "evaluation" in results["sweeps"][0]

    def late_crash(*args, **kwargs):
        raise RuntimeError("late crash")

    monkeypatch.setattr(gate_sam, "evaluate_sweep", late_crash)
    out2 = tmp_path / "sam2"
    with pytest.raises(RuntimeError):
        gate_sam.run(gate_sam.build_parser().parse_args(argv + ["--out", str(out2)]))
    data = json.loads((out2 / "gate_sam.json").read_text())
    assert data["partial"] is True and "D1 E2 car door" in data["sweeps"][0]["roles"]["pickup"]["lanes"]  # the paid lanes survive


def test_run_gate_reads_a_partial_gate_and_sets_a_stale_one_aside(tmp_path, monkeypatch):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    gate = scripts / "gate_sam.py"
    gate.write_text("import json, sys\nout = sys.argv[sys.argv.index('--out') + 1]\nopen(out + '/gate_sam.json', 'w').write(json.dumps({'partial': True, 'sweeps': []}))\nsys.exit(1)\n")
    monkeypatch.setattr(run_d1, "SCRIPTS", scripts)
    ctx = live_ctx(tmp_path / "run")
    (tmp_path / "run" / "sam").mkdir(parents=True)
    (tmp_path / "run" / "sam" / "gate_sam.json").write_text(json.dumps({"stale": True}))
    row = run_d1.run_gate(ctx, "sam", "gate_sam.py", [])
    assert row["status"] == "partial" and json.loads((tmp_path / "run" / "sam" / "gate_sam.previous.json").read_text()) == {"stale": True}
    ctx.results["steps"]["sam"] = row
    assert run_d1.gate_data(ctx, "sam") == {"partial": True, "sweeps": []}
    gate.write_text("import sys\nsys.exit(1)\n")
    assert run_d1.run_gate(ctx, "sam", "gate_sam.py", [])["status"] == "failed"


def test_usage_line_counts_the_cut_its_retry_hides(isolated):
    answer = json.dumps({"marks": [], "conditions": {"wet": False, "dirty": False, "glare": False, "dark": False}, "landmarks": [], "neighbour_car_visible": False})

    class Responses:
        def __init__(self) -> None:
            self.budgets: list[int] = []

        def create(self, **request):
            budget = request["max_output_tokens"]
            self.budgets.append(budget)
            usage = {"input_tokens": 3400, "output_tokens": 4000 if budget == 4000 else 1200, "output_tokens_details": {"reasoning_tokens": 900}, "input_tokens_details": {"cached_tokens": 0}}
            if budget == 4000:
                return {"id": f"cut{len(self.budgets)}", "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}, "usage": usage, "output": []}
            return {"id": f"ok{len(self.budgets)}", "status": "completed", "usage": usage, "output": [{"type": "message", "content": [{"type": "output_text", "text": answer}]}]}

    client = SimpleNamespace(responses=Responses())
    spark_client.set_client(client)
    try:
        crop = gate_spark.synthetic_crop("t", [])
        results = [spark_client.find_marks(crop, contracts.PanelId.LEFT_FRONT_DOOR, landmarks=[], run_id="d1test_E6") for _ in range(2)]
    finally:
        spark_client.set_client(None)
    assert client.responses.budgets == [4000, 8000, 4000, 8000]
    line = gate_spark.usage_line(results)
    assert line["incomplete_at_4000"] == 2 and line["retries"] == 2 and line["attempts"] == 4
    billed = sum(float(r["usd"]) for r in ledger.rows(ledger.LEDGER_PATH))
    assert line["dollars_total"] == pytest.approx(billed, abs=1e-6) and line["dollars_mean"] == pytest.approx(billed / 2, abs=1e-6)
    assert line["output_tokens_mean"] == 5200.0  # 4000 cut plus 1200 answered, per call


def test_vote_agreement_with_a_missing_request(fake_sam, monkeypatch):
    votes = {"winner": "car door", "complete": False, "failed_nouns": ["hood"]}
    assert gate_spark.vote_agrees("car door", votes) is None  # the hood might have won
    assert gate_spark.vote_agrees("bumper", votes) is False  # scored and lost; no missing score undoes that
    assert gate_spark.vote_agrees("hood", votes) is None
    assert gate_spark.vote_agrees("car door", {"winner": "car door", "complete": True}) is True
    failing_images(monkeypatch, "hood")
    image = np.zeros((400, 600, 3), np.uint8)
    out = asyncio.run(gate_spark.sam_votes(image, (300.0, 250.0), run_id="d1test_E11", th=TH))
    assert out["complete"] is False and out["failed_nouns"] == ["hood"] and out["winner"] == "car door"


def test_memo_reads_unmeasured_rows_as_not_run():
    missing = lambda key: run_d1.Verdict("not run", "", key)  # noqa: E731
    clip = {"label": "a", "clip_type": "mark", "is_control": False, "panned_pass": False, "backtrack": False, "error": "x", "metrics": {"verdict": "not_run"}}
    sp = {"clips": [clip], "entries": [], "evaluation": {"kill": None, "demo_pass_any": None, "control_pass": None, "not_measured": ["a/mark: x"]}}
    out = run_d1.eval_e5(SimpleNamespace(g={"staysput": sp}, missing=missing))
    assert out["E5.kill"].status == "not run" and out["E5.demo_mark"].status == "not run" and "a/mark" in out["E5.kill"].note
    ruler = {"plate": None, "wheel": None, "plate_checked": False, "wheel_checked": True, "errors": {"D1 E11 ruler plate": "SamRequestError: x"}}
    out = run_d1.eval_spark(SimpleNamespace(g={"spark": {"evaluation": {}, "ruler": ruler, "crops": []}}, missing=missing))
    assert out["E11.plate_ruler"].status == "not run" and out["E11.wheel_ruler"].status == "fail"


def test_staysput_entries_on_a_swept_walk_get_its_lanes(tmp_path):
    sam = synthetic_sam(tmp_path)
    for role in sam["sweeps"][0]["roles"].values():
        role["lanes"]["D1 E2 license plate"] = {"noun": "license plate", "lane_file": str(tmp_path / "plate.ndjson.gz"), "media_size": [960, 540], "status": "completed"}
        role["lanes"]["D1 E2 face"] = {"noun": "face", "lane_file": str(tmp_path / "face.ndjson.gz"), "media_size": [960, 540], "status": "zero_matches"}
    plan = {"staysput": [
        {"src": str(CLIP), "pts": 1.0, "mark": [1, 1, 5, 5], "panel": [0, 0, 50, 50], "label": "m", "role": "pickup", "panel_noun": "car door"},
        {"src": str(tmp_path / "panned.mov"), "pts": 1.0, "mark": [1, 1, 5, 5], "panel": [0, 0, 50, 50], "label": "p", "role": "pickup", "panned": True},
    ]}
    argv, picked = run_d1.staysput_args(live_ctx(tmp_path, plan), sam, None)
    swept, panned = json.loads(Path(argv[argv.index("--plan") + 1]).read_text())
    assert swept["panel_lane"] == str(tmp_path / "car door.ndjson.gz") and swept["lane_fps"] == 15.0 and swept["lane_size"] == [960, 540]
    assert [lane["noun"] for lane in swept["privacy_lanes"]] == ["license plate", "face"] and swept["privacy_lanes"][1]["status"] == "zero_matches"
    assert "panel_lane" not in panned and "privacy_lanes" not in panned
    assert any("panned.mov is not the pickup walk" in note for note in picked["notes"])
    assert "panel_lane" not in plan["staysput"][0]  # the plan itself is left as written


def test_delete_unblurred_keeps_only_track_blurred_e5_clips(tmp_path):
    ctx = live_ctx(tmp_path)
    clips = tmp_path / "staysput" / "clips"
    clips.mkdir(parents=True)
    kept, dropped = clips / "pickup_m_mark.mp4", clips / "pickup_p_mark.mp4"
    kept.write_bytes(b"a")
    dropped.write_bytes(b"b")
    rows = [{"clip_file": str(kept), "blur": {"track_based": True}}, {"clip_file": str(dropped), "blur": {"track_based": False}}]
    (tmp_path / "staysput" / "gate_stays_put.json").write_text(json.dumps({"clips": rows}))
    deleted = run_d1.delete_unblurred(ctx)
    assert [d["path"] for d in deleted] == ["staysput/clips/pickup_p_mark.mp4"] and kept.exists() and not dropped.exists()


def test_gate_spark_sends_nothing_whose_blur_did_not_complete(fake_sam, fake_spark, tmp_path, monkeypatch):
    from PIL import Image

    failing_images(monkeypatch, "face")
    crops = [{"name": "door", "source": str(CLIP), "t": 2.0, "crop": [300, 150, 1000, 650], "panel": "car door", "clean": False, "targets": []}]
    path = tmp_path / "crops.json"
    path.write_text(json.dumps(crops))
    claim = tmp_path / "claim.png"
    Image.fromarray(gate_spark.synthetic_crop("claim", [(500, 400)])).save(claim)
    argv = ["--crops", str(path), "--efforts", "low", "--n", "2", "--claim", str(claim), "--ruler-frame", str(CLIP), "--ruler-t", "2.0", "--run-prefix", "d1test", "--out", str(tmp_path / "sp")]
    results = gate_spark.run(gate_spark.build_parser().parse_args(argv))
    assert [c["name"] for c in results["crops"]] == ["door"] and "blur checks did not complete" in results["crops"][0]["skipped"]
    assert results["claims"][0]["agreed"] is None and "blur checks did not complete" in results["claims"][0]["error"]
    assert results["evaluation"]["claims_total"] == 0 and results["evaluation"]["claims_not_judged"] == 1
    assert results["ruler"]["plate"]["qualifies"] is True  # the ruler reads the unblurred frame in memory and writes nothing
    assert not [r for r in ledger.rows(ledger.LEDGER_PATH) if r["kind"] == "spark"]  # Spark saw no frame
    assert not (tmp_path / "sp" / "crop_door.png").exists()
