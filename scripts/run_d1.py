"""D1 gate runner (docs/ARCHITECTURE.md section 11): every E row in order, one results.json, the decision memo.

    LIVE=1 python scripts/run_d1.py --pickup PICKUP.mp4 --return RETURN.mp4 [--plan PLAN.json] [--budget 25] [--out DIR]
    python scripts/run_d1.py --dry-run [--out DIR] [--budget DOLLARS]
    python scripts/run_d1.py --plan-template > runs/d1_plan.json
    python scripts/run_d1.py --render-only runs/d1/<timestamp>

Order, as far as the gate scripts bundle the rows: E0 setup and E1 proxy timing here; gate_sam.py (E2, with E3, E4,
E8, E9, E15 and E17 on its lanes and E12's store echo, Files probe and DELETE bodies); gate_spark.py (E6, E7, E11,
E18); gate_stays_put.py (E5, last on section 11's critical path E0 + E2 + E6 + E5); gate_voice.py (E10 against
gate_sam's panel windows); the E12 ots upgrade; E13 from the measured parts. E14, E16 and E19 need a browser,
Baseten or the fal adapter and are recorded as not run with the reason.

Each gate runs as its own process with --run-prefix and --budget, so every ledger row carries the D1 prefix and a
gate stops starting work at the ceiling. Before each step this runner reads the prefix's dollars back from the
ledger and skips the step when they plus the step's planned dollars would pass --budget (default
thresholds cost.d1_budget_usd). --dry-run replays the 10/26 probe fixtures through every gate: no model or network
call, ledger rows go to <out>/dryrun_runs and each gate's ledger.dryrun.jsonl, and the memo says DRY RUN on top.
Without LIVE=1 and without --dry-run the runner prints its plan and exits 2.

The plan file (optional; --plan-template prints one) names what only a person can pick: the E6 crops with the clean
door and hand-measured targets, the second-look pairs, the E5 marks and the reflection control, the E11 claim
photos and tire code, take 2 for E8, a separate narration file for E10. Whatever it leaves out is picked from
gate_sam's lanes: the largest door and bumper frames for E6, the best repeated E6 mark and the return hood's
brightest highlight for E5, the largest plate frame for the E11 ruler. E5 entries on a swept walk get that walk's
panel, plate and face lanes from gate_sam. Unblurred intermediates (the SAM proxies, the bridge clips cut from them,
and any E5 clip not pixelated from plate and face tracks) are deleted at the end unless --keep-unblurred; deletions
are listed in results.json. A gate that stops part way is 'partial': the JSON it wrote after each paid stage still
feeds the memo, and the runner exits 1.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
for entry in (str(REPO), str(SCRIPTS)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import gate_sam  # noqa: E402
from walkaround import contracts, ledger, paths  # noqa: E402
from walkaround.contracts import LANE_NOUNS  # noqa: E402

TEMPLATE = SCRIPTS / "d1_memo_template.md"
FIXTURES = {
    "sam_text": REPO / "tests" / "fixtures" / "sam" / "pingpong_text.events.ndjson",
    "sam_box": REPO / "tests" / "fixtures" / "sam" / "pingpong_box.events.ndjson",
    "clip": REPO / "data" / "test" / "pingpong.mp4",
    "clip_return": REPO / "data" / "test" / "shiba_jump.mp4",
    "spark": REPO / "tests" / "fixtures" / "d1" / "spark_fixture.json",
    "voice": REPO / "tests" / "fixtures" / "d1" / "voice_fixture.json",
}
GATE_JSON = {"sam": "gate_sam.json", "spark": "gate_spark.json", "staysput": "gate_stays_put.json", "voice": "gate_voice.json"}
SPARK_CALL_USD = 0.012  # per marks_v1 call at 1280 px, from the 10/26 probe ($0.007 to $0.012 at 1600 px); E6 replaces it
SAM_IMAGE_USD = 0.0025
CLIP_USD = 91 * 0.0002  # one Stays Put clip, 91 frames


@dataclass(frozen=True)
class Step:
    key: str
    rows: tuple[str, ...]
    title: str


STEPS: tuple[Step, ...] = (
    Step("E0", ("E0",), "tools, hashes of the originals, proxies, the ots and freetsa smoke"),
    Step("E1", ("E1",), "proxy encode time on this Mac at medium and veryfast"),
    Step("sam", ("E2", "E3", "E4", "E8", "E9", "E15", "E17"), "scripts/gate_sam.py"),
    Step("spark", ("E6", "E7", "E11", "E18"), "scripts/gate_spark.py"),
    Step("staysput", ("E5",), "scripts/gate_stays_put.py"),
    Step("voice", ("E10",), "scripts/gate_voice.py"),
    Step("E12", ("E12",), "ots upgrade on the E0 stamps; the rest of E12 comes from E0 and the gates"),
    Step("E13", ("E13",), "wall clock and dollars from the measured parts"),
)
STEP_KEYS = tuple(s.key for s in STEPS)
ROWS: dict[str, tuple[str, str]] = {
    "E0": ("Setup", "20 min, $0"),
    "E1": ("Upload and proxy timing", "30 min, $0"),
    "E2": ("Panel sweep", "60 min, about $7 at 15 and 10 fps on both walks"),
    "E3": ("c semantics and states", "15 min, in E2's data"),
    "E4": ("Anchor bridge", "20 min, $0.08"),
    "E5": ("Stays Put", "45 min, about $0.20"),
    "E6": ("Spark jitter and the second look", "30 min, about $0.70"),
    "E7": ("Count mode", "10 min, $0.08"),
    "E8": ("Door handle and the cross-take residual", "25 min, $0.13"),
    "E9": ("Blur lanes", "15 min, in E2's data plus $0.02"),
    "E10": ("Voice", "10 min, $0.01"),
    "E11": ("Claim-first probes", "30 min, $0.15"),
    "E12": ("Seal, Files and storage", "15 min, $0"),
    "E13": ("Wall clock and dollars", "30 min, about $9"),
    "E14": ("Capture path and browsers", "20 min, $0"),
    "E15": ("Lane size and worker parse", "10 min, $0"),
    "E16": ("S1 probe, Spark Open on Baseten", "30 min, about $0.10"),
    "E17": ("S2 probe, timeline over the blurred proxy", "15 min, under $0.50"),
    "E18": ("S2b probe, note prefill", "15 min, under $0.20"),
    "E19": ("fal.ai adapter", "$0 until used"),
}
ROW_STEP = {row: s.key for s in STEPS for row in s.rows}
NOT_RUN_HERE = {
    "E14": "needs the replay route in a browser and screencapture of the Claude Chrome window; run by hand",
    "E16": "needs Baseten (Muse Spark Open); scripts/gate_s1.py in another lane owns it",
    "E19": "needs the fal.ai adapter; scripts/gate_fal.py in another lane owns it",
}

# (id, row, the rule as section 11 writes it, what the fail branch says)
CRITERIA: tuple[tuple[str, str, str, str], ...] = (
    ("E0.tools", "E0", "every command answers: ots --version, modal token, railway whoami, node --version, ffmpeg -version with libx265", "Sanat installs what is missing (uv pip install opentimestamps-client modal; railway login)"),
    ("E0.ots", "E0", "ots and freetsa smoke on a text file: two .ots files exist", "stamps print pending; the freetsa token stands alone until ots works"),
    ("E0.hashes", "E0", "the originals are hashed before any conversion", "-"),
    ("E0.proxies", "E0", "proxies of the pickup walk at 15, 10 and 30 fps", "-"),
    ("E1.upload_wifi", "E1", "300 MB through the chunk route to the deploy from the Mac on wifi under 60 s, and the phone on LTE", "the pitch line follows section 5"),
    ("E1.proxy_mac", "E1", "the 4K30 take to the 960 px 15 fps proxy on the M4 Max at preset medium and veryfast", "-"),
    ("E1.proxy_container", "E1", "container proxy under 90 s at medium", "veryfast by default; over 120 s resize the service"),
    ("E2.tracks", "E2", "door and bumper tracks >= 70% of visible frames with <= 3 fragments after 5.1 at the chosen fps", "anchor pass on every panel by default and re-test; still failing 3 h after the footage: SAM_ENGINE=fal; fal failing too: One Walk"),
    ("E2.completion", "E2", "p50 completion under 120 s at 400 frames and under 180 s at 600", "fewer frames per request (10 fps) or fewer lanes"),
    ("E2.wire_parse", "E2", "wire parse under 5 s per lane", "profile the reader before the build"),
    ("E2.decode", "E2", "keyframe decodes under 60 s per video on the M4 Max", "the Node sidecar of step 5"),
    ("E2.no_429_loss", "E2", "no request lost to 429 after backoff", "priority waves of 6"),
    ("E2.concurrency", "E2", "16 concurrent accepted", "priority waves of 6"),
    ("E2.zero_matches", "E2", "a completed lane with no delta reads as zero_matches, never incomplete", "fix the reader before any coverage state ships"),
    ("E2.frames_billed", "E2", "usage.video_frames_processed equals frames_sent, the difference logged", "bill from the API's count; the method page notes the difference"),
    ("E2.oracle", "E2", "the wire reader agrees with meta_sam_parser on one lane (boxes, ids, c, frames)", "fix the reader"),
    ("E2.overlay", "E2", "raster placement checked by overlay on 3 frames", "fix the paste offset before keyframes"),
    ("E2.prop_both", "E2", "one text lane with propagation_direction both: accepted, ignored or a 400", "-"),
    ("E2.variance", "E2", "'car door' three times for run-to-run variance", "-"),
    ("E3.states", "E3", "states agree across the three 'car door' runs on >= 11 of 12 panels, c basis", "area-only states; the 25th-percentile rule is off"),
    ("E3.c_semantics", "E3", "c on first, mid and propagated frames per noun; c_present_share; whether c varies per frame", "-"),
    ("E4.joins", "E4", "the bridge joins the fragments", "the anchor pass stays off"),
    ("E4.negative_point", "E4", "the negative glass point lowers glass overlap", "keep the bridge, drop the negative point"),
    ("E5.demo_mark", "E5", "the prompted id returns a box on >= 80% of the 90 frames and lies inside the panel on >= 80% of complete-panel frames with RMS drift under 0.08", "see E5.kill"),
    ("E5.control", "E5", "the reflection drifts over 0.16 with the sign test over 80% on >= 8 pairs", "the moment loses its reflection inset"),
    ("E5.kill", "E5", "kill when both demo-mark clips fail on both clip types", "C5 degrades to multi-keyframe Spark consistency; the tape becomes the tracked object; the drift number leaves the README"),
    ("E5.carried_moments", "E5", "the carried-moments check of 5.5 on 5 frames of one clip, under 0.02 units", "decode the panel mask on every clip frame"),
    ("E5.panned", "E5", "inconclusive share on the deliberately panned pass", "over half: the caption gains 'when the camera moves' and the README calls the test a bonus"),
    ("E5.scuff_text", "E5", "'scuff' as a text prompt once", "-"),
    ("E6.sigma", "E6", "sigma <= 2% of crop width (about 0.07 units) at effort low", "2 to 3% with medium meeting 2%: medium becomes the default; over 3% at both: tau_pre past the 0.4 cap, hoods and corners go needs your eye"),
    ("E6.jaccard", "E6", "median Jaccard of the mark sets >= 0.7 at low", "dedup, count union and the second look carry more weight; README says so"),
    ("E6.no_truncation", "E6", "no completed call truncated at 4000", "max_output_tokens 8000 by default"),
    ("E6.truncation_signal", "E6", "one call at max_output_tokens 600 shows status incomplete, reason max_output_tokens", "-"),
    ("E6.second_look_valid", "E6", "the second look returns valid JSON", "second look disabled, README says so"),
    ("E6.second_look_control", "E6", "the second look says unsure or different on the control", "second look disabled, README says so"),
    ("E6.false_marks", "E6", "false marks on the clean crop", "-"),
    ("E6.targets", "E6", "error on the hand-measured targets", "-"),
    ("E7.count", "E7", "count mode finds a target the box pass missed on the 3 crops; kill when it finds none", "count mode cut"),
    ("E8.sigma_frame", "E8", "sigma_frame under 0.05 from one 'door handle' video request over 10 consecutive frames", "the band widens; tau_fix and tau_pre0 follow"),
    ("E8.cross_take", "E8", "cross-take residual under 0.15 between takes", "the band widens and the method page says why"),
    ("E8.cross_walk", "E8", "residual under 0.25 across walks", "the band widens and the method page says why"),
    ("E9.plate_coverage", "E9", "plate-track coverage >= 90% where the plate is visible, with the 0.5 s carry-over", "the display proxy blurs by keyframe image boxes carried over time; README says the proxy blur is approximate"),
    ("E9.image_checks", "E9", "image checks on 3 keyframes, run on the unblurred frame first, plate extent recorded", "-"),
    ("E10.turns", "E10", "ENDPOINTING turns non-empty, the mode string validated", "-"),
    ("E10.mode_applied", "E10", "no 'mode not applied' (empty turns with a transcript)", "inspect the request JSON before anything else"),
    ("E10.pins", "E10", ">= 80% of noun-naming turns land where SAM shows that panel", "pin to the panel at turn start and print 'about'; the seed uses the turn-start panel"),
    ("E10.diarization", "E10", "DIARIZATION as the comparison", "-"),
    ("E11.naming", "E11", "Spark naming agrees with the SAM vote on 3 of 3 claim photos", "layer 1 ships with the one-tap chooser as the default"),
    ("E11.plate_ruler", "E11", "plate mask over 60 px with c >= 0.5 on a 4K frame", "the ruler is cut"),
    ("E11.wheel_ruler", "E11", "wheel major axis over 120 px", "the ruler is cut"),
    ("E11.ppi", "E11", "ppi from plate and wheel within 15% of each other (estimate)", "-"),
    ("E11.pixelation", "E11", "plate ppi unblurred against the same frame pixelated, over 20% apart (the step 7 order)", "-"),
    ("E11.door_pair", "E11", "the same door whitened from a walk frame and the close photo (r_claim)", "-"),
    ("E12.ots_latency", "E12", "ots stamp under 5 s", "-"),
    ("E12.ots_upgrade", "E12", "whether ots upgrade lands within the evening", "-"),
    ("E12.freetsa", "E12", "the freetsa round trip", "DigiCert's TSA as the fallback with the issuer named"),
    ("E12.expires_after", "E12", "expires_after accepted by the Files API", "-"),
    ("E12.files_delete", "E12", "DELETE /v1/files returns a printable body, stored", "-"),
    ("E12.store_false_sam", "E12", "sam-3.1 accepts store false (the response echo)", "the per-request deletion path of section 8"),
    ("E12.store_false_spark", "E12", "muse-spark-1.3 accepts store false (the response echo)", "the per-request deletion path of section 8"),
    ("E12.responses_delete", "E12", "whether DELETE /v1/responses/{id} exists and what it returns, stored", "-"),
    ("E13.pickup_wall", "E13", "pickup-only check under 3 min keeps 'check coverage before you drive off'", "3 to 6 min: 'before you leave the lot'; over 6 min the words leave"),
    ("E13.pair_dollars", "E13", "a pair under $7", "-"),
    ("E13.pickup_dollars", "E13", "a pickup-only check under $3", "-"),
    ("E14.capture", "E14", "capture path (a) at 60 fps", "path (b), the Playwright frame-stepped render, then a 30 fps timeline"),
    ("E14.browsers", "E14", "@meta-sam/video on the H.264 proxy in Safari 18 and on the iPhone; the chunk upload from the phone", "overlay.mp4 as the Safari default"),
    ("E15.lane_bytes", "E15", "the 14 display nouns under about 30 MB per pair (estimate)", "lanes downsampled to every other frame for the judge path, then boxes plus outlines only"),
    ("E15.worker", "E15", "no main-thread task over 50 ms while the Web Worker parses", "-"),
    ("E16.s1", "E16", "Spark Open on Baseten: sigma <= 2% and >= 90% valid JSON first try; the second look valid", "S1 cut"),
    ("E17.timeline", "E17", ">= 70% of panels within 1 s of SAM's windows", "S2 dropped"),
    ("E18.note_prefill", "E18", "5 of 5 fields agree on 2 of 2 synthetic agreements", "S2b dropped"),
    ("E19.fal", "E19", "the fal adapter round-trips its recorded fixture", "-"),
)
CRITERION_IDS = tuple(c[0] for c in CRITERIA)

PLAN_TEMPLATE: dict[str, Any] = {
    "_note": "Paths are relative to this file. Delete any key you do not have; run_d1 picks it from gate_sam's lanes. Boxes are x1 y1 x2 y2 in source pixels (the 4K frame after rotation). Roles are pickup or return. A staysput entry on the pickup or return walk gets gate_sam's panel, plate and face lanes; an entry on another file needs panel_start and panel_end (the panel box on the clip's first and last frame) or it is refused, and its clip is deleted at the end because no plate or face track blurred it.",
    "pickup": "pickup_take1.mov",
    "return": "return.mov",
    "take2": "pickup_take2.mov",
    "narration": "pickup_take1.mov",
    "tire_code": "225/60R16",
    "crops": [
        {"name": "mark_door", "source": "pickup_take1.mov", "t": 12.4, "crop": [1500, 700, 2900, 1700], "panel": "left_front_door", "clean": False, "targets": [[640, 410]]},
        {"name": "bumper", "source": "pickup_take1.mov", "t": 3.1, "crop": [900, 1300, 2900, 2000], "panel": "front_bumper", "clean": False, "targets": []},
        {"name": "clean_door", "source": "pickup_take1.mov", "t": 30.0, "crop": [1400, 650, 2800, 1650], "panel": "right_rear_door", "clean": True, "targets": []},
    ],
    "second_look": ["look_pickup_mark.jpg", "look_return_mark.jpg"],
    "second_look_control": ["look_pickup_mark.jpg", "look_return_reflection.jpg"],
    "claims": ["claim_1.jpg", "claim_2.jpg", "claim_3.jpg"],
    "ruler_frame": {"source": "pickup_take1.mov", "t": 8.0},
    "door_pair": ["walk_frame.jpg", "close_photo.jpg"],
    "staysput": [
        {"src": "pickup_take1.mov", "pts": 12.4, "mark": [2140, 1080, 2210, 1130], "panel": [1500, 700, 2900, 1700], "label": "demo_mark", "role": "pickup", "panel_noun": "car door", "scuff_text": True},
        {"src": "return.mov", "pts": 14.0, "mark": [2000, 900, 2100, 960], "panel": [1400, 600, 2800, 1600], "label": "tape", "role": "return", "panel_noun": "car door"},
        {"src": "return.mov", "pts": 22.0, "mark": [1700, 500, 1780, 560], "panel": [1000, 300, 2900, 1200], "label": "hood_reflection", "role": "return", "panel_noun": "hood", "control": True},
        {"src": "pickup_panned.mov", "pts": 4.0, "mark": [2140, 1080, 2210, 1130], "panel": [1500, 700, 2900, 1700], "panel_start": [1450, 720, 2950, 1690], "panel_end": [1560, 690, 2840, 1710], "label": "demo_mark_panned", "role": "pickup", "panel_noun": "car door", "panned": True},
    ],
}


# ------------------------------------------------------------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------------------------------------------------------------


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def fmt(value: Any, places: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{places}f}"
    return str(value)


def pct(value: float | None, places: int = 1) -> str:
    return "-" if value is None else f"{100.0 * value:.{places}f}%"


def ascii_text(text: str) -> str:
    """ASCII only: common typographic characters mapped, anything else becomes '?'."""
    table = {chr(0x2018): "'", chr(0x2019): "'", chr(0x201C): '"', chr(0x201D): '"', chr(0x2013): "-", chr(0x2014): "-", chr(0x2212): "-", chr(0x2026): "...", chr(0xA0): " ", chr(0xD7): "x"}
    return "".join(table.get(ch, ch) for ch in text).encode("ascii", "replace").decode("ascii")


def cell(text: Any) -> str:
    """One markdown table cell: no pipe, no newline, ASCII."""
    return ascii_text(str(text)).replace("|", "/").replace("\n", " ").strip()


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rel(path: Path | str | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(Path(path).resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


# ------------------------------------------------------------------------------------------------------------------
# The run context: inputs, plan, budget, spend
# ------------------------------------------------------------------------------------------------------------------


@dataclass
class Ctx:
    args: argparse.Namespace
    th: dict
    out: Path
    prefix: str
    dry: bool
    budget: float
    plan: dict[str, Any]
    inputs: dict[str, Path | None]
    results: dict[str, Any]


def load_plan(path: str | None) -> dict[str, Any]:
    """The plan JSON with every path made absolute against the plan's own folder."""
    if not path:
        return {}
    plan_path = Path(path).resolve()
    plan = read_json(plan_path)
    if not isinstance(plan, dict):
        raise SystemExit(f"{plan_path}: the plan must be a JSON object")
    base = plan_path.parent

    def absolute(value: Any) -> Any:
        return str((base / value).resolve()) if isinstance(value, str) and value and not Path(value).is_absolute() else value

    for key in ("pickup", "return", "take2", "narration"):
        if plan.get(key):
            plan[key] = absolute(plan[key])
    for key in ("second_look", "second_look_control", "claims", "door_pair"):
        if isinstance(plan.get(key), list):
            plan[key] = [absolute(v) for v in plan[key]]
    if isinstance(plan.get("ruler_frame"), dict):
        plan["ruler_frame"]["source"] = absolute(plan["ruler_frame"].get("source"))
    for spec in plan.get("crops") or []:
        spec["source"] = absolute(spec.get("source"))
        if spec.get("mask"):
            spec["mask"] = absolute(spec["mask"])
    for entry in plan.get("staysput") or []:
        entry["src"] = absolute(entry.get("src"))
        if entry.get("panel_lane"):
            entry["panel_lane"] = absolute(entry["panel_lane"])
        for lane in entry.get("privacy_lanes") or []:
            if isinstance(lane, dict) and lane.get("path"):
                lane["path"] = absolute(lane["path"])
        if entry.get("privacy_lanes"):
            entry["privacy_lanes"] = [absolute(v) if isinstance(v, str) else v for v in entry["privacy_lanes"]]
        if str(entry.get("role") or "pickup") not in ("pickup", "return"):
            raise SystemExit(f"plan staysput entry {entry.get('label')}: role must be pickup or return")
    plan["_path"] = str(plan_path)
    return plan


def resolve_inputs(args: argparse.Namespace, plan: dict[str, Any], dry: bool) -> dict[str, Path | None]:
    if dry:
        return {"pickup": FIXTURES["clip"], "return": FIXTURES["clip_return"], "take2": None, "narration": None}

    def pick(name: str, flag: str | None) -> Path | None:
        value = flag or plan.get(name)
        return Path(value).expanduser().resolve() if value else None

    return {
        "pickup": pick("pickup", args.pickup),
        "return": pick("return", args.return_),
        "take2": pick("take2", args.take2),
        "narration": pick("narration", args.narration),
    }


def input_problems(inputs: dict[str, Path | None]) -> list[str]:
    """Any pickup, return, take 2 or narration file that is missing or that ffprobe cannot read as a video, one line each.
    Any path works (data/raw, data/eval_internal, a phone export elsewhere, MOV, MP4, WebM, MPEG-4 Part 2): the gates
    make their own H.264 proxies."""
    problems = []
    for role, path in inputs.items():
        if path is None:
            continue
        if not path.is_file():
            problems.append(f"{role}: no such file: {path}")
            continue
        try:
            info = gate_sam.probe_media(path)
        except Exception as exc:  # noqa: BLE001 - say what is wrong with the file instead of failing inside a gate
            problems.append(f"{role}: ffprobe cannot read {path} as a video ({type(exc).__name__}: {exc})")
            continue
        if role != "narration" and not info.get("duration_s"):
            problems.append(f"{role}: {path} has no video duration")
    return problems


def ledger_files(ctx: Ctx) -> list[Path]:
    """Where the spend is read: the real ledger, or every dry-run ledger under the run folder."""
    if ctx.dry:
        return sorted(p for p in ctx.out.rglob("ledger*.jsonl") if p.is_file())
    return [Path(ledger.LEDGER_PATH)]


def spend(ctx: Ctx) -> float:
    """Dollars of the ledger rows whose run_id starts with this D1 run's prefix."""
    total = 0.0
    for path in ledger_files(ctx):
        for row in ledger.rows(path):
            if str(row.get("run_id") or "").startswith(ctx.prefix):
                total += float(row.get("usd") or 0.0)
    return round(total, 6)


def duration_s(path: Path | None, default: float = 40.0) -> float:
    if path is None or not Path(path).exists():
        return default
    try:
        return float(gate_sam.probe_media(Path(path))["duration_s"]) or default
    except Exception:  # noqa: BLE001 - a plan printout must not die on a bad file
        return default


def sweep_fps(th: dict) -> tuple[list[float], float | None]:
    """The E2 order of section 16 item 2: 15 then 10, and 30 only when 15 fails."""
    fallbacks = [float(f) for f in th["media"]["proxy_fps_fallbacks"]]
    first = [float(th["media"]["proxy_fps"])] + fallbacks[:1]
    return first, (fallbacks[1] if len(fallbacks) > 1 else None)


def planned_usd(ctx: Ctx, step: str) -> float:
    """What a step plans to spend, priced from thresholds and section 11 (retries and the 30 fps fallback excluded)."""
    th = ctx.th
    if step == "sam":
        fps_list, _ = sweep_fps(th)
        lanes = len(LANE_NOUNS) + 2 + 1 + 1  # the 16 lanes, two more 'car door' runs, the handle lane, the prop-both lane
        total = 0.0
        for role in ("pickup", "return"):
            if ctx.inputs.get(role) is None:
                continue
            seconds = duration_s(ctx.inputs.get(role))
            total += sum(gate_sam.planned_dollars(int(round(seconds * fps)), lanes, th) for fps in fps_list)
        return round(total + 0.3, 4)  # bridges, plate checks, handle keyframes, the timeline
    if step == "spark":
        crops = max(3, len(ctx.plan.get("crops") or []))
        claims = len(ctx.plan.get("claims") or []) or crops
        calls = crops * 2 * ctx.args.n + crops + 1 + 2 + claims * 3 + 2
        images = 2 * crops + claims * 16 + 3 + 4
        return round(calls * SPARK_CALL_USD + images * SAM_IMAGE_USD, 4)
    if step == "staysput":
        entries = max(2, len(ctx.plan.get("staysput") or []))
        return round(entries * (2 * CLIP_USD + 3 * SAM_IMAGE_USD) + CLIP_USD, 4)
    if step == "voice":
        seconds = duration_s(ctx.inputs.get("narration") or ctx.inputs.get("pickup"))
        return round(2 * math.ceil(seconds) * th["cost"]["voice_usd_per_hour"] / 3600.0, 4)
    return 0.0


# ------------------------------------------------------------------------------------------------------------------
# E0 and E1 (in this process)
# ------------------------------------------------------------------------------------------------------------------


TOOL_VERSIONS = (
    ("ots", ["ots", "--version"]),
    ("modal", ["modal", "--version"]),
    ("railway", ["railway", "--version"]),
    ("node", ["node", "--version"]),
    ("ffmpeg", ["ffmpeg", "-hide_banner", "-version"]),
    ("ffprobe", ["ffprobe", "-hide_banner", "-version"]),
    ("openssl", ["openssl", "version"]),
)
TOOL_AUTH = (("modal token", ["modal", "profile", "current"]), ("railway whoami", ["railway", "whoami"]))
INSTALL_LINE = "uv pip install opentimestamps-client modal (into walkaround/.venv; Sanat runs it, this runner never installs)"


def run_tool(cmd: list[str], timeout: float = 30.0) -> dict[str, Any]:
    if shutil.which(cmd[0]) is None:
        return {"present": False, "answered": False, "error": f"{cmd[0]} not on PATH"}
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {"present": True, "answered": False, "error": f"{type(exc).__name__}: {exc}"}
    text = (proc.stdout or proc.stderr or "").strip()
    return {"present": True, "answered": proc.returncode == 0, "returncode": proc.returncode, "first_line": ascii_text(text.splitlines()[0])[:160] if text else "", "_text": text}


def check_tools(*, auth: bool) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, cmd in TOOL_VERSIONS:
        row = run_tool(cmd)
        if name == "ffmpeg":
            row["libx265"] = "--enable-libx265" in row.get("_text", "")
        row.pop("_text", None)
        out[name] = row
    if auth:
        for name, cmd in TOOL_AUTH:
            row = run_tool(cmd)
            row.pop("_text", None)
            out[name] = row
    missing = [k for k, v in out.items() if not v.get("answered")]
    return {"tools": out, "missing": missing, "install_line": INSTALL_LINE if {"ots", "modal"} & set(missing) else None}


def hash_inputs(ctx: Ctx) -> dict[str, Any]:
    """sha256 of every original before any conversion, with the camera time ffprobe reports."""
    rows: dict[str, Any] = {}
    for role, path in ctx.inputs.items():
        if path is None:
            continue
        if not Path(path).exists():
            rows[role] = {"path": str(path), "error": "file not found"}
            continue
        t0 = time.monotonic()
        digest = gate_sam.sha256_file(Path(path))
        info = gate_sam.probe_media(Path(path))
        rows[role] = {"path": str(path), "sha256": digest, "bytes": Path(path).stat().st_size, "hash_s": round(time.monotonic() - t0, 3), "hashed_at": utc_now(), "camera_time": info.get("creation_time"), "width": info["width"], "height": info["height"], "fps": info["fps"], "duration_s": info["duration_s"]}
    return rows


def proxy_dir(ctx: Ctx) -> Path:
    """Live: gate_sam's proxy folder, so its sweep reuses these encodes. Dry run: E0's own folder."""
    return (ctx.out / "E0" / "proxies") if ctx.dry else (ctx.out / "sam" / "proxies")


def e0_proxies(ctx: Ctx) -> list[dict[str, Any]]:
    src = ctx.inputs.get("pickup")
    if src is None:
        return []
    m = ctx.th["media"]
    fps_list, fallback = sweep_fps(ctx.th)
    rows = []
    for fps in fps_list + ([fallback] if fallback else []):
        try:
            info = gate_sam.make_proxy(Path(src), proxy_dir(ctx) / f"pickup_{int(fps)}fps.mp4", fps=fps, max_edge=m["proxy_max_edge"], preset=m["proxy_preset"], crf=m["proxy_crf"])
            rows.append({"fps": fps, "path": info["path"], "frames": info["frames"], "width": info["width"], "height": info["height"], "encode_s": info["encode_s"], "sha256": info["sha256"]})
        except Exception as exc:  # noqa: BLE001
            rows.append({"fps": fps, "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
    return rows


def ots_stamp(path: Path) -> dict[str, Any]:
    seal = gate_sam.optional_module("seal")
    t0 = time.monotonic()
    try:
        if seal is not None and gate_sam.implemented(seal, "stamp_ots"):
            ots_path = Path(seal.stamp_ots(path))
            how = "seal.stamp_ots"
        elif shutil.which("ots"):
            subprocess.run(["ots", "stamp", str(path)], check=True, capture_output=True, text=True, timeout=120)
            ots_path = Path(str(path) + ".ots")
            how = "ots stamp"
        else:
            return {"file": path.name, "error": "ots not installed", "install_line": INSTALL_LINE}
    except Exception as exc:  # noqa: BLE001 - the error is the finding
        return {"file": path.name, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    return {"file": path.name, "how": how, "seconds": round(time.monotonic() - t0, 3), "ots_path": str(ots_path), "exists": ots_path.exists()}


def tsa_stamp(path: Path) -> dict[str, Any]:
    """RFC 3161 token from freetsa: seal.stamp_tsa when built, else openssl ts -query plus one POST."""
    seal = gate_sam.optional_module("seal")
    t0 = time.monotonic()
    try:
        if seal is not None and gate_sam.implemented(seal, "stamp_tsa"):
            tsq, tsr, tsr_time, serial = seal.stamp_tsa(path)
            return {"how": "seal.stamp_tsa", "seconds": round(time.monotonic() - t0, 3), "tsr": str(tsr), "time": tsr_time, "serial": serial, "ok": True}
        import httpx

        tsq = Path(str(path) + ".tsq")
        tsr = Path(str(path) + ".tsr")
        subprocess.run(["openssl", "ts", "-query", "-data", str(path), "-no_nonce", "-sha256", "-cert", "-out", str(tsq)], check=True, capture_output=True, timeout=30)
        reply = httpx.post("https://freetsa.org/tsr", content=tsq.read_bytes(), headers={"Content-Type": "application/timestamp-query"}, timeout=20.0)
        reply.raise_for_status()
        tsr.write_bytes(reply.content)
        text = subprocess.run(["openssl", "ts", "-reply", "-in", str(tsr), "-text"], capture_output=True, text=True, timeout=30).stdout
        when = next((line.split(":", 1)[1].strip() for line in text.splitlines() if line.strip().startswith("Time stamp:")), None)
        return {"how": "openssl ts plus a POST to freetsa", "seconds": round(time.monotonic() - t0, 3), "tsr": str(tsr), "time": when, "bytes": len(reply.content), "ok": bool(when)}
    except Exception as exc:  # noqa: BLE001
        return {"seconds": round(time.monotonic() - t0, 3), "error": f"{type(exc).__name__}: {str(exc)[:300]}", "ok": False}


def seal_smoke(folder: Path) -> dict[str, Any]:
    """Two small text files stamped with ots, one through freetsa (free calls, no model)."""
    folder.mkdir(parents=True, exist_ok=True)
    files = []
    for name in ("smoke_a.txt", "smoke_b.txt"):
        path = folder / name
        path.write_text(f"walkaround D1 seal smoke {name} {utc_now()}\n", encoding="ascii")
        files.append(path)
    ots = [ots_stamp(p) for p in files]
    return {"ots": ots, "ots_files": sum(1 for r in ots if r.get("exists")), "freetsa": tsa_stamp(files[0])}


def step_e0(ctx: Ctx) -> dict[str, Any]:
    row: dict[str, Any] = {"tools": check_tools(auth=not ctx.dry)}
    row["originals"] = hash_inputs(ctx)  # before any proxy is made
    row["proxies"] = e0_proxies(ctx)
    row["smoke"] = {"note": "dry run: no network call"} if ctx.dry else seal_smoke(ctx.out / "E0")
    row["repo"] = {"walkaround_git_dir": (REPO / ".git").is_dir(), "note": "checked as a folder; this runner runs no git command"}
    return {"status": "done", **row}


def step_e1(ctx: Ctx) -> dict[str, Any]:
    """Fresh encodes of the pickup walk at medium and veryfast (deleted after timing); the upload and container
    halves of E1 need the deploy and a phone."""
    src = ctx.inputs.get("pickup")
    if src is None:
        return {"status": "not run", "reason": "no pickup walk"}
    m = ctx.th["media"]
    fps = float(m["proxy_fps"])
    rows = {}
    for preset in ("medium", "veryfast"):
        dst = ctx.out / "E1" / f"pickup_{int(fps)}fps_{preset}.mp4"
        dst.unlink(missing_ok=True)
        try:
            info = gate_sam.make_proxy(Path(src), dst, fps=fps, max_edge=m["proxy_max_edge"], preset=preset, crf=m["proxy_crf"])
            rows[preset] = {"encode_s": info["encode_s"], "frames": info["frames"], "bytes": dst.stat().st_size, "size": [info["width"], info["height"]]}
        except Exception as exc:  # noqa: BLE001
            rows[preset] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        dst.unlink(missing_ok=True)  # an unblurred intermediate; only the time is kept
    source = ctx.results.get("steps", {}).get("E0", {}).get("originals", {}).get("pickup", {})
    return {"status": "done", "source": {k: source.get(k) for k in ("width", "height", "fps", "duration_s", "bytes")}, "proxy_mac": rows, "upload": {"note": "needs the deploy and the chunk route; run from the Mac on wifi and the phone on LTE"}, "container": {"note": "needs the Railway container"}}


# ------------------------------------------------------------------------------------------------------------------
# Gates (each its own process)
# ------------------------------------------------------------------------------------------------------------------


def gate_env(ctx: Ctx) -> dict[str, str]:
    env = dict(os.environ)
    if ctx.dry:
        env.pop("LIVE", None)
        env["WALKAROUND_RUNS"] = str(ctx.out / "dryrun_runs")
    return env


USABLE = ("done", "partial")  # a partial step's gate JSON holds the stages that finished before it stopped


def run_gate(ctx: Ctx, step: str, script: str, argv: list[str]) -> dict[str, Any]:
    """One gate as its own process. The gates write their JSON after every paid stage, so a gate that exits non-zero
    with a JSON on disk is 'partial' and its finished stages still reach the memo; a JSON from an earlier attempt is
    moved aside first so it never reads as this one's."""
    out = ctx.out / step
    out.mkdir(parents=True, exist_ok=True)
    data_path = out / GATE_JSON[step]
    if data_path.exists():
        data_path.replace(data_path.with_name(data_path.stem + ".previous.json"))
    cmd = [sys.executable, str(SCRIPTS / script), *argv, "--out", str(out)]
    log = out / "gate.log"
    t0 = time.monotonic()
    try:
        with log.open("w", encoding="utf-8") as fh:
            proc = subprocess.run(cmd, cwd=str(REPO), env=gate_env(ctx), stdout=fh, stderr=subprocess.STDOUT, timeout=ctx.args.step_timeout)
        code = proc.returncode
    except subprocess.TimeoutExpired:
        code = -9
    seconds = round(time.monotonic() - t0, 3)
    row: dict[str, Any] = {"cmd": [Path(cmd[1]).name] + [rel(a, REPO) if isinstance(a, str) and a.startswith(str(REPO)) else a for a in cmd[2:]], "returncode": code, "seconds": seconds, "log": rel(log, ctx.out), "json": rel(data_path, ctx.out)}
    if code == 0 and data_path.exists():
        row["status"] = "done"
    else:
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-15:] if log.exists() else []
        reason = f"exit {code}" + (" (timeout)" if code == -9 else "")
        if data_path.exists():
            reason += "; the gate JSON holds the stages that finished"
        row.update({"status": "partial" if data_path.exists() else "failed", "reason": reason, "log_tail": [ascii_text(t)[:300] for t in tail]})
    return row


def gate_data(ctx: Ctx, step: str) -> dict[str, Any] | None:
    path = ctx.out / step / GATE_JSON[step]
    if ctx.results.get("steps", {}).get(step, {}).get("status") not in USABLE or not path.exists():
        return None
    return read_json(path)


def sam_args(ctx: Ctx) -> list[str]:
    if ctx.dry:
        info = gate_sam.probe_media(FIXTURES["clip"])
        return ["--fixture", str(FIXTURES["sam_text"]), "--bridge-fixture", str(FIXTURES["sam_box"]), "--clip", str(FIXTURES["clip"]), "--fps", f"{info['fps']:g}", "--nouns", *[n.value for n in LANE_NOUNS], "--run-prefix", ctx.prefix, "--budget", str(ctx.budget)]
    fps_list, fallback = sweep_fps(ctx.th)
    argv = ["--pickup", str(ctx.inputs["pickup"]), "--fps", *[f"{f:g}" for f in fps_list], "--fps-fallback", f"{fallback:g}" if fallback else "0", "--repeats", "3", "--handle", "--bridge", "--plate-checks", "--timeline", "--delete-response-probe", "--run-prefix", ctx.prefix, "--budget", str(ctx.budget)]
    if ctx.inputs.get("return"):
        argv += ["--return", str(ctx.inputs["return"])]
    if ctx.inputs.get("take2"):
        argv += ["--take2", str(ctx.inputs["take2"])]
    return argv


# ------------------------------------------------------------------------------------------------------------------
# Picks from gate_sam's lanes (used when the plan names nothing)
# ------------------------------------------------------------------------------------------------------------------


def chosen_sweep(sam: dict[str, Any] | None, th: dict) -> tuple[dict[str, Any] | None, float | None]:
    """(the sweep the E2 criteria read, the fps chosen): 15 when it passes tracks (section 16 item 2), else the
    highest passing fps (the 30 fps fallback), else None with the 15 fps sweep (or the first) for the rest."""
    sweeps = [s for s in (sam or {}).get("sweeps", []) if "evaluation" in s]
    if not sweeps:
        return None, None
    default = float(th["media"]["proxy_fps"])
    by_fps = {float(s["fps"]): s for s in sweeps}
    passing = [s for s in sweeps if s["evaluation"]["tracks"]["pass"]]
    if default in by_fps and by_fps[default] in passing:
        return by_fps[default], default
    if passing:
        best = max(passing, key=lambda s: float(s["fps"]))
        return best, float(best["fps"])
    return by_fps.get(default) or sweeps[0], None


def pad_box(box: list[float], size: tuple[int, int], frac: float) -> list[int]:
    w, h = box[2] - box[0], box[3] - box[1]
    x1, y1 = max(0, int(box[0] - frac * w)), max(0, int(box[1] - frac * h))
    x2, y2 = min(size[0], int(math.ceil(box[2] + frac * w))), min(size[1], int(math.ceil(box[3] + frac * h)))
    return [x1, y1, x2, y2]


def auto_crops(ctx: Ctx, sam: dict[str, Any] | None) -> tuple[list[dict[str, Any]], list[str]]:
    """E6 crops from gate_sam's suggestions: the largest door and bumper frames of the pickup walk and a door from
    the return walk (or the pickup's second door). None is labelled clean, so false marks are not counted."""
    sweep, _ = chosen_sweep(sam, ctx.th)
    if sweep is None:
        return [], ["no gate_sam sweep to pick crops from"]
    pad = ctx.th["panels"]["crop_pad_frac"]
    specs: list[dict[str, Any]] = []
    notes: list[str] = []

    def spec(name: str, role: str, row: dict[str, Any], noun: str) -> dict[str, Any] | None:
        src = ctx.inputs.get(role)
        role_out = sweep["roles"].get(role) or {}
        if src is None:
            return None
        info = ctx.results.get("steps", {}).get("E0", {}).get("originals", {}).get(role) or gate_sam.probe_media(Path(src))
        lane = (role_out.get("lanes") or {}).get(f"D1 E2 {noun}") or {}
        return {
            "name": name, "source": str(src), "t": row["t"], "crop": pad_box(row["box_src"], (int(info["width"]), int(info["height"])), pad),
            "panel": noun, "clean": False, "targets": [], "auto": True, "role": role, "panel_box_src": row["box_src"],
            "object_id": row.get("object_id"), "lane_file": lane.get("lane_file"), "lane_fps": float(sweep["fps"]), "lane_size": lane.get("media_size"),
        }

    pick = (sweep["roles"].get("pickup") or {}).get("suggestions") or {}
    back = (sweep["roles"].get("return") or {}).get("suggestions") or {}
    doors, bumpers, doors_back = pick.get("car door") or [], pick.get("bumper") or [], back.get("car door") or []
    for name, role, rows, noun in (("auto_door", "pickup", doors[:1], "car door"), ("auto_bumper", "pickup", bumpers[:1], "bumper"), ("auto_door_return", "return", doors_back[:1], "car door")):
        for row in rows:
            made = spec(name, role, row, noun)
            if made:
                specs.append(made)
    if not doors_back and len(doors) > 1:
        made = spec("auto_door_2", "pickup", doors[1], "car door")
        if made:
            specs.append(made)
    if not specs:
        notes.append("gate_sam suggested no door or bumper frame")
    return specs, notes


def auto_ruler(ctx: Ctx, sam: dict[str, Any] | None) -> dict[str, Any] | None:
    sweep, _ = chosen_sweep(sam, ctx.th)
    rows = ((sweep or {}).get("roles", {}).get("pickup") or {}).get("suggestions", {}).get("license plate") or []
    if not rows or ctx.inputs.get("pickup") is None:
        return None
    return {"source": str(ctx.inputs["pickup"]), "t": rows[0]["t"], "auto": True}


def spark_args(ctx: Ctx, sam: dict[str, Any] | None) -> tuple[list[str] | None, dict[str, Any]]:
    """gate_spark's arguments plus what was picked automatically; None when there is no crop to run on."""
    picked: dict[str, Any] = {"notes": []}
    if ctx.dry:
        crops, notes = auto_crops(ctx, sam)
        picked.update({"crops": crops, "notes": notes})
        argv = ["--fixture", str(FIXTURES["spark"]), "--auto-looks", "--claims-from-crops", "--n", str(ctx.args.n), "--run-prefix", ctx.prefix, "--budget", str(ctx.budget)]
        if crops:
            path = ctx.out / "spark" / "auto_crops.json"
            gate_sam.write_json(path, crops)
            argv += ["--crops", str(path)]
        return argv, picked
    crops = ctx.plan.get("crops") or []
    if not crops:
        crops, notes = auto_crops(ctx, sam)
        picked.update({"crops": crops, "notes": notes})
    if not crops:
        return None, picked
    path = ctx.out / "spark" / "crops.json"
    gate_sam.write_json(path, crops)
    argv = ["--crops", str(path), "--efforts", "low", "medium", "--n", str(ctx.args.n), "--count", "--truncate", "600", "--note-prefill-synthetic", "--run-prefix", ctx.prefix, "--budget", str(ctx.budget)]
    if ctx.plan.get("second_look"):
        argv += ["--second-look", *ctx.plan["second_look"][:2]]
    if ctx.plan.get("second_look_control"):
        argv += ["--second-look-control", *ctx.plan["second_look_control"][:2]]
    if not (ctx.plan.get("second_look") and ctx.plan.get("second_look_control")):
        argv.append("--auto-looks")
        picked["notes"].append("second look: auto pairs (weak), the plan names no pair")
    if ctx.plan.get("claims"):
        argv += ["--claim", *ctx.plan["claims"]]
    else:
        argv.append("--claims-from-crops")
        picked["notes"].append("claims: the blurred crops stand in for staged claim photos")
    ruler = ctx.plan.get("ruler_frame") or auto_ruler(ctx, sam)
    if ruler:
        argv += ["--ruler-frame", str(ruler["source"]), "--ruler-t", str(ruler["t"])]
        picked["ruler"] = ruler
    if ctx.plan.get("tire_code"):
        argv += ["--tire-code", str(ctx.plan["tire_code"])]
    if ctx.plan.get("door_pair"):
        argv += ["--door-pair", *ctx.plan["door_pair"][:2]]
    return argv, picked


def mark_clusters(rows_by_run: list[list[dict[str, Any]]], radius: float) -> list[list[dict[str, Any]]]:
    """Greedy clusters of mark rows across repeated calls (at most one mark per call per cluster)."""
    clusters: list[list[dict[str, Any]]] = []
    for run_index, rows in enumerate(rows_by_run):
        used: set[int] = set()
        for row in rows:
            best, best_d = None, radius
            for k, cluster in enumerate(clusters):
                if k in used:
                    continue
                cx = statistics.mean(m["px"][0] for m in cluster)
                cy = statistics.mean(m["px"][1] for m in cluster)
                d = math.dist((cx, cy), row["px"])
                if d <= best_d:
                    best, best_d = k, d
            if best is None:
                clusters.append([{**row, "_run": run_index}])
                used.add(len(clusters) - 1)
            else:
                clusters[best].append({**row, "_run": run_index})
                used.add(best)
    return clusters


def glare_box(rgb: np.ndarray, box: list[float], th: dict) -> list[float] | None:
    """The largest bright, unsaturated blob inside a box (the hood glare rule of 5.9 item 4), at least min_bbox_px."""
    import cv2

    h, w = rgb.shape[:2]
    x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
    x2, y2 = min(w, int(math.ceil(box[2]))), min(h, int(math.ceil(box[3])))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb[y1:y2, x1:x2]), cv2.COLOR_RGB2HSV)
    mask = ((hsv[:, :, 2] > th["panels"]["glare_v_min"]) & (hsv[:, :, 1] < th["panels"]["glare_s_max"])).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if int(stats[i, cv2.CC_STAT_AREA]) < 20:
        return None
    bx, by, bw, bh = (int(v) for v in stats[i, :4])
    side = th["staysput"]["min_bbox_px_at_4k"] * max(w, h) / 3840.0
    cx, cy = x1 + bx + bw / 2.0, y1 + by + bh / 2.0
    bw, bh = max(bw, side), max(bh, side)
    return [round(cx - bw / 2.0, 1), round(cy - bh / 2.0, 1), round(cx + bw / 2.0, 1), round(cy + bh / 2.0, 1)]


def auto_staysput(ctx: Ctx, sam: dict[str, Any] | None, spark: dict[str, Any] | None) -> tuple[list[dict[str, Any]], list[str]]:
    """E5 entries when the plan names none: the uncapped mark seen in the most low-effort calls on the first auto
    door crop (median box), and the return walk's hood glare as the reflection control."""
    entries: list[dict[str, Any]] = []
    notes: list[str] = []
    sweep, _ = chosen_sweep(sam, ctx.th)
    sigma_frame = None
    for role_out in ((sweep or {}).get("roles") or {}).values():
        sigma_frame = sigma_frame or ((role_out.get("handle") or {}).get("sigma") or {}).get("sigma_frame")
    specs = {s["name"]: s for s in (spark or {}).get("crop_specs") or [] if s.get("auto")}
    crop_row = next((c for c in (spark or {}).get("crops", []) if c.get("name") in specs and specs[c["name"]].get("panel") == "car door" and c.get("efforts")), None)
    if crop_row is None:
        notes.append("no auto door crop in gate_spark's results, so no mark to track")
    else:
        spec = specs[crop_row["name"]]
        effort = "low" if "low" in crop_row["efforts"] else next(iter(crop_row["efforts"]))
        runs = [[m for m in r.get("marks", []) if not m.get("capped")] for r in crop_row["efforts"][effort]["runs"]]
        w, h = crop_row["crop_size"]
        clusters = [c for c in mark_clusters(runs, 0.1 * max(w, h)) if len(c) >= 2]
        if not clusters:
            notes.append(f"{crop_row['name']}: no uncapped mark repeated across calls")
        else:
            best = max(clusters, key=lambda c: (len({m['_run'] for m in c}), sum(1 for m in c if m.get("confidence") == "high")))
            box = [statistics.median(m["bbox_px"][i] for m in best) for i in range(4)]
            ox, oy = spec["crop"][0], spec["crop"][1]
            entries.append({
                "src": spec["source"], "pts": spec["t"], "mark": [round(box[0] + ox, 1), round(box[1] + oy, 1), round(box[2] + ox, 1), round(box[3] + oy, 1)],
                "panel": spec["panel_box_src"], "label": "auto_mark", "role": spec.get("role", "pickup"), "panel_noun": "car door", "scuff_text": True,
                "panel_lane": spec.get("lane_file"), "lane_fps": spec.get("lane_fps"), "lane_size": spec.get("lane_size"), "panel_object_id": spec.get("object_id"),
                "sigma_frame": sigma_frame, "auto": f"{best[0]['type']} seen in {len({m['_run'] for m in best})} of {len(runs)} calls on {crop_row['name']}",
            })
    hood = (((sweep or {}).get("roles") or {}).get("return") or {}).get("suggestions", {}).get("hood") or []
    src = ctx.inputs.get("return")
    if hood and src is not None:
        row = hood[0]
        try:
            rgb = gate_sam.frame_at(Path(src), row["t"])
            glare = glare_box(rgb, row["box_src"], ctx.th)
        except Exception as exc:  # noqa: BLE001
            glare = None
            notes.append(f"hood frame not read: {type(exc).__name__}: {str(exc)[:200]}")
        if glare is not None:
            lane = (((sweep or {}).get("roles") or {}).get("return") or {}).get("lanes", {}).get("D1 E2 hood") or {}
            entries.append({
                "src": str(src), "pts": row["t"], "mark": glare, "panel": row["box_src"], "label": "auto_hood_glare", "role": "return", "panel_noun": "hood", "control": True,
                "panel_lane": lane.get("lane_file"), "lane_fps": float(sweep["fps"]), "lane_size": lane.get("media_size"), "panel_object_id": row.get("object_id"), "sigma_frame": sigma_frame,
                "auto": "the brightest unsaturated blob on the return walk's largest hood frame (5.9 item 4)",
            })
        else:
            notes.append("no glare blob on the return walk's hood frame")
    else:
        notes.append("no hood frame in the return walk for the reflection control")
    return entries, notes


def same_file(a: Any, b: Any) -> bool:
    try:
        return a is not None and b is not None and Path(a).resolve() == Path(b).resolve()
    except (OSError, ValueError):
        return False


def attach_lanes(ctx: Ctx, sam: dict[str, Any] | None, entries: list[dict[str, Any]]) -> list[str]:
    """E5 entries on the pickup or return walk gate_sam swept get that walk's lanes: the panel lane when the entry
    names none (the window follows the panel, 5.9 item 1) and the plate and face lanes (the clip is pixelated from
    tracks, 5.9 item 2). An entry on another file (the panned pass) gets neither and says so."""
    sweep, _ = chosen_sweep(sam, ctx.th)
    if sweep is None:
        return ["E5: no gate_sam sweep, so no lane moves the windows or blurs the clips; entries need panel_start and panel_end"] if entries else []
    notes = []
    for entry in entries:
        role = str(entry.get("role") or "pickup")
        if not same_file(entry.get("src"), ctx.inputs.get(role)):
            notes.append(f"E5 {entry.get('label')}: {Path(str(entry.get('src'))).name} is not the {role} walk gate_sam swept, so no lane moves its window or blurs it from tracks")
            continue
        lanes = ((sweep.get("roles") or {}).get(role) or {}).get("lanes") or {}
        fps = float(sweep["fps"])
        panel = lanes.get(f"D1 E2 {entry.get('panel_noun') or 'car door'}") or {}
        if not entry.get("panel_lane") and panel.get("lane_file"):
            entry.update({"panel_lane": panel["lane_file"], "lane_fps": fps, "lane_size": panel.get("media_size")})
        if not entry.get("privacy_lanes"):
            entry["privacy_lanes"] = [
                {"noun": noun, "path": m.get("lane_file"), "status": m.get("status"), "fps": fps, "size": m.get("media_size")}
                for noun in ("license plate", "face")
                for m in [lanes.get(f"D1 E2 {noun}") or {}]
            ]
    return notes


def staysput_args(ctx: Ctx, sam: dict[str, Any] | None, spark: dict[str, Any] | None) -> tuple[list[str] | None, dict[str, Any]]:
    folder = ctx.out / "staysput"
    folder.mkdir(parents=True, exist_ok=True)
    if ctx.dry:
        entries, notes = auto_staysput(ctx, sam, spark)
        notes += attach_lanes(ctx, sam, entries)
        gate_sam.write_json(folder / "auto_plan.json", entries)
        return ["--fixture", str(FIXTURES["sam_box"]), "--text-fixture", str(FIXTURES["sam_text"]), "--cut-clip", "--clip", str(FIXTURES["clip"]), "--run-prefix", ctx.prefix], {"entries": entries, "notes": notes + ["dry run: the auto plan is written, the fixture runs"]}
    entries = [dict(e) for e in ctx.plan.get("staysput") or []]
    picked: dict[str, Any] = {"notes": []}
    if not entries:
        entries, notes = auto_staysput(ctx, sam, spark)
        picked.update({"entries": entries, "notes": notes})
    if not entries:
        return None, picked
    picked["notes"] = picked.get("notes", []) + attach_lanes(ctx, sam, entries)
    path = folder / "plan.json"
    gate_sam.write_json(path, entries)
    return ["--plan", str(path), "--run-prefix", ctx.prefix, "--budget", str(ctx.budget)], picked


def voice_args(ctx: Ctx) -> list[str] | None:
    pins = ctx.out / "sam" / GATE_JSON["sam"]
    pin_args = ["--pins", str(pins)] if pins.exists() else []
    if ctx.dry:
        return ["--fixture", str(FIXTURES["voice"]), *pin_args, "--run-prefix", ctx.prefix]
    source = ctx.inputs.get("narration") or ctx.inputs.get("pickup")
    if source is None:
        return None
    return [str(source), "--modes", "ENDPOINTING", "DIARIZATION", *pin_args, "--role", "pickup", "--run-prefix", ctx.prefix]


def step_e12(ctx: Ctx) -> dict[str, Any]:
    """ots upgrade on E0's stamps (the hourly task's first try); the other E12 answers come from E0 and the gates."""
    if ctx.dry:
        return {"status": "done", "ots_upgrade": {"note": "dry run: no network call"}}
    smoke = ctx.results.get("steps", {}).get("E0", {}).get("smoke") or {}
    rows = []
    for stamp in smoke.get("ots", []):
        if not stamp.get("exists"):
            continue
        path = Path(stamp["ots_path"])
        seal = gate_sam.optional_module("seal")
        try:
            if seal is not None and gate_sam.implemented(seal, "upgrade_ots"):
                status = seal.upgrade_ots(path)
            else:
                proc = subprocess.run(["ots", "upgrade", str(path)], capture_output=True, text=True, timeout=120)
                status = "confirmed" if proc.returncode == 0 else "pending"
                status += f" ({ascii_text((proc.stdout or proc.stderr).strip().splitlines()[-1])[:160]})" if (proc.stdout or proc.stderr).strip() else ""
        except Exception as exc:  # noqa: BLE001
            status = f"failed: {type(exc).__name__}: {str(exc)[:200]}"
        rows.append({"file": path.name, "status": status, "at": utc_now()})
    return {"status": "done", "ots_upgrade": {"rows": rows} if rows else {"note": "no ots stamp from E0 to upgrade"}}


# ------------------------------------------------------------------------------------------------------------------
# E13 from the measured parts
# ------------------------------------------------------------------------------------------------------------------


def plain_lanes(role_out: dict[str, Any]) -> list[dict[str, Any]]:
    return [m for k, m in (role_out.get("lanes") or {}).items() if m.get("noun") and k == f"D1 E2 {m['noun']}"]


def lane_dollars(rows: list[dict[str, Any]], th: dict) -> float:
    """The lanes' ledger dollars, or priced from frames when a dry run left them at zero."""
    total = 0.0
    for m in rows:
        usd = float(m.get("dollars") or 0.0)
        if usd <= 0:
            frames = int(m.get("video_frames_processed") or m.get("frames_sent") or 0)
            usd = max(frames, th["cost"]["sam_video_min_frames"]) * th["cost"]["sam_video_usd_per_frame"]
        total += usd
    return total


def mean_or(values: list[float], default: float) -> tuple[float, str]:
    vals = [v for v in values if v is not None and v > 0]
    return (statistics.mean(vals), "measured") if vals else (default, "estimate")


def step_e13(ctx: Ctx) -> dict[str, Any]:
    """Pickup-only and pair dollars from measured unit prices times section 5's counts, and the pickup-only wall
    clock from the measured parts. Not an end-to-end run: gate_wallclock.py through the orchestrator replaces it."""
    th = ctx.th
    sam = gate_data(ctx, "sam")
    sweep, fps = chosen_sweep(sam, th)
    if sweep is None:
        return {"status": "not run", "reason": "no gate_sam sweep to read"}
    spark = gate_data(ctx, "spark") or {}
    sp = gate_data(ctx, "staysput") or {}
    voice = gate_data(ctx, "voice") or {}
    roles = sweep.get("roles") or {}
    pickup = roles.get("pickup") or {}
    lane_usd = {role: lane_dollars(plain_lanes(r), th) for role, r in roles.items()}
    pickup_lanes = lane_usd.get("pickup", 0.0)
    pair_lanes = sum(lane_usd.values()) if len(lane_usd) > 1 else 2 * pickup_lanes
    ev = spark.get("evaluation") or {}
    effort = ev.get("effort_default") or th["spark"]["effort"]
    call, call_basis = mean_or([(ev.get("per_effort") or {}).get(effort, {}).get("dollars_mean")], SPARK_CALL_USD)
    count, count_basis = mean_or([c["count_mode"].get("dollars") for c in spark.get("crops", []) if c.get("count_mode")], call)
    look, look_basis = mean_or([v.get("dollars") for v in (spark.get("second_look") or {}).values()], call)
    clip, clip_basis = mean_or([c.get("metrics", {}).get("dollars") for c in sp.get("clips", [])], CLIP_USD)
    bridges = [v.get("dollars") for r in roles.values() for v in ((r.get("anchor_bridge") or {}).get("variants") or {}).values() if isinstance(v, dict)]
    bridge, bridge_basis = mean_or(bridges, 0.036)
    voice_usd, voice_basis = mean_or([m.get("usd") for m in (voice.get("modes") or {}).values() if isinstance(m, dict)], 0.0025)
    img = th["cost"]["sam_image_usd"]

    def item(name: str, n: float, unit: float, basis: str) -> dict[str, Any]:
        return {"item": name, "count": n, "unit_usd": round(unit, 6), "usd": round(n * unit, 4), "basis": basis}

    pair = [
        item("16 lanes per walk, both walks (ledger)", 1, pair_lanes, "measured" if len(lane_usd) > 1 else "measured on one walk, doubled"),
        item("anchor bridges", 12, bridge, bridge_basis),
        item("keyframe blur image checks (78 keyframes x 2)", 156, img, "price"),
        item("door handle image requests", 24, img, "price"),
        item("Stays Put clips", 24, clip, clip_basis),
        item("clip blur checks (24 clips x 2)", 48, img, "price"),
        item("Spark box marks (78 crops)", 78, call, call_basis),
        item("Spark count mode (78 crops)", 78, count, count_basis),
        item("second looks", 12, look, look_basis),
        item("narration seeds", 6, call, call_basis),
        item("voice, both walks", 2, voice_usd, voice_basis),
    ]
    pickup_items = [
        item("16 lanes on the pickup walk (ledger)", 1, pickup_lanes, "measured"),
        item("keyframe blur image checks (39 keyframes x 2)", 78, img, "price"),
        item("Spark box marks (39 crops)", 39, call, call_basis),
        item("Stays Put clips on one walk", 12, clip, clip_basis),
        item("clip blur checks (12 clips x 2)", 24, img, "price"),
    ]
    e0 = ctx.results.get("steps", {}).get("E0", {})
    proxy_s = next((p.get("encode_s") for p in e0.get("proxies", []) if fps and float(p.get("fps", 0)) == fps), None)
    proxy_s = proxy_s or (ctx.results.get("steps", {}).get("E1", {}).get("proxy_mac", {}).get("medium") or {}).get("encode_s")
    proxy_s = proxy_s or (pickup.get("proxy") or {}).get("encode_s")
    upload_s = (pickup.get("upload") or {}).get("upload_s")
    sweep_wall = pickup.get("sweep_wall_s")
    if sweep_wall is None:
        done = [m.get("completed_s") for m in plain_lanes(pickup) if m.get("completed_s") is not None]
        sweep_wall = max(done) if done else None
    parts = {"proxy_encode_s": proxy_s, "files_upload_s": upload_s, "lanes_wall_s": sweep_wall}
    wall = round(sum(v for v in parts.values() if v is not None), 2) if sweep_wall is not None else None
    return {
        "status": "done", "fps": fps or float(sweep["fps"]), "effort": effort,
        "pair_items": pair, "pair_usd": round(sum(i["usd"] for i in pair), 4),
        "pickup_items": pickup_items, "pickup_usd": round(sum(i["usd"] for i in pickup_items), 4),
        "pickup_wall_parts": parts, "pickup_wall_s": wall,
        "note": "measured unit prices times section 5's counts; the browser upload of E1 is not in the wall clock; gate_wallclock.py through the orchestrator replaces this",
    }


# ------------------------------------------------------------------------------------------------------------------
# Criteria
# ------------------------------------------------------------------------------------------------------------------


@dataclass
class Verdict:
    status: str  # pass, fail, not run, info
    value: str = ""
    note: str = ""


def ok(flag: Any, value: str = "", note: str = "") -> Verdict:
    if flag is None:
        return Verdict("not run", value, note or "no data")
    return Verdict("pass" if flag else "fail", value, note)


class Res:
    """Read access to one run's results and the gate JSONs it points at."""

    def __init__(self, results: dict[str, Any], out: Path, th: dict) -> None:
        self.r = results
        self.out = out
        self.th = th
        self.dry = results.get("mode") == "dry_run"
        self.g: dict[str, Any] = {}
        for step, name in GATE_JSON.items():
            path = out / step / name
            if self.step(step).get("status") in USABLE and path.exists():
                self.g[step] = read_json(path)
        self.sweep, self.fps = chosen_sweep(self.g.get("sam"), th)

    def step(self, key: str) -> dict[str, Any]:
        return (self.r.get("steps") or {}).get(key) or {}

    def why(self, key: str) -> str:
        row = self.step(key)
        if not row:
            return "step not run (not in --steps)"
        status = row.get("status")
        if status == "done":
            return "no data in the step's output"
        return f"step {status}" + (f": {row['reason']}" if row.get("reason") else "")

    def missing(self, key: str) -> Verdict:
        return Verdict("not run", "", self.why(key))

    def roles(self) -> dict[str, Any]:
        return (self.sweep or {}).get("roles") or {}

    def first_roles(self) -> dict[str, Any]:
        sweeps = [s for s in (self.g.get("sam") or {}).get("sweeps", []) if s.get("roles")]
        return sweeps[0]["roles"] if sweeps else {}


def eval_e0(res: Res) -> dict[str, Verdict]:
    e0 = res.step("E0")
    if e0.get("status") != "done":
        return {k: res.missing("E0") for k in ("E0.tools", "E0.ots", "E0.hashes", "E0.proxies")}
    tools = e0["tools"]
    missing = tools.get("missing") or []
    out = {"E0.tools": ok(not missing, f"missing: {', '.join(missing)}" if missing else "all answered", tools.get("install_line") or "")}
    smoke = e0.get("smoke") or {}
    if "ots_files" in smoke:
        out["E0.ots"] = ok(smoke["ots_files"] >= 2 and (smoke.get("freetsa") or {}).get("ok", False), f"{smoke['ots_files']} .ots files; freetsa {fmt((smoke.get('freetsa') or {}).get('ok'))}", "; ".join(r.get("error", "") for r in smoke.get("ots", []) if r.get("error")))
    else:
        out["E0.ots"] = Verdict("not run", "", smoke.get("note", "no smoke"))
    originals = e0.get("originals") or {}
    out["E0.hashes"] = ok(bool(originals) and all(r.get("sha256") for r in originals.values()), ", ".join(f"{k} {r.get('sha256', '?')[:12]}" for k, r in originals.items()), "hashed before the proxies")
    proxies = [p for p in e0.get("proxies", []) if p.get("path")]
    out["E0.proxies"] = ok(len(proxies) >= 3, ", ".join(f"{p['fps']:g} fps {p['frames']} frames in {fmt(p['encode_s'], 1)} s" for p in proxies))
    return out


def eval_e1(res: Res) -> dict[str, Verdict]:
    e1 = res.step("E1")
    out = {"E1.upload_wifi": Verdict("not run", "", "needs the deploy and the chunk route; run by hand from the Mac on wifi and the phone on LTE"), "E1.proxy_container": Verdict("not run", "", "needs the Railway container")}
    if e1.get("status") != "done":
        out["E1.proxy_mac"] = res.missing("E1")
        return out
    rows = e1.get("proxy_mac") or {}
    medium = (rows.get("medium") or {}).get("encode_s")
    fast = (rows.get("veryfast") or {}).get("encode_s")
    out["E1.proxy_mac"] = Verdict("info", f"medium {fmt(medium, 1)} s, veryfast {fmt(fast, 1)} s for a {fmt((e1.get('source') or {}).get('duration_s'), 1)} s source", "the 90 s container bar is about a shared vCPU; this Mac number is a lower bound")
    return out


def _role_lane(role_out: dict[str, Any], noun: str) -> dict[str, Any]:
    m = (role_out.get("lanes") or {}).get(f"D1 E2 {noun}") or {}
    return m.get("after_car_filter") or m


def eval_e2(res: Res) -> dict[str, Verdict]:
    keys = ("E2.tracks", "E2.completion", "E2.wire_parse", "E2.decode", "E2.no_429_loss", "E2.concurrency", "E2.zero_matches", "E2.frames_billed", "E2.oracle", "E2.overlay", "E2.prop_both", "E2.variance")
    if res.sweep is None:
        return {k: res.missing("sam") for k in keys}
    ev = res.sweep["evaluation"]
    sweeps = [s for s in res.g["sam"].get("sweeps", [])]
    sweep_line = "; ".join(f"{s['fps']:g} fps " + ("skipped" if "skipped" in s else ("pass" if s["evaluation"]["tracks"]["pass"] else "fail")) for s in sweeps)
    per_role = []
    for role, role_out in res.roles().items():
        for noun in ("car door", "bumper"):
            st = _role_lane(role_out, noun)
            per_role.append(f"{role} {noun} {fmt(st.get('longest_track_share'), 2)}/{fmt(st.get('longest_track_fragments'))} frags")
    out: dict[str, Verdict] = {}
    out["E2.tracks"] = Verdict("pass" if res.fps is not None else "fail", f"{sweep_line}; at {res.sweep['fps']:g} fps: " + ", ".join(per_role), "the chosen fps is the first that passes in section 16 item 2's order" if res.fps else "no fps passed")
    c = ev["completion"]
    out["E2.completion"] = ok(c["pass"], f"p50 {fmt(c['p50_s'], 1)} s against {fmt(c['limit_s'], 0)} s at {c['frames_sent']} frames")
    out["E2.wire_parse"] = ok(ev["wire_parse"]["pass"], f"max {fmt(ev['wire_parse']['max_parse_cpu_s'])} s CPU per lane")
    d = ev["decode"]
    out["E2.decode"] = ok(d["pass"], f"{d.get('masks_decoded')} masks in {fmt(d.get('decode_cpu_s'), 2)} s CPU")
    n = ev["no_429_loss"]
    other = n.get("failed_other") or []
    out["E2.no_429_loss"] = ok(n["pass"], f"lost to 429: {n['failed_lanes'] or 'none'}; failed otherwise: {other or 'none'}; retries {n['retries_total']}", "sam_client counts 429 backoffs with the one retry of a broken stream; the propagation_direction probe is read under E2.prop_both")
    cc = ev["concurrency"]
    out["E2.concurrency"] = ok(cc["pass"], f"{cc['lanes_started']} lanes started, failed {cc.get('failed_lanes') or 'none'}; first delta grows with request index: {fmt(cc['first_delta_grows'])}")
    zero = sorted({lane for s in sweeps if "evaluation" in s for lane in s["evaluation"]["zero_match_lanes"]})
    incomplete = sorted({lane for s in sweeps if "evaluation" in s for lane in s["evaluation"]["incomplete_lanes"]})
    if incomplete:
        out["E2.zero_matches"] = Verdict("fail", f"incomplete lanes: {', '.join(incomplete)}", "check whether these are real breaks or misread zero-match lanes")
    elif zero:
        out["E2.zero_matches"] = Verdict("pass", f"zero_matches: {', '.join(zero)}; none incomplete")
    else:
        out["E2.zero_matches"] = Verdict("info", "no lane came back empty", "the face lane and the trunk lid lane on a hatchback were the expected empty cases")
    vf = ev["vfp_equals_sent"]
    out["E2.frames_billed"] = ok(vf["all_equal"], "equal on every lane" if vf["all_equal"] else f"differences {vf['differences']}")
    oracle = [r.get("oracle_check") for r in res.roles().values() if r.get("oracle_check")]
    out["E2.oracle"] = ok(all(o["agree"] for o in oracle) if oracle else None, "; ".join(f"{o['lane_id']}: agree {fmt(o['agree'])}, reader {fmt(o['reader_cpu_s'])} s vs parser {fmt(o['parser_cpu_s'])} s" for o in oracle))
    overlays = [r.get("overlay") for r in res.roles().values() if (r.get("overlay") or {}).get("path")]
    out["E2.overlay"] = ok(all(o["placement_mismatch"] == 0 and o["masks_drawn"] > 0 for o in overlays) if overlays else None, "; ".join(f"{o['masks_drawn']} masks drawn, {o['placement_mismatch']} misplaced, {o.get('frames_pixelated', 0)} frames pixelated" for o in overlays))
    pb = [(r.get("prop_both_lane") or {}, (r.get("lanes") or {}).get(gate_sam.PROP_BOTH_LANE) or {}) for r in res.roles().values() if (r.get("prop_both_lane") or {}).get("present")]
    out["E2.prop_both"] = Verdict("info", "; ".join(f"status {p.get('status')}, metadata echo {p.get('metadata_echo')}, {p.get('frames_returned')} frames" + (f", error {m['error']}" if m.get("error") else "") for p, m in pb) or "not sent")
    var = [r.get("variance_car_door") for r in res.roles().values() if (r.get("variance_car_door") or {}).get("pairs")]
    out["E2.variance"] = Verdict("info", "; ".join(", ".join(f"mean best IoU {fmt(p['mean_best_iou'])} on {p['shared_frames']} shared frames" for p in v["pairs"]) for v in var) or "fewer than two runs")
    return out


def eval_e3(res: Res) -> dict[str, Verdict]:
    if res.sweep is None:
        return {k: res.missing("sam") for k in ("E3.states", "E3.c_semantics")}
    rows = {role: r.get("states_agreement") or {} for role, r in res.roles().items()}
    have = {k: v for k, v in rows.items() if v.get("pass") is not None}
    value = "; ".join(f"{k} c basis {fmt(v.get('agree_c_share'), 2)} ({v.get('agree_c_basis')} of {v.get('tracks_matched_across_runs')}), area-only {fmt(v.get('agree_area_share'), 2)}" for k, v in have.items())
    out = {"E3.states": ok(all(v["pass"] for v in have.values()) if have else None, value, f"pass at >= {11}/{12} of the tracks matched across runs")}
    text_lanes = [m for r in res.roles().values() for m in plain_lanes(r) if m.get("kind") == "text"]
    varies = sum(1 for m in text_lanes if (m.get("c") or {}).get("c_varies_per_frame"))
    present = [(m.get("c") or {}).get("c_present_share") for m in text_lanes if (m.get("c") or {}).get("c_present_share") is not None]
    out["E3.c_semantics"] = Verdict("info", f"c varies per frame on {varies} of {len(text_lanes)} text lanes; c present on {pct(min(present) if present else None)} of records at worst", "10/26 measured c constant per track on text prompts")
    return out


def eval_e4(res: Res) -> dict[str, Verdict]:
    rows = [r.get("anchor_bridge") for r in res.roles().values() if r.get("anchor_bridge")]
    if not rows:
        return {k: res.missing("sam") for k in ("E4.joins", "E4.negative_point")}
    box_only = [((b.get("variants") or {}).get("box_only") or {}) for b in rows]
    joined = [(v.get("absorb") or {}).get("joined") for v in box_only]
    neg = [b.get("negative_point_lowers_glass_overlap") for b in rows]
    notes = "; ".join(dict.fromkeys(str(t) for b, v in zip(rows, box_only) for t in (b.get("note"), (v.get("absorb") or {}).get("note"), v.get("error"), b.get("error")) if t))
    j = [v for v in joined if v is not None]
    n = [v for v in neg if v is not None]
    if j:
        joins = ok(any(j), ", ".join(fmt(v) for v in joined), notes)
    elif "nothing to join" in notes:
        joins = Verdict("info", "nothing to join: no door track came back in 2 or more fragments", notes)  # measured, no bar applies
    else:
        joins = Verdict("not run", "", notes or "no data")
    return {
        "E4.joins": joins,
        "E4.negative_point": ok(any(n) if n else None, ", ".join(fmt(v) for v in neg), "needs a car window inside the door box at the first prompt frame"),
    }


def eval_e5(res: Res) -> dict[str, Verdict]:
    keys = ("E5.demo_mark", "E5.control", "E5.kill", "E5.carried_moments", "E5.panned", "E5.scuff_text")
    sp = res.g.get("staysput")
    if sp is None:
        return {k: res.missing("staysput") for k in keys}
    ev = sp.get("evaluation") or {}
    clips = sp.get("clips") or []
    demo = [c for c in clips if not c.get("is_control") and not c.get("panned_pass") and not c.get("backtrack")]
    refused = "; ".join(f"{e.get('label')}: {e.get('skipped') or e.get('error')}" for e in sp.get("entries", []) if e.get("skipped") or (e.get("error") and not e.get("clips")))
    unmeasured = "; ".join(ev.get("not_measured") or [])
    out: dict[str, Verdict] = {}
    value = "; ".join(f"{c['label']} {c['clip_type']}: box {fmt(c['metrics'].get('box_share'), 2)}, inside {fmt(c['metrics'].get('inside_panel_share'), 2)}, drift {fmt(c['metrics'].get('drift_rms_units'))}, {c['metrics'].get('verdict')}" for c in demo)
    out["E5.demo_mark"] = ok(ev.get("demo_pass_any") if demo else None, value, "; ".join(t for t in (unmeasured and f"not measured: {unmeasured}", refused and f"refused: {refused}") if t))
    controls = [c for c in clips if c.get("is_control")]
    out["E5.control"] = ok(ev.get("control_pass") if controls else None, "; ".join(f"{c['label']} {c['clip_type']}: drift {fmt(c['metrics'].get('drift_rms_units'))}, sign {fmt(c['metrics'].get('sign_agreement'), 2)} on {c['metrics'].get('sign_pairs')} pairs" for c in controls), "" if controls else "no reflection control clip (the plan names none and no hood glare was found, or a dry run)")
    kill = ev.get("kill")
    if kill is True:
        out["E5.kill"] = Verdict("fail", "both clip types failed on every demo mark")
    elif kill is False:
        out["E5.kill"] = Verdict("pass", "not killed: a demo-mark clip passed")
    else:
        out["E5.kill"] = Verdict("not run", "", "a demo mark lacks a measured clip of each type" + (f" ({unmeasured})" if unmeasured else "") if demo else ("no demo-mark clip" + (f"; refused: {refused}" if refused else "")))
    cms = [c.get("carried_moments_check") for c in clips if c.get("carried_moments_check")] + ([sp["carried_moments_check"]] if sp.get("carried_moments_check") else [])
    cms = [c for c in cms if c.get("frames_checked")]
    out["E5.carried_moments"] = ok(all(c.get("within_limit") for c in cms) if cms else None, "; ".join(f"{c['frames_checked']} frames, max residual {fmt(c.get('residual_max_units'), 4)} units" for c in cms), "; ".join(c.get("note", "") for c in cms if c.get("note")))
    out["E5.panned"] = Verdict("info", f"{fmt(ev.get('panned_inconclusive_share'), 2)} of {ev.get('panned_clips', 0)} panned clips inconclusive") if ev.get("panned_clips") else Verdict("not run", "", "no panned clip in the plan")
    scuff = sp.get("scuff_text") or next((e.get("scuff_text") for e in sp.get("entries", []) if e.get("scuff_text")), None)
    if not scuff:
        out["E5.scuff_text"] = Verdict("not run", "", "no scuff prompt")
    elif "status" in scuff:
        out["E5.scuff_text"] = Verdict("info", f"{scuff['status']}, ids {scuff.get('ids')}, overlap with the tracked mark {fmt(scuff.get('overlap_share'), 2)}", scuff.get("note", ""))
    else:
        out["E5.scuff_text"] = Verdict("info", scuff.get("error", "no answer"))
    return out


def eval_spark(res: Res) -> dict[str, Verdict]:
    keys = ("E6.sigma", "E6.jaccard", "E6.no_truncation", "E6.truncation_signal", "E6.second_look_valid", "E6.second_look_control", "E6.false_marks", "E6.targets", "E7.count", "E11.naming", "E11.plate_ruler", "E11.wheel_ruler", "E11.ppi", "E11.pixelation", "E11.door_pair", "E18.note_prefill")
    sp = res.g.get("spark")
    if sp is None:
        return {k: res.missing("spark") for k in keys}
    ev = sp.get("evaluation") or {}
    pe = ev.get("per_effort") or {}
    low, med = pe.get("low") or {}, pe.get("medium") or {}
    out: dict[str, Verdict] = {}
    out["E6.sigma"] = ok(low.get("sigma_pass"), f"low {pct(low.get('sigma_frac_max'), 2)} of crop width ({fmt(low.get('sigma_units_max'), 4)} units), medium {pct(med.get('sigma_frac_max'), 2)}", f"effort default: {ev.get('effort_default')}")
    out["E6.jaccard"] = ok(low.get("jaccard_pass"), f"low worst crop median {fmt(low.get('jaccard_median_min'), 3)}, pooled {fmt(low.get('jaccard_pooled_median'), 3)}; medium {fmt(med.get('jaccard_median_min'), 3)}")
    out["E6.no_truncation"] = ok(ev.get("no_truncation_at_4000"), ", ".join(f"{k}: {v.get('incomplete_at_4000')} incomplete, {v.get('truncated_after_retry')} truncated" for k, v in pe.items()))
    out["E6.truncation_signal"] = ok(ev.get("truncation_signal_seen"), f"reason {ev.get('truncation_reason')}")
    looks = sp.get("second_look") or {}
    out["E6.second_look_valid"] = ok(ev.get("second_look_valid"), "; ".join(f"{k}: {v['verdict']} ({v.get('source')})" for k, v in looks.items()))
    out["E6.second_look_control"] = ok(ev.get("second_look_control_ok"), (looks.get("control") or {}).get("verdict", ""), (looks.get("control") or {}).get("source") or "")
    fm = ev.get("false_marks_uncapped_mean_on_clean")
    out["E6.false_marks"] = Verdict("info", f"{fmt(fm, 2)} uncapped marks per call on the clean crop") if fm is not None else Verdict("not run", "", "no crop labelled clean")
    targets = [(c["name"], e.get("targets")) for c in sp.get("crops", []) for k, e in c.get("efforts", {}).items() if k == "low" and e.get("targets")]
    out["E6.targets"] = Verdict("info", "; ".join(f"{n}: {t['targets_hit_in_any_run']} of {t['targets_total']} hit within {t['radius_px']} px" for n, t in targets)) if targets else Verdict("not run", "", "no hand-measured targets")
    counts = [c.get("count_mode") for c in sp.get("crops", []) if c.get("count_mode")]
    missed = sum(len(r.get("targets_missed_by_box_pass") or []) for r in counts)
    if not counts:
        out["E7.count"] = Verdict("not run", "", "count mode not run")
    elif not targets:
        out["E7.count"] = Verdict("info", f"{ev.get('count_mode_new_points')} new points beyond the box pass", "no hand-measured targets, so the kill rule cannot be read")
    else:
        out["E7.count"] = ok(ev.get("count_mode_recovered_any"), f"recovered a missed target: {fmt(ev.get('count_mode_recovered_any'))}; the box pass missed {missed}; {ev.get('count_mode_new_points')} new points")
    total = ev.get("claims_total") or 0
    unjudged = ev.get("claims_not_judged") or 0
    claim_note = "; ".join(t for t in ("" if total >= 3 else "fewer than 3 photos judged", f"{unjudged} photo(s) not judged (a blur check or a vote request did not complete)" if unjudged else "") if t)
    out["E11.naming"] = ok(ev.get("claims_agreed") == total if total else None, f"{ev.get('claims_agreed')} of {total} agree", claim_note)
    ruler = sp.get("ruler") or {}
    plate, wheel = ruler.get("plate") or {}, ruler.get("wheel") or {}
    errors = "; ".join(f"{k}: {v}" for k, v in (ruler.get("errors") or {}).items()) or ruler.get("error", "")
    out["E11.plate_ruler"] = ok(plate.get("qualifies") if ruler else None, f"{fmt(plate.get('long_px'), 1)} px at c {fmt(plate.get('c'), 2)}" if plate else "no plate over c 0.5 off the border")
    if ruler and not plate:
        # a request that did not complete measured nothing; only an answer with no usable plate cuts the ruler
        out["E11.plate_ruler"] = Verdict("fail", "no plate over c 0.5 off the border") if ruler.get("plate_checked", "error" not in ruler) else Verdict("not run", "", f"the plate request did not complete: {errors}")
    out["E11.wheel_ruler"] = ok(wheel.get("qualifies") if ruler else None, f"major axis {fmt(wheel.get('major_axis_px'), 1)} px" if wheel else "no wheel found")
    if ruler and not wheel:
        out["E11.wheel_ruler"] = Verdict("fail", "no wheel over c 0.5 off the border") if ruler.get("wheel_checked", "error" not in ruler) else Verdict("not run", "", f"the wheel request did not complete: {errors}")
    ppi = ruler.get("ppi_agreement") or {}
    out["E11.ppi"] = ok(ppi.get("within_15pct") if ppi else None, f"abs log ratio {fmt(ppi.get('abs_log_ratio'), 3)}" if ppi else "", "" if ppi else "needs both rulers and a tire code")
    after = ruler.get("plate_after_pixelation") or {}
    if plate and "plate_after_pixelation" in ruler and ruler["plate_after_pixelation"] is None:
        out["E11.pixelation"] = Verdict("not run", "", f"the request on the pixelated frame did not complete: {errors}")
    elif after.get("matches") == 0:
        out["E11.pixelation"] = Verdict("pass", "no plate found on the pixelated frame")
    else:
        out["E11.pixelation"] = ok(after.get("over_20pct") if after else None, f"relative change {fmt(after.get('relative_change'), 3)}" if after else "")
    dp = sp.get("door_pair") or {}
    out["E11.door_pair"] = Verdict("info", f"residual {fmt(dp.get('residual_units'), 4)} units, r_claim {fmt(dp.get('r_claim_suggested'), 3)}") if dp.get("residual_units") is not None else Verdict("not run", "", "no door pair, or the handle not found in both")
    note = sp.get("note_prefill") or {}
    out["E18.note_prefill"] = ok(ev.get("note_prefill_all_fields") if note else None, f"fields agreed per agreement {ev.get('note_prefill_fields')}")
    return out


def eval_e8(res: Res) -> dict[str, Verdict]:
    sam = res.g.get("sam")
    if sam is None:
        return {k: res.missing("sam") for k in ("E8.sigma_frame", "E8.cross_take", "E8.cross_walk")}
    dry_note = "dry run: the handle lane and the door keyframe image requests need LIVE=1"
    rows = [((r.get("handle") or {}).get("sigma") or {}) for r in res.first_roles().values()]
    sig = [s["sigma_frame"] for s in rows if s.get("sigma_frame") is not None]
    why = next((s.get("note") for s in rows if s.get("note")), None) or (dry_note if res.dry else "no 'door handle' lane in this run")
    out = {"E8.sigma_frame": ok(max(sig) < 0.05 if sig else None, ", ".join(fmt(s, 4) for s in sig), "the bar is an estimate" if sig else why)}
    for key, name, bar in (("E8.cross_take", "handle_cross_take", 0.15), ("E8.cross_walk", "handle_cross_walk", 0.25)):
        row = sam.get(name) or {}
        r = row.get("residual")
        if r is None:
            out[key] = Verdict("not run", "", row.get("note") or (dry_note if res.dry else "not measured"))
        else:
            out[key] = ok(r < bar, f"{fmt(r, 4)} units ({row.get('used')}; {row.get('basis')})", "the bar is an estimate; side-agnostic because the doors are not matched by side")
    return out


def eval_e9(res: Res) -> dict[str, Verdict]:
    if res.sweep is None:
        return {k: res.missing("sam") for k in ("E9.plate_coverage", "E9.image_checks")}
    cov = {role: r.get("blur_coverage") or {} for role, r in res.roles().items()}
    have = {k: v for k, v in cov.items() if "coverage_bridged" in v}
    value = "; ".join(f"{k} raw {fmt(v['coverage_raw'], 2)}, carried {fmt(v['coverage_bridged'], 2)} over {len(v.get('windows') or [])} visible window(s) {v.get('windows_s', '')}" for k, v in have.items()) or "; ".join(f"{k}: {v.get('note', '-')}" for k, v in cov.items())
    note = f"visible windows split at gaps over {gate_sam.PLATE_WINDOW_GAP_S:g} s between detections (an estimate); check each split gap on the overlay" if have else ""
    out = {"E9.plate_coverage": ok(all(v["coverage_bridged"] >= 0.9 for v in have.values()) if have else None, value, note)}
    checks = [(role, r.get("plate_image_checks")) for role, r in res.roles().items() if r.get("plate_image_checks")]
    out["E9.image_checks"] = Verdict("info", "; ".join(f"{role}: {len(c.get('frames', []))} frames, extents {[f.get('plate_extent_px_hw_tilt') for f in c.get('frames', [])]}" for role, c in checks)) if checks else Verdict("not run", "", "plate checks not run" if not res.dry else "dry run: image requests need LIVE=1")
    return out


def eval_e10(res: Res) -> dict[str, Verdict]:
    keys = ("E10.turns", "E10.mode_applied", "E10.pins", "E10.diarization")
    v = res.g.get("voice")
    if v is None:
        return {k: res.missing("voice") for k in keys}
    ev = v.get("evaluation") or {}
    main = (v.get("modes") or {}).get(ev.get("mode")) or {}
    shares = main.get("pin_share") or {}
    split = f"; after split_long_turns {fmt(ev.get('pin_share_split'), 2)} over {ev.get('split_turns')} turns" if ev.get("pin_share_split") is not None else ""
    if ev.get("pin_share") is not None:
        pins = ok(ev.get("pins_pass"), f"{shares.get('landed')} of {shares.get('noun_naming_turns')} noun-naming turns landed ({fmt(ev.get('pin_share'), 2)}){split}", f"fallback: {ev.get('pin_fallback')}" if ev.get("pin_fallback") else "the rule reads the API's own turns")
    elif not v.get("windows_ms"):
        pins = Verdict("not run", "", "no panel windows from gate_sam to pin against")
    else:
        pins = Verdict("not run", f"{main.get('turns', 0)} turns, none names a panel noun", "the narration has to name panels for the pins to be read")
    return {
        "E10.turns": ok(ev.get("turns_non_empty"), f"{main.get('turns')} turns, {main.get('timed_turns')} timed"),
        "E10.mode_applied": ok(ev.get("mode_applied"), main.get("verdict", "")),
        "E10.pins": pins,
        "E10.diarization": Verdict("info", f"{ev.get('diarization_turns')} turns, speakers {ev.get('diarization_speakers')}"),
    }


def eval_e12(res: Res) -> dict[str, Verdict]:
    out: dict[str, Verdict] = {}
    smoke = res.step("E0").get("smoke") or {}
    stamps = [r for r in smoke.get("ots", []) if r.get("seconds") is not None]
    out["E12.ots_latency"] = ok(max(r["seconds"] for r in stamps) < 5.0 if stamps else None, ", ".join(f"{r['seconds']:.1f} s" for r in stamps), smoke.get("note") or "; ".join(r.get("error", "") for r in smoke.get("ots", []) if r.get("error")))
    up = res.step("E12").get("ots_upgrade") or {}
    out["E12.ots_upgrade"] = Verdict("info", "; ".join(f"{r['file']}: {r['status']}" for r in up.get("rows", []))) if up.get("rows") else Verdict("not run", "", up.get("note") or res.why("E12"))
    tsa = smoke.get("freetsa") or {}
    out["E12.freetsa"] = ok(tsa.get("ok") if tsa else None, f"{fmt(tsa.get('seconds'), 1)} s, time {tsa.get('time')}" if tsa else "", tsa.get("error", "") or smoke.get("note", ""))
    roles = res.first_roles()
    probes = [r.get("files_probe") for r in roles.values() if r.get("files_probe")]
    accepted = [p.get("expires_after_accepted") for p in probes if "expires_after_accepted" in p]
    out["E12.expires_after"] = ok(all(accepted) if accepted else None, "; ".join(f"expires_at {p.get('expires_at')}" for p in probes) or "", "dry run: no upload" if res.dry else "")
    deletes = [r.get("delete_file") for r in roles.values() if r.get("delete_file")]
    out["E12.files_delete"] = ok(all(isinstance(d, dict) and "error" not in d for d in deletes) if deletes else None, "; ".join(json.dumps(d, default=str)[:120] for d in deletes), "dry run: no upload" if res.dry else "")
    if res.sweep is not None:
        echo = res.sweep["evaluation"].get("store_echo_false_everywhere")
        note = "dry run: the 10/26 probe sent no store field, so its echo is true" if res.dry else ""
        out["E12.store_false_sam"] = ok(echo, "store false echoed on every lane" if echo else "some lane echoed store true or nothing", note)
    else:
        out["E12.store_false_sam"] = res.missing("sam")
    sp = res.g.get("spark")
    echo = (sp or {}).get("store_echo") or {}
    out["E12.store_false_spark"] = ok(echo.get("all_false"), f"{echo.get('counts')} over {echo.get('answers')} answers" if echo else "", echo.get("note", "")) if sp else res.missing("spark")
    dr = [r.get("delete_response_probe") for r in roles.values() if r.get("delete_response_probe")]
    out["E12.responses_delete"] = Verdict("info", "; ".join(json.dumps(d, default=str)[:160] for d in dr)) if dr else Verdict("not run", "", "dry run: no response to delete" if res.dry else "not probed")
    return out


def eval_e13(res: Res) -> dict[str, Verdict]:
    e13 = res.step("E13")
    if e13.get("status") != "done":
        return {k: res.missing("E13") for k in ("E13.pickup_wall", "E13.pair_dollars", "E13.pickup_dollars")}
    wall = e13.get("pickup_wall_s")
    parts = e13.get("pickup_wall_parts") or {}
    note = "computed from parts, not an end-to-end run"
    return {
        "E13.pickup_wall": ok(wall < 180 if wall is not None else None, f"{fmt(wall, 1)} s (proxy {fmt(parts.get('proxy_encode_s'), 1)} + Files {fmt(parts.get('files_upload_s'), 1)} + lanes {fmt(parts.get('lanes_wall_s'), 1)})", note + "; the browser upload is not in it"),
        "E13.pair_dollars": ok(e13["pair_usd"] < 7.0, f"${e13['pair_usd']:.2f} at {e13['fps']:g} fps", note),
        "E13.pickup_dollars": ok(e13["pickup_usd"] < 3.0, f"${e13['pickup_usd']:.2f}", note),
    }


def eval_e15(res: Res) -> dict[str, Verdict]:
    out = {"E15.worker": Verdict("not run", "", "a browser measurement (the Web Worker on the E2 fixture lanes)")}
    if res.sweep is None:
        out["E15.lane_bytes"] = res.missing("sam")
        return out
    sizes = {role: (r.get("lane_sizes") or {}).get("display_nouns_bytes") or 0 for role, r in res.roles().items()}
    total = sum(sizes.values()) if len(sizes) > 1 else 2 * sum(sizes.values())
    counted = max((len((r.get("lane_sizes") or {}).get("display_nouns_counted") or []) for r in res.roles().values()), default=0)
    out["E15.lane_bytes"] = ok(total / 1e6 < 30.0 if total else None, f"{total / 1e6:.2f} MB for {counted} display nouns per pair" + ("" if len(sizes) > 1 else " (one walk doubled)"))
    return out


def eval_e17(res: Res) -> dict[str, Verdict]:
    tl = (res.first_roles().get("pickup") or {}).get("timeline") if res.g.get("sam") else None
    if not tl:
        return {"E17.timeline": Verdict("not run", "", "dry run: the timeline call needs LIVE=1" if res.dry else res.why("sam") if res.g.get("sam") is None else "timeline not run")}
    ag = tl.get("agreement") or {}
    return {"E17.timeline": ok(ag.get("share") >= 0.7 if ag.get("share") is not None else None, f"{ag.get('within_1s')} of {ag.get('panels')} panels within 1 s; {tl.get('usage')} tokens, ${tl.get('dollars')}", tl.get("note", ""))}


def evaluate(res: Res) -> list[dict[str, Any]]:
    verdicts: dict[str, Verdict] = {}
    for fn in (eval_e0, eval_e1, eval_e2, eval_e3, eval_e4, eval_e5, eval_spark, eval_e8, eval_e9, eval_e10, eval_e12, eval_e13, eval_e15, eval_e17):
        verdicts.update(fn(res))
    verdicts["E14.capture"] = Verdict("not run", "", NOT_RUN_HERE["E14"])
    verdicts["E14.browsers"] = Verdict("not run", "", NOT_RUN_HERE["E14"])
    verdicts["E16.s1"] = Verdict("not run", "", NOT_RUN_HERE["E16"])
    verdicts["E19.fal"] = Verdict("not run", "", NOT_RUN_HERE["E19"])
    rows = []
    for cid, row, rule, on_fail in CRITERIA:
        v = verdicts.get(cid) or Verdict("not run", "", "no evaluator")
        rows.append({"id": cid, "row": row, "rule": rule, "on_fail": on_fail, "status": v.status, "value": v.value, "note": v.note})
    return rows


# ------------------------------------------------------------------------------------------------------------------
# Decisions and the recommendation
# ------------------------------------------------------------------------------------------------------------------


def calibration(sigma_frame: float | None, sigma_spark: float | None, th: dict) -> dict[str, Any]:
    """tau_pre0, tau_eye and tau_fix through match.calibrate when it is built (5.10), else the same formulas here."""
    match = gate_sam.optional_module("match")
    note = "match.calibrate not built"
    if match is not None and gate_sam.implemented(match, "calibrate"):
        try:
            cal = match.calibrate(sigma_frame=sigma_frame, sigma_spark=sigma_spark, sigma_view=None, pseudo_pair_d=None, th=th)
            return {**cal.model_dump(), "via": "match.calibrate"}
        except Exception as exc:  # noqa: BLE001 - fall back to the formulas, and say so
            note = f"match.calibrate raised {type(exc).__name__}"
    m = th["matching"]
    within = math.sqrt(sum(s * s for s in (sigma_frame, sigma_spark) if s is not None)) if (sigma_frame or sigma_spark) else None
    tau_pre = max(m["tau_pre0_floor"], m["tau_pre0_sigma_factor"] * within) if within is not None else m["tau_pre0_floor"]
    tau_fix = max(th["staysput"]["tau_fix_floor"], th["staysput"]["tau_fix_sigma_factor"] * sigma_frame) if sigma_frame else th["staysput"]["tau_fix_floor"]
    return {"sigma_within": within, "tau_pre": round(min(tau_pre, m["tau_pre_cap"]), 4), "tau_eye": round(m["tau_eye_factor"] * min(tau_pre, m["tau_pre_cap"]), 4), "tau_fix": round(tau_fix, 4), "calibrated": tau_pre <= m["tau_pre_cap"], "via": "formulas of 5.10 (" + note + ")"}


def by_id(criteria: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {c["id"]: c for c in criteria}


def recommend(res: Res, criteria: list[dict[str, Any]]) -> dict[str, Any]:
    c = by_id(criteria)
    st = {k: v["status"] for k, v in c.items()}
    lines: list[str] = []
    if st["E2.tracks"] == "not run":
        headline = "No engine decision yet: E2 did not run"
        lines.append(f"E2.tracks: {c['E2.tracks']['note']}. Run the sam step (LIVE=1 python scripts/run_d1.py --steps sam ...) before choosing.")
    elif st["E2.tracks"] == "pass":
        headline = f"Use hosted SAM 3.1 (meta-model-api) at {res.fps:g} fps"
        lines.append(f"E2.tracks passed: {c['E2.tracks']['value']}.")
    elif st["E4.joins"] == "fail":
        # only a measured join failure skips section 11's first branch: the anchor pass is known not to help
        headline = "Fall back: SAM_ENGINE=fal (the fal.ai SAM 3 adapter) and re-run E2 through it"
        lines.append(f"E2.tracks failed at every fps tried ({c['E2.tracks']['value']}) and the E4 bridge did not join the fragments it was given ({c['E4.joins']['value']}).")
        lines.append("If fal fails too: open-weight SAM 3 on a GPU the same day (CLAUDE.md), else One Walk on Wed morning (section 11).")
    else:
        headline = "Use hosted SAM 3.1 with the anchor pass on every panel by default, then re-run E2"
        if st["E4.joins"] == "pass":
            lines.append(f"E2.tracks failed at every fps tried ({c['E2.tracks']['value']}), but the E4 bridge joined the fragments ({c['E4.joins']['value']}).")
        else:
            why = c["E4.joins"]["value"] or c["E4.joins"]["note"] or "no data"
            lines.append(f"E2.tracks failed at every fps tried ({c['E2.tracks']['value']}); E4 measured no join ({c['E4.joins']['status']}: {why}), so section 11's first branch stands.")
        lines.append("Section 11's branch: run the anchor pass on every panel by default and re-test; if tracks still fail 3 hours after the footage landed, switch to SAM_ENGINE=fal and re-run E2 through it.")
    if st["E2.concurrency"] == "fail" or st["E2.no_429_loss"] == "fail" or "grows with request index: yes" in c["E2.concurrency"]["value"]:
        lines.append("Concurrency: start the lanes in priority waves of 6 in section 5's order (E2.concurrency, E2.no_429_loss or a growing first delta).")
    if st["E2.decode"] == "fail":
        lines.append("Decode: the Node sidecar of step 5 replaces the Python pool for bulk decodes (E2.decode).")
    if "fail" in (st["E12.store_false_sam"], st["E12.store_false_spark"]):
        lines.append("Storage: a model did not echo store false; switch on the per-request deletion path of section 8 (E12).")
    if st["E5.kill"] == "fail":
        lines.append("Stays Put is killed (E5.kill): C5 degrades to multi-keyframe Spark consistency; this does not change the engine.")
    lines.append("Named fallback during the build or judging: SAM_ENGINE=fal flips every video lane without a deploy (section 14 item 33).")
    if res.dry:
        headline = "DRY RUN, not a decision. " + headline
    return {"headline": headline, "lines": lines}


def decisions(res: Res, criteria: list[dict[str, Any]], rec: dict[str, Any]) -> list[dict[str, str]]:
    c = by_id(criteria)
    st = {k: v["status"] for k, v in c.items()}
    th = res.th
    sam = res.g.get("sam") or {}
    spark = res.g.get("spark") or {}
    sp = res.g.get("staysput") or {}
    e13 = res.step("E13")
    rows: list[dict[str, str]] = []

    def add(item: str, value: str, source: str) -> None:
        rows.append({"item": item, "value": value, "source": source})

    sweeps = [s for s in sam.get("sweeps", []) if "evaluation" in s]
    other = [f"{s['fps']:g} fps {'passes' if s['evaluation']['tracks']['pass'] else 'fails'}" for s in sweeps if res.fps is None or float(s["fps"]) != res.fps]
    add("fps", (f"{res.fps:g}" if res.fps else "none passed E2.tracks") + (f" ({'; '.join(other)})" if other else ""), "E2.tracks")
    zero = c["E2.zero_matches"]["value"]
    add("noun list", f"the 16 lanes of section 5; door handle as image requests only; empty lanes this run: {zero}", "E2.zero_matches")
    waves = st["E2.concurrency"] == "fail" or st["E2.no_429_loss"] == "fail" or "grows with request index: yes" in c["E2.concurrency"]["value"]
    add("concurrency policy", "priority waves of 6" if waves else ("16 at once per video" if st["E2.concurrency"] == "pass" else "16 at once until E2 says otherwise"), "E2.concurrency, E2.no_429_loss")
    add("SAM engine", rec["headline"], "E2.tracks, E4.joins")
    add("decode path", "Node sidecar" if st["E2.decode"] == "fail" else ("Python process pool" if st["E2.decode"] == "pass" else "Python process pool until E2 measures it"), "E2.decode")
    if "fail" in (st["E12.store_false_sam"], st["E12.store_false_spark"]):
        store = "per-request deletion path (a model did not echo store false)"
    elif st["E12.store_false_sam"] == "pass" and st["E12.store_false_spark"] == "pass":
        store = "store false on every request, accepted by both models"
    else:
        store = "unknown until E12 sees both echoes"
    add("store false or the deletion path", store, "E12.store_false_sam, E12.store_false_spark")
    ev5 = sp.get("evaluation") or {}
    by_type: dict[str, bool] = {}
    for passes in (ev5.get("demo_pass_by_label") or {}).values():
        for k, v in passes.items():
            if v is not None:  # None: that clip type measured nothing on this mark
                by_type[k] = by_type.get(k, False) or bool(v)
    if st["E5.kill"] == "fail":
        state = "killed: C5 degrades to multi-keyframe Spark consistency"
    elif by_type:
        state = "alive; " + ", ".join(f"{k}-centered clip {'passes' if v else 'fails'}" if k == "mark" else f"{k}-crop clip {'passes' if v else 'fails'}" for k, v in by_type.items())
    else:
        state = "not measured"
    add("Stays Put input type and state", state, "E5.demo_mark, E5.kill")
    def span(obs: dict[str, Any] | None) -> str:
        return f"{fmt(obs['min'])} to {fmt(obs['max'])}" if obs else "not measured"

    add("b floors and the panned share", f"b1 observed {span(ev5.get('b1_observed'))} against the {ev5.get('b1_floor', th['staysput']['b1_floor'])} floor; b2 observed {span(ev5.get('b2_observed'))} against {ev5.get('b2_floor', th['staysput']['b2_floor'])}; panned: {c['E5.panned']['value'] or c['E5.panned']['note']}", "E5")
    ev6 = spark.get("evaluation") or {}
    pe = (ev6.get("per_effort") or {}).get(ev6.get("effort_default") or "low") or {}
    tokens = f"in {fmt(pe.get('input_tokens_mean'), 0)}, out {fmt(pe.get('output_tokens_mean'), 0)} of which reasoning {fmt(pe.get('reasoning_tokens_mean'), 0)}, ${fmt(pe.get('dollars_mean'), 4)} per call, {fmt(pe.get('ms_mean'), 0)} ms" if pe else "not measured"
    add("Spark effort, tokens and truncation", f"effort {ev6.get('effort_default') or 'low (default)'}; {tokens}; truncation signal seen {fmt(ev6.get('truncation_signal_seen'))}, none at 4000 {fmt(ev6.get('no_truncation_at_4000'))}", "E6")
    sig = [((r.get("handle") or {}).get("sigma") or {}).get("sigma_frame") for r in res.first_roles().values()]
    sigma_frame = max((s for s in sig if s is not None), default=None)
    cal = calibration(sigma_frame, ev6.get("sigma_spark_units"), th)
    add("tau_pre0, tau_fix, sigma_frame", f"tau_pre0 {fmt(cal.get('tau_pre'), 3)}, tau_eye {fmt(cal.get('tau_eye'), 3)}, tau_fix {fmt(cal.get('tau_fix'), 3)}, sigma_frame {fmt(sigma_frame, 4)}, sigma_spark {fmt(ev6.get('sigma_spark_units'), 4)} ({cal.get('via')})", "E6, E8")
    count = {"pass": "count mode alive", "fail": "count mode cut", "info": "count mode undecided (no targets)", "not run": "count mode not measured"}[st["E7.count"]]
    look = "second look alive" if st["E6.second_look_valid"] == "pass" and st["E6.second_look_control"] == "pass" else ("second look disabled" if "fail" in (st["E6.second_look_valid"], st["E6.second_look_control"]) else "second look not measured")
    add("count mode and the second look", f"{count}; {look}", "E7.count, E6.second_look_*")
    rulers = [name for name, key in (("license plate", "E11.plate_ruler"), ("wheel", "E11.wheel_ruler")) if st[key] == "pass"]
    ruler = ("alive: " + " and ".join(rulers)) if rulers else ("cut" if "fail" in (st["E11.plate_ruler"], st["E11.wheel_ruler"]) else "not measured")
    add("the ruler", ruler, "E11.plate_ruler, E11.wheel_ruler")
    wall = e13.get("pickup_wall_s")
    if wall is None:
        wording = "not measured; keep 'check coverage before you drive off' as the draft"
    elif wall < 180:
        wording = "check coverage before you drive off"
    elif wall <= 360:
        wording = "before you leave the lot"
    else:
        wording = "upload the pickup walk before you leave the lot, or within the hour on wifi"
    add("pickup-check wording", f"{wording} (pickup-only {fmt(wall, 0)} s from parts)", "E13.pickup_wall")
    add("capture path", "path (a), screencapture of the Claude Chrome window, until E14 runs by hand", "E14")
    s2 = {"pass": "schedule", "fail": "drop", "not run": "not measured", "info": "undecided"}[st["E17.timeline"]]
    s2b = {"pass": "schedule", "fail": "drop", "not run": "not measured", "info": "undecided"}[st["E18.note_prefill"]]
    add("S1, S2 and S2b for Mon 11/2", f"S1: not measured here (gate_s1.py); S2: {s2}; S2b: {s2b}", "E16, E17, E18")
    return rows


# ------------------------------------------------------------------------------------------------------------------
# Memo
# ------------------------------------------------------------------------------------------------------------------


def row_status(crits: list[dict[str, Any]]) -> str:
    """fail if any criterion fails; pass when every one passed or was measured; partial when some did not run."""
    statuses = [c["status"] for c in crits]
    if "fail" in statuses:
        return "fail"
    if "not run" not in statuses:
        return "pass" if "pass" in statuses else "measured"
    if any(s in ("pass", "info") for s in statuses):
        return "partial"
    return "not run"


def memo_context(results: dict[str, Any], criteria: list[dict[str, Any]], dec: list[dict[str, str]], rec: dict[str, Any], out: Path) -> dict[str, Any]:
    steps = results.get("steps") or {}
    rows = []
    for row, (title, cost) in ROWS.items():
        crits = [{k: cell(v) for k, v in c.items()} for c in criteria if c["row"] == row]
        step = ROW_STEP.get(row)
        srow = steps.get(step, {}) if step else {}
        rows.append({"id": row, "title": title, "cost": cost, "status": row_status([c for c in criteria if c["row"] == row]), "step": step or "-", "step_status": srow.get("status", "not run") if step else "not run here", "reason": cell(srow.get("reason") or (NOT_RUN_HERE.get(row) or "")), "criteria": crits})
    fails = [{k: cell(v) for k, v in c.items()} for c in criteria if c["status"] == "fail"]
    spend_rows = [{"step": s.key, "rows": ", ".join(s.rows), "status": steps.get(s.key, {}).get("status", "not run"), "seconds": fmt(steps.get(s.key, {}).get("seconds"), 1), "planned": fmt(steps.get(s.key, {}).get("planned_usd"), 2), "spent": fmt(steps.get(s.key, {}).get("spent_usd"), 4), "reason": cell(steps.get(s.key, {}).get("reason", ""))} for s in STEPS]
    e13 = steps.get("E13", {})
    counts = {s: sum(1 for c in criteria if c["status"] == s) for s in ("pass", "fail", "not run", "info")}
    meta = {k: cell(results.get(k)) for k in ("run_prefix", "started_at", "finished_at", "config_hash")}
    meta.update({"out": cell(results.get("out")), "budget": f"{results.get('budget_usd', 0):.2f}", "spend": f"{results.get('spend_usd', 0.0):.4f}", "total_ledger": cell(results.get("ledger_total_usd"))})
    inputs = [{"role": k, "path": cell(v.get("path")), "sha256": cell((v.get("sha256") or "")[:16]), "camera_time": cell(v.get("camera_time")), "size": cell(f"{v.get('width')}x{v.get('height')} at {fmt(v.get('fps'), 2)} fps, {fmt(v.get('duration_s'), 1)} s")} for k, v in (steps.get("E0", {}).get("originals") or {}).items()]
    auto = results.get("auto") or {}
    picks = [cell(n) for n in auto.get("notes", [])]
    for spec in auto.get("crops", []):
        picks.append(cell(f"E6 crop {spec['name']}: {Path(spec['source']).name} at {spec['t']} s, box {spec['crop']}"))
    for entry in auto.get("staysput", []):
        picks.append(cell(f"E5 {entry['label']}: {Path(entry['src']).name} at {entry['pts']} s, mark {entry['mark']} ({entry.get('auto')})"))
    deleted = [cell(f"{d['path']} ({d.get('bytes')} bytes, sha256 {str(d.get('sha256'))[:12]})") for d in results.get("deleted_unblurred", [])]
    return {
        "dry": results.get("mode") == "dry_run",
        "meta": meta,
        "inputs": inputs,
        "plan": cell(results.get("plan") or "none (picks below come from gate_sam's lanes)"),
        "recommendation": {"headline": cell(rec["headline"]), "lines": [cell(t) for t in rec["lines"]]},
        "decisions": [{k: cell(v) for k, v in d.items()} for d in dec],
        "rows": rows,
        "fails": fails,
        "counts": counts,
        "spend_rows": spend_rows,
        "e13_pair": [{k: cell(fmt(v, 4) if isinstance(v, float) else v) for k, v in i.items()} for i in e13.get("pair_items", [])],
        "e13_pickup": [{k: cell(fmt(v, 4) if isinstance(v, float) else v) for k, v in i.items()} for i in e13.get("pickup_items", [])],
        "e13_note": cell(e13.get("note", "")),
        "picks": picks,
        "deleted": deleted,
        "files": [cell(f"{k}: {rel(out / k / GATE_JSON[k], out)}") for k in GATE_JSON if (out / k / GATE_JSON[k]).exists()],
    }


def render_memo(context: dict[str, Any], template: Path = TEMPLATE) -> str:
    import jinja2

    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(template.parent)), undefined=jinja2.StrictUndefined, trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True, autoescape=False)
    return ascii_text(env.get_template(template.name).render(**context))


def finish(results: dict[str, Any], out: Path, th: dict) -> tuple[dict[str, Any], str]:
    """Evaluate every criterion, decide, render the memo; results.json and D1_MEMO.md written."""
    res = Res(results, out, th)
    criteria = evaluate(res)
    rec = recommend(res, criteria)
    dec = decisions(res, criteria, rec)
    results.update({"criteria": criteria, "recommendation": rec, "decisions": dec, "chosen_fps": res.fps})
    memo = render_memo(memo_context(results, criteria, dec, rec, out))
    (out / "D1_MEMO.md").write_text(memo, encoding="ascii")
    results["memo"] = "D1_MEMO.md"
    gate_sam.write_json(out / "results.json", results)
    return results, memo


# ------------------------------------------------------------------------------------------------------------------
# The run
# ------------------------------------------------------------------------------------------------------------------


def track_blurred_clips(ctx: Ctx) -> set[Path]:
    """E5 clips pixelated from the walk's plate and face tracks (5.9 item 2); every other E5 clip is approximate."""
    path = ctx.out / "staysput" / GATE_JSON["staysput"]
    if not path.exists():
        return set()
    try:
        clips = read_json(path).get("clips") or []
    except (OSError, ValueError):
        return set()
    return {Path(c["clip_file"]).resolve() for c in clips if c.get("clip_file") and (c.get("blur") or {}).get("track_based")}


def delete_unblurred(ctx: Ctx) -> list[dict[str, Any]]:
    """The SAM proxies and the bridge clips cut from them are unblurred, and an E5 clip blurred only by the prompt
    frame's boxes may still hold a plate or a face on its other frames; they go once every row has run."""
    rows = []
    keep = track_blurred_clips(ctx)
    patterns = [(ctx.out / "sam" / "proxies", "*.mp4"), (ctx.out / "sam" / "clips", "*.mp4"), (ctx.out / "E0" / "proxies", "*.mp4"), (ctx.out / "E1", "*.mp4"), (ctx.out / "staysput" / "clips", "*.mp4")]
    for folder, pattern in patterns:
        for path in sorted(folder.glob(pattern)) if folder.exists() else []:
            if path.name.endswith(".blurred.mp4") or path.resolve() in keep:
                continue
            rows.append({"path": rel(path, ctx.out), "bytes": path.stat().st_size, "sha256": gate_sam.sha256_file(path), "deleted_at": utc_now()})
            path.unlink()
    return rows


def run_step(ctx: Ctx, step: Step) -> dict[str, Any]:
    spent = spend(ctx)
    planned = planned_usd(ctx, step.key)
    if planned > 0 and spent + planned > ctx.budget:
        return {"status": "skipped", "reason": f"budget: ${spent:.2f} spent + ${planned:.2f} planned > ${ctx.budget:.2f}", "planned_usd": planned, "seconds": 0.0, "spent_usd": 0.0}
    t0 = time.monotonic()
    try:
        if step.key == "E0":
            row = step_e0(ctx)
        elif step.key == "E1":
            row = step_e1(ctx)
        elif step.key == "sam":
            if ctx.inputs.get("pickup") is None:
                row = {"status": "not run", "reason": "no pickup walk"}
            else:
                row = run_gate(ctx, "sam", "gate_sam.py", sam_args(ctx))
        elif step.key == "spark":
            argv, picked = spark_args(ctx, gate_data(ctx, "sam"))
            ctx.results.setdefault("auto", {}).setdefault("notes", []).extend(picked.get("notes", []))
            if picked.get("crops"):
                ctx.results["auto"]["crops"] = picked["crops"]
            row = run_gate(ctx, "spark", "gate_spark.py", argv) if argv else {"status": "not run", "reason": "no crop: the plan names none and gate_sam suggested none"}
        elif step.key == "staysput":
            argv, picked = staysput_args(ctx, gate_data(ctx, "sam"), gate_data(ctx, "spark"))
            ctx.results.setdefault("auto", {}).setdefault("notes", []).extend(picked.get("notes", []))
            if picked.get("entries"):
                ctx.results["auto"]["staysput"] = picked["entries"]
            row = run_gate(ctx, "staysput", "gate_stays_put.py", argv) if argv else {"status": "not run", "reason": "no mark to track: the plan names none and E6 gave none"}
        elif step.key == "voice":
            argv = voice_args(ctx)
            row = run_gate(ctx, "voice", "gate_voice.py", argv) if argv else {"status": "not run", "reason": "no narration source"}
        elif step.key == "E12":
            row = step_e12(ctx)
        else:
            row = step_e13(ctx)
    except Exception as exc:  # noqa: BLE001 - one step's crash must not stop the memo
        row = {"status": "failed", "reason": f"{type(exc).__name__}: {str(exc)[:400]}"}
    row.setdefault("seconds", round(time.monotonic() - t0, 3))
    row["planned_usd"] = planned
    row["spent_usd"] = round(spend(ctx) - spent, 6)
    return row


def make_ctx(args: argparse.Namespace, th: dict) -> Ctx:
    dry = bool(args.dry_run)
    ts = gate_sam.utc_ts()
    out = Path(args.out).resolve() if args.out else (paths.runs_root() / "d1" / (("dry_" if dry else "") + ts))
    out.mkdir(parents=True, exist_ok=True)
    previous = read_json(out / "results.json") if (out / "results.json").exists() else {}
    if previous and (previous.get("mode") == "dry_run") != dry:
        raise SystemExit(f"{out} holds a {'dry' if previous.get('mode') == 'dry_run' else 'live'} run; use another --out")
    plan = load_plan(args.plan)
    budget = float(args.budget if args.budget is not None else th["cost"]["d1_budget_usd"])
    prefix = previous.get("run_prefix") or (("d1dry_" if dry else "d1_") + ts)
    results = previous or {"schema": "d1_results_v1", "mode": "dry_run" if dry else "live", "run_prefix": prefix, "started_at": utc_now(), "steps": {}}
    results.update({"out": str(out), "budget_usd": budget, "plan": plan.get("_path"), "config_hash": contracts.config_hash(th)})
    return Ctx(args=args, th=th, out=out, prefix=prefix, dry=dry, budget=budget, plan=plan, inputs=resolve_inputs(args, plan, dry), results=results)


def run_all(ctx: Ctx) -> dict[str, Any]:
    wanted = set(ctx.args.steps or STEP_KEYS)
    ctx.results["inputs"] = {k: str(v) if v else None for k, v in ctx.inputs.items()}
    for step in STEPS:
        if step.key not in wanted:
            continue
        print(f"run_d1: {step.key} ({', '.join(step.rows)}) ...", flush=True)
        ctx.results["steps"][step.key] = run_step(ctx, step)
        row = ctx.results["steps"][step.key]
        print(f"run_d1: {step.key} {row['status']}" + (f" ({row['reason']})" if row.get("reason") else "") + f" in {row['seconds']:.1f} s, ${row['spent_usd']:.4f}", flush=True)
        gate_sam.write_json(ctx.out / "results.json", ctx.results)
    if not ctx.args.keep_unblurred:
        ctx.results["deleted_unblurred"] = ctx.results.get("deleted_unblurred", []) + delete_unblurred(ctx)
    ctx.results["finished_at"] = utc_now()
    ctx.results["spend_usd"] = spend(ctx)
    ctx.results["ledger_total_usd"] = None if ctx.dry else round(sum(float(r.get("usd") or 0.0) for r in ledger.rows(Path(ledger.LEDGER_PATH))), 4)
    results, _ = finish(ctx.results, ctx.out, ctx.th)
    return results


def print_plan(ctx: Ctx) -> None:
    print("run_d1 would run (set LIVE=1 to spend, or pass --dry-run):")
    for role, path in ctx.inputs.items():
        if path:
            print(f"  {role}: {path} ({duration_s(path):.1f} s)")
    total = 0.0
    for step in STEPS:
        usd = planned_usd(ctx, step.key)
        total += usd
        print(f"  {step.key:<9} {', '.join(step.rows):<28} planned ${usd:.2f}  {step.title}")
    print(f"  total planned ${total:.2f} against a ${ctx.budget:.2f} ceiling read from {ledger.LEDGER_PATH} (rows with the run prefix)")
    for row, why in NOT_RUN_HERE.items():
        print(f"  {row}: not run here, {why}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pickup", help="the pickup walk (4K original)")
    parser.add_argument("--return", dest="return_", help="the return walk")
    parser.add_argument("--take2", help="take 2 of the pickup walk (E8 cross-take residual)")
    parser.add_argument("--narration", help="the narration source for E10 (default: the pickup walk)")
    parser.add_argument("--plan", help="JSON plan of hand picks (see --plan-template)")
    parser.add_argument("--budget", type=float, default=None, help="dollar ceiling on ledger rows with this run's prefix (default thresholds cost.d1_budget_usd)")
    parser.add_argument("--out", help="run folder (default runs/d1/<timestamp>); an existing folder resumes with its prefix")
    parser.add_argument("--steps", nargs="+", choices=STEP_KEYS, help="run only these steps (results merge into --out)")
    parser.add_argument("--n", type=int, default=5, help="Spark calls per crop and effort in E6 (default 5)")
    parser.add_argument("--dry-run", action="store_true", help="replay the probe fixtures through every gate; no model or network call")
    parser.add_argument("--render-only", metavar="DIR", help="re-evaluate and re-render the memo of an existing run folder")
    parser.add_argument("--plan-template", action="store_true", help="print a plan skeleton and exit")
    parser.add_argument("--keep-unblurred", action="store_true", help="keep the unblurred SAM proxies, bridge clips and E5 clips not blurred from tracks")
    parser.add_argument("--step-timeout", type=float, default=None, help="seconds before a gate process is stopped (default none)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.plan_template:
        print(json.dumps(PLAN_TEMPLATE, indent=2))
        return 0
    th = contracts.load_thresholds()
    if args.render_only:
        out = Path(args.render_only).resolve()
        results, _ = finish(read_json(out / "results.json"), out, th)
        print(f"wrote {out / 'D1_MEMO.md'}: {results['recommendation']['headline']}")
        return 0
    if not args.dry_run:
        plan = load_plan(args.plan)
        if not (args.pickup or plan.get("pickup")):
            print("run_d1: --pickup (or a plan with 'pickup') is required, or --dry-run", file=sys.stderr)
            return 2
        problems = input_problems(resolve_inputs(args, plan, False))
        if problems:
            for line in problems:
                print(f"run_d1: {line}", file=sys.stderr)
            return 2
        if not gate_sam.live_allowed():
            ctx = Ctx(args=args, th=th, out=Path("."), prefix="d1_plan", dry=False, budget=float(args.budget if args.budget is not None else th["cost"]["d1_budget_usd"]), plan=plan, inputs=resolve_inputs(args, plan, False), results={})
            print_plan(ctx)
            return 2
    ctx = make_ctx(args, th)
    results = run_all(ctx)
    failed = [k for k, v in results["steps"].items() if v.get("status") in ("failed", "partial")]
    counts = {s: sum(1 for c in results["criteria"] if c["status"] == s) for s in ("pass", "fail", "not run", "info")}
    print(f"run_d1: {results['recommendation']['headline']}")
    print(f"run_d1: criteria {counts}; spend ${results['spend_usd']:.4f} of ${ctx.budget:.2f}" + (" (dry run, nothing spent)" if ctx.dry else ""))
    print(f"wrote {ctx.out / 'results.json'} and {ctx.out / 'D1_MEMO.md'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
