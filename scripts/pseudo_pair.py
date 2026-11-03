"""Pseudo pairs for label-free calibration (docs/ARCHITECTURE.md 5.10 and section 10 item 4; eval/README.md).

A pseudo pair is two walks of one car where nothing changed: a time split of one clip, or two takes from one shoot.
Every mark should match itself, so the distances of the pair's mutual-NN pairs are the noise floor. When fewer than
90% of those under tau_eye also fall under tau_pre, match.calibrate raises tau_pre to their 90th percentile (capped
at 0.4 units). This script cuts the clips, then reads a finished run of them; it never calls a model.

    pseudo_pair.py list                  the pairs, their sources and what is on disk
    pseudo_pair.py cut [PAIR ...]        cut the clips into data/derived/ (ffmpeg only) and print the run command
    pseudo_pair.py scenes CLIP           scene cuts in a clip (a time split sits on one)
    pseudo_pair.py calibrate RUN_ID      the noise floor and the tau_pre it implies; writes eval/pseudo/<run>.json

The NHTSA before and after cut (kind before_after) is listed too because it is cut the same way
(docs/DATA_HUNT.md 3.4); calibrate refuses it, finding a run's pair by its id (eval_<pair>[_n]) or by the sha256 of
both walks. Only licensed footage is listed; credits are in DATA_HUNT.md 3.8. DATA_HUNT 4.1 also names 4281322 against
the left side of 4210869 and 4228872 against 4281029; this build's eval uses only clips 4228872 and 4281322.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from walkaround import contracts, match, paths  # noqa: E402
from walkaround.contracts import Calibration, MatchState, PanelMatch  # noqa: E402

PSEUDO_OUT = REPO / "eval" / "pseudo"
LABELS_DIR = REPO / "eval" / "labels"
HIST_STEP = 0.05  # units per histogram bin
HIST_TOP = 0.5  # last closed bin edge; farther pairs go in one overflow bin
CALIBRATION_KINDS = ("time_split", "two_takes")
FLOOR_WORDS = {"time_split": "same car, one clip split in two", "two_takes": "same car, two takes from one shoot"}
# D1 E13 (docs/evidence/D1_MEMO.md), from parts on uCO3D pair C: 16 lanes on both walks cost $1.9792 for 39.9 s of video, and
# the whole pair $4.33; the rest (keyframe checks, Stays Put, Spark) is taken as a fixed $2.35 per pair. Plan lines only.
LANE_USD_PER_S = 1.9792 / 39.9
PAIR_EXTRAS_USD = 4.33 - 1.9792


@dataclass(frozen=True)
class Source:
    name: str
    folders: tuple[str, ...]  # repo-relative folders that may hold it (candidates/ until DATA_HUNT 3.1 moves it)
    sha256: str  # from the candidate manifests
    credit: str


@dataclass(frozen=True)
class Cut:
    source: Source
    t0: float = 0.0
    t1: float | None = None  # None: to the end
    out: str | None = None  # repo-relative output; None: the source is the walk as is

    def walk_path(self) -> Path:
        """The file the walk runs on: the cut, or the source itself."""
        return REPO / self.out if self.out else find_source(self.source)


@dataclass(frozen=True)
class Pair:
    pair_id: str
    kind: str  # time_split | two_takes | before_after
    table: str  # docs/DATA_HUNT.md 4.1 table letter
    pickup: Cut
    return_: Cut
    note: str


PEXELS = ("data/candidates/pexels_westarmoney_s550", "data/raw/pexels")
NHTSA = ("data/candidates/nhtsa_crash_test_db/video", "data/raw/nhtsa/video")
PEXELS_CREDIT = "Stock video: WeStarMoney Rec on Pexels, Pexels License"
PX_4228872 = Source("pexels_4228872.mp4", PEXELS, "53fbbef8c6004f973ca37425b0e24e99752045febefe489e4f05b8e6e7f291e5", PEXELS_CREDIT)
PX_4281322 = Source("pexels_4281322.mp4", PEXELS, "e25b935543511999ad05db023ac7e8adcc8a9c39ba6657fa21565bdbd7204e33", PEXELS_CREDIT)
NHTSA_15759 = Source(
    "t15759_documentation_video.mp4",
    NHTSA,
    "2e9bd5e92f0bd3a3fd3c0e06d5fc361536b2fc1f73801223c8230304d9d23ac7",
    "NHTSA crash test 15759, public information",
)

# Scene cuts measured with `pseudo_pair.py scenes` (ffmpeg scene score over 0.25) on Oct 31: 4228872 cuts at
# 13.791667 s (rear and left side, then front-left, front and right side) and at 44.875 s (the car leaves the frame).
PAIRS: dict[str, Pair] = {
    p.pair_id: p
    for p in (
        Pair(
            "px4228872_cut",
            "time_split",
            "C",
            Cut(PX_4228872, 0.0, 13.791667, "data/derived/px4228872_0000_1379.mp4"),
            Cut(PX_4228872, 13.791667, 44.875, "data/derived/px4228872_1379_4488.mp4"),
            "one clip split at its scene cut: the rear and the left side, then the front-left corner, the front and "
            "the right side; little overlap (the left side is seen square on, then from the front corner)",
        ),
        Pair(
            "px4228872_front",
            "time_split",
            "C",
            Cut(PX_4228872, 13.791667, 29.0, "data/derived/px4228872_1379_2900.mp4"),
            Cut(PX_4228872, 29.0, 44.875, "data/derived/px4228872_2900_4488.mp4"),
            "the second scene split in the middle of the front view: both halves hold the front, so the front panels "
            "can pair; the better calibration pair of the two splits",
        ),
        Pair(
            "px_two_takes",
            "two_takes",
            "C",
            Cut(PX_4228872, 0.0, 13.791667, "data/derived/px4228872_0000_1379.mp4"),
            Cut(PX_4281322),
            "two takes from one shoot: the rear and left side at a distance, then a close pass along the left side "
            "to the rear",
        ),
        Pair(
            "nhtsa15759",
            "before_after",
            "D",
            Cut(NHTSA_15759, 32.0, 52.0, "data/derived/t15759_pre.mp4"),
            Cut(NHTSA_15759, 112.0, 142.0, "data/derived/t15759_post.mp4"),
            "crash test 15759 before and after (DATA_HUNT 3.4 cut times): fixed shots of the left side, rear, right "
            "side and front before; the struck left side, front and rear after",
        ),
    )
}


# Sources and cuts ----------------------------------------------------------------------------------------------------


def find_source(source: Source) -> Path:
    """The first folder that holds the source file; the candidates path when none does (the caller reports it)."""
    for folder in source.folders:
        path = REPO / folder / source.name
        if path.is_file():
            return path
    return REPO / source.folders[0] / source.name


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cut_command(src: Path, cut: Cut, dst: Path) -> list[str]:
    """DATA_HUNT 3.4's ffmpeg line (input seek, x264 crf 16, no audio) plus metadata stripped, bitexact flags and a
    fixed encoder thread count, so a cut carries no borrowed camera time and the same ffmpeg build gives the same bytes
    on any machine."""
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", f"{cut.t0:.6f}", "-i", str(src)]
    if cut.t1 is not None:
        cmd += ["-t", f"{cut.t1 - cut.t0:.6f}"]
    cmd += ["-map", "0:v:0", "-c:v", "libx264", "-crf", "16", "-threads", "8", "-pix_fmt", "yuv420p", "-an", "-map_metadata", "-1"]
    cmd += ["-fflags", "+bitexact", "-flags:v", "+bitexact", "-movflags", "+faststart", str(dst)]
    return cmd


def probe_frames(path: Path) -> tuple[int, float]:
    """(frames, duration_s) of the first video stream, by counting packets."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries", "stream=nb_read_packets:format=duration", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    info = json.loads(out)
    return int(info["streams"][0]["nb_read_packets"]), float(info["format"]["duration"])


def manifest_path(out: Path) -> Path:
    return out.with_suffix(".cut.json")


def cut_clip(cut: Cut, *, check_source: bool = True) -> dict[str, Any]:
    """Cut one walk into data/derived/ (skipped when the output and its manifest exist) and return the manifest."""
    assert cut.out is not None
    src = find_source(cut.source)
    dst = REPO / cut.out
    record = manifest_path(dst)
    if dst.is_file() and record.is_file():
        return json.loads(record.read_text(encoding="utf-8"))
    if not src.is_file():
        raise FileNotFoundError(f"source missing: {src.relative_to(REPO)} (sha256 {cut.source.sha256[:12]}); see docs/DATA_HUNT.md section 3")
    src_sha = sha256_file(src)
    if check_source and src_sha != cut.source.sha256:
        raise ValueError(f"{src.relative_to(REPO)} hashes to {src_sha[:12]}, not the manifest's {cut.source.sha256[:12]}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cut_command(src, cut, dst), check=True)
    frames, duration = probe_frames(dst)
    manifest = {
        "schema": "walkaround_cut_v1",
        "source": src.relative_to(REPO).as_posix(),
        "source_sha256": src_sha,
        "t0": cut.t0,
        "t1": cut.t1,
        "out": cut.out,
        "out_sha256": sha256_file(dst),
        "out_bytes": dst.stat().st_size,
        "frames": frames,
        "duration_s": round(duration, 6),
        "command": " ".join(cut_command(Path(src.relative_to(REPO)), cut, Path(cut.out))),
        "credit": cut.source.credit,
    }
    contracts.write_json(record, manifest)
    return manifest


def walk_paths(pair: Pair) -> tuple[Path, Path]:
    return pair.pickup.walk_path(), pair.return_.walk_path()


def run_id_for(pair_id: str) -> str:
    return f"eval_{pair_id}"


def run_command(pair: Pair) -> str:
    pickup, ret = (p.relative_to(REPO).as_posix() for p in walk_paths(pair))
    return f"LIVE=1 .venv/bin/walkaround run {pickup} {ret} --run-id {run_id_for(pair.pair_id)}"


def video_seconds(cut: Cut) -> float | None:
    if cut.t1 is not None:
        return cut.t1 - cut.t0
    path = find_source(cut.source)
    return probe_frames(path)[1] - cut.t0 if path.is_file() else None


# Scenes --------------------------------------------------------------------------------------------------------------


def scene_cuts(path: Path, *, threshold: float = 0.25) -> list[float]:
    """pts of frames whose ffmpeg scene score is over threshold."""
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-i", str(path), "-vf", f"select='gt(scene\\,{threshold})',showinfo", "-an", "-f", "null", "-"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [float(t) for t in re.findall(r"pts_time:([0-9.]+)", proc.stderr)]


# Calibration ---------------------------------------------------------------------------------------------------------


def pseudo_rows(matches: list[PanelMatch]) -> list[tuple[float, float]]:
    """(d, r_mean) of every mutual-NN pair on a comparable panel: the rows match.calibrate reads."""
    return [(p.d, p.r_mean) for pm in matches if pm.state == MatchState.COMPARABLE for p in pm.pairs if p.mutual]


def histogram(values: list[float]) -> list[dict[str, Any]]:
    """Counts per HIST_STEP bin from 0 to HIST_TOP plus one overflow bin, as [lo, hi) rows."""
    n_bins = round(HIST_TOP / HIST_STEP)
    counts = [0] * (n_bins + 1)
    for v in values:
        counts[min(math.floor(v / HIST_STEP + 1e-9), n_bins)] += 1  # the slack keeps 0.3 out of the [0.25, 0.3) bin
    rows = [{"lo": round(i * HIST_STEP, 6), "hi": round((i + 1) * HIST_STEP, 6), "n": counts[i]} for i in range(n_bins)]
    rows.append({"lo": HIST_TOP, "hi": None, "n": counts[n_bins]})
    return rows


def noise_floor(matches: list[PanelMatch], calibration: Calibration | None, th: dict) -> dict[str, Any]:
    """The pseudo pair's numbers: rows, the histogram of d brought back to r = 0, how many marks matched, and the
    calibration the rows imply next to the run's own (sigmas carried over from calibration.json)."""
    rows = pseudo_rows(matches)
    factor = float(th["matching"]["band_radius_factor"])
    normalised = sorted(d / (1.0 + factor * r * r) for d, r in rows)
    base = calibration or match.calibrate(sigma_frame=None, sigma_spark=None, sigma_view=None, pseudo_pair_d=None, th=th)
    implied = match.calibrate(sigma_frame=base.sigma_frame, sigma_spark=base.sigma_spark, sigma_view=base.sigma_view, pseudo_pair_d=rows or None, th=th)
    under = sum(1 for d in normalised if d < base.tau_pre)
    comparable = [pm for pm in matches if pm.state == MatchState.COMPARABLE]
    paired_p = {p.pickup_mark for pm in comparable for p in pm.pairs if p.mutual and p.d < p.tau_pre_r}
    paired_r = {p.return_mark for pm in comparable for p in pm.pairs if p.mutual and p.d < p.tau_pre_r}
    n_pickup = sum(len(pm.unmatched_pickup) for pm in comparable) + len({p.pickup_mark for pm in comparable for p in pm.pairs})
    n_return = sum(len(pm.unmatched_return) for pm in comparable) + len({p.return_mark for pm in comparable for p in pm.pairs})
    return {
        "rows": [[round(d, 6), round(r, 6)] for d, r in rows],
        "n_rows": len(rows),
        "under_tau_pre": under,
        "histogram": histogram(normalised),
        "comparable_panels": len(comparable),
        "pickup_marks_matched": [len(paired_p), n_pickup],
        "return_marks_matched": [len(paired_r), n_return],
        "tau_pre_run": base.tau_pre,
        "tau_pre_implied": implied.tau_pre,
        "tau_eye_implied": implied.tau_eye,
        "calibrated_implied": implied.calibrated,
        "raised": implied.tau_pre > base.tau_pre + 1e-9,
    }


def floor_line(result: dict[str, Any], kind: str | None = None) -> str:
    """The README's sentence, with N, worded for how the pair was made (nothing measured how far apart two takes were)."""
    n, m = result["pickup_marks_matched"]
    words = FLOOR_WORDS.get(kind or "", "same car, two walks where nothing changed")
    return f"{n} of {m} pickup marks matched, {words} (noise floor)"


def _expected_sha(cut: Cut, labeled: str | None) -> str | None:
    """The sha256 a walk of this cut has: the source's for an uncut walk, else the cut manifest's, else the labels'."""
    if cut.out is None:
        return cut.source.sha256
    record = manifest_path(REPO / cut.out)
    if record.is_file():
        return json.loads(record.read_text(encoding="utf-8")).get("out_sha256")
    return labeled


def _labeled_shas(pair_id: str) -> dict[str, str | None]:
    path = LABELS_DIR / f"{pair_id}.json"
    footage = json.loads(path.read_text(encoding="utf-8")).get("footage", {}) if path.is_file() else {}
    return {role: (footage.get(role) or {}).get("sha256") for role in ("pickup", "return")}


def pair_for_run(run_id: str, shas: dict[str, str | None] | None = None) -> Pair | None:
    """The pair a run is of: by its id (eval_<pair> or eval_<pair>_<n>), else by the sha256 of both walks."""
    for pair in PAIRS.values():
        if run_id == run_id_for(pair.pair_id) or re.fullmatch(re.escape(run_id_for(pair.pair_id)) + r"_\d+", run_id):
            return pair
    if not shas or not all(shas.get(role) for role in ("pickup", "return")):
        return None
    for pair in PAIRS.values():
        labeled = _labeled_shas(pair.pair_id)
        want = (_expected_sha(pair.pickup, labeled["pickup"]), _expected_sha(pair.return_, labeled["return"]))
        if want == (shas["pickup"], shas["return"]):
            return pair
    return None


def run_shas(root: Path) -> dict[str, str | None]:
    """The sha256 of each walk as the server hashed it (videos/<role>.json)."""
    out: dict[str, str | None] = {}
    for role in ("pickup", "return"):
        path = root / "videos" / f"{role}.json"
        raw = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        out[role] = raw.get("sha256_server") or raw.get("sha256_browser")
    return out


def calibrate_run(run_id: str, *, out_dir: Path | None = None, kind: str | None = None) -> tuple[dict[str, Any], Path]:
    """Read runs/<run_id>/derived/matches.json and calibration.json with the run's frozen thresholds; write the result."""
    root = paths.runs_root() / run_id
    matches_file = root / "derived" / "matches.json"
    if not matches_file.is_file():
        raise FileNotFoundError(f"runs/{run_id}/derived/matches.json is missing; finish the run first")
    run = contracts.read_json(root / "run.json", contracts.RunManifest)
    th = run.thresholds or contracts.load_thresholds()
    cal_file = root / "derived" / "calibration.json"
    calibration = contracts.read_json(cal_file, Calibration) if cal_file.is_file() else None
    matches = contracts.read_json_list(matches_file, PanelMatch)
    result = noise_floor(matches, calibration, th)
    result.update({"schema": "walkaround_pseudo_v1", "run_id": run_id, "config_hash": run.config_hash, "kind": kind, "line": floor_line(result, kind)})
    out = (out_dir or PSEUDO_OUT) / f"{run_id}.json"
    contracts.write_json(out, result)
    return result, out


# Commands ------------------------------------------------------------------------------------------------------------


def cmd_list(_args: argparse.Namespace) -> int:
    for pair in PAIRS.values():
        print(f"{pair.pair_id} ({pair.kind}, table {pair.table}): {pair.note}")
        total = 0.0
        for role, cut in (("pickup", pair.pickup), ("return", pair.return_)):
            src = find_source(cut.source)
            span = f"{cut.t0:.3f} s to {'the end' if cut.t1 is None else f'{cut.t1:.3f} s'}"
            have = "cut" if cut.out and (REPO / cut.out).is_file() else ("source" if src.is_file() else "missing")
            target = cut.out or src.relative_to(REPO).as_posix()
            print(f"  {role}: {src.relative_to(REPO).as_posix()} {span} -> {target} [{have}]")
            seconds = video_seconds(cut) if src.is_file() else (cut.t1 - cut.t0 if cut.t1 is not None else None)
            total += seconds or 0.0
        print(f"  run: {run_command(pair)}")
        lanes = total * LANE_USD_PER_S
        print(f"  cost: lanes about ${lanes:.2f} for {total:.1f} s of video, about ${lanes + PAIR_EXTRAS_USD:.2f} in all (D1 E13 rates; estimate)")
    return 0


def cmd_cut(args: argparse.Namespace) -> int:
    if shutil.which("ffmpeg") is None:
        print("ffmpeg is not on PATH", file=sys.stderr)
        return 2
    ids = args.pairs or list(PAIRS)
    unknown = [p for p in ids if p not in PAIRS]
    if unknown:
        print(f"unknown pair: {', '.join(unknown)}; known: {', '.join(PAIRS)}", file=sys.stderr)
        return 2
    code = 0
    for pair_id in ids:
        pair = PAIRS[pair_id]
        for role, cut in (("pickup", pair.pickup), ("return", pair.return_)):
            if cut.out is None:
                print(f"{pair_id} {role}: uses {find_source(cut.source).relative_to(REPO).as_posix()} as is")
                continue
            try:
                m = cut_clip(cut)
            except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as exc:
                print(f"{pair_id} {role}: {exc}", file=sys.stderr)
                code = 1
                continue
            print(f"{pair_id} {role}: {m['out']} {m['frames']} frames, {m['duration_s']:.3f} s, sha256 {m['out_sha256']}")
        print(f"  then: {run_command(pair)}")
    return code


def cmd_scenes(args: argparse.Namespace) -> int:
    cuts = scene_cuts(Path(args.clip), threshold=args.threshold)
    print(" ".join(f"{t:.6f}" for t in cuts) if cuts else "no scene cut over the threshold")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    if args.pair and args.pair not in PAIRS:
        print(f"unknown pair: {args.pair}; known: {', '.join(PAIRS)}", file=sys.stderr)
        return 2
    found = pair_for_run(args.run_id, run_shas(paths.runs_root() / args.run_id))
    if args.pair and found is not None and found.pair_id != args.pair:
        print(f"{args.run_id} is a run of {found.pair_id}, not {args.pair}", file=sys.stderr)
        return 2
    pair = PAIRS[args.pair] if args.pair else found
    if pair is not None and pair.kind not in CALIBRATION_KINDS:
        print(f"{pair.pair_id} is a {pair.kind} pair; only time splits and two takes calibrate", file=sys.stderr)
        return 2
    if pair is None:
        print(f"{args.run_id} is not a run of a listed pair; reading it as two walks where nothing changed", file=sys.stderr)
    try:
        result, out = calibrate_run(args.run_id, kind=pair.kind if pair else None)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(result["line"])
    print(f"mutual-NN pairs: {result['n_rows']}; under the run's tau_pre {result['tau_pre_run']:.3f}: {result['under_tau_pre']} of {result['n_rows']}")
    print(f"tau_pre implied: {result['tau_pre_implied']:.3f} ({'raised' if result['raised'] else 'unchanged'}; calibrated {str(result['calibrated_implied']).lower()})")
    for row in result["histogram"]:
        hi = "inf" if row["hi"] is None else f"{row['hi']:.2f}"
        print(f"  [{row['lo']:.2f}, {hi}) {row['n']}")
    print(f"wrote {out.relative_to(REPO) if out.is_relative_to(REPO) else out}")
    if result["raised"]:
        print("thresholds.json is not edited here; the lead freezes a raised tau_pre (section 5.10) and the README says which pair set it")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pseudo_pair.py", description="pseudo pairs for label-free calibration")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="the pairs, their sources and what exists")
    cut = sub.add_parser("cut", help="cut the clips into data/derived/")
    cut.add_argument("pairs", nargs="*", metavar="PAIR")
    scenes = sub.add_parser("scenes", help="scene cuts in a clip")
    scenes.add_argument("clip")
    scenes.add_argument("--threshold", type=float, default=0.25)
    cal = sub.add_parser("calibrate", help="noise floor and implied tau_pre from a finished run")
    cal.add_argument("run_id")
    cal.add_argument("--pair", help="the pair the run is of (refuses a before_after pair)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return {"list": cmd_list, "cut": cmd_cut, "scenes": cmd_scenes, "calibrate": cmd_calibrate}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
