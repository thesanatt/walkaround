"""Copy a finished run into samples/<run_id>/ so a clean clone replays it with no key (the app copies samples/ onto
the runs volume on start, orchestrator.seed_samples, and the first sample with a run.json is the sample pair).

    make_sample.py RUN_ID [--footage-credit TEXT] [--clip REL ...] [--max-mb 60] [--dry-run]
    make_sample.py --check [SAMPLE_DIR ...]

What goes in: every file of the run that app.servable and privacy.served_file_allowed(run_owner=False) both allow,
minus what no page of the app reads and the repo's size budget cannot carry: the Stays Put lanes (sam/<role>/sp_*),
the lane index files (sam/<role>/*.index.json, read only by the rebuild), the full keyframes (keyframes/, read only by
the eval labeling page) and every Stays Put clip not named with --clip. Then run.json (footage_credit replaced when
given, live_or_replay set to replay) and events.ndjson gzipped with a fixed mtime, so building the sample twice gives
the same bytes. An original, an unblurred proxy, the WAV, quarantine, uploads, spark.json, voice.json, seeds.json and
the pending event file never pass the allowlist, and --check reads every copied file against it again.

--plate-block PX pixelates every plate and face track box again (derived/tracks.json, bridged like the run's blur) on
each blurred display video with PX-pixel blocks, and --cover ROLE:F1-F2:X1,Y1,X2,Y2 adds a box (display-proxy pixels)
on frames F1 to F2 of that walk, for a plate the run's lanes missed. On a 960 px proxy the run's own block is 5 px,
which leaves a large plate readable where nothing else covers it. A video touched this way is re-encoded, so its
bytes no longer match the hash in videos/<role>.json and bundle/manifest_run.json; the README says so.

The sample in the repo was made with:
    make_sample.py pairC_core1 --footage-credit "Footage: uCO3D (Meta), CC BY 4.0, modified; narration: our own Piper TTS track"
        --clip clips/pickup/pickup_front_bumper_kf1_sp01.mp4 --clip clips/pickup/pickup_front_bumper_kf1_sp02.mp4
        --clip clips/pickup/pickup_trunk_lid_kf1_sp01.mp4 --plate-block 16
        --cover pickup:184-191:175,85,285,245 --cover pickup:190-193:140,195,280,300 --cover pickup:194-195:225,200,340,300
        --cover pickup:196-197:300,205,420,305 --cover pickup:198-199:340,230,450,315
(the run page opens on the front bumper, whose mark clip and control clip are the first two, and the trunk lid holds
the chips the demo is about; the covers are the rear plate seen edge-on at 12.3 to 12.8 s of the pickup walk, before
SAM 3.1's plate lane picked it up, and the bumper plate just after).
"""

from __future__ import annotations

import argparse
import gzip
import re
import shutil
import sys
from pathlib import Path, PurePosixPath
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from walkaround import contracts as c  # noqa: E402
from walkaround import paths, privacy, video  # noqa: E402
from walkaround.app import servable  # noqa: E402

SAMPLES = REPO / "samples"
MAX_MB = 60
APP_READS = ("run.json", "events.ndjson.gz")  # read by the app itself, never through the files route
NEVER = ("events.ndjson", "events.pending.ndjson", "spark.json", "voice.json", "seeds.json", "uploads.json")
COVER = re.compile(r"^([a-z0-9_]+):([0-9]+)-([0-9]+):([0-9]+),([0-9]+),([0-9]+),([0-9]+)$")
Cover = tuple[str, int, int, list[float]]


def parse_cover(text: str) -> Cover:
    """ROLE:F1-F2:X1,Y1,X2,Y2 -> (role, f1, f2, box) in display-proxy pixels."""
    m = COVER.match(text)
    if not m:
        raise argparse.ArgumentTypeError(f"{text!r}: write ROLE:F1-F2:X1,Y1,X2,Y2 (proxy frames and pixels)")
    role, f1, f2 = m.group(1), int(m.group(2)), int(m.group(3))
    box = [float(m.group(i)) for i in range(4, 8)]
    if f2 < f1 or box[2] <= box[0] or box[3] <= box[1]:
        raise argparse.ArgumentTypeError(f"{text!r}: frames or box run backwards")
    return role, f1, f2, box


def plate_boxes(root: Path, role: str, frames: int, fps: float, th: dict[str, Any]) -> dict[int, list[list[float]]]:
    """Every proxy frame's plate and face track boxes for one walk, from the run's derived/tracks.json, with the gap
    bridge privacy.track_boxes_at gives the run's own blur."""
    tracks_path = root / "derived" / "tracks.json"
    tracks = [t for t in c.read_json_list(tracks_path, c.PanelTrack) if t.role == role] if tracks_path.is_file() else []
    info = video.probe(root / "media" / f"{role}.proxy_blurred.mp4")
    size = (info.width, info.height)
    out: dict[int, list[list[float]]] = {}
    for f in range(frames):
        boxes = privacy.track_boxes_at(f, tracks, fps=fps, media_size_proxy=size, media_size_src=size, th=th)
        if boxes:
            out[f] = [list(b) for b in boxes]
    return out


def repixelate(path: Path, boxes: dict[int, list[list[float]]], th: dict[str, Any], *, block_px: int | None) -> None:
    """Pixelate the boxes on one blurred display video in place: block_px pixels, or the run's own block when None,
    with the run's dilation scaled to the video."""
    info = video.probe(path)
    scale = max(info.width, info.height) / privacy.SRC_REFERENCE_EDGE
    p = th["privacy"]
    block = block_px or max(2, int(round(p["pixel_block_px_at_4k"] * scale)))
    out = path.with_name(path.name + ".repixelated.mp4")
    video.render_blurred_proxy(path, out, boxes, block_px=block, dilate_frac=p["track_box_dilation_frac"], dilate_px=int(round(p["track_box_dilation_px"] * scale)))
    out.replace(path)


def dropped(rel: str, clips: set[str]) -> str | None:
    """Why a servable file stays out of the sample, or None when it goes in."""
    p = PurePosixPath(rel)
    if p.parts[0] == "sam" and p.name.startswith("sp_"):
        return "Stays Put lane"
    if p.parts[0] == "sam" and p.name.endswith(".index.json"):
        return "lane index"
    if p.parts[0] == "keyframes":
        return "full keyframe"
    if p.parts[0] == "clips" and rel not in clips:
        return "Stays Put clip not named"
    return None


def plan(run_id: str, clips: set[str]) -> tuple[c.RunManifest, list[str], dict[str, int]]:
    """The run, the run-relative files that go in (sorted) and the bytes left out per reason."""
    root = paths.runs_root() / run_id
    run = c.RunManifest.model_validate_json((root / "run.json").read_text(encoding="utf-8"))
    keep: list[str] = []
    out: dict[str, int] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if not (servable(rel) and privacy.served_file_allowed(run, rel, run_owner=False)):
            continue
        why = dropped(rel, clips)
        if why:
            out[why] = out.get(why, 0) + path.stat().st_size
            continue
        keep.append(rel)
    return run, keep, out


def problems(sample: Path, *, max_mb: float = MAX_MB) -> list[str]:
    """What is wrong with a sample folder: a file the allowlist refuses, a missing run.json, the size budget."""
    run_json = sample / "run.json"
    if not run_json.is_file():
        return [f"{sample.name}: no run.json"]
    run = c.RunManifest.model_validate_json(run_json.read_text(encoding="utf-8"))
    found = []
    if run.run_id != sample.name:
        found.append(f"{sample.name}: run.json names run {run.run_id}")
    if not run.keep or run.keep_plate:
        found.append(f"{sample.name}: a sample is a keep run with the plate blurred (keep {run.keep}, keep_plate {run.keep_plate})")
    total = 0
    for path in sorted(p for p in sample.rglob("*") if p.is_file()):
        rel = path.relative_to(sample).as_posix()
        total += path.stat().st_size
        if rel in APP_READS:
            continue
        if rel in NEVER or not (servable(rel) and privacy.served_file_allowed(run, rel, run_owner=False)):
            found.append(f"{sample.name}: {rel} is not served to the public")
    if total > max_mb * 1_000_000:
        found.append(f"{sample.name}: {total / 1e6:.1f} MB, over {max_mb} MB")
    events = sample / "events.ndjson.gz"
    if not events.is_file():
        found.append(f"{sample.name}: no events.ndjson.gz, so nothing to replay")
    else:
        try:
            gzip.decompress(events.read_bytes())
        except OSError as exc:
            found.append(f"{sample.name}: events.ndjson.gz does not unpack: {exc}")
    return found


def write(run_id: str, *, footage_credit: str | None, clips: set[str], covers: list[Cover], plate_block: int | None, max_mb: float, dry_run: bool) -> int:
    run, files, left_out = plan(run_id, clips)
    root = paths.runs_root() / run_id
    missing = sorted(clips - set(files))
    if missing:
        sys.stderr.write(f"not servable or not in the run: {', '.join(missing)}\n")
        return 2
    proxies = {rel.split("/")[1].removesuffix(".proxy_blurred.mp4") for rel in files if rel.startswith("media/") and rel.endswith(".proxy_blurred.mp4")}
    walks = {role for role, *_ in covers} | (proxies if plate_block else set())
    no_video = sorted(walks - proxies)
    if no_video:
        sys.stderr.write(f"no blurred display video to cover for: {', '.join(no_video)}\n")
        return 2
    events = gzip.compress((root / "events.ndjson").read_bytes(), compresslevel=9, mtime=0)
    # on any server the copy is a replay: no start, top-up or claim, and it counts against no day's live runs
    run = run.model_copy(update={"live_or_replay": c.LiveOrReplay.REPLAY})
    if footage_credit:
        run = run.model_copy(update={"footage_credit": " ".join(footage_credit.split())})
    total = sum((root / rel).stat().st_size for rel in files) + len(events) + len(c.json_bytes(run))
    for why, size in sorted(left_out.items()):
        print(f"left out ({why}): {size / 1e6:.1f} MB")
    print(f"{len(files) + 2} files, {total / 1e6:.1f} MB")
    if total > max_mb * 1_000_000:
        sys.stderr.write(f"over {max_mb} MB; name fewer clips\n")
        return 1
    if dry_run:
        return 0
    target = SAMPLES / run_id
    staging = SAMPLES / ".staging" / run_id  # named like the run, so problems() reads it as that sample
    shutil.rmtree(staging.parent, ignore_errors=True)
    for rel in files:
        dest = staging / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / rel, dest)
    (staging / "events.ndjson.gz").write_bytes(events)
    c.write_json(staging / "run.json", run)
    for role in sorted(walks):
        record = c.read_json(staging / "videos" / f"{role}.json", c.VideoRecord)
        assert record.proxy is not None
        boxes = plate_boxes(staging, role, record.proxy.frames, record.proxy.fps, run.thresholds) if plate_block else {}
        for _role, f1, f2, box in (cv for cv in covers if cv[0] == role):
            for f in range(f1, f2 + 1):
                boxes.setdefault(f, []).append(box)
        repixelate(staging / "media" / f"{role}.proxy_blurred.mp4", boxes, run.thresholds, block_px=plate_block)
        print(f"pixelated plate boxes again on {len(boxes)} frames of the {role} display video")
    found = problems(staging, max_mb=max_mb)
    if found:
        shutil.rmtree(staging.parent, ignore_errors=True)
        sys.stderr.write("\n".join(found) + "\n")
        return 1
    shutil.rmtree(target, ignore_errors=True)
    staging.rename(target)
    staging.parent.rmdir()
    print(f"wrote {target}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="copy a finished run's public files into samples/<run_id>/")
    parser.add_argument("run_id", nargs="?")
    parser.add_argument("--footage-credit", help="replace run.json's footage_credit (shown under the walk player)")
    parser.add_argument("--clip", action="append", default=[], metavar="REL", help="a Stays Put clip to keep, run-relative (repeat)")
    parser.add_argument("--cover", action="append", default=[], type=parse_cover, metavar="ROLE:F1-F2:X1,Y1,X2,Y2", help="pixelate one more box on a walk's display video (repeat)")
    parser.add_argument("--plate-block", type=int, metavar="PX", help="pixelate every plate and face track box again with PX-pixel blocks")
    parser.add_argument("--max-mb", type=float, default=MAX_MB)
    parser.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    parser.add_argument("--check", nargs="*", metavar="SAMPLE_DIR", help="check sample folders (default every samples/<id>/)")
    args = parser.parse_args(argv)
    if args.check is not None:
        if args.check:
            folders = [Path(p) for p in args.check]
        else:
            folders = sorted(p for p in SAMPLES.iterdir() if p.is_dir() and not p.name.startswith(".")) if SAMPLES.is_dir() else []
        found = [line for folder in folders for line in problems(folder, max_mb=args.max_mb)]
        print("\n".join(found) if found else f"{len(folders)} sample folder(s): every file is served to the public, under {args.max_mb:g} MB")
        return 1 if found else 0
    if not args.run_id:
        parser.error("name a run, or pass --check")
    return write(args.run_id, footage_credit=args.footage_credit, clips=set(args.clip), covers=args.cover, plate_block=args.plate_block, max_mb=args.max_mb, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
