"""Ask Muse Spark Open on Baseten the stored second looks again, on the same outlined crops, and print each new answer
beside the stored one. The default folder is docs/evidence/second_look: three looks on the uCO3D pair C trunk lid from
run b11_fix_smoke_1031 (one true pair, two controls). Any run's derived/looks.json works too, with its crops/ folder.

    LIVE=1 second_look_again.py [FOLDER]

Needs BASETEN_HACKATHON_API_KEY (meta-models/muse-spark-open-1026) or BASETEN_API_KEY (meta-models/muse-spark-open)
in walkaround/.env; the hackathon key is tried first. Each request writes a ledger row (runs/ledger.jsonl) and a
provenance line (runs/<run_id>/open_weights.ndjson, run id second_look_again) naming the key slot that asked; about
$0.004 a look. Without LIVE=1 it prints what it would ask and spends nothing. The answer can differ from the stored
one: a model is not bit-deterministic, which is why the run keeps every answer and a model can only lower a label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from walkaround import open_weights  # noqa: E402
from walkaround.contracts import InputRef, PanelId  # noqa: E402

DEFAULT = REPO / "docs" / "evidence" / "second_look"
RUN_ID = "second_look_again"


def by_sha(folder: Path) -> dict[str, Path]:
    """sha256 -> file for every JPEG under folder, so stored input_refs find their crops wherever they sit."""
    return {hashlib.sha256(p.read_bytes()).hexdigest(): p for p in sorted(folder.rglob("*.jpg"))}


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"{path}: not an image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def looks(folder: Path) -> list[dict]:
    path = folder / "looks.json" if (folder / "looks.json").is_file() else folder / "derived" / "looks.json"
    return json.loads(path.read_text(encoding="utf-8"))


def plan(folder: Path) -> list[tuple[dict, list[Path]]]:
    """Each stored look with its two crops (pickup, return), found by sha256; a look whose crops are missing is skipped."""
    files = by_sha(folder)
    out = []
    for row in looks(folder):
        refs = row.get("input_refs") or []
        crops = [files.get(ref["sha256"]) for ref in refs]
        if len(crops) == 2 and all(crops):
            out.append((row, crops))
        else:
            print(f"skip {row.get('pickup_mark')} / {row.get('return_mark')}: crops not in {folder}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ask the stored second looks again on Muse Spark Open (Baseten)")
    parser.add_argument("folder", nargs="?", type=Path, default=DEFAULT)
    args = parser.parse_args(argv)
    todo = plan(args.folder)
    if os.environ.get("LIVE") != "1":
        for row, crops in todo:
            print(f"would ask {row['panel_id']}: {row['pickup_mark']} vs {row['return_mark']} on {crops[0].name} and {crops[1].name}; stored {row['verdict']} from {row['model']}")
        print("set LIVE=1 to spend (about $0.004 a look on Baseten)")
        return 2
    for row, crops in todo:
        refs = [InputRef(path=p.relative_to(REPO).as_posix() if p.is_relative_to(REPO) else p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in crops]
        result = open_weights.second_look(read_rgb(crops[0]), read_rgb(crops[1]), panel_id=PanelId(row["panel_id"]), run_id=RUN_ID, input_refs=refs, pair=(row["pickup_mark"], row["return_mark"]))
        slot = open_weights.key_slot_of(result.record) or "hosted"
        print(f"{row['pickup_mark']} vs {row['return_mark']}: now {result.verdict.value} ({result.record.model}), stored {row['verdict']} ({row['model']})")
        print(f"  {slot} key, {result.record.ms} ms: {result.reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
