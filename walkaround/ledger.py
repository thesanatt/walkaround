"""Every model call writes one row here. LEDGER.md and the README copy numbers from this file,
never from memory."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from .config import PRICES, RUNS_DIR

LEDGER_PATH = RUNS_DIR / "ledger.jsonl"
_WRITE_LOCK = threading.Lock()  # marks.derive_marks records Spark rows from several threads at once


def cost_for(
    kind: str,
    *,
    frames: int | None = None,
    images: int | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cached_tokens: int | None = None,
    audio_seconds: float | None = None,
) -> float:
    """Dollars for one call, from the price table. kind is sam_video, sam_image, spark or voice."""
    if kind == "sam_video":
        billed = max(int(frames or 0), PRICES["sam_video_min_frames"])
        return billed * PRICES["sam_video_frame"]
    if kind == "sam_image":
        return (images or 1) * PRICES["sam_image"]
    if kind == "spark":
        cached = cached_tokens or 0
        uncached = max((input_tokens or 0) - cached, 0)
        return (
            uncached * PRICES["spark_input_token"]
            + cached * PRICES["spark_cached_token"]
            + (output_tokens or 0) * PRICES["spark_output_token"]
        )
    if kind == "voice":
        return int(audio_seconds or 0) * PRICES["voice_second"]
    raise ValueError(f"unknown ledger kind {kind!r}")


def record(kind: str, model: str, *, note: str = "", run_id: str | None = None, **usage: Any) -> dict[str, Any]:
    """Append one row and return it. usage keys match cost_for plus anything extra (seconds, reasoning_tokens)."""
    cost_keys = {"frames", "images", "input_tokens", "output_tokens", "cached_tokens", "audio_seconds"}
    cost = cost_for(kind, **{k: v for k, v in usage.items() if k in cost_keys})
    row = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "kind": kind,
        "model": model,
        "run_id": run_id,
        "note": note,
        "usd": round(cost, 6),
        **usage,
    }
    line = json.dumps(row) + "\n"
    with _WRITE_LOCK:
        LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LEDGER_PATH.open("a", encoding="utf-8") as fh:
            fh.write(line)
    return row


def rows(path: Path | None = None) -> list[dict[str, Any]]:
    """Every row of the ledger (LEDGER_PATH read at call time, so a test that points it elsewhere is honoured)."""
    path = Path(path) if path is not None else LEDGER_PATH
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summary(path: Path | None = None) -> dict[str, Any]:
    """Totals by kind and overall, in dollars."""
    out: dict[str, Any] = {"total_usd": 0.0, "by_kind": {}}
    for row in rows(path):
        kind = row["kind"]
        bucket = out["by_kind"].setdefault(kind, {"calls": 0, "usd": 0.0})
        bucket["calls"] += 1
        bucket["usd"] += row["usd"]
        out["total_usd"] += row["usd"]
    out["total_usd"] = round(out["total_usd"], 4)
    for bucket in out["by_kind"].values():
        bucket["usd"] = round(bucket["usd"], 4)
    return out
