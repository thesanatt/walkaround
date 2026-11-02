"""The marks_v1 call asked again on a stored run's keyframe crops with the Nov 2 prompt switches on
(matching.colour_on_paint and matching.conditions_on_panel), the run's own fixtures, seams and effort otherwise. LIVE:
one Spark call per (crop sha256, panel) not yet stored, ledger run id prompt_reask_1102, a stop on the ET Meta day.
Answers go to reask_<run>.json (SparkRequestRecord rows), read by replay.py. ARM=old asks the same crops with the old
prompt (both switches off) into reask_old_<run>.json, the control: a fresh answer differs from the stored one even with
the same prompt.

Usage: [ARM=old] LIVE=1 .venv/bin/python docs/evidence/hero_1102/reask.py RUN_ID [kf_id_substring,...]
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
from walkaround import config, marks as marks_mod, match, spark_client  # noqa: E402
from walkaround.contracts import InputRef, Keyframe, MarksStatus, SparkRequestRecord, read_json_list, write_json  # noqa: E402

RUN_ID = "prompt_reask_1102"
PER_CALL = 0.0075
DAY_STOP_USD = float(os.environ.get("DAY_STOP_USD", "40.00"))


def day_spent() -> float:
    et = ZoneInfo("America/New_York")
    day = dt.datetime.now(et).date()
    total = 0.0
    for line in (REPO / "runs" / "ledger.jsonl").open():
        row = json.loads(line)
        try:
            t = dt.datetime.fromisoformat(str(row.get("ts", "")).replace("Z", "+00:00"))
        except ValueError:
            continue
        if t.tzinfo is not None and t.astimezone(et).date() == day and row.get("engine") == "meta-model-api":
            total += float(row.get("usd") or 0)
    return total


def run_th(run: str) -> dict:
    return json.loads((REPO / "runs" / run / "run.json").read_text())["thresholds"]


async def main(run: str, only: list[str]) -> None:
    config.load_env()
    th = run_th(run)
    old = os.environ.get("ARM") == "old"
    out = HERE / (f"reask_old_{run}.json" if old else f"reask_{run}.json")
    stored = read_json_list(out, SparkRequestRecord) if out.exists() else []
    have = {(r.input_refs[0].sha256, r.input_refs[0].path) for r in stored if r.valid and r.input_refs}
    kfs = [k for k in read_json_list(REPO / "runs" / run / "derived" / "keyframes.json", Keyframe) if k.crop is not None and k.marks_status != MarksStatus.NOT_RUN]
    if only:
        kfs = [k for k in kfs if any(s in k.kf_id for s in only)]
    todo, seen = [], set()
    for k in kfs:
        key = (k.crop.sha256, k.crop.path)
        if key in have or key in seen:
            continue
        seen.add(key)
        todo.append(k)
    spent = day_spent()
    print(f"{run}: {len(todo)} crops to ask (about ${len(todo) * PER_CALL:.2f}); Meta day ${spent:.2f}, stop ${DAY_STOP_USD:.2f}")
    if not todo or os.environ.get("LIVE") != "1" or spent + len(todo) * PER_CALL > DAY_STOP_USD:
        print("not asking")
        return
    sem = asyncio.Semaphore(6)
    flags = {"fixtures": match.fixtures_not_damage(th), "seams": match.seams_not_marks(th), "colour": not old, "panel_conditions": not old}

    async def one(k: Keyframe) -> SparkRequestRecord:
        from PIL import Image
        import numpy as np

        rgb = np.asarray(Image.open(REPO / "runs" / run / k.crop.path).convert("RGB"))
        async with sem:
            res = await asyncio.to_thread(
                spark_client.find_marks, rgb, k.panel_id, landmarks=marks_mod.landmarks_for(k.panel_id, th), effort=str(th["spark"]["effort"]),
                run_id=RUN_ID, input_ref=InputRef(path=k.crop.path, sha256=k.crop.sha256), **flags,
            )
        return res.record

    got = await asyncio.gather(*(one(k) for k in todo))
    write_json(out, [*stored, *got])
    print(f"asked {len(got)}; valid {sum(r.valid for r in got)}; Meta day now ${day_spent():.2f}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], [s for s in (sys.argv[2].split(",") if len(sys.argv) > 2 else []) if s]))
