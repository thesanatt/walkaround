"""The majority look on stored crops, live (lead decision 2 of 11/2): the match stage of a scratch copy of each run is
derived again under its frozen thresholds plus the matching leaves config/thresholds.json has and the frozen file lacks
(K20 on hero_focus2, and the two R24 leaves), with the second look live on Spark Open. Each look's first answer is the
stored one wherever the question is the same (SecondLook reuses a stored row by its question hash), so only the repeats
and the looks the new rules newly ask are calls. Then the scratch copy is rebuilt with the looks replayed, which must
give the same verdicts.

Calls go to Baseten on the hackathon key only: the account key slot is left out and hosted Muse Spark is blocked (a
look Spark Open cannot answer does not run). Ledger rows carry run id r24_<run>, so they never count toward a run.

Usage: .venv/bin/python docs/evidence/vouch_majority_1102/live.py SCRATCH_DIR RUN_ID [RUN_ID ...]
writes live_<run>.json (rows, truth, every look answer) here.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))
from replay import NEVER, variant_th  # noqa: E402
from truth import truth_rows  # noqa: E402

from walkaround import contracts as c  # noqa: E402
from walkaround import open_weights, paths, pipeline, second_look, spark_client  # noqa: E402
from walkaround.orchestrator import _runs_root  # noqa: E402

HACKATHON = [s for s in open_weights.KEY_SLOTS if s.name == "hackathon"]
_real_key_slots = open_weights.key_slots
open_weights.key_slots = lambda: [s for s in _real_key_slots() if s.name == "hackathon"]  # the account key is never asked
assert HACKATHON and open_weights.key_slots() == HACKATHON, "BASETEN_HACKATHON_API_KEY is not set"


def _no_hosted(*_a: object, **_k: object) -> None:
    raise RuntimeError("hosted Muse Spark is off for this study")


spark_client.second_look = _no_hosted
_real_open = open_weights.second_look


def _billed(*a: object, **kw: object) -> object:
    kw["ledger_run_id"] = f"r24_{kw.get('run_id')}"
    return _real_open(*a, **kw)


open_weights.second_look = _billed

if __import__("os").environ.get("DRY") == "1":  # count the calls with a fixed answer and no network
    from tests.test_counter_confirm import _record

    def _dry(*_a: object, **kw: object) -> object:
        schema = str(kw.get("schema_name", "look_v1"))
        rec = _record({"mark_in_box": "yes", "same_mark_at_spot": "no", "reason": "dry"} if schema == "counter2_v1" else {"verdict": "same", "reason": "dry"}, schema=schema)
        if schema == "counter2_v1":
            seen, verdict, reason = open_weights.counter2_answer(rec)
            return spark_client.LookResult(verdict=verdict, reason=reason, record=rec, attempts=[rec], mark_seen=seen)
        verdict, reason = open_weights.look_answer(rec)
        return spark_client.LookResult(verdict=verdict, reason=reason, record=rec, attempts=[rec])

    open_weights.second_look = _dry


def summary(vs: list[c.Verdict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in vs:
        out[v.label.value] = out.get(v.label.value, 0) + 1
    return out


def one(orch: object, scratch: Path, run_id: str) -> dict:
    run = orch.load_run(run_id)  # type: ignore[attr-defined]
    th = variant_th(orch.run_th(run), "both", current=True)  # type: ignore[attr-defined]
    target = scratch / run_id
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(paths.run_dir(run_id), target, ignore=shutil.ignore_patterns(*NEVER))
    stored = json.loads((target / "derived" / "looks.json").read_text())
    with _runs_root(scratch):
        # the product's rebuild, with the live look in place of the replay: every stage as a rebuild runs it
        look = second_look.make_second_look(run_id, th=th, engine="spark_open")
        orch.overrides["second_look.replay_second_look"] = lambda _rid, *, th=None: look  # type: ignore[attr-defined]
        try:
            orch._rebuild_in_place(run, th)  # type: ignore[attr-defined]
        finally:
            orch.overrides.pop("second_look.replay_second_look", None)  # type: ignore[attr-defined]
        live_bytes = (paths.matches_json(run_id).read_bytes(), paths.verdicts_json(run_id).read_bytes())
        rows = json.loads(paths.looks_json(run_id).read_text())
        orch._rebuild_in_place(run, th)  # type: ignore[attr-defined]  # now the looks replayed from looks.json
        replayed = (paths.matches_json(run_id).read_bytes(), paths.verdicts_json(run_id).read_bytes())
        vs = c.read_json_list(paths.verdicts_json(run_id), c.Verdict)
    reused = {(r["pickup_mark"], r["return_mark"]) for r in stored}
    looks = [
        {
            "key": [r["pickup_mark"], r["return_mark"]],
            "kind": r.get("kind") or "pair",
            "ask": r.get("ask") or 0,
            "stored": (r.get("ask") or 0) == 0 and (r["pickup_mark"], r["return_mark"]) in reused and any(s.get("question_sha256") == r.get("question_sha256") and s["pickup_mark"] == r["pickup_mark"] and s["return_mark"] == r["return_mark"] for s in stored),
            "mark_seen": r.get("mark_seen"),
            "verdict": r.get("verdict"),
            "key_slot": r.get("key_slot"),
            "engine": r.get("engine"),
            "request_id": r.get("request_id"),
            "reason": r.get("reason"),
        }
        for r in rows
    ]
    out = {
        "run": run_id,
        "summary": summary(vs),
        "rows": {"|".join([v.panel_id.value, *v.mark_ids]): [v.label.value, v.trace.rule_ids, v.counter_note, v.type_note] for v in vs},
        "truth": truth_rows(run_id, vs),
        "looks": looks,
        "calls": sum(1 for x in looks if not x["stored"] and x["engine"] is not None),
        "rebuild_replays_live": replayed == live_bytes,
    }
    (HERE / f"{'dry' if __import__('os').environ.get('DRY') == '1' else 'live'}_{run_id}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"run": run_id, "summary": out["summary"], "truth": [t["label"] for t in out["truth"]], "calls": out["calls"], "rebuild_replays_live": out["rebuild_replays_live"]}))
    return out


def main() -> None:
    scratch, runs = Path(sys.argv[1]), sys.argv[2:]
    orch = pipeline.make_orchestrator()
    for run_id in runs:
        one(orch, scratch, run_id)


if __name__ == "__main__":
    main()
