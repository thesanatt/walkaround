"""Verdict changes between the base and reg replays, with truth where a run has it. Prints a table row per run and
every verdict whose label or rule ids changed; writes compare.json.

Truth for a verdict: synth runs, the generator mark under each box (study.synth_truth_for) and its walks; friend_pair2,
Sanat's verdict checks (eval/verdict_checks/friend_pair2.json). A false "new" or "not seen at return" is a one-walk label
on a mark the truth has in both walks (or on no real mark); a false "already there" is a pair the truth does not have
as one old mark.

Usage: compare.py RUN [RUN ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import study  # noqa: E402

DECIDED = ("already_there", "new_or_unseen", "not_seen_at_return")


def truth_of(run: str, v: dict) -> str | None:
    """already_there, new_or_unseen, not_seen_at_return, no_mark or None (unknown) for a verdict."""
    if run.startswith("synth"):
        hits = []
        for mid in v["mark_ids"]:
            m = study.marks(run).get(mid)
            if m is None or m.bbox_crop is None:
                return None
            other = next((k for k in study.keyframes(run) if str(study.keyframes(run)[k].role) != str(m.role) and study.keyframes(run)[k].panel_id == m.panel_id), None)
            hit, _ = study.synth_truth_for(run, m, other) if other else (None, None)
            hits.append(hit)
        if not any(hits):
            return "no_mark"
        gm = next(g for g in study.truth()["marks"] if g["id"] == next(h for h in hits if h))
        walks = set(gm["walks"])
        return "already_there" if walks == {"pickup", "return"} else ("new_or_unseen" if walks == {"return"} else "not_seen_at_return")
    checks = study.checks(run)
    labels = {checks[m] for m in v["mark_ids"] if m in checks}
    return labels.pop() if len(labels) == 1 else None


def load(run: str, variant: str) -> dict:
    return json.loads((HERE / f"replay_{run}_{variant}.json").read_text())


def main(runs: list[str]) -> None:
    out = []
    for run in runs:
        base, reg = load(run, "base"), load(run, "reg")
        key = lambda v: tuple(sorted(v["mark_ids"]))  # noqa: E731
        b = {key(v): v for v in base["verdicts"]}
        r = {key(v): v for v in reg["verdicts"]}
        changes = []
        for k in sorted(set(b) | set(r)):
            vb, vr = b.get(k), r.get(k)
            if vb is not None and vr is not None and vb["label"] == vr["label"] and vb["rule_ids"] == vr["rule_ids"]:
                continue
            ref = vr or vb
            t = truth_of(run, ref)
            changes.append({"marks": list(k), "panel": ref["panel"], "base": vb and vb["label"], "reg": vr and vr["label"], "base_rules": vb and vb["rule_ids"], "reg_rules": vr and vr["rule_ids"], "truth": t})
        def bad(vs: list[dict], label: str) -> int:
            n = 0
            for v in vs:
                if v["label"] != label:
                    continue
                t = truth_of(run, v)
                if t is not None and t != label:
                    n += 1
            return n
        row = {
            "run": run,
            "base": base["summary"],
            "reg": reg["summary"],
            "false_new_base": bad(base["verdicts"], "new_or_unseen") + bad(base["verdicts"], "not_seen_at_return"),
            "false_new_reg": bad(reg["verdicts"], "new_or_unseen") + bad(reg["verdicts"], "not_seen_at_return"),
            "false_already_base": bad(base["verdicts"], "already_there"),
            "false_already_reg": bad(reg["verdicts"], "already_there"),
            "k18": sum("K18" in v["rule_ids"] for v in reg["verdicts"]),
            "registered_pairs": sum(v["registered"] for v in reg["verdicts"]),
            "changes": changes,
        }
        out.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "changes"}))
        for ch in changes:
            print("   ", ch["panel"], ch["marks"], ch["base"], "->", ch["reg"], ch["reg_rules"], "truth:", ch["truth"])
    (HERE / "compare.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
