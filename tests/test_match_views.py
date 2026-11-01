"""K15 end to end (lead 10/31): a run built with today's code on the offline self-pair, with Muse Spark flagging glare
on every crop, compares its panels (G2 is retired), notes the flagged views on marks seen in both walks, prints the
note on the sheet, and verifies "rebuild identical". Reuses the offline pipeline fixture of tests/test_pipeline.py: no
key, no socket, the test clip only."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from typing import Any

from walkaround import contracts as c, paths, pipeline, spark_client

from .test_pipeline import CLIP, SPARK, offline  # noqa: F401  (offline is the fixture)

NOTE = "pickup view flagged glare, return view flagged glare: the mark is in the same place in both walks, so it was there at pickup, whatever it is"


def _glare_answer() -> dict[str, Any]:
    answer = copy.deepcopy(SPARK)
    for item in answer["output"]:
        for part in item.get("content") or []:
            if part.get("type") == "output_text":
                parsed = json.loads(part["text"])
                parsed["conditions"]["glare"] = True
                part["text"] = json.dumps(parsed)
    return answer


class GlareSpark:
    def __init__(self, calls: dict[str, list[str]]) -> None:
        self.calls = calls
        self.answer = _glare_answer()
        self.responses = SimpleNamespace(create=self.create)

    def create(self, **request: Any) -> Any:
        self.calls["spark"].append(request["model"])
        answer = copy.deepcopy(self.answer)
        answer["id"] = f"resp_spark_{len(self.calls['spark'])}"
        text = next(p["text"] for item in answer["output"] if item.get("type") == "message" for p in item["content"] if p.get("type") == "output_text")
        return SimpleNamespace(model_dump=lambda: copy.deepcopy(answer), output_text=text)


async def test_a_run_with_flagged_views_compares_notes_and_verifies_identical(offline: dict[str, Any]) -> None:  # noqa: F811
    orch, calls = offline["orch"], offline["calls"]
    spark_client.set_client(GlareSpark(calls))
    final = await pipeline.run_pair_async(CLIP, CLIP, "selfpair_glare", ["ping pong ball"], orch=orch)
    run_id = final.run_id
    assert final.status == c.RunStatus.DONE and final.errors == [], final.errors
    kfs = c.read_json_list(paths.keyframes_json(run_id), c.Keyframe)
    assert kfs and all(k.conditions.glare for k in kfs)
    matches = c.read_json_list(paths.matches_json(run_id), c.PanelMatch)
    compared = [pm for pm in matches if pm.state == c.MatchState.COMPARABLE]
    assert compared and not any("G2" in pm.reasons for pm in matches)  # before 10/31 every flagged panel failed G2
    verdicts = c.read_json_list(paths.verdicts_json(run_id), c.Verdict)
    on_compared = [v for v in verdicts if v.panel_id in {pm.panel_id for pm in compared}]
    # every mark matches itself in the same place: L3 with the note and never K15 (a mark's own cap, K3 here, may lower it)
    bad = [(v.verdict_id, v.trace.rule_ids, v.view_note) for v in on_compared if not ("L3" in v.trace.rule_ids and v.view_note == NOTE and "K15" not in v.trace.rule_ids)]
    assert on_compared and not bad, bad
    assert any(v.label == c.VerdictLabel.ALREADY_THERE for v in on_compared)
    sheet = c.read_json(paths.sheet_json(run_id), c.SheetModel)
    assert NOTE in [line.view_line for block in sheet.panels for line in block.verdicts]
    assert NOTE in paths.sheet_html(run_id).read_text() and "K15" in sheet.appendices["B glossary"]
    counts = {k: len(v) for k, v in calls.items()}
    assert pipeline.verify(run_id, orch=orch) == []  # built with today's code: rebuild identical
    assert {k: len(v) for k, v in calls.items()} == counts  # and verify asked nothing
