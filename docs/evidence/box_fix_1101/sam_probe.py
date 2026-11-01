"""A handful of SAM 3.1 image requests on stored mark crops (LIVE=1 to call; $0.0025 each): Spark's box as the
prompt, the box grown by one Spark patch (28 px) a side, and a point at the box centre. Answers kept in
sam_probe.json (claim.SamAnswer rows, masks as their wire payload) so the scoring re-runs with no call."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import truth as T  # noqa: E402

from walkaround import sam_client  # noqa: E402
from walkaround.claim import SamAnswer, sam_answer_from_run  # noqa: E402
from walkaround.contracts import Mark, read_json_list  # noqa: E402
from walkaround import paths  # noqa: E402
from meta_sam_parser import FramePrompt, PromptObject, PromptPoint  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "sam_probe.json"
RUN_ID = "box_fix_1101"
SYNTH = [("synth_pair2_r8", "pickup_left_rear_door_kf1_m1"), ("synth_pair2", "return_hood_kf1_m1"),
         ("synth_pair2_r8", "return_left_front_fender_kf2_m1"), ("synth_pair2_r9", "return_hood_kf2_m1"),
         ("synth_pair2_r9", "pickup_right_rear_fender_kf1_m1")]
PAIRC = [("pickup_trunk_chips", 0), ("return_trunk_chips", 0), ("alto_hood_dent", 0)]


def probe_marks() -> list[dict]:
    out = []
    items = {(it["run"], it["kf"].kf_id): it for it in T.synth_items([r for r, _ in SYNTH])}
    for run, mid in SYNTH:
        m = next(m for m in read_json_list(paths.marks_json(run), Mark) if m.mark_id == mid)
        it = items[(run, m.kf_id)]
        out.append({"id": f"{run}:{mid}", "path": it["path"], "w": it["w"], "h": it["h"], "spark": m.bbox_crop, "truth": [g["box"] for g in it["truth"]]})
    specs = {s["name"]: s for s in T.pairc_specs()}
    answers = [json.loads(line) for line in T.STUDY.read_text().splitlines()]
    for name, rep in PAIRC:
        s = specs[name]
        w, h = Image.open(s["path"]).size
        a = next(a for a in answers if a["crop_id"] == f"pairC_{name}" and a["arm"] == "base" and a["effort"] == "low" and a["rep"] == rep)
        box = next(b["box"] for b in a["boxes"] if any(T.target_hit(t, b["box"], max(w, h)) for t in s["targets"]))
        out.append({"id": f"pairC:{name}", "path": s["path"], "w": w, "h": h, "spark": box, "targets": s["targets"]})
    return out


def prompts(m: dict) -> dict[str, str]:
    size = (m["w"], m["h"])
    b = m["spark"]
    grown = [b[0] - 28, b[1] - 28, b[2] + 28, b[3] + 28]
    cx, cy = int(round((b[0] + b[2]) / 2)), int(round((b[1] + b[3]) / 2))
    one = lambda fp: sam_client.object_prompt([PromptObject(id=1, prompts=[fp])], size)  # noqa: E731
    return {"box": one(FramePrompt(frame=0, box=sam_client.prompt_box(b, size))),
            "grown": one(FramePrompt(frame=0, box=sam_client.prompt_box(grown, size, pad_px=0))),
            "point": one(FramePrompt(frame=0, points=[PromptPoint(x=cx, y=cy)]))}


async def main() -> int:
    live = os.environ.get("LIVE") == "1"
    stored = json.loads(OUT.read_text()) if OUT.exists() else {}
    marks = probe_marks()
    todo = []
    for i, m in enumerate(marks):
        for variant, concept in prompts(m).items():
            if variant == "point" and i % 2:
                continue  # points on half of them
            key = f"{m['id']}|{variant}"
            if key not in stored:
                todo.append((key, m, concept))
    if live:
        async def one(key: str, m: dict, concept: str) -> None:
            try:
                run = await sam_client.segment_image(Path(m["path"]), concept, run_id=RUN_ID, tag="box_fix")
                stored[key] = sam_answer_from_run(run).model_dump(mode="json")
            except sam_client.SamRequestError as exc:
                stored[key] = SamAnswer(concept=concept, status="failed", error=str(exc)[:300]).model_dump(mode="json")
        await asyncio.gather(*(one(*t) for t in todo))
        OUT.write_text(json.dumps(stored, indent=1) + "\n")
    print(len(marks), "marks;", len(todo), "asked" if live else "not asked (no LIVE=1)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
