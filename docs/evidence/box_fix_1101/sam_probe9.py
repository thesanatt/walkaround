"""SAM 3.1 with nine box prompts in one image request (LIVE=1 to call; $0.0025 a request): Spark's box and its eight
shifts by one Spark patch (28 px), object ids 1 to 9 (1 is Spark's own box). Answers in sam_probe9.json."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sam_probe as P  # noqa: E402

from walkaround import sam_client  # noqa: E402
from walkaround.claim import SamAnswer, sam_answer_from_run  # noqa: E402
from meta_sam_parser import FramePrompt, PromptObject  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "sam_probe9.json"
STEP = 28
SHIFTS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]


def shifted(b: list[float], size: tuple[int, int]) -> list[list[float]]:
    w, h = size
    out = []
    for sx, sy in SHIFTS:
        dx, dy = sx * STEP, sy * STEP
        dx = min(max(dx, -b[0]), w - b[2])  # kept inside the crop
        dy = min(max(dy, -b[1]), h - b[3])
        out.append([b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy])
    return out


def concept(b: list[float], size: tuple[int, int]) -> str:
    objs = [PromptObject(id=i + 1, prompts=[FramePrompt(frame=0, box=sam_client.prompt_box(s, size))]) for i, s in enumerate(shifted(b, size))]
    return sam_client.object_prompt(objs, size)


async def main() -> int:
    stored = json.loads(OUT.read_text()) if OUT.exists() else {}
    marks = P.probe_marks()
    todo = [m for m in marks if m["id"] not in stored]
    if os.environ.get("LIVE") == "1":
        async def one(m: dict) -> None:
            c = concept(m["spark"], (m["w"], m["h"]))
            try:
                run = await sam_client.segment_image(Path(m["path"]), c, run_id=P.RUN_ID, tag="box_fix9")
                stored[m["id"]] = sam_answer_from_run(run).model_dump(mode="json")
            except sam_client.SamRequestError as exc:
                stored[m["id"]] = SamAnswer(concept=c, status="failed", error=str(exc)[:300]).model_dump(mode="json")
        await asyncio.gather(*(one(m) for m in todo))
        OUT.write_text(json.dumps(stored, indent=1) + "\n")
    print(len(todo), "asked" if os.environ.get("LIVE") == "1" else "not asked")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
