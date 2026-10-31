"""Does SAM 3.1 answer two identical requests identically? Two 'person' video lanes on one Files API upload of the
test clip's 15 fps proxy, then two 'face' image checks on one proxy frame. About $0.04; needs LIVE=1.

    LIVE=1 .venv/bin/python scripts/probe_sam_determinism.py [--clip data/test/pingpong.mp4] [--out runs/probe_det]

Measured 10/31 (run id probe_det_1031): 18 of 178 video boxes one pixel apart, lane texts different, image boxes
identical with one face mask 2 px apart. The orchestrator therefore reuses answers for byte-identical inputs."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from walkaround import contracts as c  # noqa: E402
from walkaround import paths, sam_client, sam_wire, video  # noqa: E402


async def probe(clip: Path, out: Path, run_id: str) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    m = c.load_thresholds()["media"]
    proxy = out / "proxy_15fps.mp4"
    info = video.make_proxy(clip, proxy, fps=float(m["proxy_fps"]), max_edge=int(m["proxy_max_edge"]), preset=m["proxy_preset"], crf=int(m["proxy_crf"]))
    pts = video.pts_map(proxy)
    upload = await sam_client.upload_file(proxy, run_id=run_id)
    try:
        meta, size = sam_client.sam_metadata(), (info.width, info.height)
        lanes = await asyncio.gather(
            *(sam_client.segment_video(upload.file_id, "person", run_id=run_id, role="pickup", lane=f"person_{k}", frames_sent=len(pts), media_size=size, metadata=meta) for k in "ab")
        )
        boxes = [[(f.frame, o.id, o.box, o.c) for f in frames for o in f.objects] for _, frames in lanes]
        texts = [sam_wire.lane_text(sam_wire.read_events(paths.run_dir(run_id) / record.lane_path)) for record, _ in lanes]
        frame = video.read_frame_at(proxy, pts[min(40, len(pts) - 1)])  # frame 40, the one measured on 10/31
        checks = [await sam_client.segment_image(frame, "face", run_id=run_id, role="pickup", tag=f"det_{k}") for k in "ab"]
        dets = [[(d.box.left, d.box.top, d.box.right, d.box.bottom, d.confidence, int(sam_client.mask_array(d).sum())) for d in run.frames.get(0, [])] for run in checks]
    finally:
        await sam_client.delete_file(upload.file_id)
    result = {
        "video_records": [len(b) for b in boxes],
        "video_records_differing": sum(1 for x, y in zip(*boxes) if x != y) + abs(len(boxes[0]) - len(boxes[1])),
        "lane_text_identical": texts[0] == texts[1],
        "image_detections": dets,
        "image_identical": dets[0] == dets[1],
    }
    (out / "result.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clip", type=Path, default=REPO / "data" / "test" / "pingpong.mp4")
    parser.add_argument("--out", type=Path, default=REPO / "runs" / "probe_det")
    parser.add_argument("--run-id", default="probe_det")
    args = parser.parse_args(argv)
    if os.environ.get("LIVE") != "1":
        print(f"would send two 'person' lanes and two 'face' image checks on {args.clip} (about $0.04); set LIVE=1")
        return 2
    print(json.dumps(asyncio.run(probe(args.clip, args.out, args.run_id)), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
