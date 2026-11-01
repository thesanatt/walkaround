"""The test clips under data/test/ are not in the repo (the ping pong clip shows two people's faces, and data/ is
ignored), so a fresh clone gets synthetic stand-ins with the same size, rate and length before any test runs."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "test"
STAND_INS = {
    # 1280x720 at 24000/1001 fps, 141 frames, no audio: the geometry the saved 10/26 SAM streams were made on
    "pingpong.mp4": ["-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=24000/1001,format=yuv420p", "-frames:v", "141", "-video_track_timescale", "24000"],
    # 720x1280 portrait at 24 fps, 10 s, with an AAC track: run_d1's dry-run return walk
    "shiba_jump.mp4": ["-f", "lavfi", "-i", "testsrc2=size=720x1280:rate=24,format=yuv420p", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", "10", "-c:a", "aac"],
}


def pytest_configure(config) -> None:
    if shutil.which("ffmpeg") is None:
        return
    for name, args in STAND_INS.items():
        target = DATA / name
        if target.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["ffmpeg", "-loglevel", "error", "-y", *args, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-map_metadata", "-1", str(target)]
        subprocess.run(cmd, check=True)


import os  # noqa: E402

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _no_live_sam_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without LIVE=1 no test reaches SAM 3.1 on the Meta Model API: sam_client.client raises unless the test fakes it
    (or segment_image) itself. Box fix 11/1: a marks test paid for 5 SAM image requests when the SAM box leaf came on.
    Spark tests already inject fakes through spark_client.set_client."""
    if os.environ.get("LIVE") == "1":
        return
    from walkaround import sam_client

    def no_sam() -> None:
        raise RuntimeError("no live SAM call in tests; fake sam_client.client or segment_image (LIVE=1 to allow)")

    monkeypatch.setattr(sam_client, "client", no_sam)
