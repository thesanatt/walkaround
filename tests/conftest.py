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
