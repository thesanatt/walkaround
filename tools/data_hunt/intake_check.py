# Intake check for one phone clip, before any run. Reads the clip only through ffprobe and ffmpeg, never uploads it.
#   python3 -I intake_check.py CLIP [FRAMES_DIR] [--min-sec 45 --max-sec 60 --frames 8]
# Prints one JSON object. Exit 1 when a hard check fails (codec, 1080p or more, landscape, HDR off); fps, duration and
# audio are warnings, since the pipeline makes a 15 fps proxy and runs without narration.
import argparse
import hashlib
import json
import os
import subprocess
import sys

HARD = ("codec_h264_or_hevc", "at_least_1080p", "landscape", "hdr_off")

ap = argparse.ArgumentParser()
ap.add_argument("clip")
ap.add_argument("frames_dir", nargs="?")
ap.add_argument("--min-sec", type=float, default=45)
ap.add_argument("--max-sec", type=float, default=60)
ap.add_argument("--frames", type=int, default=8)
a = ap.parse_args()

h = hashlib.sha256()
with open(a.clip, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)

p = json.loads(subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", a.clip],
                              capture_output=True, text=True, check=True).stdout)
vids = [s for s in p["streams"] if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")]
auds = [s for s in p["streams"] if s.get("codec_type") == "audio"]
if not vids:
    print(json.dumps({"file": os.path.basename(a.clip), "error": "no video stream"}))
    sys.exit(1)
v = vids[0]
rot = 0
for sd in v.get("side_data_list") or []:
    if "rotation" in sd:
        rot = int(round(float(sd["rotation"])))
if not rot and (v.get("tags") or {}).get("rotate"):
    rot = int(v["tags"]["rotate"])
w, ht = v["width"], v["height"]
dw, dh = (ht, w) if abs(rot) % 180 == 90 else (w, ht)
num, den = (float(x) for x in v.get("avg_frame_rate", "0/1").split("/"))
fps = num / den if den else 0.0
dur = float(p["format"].get("duration") or v.get("duration") or 0)
tags = {**(p["format"].get("tags") or {}), **(v.get("tags") or {})}
gps = sorted(k for k in tags if "location" in k.lower() or "iso6709" in k.lower())
hdr = v.get("color_transfer") in ("arib-std-b67", "smpte2084")

checks = {
    "codec_h264_or_hevc": v.get("codec_name") in ("h264", "hevc"),
    "landscape": dw > dh,
    "at_least_1080p": min(dw, dh) >= 1080,
    "fps_about_30": 29 <= fps <= 31,
    "duration_in_range": a.min_sec <= dur <= a.max_sec,
    "audio_present": bool(auds),
    "hdr_off": not hdr,
}
warnings = [f"{k} failed" for k, ok in checks.items() if not ok and k not in HARD]
if gps:
    warnings.append("GPS tag present (%s): never share the original; the run's proxy drops all metadata" % ", ".join(gps))
if hdr:
    warnings.append("HDR transfer %s: set HDR Video off and film again" % v.get("color_transfer"))

frames = []
if a.frames_dir and a.frames > 0:
    os.makedirs(a.frames_dir, exist_ok=True)
    for i in range(a.frames):
        t = dur * (i + 0.5) / a.frames
        out = os.path.join(a.frames_dir, "frame_%02d_%06.2fs.jpg" % (i, t))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "%.3f" % t, "-i", a.clip, "-frames:v", "1", "-q:v", "2", out], check=True)
        frames.append(out)

hard_pass = all(checks[k] for k in HARD)
print(json.dumps({
    "file": os.path.basename(a.clip), "bytes": os.path.getsize(a.clip), "sha256": h.hexdigest(),
    "codec": v.get("codec_name"), "coded_wh": [w, ht], "rotation": rot, "display_wh": [dw, dh],
    "fps": round(fps, 3), "duration_s": round(dur, 2), "audio": [s.get("codec_name") for s in auds],
    "pix_fmt": v.get("pix_fmt"), "color_transfer": v.get("color_transfer"),
    "creation_time": tags.get("com.apple.quicktime.creationdate") or tags.get("creation_time"),
    "checks": checks, "hard_pass": hard_pass, "warnings": warnings, "frames": frames,
}, indent=2))
sys.exit(0 if hard_pass else 1)
