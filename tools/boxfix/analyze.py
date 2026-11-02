"""Report on hand corrections from the box fix page (eval/box_corrections/*.json).

    .venv/bin/python tools/boxfix/analyze.py [--dir eval/box_corrections]

Prints counts and offsets per run and overall, writes summary.json, and draws the 12 worst moved boxes into
eval/box_corrections/worst/.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNS = REPO / "runs"
OUT = REPO / "eval" / "box_corrections"
WORST = OUT / "worst"
PATCH_SENT_PX = 28  # one Muse Spark vision patch on the sent image (BUILD_STATUS R11)
SENT_LONG_EDGE = 1280  # spark_client.IMAGE_LONG_EDGE
SIZE_BINS = [(0, 16, "under 16 px"), (16, 32, "16-32 px"), (32, 64, "32-64 px"), (64, 128, "64-128 px"),
             (128, 1e9, "128 px and up")]
PROD, FIX = (255, 210, 63), (61, 220, 132)


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def centre(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def sign_p(k: int, n: int) -> float:
    # two-sided binomial sign test, p = 0.5
    if n == 0:
        return 1.0
    lo = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(lo + 1)) / 2 ** n)


def q(vals, p):
    return float(np.percentile(vals, p)) if len(vals) else None


def spread(vals) -> dict | None:
    if not vals:
        return None
    v = np.asarray(vals, float)
    return {"median": float(np.median(v)), "p25": q(v, 25), "p75": q(v, 75), "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0}


def rows_of(doc: dict) -> list[dict]:
    out = []
    for mid, r in doc.get("marks", {}).items():
        r = dict(r, mark_id=mid, run_id=doc["run_id"])
        if r["status"] in ("correct", "moved") and r.get("product_box") and r.get("box"):
            p, b = r["product_box"], r["box"]
            pc, bc = centre(p), centre(b)
            w, h = max(p[2] - p[0], 1e-6), max(p[3] - p[1], 1e-6)
            dx, dy = bc[0] - pc[0], bc[1] - pc[1]
            s = min(1.0, SENT_LONG_EDGE / max(r.get("crop_w") or 1, r.get("crop_h") or 1))
            r.update(dx=dx, dy=dy, dist=math.hypot(dx, dy), fx=dx / w, fy=dy / h, frac=math.hypot(dx / w, dy / h),
                     patches=math.hypot(dx, dy) * s / PATCH_SENT_PX, iou=iou(p, b), long_side=max(w, h),
                     w_ratio=(b[2] - b[0]) / w, h_ratio=(b[3] - b[1]) / h,
                     spark_iou=iou(r["spark_box"], b) if r.get("spark_box") else None)
        out.append(r)
    return out


def direction(moved: list[dict]) -> dict | None:
    if not moved:
        return None
    dx = [r["dx"] for r in moved]
    dy = [r["dy"] for r in moved]
    n = len(moved)
    right, down = sum(v > 0 for v in dx), sum(v > 0 for v in dy)
    nzx, nzy = sum(v != 0 for v in dx), sum(v != 0 for v in dy)
    units = [(r["dx"] / r["dist"], r["dy"] / r["dist"]) for r in moved if r["dist"] > 0]
    R = math.hypot(sum(u[0] for u in units), sum(u[1] for u in units)) / len(units) if units else 0.0
    ang = math.degrees(math.atan2(sum(u[1] for u in units), sum(u[0] for u in units))) if units else None
    return {"n": n, "right": right, "left": sum(v < 0 for v in dx), "down": down, "up": sum(v < 0 for v in dy),
            "p_x": sign_p(right, nzx), "p_y": sign_p(down, nzy), "mean_dx": float(np.mean(dx)), "mean_dy": float(np.mean(dy)),
            "resultant": R, "mean_angle_deg": ang}


def compass(deg: float | None) -> str:
    if deg is None:
        return "none"
    names = ["right", "down-right", "down", "down-left", "left", "up-left", "up", "up-right"]  # image y points down
    return names[int(((deg % 360) + 22.5) // 45) % 8]


def group(moved: list[dict], key) -> dict:
    g: dict[str, list[dict]] = {}
    for r in moved:
        g.setdefault(key(r), []).append(r)
    return {k: {"n": len(v), "median_dist_px": float(np.median([r["dist"] for r in v])),
                "median_frac": float(np.median([r["frac"] for r in v])),
                "median_dx": float(np.median([r["dx"] for r in v])), "median_dy": float(np.median([r["dy"] for r in v])),
                "median_iou": float(np.median([r["iou"] for r in v]))} for k, v in sorted(g.items())}


def size_bin(r) -> str:
    return next(name for lo, hi, name in SIZE_BINS if lo <= r["long_side"] < hi)


def summarize(rows: list[dict], missed: list[dict], n_marks: int | None, n_crops_seen: int) -> dict:
    st = {s: sum(r["status"] == s for r in rows) for s in ("correct", "moved", "not_mark")}
    moved = [r for r in rows if r["status"] == "moved" and "dx" in r]
    kept = [r for r in rows if "iou" in r]
    sp = [r for r in moved if r.get("spark_iou") is not None]
    return {
        "checked": len(rows), "of_marks": n_marks, **st, "missed": len(missed),
        "crops_with_missed": len({(m.get("run_id"), m["kf_id"]) for m in missed}), "crops_seen": n_crops_seen,
        "moved_offset_px": {"dx": spread([r["dx"] for r in moved]), "dy": spread([r["dy"] for r in moved]),
                            "dist": spread([r["dist"] for r in moved])},
        "moved_offset_frac": {"fx": spread([r["fx"] for r in moved]), "fy": spread([r["fy"] for r in moved]),
                              "frac": spread([r["frac"] for r in moved])},
        "moved_offset_spark_patches": spread([r["patches"] for r in moved]),
        "moved_size_ratio": {"w": spread([r["w_ratio"] for r in moved]), "h": spread([r["h_ratio"] for r in moved])},
        "direction": direction([r for r in moved if r["dist"] > 0]),
        "iou_moved": spread([r["iou"] for r in moved]),
        "iou_all_kept": spread([r["iou"] for r in kept]),
        "iou_spark_box_moved": spread([r["spark_iou"] for r in sp]),
        "by_type": group(moved, lambda r: r.get("type") or "unknown"),
        "by_size": group(moved, size_bin),
    }


def fmt(s: dict | None, unit: str = "", nd: int = 1) -> str:
    if not s:
        return "n/a"
    return f"median {s['median']:+.{nd}f}{unit} (middle half {s['p25']:+.{nd}f} to {s['p75']:+.{nd}f}, sd {s['sd']:.{nd}f})"


def fmt_abs(s: dict | None, unit: str = "", nd: int = 1) -> str:
    if not s:
        return "n/a"
    return f"median {s['median']:.{nd}f}{unit} (middle half {s['p25']:.{nd}f} to {s['p75']:.{nd}f})"


def report(name: str, s: dict) -> None:
    print(f"\n== {name}")
    of = f" of {s['of_marks']}" if s["of_marks"] is not None else ""
    print(f"checked {s['checked']}{of} boxes: correct {s['correct']}, moved {s['moved']}, not a mark {s['not_mark']}; "
          f"missed {s['missed']} on {s['crops_with_missed']} crops ({s['crops_seen']} crops looked at)")
    if not s["moved"]:
        return
    o, f = s["moved_offset_px"], s["moved_offset_frac"]
    print(f"moved, centre offset in crop px: dx {fmt(o['dx'])}, dy {fmt(o['dy'])}")
    print(f"  distance {fmt_abs(o['dist'], ' px')}, in Spark patches {fmt_abs(s['moved_offset_spark_patches'], '', 2)}")
    print(f"  as a fraction of the box: x {fmt(f['fx'], '', 2)}, y {fmt(f['fy'], '', 2)}, length {fmt_abs(f['frac'], '', 2)}")
    r = s["moved_size_ratio"]
    print(f"  corrected size over product size: width {fmt_abs(r['w'], 'x', 2)}, height {fmt_abs(r['h'], 'x', 2)}")
    d = s["direction"]
    if d:
        print(f"  direction: right {d['right']}, left {d['left']} (sign test p={d['p_x']:.2f}); down {d['down']}, up {d['up']} "
              f"(p={d['p_y']:.2f}); mean unit vector length {d['resultant']:.2f} toward {compass(d['mean_angle_deg'])}"
              f" (0 = no common direction, 1 = all the same way)")
    print(f"IoU product vs corrected: moved {fmt_abs(s['iou_moved'], '', 2)}; all kept boxes {fmt_abs(s['iou_all_kept'], '', 2)}")
    if s["iou_spark_box_moved"]:
        print(f"IoU Spark's own box vs corrected, moved: {fmt_abs(s['iou_spark_box_moved'], '', 2)}")
    for title, g in (("by type", s["by_type"]), ("by size (product box long side, crop px)", s["by_size"])):
        print(f"  {title}:")
        for k, v in g.items():
            print(f"    {k:<16} n={v['n']:<3} median offset {v['median_dist_px']:5.1f} px ({v['median_frac']:.2f} of box), "
                  f"dx {v['median_dx']:+.1f}, dy {v['median_dy']:+.1f}, IoU {v['median_iou']:.2f}")


def crop_path(run_id: str, kf_id: str, cache: dict) -> Path | None:
    if run_id not in cache:
        f = RUNS / run_id / "derived" / "keyframes.json"
        cache[run_id] = {k["kf_id"]: k["crop"]["path"] for k in json.loads(f.read_text()) if k.get("crop")} if f.exists() else {}
    p = cache[run_id].get(kf_id)
    return RUNS / run_id / p if p else None


def tile(r: dict, cache: dict, size: int = 520) -> Image.Image | None:
    p = crop_path(r["run_id"], r["kf_id"], cache)
    if not p or not p.exists():
        return None
    im = Image.open(p).convert("RGB")
    a, b = r["product_box"], r["box"]
    u = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
    pad = max(40.0, 1.0 * max(u[2] - u[0], u[3] - u[1]))
    side = max(u[2] - u[0], u[3] - u[1]) + 2 * pad
    cx, cy = centre(u)
    x0, y0 = max(0, cx - side / 2), max(0, cy - side / 2)
    x1, y1 = min(im.width, x0 + side), min(im.height, y0 + side)
    x0, y0 = max(0, x1 - side), max(0, y1 - side)
    region = im.crop((int(x0), int(y0), int(math.ceil(x1)), int(math.ceil(y1))))
    k = size / max(region.width, region.height)
    region = region.resize((max(1, round(region.width * k)), max(1, round(region.height * k))), Image.Resampling.LANCZOS)
    rd = ImageDraw.Draw(region)  # drawn on the region so a box past the crop edge is clipped

    def box(bb, color):
        xy = [(bb[0] - int(x0)) * k, (bb[1] - int(y0)) * k, (bb[2] - int(x0)) * k, (bb[3] - int(y0)) * k]
        rd.rectangle([xy[0] - 1, xy[1] - 1, xy[2] + 1, xy[3] + 1], outline=(0, 0, 0), width=4)
        rd.rectangle(xy, outline=color, width=2)

    box(a, PROD)
    box(b, FIX)
    canvas = Image.new("RGB", (size, size + 58), (21, 23, 26))
    canvas.paste(region, (0, 0))
    dr = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    dr.text((8, size + 6), f"{r['run_id']}  {r['mark_id']}", fill=(236, 234, 228), font=font)
    dr.text((8, size + 30), f"{r.get('type')}  IoU {r['iou']:.2f}  offset {r['dx']:+.0f},{r['dy']:+.0f} px  "
            f"yellow product, green fixed", fill=(163, 166, 171), font=font)
    return canvas


def worst(moved: list[dict]) -> list[Path]:
    WORST.mkdir(parents=True, exist_ok=True)
    for old in WORST.glob("*.png"):
        old.unlink()
    picks = sorted(moved, key=lambda r: (r["iou"], -r["dist"]))[:12]
    cache: dict = {}
    tiles, paths = [], []
    for i, r in enumerate(picks, 1):
        t = tile(r, cache)
        if t is None:
            continue
        p = WORST / f"{i:02d}_{r['run_id']}_{r['mark_id']}.png"
        t.save(p)
        tiles.append(t)
        paths.append(p)
    if tiles:
        cols = 4
        rows = math.ceil(len(tiles) / cols)
        tw, th = tiles[0].size
        sheet = Image.new("RGB", (cols * tw + (cols - 1) * 6, rows * th + (rows - 1) * 6), (60, 63, 68))
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % cols) * (tw + 6), (i // cols) * (th + 6)))
        p = WORST / "worst12.png"
        sheet.save(p)
        paths.append(p)
    return paths


def main() -> None:
    global OUT, WORST
    ap = argparse.ArgumentParser(description="Report on box fix corrections.")
    ap.add_argument("--dir", type=Path, default=OUT, help="folder of <run_id>.json corrections")
    a = ap.parse_args()
    OUT, WORST = a.dir, a.dir / "worst"
    files = sorted(p for p in OUT.glob("*.json") if p.name != "summary.json")
    if not files:
        sys.exit(f"no corrections in {OUT}")
    all_rows, all_missed, out, seen_total, marks_total = [], [], {}, 0, 0
    for f in files:
        doc = json.loads(f.read_text())
        run = doc.get("run_id") or f.stem
        doc["run_id"] = run
        rows = rows_of(doc)
        missed = [dict(m, run_id=run) for m in doc.get("missed", [])]
        mj = RUNS / run / "derived" / "marks.json"
        n_marks = len(json.loads(mj.read_text())) if mj.exists() else None
        seen = len(doc.get("crops_seen", []))
        s = summarize(rows, missed, n_marks, seen)
        out[run] = s
        report(run, s)
        all_rows += rows
        all_missed += missed
        seen_total += seen
        marks_total += n_marks or 0
    s = summarize(all_rows, all_missed, marks_total, seen_total)
    out["overall"] = s
    if len(files) > 1:
        report("overall", s)
    (OUT / "summary.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    paths = worst([r for r in all_rows if r["status"] == "moved" and "dx" in r])
    print(f"\nsummary: {OUT / 'summary.json'}")
    print(f"worst cases: {len([p for p in paths if p.name != 'worst12.png'])} images in {WORST}" if paths else "worst cases: no moved boxes yet")


if __name__ == "__main__":
    main()
