"""Where Spark's boxes sit against ground truth, in crop pixels (no call). Spark centre minus truth centre, split by
the truth's position in the crop and by the crop's aspect, to find any systematic shift. Writes bias.json."""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import truth as T  # noqa: E402

from walkaround import paths  # noqa: E402
from walkaround.contracts import Keyframe, read_json_list  # noqa: E402


def pairs_synth() -> list[dict]:
    rows = []
    for it in T.synth_items():
        for m in it["marks"]:
            rows.extend(_match(it, m.bbox_crop, f"stored:{it['run']}", m.mark_id))
    # the R10 study's answers on synth_pair2_r9 crops (full crop at effort low, medium, high)
    kfs = {k.kf_id: k for k in read_json_list(paths.keyframes_json("synth_pair2_r9"), Keyframe)}
    for line in T.STUDY.read_text().splitlines():
        a = json.loads(line)
        if a["arm"] != "base" or a["crop_id"] not in kfs:
            continue
        k = kfs[a["crop_id"]]
        it = {"kf": k, "truth": T.truth_in_crop("pair2", k), "w": k.crop.w, "h": k.crop.h}
        for i, b in enumerate(a["boxes"]):
            rows.extend(_match(it, b["box"], f"study:{a['effort']}", f"{a['crop_id']}#{a['rep']}#{i}"))
    return rows


def _match(it: dict, box: list[float], src: str, ident: str) -> list[dict]:
    cand = [g for g in it["truth"] if T.near(box, g["box"])]
    if not cand:
        return []
    cx, cy = T.centre(box)
    g = min(cand, key=lambda g: (cx - T.centre(g["box"])[0]) ** 2 + (cy - T.centre(g["box"])[1]) ** 2)
    gx, gy = T.centre(g["box"])
    return [{"src": src, "id": ident, "truth": g["id"], "w": it["w"], "h": it["h"], "dx": cx - gx, "dy": cy - gy,
             "gx": gx / it["w"], "gy": gy / it["h"], "iou": T.iou(box, g["box"]), "strict": T.hit(box, g["box"]),
             "gw": g["box"][2] - g["box"][0], "gh": g["box"][3] - g["box"][1]}]


def pairs_pairc() -> list[dict]:
    rows = []
    specs = {s["name"]: s for s in T.pairc_specs()}
    from PIL import Image
    for line in T.STUDY.read_text().splitlines():
        a = json.loads(line)
        name = a["crop_id"].removeprefix("pairC_")
        if a["arm"] != "base" or name not in specs:
            continue
        s = specs[name]
        w, h = Image.open(s["path"]).size
        for i, b in enumerate(a["boxes"]):
            hits = [t for t in s["targets"] if T.target_hit(t, b["box"], max(w, h))]
            if not hits:
                continue
            cx, cy = T.centre(b["box"])
            t = min(hits, key=lambda t: (cx - t[0]) ** 2 + (cy - t[1]) ** 2)
            rows.append({"src": f"pairC:{a['effort']}", "id": f"{name}#{a['rep']}#{i}", "w": w, "h": h, "dx": cx - t[0], "dy": cy - t[1],
                         "gx": t[0] / w, "gy": t[1] / h})
    return rows


def summary(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    def med(k, rs):
        return round(st.median(r[k] for r in rs), 1) if rs else None
    out = {"n": len(rows), "dx_med": med("dx", rows), "dy_med": med("dy", rows),
           "dx_mean": round(st.mean(r["dx"] for r in rows), 1), "dy_mean": round(st.mean(r["dy"] for r in rows), 1),
           "dx_grid_med": round(st.median(1000 * r["dx"] / r["w"] for r in rows), 1),
           "dy_grid_med": round(st.median(1000 * r["dy"] / r["h"] for r in rows), 1),
           "abs_off_med": round(st.median((r["dx"] ** 2 + r["dy"] ** 2) ** 0.5 for r in rows), 1)}
    if "iou" in rows[0]:
        out["iou_med"] = round(st.median(r["iou"] for r in rows), 3)
    for axis, k in (("x", "gx"), ("y", "gy")):
        for lo, hi, name in ((0, 1 / 3, "first third"), (1 / 3, 2 / 3, "middle third"), (2 / 3, 1.01, "last third")):
            rs = [r for r in rows if lo <= r[k] < hi]
            out[f"by_{axis}: {name}"] = {"n": len(rs), "dx": med("dx", rs), "dy": med("dy", rs)}
    for lo, hi in ((0, 1.5), (1.5, 2.5), (2.5, 99)):
        rs = [r for r in rows if lo <= r["w"] / r["h"] < hi]
        out[f"aspect {lo}-{hi}"] = {"n": len(rs), "dx": med("dx", rs), "dy": med("dy", rs)}
    return out


def main() -> int:
    synth = pairs_synth()
    real = pairs_pairc()
    res = {
        "synthetic, every box near a planted mark": summary(synth),
        "synthetic, stored product boxes only": summary([r for r in synth if r["src"].startswith("stored")]),
        "synthetic, study boxes only": summary([r for r in synth if r["src"].startswith("study")]),
        "synthetic, strict hits only": summary([r for r in synth if r["strict"]]),
        "pair C, D1 crops (study answers, every effort)": summary(real),
    }
    (Path(__file__).with_name("bias.json")).write_text(json.dumps({"summary": res, "rows": synth + real}, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
