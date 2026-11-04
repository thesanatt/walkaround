"""Registration study (R20, 11/2): alignment per compared panel and the presence test per mark, on stored runs.

No model call. Reads each run's frozen keyframes, crops, masks, marks and matches. For every panel the stored match
compared, aligns the chosen keyframe pair, and the keyframe of every mark with the other walk's chosen keyframe.

Truth for alignment: on synth runs the exact boxes of the generator's marks in both frames (synthetic/out/pair2/
ground_truth.json); a mark visible in both crops is a landmark the fit never saw as a feature target (it is not cut
out unless the product boxed it), so its centre mapped through the fit against its true centre is the error.

Usage: study.py align RUN [RUN ...]   prints one line per alignment, a JSON summary per run on stdout's last line
"""

from __future__ import annotations

import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
from walkaround import contracts as c  # noqa: E402
from walkaround import register  # noqa: E402

TRUTH = REPO.parent / "synthetic" / "out" / "pair2" / "ground_truth.json"
SYNTH_PANEL = {
    "front_left_door": "left_front_door",
    "front_right_door": "right_front_door",
    "front_left_fender": "left_front_fender",
    "front_right_fender": "right_front_fender",
    "left_rear_quarter": "left_rear_fender",
    "right_rear_quarter": "right_rear_fender",
    "trunk_lid": "trunk_lid",
    "hood": "hood",
    "rear_bumper": "rear_bumper",
}


def cfg_for(th: dict | None = None) -> dict:
    base = dict(register.DEFAULTS)
    base["enabled"] = True
    leaf = (c.load_thresholds().get("matching", {}).get("registration") or {}) if th is None else (th.get("matching", {}).get("registration") or {})
    base.update(leaf)
    return base


def root(run: str) -> Path:
    return REPO / "runs" / run


@lru_cache(maxsize=None)
def keyframes(run: str) -> dict[str, c.Keyframe]:
    return {k.kf_id: k for k in c.read_json_list(root(run) / "derived" / "keyframes.json", c.Keyframe)}


@lru_cache(maxsize=None)
def marks(run: str) -> dict[str, c.Mark]:
    return {m.mark_id: m for m in c.read_json_list(root(run) / "derived" / "marks.json", c.Mark)}


def matches(run: str) -> list[dict]:
    return json.loads((root(run) / "derived" / "matches.json").read_text())


_img: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}


def pixels(run: str, kf_id: str) -> tuple[np.ndarray, np.ndarray] | None:
    key = (run, kf_id)
    if key not in _img:
        kf = keyframes(run)[kf_id]
        if kf.crop is None or not kf.mask_png:
            return None
        img = cv2.imread(str(root(run) / kf.crop.path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(root(run) / kf.mask_png), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            return None
        _img[key] = (img, mask > 0)
    return _img[key]


def boxes_in(run: str, kf_id: str) -> list[tuple[float, float, float, float]]:
    return [tuple(m.bbox_crop) for m in marks(run).values() if m.kf_id == kf_id and m.bbox_crop is not None]


def align(run: str, kf_a: str, kf_b: str, cfg: dict) -> register.Aligned | None:
    pa, pb = pixels(run, kf_a), pixels(run, kf_b)
    if pa is None or pb is None:
        return None
    ka, kb = keyframes(run)[kf_a], keyframes(run)[kf_b]
    prior = register.prior_map(ka.moments, ka.front_sign, kb.moments, kb.front_sign)
    return register.align(kf_a, pa[0], pa[1], kf_b, pb[0], pb[1], cfg, prior=prior, boxes_a=boxes_in(run, kf_a), boxes_b=boxes_in(run, kf_b))


@lru_cache(maxsize=1)
def truth() -> dict:
    return json.loads(TRUTH.read_text())


def truth_points(run: str, kf_id: str) -> dict[str, tuple[float, float, float, float]]:
    """Generator mark id -> its visible box in this crop's pixels, for marks visible in the keyframe's frame."""
    kf = keyframes(run)[kf_id]
    if kf.crop is None:
        return {}
    frame = int(round(kf.pts * 30.0))
    out = {}
    for m in truth()["marks"]:
        rows = m["frames"].get(str(kf.role), [])
        row = next((r for r in rows if r["frame"] == frame), None)
        if row is None or not row["visible"] or row["visible_bbox_xyxy"] is None:
            continue
        x1, y1, x2, y2 = row["visible_bbox_xyxy"]
        s, dx, dy = kf.crop.scale, kf.crop.dx, kf.crop.dy
        b = ((x1 - dx) * s, (y1 - dy) * s, (x2 - dx) * s, (y2 - dy) * s)
        if b[2] <= 0 or b[3] <= 0 or b[0] >= kf.crop.w or b[1] >= kf.crop.h:
            continue
        out[m["id"]] = b
    return out


def landmark_errors(run: str, al: register.Aligned) -> list[tuple[str, float]]:
    if al.H is None:
        return []
    ta, tb = truth_points(run, al.kf_a), truth_points(run, al.kf_b)
    mask_a = pixels(run, al.kf_a)[1]
    out = []
    for mid in sorted(set(ta) & set(tb)):
        a, b = ta[mid], tb[mid]
        x, y = int((a[0] + a[2]) / 2), int((a[1] + a[3]) / 2)
        if not (0 <= y < mask_a.shape[0] and 0 <= x < mask_a.shape[1]) or not mask_a[y, x]:
            continue  # off the panel: not a point the panel's fit is meant to carry
        ca = np.array([[(a[0] + a[2]) / 2, (a[1] + a[3]) / 2]])
        cb = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
        out.append((mid, float(np.linalg.norm(register.apply(al.H, ca)[0] - cb)), register.uncertainty_px(al, (float(ca[0][0]), float(ca[0][1])))))
    return out


def pairs_to_align(run: str) -> list[tuple[str, str, str, str]]:
    """(panel, kind, kf_a, kf_b): the chosen pair of every compared panel, then each other keyframe holding a mark
    with the other walk's chosen keyframe (a pickup keyframe first in every pair)."""
    out = []
    for pm in matches(run):
        if pm["state"] != "comparable" or not pm.get("kf_pair") or pm["panel_id"].endswith("_wheel"):
            continue
        kp, kr = pm["kf_pair"]
        out.append((pm["panel_id"], "chosen", kp, kr))
        ids = [p["pickup_mark"] for p in pm["pairs"]] + [p["return_mark"] for p in pm["pairs"]] + pm["unmatched_pickup"] + pm["unmatched_return"]
        seen = set()
        for mid in ids:
            m = marks(run).get(mid)
            if m is None:
                continue
            a, b = (m.kf_id, kr) if str(m.role) == "pickup" else (kp, m.kf_id)
            if (a, b) != (kp, kr) and (a, b) not in seen:
                seen.add((a, b))
                out.append((pm["panel_id"], "mark", a, b))
    return out


def cmd_align(runs: list[str]) -> None:
    import os

    cfg = cfg_for()
    cfg.update(json.loads(os.environ.get("CFG", "{}")))
    for run in runs:
        rows = []
        for panel, kind, a, b in pairs_to_align(run):
            al = align(run, a, b, cfg)
            if al is None:
                continue
            errs = landmark_errors(run, al) if run.startswith("synth") else []
            row = {"run": run, "panel": panel, "kind": kind, **al.summary(), "landmarks": [(m, round(e, 1), round(u, 1)) for m, e, u in errs]}
            rows.append(row)
            print(f"{run:22s} {panel:20s} {kind:6s} {'OK ' if al.ok else 'NO '} {al.model[:4]} in={al.n_inliers:4d}/{al.n_matches:<4d} sp={al.spread:.2f} iou={al.mask_iou:.2f} rp={al.reproj_px:.1f} lm={[(round(e), round(u)) for _, e, u in errs]} {al.why or ''}")
        ok = [r for r in rows if r["ok"] and r["kind"] == "chosen"]
        print(json.dumps({"run": run, "chosen": sum(r["kind"] == "chosen" for r in rows), "chosen_ok": len(ok)}))
        (HERE / f"align_{run}.json").write_text(json.dumps(rows, indent=1) + "\n")




def overlay(run: str, kf_a: str, kf_b: str, out_path: Path, cfg: dict, mark_rows: list[dict] | None = None) -> register.Aligned | None:
    """Warped pickup crop blended over the return crop: return outline green, warped pickup outline magenta, pickup
    marks mapped (yellow, with the search window when tested), return marks (cyan); the fit's numbers on top."""
    al = align(run, kf_a, kf_b, cfg)
    if al is None or al.H is None:
        return al
    (ia, ma), (ib, mb) = pixels(run, kf_a), pixels(run, kf_b)
    h, w = ib.shape[:2]
    warped = cv2.warpPerspective(ia, al.H, (w, h))
    blend = cv2.addWeighted(ib, 0.5, warped, 0.5, 0)
    wm = cv2.warpPerspective(ma.astype(np.uint8), al.H, (w, h), flags=cv2.INTER_NEAREST)
    for m, col in ((mb.astype(np.uint8), (0, 255, 0)), (wm, (255, 0, 255))):
        cnt, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blend, cnt, -1, col, 2)
    for mk in marks(run).values():
        if mk.bbox_crop is None:
            continue
        if mk.kf_id == kf_a:
            x1, y1, x2, y2 = (int(v) for v in register.map_box(al.H, tuple(mk.bbox_crop)))
            cv2.rectangle(blend, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(blend, mk.mark_id.split("_")[-1], (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        elif mk.kf_id == kf_b:
            x1, y1, x2, y2 = (int(v) for v in mk.bbox_crop)
            cv2.rectangle(blend, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(blend, mk.mark_id.split("_")[-1], (x1, min(h - 4, y2 + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    for row in mark_rows or []:
        if row.get("found_box"):
            x1, y1, x2, y2 = (int(v) for v in row["found_box"])
            cv2.rectangle(blend, (x1, y1), (x2, y2), (0, 128, 255), 1)
    s = al.summary()
    text = f"{s['model']} {'ok' if s['ok'] else 'refused'} iou {s['mask_iou']} inliers {s['n_inliers']}/{s['n_matches']} rp {s['reproj_px']}"
    cv2.rectangle(blend, (0, 0), (w, 22), (0, 0, 0), -1)
    cv2.putText(blend, f"{kf_a} -> {kf_b}: {text}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), blend, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return al


# Presence: the product registration step on the frozen match, joined with truth --------------------------------


def run_step(run: str, cfg: dict) -> list[c.PanelMatch]:
    from walkaround import match

    th = {"matching": {"registration": cfg}}
    full = json.loads((root(run) / "run.json").read_text())["thresholds"]
    th = {**full, "matching": {**full["matching"], "registration": cfg}}
    kfs = keyframes(run)
    ms = list(marks(run).values())
    cal = c.read_json(root(run) / "derived" / "calibration.json", c.Calibration)
    cache = match.RegistrationCache(run, kfs, ms, match.registration_config(th))
    from walkaround.orchestrator import _runs_root

    out = []
    with _runs_root(REPO / "runs"):
        for pm in c.read_json_list(root(run) / "derived" / "matches.json", c.PanelMatch):
            panel_marks = {role: [marks(run)[i] for i in ids if i in marks(run)] for role, ids in (("pickup", [p.pickup_mark for p in pm.pairs] + pm.unmatched_pickup), ("return", [p.return_mark for p in pm.pairs] + pm.unmatched_return))}
            out.append(match.registration_step(pm, panel_marks, cache, cal, th))
    return out


def synth_truth_for(run: str, mark: c.Mark, other_kf: str) -> tuple[str | None, str | None]:
    """(generator mark under the box or None, what the other crop should show: present, absent or None)."""
    tb = truth_points(run, mark.kf_id)
    x1, y1, x2, y2 = mark.bbox_crop
    hit = None
    for mid, b in tb.items():
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        if x1 - 15 <= cx <= x2 + 15 and y1 - 15 <= cy <= y2 + 15:
            hit = mid
            break
    if hit is None:
        return None, None
    kf_o = keyframes(run)[other_kf]
    frame = int(round(kf_o.pts * 30.0))
    gm = next(m for m in truth()["marks"] if m["id"] == hit)
    row = next((r for r in gm["frames"].get(str(kf_o.role), []) if r["frame"] == frame), None)
    if row is None:
        return hit, None
    if not row["present"]:
        return hit, "absent"
    return hit, ("present" if hit in truth_points(run, other_kf) else None)


def labels(run: str) -> dict[str, str]:
    """Sanat's box labels: mark id -> real or not_mark (eval/box_corrections)."""
    f = REPO / "eval" / "box_corrections" / f"{run}.json"
    if not f.is_file():
        return {}
    d = json.loads(f.read_text())
    return {mid: ("not_mark" if v["status"] == "not_mark" else "real") for mid, v in d["marks"].items()}


def checks(run: str) -> dict[str, str]:
    """Sanat's verdict checks: mark id -> the label he says is right, where he gave one (eval/verdict_checks)."""
    f = REPO / "eval" / "verdict_checks" / f"{run}.json"
    if not f.is_file():
        return {}
    out = {}
    for v in json.loads(f.read_text())["checks"].values():
        label = v.get("correct_label") or (v["product_label"] if v["status"] == "right" else None)
        for mid in v["mark_ids"]:
            if label:
                out[mid] = label
    return out


FRIEND_NEW_PANELS = {"trunk_lid", "right_rear_door"}  # M5, M7; M6 (left rear door, bottom) is told apart by hand below


def expected(run: str, mark: c.Mark, other_kf: str) -> tuple[str, str | None]:
    """(truth source, expected presence) for one tested mark."""
    if run.startswith("synth"):
        hit, exp = synth_truth_for(run, mark, other_kf)
        return ("synth:" + (hit or "no_mark")), (exp if hit else "no_mark")
    ch = checks(run).get(mark.mark_id)
    if ch:
        return "check:" + ch, {"already_there": "present", "new_or_unseen": "absent", "not_seen_at_return": "absent"}.get(ch)
    lab = labels(run).get(mark.mark_id)
    if lab == "not_mark":
        return "label:not_mark", "no_mark"
    if lab == "real" and run.startswith("friend") and mark.panel_id.value not in FRIEND_NEW_PANELS:
        return "label:real (old on friend)", "present"
    if lab == "real":
        return "label:real", None
    return "none", None


def cmd_presence(runs: list[str]) -> None:
    import os

    cfg = cfg_for()
    cfg.update(json.loads(os.environ.get("CFG", "{}")))
    allrows = []
    for run in runs:
        for pm in run_step(run, cfg):
            for r in pm.registration_marks:
                m = marks(run)[r.mark_id]
                src, exp = expected(run, m, r.other_kf)
                row = {"run": run, "panel": pm.panel_id.value, **r.model_dump(mode="json"), "truth": src, "expected": exp, "type": f"{m.type.value}/{m.confidence.value}"}
                allrows.append(row)
                print(f"{run[:12]:12s} {r.mark_id:34s} {r.result:11s} {str(r.used):9s} exp={str(exp):8s} ncc={r.ncc} s_own={r.contrast_own} s_oth={r.contrast_other} u={r.uncertainty_px} {src} {(r.why or '')[:50]}")
    (HERE / "presence.json").write_text(json.dumps(allrows, indent=1) + "\n")
    tab: dict[str, dict[str, int]] = {}
    for r in allrows:
        key = str(r["expected"])
        tab.setdefault(key, {}).setdefault(r["result"], 0)
        tab[key][r["result"]] += 1
    print(json.dumps(tab))


def truth_boxes(run: str, kf_id: str) -> list[tuple[str, tuple[float, float, float, float], str]]:
    """(id, box, kind) of the true marks in a crop: the generator's visible boxes on synth runs, Sanat's corrected
    boxes (moved or correct, plus the marks he added as missed) elsewhere."""
    if run.startswith("synth"):
        mask = pixels(run, kf_id)[1]
        out = []
        for mid, b in truth_points(run, kf_id).items():
            x, y = int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
                # the generator's box is the bare mark; pad it to at least 12 px like a drawn box
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                hw, hh = max(6.0, (b[2] - b[0]) / 2 + 2), max(6.0, (b[3] - b[1]) / 2 + 2)
                out.append((mid, (cx - hw, cy - hh, cx + hw, cy + hh), "synth"))
        return out
    f = REPO / "eval" / "box_corrections" / f"{run}.json"
    if not f.is_file():
        return []
    d = json.loads(f.read_text())
    out = [(mid, tuple(v["box"]), "label") for mid, v in d["marks"].items() if v["kf_id"] == kf_id and v["status"] != "not_mark" and v.get("box")]
    out += [(v["id"], tuple(v["box"]), "missed") for v in d.get("missed", []) if v["kf_id"] == kf_id and v.get("box")]
    return out


HAND = HERE / "hand_labels.json"  # boxes I drew by eye where no label file covers a compared crop (friend_pair2)


def hand_boxes(run: str, kf_id: str) -> list[tuple[str, tuple[float, float, float, float], str, str]]:
    if not HAND.is_file():
        return []
    rows = json.loads(HAND.read_text()).get(run, [])
    return [(r["id"], tuple(r["box"]), "hand", r["expected"]) for r in rows if r["kf_id"] == kf_id]


def synth_expected(run: str, mid: str, kf_other: str) -> str | None:
    kf_o = keyframes(run)[kf_other]
    gm = next(m for m in truth()["marks"] if m["id"] == mid)
    if str(kf_o.role) not in gm["walks"]:
        return "absent"
    return "present" if mid in truth_points(run, kf_other) else None


def cmd_truthbox(runs: list[str]) -> None:
    """The presence test on true boxes, every pickup x return keyframe pair of each panel the stored match compared."""
    import os

    cfg = cfg_for()
    cfg.update(json.loads(os.environ.get("CFG", "{}")))
    rows = []
    for run in runs:
        compared = {pm["panel_id"] for pm in matches(run) if pm["state"] == "comparable" and not pm["panel_id"].endswith("_wheel")}
        kfs = keyframes(run)
        for panel in sorted(compared):
            pk = sorted(k for k, v in kfs.items() if v.panel_id.value == panel and str(v.role) == "pickup")
            rk = sorted(k for k, v in kfs.items() if v.panel_id.value == panel and str(v.role) == "return")
            for a in pk:
                for b in rk:
                    al = align(run, a, b, cfg)
                    if al is None:
                        continue
                    for own, other, fit in ((a, b, al), (b, a, register.inverse(al))):
                        img_o, mask_o = pixels(run, own)
                        ys, xs = np.nonzero(mask_o)
                        diag = math.hypot(float(xs.max() - xs.min()), float(ys.max() - ys.min()))
                        items = [(i, bx, k, None) for i, bx, k in truth_boxes(run, own)] + hand_boxes(run, own)
                        for mid, box, kind, exp in items:
                            if run.startswith("synth"):
                                exp = synth_expected(run, mid, other)
                            elif exp is None and run.startswith("friend"):
                                exp = "present" if panel not in FRIEND_NEW_PANELS and panel != "left_rear_door" else None
                            pr = register.presence(fit, img_o, pixels(run, other)[0], pixels(run, other)[1], box, cfg, diag_a=diag)
                            rows.append({"run": run, "panel": panel, "own": own, "other": other, "id": mid, "kind": kind, "expected": exp, "result": pr.result, "why": pr.why, "ncc": pr.ncc, "ncc_mark": pr.ncc_mark, "s_own": pr.contrast_a, "s_other": pr.contrast_b, "u": pr.uncertainty_px, "fit_ok": al.ok})
    (HERE / "truthbox.json").write_text(json.dumps(rows, indent=1) + "\n")
    tab: dict[str, dict[str, int]] = {}
    for r in rows:
        if not r["fit_ok"]:
            continue
        key = f"{r['run'][:5]}:{r['expected']}"
        tab.setdefault(key, {}).setdefault(r["result"], 0)
        tab[key][r["result"]] += 1
    for k in sorted(tab):
        print(k, tab[k])
    print("refused fits:", sum(not r["fit_ok"] for r in rows), "of", len(rows))


COMMANDS = {"truthbox": cmd_truthbox}


OVERLAY_DIR = REPO.parent / "docs" / "registration"
OVERLAYS = [
    ("friend_pair2", "pickup_left_rear_door_kf1", "return_left_rear_door_kf2", "friend_pair2_left_rear_door.jpg"),
    ("friend_pair1", "pickup_left_rear_fender_kf1", "return_left_rear_fender_kf2", "friend_pair1_left_rear_fender_M1.jpg"),
    ("synth_pair2_r9", "pickup_left_front_door_kf1", "return_left_front_door_kf1", "synth_pair2_r9_left_front_door.jpg"),
    ("synth_pair2_r8", "pickup_left_front_door_kf2", "return_left_front_door_kf1", "synth_pair2_r8_left_front_door_K18.jpg"),
    ("pairC_r16", "pickup_right_front_door_kf1", "return_right_front_door_kf1", "pairC_r16_right_front_door.jpg"),
    ("friend_pair2", "pickup_left_rear_fender_kf1", "return_left_rear_fender_kf1", "friend_pair2_left_rear_fender_refused.jpg"),
]


def cmd_overlays(_: list[str]) -> None:
    """Warped pickup over return for a few pairs, with the presence test drawn for the true or hand boxes on the
    pickup crop (orange: where the patch search found it in the return crop; the text says present/absent/cannot)."""
    cfg = cfg_for()
    for run, a, b, name in OVERLAYS:
        path = OVERLAY_DIR / name
        al = overlay(run, a, b, path, cfg)
        if al is None or al.H is None:
            print(name, "no fit")
            continue
        img = cv2.imread(str(path))
        (ia, ma), (ib, mb) = pixels(run, a), pixels(run, b)
        ys, xs = np.nonzero(ma)
        diag = math.hypot(float(xs.max() - xs.min()), float(ys.max() - ys.min()))
        items = [(i, bx) for i, bx, _ in truth_boxes(run, a)] + [(i, bx) for i, bx, _, _ in hand_boxes(run, a)]
        inv = register.inverse(al)
        items_b = [(i, bx) for i, bx, _ in truth_boxes(run, b)] + [(i, bx) for i, bx, _, _ in hand_boxes(run, b)]
        lines = []
        for mid, box in items:
            pr = register.presence(al, ia, ib, mb, box, cfg, diag_a=diag)
            if pr.found_box:
                x1, y1, x2, y2 = (int(v) for v in pr.found_box)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 140, 255), 2)
                cv2.putText(img, pr.result, (x1, y2 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1)
            lines.append(f"pickup {mid[:24]}: {pr.result} (ncc {pr.ncc}, saliency {pr.contrast_a} -> {pr.contrast_b})")
        ys, xs = np.nonzero(mb)
        diag_b = math.hypot(float(xs.max() - xs.min()), float(ys.max() - ys.min()))
        for mid, box in items_b:
            pr = register.presence(inv, ib, ia, ma, box, cfg, diag_a=diag_b)
            x1, y1, x2, y2 = (int(v) for v in box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 140, 255), 1)
            cv2.putText(img, pr.result, (x1, max(36, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1)
            lines.append(f"return {mid[:24]}: {pr.result} (ncc {pr.ncc}, saliency {pr.contrast_a} -> {pr.contrast_b})")
        h = img.shape[0]
        for k, text in enumerate(lines[:8]):
            y = h - 10 - 20 * (len(lines[:8]) - 1 - k)
            cv2.rectangle(img, (0, y - 15), (min(img.shape[1], 9 * len(text) + 12), y + 5), (0, 0, 0), -1)
            cv2.putText(img, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(name, al.summary()["model"], al.ok, lines)


COMMANDS["overlays"] = cmd_overlays


if __name__ == "__main__":
    {"align": cmd_align, "presence": cmd_presence, **COMMANDS}[sys.argv[1]](sys.argv[2:])
