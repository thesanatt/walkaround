"""Box fix: a local page for checking the product's mark boxes by hand.

Serves blurred crops and marks read-only from runs/, and writes corrections to eval/box_corrections/<run_id>.json.
Standard library only. Binds to 127.0.0.1.

    .venv/bin/python tools/boxfix/serve.py [extra_run_id ...] [--port 8790] [--only]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNS = REPO / "runs"
OUT = REPO / "eval" / "box_corrections"
SYNTH = REPO.parent / "synthetic" / "out"
DEFAULT_RUNS = ["pairC_core3", "eval_nhtsa15759_r6", "synth_pair2_r9"]
RUN_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
MAX_BODY = 5_000_000

_gt_cache: dict[str, dict] = {}
_write_lock = threading.Lock()


def read_json(p: Path):
    return json.loads(p.read_text())


def synth_pair(run_id: str) -> str | None:
    m = re.match(r"^synth_(pair\d+)", run_id)
    if m and (SYNTH / m.group(1) / "ground_truth.json").exists():
        return m.group(1)
    return None


def ground_truth(pair: str) -> dict:
    if pair not in _gt_cache:
        _gt_cache[pair] = read_json(SYNTH / pair / "ground_truth.json")
    return _gt_cache[pair]


def truth_in_crop(pair: str, kf: dict) -> list[dict]:
    # Same projection as docs/evidence/box_fix_1101/truth.py: render boxes are in source pixels.
    g = ground_truth(pair)
    c = kf["crop"]
    f = int(round(kf["pts"] * g["fps"]))
    out = []
    for m in g["marks"]:
        rows = m["frames"].get(kf["role"]) or []
        row = rows[f] if f < len(rows) and rows[f].get("frame") == f else next((r for r in rows if r.get("frame") == f), None)
        if not row or not row.get("visible") or not row.get("visible_bbox_xyxy"):
            continue
        b = row["visible_bbox_xyxy"]
        box = [(b[0] - c["dx"]) * c["scale"], (b[1] - c["dy"]) * c["scale"],
               (b[2] - c["dx"]) * c["scale"], (b[3] - c["dy"]) * c["scale"]]
        if box[2] > 0 and box[0] < c["w"] and box[3] > 0 and box[1] < c["h"]:
            out.append({"id": m["id"], "kind": m.get("kind"), "box": [round(v, 2) for v in box]})
    return out


def mark_item(m: dict) -> dict:
    br = m.get("box_refine") or {}
    box = m.get("bbox_crop")
    point_only = False
    if not box and m.get("point_crop"):
        x, y = m["point_crop"]
        box, point_only = [x - 12, y - 12, x + 12, y + 12], True
    spark = br.get("spark_bbox_crop")
    return {
        "mark_id": m["mark_id"],
        "type": m.get("type"),
        "description": m.get("description") or "",
        "confidence": m.get("confidence"),
        "source": m.get("source"),
        "size_class": m.get("size_class"),
        "box": box,
        "point_only": point_only,
        "spark_box": spark if spark and spark != box else None,
        "box_used": br.get("used"),
        "box_note": br.get("note") or "",
    }


def load_run(run_id: str) -> dict:
    d = RUNS / run_id
    kfs = read_json(d / "derived" / "keyframes.json")
    marks = read_json(d / "derived" / "marks.json")
    pair = synth_pair(run_id)
    by_kf: dict[str, list[dict]] = {}
    for m in marks:
        by_kf.setdefault(m["kf_id"], []).append(m)
    crops = []
    for kf in kfs:
        c = kf.get("crop")
        if not c or not c.get("path") or kf.get("marks_status") not in ("complete", "truncated"):
            continue
        if not (d / c["path"]).is_file():
            continue
        crops.append({
            "kf_id": kf["kf_id"],
            "role": kf["role"],
            "panel_id": kf["panel_id"],
            "w": c["w"],
            "h": c["h"],
            "url": f"/crop/{run_id}/{kf['kf_id']}",
            "marks": [mark_item(m) for m in by_kf.get(kf["kf_id"], [])],
            "gt": truth_in_crop(pair, kf) if pair else None,
        })
    crops.sort(key=lambda c: (c["role"] != "pickup", c["role"]))  # stable: pickup first, keyframes.json order kept
    return {"run_id": run_id, "synthetic": pair is not None, "gt_source": str(SYNTH / pair / "ground_truth.json") if pair else None,
            "crops": crops, "n_marks": sum(len(c["marks"]) for c in crops)}


class State:
    def __init__(self, run_ids: list[str]):
        self.run_ids = run_ids
        self.runs = {r: load_run(r) for r in run_ids}
        self.crop_paths = {}
        for r in run_ids:
            for kf in read_json(RUNS / r / "derived" / "keyframes.json"):
                if kf.get("crop") and kf["crop"].get("path"):
                    self.crop_paths[(r, kf["kf_id"])] = (RUNS / r / kf["crop"]["path"]).resolve()


def atomic_write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=1, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def empty_doc(run_id: str) -> dict:
    return {"run_id": run_id, "schema": 1, "marks": {}, "missed": [], "crops_seen": []}


def valid_box(b) -> bool:
    return isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b) and b[2] > b[0] and b[3] > b[1]


def check_doc(doc, run_id: str) -> str | None:
    if not isinstance(doc, dict) or doc.get("run_id") != run_id:
        return "run_id does not match"
    if not isinstance(doc.get("marks"), dict) or not isinstance(doc.get("missed"), list):
        return "marks must be an object and missed a list"
    for mid, row in doc["marks"].items():
        if not isinstance(row, dict) or row.get("status") not in ("correct", "moved", "not_mark"):
            return f"bad status for {mid}"
        if row["status"] != "not_mark" and not valid_box(row.get("box")):
            return f"bad box for {mid}"
    for row in doc["missed"]:
        if not isinstance(row, dict) or not valid_box(row.get("box")) or not row.get("kf_id"):
            return "bad missed box"
    return None


class Handler(BaseHTTPRequestHandler):
    state: State
    port: int

    def log_message(self, fmt, *args):  # quiet: only errors
        if args and str(args[1]).startswith(("4", "5")):
            sys.stderr.write("%s %s\n" % (self.log_date_time_string(), fmt % args))

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").lower()
        return host in (f"127.0.0.1:{self.port}", f"localhost:{self.port}")

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store" if ctype.startswith(("application/json", "text/html")) else "max-age=3600")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        if not self._host_ok():
            return self._json(HTTPStatus.FORBIDDEN, {"error": "bad host"})
        path = unquote(urlparse(self.path).path)
        if path in ("/", "/index.html"):
            return self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/api/items":
            return self._json(200, {"runs": [self.state.runs[r] for r in self.state.run_ids]})
        m = re.match(r"^/crop/([^/]+)/([^/]+)$", path)
        if m:
            p = self.state.crop_paths.get((m.group(1), m.group(2)))
            if not p or not p.is_file():
                return self._json(404, {"error": "no such crop"})
            ctype = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
            return self._send(200, p.read_bytes(), ctype)
        m = re.match(r"^/api/corrections/([^/]+)$", path)
        if m:
            run_id = m.group(1)
            if run_id not in self.state.runs:
                return self._json(404, {"error": "run not served"})
            f = OUT / f"{run_id}.json"
            return self._json(200, read_json(f) if f.exists() else empty_doc(run_id))
        self._json(404, {"error": "not found"})

    def do_PUT(self):
        if not self._host_ok():
            return self._json(HTTPStatus.FORBIDDEN, {"error": "bad host"})
        path = unquote(urlparse(self.path).path)
        m = re.match(r"^/api/corrections/([^/]+)$", path)
        if not m or m.group(1) not in self.state.runs:
            return self._json(404, {"error": "run not served"})
        run_id = m.group(1)
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > MAX_BODY:
            return self._json(413, {"error": "body size"})
        try:
            doc = json.loads(self.rfile.read(n))
        except ValueError:
            return self._json(400, {"error": "not JSON"})
        err = check_doc(doc, run_id)
        if err:
            return self._json(400, {"error": err})
        doc["schema"] = 1
        doc["saved_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with _write_lock:
            atomic_write(OUT / f"{run_id}.json", doc)
        self._json(200, {"ok": True, "saved_at": doc["saved_at"]})

    do_POST = do_PUT


def main() -> None:
    ap = argparse.ArgumentParser(description="Box fix: check mark boxes by hand.")
    ap.add_argument("runs", nargs="*", help="extra run ids under runs/")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--only", action="store_true", help="serve only the runs given, not the three defaults")
    a = ap.parse_args()
    run_ids = [] if a.only else list(DEFAULT_RUNS)
    for r in a.runs:
        if r not in run_ids:
            run_ids.append(r)
    for r in run_ids:
        if not RUN_ID.match(r) or not (RUNS / r / "derived" / "marks.json").is_file():
            sys.exit(f"no marks for run {r!r} under {RUNS}")
    state = State(run_ids)
    Handler.state, Handler.port = state, a.port
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    for r in run_ids:
        d = state.runs[r]
        print(f"  {r}: {d['n_marks']} boxes on {sum(1 for c in d['crops'] if c['marks'])} crops"
              f"{' (ground truth available)' if d['synthetic'] else ''}")
    print(f"Box fix on http://127.0.0.1:{a.port}/  (corrections go to {OUT})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
