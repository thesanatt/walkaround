"""Make a walk's blurred display video again from the renter's original, outside the run, and say whether it is the
run's own file byte for byte. No model call: the plate and face tracks come from the run's stored lanes.

    rederive_display.py RUN_ID ORIGINAL [--role pickup] --out DIR [--json] [--install --reason TEXT]

The original must carry the SHA-256 sealed in runs/<run_id>/videos/<role>.json. The proxy is cut again with the
run's media settings and compared with the sealed proxy hash, then pixelated by privacy.render_display with the run's
thresholds (keys a run frozen before Oct 31 lacks take today's values). The unblurred proxy lives only in DIR and is
deleted before the script ends; DIR may not be inside the run. A run made with this blur code prints "matches the
sealed display video"; an older one prints both hashes, so a re-derived file is never passed off as the sealed one.

--install puts the new file in place of the run's display video and writes the sealed and new hashes, the time and
--reason to runs/<run_id>/rederived_display.json, which walkaround verify then names on the media lines it reports
(the sealed records stay as sealed). Used once, on Nov 1, for pairC_core1, whose 5 px blocks left a plate readable:
    rederive_display.py pairC_core1 data/candidates/uco3d_car_automobile/uco3d_50152-5241-46685.mp4 --role return
        --out DIR --install --reason "the sealed one left a plate readable under 5 px blocks at 4.4 s"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from walkaround import privacy  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_id")
    ap.add_argument("original", type=Path)
    ap.add_argument("--role", default="pickup")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    ap.add_argument("--install", action="store_true", help="replace the run's display video with the new one and record why")
    ap.add_argument("--reason", help="why the display video is replaced (with --install)")
    args = ap.parse_args(argv)
    if args.install and not args.reason:
        ap.error("--install needs --reason")
    try:
        got = privacy.rederive_display(args.run_id, args.role, args.original, args.out)
    except (ValueError, FileNotFoundError) as exc:
        print(f"rederive_display: {exc}", file=sys.stderr)
        return 2
    if args.install:
        try:
            got["installed"] = privacy.install_rederived_display(args.run_id, got, reason=args.reason)
        except ValueError as exc:
            print(f"rederive_display: {exc}", file=sys.stderr)
            return 2
    if args.json:
        print(json.dumps(got, indent=1, sort_keys=True))
        return 0
    print(f"{args.role}: original {got['original_sha256'][:12]} matches the sealed upload")
    proxy = "matches" if got["proxy_matches_sealed"] else "differs from"
    print(f"{args.role}: proxy {got['proxy_sha256'][:12]} {proxy} the sealed proxy")
    if got["matches_sealed"]:
        print(f"{args.role}: {got['blurred_path']} matches the sealed display video ({got['blurred_sha256'][:12]})")
    else:
        sealed = (got["sealed_blurred_sha256"] or "none")[:12]
        print(f"{args.role}: {got['blurred_path']} is {got['blurred_sha256'][:12]}; the run sealed {sealed} (made with other blur code)")
    if "installed" in got:
        print(f"{args.role}: installed as {got['installed']['path']}; recorded in runs/{args.run_id}/rederived_display.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
