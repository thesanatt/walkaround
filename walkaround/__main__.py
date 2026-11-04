"""Command line: walkaround (or python -m walkaround) {run, rebuild, verify, serve}.

run <pickup> [<return>] runs the floor end to end through walkaround.pipeline (the browser's upload path, seal, every
stage) and prints every event as one JSON line, then the final run.json; it needs LIVE=1 because it spends API credit.
--run-id names the run (an unfinished run of that id resumes); --nouns sends stand-in prompts for a test clip with no
car (pipeline.stand_in_prompts); --tire-code stores the sidewall size on a new run's run.json so the wheel ruler can
state sizes. run --replay <run_id> re-emits a stored run's events with the pacing of --mode (fixed
by default) and spends nothing.
rebuild <run_id> [--out DIR] regenerates the derived files from the frozen facts with no network call; a closed run
(sealed in bundle/manifest_run.json) is never rebuilt in place, so without --out it goes to runs/_rebuild/.
verify <run_id> prints "rebuild identical" or each problem, the first differing file first, and exits 1 on any.
serve starts uvicorn on the FastAPI app.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from . import contracts as c
from . import paths, pipeline, ruler
from .orchestrator import Orchestrator, RebuildRefused, RunLocked


def make_orchestrator(th: dict[str, Any] | None = None) -> Orchestrator:
    return pipeline.make_orchestrator(th)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="walkaround", description="pickup video plus return video in, evidence sheet out")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the pipeline on local files, or replay a stored run")
    run.add_argument("pickup", nargs="?", type=Path, help="the pickup walk (MP4 or MOV)")
    run.add_argument("return_", nargs="?", type=Path, metavar="return", help="the return walk; omit for a pickup-only check")
    run.add_argument("--replay", metavar="RUN_ID", help="re-emit a stored run's events instead of calling any model")
    run.add_argument("--mode", choices=[m.value for m in c.ReplayMode], default=c.ReplayMode.FIXED.value, help="replay pacing")
    run.add_argument("--keep-plate", action="store_true", help="the renter's choice: keep the plate on the sheet")
    run.add_argument("--quiet", action="store_true", help="print only the final run.json")
    run.add_argument("--run-id", help="name the run (letters, digits, _ and -); an unfinished run of that id resumes")
    run.add_argument("--nouns", nargs="+", metavar="NOUN", help="stand-in prompts for a test clip with no car: each stands in for one single-slot panel")
    run.add_argument("--container", metavar="NOUN", help="with --nouns: the prompt sent as the car lane (default the last of --nouns)")
    run.add_argument("--keep", action="store_true", help="keep this run past the retention sweep (sample and eval runs); a kept run is public and takes no claim or top-up")
    run.add_argument("--footage-credit", metavar="TEXT", help="the footage credit line shown under the walk player (licensed sample footage)")
    run.add_argument("--tire-code", metavar="CODE", help="the tire size on the sidewall, such as 155/80R13, so the wheel ruler can state sizes (a new run only)")

    rebuild = sub.add_parser("rebuild", help="regenerate the derived files from the frozen facts, no network")
    rebuild.add_argument("run_id")
    rebuild.add_argument("--out", type=Path, help="write the rebuilt run under this directory instead of in place")
    rebuild.add_argument("--thresholds", type=Path, help="a thresholds.json to use instead of the run's frozen one")

    verify = sub.add_parser("verify", help="rebuild and compare; print 'rebuild identical' or the first differing file")
    verify.add_argument("run_id")
    verify.add_argument("--out", type=Path, help="scratch directory for the rebuild (default runs/_verify)")

    rederive = sub.add_parser("rederive", help="a new run from a run's sealed originals and stored model answers, under this version's rules, no model call")
    rederive.add_argument("run_id")
    rederive.add_argument("new_run_id")

    serve = sub.add_parser("serve", help="start the web app")
    serve.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    serve.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    return parser


def _print_event(event: c.Event) -> None:
    sys.stdout.write(event.to_json() + "\n")
    sys.stdout.flush()


async def _replay(orch: Orchestrator, run_id: str, mode: str) -> int:
    if not orch.run_exists(run_id):
        sys.stderr.write(f"no run {run_id}\n")
        return 1
    async for event in orch.replay_events(run_id, mode=c.ReplayMode(mode)):
        _print_event(event)
    return 0


async def _run_local(orch: Orchestrator, args: argparse.Namespace) -> int:
    if args.tire_code and args.run_id and orch.run_exists(args.run_id) and orch.load_run(args.run_id).tire_code != args.tire_code:
        kept = orch.load_run(args.run_id).tire_code or "no tire code"
        sys.stderr.write(f"--tire-code applies to a new run; run {args.run_id} keeps {kept}\n")
    final = await pipeline.run_pair_async(
        args.pickup,
        args.return_,
        args.run_id,
        args.nouns,
        container=args.container,
        keep_plate=args.keep_plate,
        keep=args.keep,
        footage_credit=args.footage_credit,
        tire_code=args.tire_code,
        orch=orch,
        on_event=None if args.quiet else _print_event,
    )
    sys.stdout.write(final.to_json() + "\n")
    return 0 if final.status in (c.RunStatus.DONE, c.RunStatus.PARTIAL) else 1


def cmd_run(args: argparse.Namespace, orch: Orchestrator) -> int:
    if args.replay:
        return asyncio.run(_replay(orch, args.replay, args.mode))
    if args.pickup is None:
        sys.stderr.write("run needs a pickup file, or --replay RUN_ID\n")
        return 2
    for path in (args.pickup, args.return_):
        if path is not None and not path.is_file():
            sys.stderr.write(f"not a file: {path}\n")
            return 2
    if args.container and not args.nouns:
        sys.stderr.write("--container goes with --nouns\n")
        return 2
    if args.tire_code is not None:
        code = ruler.normalize_tire_code(args.tire_code)
        if code is None:
            sys.stderr.write(f"{ruler.TIRE_UNREADABLE}: {args.tire_code!r} (a sidewall size such as 155/80R13)\n")
            return 2
        args.tire_code = code
    try:
        lanes, prompts = pipeline.stand_in_prompts(args.nouns, container=args.container) if args.nouns else (None, {})
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    if os.environ.get("LIVE") != "1":
        kind = "pair" if args.return_ is not None else "pickup_check"
        plan = f"would run a {kind} on {args.pickup}" + (f" and {args.return_}" if args.return_ else "")
        if prompts:
            plan += " with lanes " + ", ".join(f"{n.value}='{prompts.get(n.value, n.value)}'" if prompts.get(n.value, n.value) else f"{n.value}=(whole frame, no request)" for n in lanes)
        if args.tire_code:
            plan += f", tire code {args.tire_code} ({ruler.tire_diameter_in(args.tire_code):.1f} in)"
        sys.stderr.write(plan + "; set LIVE=1 to spend API credit\n")
        return 2
    try:
        return asyncio.run(_run_local(orch, args))
    except (FileExistsError, ValueError, RunLocked) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2


def sample_copy(orch: Orchestrator, run_id: str) -> str | None:
    """Why a run cannot be rebuilt here: a replay copy (the sample pair) holds only the files the app serves, and a
    rebuild needs the stored model answers too."""
    if orch.load_run(run_id).live_or_replay != c.LiveOrReplay.REPLAY or paths.spark_json(run_id).is_file():
        return None
    return (
        f"run {run_id} is a replay copy that holds only the files the app serves; a rebuild needs its stored model "
        "answers (spark.json, voice.json), so verify and rebuild work on your own runs, not on the sample pair\n"
    )


def cmd_rebuild(args: argparse.Namespace, orch: Orchestrator) -> int:
    if not orch.run_exists(args.run_id):
        sys.stderr.write(f"no run {args.run_id}\n")
        return 1
    if why := sample_copy(orch, args.run_id):
        sys.stderr.write(why)
        return 2
    thresholds = c.threshold_values(json.loads(args.thresholds.read_text(encoding="utf-8"))) if args.thresholds else None
    if args.out is None and orch.is_closed(args.run_id):
        sys.stderr.write(f"run {args.run_id} is closed, so its sealed files stay as they are; rebuilding into runs/_rebuild/{args.run_id}\n")
    try:
        result, _out = pipeline.rebuild(args.run_id, out=args.out, thresholds=thresholds, orch=orch)
    except RebuildRefused as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    for name in sorted(result):
        sys.stdout.write(f"{name}: {result[name]}\n")
    return 0


def cmd_verify(args: argparse.Namespace, orch: Orchestrator) -> int:
    if not orch.run_exists(args.run_id):
        sys.stderr.write(f"no run {args.run_id}\n")
        return 1
    if why := sample_copy(orch, args.run_id):
        sys.stderr.write(why)
        return 2
    try:
        problems = pipeline.verify(args.run_id, out=args.out, orch=orch)
    except RebuildRefused as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    if not problems:
        sys.stdout.write("rebuild identical\n")
        return 0
    said: set[str] = set()
    for problem in problems:
        why = why_differs(args.run_id, problem)
        if why:
            again = "the same re-derivation" if why.startswith("display video re-derived") else "the same rule change"
            problem = f"{problem}; {again if why in said else why}"
            said.add(why)
        sys.stdout.write(problem + "\n")
    return 1


def why_differs(run_id: str, problem: str) -> str | None:
    """For a "rebuild differs" line, the known rule change that explains it when the stored file shows one (verify
    rebuilds an older run with today's code): match.why_rebuild_differs (the 10/31 view-flag rule) or
    narration.why_rebuild_differs (Voice's speaker markers left out of the words shown). For a run manifest line on a
    display video, the row privacy.install_rederived_display wrote when it replaced a sealed one (pairC_core1, 11/1)."""
    from . import narration, privacy
    from .match import why_rebuild_differs

    for media in ("run manifest: size differs: ", "run manifest: hash differs: "):
        if problem.startswith(media):
            return privacy.rederived_display_note(run_id, problem[len(media):])
    prefix = "rebuild differs: "
    if not problem.startswith(prefix):
        return None
    name = problem[len(prefix):]
    return why_rebuild_differs(run_id, name) or narration.why_rebuild_differs(run_id, name)


def cmd_rederive(args: argparse.Namespace, orch: Orchestrator) -> int:
    if not orch.run_exists(args.run_id):
        sys.stderr.write(f"no run {args.run_id}\n")
        return 1
    if why := sample_copy(orch, args.run_id):
        sys.stderr.write(why)
        return 2
    try:
        run = pipeline.rederive(args.run_id, args.new_run_id, orch=orch)
    except (RebuildRefused, FileExistsError, ValueError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    sys.stdout.write(f"{run.run_id}: {run.rederived.text if run.rederived else ''}\n")
    return 0


def cmd_serve(args: argparse.Namespace, orch: Orchestrator) -> int:
    import uvicorn

    from .app import create_app

    uvicorn.run(create_app(orch), host=args.host, port=args.port)
    return 0


COMMANDS = {"run": cmd_run, "rebuild": cmd_rebuild, "verify": cmd_verify, "rederive": cmd_rederive, "serve": cmd_serve}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return COMMANDS[args.command](args, make_orchestrator())


if __name__ == "__main__":
    sys.exit(main())
