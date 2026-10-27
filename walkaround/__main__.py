"""Command line: python -m walkaround {run, rebuild, verify, serve}.

run <pickup> [<return>] runs the pipeline on local files through the same path the browser uses (chunks, finalize,
seal, stages) and prints every event as one JSON line; it needs LIVE=1 because it spends API credit.
run --replay <run_id> re-emits a stored run's events with the pacing of --mode (fixed by default) and spends nothing.
rebuild <run_id> [--out DIR] regenerates the derived files from the frozen facts with no network call.
verify <run_id> prints "rebuild identical" or the first differing file and exits 1 on any problem.
serve starts uvicorn on the FastAPI app.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from . import contracts as c
from .orchestrator import Orchestrator, StageGap  # noqa: F401  (StageGap re-exported for scripts)


def make_orchestrator(th: dict[str, Any] | None = None) -> Orchestrator:
    th = th or c.load_thresholds()
    return Orchestrator(
        th=th,
        engine=c.SamEngine(os.environ.get("SAM_ENGINE", "meta-model-api")),
        max_concurrent_runs=int(os.environ.get("MAX_CONCURRENT_RUNS", th["orchestrator"]["max_concurrent_runs"])),
        media_workers=int(os.environ.get("MEDIA_WORKERS", th["orchestrator"]["media_workers"])),
        decode_workers=int(os.environ.get("DECODE_WORKERS", th["orchestrator"]["decode_workers"])),
    )


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

    rebuild = sub.add_parser("rebuild", help="regenerate the derived files from the frozen facts, no network")
    rebuild.add_argument("run_id")
    rebuild.add_argument("--out", type=Path, help="write the rebuilt run under this directory instead of in place")
    rebuild.add_argument("--thresholds", type=Path, help="a thresholds.json to use instead of the run's frozen one")

    verify = sub.add_parser("verify", help="rebuild and compare; print 'rebuild identical' or the first differing file")
    verify.add_argument("run_id")
    verify.add_argument("--out", type=Path, help="scratch directory for the rebuild (default runs/_verify)")

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


async def _stage_local_file(orch: Orchestrator, run_id: str, role: str, src: Path) -> c.VideoRecord:
    """A local file goes through the upload path: 8 MB chunks, then finalize with the hash of the bytes sent."""
    chunk_bytes = int(orch.th["media"]["chunk_bytes"])
    upload_id = f"cli-{role}"
    digest = hashlib.sha256()
    size = 0
    with open(src, "rb") as fh:
        for index, block in enumerate(iter(lambda: fh.read(chunk_bytes), b"")):
            digest.update(block)
            size += len(block)
            orch.chunk_path(upload_id, index, run_id).write_bytes(block)
    if size == 0:
        orch.chunk_path(upload_id, 0, run_id).write_bytes(b"")
    return await orch.finalize_upload(run_id, upload_id=upload_id, role=role, sha256_browser=digest.hexdigest(), bytes=size, filename=src.name)


async def _run_local(orch: Orchestrator, pickup: Path, return_: Path | None, *, keep_plate: bool, quiet: bool) -> int:
    kind = c.RunKind.PAIR if return_ is not None else c.RunKind.PICKUP_CHECK
    run = await orch.create_run(mode=c.RunMode.CAR, kind=kind, keep_plate=keep_plate)
    printer = asyncio.create_task(_tail(orch, run.run_id)) if not quiet else None
    await _stage_local_file(orch, run.run_id, "pickup", pickup)
    if return_ is not None:
        await _stage_local_file(orch, run.run_id, "return", return_)
    final = await orch.run_pipeline(run.run_id)
    if printer is not None:
        await asyncio.sleep(0)
        printer.cancel()
    sys.stdout.write(final.to_json() + "\n")
    return 0 if final.status in (c.RunStatus.DONE, c.RunStatus.PARTIAL) else 1


async def _tail(orch: Orchestrator, run_id: str) -> None:
    try:
        async for event in orch.events.subscribe(run_id, since_seq=-1):
            _print_event(event)
    except asyncio.CancelledError:
        return


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
    if os.environ.get("LIVE") != "1":
        kind = "pair" if args.return_ is not None else "pickup_check"
        sys.stderr.write(f"would run a {kind} on {args.pickup}" + (f" and {args.return_}" if args.return_ else "") + "; set LIVE=1 to spend API credit\n")
        return 2
    return asyncio.run(_run_local(orch, args.pickup, args.return_, keep_plate=args.keep_plate, quiet=args.quiet))


def cmd_rebuild(args: argparse.Namespace, orch: Orchestrator) -> int:
    if not orch.run_exists(args.run_id):
        sys.stderr.write(f"no run {args.run_id}\n")
        return 1
    thresholds = c.threshold_values(json.loads(args.thresholds.read_text(encoding="utf-8"))) if args.thresholds else None
    result = orch.rebuild(args.run_id, out=args.out, thresholds=thresholds)
    for name in sorted(result):
        sys.stdout.write(f"{name}: {result[name]}\n")
    return 0


def cmd_verify(args: argparse.Namespace, orch: Orchestrator) -> int:
    if not orch.run_exists(args.run_id):
        sys.stderr.write(f"no run {args.run_id}\n")
        return 1
    problems = orch.verify(args.run_id, out=args.out)
    if not problems:
        sys.stdout.write("rebuild identical\n")
        return 0
    for problem in problems:
        sys.stdout.write(problem + "\n")
    return 1


def cmd_serve(args: argparse.Namespace, orch: Orchestrator) -> int:
    import uvicorn

    from .app import create_app

    uvicorn.run(create_app(orch), host=args.host, port=args.port)
    return 0


COMMANDS = {"run": cmd_run, "rebuild": cmd_rebuild, "verify": cmd_verify, "serve": cmd_serve}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return COMMANDS[args.command](args, make_orchestrator())


if __name__ == "__main__":
    sys.exit(main())
