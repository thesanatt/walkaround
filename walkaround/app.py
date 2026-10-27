"""FastAPI routes (M15, docs/ARCHITECTURE.md section 9 and docs/CONTRACTS.md).

`create_app(orchestrator)` is the one factory. Routes never compute anything: they move bytes into the run
directory, start the orchestrator, and serve the files it wrote. Body limits: the chunk route reads at most
`chunk_route_max_body_bytes` and answers 413 past it; every other JSON route is capped by a plain ASGI middleware
at `other_route_max_body_bytes` (multipart routes for top-up clips and claim photos are exempt).
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import shutil
import zipfile
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse, ServerSentEvent

from . import contracts as c
from . import paths
from .orchestrator import (
    TERMINAL_EVENTS,
    TERMINAL_STATUSES,
    CapsExceeded,
    HashMismatch,
    MissingChunks,
    NoSpace,
    Orchestrator,
    et_day_of,
)

SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
ROLE = re.compile(r"^(pickup|return|topup_[0-9]+|claim_[0-9]+)$")
SPA_ROUTES = ("/", "/run/{run_id}", "/run/{run_id}/sheet", "/claim", "/kiosk/{run_id}", "/verify", "/eval/label.html")
SERVED_PREFIXES = ("derived/", "keyframes/", "crops/", "clips/", "chips/", "sheet/", "bundle/", "claim/")
MULTIPART_MAX_BYTES = 1 << 30  # top-up clips go through the same 1 GiB refusal as uploads


class RunCreate(BaseModel):
    mode: c.RunMode = c.RunMode.CAR
    kind: c.RunKind = c.RunKind.PAIR
    keep_plate: bool = False


class FinalizeBody(BaseModel):
    run_id: str
    role: str
    sha256_browser: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    bytes: int = Field(ge=0)
    filename: str = "upload"
    keep_plate: bool | None = None


def canonical(value: Any, status_code: int = 200) -> Response:
    """Contract models leave the server as the same canonical bytes they have on disk."""
    return Response(content=c.json_bytes(value), media_type="application/json", status_code=status_code)


class BodyLimit:
    """Pure ASGI middleware: 413 when Content-Length is over the limit on a non-exempt write route."""

    def __init__(self, app: Any, *, limit: int, exempt: Callable[[str], bool]) -> None:
        self.app = app
        self.limit = limit
        self.exempt = exempt

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] == "http" and scope.get("method") in ("POST", "PUT", "PATCH") and not self.exempt(scope["path"]):
            headers = {k.lower(): v for k, v in scope.get("headers", [])}
            length = headers.get(b"content-length")
            if length and int(length) > self.limit:
                response = JSONResponse({"detail": f"body over {self.limit} bytes"}, status_code=413)
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


def _default_served_allowed(run: c.RunManifest, rel_path: str, *, run_owner: bool) -> bool:
    """The allowlist used until privacy.served_file_allowed lands: blurred artifacts only, never quarantine."""
    if rel_path.startswith("quarantine/") or rel_path.endswith((".orig", ".wav", ".part")):
        return False
    if rel_path.startswith("sam/"):
        return rel_path.endswith((".ndjson.gz", ".index.json", ".summary.json"))
    if rel_path.startswith("media/"):
        return rel_path.endswith((".proxy_blurred.mp4", ".pts.json", ".overlay.mp4"))
    if rel_path in ("spark.json", "voice.json", "deletions.json", "events.ndjson", "events.ndjson.gz"):
        return True
    return rel_path.startswith(SERVED_PREFIXES)


async def _stage_stream(orch: Orchestrator, upload_id: str, read: Callable[[int], Awaitable[bytes]]) -> tuple[int, str]:
    """Write an incoming stream as 8 MB staging chunks; returns (bytes, sha256) so finalize_upload can be reused."""
    chunk_bytes = int(orch.th["media"]["chunk_bytes"])
    digest = hashlib.sha256()
    total = 0
    index = 0
    buffer = bytearray()
    while True:
        block = await read(1 << 20)
        if not block:
            break
        digest.update(block)
        total += len(block)
        buffer += block
        while len(buffer) >= chunk_bytes:
            orch.chunk_path(upload_id, index).write_bytes(bytes(buffer[:chunk_bytes]))
            del buffer[:chunk_bytes]
            index += 1
    if buffer or index == 0:
        orch.chunk_path(upload_id, index).write_bytes(bytes(buffer))
    return total, digest.hexdigest()


def create_app(orchestrator: Orchestrator, *, background: bool = True) -> FastAPI:
    """The one factory. Tests pass an orchestrator with fakes and background=False."""
    orch = orchestrator
    th = orch.th
    dist = paths.web_dist_dir()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        tasks: list[asyncio.Task[Any]] = []
        if background:
            await orch.resume_all()
            tasks.append(asyncio.create_task(_loop(orch.health_probe, float(th["orchestrator"]["health_probe_interval_s"]))))
            tasks.append(asyncio.create_task(_loop(orch.sweep, float(th["orchestrator"]["sweeper_interval_s"]))))
        try:
            yield
        finally:
            for task in tasks:
                task.cancel()
            orch.close()

    app = FastAPI(title="Walkaround", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.orchestrator = orch

    def exempt(path: str) -> bool:
        return "/chunks/" in path or path.endswith("/topup") or path == "/api/claims"

    app.add_middleware(BodyLimit, limit=int(th["media"]["other_route_max_body_bytes"]), exempt=exempt)

    def run_or_404(run_id: str) -> c.RunManifest:
        if not SAFE_ID.match(run_id) or not orch.run_exists(run_id):
            raise HTTPException(404, "no such run")
        return orch.load_run(run_id)

    # Uploads ---------------------------------------------------------------------------------------------------------

    @app.put("/api/uploads/{upload_id}/chunks/{n}", status_code=204)
    async def put_chunk(upload_id: str, n: int, request: Request, run_id: str | None = None) -> Response:
        if not SAFE_ID.match(upload_id) or not (0 <= n < 4096) or (run_id and not SAFE_ID.match(run_id)):
            raise HTTPException(400, "bad upload id or chunk index")
        limit = int(th["media"]["chunk_route_max_body_bytes"])
        declared = request.headers.get("content-length")
        if declared and int(declared) > limit:
            raise HTTPException(413, f"chunk over {limit} bytes")
        data = bytearray()
        async for part in request.stream():
            data += part
            if len(data) > limit:
                raise HTTPException(413, f"chunk over {limit} bytes")
        target = orch.chunk_path(upload_id, n, run_id)
        tmp = target.with_suffix(".part")
        tmp.write_bytes(bytes(data))
        tmp.replace(target)  # idempotent by index: a retried PUT replaces the same file
        return Response(status_code=204)

    @app.post("/api/uploads/{upload_id}/finalize")
    async def finalize(upload_id: str, body: FinalizeBody) -> Response:
        if not SAFE_ID.match(upload_id) or not ROLE.match(body.role):
            raise HTTPException(400, "bad upload id or role")
        if body.bytes > int(th["media"]["upload_max_bytes"]):
            raise HTTPException(413, "over 1 GiB; refused")
        run = run_or_404(body.run_id)
        try:
            record = await orch.finalize_upload(body.run_id, upload_id=upload_id, role=body.role, sha256_browser=body.sha256_browser, bytes=body.bytes, filename=body.filename)
        except HashMismatch as exc:
            return JSONResponse({"sha256_browser": exc.sha256_browser, "sha256_server": exc.sha256_server, "detail": "hash mismatch; upload again"}, status_code=409)
        except MissingChunks as exc:
            return JSONResponse({"detail": "chunks missing; resume upload", "missing": exc.missing}, status_code=400)
        if body.keep_plate is not None and body.keep_plate != run.keep_plate:
            run.keep_plate = body.keep_plate
            orch.save_run(run)
        if run.status == c.RunStatus.CREATED and orch.uploads_complete(run):
            orch.start(run.run_id)
        return canonical(record)

    # Runs ------------------------------------------------------------------------------------------------------------

    @app.post("/api/runs", status_code=201)
    async def create_run(body: RunCreate) -> Response:
        try:
            orch.check_can_start(body.kind)
            run = await orch.create_run(mode=body.mode, kind=body.kind, keep_plate=body.keep_plate)
        except CapsExceeded as exc:
            return JSONResponse({"reason": exc.reason, "pickup_left": exc.pickup_left, "pairs_left": exc.pairs_left}, status_code=429)
        except NoSpace as exc:
            return JSONResponse({"reason": str(exc)}, status_code=507)
        return canonical(run, 201)

    @app.get("/api/runs")
    async def run_index() -> Response:
        """Every run, newest first, without the frozen threshold dict (GET /api/runs/{id} has it)."""
        return canonical([run.model_copy(update={"thresholds": {}}) for run in orch.list_runs()])

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str) -> Response:
        return canonical(run_or_404(run_id))

    @app.post("/api/runs/{run_id}/start", status_code=202)
    async def start_run(run_id: str) -> Response:
        """Start (or resume) the pipeline once every expected upload is in quarantine; idempotent."""
        run = run_or_404(run_id)
        if run.status in TERMINAL_STATUSES:
            return canonical(run)
        if not orch.uploads_complete(run):
            missing = [r for r in orch.expected_roles(run) if not paths.quarantine_original(run_id, r).is_file()]
            raise HTTPException(409, f"upload {', '.join(missing)} first")
        orch.start(run_id)
        return canonical(run, 202)

    @app.get("/api/runs/{run_id}/events")
    async def run_events(run_id: str, request: Request) -> EventSourceResponse:
        run_or_404(run_id)
        last = request.headers.get("last-event-id")
        since = int(last) if last and last.lstrip("-").isdigit() else -1

        async def stream() -> AsyncIterator[ServerSentEvent]:
            if orch.load_run(run_id).status in TERMINAL_STATUSES:
                history: Any = orch.read_events(run_id, since_seq=since)  # the run ended: no tail to wait for
            else:
                history = orch.events.subscribe(run_id, since_seq=since)
            if hasattr(history, "__aiter__"):
                async for event in history:
                    yield ServerSentEvent(data=event.to_json(), event=event.kind.value, id=str(event.seq))
                    if event.kind in TERMINAL_EVENTS:
                        break
            else:
                for event in history:
                    yield ServerSentEvent(data=event.to_json(), event=event.kind.value, id=str(event.seq))

        return EventSourceResponse(stream(), ping=int(th["orchestrator"]["sse_ping_s"]), ping_message_factory=lambda: ServerSentEvent(comment="ping"))

    @app.get("/api/runs/{run_id}/replay")
    async def run_replay(run_id: str, mode: str = "fixed") -> Response:
        run_or_404(run_id)
        try:
            replay_mode = c.ReplayMode(mode)
        except ValueError:
            raise HTTPException(400, "mode is fixed, recorded or instant")
        return canonical(orch.replay_payload(run_id, mode=replay_mode))

    # Sheet and files -------------------------------------------------------------------------------------------------

    def sheet_file(run_id: str, name: str, media_type: str, missing: str) -> FileResponse:
        run_or_404(run_id)
        path = paths.sheet_dir(run_id) / name
        if not path.is_file():
            raise HTTPException(404, missing)
        return FileResponse(path, media_type=media_type)

    @app.get("/api/runs/{run_id}/sheet")
    async def sheet_html(run_id: str) -> FileResponse:
        return sheet_file(run_id, "sheet.html", "text/html", "sheet not written yet")

    @app.get("/api/runs/{run_id}/sheet.json")
    async def sheet_json(run_id: str) -> FileResponse:
        return sheet_file(run_id, "sheet.json", "application/json", "sheet not written yet")

    @app.get("/api/runs/{run_id}/sheet.pdf")
    async def sheet_pdf(run_id: str) -> FileResponse:
        return sheet_file(run_id, "evidence.pdf", "application/pdf", "PDF retry")

    @app.get("/api/runs/{run_id}/bundle.zip")
    async def bundle_zip(run_id: str) -> FileResponse:
        run_or_404(run_id)
        bundle = paths.bundle_dir(run_id)
        files = sorted(p for p in bundle.rglob("*") if p.is_file())
        if not files:
            raise HTTPException(404, "bundle not written yet")
        target = paths.bundle_zip(run_id)
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in files:
                zf.write(path, arcname=str(path.relative_to(bundle)))
        return FileResponse(target, media_type="application/zip", filename=f"walkaround-{run_id}-bundle.zip")

    @app.get("/api/runs/{run_id}/files/{rel_path:path}")
    async def run_file(run_id: str, rel_path: str, request: Request) -> FileResponse:
        run = run_or_404(run_id)
        root = paths.run_dir(run_id).resolve()
        target = (root / rel_path).resolve()
        if rel_path.startswith("/") or ".." in Path(rel_path).parts or root not in target.parents or not target.is_file():
            raise HTTPException(404, "no such file")
        allowed = orch.dep("privacy.served_file_allowed") if orch.has_dep("privacy.served_file_allowed") else _default_served_allowed
        if not allowed(run, rel_path, run_owner=request.query_params.get("view") != "kiosk"):
            raise HTTPException(404, "no such file")
        return FileResponse(target)

    # Top-up and claims -----------------------------------------------------------------------------------------------

    @app.post("/api/runs/{run_id}/topup")
    async def topup(run_id: str, video: UploadFile = File(...)) -> Response:
        run = run_or_404(run_id)
        if not paths.coverage_json(run_id).is_file():
            raise HTTPException(409, "coverage is not derived yet; the top-up waits for the first pass")
        n = 1 + len(list(paths.videos_dir(run_id).glob("topup_*.json")))
        role = f"topup_{n}"
        upload_id = f"topup-{run_id}-{n}"
        size, digest = await _stage_stream(orch, upload_id, video.read)
        if size > MULTIPART_MAX_BYTES:
            raise HTTPException(413, "over 1 GiB; refused")
        await orch.finalize_upload(run_id, upload_id=upload_id, role=role, sha256_browser=digest, bytes=size, filename=video.filename or "topup.mp4")
        coverage = await orch.run_topup(run.run_id, role)
        return canonical(coverage)

    @app.post("/api/claims")
    async def claims(image: UploadFile = File(...), run_id: str = Form(...), text: str = Form(""), tire_code: str | None = Form(None)) -> Response:
        run_or_404(run_id)
        if not orch.has_dep("claim.derive_claim"):
            raise HTTPException(501, "claim-first is not wired yet")
        n = 1 + len(list(paths.claim_dir(run_id).glob("claim_*.jpg")))
        target = paths.claim_image_path(run_id, f"claim_{n}")
        target.write_bytes(await image.read())
        claim = await orch.dep("claim.derive_claim")(run_id, image=target, text=text, tire_code=tire_code, th=orch.run_th(orch.load_run(run_id)))
        return canonical(claim)

    # Status ----------------------------------------------------------------------------------------------------------

    @app.get("/api/status")
    async def status(run_id: str | None = None) -> dict[str, Any]:
        left = orch.caps_remaining(et_day=et_day_of(orch.now()))
        position = orch.queue_position(run_id) if run_id else None
        return {"paused": orch.paused, "pickup_left": left["pickup"], "pairs_left": left["pair"], "queue_position": position, "engine": orch.engine.value}

    @app.get("/api/sample")
    async def sample() -> dict[str, Any]:
        """The precomputed demo pair: never a live run, always flagged as such."""
        run_id = orch.sample_run_id()
        if not run_id or not orch.run_exists(run_id):
            raise HTTPException(404, "no sample pair on this server")
        run = orch.load_run(run_id)
        return {"run_id": run_id, "precomputed": True, "recorded_on": run.created_at[:10], "badge": f"replayed from a run on {run.created_at[:10]}", "replay": f"/api/runs/{run_id}/replay?mode=fixed"}

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"ok": True, "engine": orch.engine.value, "probe_ok": orch.probe_ok, "free_bytes": shutil.disk_usage(paths.runs_root()).free}

    # The web build ---------------------------------------------------------------------------------------------------

    index = dist / "index.html"
    if (dist / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    async def spa(request: Request) -> HTMLResponse:
        if index.is_file():
            return HTMLResponse(index.read_text(encoding="utf-8"))
        return HTMLResponse(FALLBACK_HOME)

    for route in SPA_ROUTES:
        app.add_api_route(route, spa, methods=["GET"], include_in_schema=False)

    return app


FALLBACK_HOME = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Walkaround</title></head>
<body style="font-family: sans-serif; max-width: 40em; margin: 3em auto; line-height: 1.5">
<h1>Walkaround</h1>
<p>The API is running. The web build is not on this server yet, so there is no upload page here.</p>
<p>Health: <a href="/healthz">/healthz</a>. Status and today's remaining live runs: <a href="/api/status">/api/status</a>.</p>
</body></html>
"""


async def _loop(fn: Callable[[], Awaitable[Any]], interval_s: float) -> None:
    while True:
        try:
            await fn()
        except Exception:
            pass
        await asyncio.sleep(interval_s)


def default_app() -> FastAPI:
    """The app uvicorn serves: thresholds from config/, engine from SAM_ENGINE, limits from the environment."""
    import os

    th = c.load_thresholds()
    orch = Orchestrator(
        th=th,
        engine=c.SamEngine(os.environ.get("SAM_ENGINE", "meta-model-api")),
        max_concurrent_runs=int(os.environ.get("MAX_CONCURRENT_RUNS", th["orchestrator"]["max_concurrent_runs"])),
        media_workers=int(os.environ.get("MEDIA_WORKERS", th["orchestrator"]["media_workers"])),
        decode_workers=int(os.environ.get("DECODE_WORKERS", th["orchestrator"]["decode_workers"])),
    )
    return create_app(orch)


__all__ = ["BodyLimit", "FinalizeBody", "RunCreate", "canonical", "create_app", "default_app"]
