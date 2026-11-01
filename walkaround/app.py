"""FastAPI routes (M15; docs/ARCHITECTURE.md sections 7, 9 and 14; docs/CONTRACTS.md).

create_app(orchestrator) is the one factory. Routes move bytes into the run directory, ask the orchestrator to act and
serve the files it wrote; they compute nothing. Body limits: the chunk route reads at most chunk_route_max_body_bytes
(413 past it); the top-up route streams its multipart body straight into upload chunks inside the run directory, so
no unblurred clip lands in a temp file outside it, and refuses past 1 GiB; every other write route is held to
other_route_max_body_bytes by a small ASGI middleware, with or without a Content-Length header.

Who sees what (section 8 item 5): the run id is the renter's receipt, so the run index lists only public runs (the
sample pair and keep runs), and a file is served only when the app's own floor (never quarantine, upload chunks, an
original, the unblurred proxy or the pending event file) and privacy.served_file_allowed both allow it; that holds for
the files route, the sheet routes and the bundle zip alike. The sample pair, keep runs and the kiosk view
(?view=kiosk) are never the run owner, and a public run takes no top-up or claim from whoever holds its listed id.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import re
import secrets
import zipfile
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from pathlib import PurePosixPath
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from python_multipart import MultipartParser
from python_multipart.multipart import parse_options_header
from sse_starlette.sse import EventSourceResponse, ServerSentEvent

from . import contracts as c
from . import paths
from .config import redact
from .orchestrator import (
    CREATABLE_KINDS,
    PENDING_EVENTS_NAME,
    TERMINAL_EVENTS,
    TERMINAL_STATUSES,
    CapsExceeded,
    HashMismatch,
    MissingChunks,
    NoSpace,
    Orchestrator,
    UploadRefused,
    VideoUnreadable,
    et_day_of,
)

SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
ROLE = re.compile(r"^(pickup|return|topup_[0-9]+|claim_[0-9]+)$")
SPA_ROUTES = ("/", "/upload", "/run/{run_id}", "/run/{run_id}/sheet", "/claim", "/kiosk/{run_id}", "/verify")
EXEMPT_PATHS = re.compile(r"^/api/(uploads/[^/]+/chunks/[^/]+|runs/[^/]+/topup|claims)$")
NEVER_SERVED_TOPS = ("quarantine", "uploads")
NEVER_SERVED_SUFFIXES = (".orig", ".part", ".tmp", ".proxy.mp4")
CLAIM_MAX_BYTES = 52_428_800  # a claim photo; the same 50 MB cap as an inline image
FIELD_MAX_BYTES = 4096
SSE_POLL_S = 5.0  # how often a quiet event stream checks whether its run ended without a chained terminal event


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
    """Pure ASGI middleware: 413 past the limit on a non-exempt write route, by Content-Length or by counting."""

    def __init__(self, app: Any, *, limit: int, exempt: Callable[[str], bool]) -> None:
        self.app = app
        self.limit = limit
        self.exempt = exempt

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http" or scope.get("method") not in ("POST", "PUT", "PATCH") or self.exempt(scope["path"]):
            await self.app(scope, receive, send)
            return
        headers = {k.lower(): v for k, v in scope.get("headers", [])}
        length = headers.get(b"content-length")
        if length is not None:
            if not length.isdigit() or int(length) > self.limit:
                await self._refuse(scope, receive, send)
                return
            await self.app(scope, receive, send)
            return
        body, more = bytearray(), True  # no Content-Length (chunked): read up to the limit, then replay it
        while more:
            message = await receive()
            if message["type"] != "http.request":
                await self.app(scope, _replay([message]), send)
                return
            body += message.get("body", b"")
            more = message.get("more_body", False)
            if len(body) > self.limit:
                await self._refuse(scope, receive, send)
                return
        await self.app(scope, _replay([{"type": "http.request", "body": bytes(body), "more_body": False}]), send)

    async def _refuse(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        await JSONResponse({"detail": f"body over {self.limit} bytes"}, status_code=413)(scope, receive, send)


def _replay(messages: list[dict[str, Any]]) -> Callable[[], Awaitable[dict[str, Any]]]:
    queue = list(messages)

    async def receive() -> dict[str, Any]:
        if queue:
            return queue.pop(0)
        await asyncio.Event().wait()  # the request is fully read; wait like Starlette does until the app returns
        return {"type": "http.disconnect"}

    return receive


def servable(rel_path: str) -> bool:
    """The app's floor under every served file, whatever privacy.served_file_allowed says."""
    p = PurePosixPath(rel_path)
    if not p.parts or p.is_absolute() or any(part in ("", ".", "..") or part.startswith(".") for part in p.parts):
        return False
    top, name = p.parts[0].casefold(), p.name.casefold()  # APFS matches names case-insensitively
    if top in NEVER_SERVED_TOPS or name == PENDING_EVENTS_NAME:
        return False
    return not name.endswith(NEVER_SERVED_SUFFIXES)


class _ChunkSink:
    """A byte stream written as the chunk route's 8 MB chunk files, hashed on the way."""

    def __init__(self, orch: Orchestrator, upload_id: str, run_id: str) -> None:
        self.orch, self.upload_id, self.run_id = orch, upload_id, run_id
        self.size = int(orch.th["media"]["chunk_bytes"])
        self.digest = hashlib.sha256()
        self.total = 0
        self.index = 0
        self.buffer = bytearray()

    def write(self, data: bytes) -> None:
        self.digest.update(data)
        self.total += len(data)
        self.buffer += data
        while len(self.buffer) >= self.size:
            self._flush(bytes(self.buffer[: self.size]))
            del self.buffer[: self.size]

    def _flush(self, block: bytes) -> None:
        self.orch.chunk_path(self.upload_id, self.index, self.run_id).write_bytes(block)
        self.index += 1

    def close(self) -> None:
        if self.buffer or self.index == 0:
            self._flush(bytes(self.buffer))
            self.buffer.clear()


class _MemorySink:
    """A small upload held in memory (a claim photo); read_multipart's max_bytes caps it."""

    def __init__(self) -> None:
        self.data = bytearray()

    def write(self, data: bytes) -> None:
        self.data += data

    def close(self) -> None:
        pass


def next_claim_number(run_id: str) -> int:
    """1 + the highest n among the run's claim_<n> files (claim_3.json, claim_3.blurred.jpg, claim_3.answers.json)."""
    folder = paths.runs_root() / run_id / "claim"
    names = [p.name for p in folder.iterdir()] if folder.is_dir() else []
    return 1 + max((int(m.group(1)) for name in names if (m := re.match(r"claim_([0-9]+)\.", name))), default=0)


async def read_multipart(request: Request, *, file_field: str, sink: _ChunkSink | _MemorySink, max_bytes: int, too_big: str = "over 1 GiB; refused") -> tuple[dict[str, str], str | None]:
    """Stream a multipart body: the file field goes to sink as it arrives, small text fields come back in a dict."""
    ctype, params = parse_options_header(request.headers.get("content-type", ""))
    boundary = params.get(b"boundary")
    if ctype != b"multipart/form-data" or not boundary:
        raise HTTPException(400, f"send the clip as multipart/form-data with a '{file_field}' file field")
    fields: dict[str, str] = {}
    part: dict[str, Any] = {}
    header: dict[str, bytes] = {"field": b"", "value": b""}
    found: list[str | None] = []

    def on_part_begin() -> None:
        part.clear()
        part.update(headers={}, name="", filename=None, text=bytearray())

    def on_header_field(data: bytes, start: int, end: int) -> None:
        header["field"] += data[start:end]

    def on_header_value(data: bytes, start: int, end: int) -> None:
        header["value"] += data[start:end]

    def on_header_end() -> None:
        part["headers"][header["field"].strip().lower()] = header["value"].strip()
        header["field"], header["value"] = b"", b""

    def on_headers_finished() -> None:
        _, disposition = parse_options_header(part["headers"].get(b"content-disposition", b""))
        part["name"] = disposition.get(b"name", b"").decode("utf-8", "replace")
        filename = disposition.get(b"filename")
        part["filename"] = filename.decode("utf-8", "replace") if filename is not None else None

    def on_part_data(data: bytes, start: int, end: int) -> None:
        if part["name"] == file_field and part["filename"] is not None:
            sink.write(data[start:end])
            return
        part["text"] += data[start:end]
        if len(part["text"]) > FIELD_MAX_BYTES:
            raise ValueError(f"form field {part['name']} is over {FIELD_MAX_BYTES} bytes")

    def on_part_end() -> None:
        if part["name"] == file_field and part["filename"] is not None:
            found.append(part["filename"])
        elif part["name"]:
            fields[part["name"]] = part["text"].decode("utf-8", "replace")

    parser = MultipartParser(
        boundary,
        {
            "on_part_begin": on_part_begin,
            "on_header_field": on_header_field,
            "on_header_value": on_header_value,
            "on_header_end": on_header_end,
            "on_headers_finished": on_headers_finished,
            "on_part_data": on_part_data,
            "on_part_end": on_part_end,
        },
    )
    received = 0
    try:
        async for block in request.stream():
            received += len(block)
            if received > max_bytes:
                raise HTTPException(413, too_big)
            parser.write(block)
        parser.finalize()
    except ValueError as exc:
        raise HTTPException(400, f"could not read the multipart body: {exc}") from exc
    if not found:
        raise HTTPException(400, f"no '{file_field}' file field in the form")
    sink.close()
    return fields, found[0]


def create_app(orchestrator: Orchestrator, *, background: bool = True) -> FastAPI:
    """The one factory. Tests pass an orchestrator with fakes and background=False."""
    orch = orchestrator
    th = orch.th
    dist = paths.web_dist_dir()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        tasks: list[asyncio.Task[Any]] = []
        if background:
            orch.seed_samples()
            await orch.resume_all()
            tasks.append(asyncio.create_task(_loop(orch.health_probe, float(th["orchestrator"]["health_probe_interval_s"]))))
            tasks.append(asyncio.create_task(_loop(orch.sweep, float(th["orchestrator"]["sweeper_interval_s"]))))
        try:
            yield
        finally:
            for task in tasks:
                task.cancel()
            orch.close()

    app = FastAPI(title="Walkaround", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.orchestrator = orch
    app.add_middleware(BodyLimit, limit=int(th["media"]["other_route_max_body_bytes"]), exempt=lambda path: bool(EXEMPT_PATHS.match(path)))
    max_chunks = -(-int(th["media"]["upload_max_bytes"]) // int(th["media"]["chunk_bytes"]))

    def run_or_404(run_id: str) -> c.RunManifest:
        if not SAFE_ID.match(run_id) or not orch.run_exists(run_id):
            raise HTTPException(404, "no such run")
        return orch.load_run(run_id)

    def may_serve(run: c.RunManifest, rel_path: str, request: Request) -> bool:
        """The app's floor, then privacy.served_file_allowed for this viewer; without that rule nothing is served."""
        if not servable(rel_path) or not orch.has_dep("privacy.served_file_allowed"):
            return False
        owner = request.query_params.get("view") != "kiosk" and not orch.is_public(run)
        return bool(orch.dep("privacy.served_file_allowed")(run, rel_path, run_owner=owner))

    claim_locks: dict[str, asyncio.Lock] = {}  # one claim at a time per run: claim_<n> numbering and its files

    def refused(exc: CapsExceeded) -> JSONResponse:
        return JSONResponse({"reason": exc.reason, "pickup_left": exc.pickup_left, "pairs_left": exc.pairs_left}, status_code=429)

    async def start_admitted(run: c.RunManifest) -> JSONResponse | None:
        """Start a run whose uploads are in, after admitting it again (its reservation may have lapsed). A refused run
        keeps none of its unblurred bytes; the 429 carries the same body POST /api/runs answers."""
        try:
            orch.admit(run)
        except CapsExceeded as exc:
            await orch.discard_uploads(run.run_id, reason=exc.reason)
            return refused(exc)
        orch.start(run.run_id)
        return None

    # Uploads ---------------------------------------------------------------------------------------------------------

    @app.put("/api/uploads/{upload_id}/chunks/{n}", status_code=204)
    async def put_chunk(upload_id: str, n: int, request: Request, run_id: str | None = None) -> Response:
        if not SAFE_ID.match(upload_id) or not (0 <= n < max_chunks):
            raise HTTPException(400, "bad upload id or chunk index")
        if run_id is not None:
            run_or_404(run_id)
        limit = int(th["media"]["chunk_route_max_body_bytes"])
        declared = request.headers.get("content-length")
        if declared and (not declared.isdigit() or int(declared) > limit):
            raise HTTPException(413, f"chunk over {limit} bytes")
        data = bytearray()
        async for part in request.stream():
            data += part
            if len(data) > limit:
                raise HTTPException(413, f"chunk over {limit} bytes")
        target = orch.chunk_path(upload_id, n, run_id)
        tmp = target.with_name(f"{target.name}.{secrets.token_hex(4)}.part")
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
        except UploadRefused as exc:
            return JSONResponse({"detail": str(exc)}, status_code=400)
        run = orch.load_run(body.run_id)
        if body.keep_plate is not None and body.keep_plate != run.keep_plate and run.status == c.RunStatus.CREATED:
            run = orch.update_run(run.run_id, lambda r: setattr(r, "keep_plate", body.keep_plate))
        if run.status == c.RunStatus.CREATED and orch.uploads_complete(run):
            refusal = await start_admitted(run)
            if refusal is not None:
                return refusal
        return canonical(record)

    # Runs ------------------------------------------------------------------------------------------------------------

    @app.post("/api/runs", status_code=201)
    async def create_run(body: RunCreate) -> Response:
        if body.mode != c.RunMode.CAR:
            raise HTTPException(400, "only car mode is built")
        if body.kind not in CREATABLE_KINDS:
            raise HTTPException(400, "a run is a pair or a pickup_check; top-ups and claims attach to an existing run")
        try:
            orch.check_can_start(body.kind)
            run = await orch.create_run(mode=body.mode, kind=body.kind, keep_plate=body.keep_plate)
        except CapsExceeded as exc:
            return refused(exc)
        except NoSpace as exc:
            return JSONResponse({"reason": str(exc)}, status_code=507)
        return canonical(run, 201)

    @app.get("/api/runs")
    async def run_index() -> Response:
        """The public runs, newest first, without the frozen threshold dict; a renter's run id is a receipt, never listed."""
        return canonical([run.model_copy(update={"thresholds": {}}) for run in orch.list_runs() if orch.is_public(run)])

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str) -> Response:
        return canonical(run_or_404(run_id))

    @app.post("/api/runs/{run_id}/start", status_code=202)
    async def start_run(run_id: str) -> Response:
        """Start (or resume) the pipeline once every expected upload is in quarantine; idempotent."""
        run = run_or_404(run_id)
        if run.status in TERMINAL_STATUSES or run.live_or_replay != c.LiveOrReplay.LIVE:
            return canonical(run)
        if not orch.uploads_complete(run):
            missing = [r for r in orch.expected_roles(run) if not paths.quarantine_original(run_id, r).is_file()]
            raise HTTPException(409, f"upload {', '.join(missing)} first")
        refusal = await start_admitted(run)
        if refusal is not None:
            return refusal
        return canonical(run, 202)

    @app.get("/api/runs/{run_id}/events", response_model=None)
    async def run_events(run_id: str, request: Request) -> Response:
        """SSE: id = seq, event = kind, data = the Event JSON; Last-Event-ID resumes at seq + 1; a ping comment every
        sse_ping_s. An ended run with nothing left to send answers 204, which stops the browser reconnecting. A run that
        failed before its chain opened sends its waiting events (orchestrator.visible_events), run.failed last."""
        run = run_or_404(run_id)
        last = request.headers.get("last-event-id", "").strip()
        since = int(last) if last.lstrip("-").isdigit() else -1
        if run.status in TERMINAL_STATUSES and not orch.visible_events(run_id, since_seq=since):
            return Response(status_code=204)
        return EventSourceResponse(
            _event_stream(orch, run_id, since),
            ping=int(th["orchestrator"]["sse_ping_s"]),
            ping_message_factory=lambda: ServerSentEvent(comment="ping"),
        )

    @app.get("/api/runs/{run_id}/replay")
    async def run_replay(run_id: str, mode: str = "fixed") -> Response:
        run_or_404(run_id)
        try:
            replay_mode = c.ReplayMode(mode)
        except ValueError:
            raise HTTPException(400, "mode is fixed, recorded or instant")
        return canonical(orch.replay_payload(run_id, mode=replay_mode))

    # Sheet and files -------------------------------------------------------------------------------------------------

    def sheet_file(run_id: str, name: str, media_type: str, missing: str, request: Request) -> FileResponse:
        """The sheet files under the files route's rule: with keep_plate the page inlines the owner's frames with the
        plate kept, so a public run or the kiosk view gets a 404."""
        run = run_or_404(run_id)
        if not may_serve(run, f"sheet/{name}", request):
            raise HTTPException(404, "no such file")
        path = paths.runs_root() / run_id / "sheet" / name
        if not path.is_file():
            raise HTTPException(404, missing)
        return FileResponse(path, media_type=media_type)

    @app.api_route("/api/runs/{run_id}/sheet", methods=["GET", "HEAD"])
    async def sheet_html(run_id: str, request: Request) -> FileResponse:
        return sheet_file(run_id, "sheet.html", "text/html", "sheet not written yet", request)

    @app.api_route("/api/runs/{run_id}/sheet.json", methods=["GET", "HEAD"])
    async def sheet_json(run_id: str, request: Request) -> FileResponse:
        return sheet_file(run_id, "sheet.json", "application/json", "sheet not written yet", request)

    @app.api_route("/api/runs/{run_id}/sheet.pdf", methods=["GET", "HEAD"])
    async def sheet_pdf(run_id: str, request: Request) -> FileResponse:
        return sheet_file(run_id, "evidence.pdf", "application/pdf", "PDF retry", request)

    @app.get("/api/runs/{run_id}/bundle.zip")
    async def bundle_zip(run_id: str, request: Request) -> Response:
        """The zip of bundle/, built in memory: a zip written into the run would change the run's own manifest."""
        run = run_or_404(run_id)
        bundle = paths.runs_root() / run_id / "bundle"
        files = sorted(p for p in bundle.rglob("*") if p.is_file() and may_serve(run, f"bundle/{p.relative_to(bundle).as_posix()}", request)) if bundle.is_dir() else []
        if not files:
            raise HTTPException(404, "bundle not written yet")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in files:
                zf.write(path, arcname=path.relative_to(bundle).as_posix())
        headers = {"Content-Disposition": f'attachment; filename="walkaround-{run_id}-bundle.zip"'}
        return Response(content=buffer.getvalue(), media_type="application/zip", headers=headers)

    @app.api_route("/api/runs/{run_id}/files/{rel_path:path}", methods=["GET", "HEAD"])
    async def run_file(run_id: str, rel_path: str, request: Request) -> FileResponse:
        run = run_or_404(run_id)
        root = (paths.runs_root() / run_id).resolve()
        target = (root / rel_path).resolve()
        if not may_serve(run, rel_path, request) or root not in target.parents or not target.is_file():
            raise HTTPException(404, "no such file")
        return FileResponse(target)

    # Top-up and claims -----------------------------------------------------------------------------------------------

    @app.post("/api/runs/{run_id}/topup")
    async def topup(run_id: str, request: Request) -> Response:
        """The missing panels on a new clip (multipart field 'video', optional 'sha256_browser'): the clip is
        streamed into upload chunks, finalized into quarantine like a walk, then only nouns_to_rerun run on it. It
        spends model calls, so the pause, the card ceiling and the run's ceiling apply before any byte is read, and a
        public run (its id is listed) takes none."""
        run = run_or_404(run_id)
        if run.live_or_replay != c.LiveOrReplay.LIVE:
            raise HTTPException(409, "a replayed run takes no top-up")
        if orch.is_public(run):
            raise HTTPException(403, "the sample pair and kept runs take no top-up")
        if not (paths.runs_root() / run_id / "derived" / "coverage.json").is_file():
            raise HTTPException(409, "coverage is not derived yet; the top-up waits for the first pass")
        try:
            orch.check_can_topup(run)
        except CapsExceeded as exc:
            return refused(exc)
        role = orch.reserve_role(run_id, "topup")
        upload_id = f"{role}-{secrets.token_hex(4)}"
        try:
            sink = _ChunkSink(orch, upload_id, run_id)
            fields, filename = await read_multipart(request, file_field="video", sink=sink, max_bytes=int(th["media"]["upload_max_bytes"]))
            browser = fields.get("sha256_browser", "").strip().lower() or None
            if browser is not None and not re.fullmatch(r"[0-9a-f]{64}", browser):
                raise HTTPException(400, "sha256_browser is 64 hex characters")
            await orch.finalize_upload(run_id, upload_id=upload_id, role=role, sha256_browser=browser, bytes=sink.total, filename=filename or "topup.mp4")
        except HashMismatch as exc:
            (paths.runs_root() / run_id / "videos" / f"{role}.json").unlink(missing_ok=True)  # a refused clip is not an original
            return JSONResponse({"sha256_browser": exc.sha256_browser, "sha256_server": exc.sha256_server, "detail": "hash mismatch; upload again"}, status_code=409)
        except UploadRefused as exc:
            return JSONResponse({"detail": str(exc)}, status_code=400)
        finally:
            with suppress(Exception):
                for chunk in (paths.runs_root() / run_id / "uploads" / upload_id).rglob("*"):
                    if chunk.is_file():
                        chunk.unlink()  # a refused or broken body leaves no unblurred bytes behind
            orch.release_role(run_id, role)
        try:
            coverage = await orch.run_topup(run_id, role)
        except VideoUnreadable as exc:
            return JSONResponse({"detail": str(exc)}, status_code=400)
        return canonical(coverage)

    @app.post("/api/claims")
    async def claims(request: Request) -> Response:
        """One claim photo against a finished run (multipart 'image', 'run_id', 'text', optional 'tire_code' and
        'panel_id', the renter's one tap after E2 or E2b). It spends model calls (about $0.05), so the pause and the
        card ceiling apply before the body is read. The photo is held in memory under the 50 MB cap and written to the
        run's quarantine, which is never served; claim.derive_claim deletes it once the pixelated copy exists. Claims
        on one run go one at a time, numbered after the highest claim_<n> on disk."""
        if not orch.has_dep("claim.derive_claim"):
            raise HTTPException(501, "claim-first is not wired yet")
        length = request.headers.get("content-length", "")
        if length.isdigit() and int(length) > CLAIM_MAX_BYTES:
            raise HTTPException(413, "the claim photo is over 50 MB")
        if orch.paused or orch.card_ceiling_reached(et_day_of(orch.now())):
            left = orch.caps_remaining(et_day=et_day_of(orch.now()))
            reason = "live runs paused; the sample pair is precomputed" if orch.paused else "live runs paused for today; the sample pair is precomputed"
            return refused(CapsExceeded(reason, pickup_left=left["pickup"], pairs_left=left["pair"]))
        sink = _MemorySink()
        fields, _filename = await read_multipart(request, file_field="image", sink=sink, max_bytes=CLAIM_MAX_BYTES, too_big="the claim photo is over 50 MB")
        run = run_or_404(fields.get("run_id", "").strip())
        run_id = run.run_id
        if orch.is_public(run):
            raise HTTPException(403, "the sample pair and kept runs take no claim")
        if run.status not in TERMINAL_STATUSES:
            raise HTTPException(409, "the run is still working; send the claim once it has finished")
        tire_code = fields.get("tire_code", "").strip() or None
        panel_id = fields.get("panel_id", "").strip() or None
        async with claim_locks.setdefault(run_id, asyncio.Lock()):
            claim_id = f"claim_{next_claim_number(run_id)}"
            target = paths.quarantine_temp(run_id, f"{claim_id}.jpg")
            target.write_bytes(bytes(sink.data))
            try:
                claim = await orch.dep("claim.derive_claim")(run_id, image=target, text=fields.get("text", ""), tire_code=tire_code, th=orch.run_th(run), panel_id=panel_id, claim_id=claim_id)
            except ValueError as exc:  # claim.ClaimRefused: a photo, panel id or run the claim cannot use
                text = redact(str(exc))
                status = 409 if "coverage" in text else 503 if "did not complete" in text else 400
                return JSONResponse({"detail": text}, status_code=status)
            finally:
                target.unlink(missing_ok=True)  # derive_claim removes it; a failure before that must not leave it
        return canonical(claim)

    # Status ----------------------------------------------------------------------------------------------------------

    @app.get("/api/status")
    async def status(run_id: str | None = None) -> dict[str, Any]:
        left = orch.caps_remaining(et_day=et_day_of(orch.now()))
        position = orch.queue_position(run_id) if run_id else None
        return {
            "paused": orch.paused,
            "pickup_left": left["pickup"],
            "pairs_left": left["pair"],
            "queue_position": position,
            "queue_eta_minutes": orch.queue_eta_minutes(position),
            "engine": orch.engine.value,
        }

    @app.get("/api/sample")
    async def sample() -> dict[str, Any]:
        """The precomputed demo pair: never a live run, always flagged as such."""
        run_id = orch.sample_run_id()
        if not run_id or not SAFE_ID.match(run_id) or not orch.run_exists(run_id):
            raise HTTPException(404, "no sample pair on this server")
        recorded_on = orch.recorded_on(orch.load_run(run_id))
        return {"run_id": run_id, "precomputed": True, "recorded_on": recorded_on, "badge": f"replayed from a run on {recorded_on}", "replay": f"/api/runs/{run_id}/replay?mode=fixed"}

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"ok": True, "engine": orch.engine.value, "probe_ok": orch.probe_ok, "free_bytes": orch.free_bytes()}

    # The web build ---------------------------------------------------------------------------------------------------

    index = dist / "index.html"
    if (dist / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    async def spa(request: Request) -> HTMLResponse:
        if index.is_file():
            return HTMLResponse(index.read_text(encoding="utf-8"))
        return HTMLResponse(FALLBACK_HOME)

    label_page = dist / "eval" / "label.html"

    async def eval_label_page() -> FileResponse:
        """The eval labeling page (web/public/eval/label.html), a static file outside the SPA; it reads only run files
        the allowlist already serves and saves nothing."""
        if not label_page.is_file():
            raise HTTPException(404, "the labeling page is not in this build; run npm run build in web/")
        return FileResponse(label_page, media_type="text/html")

    app.add_api_route("/eval/label.html", eval_label_page, methods=["GET", "HEAD"], include_in_schema=False)
    for route in SPA_ROUTES:
        app.add_api_route(route, spa, methods=["GET", "HEAD"], include_in_schema=False)

    return app


async def _event_stream(orch: Orchestrator, run_id: str, since: int) -> AsyncIterator[ServerSentEvent]:
    """History after since, then live events until a terminal one. A pump task feeds a queue so the wait can time
    out safely: once the run's status is terminal (set in the same synchronous step as its terminal event), whatever
    is left is sent and the stream closes. A run that ended before its chain opened sends its waiting events at once."""
    if not orch.chain_open(run_id) and orch.load_run(run_id).status in TERMINAL_STATUSES:
        for event in orch.visible_events(run_id, since_seq=since):
            yield _sse(event)
        return
    queue: asyncio.Queue[c.Event | None] = asyncio.Queue()

    async def pump() -> None:
        try:
            async for event in orch.events.subscribe(run_id, since_seq=since):
                await queue.put(event)
        finally:
            await queue.put(None)

    task = asyncio.create_task(pump())
    last = since
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=SSE_POLL_S)
            except TimeoutError:
                if orch.load_run(run_id).status not in TERMINAL_STATUSES:
                    continue
                for event in orch.visible_events(run_id, since_seq=last):
                    yield _sse(event)
                return
            if event is None:
                return
            if event.seq <= last:
                continue
            last = event.seq
            yield _sse(event)
            if event.kind in TERMINAL_EVENTS:
                return
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await task


def _sse(event: c.Event) -> ServerSentEvent:
    return ServerSentEvent(data=event.to_json(), event=event.kind.value, id=str(event.seq))


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
    """The app uvicorn serves (uvicorn walkaround.app:default_app --factory): Orchestrator.from_env()."""
    return create_app(Orchestrator.from_env())


__all__ = ["BodyLimit", "FinalizeBody", "RunCreate", "canonical", "create_app", "default_app", "read_multipart", "servable"]
