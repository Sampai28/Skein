"""The FastAPI application.

Fully async: every handler is ``async def`` and nothing in a request path
blocks. That is not stylistic — a single synchronous call of any duration in a
handler blocks the event loop and therefore every other in-flight request and
every running workflow step in the process.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request, Response, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from skein import __version__
from skein.api.models import (
    GraphView,
    HealthView,
    RunView,
    SubmitRequest,
    SubmitResponse,
    TraceView,
)
from skein.api.problems import install_error_handlers
from skein.api.ws import stream_run
from skein.config import Settings, load_settings
from skein.errors import SkeinError
from skein.model.workflow import StepKind
from skein.observability import metrics
from skein.observability.otel import setup_tracing, shutdown_tracing
from skein.runtime.engine import Engine, EngineConfig

logger = logging.getLogger("skein.api")

WEB_DIR = Path(__file__).resolve().parents[3] / "web"


def create_app(settings: Settings | None = None, engine: Engine | None = None) -> FastAPI:
    resolved = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        setup_tracing(endpoint=resolved.otlp_endpoint, enabled=resolved.tracing_enabled)
        app.state.engine = engine or Engine(EngineConfig(settings=resolved))
        await app.state.engine.start()
        try:
            yield
        finally:
            # Ordering matters: drain the engine before tearing down tracing, so
            # the spans produced during shutdown are exported rather than lost
            # with the provider.
            await app.state.engine.stop()
            shutdown_tracing()

    app = FastAPI(
        title="Skein",
        version=__version__,
        description="An asyncio runtime for executing agent workflows defined as DAGs.",
        lifespan=lifespan,
    )
    install_error_handlers(app)

    def get_engine(request: Request) -> Engine:
        return request.app.state.engine  # type: ignore[no-any-return]

    # -- workflows ---------------------------------------------------------

    @app.post("/workflows", response_model=SubmitResponse, status_code=202)
    async def submit_workflow(request: Request, body: SubmitRequest) -> SubmitResponse:
        engine = get_engine(request)
        if engine.shutdown.is_draining():
            raise SkeinError("instance is draining and not accepting new work")

        replay_path = Path(body.replay_from) if body.replay_from else None
        run = await engine.submit(body.workflow, body.inputs, replay_from=replay_path)
        return SubmitResponse(
            run_id=run.run_id,
            status=run.status.value,
            workflow=run.workflow.name,
            queued_behind=max(0, engine.queue.qsize() - 1),
        )

    # -- runs --------------------------------------------------------------

    @app.get("/runs/{run_id}", response_model=RunView)
    async def get_run(request: Request, run_id: str) -> RunView:
        handle = get_engine(request).store.get(run_id)
        return RunView.model_validate(handle.state.to_dict())

    @app.get("/runs/{run_id}/trace", response_model=TraceView)
    async def get_trace(request: Request, run_id: str) -> TraceView:
        handle = get_engine(request).store.get(run_id)
        return TraceView(
            run_id=run_id,
            trajectory_hash=handle.recorder.trajectory_hash(),
            event_count=len(handle.recorder.events),
            events=[event.model_dump(exclude_none=True) for event in handle.recorder.events],
        )

    @app.get("/runs/{run_id}/graph", response_model=GraphView)
    async def get_graph(request: Request, run_id: str) -> GraphView:
        handle = get_engine(request).store.get(run_id)
        workflow = handle.state.workflow
        nodes = [
            {
                "id": step.id,
                "kind": step.kind.value,
                "target": step.tool or step.model,
                "status": handle.state.step(step.id).status.value,
            }
            for step in workflow.steps
        ]
        edges = [
            {"source": dependency, "target": step.id}
            for step in workflow.steps
            for dependency in step.depends_on
        ]
        # Branch edges are drawn too, dashed in the viewer, because a branch's
        # control flow is not visible from depends_on alone.
        for step in workflow.steps:
            if step.kind is StepKind.BRANCH:
                for target in (*step.on_true, *step.on_false):
                    edges.append({"source": step.id, "target": target, "kind": "branch"})
        return GraphView(
            run_id=run_id, workflow=workflow.name, nodes=nodes, edges=edges
        )

    @app.delete("/runs/{run_id}", response_model=RunView)
    async def cancel_run(request: Request, run_id: str) -> RunView:
        engine = get_engine(request)
        state = await engine.store.cancel(run_id, timeout_s=engine.settings.cancel_timeout_s)
        return RunView.model_validate(state.to_dict())

    @app.websocket("/runs/{run_id}/stream")
    async def stream(websocket: WebSocket, run_id: str) -> None:
        engine: Engine = websocket.app.state.engine
        try:
            handle = engine.store.get(run_id)
        except SkeinError:
            # A WebSocket cannot return a problem document — the handshake has
            # not completed. 4404 is in the application-defined close-code range
            # and mirrors the HTTP status.
            await websocket.close(code=4404, reason="run not found")
            return
        await stream_run(websocket, handle)

    # -- introspection -----------------------------------------------------

    @app.get("/tools")
    async def list_tools(request: Request) -> list[dict[str, Any]]:
        return get_engine(request).registry.describe()

    @app.get("/metrics")
    async def prometheus_metrics() -> Response:
        return Response(content=metrics.render(), media_type="text/plain; version=0.0.4")

    @app.get("/healthz", response_model=HealthView)
    async def healthz(request: Request) -> HealthView:
        # Liveness: is the process functioning. Deliberately does NOT check
        # Ollama — a dead dependency would restart a healthy pod, and a restart
        # loop caused by someone else's outage is worse than degraded service.
        return HealthView(
            status="ok", version=__version__, detail=get_engine(request).health()
        )

    @app.get("/readyz")
    async def readyz(request: Request) -> JSONResponse:
        engine = get_engine(request)
        ready = await engine.ready()
        return JSONResponse(
            status_code=200 if ready else 503,
            content={
                "ready": ready,
                "draining": engine.shutdown.is_draining(),
                "queue_depth": engine.queue.qsize(),
                "queue_capacity": engine.queue.maxsize,
            },
        )

    # -- static viewer -----------------------------------------------------

    if WEB_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

        @app.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(str(WEB_DIR / "index.html"))

    return app


app = create_app()
