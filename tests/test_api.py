"""The HTTP and WebSocket surface.

Uses FastAPI's TestClient, which runs the app's lifespan — so the engine really
starts, workers really run, and a submitted workflow really executes. That is
the point: mocking the engine here would test the routing and nothing else.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi.testclient import TestClient

from skein.api.app import create_app
from skein.runtime.engine import Engine, EngineConfig
from tests.conftest import make_step, make_workflow


@pytest.fixture
def client(settings, registry):
    async def quick(**kwargs: Any) -> dict[str, Any]:
        await asyncio.sleep(0)
        return {"ok": True, "echo": kwargs.get("label")}

    async def slow(**kwargs: Any) -> dict[str, Any]:
        await asyncio.sleep(5.0)
        return {"ok": True}

    async def boom(**kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("injected")

    registry.register("quick", quick)
    registry.register("slow", slow)
    registry.register("boom", boom)

    engine = Engine(EngineConfig(settings=settings, registry=registry))
    app = create_app(settings=settings, engine=engine)
    # The context manager form is what triggers startup and shutdown; without it
    # app.state.engine is never set and every request 500s.
    with TestClient(app) as test_client:
        yield test_client


def _payload(steps=None, **kwargs):
    workflow = make_workflow(steps or [make_step("a", tool="quick")], **kwargs)
    return {"workflow": workflow.model_dump(mode="json"), "inputs": {}}


def _wait_for_terminal(client: TestClient, run_id: str, tries: int = 200) -> dict:
    for _ in range(tries):
        body = client.get(f"/runs/{run_id}").json()
        if body["status"] in {"succeeded", "failed", "cancelled"}:
            return body
        import time

        time.sleep(0.01)
    raise AssertionError(f"run {run_id} did not finish")


def test_healthz_reports_ok(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readyz_reports_ready(client):
    response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json()["ready"] is True


def test_metrics_are_exposed_in_prometheus_format(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "skein_runs_started_total" in response.text


def test_tools_are_listed_with_schemas(client):
    response = client.get("/tools")
    assert response.status_code == 200
    assert {tool["name"] for tool in response.json()} >= {"quick", "slow", "boom"}


def test_submitting_a_workflow_returns_202_and_a_run_id(client):
    response = client.post("/workflows", json=_payload())
    assert response.status_code == 202
    body = response.json()
    assert body["run_id"]
    assert body["status"] == "queued"


def test_a_submitted_workflow_actually_runs(client):
    run_id = client.post("/workflows", json=_payload()).json()["run_id"]
    body = _wait_for_terminal(client, run_id)

    assert body["status"] == "succeeded"
    assert body["counts"]["succeeded"] == 1
    assert body["trajectory_hash"]


def test_an_invalid_workflow_is_rejected_as_a_problem_document(client):
    payload = _payload(
        [
            make_step("a", tool="quick"),
            make_step("b", tool="quick", depends_on=["missing"]),
        ]
    )
    response = client.post("/workflows", json=payload)

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/problem+json")
    body = response.json()
    assert body["code"] == "unknown_step_reference"
    assert body["instance"] == "/workflows"


def test_a_malformed_body_is_a_problem_document_too(client):
    response = client.post("/workflows", json={"workflow": {"nope": True}})
    assert response.status_code == 422
    assert response.json()["code"] == "request_validation"


def test_an_unknown_run_is_404_with_a_code(client):
    response = client.get("/runs/does-not-exist")
    assert response.status_code == 404
    assert response.json()["code"] == "run_not_found"


def test_the_trace_endpoint_returns_events_and_a_hash(client):
    run_id = client.post("/workflows", json=_payload()).json()["run_id"]
    _wait_for_terminal(client, run_id)

    body = client.get(f"/runs/{run_id}/trace").json()
    assert body["event_count"] > 0
    assert body["trajectory_hash"]
    kinds = {event["kind"] for event in body["events"]}
    assert {"run_started", "step_started", "step_succeeded", "run_finished"} <= kinds


def test_the_graph_endpoint_describes_nodes_and_edges(client):
    payload = _payload(
        [
            make_step("a", tool="quick"),
            make_step("b", tool="quick", depends_on=["a"]),
        ]
    )
    run_id = client.post("/workflows", json=payload).json()["run_id"]
    body = client.get(f"/runs/{run_id}/graph").json()

    assert {node["id"] for node in body["nodes"]} == {"a", "b"}
    assert {"source": "a", "target": "b"} in body["edges"]


def test_a_failing_step_is_reported_with_its_error_code(client):
    payload = _payload([make_step("a", tool="boom")])
    run_id = client.post("/workflows", json=payload).json()["run_id"]
    body = _wait_for_terminal(client, run_id)

    assert body["status"] == "failed"
    assert body["steps"][0]["status"] == "failed"
    assert body["steps"][0]["error_code"]


def test_cancelling_a_running_workflow(client):
    payload = _payload([make_step("a", tool="slow", timeout_s=30)])
    run_id = client.post("/workflows", json=payload).json()["run_id"]

    import time

    # Wait for it to actually start, otherwise the cancel races the worker.
    for _ in range(200):
        if client.get(f"/runs/{run_id}").json()["status"] == "running":
            break
        time.sleep(0.01)

    response = client.delete(f"/runs/{run_id}")
    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"


def test_the_websocket_streams_events(client):
    payload = _payload(
        [
            make_step("a", tool="quick"),
            make_step("b", tool="quick", depends_on=["a"]),
        ]
    )
    run_id = client.post("/workflows", json=payload).json()["run_id"]

    kinds: list[str] = []
    with client.websocket_connect(f"/runs/{run_id}/stream") as websocket:
        # The stream replays history first, so a client connecting after the run
        # finished still receives every event rather than an empty stream.
        for _ in range(60):
            message = websocket.receive_json()
            kinds.append(message.get("kind", ""))
            if message.get("kind") == "stream_end":
                break

    assert "stream_end" in kinds
    assert "step_started" in kinds or "run_started" in kinds


def test_the_websocket_closes_with_4404_for_an_unknown_run(client):
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect) as excinfo:
        with client.websocket_connect("/runs/nope/stream") as websocket:
            websocket.receive_json()
    assert excinfo.value.code == 4404


def test_the_dag_viewer_is_served(client):
    response = client.get("/")
    # The page is optional (it is mounted only if web/ exists next to the
    # package), so a 404 here is a packaging problem rather than an app bug.
    assert response.status_code in (200, 404)
    if response.status_code == 200:
        assert "Skein" in response.text
