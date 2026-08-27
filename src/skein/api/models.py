"""Request and response bodies for the HTTP surface."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from skein.model.workflow import Workflow


class SubmitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workflow: Workflow
    inputs: dict[str, Any] = Field(default_factory=dict)
    #: Path to a recorded trace. Present means replay: tool and LLM calls are
    #: served from the recording instead of the world.
    replay_from: str | None = None


class SubmitResponse(BaseModel):
    run_id: str
    status: str
    workflow: str
    queued_behind: int


class StepView(BaseModel):
    id: str
    status: str
    attempts: int
    duration_s: float | None = None
    error_code: str | None = None
    error_message: str | None = None
    skipped_reason: str | None = None


class RunView(BaseModel):
    run_id: str
    workflow: str
    version: str
    status: str
    steps: list[StepView]
    counts: dict[str, int]
    duration_s: float | None = None
    error_code: str | None = None
    error_message: str | None = None
    trajectory_hash: str | None = None


class TraceView(BaseModel):
    run_id: str
    trajectory_hash: str
    event_count: int
    events: list[dict[str, Any]]


class GraphView(BaseModel):
    """The DAG shape, for the viewer.

    Served separately from run state so the page can draw the graph once and
    then apply status updates from the WebSocket, rather than re-laying out on
    every event.
    """

    run_id: str
    workflow: str
    nodes: list[dict[str, Any]]
    edges: list[dict[str, str]]


class HealthView(BaseModel):
    status: str
    version: str
    detail: dict[str, Any] = Field(default_factory=dict)
