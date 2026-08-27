"""Mutable state for one run.

Plain objects rather than Pydantic models. This is hot, mutated on every
scheduling turn, and Pydantic's validate-on-assignment would run a validator on
each status change for no benefit — the values come from the runtime itself, not
from a user.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from skein.model.workflow import Workflow
from skein.trace.events import StepStatus


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}


@dataclass
class StepState:
    id: str
    status: StepStatus = StepStatus.PENDING
    attempts: int = 0
    output: Any = None
    error_code: str | None = None
    error_message: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    skipped_reason: str | None = None

    @property
    def duration_s(self) -> float | None:
        if self.started_at is None or self.finished_at is None:
            return None
        return round(self.finished_at - self.started_at, 6)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status.value,
            "attempts": self.attempts,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "duration_s": self.duration_s,
            "skipped_reason": self.skipped_reason,
        }


@dataclass
class RunState:
    workflow: Workflow
    inputs: dict[str, Any] = field(default_factory=dict)
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: RunStatus = RunStatus.QUEUED
    steps: dict[str, StepState] = field(default_factory=dict)
    error_code: str | None = None
    error_message: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    trajectory_hash: str | None = None

    def __post_init__(self) -> None:
        if not self.steps:
            self.steps = {step.id: StepState(id=step.id) for step in self.workflow.steps}

    # -- queries used by the scheduler ------------------------------------

    def step(self, step_id: str) -> StepState:
        return self.steps[step_id]

    def all_terminal(self) -> bool:
        return all(state.status.is_terminal for state in self.steps.values())

    def outputs(self) -> dict[str, Any]:
        """Outputs of steps that actually succeeded.

        Deliberately excludes failed and skipped steps rather than including
        them as None. A dependent binding to a failed step's output should fail
        to resolve loudly, not receive a null that looks like a legitimate
        empty result.
        """
        return {
            step_id: state.output
            for step_id, state in self.steps.items()
            if state.status is StepStatus.SUCCEEDED
        }

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {status.value: 0 for status in StepStatus}
        for state in self.steps.values():
            counts[state.status.value] += 1
        return counts

    @property
    def duration_s(self) -> float | None:
        if self.started_at is None or self.finished_at is None:
            return None
        return round(self.finished_at - self.started_at, 6)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "workflow": self.workflow.name,
            "version": self.workflow.version,
            "status": self.status.value,
            "steps": [state.to_dict() for state in self.steps.values()],
            "counts": self.counts(),
            "error_code": self.error_code,
            "error_message": self.error_message,
            "duration_s": self.duration_s,
            "trajectory_hash": self.trajectory_hash,
        }
