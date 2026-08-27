"""The trace event vocabulary.

One JSON object per line, append-only. The format is deliberately flat and
self-describing: a trace has to be readable with ``jq`` on a machine that does
not have Skein installed, which rules out anything pickled or schema-registry
shaped.

Two invariants the rest of the system relies on:

* Every executed step emits exactly one ``step_started`` and exactly one
  terminal event (``succeeded``, ``failed``, ``cancelled`` or ``skipped``).
  Tests assert this as *trace completeness*, and it is the check that catches a
  step whose cleanup path forgot to record its own failure.
* Response events (``tool_response``, ``llm_response``) carry the attempt
  number. Without it a replay cannot distinguish the first attempt's response
  from the retry's, and a recorded run with retries would replay incorrectly.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"

    @property
    def is_terminal(self) -> bool:
        return self in {
            StepStatus.SUCCEEDED,
            StepStatus.FAILED,
            StepStatus.CANCELLED,
            StepStatus.SKIPPED,
        }


class EventKind(str, Enum):
    RUN_STARTED = "run_started"
    RUN_FINISHED = "run_finished"
    STEP_STARTED = "step_started"
    STEP_SUCCEEDED = "step_succeeded"
    STEP_FAILED = "step_failed"
    STEP_CANCELLED = "step_cancelled"
    STEP_SKIPPED = "step_skipped"
    STEP_RETRY = "step_retry"
    TOOL_RESPONSE = "tool_response"
    LLM_RESPONSE = "llm_response"
    BREAKER_TRANSITION = "breaker_transition"
    BUDGET_TRIPPED = "budget_tripped"

    @property
    def terminal_status(self) -> StepStatus | None:
        return {
            EventKind.STEP_SUCCEEDED: StepStatus.SUCCEEDED,
            EventKind.STEP_FAILED: StepStatus.FAILED,
            EventKind.STEP_CANCELLED: StepStatus.CANCELLED,
            EventKind.STEP_SKIPPED: StepStatus.SKIPPED,
        }.get(self)


class TraceEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: EventKind
    run_id: str
    #: Wall-clock seconds. Only used for display and ordering within a run; the
    #: sequence number below is what actually establishes order, because two
    #: events on the same event-loop turn can share a timestamp exactly.
    ts: float = Field(default_factory=time.time)
    seq: int = 0

    step_id: str | None = None
    attempt: int | None = None
    status: StepStatus | None = None

    #: Tool or model name, where applicable.
    target: str | None = None

    #: Structured payload — inputs, outputs, error details. Excluded from the
    #: trajectory hash except where explicitly digested, so that a run whose
    #: outputs differ only in a timestamp still compares equal.
    payload: dict[str, Any] = Field(default_factory=dict)

    error_code: str | None = None
    error_message: str | None = None
    duration_s: float | None = None

    def to_json_line(self) -> str:
        # exclude_none keeps the file readable; a trace with forty null fields
        # per line is unpleasant to grep and three times the size.
        return self.model_dump_json(exclude_none=True)

    @classmethod
    def from_json_line(cls, line: str) -> TraceEvent:
        return cls.model_validate_json(line)
