"""Pydantic v2 models describing a workflow as a DAG of steps.

These models cover *shape* — types, ranges, required fields for a step kind.
Graph-level questions (cycles, unknown references, reachability) live in
:mod:`skein.model.validation`, because they need the whole workflow and a
field validator only ever sees one field.

Both run at submission. Nothing here is checked during execution: a workflow
that reached the scheduler has already been proven well-formed, so the
scheduler never has to ask whether a dependency exists.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ``${steps.fetch.output.items}``, ``${inputs.query}``, or — inside a map step's
# inputs only — ``${item.score}`` referring to the element being processed.
# Anything else in a binding position is treated as a literal value.
BINDING_RE = re.compile(
    r"^\$\{("
    r"steps\.[A-Za-z0-9_\-]+\.output(?:\.[A-Za-z0-9_\-]+)*"
    r"|inputs(?:\.[A-Za-z0-9_\-]+)*"
    r"|item(?:\.[A-Za-z0-9_\-]+)*"
    r")\}$"
)
STEP_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_\-]{0,63}$")

MAX_STEPS_HARD_CAP = 512


class StepKind(str, Enum):
    TOOL_CALL = "tool_call"
    LLM_CALL = "llm_call"
    MAP = "map"
    REDUCE = "reduce"
    BRANCH = "branch"


class DependentsPolicy(str, Enum):
    """What happens to the dependents of a step that did not succeed.

    Default is ``SKIP``. Reasoning, since this is the choice people argue about:
    ``FAIL_FAST`` throws away work that was going to succeed and is wrong for an
    agent workflow where one branch failing is normal. ``CONTINUE`` is worse — it
    runs a dependent against a missing input, so the failure resurfaces later as
    a confusing binding error rather than at its actual cause. ``SKIP`` marks the
    subtree skipped, records why, and lets independent branches finish.
    """

    FAIL_FAST = "fail_fast"
    SKIP = "skip"
    CONTINUE = "continue"


class OverflowPolicy(str, Enum):
    """What the ingress queue does when it is full."""

    REJECT = "reject"
    BLOCK = "block"


class RetryPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_attempts: int = Field(default=3, ge=1, le=10)
    initial_backoff_s: float = Field(default=0.1, gt=0, le=60)
    max_backoff_s: float = Field(default=10.0, gt=0, le=600)
    multiplier: float = Field(default=2.0, ge=1.0, le=10.0)
    # Full jitter by default. Deterministic backoff synchronises every retrying
    # caller onto the same instant, so a tool that just recovered is hit by the
    # entire backlog at once.
    jitter: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _backoff_ordering(self) -> RetryPolicy:
        if self.max_backoff_s < self.initial_backoff_s:
            raise ValueError("max_backoff_s must be >= initial_backoff_s")
        return self


class Budget(BaseModel):
    """Runaway guards. Every field is a hard stop, not a warning threshold."""

    model_config = ConfigDict(extra="forbid")

    max_steps: int = Field(default=200, ge=1, le=MAX_STEPS_HARD_CAP)
    max_duration_s: float = Field(default=300.0, gt=0, le=86_400)
    max_tool_calls: int = Field(default=500, ge=1)
    max_tokens: int = Field(default=200_000, ge=1)


class Step(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: StepKind
    depends_on: list[str] = Field(default_factory=list)

    # Binding expressions or literals. Resolved against completed step outputs
    # and the run's inputs immediately before execution.
    inputs: dict[str, Any] = Field(default_factory=dict)

    timeout_s: float | None = Field(default=None, gt=0, le=3600)
    retry: RetryPolicy | None = None

    # tool_call
    tool: str | None = None
    # Name of a schema registered alongside the tool. Its absence means the
    # output is passed through unvalidated, which is allowed but noted.
    output_schema: str | None = None

    # llm_call
    model: str | None = None
    prompt: str | None = None
    response_model: str | None = None

    # map — apply `tool` to every element of `over`
    over: str | None = None
    # Bounds the fan-out of a single map step independently of the global
    # limits. An LLM that returns a 10,000-element list should not be able to
    # queue 10,000 tool calls just because the global cap would eventually
    # throttle them.
    max_fanout: int = Field(default=64, ge=1, le=1024)

    # reduce
    reducer: Literal["concat", "merge", "sum", "first", "last"] | None = None

    # branch
    when: str | None = None
    on_true: list[str] = Field(default_factory=list)
    on_false: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _valid_id(cls, value: str) -> str:
        if not STEP_ID_RE.match(value):
            raise ValueError(
                "step id must start with a letter and contain only letters, "
                "digits, underscore or hyphen (max 64 chars)"
            )
        return value

    @field_validator("depends_on")
    @classmethod
    def _no_self_dependency(cls, value: list[str], info: Any) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("depends_on contains duplicates")
        return value

    @model_validator(mode="after")
    def _required_fields_for_kind(self) -> Step:
        """Each kind needs different fields; enforce that here rather than
        discovering it in the executor with an AttributeError."""
        if self.id in self.depends_on:
            raise ValueError(f"step {self.id!r} depends on itself")

        if self.kind is StepKind.TOOL_CALL and not self.tool:
            raise ValueError(f"step {self.id!r}: tool_call requires 'tool'")

        if self.kind is StepKind.LLM_CALL and not self.prompt:
            raise ValueError(f"step {self.id!r}: llm_call requires 'prompt'")

        if self.kind is StepKind.MAP:
            if not self.over:
                raise ValueError(f"step {self.id!r}: map requires 'over'")
            if not self.tool:
                raise ValueError(f"step {self.id!r}: map requires 'tool' to apply per item")

        if self.kind is StepKind.REDUCE and not self.reducer:
            raise ValueError(f"step {self.id!r}: reduce requires 'reducer'")

        if self.kind is StepKind.BRANCH:
            if not self.when:
                raise ValueError(f"step {self.id!r}: branch requires 'when'")
            if not self.on_true and not self.on_false:
                raise ValueError(f"step {self.id!r}: branch requires on_true or on_false")

        return self

    def binding_references(self) -> set[str]:
        """Step ids this step reads output from, via binding expressions.

        Distinct from ``depends_on``: a step can depend on another for ordering
        without reading its output, and — the case that matters — can reference
        an output it forgot to declare a dependency on. Validation compares the
        two sets and rejects the second.
        """
        found: set[str] = set()
        for value in self._binding_strings():
            match = BINDING_RE.match(value)
            if match and match.group(1).startswith("steps."):
                found.add(match.group(1).split(".")[1])
        return found

    def _binding_strings(self) -> list[str]:
        strings: list[str] = []

        def walk(node: Any) -> None:
            if isinstance(node, str):
                strings.append(node)
            elif isinstance(node, dict):
                for item in node.values():
                    walk(item)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(self.inputs)
        for extra in (self.over, self.when, self.prompt):
            if extra:
                strings.append(extra)
        return strings


class Workflow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=128)
    version: str = "1"
    steps: list[Step] = Field(min_length=1)

    budget: Budget = Field(default_factory=Budget)
    defaults_timeout_s: float = Field(default=30.0, gt=0, le=3600)
    defaults_retry: RetryPolicy = Field(default_factory=RetryPolicy)

    on_step_failure: DependentsPolicy = DependentsPolicy.SKIP

    # Per-workflow concurrency ceiling, applied on top of the global one.
    max_concurrency: int = Field(default=8, ge=1, le=256)

    # Total retries across the whole run. One flapping tool must not be able to
    # consume the run's time by itself while other branches wait.
    retry_budget: int = Field(default=20, ge=0, le=1000)

    @model_validator(mode="after")
    def _within_hard_cap(self) -> Workflow:
        if len(self.steps) > self.budget.max_steps:
            raise ValueError(
                f"workflow declares {len(self.steps)} steps, "
                f"budget.max_steps is {self.budget.max_steps}"
            )
        return self

    def step_map(self) -> dict[str, Step]:
        return {step.id: step for step in self.steps}

    def timeout_for(self, step: Step) -> float:
        return step.timeout_s if step.timeout_s is not None else self.defaults_timeout_s

    def retry_for(self, step: Step) -> RetryPolicy:
        return step.retry if step.retry is not None else self.defaults_retry
