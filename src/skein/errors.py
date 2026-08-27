"""Every way Skein refuses or fails, named.

One exception hierarchy with a stable ``code`` on each class. The code is what
the API returns in an RFC 7807 body, what the Prometheus counter is tagged with,
and what tests assert on. Free-text messages change; codes do not.

``SkeinError`` deliberately does **not** inherit from ``asyncio.CancelledError``,
and nothing here ever wraps one. That matters more than it looks: since Python
3.8 ``CancelledError`` inherits from ``BaseException``, not ``Exception``, so
that a ``except Exception`` handler cannot accidentally swallow a cancellation.
Any code here that caught ``SkeinError`` and returned a value would break the
cancellation contract if ``CancelledError`` were in the hierarchy.
"""

from __future__ import annotations

from typing import Any


class SkeinError(Exception):
    """Base for everything Skein raises deliberately."""

    code: str = "skein_error"
    http_status: int = 500

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details

    def to_problem(self) -> dict[str, Any]:
        """RFC 7807 body. Extension members carry the structured details."""
        return {
            "type": f"https://skein.dev/problems/{self.code.replace('_', '-')}",
            "title": self.__class__.__name__,
            "status": self.http_status,
            "detail": self.message,
            "code": self.code,
            **self.details,
        }


# ---------------------------------------------------------------------------
# Workflow definition — raised at submission, never during execution
# ---------------------------------------------------------------------------

class ValidationError(SkeinError):
    code = "validation_error"
    http_status = 422


class DagCycleError(ValidationError):
    code = "dag_cycle"


class UnknownStepReferenceError(ValidationError):
    code = "unknown_step_reference"


class DuplicateStepIdError(ValidationError):
    code = "duplicate_step_id"


class UnreachableStepError(ValidationError):
    code = "unreachable_step"


class MissingBindingError(ValidationError):
    code = "missing_binding"


class InvalidPolicyValueError(ValidationError):
    code = "invalid_policy_value"


class StepCountExceededError(ValidationError):
    code = "step_count_exceeded"


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

class ToolError(SkeinError):
    """A tool failed. Retryable unless a subclass says otherwise."""

    code = "tool_error"
    http_status = 502
    retryable = True


class ToolNotFoundError(ToolError):
    code = "tool_not_found"
    http_status = 400
    # A missing tool will still be missing on the next attempt.
    retryable = False


class ToolOutputSchemaError(ToolError):
    """A tool returned something its declared output schema rejects.

    Classified as a tool failure rather than allowed to propagate as a Pydantic
    error, so that a malformed tool response is retried and circuit-broken like
    any other tool fault instead of crashing the run.
    """

    code = "tool_output_schema"
    retryable = False


class ToolTimeoutError(ToolError):
    code = "tool_timeout"
    http_status = 504
    retryable = True


class LlmError(SkeinError):
    code = "llm_error"
    http_status = 502
    retryable = True


class LlmParseError(LlmError):
    """The model produced text that will not parse into the declared type."""

    code = "llm_parse_error"
    # Retryable: resampling is the documented fallback, and it often works.
    retryable = True


class StepTimeoutError(SkeinError):
    code = "step_timeout"
    http_status = 504


class WorkflowTimeoutError(SkeinError):
    code = "workflow_timeout"
    http_status = 504


class DependencyFailedError(SkeinError):
    """A step was skipped because something it depends on did not succeed."""

    code = "dependency_failed"
    http_status = 424


class CircuitOpenError(ToolError):
    code = "circuit_open"
    http_status = 503
    # Not retryable *within* an attempt loop — the breaker is open precisely to
    # stop that. The scheduler surfaces it and the next workflow tries again
    # once the breaker moves to half-open.
    retryable = False


class RetryBudgetExhaustedError(SkeinError):
    code = "retry_budget_exhausted"
    http_status = 503


class BudgetExceededError(SkeinError):
    """A runaway guard tripped: steps, duration, tool calls or tokens."""

    code = "budget_exceeded"
    http_status = 429


class QueueFullError(SkeinError):
    code = "queue_full"
    http_status = 429


class RunNotFoundError(SkeinError):
    code = "run_not_found"
    http_status = 404


class ReplayMismatchError(SkeinError):
    """A replayed run diverged from its recording."""

    code = "replay_mismatch"
    http_status = 409


class BindingResolutionError(SkeinError):
    code = "binding_resolution"
    http_status = 422


def is_retryable(exc: BaseException) -> bool:
    """Classify an exception for the retry layer.

    Only errors explicitly marked retryable are retried. The default for
    anything unrecognised is **no** — retrying an unknown failure is how a
    deterministic bug turns into three deterministic bugs and a longer run.
    """
    return bool(getattr(exc, "retryable", False))
