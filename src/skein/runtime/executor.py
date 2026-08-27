"""Executing a single step: timeout, retry, breaker, and the five step kinds.

The scheduler decides *when* a step runs. This decides *what happens* when it
does, and owns the layering of the three resilience mechanisms, which have to
nest in one specific order:

    timeout( retry( breaker( call ) ) )

The timeout is outermost so it bounds the whole attempt sequence — a step with a
5 s timeout and three retries takes 5 s, not 15 s. Retry sits inside so each
attempt is a fresh call. The breaker is innermost so it observes individual
calls, which is what a failure *rate* has to be computed from; putting it
outside retry would record one outcome per logical step and the rate would be
wrong by a factor of the attempt count.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from skein.budget.guards import BudgetTracker
from skein.errors import (
    BudgetExceededError,
    CircuitOpenError,
    LlmError,
    SkeinError,
    StepTimeoutError,
    ToolError,
)
from skein.llm.base import LlmClient
from skein.model.workflow import Step, StepKind, Workflow
from skein.resilience.breaker import BreakerRegistry
from skein.resilience.clock import Clock
from skein.resilience.retry import RetryBudget, retry_async
from skein.runtime.bindings import evaluate_condition, resolve_value
from skein.trace.events import EventKind
from skein.trace.recorder import TraceRecorder
from skein.tools.registry import ToolRegistry
from skein.trace.replay import ReplaySource


@dataclass
class ExecutionContext:
    """Everything a step execution needs, passed as one object.

    Bundled rather than passed as nine arguments so that adding a dependency
    does not require editing every call site — and so the scheduler's own code
    stays about scheduling.
    """

    run_id: str
    workflow: Workflow
    registry: ToolRegistry
    llm: LlmClient
    breakers: BreakerRegistry
    clock: Clock
    recorder: TraceRecorder
    budget: BudgetTracker
    retry_budget: RetryBudget
    run_inputs: dict[str, Any]
    replay: ReplaySource | None = None


class StepExecutor:
    def __init__(self, context: ExecutionContext) -> None:
        self.ctx = context

    async def execute(self, step: Step, outputs: dict[str, Any]) -> Any:
        """Run one step to a value, or raise.

        The caller (the scheduler) owns concurrency permits and status
        transitions. This function's only jobs are producing the value and
        recording what happened.
        """
        timeout_s = self.ctx.workflow.timeout_for(step)
        policy = self.ctx.workflow.retry_for(step)

        async def attempt(attempt_number: int) -> Any:
            # No event recorded here. STEP_STARTED is emitted once by the
            # scheduler and each retry is recorded by note_retry below;
            # recording in both places would break the trace-completeness
            # invariant, which counts exactly one start per executed step.
            return await self._dispatch(step, outputs, attempt_number)

        def note_retry(attempt_number: int, delay: float, exc: BaseException) -> None:
            self.ctx.recorder.record(
                EventKind.STEP_RETRY,
                step_id=step.id,
                attempt=attempt_number,
                target=step.tool or step.model,
                error_code=getattr(exc, "code", type(exc).__name__),
                error_message=str(exc),
                payload={"backoff_s": round(delay, 4)},
            )

        try:
            # asyncio.timeout cancels the enclosed block when the deadline
            # passes. That cancellation is delivered as CancelledError at the
            # innermost await, so a cooperating tool stops; TimeoutError is what
            # surfaces here afterwards. This is why nothing inside may catch
            # CancelledError and continue.
            async with asyncio.timeout(timeout_s):
                return await retry_async(
                    attempt,
                    policy=policy,
                    clock=self.ctx.clock,
                    budget=self.ctx.retry_budget,
                    on_retry=note_retry,
                )
        except TimeoutError as exc:
            raise StepTimeoutError(
                f"step {step.id!r} exceeded its timeout of {timeout_s}s",
                step_id=step.id,
                timeout_s=timeout_s,
            ) from exc

    # -- dispatch ---------------------------------------------------------

    async def _dispatch(self, step: Step, outputs: dict[str, Any], attempt: int) -> Any:
        resolved = resolve_value(step.inputs, outputs, self.ctx.run_inputs)

        if step.kind is StepKind.TOOL_CALL:
            return await self._tool_call(step, resolved, attempt)
        if step.kind is StepKind.LLM_CALL:
            return await self._llm_call(step, outputs, attempt)
        if step.kind is StepKind.MAP:
            return await self._map(step, outputs, resolved, attempt)
        if step.kind is StepKind.REDUCE:
            return self._reduce(step, resolved)
        if step.kind is StepKind.BRANCH:
            return self._branch(step, outputs)
        raise SkeinError(f"unhandled step kind {step.kind!r}", step_id=step.id)

    # -- tool -------------------------------------------------------------

    async def _tool_call(self, step: Step, inputs: dict[str, Any], attempt: int) -> Any:
        assert step.tool is not None  # guaranteed by model validation
        return await self._invoke_tool(step.id, step.tool, inputs, attempt)

    async def _invoke_tool(
        self, step_id: str, tool_name: str, inputs: dict[str, Any], attempt: int
    ) -> Any:
        if self.ctx.replay is not None:
            # Replay short-circuits before the breaker: a recorded run's
            # breaker decisions are part of what is being reproduced, not
            # re-derived from a live dependency that no longer exists.
            return self.ctx.replay.get(step_id, attempt)

        self.ctx.budget.reserve_tool_calls(1)
        breaker = self.ctx.breakers.get(tool_name)
        breaker.acquire()

        tool = self.ctx.registry.get(tool_name)
        started = self.ctx.clock.now()
        try:
            result = await tool.invoke(inputs)
        except CircuitOpenError:
            raise
        except Exception:
            # Only tool-attributable failures move the breaker. A binding error
            # or a budget trip is our fault, and counting it would open a
            # breaker on a healthy dependency.
            breaker.record_failure()
            raise
        else:
            breaker.record_success()

        self.ctx.recorder.record(
            EventKind.TOOL_RESPONSE,
            step_id=step_id,
            attempt=attempt,
            target=tool_name,
            duration_s=round(self.ctx.clock.now() - started, 6),
            payload={"response": result},
        )
        return result

    # -- llm --------------------------------------------------------------

    async def _llm_call(self, step: Step, outputs: dict[str, Any], attempt: int) -> Any:
        assert step.prompt is not None
        prompt = resolve_value(step.prompt, outputs, self.ctx.run_inputs)
        if not isinstance(prompt, str):
            prompt = str(prompt)

        if self.ctx.replay is not None:
            return self.ctx.replay.get(step.id, attempt)

        started = self.ctx.clock.now()
        try:
            response = await self.ctx.llm.complete(prompt, model=step.model)
        except SkeinError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalise backend errors
            raise LlmError(f"llm call failed for step {step.id!r}: {exc}") from exc

        # Recorded before the budget check so the trace shows the call that
        # tripped the token ceiling. Checking first would leave the run's most
        # important event missing from its own trace.
        payload = {
            "text": response.text,
            "model": response.model,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "estimated": response.estimated,
        }
        self.ctx.recorder.record(
            EventKind.LLM_RESPONSE,
            step_id=step.id,
            attempt=attempt,
            target=response.model,
            duration_s=round(self.ctx.clock.now() - started, 6),
            payload={"response": payload},
        )
        self.ctx.budget.record_tokens(response.total_tokens)
        return payload

    # -- map --------------------------------------------------------------

    async def _map(
        self, step: Step, outputs: dict[str, Any], resolved_inputs: dict[str, Any], attempt: int
    ) -> Any:
        assert step.over is not None and step.tool is not None
        collection = resolve_value(step.over, outputs, self.ctx.run_inputs)

        if not isinstance(collection, list):
            raise ToolError(
                f"map step {step.id!r}: 'over' resolved to {type(collection).__name__}, "
                f"expected a list",
                step_id=step.id,
            )

        if len(collection) > step.max_fanout:
            raise ToolError(
                f"map step {step.id!r} would fan out to {len(collection)} calls, "
                f"exceeding max_fanout of {step.max_fanout}",
                step_id=step.id,
                fanout=len(collection),
                max_fanout=step.max_fanout,
            )

        # Reserved up front rather than per item, so a fan-out that would blow
        # the tool-call budget is refused before any of it runs instead of
        # halfway through.
        self.ctx.budget.reserve_tool_calls(len(collection))

        results: list[Any] = [None] * len(collection)

        async def run_item(index: int, item: Any) -> None:
            # Inputs are re-resolved per item so that ${item.…} bindings see
            # this element. resolved_inputs (computed once, without an item in
            # scope) is not reused, because any ${item...} in it would already
            # have raised.
            item_inputs = resolve_value(
                step.inputs, outputs, self.ctx.run_inputs, item=item
            )
            # Each fan-out call records under its own synthetic step id, so a
            # replay can address them individually; without the index every item
            # of a map step would share one recorded response.
            results[index] = await self._invoke_tool(
                f"{step.id}[{index}]", step.tool or "", item_inputs, attempt
            )

        # A nested TaskGroup. The fan-out is structured concurrency inside a
        # single step: if one item fails, its siblings are cancelled and the
        # group raises, so a map step is atomic — it does not return a half-
        # filled list. Cancellation of the parent step propagates in here
        # automatically because these tasks are children of this block.
        try:
            async with asyncio.TaskGroup() as group:
                for index, item in enumerate(collection):
                    group.create_task(run_item(index, item))
        except* Exception as group_error:
            # `except*` unpacks an ExceptionGroup. TaskGroup always raises one,
            # even for a single failure, and an ExceptionGroup is opaque to the
            # retry layer — is_retryable() inspects the exception type, and a
            # group is never itself retryable. Unwrapping to the first real
            # failure preserves the classification.
            #
            # Note CancelledError is deliberately NOT caught here: it is a
            # BaseException, so `except* Exception` does not match it, and a
            # cancelled fan-out propagates outward untouched.
            raise group_error.exceptions[0] from None

        return {"items": results, "count": len(results)}

    # -- reduce -----------------------------------------------------------

    def _reduce(self, step: Step, resolved_inputs: dict[str, Any]) -> Any:
        values = resolved_inputs.get("values")
        if values is None:
            raise ToolError(
                f"reduce step {step.id!r} requires an input named 'values'",
                step_id=step.id,
            )
        if not isinstance(values, list):
            values = [values]

        reducer = step.reducer
        if reducer == "concat":
            joined: list[Any] = []
            for value in values:
                joined.extend(value if isinstance(value, list) else [value])
            return {"result": joined}
        if reducer == "merge":
            merged: dict[str, Any] = {}
            for value in values:
                if isinstance(value, dict):
                    merged.update(value)
            return {"result": merged}
        if reducer == "sum":
            total = 0.0
            for value in values:
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    total += float(value)
                else:
                    raise ToolError(
                        f"reduce step {step.id!r}: 'sum' got a non-numeric value "
                        f"({type(value).__name__})",
                        step_id=step.id,
                    )
            return {"result": total}
        if reducer == "first":
            return {"result": values[0] if values else None}
        if reducer == "last":
            return {"result": values[-1] if values else None}

        raise ToolError(f"unknown reducer {reducer!r}", step_id=step.id)

    # -- branch -----------------------------------------------------------

    def _branch(self, step: Step, outputs: dict[str, Any]) -> Any:
        assert step.when is not None
        resolved = resolve_value(step.when, outputs, self.ctx.run_inputs)
        taken = evaluate_condition(resolved)
        return {
            "condition": taken,
            "taken": step.on_true if taken else step.on_false,
            "not_taken": step.on_false if taken else step.on_true,
        }


def classify_terminal(exc: BaseException) -> EventKind:
    """Which terminal event an exception corresponds to."""
    if isinstance(exc, asyncio.CancelledError):
        return EventKind.STEP_CANCELLED
    if isinstance(exc, BudgetExceededError):
        return EventKind.BUDGET_TRIPPED
    return EventKind.STEP_FAILED
