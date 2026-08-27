"""Shared fixtures.

Two things here are load-bearing for the whole suite.

``virtual_clock`` removes real time from every timing assertion, so a retry
sequence spanning minutes is verified instantly and identically on a loaded CI
runner.

``no_leaked_tasks`` runs after every test and fails if any task created during
it is still alive. Task leaks are the failure this project exists to prevent,
and a leak is invisible unless something looks for it — the test that leaked
usually passes.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Awaitable, Callable

import pytest

from skein.budget.guards import BudgetTracker
from skein.config import Settings
from skein.limits.concurrency import LimitSet
from skein.llm.base import StubLlmClient
from skein.model.workflow import Budget, RetryPolicy, Step, StepKind, Workflow
from skein.resilience.breaker import BreakerRegistry, CircuitBreakerConfig
from skein.resilience.clock import SystemClock, VirtualClock
from skein.resilience.retry import RetryBudget
from skein.runtime.executor import ExecutionContext
from skein.runtime.scheduler import Scheduler
from skein.runtime.state import RunState
from skein.tools.registry import ToolRegistry
from skein.trace.recorder import TraceRecorder


@pytest.fixture
def virtual_clock() -> VirtualClock:
    return VirtualClock()


@pytest.fixture
def system_clock() -> SystemClock:
    return SystemClock()


@pytest.fixture
def registry() -> ToolRegistry:
    """An empty registry. Tests add exactly the tools they need, so a test's
    behaviour cannot depend on a demo tool it never mentions."""
    return ToolRegistry()


@pytest.fixture
def llm() -> StubLlmClient:
    return StubLlmClient(default="stub response")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        global_concurrency=8,
        default_tool_concurrency=4,
        queue_size=8,
        workers=2,
        trace_dir=tmp_path / "traces",
        trace_to_disk=False,
        tracing_enabled=False,
        use_stub_llm=True,
        drain_timeout_s=1.0,
        cancel_timeout_s=0.5,
    )


@pytest.fixture
async def no_leaked_tasks() -> Any:
    """Fail the test if it leaves a task running.

    The set is captured before and compared after. Tasks belonging to the test
    framework itself (the one running the test) are excluded by taking the
    difference rather than asserting the set is empty.
    """
    before = asyncio.all_tasks()
    yield
    # One turn so tasks that were cancelled during teardown can actually finish
    # unwinding; without it a correctly-cancelled task can still be in the set.
    await asyncio.sleep(0)
    leaked = {
        task
        for task in asyncio.all_tasks() - before
        if not task.done() and task is not asyncio.current_task()
    }
    if leaked:
        names = sorted(task.get_name() for task in leaked)
        for task in leaked:
            task.cancel()
        pytest.fail(f"{len(leaked)} task(s) leaked: {names}")


# ---------------------------------------------------------------------------
# Workflow construction helpers
# ---------------------------------------------------------------------------

def make_step(
    step_id: str,
    kind: StepKind = StepKind.TOOL_CALL,
    depends_on: list[str] | None = None,
    tool: str | None = "noop",
    **kwargs: Any,
) -> Step:
    return Step(
        id=step_id,
        kind=kind,
        depends_on=depends_on or [],
        tool=tool,
        **kwargs,
    )


def make_workflow(
    steps: list[Step],
    *,
    name: str = "test",
    max_concurrency: int = 8,
    budget: Budget | None = None,
    retry_budget: int = 20,
    **kwargs: Any,
) -> Workflow:
    return Workflow(
        name=name,
        steps=steps,
        max_concurrency=max_concurrency,
        budget=budget or Budget(max_steps=100, max_duration_s=600, max_tool_calls=500),
        retry_budget=retry_budget,
        **kwargs,
    )


@pytest.fixture
def build_runner(
    registry: ToolRegistry, llm: StubLlmClient
) -> Callable[..., Awaitable[RunState]]:
    """Returns a coroutine that runs a workflow through a real Scheduler.

    Bypasses the Engine deliberately. Most tests are about scheduling and
    cancellation, and going through the queue and worker pool would add
    indirection between the assertion and the thing being asserted.
    """

    async def run(
        workflow: Workflow,
        inputs: dict[str, Any] | None = None,
        clock: Any = None,
        limits: LimitSet | None = None,
        breakers: BreakerRegistry | None = None,
        recorder: TraceRecorder | None = None,
        replay: Any = None,
    ) -> RunState:
        used_clock = clock or SystemClock()
        state = RunState(workflow=workflow, inputs=inputs or {})
        used_recorder = recorder or TraceRecorder(state.run_id)
        used_limits = limits or LimitSet(
            global_limit=32, default_tool_limit=8, tool_limits={}
        )
        context = ExecutionContext(
            run_id=state.run_id,
            workflow=workflow,
            registry=registry,
            llm=llm,
            breakers=breakers
            or BreakerRegistry(config=CircuitBreakerConfig(), clock=used_clock),
            clock=used_clock,
            recorder=used_recorder,
            budget=BudgetTracker(budget=workflow.budget, clock=used_clock),
            retry_budget=RetryBudget(limit=workflow.retry_budget),
            run_inputs=inputs or {},
            replay=replay,
        )
        scheduler = Scheduler(context, used_limits)
        await scheduler.run(state)
        return state

    return run


@pytest.fixture
def build_context(
    registry: ToolRegistry, llm: StubLlmClient
) -> Callable[..., tuple[ExecutionContext, RunState, LimitSet]]:
    """Same wiring as build_runner, but returns the pieces unrun.

    Needed by tests that must start the run as a task they can cancel, or that
    inspect the recorder after a failure.
    """

    def build(
        workflow: Workflow,
        inputs: dict[str, Any] | None = None,
        clock: Any = None,
        limits: LimitSet | None = None,
    ) -> tuple[ExecutionContext, RunState, LimitSet]:
        used_clock = clock or SystemClock()
        state = RunState(workflow=workflow, inputs=inputs or {})
        used_limits = limits or LimitSet(global_limit=32, default_tool_limit=8)
        context = ExecutionContext(
            run_id=state.run_id,
            workflow=workflow,
            registry=registry,
            llm=llm,
            breakers=BreakerRegistry(config=CircuitBreakerConfig(), clock=used_clock),
            clock=used_clock,
            recorder=TraceRecorder(state.run_id),
            budget=BudgetTracker(budget=workflow.budget, clock=used_clock),
            retry_budget=RetryBudget(limit=workflow.retry_budget),
            run_inputs=inputs or {},
        )
        return context, state, used_limits

    return build


@pytest.fixture
def fast_retry() -> RetryPolicy:
    return RetryPolicy(max_attempts=3, initial_backoff_s=0.01, max_backoff_s=0.1, jitter=0.0)
