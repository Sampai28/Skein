"""Per-step and per-workflow timeouts, and the dependents policy."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from skein.errors import WorkflowTimeoutError
from skein.model.workflow import Budget, DependentsPolicy
from skein.runtime.scheduler import Scheduler
from skein.runtime.state import RunStatus
from skein.trace.events import StepStatus
from tests.conftest import make_step, make_workflow


async def test_a_step_that_overruns_is_failed_with_step_timeout(
    registry, build_runner, no_leaked_tasks
):
    async def slow(**kwargs: Any) -> dict[str, Any]:
        await asyncio.sleep(1.0)
        return {"ok": True}

    registry.register("slow", slow)

    workflow = make_workflow([make_step("a", tool="slow", timeout_s=0.05)])
    state = await build_runner(workflow)

    assert state.step("a").status is StepStatus.FAILED
    assert state.step("a").error_code == "step_timeout"


async def test_a_step_timeout_leaves_no_task_running(registry, build_runner):
    before = asyncio.all_tasks()
    entered = asyncio.Event()

    async def slow(**kwargs: Any) -> dict[str, Any]:
        entered.set()
        await asyncio.sleep(5.0)
        return {"ok": True}

    registry.register("slow", slow)

    workflow = make_workflow([make_step("a", tool="slow", timeout_s=0.05)])
    await build_runner(workflow)
    await asyncio.sleep(0)

    leaked = {task for task in asyncio.all_tasks() - before if not task.done()}
    leaked.discard(asyncio.current_task())
    assert leaked == set()


async def test_the_timeout_bounds_the_whole_retry_sequence(
    registry, build_runner, no_leaked_tasks
):
    """Three attempts inside a 0.15s budget must not take 3 x 0.15s.

    This is what the timeout-outside-retry nesting buys. Inverted, each attempt
    would get its own deadline and the step could run for the sum.
    """
    from skein.model.workflow import RetryPolicy

    attempts = 0

    async def slow_and_flaky(**kwargs: Any) -> dict[str, Any]:
        nonlocal attempts
        attempts += 1
        await asyncio.sleep(0.5)
        return {"ok": True}

    registry.register("flaky", slow_and_flaky)

    workflow = make_workflow(
        [
            make_step(
                "a",
                tool="flaky",
                timeout_s=0.15,
                retry=RetryPolicy(max_attempts=3, initial_backoff_s=0.01, jitter=0.0),
            )
        ]
    )

    loop = asyncio.get_running_loop()
    started = loop.time()
    state = await build_runner(workflow)
    elapsed = loop.time() - started

    assert state.step("a").error_code == "step_timeout"
    # Comfortably under 3 x 0.5s; generous bound so a slow runner cannot flake it.
    assert elapsed < 0.45


async def test_a_workflow_timeout_terminates_the_run(registry, build_context, no_leaked_tasks):
    async def slow(**kwargs: Any) -> dict[str, Any]:
        await asyncio.sleep(5.0)
        return {"ok": True}

    registry.register("slow", slow)

    workflow = make_workflow(
        [make_step("a", tool="slow", timeout_s=4.0)],
        budget=Budget(max_duration_s=0.1, max_steps=10),
    )
    context, state, limits = build_context(workflow)

    with pytest.raises(WorkflowTimeoutError):
        await Scheduler(context, limits).run(state)

    assert state.status is RunStatus.FAILED
    assert state.error_code == "workflow_timeout"


async def test_dependents_are_skipped_when_a_step_times_out(
    registry, build_runner, no_leaked_tasks
):
    async def slow(**kwargs: Any) -> dict[str, Any]:
        await asyncio.sleep(1.0)
        return {"ok": True}

    async def quick(**kwargs: Any) -> dict[str, Any]:
        return {"ok": True}

    registry.register("slow", slow)
    registry.register("quick", quick)

    workflow = make_workflow(
        [
            make_step("slow_step", tool="slow", timeout_s=0.05),
            make_step("dependent", tool="quick", depends_on=["slow_step"]),
            make_step("independent", tool="quick"),
        ],
        on_step_failure=DependentsPolicy.SKIP,
    )
    state = await build_runner(workflow)

    assert state.step("slow_step").error_code == "step_timeout"
    assert state.step("dependent").status is StepStatus.SKIPPED
    assert state.step("independent").status is StepStatus.SUCCEEDED


async def test_continue_policy_lets_dependents_attempt_and_fail_honestly(
    registry, build_runner, no_leaked_tasks
):
    """CONTINUE does not fabricate a value for the missing output."""

    async def boom(**kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("nope")

    async def consume(**kwargs: Any) -> dict[str, Any]:
        return {"ok": True}

    registry.register("boom", boom)
    registry.register("consume", consume)

    workflow = make_workflow(
        [
            make_step("bad", tool="boom"),
            make_step(
                "dependent",
                tool="consume",
                depends_on=["bad"],
                inputs={"x": "${steps.bad.output.value}"},
            ),
        ],
        on_step_failure=DependentsPolicy.CONTINUE,
    )
    state = await build_runner(workflow)

    assert state.step("bad").status is StepStatus.FAILED
    # Never becomes eligible, because eligibility requires dependencies to have
    # SUCCEEDED; it settles as skipped rather than running against a null.
    assert state.step("dependent").status is StepStatus.SKIPPED


async def test_a_run_where_everything_times_out_still_reaches_a_terminal_state(
    registry, build_runner, no_leaked_tasks
):
    async def slow(**kwargs: Any) -> dict[str, Any]:
        await asyncio.sleep(1.0)
        return {"ok": True}

    registry.register("slow", slow)

    workflow = make_workflow(
        [make_step(f"s{index}", tool="slow", timeout_s=0.03) for index in range(4)]
    )
    state = await build_runner(workflow)

    assert state.status.is_terminal
    assert all(step.status.is_terminal for step in state.steps.values())
