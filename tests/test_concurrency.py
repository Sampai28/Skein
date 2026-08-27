"""Concurrency caps, asserted against observed simultaneity.

Every assertion here is on ``observer.peak`` — what actually happened — rather
than on how a semaphore was constructed. The bug this catches is a limiter that
is acquired and released around the wrong span of code, which a configuration
assertion would pass with flying colours.
"""

from __future__ import annotations

import asyncio
from typing import Any

from skein.limits.concurrency import LimitScope, LimitSet
from tests.conftest import make_step, make_workflow


def _tracking_tool(peak_holder: dict[str, int], delay: float = 0.02):
    async def tool(**kwargs: Any) -> dict[str, Any]:
        peak_holder["current"] += 1
        peak_holder["peak"] = max(peak_holder["peak"], peak_holder["current"])
        try:
            await asyncio.sleep(delay)
            return {"ok": True}
        finally:
            peak_holder["current"] -= 1

    return tool


async def test_global_limit_is_never_exceeded_under_burst(
    registry, build_context, no_leaked_tasks
):
    from skein.runtime.scheduler import Scheduler

    observed = {"current": 0, "peak": 0}
    registry.register("burst", _tracking_tool(observed))

    limits = LimitSet(global_limit=3, default_tool_limit=99)
    workflow = make_workflow(
        [make_step(f"s{index}", tool="burst") for index in range(24)],
        max_concurrency=99,
    )
    context, state, _ = build_context(workflow, limits=limits)
    await Scheduler(context, limits).run(state)

    assert observed["peak"] <= 3
    assert limits.observer.peak(LimitScope.GLOBAL) <= 3
    # And the cap actually bound — otherwise the assertion above is vacuous.
    assert observed["peak"] > 1


async def test_per_tool_limit_is_independent_of_the_global_one(
    registry, build_context, no_leaked_tasks
):
    from skein.runtime.scheduler import Scheduler

    slow_observed = {"current": 0, "peak": 0}
    fast_observed = {"current": 0, "peak": 0}
    registry.register("slow_tool", _tracking_tool(slow_observed, delay=0.03))
    registry.register("fast_tool", _tracking_tool(fast_observed, delay=0.01))

    limits = LimitSet(
        global_limit=16, default_tool_limit=16, tool_limits={"slow_tool": 2}
    )
    steps = [make_step(f"slow{i}", tool="slow_tool") for i in range(8)]
    steps += [make_step(f"fast{i}", tool="fast_tool") for i in range(8)]
    workflow = make_workflow(steps, max_concurrency=16)

    context, state, _ = build_context(workflow, limits=limits)
    await Scheduler(context, limits).run(state)

    assert slow_observed["peak"] <= 2
    # The constrained tool did not constrain the unconstrained one.
    assert fast_observed["peak"] > 2


async def test_per_workflow_limit_applies(registry, build_context, no_leaked_tasks):
    from skein.runtime.scheduler import Scheduler

    observed = {"current": 0, "peak": 0}
    registry.register("burst", _tracking_tool(observed))

    limits = LimitSet(global_limit=99, default_tool_limit=99)
    workflow = make_workflow(
        [make_step(f"s{index}", tool="burst") for index in range(16)],
        max_concurrency=4,
    )
    context, state, _ = build_context(workflow, limits=limits)
    await Scheduler(context, limits).run(state)

    assert observed["peak"] <= 4


async def test_the_tightest_limit_wins(registry, build_context, no_leaked_tasks):
    from skein.runtime.scheduler import Scheduler

    observed = {"current": 0, "peak": 0}
    registry.register("burst", _tracking_tool(observed))

    limits = LimitSet(global_limit=6, default_tool_limit=2)
    workflow = make_workflow(
        [make_step(f"s{index}", tool="burst") for index in range(12)],
        max_concurrency=4,
    )
    context, state, _ = build_context(workflow, limits=limits)
    await Scheduler(context, limits).run(state)

    assert observed["peak"] <= 2


async def test_permits_are_released_after_a_failure(registry, build_context, no_leaked_tasks):
    """A failing step must return its permit, or capacity bleeds away."""
    from skein.runtime.scheduler import Scheduler

    calls = {"n": 0}

    async def sometimes_bad(**kwargs: Any) -> dict[str, Any]:
        calls["n"] += 1
        await asyncio.sleep(0.001)
        if calls["n"] % 2 == 0:
            raise RuntimeError("nope")
        return {"ok": True}

    registry.register("flaky", sometimes_bad)

    limits = LimitSet(global_limit=2, default_tool_limit=2)
    workflow = make_workflow(
        [make_step(f"s{index}", tool="flaky") for index in range(10)],
        max_concurrency=2,
    )
    context, state, _ = build_context(workflow, limits=limits)
    await Scheduler(context, limits).run(state)

    assert limits.observer.current(LimitScope.GLOBAL) == 0
    assert all(step.status.is_terminal for step in state.steps.values())


async def test_acquire_order_prevents_deadlock(no_leaked_tasks):
    """Two runs contending for the same tool from opposite directions.

    With independent acquisition this is the classic AB/BA deadlock. With the
    fixed global order in LimitSet it cannot form, and both runs complete.
    """
    limits = LimitSet(global_limit=2, default_tool_limit=1)
    limits.register_workflow("run-a", 2)
    limits.register_workflow("run-b", 2)

    async def hold(run_id: str, tool: str) -> None:
        async with limits.acquire(run_id, tool):
            await asyncio.sleep(0.01)

    async with asyncio.timeout(2.0):
        async with asyncio.TaskGroup() as group:
            group.create_task(hold("run-a", "alpha"))
            group.create_task(hold("run-b", "alpha"))
            group.create_task(hold("run-a", "beta"))
            group.create_task(hold("run-b", "beta"))

    assert limits.observer.current(LimitScope.GLOBAL) == 0


async def test_observer_reports_waits_when_a_limit_binds(no_leaked_tasks):
    limits = LimitSet(global_limit=1, default_tool_limit=1)
    limits.register_workflow("run", 1)

    async def hold() -> None:
        async with limits.acquire("run", "tool"):
            await asyncio.sleep(0.005)

    async with asyncio.TaskGroup() as group:
        for _ in range(4):
            group.create_task(hold())

    # Distinguishes "the limit was configured" from "the limit did something".
    assert limits.observer.waited(LimitScope.GLOBAL) >= 1
