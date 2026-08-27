"""Retry: attempt counts, backoff sequence against virtual time, and budgets.

Every timing assertion runs on the virtual clock. Nothing here sleeps.
"""

from __future__ import annotations

import random
from typing import Any

import pytest

from skein.errors import RetryBudgetExhaustedError, ToolError, ToolNotFoundError
from skein.model.workflow import RetryPolicy
from skein.resilience.clock import VirtualClock
from skein.resilience.retry import RetryBudget, backoff_delays, retry_async


def test_backoff_grows_exponentially_and_is_capped():
    policy = RetryPolicy(
        max_attempts=6, initial_backoff_s=1.0, multiplier=2.0, max_backoff_s=8.0, jitter=0.0
    )
    assert backoff_delays(policy) == [1.0, 2.0, 4.0, 8.0, 8.0]


def test_jitter_keeps_delays_inside_the_computed_bound():
    policy = RetryPolicy(
        max_attempts=5, initial_backoff_s=1.0, multiplier=2.0, max_backoff_s=100.0, jitter=1.0
    )
    rng = random.Random(7)
    delays = backoff_delays(policy, rng)
    # Full jitter draws uniformly in [0, computed]; never above.
    for index, delay in enumerate(delays):
        assert 0.0 <= delay <= 1.0 * (2.0**index)


def test_partial_jitter_keeps_a_deterministic_floor():
    policy = RetryPolicy(
        max_attempts=3, initial_backoff_s=4.0, multiplier=1.0, max_backoff_s=4.0, jitter=0.5
    )
    for delay in backoff_delays(policy, random.Random(1)):
        assert 2.0 <= delay <= 4.0


async def test_a_retryable_failure_is_retried_and_can_succeed():
    clock = VirtualClock()
    attempts: list[int] = []

    async def flaky(attempt: int) -> str:
        attempts.append(attempt)
        if attempt < 3:
            raise ToolError("transient")
        return "ok"

    policy = RetryPolicy(max_attempts=5, initial_backoff_s=1.0, jitter=0.0)

    import asyncio

    task = asyncio.create_task(retry_async(flaky, policy=policy, clock=clock))
    # Advance past each backoff. The whole sequence resolves in microseconds.
    for _ in range(3):
        await clock.advance(10.0)
    result = await task

    assert result == "ok"
    assert attempts == [1, 2, 3]
    # Two sleeps for three attempts.
    assert clock.slept == [1.0, 2.0]


async def test_a_non_retryable_failure_is_raised_immediately():
    clock = VirtualClock()
    attempts: list[int] = []

    async def broken(attempt: int) -> str:
        attempts.append(attempt)
        raise ToolNotFoundError("no such tool", tool="ghost")

    policy = RetryPolicy(max_attempts=5, initial_backoff_s=1.0, jitter=0.0)

    with pytest.raises(ToolNotFoundError):
        await retry_async(broken, policy=policy, clock=clock)

    assert attempts == [1]
    assert clock.slept == []


async def test_an_unclassified_exception_is_not_retried():
    """The default is no. Retrying an unknown failure multiplies a deterministic
    bug rather than working around a transient one."""
    clock = VirtualClock()
    attempts: list[int] = []

    async def broken(attempt: int) -> str:
        attempts.append(attempt)
        raise ValueError("programming error")

    with pytest.raises(ValueError):
        await retry_async(
            broken,
            policy=RetryPolicy(max_attempts=4, initial_backoff_s=1.0, jitter=0.0),
            clock=clock,
        )

    assert attempts == [1]


async def test_attempts_stop_at_max_attempts():
    import asyncio

    clock = VirtualClock()
    attempts: list[int] = []

    async def always_bad(attempt: int) -> str:
        attempts.append(attempt)
        raise ToolError("still broken")

    policy = RetryPolicy(max_attempts=3, initial_backoff_s=1.0, jitter=0.0)
    task = asyncio.create_task(retry_async(always_bad, policy=policy, clock=clock))
    for _ in range(3):
        await clock.advance(10.0)

    with pytest.raises(ToolError):
        await task
    assert attempts == [1, 2, 3]


async def test_the_retry_budget_stops_a_flapping_tool():
    import asyncio

    clock = VirtualClock()
    budget = RetryBudget(limit=2)
    attempts: list[int] = []

    async def always_bad(attempt: int) -> str:
        attempts.append(attempt)
        raise ToolError("flapping")

    policy = RetryPolicy(max_attempts=10, initial_backoff_s=1.0, jitter=0.0)
    task = asyncio.create_task(
        retry_async(always_bad, policy=policy, clock=clock, budget=budget)
    )
    for _ in range(4):
        await clock.advance(10.0)

    with pytest.raises(RetryBudgetExhaustedError):
        await task

    # Three calls: the original plus two budgeted retries.
    assert attempts == [1, 2, 3]
    assert budget.remaining == 0


async def test_the_budget_is_shared_across_steps():
    """One flapping tool cannot consume the budget of a whole run and then
    starve a second tool of its own retries."""
    import asyncio

    clock = VirtualClock()
    budget = RetryBudget(limit=1)

    async def bad(attempt: int) -> str:
        raise ToolError("bad")

    policy = RetryPolicy(max_attempts=5, initial_backoff_s=1.0, jitter=0.0)

    first = asyncio.create_task(retry_async(bad, policy=policy, clock=clock, budget=budget))
    for _ in range(3):
        await clock.advance(10.0)
    with pytest.raises(RetryBudgetExhaustedError):
        await first

    with pytest.raises(RetryBudgetExhaustedError):
        await retry_async(bad, policy=policy, clock=clock, budget=budget)


async def test_on_retry_callback_reports_attempt_and_delay():
    import asyncio

    clock = VirtualClock()
    seen: list[tuple[int, float]] = []

    async def flaky(attempt: int) -> str:
        if attempt < 3:
            raise ToolError("transient")
        return "ok"

    task = asyncio.create_task(
        retry_async(
            flaky,
            policy=RetryPolicy(max_attempts=4, initial_backoff_s=2.0, jitter=0.0),
            clock=clock,
            on_retry=lambda attempt, delay, exc: seen.append((attempt, delay)),
        )
    )
    for _ in range(3):
        await clock.advance(20.0)
    await task

    assert seen == [(1, 2.0), (2, 4.0)]


async def test_a_step_records_its_attempts_in_the_trace(registry, build_context, no_leaked_tasks):
    from skein.runtime.scheduler import Scheduler
    from skein.trace.events import EventKind
    from tests.conftest import make_step, make_workflow

    calls = {"n": 0}

    async def flaky(**kwargs: Any) -> dict[str, Any]:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ToolError("transient")
        return {"ok": True}

    registry.register("flaky", flaky)

    workflow = make_workflow(
        [
            make_step(
                "a",
                tool="flaky",
                timeout_s=5.0,
                retry=RetryPolicy(max_attempts=4, initial_backoff_s=0.001, jitter=0.0),
            )
        ]
    )
    context, state, limits = build_context(workflow)
    await Scheduler(context, limits).run(state)

    retries = [e for e in context.recorder.events if e.kind is EventKind.STEP_RETRY]
    assert len(retries) == 2
    assert state.step("a").status.value == "succeeded"
