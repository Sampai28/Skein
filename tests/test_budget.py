"""Runaway guards: steps, duration, tool calls, tokens."""

from __future__ import annotations

from typing import Any

import pytest

from skein.budget.guards import BudgetTracker
from skein.errors import BudgetExceededError
from skein.model.workflow import Budget, Step, StepKind
from skein.resilience.clock import VirtualClock
from skein.runtime.state import RunStatus
from tests.conftest import make_step, make_workflow


def test_step_reservations_stop_at_the_ceiling():
    tracker = BudgetTracker(budget=Budget(max_steps=3), clock=VirtualClock())
    for _ in range(3):
        tracker.reserve_step()
    with pytest.raises(BudgetExceededError) as excinfo:
        tracker.reserve_step()
    assert excinfo.value.details["guard"] == "max_steps"


def test_tool_calls_are_reserved_before_the_work_happens():
    """Counting afterwards lets a fan-out exceed the ceiling by its own width."""
    tracker = BudgetTracker(budget=Budget(max_tool_calls=10), clock=VirtualClock())
    tracker.reserve_tool_calls(8)
    with pytest.raises(BudgetExceededError) as excinfo:
        tracker.reserve_tool_calls(5)
    assert excinfo.value.details["requested"] == 5
    assert tracker.tool_calls == 8  # the failed reservation did not count


def test_duration_is_checked_against_the_clock():
    clock = VirtualClock()
    tracker = BudgetTracker(budget=Budget(max_duration_s=10.0), clock=clock)
    tracker.check_duration()

    clock._now += 11.0
    with pytest.raises(BudgetExceededError) as excinfo:
        tracker.check_duration()
    assert excinfo.value.details["guard"] == "max_duration_s"


def test_tokens_are_recorded_after_the_fact():
    tracker = BudgetTracker(budget=Budget(max_tokens=100), clock=VirtualClock())
    tracker.record_tokens(60)
    with pytest.raises(BudgetExceededError):
        tracker.record_tokens(50)
    # The overshoot is visible rather than clamped — the call already happened.
    assert tracker.tokens_used == 110


def test_snapshot_reports_every_guard():
    tracker = BudgetTracker(budget=Budget(), clock=VirtualClock())
    snapshot = tracker.snapshot()
    for key in ("max_steps", "max_tool_calls", "max_tokens", "max_duration_s"):
        assert key in snapshot


async def test_a_run_that_exceeds_max_steps_terminates(registry, build_runner, no_leaked_tasks):
    async def quick(**kwargs: Any) -> dict[str, Any]:
        return {"ok": True}

    registry.register("quick", quick)

    workflow = make_workflow(
        [make_step(f"s{index}", tool="quick") for index in range(6)],
        budget=Budget(max_steps=3, max_duration_s=60),
        max_concurrency=1,
    )
    state = await build_runner(workflow)

    assert state.status is RunStatus.FAILED
    assert state.error_code == "budget_exceeded"


async def test_a_map_fanout_beyond_the_tool_budget_is_refused_before_running(
    registry, build_runner, no_leaked_tasks
):
    calls = {"n": 0}

    async def listify(**kwargs: Any) -> dict[str, Any]:
        return {"items": list(range(20))}

    async def counted(**kwargs: Any) -> dict[str, Any]:
        calls["n"] += 1
        return {"ok": True}

    registry.register("listify", listify)
    registry.register("counted", counted)

    workflow = make_workflow(
        [
            make_step("src", tool="listify"),
            Step(
                id="fan",
                kind=StepKind.MAP,
                tool="counted",
                depends_on=["src"],
                over="${steps.src.output.items}",
                max_fanout=64,
                inputs={"value": "${item}"},
            ),
        ],
        budget=Budget(max_steps=10, max_tool_calls=5, max_duration_s=60),
    )
    state = await build_runner(workflow)

    assert state.status is RunStatus.FAILED
    # None of the fan-out ran: the reservation is made up front.
    assert calls["n"] == 0


async def test_max_fanout_caps_a_single_map_step(registry, build_runner, no_leaked_tasks):
    async def listify(**kwargs: Any) -> dict[str, Any]:
        return {"items": list(range(50))}

    async def counted(**kwargs: Any) -> dict[str, Any]:
        return {"ok": True}

    registry.register("listify", listify)
    registry.register("counted", counted)

    workflow = make_workflow(
        [
            make_step("src", tool="listify"),
            Step(
                id="fan",
                kind=StepKind.MAP,
                tool="counted",
                depends_on=["src"],
                over="${steps.src.output.items}",
                max_fanout=8,
                inputs={"value": "${item}"},
            ),
        ],
        budget=Budget(max_steps=20, max_tool_calls=500, max_duration_s=60),
    )
    state = await build_runner(workflow)

    assert state.step("fan").status.value == "failed"
    assert "max_fanout" in (state.step("fan").error_message or "")


async def test_budget_trip_is_recorded_in_the_trace(registry, build_context, no_leaked_tasks):
    from skein.runtime.scheduler import Scheduler
    from skein.trace.events import EventKind

    async def quick(**kwargs: Any) -> dict[str, Any]:
        return {"ok": True}

    registry.register("quick", quick)

    workflow = make_workflow(
        [make_step(f"s{index}", tool="quick") for index in range(5)],
        budget=Budget(max_steps=2, max_duration_s=60),
        max_concurrency=1,
    )
    context, state, limits = build_context(workflow)
    await Scheduler(context, limits).run(state)

    kinds = [event.kind for event in context.recorder.events]
    assert EventKind.BUDGET_TRIPPED in kinds
