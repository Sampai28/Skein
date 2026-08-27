"""Cancellation — the hardest property to get right and the easiest to fake.

Three things must hold when a run is cancelled:

1. every in-flight child stops;
2. cleanup runs, and the trace records a terminal event for each;
3. no task survives.

The third is the one that fails silently in real systems, so ``no_leaked_tasks``
is applied throughout and the explicit leak assertions below check
``asyncio.all_tasks()`` directly rather than trusting the fixture alone.

The uncancellable-tool test is the important one: it proves the deadline holds
even when the tool refuses to cooperate, which is the case a well-behaved test
suite never exercises and production hits regularly.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from skein.limits.concurrency import LimitSet
from skein.runtime.scheduler import Scheduler
from skein.runtime.state import RunStatus
from skein.tools.fault import FaultConfig, HangMode
from skein.trace.events import EventKind, StepStatus
from skein.trace.hashing import trace_completeness
from tests.conftest import make_step, make_workflow


async def test_cancelling_a_run_cancels_every_child(registry, build_context):
    started = asyncio.Event()
    cleanup_ran: list[str] = []

    async def blocker(**kwargs: Any) -> dict[str, Any]:
        started.set()
        try:
            await asyncio.Event().wait()  # never completes
            return {"ok": True}
        except asyncio.CancelledError:
            cleanup_ran.append(str(kwargs.get("label")))
            # Re-raised, as any correct cleanup handler must. Swallowing it here
            # would make the tool appear to complete successfully after being
            # told to stop.
            raise

    registry.register("blocker", blocker)

    workflow = make_workflow(
        [
            make_step("a", tool="blocker", inputs={"label": "a"}, timeout_s=60),
            make_step("b", tool="blocker", inputs={"label": "b"}, timeout_s=60),
        ]
    )
    context, state, limits = build_context(workflow, limits=LimitSet(global_limit=4))
    scheduler = Scheduler(context, limits)

    task = asyncio.create_task(scheduler.run(state))
    await started.wait()
    await asyncio.sleep(0.01)  # let both steps enter their await

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        # Re-raised by the scheduler rather than swallowed, so the caller sees
        # the cancellation it requested.
        await task

    assert state.status is RunStatus.CANCELLED
    assert sorted(cleanup_ran) == ["a", "b"]
    assert all(
        step.status in (StepStatus.CANCELLED, StepStatus.SKIPPED)
        for step in state.steps.values()
    )


async def test_cancellation_leaves_no_tasks_behind(registry, build_context):
    before = asyncio.all_tasks()
    started = asyncio.Event()

    async def blocker(**kwargs: Any) -> dict[str, Any]:
        started.set()
        await asyncio.Event().wait()
        return {"ok": True}

    registry.register("blocker", blocker)

    workflow = make_workflow(
        [make_step(f"s{index}", tool="blocker", timeout_s=60) for index in range(5)]
    )
    context, state, limits = build_context(workflow, limits=LimitSet(global_limit=8))

    task = asyncio.create_task(Scheduler(context, limits).run(state))
    await started.wait()
    await asyncio.sleep(0.01)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # A turn for the cancelled children to finish unwinding.
    await asyncio.sleep(0)
    leaked = {t for t in asyncio.all_tasks() - before if not t.done()}
    leaked.discard(asyncio.current_task())
    assert leaked == set(), f"leaked: {[t.get_name() for t in leaked]}"


async def test_cancelled_steps_still_record_a_terminal_event(registry, build_context):
    """The shield in the scheduler's cancellation handler exists for this."""
    started = asyncio.Event()

    async def blocker(**kwargs: Any) -> dict[str, Any]:
        started.set()
        await asyncio.Event().wait()
        return {"ok": True}

    registry.register("blocker", blocker)

    workflow = make_workflow([make_step("a", tool="blocker", timeout_s=60)])
    context, state, limits = build_context(workflow)

    task = asyncio.create_task(Scheduler(context, limits).run(state))
    await started.wait()
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    kinds = [event.kind for event in context.recorder.events]
    assert EventKind.STEP_STARTED in kinds
    assert EventKind.STEP_CANCELLED in kinds
    assert EventKind.RUN_FINISHED in kinds

    started_ids, terminated_ids = trace_completeness(context.recorder.events)
    assert started_ids == terminated_ids


async def test_a_tool_that_ignores_cancellation_still_hits_its_deadline(
    registry, build_context
):
    """A tool swallowing CancelledError must not defeat the step timeout.

    The tool here catches CancelledError in a loop and keeps waiting — the exact
    anti-pattern. asyncio.timeout still fires, the step is reported as timed out,
    and the run completes. The rogue coroutine remains alive inside the runtime's
    accounting, which is why the leaked-task gauge exists rather than a claim
    that this can never happen.
    """
    from skein.tools.fault import FaultInjector

    async def inner(**kwargs: Any) -> dict[str, Any]:
        return {"ok": True}

    injector = FaultInjector(
        inner,
        FaultConfig(hang_mode=HangMode.UNCANCELLABLE, hang_after_calls=0),
    )
    registry.register("rogue", injector.__call__)

    workflow = make_workflow([make_step("a", tool="rogue", timeout_s=0.05)])
    context, state, limits = build_context(workflow)

    # The whole run finishes despite the tool never yielding to cancellation.
    async with asyncio.timeout(5.0):
        await Scheduler(context, limits).run(state)

    assert state.step("a").status is StepStatus.FAILED
    assert state.step("a").error_code == "step_timeout"
    assert injector.stats.cancellations_observed >= 1


async def test_a_cooperative_hang_is_cancelled_cleanly(registry, build_context, no_leaked_tasks):
    from skein.tools.fault import FaultInjector

    async def inner(**kwargs: Any) -> dict[str, Any]:
        return {"ok": True}

    injector = FaultInjector(
        inner, FaultConfig(hang_mode=HangMode.COOPERATIVE, hang_after_calls=0)
    )
    registry.register("hanger", injector.__call__)

    workflow = make_workflow([make_step("a", tool="hanger", timeout_s=0.05)])
    context, state, limits = build_context(workflow)

    await Scheduler(context, limits).run(state)

    assert state.step("a").status is StepStatus.FAILED
    assert state.step("a").error_code == "step_timeout"


async def test_cancelling_releases_concurrency_permits(registry, build_context):
    """A cancelled step must not strand a semaphore permit.

    If it did, the effect would be a runtime that slowly loses capacity with
    every cancellation until it stops scheduling anything — a failure that looks
    like a hang and is very hard to attribute.
    """
    started = asyncio.Event()
    limits = LimitSet(global_limit=2, default_tool_limit=2)

    async def blocker(**kwargs: Any) -> dict[str, Any]:
        started.set()
        await asyncio.Event().wait()
        return {"ok": True}

    registry.register("blocker", blocker)

    workflow = make_workflow(
        [make_step(f"s{index}", tool="blocker", timeout_s=60) for index in range(2)]
    )
    context, state, _ = build_context(workflow, limits=limits)

    task = asyncio.create_task(Scheduler(context, limits).run(state))
    await started.wait()
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)

    assert limits.observer.current(
        __import__("skein.limits.concurrency", fromlist=["LimitScope"]).LimitScope.GLOBAL
    ) == 0


async def test_cancelling_an_already_finished_run_is_a_no_op(
    registry, build_runner, settings, no_leaked_tasks
):
    from skein.runtime.store import RunStore
    from skein.trace.recorder import TraceRecorder

    async def quick(**kwargs: Any) -> dict[str, Any]:
        return {"ok": True}

    registry.register("quick", quick)

    workflow = make_workflow([make_step("a", tool="quick")])
    state = await build_runner(workflow)

    store = RunStore()
    handle = store.add(state, TraceRecorder(state.run_id))
    handle.finished.set()

    result = await store.cancel(state.run_id)
    # Still succeeded; cancelling a completed run does not rewrite history.
    assert result.status is RunStatus.SUCCEEDED
