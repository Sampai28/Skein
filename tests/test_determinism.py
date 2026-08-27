"""Record, then replay, and compare trajectory hashes."""

from __future__ import annotations

from typing import Any

import pytest

from skein.errors import ReplayMismatchError
from skein.trace.events import EventKind, StepStatus
from skein.trace.hashing import digest_value, trajectory_hash
from skein.trace.replay import ReplaySource
from tests.conftest import make_step, make_workflow


def test_the_hash_ignores_completion_order_of_concurrent_steps():
    """Two concurrent steps may finish in either order across runs. Hashing
    completion order would make the hash a measure of scheduling luck."""
    from skein.trace.events import EventKind, StepStatus, TraceEvent

    def terminal(step_id: str, seq: int) -> TraceEvent:
        return TraceEvent(
            kind=EventKind.STEP_SUCCEEDED,
            run_id="r",
            seq=seq,
            step_id=step_id,
            status=StepStatus.SUCCEEDED,
            payload={"output": {"v": step_id}},
        )

    forward = [terminal("a", 1), terminal("b", 2)]
    reverse = [terminal("b", 1), terminal("a", 2)]
    assert trajectory_hash(forward) == trajectory_hash(reverse)


def test_the_hash_changes_when_an_output_changes():
    from skein.trace.events import EventKind, StepStatus, TraceEvent

    def terminal(value: str) -> list[TraceEvent]:
        return [
            TraceEvent(
                kind=EventKind.STEP_SUCCEEDED,
                run_id="r",
                seq=1,
                step_id="a",
                status=StepStatus.SUCCEEDED,
                payload={"output": {"v": value}},
            )
        ]

    assert trajectory_hash(terminal("x")) != trajectory_hash(terminal("y"))


def test_the_hash_ignores_timestamps_and_run_ids():
    from skein.trace.events import EventKind, StepStatus, TraceEvent

    def terminal(run_id: str, ts: float) -> list[TraceEvent]:
        return [
            TraceEvent(
                kind=EventKind.STEP_SUCCEEDED,
                run_id=run_id,
                ts=ts,
                seq=1,
                step_id="a",
                status=StepStatus.SUCCEEDED,
                payload={"output": {"v": 1}},
                duration_s=ts,
            )
        ]

    assert trajectory_hash(terminal("run-1", 100.0)) == trajectory_hash(
        terminal("run-2", 999.0)
    )


def test_canonical_digest_is_key_order_independent():
    assert digest_value({"a": 1, "b": 2}) == digest_value({"b": 2, "a": 1})


async def test_a_recorded_run_replays_to_the_same_trajectory(
    registry, build_context, no_leaked_tasks
):
    from skein.runtime.scheduler import Scheduler

    counter = {"n": 0}

    async def nondeterministic(**kwargs: Any) -> dict[str, Any]:
        # Returns a different value on every call. A replay that actually
        # substitutes recorded responses will not see these later values.
        counter["n"] += 1
        return {"value": counter["n"]}

    registry.register("nd", nondeterministic)

    workflow = make_workflow(
        [
            make_step("a", tool="nd"),
            make_step("b", tool="nd", depends_on=["a"]),
        ]
    )

    context, state, limits = build_context(workflow)
    await Scheduler(context, limits).run(state)
    original_hash = state.trajectory_hash
    recorded_events = list(context.recorder.events)

    # Replay against the recording. The live tool would now return 3 and 4.
    replay = ReplaySource(recorded_events)
    context2, state2, limits2 = build_context(workflow)
    context2.replay = replay
    await Scheduler(context2, limits2).run(state2)

    assert state2.trajectory_hash == original_hash
    assert state2.step("a").output == {"value": 1}
    assert state2.step("b").output == {"value": 2}
    # The live tool was not called during replay.
    assert counter["n"] == 2


async def test_replay_detects_divergence(registry, build_context, no_leaked_tasks):
    from skein.runtime.scheduler import Scheduler
    from skein.trace.events import EventKind, StepStatus, TraceEvent

    async def fixed(**kwargs: Any) -> dict[str, Any]:
        return {"value": 1}

    registry.register("fixed", fixed)

    workflow = make_workflow([make_step("a", tool="fixed")])
    context, state, limits = build_context(workflow)
    await Scheduler(context, limits).run(state)

    # Tamper with the recording so the replayed trajectory cannot match.
    tampered = list(context.recorder.events)
    tampered.append(
        TraceEvent(
            kind=EventKind.STEP_SUCCEEDED,
            run_id=state.run_id,
            seq=999,
            step_id="ghost",
            status=StepStatus.SUCCEEDED,
            payload={"output": {"v": "unexpected"}},
        )
    )
    source = ReplaySource(tampered)

    with pytest.raises(ReplayMismatchError) as excinfo:
        source.compare(context.recorder.events)
    assert "original_hash" in excinfo.value.details


async def test_replay_refuses_to_fall_through_to_the_live_tool(
    registry, build_context, no_leaked_tasks
):
    """A missing recorded response is an error, not a silent live call.

    Otherwise a replay would be part recording and part live, and its hash would
    mean nothing.
    """
    from skein.runtime.scheduler import Scheduler

    async def fixed(**kwargs: Any) -> dict[str, Any]:
        return {"value": 1}

    registry.register("fixed", fixed)

    workflow = make_workflow([make_step("a", tool="fixed"), make_step("b", tool="fixed")])

    context, state, limits = build_context(workflow)
    await Scheduler(context, limits).run(state)

    # Drop b's response from the recording.
    partial = [
        event
        for event in context.recorder.events
        if not (event.step_id == "b" and event.kind is EventKind.TOOL_RESPONSE)
    ]
    context2, state2, limits2 = build_context(workflow)
    context2.replay = ReplaySource(partial)
    await Scheduler(context2, limits2).run(state2)

    assert state2.step("b").status is StepStatus.FAILED
    assert state2.step("b").error_code == "replay_mismatch"


async def test_a_trace_round_trips_through_jsonl(tmp_path, registry, build_context):
    from skein.runtime.scheduler import Scheduler
    from skein.trace.recorder import TraceRecorder

    async def fixed(**kwargs: Any) -> dict[str, Any]:
        return {"value": 7}

    registry.register("fixed", fixed)

    workflow = make_workflow([make_step("a", tool="fixed")])
    path = tmp_path / "trace.jsonl"

    context, state, limits = build_context(workflow)
    context.recorder = TraceRecorder(state.run_id, path)
    await Scheduler(context, limits).run(state)
    context.recorder.close()

    loaded = TraceRecorder.load(path)
    assert trajectory_hash(loaded) == state.trajectory_hash


def test_a_truncated_trace_line_is_tolerated(tmp_path):
    """A trace from a killed process ends mid-line. Those are the most
    interesting traces and must remain readable."""
    from skein.trace.recorder import TraceRecorder

    path = tmp_path / "partial.jsonl"
    path.write_text(
        '{"kind":"run_started","run_id":"r","ts":1.0,"seq":1}\n{"kind":"step_star',
        encoding="utf-8",
    )
    events = TraceRecorder.load(path)
    assert len(events) == 1
