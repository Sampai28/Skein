"""Scheduling: dependency order, single execution, eligibility, branch, reduce.

The invariants asserted here are the ones the whole design rests on. Two in
particular:

* **no step begins before all its dependencies complete** — checked by recording
  the order of starts and finishes and comparing, not by trusting the graph;
* **no step executes twice** — checked with a call counter per step, because a
  scheduler bug that re-queues a completed step is silent otherwise.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

from skein.model.workflow import DependentsPolicy, Step, StepKind
from skein.tools.registry import ToolRegistry
from skein.trace.events import StepStatus
from tests.conftest import make_step, make_workflow


class Recorder:
    """Records when each tool call starts and finishes."""

    def __init__(self) -> None:
        self.starts: list[str] = []
        self.finishes: list[str] = []
        self.calls: dict[str, int] = defaultdict(int)
        self.concurrent = 0
        self.peak = 0

    def install(self, registry: ToolRegistry, delay: float = 0.0) -> None:
        async def noop(**kwargs: Any) -> dict[str, Any]:
            label = str(kwargs.get("label", "?"))
            self.calls[label] += 1
            self.starts.append(label)
            self.concurrent += 1
            self.peak = max(self.peak, self.concurrent)
            try:
                if delay:
                    await asyncio.sleep(delay)
                else:
                    # Always yield at least once. A tool that never awaits runs
                    # to completion synchronously, which would make every test
                    # here observe a concurrency of exactly one and prove
                    # nothing about scheduling.
                    await asyncio.sleep(0)
                return {"label": label, "ok": True}
            finally:
                self.concurrent -= 1
                self.finishes.append(label)

        registry.register("noop", noop)


async def test_a_linear_chain_runs_in_order(registry, build_runner, no_leaked_tasks):
    recorder = Recorder()
    recorder.install(registry)

    workflow = make_workflow(
        [
            make_step("a", inputs={"label": "a"}),
            make_step("b", depends_on=["a"], inputs={"label": "b"}),
            make_step("c", depends_on=["b"], inputs={"label": "c"}),
        ]
    )
    state = await build_runner(workflow)

    assert recorder.starts == ["a", "b", "c"]
    assert all(step.status is StepStatus.SUCCEEDED for step in state.steps.values())


async def test_independent_branches_run_concurrently(registry, build_runner, no_leaked_tasks):
    recorder = Recorder()
    recorder.install(registry, delay=0.02)

    workflow = make_workflow(
        [
            make_step("root", inputs={"label": "root"}),
            make_step("left", depends_on=["root"], inputs={"label": "left"}),
            make_step("right", depends_on=["root"], inputs={"label": "right"}),
            make_step("join", depends_on=["left", "right"], inputs={"label": "join"}),
        ]
    )
    await build_runner(workflow)

    # The two arms overlapped rather than serialising.
    assert recorder.peak >= 2
    assert recorder.starts[0] == "root"
    assert recorder.starts[-1] == "join"


async def test_a_fast_arm_does_not_wait_for_a_slow_sibling(
    registry, build_runner, no_leaked_tasks
):
    """The property that rules out level-by-level execution."""
    order: list[str] = []

    async def slow(**kwargs: Any) -> dict[str, Any]:
        await asyncio.sleep(0.08)
        order.append("slow")
        return {"ok": True}

    async def fast(**kwargs: Any) -> dict[str, Any]:
        await asyncio.sleep(0.001)
        order.append("fast")
        return {"ok": True}

    async def after_fast(**kwargs: Any) -> dict[str, Any]:
        order.append("after_fast")
        return {"ok": True}

    registry.register("slow", slow)
    registry.register("fast", fast)
    registry.register("after_fast", after_fast)

    workflow = make_workflow(
        [
            make_step("slow_arm", tool="slow"),
            make_step("fast_arm", tool="fast"),
            make_step("after_fast", tool="after_fast", depends_on=["fast_arm"]),
        ]
    )
    await build_runner(workflow)

    # after_fast ran before slow finished; with level scheduling it could not.
    assert order.index("after_fast") < order.index("slow")


async def test_no_step_executes_twice(registry, build_runner, no_leaked_tasks):
    recorder = Recorder()
    recorder.install(registry)

    # A wide fan-in: many steps converge on one, which is where a naive
    # scheduler re-queues the join once per satisfied dependency.
    steps = [make_step("root", inputs={"label": "root"})]
    for index in range(6):
        steps.append(
            make_step(f"leaf{index}", depends_on=["root"], inputs={"label": f"leaf{index}"})
        )
    steps.append(
        make_step(
            "join",
            depends_on=[f"leaf{index}" for index in range(6)],
            inputs={"label": "join"},
        )
    )
    await build_runner(make_workflow(steps))

    assert recorder.calls["join"] == 1
    assert all(count == 1 for count in recorder.calls.values())


async def test_dependencies_complete_before_dependents_start(
    registry, build_runner, no_leaked_tasks
):
    recorder = Recorder()
    recorder.install(registry, delay=0.01)

    workflow = make_workflow(
        [
            make_step("a", inputs={"label": "a"}),
            make_step("b", inputs={"label": "b"}),
            make_step("c", depends_on=["a", "b"], inputs={"label": "c"}),
        ]
    )
    await build_runner(workflow)

    start_of_c = recorder.starts.index("c")
    finished_before_c = set(recorder.finishes[: recorder.finishes.index("c")])
    assert {"a", "b"} <= finished_before_c
    assert start_of_c == 2


async def test_bindings_flow_between_steps(registry, build_runner, no_leaked_tasks):
    async def produce(**kwargs: Any) -> dict[str, Any]:
        return {"value": 41}

    seen: dict[str, Any] = {}

    async def consume(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {"ok": True}

    registry.register("produce", produce)
    registry.register("consume", consume)

    workflow = make_workflow(
        [
            make_step("p", tool="produce"),
            make_step(
                "c",
                tool="consume",
                depends_on=["p"],
                inputs={"incoming": "${steps.p.output.value}"},
            ),
        ]
    )
    await build_runner(workflow)

    # The int survived the binding; it was not stringified.
    assert seen["incoming"] == 41


async def test_reduce_concatenates_map_output(registry, build_runner, no_leaked_tasks):
    async def listify(**kwargs: Any) -> dict[str, Any]:
        return {"items": [1, 2, 3]}

    async def double(**kwargs: Any) -> dict[str, Any]:
        return {"doubled": kwargs["value"] * 2}

    registry.register("listify", listify)
    registry.register("double", double)

    workflow = make_workflow(
        [
            make_step("src", tool="listify"),
            Step(
                id="fan",
                kind=StepKind.MAP,
                tool="double",
                depends_on=["src"],
                over="${steps.src.output.items}",
                inputs={"value": "${item}"},
            ),
            Step(
                id="join",
                kind=StepKind.REDUCE,
                reducer="concat",
                depends_on=["fan"],
                inputs={"values": "${steps.fan.output.items}"},
            ),
        ]
    )
    state = await build_runner(workflow)

    assert state.step("fan").output["count"] == 3
    assert state.step("join").output["result"] == [
        {"doubled": 2},
        {"doubled": 4},
        {"doubled": 6},
    ]


async def test_branch_skips_the_untaken_side(registry, build_runner, no_leaked_tasks):
    recorder = Recorder()
    recorder.install(registry)

    async def decide(**kwargs: Any) -> dict[str, Any]:
        return {"ok": True}

    registry.register("decide", decide)

    workflow = make_workflow(
        [
            make_step("root", tool="decide"),
            Step(
                id="gate",
                kind=StepKind.BRANCH,
                depends_on=["root"],
                when="${steps.root.output.ok}",
                on_true=["yes_path"],
                on_false=["no_path"],
            ),
            make_step("yes_path", depends_on=["gate"], inputs={"label": "yes"}),
            make_step("no_path", depends_on=["gate"], inputs={"label": "no"}),
        ]
    )
    state = await build_runner(workflow)

    assert state.step("yes_path").status is StepStatus.SUCCEEDED
    # Skipped, not deleted — the step still exists with a recorded reason.
    assert state.step("no_path").status is StepStatus.SKIPPED
    assert state.step("no_path").skipped_reason
    assert "no" not in recorder.calls


async def test_a_failed_step_skips_its_dependents_by_default(
    registry, build_runner, no_leaked_tasks
):
    recorder = Recorder()
    recorder.install(registry)

    async def boom(**kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("nope")

    registry.register("boom", boom)

    workflow = make_workflow(
        [
            make_step("ok", inputs={"label": "ok"}),
            make_step("bad", tool="boom"),
            make_step("after_bad", depends_on=["bad"], inputs={"label": "after_bad"}),
            make_step("after_ok", depends_on=["ok"], inputs={"label": "after_ok"}),
        ]
    )
    state = await build_runner(workflow)

    assert state.step("bad").status is StepStatus.FAILED
    assert state.step("after_bad").status is StepStatus.SKIPPED
    # The independent branch still completed — the point of SKIP over FAIL_FAST.
    assert state.step("after_ok").status is StepStatus.SUCCEEDED
    assert "after_bad" not in recorder.calls


async def test_fail_fast_stops_the_run(registry, build_runner, no_leaked_tasks):
    recorder = Recorder()
    recorder.install(registry, delay=0.05)

    async def boom(**kwargs: Any) -> dict[str, Any]:
        await asyncio.sleep(0.001)
        raise RuntimeError("nope")

    registry.register("boom", boom)

    workflow = make_workflow(
        [
            make_step("bad", tool="boom"),
            make_step("slow", inputs={"label": "slow"}),
        ],
        on_step_failure=DependentsPolicy.FAIL_FAST,
    )
    state = await build_runner(workflow)

    assert state.status.value == "failed"
    # The concurrent sibling was cancelled by the TaskGroup rather than allowed
    # to finish.
    assert state.step("slow").status in (StepStatus.CANCELLED, StepStatus.SKIPPED)


async def test_every_executed_step_has_a_start_and_a_terminal_event(
    registry, build_context, no_leaked_tasks
):
    """Trace completeness, asserted directly."""
    from skein.limits.concurrency import LimitSet
    from skein.runtime.scheduler import Scheduler
    from skein.trace.hashing import trace_completeness

    recorder = Recorder()
    recorder.install(registry)

    workflow = make_workflow(
        [
            make_step("a", inputs={"label": "a"}),
            make_step("b", depends_on=["a"], inputs={"label": "b"}),
        ]
    )
    context, state, _ = build_context(workflow, limits=LimitSet(global_limit=4))
    await Scheduler(context, LimitSet(global_limit=4)).run(state)

    started, terminated = trace_completeness(context.recorder.events)
    assert started == terminated == {"a", "b"}
