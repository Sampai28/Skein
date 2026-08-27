"""The scheduler — topological execution at maximal safe concurrency.

The rule is simple and the implementation exists to keep it true: **a step
becomes eligible the moment its dependencies settle**, not when its "level" of
the graph completes. Level-by-level execution is the obvious implementation and
it is wrong — a diamond where one arm takes 10 s and the other 100 ms makes the
fast arm's dependents wait 10 s for no reason.

Structured concurrency throughout. Every task is created inside an
``asyncio.TaskGroup``, so:

* the block cannot exit while a child is still running;
* a child that raises cancels its siblings and the group re-raises;
* cancelling the run cancels every child, transitively.

That last property is what makes the cancellation contract enforceable. With
bare ``asyncio.create_task`` the runtime would hold a set of task handles and be
responsible for cancelling each one, awaiting each one, and handling the case
where cancellation itself raises — which is exactly the bookkeeping that leaks
tasks in practice.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from typing import Any

from skein.budget.guards import BudgetTracker
from skein.errors import (
    BudgetExceededError,
    DependencyFailedError,
    SkeinError,
    WorkflowTimeoutError,
)
from skein.limits.concurrency import LimitSet
from skein.model.workflow import DependentsPolicy, Step, StepKind
from skein.observability import metrics
from skein.resilience.clock import Clock
from skein.resilience.retry import RetryBudget
from skein.runtime.executor import ExecutionContext, StepExecutor
from skein.runtime.state import RunState, RunStatus
from skein.trace.events import EventKind, StepStatus


class _StepFailed(Exception):
    """Internal signal used to trip fail-fast without losing the cause."""

    def __init__(self, step_id: str, cause: BaseException) -> None:
        super().__init__(f"step {step_id} failed")
        self.step_id = step_id
        self.cause = cause


class Scheduler:
    def __init__(self, context: ExecutionContext, limits: LimitSet) -> None:
        self.ctx = context
        self.limits = limits
        self.executor = StepExecutor(context)
        self._completed: asyncio.Queue[str] = asyncio.Queue()
        self._inflight: int = 0

    # -- entry point -------------------------------------------------------

    async def run(self, run: RunState) -> RunState:
        workflow = run.workflow
        budget: BudgetTracker = self.ctx.budget
        self.limits.register_workflow(run.run_id, workflow.max_concurrency)

        run.status = RunStatus.RUNNING
        run.started_at = self.ctx.clock.now()
        self.ctx.recorder.record(
            EventKind.RUN_STARTED,
            payload={"workflow": workflow.name, "steps": len(workflow.steps)},
        )
        metrics.runs_started.labels(workflow=workflow.name).inc()

        try:
            async with asyncio.timeout(workflow.budget.max_duration_s):
                await self._drive(run)
        except TimeoutError as exc:
            self._finalise_incomplete(run, StepStatus.CANCELLED, "workflow_timeout")
            run.status = RunStatus.FAILED
            run.error_code = "workflow_timeout"
            run.error_message = (
                f"workflow exceeded its duration budget of {workflow.budget.max_duration_s}s"
            )
            self._record_finish(run)
            raise WorkflowTimeoutError(run.error_message, run_id=run.run_id) from exc
        except asyncio.CancelledError:
            # Cancellation is not an error to be reported and swallowed — it is
            # a control signal. Mark state, flush the trace, then RE-RAISE so
            # the caller (and any enclosing TaskGroup) sees the cancellation it
            # asked for. Returning normally here would make `await cancel()`
            # look like it had failed to take effect.
            self._finalise_incomplete(run, StepStatus.CANCELLED, "cancelled")
            run.status = RunStatus.CANCELLED
            run.error_code = "cancelled"
            run.error_message = "run was cancelled"
            self._record_finish(run)
            metrics.runs_cancelled.labels(workflow=workflow.name).inc()
            raise
        except _StepFailed as exc:
            self._finalise_incomplete(run, StepStatus.SKIPPED, "fail_fast")
            run.status = RunStatus.FAILED
            run.error_code = getattr(exc.cause, "code", type(exc.cause).__name__)
            run.error_message = str(exc.cause)
            self._record_finish(run)
            metrics.runs_failed.labels(
                workflow=workflow.name, code=run.error_code or "unknown"
            ).inc()
            return run
        except BudgetExceededError as exc:
            self._finalise_incomplete(run, StepStatus.SKIPPED, "budget_exceeded")
            run.status = RunStatus.FAILED
            run.error_code = exc.code
            run.error_message = exc.message
            self.ctx.recorder.record(
                EventKind.BUDGET_TRIPPED,
                error_code=exc.code,
                error_message=exc.message,
                payload=dict(exc.details),
            )
            self._record_finish(run)
            metrics.budget_trips.labels(
                workflow=workflow.name, guard=str(exc.details.get("guard", "unknown"))
            ).inc()
            return run
        else:
            failed = [
                state for state in run.steps.values() if state.status is StepStatus.FAILED
            ]
            run.status = RunStatus.FAILED if failed else RunStatus.SUCCEEDED
            if failed:
                run.error_code = failed[0].error_code
                run.error_message = failed[0].error_message
                metrics.runs_failed.labels(
                    workflow=workflow.name, code=run.error_code or "unknown"
                ).inc()
            else:
                metrics.runs_succeeded.labels(workflow=workflow.name).inc()
            self._record_finish(run)
            return run
        finally:
            # Always released, on every path including cancellation. Left
            # behind, the per-workflow semaphore is a slow memory leak keyed by
            # run id.
            self.limits.release_workflow(run.run_id)

    # -- the loop ----------------------------------------------------------

    async def _drive(self, run: RunState) -> None:
        """Spawn eligible steps, wait for one to settle, repeat."""
        async with asyncio.TaskGroup() as group:
            while True:
                self.ctx.budget.check_all()

                for step in self._eligible(run):
                    run.step(step.id).status = StepStatus.RUNNING
                    self._inflight += 1
                    # create_task inside the group: the task is owned by the
                    # group, so it is cancelled if anything else fails and
                    # awaited before the block exits. No handle bookkeeping.
                    group.create_task(self._run_step(run, step), name=f"step:{step.id}")

                if self._inflight == 0:
                    # Nothing running and nothing eligible. Either everything is
                    # terminal or the remainder is unreachable because its
                    # dependencies did not succeed; _settle_unreachable resolves
                    # the second case so the loop cannot spin.
                    if run.all_terminal():
                        return
                    if not self._settle_unreachable(run):
                        return
                    continue

                await self._completed.get()
                self._inflight -= 1

    def _eligible(self, run: RunState) -> list[Step]:
        """Steps whose dependencies have all settled favourably."""
        ready: list[Step] = []
        for step in run.workflow.steps:
            state = run.step(step.id)
            if state.status is not StepStatus.PENDING:
                continue
            if all(
                run.step(dep).status is StepStatus.SUCCEEDED for dep in step.depends_on
            ):
                ready.append(step)
        return ready

    def _settle_unreachable(self, run: RunState) -> bool:
        """Mark steps that can never run, given how their dependencies ended.

        Returns True if anything changed. Without this the loop would sit with
        zero in-flight tasks and a non-empty pending set forever.
        """
        changed = False
        for step in run.workflow.steps:
            state = run.step(step.id)
            if state.status is not StepStatus.PENDING:
                continue
            blocking = [
                dep
                for dep in step.depends_on
                if run.step(dep).status.is_terminal
                and run.step(dep).status is not StepStatus.SUCCEEDED
            ]
            if blocking:
                self._skip(run, step.id, f"dependency did not succeed: {blocking[0]}")
                changed = True
        return changed

    # -- one step ----------------------------------------------------------

    async def _run_step(self, run: RunState, step: Step) -> None:
        state = run.step(step.id)
        tool_name = step.tool if step.kind in (StepKind.TOOL_CALL, StepKind.MAP) else None

        try:
            self.ctx.budget.reserve_step()
        except BudgetExceededError:
            state.status = StepStatus.PENDING
            self._inflight_done(step.id)
            raise

        state.started_at = self.ctx.clock.now()
        self.ctx.recorder.record(
            EventKind.STEP_STARTED, step_id=step.id, target=tool_name or step.model
        )
        metrics.steps_started.labels(kind=step.kind.value, tool=tool_name or "-").inc()

        try:
            # Permits held for the whole execution, released on every exit path
            # including cancellation. Acquiring inside the task rather than
            # before spawning is what makes the cap a limit on *running* steps
            # rather than on scheduled ones.
            async with self.limits.acquire(run.run_id, tool_name):
                output = await self.executor.execute(step, run.outputs())

            state.output = output
            state.status = StepStatus.SUCCEEDED
            state.finished_at = self.ctx.clock.now()
            self.ctx.recorder.record(
                EventKind.STEP_SUCCEEDED,
                step_id=step.id,
                status=StepStatus.SUCCEEDED,
                target=tool_name or step.model,
                duration_s=state.duration_s,
                payload={"output": output},
            )
            metrics.steps_succeeded.labels(
                kind=step.kind.value, tool=tool_name or "-"
            ).inc()
            metrics.step_duration.labels(
                kind=step.kind.value, tool=tool_name or "-"
            ).observe(state.duration_s or 0.0)

            if step.kind is StepKind.BRANCH:
                self._apply_branch(run, step, output)

        except asyncio.CancelledError:
            state.status = StepStatus.CANCELLED
            state.finished_at = self.ctx.clock.now()
            # shield: this write must complete even though the surrounding task
            # is being torn down. It is one buffered write to an already-open
            # file, so the shield cannot meaningfully delay shutdown, and
            # without it a cancelled step leaves no terminal event and the trace
            # -completeness invariant fails for a run that behaved correctly.
            await asyncio.shield(
                self._record_terminal(step, state, EventKind.STEP_CANCELLED, tool_name)
            )
            metrics.steps_cancelled.labels(
                kind=step.kind.value, tool=tool_name or "-"
            ).inc()
            self._inflight_done(step.id)
            # Re-raised, always. Swallowing it would leave the TaskGroup
            # believing this child exited normally while the loop above has
            # already been told to stop.
            raise

        except BudgetExceededError:
            state.status = StepStatus.FAILED
            state.finished_at = self.ctx.clock.now()
            self._inflight_done(step.id)
            raise

        except Exception as exc:  # noqa: BLE001 - classified below
            state.status = StepStatus.FAILED
            state.finished_at = self.ctx.clock.now()
            state.error_code = getattr(exc, "code", type(exc).__name__)
            state.error_message = str(exc)
            self.ctx.recorder.record(
                EventKind.STEP_FAILED,
                step_id=step.id,
                status=StepStatus.FAILED,
                target=tool_name or step.model,
                duration_s=state.duration_s,
                error_code=state.error_code,
                error_message=state.error_message,
            )
            metrics.steps_failed.labels(
                kind=step.kind.value,
                tool=tool_name or "-",
                code=state.error_code or "unknown",
            ).inc()

            self._apply_failure_policy(run, step, exc)
            self._inflight_done(step.id)
            if run.workflow.on_step_failure is DependentsPolicy.FAIL_FAST:
                # Raising inside the TaskGroup cancels every sibling, which is
                # precisely what fail-fast means.
                raise _StepFailed(step.id, exc) from exc
            return

        else:
            self._inflight_done(step.id)

    async def _record_terminal(
        self, step: Step, state: Any, kind: EventKind, tool_name: str | None
    ) -> None:
        self.ctx.recorder.record(
            kind,
            step_id=step.id,
            status=state.status,
            target=tool_name or step.model,
            duration_s=state.duration_s,
        )

    def _inflight_done(self, step_id: str) -> None:
        # put_nowait on an unbounded queue: this is the scheduler talking to
        # itself and must never block, least of all inside a cancellation
        # handler where an await could be interrupted again.
        self._completed.put_nowait(step_id)

    # -- policies ----------------------------------------------------------

    def _apply_failure_policy(self, run: RunState, step: Step, exc: BaseException) -> None:
        policy = run.workflow.on_step_failure
        if policy is DependentsPolicy.CONTINUE:
            # Dependents stay pending and will become eligible only if their
            # other dependencies succeed. They will then fail at binding
            # resolution if they read this step's output, which is the honest
            # outcome — CONTINUE means "let them try", not "pretend it worked".
            return
        if policy is DependentsPolicy.SKIP:
            for dependent in self._transitive_dependents(run, step.id):
                self._skip(
                    run, dependent, f"dependency {step.id!r} failed: {type(exc).__name__}"
                )

    def _apply_branch(self, run: RunState, step: Step, output: dict[str, Any]) -> None:
        """Skip the untaken side of a branch.

        The untaken steps are marked SKIPPED rather than removed from the graph.
        Removing them would change the step set between a recording and its
        replay, so their trajectory hashes could never match; and it would break
        "no step executes twice" by making the invariant unstatable for steps
        that no longer exist.
        """
        for target in output.get("not_taken", []):
            self._skip(run, target, f"branch {step.id!r} took the other path")
            for dependent in self._transitive_dependents(run, target):
                self._skip(run, dependent, f"upstream branch target {target!r} was skipped")

    def _skip(self, run: RunState, step_id: str, reason: str) -> None:
        state = run.step(step_id)
        if state.status is not StepStatus.PENDING:
            return
        state.status = StepStatus.SKIPPED
        state.skipped_reason = reason
        state.finished_at = self.ctx.clock.now()
        state.error_code = DependencyFailedError.code
        state.error_message = reason
        self.ctx.recorder.record(
            EventKind.STEP_SKIPPED,
            step_id=step_id,
            status=StepStatus.SKIPPED,
            error_code=DependencyFailedError.code,
            error_message=reason,
        )
        metrics.steps_skipped.labels(reason="dependency").inc()

    def _transitive_dependents(self, run: RunState, step_id: str) -> list[str]:
        dependents: dict[str, list[str]] = defaultdict(list)
        for step in run.workflow.steps:
            for dependency in step.depends_on:
                dependents[dependency].append(step.id)

        found: list[str] = []
        seen: set[str] = {step_id}
        queue = deque(dependents[step_id])
        while queue:
            current = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            found.append(current)
            queue.extend(dependents[current])
        return found

    # -- finalisation ------------------------------------------------------

    def _finalise_incomplete(
        self, run: RunState, status: StepStatus, reason: str
    ) -> None:
        """Give every non-terminal step a terminal status.

        Trace completeness requires a terminal event for anything that started;
        run reporting requires a status for everything else. A step still
        PENDING when the run ends did not run, so it is recorded as such rather
        than left in a state that would read as "still going" long after the run
        is over.
        """
        for step_id, state in run.steps.items():
            if state.status.is_terminal:
                continue
            if state.status is StepStatus.RUNNING:
                state.status = status
                state.finished_at = self.ctx.clock.now()
                self.ctx.recorder.record(
                    EventKind.STEP_CANCELLED
                    if status is StepStatus.CANCELLED
                    else EventKind.STEP_SKIPPED,
                    step_id=step_id,
                    status=status,
                    error_message=reason,
                )
            else:
                self._skip(run, step_id, reason)

    def _record_finish(self, run: RunState) -> None:
        run.finished_at = self.ctx.clock.now()
        run.trajectory_hash = self.ctx.recorder.trajectory_hash()
        self.ctx.recorder.record(
            EventKind.RUN_FINISHED,
            payload={
                "status": run.status.value,
                "counts": run.counts(),
                "trajectory_hash": run.trajectory_hash,
                "budget": self.ctx.budget.snapshot(),
            },
            duration_s=run.duration_s,
            error_code=run.error_code,
            error_message=run.error_message,
        )


def build_context(**kwargs: Any) -> ExecutionContext:
    """Convenience constructor kept next to its only caller's expectations."""
    kwargs.setdefault("retry_budget", RetryBudget(limit=20))
    return ExecutionContext(**kwargs)


__all__ = ["Scheduler", "build_context", "SkeinError", "Clock"]
