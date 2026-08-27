"""The engine: ingress queue, worker pool, run store, lifecycle.

This is what the API talks to. Submission puts a run on the bounded queue and
returns immediately; a pool of workers pulls from it and drives one run each.

Two levels of concurrency, and they are not the same thing:

* ``workers`` bounds how many *runs* execute at once.
* ``global_concurrency`` bounds how many *steps* execute at once, across all
  runs.

Both are needed. Workers alone would let eight runs each fan out to two hundred
steps; step limits alone would let ten thousand queued runs each hold a worker
slot's worth of state.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from skein.budget.guards import BudgetTracker
from skein.config import Settings
from skein.errors import SkeinError
from skein.limits.concurrency import LimitScope, LimitSet
from skein.limits.queue import BoundedIngressQueue
from skein.llm.base import LlmClient, StubLlmClient
from skein.llm.ollama import OllamaClient
from skein.model.validation import validate_workflow
from skein.model.workflow import Workflow
from skein.observability import metrics
from skein.observability.otel import mark_error, step_span, workflow_span
from skein.resilience.breaker import BreakerRegistry, CircuitBreakerConfig
from skein.resilience.clock import Clock, SystemClock
from skein.resilience.retry import RetryBudget
from skein.runtime.executor import ExecutionContext
from skein.runtime.scheduler import Scheduler
from skein.runtime.signals import ShutdownController
from skein.runtime.state import RunState, RunStatus
from skein.runtime.store import RunHandle, RunStore
from skein.tools.registry import ToolRegistry, default_registry
from skein.trace.recorder import TraceRecorder
from skein.trace.replay import ReplaySource

logger = logging.getLogger("skein.engine")


@dataclass
class EngineConfig:
    settings: Settings
    registry: ToolRegistry = field(default_factory=default_registry)
    llm: LlmClient | None = None
    clock: Clock = field(default_factory=SystemClock)


@dataclass
class _Submission:
    run: RunState
    recorder: TraceRecorder
    replay: ReplaySource | None = None


class Engine:
    def __init__(self, config: EngineConfig) -> None:
        self.settings = config.settings
        self.registry = config.registry
        self.clock = config.clock

        self.llm: LlmClient = config.llm or (
            StubLlmClient()
            if config.settings.use_stub_llm
            else OllamaClient(
                base_url=config.settings.ollama_url,
                default_model=config.settings.ollama_model,
                timeout_s=config.settings.llm_timeout_s,
            )
        )

        self.limits = LimitSet(
            global_limit=self.settings.global_concurrency,
            default_tool_limit=self.settings.default_tool_concurrency,
            tool_limits={
                name: tool.max_concurrency
                for name, tool in self.registry.tools.items()
                if tool.max_concurrency is not None
            },
        )
        self.breakers = BreakerRegistry(
            config=CircuitBreakerConfig(
                failure_rate_threshold=self.settings.breaker_failure_rate,
                window_s=self.settings.breaker_window_s,
                min_samples=self.settings.breaker_min_samples,
                open_duration_s=self.settings.breaker_open_duration_s,
            ),
            clock=self.clock,
        )
        self.queue: BoundedIngressQueue[_Submission] = BoundedIngressQueue(
            maxsize=self.settings.queue_size,
            policy=self.settings.overflow,
        )
        self.store = RunStore(max_completed=self.settings.max_completed_runs)
        self.shutdown = ShutdownController(
            drain_timeout_s=self.settings.drain_timeout_s,
            cancel_timeout_s=self.settings.cancel_timeout_s,
        )

        self._workers: list[asyncio.Task[None]] = []
        self._started = False

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        self.shutdown.install()

        metrics.observe_queue(self.queue.qsize(), self.queue.maxsize)
        metrics.observe_concurrency(
            "global", 0, 0, self.settings.global_concurrency
        )

        # Workers are long-lived tasks held on the engine rather than in a
        # TaskGroup, because their lifetime is the process's, not a block's.
        # They are explicitly cancelled and awaited in stop(); that pairing is
        # what keeps them from becoming the leaked tasks this project is about.
        for index in range(self.settings.workers):
            self._workers.append(
                asyncio.create_task(self._worker(index), name=f"skein-worker-{index}")
            )
        logger.info("engine started with %d worker(s)", self.settings.workers)

    async def stop(self) -> None:
        """Drain, then cancel workers, then close clients."""
        await self.shutdown.drain(
            active_count=lambda: len(self.store.active()),
            cancel_all=self._cancel_all,
        )

        for worker in self._workers:
            worker.cancel()
        # return_exceptions=True: every worker will raise CancelledError and
        # gather would otherwise propagate the first one before the rest have
        # finished unwinding, leaving tasks mid-cleanup at process exit.
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

        with contextlib.suppress(Exception):
            await self.llm.close()
        self._started = False
        logger.info("engine stopped")

    async def _cancel_all(self) -> None:
        for handle in self.store.active():
            with contextlib.suppress(Exception):
                await self.store.cancel(handle.state.run_id, timeout_s=2.0)

    # -- submission --------------------------------------------------------

    async def submit(
        self,
        workflow: Workflow,
        inputs: dict[str, Any] | None = None,
        replay_from: Path | None = None,
    ) -> RunState:
        """Validate and enqueue. Raises rather than queueing invalid work."""
        try:
            validate_workflow(workflow)
        except SkeinError as exc:
            metrics.validation_rejections.labels(code=exc.code).inc()
            raise

        run = RunState(workflow=workflow, inputs=inputs or {})
        trace_path = (
            self.settings.trace_dir / f"{run.run_id}.jsonl"
            if self.settings.trace_to_disk
            else None
        )
        recorder = TraceRecorder(run.run_id, trace_path)
        self.store.add(run, recorder)

        replay = ReplaySource.from_file(replay_from) if replay_from else None

        try:
            await self.queue.put(_Submission(run=run, recorder=recorder, replay=replay))
        except SkeinError:
            metrics.queue_rejections.inc()
            # Remove the run we just registered — leaving it would show a run
            # that is permanently QUEUED and never executes.
            self.store.mark_finished(run.run_id)
            run.status = RunStatus.FAILED
            run.error_code = "queue_full"
            raise
        metrics.observe_queue(self.queue.qsize(), self.queue.maxsize)
        return run

    # -- workers -----------------------------------------------------------

    async def _worker(self, index: int) -> None:
        while True:
            submission = await self.queue.get()
            metrics.observe_queue(self.queue.qsize(), self.queue.maxsize)
            try:
                await self._execute(submission)
            except asyncio.CancelledError:
                # The worker itself is being shut down. Mark the run and
                # re-raise so the task actually stops; returning would keep the
                # worker looping through a shutdown.
                submission.run.status = RunStatus.CANCELLED
                self.store.mark_finished(submission.run.run_id)
                raise
            except Exception:  # noqa: BLE001 - a worker must outlive a bad run
                # A failure here is a bug in the runtime rather than in the
                # workflow (the scheduler handles workflow failures itself).
                # Logged and swallowed so one poisoned run does not take the
                # worker down and reduce capacity permanently.
                logger.exception("worker %d: run %s crashed", index, submission.run.run_id)
            finally:
                self.queue.task_done()

    async def _execute(self, submission: _Submission) -> None:
        run = submission.run
        handle = self.store.get(run.run_id)
        handle.task = asyncio.current_task()

        context = ExecutionContext(
            run_id=run.run_id,
            workflow=run.workflow,
            registry=self.registry,
            llm=self.llm,
            breakers=self.breakers,
            clock=self.clock,
            recorder=submission.recorder,
            budget=BudgetTracker(budget=run.workflow.budget, clock=self.clock),
            retry_budget=RetryBudget(limit=run.workflow.retry_budget),
            run_inputs=run.inputs,
            replay=submission.replay,
        )
        scheduler = Scheduler(context, self.limits)

        try:
            with workflow_span(run.run_id, run.workflow.name) as span:
                try:
                    await scheduler.run(run)
                except SkeinError as exc:
                    mark_error(span, exc)
                    raise
        except asyncio.CancelledError:
            raise
        except SkeinError:
            # Already reflected in run.status by the scheduler.
            pass
        finally:
            self._publish_limit_metrics(run.run_id)
            metrics.run_duration.labels(
                workflow=run.workflow.name, status=run.status.value
            ).observe(run.duration_s or 0.0)
            self.store.mark_finished(run.run_id)

    def _publish_limit_metrics(self, run_id: str) -> None:
        observer = self.limits.observer
        metrics.observe_concurrency(
            "global",
            observer.current(LimitScope.GLOBAL),
            observer.peak(LimitScope.GLOBAL),
            self.settings.global_concurrency,
        )
        for tool in self.registry.names():
            breaker = self.breakers.get(tool)
            metrics.observe_breaker(tool, breaker.state.value)

    # -- inspection --------------------------------------------------------

    def health(self) -> dict[str, Any]:
        return {
            "status": "draining" if self.shutdown.is_draining() else "ok",
            "workers": len(self._workers),
            "queue": {
                "depth": self.queue.qsize(),
                "capacity": self.queue.maxsize,
                "policy": self.queue.policy.value,
                "rejected": self.queue.stats.rejected,
            },
            "runs": self.store.snapshot(),
            "concurrency": self.limits.observer.snapshot(),
            "breakers": self.breakers.snapshot(),
        }

    async def ready(self) -> bool:
        """Readiness is stricter than liveness.

        A draining instance is alive but must not receive new traffic, so it
        reports not-ready and the load balancer removes it *before* the drain
        window starts. That ordering is what makes a rolling restart lossless.
        """
        if self.shutdown.is_draining():
            return False
        if not self._started or not self._workers:
            return False
        return not self.queue.full()


__all__ = ["Engine", "EngineConfig", "RunHandle", "step_span"]
