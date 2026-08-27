"""In-memory run store.

A dict with a bounded history and the cancellation handles. Deliberately not a
database: persistence is a stated limitation of this runtime, and a half-real
store that survives some restarts and not others would be worse than an honest
in-memory one.

The bound matters. An unbounded dict of completed runs, each holding a full
trace, is a memory leak whose rate is proportional to throughput — the classic
way a long-running service dies after a week rather than in testing.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field

from skein.errors import RunNotFoundError
from skein.runtime.state import RunState, RunStatus
from skein.trace.recorder import TraceRecorder


@dataclass
class RunHandle:
    state: RunState
    recorder: TraceRecorder
    #: The task executing this run, so it can be cancelled. Held weakly in
    #: spirit — cleared once terminal, because keeping a reference to a finished
    #: task keeps its whole frame stack alive.
    task: asyncio.Task[None] | None = None
    #: Set when the run reaches a terminal state. Lets the drain wait on
    #: completion without polling.
    finished: asyncio.Event = field(default_factory=asyncio.Event)


class RunStore:
    def __init__(self, max_completed: int = 500) -> None:
        self._runs: OrderedDict[str, RunHandle] = OrderedDict()
        self._max_completed = max_completed

    def add(self, state: RunState, recorder: TraceRecorder) -> RunHandle:
        handle = RunHandle(state=state, recorder=recorder)
        self._runs[state.run_id] = handle
        return handle

    def get(self, run_id: str) -> RunHandle:
        handle = self._runs.get(run_id)
        if handle is None:
            raise RunNotFoundError(f"no run with id {run_id!r}", run_id=run_id)
        return handle

    def has(self, run_id: str) -> bool:
        return run_id in self._runs

    def active(self) -> list[RunHandle]:
        return [
            handle for handle in self._runs.values() if not handle.state.status.is_terminal
        ]

    def all(self) -> list[RunHandle]:
        return list(self._runs.values())

    def mark_finished(self, run_id: str) -> None:
        handle = self._runs.get(run_id)
        if handle is None:
            return
        handle.task = None
        handle.finished.set()
        handle.recorder.close()
        self._evict()

    def _evict(self) -> None:
        """Drop the oldest terminal runs once over the cap.

        Only terminal runs are evictable — evicting a running one would orphan
        its task and lose the only handle able to cancel it.
        """
        terminal = [
            run_id
            for run_id, handle in self._runs.items()
            if handle.state.status.is_terminal
        ]
        overflow = len(terminal) - self._max_completed
        for run_id in terminal[: max(0, overflow)]:
            self._runs.pop(run_id, None)

    async def cancel(self, run_id: str, timeout_s: float = 10.0) -> RunState:
        """Cancel a run and wait for it to actually stop.

        Waiting is the point. ``task.cancel()`` only *requests* cancellation —
        it schedules a ``CancelledError`` at the task's next await. Returning
        without awaiting would tell the caller the run is cancelled while its
        cleanup is still running, and a subsequent ``GET`` could still show it
        as RUNNING.
        """
        handle = self.get(run_id)
        if handle.state.status.is_terminal:
            return handle.state

        if handle.task is not None:
            handle.task.cancel()
            try:
                async with asyncio.timeout(timeout_s):
                    await handle.finished.wait()
            except TimeoutError:
                # The run did not stop within the deadline — a tool ignoring
                # cancellation. The state is recorded as cancelled regardless,
                # because from the caller's point of view it is; the stuck task
                # is surfaced through the leaked-task gauge rather than hidden.
                handle.state.status = RunStatus.CANCELLED
                handle.state.error_code = "cancel_timeout"
                handle.state.error_message = (
                    f"run did not stop within {timeout_s}s of cancellation"
                )
        else:
            handle.state.status = RunStatus.CANCELLED
            handle.finished.set()

        return handle.state

    def snapshot(self) -> dict[str, int]:
        counts: dict[str, int] = {status.value: 0 for status in RunStatus}
        for handle in self._runs.values():
            counts[handle.state.status.value] += 1
        counts["total"] = len(self._runs)
        return counts
