"""Three independent concurrency limits, acquired in a fixed global order.

Skein caps in-flight steps three ways: globally across the process, per
workflow run, and per tool. Each is a plain ``asyncio.Semaphore``.

**The ordering is the important part.** Acquiring three semaphores in an
arbitrary order is a deadlock waiting for load: run A holds the global permit
and waits for the ``fetch`` permit while run B holds ``fetch`` and waits for the
global one. Neither can proceed and nothing times out, because both are blocked
on an ``acquire`` that never fails. Establishing one total order —
global, then workflow, then tool — makes the cycle impossible to construct,
which is why :class:`LimitSet` owns acquisition rather than exposing the
semaphores for callers to take themselves.

A semaphore is the right primitive here rather than a bounded queue because
these limits govern *simultaneity* of work already accepted for execution. The
bounded queue in :mod:`skein.limits.queue` governs *admission*, which is a
different question: it decides whether to take the work at all. Using a queue
for both would mean a step that cannot start right now is indistinguishable
from a workflow that was never accepted.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator


class LimitScope(str, Enum):
    GLOBAL = "global"
    WORKFLOW = "workflow"
    TOOL = "tool"


@dataclass
class _ScopeCounters:
    current: int = 0
    peak: int = 0
    total_acquired: int = 0
    total_waited: int = 0


class ConcurrencyObserver:
    """Records what concurrency actually happened, not what was configured.

    Tests assert against ``peak``. That distinction matters: a test that only
    checks the semaphore was constructed with the right value proves nothing
    about whether the code holds the permit for the whole duration of the work,
    which is the bug that actually occurs.

    Not thread-safe and does not need to be — every mutation happens on the
    event loop thread between awaits, so the increment and the peak comparison
    cannot interleave.
    """

    def __init__(self) -> None:
        self._scopes: dict[tuple[LimitScope, str], _ScopeCounters] = {}

    def _counters(self, scope: LimitScope, key: str) -> _ScopeCounters:
        return self._scopes.setdefault((scope, key), _ScopeCounters())

    def enter(self, scope: LimitScope, key: str) -> None:
        counters = self._counters(scope, key)
        counters.current += 1
        counters.total_acquired += 1
        counters.peak = max(counters.peak, counters.current)

    def exit(self, scope: LimitScope, key: str) -> None:
        counters = self._counters(scope, key)
        counters.current -= 1

    def record_wait(self, scope: LimitScope, key: str) -> None:
        self._counters(scope, key).total_waited += 1

    def current(self, scope: LimitScope, key: str = "") -> int:
        return self._counters(scope, key).current

    def peak(self, scope: LimitScope, key: str = "") -> int:
        return self._counters(scope, key).peak

    def waited(self, scope: LimitScope, key: str = "") -> int:
        return self._counters(scope, key).total_waited

    def snapshot(self) -> dict[str, dict[str, int]]:
        return {
            f"{scope.value}:{key}" if key else scope.value: {
                "current": counters.current,
                "peak": counters.peak,
                "acquired": counters.total_acquired,
                "waited": counters.total_waited,
            }
            for (scope, key), counters in self._scopes.items()
        }

    def reset_peaks(self) -> None:
        for counters in self._scopes.values():
            counters.peak = counters.current


@dataclass
class LimitSet:
    """Owns every semaphore and the only correct way to acquire them."""

    global_limit: int = 32
    default_tool_limit: int = 8
    tool_limits: dict[str, int] = field(default_factory=dict)
    observer: ConcurrencyObserver = field(default_factory=ConcurrencyObserver)

    _global: asyncio.Semaphore = field(init=False)
    _per_workflow: dict[str, asyncio.Semaphore] = field(init=False, default_factory=dict)
    _per_tool: dict[str, asyncio.Semaphore] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._global = asyncio.Semaphore(self.global_limit)

    def register_workflow(self, run_id: str, limit: int) -> None:
        self._per_workflow[run_id] = asyncio.Semaphore(limit)

    def release_workflow(self, run_id: str) -> None:
        """Drop the run's semaphore once it is finished.

        Without this the dict grows for the life of the process — a slow leak
        that only shows up after a few hundred thousand runs, which is exactly
        the kind of thing that never appears in testing.
        """
        self._per_workflow.pop(run_id, None)

    def _tool_semaphore(self, tool: str) -> asyncio.Semaphore:
        if tool not in self._per_tool:
            limit = self.tool_limits.get(tool, self.default_tool_limit)
            self._per_tool[tool] = asyncio.Semaphore(limit)
        return self._per_tool[tool]

    @asynccontextmanager
    async def acquire(self, run_id: str, tool: str | None = None) -> AsyncIterator[None]:
        """Take every applicable permit, in order, and release in reverse.

        Cancellation safety: ``asyncio.Semaphore.acquire`` either completes or
        raises, and the ``finally`` blocks below unwind whatever was taken. A
        cancellation arriving while waiting on the tool permit still releases
        the global and workflow permits on the way out, so a cancelled run
        cannot strand capacity.
        """
        workflow_semaphore = self._per_workflow.get(run_id)
        tool_semaphore = self._tool_semaphore(tool) if tool else None

        await self._acquire_one(self._global, LimitScope.GLOBAL, "")
        try:
            if workflow_semaphore is not None:
                await self._acquire_one(workflow_semaphore, LimitScope.WORKFLOW, run_id)
            try:
                if tool_semaphore is not None:
                    await self._acquire_one(tool_semaphore, LimitScope.TOOL, tool or "")
                try:
                    yield
                finally:
                    if tool_semaphore is not None:
                        tool_semaphore.release()
                        self.observer.exit(LimitScope.TOOL, tool or "")
            finally:
                if workflow_semaphore is not None:
                    workflow_semaphore.release()
                    self.observer.exit(LimitScope.WORKFLOW, run_id)
        finally:
            self._global.release()
            self.observer.exit(LimitScope.GLOBAL, "")

    async def _acquire_one(
        self, semaphore: asyncio.Semaphore, scope: LimitScope, key: str
    ) -> None:
        if semaphore.locked():
            # `locked()` is true when the counter is zero, i.e. this acquire is
            # going to wait. Recorded so the metrics can distinguish "the limit
            # is doing something" from "the limit is set high enough never to
            # bind", which look identical from peak concurrency alone.
            self.observer.record_wait(scope, key)
        await semaphore.acquire()
        self.observer.enter(scope, key)
