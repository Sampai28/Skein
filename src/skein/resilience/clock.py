"""An injectable clock, so timing behaviour can be tested without waiting.

Every part of Skein that measures or waits takes a :class:`Clock` rather than
calling ``asyncio.sleep`` or ``time.monotonic`` directly. In production that is
:class:`SystemClock` and costs nothing. In tests it is :class:`VirtualClock`,
where a five-minute backoff sequence is verified in microseconds and — more
importantly — *deterministically*.

The alternative is tests that sleep. Those tests are slow, and they are flaky in
a specific and maddening way: a CI runner under load turns a 100 ms sleep into
130 ms, and an assertion about retry timing fails for reasons unrelated to the
code. Virtual time removes the scheduler from the assertion entirely.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import time
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """The only time surface the rest of the runtime is allowed to use."""

    def now(self) -> float:
        """Monotonic seconds. Only differences are meaningful."""
        ...

    async def sleep(self, seconds: float) -> None:
        ...


class SystemClock:
    """Real time. ``monotonic`` rather than ``time()`` because the wall clock
    can step backwards over an NTP correction, which would make a timeout
    computed as ``deadline - now()`` briefly negative."""

    def now(self) -> float:
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        if seconds > 0:
            await asyncio.sleep(seconds)


class VirtualClock:
    """Time that only moves when a test moves it.

    Sleepers are held in a heap keyed by deadline. :meth:`advance` moves the
    clock, wakes everything now due, and yields to the event loop so the woken
    coroutines actually get to run before ``advance`` returns — without that
    final yield a test would advance the clock and immediately assert on state
    that has not been updated yet, which is the single most confusing failure
    mode of a fake clock.

    Not safe across threads; like everything else here it assumes a single
    event loop.
    """

    def __init__(self, start: float = 0.0) -> None:
        self._now = start
        # (deadline, sequence, future). The sequence counter breaks ties so the
        # heap never has to compare two Futures, which are not orderable.
        self._waiters: list[tuple[float, int, asyncio.Future[None]]] = []
        self._counter = itertools.count()
        self.slept: list[float] = []

    def now(self) -> float:
        return self._now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        if seconds <= 0:
            # Still yield. A zero sleep that returns synchronously would let a
            # tight retry loop starve the event loop entirely.
            await asyncio.sleep(0)
            return

        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[None] = loop.create_future()
        heapq.heappush(self._waiters, (self._now + seconds, next(self._counter), waiter))
        try:
            await waiter
        except asyncio.CancelledError:
            # Drop the cancelled waiter so `advance` does not later try to
            # resolve a future nobody is awaiting. Re-raise: swallowing this
            # would leave the cancelling caller believing the sleep completed.
            self._waiters = [entry for entry in self._waiters if entry[2] is not waiter]
            heapq.heapify(self._waiters)
            raise

    async def advance(self, seconds: float) -> None:
        """Move time forward, waking whatever becomes due."""
        if seconds < 0:
            raise ValueError("virtual time does not move backwards")
        target = self._now + seconds

        while self._waiters and self._waiters[0][0] <= target:
            deadline, _, waiter = heapq.heappop(self._waiters)
            # Step the clock to each waiter's deadline in turn rather than
            # jumping straight to the target, so a coroutine that reads now()
            # after waking sees its own deadline and not the end of the window.
            self._now = deadline
            if not waiter.done():
                waiter.set_result(None)
            # Let the woken coroutine run before waking the next one; ordering
            # between sleepers is otherwise unobservable and tests that assert
            # on sequence become order-dependent on heap internals.
            await asyncio.sleep(0)

        self._now = target
        # A couple of extra turns so anything woken above can progress through
        # its own awaits before the caller inspects state.
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    @property
    def pending_sleepers(self) -> int:
        return len(self._waiters)
