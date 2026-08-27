"""The bounded ingress queue — admission control for submitted workflows.

This is where backpressure lives. Submissions arrive faster than the runtime
drains them, and something has to give; the only question is what.

``REJECT`` (the default) answers immediately with 429 and a ``Retry-After``.
``BLOCK`` holds the producer until space appears.

**Why reject is the default.** Blocking looks kinder and is worse under exactly
the conditions backpressure exists for. A blocked submission occupies an ASGI
worker and its socket for as long as it waits, so a sustained overload converts
into exhausted server capacity and the service stops answering *everything*,
including the health checks that would have told an autoscaler to add capacity.
Rejecting keeps the cost of an overload proportional to the overload and leaves
the decision — retry, shed, queue elsewhere — with the caller, who has context
the runtime does not. ``BLOCK`` remains available for a trusted in-process
producer that genuinely has nowhere else to put the work.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from skein.errors import QueueFullError
from skein.model.workflow import OverflowPolicy

T = TypeVar("T")


@dataclass
class QueueStats:
    depth: int = 0
    max_depth_seen: int = 0
    enqueued: int = 0
    dequeued: int = 0
    rejected: int = 0
    blocked_producers: int = 0


class BoundedIngressQueue(Generic[T]):
    """A size-capped queue with an explicit overflow policy.

    Wraps ``asyncio.Queue`` rather than reimplementing it. The value added is
    the policy, the statistics, and refusing to expose an unbounded constructor:
    ``asyncio.Queue()`` with no maxsize is unbounded, and an unbounded ingress
    queue is not backpressure, it is a memory leak with extra steps.
    """

    def __init__(
        self,
        maxsize: int = 128,
        policy: OverflowPolicy = OverflowPolicy.REJECT,
        retry_after_s: int = 1,
    ) -> None:
        if maxsize < 1:
            raise ValueError("ingress queue must be bounded (maxsize >= 1)")
        self._queue: asyncio.Queue[T] = asyncio.Queue(maxsize=maxsize)
        self.maxsize = maxsize
        self.policy = policy
        self.retry_after_s = retry_after_s
        self.stats = QueueStats()

    async def put(self, item: T) -> None:
        """Admit an item, or apply the overflow policy."""
        if self.policy is OverflowPolicy.REJECT:
            try:
                # put_nowait raises QueueFull rather than waiting, which is the
                # whole point: the decision is made now, not eventually.
                self._queue.put_nowait(item)
            except asyncio.QueueFull:
                self.stats.rejected += 1
                raise QueueFullError(
                    f"ingress queue is full ({self.maxsize} items); retry shortly",
                    depth=self._queue.qsize(),
                    maxsize=self.maxsize,
                    retry_after_s=self.retry_after_s,
                ) from None
        else:
            if self._queue.full():
                self.stats.blocked_producers += 1
            await self._queue.put(item)

        self.stats.enqueued += 1
        self._note_depth()

    async def get(self) -> T:
        item = await self._queue.get()
        self.stats.dequeued += 1
        self._note_depth()
        return item

    def task_done(self) -> None:
        self._queue.task_done()

    async def join(self) -> None:
        """Wait until every admitted item has been marked done.

        Used by the SIGTERM drain. Note this waits for *processing* to finish,
        not merely for the queue to empty — an item pulled off the queue but
        still running has not been accounted for until ``task_done``.
        """
        await self._queue.join()

    def qsize(self) -> int:
        return self._queue.qsize()

    def full(self) -> bool:
        return self._queue.full()

    def empty(self) -> bool:
        return self._queue.empty()

    def _note_depth(self) -> None:
        depth = self._queue.qsize()
        self.stats.depth = depth
        self.stats.max_depth_seen = max(self.stats.max_depth_seen, depth)
