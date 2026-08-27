"""Backpressure: the ingress queue applies its policy and memory stays bounded."""

from __future__ import annotations

import asyncio

import pytest

from skein.errors import QueueFullError
from skein.limits.queue import BoundedIngressQueue
from skein.model.workflow import OverflowPolicy


async def test_the_queue_refuses_construction_without_a_bound():
    """An unbounded ingress queue is not backpressure, it is a memory leak."""
    with pytest.raises(ValueError):
        BoundedIngressQueue(maxsize=0)


async def test_reject_policy_refuses_once_full(no_leaked_tasks):
    queue: BoundedIngressQueue[int] = BoundedIngressQueue(
        maxsize=3, policy=OverflowPolicy.REJECT
    )
    for value in range(3):
        await queue.put(value)

    with pytest.raises(QueueFullError) as excinfo:
        await queue.put(99)

    assert excinfo.value.details["maxsize"] == 3
    assert excinfo.value.details["retry_after_s"] >= 1
    assert queue.stats.rejected == 1
    assert queue.qsize() == 3


async def test_rejection_is_immediate_not_a_timeout(no_leaked_tasks):
    """The decision is made now. A caller must not be held while we think."""
    queue: BoundedIngressQueue[int] = BoundedIngressQueue(maxsize=1)
    await queue.put(1)

    loop = asyncio.get_running_loop()
    started = loop.time()
    with pytest.raises(QueueFullError):
        await queue.put(2)
    assert loop.time() - started < 0.05


async def test_block_policy_waits_for_space(no_leaked_tasks):
    queue: BoundedIngressQueue[int] = BoundedIngressQueue(
        maxsize=1, policy=OverflowPolicy.BLOCK
    )
    await queue.put(1)

    producer = asyncio.create_task(queue.put(2))
    await asyncio.sleep(0.01)
    assert not producer.done()
    assert queue.stats.blocked_producers == 1

    assert await queue.get() == 1
    await producer
    assert queue.qsize() == 1


async def test_depth_never_exceeds_the_bound_under_sustained_load(no_leaked_tasks):
    """The property that matters: memory does not grow with offered load."""
    queue: BoundedIngressQueue[int] = BoundedIngressQueue(
        maxsize=8, policy=OverflowPolicy.REJECT
    )

    accepted = 0
    rejected = 0
    for value in range(500):
        try:
            await queue.put(value)
            accepted += 1
        except QueueFullError:
            rejected += 1
        # Drain occasionally, far slower than the producer.
        if value % 10 == 0 and not queue.empty():
            await queue.get()

    assert queue.qsize() <= 8
    assert queue.stats.max_depth_seen <= 8
    assert rejected > 0
    assert accepted + rejected == 500


async def test_stats_track_throughput(no_leaked_tasks):
    queue: BoundedIngressQueue[int] = BoundedIngressQueue(maxsize=4)
    for value in range(4):
        await queue.put(value)
    for _ in range(4):
        await queue.get()

    assert queue.stats.enqueued == 4
    assert queue.stats.dequeued == 4
    assert queue.qsize() == 0


async def test_join_waits_for_processing_not_just_draining(no_leaked_tasks):
    """A dequeued-but-still-running item is not done. The drain depends on this
    distinction."""
    queue: BoundedIngressQueue[int] = BoundedIngressQueue(maxsize=4)
    await queue.put(1)

    item = await queue.get()
    assert item == 1

    joiner = asyncio.create_task(queue.join())
    await asyncio.sleep(0.01)
    assert not joiner.done()  # taken off the queue, not yet finished

    queue.task_done()
    await joiner


async def test_engine_rejects_submissions_when_saturated(settings, registry, no_leaked_tasks):
    """End to end: a full queue surfaces as a typed error, not a hang."""
    from typing import Any

    from skein.runtime.engine import Engine, EngineConfig
    from tests.conftest import make_step, make_workflow

    async def slow(**kwargs: Any) -> dict[str, Any]:
        await asyncio.sleep(0.2)
        return {"ok": True}

    registry.register("slow", slow)

    # One worker, tiny queue: saturation is reachable in a handful of submits.
    settings.workers = 1
    settings.queue_size = 2
    engine = Engine(EngineConfig(settings=settings, registry=registry))
    await engine.start()
    try:
        workflow = make_workflow([make_step("a", tool="slow")])
        rejected = 0
        for _ in range(12):
            try:
                await engine.submit(workflow)
            except QueueFullError:
                rejected += 1
        assert rejected > 0
        assert engine.queue.qsize() <= settings.queue_size
    finally:
        await engine.stop()
