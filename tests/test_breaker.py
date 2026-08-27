"""Circuit breaker: the full state sequence across a failure-rate window."""

from __future__ import annotations

import pytest

from skein.errors import CircuitOpenError
from skein.resilience.breaker import (
    BreakerRegistry,
    BreakerState,
    CircuitBreaker,
    CircuitBreakerConfig,
)
from skein.resilience.clock import VirtualClock


def make_breaker(clock: VirtualClock, **overrides) -> CircuitBreaker:
    config = CircuitBreakerConfig(
        failure_rate_threshold=0.5,
        window_s=30.0,
        min_samples=10,
        open_duration_s=15.0,
        half_open_successes=3,
        half_open_max_calls=3,
        **overrides,
    )
    return CircuitBreaker("tool", config, clock)


def test_a_new_breaker_is_closed():
    breaker = make_breaker(VirtualClock())
    assert breaker.state is BreakerState.CLOSED
    assert breaker.allows()


def test_it_does_not_trip_below_the_sample_floor():
    """Nine failures out of nine is a 100% rate and must not open the breaker.

    Without a floor, the first failed call after a deploy trips every breaker in
    the process.
    """
    breaker = make_breaker(VirtualClock())
    for _ in range(9):
        breaker.record_failure()
    assert breaker.state is BreakerState.CLOSED


def test_it_trips_once_the_rate_is_breached_with_enough_samples():
    breaker = make_breaker(VirtualClock())
    for _ in range(10):
        breaker.record_failure()
    assert breaker.state is BreakerState.OPEN
    assert not breaker.allows()


def test_a_low_failure_rate_does_not_trip_it():
    breaker = make_breaker(VirtualClock())
    for index in range(20):
        if index % 5 == 0:
            breaker.record_failure()
        else:
            breaker.record_success()
    # 20% failures against a 50% threshold.
    assert breaker.state is BreakerState.CLOSED
    assert breaker.failure_rate() == pytest.approx(0.2)


def test_old_failures_leave_the_window():
    """A breaker must describe current health, not worst recent moment."""
    clock = VirtualClock()
    breaker = make_breaker(clock)

    for _ in range(9):
        breaker.record_failure()

    # Move past the window; those failures no longer count.
    clock._now += 31.0
    for _ in range(9):
        breaker.record_success()

    assert breaker.state is BreakerState.CLOSED
    assert breaker.failure_rate() == 0.0


async def test_the_full_state_sequence():
    """CLOSED -> OPEN -> HALF_OPEN -> CLOSED."""
    clock = VirtualClock()
    breaker = make_breaker(clock)

    assert breaker.state is BreakerState.CLOSED

    for _ in range(10):
        breaker.record_failure()
    assert breaker.state is BreakerState.OPEN

    # Still open before the cooldown elapses.
    clock._now += 14.0
    assert breaker.state is BreakerState.OPEN

    clock._now += 2.0
    assert breaker.state is BreakerState.HALF_OPEN

    for _ in range(3):
        breaker.acquire()
        breaker.record_success()
    assert breaker.state is BreakerState.CLOSED


def test_a_failed_probe_reopens_immediately():
    clock = VirtualClock()
    breaker = make_breaker(clock)

    for _ in range(10):
        breaker.record_failure()
    clock._now += 16.0
    assert breaker.state is BreakerState.HALF_OPEN

    breaker.acquire()
    breaker.record_failure()
    # One bad probe is enough; further probes would just be load it cannot serve.
    assert breaker.state is BreakerState.OPEN


def test_half_open_limits_concurrent_probes():
    clock = VirtualClock()
    breaker = make_breaker(clock, half_open_max_calls=2)

    for _ in range(10):
        breaker.record_failure()
    clock._now += 16.0
    assert breaker.state is BreakerState.HALF_OPEN

    breaker.acquire()
    breaker.acquire()
    with pytest.raises(CircuitOpenError):
        breaker.acquire()


def test_acquire_raises_with_a_retry_hint_when_open():
    clock = VirtualClock()
    breaker = make_breaker(clock)
    for _ in range(10):
        breaker.record_failure()

    with pytest.raises(CircuitOpenError) as excinfo:
        breaker.acquire()
    assert excinfo.value.details["state"] == "open"
    assert excinfo.value.details["retry_after_s"] > 0


def test_closing_clears_the_window():
    """Otherwise the first failure after recovery re-opens the breaker."""
    clock = VirtualClock()
    breaker = make_breaker(clock)

    for _ in range(10):
        breaker.record_failure()
    clock._now += 16.0
    for _ in range(3):
        breaker.acquire()
        breaker.record_success()
    assert breaker.state is BreakerState.CLOSED

    breaker.record_failure()
    assert breaker.state is BreakerState.CLOSED
    assert breaker.failure_rate() == 1.0  # one sample, below min_samples


def test_breakers_are_per_tool():
    clock = VirtualClock()
    registry = BreakerRegistry(config=CircuitBreakerConfig(min_samples=4), clock=clock)

    for _ in range(4):
        registry.get("flaky").record_failure()

    assert registry.get("flaky").state is BreakerState.OPEN
    # A different tool is untouched — the blast radius of one bad dependency is
    # that dependency.
    assert registry.get("healthy").state is BreakerState.CLOSED


async def test_an_open_breaker_fails_the_step_without_calling_the_tool(
    registry, build_context, no_leaked_tasks
):
    from typing import Any

    from skein.resilience.breaker import BreakerRegistry as Registry
    from skein.runtime.scheduler import Scheduler
    from tests.conftest import make_step, make_workflow

    calls = {"n": 0}

    async def counted(**kwargs: Any) -> dict[str, Any]:
        calls["n"] += 1
        return {"ok": True}

    registry.register("counted", counted)

    clock = VirtualClock()
    breakers = Registry(config=CircuitBreakerConfig(min_samples=1), clock=clock)
    breakers.get("counted")._transition(BreakerState.OPEN)

    workflow = make_workflow([make_step("a", tool="counted")])
    context, state, limits = build_context(workflow, clock=clock)
    context.breakers = breakers

    await Scheduler(context, limits).run(state)

    assert state.step("a").error_code == "circuit_open"
    # The tool was never invoked — the point of an open breaker.
    assert calls["n"] == 0


def test_snapshot_exposes_state_for_the_api():
    clock = VirtualClock()
    registry = BreakerRegistry(config=CircuitBreakerConfig(min_samples=2), clock=clock)
    registry.get("a").record_success()
    registry.get("b").record_failure()
    registry.get("b").record_failure()

    snapshot = {entry["tool"]: entry for entry in registry.snapshot()}
    assert snapshot["a"]["state"] == "closed"
    assert snapshot["b"]["state"] == "open"
    assert snapshot["b"]["opened_count"] == 1
