"""Per-tool circuit breakers driven by failure rate over a sliding window.

**Rate, not count.** A breaker that opens after N consecutive failures cannot
distinguish a tool that fails one call in three — which is broken — from one
that failed three times during a deploy and has served ten thousand calls since.
A rate over a time window describes the tool's current health; a raw count
describes its worst recent moment.

**Per tool, not global.** A global breaker means a flaky search API takes the
calculator offline with it. Tools fail independently, so they get independent
breakers; the blast radius of one bad dependency is that dependency.

States: ``CLOSED`` → ``OPEN`` on a breach → ``HALF_OPEN`` after a cooldown →
``CLOSED`` on enough consecutive probe successes, or straight back to ``OPEN``
on any probe failure.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from skein.errors import CircuitOpenError
from skein.resilience.clock import Clock, SystemClock


class BreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    #: Fraction of failures in the window that trips the breaker.
    failure_rate_threshold: float = 0.5
    #: Length of the sliding window.
    window_s: float = 30.0
    #: Never trip on fewer than this many samples. Without a floor, the very
    #: first call failing is a 100% failure rate and the breaker opens on a
    #: sample of one.
    min_samples: int = 10
    #: How long OPEN lasts before a probe is allowed.
    open_duration_s: float = 15.0
    #: Consecutive probe successes needed to close from HALF_OPEN.
    half_open_successes: int = 3
    #: Concurrent probes permitted in HALF_OPEN. Kept small on purpose — the
    #: point of a probe is to test the water, not to resume full load against a
    #: dependency that may still be down.
    half_open_max_calls: int = 3


@dataclass
class BreakerStats:
    opened_count: int = 0
    rejected_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    last_transition_at: float = 0.0


class CircuitBreaker:
    """One tool's breaker.

    Deliberately synchronous. Every method runs to completion without awaiting,
    so state transitions cannot interleave with another coroutine's — no lock is
    needed, and adding one would be misleading about where the concurrency is.
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.clock = clock or SystemClock()
        self.stats = BreakerStats()

        self._state = BreakerState.CLOSED
        # (timestamp, succeeded) newest last.
        self._window: deque[tuple[float, bool]] = deque()
        self._opened_at: float = 0.0
        self._half_open_successes = 0
        self._half_open_inflight = 0

    # -- state ------------------------------------------------------------

    @property
    def state(self) -> BreakerState:
        """Current state, after accounting for an elapsed open period.

        Computed on read rather than by a timer. A timer would need a task per
        breaker whose only job is to flip a flag, and that task would have to be
        cancelled correctly on shutdown — an entire class of leak avoided by
        deriving the state from the clock instead.
        """
        if self._state is BreakerState.OPEN:
            if self.clock.now() - self._opened_at >= self.config.open_duration_s:
                self._transition(BreakerState.HALF_OPEN)
        return self._state

    def _transition(self, new_state: BreakerState) -> None:
        if new_state is self._state:
            return
        self._state = new_state
        self.stats.last_transition_at = self.clock.now()
        if new_state is BreakerState.OPEN:
            self._opened_at = self.clock.now()
            self.stats.opened_count += 1
            self._half_open_successes = 0
            self._half_open_inflight = 0
        elif new_state is BreakerState.HALF_OPEN:
            self._half_open_successes = 0
            self._half_open_inflight = 0
        elif new_state is BreakerState.CLOSED:
            # Clear the window on close. Keeping the failures that caused the
            # trip would re-open the breaker on the first post-recovery failure,
            # regardless of how healthy the tool now is.
            self._window.clear()
            self._half_open_successes = 0
            self._half_open_inflight = 0

    # -- admission --------------------------------------------------------

    def allows(self) -> bool:
        state = self.state
        if state is BreakerState.CLOSED:
            return True
        if state is BreakerState.OPEN:
            return False
        return self._half_open_inflight < self.config.half_open_max_calls

    def acquire(self) -> None:
        """Raise if the call must not proceed; otherwise reserve a probe slot."""
        if not self.allows():
            self.stats.rejected_count += 1
            raise CircuitOpenError(
                f"circuit for tool {self.name!r} is {self.state.value}",
                tool=self.name,
                state=self.state.value,
                retry_after_s=max(
                    0.0,
                    self.config.open_duration_s - (self.clock.now() - self._opened_at),
                ),
            )
        if self.state is BreakerState.HALF_OPEN:
            self._half_open_inflight += 1

    # -- outcomes ---------------------------------------------------------

    def record_success(self) -> None:
        self.stats.success_count += 1
        state = self.state
        if state is BreakerState.HALF_OPEN:
            self._half_open_inflight = max(0, self._half_open_inflight - 1)
            self._half_open_successes += 1
            if self._half_open_successes >= self.config.half_open_successes:
                self._transition(BreakerState.CLOSED)
            return
        self._append(True)

    def record_failure(self) -> None:
        self.stats.failure_count += 1
        state = self.state
        if state is BreakerState.HALF_OPEN:
            # One failed probe is enough. The dependency is still unhealthy and
            # further probes would just be load it cannot serve.
            self._half_open_inflight = max(0, self._half_open_inflight - 1)
            self._transition(BreakerState.OPEN)
            return
        self._append(False)
        self._maybe_trip()

    def _append(self, succeeded: bool) -> None:
        now = self.clock.now()
        self._window.append((now, succeeded))
        self._evict(now)

    def _evict(self, now: float) -> None:
        cutoff = now - self.config.window_s
        while self._window and self._window[0][0] < cutoff:
            self._window.popleft()

    def _maybe_trip(self) -> None:
        if self._state is not BreakerState.CLOSED:
            return
        self._evict(self.clock.now())
        if len(self._window) < self.config.min_samples:
            return
        failures = sum(1 for _, ok in self._window if not ok)
        if failures / len(self._window) >= self.config.failure_rate_threshold:
            self._transition(BreakerState.OPEN)

    # -- inspection -------------------------------------------------------

    def failure_rate(self) -> float:
        self._evict(self.clock.now())
        if not self._window:
            return 0.0
        return sum(1 for _, ok in self._window if not ok) / len(self._window)

    def snapshot(self) -> dict[str, object]:
        return {
            "tool": self.name,
            "state": self.state.value,
            "failure_rate": round(self.failure_rate(), 4),
            "samples": len(self._window),
            "opened_count": self.stats.opened_count,
            "rejected_count": self.stats.rejected_count,
            "successes": self.stats.success_count,
            "failures": self.stats.failure_count,
        }


@dataclass
class BreakerRegistry:
    """One breaker per tool, created on first use."""

    config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    clock: Clock = field(default_factory=SystemClock)
    _breakers: dict[str, CircuitBreaker] = field(default_factory=dict)

    def get(self, tool: str) -> CircuitBreaker:
        if tool not in self._breakers:
            self._breakers[tool] = CircuitBreaker(tool, self.config, self.clock)
        return self._breakers[tool]

    def snapshot(self) -> list[dict[str, object]]:
        return [breaker.snapshot() for breaker in self._breakers.values()]
