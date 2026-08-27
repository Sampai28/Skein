"""A fault-injecting wrapper — the instrument the resilience tests measure with.

Retry, timeout and circuit-breaker code is only as trustworthy as the failures
it has been shown. Waiting for a real dependency to misbehave is not a test
strategy, so this wraps any tool and makes it misbehave on demand: added
latency, a failure rate, and a hang mode that ignores cancellation.

The hang mode is the important one. It reproduces the failure that actually
breaks async services in production — a tool that does not cooperate with
cancellation, so that cancelling it does not stop it. A runtime that only ever
meets well-behaved tools will pass its cancellation tests and still leak tasks
in the real world.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable

from skein.errors import ToolError
from skein.resilience.clock import Clock, SystemClock


class LatencyDistribution(str, Enum):
    FIXED = "fixed"
    UNIFORM = "uniform"
    #: Most calls fast, a long tail. The shape that makes p99 diverge from p50
    #: and the reason timeouts exist.
    LOGNORMAL = "lognormal"


class HangMode(str, Enum):
    NONE = "none"
    #: Awaits forever but cooperates: cancellation stops it at the await point.
    COOPERATIVE = "cooperative"
    #: Catches CancelledError and keeps going. Models a tool wrapping its work
    #: in `except BaseException` or a C extension that ignores interruption.
    #: The runtime must still enforce its deadline.
    UNCANCELLABLE = "uncancellable"


@dataclass
class FaultConfig:
    error_rate: float = 0.0
    latency_s: float = 0.0
    latency_distribution: LatencyDistribution = LatencyDistribution.FIXED
    #: For UNIFORM, the upper bound. For LOGNORMAL, the tail multiplier.
    latency_spread: float = 0.0
    hang_mode: HangMode = HangMode.NONE
    hang_after_calls: int = 0
    #: Fail the first N calls then behave. Used to drive a breaker through a
    #: full open → half-open → closed cycle deterministically.
    fail_first_n: int = 0
    #: Deterministic failures beat probabilistic ones in a test. A seeded RNG
    #: makes an error_rate reproducible.
    seed: int | None = None
    error_message: str = "injected fault"
    retryable: bool = True


@dataclass
class FaultStats:
    calls: int = 0
    failures: int = 0
    hangs: int = 0
    cancellations_observed: int = 0
    total_latency_s: float = 0.0


class FaultInjector:
    """Wraps an async callable and applies :class:`FaultConfig` to every call."""

    def __init__(
        self,
        inner: Callable[..., Awaitable[Any]],
        config: FaultConfig | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.inner = inner
        self.config = config or FaultConfig()
        self.clock = clock or SystemClock()
        self.stats = FaultStats()
        self._rng = random.Random(self.config.seed)

    async def __call__(self, **kwargs: Any) -> Any:
        self.stats.calls += 1
        call_index = self.stats.calls

        await self._apply_latency()
        await self._maybe_hang(call_index)
        self._maybe_fail(call_index)

        return await self.inner(**kwargs)

    # -- behaviours --------------------------------------------------------

    async def _apply_latency(self) -> None:
        delay = self._draw_latency()
        if delay <= 0:
            return
        self.stats.total_latency_s += delay
        await self.clock.sleep(delay)

    def _draw_latency(self) -> float:
        base = self.config.latency_s
        if base <= 0:
            return 0.0
        if self.config.latency_distribution is LatencyDistribution.FIXED:
            return base
        if self.config.latency_distribution is LatencyDistribution.UNIFORM:
            return self._rng.uniform(base, base + max(0.0, self.config.latency_spread))
        # Lognormal: median at `base`, a tail controlled by spread. sigma is
        # clamped because an unclamped draw can produce a delay long enough to
        # look like a hang, which would confuse the two failure modes this class
        # is meant to keep separate.
        sigma = min(2.0, max(0.01, self.config.latency_spread))
        return float(self._rng.lognormvariate(0.0, sigma)) * base

    async def _maybe_hang(self, call_index: int) -> None:
        if self.config.hang_mode is HangMode.NONE:
            return
        if call_index <= self.config.hang_after_calls:
            return

        self.stats.hangs += 1

        if self.config.hang_mode is HangMode.COOPERATIVE:
            # An Event that is never set. Cancellation raises here, which is the
            # well-behaved case.
            await asyncio.Event().wait()
            return

        # UNCANCELLABLE. Swallow cancellation and keep waiting.
        #
        # This is deliberately the anti-pattern: `except BaseException` around
        # an await, absorbing CancelledError and continuing. It exists here so
        # the test suite can prove the runtime's deadline still holds when a
        # tool does this, because in production some of them do.
        while True:
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                self.stats.cancellations_observed += 1
                continue

    def _maybe_fail(self, call_index: int) -> None:
        should_fail = call_index <= self.config.fail_first_n
        if not should_fail and self.config.error_rate > 0:
            should_fail = self._rng.random() < self.config.error_rate

        if not should_fail:
            return

        self.stats.failures += 1
        error = ToolError(
            f"{self.config.error_message} (call {call_index})",
            injected=True,
            call_index=call_index,
        )
        # ToolError.retryable is a class attribute; overriding per instance lets
        # a test exercise the non-retryable path without a second exception type.
        error.retryable = self.config.retryable  # type: ignore[attr-defined]
        raise error


def wrap_with_faults(
    registry: Any,
    tool_name: str,
    config: FaultConfig,
    clock: Clock | None = None,
) -> FaultInjector:
    """Replace a registered tool with a fault-injecting version of itself.

    Returns the injector so a test can read its stats. The tool keeps its name,
    its schemas and its concurrency limit — only the callable changes — so the
    rest of the runtime cannot tell the difference, which is the whole point.
    """
    from dataclasses import replace

    original = registry.get(tool_name)
    injector = FaultInjector(original.fn, config, clock)
    registry.tools[tool_name] = replace(original, fn=injector)
    return injector
