"""Retry with jittered exponential backoff and a per-run budget.

Two rules shape everything here.

**Only classified-retryable errors are retried.** A schema violation or an
unknown tool will fail identically on the next attempt; retrying it wastes the
budget and delays the inevitable. :func:`skein.errors.is_retryable` decides, and
the default for an unrecognised exception is no.

**The budget is per run, not per step.** A single flapping tool with
``max_attempts=5`` across forty steps can otherwise consume two hundred attempts
and the whole wall-clock budget while healthy branches sit waiting for permits.
The run-level budget puts a ceiling on total retry work regardless of how it is
distributed.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import Awaitable, Callable, TypeVar

from skein.errors import RetryBudgetExhaustedError, is_retryable
from skein.model.workflow import RetryPolicy
from skein.resilience.clock import Clock

T = TypeVar("T")


@dataclass
class RetryBudget:
    """A shared allowance of retry attempts for one workflow run."""

    limit: int
    consumed: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.consumed)

    def try_consume(self) -> bool:
        if self.consumed >= self.limit:
            return False
        self.consumed += 1
        return True


@dataclass
class RetryOutcome:
    attempts: int = 0
    delays: list[float] = field(default_factory=list)
    last_error: BaseException | None = None


def backoff_delays(
    policy: RetryPolicy,
    rng: random.Random | None = None,
) -> list[float]:
    """The delay sequence for one full attempt budget.

    Full jitter (``delay = uniform(0, computed)``) when ``jitter == 1.0``, which
    is the default. Equal jitter and no jitter are reachable by lowering it.

    Exposed separately from :func:`retry_async` so tests can assert on the
    sequence without executing anything, and so the shape is inspectable from a
    REPL when tuning a policy.
    """
    generator = rng or random
    delays: list[float] = []
    computed = policy.initial_backoff_s
    for _ in range(max(0, policy.max_attempts - 1)):
        capped = min(computed, policy.max_backoff_s)
        if policy.jitter <= 0:
            delays.append(capped)
        else:
            # Interpolate between the full delay and a uniform draw over it, so
            # jitter=0.5 keeps half the delay deterministic.
            floor = capped * (1.0 - policy.jitter)
            delays.append(floor + generator.uniform(0.0, capped - floor))
        computed *= policy.multiplier
    return delays


async def retry_async(
    operation: Callable[[int], Awaitable[T]],
    *,
    policy: RetryPolicy,
    clock: Clock,
    budget: RetryBudget | None = None,
    rng: random.Random | None = None,
    on_retry: Callable[[int, float, BaseException], None] | None = None,
) -> T:
    """Run ``operation(attempt)`` until it succeeds or the policy is exhausted.

    ``operation`` receives the 1-based attempt number, which is what the trace
    recorder keys recorded responses on — without it a replay could not tell the
    first attempt's response from the third's.

    Cancellation passes straight through. ``CancelledError`` derives from
    ``BaseException``, so the ``except Exception`` below cannot catch it, and
    that is deliberate rather than incidental: a retry loop that caught
    cancellation would keep retrying a step whose workflow has already been
    told to stop.
    """
    outcome = RetryOutcome()
    delays = backoff_delays(policy, rng)

    for attempt in range(1, policy.max_attempts + 1):
        outcome.attempts = attempt
        try:
            return await operation(attempt)
        except Exception as exc:  # noqa: BLE001 - re-raised below unless retryable
            outcome.last_error = exc

            if attempt >= policy.max_attempts or not is_retryable(exc):
                raise

            if budget is not None and not budget.try_consume():
                raise RetryBudgetExhaustedError(
                    f"retry budget of {budget.limit} exhausted; "
                    f"last error: {type(exc).__name__}: {exc}",
                    budget=budget.limit,
                    attempts=attempt,
                ) from exc

            delay = delays[attempt - 1]
            outcome.delays.append(delay)
            if on_retry is not None:
                on_retry(attempt, delay, exc)

            await clock.sleep(delay)

    # Unreachable: the loop either returns or raises. Present so a future edit
    # to the bounds cannot silently fall through to None.
    raise AssertionError("retry_async exhausted its loop without returning or raising")


async def sleep_with_deadline(clock: Clock, seconds: float, deadline: float | None) -> None:
    """Sleep, but never past a deadline.

    Used where a backoff would otherwise push a step beyond its own timeout —
    sleeping 8 seconds inside a 5-second budget guarantees a timeout that the
    caller could have known about immediately.
    """
    if deadline is None:
        await clock.sleep(seconds)
        return
    remaining = deadline - clock.now()
    if remaining <= 0:
        # Yield so the surrounding timeout has a chance to fire rather than
        # returning synchronously into more work.
        await asyncio.sleep(0)
        return
    await clock.sleep(min(seconds, remaining))
