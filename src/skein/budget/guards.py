"""Hard stops for a run that will not stop by itself.

An agent workflow can loop, fan out on model output, or retry into the ground.
None of that is caught by a timeout alone — a run can burn a hundred thousand
tokens well inside its wall-clock budget, and a map step over a model-generated
list can queue thousands of tool calls without any single step misbehaving.

Four independent ceilings, checked before work is committed rather than after:
steps, wall clock, tool invocations, tokens. Any breach terminates the run with
:class:`~skein.errors.BudgetExceededError`, which the scheduler treats as
terminal and never retries.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from skein.errors import BudgetExceededError
from skein.model.workflow import Budget
from skein.resilience.clock import Clock, SystemClock


@dataclass
class BudgetTracker:
    """Mutable counters for one run, checked against an immutable Budget."""

    budget: Budget
    clock: Clock = field(default_factory=SystemClock)
    started_at: float = field(default=0.0)

    steps_started: int = 0
    tool_calls: int = 0
    tokens_used: int = 0

    def __post_init__(self) -> None:
        if self.started_at == 0.0:
            self.started_at = self.clock.now()

    # -- reservations -----------------------------------------------------
    #
    # Each of these is called *before* the work happens. Counting afterwards
    # would let a run exceed every ceiling by exactly one unit of whatever it
    # was doing, which for a map step fanning out over a large collection is not
    # a rounding error.

    def reserve_step(self) -> None:
        if self.steps_started >= self.budget.max_steps:
            raise BudgetExceededError(
                f"step budget exhausted ({self.budget.max_steps} steps)",
                guard="max_steps",
                limit=self.budget.max_steps,
                used=self.steps_started,
            )
        self.steps_started += 1

    def reserve_tool_calls(self, count: int = 1) -> None:
        if self.tool_calls + count > self.budget.max_tool_calls:
            raise BudgetExceededError(
                f"tool-call budget exhausted ({self.budget.max_tool_calls} calls); "
                f"requested {count} more with {self.tool_calls} used",
                guard="max_tool_calls",
                limit=self.budget.max_tool_calls,
                used=self.tool_calls,
                requested=count,
            )
        self.tool_calls += count

    def record_tokens(self, count: int) -> None:
        """Tokens are recorded after the fact — the count is not known until the
        model has answered. The ceiling therefore stops the *next* call rather
        than truncating this one."""
        self.tokens_used += count
        if self.tokens_used > self.budget.max_tokens:
            raise BudgetExceededError(
                f"token budget exhausted ({self.budget.max_tokens} tokens)",
                guard="max_tokens",
                limit=self.budget.max_tokens,
                used=self.tokens_used,
            )

    def check_duration(self) -> None:
        elapsed = self.elapsed()
        if elapsed > self.budget.max_duration_s:
            raise BudgetExceededError(
                f"duration budget exhausted ({self.budget.max_duration_s}s)",
                guard="max_duration_s",
                limit=self.budget.max_duration_s,
                used=round(elapsed, 3),
            )

    def check_all(self) -> None:
        """Called at each scheduling turn. Only duration can lapse without an
        explicit reservation, so it is the only one that needs polling."""
        self.check_duration()

    # -- inspection -------------------------------------------------------

    def elapsed(self) -> float:
        return self.clock.now() - self.started_at

    def remaining_duration(self) -> float:
        return max(0.0, self.budget.max_duration_s - self.elapsed())

    def snapshot(self) -> dict[str, float | int]:
        return {
            "steps_started": self.steps_started,
            "max_steps": self.budget.max_steps,
            "tool_calls": self.tool_calls,
            "max_tool_calls": self.budget.max_tool_calls,
            "tokens_used": self.tokens_used,
            "max_tokens": self.budget.max_tokens,
            "elapsed_s": round(self.elapsed(), 3),
            "max_duration_s": self.budget.max_duration_s,
        }
