"""Replay: run a workflow again, substituting recorded responses.

The runtime is deterministic given its inputs; what is not deterministic is the
world — a tool that returns different data, a model that samples differently, a
service that is down today. Replay pins the world so that a change in behaviour
can be attributed to the code.

Substitution happens at the *edge* only. Tool and LLM calls return recorded
values; everything else — scheduling, dependency resolution, retry decisions,
budget accounting, circuit-breaker state — runs for real. That boundary is the
point: replaying the scheduler's decisions too would only prove the recording
could be read back.

Divergence is detected by comparing trajectory hashes. A mismatch means the same
inputs and the same responses produced a different set of step outcomes, which
is a real behavioural change and worth failing on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from skein.errors import ReplayMismatchError
from skein.trace.events import EventKind, TraceEvent
from skein.trace.hashing import trajectory_hash


class ReplaySource:
    """Recorded responses, addressable by ``(step_id, attempt)``."""

    def __init__(self, events: list[TraceEvent]) -> None:
        self.events = events
        self._responses: dict[tuple[str, int], Any] = {}
        self._consumed: set[tuple[str, int]] = set()

        for event in events:
            if event.kind in (EventKind.TOOL_RESPONSE, EventKind.LLM_RESPONSE):
                if event.step_id is not None:
                    self._responses[(event.step_id, event.attempt or 1)] = event.payload.get(
                        "response"
                    )

    @classmethod
    def from_file(cls, path: Path) -> ReplaySource:
        from skein.trace.recorder import TraceRecorder

        return cls(TraceRecorder.load(path))

    def has(self, step_id: str, attempt: int) -> bool:
        return (step_id, attempt) in self._responses

    def get(self, step_id: str, attempt: int) -> Any:
        """Fetch a recorded response.

        A miss is an error rather than a fall-through to the live tool. Silently
        calling the real world during a replay would produce a run that is part
        recording and part live, and whose trajectory hash means nothing.
        """
        key = (step_id, attempt)
        if key not in self._responses:
            raise ReplayMismatchError(
                f"no recorded response for step {step_id!r} attempt {attempt}; "
                f"the workflow took a path the recording did not",
                step_id=step_id,
                attempt=attempt,
                recorded_steps=sorted({sid for sid, _ in self._responses}),
            )
        self._consumed.add(key)
        return self._responses[key]

    @property
    def original_hash(self) -> str:
        return trajectory_hash(self.events)

    def unconsumed(self) -> list[tuple[str, int]]:
        """Recorded responses the replay never asked for.

        Not an error on its own — a workflow that legitimately short-circuits
        will leave some — but a useful signal when diagnosing a mismatch,
        because it names the steps the replay declined to take.
        """
        return sorted(set(self._responses) - self._consumed)

    def compare(self, replayed_events: list[TraceEvent]) -> None:
        """Raise if the replay diverged from the recording."""
        replayed = trajectory_hash(replayed_events)
        original = self.original_hash
        if replayed != original:
            raise ReplayMismatchError(
                "replay trajectory differs from the recording",
                original_hash=original,
                replayed_hash=replayed,
                unconsumed=self.unconsumed(),
            )
