"""Records events to memory, to a JSONL file, and to live subscribers.

Three consumers, one write path. The in-memory list backs ``GET /runs/{id}/trace``
and the trajectory hash; the file is the durable artefact a replay reads; the
subscribers are WebSocket clients watching the DAG viewer.

File writes are intentionally **synchronous**. Appending a line to an open file
is a buffered ``write`` syscall in the low microseconds, and making it async
would mean either a thread pool (adding a context switch per event to save
nothing) or ``aiofiles`` (another dependency for the same non-saving). The one
thing that would justify async here is fsync-per-event durability, which this
does not need — a trace whose last few lines are lost in a hard kill is
acceptable, since the run itself did not survive either.

See :meth:`TraceRecorder.publish` for the fan-out policy, which is where the
interesting failure mode is.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from pathlib import Path
from typing import Any, TextIO

from skein.trace.events import EventKind, StepStatus, TraceEvent
from skein.trace.hashing import trajectory_hash


class TraceRecorder:
    """One recorder per run."""

    def __init__(
        self,
        run_id: str,
        path: Path | None = None,
        subscriber_queue_size: int = 256,
    ) -> None:
        self.run_id = run_id
        self.path = path
        self.events: list[TraceEvent] = []
        self._seq = 0
        self._handle: TextIO | None = None
        self._subscribers: set[asyncio.Queue[TraceEvent]] = set()
        self._subscriber_queue_size = subscriber_queue_size
        self._closed = False

        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = path.open("a", encoding="utf-8")

    # -- recording ---------------------------------------------------------

    def record(
        self,
        kind: EventKind,
        *,
        step_id: str | None = None,
        attempt: int | None = None,
        status: StepStatus | None = None,
        target: str | None = None,
        payload: dict[str, Any] | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        duration_s: float | None = None,
    ) -> TraceEvent:
        self._seq += 1
        event = TraceEvent(
            kind=kind,
            run_id=self.run_id,
            ts=time.time(),
            seq=self._seq,
            step_id=step_id,
            attempt=attempt,
            status=status,
            target=target,
            payload=payload or {},
            error_code=error_code,
            error_message=error_message,
            duration_s=duration_s,
        )
        self.events.append(event)

        if self._handle is not None and not self._closed:
            self._handle.write(event.to_json_line() + "\n")
            self._handle.flush()

        self.publish(event)
        return event

    # -- live subscribers --------------------------------------------------

    def subscribe(self) -> asyncio.Queue[TraceEvent]:
        queue: asyncio.Queue[TraceEvent] = asyncio.Queue(
            maxsize=self._subscriber_queue_size
        )
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[TraceEvent]) -> None:
        self._subscribers.discard(queue)

    def publish(self, event: TraceEvent) -> None:
        """Fan out to subscribers, dropping for any that has fallen behind.

        ``put_nowait`` rather than ``await put``. This is the load-bearing
        decision in the whole file: awaiting would make the *scheduler* wait on
        the slowest WebSocket client, so a browser on a bad connection would
        throttle workflow execution. A subscriber that cannot keep up loses
        events instead, which is the correct trade — the durable trace on disk
        is complete regardless, and the viewer is a live view, not a ledger.

        The queue is bounded for the same reason: an unbounded per-subscriber
        queue turns a stalled client into unbounded memory growth.
        """
        if not self._subscribers:
            return
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # Drop and move on. Deliberately not logged per-event; a slow
                # client would otherwise generate more log volume than the run
                # generates work.
                continue

    # -- inspection --------------------------------------------------------

    def trajectory_hash(self) -> str:
        return trajectory_hash(self.events)

    def responses(self) -> dict[tuple[str, int], Any]:
        """Recorded tool and LLM responses, keyed by ``(step_id, attempt)``.

        This is what a replay consumes. Keying on the attempt matters: a step
        that failed twice and succeeded on the third try recorded three
        responses, and replaying the first for all three would not reproduce
        the run.
        """
        out: dict[tuple[str, int], Any] = {}
        for event in self.events:
            if event.kind in (EventKind.TOOL_RESPONSE, EventKind.LLM_RESPONSE):
                if event.step_id is not None:
                    out[(event.step_id, event.attempt or 1)] = event.payload.get("response")
        return out

    def close(self) -> None:
        self._closed = True
        if self._handle is not None:
            with contextlib.suppress(Exception):
                self._handle.flush()
                self._handle.close()
            self._handle = None
        self._subscribers.clear()

    @classmethod
    def load(cls, path: Path) -> list[TraceEvent]:
        """Read a trace file back.

        Tolerates a truncated final line, which is what a trace from a killed
        process looks like. Raising there would make the most interesting traces
        — the ones from runs that died — the only ones that cannot be inspected.
        """
        events: list[TraceEvent] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(TraceEvent.from_json_line(line))
                except Exception:  # noqa: BLE001 - partial trailing line
                    continue
        return events
