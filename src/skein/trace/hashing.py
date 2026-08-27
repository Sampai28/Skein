"""Trajectory hashing — one value that says whether two runs did the same thing.

A replay is only useful if divergence is detectable, and comparing whole traces
is impractical: timestamps, durations and run ids differ on every run by
design. The trajectory hash is a digest over only the parts that *should* be
reproducible — which steps ran, in what dependency-settled order, with what
status, producing what outputs.

Deliberately excluded: timestamps, durations, run ids, attempt counts, and the
completion order of steps that are concurrent with each other. That last one is
the subtle one. Two concurrent steps can finish in either order across runs
without anything being wrong, so the trajectory is sorted by step id rather than
taken in completion order; hashing completion order would make the hash a
measure of scheduling luck.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

from skein.trace.events import EventKind, TraceEvent


def canonical_json(value: Any) -> str:
    """Stable JSON for hashing.

    ``sort_keys`` because dict ordering is insertion-ordered in Python and two
    equal dicts built differently would otherwise digest differently.
    ``separators`` to remove whitespace, and ``default=str`` so an unexpected
    type degrades to its string form instead of raising during a hash.
    """
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def digest_value(value: Any) -> str:
    """A short digest of any JSON-able value."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:16]


def trajectory_hash(events: Iterable[TraceEvent]) -> str:
    """Digest the reproducible shape of a run.

    Built from terminal step events only. A step that was retried three times
    and then succeeded has the same trajectory contribution as one that
    succeeded immediately — the retries are a property of the environment, not
    of what the workflow decided to do.
    """
    entries: list[tuple[str, str, str]] = []
    for event in events:
        status = event.kind.terminal_status
        if status is None or event.step_id is None:
            continue
        output_digest = digest_value(event.payload.get("output"))
        entries.append((event.step_id, status.value, output_digest))

    entries.sort()
    return hashlib.sha256(canonical_json(entries).encode("utf-8")).hexdigest()


def trace_completeness(events: Iterable[TraceEvent]) -> tuple[set[str], set[str]]:
    """Return ``(started, terminated)`` step ids.

    The invariant is that these are equal. A step id in ``started`` but not
    ``terminated`` means a code path exited without recording an outcome, which
    is the signature of a cleanup block that raised or a cancellation that was
    swallowed.
    """
    started: set[str] = set()
    terminated: set[str] = set()
    for event in events:
        if event.step_id is None:
            continue
        if event.kind is EventKind.STEP_STARTED:
            started.add(event.step_id)
        elif event.kind.terminal_status is not None:
            terminated.add(event.step_id)
    return started, terminated
