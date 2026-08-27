"""Prometheus metrics.

Label cardinality is the thing to watch and the reason nothing here is labelled
with a run id or a step id. Those are unbounded — one new time series per run
forever — and a Prometheus server ingesting them degrades in a way that is hard
to attribute back to the service that caused it. Labels here are workflow name,
step kind, tool name and error code: all drawn from a fixed set that only
changes when someone edits code.

Every counter is created at import so the series exists at zero. A counter that
springs into existence on first failure makes ``rate()`` and alerting behave
differently before and after the first occurrence, which is precisely when you
want them to behave predictably.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from prometheus_client.core import REGISTRY as DEFAULT_REGISTRY

REGISTRY: CollectorRegistry = DEFAULT_REGISTRY

# -- runs -------------------------------------------------------------------

runs_started = Counter("skein_runs_started_total", "Workflow runs started", ["workflow"])
runs_succeeded = Counter(
    "skein_runs_succeeded_total", "Workflow runs that completed with no failed step", ["workflow"]
)
runs_failed = Counter(
    "skein_runs_failed_total", "Workflow runs that ended with a failure", ["workflow", "code"]
)
runs_cancelled = Counter(
    "skein_runs_cancelled_total", "Workflow runs cancelled by a caller or shutdown", ["workflow"]
)
run_duration = Histogram(
    "skein_run_duration_seconds",
    "Wall-clock duration of a workflow run",
    ["workflow", "status"],
    # Buckets span three orders of magnitude: agent workflows are dominated by
    # model latency, so the interesting range is seconds to minutes, not the
    # default's sub-second emphasis.
    buckets=(0.1, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600),
)

# -- steps ------------------------------------------------------------------

steps_started = Counter("skein_steps_started_total", "Steps started", ["kind", "tool"])
steps_succeeded = Counter("skein_steps_succeeded_total", "Steps succeeded", ["kind", "tool"])
steps_failed = Counter("skein_steps_failed_total", "Steps failed", ["kind", "tool", "code"])
steps_cancelled = Counter("skein_steps_cancelled_total", "Steps cancelled", ["kind", "tool"])
steps_skipped = Counter("skein_steps_skipped_total", "Steps skipped", ["reason"])
step_duration = Histogram(
    "skein_step_duration_seconds",
    "Step execution duration including retries",
    ["kind", "tool"],
    buckets=(0.005, 0.025, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120),
)

# -- resilience -------------------------------------------------------------

retries = Counter("skein_retries_total", "Retry attempts", ["tool", "code"])
retry_budget_exhausted = Counter(
    "skein_retry_budget_exhausted_total", "Runs that ran out of retry budget", ["workflow"]
)
breaker_transitions = Counter(
    "skein_breaker_transitions_total", "Circuit breaker state changes", ["tool", "state"]
)
#: 0 closed, 1 half-open, 2 open. A gauge rather than three booleans so a single
#: panel can show every tool's state as a step function over time.
breaker_state = Gauge(
    "skein_breaker_state", "Circuit breaker state (0=closed, 1=half_open, 2=open)", ["tool"]
)
breaker_rejections = Counter(
    "skein_breaker_rejections_total", "Calls refused by an open breaker", ["tool"]
)

# -- limits and backpressure ------------------------------------------------

inflight_steps = Gauge("skein_inflight_steps", "Steps currently executing", ["scope"])
concurrency_limit = Gauge("skein_concurrency_limit", "Configured concurrency ceiling", ["scope"])
concurrency_peak = Gauge(
    "skein_concurrency_peak", "Highest simultaneous step count observed", ["scope"]
)
limit_waits = Counter(
    "skein_limit_waits_total", "Times a step had to wait for a permit", ["scope"]
)

queue_depth = Gauge("skein_queue_depth", "Ingress queue depth")
queue_capacity = Gauge("skein_queue_capacity", "Ingress queue capacity")
queue_rejections = Counter(
    "skein_queue_rejections_total", "Submissions refused because the queue was full"
)
queue_blocked_producers = Counter(
    "skein_queue_blocked_producers_total", "Submissions that waited for queue space"
)

# -- budgets and validation -------------------------------------------------

budget_trips = Counter(
    "skein_budget_trips_total", "Runs terminated by a runaway guard", ["workflow", "guard"]
)
validation_rejections = Counter(
    "skein_validation_rejections_total", "Workflows refused at submission", ["code"]
)
tokens_used = Counter("skein_tokens_total", "Tokens consumed", ["model", "kind"])

# -- invariants -------------------------------------------------------------
#
# Exported as gauges so the same properties the tests assert are also visible in
# production. A test proves the code cannot violate them along the paths it
# exercises; a gauge proves the running system is not violating them now.

invariant_violations = Gauge(
    "skein_invariant_violations",
    "Runtime invariant violations; anything above zero is a defect",
    ["invariant"],
)
leaked_tasks = Gauge(
    "skein_leaked_tasks", "Tasks still alive that belong to a finished run"
)

INVARIANTS = (
    "dependency_order",
    "single_execution",
    "concurrency_cap",
    "no_leaked_tasks",
    "trace_completeness",
)

for _invariant in INVARIANTS:
    invariant_violations.labels(invariant=_invariant).set(0)


# -- helpers ----------------------------------------------------------------

_BREAKER_STATE_VALUES = {"closed": 0, "half_open": 1, "open": 2}


def observe_breaker(tool: str, state: str) -> None:
    breaker_state.labels(tool=tool).set(_BREAKER_STATE_VALUES.get(state, 0))


def observe_queue(depth: int, capacity: int) -> None:
    queue_depth.set(depth)
    queue_capacity.set(capacity)


def observe_concurrency(scope: str, current: int, peak: int, limit: int) -> None:
    inflight_steps.labels(scope=scope).set(current)
    concurrency_peak.labels(scope=scope).set(peak)
    concurrency_limit.labels(scope=scope).set(limit)


def render() -> bytes:
    """The exposition payload for ``GET /metrics``."""
    return generate_latest(REGISTRY)
