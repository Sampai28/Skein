"""OpenTelemetry tracing, exported to Jaeger over OTLP.

Two span levels: one per workflow run, one per step, with steps as children of
the run. That nesting is what makes a trace readable — the run span shows total
duration and the step spans show where it went, including the gaps where a step
was waiting on a concurrency permit rather than doing work.

Tracing is optional at runtime. If the exporter cannot be constructed the app
still starts with a no-op tracer, because an observability backend being down is
not a reason for the service to be down.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any, Iterator

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, Status, StatusCode

_provider: TracerProvider | None = None


def setup_tracing(
    service_name: str = "skein",
    endpoint: str | None = None,
    enabled: bool | None = None,
) -> None:
    """Install a tracer provider. Safe to call once at startup."""
    global _provider

    if enabled is None:
        enabled = os.getenv("SKEIN_TRACING_ENABLED", "true").lower() == "true"
    if not enabled or _provider is not None:
        return

    endpoint = endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://jaeger:4317")

    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": os.getenv("SKEIN_VERSION", "0.1.0"),
        }
    )
    provider = TracerProvider(resource=resource)

    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        # BatchSpanProcessor rather than SimpleSpanProcessor: simple exports on
        # the calling thread, so every span end becomes a network round trip
        # inside the event loop. Batching moves that to a background thread,
        # which is the difference between tracing being free and tracing being
        # the slowest thing in the step.
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True))
        )
    except Exception:  # noqa: BLE001 - tracing must never block startup
        # No exporter: spans are created and dropped. The API keeps working.
        pass

    trace.set_tracer_provider(provider)
    _provider = provider


def shutdown_tracing(timeout_s: float = 5.0) -> None:
    """Flush pending spans on shutdown.

    Without this the last batch — which is the one covering whatever was
    happening when the process was told to stop, i.e. the interesting part — is
    lost.
    """
    global _provider
    if _provider is not None:
        with contextlib.suppress(Exception):
            _provider.shutdown()
        _provider = None


def tracer() -> trace.Tracer:
    return trace.get_tracer("skein")


@contextlib.contextmanager
def workflow_span(run_id: str, workflow: str) -> Iterator[Span]:
    with tracer().start_as_current_span(
        f"workflow {workflow}",
        attributes={"skein.run_id": run_id, "skein.workflow": workflow},
    ) as span:
        yield span


@contextlib.contextmanager
def step_span(run_id: str, step_id: str, kind: str, target: str | None = None) -> Iterator[Span]:
    attributes: dict[str, Any] = {
        "skein.run_id": run_id,
        "skein.step_id": step_id,
        "skein.step_kind": kind,
    }
    if target:
        attributes["skein.target"] = target
    with tracer().start_as_current_span(f"step {step_id}", attributes=attributes) as span:
        yield span


def mark_error(span: Span, exc: BaseException) -> None:
    span.set_status(Status(StatusCode.ERROR, str(exc)))
    span.set_attribute("skein.error_code", getattr(exc, "code", type(exc).__name__))
