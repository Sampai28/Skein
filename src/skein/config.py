"""Runtime configuration, from environment with sane defaults.

Plain dataclass rather than pydantic-settings — one fewer dependency for
something that reads a dozen environment variables. Every default is chosen to
be safe on a laptop; the Compose file and the ConfigMap both override.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from skein.model.workflow import OverflowPolicy


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # -- concurrency --------------------------------------------------------
    #: Global ceiling on simultaneously executing steps. 64 is chosen against
    #: asyncio's cost model, not arbitrarily: see docs/design-notes.md.
    global_concurrency: int = field(default_factory=lambda: _int("SKEIN_GLOBAL_CONCURRENCY", 64))
    default_tool_concurrency: int = field(
        default_factory=lambda: _int("SKEIN_TOOL_CONCURRENCY", 8)
    )

    # -- ingress ------------------------------------------------------------
    queue_size: int = field(default_factory=lambda: _int("SKEIN_QUEUE_SIZE", 128))
    overflow_policy: str = field(
        default_factory=lambda: os.getenv("SKEIN_OVERFLOW_POLICY", OverflowPolicy.REJECT.value)
    )
    #: Workers pulling from the ingress queue. Each drives one run at a time, so
    #: this bounds concurrent *runs* while global_concurrency bounds concurrent
    #: *steps*.
    workers: int = field(default_factory=lambda: _int("SKEIN_WORKERS", 8))

    # -- shutdown -----------------------------------------------------------
    drain_timeout_s: float = field(default_factory=lambda: _float("SKEIN_DRAIN_TIMEOUT_S", 25.0))
    cancel_timeout_s: float = field(default_factory=lambda: _float("SKEIN_CANCEL_TIMEOUT_S", 5.0))

    # -- llm ----------------------------------------------------------------
    ollama_url: str = field(
        default_factory=lambda: os.getenv("SKEIN_OLLAMA_URL", "http://localhost:11434")
    )
    ollama_model: str = field(
        default_factory=lambda: os.getenv("SKEIN_OLLAMA_MODEL", "llama3.2:3b")
    )
    llm_timeout_s: float = field(default_factory=lambda: _float("SKEIN_LLM_TIMEOUT_S", 120.0))
    #: When true the runtime uses StubLlmClient. Lets the whole stack run and
    #: the demo workflow execute without a model download.
    use_stub_llm: bool = field(default_factory=lambda: _bool("SKEIN_STUB_LLM", False))

    # -- trace --------------------------------------------------------------
    trace_dir: Path = field(
        default_factory=lambda: Path(os.getenv("SKEIN_TRACE_DIR", "traces"))
    )
    trace_to_disk: bool = field(default_factory=lambda: _bool("SKEIN_TRACE_TO_DISK", True))
    max_completed_runs: int = field(default_factory=lambda: _int("SKEIN_MAX_COMPLETED_RUNS", 500))

    # -- observability ------------------------------------------------------
    tracing_enabled: bool = field(default_factory=lambda: _bool("SKEIN_TRACING_ENABLED", True))
    otlp_endpoint: str = field(
        default_factory=lambda: os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://jaeger:4317")
    )

    # -- breaker ------------------------------------------------------------
    breaker_failure_rate: float = field(
        default_factory=lambda: _float("SKEIN_BREAKER_FAILURE_RATE", 0.5)
    )
    breaker_window_s: float = field(default_factory=lambda: _float("SKEIN_BREAKER_WINDOW_S", 30.0))
    breaker_min_samples: int = field(
        default_factory=lambda: _int("SKEIN_BREAKER_MIN_SAMPLES", 10)
    )
    breaker_open_duration_s: float = field(
        default_factory=lambda: _float("SKEIN_BREAKER_OPEN_S", 15.0)
    )

    @property
    def overflow(self) -> OverflowPolicy:
        try:
            return OverflowPolicy(self.overflow_policy)
        except ValueError:
            return OverflowPolicy.REJECT


def load_settings() -> Settings:
    return Settings()
