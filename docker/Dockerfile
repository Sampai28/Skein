# Python 3.12, not 3.14 — see docs/notes.md. The observability stack is the
# binding constraint, not the runtime.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependency metadata first so editing a source file does not invalidate the
# layer that installed the dependency tree.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --upgrade pip && pip install .

COPY web/ ./web/
COPY examples/ ./examples/

# Non-root. The container writes only to /app/traces, which is created and
# owned here; everything else can stay read-only.
RUN useradd --create-home --shell /usr/sbin/nologin skein \
    && mkdir -p /app/traces \
    && chown -R skein:skein /app
USER skein

EXPOSE 8000

# No --reload and a single worker process. Uvicorn's multi-worker mode forks
# separate processes, and this runtime's state — the run store, the concurrency
# semaphores, the circuit breakers — is per-process. Two workers would mean a
# run submitted to one being invisible to the other, and concurrency limits
# enforced twice over at half strength. Scaling is horizontal, via replicas.
CMD ["python", "-m", "uvicorn", "skein.api.app:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--timeout-graceful-shutdown", "30"]


# ---------------------------------------------------------------------------
# Test stage: `docker build --target test .`
# ---------------------------------------------------------------------------
FROM base AS test

USER root
COPY tests/ ./tests/
RUN pip install ".[dev]" && chown -R skein:skein /app
USER skein

CMD ["python", "-m", "pytest", "-q"]
