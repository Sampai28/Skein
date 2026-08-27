# Working notes

Practical things you need on the way to a first run.

## Version assumptions

Nothing here was resolved against a package index. These are the pins in
`pyproject.toml`; if any of them has moved, that file is the only place to
change it.

| Package | Pinned | Why this one |
|---|---|---|
| pydantic | 2.9.2 | v2 API throughout (`model_validate`, `model_config`, `@model_validator`) |
| fastapi | 0.115.4 | lifespan context manager, `WebSocket` routing |
| uvicorn[standard] | 0.32.0 | `--timeout-graceful-shutdown`, websockets extra |
| httpx | 0.27.2 | async client for the fetch tool and Ollama |
| pyyaml | 6.0.2 | workflow files |
| prometheus-client | 0.21.0 | `generate_latest`, `CollectorRegistry` |
| opentelemetry-* | 1.28.1 / 0.49b1 | SDK and instrumentation versions move together; the `0.49b1` on the instrumentation package is not a typo |
| pytest | 8.3.3 | |
| pytest-asyncio | 0.24.0 | `asyncio_mode`, `asyncio_default_fixture_loop_scope` |
| hypothesis | 6.115.5 | `HealthCheck.function_scoped_fixture` |
| ruff | 0.7.2 | the `ASYNC` rule set |
| mypy | 1.13.0 | |

Container images: `python:3.12-slim`, `ollama/ollama:0.4.7`,
`jaegertracing/all-in-one:1.62.0`, `prom/prometheus:v2.55.1`,
`grafana/grafana:11.3.1`, `rancher/k3s:v1.31.2-k3s1`.

**Python 3.12 in containers, not 3.14.** The runtime itself needs only 3.11+
(`TaskGroup`, `asyncio.timeout`, `except*`). The constraint is the observability
stack: OpenTelemetry's instrumentation packages and the `grpcio` wheel that the
OTLP exporter depends on lag new interpreter releases, and `grpcio` in
particular falls back to building from source when no wheel exists — which on a
slim image with no compiler fails outright. If you run the test suite on 3.14 on
the host, expect the OTel imports to be the first thing to break; `SKEIN_TRACING_ENABLED=false`
sidesteps it, since `otel.setup_tracing` returns immediately and the app runs
with a no-op tracer.

## First run

```bash
python -m venv .venv
.venv\Scripts\activate            # PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
```

The suite needs no Docker, no Ollama and no network. Every timing test runs on
the virtual clock.

Then the stack:

```bash
docker compose -f docker/docker-compose.yml up -d --build
docker exec skein-ollama ollama pull llama3.2:3b
python -m skein.cli submit examples/research.yaml --input question="what is backpressure" --follow
```

The model pull is separate on purpose — it is a multi-gigabyte download, and
doing it inside a container healthcheck makes `docker compose up` look like it
has hung. Until it completes, LLM steps fail with an `llm_error`; the rest of
the workflow still runs, which is a reasonable way to see the failure-policy
behaviour. To skip Ollama entirely, set `SKEIN_STUB_LLM=true` and the runtime
uses a deterministic fake.

## Rough edges, most likely first

**pytest-asyncio mode.** `asyncio_mode = "auto"` is set in `pyproject.toml`. In
the default `strict` mode every async test needs `@pytest.mark.asyncio`, and —
this is the part that wastes an afternoon — a missing marker does not error. The
coroutine is never awaited, pytest reports the test as passed, and a
`RuntimeWarning: coroutine was never awaited` scrolls past. If tests pass
suspiciously fast, check the mode first. `filterwarnings = ["error::RuntimeWarning"]`
is set to turn that particular warning into a failure.

Also pinned: `asyncio_default_fixture_loop_scope = "function"`. Without it,
newer pytest-asyncio emits a deprecation warning on every run, and a
session-scoped loop lets a task leaked by one test fail a later, unrelated one.

**Pydantic v2 validator syntax.** The v1 spellings do not work and the errors
are unhelpful. `@validator` → `@field_validator` plus `@classmethod` (order
matters — the classmethod decorator goes underneath). `@root_validator` →
`@model_validator(mode="after")`, which receives and returns the model instance
rather than a dict. `class Config` → `model_config = ConfigDict(...)`.
`.dict()` → `.model_dump()`, `.json()` → `.model_dump_json()`.

The one that bites hardest: `model_validator(mode="after")` methods must
`return self`. Forgetting silently makes the model `None`.

**OpenTelemetry exporter wiring.** Three things fail independently. The endpoint
must be reachable — `http://jaeger:4317` inside Compose, `http://localhost:4317`
from the host — and pointing at 16686 (the UI) instead of 4317 (OTLP) produces a
connection that establishes and then fails on every export. `insecure=True` is
required for a plaintext gRPC endpoint or the exporter tries TLS and hangs. And
spans only leave on batch flush, so a process that exits without
`shutdown_tracing()` loses the last batch — which is always the interesting one.
Failures are swallowed at setup on purpose: a missing tracing backend must not
stop the service starting.

**WebSocket lifecycle in tests.** `TestClient.websocket_connect` is a context
manager and must be used as one; leaving it open leaks the connection into the
next test. The connection is only established inside the `with`. Also note the
app's lifespan does not run unless `TestClient` is itself used as a context
manager — `TestClient(app)` alone leaves `app.state.engine` unset and every
request 500s with an `AttributeError` that points nowhere near the cause.

The stream replays history before streaming live events, so a test that connects
after a fast run has finished still receives the full event sequence rather than
an empty stream followed by a close.

**Windows.** `loop.add_signal_handler` raises `NotImplementedError` on the
Proactor event loop; `ShutdownController.install` falls back to `signal.signal`.
The fallback works for Ctrl-C but Windows has no real SIGTERM, so the drain path
is best exercised in a container.

## Reading an asyncio traceback

The least legible error you will hit, so: the traceback shows where the
exception *surfaced*, which for async code is often not where it originated.

- **`ExceptionGroup` / `+ Exception Group Traceback`.** A `TaskGroup` failed. The
  real exceptions are the nested ones, indented under `+-+---------------`; the
  outer frame is just the group. If several children failed, they are all there
  — the first is not necessarily the cause of the others, since one failure
  cancels its siblings and cancellation shows up as more entries.

- **`CancelledError` with a short traceback.** Almost always a symptom rather
  than a cause. Look upward for what did the cancelling: a `TaskGroup` sibling
  that raised, an `asyncio.timeout` that expired, or an explicit `.cancel()`.
  A `TimeoutError` immediately after is the signature of `asyncio.timeout`,
  which cancels the block and *then* raises.

- **`RuntimeWarning: coroutine 'x' was never awaited`.** A missing `await`. The
  line number is where the coroutine was created, which is usually the line
  with the missing keyword.

- **`Task was destroyed but it is pending!`** on shutdown. A task was garbage
  collected while still running — the leak this project's tests exist to catch.
  It names the task, and `Task-N` names are why every `create_task` here passes
  `name=`.

- **`RuntimeError: no running event loop`.** Calling something loop-bound from
  synchronous code. Usually a fixture or a constructor that creates an
  `asyncio.Queue` or `Event` outside a coroutine.

- **`RuntimeError: attached to a different loop`.** An object created on one
  event loop used on another. In tests this is a fixture with a wider scope than
  the loop; the function-scoped loop setting above is the fix.

Practical habit: read an async traceback from the bottom, and when you reach a
`CancelledError`, stop and go looking for the canceller rather than reading
further up.

## Uncertainties recorded rather than resolved

Nothing here was executed, so the following are judgement calls that a first run
will confirm or correct:

- `opentelemetry-instrumentation-fastapi==0.49b1` is imported but never wired
  up; the app instruments manually via `workflow_span` / `step_span`. If the
  version pin conflicts, dropping the dependency costs nothing.
- `prometheus_client.core.REGISTRY` is used as the default registry. If the
  import path has moved, `metrics.REGISTRY` is the single place to change.
- The `except* Exception` unwrapping in `executor._map` assumes `TaskGroup`
  always raises an `ExceptionGroup` even for one failure. That is the documented
  behaviour; if a future version raises bare, the `except*` clause simply will
  not match and the exception propagates unwrapped, which is still correct but
  loses the classification.
- `HealthCheck.function_scoped_fixture` exists in the pinned Hypothesis version.
  On an older one the suppression list will raise an `AttributeError` at import.
