# Skein

An asyncio runtime for executing agent workflows defined as DAGs. You describe
steps and their dependencies; Skein runs them at maximal safe concurrency, with
bounded parallelism, real cancellation, timeouts that don't leak tasks, and a
recorded trace you can replay.

## Why this is harder than it looks

Calling three tools concurrently is `asyncio.gather` and takes a line. What
takes the rest of the project is everything that happens when it goes wrong. One
tool fails and you have to decide what its dependents do — and *not* run them
against a missing input, which is how a failure resurfaces two steps later as a
confusing binding error. A user cancels mid-run, and every in-flight call has to
stop, clean up, and leave nothing behind; a task that outlives its run is a leak
you will find weeks later as unexplained memory growth. A model returns a list
of two thousand items and a `map` step tries to fan out to two thousand
concurrent tool calls. A flaky dependency starts failing and naive retries turn
a partial outage into a full one.

None of that is about agents specifically. It is about concurrency with partial
failure, which is a solved problem in principle and a badly-solved one in most
agent frameworks.

## Quickstart

```bash
docker compose -f docker/docker-compose.yml up -d --build
docker exec skein-ollama ollama pull llama3.2:3b
```

The model pull is separate because it's a multi-gigabyte download, and putting
it in a healthcheck makes `docker compose up` look like it has hung. To skip
Ollama entirely, set `SKEIN_STUB_LLM=true` and the runtime uses a deterministic
fake.

- API and DAG viewer — http://localhost:8000
- Jaeger — http://localhost:16686
- Prometheus — http://localhost:9090
- Grafana (no login) — http://localhost:3000

Submit the example:

```bash
python -m skein.cli submit examples/research.yaml --input question="what is backpressure" --follow
```

Paste the run id into the viewer at http://localhost:8000 to watch step statuses
change over the WebSocket.

Tests need no Docker, no Ollama and no network — timing tests run on an
injectable virtual clock rather than sleeping:

```bash
make install
make test
```

## A workflow

```yaml
name: research
max_concurrency: 6

budget:
  max_steps: 40
  max_duration_s: 180
  max_tool_calls: 60
  max_tokens: 50000

on_step_failure: skip

steps:
  - id: plan
    kind: llm_call
    prompt: "Rewrite as a search query: ${inputs.question}"

  - id: search
    kind: tool_call
    tool: retrieval
    depends_on: [plan]
    inputs:
      query: ${steps.plan.output.text}

  - id: score
    kind: map
    tool: calculator
    depends_on: [search]
    over: ${steps.search.output.documents}
    max_fanout: 8
    inputs:
      expression: "${item.score} * 100"

  - id: arithmetic
    kind: tool_call
    tool: calculator
    depends_on: [search]
    inputs:
      expression: "42 * (7 + 3)"

  - id: combine
    kind: reduce
    reducer: concat
    depends_on: [score, arithmetic]
    inputs:
      values: ${steps.score.output.items}
```

`arithmetic` depends only on `search`, so it starts the moment `search` finishes
and runs concurrently with the whole `score` fan-out instead of queueing behind
it. That is the property the scheduler exists to provide: eligibility is per
step, not per graph level. Reading `${steps.search.output.documents}` without
declaring `search` in `depends_on` is rejected at submission — it would
otherwise work whenever `search` happened to finish first and fail under load.

## asyncio notes

If you're coming to this from synchronous Python:

- **Structured concurrency.** `async with asyncio.TaskGroup()` cannot exit while
  a child task is alive, and a child that raises cancels its siblings. Every
  task in Skein is created inside one, which is why cancellation correctness is
  structural rather than bookkeeping the scheduler has to get right.

- **Cancellation is an exception.** `CancelledError` is raised at the next
  `await` inside the cancelled task. Code between `await`s runs to completion —
  a coroutine that never awaits cannot be cancelled at all.

- **`except Exception` does not catch it.** `CancelledError` inherits from
  `BaseException` specifically so broad handlers don't swallow it. If you *do*
  catch it, re-raise: returning normally tells your caller the work finished.

- **`TaskGroup` over `gather`.** `gather(return_exceptions=False)` leaves the
  other coroutines running after the first exception propagates;
  `return_exceptions=True` puts `CancelledError` in the result list where it
  looks like a value. Neither is acceptable if the job is to not leak work. The
  cost of `TaskGroup` is that failures arrive wrapped in an `ExceptionGroup`, so
  you need `except*`.

- **Semaphores and queues do different jobs.** A semaphore bounds how much runs
  at once, for work already accepted. A bounded queue bounds how much you
  *accept*. Skein uses both — semaphores for step concurrency, a queue for
  ingress — because "wait your turn" and "come back later" are different answers.

- **One blocking call stalls everything.** A synchronous call that takes 200 ms
  blocks the event loop for 200 ms, freezing every other in-flight step in the
  process. This is the most common way an async service quietly loses its
  concurrency. `ToolRegistry.register_sync` pushes blocking work to a thread;
  note that fixes the blocking, not the parallelism, since the GIL still applies.

## Design decisions

- **Ingress overflow rejects rather than blocks.** Blocking a submission holds a
  worker and a socket for as long as it waits, so a sustained overload becomes
  exhausted server capacity and the service stops answering everything — health
  checks included. Rejecting keeps the cost of overload proportional to the
  overload. `BLOCK` is available for a trusted in-process producer.

- **A failed step skips its dependents; it doesn't fail the run.** `FAIL_FAST`
  throws away work that was going to succeed. `CONTINUE` runs a dependent
  against a missing input, so the failure resurfaces later somewhere confusing.
  `SKIP` marks the subtree, records why, and lets independent branches finish.
  All three are configurable per workflow.

- **Circuit breakers are per tool, driven by failure rate.** A global breaker
  means a flaky search API takes the calculator offline with it. A rate over a
  sliding window describes a tool's *current* health, where a consecutive-failure
  count describes its worst recent moment; a minimum sample floor stops the
  first failed call after a deploy tripping everything.

- **Three concurrency limits, acquired in one fixed order** — global, then
  workflow, then tool. Three semaphores taken in arbitrary order is a textbook
  deadlock, so `LimitSet` owns acquisition rather than exposing them.

- **`shield` is used exactly twice**, both around cleanup that must survive the
  cancellation that triggered it: writing a cancelled step's terminal trace
  event, and the final cancel-everything pass during shutdown. Anywhere else it
  would only delay a shutdown that has already been requested.

- **Timeout wraps retry, retry wraps the breaker.** The timeout is outermost so
  it bounds the whole attempt sequence — 5s with three retries is 5s, not 15s.
  The breaker is innermost so it sees individual calls, which is what a failure
  rate has to be computed from.

- **The retry budget is per run.** One flapping tool must not be able to consume
  the run's time while healthy branches wait for permits.

- **Branch targets are skipped, not removed from the graph.** Mutating the DAG
  mid-run would make "no step executes twice" unstatable for steps that no
  longer exist, and would give a recording and its replay different step sets so
  their trajectory hashes could never match.

## Limitations

Single process — one instance owns a run start to finish. The run store is in
memory, so a restart loses in-flight runs and a `GET` that lands on a different
replica returns 404 (hence the client-IP affinity in the Service manifest, which
is a workaround rather than a design). No distributed scheduling, no persistence,
no shared state between replicas. A tool that catches `CancelledError` and
refuses to stop cannot be reclaimed; the deadline still holds and the step is
failed, but the coroutine lives until the process ends.

More detail in [docs/design-notes.md](docs/design-notes.md), and the things most
likely to need attention on a first run are in [docs/notes.md](docs/notes.md).
