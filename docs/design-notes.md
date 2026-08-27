# Design notes

## Why TaskGroup rather than managing tasks by hand

The scheduler could have kept a `set[asyncio.Task]`, called `create_task` for
each eligible step, and cleaned up in a `finally`. That is the shape most
schedulers start as, and it is where the leaks come from.

The manual version has to get all of the following right, every time:

- cancel every outstanding task when one fails, without racing a task that is
  finishing at the same moment;
- await each cancelled task, because `cancel()` only *requests* cancellation and
  a task that is not awaited can still be running after you return;
- handle the case where cleanup itself raises, without losing the original
  exception;
- avoid holding references to completed tasks, which keeps their frames alive.

`TaskGroup` does all of that as a language construct. The block cannot exit
while a child is alive, a child that raises cancels its siblings, and exceptions
are collected into an `ExceptionGroup` rather than the first one winning. The
practical consequence for this codebase is that cancellation correctness is
structural rather than something the scheduler has to remember.

The cost is the `ExceptionGroup`. Every failure arrives wrapped, so
`except ToolError` no longer matches and `except*` is needed — which is why
`executor._map` unwraps to the first real exception before letting it reach the
retry layer, whose classification works on exception type.

`gather` was rejected for a different reason: `return_exceptions=False` leaves
the other coroutines running after the first exception propagates, and
`return_exceptions=True` swallows `CancelledError` into the result list, where
it is indistinguishable from a returned value. Neither is acceptable for
something whose job is to not leak work.

## The cancellation contract

Stated plainly, because everything else depends on it:

1. **Cancellation is an exception, not a return value.** `CancelledError` is
   raised at the next `await` inside the cancelled task. Code between `await`s
   runs to completion.
2. **Nothing catches it without re-raising.** Every `except asyncio.CancelledError`
   in this codebase ends in `raise`. A handler that returns normally tells its
   caller the work completed, which is a lie the caller has no way to detect.
3. **`except Exception` cannot catch it.** Since 3.8, `CancelledError` derives
   from `BaseException` specifically so that broad handlers do not swallow it.
   This is why `retry_async` can use `except Exception` safely — a cancelled
   step is not retried.
4. **Cleanup ordering is guaranteed by the stack, not by bookkeeping.**
   Cancellation unwinds inward-out: the innermost `finally` runs first, then its
   caller's, and so on. The concurrency permits in `LimitSet.acquire` are
   released by nested `finally` blocks in reverse acquisition order, and the
   step's terminal trace event is written by the handler above them.
5. **`shield` is used exactly twice**, both times around work that must survive
   the cancellation that triggered it:
   - `scheduler._run_step`, writing the terminal trace event for a cancelled
     step. One buffered write; without it a cancelled run has no terminal event
     and violates trace completeness.
   - `signals.ShutdownController.drain`, around the final cancel-everything
     pass, so an interrupted shutdown still tears runs down cleanly. Bounded by
     `cancel_timeout_s` so it cannot hang.

   Everywhere else, shielding would just delay a shutdown that has already been
   requested.

The property that ties it together is the one the tests assert directly: after a
cancelled run, `asyncio.all_tasks()` contains nothing attributable to it.

### The tool that will not be cancelled

A tool can catch `CancelledError` and keep going — through a bare
`except BaseException`, or a C extension that ignores interruption. `FaultInjector`
reproduces this deliberately.

The runtime's deadline still holds: `asyncio.timeout` fires, the step is failed
as `step_timeout`, dependents are skipped, and the run completes. What the
runtime *cannot* do is reclaim the rogue coroutine, which stays alive until the
process ends. That is a real limit rather than something the design papers over,
which is why `skein_leaked_tasks` is a gauge and not an assertion.

## Retry and the circuit breaker, and why the nesting order matters

The layering is `timeout( retry( breaker( call ) ) )`.

- **Timeout outermost** so it bounds the whole attempt sequence. A step with a
  5 s timeout and three retries takes 5 s, not 15 s. Inverted, a step's worst
  case would be the product of its timeout and its attempt count — which is
  also why `validation._check_policy_values` rejects a workflow whose worst case
  cannot fit the run budget.
- **Breaker innermost** so it observes individual calls. A failure *rate* has to
  be computed per call; outside the retry loop it would see one outcome per
  logical step and the rate would be wrong by a factor of the attempt count.

The two interact in a way worth stating: `CircuitOpenError` is classified
**not retryable**. When the breaker is open, retrying inside the same step is
precisely the load the breaker exists to prevent. The step fails immediately,
and the next workflow to reach that tool finds the breaker in half-open and
probes it.

The retry budget is per **run**, not per step, so one flapping tool cannot
consume the entire run's time while healthy branches wait for permits.

## The trace and replay format

JSONL, one event per line, append-only. Flat and self-describing so a trace can
be read with `jq` on a machine that has never had Skein installed.

Two properties the rest of the system relies on:

- every executed step emits exactly one `step_started` and exactly one terminal
  event;
- every tool and LLM response carries its attempt number, so a step that failed
  twice and succeeded on the third try replays as three distinct responses
  rather than the first one three times.

**Trajectory hashing** is what makes replay useful. It digests only what should
be reproducible: which steps ran, with what status, producing what output —
sorted by step id. Excluded are timestamps, durations, run ids, attempt counts,
and the completion order of concurrent steps. That last exclusion is the subtle
one: two independent steps can finish in either order across runs without
anything being wrong, so hashing completion order would make the hash a measure
of scheduling luck rather than of behaviour.

Replay substitutes at the edge only. Tool and LLM calls return recorded values;
scheduling, dependency resolution, retry decisions, budget accounting and
breaker state all run for real. A recorded response that the replay never asks
for is not an error on its own, but a *missing* one is — falling through to the
live tool would produce a run that is part recording and part live, whose hash
means nothing.

## Design targets

These are the characteristics the runtime is built to hit, reasoned from
asyncio's cost model. They are targets, not observations.

**Per-task scheduling overhead.** Creating and running a trivial task costs on
the order of tens of microseconds — allocation, a `Task` object, and a trip
through the event loop's ready queue. The design target is that scheduler
overhead stays a small fraction of step duration for any step doing real work.
For an LLM call measured in seconds, or an HTTP fetch in tens of milliseconds,
that is comfortable. For a step that returns immediately, the overhead dominates
and the honest answer is that such steps should not be separate steps.

**The practical ceiling on concurrent tasks.** The event loop is a single
thread running a ready queue; every additional runnable task adds to the work
done per iteration. Somewhere in the low thousands of simultaneously runnable
tasks, loop iteration itself becomes the bottleneck and latency degrades for
everything, including tasks that are ready to run. `SKEIN_GLOBAL_CONCURRENCY`
defaults to 64 — well below any such ceiling — because for I/O-bound agent
steps, concurrency beyond the point where the *downstream* dependency saturates
buys nothing and costs queueing. The limit is there to protect the dependencies
and the loop, not to be maximised.

**CPU-bound tools need a pool.** A tool that computes for 200 ms blocks the
event loop for 200 ms, stalling every other in-flight step in the process — the
single most common way an asyncio service loses its concurrency. `to_thread`
(via `ToolRegistry.register_sync`) fixes the *blocking*, but not the
parallelism: the GIL means threaded CPU work still serialises. Genuinely
CPU-bound tools want a `ProcessPoolExecutor`, at the cost of pickling arguments
and results across a process boundary. The target here is that no tool in the
default registry blocks for a measurable time, and that the sync-adapter path
exists so a tool that must block does not take the loop with it.

**WebSocket fan-out.** Each connected client costs one bounded queue and one
`put_nowait` per event. Fan-out is therefore linear in subscribers and the
per-event cost is small, but a workflow with a large `map` step emits events at
a high rate, and a client on a slow link cannot keep up. The design target is
that a slow client never applies backpressure to the scheduler: the recorder
drops events for a full subscriber queue rather than awaiting it, and the
durable trace on disk stays complete regardless. The viewer is a live view, not
a ledger.

**Where latency actually goes.** For a typical agent workflow the wall clock is
dominated by model inference, then by outbound HTTP, then by everything else
together. The scheduler's contribution is the time between a step's last
dependency settling and that step starting — which is one queue `get`, one
eligibility pass over the step list, and up to three semaphore acquisitions. The
eligibility pass is O(steps) per completion, so it is O(steps²) over a run;
fine for the hundreds of steps the budget caps at, and the thing to revisit
first if that cap is ever raised substantially.

## Things deliberately not done

- **No distributed scheduling.** One process owns a run start to finish. Runs
  are independent, so scaling is horizontal by replica, and the `Service`
  manifest uses client-IP affinity to keep a caller talking to the pod that owns
  its run. That is a workaround for the in-memory store rather than a design.
- **No persistence.** A restart loses in-flight runs. The drain handler exists
  so that a *planned* restart does not, but a crash does.
- **No expression language in `branch`.** `when` resolves to a value and its
  truthiness decides. Adding an evaluator would mean shipping `eval` — arbitrary
  code from a workflow definition, which may itself have come from a model — or
  maintaining a parser. The producing step can return a boolean.
- **Branch targets are skipped, not removed.** Mutating the graph mid-run would
  break "no step executes twice" for steps that no longer exist, and would make
  a recording and its replay have different step sets, so their hashes could
  never match.
