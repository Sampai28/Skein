"""Graceful shutdown: drain in-flight runs, then cancel what is left.

A SIGTERM arrives when Kubernetes decides to replace the pod, and the difference
between a good and a bad shutdown is entirely in what happens next. Exiting
immediately abandons in-flight runs mid-step. Waiting forever means the pod is
eventually SIGKILLed, which abandons them anyway *and* skips every cleanup path.

So: stop accepting new work, give what is running a bounded window to finish,
then cancel the remainder and let their cleanup run.

The drain deadline must be shorter than the container's
``terminationGracePeriodSeconds`` or the kernel kills the process partway
through cancellation — see ``k8s/deployment.yaml``, where the two are set
together for exactly this reason.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
from dataclasses import dataclass, field
from typing import Awaitable, Callable

logger = logging.getLogger("skein.shutdown")


@dataclass
class ShutdownController:
    """Owns the shutdown sequence and the flag everything else reads."""

    drain_timeout_s: float = 25.0
    cancel_timeout_s: float = 5.0

    #: Set as soon as a signal arrives. The API checks it to start refusing new
    #: submissions with 503, and the readiness probe checks it so the load
    #: balancer stops routing before the drain begins.
    draining: asyncio.Event = field(default_factory=asyncio.Event)
    _installed: bool = False
    _original: dict[int, object] = field(default_factory=dict)

    def install(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Register handlers for SIGTERM and SIGINT.

        ``loop.add_signal_handler`` where available. The alternative,
        ``signal.signal``, runs the handler on an arbitrary bytecode boundary
        rather than on the event loop, so it cannot safely touch loop state —
        setting an asyncio.Event from it is a race. On Windows the loop method
        is unimplemented, so the fallback is used and the handler does nothing
        but set a thread-safe flag.
        """
        if self._installed:
            return
        loop = loop or asyncio.get_running_loop()

        for signum in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(signum, self._on_signal, signum)
            except (NotImplementedError, RuntimeError):
                # Windows, or a non-main thread.
                with contextlib.suppress(ValueError, OSError):
                    self._original[signum] = signal.getsignal(signum)
                    signal.signal(signum, lambda s, _f: self._on_signal(s))
        self._installed = True

    def _on_signal(self, signum: int) -> None:
        if self.draining.is_set():
            # A second signal means someone is impatient. Honour it: the
            # supervisor is about to SIGKILL anyway, and a fast exit at least
            # runs atexit handlers.
            logger.warning("second signal %s received; exiting without drain", signum)
            raise SystemExit(1)
        logger.info("signal %s received; draining", signum)
        self.draining.set()

    def is_draining(self) -> bool:
        return self.draining.is_set()

    async def drain(
        self,
        active_count: Callable[[], int],
        cancel_all: Callable[[], Awaitable[None]],
    ) -> None:
        """Wait for in-flight work, then cancel whatever remains.

        ``active_count`` is polled rather than awaited on an event, because runs
        finish independently and there is no single future that represents "all
        of them". The poll interval is short enough to be invisible against a
        multi-second drain window.
        """
        deadline = asyncio.get_running_loop().time() + self.drain_timeout_s

        while active_count() > 0:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                logger.warning(
                    "drain window elapsed with %d run(s) still active; cancelling",
                    active_count(),
                )
                break
            await asyncio.sleep(min(0.1, remaining))

        if active_count() > 0:
            # shield: cancellation cleanup must complete even if the caller's
            # own shutdown path is itself cancelled — otherwise a shutdown that
            # is interrupted leaves runs half-torn-down with no terminal trace
            # events. Bounded by cancel_timeout_s so it cannot hang forever.
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(self.cancel_timeout_s):
                    await asyncio.shield(cancel_all())

        logger.info("drain complete; %d run(s) still active", active_count())

    def uninstall(self) -> None:
        for signum, handler in self._original.items():
            with contextlib.suppress(ValueError, OSError, TypeError):
                signal.signal(signum, handler)  # type: ignore[arg-type]
        self._original.clear()
        self._installed = False
