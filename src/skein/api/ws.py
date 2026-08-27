"""The WebSocket stream: live step status for the DAG viewer.

One subscriber queue per connection, fed by the run's
:class:`~skein.trace.recorder.TraceRecorder`. The recorder drops events for a
subscriber whose queue is full rather than awaiting it, so a slow browser cannot
apply backpressure to the scheduler — see the note on ``TraceRecorder.publish``.

The lifecycle here is the fiddly part, and it is where WebSocket handlers
usually leak. Three things must happen no matter how the connection ends:
unsubscribe from the recorder, stop the sender task, and not raise out of the
handler on a disconnect (which is a normal event, not an error).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from skein.runtime.store import RunHandle

logger = logging.getLogger("skein.ws")

#: How long to wait for a new event before sending a keepalive. Proxies and load
#: balancers close idle WebSockets, commonly at 60s, and a long-running workflow
#: can legitimately produce no events for minutes.
KEEPALIVE_S = 20.0


async def stream_run(websocket: WebSocket, handle: RunHandle) -> None:
    await websocket.accept()
    queue = handle.recorder.subscribe()

    try:
        # Replay history first, then stream. Without this a client that connects
        # a second after submitting misses everything that already happened and
        # renders a graph of pending nodes that are actually finished.
        for event in list(handle.recorder.events):
            await websocket.send_text(event.to_json_line())

        if handle.state.status.is_terminal:
            await websocket.send_json(
                {"kind": "stream_end", "run_id": handle.state.run_id,
                 "status": handle.state.status.value}
            )
            return

        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=KEEPALIVE_S)
            except TimeoutError:
                if handle.state.status.is_terminal:
                    break
                # A ping frame, not a JSON message: it keeps the connection
                # alive without the client needing to filter it out of the
                # event stream.
                await websocket.send_json({"kind": "keepalive"})
                continue

            await websocket.send_text(event.to_json_line())

            if handle.state.status.is_terminal and queue.empty():
                break

        await websocket.send_json(
            {"kind": "stream_end", "run_id": handle.state.run_id,
             "status": handle.state.status.value}
        )

    except WebSocketDisconnect:
        # The client went away. Entirely normal — no log, no error.
        pass
    except asyncio.CancelledError:
        # Server shutdown. Close politely if the socket is still usable, then
        # re-raise so the task actually stops.
        with contextlib.suppress(Exception):
            if websocket.client_state is WebSocketState.CONNECTED:
                await websocket.close(code=1012)  # service restart
        raise
    except Exception:  # noqa: BLE001 - one bad socket must not kill the app
        logger.exception("websocket stream failed for run %s", handle.state.run_id)
    finally:
        # Always unsubscribe. A recorder holding queues for departed clients
        # accumulates them for the life of the run and publishes into each one
        # on every event.
        handle.recorder.unsubscribe(queue)
        with contextlib.suppress(Exception):
            if websocket.client_state is WebSocketState.CONNECTED:
                await websocket.close()
