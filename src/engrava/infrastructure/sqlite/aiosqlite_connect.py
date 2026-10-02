"""A connect helper that waits for aiosqlite's worker thread after a failure.

In aiosqlite 0.22, awaiting ``aiosqlite.connect()`` starts its background worker thread
before knowing whether the underlying ``sqlite3`` connect call will succeed.
On failure the worker is told to stop, with a future on the running loop, but
nothing waits for it, so a caller whose event loop closes right after can leave
that thread posting to a loop that is no longer there.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Final

import aiosqlite

#: Seconds of event-loop time after which :func:`connect` starts no new poll of
#: aiosqlite's worker thread after a failed connect.
WORKER_STOP_TIMEOUT_SECONDS: Final = 2.0

_POLL_INTERVAL_SECONDS: Final = 0.05


async def connect(
    database: str,
    **kwargs: Any,  # noqa: ANN401 -- forwarded verbatim to aiosqlite.connect
) -> aiosqlite.Connection:
    """Open an aiosqlite connection, as ``await aiosqlite.connect(...)`` does.

    On a failed connect under aiosqlite 0.22, polls its worker thread until it
    stops or :data:`WORKER_STOP_TIMEOUT_SECONDS` of loop time have passed, then
    re-raises the original exception. A cancellation or ``KeyboardInterrupt``
    raised while it sleeps between polls is held and re-raised instead, once
    the polling ends; an interruption raised outside those sleeps ends the
    polling early.
    """
    conn = aiosqlite.connect(database, **kwargs)
    try:
        return await conn
    except BaseException:
        await _wait_for_worker_stop(_stopping_worker_thread(conn))
        raise


def _stopping_worker_thread(conn: object) -> threading.Thread | None:
    """Return the worker thread a failed aiosqlite 0.22 connect told to stop.

    Returns ``None`` for older aiosqlite, where the connection is itself the
    thread and stopping it posts nothing back to the loop.
    """
    if isinstance(conn, threading.Thread):
        return None
    # aiosqlite 0.22 keeps the worker on the private `_thread` attribute, the
    # only handle on the thread its failed connect has just told to stop.
    found = getattr(conn, "_thread", None)
    return found if isinstance(found, threading.Thread) else None


async def _wait_for_worker_stop(thread: threading.Thread | None) -> None:
    """Poll *thread* until it stops or the timeout has passed.

    A cancellation or ``KeyboardInterrupt`` raised while it sleeps between polls
    does not end the polling early; the last one is re-raised after it. An
    interruption raised outside those sleeps ends the polling early.
    """
    if thread is None:
        return
    loop = asyncio.get_running_loop()
    deadline = loop.time() + WORKER_STOP_TIMEOUT_SECONDS
    interrupted: asyncio.CancelledError | KeyboardInterrupt | None = None
    while thread.is_alive() and loop.time() < deadline:
        try:
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)
        except (asyncio.CancelledError, KeyboardInterrupt) as exc:
            interrupted = exc
    if interrupted is not None:
        raise interrupted
