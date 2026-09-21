"""A small notes memory built with engrava — the companion to the tutorial.

This is the complete, runnable version of ``docs/tutorial.md``: ingest a few
notes, embed them, link related ones with an edge, and search them. It uses a
tiny deterministic hash function for the embeddings so it runs with no external
services. A hash carries no meaning: the vector scores are a function of the
text's digest rather than of what it says, so the keyword (FTS5) signal is the
only one related to the query while meaningless vector scores still influence
where everything lands — and only one of the two coffee notes reaches
``top_k=3``. See the tutorial for the
walk-through. Swap in a provider backed by a semantic embedding model (see the
Embeddings guide) to search by meaning.

Run directly::

    python examples/notes_memory.py
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import sys
import uuid

import aiosqlite

from engrava import (
    CallbackProvider,
    EdgeRecord,
    EdgeType,
    LifecycleStatus,
    Priority,
    SqliteEngravaCore,
    ThoughtRecord,
    ThoughtType,
)

EMBED_DIM = 32

NOTES = [
    "Buy oat milk and coffee beans on the way home.",
    "The espresso machine descaling is overdue.",
    "Standup moved to 10am on Thursdays.",
    "Coffee tastes better with freshly ground beans.",
]


def embed(text: str) -> list[float]:
    """A tiny deterministic stand-in. Use a real provider in production."""
    digest = hashlib.sha256(text.lower().encode("utf-8")).digest()
    return [byte / 255.0 for byte in (digest * 2)[:EMBED_DIM]]


async def ingest(store: SqliteEngravaCore, notes: list[str]) -> list[ThoughtRecord]:
    """Store each note as an OBSERVATION thought; return the persisted records."""
    records: list[ThoughtRecord] = []
    for index, text in enumerate(notes):
        record = ThoughtRecord(
            thought_id=str(uuid.uuid4()),
            thought_type=ThoughtType.OBSERVATION,
            essence=text[:200],
            content=text,
            priority=Priority.P3,
            lifecycle_status=LifecycleStatus.ACTIVE,
            created_cycle=index,
            updated_cycle=index,
            source="notes",
        )
        records.append(await store.create_thought(record))
    return records


async def link(
    store: SqliteEngravaCore,
    a: ThoughtRecord,
    b: ThoughtRecord,
    weight: float = 0.8,
) -> None:
    """Connect two related notes with an ASSOCIATED edge."""
    await store.create_edge(
        EdgeRecord(
            edge_id=str(uuid.uuid4()),
            from_thought_id=a.thought_id,
            to_thought_id=b.thought_id,
            edge_type=EdgeType.ASSOCIATED,
            weight=weight,
            created_cycle=0,
        )
    )


async def search(store: SqliteEngravaCore, query: str, cycle: int) -> None:
    """Print the top matches for a query (search embeds the query for you)."""
    result = await store.search_hybrid(query, top_k=3, current_cycle=cycle)
    print(f"\nQuery: {query!r}  (signals: {sorted(result.backends_used)})")
    for thought_id, score in result.results:
        record = await store.get_thought(thought_id)
        if record is not None:
            print(f"  {score:.3f}  {record.essence}")


async def _close_quietly(conn: aiosqlite.Connection) -> None:
    """Close *conn* during cleanup without letting an ordinary close failure
    replace the body's error.

    Used only from the exception path below, after the body has already
    raised: an unconditional ``await conn.close()`` there would let a close
    failure overwrite whatever the block above actually raised. A failure
    that is an ``Exception`` -- a locked database, a disk error -- is
    reported to stderr and swallowed, so the body's error is the one that
    ends up propagating.

    A ``BaseException`` raised **by the close itself** that is not an
    ``Exception`` -- ``CancelledError``, ``KeyboardInterrupt``,
    ``SystemExit``, an exception group with at least one non-``Exception``
    leaf (an ``ExceptionGroup`` is itself an ``Exception`` -- constructing
    ``BaseExceptionGroup(msg, [RuntimeError(...)])`` actually returns one --
    so it is caught and reported below like any other close failure; only a
    group with a genuine ``BaseException`` leaf is a real
    ``BaseExceptionGroup`` and escapes) -- is deliberately left uncaught
    here: it propagates and supersedes the body's error, because a
    cancellation or interrupt arriving while the close itself is running is
    current, real information the caller needs, not a subordinate detail of
    a failure that already happened.

    Producing the report of an ordinary close failure is different: nothing
    downstream depends on that warning actually being printed, so this is
    deliberately broader than the catch above and swallows *any*
    ``BaseException`` -- ``str(exc)`` raising, ``print`` raising (a broken
    or already-closed stderr), or even a cancellation or interrupt landing
    in that narrow window -- rather than let a failure in a best-effort log
    line replace the body's error the way an unreported close failure
    would.
    """
    try:
        await conn.close()
    except Exception as exc:  # noqa: BLE001 - reports Exception only; a BaseException
        # raised by the close itself is left to propagate and supersede
        # the body's error instead, see above.
        with contextlib.suppress(BaseException):
            # Deliberately broader than the catch above -- see the
            # docstring: this is a best-effort report, so even a
            # cancellation landing inside str(exc) or print() itself must
            # not replace the body's error either.
            print(f"warning: failed to close the database connection: {exc}", file=sys.stderr)


async def main() -> None:
    """Build the notes memory and run a search over it."""
    provider = CallbackProvider(callback=embed, dimension=EMBED_DIM, model_name="tutorial")
    conn = await aiosqlite.connect(":memory:")
    try:
        conn.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(conn, embedding_provider=provider, auto_embed=True)
        await store.ensure_schema()

        notes = await ingest(store, NOTES)

        # link the two coffee-related notes
        await link(store, notes[0], notes[3])

        await search(store, "anything about coffee?", cycle=len(NOTES))

        total = await store.count_thoughts()
        print(f"\nStored {total} notes.")
    except BaseException:
        await _close_quietly(conn)
        raise
    else:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
