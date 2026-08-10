#!/usr/bin/env python3
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
import hashlib
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


async def main() -> None:
    """Build the notes memory and run a search over it."""
    provider = CallbackProvider(callback=embed, dimension=EMBED_DIM, model_name="tutorial")
    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(conn, embedding_provider=provider, auto_embed=True)
        await store.ensure_schema()

        notes = await ingest(store, NOTES)

        # link the two coffee-related notes
        await link(store, notes[0], notes[3])

        await search(store, "anything about coffee?", cycle=len(NOTES))

        total = await store.count_thoughts()
        print(f"\nStored {total} notes.")


if __name__ == "__main__":
    asyncio.run(main())
