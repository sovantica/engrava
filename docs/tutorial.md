# Tutorial: a small notes memory

The [Quick Start](quickstart.md) shows the primitives in isolation. This
tutorial builds one small, real thing end to end — a personal-notes memory you
can search, link, and consolidate — typing each step yourself. By the end
you'll have a script that runs.

It uses no external services: embeddings come from a tiny deterministic hash
function. That keeps the tutorial dependency-free, but a hash carries no meaning.
The vector arm still produces a score for every note, and those scores still move
the ranking — they are just a function of the text's hash rather than of what it
says. So the keyword (FTS5) signal is the only one carrying a real relationship
to your query, while meaningless vector scores still influence where everything
lands. Swap in a provider backed by a semantic embedding model — see the
[Embeddings guide](guides/embeddings.md) — to search by meaning.
Read [Core Concepts](concepts.md) first if "thought", "cycle", or "reflection"
are unfamiliar.

## 1. Imports and a store

Open an in-memory store with a (toy) embedding provider, so the vector arm of
hybrid search is wired up end to end. The vectors are hashes, not meanings — see
the note above:

```python
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


def embed(text: str) -> list[float]:
    """A tiny deterministic stand-in. Use a real provider in production."""
    digest = hashlib.sha256(text.lower().encode("utf-8")).digest()
    return [byte / 255.0 for byte in (digest * 2)[:32]]
```

## 2. Ingest some notes

Each note becomes an `OBSERVATION` thought. We keep the returned records so we
can link them next:

```python
NOTES = [
    "Buy oat milk and coffee beans on the way home.",
    "The espresso machine descaling is overdue.",
    "Standup moved to 10am on Thursdays.",
    "Coffee tastes better with freshly ground beans.",
]


async def ingest(store, notes):
    records = []
    for index, text in enumerate(notes):
        record = ThoughtRecord(
            thought_id=str(uuid.uuid4()),
            thought_type=ThoughtType.OBSERVATION,
            essence=text[:200],
            content=text,
            priority=Priority.P3,
            lifecycle_status=LifecycleStatus.ACTIVE,
            created_cycle=index,        # one cycle per note here
            updated_cycle=index,
            source="notes",
        )
        records.append(await store.create_thought(record))
    return records
```

With `auto_embed=True` (step 5) each note is embedded on write.

## 3. Link related notes

Connect notes that are about the same thing with an `ASSOCIATED` edge — this is
what makes the memory a *graph*:

```python
async def link(store, a, b, weight=0.8):
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
```

## 4. Search the notes

Ask a question; `search_hybrid` embeds the query for you and returns ranked
`(thought_id, score)` tuples, which we turn back into text:

```python
async def search(store, query, cycle):
    result = await store.search_hybrid(query, top_k=3, current_cycle=cycle)
    print(f"\nQuery: {query!r}  (signals: {sorted(result.backends_used)})")
    for thought_id, score in result.results:
        record = await store.get_thought(thought_id)
        if record is not None:
            print(f"  {score:.3f}  {record.essence}")
```

## 5. Put it together

Wire the pieces into a `main()` and run it:

```python
async def main():
    provider = CallbackProvider(callback=embed, dimension=32, model_name="tutorial")
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
```

Run it and you'll see the top three notes for the coffee query, plus the total
count:

```text
Query: 'anything about coffee?'  (signals: ['fts5', 'priority', 'vector'])
  0.844  Coffee tastes better with freshly ground beans.
  0.504  The espresso machine descaling is overdue.
  0.494  Standup moved to 10am on Thursdays.

Stored 4 notes.
```

Of the two notes containing the word `coffee`, only one reaches the top three.
The other — `"Buy oat milk and coffee beans on the way home."` — never reaches
`top_k=3` at all, while a standup note that mentions neither coffee nor espresso
does. That is the toy `embed` showing. Two things
combine to produce it:

- Only two notes match the keyword query, and the FTS arm is min-max normalized
  across its candidates, so the weaker of the two matches is pinned to `0.0` —
  the oat-milk note loses its entire keyword contribution.
- The hash vectors still score every note, but a sha256 digest has no relation
  to meaning, so those scores are effectively arbitrary. They are what separates
  the bottom three: run the same query with `vector_weight=0.0` and those three
  come back with one and the same score (`0.043`, to three decimals). Here the
  arbitrary scores happen to put the standup note above the oat-milk note.

Swap the toy `embed` for a provider backed by a semantic embedding model (see the
[Embeddings guide](guides/embeddings.md)) and the vector scores start reflecting
what the notes say instead of what their digests do.

That's a working memory: ingest, embed, link, search.

The complete script is also shipped as
[`examples/notes_memory.py`](https://github.com/sovantica/engrava/blob/main/examples/notes_memory.py)
— run it directly with `python examples/notes_memory.py`.

## Where to go next

- **Make it an agent.** [Building a memory-backed agent](guides/agent-memory.md)
  turns this into a per-turn loop (retrieve before you answer, store the reply).
- **More tasks.** The [Recipes](recipes/index.md) cover TTL, dedup, session
  scoping, and scheduled consolidation.
- **Real embeddings.** Swap the toy `embed` for a provider in the
  [Embeddings guide](guides/embeddings.md).
- **Consolidation.** [Dreaming](dreaming.md) turns accumulating notes into
  higher-level reflections over time.
