#!/usr/bin/env python3
"""engrava quickstart — a five-minute introduction.

Requirements:
    pip install 'engrava[embeddings-local]'

The ``[embeddings-local]`` extra pulls ``sentence-transformers`` and
``torch`` so vector search has a real ENCODER (~30-90 MB model on first
run, then cached locally). The encoder is NOT a language model: it
turns text into a fixed-size vector and is fully local — no API keys
and no network after the first download.

Run this directly:
    python examples/quickstart.py

What it does:
  1. Boots an in-memory engrava store with a local embedding encoder
     and auto-embedding turned on.
  2. Ingests a handful of percepts (things the agent learned about the
     user) and two utterances (replies the agent already sent), using
     the self-anchored metadata helpers shipped in ``engrava.metadata``.
  3. Runs one dreaming consolidation cycle (deterministic, no LLM).
  4. Queries the memory with hybrid search and prints the top results.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys

import aiosqlite

from engrava import (
    DreamingConfig,
    DreamingExtension,
    DreamingGates,
    LifecycleStatus,
    Priority,
    SqliteEngravaCore,
    ThoughtRecord,
    ThoughtType,
    percept,
    utterance,
)

# The local encoder is required for vector search. Surface a clear
# actionable hint instead of letting the import fail deep inside the
# asyncio stack (same pattern as the synthetic benchmark uses).
if importlib.util.find_spec("sentence_transformers") is None:
    sys.stderr.write(
        "This example requires the local embedding encoder:\n"
        "    pip install 'engrava[embeddings-local]'\n"
        "(This is an ENCODER model, not an LLM. No API keys needed.)\n",
    )
    sys.exit(2)


CONTENT_PRINT_LIMIT = 80
CONTENT_TRUNCATED_LEN = 77

PERCEPTS = [
    "My favorite color is teal.",
    "I started learning piano on March 15.",
    "Last weekend I hiked Mount Tammany.",
    "My dog's name is Atlas.",
    "I prefer sparkling water over still.",
    "I usually go for a long run on Sunday mornings.",
    "I learned Spanish in college but rarely use it now.",
    "I bought a sourdough starter from a local bakery.",
]
UTTERANCES = [
    "That sounds fun!",
    "I see, thanks for sharing.",
]
QUERY = "What is the user's favorite color?"
TOP_K = 3


def _percept_thought(thought_id: str, content: str, cycle: int) -> ThoughtRecord:
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.OBSERVATION,
        essence=content[:200],
        content=content,
        priority=Priority.P3,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=cycle,
        updated_cycle=cycle,
        source="quickstart",
        metadata=percept(source_id="user-demo", label="user"),
    )


def _utterance_thought(thought_id: str, content: str, cycle: int) -> ThoughtRecord:
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.OUTPUT_DRAFT,
        essence=content[:200],
        content=content,
        priority=Priority.P3,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=cycle,
        updated_cycle=cycle,
        source="quickstart",
        metadata=utterance(),
    )


async def main() -> None:
    """Run the quickstart walkthrough end-to-end."""
    from engrava.embeddings.sentence_transformer import (  # noqa: PLC0415
        SentenceTransformerProvider,
    )

    provider = SentenceTransformerProvider(model_name="all-MiniLM-L6-v2")
    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(conn, embedding_provider=provider, auto_embed=True)
        await store.ensure_schema()

        for index, content in enumerate(PERCEPTS):
            await store.create_thought(_percept_thought(f"p{index:02d}", content, cycle=index))
        for index, content in enumerate(UTTERANCES):
            base_cycle = len(PERCEPTS) + index
            await store.create_thought(_utterance_thought(f"u{index:02d}", content, base_cycle))

        dreaming = DreamingExtension(
            config=DreamingConfig(
                enabled=True,
                gates=DreamingGates(
                    min_confirmations=0,
                    min_age_cycles=0,
                    allow_zero_confirmation=True,
                ),
            ),
        )
        await dreaming.run_consolidation(store, current_cycle=len(PERCEPTS) + len(UTTERANCES))

        result = await store.search_hybrid(QUERY, top_k=TOP_K)

        print(f"Query: {QUERY}")
        print()
        for rank, (thought_id, score) in enumerate(result.results, start=1):
            record = await store.get_thought(thought_id)
            if record is None:
                continue
            snippet = (
                record.content
                if len(record.content) <= CONTENT_PRINT_LIMIT
                else f"{record.content[:CONTENT_TRUNCATED_LEN]}..."
            )
            print(f"  {rank}. [{record.thought_type.value}] {snippet}  (score={score:.3f})")


if __name__ == "__main__":
    asyncio.run(main())
