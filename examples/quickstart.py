"""engrava quickstart — a five-minute introduction.

Requirements:
    pip install 'engrava[embeddings-local]'

The ``[embeddings-local]`` extra pulls ``sentence-transformers`` and
``torch`` — a large one-time download (~550+ MB on the PyPI
Linux/x86_64 wheel for Python 3.11; see docs/configuration.md's
"Quick-start profiles" section for the exact, platform-scoped numbers)
— so vector search has a real ENCODER, plus a further encoder model
download on first run, then cached locally. The encoder is NOT a
language model: it turns text into a fixed-size vector and is fully
local — no API keys, and no network after the first download once
``HF_HUB_OFFLINE=1`` and ``TRANSFORMERS_OFFLINE=1`` are set (see
docs/configuration.md's "local" profile for why those two variables
are needed).

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
import contextlib
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
    """Run the quickstart walkthrough end-to-end."""
    from engrava.embeddings.sentence_transformer import (  # noqa: PLC0415
        SentenceTransformerProvider,
    )

    provider = SentenceTransformerProvider(model_name="all-MiniLM-L6-v2")
    conn = await aiosqlite.connect(":memory:")
    try:
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
    except BaseException:
        await _close_quietly(conn)
        raise
    else:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
