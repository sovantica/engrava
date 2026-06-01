#!/usr/bin/env python3
"""Minimal agent using only engrava — no external services, no LLM.

Demonstrates:
  1. Creating a thought-graph from plain observations
  2. Building edges based on keyword overlap
  3. Querying with ``search_similar`` (cosine similarity)
  4. Custom scoring hooks

Usage::

    pip install -e packages/engrava
    python packages/engrava/examples/simple_agent.py

Requires ``sentence-transformers`` for embeddings::

    pip install sentence-transformers

"""

from __future__ import annotations

import asyncio
import uuid

import numpy as np

from engrava import (
    DefaultEngravaHooks,
    EdgeRecord,
    EdgeType,
    LifecycleStatus,
    Priority,
    ScoringContext,
    SqliteEngravaCore,
    ThoughtRecord,
    ThoughtType,
)

# ---------------------------------------------------------------------------
# 1. Custom hooks — weight recent thoughts higher
# ---------------------------------------------------------------------------


class RecencyHooks(DefaultEngravaHooks):
    """Score thoughts where newer cycle numbers rank higher."""

    async def score_function(
        self,
        thought: ThoughtRecord,
        _context: ScoringContext,
    ) -> float:
        """Return a recency-biased relevance score.

        Args:
            thought: The thought to score.
            context: Scoring context with query metadata.

        Returns:
            Float score combining base priority and recency bonus.

        """
        priority_scores = {"P1": 4.0, "P2": 3.0, "P3": 2.0, "P4": 1.0}
        base = priority_scores.get(thought.priority.value, 0.0)
        recency_bonus = min(thought.created_cycle / 100.0, 0.2)
        return base + recency_bonus


# ---------------------------------------------------------------------------
# 2. Simple embedding function (bag of words → random-but-deterministic)
# ---------------------------------------------------------------------------

EMBED_DIM = 64


def embed(text: str) -> list[float]:
    """Produce a deterministic pseudo-embedding from text.

    Real agents should use ``sentence-transformers`` or an API.

    Args:
        text: Input text to embed.

    Returns:
        List of floats (length EMBED_DIM).

    """
    rng = np.random.default_rng(seed=hash(text) % (2**31))
    return rng.standard_normal(EMBED_DIM).tolist()


# ---------------------------------------------------------------------------
# 3. Build the graph
# ---------------------------------------------------------------------------

OBSERVATIONS = [
    "The cat sat on the warm windowsill",
    "Rain is expected later this afternoon",
    "Cats dislike getting wet in the rain",
    "The windowsill overlooks the garden",
    "Garden plants need rain to grow",
]


def _make_thought(text: str, cycle: int) -> ThoughtRecord:
    return ThoughtRecord(
        thought_id=uuid.uuid4().hex,
        essence=text[:200],
        content=text,
        thought_type=ThoughtType.OBSERVATION,
        source="example-agent",
        lifecycle_status=LifecycleStatus.ACTIVE,
        priority=Priority.P2,
        created_cycle=cycle,
        updated_cycle=cycle,
    )


def _find_overlaps(texts: list[str]) -> list[tuple[int, int]]:
    """Return index pairs sharing at least one non-trivial keyword."""
    stop = {"the", "a", "an", "is", "on", "in", "to", "of", "this"}
    word_sets = [set(t.lower().split()) - stop for t in texts]
    pairs: list[tuple[int, int]] = []
    for i in range(len(texts)):
        pairs.extend((i, j) for j in range(i + 1, len(texts)) if word_sets[i] & word_sets[j])
    return pairs


async def main() -> None:
    """Run the minimal agent demo."""
    import aiosqlite  # noqa: PLC0415

    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row

    hooks = RecencyHooks()
    store = SqliteEngravaCore(conn, hooks=hooks)
    await store.ensure_schema()

    # --- create thoughts ---
    thoughts: list[ThoughtRecord] = []
    for idx, obs in enumerate(OBSERVATIONS):
        t = _make_thought(obs, cycle=idx + 1)
        t = await store.create_thought(t)
        await store.store_embedding(t.thought_id, embed(obs))
        thoughts.append(t)
        print(f"  [+] thought {t.thought_id[:8]}  cycle={t.created_cycle}  {obs!r}")

    # --- create edges from keyword overlap ---
    overlaps = _find_overlaps(OBSERVATIONS)
    for i, j in overlaps:
        edge = EdgeRecord(
            edge_id=uuid.uuid4().hex,
            from_thought_id=thoughts[i].thought_id,
            to_thought_id=thoughts[j].thought_id,
            edge_type=EdgeType.ASSOCIATED,
            weight=0.8,
            created_cycle=1,
        )
        await store.create_edge(edge)
        print(f"  [~] edge {i} -> {j}  ({OBSERVATIONS[i][:30]}... -> {OBSERVATIONS[j][:30]}...)")

    # --- similarity search ---
    query_text = "wet cats and rain"
    query_vec = embed(query_text)
    results = await store.search_similar(query_vec, top_k=3, threshold=0.0)

    print(f"\n  Query: {query_text!r}")
    print("  Top results:")
    for tid, score in results:
        t = await store.get_thought(tid)
        if t:
            print(f"    score={score:.3f}  {t.content!r}")

    # --- graph traversal (1-hop from top result) ---
    if results:
        top_id = results[0][0]
        edges = await store.get_edges(top_id)
        print(f"\n  Edges from top result ({top_id[:8]}):")
        for e in edges:
            neighbor_id = e.to_thought_id if e.from_thought_id == top_id else e.from_thought_id
            neighbor = await store.get_thought(neighbor_id)
            if neighbor:
                print(f"    -> {neighbor.content!r}")

    await conn.close()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
