#!/usr/bin/env python3
"""A memory-backed agent loop using only engrava — no external services.

This is the canonical "wire engrava into an agent" example: a per-turn loop
that, for each user message,

  1. stores the message as a ``percept`` thought,
  2. retrieves relevant prior memory with ``search_hybrid``,
  3. builds a prompt from the retrieved essences and calls an LLM
     (a deterministic stand-in here — swap in your real model),
  4. stores the agent's reply as an ``utterance`` thought,
  5. records the action it took (an ``ActionRecord``),
  6. advances the cycle counter, and
  7. runs dreaming consolidation every N turns.

The cycle counter is the agent's logical clock: engrava never advances it for
you, so this loop owns it and increments it once per turn (see the Core
Concepts docs). On restart you would recover it from the maximum stored
``created_cycle``; this in-memory demo just starts at 0.

No LLM and no embedding API are required: the "LLM" is a canned responder and
embeddings come from a deterministic ``CallbackProvider``. Run directly::

    python examples/agent_loop.py
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid

import aiosqlite

from engrava import (
    ActionRecord,
    ActionStatus,
    ActionType,
    CallbackProvider,
    DreamingConfig,
    DreamingExtension,
    DreamingGates,
    LifecycleStatus,
    Priority,
    SqliteEngravaCore,
    ThoughtRecord,
    ThoughtType,
    VerificationStatus,
    percept,
    utterance,
)

EMBED_DIM = 64
CONSOLIDATE_EVERY = 3
RETRIEVE_TOP_K = 3


def _deterministic_embed(text: str) -> list[float]:
    """Map text to a stable pseudo-embedding (no model, fully reproducible).

    A real agent passes a real provider (sentence-transformers, OpenAI, …);
    this keeps the example dependency-free and deterministic across runs.
    """
    digest = hashlib.sha256(text.lower().encode("utf-8")).digest()
    # Repeat the 32-byte digest to fill EMBED_DIM bytes, then scale to [0, 1].
    repeats = (EMBED_DIM // len(digest)) + 1
    stretched = (digest * repeats)[:EMBED_DIM]
    return [byte / 255.0 for byte in stretched]


def _mock_llm(prompt: str) -> str:
    """Stand in for an LLM call. Replace with your provider."""
    return f"(reply based on {prompt.count('-')} retrieved memories)"


async def _store_percept(
    store: SqliteEngravaCore,
    text: str,
    cycle: int,
    user_id: str,
    session_id: str,
    turn_index: int,
) -> ThoughtRecord:
    """Persist an incoming user message as an OBSERVATION percept.

    The percept metadata is extended with ``session_id`` and ``turn_index`` so
    every memory is anchored to its conversation and position within it — the
    keys you'd later filter or post-filter on for per-session retrieval.
    """
    record = ThoughtRecord(
        thought_id=str(uuid.uuid4()),
        thought_type=ThoughtType.OBSERVATION,
        essence=text[:200],
        content=text,
        priority=Priority.P2,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=cycle,
        updated_cycle=cycle,
        source=user_id,
        metadata={
            **percept(source_id=user_id, label="user"),
            "session_id": session_id,
            "turn_index": turn_index,
        },
    )
    return await store.create_thought(record)


async def _retrieve_context(store: SqliteEngravaCore, query: str, cycle: int) -> list[str]:
    """Return the essences of the most relevant prior memories."""
    result = await store.search_hybrid(
        query,
        query_vector=_deterministic_embed(query),
        top_k=RETRIEVE_TOP_K,
        current_cycle=cycle,  # the agent clock — drives the recency signal
    )
    essences: list[str] = []
    for thought_id, _score in result.results:
        record = await store.get_thought(thought_id)
        if record is not None:
            essences.append(record.essence)
    return essences


async def _store_utterance(
    store: SqliteEngravaCore,
    reply: str,
    cycle: int,
    session_id: str,
    turn_index: int,
) -> ThoughtRecord:
    """Persist the agent's own reply as an OUTPUT_DRAFT utterance.

    Tagged with the same ``session_id``/``turn_index`` as the percept it
    answered, so a turn's input and output stay linked.
    """
    record = ThoughtRecord(
        thought_id=str(uuid.uuid4()),
        thought_type=ThoughtType.OUTPUT_DRAFT,
        essence=reply[:200],
        content=reply,
        priority=Priority.P3,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=cycle,
        updated_cycle=cycle,
        source="agent",
        metadata={
            **utterance(),
            "session_id": session_id,
            "turn_index": turn_index,
        },
    )
    return await store.create_thought(record)


async def _record_action(store: SqliteEngravaCore, source_thought_id: str, intent: str) -> None:
    """Record that the agent took an action, linked to the source thought."""
    await store.create_action(
        ActionRecord(
            action_id=str(uuid.uuid4()),
            source_thought_id=source_thought_id,
            action_type=ActionType.MESSAGE,
            intent=intent,
            status=ActionStatus.CONFIRMED,
            verification_status=VerificationStatus.CONFIRMED,
        )
    )


async def main() -> None:
    """Run a few turns of a memory-backed agent over an in-memory store."""
    provider = CallbackProvider(
        callback=_deterministic_embed,
        dimension=EMBED_DIM,
        model_name="demo-deterministic",
    )
    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(conn, embedding_provider=provider, auto_embed=True)
        await store.ensure_schema()

        dreaming = DreamingExtension(
            config=DreamingConfig(
                enabled=True,
                gates=DreamingGates(min_confirmations=0, min_age_cycles=0),
            ),
        )

        user_id = "user-demo"
        session_id = str(uuid.uuid4())  # one conversation; tag every memory with it
        conversation = [
            "I'm planning a trip to Japan in spring.",
            "What's the weather like in Kyoto in April?",
            "Remind me which city I'm visiting.",
            "I prefer trains over flights for getting around.",
        ]

        cycle = 0  # the agent's logical clock; advance once per turn
        for turn_index, user_message in enumerate(conversation):
            # 1. store the incoming message (anchored to session + turn)
            percept_thought = await _store_percept(
                store, user_message, cycle, user_id, session_id, turn_index
            )

            # 2. retrieve relevant prior memory
            context = await _retrieve_context(store, user_message, cycle)

            # 3. build a prompt and call the LLM (stand-in)
            prompt = "Context:\n" + "\n".join(f"- {c}" for c in context)
            prompt += f"\n\nUser: {user_message}\nAssistant:"
            reply = _mock_llm(prompt)

            # 4. store the agent's reply (same session + turn as its percept)
            await _store_utterance(store, reply, cycle, session_id, turn_index)

            # 5. record the action taken
            await _record_action(store, percept_thought.thought_id, intent="answered user")

            print(f"cycle {cycle}: user={user_message!r}")
            print(f"          retrieved {len(context)} memory(ies); reply={reply!r}")

            # 6. advance the clock
            cycle += 1

            # 7. consolidate periodically
            if cycle % CONSOLIDATE_EVERY == 0:
                result = await dreaming.run_consolidation(store, current_cycle=cycle)
                print(f"          [dreaming] promoted={result.promoted_count}")

        total = await store.count_thoughts()
        print(f"\nDone. {total} thoughts stored across {cycle} turns.")


if __name__ == "__main__":
    asyncio.run(main())
