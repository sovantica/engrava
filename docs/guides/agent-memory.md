# Building a memory-backed agent

This guide shows the canonical way to wire Engrava into an agent's turn loop:
give a chat/agent long-term memory that persists across sessions and surfaces
relevant context on every turn. It's the end-to-end pattern behind Engrava's
one-line pitch — "the memory database for AI agents."

A complete, runnable version of everything here ships as
[`examples/agent_loop.py`](https://github.com/sovantica/engrava/blob/main/examples/agent_loop.py)
— no LLM or embedding API required (it uses a canned responder and a
deterministic embedder). This page walks through the shape of that loop.

> New to the model (thought, edge, reflection, **cycle**)? Read
> [Core Concepts](../concepts.md) first — this guide assumes those terms.

## The loop, in one picture

Per user turn:

```
user message
   │
   ▼
1. store it as a percept  ──────────────►  create_thought(OBSERVATION)
2. retrieve relevant memory  ───────────►  search_hybrid(query, current_cycle)
3. build prompt from retrieved essences ─►  call your LLM
4. store the reply as an utterance ─────►  create_thought(OUTPUT_DRAFT)
5. record the action taken ─────────────►  create_action(ActionRecord)
6. advance the cycle counter ───────────►  cycle += 1   (you own this clock)
   │
   └─ every N turns ────────────────────►  dreaming.run_consolidation(current_cycle)
```

## Setup

Create one store for the lifetime of the agent. Configure an embedding provider
so retrieval is semantic (the example uses a deterministic stand-in; in
production pass a real provider such as `SentenceTransformerProvider` or
`OpenAICompatibleProvider`, configurable via
[`engrava.yaml`](../configuration.md)):

```python
import aiosqlite
from engrava import SqliteEngravaCore, CallbackProvider

provider = CallbackProvider(
    callback=my_embed_fn,       # swap in a real provider in production
    dimension=64,
    model_name="demo",
)
conn = await aiosqlite.connect("agent-memory.db")   # a file persists across runs
conn.row_factory = aiosqlite.Row
store = SqliteEngravaCore(conn, embedding_provider=provider, auto_embed=True)
await store.ensure_schema()
```

`auto_embed=True` means thoughts are embedded on write. At search time you may
pass an explicit `query_vector`; if you omit it, the store embeds the query
text for you **when an embedding provider is configured**. Passing it yourself
is handy when you've already computed the vector or want a different query
representation.

## Step 1 — store the incoming message (a *percept*)

Each user message becomes an `OBSERVATION` thought, tagged with `percept(...)`
metadata so its origin is recorded. Extend that metadata with a `session_id`
(which conversation) and `turn_index` (position within it) so every memory is
anchored to its conversation — these are the keys you'd later filter on (or
post-filter on) to scope retrieval to one session or user:

```python
import uuid
from engrava import ThoughtRecord, ThoughtType, Priority, LifecycleStatus, percept

async def store_percept(store, text, cycle, user_id, session_id, turn_index):
    record = ThoughtRecord(
        thought_id=str(uuid.uuid4()),
        thought_type=ThoughtType.OBSERVATION,
        essence=text[:200],          # the prompt-facing one-liner
        content=text,                # the full message
        priority=Priority.P2,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=cycle,         # the agent clock (see step 6)
        updated_cycle=cycle,
        source=user_id,
        metadata={
            **percept(source_id=user_id, label="user"),
            "session_id": session_id,
            "turn_index": turn_index,
        },
    )
    return await store.create_thought(record)
```

## Step 2 — retrieve relevant memory

Before calling the LLM, pull the most relevant prior memories with
`search_hybrid`. Pass `current_cycle` so the recency signal works, and turn the
returned `(thought_id, score)` tuples back into text via `get_thought`:

```python
async def retrieve_context(store, query, cycle):
    result = await store.search_hybrid(
        query,
        query_vector=my_embed_fn(query),   # optional: omit to let the provider embed `query`
        top_k=3,
        current_cycle=cycle,
    )
    essences = []
    for thought_id, _score in result.results:
        record = await store.get_thought(thought_id)
        if record is not None:
            essences.append(record.essence)   # essence = prompt-ready text
    return essences
```

`result.results` is a list of `(thought_id, score)` — Engrava returns IDs, not
records, so you fetch the ones you want. `result.backends_used` tells you which
signals contributed (e.g. `{"fts5", "vector", "recency"}`).

## Step 3 — build the prompt and call your LLM

This is the only step that touches your model. Engrava is LLM-free; you own the
call:

```python
prompt = "Context:\n" + "\n".join(f"- {c}" for c in context)
prompt += f"\n\nUser: {user_message}\nAssistant:"
reply = await my_llm(prompt)        # your provider here
```

## Step 4 — store the agent's reply (an *utterance*)

Persist what the agent said as an `OUTPUT_DRAFT` thought with `utterance(...)`
metadata, so the agent's own outputs are part of memory too:

```python
from engrava import utterance

async def store_utterance(store, reply, cycle, session_id, turn_index):
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
        metadata={                       # same session + turn as the percept it answered
            **utterance(),
            "session_id": session_id,
            "turn_index": turn_index,
        },
    )
    return await store.create_thought(record)
```

## Step 5 — record the action taken (optional)

If your agent *does* things (sends a message, calls a tool), record each as an
`ActionRecord` linked to the source thought. This is how the audit/action
surface tracks what the agent did and whether it succeeded:

```python
from engrava import ActionRecord, ActionType, ActionStatus, VerificationStatus

await store.create_action(
    ActionRecord(
        action_id=str(uuid.uuid4()),
        source_thought_id=percept_thought.thought_id,
        action_type=ActionType.MESSAGE,        # or TOOL_CALL / CLI_OUTPUT / STATE_UPDATE
        intent="answered user",
        status=ActionStatus.CONFIRMED,
        verification_status=VerificationStatus.CONFIRMED,
    )
)
```

Read them back with `await store.get_actions(thought_id)`.

## Step 6 — advance the cycle

A **cycle** is the agent's logical clock, and **you own it** — Engrava never
advances or persists it. Increment it once per turn and use it for
`created_cycle`/`updated_cycle` and the `current_cycle` you pass to search and
consolidation:

```python
cycle = 0
while running:
    ...                  # steps 1–5 use `cycle`
    cycle += 1
```

If you leave it at a constant, recency can't distinguish old memories from new
and dreaming's age gate never opens (see
[Cycle (the agent clock)](../concepts.md#cycle-the-agent-clock)). On restart,
recover it so it stays monotonic — see [Persistence across restarts](#persistence-across-restarts).

## Step 7 — consolidate periodically

Dreaming turns accumulated observations into higher-order REFLECTIONs. In a
long-running agent, run it on a cadence — e.g. every N turns — rather than every
turn:

```python
from engrava import DreamingExtension, DreamingConfig

dreaming = DreamingExtension(config=DreamingConfig(enabled=True))

# inside the loop, after advancing the cycle:
if cycle % 20 == 0:
    result = await dreaming.run_consolidation(store, current_cycle=cycle)
```

The cadence is yours to choose: every-N-turns (as above), a background asyncio
task on a timer, or an out-of-process job. Engrava is single-writer, so run
consolidation on the same writer that handles turns (or coordinate so they don't
write concurrently). A brand-new store has little to consolidate — REFLECTIONs
emerge as memories accumulate and repeat. See [Dreaming](../dreaming.md) for the
knobs.

## Persistence across restarts

- **Embeddings persist.** They are stored in the database; you do **not**
  re-embed on a normal restart. Re-embedding is a deliberate migration when
  you change the embedding model; the CLI supports it while restoring a
  snapshot into a configured service (see [CLI restore](../cli.md#restore)).
- **The cycle counter does not persist** — Engrava doesn't store it. Recover it
  on startup so it keeps increasing. `list_thoughts` returns rows ordered by
  `updated_cycle` descending, so the most recent thought carries the highest
  cycle you've used; resume one past it:

  ```python
  recent = await store.list_thoughts(limit=1)   # ordered by updated_cycle desc
  cycle = (recent[0].updated_cycle + 1) if recent else 0
  ```

- **Model lock.** If you configured an embedding provider, the store remembers
  which model produced its vectors; calling `store_embedding` later with a
  different model raises `EmbeddingModelMismatchError`. Keep the same provider
  across restarts (or migrate deliberately).

## Full example

The complete, runnable loop — including the deterministic embedder and the
mock LLM so it runs with zero external dependencies — is in
[`examples/agent_loop.py`](https://github.com/sovantica/engrava/blob/main/examples/agent_loop.py):

```bash
python examples/agent_loop.py
```

## Next

- [Core Concepts](../concepts.md) — thought / edge / reflection / cycle.
- [Hybrid Search](../search.md) — how the retrieval ranking works.
- [Dreaming](../dreaming.md) — consolidation in depth.
- [Configuration](../configuration.md) — wiring an embedding provider via `engrava.yaml`.
