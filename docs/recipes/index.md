# Recipes

Short, copy-paste snippets for the things you actually do with an agent-memory
database. Each assumes you already have an open `store` (see the
[Quick Start](../quickstart.md)); imports are shown once per recipe.

> New to the model? Read [Core Concepts](../concepts.md) first. For the full
> agent turn loop, see [Building a memory-backed agent](../guides/agent-memory.md).

## Store a conversation turn

Persist a user message and the agent's reply, tagged with conversation metadata
so you can scope retrieval later:

```python
import uuid
from engrava import ThoughtRecord, ThoughtType, Priority, LifecycleStatus, percept, utterance

async def store_turn(store, user_text, agent_text, *, cycle, session_id, turn_index, user_id):
    user_thought = ThoughtRecord(
        thought_id=str(uuid.uuid4()),
        thought_type=ThoughtType.OBSERVATION,
        essence=user_text[:200], content=user_text,
        priority=Priority.P2, lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=cycle, updated_cycle=cycle, source=user_id,
        metadata={**percept(source_id=user_id, label="user"),
                  "session_id": session_id, "turn_index": turn_index},
    )
    await store.create_thought(user_thought)

    agent_thought = ThoughtRecord(
        thought_id=str(uuid.uuid4()),
        thought_type=ThoughtType.OUTPUT_DRAFT,
        essence=agent_text[:200], content=agent_text,
        priority=Priority.P3, lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=cycle, updated_cycle=cycle, source="agent",
        metadata={**utterance(), "session_id": session_id, "turn_index": turn_index},
    )
    await store.create_thought(agent_thought)
```

## Retrieve context for a prompt

Get the most relevant prior memories and turn them into prompt-ready text. With
an embedding provider configured, `search_hybrid` embeds the query for you:

```python
async def context_for(store, query, cycle, top_k=5):
    result = await store.search_hybrid(query, top_k=top_k, current_cycle=cycle)
    lines = []
    for thought_id, _score in result.results:
        record = await store.get_thought(thought_id)
        if record is not None:
            lines.append(record.essence)        # essence = the prompt-facing one-liner
    return "\n".join(f"- {line}" for line in lines)
```

## Filter retrieval by session (or user)

Use the ranked methods' native typed metadata filters. They constrain candidate
eligibility inside each ranking arm before its candidate limit, so unrelated
rows cannot consume the pool and starve the requested `top_k`:

```python
from engrava import FieldOp, FieldPredicate, MetadataFilter

async def search_in_session(store, query, session_id, cycle, want=5):
    result = await store.search_hybrid(
        query,
        top_k=want,
        current_cycle=cycle,
        filters=MetadataFilter(
            [FieldPredicate("$.session_id", FieldOp.EQ, session_id)]
        ),
    )
    scoped = []
    for thought_id, _score in result.results:
        record = await store.get_thought(thought_id)
        if record is not None:
            scoped.append(record)
    return scoped
```

> For *hard* isolation between users/tenants (separate databases rather than a
> shared one with a metadata tag), use [`EngravaManager`](../api-reference.md) —
> one `<name>.db` per service. That trades cross-tenant search for strong
> isolation; the metadata approach keeps one searchable store. `filters=` is a
> query refinement, not access control: a caller can omit or forge it.

## Set a TTL on transient memories

Give a thought an expiry, then expire due thoughts. The default strategy is
`archive` (soft — marks `ARCHIVED`); switch to `delete` for hard removal:

```python
# expire this thought one hour from now
await store.create_thought(transient_thought, expires_after_seconds=3600)

# later: process everything past its expiry (archive or delete per ttl_strategy)
result = await store.cleanup_expired()
print(f"{result.expired_count} thoughts expired via '{result.strategy_applied}'")
```

A store-wide default TTL and the archive-vs-delete strategy are set in config —
see the [`ttl` configuration](../configuration.md). Archived thoughts leave disk
only on a later `engrava gc`.

## Deduplicate repeated facts

Pass `deduplicate=True` so identical `content` collapses into one thought with a
bumped `confirmation_count` instead of a duplicate row:

```python
first = await store.create_thought(fact, deduplicate=True)
again = await store.create_thought(same_fact, deduplicate=True)
# again.thought_id == first.thought_id; confirmation_count incremented, no new row
```

The growing `confirmation_count` is also a reliability signal dreaming uses (a
fact re-confirmed many times ranks as more trustworthy) — see
[Core Concepts](../concepts.md#reliability-confidence-vs-confirmation_count).

## Run consolidation on a schedule

In a long-running agent, run dreaming every N turns rather than every turn:

```python
from engrava import DreamingExtension, DreamingConfig

dreaming = DreamingExtension(config=DreamingConfig(enabled=True))

# inside your turn loop, after advancing the cycle counter:
if cycle % 20 == 0:
    result = await dreaming.run_consolidation(store, current_cycle=cycle)
    print(f"consolidation: promoted {result.promoted_count}")
```

A fresh store has little to consolidate — REFLECTIONs emerge as memories
accumulate and repeat. See [Dreaming](../dreaming.md) for the cadence and knobs.

## Inspect what changed (audit trail)

With the [audit journal](../audit-trail.md) enabled, read the history of any
thought:

```python
history = await store.journal.get_entries(target_id=some_thought_id)
for entry in history:
    print(entry.sequence_number, entry.mutation_type, entry.created_at)
```

## Record a tool result / action

If your agent *does* things (calls a tool, sends a message), record each as an
`ActionRecord` linked to the thought that prompted it, so what the agent did —
and whether it worked — is part of memory:

```python
import uuid
from engrava import ActionRecord, ActionType, ActionStatus, VerificationStatus

action = await store.create_action(
    ActionRecord(
        action_id=str(uuid.uuid4()),
        source_thought_id=prompting_thought_id,
        action_type=ActionType.TOOL_CALL,     # or MESSAGE / CLI_OUTPUT / STATE_UPDATE
        intent="search the web for flight prices",
        status=ActionStatus.PLANNED,
        verification_status=VerificationStatus.PENDING,
    )
)

# Execution follows PLANNED -> EXECUTING -> CONFIRMED or FAILED.
await store.update_action(action.action_id, status=ActionStatus.EXECUTING)
await store.update_action(
    action.action_id,
    status=ActionStatus.CONFIRMED,
    verification_status=VerificationStatus.CONFIRMED,
)

# BLOCKED is a planning detour only: PLANNED -> BLOCKED -> PLANNED.
# It is not reachable from EXECUTING and does not lead directly to a terminal state.

# read an entity's actions back:
actions = await store.get_actions(prompting_thought_id)
```

## Restore the cycle counter after a restart

The cycle is the agent's logical clock and Engrava does **not** persist it — on
startup, seed it from the highest cycle already stored so it keeps increasing.
`list_thoughts` returns rows ordered by `updated_cycle` descending, so the most
recent thought carries the highest value:

```python
recent = await store.list_thoughts(limit=1)        # ordered by updated_cycle desc
cycle = (recent[0].updated_cycle + 1) if recent else 0
```

See [Cycle (the agent clock)](../concepts.md#cycle-the-agent-clock) for why this
matters (a frozen clock disables recency and stalls dreaming).

## Next

- [Building a memory-backed agent](../guides/agent-memory.md) — these recipes assembled into a loop.
- [Tutorial](../tutorial.md) — build a small notes memory from scratch.
- [Core Concepts](../concepts.md) — the model behind the snippets.
- [Hybrid Search](../search.md) · [Dreaming](../dreaming.md) · [Configuration](../configuration.md).
