# Extension hooks

Engrava exposes a **functional hook interface** that lets external code observe
and transform data flowing through the core pipeline:

| Interface | File | Semantic |
|-----------|------|----------|
| `EngravaHooksProtocol` | `domain/protocols/hooks.py` | **Transformation** — methods return modified data (custom scoring, MindQL extensions) |

For higher-level extension patterns (embedding providers, custom MindQL commands)
see [extensions.md](extensions.md).

---

## 1. EngravaHooksProtocol

### 1.1 Contract

- All four data-flow methods are `async` and return a value.
- Hooks **must not raise** — unexpected exceptions will propagate to the caller.
- Hooks **must not have side effects** that modify shared state; return an
  enriched copy instead.
- Engrava is `frozen=True`-first — if you need to mutate a `ThoughtRecord`,
  return `thought.model_copy(update={...})`.

### 1.2 Available hooks

| Method | When | Returns |
|--------|------|---------|
| `on_store` | After a thought is persisted | `ThoughtRecord` (enriched or unchanged) |
| `on_retrieve` | After a thought is loaded from storage | `ThoughtRecord` (enriched or unchanged) |
| `score_function` | During hybrid search, to compute a custom relevance score | `float` |
| `decay_function` | During dreaming consolidation, to compute per-thought decay | `float` in `[0.0, 1.0]` |
| `mindql_extension_registry` | At `SqliteEngravaCore` init, to register custom MindQL verbs | `dict[str, MindQLExtension]` |

---

## 2. Write your own hook in 20 lines

```python
from __future__ import annotations

from engrava.domain.protocols.hooks import DefaultEngravaHooks, ScoringContext
from engrava.domain.models.thought import ThoughtRecord


class RecencyBoostHooks(DefaultEngravaHooks):
    """Boosts score for recently updated thoughts."""

    async def score_function(
        self,
        thought: ThoughtRecord,
        context: ScoringContext,
    ) -> float:
        # Add a small recency bonus if the thought was updated recently.
        base = 0.0
        if thought.updated_at and context.current_cycle > 0:
            age = context.current_cycle - thought.updated_at
            base = max(0.0, 1.0 - age / 100)
        return base


# Registration:
import aiosqlite
from engrava import SqliteEngravaCore

async def build_store(db_path: str) -> SqliteEngravaCore:
    conn = await aiosqlite.connect(db_path)
    conn.row_factory = aiosqlite.Row
    store = SqliteEngravaCore(conn, hooks=RecencyBoostHooks())
    await store.ensure_schema()
    return store
```

`DefaultEngravaHooks` is a no-op base class — override only the methods you
care about.

---

## 3. Custom MindQL verb

Use `mindql_extension_registry` to register a new MindQL command:

```python
from __future__ import annotations

from engrava.domain.protocols.hooks import DefaultEngravaHooks, MindQLExtension


async def _pinned_handler(**kwargs: object) -> list[dict[str, object]]:
    """Return all thoughts tagged 'pinned'."""
    store = kwargs["store"]
    results = await store.search("", filters={"tags": ["pinned"]}, top_k=100)
    return [{"id": str(t.id), "content": t.content} for t in results]


class PinnedMindQLHooks(DefaultEngravaHooks):
    def mindql_extension_registry(self) -> dict[str, MindQLExtension]:
        return {
            "PINNED": MindQLExtension(
                command_name="PINNED",
                handler=_pinned_handler,
                description="Return all pinned thoughts.",
                category="custom",
            ),
        }
```

Usage: `FIND PINNED` — treated as a first-class MindQL verb after registration.

---

## 4. Implementing a contract test

Add a contract test to verify your implementation satisfies the protocol:

```python
from engrava.domain.protocols.hooks import EngravaHooksProtocol


def test_my_hooks_satisfy_protocol() -> None:
    assert isinstance(RecencyBoostHooks(), EngravaHooksProtocol)
```

`EngravaHooksProtocol` is `@runtime_checkable`, so `isinstance` works without
meta-class magic.
