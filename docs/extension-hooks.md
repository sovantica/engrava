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

Only `on_store` and `on_retrieve` are invoked by the public engrava package
today. The remaining three are part of the protocol contract but are
**reserved** — core engrava does not call them (they exist for downstream
consumers and future use). Implement them if you want protocol conformance,
but do not expect core to invoke them.

| Method | When | Returns | Status in core |
|--------|------|---------|----------------|
| `on_store` | After a thought is persisted | `ThoughtRecord` (enriched or unchanged) | **active** |
| `on_retrieve` | After a thought is loaded from storage | `ThoughtRecord` (enriched or unchanged) | **active** |
| `score_function(thought, context)` | Custom relevance score | `float` | reserved — not called by core |
| `decay_function(thought, elapsed_cycles)` | Per-thought decay factor | `float` in `[0.0, 1.0]` | reserved — not called by core |
| `mindql_extension_registry()` | Register custom MindQL verbs | `dict[str, MindQLExtension]` | reserved — core wires MindQL verbs via `ExtensionManifest.mindql_extensions`, not this hook |

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
        # Add a small recency bonus for thoughts updated in a recent cycle.
        # (updated_cycle is an int; updated_at is an ISO-8601 string, not a cycle.)
        if context.current_cycle > 0:
            age = context.current_cycle - thought.updated_cycle
            return max(0.0, 1.0 - age / 100)
        return 0.0


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

A custom MindQL command is an `MindQLExtension` whose `handler` is an async
callable. The executor invokes the handler with **two positional arguments** —
the open `aiosqlite.Connection` and the parsed extension-argument list — and
expects a `list[dict[str, object]]` back:

```python
from __future__ import annotations

import aiosqlite

from engrava.domain.protocols.hooks import MindQLExtension


async def _recent_handler(
    db: aiosqlite.Connection,
    args: list[str],  # noqa: ARG001 — this command takes no args
) -> list[dict[str, object]]:
    """Return the 100 most recently updated thoughts."""
    cursor = await db.execute(
        "SELECT thought_id, content FROM thought "
        "ORDER BY updated_cycle DESC LIMIT 100"
    )
    rows = await cursor.fetchall()
    return [{"thought_id": row["thought_id"], "content": row["content"]} for row in rows]


RECENT_COMMAND = MindQLExtension(
    command_name="RECENT",
    handler=_recent_handler,
    description="Return the most recently updated thoughts.",
    category="custom",
)
```

The command is registered by listing it in an extension's
`ExtensionManifest.mindql_extensions` (the discovery path), or by passing it
through `MindQLExtension`-keyed `extensions=` when constructing the executor:

```python
from engrava import MindQLExecutor, parse

executor = MindQLExecutor(conn, extensions={"RECENT": RECENT_COMMAND})
# parse() needs the registered verb names to recognise an extension command.
result = await executor.execute(parse("RECENT", known_extensions={"RECENT"}))
```

> Note: `EngravaHooksProtocol.mindql_extension_registry()` is **not** consulted
> by core engrava (see §1.2) — declare custom verbs via `ExtensionManifest`
> or the executor's `extensions=` argument as shown above.

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
