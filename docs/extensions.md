# Extensions

engrava provides a hook-based extension system that lets you plug into
the thought lifecycle without modifying core code.

## EngravaHooksProtocol

All extensions implement the `EngravaHooksProtocol`:

```python
from engrava import (
    EngravaHooksProtocol,
    ThoughtRecord,
    ScoringContext,
    MindQLExtension,
)

class MyHooks(EngravaHooksProtocol):
    async def on_store(self, thought: ThoughtRecord) -> ThoughtRecord:
        """Called before a thought is persisted. Return modified thought."""
        return thought

    async def on_retrieve(self, thought: ThoughtRecord) -> ThoughtRecord:
        """Called after a thought is loaded from DB. Return modified thought."""
        return thought

    async def score_function(
        self, thought: ThoughtRecord, context: ScoringContext
    ) -> float:
        """Custom relevance scoring for search results."""
        return thought.confidence or 0.5

    async def decay_function(
        self, thought: ThoughtRecord, elapsed_cycles: int
    ) -> float:
        """Decay multiplier for edge weights over time."""
        return 1.0

    def mindql_extension_registry(self) -> dict[str, MindQLExtension]:
        """Register custom MindQL commands."""
        return {}
```

## Using Hooks

Pass hooks when creating a store:

```python
from engrava import SqliteEngravaCore

hooks = MyHooks()
store = SqliteEngravaCore("my.db", hooks=hooks)
await store.ensure_schema()

# Hooks are now called automatically during CRUD operations
```

## Default Hooks

If no hooks are provided, `DefaultEngravaHooks` is used — all methods
are no-ops that pass through data unchanged.

## Custom MindQL Commands

Register custom commands via `mindql_extension_registry()`:

```python
from engrava import MindQLExtension

class StatsHooks(EngravaHooksProtocol):
    # ... other hook methods ...

    def mindql_extension_registry(self) -> dict[str, MindQLExtension]:
        return {
            "STATS": MindQLExtension(
                command_name="STATS",
                handler=self._handle_stats,
                help_text="Show thought statistics",
            ),
        }

    async def _handle_stats(
        self, store: EngravaCoreProtocol, args: str
    ) -> list[dict[str, object]]:
        thoughts = await store.list_thoughts(limit=1000)
        by_type: dict[str, int] = {}
        for t in thoughts:
            by_type[t.thought_type] = by_type.get(t.thought_type, 0) + 1
        return [by_type]
```

Then use it via MindQL:

```python
from engrava import MindQLExecutor

executor = MindQLExecutor(store)
result = await executor.execute("STATS")
```

## Dreaming Extension

The built-in `DreamingExtension` performs periodic memory consolidation:

```python
from engrava import DreamingExtension, DreamingConfig, DreamingGates

config = DreamingConfig(
    enabled=True,
    candidates_limit=100,
    gates=DreamingGates(
        min_confidence=0.6,
        min_confirmation_count=2,
        min_composite_score=0.5,
    ),
)

dreaming = DreamingExtension(config=config)
result = await dreaming.consolidate(store, current_cycle=42)
print(f"Promoted {result.promoted_count} thoughts")
```

### Custom Signals

Implement `DreamingSignalProtocol` to add custom scoring signals:

```python
from engrava import DreamingSignalProtocol, DreamingContext, ThoughtRecord

class ImportanceSignal(DreamingSignalProtocol):
    name: str = "importance"
    weight: float = 0.3

    def score(self, thought: ThoughtRecord, context: DreamingContext) -> float:
        # Custom importance scoring logic
        if thought.priority == "P1":
            return 1.0
        if thought.priority == "P2":
            return 0.7
        return 0.3
```

## Extension Manifest

For distributing extensions as packages, use `ExtensionManifest`:

```python
from pathlib import Path
from engrava import ExtensionManifest

manifest = ExtensionManifest(
    name="my-engrava-plugin",
    version="1.0.0",
    hooks_class=MyHooks,
    mindql_extensions=[],
    schema_migrations=[
        Path("migrations/001_initial.sql"),
        Path("migrations/002_add_tags.sql"),
    ],
)
```

### Migration files

Place SQL migration scripts alongside your extension package using the
convention `NNN_slug.sql` (e.g. `001_initial.sql`, `002_add_tags.sql`).
The runner sorts files lexicographically and applies them in order:

```
my_extension/
├── __init__.py
├── hooks.py
├── manifest.py          # exports MANIFEST
└── migrations/
    ├── 001_initial.sql
    └── 002_add_tags.sql
```

Each `.sql` file should contain valid SQLite DDL or DML.  Use
`CREATE TABLE IF NOT EXISTS` to keep migrations idempotent.

### Migration file resolution

Relative paths in `schema_migrations` are resolved in this order:

1. **Absolute path** — used as-is (CI / developer override).
2. **`manifest.package_root` is set** — joined with `package_root`
   (useful for test fixtures or non-installable manifests).
3. **Default** — resolved via `importlib.resources.files` against the
   top-level package that contains `hooks_class`.  Works correctly for
   installed wheels, editable installs, and zipapps.

```python
from pathlib import Path
from engrava import ExtensionManifest

# Default (importlib.resources — recommended for distributed packages)
manifest = ExtensionManifest(
    name="my-plugin",
    version="1.0.0",
    hooks_class=MyHooks,
    schema_migrations=[Path("migrations/001_initial.sql")],
)

# Absolute path (CI / local dev)
manifest = ExtensionManifest(
    name="my-plugin",
    version="1.0.0",
    hooks_class=MyHooks,
    schema_migrations=[Path("/abs/path/to/001_initial.sql")],
)

# package_root override (test fixtures)
manifest = ExtensionManifest(
    name="my-plugin",
    version="1.0.0",
    hooks_class=MyHooks,
    schema_migrations=[Path("migrations/001_initial.sql")],
    package_root=Path(__file__).parent,
)
```

### Loading extensions with migrations

Pass manifests explicitly to `SqliteEngravaCore`.  Schema migrations are
applied automatically during `ensure_schema()`:

```python
import aiosqlite
from engrava import SqliteEngravaCore

async with aiosqlite.connect("my.db") as db:
    store = SqliteEngravaCore(db, manifests=[manifest])
    await store.ensure_schema()
    # migrations/001_initial.sql and 002_add_tags.sql are now applied
```

Or use the opt-in discovery helper to load all installed extensions:

```python
from engrava import SqliteEngravaCore
from engrava.extensions.discovery import discover_manifests

store = SqliteEngravaCore(db, manifests=discover_manifests())
await store.ensure_schema()
```

> **Note:** Discovery is **never** automatic — always opt in explicitly.
> Schema migrations have side-effects (ALTER TABLE, CREATE TABLE) and
> should only run when the caller is aware of them.

### YAML configuration

Manifests can also be declared in `engrava.yaml`:

```yaml
# Explicit dotted paths
manifests:
  - "my_plugin.manifest:MANIFEST"

# Auto-discover via entry points
manifests:
  discover: true

# Both
manifests:
  discover: true
  paths:
    - "my_plugin.manifest:MANIFEST"
```

### Version tracking

The runner tracks per-extension migration state in the
`extension_schema_versions` table (added in core schema v9).  Each row
records the extension name, the count of applied migrations, the timestamp,
the last applied filename, and the extension version at apply time.

Runner behavior at startup:

| State | Action |
|---|---|
| No row (fresh install) | Apply all migration files |
| Row with `version < len(files)` | Apply only pending files |
| Row with `version == len(files)` | No-op |
| Row with `version > len(files)` | Raise `ExtensionMigrationError` (downgrade detected) |

On SQL failure the version counter is **not** advanced.
`ExtensionMigrationError` is raised with the extension name and failing
filename so the caller can surface a clear error message.

## Subclassing SqliteEngravaCore

For deeper customization, subclass `SqliteEngravaCore` and override
the template methods:

```python
from engrava import SqliteEngravaCore, ThoughtRecord

class ExtendedStore(SqliteEngravaCore):
    def _row_to_thought(self, row: dict) -> ThoughtRecord:
        """Override to produce a richer model type."""
        # Add custom field mapping here
        return super()._row_to_thought(row)
```

This is the recommended pattern for adding domain-specific fields to
the thought model without forking the core.
