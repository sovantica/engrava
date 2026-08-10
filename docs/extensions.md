# Extensions

engrava provides lifecycle hooks, derived-record producers, embedding providers,
MindQL commands, and package manifests for extending the store without modifying
core code. This page covers the package-level patterns; the complete hook and
derived-record contracts are in [Extension hooks](extension-hooks.md).

## EngravaHooksProtocol

Lifecycle-hook implementations satisfy `EngravaHooksProtocol`:

```python
from engrava import (
    EngravaHooksProtocol,
    ThoughtRecord,
    ScoringContext,
    MindQLExtension,
)

class MyHooks(EngravaHooksProtocol):
    async def on_store(self, thought: ThoughtRecord) -> ThoughtRecord:
        """Called after a thought is persisted. Return the (enriched) thought."""
        return thought

    async def on_retrieve(self, thought: ThoughtRecord) -> ThoughtRecord:
        """Called after a thought is loaded from DB. Return the (enriched) thought."""
        return thought

    async def score_function(
        self, thought: ThoughtRecord, context: ScoringContext
    ) -> float:
        """Custom relevance score (reserved — not currently called by core)."""
        return thought.confidence or 0.5

    async def decay_function(
        self, thought: ThoughtRecord, elapsed_cycles: int
    ) -> float:
        """Decay multiplier used by an enabled Memory Hygiene pass."""
        return 1.0

    def mindql_extension_registry(self) -> dict[str, MindQLExtension]:
        """Reserved — core wires MindQL verbs via ExtensionManifest, not this hook."""
        return {}
```

> Core invokes `on_store`, `on_retrieve`, and `decay_function`.
> `decay_function` is called once per candidate when an enabled `run_hygiene()`
> pass reaches archive scoring; its return is clamped to `[0.0, 1.0]`, and a
> non-finite value fails safe to `1.0`. It is not a search or promotion hook. Only
> `score_function` and `mindql_extension_registry()` are reserved and not called
> by core; MindQL verbs are wired through `ExtensionManifest`. See
> [Available extension hooks](extension-hooks.md). Subclass
> `DefaultEngravaHooks` if you only want to override selected methods.

## Using Hooks

Pass hooks when creating a store (the store wraps an open connection):

```python
import aiosqlite
from engrava import SqliteEngravaCore

hooks = MyHooks()
async with aiosqlite.connect("my.db") as conn:
    conn.row_factory = aiosqlite.Row
    store = SqliteEngravaCore(conn, hooks=hooks)
    await store.ensure_schema()
    # on_store / on_retrieve now run during CRUD; decay_function runs only when
    # this store executes an enabled Memory Hygiene pass.
```

## Default Hooks

If no hooks are provided, `DefaultEngravaHooks` is used — all methods
are no-ops that pass through data unchanged.

## Custom MindQL Commands

A custom command is an `MindQLExtension`. Its `handler` is an async callable
that the executor invokes with two positional arguments — the open
`aiosqlite.Connection` and the parsed argument list — and returns a
`list[dict[str, object]]`. The `MindQLExtension` fields are `command_name`,
`handler`, `description`, and `category` (there is no `help_text` field):

```python
import aiosqlite
from engrava import MindQLExtension


async def _handle_stats(
    db: aiosqlite.Connection,
    args: list[str],  # noqa: ARG001 — STATS takes no args
) -> list[dict[str, object]]:
    cursor = await db.execute(
        "SELECT thought_type, COUNT(*) AS n FROM thought GROUP BY thought_type"
    )
    rows = await cursor.fetchall()
    return [{row["thought_type"]: row["n"]} for row in rows]


STATS_COMMAND = MindQLExtension(
    command_name="STATS",
    handler=_handle_stats,
    description="Show thought statistics",
)
```

Then run it through the executor, passing the command in `extensions=` and
telling `parse()` which verbs are registered:

```python
from engrava import MindQLExecutor, parse

executor = MindQLExecutor(conn, extensions={"STATS": STATS_COMMAND})
result = await executor.execute(parse("STATS", known_extensions={"STATS"}))
```

## Dreaming Extension

The built-in `DreamingExtension` performs periodic memory consolidation:

```python
from engrava import DreamingExtension, DreamingConfig, DreamingGates

config = DreamingConfig(
    enabled=True,
    candidates_limit=100,
    promote_threshold=0.6,
    gates=DreamingGates(
        min_confirmations=2,
        min_age_cycles=1,
        max_promoted_per_run=20,
    ),
)

dreaming = DreamingExtension(config=config)
result = await dreaming.run_consolidation(store, current_cycle=42)
print(f"Promoted {result.promoted_count} thoughts")
```

The weighted-score cutoff is `DreamingConfig.promote_threshold`;
`DreamingGates` controls eligibility (confirmations, age, per-run cap, and the
clustering/quality thresholds). See [Dreaming](dreaming.md) for the full
configuration surface.

### Custom Signals

`DreamingSignalProtocol` is a callable protocol — implement `__call__(thought,
ctx)` returning a score in `[0.0, 1.0]`. There is no `name`/`weight` attribute
or `score()` method; a signal's weight is set separately in
`DreamingConfig.signals`, and the instance is wired in via
`DreamingExtension(config, custom_signals={...})`.

```python
from engrava import DreamingContext, DreamingExtension, DreamingConfig, ThoughtRecord
from engrava import Priority


class ImportanceSignal:
    """Custom scoring signal — must be callable as (thought, ctx) -> float."""

    def __call__(self, thought: ThoughtRecord, ctx: DreamingContext) -> float:
        if thought.priority == Priority.P1:
            return 1.0
        if thought.priority == Priority.P2:
            return 0.7
        return 0.3


# Register the signal AND give it a weight in the signals map, or it never runs.
dreaming = DreamingExtension(
    config=DreamingConfig(enabled=True, signals={"importance": 0.3}),
    custom_signals={"importance": ImportanceSignal()},
)
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

Or use the opt-in discovery helper to load all installed extensions into a
library-created store:

```python
from engrava import SqliteEngravaCore
from engrava.extensions.discovery import discover_manifests

store = SqliteEngravaCore(db, manifests=discover_manifests())
await store.ensure_schema()
```

> **Library boundary:** `SqliteEngravaCore` does not scan entry points by
> itself. Library callers opt in by calling `discover_manifests()` or by setting
> `manifests.discover: true` in a configuration loaded by `from_config()`.
> Discovered manifests are then passed to the store, so their schema migrations
> run during `ensure_schema()`.

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

### CLI discovery and its disable control

The `engrava` CLI has two additional discovery paths that are independent of
the library opt-in above:

- root CLI help and resolution of an otherwise unknown command scan the
  `engrava.cli` entry-point group and load values that provide Click commands;
  built-in commands resolve without that scan;
- the built-in `engrava query` command also scans `engrava.extensions`, loads
  manifests, and registers their `mindql_extensions` for that query.

These CLI scans do not pass discovered manifests to a store and therefore do
not apply their schema migrations. They do import and execute installed Python
entry-point code. The global `--no-extensions` option prevents both scans,
including during root help, built-in commands, and `query`. Set
`ENGRAVA_DISABLE_EXTENSIONS=1` for the equivalent process-level control. This
does not disable explicit manifest paths or library-side discovery configured on
an application-created store. Run the CLI in an environment containing only
trusted packages; use the Python API with explicit manifests when you need an
allow-listed extension set. See [Security and Trust Boundaries](security.md).

### Version tracking

The runner uses two bookkeeping tables:

- `extension_schema_migrations` is append-only history keyed by extension name
  and one-based migration index. Each row records the filename, a SHA-256
  content checksum, timestamp, and extension version at apply time.
- `extension_schema_versions` is a one-row summary containing the latest
  applied count, filename, timestamp, and extension version.

Migration identity is the unique basename in lexicographic order. After a file
has been applied, its position, basename, and SQL content are immutable: add a
new later migration instead of editing, renaming, reordering, or inserting a
file before history. Duplicate basenames are rejected even when their paths
differ.

Runner behavior at startup:

| State | Action |
|---|---|
| No history or summary (fresh install) | Prepare and apply all migration files |
| History is a valid prefix of the current files | Re-verify every applied filename/checksum, then apply only the pending suffix |
| History covers every current file | No-op after drift verification |
| Recorded count exceeds the current file count | Raise `ExtensionMigrationError` (downgrade detected) |
| History is non-contiguous or disagrees with the summary | Raise `ExtensionMigrationError` (bookkeeping corruption) |
| Applied filename, order, or checksum changed | Raise `ExtensionMigrationError` before pending SQL runs |

Databases created by older Engrava versions may have only the summary row. On
the first run with checksum history, the runner adopts the first recorded
number of current files as the baseline and stores their current checksums.
Because no older checksum exists, this cannot detect edits made before adoption;
drift detection for those files is prospective from that point onward.

Every pending file is read, split into complete semicolon-terminated SQLite
statements, checksummed, and checked for forbidden transaction-control commands
before the first pending migration runs. The runner owns transaction state;
`BEGIN`, `COMMIT`, `END`, `ROLLBACK`, `SAVEPOINT`, and `RELEASE` statements are
rejected.

Each migration file runs in its own savepoint together with its history append
and summary update. A failure rolls back that file's SQL and bookkeeping, then
raises `ExtensionMigrationError` with the extension and filename. Successfully
completed earlier files have already been committed and remain applied; the
pending set is not one all-or-nothing transaction. Re-running resumes from the
first unapplied file after the problem is corrected by adding/fixing a migration
that has not yet been recorded.

Use a WAL-safe backup before applying migrations. Checksums protect migration
history from accidental drift; they do not authenticate an extension publisher
or make untrusted SQL safe.

## Subclassing SqliteEngravaCore

For deeper customization, subclass `SqliteEngravaCore` and override
the template methods:

```python
import aiosqlite

from engrava import SqliteEngravaCore, ThoughtRecord

class ExtendedStore(SqliteEngravaCore):
    def _row_to_thought(self, row: aiosqlite.Row) -> ThoughtRecord:
        """Override to produce a richer model type."""
        # Add custom field mapping here
        return super()._row_to_thought(row)
```

This is the recommended pattern for adding domain-specific fields to
the thought model without forking the core.
