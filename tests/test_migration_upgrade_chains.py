"""Whole-ladder schema migration hardening for ``ensure_schema``.

``ensure_schema`` migrates a database up to the head ``user_version`` (20)
through an ordered migration registry plus a loop over a chain of per-version
``_migrate_core_v*`` helpers. Existing suites cover single rungs (the
v13->v14 hot-path indexes and the v18->v19 edge metadata column); this module
generalises that to the **entire** ladder so an off-by-one step (a wrong start
version or a skipped step) can no longer strand one specific source version
undetected.

Four properties are asserted:

* **Exhaustive convergence.** For every seed version ``v`` in
  ``{fresh, 2..19}`` a *real-shape* schema-at-``v`` fixture (the actual tables,
  columns, indexes and FTS tokenizer that version shipped — reconstructed from
  the migration helpers and cross-checked against the historical
  ``schema_core.sql``) is stamped, pre-migration rows are written, and
  ``ensure_schema`` must land at v20, be idempotent on a second run, and leave
  every pre-existing thought / edge / embedding queryable through the public
  read surface (get / list / FTS / hybrid). A post-migration API write must
  also succeed — the exact failure mode ("a missing column makes the first
  write raise") the ladder hardening guards against.

* **Fresh-vs-migrated parity.** A database bootstrapped fresh at v20 and one
  migrated up from an old version must be structurally equivalent — same
  columns (name, type, nullability, default, PK), same foreign keys, same
  indexes, same triggers, same FTS config. A discriminating revert (a
  migration that adds a column with a different default) must break parity, so
  the check is not a tautology.

* **Real pip-install upgrade matrix.** Wired into CI via a dedicated job (see
  ``.github/workflows/ci.yml``); the opt-in env gate on
  ``tests/upgrade/test_upgrade_matrix.py`` is intentionally kept so local runs
  stay offline.

* **Static structure of the bootstrap script.** ``schema_core.sql`` must stamp
  ``PRAGMA user_version`` as its final statement, exactly once. Runtime
  failure-injection can only probe the offsets it injects at, so the invariant
  is also asserted statically against the script text, where it cannot drift
  with the script.

Reconstruction fidelity note: because the parity test asserts a migrated
database equals a freshly bootstrapped one, a fixture that invented a column /
table a version never had, or omitted one it did, would surface as a parity
divergence — so the fixtures are self-checking against the real schema.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from importlib import resources
from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import (
    EdgeRecord,
    EdgeType,
    LifecycleStatus,
    Priority,
    SqliteEngravaCore,
    ThoughtRecord,
    ThoughtType,
)
from engrava.domain.exceptions import CoreMigrationError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_HEAD_VERSION = 20

# The user (non-shadow) tables that exist at head v20, compared column-by-column
# for fresh-vs-migrated parity. This list mirrors a source of truth (the tables a
# head database actually carries), so ``_capture_schema_shape`` derives that set
# from ``PRAGMA table_list`` on every capture and refuses to compare against a
# stale list — otherwise a table outside the tuple is invisible to the parity
# check and its columns, defaults and foreign keys are never read at all.
_CORE_TABLES = (
    "thought",
    "edge",
    "embedding",
    "action",
    "_metadata",
    "journal_entry",
    "extension_schema_versions",
)

# The FTS5 virtual table at head. Its structure is fingerprinted through
# ``_SchemaShape.fts`` rather than column-by-column, but it is still named here
# so the derived table set covers it: a virtual table added to one bootstrap
# path and not the other is as much of a divergence as a missing plain table.
_FTS_VIRTUAL_TABLE = "thought_fts"

# Distinctive FTS terms seeded into the legacy thought so a post-migration
# search proves the index survived (and was rebuilt where the ladder rebuilds
# it, at v2->v3).
_LEGACY_ESSENCE = "migration probe beacon essence"
_LEGACY_CONTENT = "zephyr alpha vector content"
_LEGACY_TERM = "beacon"


# ---------------------------------------------------------------------------
# Real-shape schema-at-version builder
# ---------------------------------------------------------------------------
#
# Each column / table / index / trigger is emitted guarded by the version that
# first introduced it, reconstructing exactly what a database freshly created
# at ``target`` carried. The introduction versions come straight from the
# ``_migrate_core_v*`` helpers:
#
#   v4  thought.access_count/last_accessed_at/created_at/updated_at
#   v5  _metadata table
#   v6  journal_entry table + its three indexes
#   v7  thought.expires_at + idx_thought_expires
#   v8  idx_edge_type_from
#   v9  extension_schema_versions table
#   v10 thought.content_hash + idx_thought_content_hash
#   v11 thought.metadata_json
#   v12 FK + ON DELETE CASCADE on edge / embedding / action
#   v13 valid-time columns + indexes on thought and edge
#   v14 hot-path indexes (edge/embedding/thought)
#   v15 idx_edge_type_to
#   v16 thought.action_outcome_score + idx_action_source_thought
#   v17 thought.provenance + its two json-expression indexes
#   v18 thought.pinned + thought.archived_at_cycle
#   v19 edge.metadata_json
#   v20 thought.archived_at
#
# The base (v2/v3) tables predate the public git history; v2 differs from v3
# only in the FTS tokenizer (v2 shipped the pre-hyphen-aware config that
# ``_rebuild_fts_index`` replaces at v2->v3). The reconstruction of v12..v18
# reproduces the historical ``schema_core.sql`` at those tags column-for-column.


def _thought_columns(target: int) -> list[str]:
    """Return the ``thought`` column definitions present at ``target``.

    Column order matches the historical fresh DDL: ``content_hash`` sits at
    position five (as it has since the first public release, v12), while every
    later column is appended in introduction order.
    """
    cols = [
        "thought_id        TEXT    PRIMARY KEY",
        "thought_type      TEXT    NOT NULL",
        "essence           TEXT    NOT NULL",
        "content           TEXT    NOT NULL",
    ]
    if target >= 10:
        cols.append("content_hash      TEXT")
    cols += [
        "priority          TEXT    NOT NULL",
        "lifecycle_status  TEXT    NOT NULL DEFAULT 'CREATED'",
        "created_cycle     INTEGER NOT NULL DEFAULT 0",
        "updated_cycle     INTEGER NOT NULL DEFAULT 0",
        "source            TEXT    NOT NULL DEFAULT 'human'",
        "confidence        REAL",
        "embedding_ref     TEXT",
        "source_type       TEXT    NOT NULL DEFAULT 'EXPERIENCE'",
        "confirmation_count INTEGER NOT NULL DEFAULT 0",
        "consolidated_from TEXT",
        "visibility        TEXT    NOT NULL DEFAULT 'selective'",
    ]
    if target >= 4:
        cols += [
            "access_count      INTEGER NOT NULL DEFAULT 0",
            "last_accessed_at  TEXT",
            "created_at        TEXT",
            "updated_at        TEXT",
        ]
    if target >= 7:
        cols.append("expires_at        TEXT")
    if target >= 11:
        cols.append("metadata_json     TEXT    NOT NULL DEFAULT '{}'")
    if target >= 13:
        cols += ["valid_from        TEXT", "valid_until       TEXT"]
    if target >= 16:
        cols.append("action_outcome_score REAL")
    if target >= 17:
        cols.append("provenance        TEXT")
    if target >= 18:
        cols += [
            "pinned            INTEGER NOT NULL DEFAULT 0",
            "archived_at_cycle INTEGER",
        ]
    if target >= 20:
        cols.append("archived_at       TEXT")
    return cols


def _edge_table(target: int) -> str:
    """Return the ``edge`` ``CREATE TABLE`` for ``target``.

    Foreign keys arrive at v12, valid-time columns at v13, ``metadata_json``
    at v19 — each appended after the base columns exactly as the in-place
    ``ALTER``/recreate paths leave them.
    """
    cols = [
        "edge_id           TEXT PRIMARY KEY",
        "from_thought_id   TEXT NOT NULL",
        "to_thought_id     TEXT NOT NULL",
        "edge_type         TEXT NOT NULL",
        "weight            REAL NOT NULL DEFAULT 0.5",
        "created_cycle     INTEGER NOT NULL DEFAULT 0",
        "source            TEXT NOT NULL DEFAULT 'EXPERIENCE'",
        "decay_multiplier  REAL NOT NULL DEFAULT 1.0",
    ]
    if target >= 13:
        cols += ["valid_from        TEXT", "valid_until       TEXT"]
    if target >= 19:
        cols.append("metadata_json     TEXT NOT NULL DEFAULT '{}'")
    cols.append("UNIQUE(from_thought_id, to_thought_id, edge_type)")
    if target >= 12:
        cols += [
            "FOREIGN KEY (from_thought_id) REFERENCES thought(thought_id) ON DELETE CASCADE",
            "FOREIGN KEY (to_thought_id) REFERENCES thought(thought_id) ON DELETE CASCADE",
        ]
    return "CREATE TABLE edge (\n  " + ",\n  ".join(cols) + "\n)"


def _embedding_table(target: int) -> str:
    """Return the ``embedding`` ``CREATE TABLE`` for ``target`` (FK from v12)."""
    cols = [
        "embedding_id TEXT PRIMARY KEY",
        "owner_type   TEXT    NOT NULL",
        "owner_id     TEXT    NOT NULL",
        "model_name   TEXT    NOT NULL",
        "dimension    INTEGER NOT NULL",
        "vector_blob  BLOB    NOT NULL",
        "created_at   TEXT    NOT NULL",
    ]
    if target >= 12:
        cols.append("FOREIGN KEY (owner_id) REFERENCES thought(thought_id) ON DELETE CASCADE")
    return "CREATE TABLE embedding (\n  " + ",\n  ".join(cols) + "\n)"


def _action_table(target: int) -> str:
    """Return the ``action`` ``CREATE TABLE`` for ``target`` (FK from v12)."""
    cols = [
        "action_id           TEXT PRIMARY KEY",
        "source_thought_id   TEXT NOT NULL",
        "action_type         TEXT NOT NULL",
        "intent              TEXT NOT NULL",
        "status              TEXT NOT NULL DEFAULT 'PLANNED'",
        "verification_status TEXT NOT NULL DEFAULT 'PENDING'",
        "raw_metrics_json    TEXT",
    ]
    if target >= 12:
        cols.append(
            "FOREIGN KEY (source_thought_id) REFERENCES thought(thought_id) ON DELETE CASCADE"
        )
    return "CREATE TABLE action (\n  " + ",\n  ".join(cols) + "\n)"


def _fts_and_triggers(target: int) -> list[str]:
    """Return the FTS virtual table + sync triggers for ``target``.

    v2 shipped the plain ``unicode61`` tokenizer (``-`` treated as an
    operator); v3 onward carries the hyphen-aware ``tokenchars '-_'`` config
    that ``_rebuild_fts_index`` installs at the v2->v3 rung.
    """
    tokenizer = "tokenize = \"unicode61 tokenchars '-_'\", " if target >= 3 else ""
    return [
        "CREATE VIRTUAL TABLE thought_fts USING fts5("
        "essence, content, " + tokenizer + "content='thought', content_rowid='rowid');",
        "CREATE TRIGGER thought_fts_insert AFTER INSERT ON thought BEGIN "
        "INSERT INTO thought_fts(rowid, essence, content) "
        "VALUES (new.rowid, new.essence, new.content); END;",
        "CREATE TRIGGER thought_fts_delete AFTER DELETE ON thought BEGIN "
        "INSERT INTO thought_fts(thought_fts, rowid, essence, content) "
        "VALUES ('delete', old.rowid, old.essence, old.content); END;",
        "CREATE TRIGGER thought_fts_update AFTER UPDATE OF essence, content ON thought BEGIN "
        "INSERT INTO thought_fts(thought_fts, rowid, essence, content) "
        "VALUES ('delete', old.rowid, old.essence, old.content); "
        "INSERT INTO thought_fts(rowid, essence, content) "
        "VALUES (new.rowid, new.essence, new.content); END;",
    ]


def _indexes(target: int) -> list[str]:
    """Return the explicit named indexes present at ``target``."""
    stmts: list[str] = []
    if target >= 6:
        stmts += [
            "CREATE INDEX idx_journal_target ON journal_entry(target_id, sequence_number);",
            "CREATE INDEX idx_journal_type ON journal_entry(mutation_type);",
            "CREATE INDEX idx_journal_seq ON journal_entry(sequence_number);",
        ]
    if target >= 7:
        stmts.append(
            "CREATE INDEX idx_thought_expires ON thought(expires_at) WHERE expires_at IS NOT NULL;"
        )
    if target >= 8:
        stmts.append("CREATE INDEX idx_edge_type_from ON edge(edge_type, from_thought_id);")
    if target >= 10:
        stmts.append("CREATE INDEX idx_thought_content_hash ON thought(content_hash);")
    if target >= 13:
        stmts += [
            "CREATE INDEX idx_thought_valid_from ON thought(valid_from);",
            "CREATE INDEX idx_thought_valid_until ON thought(valid_until) "
            "WHERE valid_until IS NOT NULL;",
            "CREATE INDEX idx_thought_valid_range ON thought(valid_from, valid_until);",
            "CREATE INDEX idx_edge_valid_from ON edge(valid_from);",
            "CREATE INDEX idx_edge_valid_until ON edge(valid_until) WHERE valid_until IS NOT NULL;",
            "CREATE INDEX idx_edge_valid_range ON edge(valid_from, valid_until);",
        ]
    if target >= 14:
        stmts += [
            "CREATE INDEX idx_edge_to_thought ON edge(to_thought_id);",
            "CREATE INDEX idx_embedding_owner ON embedding(owner_id);",
            "CREATE INDEX idx_thought_updated_cycle ON thought(updated_cycle);",
            "CREATE INDEX idx_thought_type ON thought(thought_type);",
        ]
    if target >= 15:
        stmts.append("CREATE INDEX idx_edge_type_to ON edge(edge_type, to_thought_id);")
    if target >= 16:
        stmts.append("CREATE INDEX idx_action_source_thought ON action(source_thought_id);")
    if target >= 17:
        stmts += [
            "CREATE INDEX idx_thought_prov_session "
            "ON thought(json_extract(provenance, '$.session_id'));",
            "CREATE INDEX idx_thought_prov_actor "
            "ON thought(json_extract(provenance, '$.actor_id'));",
        ]
    return stmts


def _schema_sql_at_version(target: int) -> str:
    """Assemble the full real-shape schema script for ``target`` (2..19)."""
    parts = [
        "CREATE TABLE thought (\n  " + ",\n  ".join(_thought_columns(target)) + "\n);",
        _edge_table(target) + ";",
        _embedding_table(target) + ";",
        _action_table(target) + ";",
    ]
    if target >= 5:
        parts.append("CREATE TABLE _metadata (key TEXT PRIMARY KEY, value TEXT);")
    if target >= 6:
        parts.append(
            "CREATE TABLE journal_entry ("
            "entry_id TEXT PRIMARY KEY, sequence_number INTEGER NOT NULL UNIQUE, "
            "mutation_type TEXT NOT NULL, target_id TEXT, delta TEXT NOT NULL, "
            "parent_hash TEXT, entry_hash TEXT NOT NULL, created_at TEXT NOT NULL);"
        )
    if target >= 9:
        parts.append(
            "CREATE TABLE extension_schema_versions ("
            "extension_name TEXT PRIMARY KEY, version INTEGER NOT NULL DEFAULT 0, "
            "applied_at REAL NOT NULL, migration_file TEXT, extension_version TEXT);"
        )
    parts += _fts_and_triggers(target)
    parts += _indexes(target)
    parts.append(f"PRAGMA user_version = {target};")
    return "\n".join(parts)


async def _bootstrap_core_at_version(db: aiosqlite.Connection, target: int) -> None:
    """Stamp a real-shape schema-at-``target`` database (2..19)."""
    await db.executescript(_schema_sql_at_version(target))
    await db.commit()


async def _seed_legacy_rows(db: aiosqlite.Connection) -> None:
    """Insert a thought pair, an edge and an embedding using base-version columns.

    Only columns present at every seed version (>=2) are written, so the same
    insert works against any real-shape base. The edge and embedding reference
    an existing thought so the v11->v12 orphan purge never removes them.
    """
    await db.execute(
        "INSERT INTO thought (thought_id, thought_type, essence, content, priority, "
        "lifecycle_status) VALUES (?, 'OBSERVATION', ?, ?, 'P2', 'ACTIVE')",
        ("t-1", _LEGACY_ESSENCE, _LEGACY_CONTENT),
    )
    await db.execute(
        "INSERT INTO thought (thought_id, thought_type, essence, content, priority, "
        "lifecycle_status) VALUES (?, 'OBSERVATION', ?, ?, 'P2', 'ACTIVE')",
        ("t-2", "second thought essence", "second thought content"),
    )
    await db.execute(
        "INSERT INTO edge (edge_id, from_thought_id, to_thought_id, edge_type) "
        "VALUES ('e-1', 't-1', 't-2', 'ASSOCIATED')",
    )
    await db.execute(
        "INSERT INTO embedding (embedding_id, owner_type, owner_id, model_name, "
        "dimension, vector_blob, created_at) "
        "VALUES ('emb-1', 'THOUGHT', 't-1', 'probe-model', 3, ?, '2026-01-01T00:00:00+00:00')",
        (struct.pack("3f", 0.1, 0.2, 0.3),),
    )
    await db.commit()


# ---------------------------------------------------------------------------
# Schema shape capture + comparison
# ---------------------------------------------------------------------------


def _norm_sql(sql: str) -> str:
    """Collapse whitespace and tighten spacing around ``(`` ``)`` ``,``.

    ``sqlite_master.sql`` preserves the original statement text, so the fresh
    multi-line ``schema_core.sql`` DDL and the single-line migration DDL differ
    only in benign whitespace. Normalising both to a canonical spacing lets a
    real structural divergence (a renamed column, a changed predicate) still be
    caught while cosmetic layout differences are ignored.
    """
    collapsed = " ".join(sql.split())
    return re.sub(r"\s*([(),])\s*", r"\1", collapsed)


@dataclass(frozen=True)
class _SchemaShape:
    """A comparable structural fingerprint of a core schema.

    ``columns`` is keyed by name (order-independent) so the two legitimate
    historical column orders of ``content_hash`` do not force a false
    divergence; ``columns_ordered`` preserves cid order for the strict
    column-order parity check on the public (v12+) range.
    """

    columns: dict[str, dict[str, tuple[str, int, str | None, int]]]
    columns_ordered: dict[str, tuple[str, ...]]
    foreign_keys: dict[str, tuple[tuple[str, str, str, str], ...]]
    index_names: dict[str, frozenset[str]]
    index_sql: dict[str, str]
    triggers: dict[str, str]
    fts: str

    def semantic_equal(self, other: _SchemaShape) -> bool:
        """Return ``True`` when two shapes are structurally equivalent.

        Column *order* is deliberately excluded (see ``columns_ordered`` for
        the strict variant); every other facet — column definitions, foreign
        keys, index presence and DDL, triggers, FTS config — must match.
        """
        return (
            self.columns == other.columns
            and self.foreign_keys == other.foreign_keys
            and self.index_names == other.index_names
            and self.index_sql == other.index_sql
            and self.triggers == other.triggers
            and self.fts == other.fts
        )


#: The ``PRAGMA table_list`` types that belong to the declared schema. Only
#: SQLite's own ``shadow`` classification is dropped: shadow tables are an
#: implementation detail of the virtual table that owns them, and that virtual
#: table is compared in its own right.
_DECLARED_TABLE_TYPES = frozenset({"table", "virtual", "view"})


async def _schema_table_names(db: aiosqlite.Connection) -> set[str]:
    """Return the declared objects of ``db``'s main schema, as SQLite classifies them.

    ``PRAGMA table_list`` reports every entry as ``table``, ``virtual``,
    ``shadow`` or ``view``, so the FTS5 shadow tables are recognised **by SQLite
    itself** instead of by guessing at a name prefix. That matters three times
    over: a plain table that merely happens to share a virtual table's name
    prefix stays visible; a virtual table added to one bootstrap path and not the
    other is reported rather than skipped; and a view is reported too — each
    would otherwise be exactly as invisible as a missing plain table. SQLite's
    own ``sqlite_*`` objects are dropped, as is anything outside ``main``.

    ``table_list`` needs SQLite 3.37 (2021). That is a requirement of this test
    module only — nothing in the package depends on it — and an unknown pragma
    is a silent no-op rather than an error, so the empty result is caught below
    instead of quietly reporting that the database has no tables at all.
    """
    cursor = await db.execute("PRAGMA table_list")
    rows = await cursor.fetchall()
    assert rows, "PRAGMA table_list returned nothing — this test module needs SQLite 3.37+"
    return {
        str(row["name"])
        for row in rows
        if str(row["schema"]) == "main"
        and str(row["type"]) in _DECLARED_TABLE_TYPES
        and not str(row["name"]).startswith("sqlite_")
    }


async def _capture_schema_shape(db: aiosqlite.Connection) -> _SchemaShape:
    """Read a :class:`_SchemaShape` from a live connection.

    The captured facets are read per table from ``_CORE_TABLES`` (plus the FTS
    virtual table, fingerprinted separately), so the first thing asserted is that
    those names still *are* the tables the database carries. Without that, a
    table present in one database and absent from the other is simply never
    looked at, and every parity comparison below is blind to exactly the
    divergence it exists to catch.
    """
    present = await _schema_table_names(db)
    expected = {*_CORE_TABLES, _FTS_VIRTUAL_TABLE}
    assert present == expected, (
        "the captured schema shape only covers the tables named in _CORE_TABLES "
        "and the FTS virtual table, so the parity check cannot see a table "
        f"outside them. Only in the database: {sorted(present - expected)}; "
        f"only in the expected set: {sorted(expected - present)}"
    )

    columns: dict[str, dict[str, tuple[str, int, str | None, int]]] = {}
    columns_ordered: dict[str, tuple[str, ...]] = {}
    foreign_keys: dict[str, tuple[tuple[str, str, str, str], ...]] = {}
    index_names: dict[str, frozenset[str]] = {}

    for table in _CORE_TABLES:
        info = await db.execute(f"PRAGMA table_info({table})")
        rows = await info.fetchall()
        columns[table] = {
            str(row["name"]): (
                str(row["type"]),
                int(row["notnull"]),
                None if row["dflt_value"] is None else str(row["dflt_value"]),
                int(row["pk"]),
            )
            for row in rows
        }
        columns_ordered[table] = tuple(str(row["name"]) for row in rows)

        fk = await db.execute(f"PRAGMA foreign_key_list({table})")
        fk_rows = await fk.fetchall()
        foreign_keys[table] = tuple(
            sorted(
                (
                    str(row["table"]),
                    str(row["from"]),
                    str(row["to"]),
                    str(row["on_delete"]),
                )
                for row in fk_rows
            )
        )

        idx = await db.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = ?",
            (table,),
        )
        index_names[table] = frozenset(str(row["name"]) for row in await idx.fetchall())

    idx_sql_cur = await db.execute(
        "SELECT name, sql FROM sqlite_master WHERE type = 'index' AND sql IS NOT NULL"
    )
    index_sql = {
        str(row["name"]): _norm_sql(str(row["sql"])) for row in await idx_sql_cur.fetchall()
    }

    trg_cur = await db.execute("SELECT name, sql FROM sqlite_master WHERE type = 'trigger'")
    triggers = {str(row["name"]): _norm_sql(str(row["sql"])) for row in await trg_cur.fetchall()}

    fts_cur = await db.execute("SELECT sql FROM sqlite_master WHERE name = 'thought_fts'")
    fts_row = await fts_cur.fetchone()
    assert fts_row is not None
    fts = _norm_sql(str(fts_row["sql"]))

    return _SchemaShape(
        columns=columns,
        columns_ordered=columns_ordered,
        foreign_keys=foreign_keys,
        index_names=index_names,
        index_sql=index_sql,
        triggers=triggers,
        fts=fts,
    )


async def _user_version(db: aiosqlite.Connection) -> int:
    """Return the current ``PRAGMA user_version``."""
    cursor = await db.execute("PRAGMA user_version")
    row = await cursor.fetchone()
    assert row is not None
    return int(row[0])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def fresh_db() -> AsyncIterator[aiosqlite.Connection]:
    """Empty in-memory SQLite (``user_version`` starts at 0)."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    yield conn
    await conn.close()


async def _new_migrated_db(target: int, *, seed: bool) -> aiosqlite.Connection:
    """Return a v20 connection migrated from a real-shape ``target`` base."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await _bootstrap_core_at_version(conn, target)
    if seed:
        await _seed_legacy_rows(conn)
    await SqliteEngravaCore(conn).ensure_schema()
    return conn


async def _new_fresh_db() -> aiosqlite.Connection:
    """Return a v20 connection bootstrapped fresh from ``schema_core.sql``."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await SqliteEngravaCore(conn).ensure_schema()
    return conn


# ---------------------------------------------------------------------------
# Data-survival assertions (public read/write surface)
# ---------------------------------------------------------------------------


async def _assert_legacy_rows_survive(store: SqliteEngravaCore) -> None:
    """The pre-migration thought / edge / embedding are all queryable at head."""
    thought = await store.get_thought("t-1")
    assert thought is not None
    assert thought.essence == _LEGACY_ESSENCE

    listed = {t.thought_id for t in await store.list_thoughts()}
    assert {"t-1", "t-2"} <= listed

    edges = {e.edge_id: e for e in await store.get_edges("t-1")}
    assert "e-1" in edges
    # The v18->v19 upgrade backfills the empty-mapping default for legacy edges.
    assert edges["e-1"].metadata == {}

    embedding = await store.get_embedding("t-1")
    assert embedding is not None
    assert embedding.dimension == 3

    fts_hits = await store.search_fts(_LEGACY_TERM)
    assert any(tid == "t-1" for tid, _ in fts_hits)

    hybrid = await store.search_hybrid(_LEGACY_TERM)
    assert any(tid == "t-1" for tid, _ in hybrid.results)


async def _assert_api_roundtrip(store: SqliteEngravaCore) -> None:
    """A fresh thought/edge/embedding written via the public API round-trips.

    This is the direct guard against the ladder's failure mode: a stranded
    source version leaves a column missing, and the *first write* through the
    typed API raises. A clean round-trip proves the migrated head schema is
    fully writable.
    """
    first = ThoughtRecord(
        thought_id="api-t-1",
        thought_type=ThoughtType.OBSERVATION,
        essence="post upgrade widget essence",
        content="post upgrade widget content",
        priority=Priority.P2,
        lifecycle_status=LifecycleStatus.ACTIVE,
        source="upgrade-test",
    )
    second = ThoughtRecord(
        thought_id="api-t-2",
        thought_type=ThoughtType.OBSERVATION,
        essence="post upgrade sibling essence",
        content="post upgrade sibling content",
        priority=Priority.P2,
        lifecycle_status=LifecycleStatus.ACTIVE,
        source="upgrade-test",
    )
    await store.create_thought(first)
    await store.create_thought(second)
    await store.create_edge(
        EdgeRecord(
            edge_id="api-e-1",
            from_thought_id="api-t-1",
            to_thought_id="api-t-2",
            edge_type=EdgeType.ASSOCIATED,
            weight=0.7,
            created_cycle=0,
            metadata={"origin": "upgrade-test"},
        )
    )
    await store.store_embedding("api-t-1", [0.4, 0.5, 0.6], model_name="probe-model")

    reloaded = await store.get_thought("api-t-1")
    assert reloaded is not None
    assert reloaded.essence == "post upgrade widget essence"

    edges = {e.edge_id: e for e in await store.get_edges("api-t-1")}
    assert edges["api-e-1"].metadata == {"origin": "upgrade-test"}

    assert any(tid == "api-t-1" for tid, _ in await store.search_fts("widget"))


# ---------------------------------------------------------------------------
# 1. Exhaustive version-ladder convergence matrix
# ---------------------------------------------------------------------------

_LADDER_SEEDS: list[int | None] = [None, *range(2, 20)]


@pytest.mark.parametrize(
    "seed_version",
    _LADDER_SEEDS,
    ids=lambda v: "fresh" if v is None else f"v{v}",
)
async def test_version_ladder_converges_and_preserves_data(
    fresh_db: aiosqlite.Connection,
    seed_version: int | None,
) -> None:
    """Every seed version converges to head, is idempotent, and keeps its data.

    ``seed_version is None`` exercises the empty-database fresh bootstrap; every
    other value stamps a real-shape schema-at-``v`` fixture, writes legacy rows,
    then upgrades. Post-upgrade the ladder must have reached v20, a second
    ``ensure_schema`` must be a structural no-op, and the pre-existing rows
    (when seeded) plus a fresh API write must all be queryable.
    """
    store = SqliteEngravaCore(fresh_db)
    seeded = seed_version is not None
    if seed_version is not None:
        await _bootstrap_core_at_version(fresh_db, seed_version)
        await _seed_legacy_rows(fresh_db)
        assert await _user_version(fresh_db) == seed_version

    await store.ensure_schema()
    assert await _user_version(fresh_db) == _HEAD_VERSION
    shape_after_first = await _capture_schema_shape(fresh_db)

    if seeded:
        await _assert_legacy_rows_survive(store)

    # A second run must be a genuine no-op: version unchanged and the full
    # structural fingerprint (columns in order, indexes, triggers, FTS) stable.
    await store.ensure_schema()
    assert await _user_version(fresh_db) == _HEAD_VERSION
    assert await _capture_schema_shape(fresh_db) == shape_after_first

    if seeded:
        await _assert_legacy_rows_survive(store)
    await _assert_api_roundtrip(store)


# ---------------------------------------------------------------------------
# 2. Fresh-vs-migrated schema-drift parity
# ---------------------------------------------------------------------------


async def test_core_tables_list_matches_the_bootstrapped_schema() -> None:
    """``_CORE_TABLES`` + the FTS table are the tables a head database carries.

    The parity comparison reads its facets per table from this hand-maintained
    tuple, so a table the bootstrap gains (or loses) without the tuple following
    is invisible to every parity case — the list cannot detect its own rot. This
    names that invariant directly, so a stale tuple fails here with an
    attributable message rather than only as a side effect of a parity case.
    """
    expected = {*_CORE_TABLES, _FTS_VIRTUAL_TABLE}
    fresh = await _new_fresh_db()
    migrated = await _new_migrated_db(2, seed=False)
    try:
        assert await _schema_table_names(fresh) == expected
        assert await _schema_table_names(migrated) == expected
    finally:
        await fresh.close()
        await migrated.close()


async def test_fresh_equals_migrated_from_v2() -> None:
    """A fresh v20 database is structurally equivalent to one migrated from v2.

    This walks the entire ladder (v2 rebuilds FTS, v11->v12 recreates the FK
    tables, v18->v19 adds the edge metadata column, v19->v20 adds the
    thought.archived_at column) and asserts the end state matches a clean
    bootstrap column-for-column, FK-for-FK, index-for-index, trigger-for-trigger,
    and on the FTS config.
    """
    fresh = await _new_fresh_db()
    migrated = await _new_migrated_db(2, seed=True)
    try:
        assert await _user_version(fresh) == await _user_version(migrated) == _HEAD_VERSION
        fresh_shape = await _capture_schema_shape(fresh)
        migrated_shape = await _capture_schema_shape(migrated)
        assert fresh_shape.semantic_equal(migrated_shape)
    finally:
        await fresh.close()
        await migrated.close()


@pytest.mark.parametrize("seed_version", list(range(2, 20)))
async def test_fresh_equals_migrated_every_seed(seed_version: int) -> None:
    """Parity holds for every seed version, not just v2.

    Doubles as a fidelity check on the real-shape fixtures: a reconstruction
    that invented or dropped a column/table/index a version never had would
    diverge from the fresh bootstrap here.
    """
    fresh = await _new_fresh_db()
    migrated = await _new_migrated_db(seed_version, seed=False)
    try:
        fresh_shape = await _capture_schema_shape(fresh)
        migrated_shape = await _capture_schema_shape(migrated)
        assert fresh_shape.semantic_equal(migrated_shape)
    finally:
        await fresh.close()
        await migrated.close()


@pytest.mark.parametrize("seed_version", list(range(12, 20)))
async def test_fresh_equals_migrated_strict_column_order_public_range(seed_version: int) -> None:
    """Full cid-order column parity holds across the public (v12+) range.

    Every public release (v12 onward) carried ``content_hash`` at its head
    position and only ever *appended* later columns, so a migrated database
    must match a fresh one in exact column order — the "append last for
    column-order parity" contract the ``schema_core.sql`` comments promise.

    Only the public range is checked strictly: a pre-v10 base has
    ``content_hash`` added by an in-place ``ALTER`` (which can only append),
    a legitimate second historical order that parity compares
    order-independently above; no public database predates v12.
    """
    fresh = await _new_fresh_db()
    migrated = await _new_migrated_db(seed_version, seed=False)
    try:
        fresh_shape = await _capture_schema_shape(fresh)
        migrated_shape = await _capture_schema_shape(migrated)
        for table in _CORE_TABLES:
            assert migrated_shape.columns_ordered[table] == fresh_shape.columns_ordered[table]
    finally:
        await fresh.close()
        await migrated.close()


async def test_parity_check_detects_default_drift(
    fresh_db: aiosqlite.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The parity check is discriminating: a changed default breaks equivalence.

    Reverting the v18->v19 migration to add ``edge.metadata_json`` with a
    non-default value must make the migrated shape diverge from a fresh one —
    proving :meth:`_SchemaShape.semantic_equal` is not a tautology. The fresh
    bootstrap runs ``schema_core.sql`` directly (never the ladder), so it is
    unaffected by the patch and keeps the correct ``'{}'`` default.
    """

    async def _broken_v18_to_v19(self: SqliteEngravaCore) -> None:
        if not await self._table_exists("edge"):
            return
        if not await self._column_exists("edge", "metadata_json"):
            await self._db.execute(
                "ALTER TABLE edge ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{\"drift\": 1}'"
            )

    monkeypatch.setattr(SqliteEngravaCore, "_migrate_core_v18_to_v19", _broken_v18_to_v19)

    await _bootstrap_core_at_version(fresh_db, 18)
    await SqliteEngravaCore(fresh_db).ensure_schema()
    migrated_shape = await _capture_schema_shape(fresh_db)

    fresh = await _new_fresh_db()
    try:
        fresh_shape = await _capture_schema_shape(fresh)
    finally:
        await fresh.close()

    assert not fresh_shape.semantic_equal(migrated_shape)
    assert (
        migrated_shape.columns["edge"]["metadata_json"]
        != fresh_shape.columns["edge"]["metadata_json"]
    )


# ---------------------------------------------------------------------------
# 3. Registry-driven dispatch: ordering, postcondition gating, retryability
# ---------------------------------------------------------------------------


async def test_migration_registry_is_contiguous_and_ordered(
    fresh_db: aiosqlite.Connection,
) -> None:
    """The registry is the single source of upgrade order: contiguous 3..20.

    Every entry's target version is strictly greater than the previous one and
    the sequence is gap-free from the first post-bootstrap step (``v2 -> v3``)
    to head (``v19 -> v20``), so the loop applies exactly the right tail for any
    starting version and a future migration is one appended entry.
    """
    store = SqliteEngravaCore(fresh_db)
    targets = [target for target, _ in store._core_migration_steps()]

    assert targets == list(range(3, _HEAD_VERSION + 1))
    assert targets == sorted(targets)
    assert len(targets) == len(set(targets))


async def test_loop_runs_only_pending_steps(
    fresh_db: aiosqlite.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A step whose target is at or below the current version is never run.

    A database stamped at v19 must upgrade by running only ``v19 -> v20``. If
    the loop re-ran an already-applied step it would call the patched
    ``v18 -> v19`` helper below and raise; reaching head proves the loop skips
    every step whose target does not exceed the current version.
    """

    async def _must_not_run(self: SqliteEngravaCore) -> None:
        message = "already-applied step re-run"
        raise AssertionError(message)

    monkeypatch.setattr(SqliteEngravaCore, "_migrate_core_v18_to_v19", _must_not_run)

    await _bootstrap_core_at_version(fresh_db, 19)
    await _seed_legacy_rows(fresh_db)
    store = SqliteEngravaCore(fresh_db)

    await store.ensure_schema()

    assert await _user_version(fresh_db) == _HEAD_VERSION


async def test_double_run_after_head_is_a_noop(
    fresh_db: aiosqlite.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-running ``ensure_schema`` at head runs no migration step at all.

    Once the database is at head, a second ``ensure_schema`` must leave the
    version untouched and dispatch no step — patching every registered helper
    to raise proves the loop body is never entered on the second pass.
    """
    store = SqliteEngravaCore(fresh_db)
    await store.ensure_schema()
    assert await _user_version(fresh_db) == _HEAD_VERSION

    async def _must_not_run(self: SqliteEngravaCore) -> None:
        message = "no step may run at head"
        raise AssertionError(message)

    for _target, step in store._core_migration_steps():
        monkeypatch.setattr(SqliteEngravaCore, step.__name__, _must_not_run)

    await store.ensure_schema()
    assert await _user_version(fresh_db) == _HEAD_VERSION


async def test_unexpected_index_ddl_failure_leaves_version_retryable() -> None:
    """An isolated index-creation failure propagates and never marks the DB current.

    The ``v7 -> v8`` step creates ``idx_edge_type_from``. Seeding a conflicting
    object of that name makes the ``CREATE INDEX`` raise a real SQLite error
    that is **not** the absent-``edge``-table case the step tolerates. The error
    must propagate (not be swallowed), the version must stay at 7 (the failed
    step never bumps it), and once the conflict is removed a re-run must
    complete the upgrade to head with the pre-existing rows intact — proving the
    failure left the database retryable rather than falsely current.
    """
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    try:
        await _bootstrap_core_at_version(conn, 7)
        await _seed_legacy_rows(conn)
        # A non-index object occupying the index name the migration will create.
        await conn.execute("CREATE TABLE idx_edge_type_from (placeholder TEXT)")
        await conn.commit()

        store = SqliteEngravaCore(conn)
        with pytest.raises(aiosqlite.OperationalError, match="idx_edge_type_from"):
            await store.ensure_schema()

        # The failed step must not have advanced the version past its input.
        assert await _user_version(conn) == 7

        # Remove the conflict and retry: the upgrade resumes and completes.
        await conn.execute("DROP TABLE idx_edge_type_from")
        await conn.commit()
        await store.ensure_schema()

        assert await _user_version(conn) == _HEAD_VERSION
        await _assert_legacy_rows_survive(store)
        await _assert_api_roundtrip(store)
    finally:
        await conn.close()


async def test_failed_step_stops_at_last_successful_version(
    fresh_db: aiosqlite.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mid-ladder failure bumps only through the last fully-applied step.

    Seeded at v18, the loop applies ``v18 -> v19`` (bumping to 19) and then
    fails on a patched ``v19 -> v20``. The version must be exactly 19 — the last
    step that fully applied — never 20. Restoring the helper and re-running must
    resume from 19 and reach head, so the interrupted upgrade was retryable.
    """
    await _bootstrap_core_at_version(fresh_db, 18)
    await _seed_legacy_rows(fresh_db)
    store = SqliteEngravaCore(fresh_db)

    async def _boom(self: SqliteEngravaCore) -> None:
        message = "injected v19->v20 failure"
        raise RuntimeError(message)

    monkeypatch.setattr(SqliteEngravaCore, "_migrate_core_v19_to_v20", _boom)
    with pytest.raises(RuntimeError, match="injected"):
        await store.ensure_schema()
    assert await _user_version(fresh_db) == 19

    monkeypatch.undo()
    await store.ensure_schema()
    assert await _user_version(fresh_db) == _HEAD_VERSION
    await _assert_legacy_rows_survive(store)
    await _assert_api_roundtrip(store)


@pytest.mark.parametrize("stamped_version", [0, 1])
async def test_below_floor_versions_bootstrap_fresh(
    fresh_db: aiosqlite.Connection,
    stamped_version: int,
) -> None:
    """Any ``user_version`` below the bootstrap floor loads the full head schema.

    An empty database at version 0 (the default) and one explicitly stamped at
    version 1 both sit below the migration-ladder floor, so ``ensure_schema``
    must run ``schema_core.sql`` directly and land at head with a fully writable
    schema — never entering the incremental loop.
    """
    await fresh_db.execute(f"PRAGMA user_version = {stamped_version}")
    await fresh_db.commit()

    store = SqliteEngravaCore(fresh_db)
    await store.ensure_schema()

    assert await _user_version(fresh_db) == _HEAD_VERSION
    await _assert_api_roundtrip(store)


async def test_postcondition_failure_raises_and_leaves_version_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A step whose postcondition does not hold raises and never bumps the version.

    Forcing ``_index_exists`` to report the ``v7 -> v8`` index absent even after
    its ``CREATE INDEX`` runs simulates the exact "version advanced without the
    required index" hole the postcondition closes: the step must raise
    :class:`CoreMigrationError`, the version must stay at 7, and once the probe
    behaves again a re-run must reach head — proving the postcondition gates the
    bump and leaves the database retryable.
    """
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    try:
        await _bootstrap_core_at_version(conn, 7)
        await _seed_legacy_rows(conn)
        store = SqliteEngravaCore(conn)

        async def _index_never_present(self: SqliteEngravaCore, index: str) -> bool:
            return False

        monkeypatch.setattr(SqliteEngravaCore, "_index_exists", _index_never_present)
        with pytest.raises(CoreMigrationError, match="idx_edge_type_from"):
            await store.ensure_schema()
        assert await _user_version(conn) == 7

        monkeypatch.undo()
        await store.ensure_schema()
        assert await _user_version(conn) == _HEAD_VERSION
        await _assert_legacy_rows_survive(store)
        await _assert_api_roundtrip(store)
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# 4. Static structure of the bootstrap script
# ---------------------------------------------------------------------------
#
# ``schema_core.sql`` stamps ``PRAGMA user_version`` as its very last statement
# so a bootstrap that fails part-way leaves the database unstamped (0) and
# therefore retryable. ``executescript`` is not atomic: every statement that
# already ran is durable, so a stamp placed anywhere but last would mark a
# database current over a schema still missing everything that follows the
# stamp — permanently, because ``ensure_schema`` neither re-bootstraps nor
# migrates a database that already reads as head.
#
# Runtime failure-injection (``TestBootstrapAtomicity`` in
# ``test_referential_integrity.py``) can only ever probe the offsets it injects
# at. The assertions below read the script text instead, so they hold for every
# offset at once and cannot drift as statements are added to the script.

#: Matches a quoted token — SQLite's four quoting styles, ``'x'`` ``"x"`` ``[x]``
#: and ```x``` — or a ``--`` line comment or a ``/* */`` block comment. Quoted
#: tokens come first so a ``--``, ``;``, ``/*`` or ``*/`` *inside* one is never
#: mistaken for a comment or a statement boundary; a token spelled ``[/*]`` would
#: otherwise open a comment that swallowed the statements after it, a version
#: stamp among them. A block comment may run to end-of-input without its closing
#: ``*/`` — SQLite accepts that, so a scan demanding the terminator would go red
#: on a purely cosmetic trailing note.
_SQL_LITERAL_OR_COMMENT = re.compile(
    r"""
      (?P<quoted>'(?:[^']|'')*'|"(?:[^"]|"")*"|`(?:[^`]|``)*`|\[[^\]]*\])
    | (?P<comment>--[^\n]*|/\*.*?(?:\*/|\Z))
    """,
    re.VERBOSE | re.DOTALL,
)

#: Characters a quoted token may keep once unquoted: ASCII word characters plus
#: a leading sign, since SQLite takes a quoted value too (``= '+20'`` stamps 20).
#: Everything else is neutralised, so ``'a;b'`` cannot forge a statement boundary
#: while ``'user_version'`` still reads as the pragma it names. ``re.ASCII``
#: matters: without it a confusable (a LATIN SMALL LETTER LONG S standing in
#: for the plain one) survives as a word character and Python's Unicode
#: case-folding then equates it with the real pragma, while SQLite treats it
#: as an unknown pragma and stamps nothing.
_QUOTED_TOKEN_SAFE = re.compile(r"[^\w+-]", re.ASCII)

#: The version stamp: an optional ``main.`` qualifier, then either the assignment
#: form (``PRAGMA main.user_version = 20``) or the call form
#: (``PRAGMA user_version(20)``). The two forms are separate alternatives rather
#: than a shared optional bracket, so an unbalanced ``user_version(20`` is not
#: accepted. The qualifier is restricted to ``main`` on purpose: SQLite accepts
#: ``PRAGMA temp.user_version = 20`` and it leaves the durable database at 0, so
#: treating it as the stamp would be exactly wrong. Quoted spellings need no
#: alternatives here — the scrubber has already unquoted them.
_VERSION_STAMP = re.compile(
    r"""PRAGMA\s+(?:main\s*\.\s*)?user_version\s*
        (?: =\s*(?P<assigned>[+-]?\d+)
          | \(\s*(?P<called>[+-]?\d+)\s*\)
        )""",
    re.ASCII | re.IGNORECASE | re.VERBOSE,
)

#: A lexical tripwire, deliberately broader than the stamp pattern: any
#: occurrence of the token at all. The "exactly once" check counts these rather
#: than recognised stamps, so a spelling this module failed to anticipate trips
#: it instead of being silently ignored. It fails *closed* — a future statement
#: merely naming a ``user_version`` column would also trip it, which is the
#: intended direction for a bootstrap invariant. ASCII-only for the same reason
#: as the stamp pattern: a Unicode confusable is not this pragma.
_MENTIONS_USER_VERSION = re.compile(r"\buser_version\b", re.ASCII | re.IGNORECASE)

#: The first statement of the trailing index block. Relocating the stamp above
#: this line is the mutation these assertions exist to catch: every statement
#: after it is a ``CREATE INDEX``, so a failure there would leave a database
#: stamped at head while missing its hot-path, valid-time and provenance indexes.
_INDEX_BLOCK_HEAD = "CREATE INDEX IF NOT EXISTS idx_thought_valid_from"

#: The stamp exactly as the script spells it, for the mutation helpers below.
_STAMP_STATEMENT = f"PRAGMA user_version = {_HEAD_VERSION};"


def _scrub_sql(sql: str) -> str:
    """Return ``sql`` normalised so statement boundaries and pragma names are readable.

    Comments become a single space — never nothing, so removing one cannot glue
    two tokens together.

    Every **quoted token** is unquoted: word characters are kept, everything else
    is neutralised, and the result is padded so it cannot fuse with the token
    before it. Both halves of that are load-bearing. Neutralising is what makes
    the ``;`` split safe against a value or identifier containing a semicolon or
    a comment marker. *Keeping* the word characters is what stops a quoted pragma
    name from hiding: SQLite accepts all four quoting styles for a pragma name —
    ``PRAGMA 'user_version' = 20`` and ``PRAGMA [user_version] = 20`` both stamp
    the database for real — so a token whose contents were blanked out would let
    an early stamp pass unseen. Treating value literals the same way costs only
    a fail-*closed* over-report: a literal that happens to contain the bare word
    would trip the mention counter, which is the safe direction here.

    Unquoting also means the patterns above never need a quoted alternative,
    which is what lets the schema qualifier stay strict. The one thing it cannot
    distinguish is a quoted token in *keyword* position (``"PRAGMA"
    "user_version" = 20`` reads as a stamp after unquoting) — but SQLite rejects
    that outright, so such a script never bootstraps and the rest of this module
    fails loudly rather than silently.
    """

    def _normalise(match: re.Match[str]) -> str:
        if match.lastgroup != "quoted":
            return " "
        return f" {_QUOTED_TOKEN_SAFE.sub('x', match.group()[1:-1])} "

    return _SQL_LITERAL_OR_COMMENT.sub(_normalise, sql)


def _sql_fragments(sql: str) -> list[str]:
    """Return the non-empty ``;``-delimited fragments of ``sql``, whitespace-collapsed.

    A ``CREATE TRIGGER`` body splits into several fragments here, because its
    inner ``;`` cannot be told apart from a statement terminator without a real
    parser — hence "fragments", not "statements". That is irrelevant to the
    tail-of-script invariant asserted below: the final fragment is the final
    statement, whatever precedes it.
    """
    return [" ".join(part.split()) for part in _scrub_sql(sql).split(";") if part.strip()]


def _read_schema_core_sql() -> str:
    """Return the bundled bootstrap script exactly as ``ensure_schema`` reads it."""
    return (
        resources.files("engrava.infrastructure.sqlite")
        .joinpath("schema_core.sql")
        .read_text(encoding="utf-8")
    )


def _stamped_version(fragment: str) -> int | None:
    """Return the version ``fragment`` stamps, or ``None`` if it is not a stamp."""
    stamp = _VERSION_STAMP.fullmatch(fragment)
    if stamp is None:
        return None
    return int(stamp.group("assigned") or stamp.group("called"))


def _count_user_version_mentions(fragments: list[str]) -> int:
    """Return how many of ``fragments`` name ``user_version`` at all.

    Lexical, not syntactic: it counts the token rather than parsing pragmas, so
    it over-reports rather than under-reports. That is the safe direction here —
    an unrecognised early stamp must not be silently ignored.
    """
    return sum(1 for fragment in fragments if _MENTIONS_USER_VERSION.search(fragment))


def _replace_once(sql: str, needle: str, replacement: str) -> str:
    """Return ``sql`` with the single occurrence of ``needle`` replaced.

    Asserting the count rather than mere presence keeps a mutation from landing
    somewhere other than the statement it names — a second textual occurrence
    (a longer object name sharing the prefix, a stray fragment) would otherwise
    silently redirect it and the mutation would prove nothing.
    """
    assert sql.count(needle) == 1, f"expected exactly one occurrence of {needle!r}"
    return sql.replace(needle, replacement, 1)


def _with_stamp_moved_above_the_index_block(sql: str) -> str:
    """Return ``sql`` with the version stamp relocated ahead of the index block."""
    without_stamp = _replace_once(sql, _STAMP_STATEMENT, "")
    return _replace_once(
        without_stamp, _INDEX_BLOCK_HEAD, f"{_STAMP_STATEMENT}\n{_INDEX_BLOCK_HEAD}"
    )


def test_schema_core_stamps_the_version_as_its_last_statement() -> None:
    """The final statement of ``schema_core.sql`` is the head version stamp.

    Structural, not positional: it holds for every DDL statement in the script
    at once, rather than for the one a failure-injection test happens to target.
    """
    fragments = _sql_fragments(_read_schema_core_sql())
    assert fragments, "schema_core.sql yielded no statements at all"

    assert _stamped_version(fragments[-1]) == _HEAD_VERSION, (
        "the version stamp must be the last statement of schema_core.sql: "
        "executescript is not atomic, so a stamp placed any earlier marks the "
        "database current over whatever DDL fails after it, and ensure_schema "
        f"never revisits a database that reads as head. Last statement: {fragments[-1]!r}"
    )


def test_schema_core_names_user_version_exactly_once() -> None:
    """No second stamp precedes the last one, in any spelling.

    "The last statement is a stamp" is on its own satisfied by a script that
    *also* stamps early, which reopens the same hole without moving anything —
    and SQLite accepts several spellings of the pragma, so this counts every
    statement that mentions ``user_version`` rather than only recognised stamps.
    """
    assert _count_user_version_mentions(_sql_fragments(_read_schema_core_sql())) == 1


# Each spelling is checked through the real pipeline (scrub, split, recognise),
# not against the pattern in isolation — the quoted forms only reduce to
# something the pattern can see because the scrubber unquotes them first.
# Verified against SQLite 3.53: every "accepted" case below really does stamp
# ``main.user_version``, and every rejected one either does not stamp it or is
# not executable at all.
#: A stamp whose pragma name carries a Unicode confusable (LATIN SMALL LETTER
#: LONG S in place of the plain one), written as an escape so this file stays
#: ASCII. SQLite reads it as an unknown pragma and stamps nothing.
_CONFUSABLE_STAMP = "PRAGMA 'u\u017fer_version' = 20"

_STAMP_SPELLINGS = [
    pytest.param("PRAGMA user_version = 20", 20, id="assignment"),
    pytest.param("PRAGMA user_version(20)", 20, id="call-form"),
    pytest.param("PRAGMA main.user_version = 20", 20, id="schema-qualified"),
    pytest.param('PRAGMA "main".user_version = 20', 20, id="quoted-schema-qualified"),
    pytest.param("PRAGMA [user_version] = 20", 20, id="bracketed-pragma-name"),
    pytest.param('PRAGMA "user_version" = 20', 20, id="double-quoted-pragma-name"),
    pytest.param("PRAGMA `user_version` = 20", 20, id="backticked-pragma-name"),
    pytest.param("PRAGMA 'user_version' = 20", 20, id="single-quoted-pragma-name"),
    pytest.param("PRAGMA 'main'.user_version = 20", 20, id="single-quoted-qualifier"),
    pytest.param("PRAGMA user_version = +20", 20, id="explicitly-signed"),
    pytest.param("PRAGMA user_version = '20'", 20, id="quoted-value"),
    pytest.param("PRAGMA user_version = '+20'", 20, id="quoted-signed-value"),
    pytest.param("pragma  USER_VERSION=20", 20, id="restyled"),
    # Accepted by SQLite, but it leaves main.user_version at 0 — recognising any
    # of these as the stamp would report an unstamped database as stamped.
    pytest.param("PRAGMA temp.user_version = 20", None, id="wrong-schema"),
    pytest.param(_CONFUSABLE_STAMP, None, id="unicode-confusable-name"),
    pytest.param("PRAGMA user_version(20", None, id="unclosed-call-form"),
    pytest.param("PRAGMA user_version = 20)", None, id="stray-closing-bracket"),
    pytest.param("PRAGMA main].user_version = 20", None, id="malformed-qualifier"),
    pytest.param("CREATE INDEX idx_x ON thought(essence)", None, id="not-a-stamp"),
]


@pytest.mark.parametrize(("statement", "expected"), _STAMP_SPELLINGS)
def test_stamp_recognition_covers_the_spellings_sqlite_accepts(
    statement: str,
    expected: int | None,
) -> None:
    """Every spelling that really stamps is recognised; nothing else is.

    Both directions matter. Failing to recognise a legitimate spelling makes the
    last-statement assertion fire on a restyled script — a false positive that
    gets the assertion deleted. Accepting something that does *not* stamp the
    durable database (a ``temp.`` qualifier, an unbalanced bracket) lets a
    script satisfy the invariant while leaving it broken.
    """
    assert _stamped_version(_sql_fragments(f"{statement};")[-1]) == expected


#: SQLite's four quoting styles, as (open, close) pairs. Every one of them can
#: spell a pragma name and every one of them can contain a comment marker, so
#: each is exercised in both roles below.
_QUOTING_STYLES = [
    pytest.param("'", "'", id="single-quoted"),
    pytest.param('"', '"', id="double-quoted"),
    pytest.param("[", "]", id="bracketed"),
    pytest.param("`", "`", id="backticked"),
]


@pytest.mark.parametrize(("open_quote", "close_quote"), _QUOTING_STYLES)
def test_quoted_tokens_cannot_open_a_comment(open_quote: str, close_quote: str) -> None:
    """A comment marker inside a quoted token does not start a comment.

    A scan that did not know a quoting style would read ``[/*]`` as opening a
    block comment — which would swallow every statement up to the next ``*/``,
    an early version stamp among them, and report the script as clean.
    """
    opener = f"{open_quote}/*{close_quote}"
    closer = f"{open_quote}*/{close_quote}"
    fragments = _sql_fragments(
        f"CREATE TABLE {opener}(x);\n"
        "PRAGMA user_version = 19;\n"
        f"CREATE TABLE {closer}(x);\n"
        "PRAGMA user_version = 20;\n"
    )

    assert _count_user_version_mentions(fragments) == 2
    assert _stamped_version(fragments[-1]) == _HEAD_VERSION


@pytest.mark.parametrize(("open_quote", "close_quote"), _QUOTING_STYLES)
def test_an_early_stamp_with_a_quoted_pragma_name_is_still_counted(
    open_quote: str,
    close_quote: str,
) -> None:
    """A quoted pragma name does not hide an early stamp from the mention counter.

    ``PRAGMA [user_version] = 20`` and ``PRAGMA 'user_version' = 20`` stamp the
    database exactly like the bare spelling (verified against SQLite). A scrubber
    that blanked the quoted token out would leave the script reporting a single
    mention and both invariants green, while the bootstrap durably marked an
    incomplete schema as current.
    """
    quoted_name = f"{open_quote}user_version{close_quote}"
    fragments = _sql_fragments(f"PRAGMA {quoted_name} = 19;\nPRAGMA user_version = 20;")

    assert _count_user_version_mentions(fragments) == 2
    assert _stamped_version(fragments[0]) == 19


def test_a_unicode_confusable_is_not_counted_as_the_pragma() -> None:
    """A confusable pragma name is not the pragma, so it must not read as one.

    A LATIN SMALL LETTER LONG S standing in for the plain one makes an *unknown*
    pragma to SQLite: a silent no-op leaving the version at 0. Python's folding
    equates the long s with a plain one, so an ASCII-blind scan would report a
    script that never stamps as correctly stamped — the fail-*open* direction.
    """
    fragments = _sql_fragments(f"{_CONFUSABLE_STAMP};")

    assert _count_user_version_mentions(fragments) == 0
    assert _stamped_version(fragments[-1]) is None


@pytest.mark.parametrize(("open_quote", "close_quote"), _QUOTING_STYLES)
def test_a_quoted_semicolon_cannot_forge_a_statement_boundary(
    open_quote: str,
    close_quote: str,
) -> None:
    """A ``;`` inside a quoted token does not split a statement.

    The other half of the unquoting trade: keeping the word characters must not
    also keep the punctuation, or a default value spelled ``'a;b'`` would make
    the scan see a statement that is not there — and the one it reported as last
    would not be the last.
    """
    quoted = f"{open_quote}a;b{close_quote}"
    fragments = _sql_fragments(
        f"CREATE TABLE t (c TEXT DEFAULT {quoted});\nPRAGMA user_version = 20;"
    )

    assert len(fragments) == 2
    assert _stamped_version(fragments[-1]) == _HEAD_VERSION


# Each case carries its own id, so there is no second list to keep in step with
# this one: a reordering cannot silently mislabel a case.
_LAYOUT_REWRITES = [
    pytest.param(
        _STAMP_STATEMENT,
        f"{_STAMP_STATEMENT}\n-- a note after the stamp\n",
        id="trailing-line-comment",
    ),
    pytest.param(
        _STAMP_STATEMENT,
        f"{_STAMP_STATEMENT}\n/* a note; with a semicolon */\n",
        id="trailing-block-comment",
    ),
    pytest.param(
        _STAMP_STATEMENT,
        f"{_STAMP_STATEMENT}\n/* a note left unterminated",
        id="unterminated-trailing-block-comment",
    ),
    pytest.param(
        _STAMP_STATEMENT,
        f"{_STAMP_STATEMENT}\n-- CREATE INDEX IF NOT EXISTS idx_note ON thought(essence);\n",
        id="commented-out-trailing-statement",
    ),
    pytest.param(
        _STAMP_STATEMENT,
        f"{_STAMP_STATEMENT}  -- head",
        id="inline-comment-on-the-stamp-line",
    ),
    pytest.param(
        _STAMP_STATEMENT,
        f"{_STAMP_STATEMENT}   \n\n\t\n   ",
        id="trailing-blank-lines-and-whitespace",
    ),
    pytest.param(
        _STAMP_STATEMENT,
        f"pragma  user_version={_HEAD_VERSION} ;",
        id="restyled-stamp-statement",
    ),
    pytest.param(
        _STAMP_STATEMENT,
        f"PRAGMA main.user_version = {_HEAD_VERSION};",
        id="schema-qualified-stamp",
    ),
    pytest.param(
        _STAMP_STATEMENT,
        f'PRAGMA "main".user_version = {_HEAD_VERSION};',
        id="quoted-schema-qualified-stamp",
    ),
    pytest.param(
        _STAMP_STATEMENT,
        f"PRAGMA user_version({_HEAD_VERSION});",
        id="call-form-stamp",
    ),
    pytest.param(
        _STAMP_STATEMENT,
        f"PRAGMA user_version = +{_HEAD_VERSION};",
        id="signed-stamp-value",
    ),
    pytest.param(
        "CREATE TABLE IF NOT EXISTS _metadata",
        'CREATE TABLE IF NOT EXISTS "_metadata"',
        id="quoted-table-identifier",
    ),
    pytest.param("\n", "\r\n", id="crlf-line-endings"),
    pytest.param("\n", "\n\n", id="extra-blank-line-everywhere"),
]


@pytest.mark.parametrize(("needle", "replacement"), _LAYOUT_REWRITES)
def test_last_statement_scan_survives_reformatting(needle: str, replacement: str) -> None:
    """Reformatting ``schema_core.sql`` does not make the stamp assertion fire.

    An assertion that goes red when somebody adds a comment gets deleted by the
    next person to touch the file, which is worse than not having it. Each case
    rewrites the real script in a way that changes only its layout, its comments
    or the spelling of the stamp itself — a *commented-out* trailing statement,
    which is what a scan that strips whitespace but not comments would trip over;
    an unterminated trailing block comment, which SQLite accepts and runs to
    end-of-input; and every pragma spelling SQLite takes. The stamp must still be
    found as the last statement, at the head version, exactly once.
    """
    original = _read_schema_core_sql()
    rewritten = original.replace(needle, replacement)
    # Without this the case would pass by rewriting nothing at all.
    assert rewritten != original, f"rewrite matched nothing: {needle!r} is not in the script"

    fragments = _sql_fragments(rewritten)

    assert _stamped_version(fragments[-1]) == _HEAD_VERSION
    assert _count_user_version_mentions(fragments) == 1


def test_last_statement_scan_detects_a_relocated_stamp() -> None:
    """Moving the stamp above the index block is seen — the scan is not a tautology.

    This is the mutation the assertion exists to catch, applied to the real
    script: the last statement is then a ``CREATE INDEX``, not the stamp.
    """
    fragments = _sql_fragments(_with_stamp_moved_above_the_index_block(_read_schema_core_sql()))

    assert _stamped_version(fragments[-1]) is None
    assert fragments[-1].startswith("CREATE INDEX")
    # Still exactly one stamp: only its position changed, which is precisely why
    # a count-based check alone would not catch this.
    assert _count_user_version_mentions(fragments) == 1


@pytest.mark.parametrize(
    "early_stamp",
    [
        f"PRAGMA user_version = {_HEAD_VERSION};",
        f"PRAGMA main.user_version = {_HEAD_VERSION};",
        f"PRAGMA user_version({_HEAD_VERSION});",
    ],
    ids=["plain", "schema-qualified", "call-form"],
)
def test_stamp_count_scan_detects_a_duplicate_stamp(early_stamp: str) -> None:
    """An *added* early stamp is seen even though the script still ends with one.

    The mirror image of the test above: the position is unchanged, so only the
    count catches it. Together the two assertions cover both ways the invariant
    breaks. Each SQLite spelling of the pragma is exercised, because an early
    stamp written in an unanticipated form is exactly how this check would be
    defeated while staying green.
    """
    duplicated = _replace_once(
        _read_schema_core_sql(), _INDEX_BLOCK_HEAD, f"{early_stamp}\n{_INDEX_BLOCK_HEAD}"
    )
    fragments = _sql_fragments(duplicated)

    assert _stamped_version(fragments[-1]) == _HEAD_VERSION
    assert _count_user_version_mentions(fragments) == 2
