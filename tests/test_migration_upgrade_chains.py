"""Whole-ladder schema migration hardening for ``ensure_schema``.

``ensure_schema`` migrates a database up to the head ``user_version`` (20)
through a hand-written ``elif`` cascade plus a chain of per-version
``_migrate_core_v*`` helpers. Existing suites cover single rungs (the
v13->v14 hot-path indexes and the v18->v19 edge metadata column); this module
generalises that to the **entire** ladder so an off-by-one arm (a wrong start
version or a skipped step) can no longer strand one specific source version
undetected.

Three properties are asserted:

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

Reconstruction fidelity note: because the parity test asserts a migrated
database equals a freshly bootstrapped one, a fixture that invented a column /
table a version never had, or omitted one it did, would surface as a parity
divergence — so the fixtures are self-checking against the real schema.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
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

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_HEAD_VERSION = 20

# The user (non-shadow) tables that exist at head v20, compared column-by-column
# for fresh-vs-migrated parity.
_CORE_TABLES = (
    "thought",
    "edge",
    "embedding",
    "action",
    "_metadata",
    "journal_entry",
    "extension_schema_versions",
)

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


async def _capture_schema_shape(db: aiosqlite.Connection) -> _SchemaShape:
    """Read a :class:`_SchemaShape` from a live connection."""
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
