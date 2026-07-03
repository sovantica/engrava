"""Tests for core-12 referential integrity (FK + ON DELETE CASCADE).

Covered surface:

* ``create_edge`` rejects orphan endpoints (both ``from_thought_id`` and
  ``to_thought_id``) and raises ``ReferentialIntegrityError``; raw
  SQLite read-back confirms zero orphan rows persist after the reject.
* ``delete_thought`` cascades to ``edge`` (both endpoints), ``embedding``
  (``owner_id``) and ``action`` (``source_thought_id``); raw read-back
  shows zero residual rows.
* The archive cleanup strategy does NOT cascade — only the parent
  transitions to ARCHIVED; children stay.
* The v11 → v12 migration recreates child tables with FK clauses,
  purges pre-existing orphans, preserves valid rows, is idempotent,
  and recovers cleanly when re-run after a partial completion.

Every assertion uses raw SQLite reads (not the public ORM) so the
data-layer guarantees stay visible even if the higher-level API ever
masks them.
"""

from __future__ import annotations

import datetime
import uuid
from typing import TYPE_CHECKING

import aiosqlite
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from engrava.domain.enums import (
    ActionStatus,
    ActionType,
    EdgeType,
    KnowledgeSource,
    LifecycleStatus,
    Priority,
    ThoughtType,
    VerificationStatus,
)
from engrava.domain.exceptions import ReferentialIntegrityError
from engrava.domain.models.action import ActionRecord
from engrava.domain.models.edge import EdgeRecord
from engrava.domain.models.thought import ThoughtRecord
from engrava.infrastructure.sqlite.engrava_core import SqliteEngravaCore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def _make_thought(tid: str, *, expires_at: str | None = None) -> ThoughtRecord:
    return ThoughtRecord(
        thought_id=tid,
        thought_type=ThoughtType.OBSERVATION,
        essence=f"essence-{tid}",
        content=f"content-{tid}",
        priority=Priority.P3,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=0,
        updated_cycle=0,
        source="test",
        confidence=0.9,
        expires_at=expires_at,
    )


def _make_edge(eid: str, src: str, dst: str) -> EdgeRecord:
    return EdgeRecord(
        edge_id=eid,
        from_thought_id=src,
        to_thought_id=dst,
        edge_type=EdgeType.ASSOCIATED,
        weight=0.5,
        created_cycle=0,
        source=KnowledgeSource.EXPERIENCE,
    )


def _make_action(aid: str, src: str) -> ActionRecord:
    return ActionRecord(
        action_id=aid,
        source_thought_id=src,
        action_type=ActionType.CLI_OUTPUT,
        intent="intent",
        status=ActionStatus.PLANNED,
        verification_status=VerificationStatus.PENDING,
    )


@pytest.fixture
async def store() -> AsyncIterator[SqliteEngravaCore]:
    """A freshly bootstrapped in-memory engrava core for one test (archive strategy)."""
    async with aiosqlite.connect(":memory:") as db:
        db.row_factory = aiosqlite.Row
        core = SqliteEngravaCore(db=db, embedding_provider=None, auto_embed=False)
        await core.ensure_schema()
        yield core


@pytest.fixture
async def delete_store() -> AsyncIterator[SqliteEngravaCore]:
    """A fresh in-memory engrava core configured with the delete TTL strategy."""
    async with aiosqlite.connect(":memory:") as db:
        db.row_factory = aiosqlite.Row
        core = SqliteEngravaCore(
            db=db,
            embedding_provider=None,
            auto_embed=False,
            ttl_strategy="delete",
        )
        await core.ensure_schema()
        yield core


class TestForeignKeysActuallyEnforced:
    """Diagnostic gate — schema-level FK declaration and runtime enforcement."""

    async def test_pragma_reports_fk_enabled(self, store: SqliteEngravaCore) -> None:
        cursor = await store._db.execute("PRAGMA foreign_keys")
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == 1

    async def test_edge_carries_two_fk_clauses(self, store: SqliteEngravaCore) -> None:
        cursor = await store._db.execute("PRAGMA foreign_key_list(edge)")
        rows = list(await cursor.fetchall())
        froms = {row["from"] for row in rows}
        assert froms == {"from_thought_id", "to_thought_id"}
        assert all(row["on_delete"] == "CASCADE" for row in rows)

    async def test_embedding_carries_fk_on_owner_id(self, store: SqliteEngravaCore) -> None:
        cursor = await store._db.execute("PRAGMA foreign_key_list(embedding)")
        rows = list(await cursor.fetchall())
        assert len(rows) == 1
        assert rows[0]["from"] == "owner_id"
        assert rows[0]["on_delete"] == "CASCADE"

    async def test_action_carries_fk_on_source_thought_id(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        cursor = await store._db.execute("PRAGMA foreign_key_list(action)")
        rows = list(await cursor.fetchall())
        assert len(rows) == 1
        assert rows[0]["from"] == "source_thought_id"
        assert rows[0]["on_delete"] == "CASCADE"

    async def test_user_version_is_head(self, store: SqliteEngravaCore) -> None:
        cursor = await store._db.execute("PRAGMA user_version")
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == 16


class TestCreateEdgeRejectsOrphans:
    """Inserting an edge whose endpoint does not exist must raise + leave nothing."""

    async def test_orphan_from_thought_id_is_rejected(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        await store.create_thought(_make_thought("t1"))
        with pytest.raises(ReferentialIntegrityError) as excinfo:
            await store.create_edge(_make_edge("e1", "ghost", "t1"))
        assert excinfo.value.column == "from_thought_id"
        assert excinfo.value.referenced_id == "ghost"

    async def test_orphan_to_thought_id_is_rejected(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        await store.create_thought(_make_thought("t1"))
        with pytest.raises(ReferentialIntegrityError) as excinfo:
            await store.create_edge(_make_edge("e1", "t1", "ghost"))
        assert excinfo.value.column == "to_thought_id"
        assert excinfo.value.referenced_id == "ghost"

    async def test_no_row_persisted_after_reject(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_make_thought("t1"))
        with pytest.raises(ReferentialIntegrityError):
            await store.create_edge(_make_edge("e1", "ghost-from", "ghost-to"))
        cursor = await store._db.execute("SELECT COUNT(*) FROM edge")
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == 0

    async def test_valid_edge_still_succeeds(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_make_thought("t1"))
        await store.create_thought(_make_thought("t2"))
        edge = await store.create_edge(_make_edge("e1", "t1", "t2"))
        assert edge.edge_id == "e1"
        cursor = await store._db.execute("SELECT COUNT(*) FROM edge")
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == 1

    async def test_unique_violation_propagates_unchanged(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """The wrapper only catches FK failures; UNIQUE violations stay raw."""
        await store.create_thought(_make_thought("t1"))
        await store.create_thought(_make_thought("t2"))
        await store.create_edge(_make_edge("e1", "t1", "t2"))
        with pytest.raises(aiosqlite.IntegrityError) as excinfo:
            # Same (from, to, type) tuple triggers UNIQUE; not FK.
            await store.create_edge(_make_edge("e2", "t1", "t2"))
        assert "UNIQUE" in str(excinfo.value).upper()


class TestCascadeOnDelete:
    """delete_thought removes children across all three child tables."""

    async def test_edges_cascade_on_from_endpoint(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        await store.create_thought(_make_thought("t1"))
        await store.create_thought(_make_thought("t2"))
        await store.create_edge(_make_edge("e1", "t1", "t2"))
        await store.delete_thought("t1")
        cursor = await store._db.execute(
            "SELECT COUNT(*) FROM edge WHERE edge_id = ?",
            ("e1",),
        )
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == 0

    async def test_edges_cascade_on_to_endpoint(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        await store.create_thought(_make_thought("t1"))
        await store.create_thought(_make_thought("t2"))
        await store.create_edge(_make_edge("e1", "t1", "t2"))
        await store.delete_thought("t2")
        cursor = await store._db.execute(
            "SELECT COUNT(*) FROM edge WHERE edge_id = ?",
            ("e1",),
        )
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == 0

    async def test_embedding_cascades(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_make_thought("t1"))
        # Insert embedding directly — bypasses the embedding model lock and
        # the provider since the schema is what we are testing.
        await store._db.execute(
            "INSERT INTO embedding "
            "(embedding_id, owner_type, owner_id, model_name, dimension, "
            " vector_blob, created_at) "
            "VALUES (?, 'THOUGHT', ?, 'test', 3, ?, ?)",
            (
                f"emb-{uuid.uuid4().hex}",
                "t1",
                b"\x00\x01\x02",
                datetime.datetime.now(tz=datetime.UTC).isoformat(),
            ),
        )
        await store.delete_thought("t1")
        cursor = await store._db.execute(
            "SELECT COUNT(*) FROM embedding WHERE owner_id = ?",
            ("t1",),
        )
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == 0

    async def test_action_cascades(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_make_thought("t1"))
        await store.create_action(_make_action("a1", "t1"))
        await store.delete_thought("t1")
        cursor = await store._db.execute(
            "SELECT COUNT(*) FROM action WHERE source_thought_id = ?",
            ("t1",),
        )
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == 0


class TestCleanupExpiredStrategies:
    """cleanup_expired delete cascades; archive does NOT."""

    async def test_delete_strategy_cascades_children(
        self,
        delete_store: SqliteEngravaCore,
    ) -> None:
        # Fixed past timestamp; ``now`` below is strictly after.
        past = "2026-01-01T00:00:00+00:00"
        now = "2026-06-01T00:00:00+00:00"
        await delete_store.create_thought(_make_thought("t1", expires_at=past))
        await delete_store.create_thought(_make_thought("t2"))
        await delete_store.create_edge(_make_edge("e1", "t1", "t2"))
        result = await delete_store.cleanup_expired(now=now)
        assert result.expired_count == 1
        cursor = await delete_store._db.execute("SELECT COUNT(*) FROM edge")
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == 0

    async def test_archive_strategy_does_not_cascade(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        past = "2026-01-01T00:00:00+00:00"
        now = "2026-06-01T00:00:00+00:00"
        await store.create_thought(_make_thought("t1", expires_at=past))
        await store.create_thought(_make_thought("t2"))
        await store.create_edge(_make_edge("e1", "t1", "t2"))
        result = await store.cleanup_expired(now=now)
        assert result.expired_count == 1
        # Edge survives — archive is a status transition, not a delete.
        cursor = await store._db.execute(
            "SELECT COUNT(*) FROM edge WHERE edge_id = ?",
            ("e1",),
        )
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == 1


class TestMigrationV11ToV12:
    """v11 -> v12 migration preserves valid data, purges orphans, idempotent."""

    @staticmethod
    async def _bootstrap_v11_schema(db: aiosqlite.Connection) -> None:
        """Create the legacy core-11 schema (no FK clauses)."""
        await db.execute("PRAGMA user_version = 11")
        await db.execute(
            "CREATE TABLE thought ("
            "  thought_id TEXT PRIMARY KEY,"
            "  thought_type TEXT NOT NULL,"
            "  essence TEXT NOT NULL,"
            "  content TEXT NOT NULL,"
            "  priority TEXT NOT NULL,"
            "  lifecycle_status TEXT NOT NULL DEFAULT 'CREATED',"
            "  created_cycle INTEGER NOT NULL DEFAULT 0,"
            "  updated_cycle INTEGER NOT NULL DEFAULT 0,"
            "  source TEXT NOT NULL DEFAULT 'human',"
            "  confidence REAL,"
            "  embedding_ref TEXT,"
            "  source_type TEXT NOT NULL DEFAULT 'EXPERIENCE',"
            "  confirmation_count INTEGER NOT NULL DEFAULT 0,"
            "  consolidated_from TEXT,"
            "  visibility TEXT NOT NULL DEFAULT 'selective',"
            "  access_count INTEGER NOT NULL DEFAULT 0,"
            "  last_accessed_at TEXT,"
            "  created_at TEXT,"
            "  updated_at TEXT,"
            "  expires_at TEXT,"
            "  metadata_json TEXT NOT NULL DEFAULT '{}'"
            ")",
        )
        await db.execute(
            "CREATE TABLE edge ("
            "  edge_id TEXT PRIMARY KEY,"
            "  from_thought_id TEXT NOT NULL,"
            "  to_thought_id TEXT NOT NULL,"
            "  edge_type TEXT NOT NULL,"
            "  weight REAL NOT NULL DEFAULT 0.5,"
            "  created_cycle INTEGER NOT NULL DEFAULT 0,"
            "  source TEXT NOT NULL DEFAULT 'EXPERIENCE',"
            "  decay_multiplier REAL NOT NULL DEFAULT 1.0,"
            "  UNIQUE(from_thought_id, to_thought_id, edge_type)"
            ")",
        )
        await db.execute(
            "CREATE TABLE embedding ("
            "  embedding_id TEXT PRIMARY KEY,"
            "  owner_type TEXT NOT NULL,"
            "  owner_id TEXT NOT NULL,"
            "  model_name TEXT NOT NULL,"
            "  dimension INTEGER NOT NULL,"
            "  vector_blob BLOB NOT NULL,"
            "  created_at TEXT NOT NULL"
            ")",
        )
        await db.execute(
            "CREATE TABLE action ("
            "  action_id TEXT PRIMARY KEY,"
            "  source_thought_id TEXT NOT NULL,"
            "  action_type TEXT NOT NULL,"
            "  intent TEXT NOT NULL,"
            "  status TEXT NOT NULL DEFAULT 'PLANNED',"
            "  verification_status TEXT NOT NULL DEFAULT 'PENDING',"
            "  raw_metrics_json TEXT"
            ")",
        )
        await db.commit()

    async def test_clean_v11_migrates_with_zero_row_loss(
        self,
        tmp_path: Path,
    ) -> None:
        """A v11 DB with no orphans keeps every row after upgrade."""
        db_path = tmp_path / "clean-v11.sqlite"
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            await self._bootstrap_v11_schema(db)
            await db.execute(
                "INSERT INTO thought (thought_id, thought_type, essence, content, priority) "
                "VALUES ('t1', 'OBSERVATION', 'a', 'a', 'P3'), "
                "       ('t2', 'OBSERVATION', 'b', 'b', 'P3')",
            )
            await db.execute(
                "INSERT INTO edge (edge_id, from_thought_id, to_thought_id, edge_type) "
                "VALUES ('e1', 't1', 't2', 'ASSOCIATED')",
            )
            await db.execute(
                "INSERT INTO embedding (embedding_id, owner_type, owner_id, model_name, "
                " dimension, vector_blob, created_at) "
                "VALUES ('emb1', 'THOUGHT', 't1', 'm', 3, ?, '2026-01-01T00:00:00+00:00')",
                (b"\x00\x01\x02",),
            )
            await db.execute(
                "INSERT INTO action (action_id, source_thought_id, action_type, intent) "
                "VALUES ('a1', 't1', 'X', 'do')",
            )
            await db.commit()
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            core = SqliteEngravaCore(db=db, embedding_provider=None, auto_embed=False)
            await core.ensure_schema()
            version_row = await (await db.execute("PRAGMA user_version")).fetchone()
            assert version_row is not None
            assert version_row[0] == 16
            for table, expected in (("edge", 1), ("embedding", 1), ("action", 1), ("thought", 2)):
                row = await (
                    await db.execute(f"SELECT COUNT(*) FROM {table}")  # noqa: S608
                ).fetchone()
                assert row is not None
                assert row[0] == expected, f"{table} row count mismatch"

    async def test_orphan_seeded_v11_purges_orphans_only(
        self,
        tmp_path: Path,
    ) -> None:
        """Orphans are removed; valid rows survive."""
        db_path = tmp_path / "orphan-v11.sqlite"
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            await self._bootstrap_v11_schema(db)
            await db.execute(
                "INSERT INTO thought (thought_id, thought_type, essence, content, priority) "
                "VALUES ('t1', 'OBSERVATION', 'a', 'a', 'P3')",
            )
            # Valid edge plus two orphans (one per endpoint).
            await db.execute(
                "INSERT INTO edge (edge_id, from_thought_id, to_thought_id, edge_type) "
                "VALUES ('e_valid', 't1', 't1', 'ASSOCIATED'), "
                "       ('e_orphan_from', 'ghost', 't1', 'ASSOCIATED'), "
                "       ('e_orphan_to', 't1', 'ghost', 'CONSOLIDATED_FROM')",
            )
            await db.execute(
                "INSERT INTO embedding (embedding_id, owner_type, owner_id, model_name, "
                " dimension, vector_blob, created_at) "
                "VALUES ('emb_valid', 'THOUGHT', 't1', 'm', 3, ?, '2026-01-01T00:00:00+00:00'), "
                "       ('emb_orphan', 'THOUGHT', 'ghost', 'm', 3, ?, '2026-01-01T00:00:00+00:00')",
                (b"\x00\x01\x02", b"\x03\x04\x05"),
            )
            await db.execute(
                "INSERT INTO action (action_id, source_thought_id, action_type, intent) "
                "VALUES ('a_valid', 't1', 'X', 'ok'), "
                "       ('a_orphan', 'ghost', 'X', 'orphan')",
            )
            await db.commit()
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            core = SqliteEngravaCore(db=db, embedding_provider=None, auto_embed=False)
            await core.ensure_schema()
            # Orphans purged, valid rows survive.
            for table, expected in (("edge", 1), ("embedding", 1), ("action", 1)):
                row = await (
                    await db.execute(f"SELECT COUNT(*) FROM {table}")  # noqa: S608
                ).fetchone()
                assert row is not None
                assert row[0] == expected, f"{table} row count mismatch"
            # Standing post-migration invariant: every FK satisfied.
            violations = list(
                await (await db.execute("PRAGMA foreign_key_check")).fetchall(),
            )
            assert violations == [], f"unexpected FK violations: {violations}"

    async def test_migration_is_idempotent(
        self,
        tmp_path: Path,
    ) -> None:
        """Running ensure_schema twice on the same DB is a no-op the second time."""
        db_path = tmp_path / "idempotent-v11.sqlite"
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            await self._bootstrap_v11_schema(db)
            await db.execute(
                "INSERT INTO thought (thought_id, thought_type, essence, content, priority) "
                "VALUES ('t1', 'OBSERVATION', 'a', 'a', 'P3')",
            )
            await db.commit()
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            core = SqliteEngravaCore(db=db, embedding_provider=None, auto_embed=False)
            await core.ensure_schema()
            await core.ensure_schema()  # second pass — must converge without error
            version_row = await (await db.execute("PRAGMA user_version")).fetchone()
            assert version_row is not None
            assert version_row[0] == 16
            # FK declarations must still be exactly 2 on edge, not duplicated.
            rows = list(await (await db.execute("PRAGMA foreign_key_list(edge)")).fetchall())
            assert len(rows) == 2
            # Standing post-migration invariant: every FK satisfied.
            violations = list(
                await (await db.execute("PRAGMA foreign_key_check")).fetchall(),
            )
            assert violations == [], f"unexpected FK violations: {violations}"

    async def test_lowercase_owner_type_orphan_embedding_is_purged(
        self,
        tmp_path: Path,
    ) -> None:
        """Embeddings with non-canonical ``owner_type`` casing still purge cleanly.

        Some legacy / CLI write paths recorded ``owner_type='thought'``
        (lowercase) instead of the canonical ``'THOUGHT'``. The FK does
        not branch on ``owner_type``, so the purge must operate on
        ``owner_id`` alone — otherwise the lowercase orphan survives
        the migration and ``PRAGMA foreign_key_check`` flags a
        violation.
        """
        db_path = tmp_path / "lowercase-orphan.sqlite"
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            await self._bootstrap_v11_schema(db)
            await db.execute(
                "INSERT INTO thought (thought_id, thought_type, essence, content, priority) "
                "VALUES ('t1', 'OBSERVATION', 'a', 'a', 'P3')",
            )
            await db.execute(
                "INSERT INTO embedding (embedding_id, owner_type, owner_id, model_name, "
                " dimension, vector_blob, created_at) "
                "VALUES ('emb_valid', 'THOUGHT', 't1', 'm', 3, ?, '2026-01-01T00:00:00+00:00'),"
                "       ('emb_lower', 'thought', 'ghost', 'm', 3, ?, '2026-01-01T00:00:00+00:00'),"
                "       ('emb_other', 'thought', 't1', 'm', 3, ?, '2026-01-01T00:00:00+00:00')",
                (b"\x00\x01\x02", b"\x03\x04\x05", b"\x06\x07\x08"),
            )
            await db.commit()
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            core = SqliteEngravaCore(db=db, embedding_provider=None, auto_embed=False)
            await core.ensure_schema()
            # Two rows survive: the two pointing at the existing thought
            # regardless of owner_type case. Only the dangling-owner row
            # is purged.
            count_row = await (await db.execute("SELECT COUNT(*) FROM embedding")).fetchone()
            assert count_row is not None
            assert count_row[0] == 2
            ghost_row = await (
                await db.execute(
                    "SELECT 1 FROM embedding WHERE embedding_id = 'emb_lower'",
                )
            ).fetchone()
            assert ghost_row is None
            violations = list(
                await (await db.execute("PRAGMA foreign_key_check")).fetchall(),
            )
            assert violations == [], f"unexpected FK violations: {violations}"

    async def test_full_ladder_path_from_oldest_supported_to_v12(
        self,
        tmp_path: Path,
    ) -> None:
        """The full migration ladder (v3 → … → v12) lands at v12 with no FK violations.

        Exercises the dispatch chain on a database whose ``user_version``
        starts at the oldest supported version (3). Every intermediate
        migration runs in sequence; the v11→v12 step must close any
        implicit transaction opened by the earlier steps before
        disabling foreign keys, otherwise the recreate fails. Valid
        rows survive the entire chain (zero loss) and the final
        database satisfies ``PRAGMA foreign_key_check``.
        """
        db_path = tmp_path / "ladder.sqlite"
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            # The dispatch chain entry below v3 enters the "bootstrap +
            # cascade" branch, which executes the full schema_core.sql
            # (already at v12). Pinning the entry point at v3 covers the
            # multi-step ladder explicitly.
            await db.execute(
                "CREATE TABLE thought ("
                "  thought_id TEXT PRIMARY KEY,"
                "  thought_type TEXT NOT NULL,"
                "  essence TEXT NOT NULL,"
                "  content TEXT NOT NULL,"
                "  priority TEXT NOT NULL"
                ")",
            )
            await db.execute(
                "CREATE TABLE edge ("
                "  edge_id TEXT PRIMARY KEY,"
                "  from_thought_id TEXT NOT NULL,"
                "  to_thought_id TEXT NOT NULL,"
                "  edge_type TEXT NOT NULL,"
                "  weight REAL NOT NULL DEFAULT 0.5,"
                "  created_cycle INTEGER NOT NULL DEFAULT 0,"
                "  source TEXT NOT NULL DEFAULT 'EXPERIENCE',"
                "  decay_multiplier REAL NOT NULL DEFAULT 1.0,"
                "  UNIQUE(from_thought_id, to_thought_id, edge_type)"
                ")",
            )
            await db.execute(
                "CREATE TABLE embedding ("
                "  embedding_id TEXT PRIMARY KEY,"
                "  owner_type TEXT NOT NULL,"
                "  owner_id TEXT NOT NULL,"
                "  model_name TEXT NOT NULL,"
                "  dimension INTEGER NOT NULL,"
                "  vector_blob BLOB NOT NULL,"
                "  created_at TEXT NOT NULL"
                ")",
            )
            await db.execute(
                "CREATE TABLE action ("
                "  action_id TEXT PRIMARY KEY,"
                "  source_thought_id TEXT NOT NULL,"
                "  action_type TEXT NOT NULL,"
                "  intent TEXT NOT NULL,"
                "  status TEXT NOT NULL DEFAULT 'PLANNED',"
                "  verification_status TEXT NOT NULL DEFAULT 'PENDING',"
                "  raw_metrics_json TEXT"
                ")",
            )
            await db.execute(
                "INSERT INTO thought (thought_id, thought_type, essence, content, priority) "
                "VALUES ('t1', 'OBSERVATION', 'a', 'a', 'P3'), "
                "       ('t2', 'OBSERVATION', 'b', 'b', 'P3')",
            )
            await db.execute(
                "INSERT INTO edge (edge_id, from_thought_id, to_thought_id, edge_type) "
                "VALUES ('e1', 't1', 't2', 'ASSOCIATED')",
            )
            await db.execute(
                "INSERT INTO embedding (embedding_id, owner_type, owner_id, model_name, "
                " dimension, vector_blob, created_at) "
                "VALUES ('emb1', 'THOUGHT', 't1', 'm', 3, ?, '2026-01-01T00:00:00+00:00')",
                (b"\x00\x01\x02",),
            )
            await db.execute(
                "INSERT INTO action (action_id, source_thought_id, action_type, intent) "
                "VALUES ('a1', 't1', 'X', 'do')",
            )
            await db.execute("PRAGMA user_version = 3")
            await db.commit()

        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            core = SqliteEngravaCore(db=db, embedding_provider=None, auto_embed=False)
            await core.ensure_schema()
            version_row = await (await db.execute("PRAGMA user_version")).fetchone()
            assert version_row is not None
            assert version_row[0] == 16
            for table, expected in (("edge", 1), ("embedding", 1), ("action", 1), ("thought", 2)):
                row = await (
                    await db.execute(f"SELECT COUNT(*) FROM {table}")  # noqa: S608
                ).fetchone()
                assert row is not None
                assert row[0] == expected, f"ladder path lost rows in {table}"
            violations = list(
                await (await db.execute("PRAGMA foreign_key_check")).fetchall(),
            )
            assert violations == [], f"ladder migration left FK violations: {violations}"

    async def test_partial_migration_recovers_on_retry(
        self,
        tmp_path: Path,
    ) -> None:
        """A DB with FK on edge but not on embedding/action finishes on re-run.

        Simulates a SIGKILL between edge recreation and embedding
        recreation. The next ensure_schema must complete the remaining
        tables; the per-table foreign_key_list probe drives that path.
        """
        db_path = tmp_path / "partial-v11.sqlite"
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            await self._bootstrap_v11_schema(db)
            # Manually pre-apply only the edge recreation, keep the rest at v11.
            await db.execute("PRAGMA foreign_keys=OFF")
            await db.execute("DROP TABLE edge")
            await db.execute(
                "CREATE TABLE edge ("
                "  edge_id TEXT PRIMARY KEY,"
                "  from_thought_id TEXT NOT NULL,"
                "  to_thought_id TEXT NOT NULL,"
                "  edge_type TEXT NOT NULL,"
                "  weight REAL NOT NULL DEFAULT 0.5,"
                "  created_cycle INTEGER NOT NULL DEFAULT 0,"
                "  source TEXT NOT NULL DEFAULT 'EXPERIENCE',"
                "  decay_multiplier REAL NOT NULL DEFAULT 1.0,"
                "  UNIQUE(from_thought_id, to_thought_id, edge_type),"
                "  FOREIGN KEY (from_thought_id) REFERENCES thought(thought_id) "
                "    ON DELETE CASCADE,"
                "  FOREIGN KEY (to_thought_id) REFERENCES thought(thought_id) "
                "    ON DELETE CASCADE"
                ")",
            )
            await db.commit()
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            core = SqliteEngravaCore(db=db, embedding_provider=None, auto_embed=False)
            await core.ensure_schema()
            # Edge declares FK on both endpoints — exactly two rows, no
            # accidental duplication or drop.
            edge_fks = list(
                await (await db.execute("PRAGMA foreign_key_list(edge)")).fetchall(),
            )
            assert len(edge_fks) == 2
            for table in ("embedding", "action"):
                rows = list(
                    await (await db.execute(f"PRAGMA foreign_key_list({table})")).fetchall(),
                )
                assert len(rows) == 1, f"{table} should declare exactly one FK after recovery"
            # Post-migration the database must satisfy every declared FK.
            violations = list(
                await (await db.execute("PRAGMA foreign_key_check")).fetchall(),
            )
            assert violations == [], f"unexpected FK violations: {violations}"
