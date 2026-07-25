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
from unittest.mock import patch

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
from engrava.domain.exceptions import (
    CoreMigrationError,
    DuplicateEdgeError,
    ReferentialIntegrityError,
)
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
        assert row[0] == 20


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

    async def test_duplicate_relationship_raises_domain_error(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        await store.create_thought(_make_thought("t1"))
        await store.create_thought(_make_thought("t2"))
        await store.create_edge(_make_edge("e1", "t1", "t2"))
        with pytest.raises(DuplicateEdgeError) as excinfo:
            await store.create_edge(_make_edge("e2", "t1", "t2"))
        assert excinfo.value.from_thought_id == "t1"
        assert excinfo.value.to_thought_id == "t2"
        assert excinfo.value.edge_type == "ASSOCIATED"

    async def test_duplicate_edge_id_remains_a_distinct_integrity_failure(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        await store.create_thought(_make_thought("t1"))
        await store.create_thought(_make_thought("t2"))
        await store.create_thought(_make_thought("t3"))
        await store.create_edge(_make_edge("e1", "t1", "t2"))
        with pytest.raises(aiosqlite.IntegrityError):
            await store.create_edge(_make_edge("e1", "t1", "t3"))

    async def test_trigger_abort_mentioning_foreign_key_is_not_misclassified(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """A trigger RAISE(ABORT, '...foreign key...') stays a raw IntegrityError.

        Classification is by ``sqlite_errorcode`` (SQLITE_CONSTRAINT_TRIGGER),
        not the message text, so a trigger abort whose message merely mentions
        "foreign key" is neither wrapped as ``ReferentialIntegrityError`` nor
        mistaken for a duplicate — it propagates unchanged, and no row persists.
        """
        await store.create_thought(_make_thought("t1"))
        await store.create_thought(_make_thought("t2"))
        # Both endpoints resolve, so no genuine FK violation occurs; the trigger
        # aborts the insert with a message that would fool substring matching.
        await store._db.execute(
            "CREATE TRIGGER edge_guard BEFORE INSERT ON edge "
            "BEGIN SELECT RAISE(ABORT, 'blocked by policy trigger: foreign key rule'); END"
        )
        with pytest.raises(aiosqlite.IntegrityError) as excinfo:
            await store.create_edge(_make_edge("e1", "t1", "t2"))
        assert not isinstance(excinfo.value, (ReferentialIntegrityError, DuplicateEdgeError))
        assert getattr(excinfo.value, "sqlite_errorname", "") == "SQLITE_CONSTRAINT_TRIGGER"
        cursor = await store._db.execute("SELECT COUNT(*) FROM edge")
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == 0


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
            assert version_row[0] == 20
            for table, expected in (("edge", 1), ("embedding", 1), ("action", 1), ("thought", 2)):
                row = await (
                    await db.execute(f"SELECT COUNT(*) FROM {table}")  # noqa: S608
                ).fetchone()
                assert row is not None
                assert row[0] == expected, f"{table} row count mismatch"
            # The success path also leaves the connection safe: the swap disables
            # foreign keys and opens a savepoint, so pin that both are restored.
            fk_row = await (await db.execute("PRAGMA foreign_keys")).fetchone()
            assert fk_row is not None
            assert fk_row[0] == 1, "foreign-key enforcement left OFF after a successful migration"
            assert not db.in_transaction, "a transaction was left open after a successful migration"

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
            assert version_row[0] == 20
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
            assert version_row[0] == 20
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


class TestBootstrapAtomicity:
    """Fresh bootstrap stamps ``user_version`` only after the full schema applies."""

    async def test_bootstrap_failure_leaves_version_unstamped_and_retryable(
        self,
        tmp_path: Path,
    ) -> None:
        """A mid-bootstrap DDL failure leaves version 0; a retry reaches v20.

        The version stamp is the last statement of ``schema_core.sql``, so a DDL
        failure before it leaves ``user_version = 0`` (never a partial 20). The
        next ``ensure_schema`` then re-runs the idempotent bootstrap rather than
        skipping every migration against an incomplete schema.
        """
        db_path = tmp_path / "bootstrap.sqlite"
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            core = SqliteEngravaCore(db=db, embedding_provider=None, auto_embed=False)

            real_executescript = db.executescript

            async def _failing_executescript(script: str) -> object:
                # Run only the FIRST table (``thought``) then fail — a genuine
                # mid-bootstrap DDL error that leaves a *partial* schema well
                # before the final ``PRAGMA user_version`` stamp.
                head, sep, _tail = script.partition("CREATE TABLE IF NOT EXISTS edge")
                assert sep, "bootstrap script must create the edge table"
                await real_executescript(head)
                msg = "injected bootstrap DDL failure"
                raise aiosqlite.OperationalError(msg)

            with (
                patch.object(db, "executescript", _failing_executescript),
                pytest.raises(aiosqlite.OperationalError, match="injected bootstrap"),
            ):
                await core.ensure_schema()

            # The failed bootstrap left a PARTIAL schema (thought only, no edge)
            # and did NOT durably stamp the version.
            version_row = await (await db.execute("PRAGMA user_version")).fetchone()
            assert version_row is not None
            assert version_row[0] == 0
            present = {
                str(row[0])
                for row in await (
                    await db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
                ).fetchall()
            }
            assert "thought" in present
            assert "edge" not in present, "partial bootstrap should not have reached the edge table"

            # A retry with the real bootstrap converges on a complete head schema.
            await core.ensure_schema()
            version_row = await (await db.execute("PRAGMA user_version")).fetchone()
            assert version_row is not None
            assert version_row[0] == 20
            for table in ("thought", "edge", "embedding", "action", "_metadata", "thought_fts"):
                row = await (
                    await db.execute(
                        "SELECT COUNT(*) FROM sqlite_master WHERE name = ?",
                        (table,),
                    )
                ).fetchone()
                assert row is not None
                assert row[0] == 1, f"{table} missing after retry"
            # The recovered schema is usable end to end.
            await core.create_thought(_make_thought("t-after-retry"))
            assert await core.get_thought("t-after-retry") is not None


class TestV11ToV12PostconditionCatchesVanishedTable:
    """The v11 -> v12 postcondition keys off entry-time existence flags.

    A standalone class (not a subclass of ``TestMigrationV11ToV12``) so pytest
    does not re-collect and re-run the whole v11 migration suite; the legacy
    schema builder is reused via a direct static-method call.
    """

    async def test_child_table_vanished_mid_migration_is_caught(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A child table present at entry but dropped mid-recreate is caught.

        Keying the postcondition off the entry-time ``*_exists`` flags (not a
        fresh existence probe) means a vanished child table fails
        ``_require_table`` rather than being silently skipped and stamped v12
        without its foreign key. ``embedding`` is used because — unlike ``edge``
        — it has no post-recreate index step that would surface the drop first,
        so the failure is caught precisely by the FK postcondition under test.
        """
        db_path = tmp_path / "vanished-v11.sqlite"
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            await TestMigrationV11ToV12._bootstrap_v11_schema(db)
            await db.execute(
                "INSERT INTO thought (thought_id, thought_type, essence, content, priority) "
                "VALUES ('t1', 'OBSERVATION', 'a', 'a', 'P3')",
            )
            await db.commit()
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            core = SqliteEngravaCore(db=db, embedding_provider=None, auto_embed=False)

            async def _drop_embedding_without_recreate() -> None:
                await db.execute("DROP TABLE embedding")

            monkeypatch.setattr(
                core,
                "_recreate_embedding_with_fk",
                _drop_embedding_without_recreate,
            )

            with pytest.raises(CoreMigrationError):
                await core.ensure_schema()

            # The version was never advanced to 12 over the incomplete schema.
            version_row = await (await db.execute("PRAGMA user_version")).fetchone()
            assert version_row is not None
            assert version_row[0] < 12

    async def test_mid_recreate_failure_rolls_back_and_retry_completes(
        self,
        tmp_path: Path,
    ) -> None:
        """A mid-recreate failure rolls the DROP back; a clean retry reaches v20.

        Under sqlite3 legacy isolation (aiosqlite's default) DDL is not enrolled
        in an implicit transaction, so without the SAVEPOINT a failure after the
        recreate's ``DROP`` could leave the child table permanently gone and a
        later attempt (``*_exists`` recomputed False) could stamp v12 over the
        missing table. The savepoint rolls the swap back so the original table
        survives with its row, foreign-key enforcement is restored, and a second
        ``ensure_schema`` (fault removed) converges on a complete v20 schema with
        every foreign key — never a v12 stamp over a missing table.
        """
        db_path = tmp_path / "midfail-v11.sqlite"
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            await TestMigrationV11ToV12._bootstrap_v11_schema(db)
            await db.execute(
                "INSERT INTO thought (thought_id, thought_type, essence, content, priority) "
                "VALUES ('t1', 'OBSERVATION', 'a', 'a', 'P3')",
            )
            await db.execute(
                "INSERT INTO embedding (embedding_id, owner_type, owner_id, model_name, "
                " dimension, vector_blob, created_at) "
                "VALUES ('emb1', 'THOUGHT', 't1', 'm', 3, ?, '2026-01-01T00:00:00+00:00')",
                (b"\x00\x01\x02",),
            )
            await db.commit()
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            core = SqliteEngravaCore(db=db, embedding_provider=None, auto_embed=False)

            async def _drop_then_fail() -> None:
                # Mimic a recreate that drops the old table and then fails before
                # completing the swap (e.g. crash before RENAME).
                await db.execute("DROP TABLE embedding")
                msg = "injected mid-recreate failure"
                raise RuntimeError(msg)

            # First attempt: the recreate fails mid-swap.
            with (
                patch.object(core, "_recreate_embedding_with_fk", _drop_then_fail),
                pytest.raises(RuntimeError, match="injected mid-recreate"),
            ):
                await core.ensure_schema()

            # The savepoint rolled the DROP back: embedding survives with its row,
            # and the version was not advanced to 12.
            present = await (
                await db.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'embedding'"
                )
            ).fetchone()
            assert present is not None, "savepoint should have rolled back the DROP"
            count_row = await (await db.execute("SELECT COUNT(*) FROM embedding")).fetchone()
            assert count_row is not None
            assert count_row[0] == 1, "the original embedding row must survive the rollback"
            version_row = await (await db.execute("PRAGMA user_version")).fetchone()
            assert version_row is not None
            assert version_row[0] < 12
            # Foreign-key enforcement was restored after the failed attempt (the
            # outer finally re-enables it even though the swap failed).
            fk_row = await (await db.execute("PRAGMA foreign_keys")).fetchone()
            assert fk_row is not None
            assert fk_row[0] == 1

            # Second attempt with the fault removed converges on a complete v20.
            await core.ensure_schema()
            version_row = await (await db.execute("PRAGMA user_version")).fetchone()
            assert version_row is not None
            assert version_row[0] == 20
            for table, column in (
                ("edge", "from_thought_id"),
                ("edge", "to_thought_id"),
                ("embedding", "owner_id"),
                ("action", "source_thought_id"),
            ):
                fks = list(
                    await (await db.execute(f"PRAGMA foreign_key_list({table})")).fetchall(),
                )
                assert any(row["from"] == column for row in fks), f"{table}.{column} FK missing"
            violations = list(
                await (await db.execute("PRAGMA foreign_key_check")).fetchall(),
            )
            assert violations == [], f"unexpected FK violations: {violations}"

    @pytest.mark.parametrize(
        ("failing_statement", "fail_body"),
        [
            ("PRAGMA foreign_keys=OFF", False),
            ("SAVEPOINT", False),
            (None, True),
            ("ROLLBACK TO", True),
            ("RELEASE", True),
            # RELEASE also runs on the success path, where it commits the swap.
            ("RELEASE", False),
        ],
        ids=[
            "off-pragma",
            "savepoint",
            "body",
            "rollback-to",
            "release-after-body-failure",
            "release-on-success-path",
        ],
    )
    async def test_control_statement_failure_never_leaks_fk_or_savepoint(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        failing_statement: str | None,
        fail_body: bool,
    ) -> None:
        """No failure path leaves FK off, a transaction open, or a ``*_new`` table.

        ``PRAGMA foreign_keys`` is per-connection and is silently ignored inside
        an open transaction, so a leaked "off" state (or a stuck savepoint that
        keeps a transaction open) would make the rest of the session accept
        orphans and skip ``ON DELETE CASCADE`` with no error at all. This injects
        a failure at every transaction-control point of the swap — disabling FK,
        establishing the savepoint, the recreate body itself, ``ROLLBACK TO`` and
        ``RELEASE`` — and asserts the connection is always left safe.
        """
        db_path = tmp_path / f"leak-{failing_statement or 'body'}-{fail_body}.sqlite"
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            await TestMigrationV11ToV12._bootstrap_v11_schema(db)
            await db.execute(
                "INSERT INTO thought (thought_id, thought_type, essence, content, priority) "
                "VALUES ('t1', 'OBSERVATION', 'a', 'a', 'P3')",
            )
            await db.execute(
                "INSERT INTO embedding (embedding_id, owner_type, owner_id, model_name, "
                " dimension, vector_blob, created_at) "
                "VALUES ('emb1', 'THOUGHT', 't1', 'm', 3, ?, '2026-01-01T00:00:00+00:00')",
                (b"\x00\x01\x02",),
            )
            await db.commit()
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            core = SqliteEngravaCore(db=db, embedding_provider=None, auto_embed=False)

            if fail_body:

                async def _fail_body() -> None:
                    # A recreate that drops the old table and then fails, i.e. the
                    # worst case the savepoint has to undo.
                    await db.execute("DROP TABLE embedding")
                    msg = "injected body failure"
                    raise RuntimeError(msg)

                monkeypatch.setattr(core, "_recreate_embedding_with_fk", _fail_body)

            if failing_statement is not None:
                real_execute = db.execute

                async def _execute(sql: str, *args: object, **kwargs: object) -> object:
                    if failing_statement in sql:
                        msg = f"injected failure at {failing_statement}"
                        raise RuntimeError(msg)
                    return await real_execute(sql, *args, **kwargs)

                monkeypatch.setattr(db, "execute", _execute)

            with pytest.raises(RuntimeError, match="injected"):
                await core.ensure_schema()

            # Restore the real driver before inspecting the connection state.
            monkeypatch.undo()

            # (a) Foreign-key enforcement is back on for the rest of the session.
            fk_row = await (await db.execute("PRAGMA foreign_keys")).fetchone()
            assert fk_row is not None
            assert fk_row[0] == 1, "foreign-key enforcement leaked OFF"

            # (b) No transaction (or savepoint) is left open.
            assert not db.in_transaction, "a transaction/savepoint was left open"

            # (c) No half-swap scratch table survives.
            leftovers = [
                str(row[0])
                for row in await (
                    await db.execute(
                        "SELECT name FROM sqlite_master "
                        "WHERE type = 'table' AND name LIKE '%\\_new' ESCAPE '\\'"
                    )
                ).fetchall()
            ]
            assert leftovers == [], f"leftover swap tables: {leftovers}"

            # (d) The ORIGINAL pre-migration child table and its row survived —
            # the swap is either fully applied or fully undone. The legacy v11
            # table declares no foreign key, so an absent FK proves this is the
            # rolled-back original rather than a half-swap that got committed.
            present = await (
                await db.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'embedding'"
                )
            ).fetchone()
            assert present is not None, "the original embedding table was lost"
            count_row = await (await db.execute("SELECT COUNT(*) FROM embedding")).fetchone()
            assert count_row is not None
            assert count_row[0] == 1, "the original embedding row was lost"
            recreated_fks = list(
                await (await db.execute("PRAGMA foreign_key_list(embedding)")).fetchall(),
            )
            assert recreated_fks == [], "a half-swapped embedding table was committed"


class TestAddColumnIfAbsentExactMatch:
    """``_add_column_if_absent`` tolerates only the exact duplicate-column race."""

    @pytest.fixture
    async def store(self) -> AsyncIterator[SqliteEngravaCore]:
        async with aiosqlite.connect(":memory:") as db:
            db.row_factory = aiosqlite.Row
            core = SqliteEngravaCore(db=db, embedding_provider=None, auto_embed=False)
            await core.ensure_schema()
            yield core

    async def test_tolerates_only_this_columns_duplicate(
        self,
        store: SqliteEngravaCore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The exact "duplicate column name: <column>" race is swallowed."""

        async def _absent(_table: str, _column: str) -> bool:
            return False

        async def _raise_dup_same(_sql: str, *_a: object, **_k: object) -> object:
            msg = "duplicate column name: mycol"
            raise aiosqlite.OperationalError(msg)

        monkeypatch.setattr(store, "_column_exists", _absent)
        monkeypatch.setattr(store._db, "execute", _raise_dup_same)
        # No raise: the exact duplicate signal for this column is the idempotent
        # re-run marker and is tolerated.
        await store._add_column_if_absent("thought", "mycol", "TEXT")

    async def test_duplicate_message_for_other_column_propagates(
        self,
        store: SqliteEngravaCore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A duplicate-column message naming a DIFFERENT column propagates."""

        async def _absent(_table: str, _column: str) -> bool:
            return False

        async def _raise_dup_other(_sql: str, *_a: object, **_k: object) -> object:
            msg = "duplicate column name: othercol"
            raise aiosqlite.OperationalError(msg)

        monkeypatch.setattr(store, "_column_exists", _absent)
        monkeypatch.setattr(store._db, "execute", _raise_dup_other)
        with pytest.raises(aiosqlite.OperationalError, match="othercol"):
            await store._add_column_if_absent("thought", "mycol", "TEXT")

    async def test_duplicate_message_for_prefix_column_propagates(
        self,
        store: SqliteEngravaCore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A duplicate for a column that HAS ours as a prefix still propagates.

        Our column ``mycol`` is a prefix of the error's ``mycol_extra``. The
        match is whole-message exact (not a substring), so the ``mycol_extra``
        duplicate is not mistaken for the ``mycol`` idempotent signal.
        """

        async def _absent(_table: str, _column: str) -> bool:
            return False

        async def _raise_dup_prefix(_sql: str, *_a: object, **_k: object) -> object:
            msg = "duplicate column name: mycol_extra"
            raise aiosqlite.OperationalError(msg)

        monkeypatch.setattr(store, "_column_exists", _absent)
        monkeypatch.setattr(store._db, "execute", _raise_dup_prefix)
        with pytest.raises(aiosqlite.OperationalError, match="mycol_extra"):
            await store._add_column_if_absent("thought", "mycol", "TEXT")

    async def test_unrelated_operational_error_propagates(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """A non-duplicate OperationalError (e.g. no such table) propagates."""
        with pytest.raises(aiosqlite.OperationalError):
            await store._add_column_if_absent("no_such_table", "c", "TEXT")
