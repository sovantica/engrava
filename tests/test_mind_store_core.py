"""Standalone tests for engrava.

Verify that the core package works without any external-consumer dependencies.
These tests use only engrava types and SqliteEngravaCore.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import (
    ActionRecord,
    ActionStatus,
    ActionType,
    CoreThoughtRecord,
    EdgeRecord,
    EdgeType,
    EngravaError,
    InvalidTransitionError,
    KnowledgeSource,
    LifecycleStatus,
    Priority,
    SqliteEngravaCore,
    StaleDataError,
    ThoughtNotFoundError,
    ThoughtType,
    ThoughtVisibility,
    VerificationStatus,
)
from engrava.domain.models.embedding import EmbeddingRecord

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def db(tmp_path: object) -> AsyncIterator[aiosqlite.Connection]:
    """Create an in-memory SQLite database with core schema."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode = WAL")
    await conn.execute("PRAGMA foreign_keys = ON")
    store = SqliteEngravaCore(conn)
    await store.ensure_schema()
    yield conn
    await conn.close()


@pytest.fixture
async def store(db: aiosqlite.Connection) -> SqliteEngravaCore:
    """Create a SqliteEngravaCore backed by the test database."""
    s = SqliteEngravaCore(db)
    await s._probe_fts()
    return s


def _make_thought(
    thought_id: str = "t-001",
    thought_type: ThoughtType = ThoughtType.TASK,
    essence: str = "Test thought",
    content: str = "Test thought content",
    priority: Priority = Priority.P2,
    lifecycle_status: LifecycleStatus = LifecycleStatus.CREATED,
    created_cycle: int = 0,
    updated_cycle: int = 0,
    source: str = "test",
    confidence: float = 0.8,
    source_type: KnowledgeSource = KnowledgeSource.EXPERIENCE,
    visibility: ThoughtVisibility = ThoughtVisibility.SELECTIVE,
) -> CoreThoughtRecord:
    """Helper to construct a core ThoughtRecord."""
    return CoreThoughtRecord(
        thought_id=thought_id,
        thought_type=thought_type,
        essence=essence,
        content=content,
        priority=priority,
        lifecycle_status=lifecycle_status,
        created_cycle=created_cycle,
        updated_cycle=updated_cycle,
        source=source,
        confidence=confidence,
        source_type=source_type,
        visibility=visibility,
    )


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestEnums:
    """Verify core enums are properly exported and functional."""

    def test_thought_type_values(self) -> None:
        assert ThoughtType.TASK.value == "TASK"
        assert ThoughtType.BELIEF.value == "BELIEF"
        assert len(ThoughtType) == 6

    def test_priority_ordering(self) -> None:
        assert Priority.P1 < Priority.P2
        assert Priority.P3 < Priority.P4

    def test_lifecycle_transitions(self) -> None:
        assert LifecycleStatus.CREATED.can_transition_to(LifecycleStatus.ACTIVE)
        assert not LifecycleStatus.ARCHIVED.can_transition_to(LifecycleStatus.CREATED)

    def test_edge_type_values(self) -> None:
        assert EdgeType.ASSOCIATED.value == "ASSOCIATED"
        assert len(EdgeType) == 7

    def test_action_status_transitions(self) -> None:
        assert ActionStatus.PLANNED.can_transition_to(ActionStatus.EXECUTING)
        assert not ActionStatus.CONFIRMED.can_transition_to(ActionStatus.PLANNED)


# ---------------------------------------------------------------------------
# Exception tests
# ---------------------------------------------------------------------------


class TestExceptions:
    """Verify exception hierarchy."""

    def test_engrava_error_is_base(self) -> None:
        assert issubclass(ThoughtNotFoundError, EngravaError)
        assert issubclass(StaleDataError, EngravaError)
        assert issubclass(InvalidTransitionError, EngravaError)

    def test_thought_not_found_error(self) -> None:
        err = ThoughtNotFoundError("t-001")
        assert "t-001" in str(err)

    def test_stale_data_error(self) -> None:
        err = StaleDataError(
            entity_type="ThoughtRecord",
            entity_id="t-001",
            expected_version=5,
        )
        assert "t-001" in str(err)


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestThoughtRecord:
    """Verify CoreThoughtRecord domain model."""

    def test_create_minimal(self) -> None:
        t = _make_thought()
        assert t.thought_id == "t-001"
        assert t.thought_type == ThoughtType.TASK
        assert t.confidence == 0.8

    def test_frozen(self) -> None:
        t = _make_thought()
        with pytest.raises(Exception):
            t.essence = "new"  # type: ignore[misc]

    def test_evolve(self) -> None:
        t = _make_thought()
        t2 = t.evolve(essence="Updated", updated_cycle=1)
        assert t2.essence == "Updated"
        assert t2.updated_cycle == 1
        assert t.essence == "Test thought"  # original unchanged

    def test_is_active(self) -> None:
        t = _make_thought(lifecycle_status=LifecycleStatus.ACTIVE)
        assert t.is_active()

    def test_can_transition_to(self) -> None:
        t = _make_thought(lifecycle_status=LifecycleStatus.CREATED)
        assert t.can_transition_to(LifecycleStatus.ACTIVE)
        assert not t.can_transition_to(LifecycleStatus.ARCHIVED)

    def test_is_archivable(self) -> None:
        t = _make_thought(lifecycle_status=LifecycleStatus.DONE)
        assert t.is_archivable()


class TestEmbeddingRecord:
    """Verify EmbeddingRecord created_at ISO-8601 validation."""

    def _make_embedding(self, created_at: str = "2026-03-11T00:00:00Z") -> EmbeddingRecord:
        import struct as _struct

        return EmbeddingRecord(
            embedding_id="emb-001",
            owner_type="THOUGHT",
            owner_id="t-001",
            model_name="all-MiniLM-L12-v2",
            dimension=3,
            vector_blob=_struct.pack("3f", 0.1, 0.2, 0.3),
            created_at=created_at,
        )

    def test_valid_iso8601_utc(self) -> None:
        emb = self._make_embedding("2026-03-11T00:00:00Z")
        assert emb.created_at == "2026-03-11T00:00:00Z"

    def test_valid_iso8601_offset(self) -> None:
        emb = self._make_embedding("2026-03-11T12:30:00+02:00")
        assert emb.created_at == "2026-03-11T12:30:00+02:00"

    def test_valid_iso8601_no_tz(self) -> None:
        emb = self._make_embedding("2026-03-11T00:00:00")
        assert emb.created_at == "2026-03-11T00:00:00"

    def test_invalid_created_at_rejects_banana(self) -> None:
        with pytest.raises(ValueError, match="ISO-8601"):
            self._make_embedding("banana")

    def test_invalid_created_at_rejects_partial(self) -> None:
        with pytest.raises(ValueError, match="ISO-8601"):
            self._make_embedding("2026-13-45")


# ---------------------------------------------------------------------------
# SqliteEngravaCore CRUD tests
# ---------------------------------------------------------------------------


class TestSqliteEngravaCoreThought:
    """Test thought CRUD on SqliteEngravaCore."""

    async def test_create_and_get(self, store: SqliteEngravaCore) -> None:
        t = _make_thought()
        created = await store.create_thought(t)
        assert created.thought_id == "t-001"

        fetched = await store.get_thought("t-001")
        assert fetched is not None
        assert fetched.thought_id == "t-001"
        assert fetched.essence == "Test thought"

    async def test_get_nonexistent(self, store: SqliteEngravaCore) -> None:
        result = await store.get_thought("nonexistent")
        assert result is None

    async def test_create_duplicate_raises(self, store: SqliteEngravaCore) -> None:
        t = _make_thought()
        await store.create_thought(t)
        with pytest.raises(ValueError, match="already exists"):
            await store.create_thought(t)

    async def test_update_thought(self, store: SqliteEngravaCore) -> None:
        t = _make_thought()
        await store.create_thought(t)
        updated = await store.update_thought("t-001", essence="Updated")
        assert updated.essence == "Updated"

    async def test_update_nonexistent_raises(self, store: SqliteEngravaCore) -> None:
        with pytest.raises(ThoughtNotFoundError):
            await store.update_thought("nonexistent", essence="x")

    async def test_delete_thought(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_make_thought())
        assert await store.delete_thought("t-001")
        assert await store.get_thought("t-001") is None

    async def test_delete_nonexistent(self, store: SqliteEngravaCore) -> None:
        assert not await store.delete_thought("nonexistent")

    async def test_list_thoughts(self, store: SqliteEngravaCore) -> None:
        for i in range(3):
            await store.create_thought(_make_thought(thought_id=f"t-{i:03d}", updated_cycle=i))
        results = await store.list_thoughts(limit=10)
        assert len(results) == 3

    async def test_list_thoughts_filter_priority(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_make_thought("t-001", priority=Priority.P1))
        await store.create_thought(_make_thought("t-002", priority=Priority.P3))
        results = await store.list_thoughts(priority="P1")
        assert len(results) == 1
        assert results[0].priority == Priority.P1


class TestSqliteEngravaCoreEdge:
    """Test edge CRUD on SqliteEngravaCore."""

    async def test_create_and_get_edges(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_make_thought("t-a"))
        await store.create_thought(_make_thought("t-b"))

        edge = EdgeRecord(
            edge_id="e-001",
            from_thought_id="t-a",
            to_thought_id="t-b",
            edge_type=EdgeType.ASSOCIATED,
            weight=0.9,
            created_cycle=0,
            source=KnowledgeSource.EXPERIENCE,
        )
        await store.create_edge(edge)

        edges = await store.get_edges("t-a", direction="OUT")
        assert len(edges) == 1
        assert edges[0].edge_type == EdgeType.ASSOCIATED

    async def test_delete_edge(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_make_thought("t-a"))
        await store.create_thought(_make_thought("t-b"))
        edge = EdgeRecord(
            edge_id="e-001",
            from_thought_id="t-a",
            to_thought_id="t-b",
            edge_type=EdgeType.ASSOCIATED,
            weight=0.9,
            created_cycle=0,
        )
        await store.create_edge(edge)
        assert await store.delete_edge("e-001")


class TestSqliteEngravaCoreEmbedding:
    """Test embedding operations on SqliteEngravaCore."""

    async def test_store_and_get_embedding(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_make_thought("t-001"))
        emb = await store.store_embedding("t-001", [0.1, 0.2, 0.3])
        assert emb.dimension == 3

        fetched = await store.get_embedding("t-001")
        assert fetched is not None
        assert fetched.dimension == 3

    async def test_search_similar(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_make_thought("t-001"))
        await store.create_thought(_make_thought("t-002"))
        await store.store_embedding("t-001", [1.0, 0.0, 0.0])
        await store.store_embedding("t-002", [0.9, 0.1, 0.0])

        results = await store.search_similar([1.0, 0.0, 0.0], top_k=5)
        assert len(results) == 2
        assert results[0][0] == "t-001"

    async def test_store_embedding_reuses_rowid_for_same_embedding_id(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        await store.create_thought(_make_thought("t-001"))

        embedding_id = "emb-stable"
        await store.store_embedding("t-001", [0.1, 0.2, 0.3], embedding_id=embedding_id)
        cursor = await store._db.execute(
            "SELECT rowid FROM embedding WHERE embedding_id = ?",
            (embedding_id,),
        )
        first_row = await cursor.fetchone()
        assert first_row is not None

        await store.store_embedding("t-001", [0.3, 0.2, 0.1], embedding_id=embedding_id)
        cursor = await store._db.execute(
            "SELECT rowid, dimension FROM embedding WHERE embedding_id = ?",
            (embedding_id,),
        )
        second_row = await cursor.fetchone()

        assert second_row is not None
        assert second_row["rowid"] == first_row["rowid"]
        assert second_row["dimension"] == 3


class TestSqliteEngravaCoreAction:
    """Test action CRUD on SqliteEngravaCore."""

    async def test_create_and_get_actions(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_make_thought("t-001"))
        action = ActionRecord(
            action_id="a-001",
            source_thought_id="t-001",
            action_type=ActionType.CLI_OUTPUT,
            intent="Test action",
            status=ActionStatus.PLANNED,
            verification_status=VerificationStatus.PENDING,
        )
        await store.create_action(action)

        actions = await store.get_actions("t-001")
        assert len(actions) == 1
        assert actions[0].action_type == ActionType.CLI_OUTPUT

    async def test_get_actions_empty(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_make_thought("t-001"))
        actions = await store.get_actions("t-001")
        assert actions == []

    async def test_multiple_actions(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_make_thought("t-001"))
        for i in range(3):
            action = ActionRecord(
                action_id=f"a-{i:03d}",
                source_thought_id="t-001",
                action_type=ActionType.CLI_OUTPUT,
                intent=f"Action {i}",
                status=ActionStatus.PLANNED,
                verification_status=VerificationStatus.PENDING,
            )
            await store.create_action(action)
        actions = await store.get_actions("t-001")
        assert len(actions) == 3


class TestActionRecordStateMachine:
    """Verify ActionRecord state machine transitions."""

    def _make_action(
        self,
        status: ActionStatus = ActionStatus.PLANNED,
    ) -> ActionRecord:
        return ActionRecord(
            action_id="a-001",
            source_thought_id="t-001",
            action_type=ActionType.CLI_OUTPUT,
            intent="Test action",
            status=status,
            verification_status=VerificationStatus.PENDING,
        )

    def test_planned_to_executing(self) -> None:
        a = self._make_action(ActionStatus.PLANNED)
        assert a.can_transition_to(ActionStatus.EXECUTING)
        a2 = a.evolve(status=ActionStatus.EXECUTING)
        assert a2.status == ActionStatus.EXECUTING

    def test_planned_to_blocked(self) -> None:
        a = self._make_action(ActionStatus.PLANNED)
        assert a.can_transition_to(ActionStatus.BLOCKED)
        a2 = a.evolve(status=ActionStatus.BLOCKED)
        assert a2.status == ActionStatus.BLOCKED

    def test_executing_to_confirmed(self) -> None:
        a = self._make_action(ActionStatus.EXECUTING)
        assert a.can_transition_to(ActionStatus.CONFIRMED)
        a2 = a.evolve(status=ActionStatus.CONFIRMED)
        assert a2.status == ActionStatus.CONFIRMED

    def test_executing_to_failed(self) -> None:
        a = self._make_action(ActionStatus.EXECUTING)
        assert a.can_transition_to(ActionStatus.FAILED)
        a2 = a.evolve(status=ActionStatus.FAILED)
        assert a2.status == ActionStatus.FAILED

    def test_blocked_to_planned(self) -> None:
        a = self._make_action(ActionStatus.BLOCKED)
        assert a.can_transition_to(ActionStatus.PLANNED)
        a2 = a.evolve(status=ActionStatus.PLANNED)
        assert a2.status == ActionStatus.PLANNED

    def test_confirmed_is_terminal(self) -> None:
        a = self._make_action(ActionStatus.CONFIRMED)
        assert not a.can_transition_to(ActionStatus.PLANNED)
        assert not a.can_transition_to(ActionStatus.EXECUTING)

    def test_failed_is_terminal(self) -> None:
        a = self._make_action(ActionStatus.FAILED)
        assert not a.can_transition_to(ActionStatus.PLANNED)
        assert not a.can_transition_to(ActionStatus.EXECUTING)

    def test_invalid_planned_to_confirmed(self) -> None:
        a = self._make_action(ActionStatus.PLANNED)
        assert not a.can_transition_to(ActionStatus.CONFIRMED)
        with pytest.raises(InvalidTransitionError):
            a.evolve(status=ActionStatus.CONFIRMED)

    def test_invalid_planned_to_failed(self) -> None:
        a = self._make_action(ActionStatus.PLANNED)
        assert not a.can_transition_to(ActionStatus.FAILED)
        with pytest.raises(InvalidTransitionError):
            a.evolve(status=ActionStatus.FAILED)

    def test_invalid_executing_to_planned(self) -> None:
        a = self._make_action(ActionStatus.EXECUTING)
        assert not a.can_transition_to(ActionStatus.PLANNED)
        with pytest.raises(InvalidTransitionError):
            a.evolve(status=ActionStatus.PLANNED)

    def test_full_happy_path(self) -> None:
        a = self._make_action(ActionStatus.PLANNED)
        a = a.evolve(status=ActionStatus.EXECUTING)
        a = a.evolve(status=ActionStatus.CONFIRMED)
        assert a.status == ActionStatus.CONFIRMED

    def test_blocked_retry_path(self) -> None:
        a = self._make_action(ActionStatus.PLANNED)
        a = a.evolve(status=ActionStatus.BLOCKED)
        a = a.evolve(status=ActionStatus.PLANNED)
        a = a.evolve(status=ActionStatus.EXECUTING)
        a = a.evolve(status=ActionStatus.CONFIRMED)
        assert a.status == ActionStatus.CONFIRMED

    def test_evolve_non_status_field(self) -> None:
        a = self._make_action()
        a2 = a.evolve(intent="Updated intent")
        assert a2.intent == "Updated intent"
        assert a.intent == "Test action"

    def test_frozen(self) -> None:
        a = self._make_action()
        with pytest.raises(Exception):
            a.status = ActionStatus.EXECUTING  # type: ignore[misc]


class TestSqliteEngravaCoreSuspendCommit:
    """Test transaction control on SqliteEngravaCore."""

    async def test_suspend_auto_commit(self, store: SqliteEngravaCore) -> None:
        async with store.suspend_auto_commit():
            await store.create_thought(_make_thought("t-001"))
        # After exiting context, auto-commit should resume
        fetched = await store.get_thought("t-001")
        assert fetched is not None


# ---------------------------------------------------------------------------
# FTS5 full-text search
# ---------------------------------------------------------------------------


class TestFTS5Schema:
    """Verify that ensure_schema creates FTS5 tables and sets _fts_available."""

    async def test_thought_has_implicit_rowid(self, db: aiosqlite.Connection) -> None:
        """thought table must have implicit rowid for FTS5 content sync."""
        store = SqliteEngravaCore(db)
        await store.create_thought(_make_thought("t-001"))
        cursor = await db.execute("SELECT rowid FROM thought LIMIT 1")
        row = await cursor.fetchone()
        assert row is not None

    async def test_fts_table_exists_after_ensure_schema(self, db: aiosqlite.Connection) -> None:
        cursor = await db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='thought_fts'"
        )
        assert await cursor.fetchone() is not None

    async def test_fts_available_flag(self, store: SqliteEngravaCore) -> None:
        assert store._fts_available is True

    async def test_fts_unavailable_on_legacy_db(self) -> None:
        """search_fts returns [] when thought_fts does not exist."""
        conn = await aiosqlite.connect(":memory:")
        conn.row_factory = aiosqlite.Row
        # Create only the thought table, no FTS5
        await conn.execute(
            "CREATE TABLE thought ("
            "  thought_id TEXT PRIMARY KEY, thought_type TEXT NOT NULL,"
            "  essence TEXT NOT NULL, content TEXT NOT NULL,"
            "  priority TEXT NOT NULL, lifecycle_status TEXT NOT NULL DEFAULT 'CREATED',"
            "  created_cycle INTEGER NOT NULL DEFAULT 0,"
            "  updated_cycle INTEGER NOT NULL DEFAULT 0,"
            "  source TEXT NOT NULL DEFAULT 'human',"
            "  confidence REAL, embedding_ref TEXT,"
            "  source_type TEXT NOT NULL DEFAULT 'EXPERIENCE',"
            "  confirmation_count INTEGER NOT NULL DEFAULT 0,"
            "  consolidated_from TEXT,"
            "  visibility TEXT NOT NULL DEFAULT 'selective'"
            ")"
        )
        await conn.commit()
        store = SqliteEngravaCore(conn)
        await store._probe_fts()
        assert store._fts_available is False
        results = await store.search_fts("anything")
        assert results == []
        await conn.close()

    async def test_user_version_set_to_current(self, db: aiosqlite.Connection) -> None:
        cursor = await db.execute("PRAGMA user_version")
        row = await cursor.fetchone()
        assert row is not None
        assert int(row[0]) == 13

    async def test_search_fts_lazy_probes_index(self, db: aiosqlite.Connection) -> None:
        """search_fts should work without an explicit _probe_fts call."""
        store = SqliteEngravaCore(db)
        await store.create_thought(
            _make_thought("t-001", essence="Kowalski report", content="Quarterly analysis")
        )
        assert store._fts_available is False
        assert store._fts_probed is False

        results = await store.search_fts("Kowalski")

        assert results == [("t-001", results[0][1])]
        assert results[0][1] > 0.0
        assert store._fts_available is True
        assert store._fts_probed is True

    async def test_ensure_schema_upgrades_v2_fts_tokenizer(self) -> None:
        """ensure_schema should rebuild core v2 FTS for hyphenated prefixes."""
        conn = await aiosqlite.connect(":memory:")
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode = WAL")
        await conn.execute("PRAGMA foreign_keys = ON")

        # Build the current core schema, then roll the FTS5 index back to the
        # legacy v1 state under test: the original virtual table tokenized ``-``
        # as an operator (no ``tokenchars '-_'``), so hyphenated prefix searches
        # such as ``REQ-FUNC*`` could not match. The ``thought`` table itself is
        # left at the current schema so the current ``create_thought`` insert
        # (all core columns) seeds a row cleanly; only the FTS index + triggers
        # are downgraded and ``user_version`` is reset to 2 to mimic a database
        # that predates the hyphen-aware FTS rebuild.
        bootstrap_store = SqliteEngravaCore(conn)
        await bootstrap_store.ensure_schema()
        await conn.execute("DROP TRIGGER IF EXISTS thought_fts_insert")
        await conn.execute("DROP TRIGGER IF EXISTS thought_fts_delete")
        await conn.execute("DROP TRIGGER IF EXISTS thought_fts_update")
        await conn.execute("DROP TABLE IF EXISTS thought_fts")
        await conn.execute(
            "CREATE VIRTUAL TABLE thought_fts USING fts5("
            "  essence, content, content='thought', content_rowid='rowid'"
            ")"
        )
        await conn.execute(
            "CREATE TRIGGER thought_fts_insert AFTER INSERT ON thought BEGIN "
            "  INSERT INTO thought_fts(rowid, essence, content) "
            "  VALUES (new.rowid, new.essence, new.content); "
            "END"
        )
        await conn.execute(
            "CREATE TRIGGER thought_fts_delete AFTER DELETE ON thought BEGIN "
            "  INSERT INTO thought_fts(thought_fts, rowid, essence, content) "
            "  VALUES ('delete', old.rowid, old.essence, old.content); "
            "END"
        )
        await conn.execute(
            "CREATE TRIGGER thought_fts_update AFTER UPDATE OF essence, content ON thought BEGIN "
            "  INSERT INTO thought_fts(thought_fts, rowid, essence, content) "
            "  VALUES ('delete', old.rowid, old.essence, old.content); "
            "  INSERT INTO thought_fts(rowid, essence, content) "
            "  VALUES (new.rowid, new.essence, new.content); "
            "END"
        )
        await conn.execute("PRAGMA user_version = 2")
        await conn.commit()

        legacy_store = SqliteEngravaCore(conn)
        await legacy_store.create_thought(
            _make_thought("t-legacy", essence="General", content="REQ-FUNC-003 compliance")
        )

        upgraded_store = SqliteEngravaCore(conn)
        await upgraded_store.ensure_schema()
        results = await upgraded_store.search_fts("REQ-FUNC*")

        assert len(results) == 1
        assert results[0][0] == "t-legacy"
        await conn.close()


class TestFTS5TriggerSync:
    """Verify that FTS5 index stays in sync with thought table mutations."""

    async def test_insert_syncs_to_fts(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(
            _make_thought("t-001", essence="Kowalski report", content="Quarterly analysis")
        )
        results = await store.search_fts("Kowalski")
        assert len(results) == 1
        assert results[0][0] == "t-001"

    async def test_update_syncs_to_fts(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(
            _make_thought("t-001", essence="Alpha project", content="Initial draft")
        )
        await store.update_thought("t-001", essence="Beta project", updated_cycle=1)
        # Old term should not be found
        old_results = await store.search_fts("Alpha")
        assert len(old_results) == 0
        # New term should be found
        new_results = await store.search_fts("Beta")
        assert len(new_results) == 1
        assert new_results[0][0] == "t-001"

    async def test_delete_syncs_to_fts(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(
            _make_thought("t-001", essence="Temporary note", content="Will be deleted")
        )
        assert len(await store.search_fts("Temporary")) == 1
        await store.delete_thought("t-001")
        assert len(await store.search_fts("Temporary")) == 0


class TestSearchFTS:
    """Verify search_fts results and BM25 ranking."""

    async def test_empty_query_returns_empty(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_make_thought("t-001"))
        assert await store.search_fts("") == []
        assert await store.search_fts("   ") == []

    async def test_no_match_returns_empty(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(
            _make_thought("t-001", essence="Hello world", content="Some content")
        )
        results = await store.search_fts("nonexistent")
        assert results == []

    async def test_basic_keyword_search(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(
            _make_thought("t-001", essence="projekt Alpha", content="Opis projektu Alpha")
        )
        await store.create_thought(
            _make_thought("t-002", essence="raport Beta", content="Opis raportu Beta")
        )
        results = await store.search_fts("Alpha")
        assert len(results) == 1
        assert results[0][0] == "t-001"

    async def test_content_field_search(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(
            _make_thought("t-001", essence="General", content="REQ-FUNC-003 compliance")
        )
        results = await store.search_fts('"REQ-FUNC-003"')
        assert len(results) == 1
        assert results[0][0] == "t-001"

    async def test_prefix_search(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(
            _make_thought("t-001", essence="General", content="REQ-FUNC-001 first")
        )
        await store.create_thought(
            _make_thought("t-002", essence="General", content="REQ-FUNC-002 second")
        )
        results = await store.search_fts("REQ-FUNC*")
        thought_ids = {r[0] for r in results}
        assert "t-001" in thought_ids
        assert "t-002" in thought_ids

    async def test_top_k_limit(self, store: SqliteEngravaCore) -> None:
        for i in range(5):
            await store.create_thought(
                _make_thought(f"t-{i:03d}", essence=f"Topic {i}", content="common keyword")
            )
        results = await store.search_fts("common", top_k=3)
        assert len(results) == 3

    async def test_bm25_ranking_order(self, store: SqliteEngravaCore) -> None:
        """Thought with keyword in both essence and content should rank higher."""
        await store.create_thought(
            _make_thought(
                "t-low",
                essence="General note",
                content="A long text that mentions Kowalski once",
            )
        )
        await store.create_thought(
            _make_thought(
                "t-high",
                essence="Kowalski report",
                content="Kowalski delivered the Kowalski analysis",
            )
        )
        results = await store.search_fts("Kowalski")
        assert len(results) == 2
        # t-high should rank first (more occurrences, appears in essence)
        assert results[0][0] == "t-high"

    async def test_scores_are_positive(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_make_thought("t-001", essence="Test", content="Test content"))
        results = await store.search_fts("Test")
        assert len(results) == 1
        assert results[0][1] > 0.0

    async def test_natural_language_query_with_question_mark(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        await store.create_thought(
            _make_thought(
                "t-001",
                essence="Alpha project summary",
                content="The alpha project shipped successfully.",
            )
        )
        results = await store.search_fts("Alpha shipped successfully?")
        assert len(results) == 1
        assert results[0][0] == "t-001"

    async def test_natural_language_query_with_currency_symbol(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        await store.create_thought(
            _make_thought(
                "t-002",
                essence="Coupon redemption note",
                content="I redeemed a $5 coupon on coffee creamer at ShopRite.",
            )
        )
        results = await store.search_fts("coffee creamer $5?")
        assert len(results) == 1
        assert results[0][0] == "t-002"
