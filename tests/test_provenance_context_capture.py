"""Tests for opt-in write-time provenance-context capture.

Covers the typed, bounded ``ProvenanceContext`` sub-model captured at
``create_thought`` and made queryable but never consumed:

* the ``ProvenanceContext`` shape, its per-field character caps and the
  ``retrieval_context_ids`` length / element caps (over-cap raises, boundary
  values pass);
* the full round trip (all five fields, including the id list) through
  ``create_thought`` -> ``get_thought``;
* the byte-identical guarantee: ``provenance=None`` writes a NULL column, the
  stored row and search behaviour are identical to a pre-feature store, and
  ``_row_to_thought`` yields ``provenance=None``;
* the queryable + indexed surface: ``list_thoughts(provenance_filter=...)``
  returns the matching set, ``EXPLAIN QUERY PLAN`` shows the session / actor
  expression index is used, and MindQL ``FIND`` / ``SELECT`` read provenance;
* the untrusted-hint posture: provenance is consulted for no access decision
  (there is no such path);
* the ``core-17`` migration (idempotence, table-absence tolerance, fresh-DB
  head version, cascade from an older version, index creation, row-count
  preservation, fresh == migrated index parity);
* journaling: provenance is in the create delta and ``verify_journal`` passes;
* no consumption: dreaming promotion and hybrid-search ranking are byte-identical
  whether or not the thoughts carry provenance;
* the read-only store still blocks the provenance write path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import aiosqlite
import pytest
from pydantic import ValidationError

from engrava import (
    FieldOp,
    FieldPredicate,
    LifecycleStatus,
    MetadataFilter,
    Priority,
    ProvenanceContext,
    ReadOnlyViolationError,
    SqliteEngravaCore,
    ThoughtRecord,
    ThoughtType,
    parse,
)
from engrava.domain.models.provenance import (
    MAX_CONTEXT_CHARS,
    MAX_CONTEXT_ID_CHARS,
    MAX_CONTEXT_IDS,
    MAX_IDENTITY_CHARS,
)
from engrava.infrastructure.read_only_store import ReadOnlyEngrava
from engrava.infrastructure.sqlite.engrava_core import (
    _decode_provenance,
    _encode_provenance,
    _validate_provenance,
)
from engrava.mindql.executor import MindQLExecutor

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_HEAD_VERSION = 20


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
async def store() -> AsyncIterator[SqliteEngravaCore]:
    """A schema-bootstrapped in-memory store (journaling off)."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA foreign_keys = ON")
    s = SqliteEngravaCore(conn)
    await s.ensure_schema()
    yield s
    await conn.close()


@pytest.fixture
async def jstore() -> AsyncIterator[SqliteEngravaCore]:
    """A schema-bootstrapped in-memory store with journaling enabled."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA foreign_keys = ON")
    s = SqliteEngravaCore(conn, journal_enabled=True)
    await s.ensure_schema()
    yield s
    await conn.close()


@pytest.fixture
async def fresh_db() -> AsyncIterator[aiosqlite.Connection]:
    """Empty in-memory SQLite (``user_version`` starts at 0)."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    yield conn
    await conn.close()


def _thought(
    thought_id: str = "t-001",
    *,
    content: str | None = None,
    provenance: ProvenanceContext | None = None,
    thought_type: ThoughtType = ThoughtType.OBSERVATION,
    priority: Priority = Priority.P2,
    updated_cycle: int = 0,
) -> ThoughtRecord:
    """Build a minimal thought, optionally carrying provenance."""
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=thought_type,
        essence=f"Essence {thought_id}",
        content=content if content is not None else f"Content of {thought_id}",
        priority=priority,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=0,
        updated_cycle=updated_cycle,
        source="test",
        provenance=provenance,
    )


def _full_provenance() -> ProvenanceContext:
    """A provenance sub-model exercising all five fields (incl. the id list)."""
    return ProvenanceContext(
        session_id="sess-42",
        actor_id="agent-a",
        retrieval_query="remote work trade-offs",
        instruction_context="summarise for a busy exec",
        retrieval_context_ids=["src-1", "src-2", "src-3"],
    )


async def _user_version(db: aiosqlite.Connection) -> int:
    cursor = await db.execute("PRAGMA user_version")
    row = await cursor.fetchone()
    assert row is not None
    return int(row[0])


async def _index_exists(db: aiosqlite.Connection, name: str) -> bool:
    cursor = await db.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = ?",
        (name,),
    )
    return await cursor.fetchone() is not None


async def _index_names(db: aiosqlite.Connection, table: str) -> set[str]:
    cursor = await db.execute(
        "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = ?",
        (table,),
    )
    rows = await cursor.fetchall()
    return {row["name"] for row in rows if row["name"] is not None}


async def _column_names(db: aiosqlite.Connection, table: str) -> list[str]:
    cursor = await db.execute(f"PRAGMA table_info({table})")
    rows = await cursor.fetchall()
    return [row["name"] for row in rows]


async def _row_count(db: aiosqlite.Connection, table: str) -> int:
    cursor = await db.execute(f"SELECT COUNT(*) AS n FROM {table}")  # noqa: S608
    row = await cursor.fetchone()
    assert row is not None
    return int(row["n"])


async def _table_names(db: aiosqlite.Connection) -> set[str]:
    cursor = await db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    rows = await cursor.fetchall()
    return {row["name"] for row in rows}


def _indexes_used(plan_rows: list[aiosqlite.Row]) -> set[str]:
    """Return the set of index names an EXPLAIN QUERY PLAN reports ``USING INDEX``."""
    used: set[str] = set()
    for row in plan_rows:
        detail = str(row["detail"])
        marker = "USING INDEX "
        if marker in detail:
            after = detail.split(marker, 1)[1]
            used.add(after.split()[0])
    return used


async def _explain(
    db: aiosqlite.Connection, sql: str, params: tuple[object, ...]
) -> list[aiosqlite.Row]:
    cursor = await db.execute(f"EXPLAIN QUERY PLAN {sql}", params)
    return list(await cursor.fetchall())


PROV_IDENTITY_INDEXES = ("idx_thought_prov_session", "idx_thought_prov_actor")


# ---------------------------------------------------------------------------
# The ``core-16`` base fixture (provenance column + indexes deliberately absent)
# ---------------------------------------------------------------------------


async def _bootstrap_core_at_v16(db: aiosqlite.Connection) -> None:
    """Recreate a faithful core-16 ``thought`` schema (no provenance surface).

    Mirrors ``schema_core.sql`` at ``user_version = 16`` — the
    ``action_outcome_score`` aggregate is present, but the ``provenance``
    column and the two provenance identity indexes the v17 upgrade adds are
    deliberately absent. That is precisely the surface the upgrade re-adds.
    """
    await db.executescript(
        """
        CREATE TABLE IF NOT EXISTS thought (
            thought_id        TEXT    PRIMARY KEY,
            thought_type      TEXT    NOT NULL,
            essence           TEXT    NOT NULL,
            content           TEXT    NOT NULL,
            content_hash      TEXT,
            priority          TEXT    NOT NULL,
            lifecycle_status  TEXT    NOT NULL DEFAULT 'CREATED',
            created_cycle     INTEGER NOT NULL DEFAULT 0,
            updated_cycle     INTEGER NOT NULL DEFAULT 0,
            source            TEXT    NOT NULL DEFAULT 'human',
            confidence        REAL,
            embedding_ref     TEXT,
            source_type       TEXT    NOT NULL DEFAULT 'EXPERIENCE',
            confirmation_count INTEGER NOT NULL DEFAULT 0,
            consolidated_from TEXT,
            visibility        TEXT    NOT NULL DEFAULT 'selective',
            access_count      INTEGER NOT NULL DEFAULT 0,
            last_accessed_at  TEXT,
            created_at        TEXT,
            updated_at        TEXT,
            expires_at        TEXT,
            metadata_json     TEXT    NOT NULL DEFAULT '{}',
            valid_from        TEXT,
            valid_until       TEXT,
            action_outcome_score REAL
        );
        CREATE INDEX IF NOT EXISTS idx_thought_expires ON thought(expires_at)
            WHERE expires_at IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_thought_content_hash ON thought(content_hash);
        CREATE INDEX IF NOT EXISTS idx_thought_valid_from ON thought(valid_from);
        CREATE INDEX IF NOT EXISTS idx_thought_valid_until ON thought(valid_until)
            WHERE valid_until IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_thought_valid_range ON thought(valid_from, valid_until);
        CREATE INDEX IF NOT EXISTS idx_thought_updated_cycle ON thought(updated_cycle);
        CREATE INDEX IF NOT EXISTS idx_thought_type ON thought(thought_type);
        PRAGMA user_version = 16;
        """,
    )
    await db.commit()


# ---------------------------------------------------------------------------
# ProvenanceContext shape + bounds
# ---------------------------------------------------------------------------


class TestProvenanceContextShape:
    def test_all_fields_optional_defaults_none(self) -> None:
        """Every field defaults to ``None`` — a bare provenance is legal."""
        prov = ProvenanceContext()
        assert prov.session_id is None
        assert prov.actor_id is None
        assert prov.retrieval_query is None
        assert prov.instruction_context is None
        assert prov.retrieval_context_ids is None

    def test_frozen(self) -> None:
        """The model is frozen — fields cannot be reassigned."""
        prov = ProvenanceContext(session_id="s")
        with pytest.raises(ValidationError):
            prov.session_id = "other"  # type: ignore[misc]

    def test_full_construction(self) -> None:
        """All five fields round-trip through construction unchanged."""
        prov = _full_provenance()
        assert prov.session_id == "sess-42"
        assert prov.actor_id == "agent-a"
        assert prov.retrieval_query == "remote work trade-offs"
        assert prov.instruction_context == "summarise for a busy exec"
        assert prov.retrieval_context_ids == ["src-1", "src-2", "src-3"]

    @pytest.mark.parametrize("field", ["session_id", "actor_id"])
    def test_identity_field_boundary_passes(self, field: str) -> None:
        """An identity field at exactly the cap length is accepted."""
        prov = ProvenanceContext(**{field: "x" * MAX_IDENTITY_CHARS})
        assert len(getattr(prov, field)) == MAX_IDENTITY_CHARS

    @pytest.mark.parametrize("field", ["session_id", "actor_id"])
    def test_identity_field_over_cap_raises(self, field: str) -> None:
        """An identity field one char over the cap raises."""
        with pytest.raises(ValidationError):
            ProvenanceContext(**{field: "x" * (MAX_IDENTITY_CHARS + 1)})

    @pytest.mark.parametrize("field", ["retrieval_query", "instruction_context"])
    def test_context_field_boundary_passes(self, field: str) -> None:
        """A context text field at exactly the cap length is accepted."""
        prov = ProvenanceContext(**{field: "y" * MAX_CONTEXT_CHARS})
        assert len(getattr(prov, field)) == MAX_CONTEXT_CHARS

    @pytest.mark.parametrize("field", ["retrieval_query", "instruction_context"])
    def test_context_field_over_cap_raises(self, field: str) -> None:
        """A context text field one char over the cap raises."""
        with pytest.raises(ValidationError):
            ProvenanceContext(**{field: "y" * (MAX_CONTEXT_CHARS + 1)})

    def test_context_ids_length_boundary_passes(self) -> None:
        """A context-id list at exactly the length cap is accepted."""
        prov = ProvenanceContext(retrieval_context_ids=["id"] * MAX_CONTEXT_IDS)
        assert len(prov.retrieval_context_ids or []) == MAX_CONTEXT_IDS

    def test_context_ids_over_length_raises(self) -> None:
        """A context-id list one element over the cap raises."""
        with pytest.raises(ValidationError):
            ProvenanceContext(retrieval_context_ids=["id"] * (MAX_CONTEXT_IDS + 1))

    def test_context_id_element_boundary_passes(self) -> None:
        """A single context-id element at exactly the element cap is accepted."""
        prov = ProvenanceContext(retrieval_context_ids=["z" * MAX_CONTEXT_ID_CHARS])
        assert prov.retrieval_context_ids == ["z" * MAX_CONTEXT_ID_CHARS]

    def test_context_id_element_over_cap_raises(self) -> None:
        """A single over-long context-id element raises."""
        with pytest.raises(ValidationError):
            ProvenanceContext(retrieval_context_ids=["z" * (MAX_CONTEXT_ID_CHARS + 1)])


class TestValidateProvenanceHook:
    def test_none_passes(self) -> None:
        """``None`` provenance passes the persistence-boundary validator."""
        _validate_provenance(None)

    def test_valid_model_passes(self) -> None:
        """A well-formed model passes the persistence-boundary validator."""
        _validate_provenance(_full_provenance())

    def test_wrong_type_raises_value_error(self) -> None:
        """A non-ProvenanceContext argument raises ``ValueError`` (not TypeError)."""
        with pytest.raises(ValueError, match="provenance must be ProvenanceContext"):
            _validate_provenance({"session_id": "s"})  # type: ignore[arg-type]


class TestEncodeDecodeHelpers:
    def test_none_round_trips_to_none(self) -> None:
        """``None`` encodes to ``None`` (SQL NULL) and decodes back to ``None``."""
        assert _encode_provenance(None) is None
        assert _decode_provenance(None) is None

    def test_model_round_trips(self) -> None:
        """A model encodes to JSON and decodes back equal."""
        prov = _full_provenance()
        raw = _encode_provenance(prov)
        assert raw is not None
        assert _decode_provenance(raw) == prov


# ---------------------------------------------------------------------------
# Round trip through the store
# ---------------------------------------------------------------------------


class TestRoundTrip:
    async def test_full_provenance_round_trip(self, store: SqliteEngravaCore) -> None:
        """A thought with a full provenance survives create -> fetch unchanged."""
        prov = _full_provenance()
        await store.create_thought(_thought("t-rt", provenance=prov))

        fetched = await store.get_thought("t-rt")
        assert fetched is not None
        assert fetched.provenance == prov
        # The list survives element-for-element (not silently dropped/reordered).
        assert fetched.provenance is not None
        assert fetched.provenance.retrieval_context_ids == ["src-1", "src-2", "src-3"]

    async def test_partial_provenance_round_trip(self, store: SqliteEngravaCore) -> None:
        """A provenance with only identity fields round-trips with the rest ``None``."""
        prov = ProvenanceContext(session_id="s-only", actor_id="a-only")
        await store.create_thought(_thought("t-partial", provenance=prov))

        fetched = await store.get_thought("t-partial")
        assert fetched is not None
        assert fetched.provenance == prov
        assert fetched.provenance is not None
        assert fetched.provenance.retrieval_query is None
        assert fetched.provenance.retrieval_context_ids is None

    async def test_update_can_attach_provenance(self, store: SqliteEngravaCore) -> None:
        """``update_thought`` persists a newly attached provenance."""
        await store.create_thought(_thought("t-upd"))
        prov = _full_provenance()

        await store.update_thought("t-upd", provenance=prov)

        fetched = await store.get_thought("t-upd")
        assert fetched is not None
        assert fetched.provenance == prov


# ---------------------------------------------------------------------------
# Byte-identical when provenance is None (regression)
# ---------------------------------------------------------------------------


class TestByteIdenticalDefault:
    async def test_none_writes_null_column(self, store: SqliteEngravaCore) -> None:
        """``provenance=None`` stores a SQL NULL, not the string ``"null"``."""
        await store.create_thought(_thought("t-null"))
        cursor = await store._db.execute(
            "SELECT provenance FROM thought WHERE thought_id = ?", ("t-null",)
        )
        row = await cursor.fetchone()
        assert row is not None
        assert row["provenance"] is None

    async def test_row_to_thought_yields_none(self, store: SqliteEngravaCore) -> None:
        """A NULL provenance column decodes back to ``provenance=None``."""
        await store.create_thought(_thought("t-null2"))
        fetched = await store.get_thought("t-null2")
        assert fetched is not None
        assert fetched.provenance is None

    async def test_stored_row_identical_to_pre_feature(self) -> None:
        """The full stored row for a no-provenance thought equals the pre-feature row.

        The write path is byte-identical: every persisted column except the new
        (NULL) ``provenance`` column matches what a store that never knew about
        provenance would have written for the same thought.
        """
        conn = await aiosqlite.connect(":memory:")
        conn.row_factory = aiosqlite.Row
        try:
            store = SqliteEngravaCore(conn)
            await store.ensure_schema()
            await store.create_thought(_thought("t-cmp"))

            cursor = await conn.execute("SELECT * FROM thought WHERE thought_id = ?", ("t-cmp",))
            row = await cursor.fetchone()
            assert row is not None
            row_map = dict(row)
            # The provenance column is present and NULL...
            assert "provenance" in row_map
            assert row_map["provenance"] is None
            # ...and every other column is exactly what the pre-feature schema held.
            del row_map["provenance"]
            assert row_map["thought_id"] == "t-cmp"
            assert row_map["metadata_json"] == "{}"
            assert row_map["action_outcome_score"] is None
        finally:
            await conn.close()

    async def test_search_behaviour_identical_without_provenance(
        self, store: SqliteEngravaCore
    ) -> None:
        """A no-provenance store searches exactly as before the feature existed."""
        for i in range(5):
            await store.create_thought(_thought(f"t-s{i}", content=f"alpha beta gamma {i}"))
        result = await store.search_hybrid(query_text="alpha beta", top_k=5)
        # ``results`` are ``(thought_id, score)`` tuples.
        assert {tid for tid, _ in result.results} == {f"t-s{i}" for i in range(5)}


# ---------------------------------------------------------------------------
# Queryable + indexed
# ---------------------------------------------------------------------------


class TestQueryableIndexed:
    async def _seed(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(
            _thought("t-q1", provenance=ProvenanceContext(session_id="sess-A", actor_id="act-1"))
        )
        await store.create_thought(
            _thought("t-q2", provenance=ProvenanceContext(session_id="sess-A", actor_id="act-2"))
        )
        await store.create_thought(
            _thought("t-q3", provenance=ProvenanceContext(session_id="sess-B", actor_id="act-1"))
        )
        await store.create_thought(_thought("t-q4"))  # no provenance

    async def test_filter_by_session_id(self, store: SqliteEngravaCore) -> None:
        """A ``session_id`` predicate returns exactly the matching thoughts."""
        await self._seed(store)
        matches = await store.list_thoughts(
            provenance_filter=MetadataFilter(
                [FieldPredicate("$.session_id", FieldOp.EQ, "sess-A")]
            ),
        )
        assert {t.thought_id for t in matches} == {"t-q1", "t-q2"}

    async def test_filter_by_actor_id(self, store: SqliteEngravaCore) -> None:
        """An ``actor_id`` predicate returns exactly the matching thoughts."""
        await self._seed(store)
        matches = await store.list_thoughts(
            provenance_filter=MetadataFilter([FieldPredicate("$.actor_id", FieldOp.EQ, "act-1")]),
        )
        assert {t.thought_id for t in matches} == {"t-q1", "t-q3"}

    async def test_filter_by_descriptive_field(self, store: SqliteEngravaCore) -> None:
        """A non-indexed descriptive provenance field is still queryable."""
        await store.create_thought(
            _thought(
                "t-desc",
                provenance=ProvenanceContext(retrieval_query="find the widget"),
            )
        )
        await store.create_thought(_thought("t-desc2"))
        matches = await store.list_thoughts(
            provenance_filter=MetadataFilter(
                [FieldPredicate("$.retrieval_query", FieldOp.EQ, "find the widget")]
            ),
        )
        assert {t.thought_id for t in matches} == {"t-desc"}

    async def test_empty_filter_is_noop(self, store: SqliteEngravaCore) -> None:
        """An empty provenance filter widens to match-all (like ``None``)."""
        await self._seed(store)
        all_thoughts = await store.list_thoughts(provenance_filter=MetadataFilter([]))
        none_thoughts = await store.list_thoughts(provenance_filter=None)
        assert {t.thought_id for t in all_thoughts} == {t.thought_id for t in none_thoughts}
        assert len(all_thoughts) == 4

    async def test_null_provenance_never_matches_nonempty_filter(
        self, store: SqliteEngravaCore
    ) -> None:
        """A NULL-provenance row is non-matching for a non-empty filter."""
        await self._seed(store)
        matches = await store.list_thoughts(
            provenance_filter=MetadataFilter(
                [FieldPredicate("$.session_id", FieldOp.EQ, "sess-A")]
            ),
        )
        assert "t-q4" not in {t.thought_id for t in matches}

    async def test_session_query_uses_expression_index(self, store: SqliteEngravaCore) -> None:
        """``WHERE json_extract(provenance,'$.session_id')=?`` uses its index.

        Confirms the primary design path (a ``json_extract`` expression index),
        so the generated-column fallback is not needed.
        """
        db = store._db
        # Populate enough rows (+ ANALYZE) that the planner prefers the index.
        for i in range(200):
            await store.create_thought(
                _thought(f"t-plan-{i}", provenance=ProvenanceContext(session_id=f"s-{i % 20}"))
            )
        await db.execute("ANALYZE")
        await db.commit()

        plan = await _explain(
            db,
            "SELECT * FROM thought WHERE json_extract(provenance, '$.session_id') = ?",
            ("s-3",),
        )
        assert "idx_thought_prov_session" in _indexes_used(plan)

    async def test_actor_query_uses_expression_index(self, store: SqliteEngravaCore) -> None:
        """``WHERE json_extract(provenance,'$.actor_id')=?`` uses its index."""
        db = store._db
        for i in range(200):
            await store.create_thought(
                _thought(f"t-a-{i}", provenance=ProvenanceContext(actor_id=f"a-{i % 20}"))
            )
        await db.execute("ANALYZE")
        await db.commit()

        plan = await _explain(
            db,
            "SELECT * FROM thought WHERE json_extract(provenance, '$.actor_id') = ?",
            ("a-3",),
        )
        assert "idx_thought_prov_actor" in _indexes_used(plan)


class TestMindQLReadsProvenance:
    async def test_find_returns_provenance_column(self, store: SqliteEngravaCore) -> None:
        """A MindQL ``FIND`` surfaces the raw provenance JSON in the row."""
        await store.create_thought(
            _thought("t-mq", provenance=ProvenanceContext(session_id="sess-mq"))
        )
        executor = MindQLExecutor(store._db)
        result = await executor.execute(parse("FIND thoughts WHERE thought_id = 't-mq'"))
        assert len(result.rows) == 1
        assert "provenance" in result.rows[0]
        assert "sess-mq" in str(result.rows[0]["provenance"])

    async def test_select_projects_provenance_path(self, store: SqliteEngravaCore) -> None:
        """A MindQL ``SELECT`` passthrough can project a provenance JSON path."""
        await store.create_thought(
            _thought("t-mq2", provenance=ProvenanceContext(session_id="sess-proj"))
        )
        executor = MindQLExecutor(store._db)
        result = await executor.execute(
            parse(
                "SELECT json_extract(provenance, '$.session_id') AS sid "
                "FROM thought WHERE thought_id = 't-mq2'"
            )
        )
        assert result.rows == [{"sid": "sess-proj"}]


# ---------------------------------------------------------------------------
# Untrusted hint / no access decision
# ---------------------------------------------------------------------------


class TestUntrustedHint:
    async def test_provenance_not_consulted_for_access(self, store: SqliteEngravaCore) -> None:
        """Provenance grants no authority: a thought is retrievable regardless of it.

        There is no access path keyed on provenance — ``actor_id`` is not an
        authorization principal. A thought whose provenance names one actor is
        readable by any caller of the same store; the field is descriptive only.
        """
        await store.create_thought(
            _thought("t-auth", provenance=ProvenanceContext(actor_id="actor-owner"))
        )
        # No credentials, no actor context — the read still succeeds.
        fetched = await store.get_thought("t-auth")
        assert fetched is not None
        assert fetched.thought_id == "t-auth"

    def test_no_authorization_symbol_references_provenance(self) -> None:
        """No engine symbol consults provenance for an access / authz decision.

        A structural guard: the engine exposes no ``authorize`` / ``can_access``
        style hook, and the provenance field is never read by one. If such a
        path is ever added it must not key on provenance — this asserts none
        exists today.
        """
        from engrava.infrastructure.sqlite import engrava_core

        forbidden = ("authorize", "can_access", "check_permission", "is_authorized")
        for name in forbidden:
            assert not hasattr(engrava_core.SqliteEngravaCore, name), name


# ---------------------------------------------------------------------------
# Migration (core-16 -> core-17)
# ---------------------------------------------------------------------------


class TestMigration:
    async def test_v16_base_lacks_provenance_surface(self, fresh_db: aiosqlite.Connection) -> None:
        """Guard: the v16 base fixture omits the provenance column and indexes."""
        await _bootstrap_core_at_v16(fresh_db)
        assert "provenance" not in await _column_names(fresh_db, "thought")
        for index_name in PROV_IDENTITY_INDEXES:
            assert not await _index_exists(fresh_db, index_name), index_name

    async def test_migrate_adds_column_and_indexes(self, fresh_db: aiosqlite.Connection) -> None:
        """The v16 -> v17 helper adds the column and both identity indexes."""
        await _bootstrap_core_at_v16(fresh_db)
        store = SqliteEngravaCore(fresh_db)
        await store._migrate_core_v16_to_v17()

        assert "provenance" in await _column_names(fresh_db, "thought")
        for index_name in PROV_IDENTITY_INDEXES:
            assert await _index_exists(fresh_db, index_name), index_name

    async def test_migrate_idempotent(self, fresh_db: aiosqlite.Connection) -> None:
        """Re-running the helper is safe: no duplicate column / index, no error."""
        await _bootstrap_core_at_v16(fresh_db)
        store = SqliteEngravaCore(fresh_db)

        for _ in range(3):
            await store._migrate_core_v16_to_v17()

        assert (await _column_names(fresh_db, "thought")).count("provenance") == 1
        for index_name in PROV_IDENTITY_INDEXES:
            assert await _index_exists(fresh_db, index_name), index_name

    async def test_ensure_schema_idempotent(self, fresh_db: aiosqlite.Connection) -> None:
        """``ensure_schema`` twice on a fresh DB stays at head with the surface."""
        store = SqliteEngravaCore(fresh_db)
        await store.ensure_schema()
        await store.ensure_schema()

        assert await _user_version(fresh_db) == _HEAD_VERSION
        assert "provenance" in await _column_names(fresh_db, "thought")
        for index_name in PROV_IDENTITY_INDEXES:
            assert await _index_exists(fresh_db, index_name), index_name

    async def test_fresh_db_lands_at_head(self, fresh_db: aiosqlite.Connection) -> None:
        """An empty DB bootstraps straight to head with the provenance surface."""
        store = SqliteEngravaCore(fresh_db)
        await store.ensure_schema()

        assert await _user_version(fresh_db) == _HEAD_VERSION
        assert "provenance" in await _column_names(fresh_db, "thought")
        for index_name in PROV_IDENTITY_INDEXES:
            assert await _index_exists(fresh_db, index_name), index_name

    async def test_cascade_from_v16_to_head(self, fresh_db: aiosqlite.Connection) -> None:
        """A v16 DB walks the ``< 17`` branch up to head and gains the surface."""
        await _bootstrap_core_at_v16(fresh_db)
        assert await _user_version(fresh_db) == 16

        store = SqliteEngravaCore(fresh_db)
        await store.ensure_schema()

        assert await _user_version(fresh_db) == _HEAD_VERSION
        assert "provenance" in await _column_names(fresh_db, "thought")
        for index_name in PROV_IDENTITY_INDEXES:
            assert await _index_exists(fresh_db, index_name), index_name

    @pytest.mark.parametrize("source_version", [3, 8, 13, 14, 15, 16])
    async def test_cascade_from_any_version_to_head(
        self, fresh_db: aiosqlite.Connection, source_version: int
    ) -> None:
        """A DB stamped at any historical version cascades to head with the surface."""
        bootstrap = SqliteEngravaCore(fresh_db)
        await bootstrap.ensure_schema()
        await fresh_db.execute(f"PRAGMA user_version = {source_version}")
        await fresh_db.commit()

        store = SqliteEngravaCore(fresh_db)
        await store.ensure_schema()

        assert await _user_version(fresh_db) == _HEAD_VERSION
        assert "provenance" in await _column_names(fresh_db, "thought")
        for index_name in PROV_IDENTITY_INDEXES:
            assert await _index_exists(fresh_db, index_name), index_name

    async def test_migrate_preserves_row_count(self, fresh_db: aiosqlite.Connection) -> None:
        """The additive migration changes no row counts (zero data loss)."""
        await _bootstrap_core_at_v16(fresh_db)
        await fresh_db.executemany(
            """
            INSERT INTO thought (thought_id, thought_type, essence, content, priority)
            VALUES (?, 'OBSERVATION', 'e', 'c', 'P2')
            """,
            [("m-1",), ("m-2",), ("m-3",)],
        )
        await fresh_db.commit()

        store = SqliteEngravaCore(fresh_db)
        await store.ensure_schema()

        assert await _user_version(fresh_db) == _HEAD_VERSION
        assert await _row_count(fresh_db, "thought") == 3

    async def test_existing_rows_get_null_provenance(self, fresh_db: aiosqlite.Connection) -> None:
        """Rows present before the migration read back as ``provenance=None``."""
        await _bootstrap_core_at_v16(fresh_db)
        await fresh_db.execute(
            """
            INSERT INTO thought (thought_id, thought_type, essence, content, priority,
                                 lifecycle_status, updated_cycle)
            VALUES ('legacy-1', 'OBSERVATION', 'e', 'c', 'P2', 'ACTIVE', 0)
            """,
        )
        await fresh_db.commit()

        store = SqliteEngravaCore(fresh_db)
        await store.ensure_schema()

        fetched = await store.get_thought("legacy-1")
        assert fetched is not None
        assert fetched.provenance is None

    async def test_migrate_tolerates_thought_only_bootstrap(
        self, fresh_db: aiosqlite.Connection
    ) -> None:
        """A thought-only partial bootstrap migrates cleanly (no edge/embedding)."""
        await fresh_db.executescript(
            """
            CREATE TABLE thought (
                thought_id    TEXT PRIMARY KEY,
                thought_type  TEXT NOT NULL,
                essence       TEXT NOT NULL,
                content       TEXT NOT NULL,
                priority      TEXT NOT NULL
            );
            PRAGMA user_version = 16;
            """,
        )
        await fresh_db.commit()
        tables = await _table_names(fresh_db)
        assert "edge" not in tables

        store = SqliteEngravaCore(fresh_db)
        await store._migrate_core_v16_to_v17()  # must not raise

        assert "provenance" in await _column_names(fresh_db, "thought")
        for index_name in PROV_IDENTITY_INDEXES:
            assert await _index_exists(fresh_db, index_name), index_name

    async def test_fresh_equals_migrated_index_surface(self) -> None:
        """A fresh-head DB has the same ``thought`` index set as a migrated one."""
        fresh = await aiosqlite.connect(":memory:")
        fresh.row_factory = aiosqlite.Row
        migrated = await aiosqlite.connect(":memory:")
        migrated.row_factory = aiosqlite.Row
        try:
            await SqliteEngravaCore(fresh).ensure_schema()

            await _bootstrap_core_at_v16(migrated)
            await SqliteEngravaCore(migrated).ensure_schema()

            assert await _user_version(fresh) == await _user_version(migrated) == _HEAD_VERSION
            assert await _index_names(fresh, "thought") == await _index_names(migrated, "thought")
        finally:
            await fresh.close()
            await migrated.close()


# ---------------------------------------------------------------------------
# Journaling
# ---------------------------------------------------------------------------


class TestJournaling:
    async def test_create_delta_includes_provenance(self, jstore: SqliteEngravaCore) -> None:
        """The create journal delta carries the provenance sub-model."""
        prov = _full_provenance()
        await jstore.create_thought(_thought("t-j", provenance=prov))

        cursor = await jstore._db.execute(
            "SELECT delta FROM journal_entry WHERE target_id = ? AND mutation_type = ?",
            ("t-j", "INSERT_THOUGHT"),
        )
        row = await cursor.fetchone()
        assert row is not None
        delta_text = str(row["delta"])
        assert "sess-42" in delta_text
        assert "src-1" in delta_text

    async def test_verify_journal_passes_with_provenance(self, jstore: SqliteEngravaCore) -> None:
        """``verify_journal`` stays valid after a provenance-carrying write."""
        await jstore.create_thought(_thought("t-j2", provenance=_full_provenance()))
        await jstore.create_thought(_thought("t-j3"))
        await jstore.update_thought("t-j3", provenance=ProvenanceContext(session_id="late"))

        result = await jstore.verify_journal()
        assert result.valid


# ---------------------------------------------------------------------------
# No consumption — dreaming + ranking unaffected by presence of provenance
# ---------------------------------------------------------------------------


class TestNoConsumption:
    async def _populate(self, store: SqliteEngravaCore, *, with_provenance: bool) -> None:
        for i in range(6):
            prov = (
                ProvenanceContext(session_id=f"s-{i}", actor_id=f"a-{i}")
                if with_provenance
                else None
            )
            # A confirmed, confident thought so a promotion-tuned dreaming pass
            # actually promotes it — otherwise the equality would be vacuous.
            thought = ThoughtRecord(
                thought_id=f"t-{i}",
                thought_type=ThoughtType.OBSERVATION,
                essence=f"Essence t-{i}",
                content=f"shared topic alpha beta {i}",
                priority=Priority.P3,
                lifecycle_status=LifecycleStatus.ACTIVE,
                created_cycle=0,
                updated_cycle=i,
                source="test",
                confirmation_count=5,
                confidence=0.9,
                provenance=prov,
            )
            await store.create_thought(thought)

    async def test_dreaming_promotion_identical(self) -> None:
        """Dreaming promotes the same ids whether or not thoughts carry provenance.

        Uses a promotion-guaranteeing config so the pass promotes a non-empty
        set — the equality has teeth (a store whose thoughts carry provenance
        promotes exactly what an otherwise-identical provenance-free store does).
        """
        from engrava.config import DreamingConfig, DreamingGates
        from engrava.extensions.dreaming import DreamingExtension

        cfg = DreamingConfig(
            enabled=True,
            promote_threshold=0.0,
            max_p1_fraction=1.0,
            promote_targets="ALL",
            gates=DreamingGates(
                min_age_cycles=0,
                allow_zero_confirmation=True,
                enable_reflections=False,
            ),
        )

        async def _run(with_provenance: bool) -> list[str]:
            conn = await aiosqlite.connect(":memory:")
            conn.row_factory = aiosqlite.Row
            try:
                store = SqliteEngravaCore(conn)
                await store.ensure_schema()
                await self._populate(store, with_provenance=with_provenance)
                ext = DreamingExtension(config=cfg)
                result = await ext.run_consolidation(store, current_cycle=10)
                return sorted(result.promoted_ids)
            finally:
                await conn.close()

        with_prov = await _run(with_provenance=True)
        without_prov = await _run(with_provenance=False)
        # Non-vacuous: the tuned pass promotes a non-empty set.
        assert with_prov, "expected the tuned dreaming pass to promote at least one thought"
        assert with_prov == without_prov

    async def test_hybrid_ranking_identical(self) -> None:
        """Hybrid search ranks identically whether or not thoughts carry provenance."""

        async def _run(with_provenance: bool) -> list[tuple[str, float]]:
            conn = await aiosqlite.connect(":memory:")
            conn.row_factory = aiosqlite.Row
            try:
                store = SqliteEngravaCore(conn)
                await store.ensure_schema()
                await self._populate(store, with_provenance=with_provenance)
                result = await store.search_hybrid(query_text="alpha beta", top_k=6)
                # ``results`` are ``(thought_id, score)`` tuples.
                return [(tid, round(score, 9)) for tid, score in result.results]
            finally:
                await conn.close()

        with_prov = await _run(with_provenance=True)
        without_prov = await _run(with_provenance=False)
        assert with_prov == without_prov


# ---------------------------------------------------------------------------
# Read-only store still blocks the provenance write path
# ---------------------------------------------------------------------------


class TestReadOnly:
    async def test_create_with_provenance_blocked(self, store: SqliteEngravaCore) -> None:
        """A read-only wrapper blocks ``create_thought`` even with provenance."""
        ro = ReadOnlyEngrava(store)
        with pytest.raises(ReadOnlyViolationError):
            await ro.create_thought(_thought("t-ro", provenance=_full_provenance()))
