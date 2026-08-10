"""Update paths write only the fields they own, and report what actually landed.

Every update in the store used to be derived from a whole-record snapshot read
at the top of the call: the record was read, evolved in memory, and written back
column by column — including columns the caller never mentioned. A telemetry
write (``record_access``) or a confirmation bump landing between that read and
that write was silently overwritten.

These tests pin the two halves of the fix:

* **Blast radius** — an update writes the columns the operation owns and leaves
  every other column exactly as it stands in storage, verified by reading the
  row back with raw SQL rather than trusting the returned record.
* **Truthfulness** — the record handed back to the caller, and the ``after``
  image written to the journal, equal the state that is actually persisted. A
  partial write that still reports the pre-write snapshot would only trade a
  database clobber for a lying API.

The interleavings are deterministic: a one-shot wrapper runs the competing write
at the exact point between an operation's read and its write, so no test depends
on wall-clock timing or task scheduling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import (
    ActionNotFoundError,
    ActionRecord,
    ActionStatus,
    ActionType,
    EdgeRecord,
    EdgeType,
    KnowledgeSource,
    LifecycleStatus,
    Priority,
    SqliteEngravaCore,
    ThoughtNotFoundError,
    ThoughtRecord,
    ThoughtType,
    VerificationStatus,
)
from engrava.domain.models import MetadataValue

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


@pytest.fixture
async def db() -> AsyncIterator[aiosqlite.Connection]:
    """In-memory SQLite with the head schema (journal table included)."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode = WAL")
    await conn.execute("PRAGMA foreign_keys = ON")
    bootstrap = SqliteEngravaCore(conn, journal_enabled=True)
    await bootstrap.ensure_schema()
    yield conn
    await conn.close()


@pytest.fixture
async def store(db: aiosqlite.Connection) -> SqliteEngravaCore:
    """Store with journaling off — the plain write path."""
    return SqliteEngravaCore(db)


@pytest.fixture
async def journaling_store(db: aiosqlite.Connection) -> SqliteEngravaCore:
    """Store with journaling on — exercises the ``after`` image."""
    return SqliteEngravaCore(db, journal_enabled=True)


def _thought(
    thought_id: str = "t-1",
    *,
    essence: str = "essence",
    metadata: dict[str, MetadataValue] | None = None,
) -> ThoughtRecord:
    """Build a minimal ACTIVE thought."""
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.OBSERVATION,
        essence=essence,
        content=f"content of {thought_id}",
        priority=Priority.P3,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=0,
        updated_cycle=0,
        source="test",
        confidence=0.75,
        consolidated_from=["t-source"],
        metadata=metadata if metadata is not None else {},
    )


#: Non-ASCII metadata: the update encoding has to agree with the insert
#: encoding byte for byte, or an unrelated edit rewrites the column.
_UNICODE_METADATA: dict[str, MetadataValue] = {"speaker": "Łukasz", "city": "Kraków"}


def _edge(
    edge_id: str = "e-1",
    *,
    weight: float = 0.5,
    decay_multiplier: float = 1.0,
    metadata: dict[str, MetadataValue] | None = None,
) -> EdgeRecord:
    """Build an edge between ``t-1`` and ``t-2``."""
    return EdgeRecord(
        edge_id=edge_id,
        from_thought_id="t-1",
        to_thought_id="t-2",
        edge_type=EdgeType.ASSOCIATED,
        weight=weight,
        created_cycle=0,
        source=KnowledgeSource.EXPERIENCE,
        decay_multiplier=decay_multiplier,
        metadata=metadata if metadata is not None else {},
    )


def _action(action_id: str = "a-1") -> ActionRecord:
    """Build a PLANNED / PENDING action rooted at ``t-1``."""
    return ActionRecord(
        action_id=action_id,
        source_thought_id="t-1",
        action_type=ActionType.CLI_OUTPUT,
        intent="do the thing",
        status=ActionStatus.PLANNED,
        verification_status=VerificationStatus.PENDING,
    )


async def _row(db: aiosqlite.Connection, table: str, key: str, value: str) -> aiosqlite.Row:
    """Read a row straight from storage — never through a return value."""
    cursor = await db.execute(f"SELECT * FROM {table} WHERE {key} = ?", (value,))  # noqa: S608 -- table/key are test literals, value is bound
    row = await cursor.fetchone()
    assert row is not None
    return row


def _interleave_once(
    store: SqliteEngravaCore,
    method_name: str,
    intruder: Callable[[], Awaitable[object]],
) -> None:
    """Run ``intruder`` once, right after ``method_name`` first returns.

    This is the seam between an operation's read and its write: the wrapped
    method is the read, and the competing write happens before the caller gets
    a chance to act on what it read.
    """
    original = getattr(store, method_name)
    fired = False

    async def wrapper(*args: object, **kwargs: object) -> object:
        nonlocal fired
        result = await original(*args, **kwargs)
        if not fired:
            fired = True
            await intruder()
        return result

    setattr(store, method_name, wrapper)


def _interleave_after_statement(
    store: SqliteEngravaCore,
    prefix: str,
    intruder: Callable[[], Awaitable[object]],
) -> None:
    """Run ``intruder`` once, right after the first statement matching ``prefix``.

    Reaches the narrower window the read-back guards: between the write itself
    and the read that confirms what the write left behind.
    """
    original = store._db.execute
    fired = False

    async def wrapper(sql: str, parameters: object = None) -> object:
        nonlocal fired
        result = await original(sql, parameters)
        if not fired and sql.lstrip().upper().startswith(prefix.upper()):
            fired = True
            await intruder()
        return result

    # Deliberate test seam: aiosqlite exposes no statement hook, so the store's
    # own connection method is shadowed on this one instance to observe the SQL
    # it issues. Nothing in the package assigns to it.
    store._db.execute = wrapper  # type: ignore[method-assign] -- deliberate test seam


def _record_statements(store: SqliteEngravaCore) -> list[str]:
    """Collect every SQL statement the store issues from now on."""
    statements: list[str] = []
    original = store._db.execute

    async def wrapper(sql: str, parameters: object = None) -> object:
        statements.append(sql)
        return await original(sql, parameters)

    # Deliberate test seam: aiosqlite exposes no statement hook, so the store's
    # own connection method is shadowed on this one instance to observe the SQL
    # it issues. Nothing in the package assigns to it.
    store._db.execute = wrapper  # type: ignore[method-assign] -- deliberate test seam
    return statements


def _set_columns(statements: list[str], table: str) -> set[str]:
    """Extract the SET column names of the single UPDATE issued for ``table``."""
    prefix = f"UPDATE {table.upper()} SET"
    updates = [s for s in statements if s.lstrip().upper().startswith(prefix)]
    assert len(updates) == 1, f"expected exactly one UPDATE {table}, got {updates}"
    body = updates[0].split(" SET ", 1)[1].split(" WHERE ", 1)[0]
    return {assignment.split("=")[0].strip() for assignment in body.split(",")}


# ---------------------------------------------------------------------------
# update_thought — blast radius
# ---------------------------------------------------------------------------


class TestUpdateThoughtBlastRadius:
    """An update touches its own columns and nothing else."""

    async def test_update_writes_only_the_columns_it_owns(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """The SET list carries the caller's field plus the update stamp."""
        await store.create_thought(_thought())
        statements = _record_statements(store)

        await store.update_thought("t-1", essence="new essence")

        assert _set_columns(statements, "thought") == {"essence", "updated_at"}

    async def test_concurrent_access_telemetry_survives_an_update(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """An access recorded between the read and the write is not rolled back."""
        await store.create_thought(_thought())
        observed: list[str] = []

        async def _telemetry() -> None:
            await store.record_access("t-1")
            row = await _row(db, "thought", "thought_id", "t-1")
            observed.append(row["last_accessed_at"])

        _interleave_once(store, "_get_thought_row", _telemetry)

        await store.update_thought("t-1", essence="new essence")

        row = await _row(db, "thought", "thought_id", "t-1")
        # The intended change landed...
        assert row["essence"] == "new essence"
        # ...and the concurrent telemetry was not overwritten.
        assert row["access_count"] == 1
        assert row["last_accessed_at"] == observed[0]

    async def test_concurrent_confirmation_bump_survives_an_update(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """A confirmation bump between the read and the write is not rolled back."""
        await store.create_thought(_thought())

        async def _confirm() -> None:
            await db.execute(
                "UPDATE thought SET confirmation_count = confirmation_count + 1 "
                "WHERE thought_id = ?",
                ("t-1",),
            )

        _interleave_once(store, "_get_thought_row", _confirm)

        await store.update_thought("t-1", priority=Priority.P1)

        row = await _row(db, "thought", "thought_id", "t-1")
        assert row["priority"] == Priority.P1.value
        assert row["confirmation_count"] == 1

    async def test_no_column_in_the_row_moves_beyond_the_owned_ones(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """Compare the whole stored row, byte for byte, across the write.

        Naming the columns to check would only catch the ones this test
        remembered; diffing every column of the row catches any that is
        silently rewritten. With a telemetry write landing mid-operation, the
        only columns allowed to move are the ones the edit owns plus the ones
        the other writer owns.
        """
        await store.create_thought(_thought(metadata=_UNICODE_METADATA))
        before = dict(await _row(db, "thought", "thought_id", "t-1"))

        async def _telemetry() -> None:
            await store.record_access("t-1")

        _interleave_once(store, "_get_thought_row", _telemetry)

        await store.update_thought("t-1", essence="new essence")

        after = dict(await _row(db, "thought", "thought_id", "t-1"))
        changed = {key for key in before if before[key] != after[key]}
        assert changed == {"essence", "updated_at", "access_count", "last_accessed_at"}


# ---------------------------------------------------------------------------
# update_thought — truthfulness of what is reported
# ---------------------------------------------------------------------------


class TestUpdateThoughtReportsWhatLanded:
    """The returned record and the journal image equal persisted state."""

    async def test_returned_record_reflects_the_concurrent_write(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """The caller gets persisted state, not the snapshot read at entry."""
        await store.create_thought(_thought())

        async def _telemetry() -> None:
            await store.record_access("t-1")

        _interleave_once(store, "_get_thought_row", _telemetry)

        returned = await store.update_thought("t-1", essence="new essence")

        row = await _row(db, "thought", "thought_id", "t-1")
        assert returned.access_count == 1
        assert returned.access_count == row["access_count"]
        assert returned.last_accessed_at == row["last_accessed_at"]
        assert returned.essence == row["essence"]

    async def test_journal_after_image_reflects_the_concurrent_write(
        self,
        journaling_store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """The journal records the row that exists, not the one that was planned."""
        await journaling_store.create_thought(_thought())

        async def _telemetry() -> None:
            await journaling_store.record_access("t-1")

        _interleave_once(journaling_store, "_get_thought_row", _telemetry)

        await journaling_store.update_thought("t-1", essence="new essence")

        assert journaling_store._journal is not None
        entries = await journaling_store._journal.get_entries(
            target_id="t-1",
            mutation_type="UPDATE_THOUGHT",
        )
        after = entries[-1].delta["after"]
        assert isinstance(after, dict)
        row = await _row(db, "thought", "thought_id", "t-1")
        assert after["access_count"] == 1
        assert after["access_count"] == row["access_count"]
        assert after["last_accessed_at"] == row["last_accessed_at"]
        assert after["updated_at"] == row["updated_at"]

    async def test_update_of_a_vanished_row_raises_instead_of_reporting_success(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """A row deleted after the write is not reported back as an updated record."""
        await store.create_thought(_thought())

        async def _delete() -> None:
            await store._db.execute("DELETE FROM thought WHERE thought_id = ?", ("t-1",))

        _interleave_after_statement(store, "UPDATE thought SET", _delete)

        with pytest.raises(ThoughtNotFoundError):
            await store.update_thought("t-1", essence="new essence")


# ---------------------------------------------------------------------------
# restore_thought
# ---------------------------------------------------------------------------


class TestRestoreThought:
    """Restore shares the update path and therefore the same obligations."""

    async def test_concurrent_access_telemetry_survives_a_restore(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """Un-archiving does not roll back an access recorded meanwhile."""
        await store.create_thought(_thought())
        await store.update_thought("t-1", lifecycle_status=LifecycleStatus.ARCHIVED)

        async def _telemetry() -> None:
            await store.record_access("t-1")

        _interleave_once(store, "_get_thought_row", _telemetry)

        returned = await store.restore_thought("t-1")

        row = await _row(db, "thought", "thought_id", "t-1")
        assert row["lifecycle_status"] == LifecycleStatus.ACTIVE.value
        assert row["access_count"] == 1
        assert returned.access_count == 1
        assert returned.last_accessed_at == row["last_accessed_at"]

    async def test_journal_after_image_reflects_the_concurrent_write(
        self,
        journaling_store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """The restore journals the row that exists, not the one it planned."""
        await journaling_store.create_thought(_thought())
        await journaling_store.update_thought("t-1", lifecycle_status=LifecycleStatus.ARCHIVED)

        async def _telemetry() -> None:
            await journaling_store.record_access("t-1")

        _interleave_once(journaling_store, "_get_thought_row", _telemetry)

        await journaling_store.restore_thought("t-1")

        assert journaling_store._journal is not None
        entries = await journaling_store._journal.get_entries(
            target_id="t-1",
            mutation_type="UPDATE_THOUGHT",
        )
        after = entries[-1].delta["after"]
        assert isinstance(after, dict)
        row = await _row(db, "thought", "thought_id", "t-1")
        assert after["access_count"] == 1
        assert after["access_count"] == row["access_count"]
        assert after["lifecycle_status"] == row["lifecycle_status"]

    async def test_restore_of_a_vanished_row_raises(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """A row deleted after the write is not reported back as a restored record."""
        await store.create_thought(_thought())
        await store.update_thought("t-1", lifecycle_status=LifecycleStatus.ARCHIVED)
        statements = _record_statements(store)

        async def _delete() -> None:
            await store._db.execute("DELETE FROM thought WHERE thought_id = ?", ("t-1",))

        _interleave_after_statement(store, "UPDATE thought SET", _delete)

        with pytest.raises(ThoughtNotFoundError):
            await store.restore_thought("t-1")
        assert any(s.lstrip().upper().startswith("UPDATE THOUGHT SET") for s in statements)


# ---------------------------------------------------------------------------
# Confirmation counter
# ---------------------------------------------------------------------------


class TestConfirmationCounter:
    """The bump is relative to stored state, not to a value read earlier."""

    async def test_concurrent_bump_is_not_lost(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """Two confirmations of the same content count twice."""
        await store.create_thought(_thought())

        async def _confirm() -> None:
            await db.execute(
                "UPDATE thought SET confirmation_count = confirmation_count + 1 "
                "WHERE thought_id = ?",
                ("t-1",),
            )

        _interleave_once(store, "_get_thought_by_content_hash", _confirm)

        await store.create_thought(_thought(), deduplicate=True)

        row = await _row(db, "thought", "thought_id", "t-1")
        assert row["confirmation_count"] == 2

    async def test_no_column_in_the_row_moves_beyond_the_bump(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """The bump moves the count and the stamp, and nothing else in the row.

        The blast-radius half is asserted over every column; the count value is
        what makes the bump relative rather than absolute.
        """
        await store.create_thought(_thought(metadata=_UNICODE_METADATA))
        before = dict(await _row(db, "thought", "thought_id", "t-1"))

        async def _confirm() -> None:
            await db.execute(
                "UPDATE thought SET confirmation_count = confirmation_count + 1 "
                "WHERE thought_id = ?",
                ("t-1",),
            )

        _interleave_once(store, "_get_thought_by_content_hash", _confirm)

        await store.create_thought(_thought(metadata=_UNICODE_METADATA), deduplicate=True)

        after = dict(await _row(db, "thought", "thought_id", "t-1"))
        changed = {key for key in before if before[key] != after[key]}
        assert changed == {"confirmation_count", "updated_at"}
        assert after["confirmation_count"] == 2

    async def test_returned_record_carries_the_persisted_count(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """The caller is told the count that is stored."""
        await store.create_thought(_thought())

        async def _confirm() -> None:
            await db.execute(
                "UPDATE thought SET confirmation_count = confirmation_count + 1 "
                "WHERE thought_id = ?",
                ("t-1",),
            )

        _interleave_once(store, "_get_thought_by_content_hash", _confirm)

        returned = await store.create_thought(_thought(), deduplicate=True)

        row = await _row(db, "thought", "thought_id", "t-1")
        assert returned.confirmation_count == 2
        assert returned.confirmation_count == row["confirmation_count"]
        assert returned.updated_at == row["updated_at"]

    async def test_journal_after_image_carries_the_persisted_count(
        self,
        journaling_store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """The journal records the count that is stored."""
        await journaling_store.create_thought(_thought())

        async def _confirm() -> None:
            await db.execute(
                "UPDATE thought SET confirmation_count = confirmation_count + 1 "
                "WHERE thought_id = ?",
                ("t-1",),
            )

        _interleave_once(journaling_store, "_get_thought_by_content_hash", _confirm)

        await journaling_store.create_thought(_thought(), deduplicate=True)

        assert journaling_store._journal is not None
        entries = await journaling_store._journal.get_entries(
            target_id="t-1",
            mutation_type="UPDATE_THOUGHT",
        )
        after = entries[-1].delta["after"]
        assert isinstance(after, dict)
        row = await _row(db, "thought", "thought_id", "t-1")
        assert after["confirmation_count"] == 2
        assert after["confirmation_count"] == row["confirmation_count"]

    async def test_bump_of_a_vanished_row_raises_instead_of_reporting_success(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """A row deleted between the hash probe and the bump is not reported as bumped."""
        await store.create_thought(_thought())

        async def _delete() -> None:
            await store._db.execute("DELETE FROM thought WHERE thought_id = ?", ("t-1",))

        _interleave_once(store, "_get_thought_by_content_hash", _delete)

        with pytest.raises(ThoughtNotFoundError):
            await store.create_thought(_thought(), deduplicate=True)

    async def test_upsert_hit_reports_the_persisted_count(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """``get_or_create`` shares the bump and therefore the same guarantee."""
        await store.create_thought(_thought())

        async def _confirm() -> None:
            await db.execute(
                "UPDATE thought SET confirmation_count = confirmation_count + 1 "
                "WHERE thought_id = ?",
                ("t-1",),
            )

        _interleave_once(store, "_get_thought_by_content_hash", _confirm)

        returned, created = await store.get_or_create(_thought())

        row = await _row(db, "thought", "thought_id", "t-1")
        assert created is False
        assert returned.confirmation_count == 2
        assert returned.confirmation_count == row["confirmation_count"]


# ---------------------------------------------------------------------------
# update_edge
# ---------------------------------------------------------------------------


class TestUpdateEdge:
    """Edges follow the same rule as thoughts."""

    async def _seed(self, store: SqliteEngravaCore, edge: EdgeRecord | None = None) -> None:
        await store.create_thought(_thought("t-1"))
        await store.create_thought(_thought("t-2"))
        await store.create_edge(edge if edge is not None else _edge())

    async def test_update_writes_only_the_columns_it_owns(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """The SET list carries exactly the changed field."""
        await self._seed(store)
        statements = _record_statements(store)

        await store.update_edge("e-1", weight=0.9)

        assert _set_columns(statements, "edge") == {"weight"}

    async def test_concurrent_weight_change_survives_an_update(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """A weight written between the read and the write is not rolled back."""
        await self._seed(store)

        async def _reweight() -> None:
            await db.execute("UPDATE edge SET weight = ? WHERE edge_id = ?", (0.25, "e-1"))

        _interleave_once(store, "_get_edge_row", _reweight)

        returned = await store.update_edge("e-1", decay_multiplier=2.0)

        row = await _row(db, "edge", "edge_id", "e-1")
        assert row["decay_multiplier"] == 2.0
        assert row["weight"] == 0.25
        assert returned.weight == 0.25
        assert returned.decay_multiplier == 2.0

    async def test_no_column_in_the_row_moves_beyond_the_owned_ones(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """Compare the whole stored edge row, byte for byte, across the write."""
        await self._seed(store, _edge(metadata=_UNICODE_METADATA))
        before = dict(await _row(db, "edge", "edge_id", "e-1"))

        async def _reweight() -> None:
            await db.execute("UPDATE edge SET weight = ? WHERE edge_id = ?", (0.25, "e-1"))

        _interleave_once(store, "_get_edge_row", _reweight)

        await store.update_edge("e-1", decay_multiplier=2.0)

        after = dict(await _row(db, "edge", "edge_id", "e-1"))
        changed = {key for key in before if before[key] != after[key]}
        assert changed == {"decay_multiplier", "weight"}

    async def test_journal_after_image_reflects_the_concurrent_write(
        self,
        journaling_store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """The journal records the edge that exists, not the one that was planned."""
        await self._seed(journaling_store)

        async def _reweight() -> None:
            await db.execute("UPDATE edge SET weight = ? WHERE edge_id = ?", (0.25, "e-1"))

        _interleave_once(journaling_store, "_get_edge_row", _reweight)

        await journaling_store.update_edge("e-1", decay_multiplier=2.0)

        assert journaling_store._journal is not None
        entries = await journaling_store._journal.get_entries(
            target_id="e-1",
            mutation_type="UPDATE_EDGE",
        )
        after = entries[-1].delta["after"]
        assert isinstance(after, dict)
        row = await _row(db, "edge", "edge_id", "e-1")
        assert after["weight"] == 0.25
        assert after["weight"] == row["weight"]
        assert after["decay_multiplier"] == row["decay_multiplier"]

    async def test_a_zero_decay_multiplier_survives_an_unrelated_update(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """``0.0`` is a valid decay and must round-trip as itself, not as the default.

        The boundary that a truthiness test in the row mapper silently rewrites:
        a stored ``0.0`` is falsy, so a mapper that falls back on falsiness
        reports ``1.0`` for a row the database says is ``0.0``.
        """
        await self._seed(store, _edge(decay_multiplier=0.0))

        returned = await store.update_edge("e-1", weight=0.9)

        row = await _row(db, "edge", "edge_id", "e-1")
        assert row["decay_multiplier"] == 0.0
        assert returned.decay_multiplier == 0.0
        assert returned.weight == 0.9

    async def test_a_zero_decay_multiplier_is_journaled_as_stored(
        self,
        journaling_store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """The ``after`` image carries the stored ``0.0``, not the model default."""
        await self._seed(journaling_store, _edge(decay_multiplier=0.0))

        await journaling_store.update_edge("e-1", weight=0.9)

        assert journaling_store._journal is not None
        entries = await journaling_store._journal.get_entries(
            target_id="e-1",
            mutation_type="UPDATE_EDGE",
        )
        after = entries[-1].delta["after"]
        assert isinstance(after, dict)
        row = await _row(db, "edge", "edge_id", "e-1")
        assert after["decay_multiplier"] == 0.0
        assert after["decay_multiplier"] == row["decay_multiplier"]

    async def test_update_without_changes_writes_nothing(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """An update that changes nothing issues no UPDATE at all."""
        await self._seed(store)
        statements = _record_statements(store)

        returned = await store.update_edge("e-1")

        assert not [s for s in statements if s.lstrip().upper().startswith("UPDATE EDGE SET")]
        assert returned.weight == 0.5

    async def test_update_of_an_edge_deleted_before_the_write_raises(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """An edge deleted between the read and the write is not reported as updated."""
        await self._seed(store)

        async def _delete() -> None:
            await store._db.execute("DELETE FROM edge WHERE edge_id = ?", ("e-1",))

        _interleave_once(store, "_get_edge_row", _delete)

        with pytest.raises(ValueError, match="Edge not found"):
            await store.update_edge("e-1", weight=0.9)

    async def test_update_of_an_edge_deleted_after_the_write_raises(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """The narrower window: the write landed, then the row went away."""
        await self._seed(store)
        statements = _record_statements(store)

        async def _delete() -> None:
            await store._db.execute("DELETE FROM edge WHERE edge_id = ?", ("e-1",))

        _interleave_after_statement(store, "UPDATE edge SET", _delete)

        with pytest.raises(ValueError, match="Edge not found"):
            await store.update_edge("e-1", weight=0.9)
        assert any(s.lstrip().upper().startswith("UPDATE EDGE SET") for s in statements)


# ---------------------------------------------------------------------------
# update_action
# ---------------------------------------------------------------------------


class TestUpdateAction:
    """Actions follow the same rule: write the field the call changes."""

    async def _seed(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_thought("t-1"))
        await store.create_action(_action())

    async def test_status_only_update_writes_only_status(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """A status move does not re-assert the verification column."""
        await self._seed(store)
        statements = _record_statements(store)

        await store.update_action("a-1", status=ActionStatus.EXECUTING)

        assert _set_columns(statements, "action") == {"status"}

    async def test_concurrent_verification_change_survives_a_status_update(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """A verification written between the read and the write is not rolled back."""
        await self._seed(store)

        async def _verify() -> None:
            await db.execute(
                "UPDATE action SET verification_status = ? WHERE action_id = ?",
                (VerificationStatus.PARTIAL.value, "a-1"),
            )

        _interleave_once(store, "_get_action", _verify)

        returned = await store.update_action("a-1", status=ActionStatus.EXECUTING)

        row = await _row(db, "action", "action_id", "a-1")
        assert row["status"] == ActionStatus.EXECUTING.value
        assert row["verification_status"] == VerificationStatus.PARTIAL.value
        assert returned.status is ActionStatus.EXECUTING
        assert returned.verification_status is VerificationStatus.PARTIAL

    async def test_no_column_in_the_row_moves_beyond_the_owned_ones(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """Compare the whole stored action row, byte for byte, across the write."""
        await self._seed(store)
        before = dict(await _row(db, "action", "action_id", "a-1"))

        async def _verify() -> None:
            await db.execute(
                "UPDATE action SET verification_status = ? WHERE action_id = ?",
                (VerificationStatus.PARTIAL.value, "a-1"),
            )

        _interleave_once(store, "_get_action", _verify)

        await store.update_action("a-1", status=ActionStatus.EXECUTING)

        after = dict(await _row(db, "action", "action_id", "a-1"))
        changed = {key for key in before if before[key] != after[key]}
        assert changed == {"status", "verification_status"}

    async def test_journal_after_image_reflects_the_concurrent_write(
        self,
        journaling_store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """The journal records the action that exists, not the one that was planned."""
        await self._seed(journaling_store)

        async def _verify() -> None:
            await db.execute(
                "UPDATE action SET verification_status = ? WHERE action_id = ?",
                (VerificationStatus.PARTIAL.value, "a-1"),
            )

        _interleave_once(journaling_store, "_get_action", _verify)

        await journaling_store.update_action("a-1", status=ActionStatus.EXECUTING)

        assert journaling_store._journal is not None
        entries = await journaling_store._journal.get_entries(
            target_id="a-1",
            mutation_type="UPDATE_ACTION",
        )
        after = entries[-1].delta["after"]
        assert isinstance(after, dict)
        row = await _row(db, "action", "action_id", "a-1")
        assert after["verification_status"] == VerificationStatus.PARTIAL.value
        assert after["verification_status"] == row["verification_status"]
        assert after["status"] == row["status"]

    async def test_update_of_an_action_deleted_before_the_write_raises(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """An action deleted between the read and the write is not reported as updated."""
        await self._seed(store)

        async def _delete() -> None:
            await store._db.execute("DELETE FROM action WHERE action_id = ?", ("a-1",))

        _interleave_once(store, "_get_action", _delete)

        with pytest.raises(ActionNotFoundError):
            await store.update_action("a-1", status=ActionStatus.EXECUTING)

    async def test_update_of_an_action_deleted_after_the_write_raises(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """The narrower window: the write landed, then the row went away."""
        await self._seed(store)
        statements = _record_statements(store)

        async def _delete() -> None:
            await store._db.execute("DELETE FROM action WHERE action_id = ?", ("a-1",))

        _interleave_after_statement(store, "UPDATE action SET", _delete)

        with pytest.raises(ActionNotFoundError):
            await store.update_action("a-1", status=ActionStatus.EXECUTING)
        assert any(s.lstrip().upper().startswith("UPDATE ACTION SET") for s in statements)


# ---------------------------------------------------------------------------
# The column maps are derived from the schema, not maintained by hand
# ---------------------------------------------------------------------------


class TestColumnMapsMatchTheSchema:
    """A column map that cannot detect its own rot is a liability."""

    async def _table_columns(self, db: aiosqlite.Connection, table: str) -> set[str]:
        cursor = await db.execute(f"PRAGMA table_info({table})")
        return {row["name"] for row in await cursor.fetchall()}

    async def test_thought_update_map_covers_every_column(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """Every thought column is either updatable or explicitly excluded."""
        mapped = set(store._thought_to_core_columns(_thought()))
        # ``thought_id`` identifies the row being updated. ``content_hash`` is
        # excluded because no update has ever written it — a known defect (an
        # edit to ``content`` leaves the stored hash pointing at the old text),
        # preserved here deliberately rather than sanctioned: changing it moves
        # deduplication behaviour and belongs to its own change.
        excluded = {"thought_id", "content_hash"}
        assert mapped | excluded == await self._table_columns(db, "thought")
        assert not mapped & excluded

    async def test_edge_update_map_covers_every_column(
        self,
        db: aiosqlite.Connection,
    ) -> None:
        """Every edge column is either updatable or explicitly excluded."""
        from engrava.infrastructure.sqlite.engrava_core import _edge_to_core_columns

        mapped = set(_edge_to_core_columns(_edge()))
        # ``edge_id`` identifies the row being updated.
        excluded = {"edge_id"}
        assert mapped | excluded == await self._table_columns(db, "edge")
        assert not mapped & excluded

    async def test_thought_update_map_values_round_trip(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """The mapped values are exactly what an insert of the same record stores.

        Every nullable numeric field is stored at its falsy boundary, because
        that is where a mapper that tests truthiness instead of ``is not None``
        substitutes a default for a value the database actually holds.
        """
        thought = _thought(metadata=_UNICODE_METADATA).model_copy(
            update={
                "confidence": 0.0,
                "action_outcome_score": 0.0,
                "archived_at_cycle": 0,
                "access_count": 0,
                "confirmation_count": 0,
                "pinned": False,
            },
        )
        await store.create_thought(thought)
        stored = dict(await _row(db, "thought", "thought_id", "t-1"))
        fetched = await store.get_thought("t-1")
        assert fetched is not None
        for column, value in store._thought_to_core_columns(fetched).items():
            assert value == stored[column], column

    async def test_edge_update_map_values_round_trip(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """The same for edges, over the falsy boundary values.

        Reading a row and re-encoding it has to be the identity, or an update
        that touches one column reports — and can rewrite — a different value
        for another. Zeros are the values a truthiness-based mapper loses.
        """
        from engrava.infrastructure.sqlite.engrava_core import _edge_to_core_columns

        await store.create_thought(_thought("t-1"))
        await store.create_thought(_thought("t-2"))
        await store.create_edge(
            _edge(weight=0.0, decay_multiplier=0.0, metadata=_UNICODE_METADATA),
        )
        stored = dict(await _row(db, "edge", "edge_id", "e-1"))
        fetched = await store._read_back_edge("e-1")
        for column, value in _edge_to_core_columns(fetched).items():
            assert value == stored[column], column
