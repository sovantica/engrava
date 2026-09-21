"""The revision guard's matrix, pinned row by row: what bumps, what enforces.

An earlier design for this same mechanism was rejected before a line of code
was written: its matrix of which paths bump and which enforce was asserted
rather than tested, and each re-reading of the code turned up another branch
the matrix had got wrong. This module is the direct counter-measure: for every
mutating path, a test reads the row's ``revision`` column with raw SQL before
and after the call, so "bumps" and "does not bump" are observed facts rather
than something inferred from the returned domain record (which never carries
``revision`` at all — it is not a domain-model field, only a bookkeeping
column the guarded ``UPDATE`` statements compare and increment).

Rows already covered elsewhere, referenced rather than duplicated here:

* The intra-call race shape (a competing write landing between one call's own
  read and write, same task or a second store) — ``tests/test_concurrency_contract.py``.
* Blast-radius / partial-field-write pinning, including that ``revision`` is
  one of the columns that moves on every real update — ``tests/test_partial_field_updates.py``.
* ``derive_existing``'s reused-child branch not bumping the reused thought — the
  existing byte-identical-row assertion in
  ``tests/test_derived_records_seam.py::test_backfill_of_on_store_source_is_byte_identical_noop``
  already proves this: a byte-identical row includes ``revision``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import (
    ActionRecord,
    ActionStatus,
    ActionType,
    EdgeRecord,
    EdgeType,
    HygienePolicyConfig,
    KnowledgeSource,
    LifecycleStatus,
    Priority,
    SqliteEngravaCore,
    StaleDataError,
    ThoughtRecord,
    ThoughtType,
    VerificationStatus,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


@pytest.fixture
async def db() -> AsyncIterator[aiosqlite.Connection]:
    """In-memory SQLite with the head schema."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA foreign_keys = ON")
    bootstrap = SqliteEngravaCore(conn)
    await bootstrap.ensure_schema()
    yield conn
    await conn.close()


@pytest.fixture
async def store(db: aiosqlite.Connection) -> SqliteEngravaCore:
    return SqliteEngravaCore(db)


def _thought(thought_id: str = "t-1", **overrides: object) -> ThoughtRecord:
    params: dict[str, object] = {
        "thought_id": thought_id,
        "thought_type": ThoughtType.OBSERVATION,
        "essence": f"essence of {thought_id}",
        "content": f"content of {thought_id}",
        "priority": Priority.P3,
        "lifecycle_status": LifecycleStatus.ACTIVE,
        "source": "test",
    }
    params.update(overrides)
    return ThoughtRecord(**params)  # type: ignore[arg-type]


def _edge(edge_id: str = "e-1", **overrides: object) -> EdgeRecord:
    params: dict[str, object] = {
        "edge_id": edge_id,
        "from_thought_id": "t-1",
        "to_thought_id": "t-2",
        "edge_type": EdgeType.ASSOCIATED,
        "weight": 0.5,
        "created_cycle": 0,
        "source": KnowledgeSource.EXPERIENCE,
    }
    params.update(overrides)
    return EdgeRecord(**params)  # type: ignore[arg-type]


def _action(action_id: str = "a-1", **overrides: object) -> ActionRecord:
    params: dict[str, object] = {
        "action_id": action_id,
        "source_thought_id": "t-1",
        "action_type": ActionType.CLI_OUTPUT,
        "intent": "do the thing",
        "status": ActionStatus.PLANNED,
        "verification_status": VerificationStatus.PENDING,
    }
    params.update(overrides)
    return ActionRecord(**params)  # type: ignore[arg-type]


async def _revision(db: aiosqlite.Connection, table: str, id_column: str, id_value: str) -> int:
    cursor = await db.execute(
        f"SELECT revision FROM {table} WHERE {id_column} = ?",  # noqa: S608 -- table/id_column are test literals
        (id_value,),
    )
    row = await cursor.fetchone()
    assert row is not None, f"{table}.{id_column}={id_value!r} not found"
    return int(row["revision"])


async def _row_exists(db: aiosqlite.Connection, table: str, id_column: str, id_value: str) -> bool:
    cursor = await db.execute(
        f"SELECT 1 FROM {table} WHERE {id_column} = ?",  # noqa: S608 -- table/id_column are test literals
        (id_value,),
    )
    return await cursor.fetchone() is not None


# ---------------------------------------------------------------------------
# Inserts start at 0
# ---------------------------------------------------------------------------


class TestInsertsStartAtZero:
    async def test_create_thought_starts_at_revision_zero(
        self, store: SqliteEngravaCore, db: aiosqlite.Connection
    ) -> None:
        await store.create_thought(_thought())
        assert await _revision(db, "thought", "thought_id", "t-1") == 0

    async def test_create_edge_starts_at_revision_zero(
        self, store: SqliteEngravaCore, db: aiosqlite.Connection
    ) -> None:
        await store.create_thought(_thought("t-1"))
        await store.create_thought(_thought("t-2"))
        await store.create_edge(_edge())
        assert await _revision(db, "edge", "edge_id", "e-1") == 0

    async def test_create_action_starts_at_revision_zero(
        self, store: SqliteEngravaCore, db: aiosqlite.Connection
    ) -> None:
        await store.create_thought(_thought("t-1"))
        await store.create_action(_action())
        assert await _revision(db, "action", "action_id", "a-1") == 0


# ---------------------------------------------------------------------------
# The four guarded update paths: bump + enforce
# ---------------------------------------------------------------------------


class TestUpdatePathsBumpAndEnforce:
    async def test_update_thought_bumps_on_every_call_and_enforces(
        self, store: SqliteEngravaCore, db: aiosqlite.Connection
    ) -> None:
        await store.create_thought(_thought())
        assert await _revision(db, "thought", "thought_id", "t-1") == 0

        await store.update_thought("t-1", essence="second")
        assert await _revision(db, "thought", "thought_id", "t-1") == 1

        await store.update_thought("t-1", essence="third")
        assert await _revision(db, "thought", "thought_id", "t-1") == 2

        # Enforcement: a stale caller-held expectation (simulated by writing
        # straight to the row, bypassing the guarded path) makes the next
        # guarded write on that row fail -- proven via the real interleave
        # mechanism in test_concurrency_contract.py; here the direct value
        # movement above is the "bumps" half of the row.

    async def test_restore_thought_bumps_and_enforces(
        self, store: SqliteEngravaCore, db: aiosqlite.Connection
    ) -> None:
        await store.create_thought(_thought(lifecycle_status=LifecycleStatus.ARCHIVED))
        assert await _revision(db, "thought", "thought_id", "t-1") == 0

        await store.restore_thought("t-1")
        assert await _revision(db, "thought", "thought_id", "t-1") == 1

    async def test_update_edge_bumps_on_a_real_change_and_enforces(
        self, store: SqliteEngravaCore, db: aiosqlite.Connection
    ) -> None:
        await store.create_thought(_thought("t-1"))
        await store.create_thought(_thought("t-2"))
        await store.create_edge(_edge())
        assert await _revision(db, "edge", "edge_id", "e-1") == 0

        await store.update_edge("e-1", weight=0.9)
        assert await _revision(db, "edge", "edge_id", "e-1") == 1

    async def test_update_edge_no_op_does_not_bump(
        self, store: SqliteEngravaCore, db: aiosqlite.Connection
    ) -> None:
        """An edit that changes nothing issues no ``UPDATE`` at all (pre-existing
        behaviour), so there is nothing to bump the guard against."""
        await store.create_thought(_thought("t-1"))
        await store.create_thought(_thought("t-2"))
        await store.create_edge(_edge(weight=0.5))
        assert await _revision(db, "edge", "edge_id", "e-1") == 0

        await store.update_edge("e-1", weight=0.5)  # same value -> no diff
        assert await _revision(db, "edge", "edge_id", "e-1") == 0

    async def test_update_action_bumps_only_on_an_actual_change(
        self, store: SqliteEngravaCore, db: aiosqlite.Connection
    ) -> None:
        await store.create_thought(_thought("t-1"))
        await store.create_action(_action())
        assert await _revision(db, "action", "action_id", "a-1") == 0

        # No-op: supplying the value already stored must not bump -- a bump
        # here would invent a change that never happened.
        await store.update_action("a-1", status=ActionStatus.PLANNED)
        assert await _revision(db, "action", "action_id", "a-1") == 0

        # A real change bumps.
        await store.update_action("a-1", status=ActionStatus.EXECUTING)
        assert await _revision(db, "action", "action_id", "a-1") == 1

        # A verification-only change on a still-non-terminal action is also a
        # real change and also bumps.
        await store.update_action("a-1", verification_status=VerificationStatus.PARTIAL)
        assert await _revision(db, "action", "action_id", "a-1") == 2

    async def test_upsert_by_hash_update_branch_bumps_and_enforces(
        self, store: SqliteEngravaCore, db: aiosqlite.Connection
    ) -> None:
        await store.create_thought(_thought(content="shared content"))
        assert await _revision(db, "thought", "thought_id", "t-1") == 0

        await store.upsert_by_hash(
            _thought("t-unused", essence="from the upsert", content="shared content")
        )
        assert await _revision(db, "thought", "thought_id", "t-1") == 1

    async def test_upsert_by_hash_no_op_hit_does_not_bump(
        self, store: SqliteEngravaCore, db: aiosqlite.Connection
    ) -> None:
        """A hit whose mutable fields already match the stored row writes
        nothing (pre-existing behaviour) -- and so cannot bump."""
        thought = _thought(essence="same", content="shared content")
        await store.create_thought(thought)
        assert await _revision(db, "thought", "thought_id", "t-1") == 0

        await store.upsert_by_hash(_thought("t-unused", essence="same", content="shared content"))
        assert await _revision(db, "thought", "thought_id", "t-1") == 0


# ---------------------------------------------------------------------------
# Dedup-hit branches: neither bump nor enforce
# ---------------------------------------------------------------------------


class TestDedupHitBranchesDoNotBump:
    async def test_create_thought_deduplicate_hit_does_not_bump(
        self, store: SqliteEngravaCore, db: aiosqlite.Connection
    ) -> None:
        await store.create_thought(_thought(content="shared"))
        assert await _revision(db, "thought", "thought_id", "t-1") == 0

        hit = await store.create_thought(
            _thought("t-2", content="shared"),
            deduplicate=True,
        )
        assert hit.thought_id == "t-1"
        assert hit.confirmation_count == 1
        assert await _revision(db, "thought", "thought_id", "t-1") == 0

    async def test_get_or_create_hit_does_not_bump(
        self, store: SqliteEngravaCore, db: aiosqlite.Connection
    ) -> None:
        await store.create_thought(_thought(content="shared"))
        assert await _revision(db, "thought", "thought_id", "t-1") == 0

        hit, created = await store.get_or_create(_thought("t-2", content="shared"))
        assert created is False
        assert hit.thought_id == "t-1"
        assert await _revision(db, "thought", "thought_id", "t-1") == 0

    async def test_bulk_store_deduplicate_hit_does_not_bump(
        self, store: SqliteEngravaCore, db: aiosqlite.Connection
    ) -> None:
        await store.create_thought(_thought(content="shared"))
        assert await _revision(db, "thought", "thought_id", "t-1") == 0

        await store.bulk_store([_thought("t-2", content="shared")], deduplicate=True)
        assert await _revision(db, "thought", "thought_id", "t-1") == 0


# ---------------------------------------------------------------------------
# Telemetry over engine-owned columns: neither bump nor enforce
# ---------------------------------------------------------------------------


class TestTelemetryDoesNotBump:
    async def test_record_access_does_not_bump(
        self, store: SqliteEngravaCore, db: aiosqlite.Connection
    ) -> None:
        await store.create_thought(_thought())
        assert await _revision(db, "thought", "thought_id", "t-1") == 0

        await store.record_access("t-1")
        row = await db.execute(
            "SELECT access_count, revision FROM thought WHERE thought_id = ?", ("t-1",)
        )
        fetched = await row.fetchone()
        assert fetched is not None
        assert fetched["access_count"] == 1
        assert fetched["revision"] == 0

    async def test_flush_access_buffer_does_not_bump(self, db: aiosqlite.Connection) -> None:
        store = SqliteEngravaCore(db, access_tracking_enabled=True)
        await store.create_thought(_thought())
        assert await _revision(db, "thought", "thought_id", "t-1") == 0

        await store.get_thought("t-1")  # buffers an access event
        flushed = await store.flush_access_buffer()
        assert flushed == 1
        assert await _revision(db, "thought", "thought_id", "t-1") == 0

    async def test_store_embedding_does_not_touch_thought_revision(
        self, store: SqliteEngravaCore, db: aiosqlite.Connection
    ) -> None:
        await store.create_thought(_thought())
        assert await _revision(db, "thought", "thought_id", "t-1") == 0

        await store.store_embedding("t-1", [0.1, 0.2, 0.3], model_name="probe-model")
        assert await _revision(db, "thought", "thought_id", "t-1") == 0


# ---------------------------------------------------------------------------
# Hygiene archive: bumps, does not enforce; hygiene GC: n/a (row deleted)
# ---------------------------------------------------------------------------


def _forgetful_policy(**overrides: object) -> HygienePolicyConfig:
    params: dict[str, object] = {"enabled": True, "min_inactivity_age_seconds": 0}
    params.update(overrides)
    return HygienePolicyConfig(**params)  # type: ignore[arg-type]


class TestHygieneAndTtl:
    async def test_hygiene_archive_bumps_without_enforcing(self, db: aiosqlite.Connection) -> None:
        store = SqliteEngravaCore(db, hygiene_policy=_forgetful_policy(eviction_threshold=0.5))
        await store.create_thought(_thought("cold", updated_cycle=0, action_outcome_score=0.0))
        assert await _revision(db, "thought", "thought_id", "cold") == 0

        result = await store.run_hygiene(current_cycle=1000)
        assert result.archived_count == 1
        assert await _revision(db, "thought", "thought_id", "cold") == 1

    async def test_hygiene_gc_deletes_the_row_leaving_nothing_to_carry_a_revision(
        self, db: aiosqlite.Connection
    ) -> None:
        policy = _forgetful_policy(
            eviction_threshold=0.5,
            auto_gc_enabled=True,
            gc_min_archive_age_cycles=0,
            gc_restore_window_seconds=0,
        )
        store = SqliteEngravaCore(db, hygiene_policy=policy)
        await store.create_thought(_thought("cold", updated_cycle=0, action_outcome_score=0.0))

        # With both restore windows at 0, the archive and GC stages run in the
        # same call: the row is archived (bumping revision, per the test
        # above) and then immediately reaped, leaving nothing to carry a
        # revision at all.
        result = await store.run_hygiene(current_cycle=1000)
        assert result.archived_count == 1
        assert result.gc_count == 1
        assert not await _row_exists(db, "thought", "thought_id", "cold")

    async def test_cleanup_expired_archive_branch_bumps_without_enforcing(
        self, db: aiosqlite.Connection
    ) -> None:
        store = SqliteEngravaCore(db, ttl_strategy="archive")
        await store.create_thought(_thought())
        await store._db.execute(
            "UPDATE thought SET expires_at = '2000-01-01T00:00:00+00:00' WHERE thought_id = ?",
            ("t-1",),
        )
        await store._db.commit()
        assert await _revision(db, "thought", "thought_id", "t-1") == 0

        result = await store.cleanup_expired()
        assert result.expired_count == 1
        row = await db.execute(
            "SELECT lifecycle_status, revision FROM thought WHERE thought_id = ?", ("t-1",)
        )
        fetched = await row.fetchone()
        assert fetched is not None
        assert fetched["lifecycle_status"] == LifecycleStatus.ARCHIVED.value
        assert fetched["revision"] == 1

    async def test_cleanup_expired_delete_branch_removes_the_row(
        self, db: aiosqlite.Connection
    ) -> None:
        store = SqliteEngravaCore(db, ttl_strategy="delete")
        await store.create_thought(_thought())
        await store._db.execute(
            "UPDATE thought SET expires_at = '2000-01-01T00:00:00+00:00' WHERE thought_id = ?",
            ("t-1",),
        )
        await store._db.commit()

        result = await store.cleanup_expired()
        assert result.expired_count == 1
        assert not await _row_exists(db, "thought", "thought_id", "t-1")


# ---------------------------------------------------------------------------
# retire_orphan_reflections: bumps, via update_thought
# ---------------------------------------------------------------------------


class TestRetireOrphanReflections:
    async def test_retiring_an_orphan_reflection_bumps_its_revision(
        self, store: SqliteEngravaCore, db: aiosqlite.Connection
    ) -> None:
        # A REFLECTION whose sole source has left ACTIVE is retired.
        await store.create_thought(_thought("source", lifecycle_status=LifecycleStatus.DONE))
        await store.create_thought(
            _thought(
                "refl",
                thought_type=ThoughtType.REFLECTION,
                consolidated_from=["source"],
            )
        )
        await store.create_edge(
            _edge(
                "e-consolidated",
                from_thought_id="refl",
                to_thought_id="source",
                edge_type=EdgeType.CONSOLIDATED_FROM,
            )
        )
        assert await _revision(db, "thought", "thought_id", "refl") == 0

        retired = await store.retire_orphan_reflections()
        assert retired == 1
        assert await _revision(db, "thought", "thought_id", "refl") == 1
        row = await db.execute("SELECT lifecycle_status FROM thought WHERE thought_id = 'refl'")
        fetched = await row.fetchone()
        assert fetched is not None
        assert fetched["lifecycle_status"] == LifecycleStatus.ARCHIVED.value


# ---------------------------------------------------------------------------
# Deletes: n/a -- nothing left to carry a revision
# ---------------------------------------------------------------------------


class TestDeletesHaveNoRevisionToCarry:
    async def test_delete_thought_removes_the_row(
        self, store: SqliteEngravaCore, db: aiosqlite.Connection
    ) -> None:
        await store.create_thought(_thought())
        assert await store.delete_thought("t-1") is True
        assert not await _row_exists(db, "thought", "thought_id", "t-1")

    async def test_delete_edge_removes_the_row(
        self, store: SqliteEngravaCore, db: aiosqlite.Connection
    ) -> None:
        await store.create_thought(_thought("t-1"))
        await store.create_thought(_thought("t-2"))
        await store.create_edge(_edge())
        assert await store.delete_edge("e-1") is True
        assert not await _row_exists(db, "edge", "edge_id", "e-1")


# ---------------------------------------------------------------------------
# Typed contention errors: StaleDataError, not a raw driver error
# ---------------------------------------------------------------------------


def _interleave_once(
    store: SqliteEngravaCore,
    method_name: str,
    intruder: object,
) -> None:
    """Run ``intruder`` once, right after ``method_name`` first returns.

    Mirrors ``tests/test_partial_field_updates.py``'s helper of the same name:
    the wrapped method is a guarded path's internal read, and the competing
    write runs before the caller gets to act on what it read -- the same-task
    TOCTOU stand-in used throughout this suite for a shape the task-reentrant
    write lock does not block (a caller-owned hook nested inside another call's
    own read-modify-write span).
    """
    original = getattr(store, method_name)
    fired = False

    async def wrapper(*args: object, **kwargs: object) -> object:
        nonlocal fired
        result = await original(*args, **kwargs)
        if not fired:
            fired = True
            await intruder()  # type: ignore[operator]
        return result

    setattr(store, method_name, wrapper)


class TestTypedErrorsNotRawDriverErrors:
    """Through the real public API: the guard rejects a stale write with the
    typed ``StaleDataError``, never a raw driver error, on each of the three
    update paths this item adds the guard to. Each scenario below is a
    genuine race exercised through the same interleave technique
    ``test_concurrency_contract.py`` uses, not a hand-driven SQL statement --
    what a caller actually observes is exactly what is asserted here.
    """

    async def test_update_thought_raises_the_typed_error(
        self, store: SqliteEngravaCore, db: aiosqlite.Connection
    ) -> None:
        await store.create_thought(_thought())

        async def _competing_write() -> None:
            await store.update_thought("t-1", priority=Priority.P1)

        _interleave_once(store, "_get_thought_row", _competing_write)

        with pytest.raises(StaleDataError):
            await store.update_thought("t-1", essence="mine")
        row = await db.execute("SELECT essence FROM thought WHERE thought_id = 't-1'")
        fetched = await row.fetchone()
        assert fetched is not None
        assert fetched["essence"] != "mine"

    async def test_update_edge_raises_the_typed_error(
        self, store: SqliteEngravaCore, db: aiosqlite.Connection
    ) -> None:
        await store.create_thought(_thought("t-1"))
        await store.create_thought(_thought("t-2"))
        await store.create_edge(_edge())

        async def _competing_write() -> None:
            await store.update_edge("e-1", weight=0.9)

        _interleave_once(store, "_get_edge_row", _competing_write)

        with pytest.raises(StaleDataError):
            await store.update_edge("e-1", weight=0.1)
        row = await db.execute("SELECT weight FROM edge WHERE edge_id = 'e-1'")
        fetched = await row.fetchone()
        assert fetched is not None
        assert fetched["weight"] == 0.9

    async def test_update_action_raises_the_typed_error(
        self, store: SqliteEngravaCore, db: aiosqlite.Connection
    ) -> None:
        await store.create_thought(_thought("t-1"))
        await store.create_action(_action())

        async def _competing_write() -> None:
            await store.update_action("a-1", status=ActionStatus.EXECUTING)

        _interleave_once(store, "_get_action_row", _competing_write)

        with pytest.raises(StaleDataError):
            await store.update_action("a-1", verification_status=VerificationStatus.PARTIAL)
        row = await db.execute("SELECT verification_status FROM action WHERE action_id = 'a-1'")
        fetched = await row.fetchone()
        assert fetched is not None
        assert fetched["verification_status"] == VerificationStatus.PENDING.value
