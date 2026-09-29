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
* The documented consequence of running *without* that migration:
  on a pre-cascade (core-11) database a deleted thought's ``embedding``
  row survives, so ``sync_embeddings`` puts its vector back and
  ``search_similar`` keeps returning the deleted **identifier** while the
  content is gone. Pinned against the head-schema control, which does not.

Every assertion uses raw SQLite reads (not the public ORM) so the
data-layer guarantees stay visible even if the higher-level API ever
masks them.
"""

from __future__ import annotations

import asyncio
import datetime
import importlib.util
import logging
import uuid
from dataclasses import dataclass
from importlib import resources
from typing import TYPE_CHECKING
from unittest.mock import patch

import aiosqlite
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from engrava.config import HygienePolicyConfig
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
    ConnectionQuarantinedError,
    CoreMigrationError,
    DuplicateEdgeError,
    ReferentialIntegrityError,
    SchemaVersionError,
)
from engrava.domain.models.action import ActionRecord
from engrava.domain.models.edge import EdgeRecord
from engrava.domain.models.thought import ThoughtRecord
from engrava.infrastructure.sqlite.engrava_core import SqliteEngravaCore
from engrava.infrastructure.sqlite.vector_sqlite_vec import SqliteVecSearchBackend
from tests.test_migration_upgrade_chains import _bootstrap_core_at_version

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
        assert row[0] == 21


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


class TestDeleteThoughtChildrenAtomicity:
    """``delete_thought`` and its child deletes when a trigger rejects an ``action`` delete.

    With foreign keys on, the parent delete's cascade reaches the trigger; with them off, the
    explicit child deletes do. The tests of a rejected ``delete_thought`` cover both. One test
    calls ``_delete_thought_children_explicit`` directly inside a transaction it holds and pins
    that the call undoes its earlier child deletes and leaves that transaction open. Two further
    tests raise ``CancelledError`` from a statement on the children savepoint and pin that the
    connection is quarantined.
    """

    @pytest.mark.parametrize("foreign_keys_on", [True, False])
    async def test_a_rejected_action_delete_leaves_the_thought_and_all_its_children_in_place(
        self,
        store: SqliteEngravaCore,
        foreign_keys_on: bool,
    ) -> None:
        if not foreign_keys_on:
            await store._db.execute("PRAGMA foreign_keys = OFF")
        await store.create_thought(_make_thought("t1"))
        await store.create_thought(_make_thought("t2"))
        await store.create_edge(_make_edge("e-out", "t1", "t2"))
        await store.create_edge(_make_edge("e-in", "t2", "t1"))
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
        await store.create_action(_make_action("a1", "t1"))

        # Rejects every delete on ``action``.
        await store._db.execute(
            "CREATE TRIGGER reject_action_delete BEFORE DELETE ON action "
            "BEGIN SELECT RAISE(ABORT, 'policy: action rows are retained'); END"
        )

        with pytest.raises(aiosqlite.IntegrityError):
            await store.delete_thought("t1")

        # This unrelated write commits its own transaction; it must not make any
        # part of the failed delete durable.
        await store.create_thought(_make_thought("unrelated-write"))

        assert await store.get_thought("t1") is not None
        cursor = await store._db.execute(
            "SELECT edge_id FROM edge WHERE edge_id IN ('e-out', 'e-in')"
        )
        surviving_edges = {row["edge_id"] for row in await cursor.fetchall()}
        assert surviving_edges == {"e-out", "e-in"}, "the edges must still be there"
        cursor = await store._db.execute("SELECT COUNT(*) FROM embedding WHERE owner_id = 't1'")
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == 1, "the embedding must still be there"
        cursor = await store._db.execute(
            "SELECT COUNT(*) FROM action WHERE source_thought_id = 't1'"
        )
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == 1, "the rejected action row itself must still be there too"

    @pytest.mark.parametrize("foreign_keys_on", [True, False])
    async def test_action_delete_rollback_trigger_surfaces_its_own_error(
        self,
        store: SqliteEngravaCore,
        foreign_keys_on: bool,
    ) -> None:
        """A ``RAISE(ROLLBACK, ...)`` trigger: the caller sees it, not a savepoint error.

        Unlike ``RAISE(ABORT, ...)`` (the previous test), ``RAISE(ROLLBACK,
        ...)`` ends the *entire* transaction, taking the savepoint down with
        it. Naively retrying ``ROLLBACK TO`` against a savepoint that is
        already gone raises ``"no such savepoint"`` and replaces the real
        failure — this pins that the caller sees the trigger's own message
        instead. Each earlier ``create_*`` call commits on its own, so by the
        time ``delete_thought`` opens its own transaction there is nothing
        else pending for the trigger's rollback to take down except that
        call's own (otherwise-uncommitted) work.
        """
        if not foreign_keys_on:
            await store._db.execute("PRAGMA foreign_keys = OFF")
        await store.create_thought(_make_thought("t1"))
        await store.create_action(_make_action("a1", "t1"))
        await store._db.execute(
            "CREATE TRIGGER reject_action_delete_rollback BEFORE DELETE ON action "
            "BEGIN SELECT RAISE(ROLLBACK, 'policy: rollback rejection'); END"
        )

        with pytest.raises(aiosqlite.IntegrityError, match="policy: rollback rejection"):
            await store.delete_thought("t1")

        assert not store._db.in_transaction, (
            "a RAISE(ROLLBACK) trigger already closes the transaction; nothing should be left open"
        )
        assert await store.get_thought("t1") is not None
        cursor = await store._db.execute(
            "SELECT COUNT(*) FROM action WHERE source_thought_id = 't1'"
        )
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == 1

    async def test_a_rejected_action_delete_undoes_the_child_deletes_before_it(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """The edge and embedding deletes run before the ``action`` delete the trigger rejects.

        ``_delete_thought_children_explicit`` is called directly inside a transaction this test
        opens. The call raises, its edge and embedding deletes are undone, and the transaction is
        still open.
        """
        db = store._db
        await store.create_thought(_make_thought("t1"))
        await store.create_thought(_make_thought("t2"))
        await store.create_edge(_make_edge("e-out", "t1", "t2"))
        await store.create_edge(_make_edge("e-in", "t2", "t1"))
        await db.execute(
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
        await store.create_action(_make_action("a1", "t1"))
        await db.execute(
            "CREATE TRIGGER reject_action_delete BEFORE DELETE ON action "
            "BEGIN SELECT RAISE(ABORT, 'policy: action rows are retained'); END"
        )
        assert not db.in_transaction

        await db.execute("BEGIN")
        with pytest.raises(aiosqlite.IntegrityError, match="policy: action rows are retained"):
            await store._delete_thought_children_explicit("t1")

        assert db.in_transaction, "the transaction the caller opened must still be open"
        cursor = await db.execute("SELECT edge_id FROM edge WHERE edge_id IN ('e-out', 'e-in')")
        surviving_edges = {row["edge_id"] for row in await cursor.fetchall()}
        assert surviving_edges == {"e-out", "e-in"}, "the edge deletes must have been undone"
        cursor = await db.execute("SELECT COUNT(*) FROM embedding WHERE owner_id = 't1'")
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == 1, "the embedding delete must have been undone"
        cursor = await db.execute("SELECT COUNT(*) FROM action WHERE source_thought_id = 't1'")
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == 1, "the rejected action row itself must still be there too"
        await db.rollback()

    async def test_cancellation_after_release_with_a_caller_transaction_quarantines(
        self,
    ) -> None:
        """A cancellation racing a completed RELEASE quarantines instead of guessing.

        Simulates the exact race a real database was confirmed to hit: the
        ``RELEASE`` genuinely executes (its SQL runs against the connection
        below), but this coroutine observes ``CancelledError`` instead of
        that success — aiosqlite does not cancel a statement already queued
        on its worker thread. With a transaction already open *before*
        ``delete_thought`` is ever called (``suspend_auto_commit`` gives one
        exactly like TTL cleanup or hygiene GC batching several deletes
        would), ``opened_transaction`` is ``False``, so this method cannot
        fall back to a full ``rollback()`` of its own — the savepoint is
        gone (``RELEASE`` already consumed it) but the transaction is not,
        and a bare retried ``ROLLBACK TO`` fails. The only safe move left is
        to quarantine: a later, unrelated write must be refused, never
        allowed to commit the three already-applied child deletes as a
        side effect.

        Builds its own connection rather than using the shared ``store``
        fixture: quarantine detaches the real connection and schedules its
        own close as an independent, un-awaited task (by design — see
        ``_quarantine_connection``), and racing that against the fixture's
        own ``async with aiosqlite.connect(...)`` teardown closing the same
        connection a second time is a test-harness hazard, not something
        this test is about.
        """
        db = await aiosqlite.connect(":memory:")
        db.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(db, embedding_provider=None, auto_embed=False)
        await store.ensure_schema()

        await store.create_thought(_make_thought("t1"))
        await store.create_thought(_make_thought("t2"))
        await store.create_edge(_make_edge("e1", "t1", "t2"))

        real_execute = db.execute
        released = {"count": 0}

        async def _execute_and_cancel_after_release(
            sql: str, *args: object, **kwargs: object
        ) -> object:
            cursor = await real_execute(sql, *args, **kwargs)
            if sql == "RELEASE delete_thought_children" and released["count"] == 0:
                released["count"] += 1
                raise asyncio.CancelledError
            return cursor

        async def _delete_inside_the_callers_transaction() -> None:
            async with store.suspend_auto_commit():
                # A transaction the caller already holds, open before
                # delete_thought is ever called -- opened_transaction is
                # False inside the helper for this call.
                await store.create_thought(_make_thought("t3"))
                assert db.in_transaction

                db.execute = _execute_and_cancel_after_release
                with pytest.raises(asyncio.CancelledError):
                    await store.delete_thought("t1")
                # Falling through to let this ``async with`` block exit:
                # suspend_auto_commit's own exit touches the connection
                # (checking whether to commit), and the connection is
                # already quarantined by this point -- that exit is
                # expected to raise too, which is exactly the point: every
                # subsequent touch of this connection fails, not only a
                # fresh, unrelated write.

        with pytest.raises(ConnectionQuarantinedError):
            await _delete_inside_the_callers_transaction()

        assert store._connection_quarantined is True
        with pytest.raises(ConnectionQuarantinedError):
            await store.create_thought(_make_thought("unrelated-after-quarantine"))

        # Quarantine's own physical close is deliberately detached (see the
        # docstring above) so *safety* never depends on it, but that leaves
        # its task still in flight when this test function returns -- and
        # pytest-asyncio closes this test's event loop immediately after.
        # Awaiting it here (never done in production, where the loop keeps
        # running) lets the close finish on the loop it was scheduled on,
        # instead of leaking aiosqlite's non-daemon worker thread to call
        # back into a now-closed loop from some unrelated, later test.
        if store._quarantine_close_task is not None:
            await store._quarantine_close_task

    async def test_cancellation_during_unwind_wins_over_the_original_error(
        self,
    ) -> None:
        """A cancellation that interrupts the unwind itself must not be swallowed.

        The original failure here is an ordinary ``RAISE(ABORT, ...)``
        trigger veto — ``delete_thought`` should recover from that cleanly
        (see the first test in this class). But if the recovery attempt
        (``ROLLBACK TO``) is itself cancelled, the cancellation must win:
        the caller sees ``CancelledError``, not the trigger's error, because
        a cleanup path that can defeat a cancellation makes shutdown and
        timeout both unreliable.

        Builds its own connection rather than using the shared ``store``
        fixture — see the previous test's docstring for why. Foreign-key
        enforcement is turned off on it, deliberately: ``delete_thought``
        now deletes the parent row first, so on an enforcement-on
        connection the parent statement's own cascade — not the explicit
        child delete this test targets — is what reaches the ``action``
        trigger, and the veto is caught by ``_delete_thought_atomic``'s
        savepoint before ``_delete_thought_children_explicit`` ever runs.
        With enforcement off there is no cascade to pre-empt it, so the
        explicit delete is what hits the trigger and the race this test
        pins — a cancellation landing during *that* method's own
        ``ROLLBACK TO`` — stays reachable, exactly as it is on a
        pre-core-12 schema or any other connection with enforcement off.
        """
        db = await aiosqlite.connect(":memory:")
        db.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(db, embedding_provider=None, auto_embed=False)
        await store.ensure_schema()
        await db.execute("PRAGMA foreign_keys = OFF")

        await store.create_thought(_make_thought("t1"))
        await store.create_action(_make_action("a1", "t1"))
        await db.execute(
            "CREATE TRIGGER reject_action_delete_for_cancel BEFORE DELETE ON action "
            "BEGIN SELECT RAISE(ABORT, 'policy: action rows are retained'); END"
        )

        real_execute = db.execute
        rollback_to_attempts = {"count": 0}

        async def _cancel_the_rollback_to(sql: str, *args: object, **kwargs: object) -> object:
            if sql == "ROLLBACK TO delete_thought_children" and rollback_to_attempts["count"] == 0:
                rollback_to_attempts["count"] += 1
                raise asyncio.CancelledError
            return await real_execute(sql, *args, **kwargs)

        db.execute = _cancel_the_rollback_to
        with pytest.raises(asyncio.CancelledError):
            await store.delete_thought("t1")

        assert store._connection_quarantined is True
        with pytest.raises(ConnectionQuarantinedError):
            await store.create_thought(_make_thought("unrelated-after-quarantine"))

        # See the previous test's closing comment: awaiting quarantine's
        # detached close task here keeps it from outliving this test's event
        # loop, which production code never has to do because its own loop
        # keeps running past this point.
        if store._quarantine_close_task is not None:
            await store._quarantine_close_task


class TestParentDeleteSeesChildrenBeforeTheyAreGone:
    """The parent delete must run before the children are gone, not after.

    ``6e4ed41`` deleted the edge / embedding / action rows first and released
    their savepoint before the ``thought`` row itself was ever touched. Two
    distinct defects followed from that ordering, both fixed by
    ``_delete_thought_atomic`` running the parent delete first, inside the
    savepoint that then covers the explicit child deletes:

    * **Lost atomicity.** Anything that then prevented, diverted or skipped
      the parent delete -- a ``BEFORE DELETE ON thought`` trigger that always
      vetoes, the concrete case exercised here -- left the already-deleted
      children sitting in the open transaction with the parent still
      present. Nothing in the raised exception said so; the children were
      simply gone the moment any later, unrelated write on the same
      connection committed.
    * **A defeated guard.** A trigger that vetoes only *conditionally* --
      ``WHEN EXISTS (... the thought still has children ...)`` -- never saw
      them: the predicate it tests is already false by the time the parent
      delete runs, so the trigger never fires and the delete the user's own
      policy meant to block **succeeds**.

    Each test installs its trigger directly on ``thought`` (not on a child
    table -- ``TestDeleteThoughtChildrenAtomicity`` above already covers a
    child table's own trigger, which is a different failure this class does
    not repeat) so the deciding question is what the parent delete itself
    observes and how its failure is handled.
    """

    async def test_delete_thought_plain_veto_does_not_lose_the_children(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        await store.create_thought(_make_thought("t1"))
        await store.create_thought(_make_thought("t2"))
        await store.create_edge(_make_edge("e1", "t1", "t2"))
        await store.create_action(_make_action("a1", "t1"))
        await store._db.execute(
            "CREATE TRIGGER thought_delete_forbidden BEFORE DELETE ON thought "
            "BEGIN SELECT RAISE(ABORT, 'policy: thought deletion forbidden'); END"
        )

        with pytest.raises(aiosqlite.IntegrityError):
            await store.delete_thought("t1")

        # The discriminating step, exactly like the child-table atomicity
        # tests above: an unrelated write's own commit must not durably
        # apply a children-delete the veto above never got to protect.
        await store.create_thought(_make_thought("unrelated-write"))

        assert await store.get_thought("t1") is not None
        cursor = await store._db.execute("SELECT COUNT(*) FROM edge WHERE edge_id = 'e1'")
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == 1, "the edge must survive a vetoed parent delete"
        cursor = await store._db.execute(
            "SELECT COUNT(*) FROM action WHERE source_thought_id = 't1'"
        )
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == 1, "the action must survive a vetoed parent delete"

    async def test_delete_thought_when_exists_guard_fires_again(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        await store.create_thought(_make_thought("t1"))
        await store.create_thought(_make_thought("t2"))
        await store.create_edge(_make_edge("e1", "t1", "t2"))
        await store._db.execute(
            "CREATE TRIGGER thought_delete_guard BEFORE DELETE ON thought "
            "WHEN EXISTS (SELECT 1 FROM edge WHERE from_thought_id = OLD.thought_id "
            "OR to_thought_id = OLD.thought_id) "
            "BEGIN SELECT RAISE(ABORT, 'policy: cannot delete a thought with live edges'); END"
        )

        with pytest.raises(aiosqlite.IntegrityError):
            await store.delete_thought("t1")

        assert await store.get_thought("t1") is not None, "the guard must block the delete"
        cursor = await store._db.execute("SELECT COUNT(*) FROM edge WHERE edge_id = 'e1'")
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == 1, "the edge the guard tests for must survive"

    async def test_ttl_delete_plain_veto_does_not_lose_the_children(
        self,
        delete_store: SqliteEngravaCore,
    ) -> None:
        past = "2026-01-01T00:00:00+00:00"
        now = "2026-06-01T00:00:00+00:00"
        await delete_store.create_thought(_make_thought("t1", expires_at=past))
        await delete_store.create_thought(_make_thought("t2"))
        await delete_store.create_edge(_make_edge("e1", "t1", "t2"))
        await delete_store._db.execute(
            "CREATE TRIGGER thought_delete_forbidden BEFORE DELETE ON thought "
            "BEGIN SELECT RAISE(ABORT, 'policy: thought deletion forbidden'); END"
        )

        with pytest.raises(aiosqlite.IntegrityError):
            await delete_store.cleanup_expired(now=now)

        assert await delete_store.get_thought("t1") is not None
        cursor = await delete_store._db.execute("SELECT COUNT(*) FROM edge WHERE edge_id = 'e1'")
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == 1, "the edge must survive a vetoed parent delete"

    async def test_ttl_delete_when_exists_guard_fires_again(
        self,
        delete_store: SqliteEngravaCore,
    ) -> None:
        past = "2026-01-01T00:00:00+00:00"
        now = "2026-06-01T00:00:00+00:00"
        await delete_store.create_thought(_make_thought("t1", expires_at=past))
        await delete_store.create_thought(_make_thought("t2"))
        await delete_store.create_edge(_make_edge("e1", "t1", "t2"))
        await delete_store._db.execute(
            "CREATE TRIGGER thought_delete_guard BEFORE DELETE ON thought "
            "WHEN EXISTS (SELECT 1 FROM edge WHERE from_thought_id = OLD.thought_id "
            "OR to_thought_id = OLD.thought_id) "
            "BEGIN SELECT RAISE(ABORT, 'policy: cannot delete a thought with live edges'); END"
        )

        with pytest.raises(aiosqlite.IntegrityError):
            await delete_store.cleanup_expired(now=now)

        assert await delete_store.get_thought("t1") is not None, "the guard must block the delete"
        cursor = await delete_store._db.execute("SELECT COUNT(*) FROM edge WHERE edge_id = 'e1'")
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == 1

    async def test_hygiene_gc_plain_veto_does_not_lose_the_children(self) -> None:
        policy = HygienePolicyConfig(
            enabled=True,
            eviction_threshold=0.0,
            auto_gc_enabled=True,
            gc_min_archive_age_cycles=0,
            gc_restore_window_seconds=0,
        )
        async with aiosqlite.connect(":memory:") as db:
            db.row_factory = aiosqlite.Row
            hygiene_store = SqliteEngravaCore(
                db, embedding_provider=None, auto_embed=False, hygiene_policy=policy
            )
            await hygiene_store.ensure_schema()

            await hygiene_store.create_thought(_make_thought("t1"))
            await hygiene_store.create_thought(_make_thought("t2"))
            await hygiene_store.create_edge(_make_edge("e1", "t1", "t2"))
            await hygiene_store.update_thought(
                "t1", lifecycle_status=LifecycleStatus.ARCHIVED, archived_at_cycle=0
            )
            await hygiene_store._db.execute(
                "CREATE TRIGGER thought_delete_forbidden BEFORE DELETE ON thought "
                "BEGIN SELECT RAISE(ABORT, 'policy: thought deletion forbidden'); END"
            )

            with pytest.raises(aiosqlite.IntegrityError):
                await hygiene_store.run_hygiene(current_cycle=1000)

            assert await hygiene_store.get_thought("t1") is not None
            cursor = await hygiene_store._db.execute(
                "SELECT COUNT(*) FROM edge WHERE edge_id = 'e1'"
            )
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == 1, "the edge must survive a vetoed parent delete"

    async def test_hygiene_gc_when_exists_guard_fires_again(self) -> None:
        policy = HygienePolicyConfig(
            enabled=True,
            eviction_threshold=0.0,
            auto_gc_enabled=True,
            gc_min_archive_age_cycles=0,
            gc_restore_window_seconds=0,
        )
        async with aiosqlite.connect(":memory:") as db:
            db.row_factory = aiosqlite.Row
            hygiene_store = SqliteEngravaCore(
                db, embedding_provider=None, auto_embed=False, hygiene_policy=policy
            )
            await hygiene_store.ensure_schema()

            await hygiene_store.create_thought(_make_thought("t1"))
            await hygiene_store.create_thought(_make_thought("t2"))
            await hygiene_store.create_edge(_make_edge("e1", "t1", "t2"))
            await hygiene_store.update_thought(
                "t1", lifecycle_status=LifecycleStatus.ARCHIVED, archived_at_cycle=0
            )
            await hygiene_store._db.execute(
                "CREATE TRIGGER thought_delete_guard BEFORE DELETE ON thought "
                "WHEN EXISTS (SELECT 1 FROM edge WHERE from_thought_id = OLD.thought_id "
                "OR to_thought_id = OLD.thought_id) "
                "BEGIN SELECT RAISE(ABORT, 'policy: cannot delete a thought with live edges'); END"
            )

            with pytest.raises(aiosqlite.IntegrityError):
                await hygiene_store.run_hygiene(current_cycle=1000)

            assert await hygiene_store.get_thought("t1") is not None, (
                "the guard must block the delete"
            )
            cursor = await hygiene_store._db.execute(
                "SELECT COUNT(*) FROM edge WHERE edge_id = 'e1'"
            )
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == 1


class TestParentDeleteSuppressedByRaiseIgnore:
    """``RAISE(IGNORE)`` reproduces the same data loss with a different trigger.

    ``RAISE(ABORT)`` (and a ``WHEN EXISTS`` guard, which uses it too) raises,
    so ``_delete_thought_atomic``'s own ``except`` branch unwinds the whole
    savepoint and the caller sees an exception -- covered above by
    ``TestParentDeleteSeesChildrenBeforeTheyAreGone``. ``RAISE(IGNORE)``
    instead *silently* reverts just the parent ``DELETE``: no exception, the
    thought stays, and the delete's own rowcount is zero -- indistinguishable
    from ``thought_id`` never having existed if rowcount is all that is
    consulted. Before the fix pinned here, ``_delete_thought_atomic`` ran the
    explicit child sweep unconditionally on that zero-row outcome, so a
    still-live parent lost its edges and its actions anyway, and
    ``delete_thought`` (or TTL cleanup, or hygiene GC) reported ``False``
    while having actually done the damage. TTL cleanup additionally never
    checked the return value at all, so it went on to purge the vector and
    append a ``DELETE_THOUGHT`` journal entry for a parent that was never
    deleted -- false history on top of the data loss.

    Exercised on both ``PRAGMA foreign_keys`` settings: the veto prevents the
    parent row from ever actually being removed, so no cascade fires either
    way, and the two settings are expected to behave identically here --
    pinning both turns that equivalence into a tested fact instead of an
    assumption. The last test in this class is the control: a genuinely
    nonexistent ``thought_id`` must keep sweeping any orphaned children it
    finds, exactly as before this fix.
    """

    @staticmethod
    async def _install_ignore_trigger(db: aiosqlite.Connection) -> None:
        await db.execute(
            "CREATE TRIGGER thought_delete_ignore BEFORE DELETE ON thought "
            "BEGIN SELECT RAISE(IGNORE); END"
        )

    @pytest.mark.parametrize("foreign_keys_on", [True, False])
    async def test_delete_thought_ignore_veto_does_not_lose_the_children(
        self,
        foreign_keys_on: bool,
    ) -> None:
        async with aiosqlite.connect(":memory:") as db:
            db.row_factory = aiosqlite.Row
            store = SqliteEngravaCore(db, embedding_provider=None, auto_embed=False)
            await store.ensure_schema()
            if not foreign_keys_on:
                await db.execute("PRAGMA foreign_keys = OFF")

            await store.create_thought(_make_thought("t1"))
            await store.create_thought(_make_thought("t2"))
            await store.create_edge(_make_edge("e1", "t1", "t2"))
            await store.create_action(_make_action("a1", "t1"))
            await self._install_ignore_trigger(db)

            deleted = await store.delete_thought("t1")

            assert deleted is False
            assert await store.get_thought("t1") is not None, "the parent must survive"
            cursor = await db.execute("SELECT COUNT(*) FROM edge WHERE edge_id = 'e1'")
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == 1, "the edge must survive a silently-ignored parent delete"
            cursor = await db.execute("SELECT COUNT(*) FROM action WHERE source_thought_id = 't1'")
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == 1, "the action must survive a silently-ignored parent delete"

    @pytest.mark.parametrize("foreign_keys_on", [True, False])
    async def test_ttl_delete_ignore_veto_does_not_lose_the_children(
        self,
        foreign_keys_on: bool,
    ) -> None:
        async with aiosqlite.connect(":memory:") as db:
            db.row_factory = aiosqlite.Row
            store = SqliteEngravaCore(
                db, embedding_provider=None, auto_embed=False, ttl_strategy="delete"
            )
            await store.ensure_schema()
            if not foreign_keys_on:
                await db.execute("PRAGMA foreign_keys = OFF")

            past = "2026-01-01T00:00:00+00:00"
            now = "2026-06-01T00:00:00+00:00"
            await store.create_thought(_make_thought("t1", expires_at=past))
            await store.create_thought(_make_thought("t2"))
            await store.create_edge(_make_edge("e1", "t1", "t2"))
            await self._install_ignore_trigger(db)

            result = await store.cleanup_expired(now=now)

            assert result.expired_count == 1
            assert await store.get_thought("t1") is not None, "the parent must survive"
            cursor = await db.execute("SELECT COUNT(*) FROM edge WHERE edge_id = 'e1'")
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == 1, "the edge must survive a silently-ignored parent delete"

    @pytest.mark.parametrize("foreign_keys_on", [True, False])
    async def test_hygiene_gc_ignore_veto_does_not_lose_the_children(
        self,
        foreign_keys_on: bool,
    ) -> None:
        policy = HygienePolicyConfig(
            enabled=True,
            eviction_threshold=0.0,
            auto_gc_enabled=True,
            gc_min_archive_age_cycles=0,
            gc_restore_window_seconds=0,
        )
        async with aiosqlite.connect(":memory:") as db:
            db.row_factory = aiosqlite.Row
            store = SqliteEngravaCore(
                db, embedding_provider=None, auto_embed=False, hygiene_policy=policy
            )
            await store.ensure_schema()
            if not foreign_keys_on:
                await db.execute("PRAGMA foreign_keys = OFF")

            await store.create_thought(_make_thought("t1"))
            await store.create_thought(_make_thought("t2"))
            await store.create_edge(_make_edge("e1", "t1", "t2"))
            await store.update_thought(
                "t1", lifecycle_status=LifecycleStatus.ARCHIVED, archived_at_cycle=0
            )
            await self._install_ignore_trigger(db)

            await store.run_hygiene(current_cycle=1000)

            assert await store.get_thought("t1") is not None, "the parent must survive"
            cursor = await db.execute("SELECT COUNT(*) FROM edge WHERE edge_id = 'e1'")
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == 1, "the edge must survive a silently-ignored parent delete"

    async def test_nonexistent_id_control_still_sweeps_orphans_and_reports_false(
        self,
    ) -> None:
        """The discrimination must not regress the pre-existing nonexistent-id path.

        With foreign keys off (no cascade to rely on), a schema can be left
        carrying orphaned child rows for a ``thought_id`` that was never
        (re)inserted into ``thought`` -- exactly the scenario the
        unconditional sweep in ``_delete_thought_atomic`` exists to clean up.
        This must keep working after the fix: ``existed_before`` is False, so
        the sweep still runs and ``delete_thought`` still reports ``False``,
        with no ``RAISE(IGNORE)`` trigger involved at all.
        """
        async with aiosqlite.connect(":memory:") as db:
            db.row_factory = aiosqlite.Row
            store = SqliteEngravaCore(db, embedding_provider=None, auto_embed=False)
            await store.ensure_schema()
            await db.execute("PRAGMA foreign_keys = OFF")

            await store.create_thought(_make_thought("t2"))
            # Enforcement is off, so this persists an edge referencing a
            # `from_thought_id` that was never inserted into `thought` --
            # the orphan a pre-cascade schema (or a connection with
            # enforcement off) can carry.
            await store.create_edge(_make_edge("e1", "ghost", "t2"))

            deleted = await store.delete_thought("ghost")

            assert deleted is False
            cursor = await db.execute("SELECT COUNT(*) FROM edge WHERE edge_id = 'e1'")
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == 0, "the orphaned edge must still be swept for a nonexistent id"
            assert db.in_transaction is False, (
                "the sweep is a real write even though `deleted` is False -- "
                "delete_thought must still commit it, not mistake `deleted is "
                "False` for 'this call wrote nothing' and roll it back"
            )


class TestChildTriggerWriteSurvivesAVetoedSweep:
    """``wrote_anything`` must reflect ``total_changes``, not a per-delete rowcount.

    ``_delete_thought_children_explicit`` used to OR together the ``rowcount``
    of its three deletes. A ``BEFORE DELETE`` trigger on any of the three
    child tables can write a real row of its own (an audit entry, say) and
    then veto its own statement with ``RAISE(IGNORE)`` -- which reverts only
    that statement, not the trigger's earlier writes, but leaves every
    rowcount involved at zero. The old computation therefore reported
    ``wrote_anything=False`` for a call that really did write something, and
    ``delete_thought`` rolled that write back instead of committing it.

    The two tests below share one scaffold (the same audit table and
    trigger) and differ only in whether the trigger ever actually fires --
    proving the fix discriminates the two states rather than merely
    happening to pass on one of them.
    """

    async def test_trigger_write_before_the_veto_is_durable(self, tmp_path: Path) -> None:
        """The trigger's own write must survive even though the sweep saw rowcount zero."""
        db_path = tmp_path / "child-trigger-write-durable.sqlite"
        db = await aiosqlite.connect(str(db_path))
        try:
            db.row_factory = aiosqlite.Row
            store = SqliteEngravaCore(db, embedding_provider=None, auto_embed=False)
            await store.ensure_schema()
            await db.execute("PRAGMA foreign_keys = OFF")
            await store.create_thought(_make_thought("t2"))
            # An orphaned edge for a `thought_id` that was never inserted --
            # only creatable with enforcement off, exactly like the sweep's
            # other tests above.
            await store.create_edge(_make_edge("e1", "ghost", "t2"))
            await db.execute("CREATE TABLE audit(note TEXT)")
            await db.execute(
                "CREATE TRIGGER edge_delete_audit BEFORE DELETE ON edge "
                "BEGIN INSERT INTO audit VALUES ('swept'); SELECT RAISE(IGNORE); END"
            )
            await db.commit()

            deleted = await store.delete_thought("ghost")

            assert deleted is False, "'ghost' was never a live thought"
            assert await store.get_thought("ghost") is None

            # Durability, not merely visibility on the writer's own
            # connection: an uncommitted row on `db` would be indistinguishable
            # from a durable one there, so this reads from a second, separate
            # connection to the same file.
            other = await aiosqlite.connect(str(db_path))
            try:
                cursor = await other.execute("SELECT COUNT(*) FROM audit")
                row = await cursor.fetchone()
                assert row is not None
                assert row[0] == 1, (
                    "the trigger's own audit insert ran before it vetoed the "
                    "edge delete, and must be committed even though the "
                    "vetoed delete's own rowcount was zero"
                )
            finally:
                await other.close()
        finally:
            await db.close()

    async def test_a_veto_that_writes_nothing_commits_nothing(self, tmp_path: Path) -> None:
        """The same trigger, never fired, must leave a caller's own transaction alone."""
        db_path = tmp_path / "child-trigger-write-free.sqlite"
        db = await aiosqlite.connect(str(db_path))
        try:
            db.row_factory = aiosqlite.Row
            store = SqliteEngravaCore(db, embedding_provider=None, auto_embed=False)
            await store.ensure_schema()
            await store.create_thought(_make_thought("unrelated"))
            await db.execute("CREATE TABLE audit(note TEXT)")
            await db.execute(
                "CREATE TRIGGER edge_delete_audit BEFORE DELETE ON edge "
                "BEGIN INSERT INTO audit VALUES ('swept'); SELECT RAISE(IGNORE); END"
            )
            await db.commit()

            # No orphaned edge, embedding, or action row exists for "ghost" --
            # the trigger above is installed but has nothing to fire on, so
            # this call is genuinely write-free, not merely vetoed.
            await db.execute("BEGIN")
            await db.execute(
                "UPDATE thought SET essence = ? WHERE thought_id = ?",
                ("edited-by-caller", "unrelated"),
            )

            deleted = await store.delete_thought("ghost")

            assert deleted is False
            assert db.in_transaction is True, (
                "the caller's own transaction, with their pending edit still "
                "inside it, must still be open after a call that wrote "
                "nothing of its own"
            )
            await db.rollback()

            row = await store.get_thought("unrelated")
            assert row is not None
            assert row.essence == "essence-unrelated", (
                "the caller's rollback must undo their own edit -- a "
                "genuinely write-free delete_thought() must not have "
                "committed it on their behalf"
            )

            other = await aiosqlite.connect(str(db_path))
            try:
                cursor = await other.execute("SELECT COUNT(*) FROM audit")
                row2 = await cursor.fetchone()
                assert row2 is not None
                assert row2[0] == 0, "the trigger never fired, so nothing was ever written"
            finally:
                await other.close()
        finally:
            await db.close()


class TestTransactionStateAfterAVetoedDelete:
    """The connection's transaction state after a delete that a trigger vetoes.

    In these tests a ``RAISE(IGNORE)`` trigger vetoes the delete without
    raising: the parent ``DELETE`` matches zero rows and the row is still
    there.

    * ``run_hygiene``'s GC, ``delete_thought`` and ``cleanup_expired``'s
      delete strategy, each called with no transaction open, return with
      ``in_transaction`` ``False`` and a second connection able to take the
      write lock at once. A transaction begun with ``BEGIN IMMEDIATE`` holds
      the write lock until it ends.
    * ``delete_thought`` called inside a ``suspend_auto_commit`` window, after
      an earlier write in the window or after ``bulk_store``, returns with the
      window's transaction still open; the window's writes and the vetoed
      thought are present once the window closes.
    * ``_delete_thought_atomic`` called directly with no transaction open
      returns with ``in_transaction`` ``False``. Called inside a transaction
      the caller began with an explicit ``BEGIN``, it returns with that
      transaction still open, with or without a pending edit in it.

    A genuine second connection to the same file is required to observe the
    lock from outside -- a shared ``:memory:`` database cannot host
    cross-connection contention (see ``tests/test_dedup_write_contention.py``,
    which uses the same file-backed pattern to simulate a stuck writer;
    here a stuck writer is exactly what must *not* be reproduced).
    """

    @staticmethod
    async def _install_ignore_trigger(db: aiosqlite.Connection) -> None:
        await db.execute(
            "CREATE TRIGGER thought_delete_ignore BEFORE DELETE ON thought "
            "BEGIN SELECT RAISE(IGNORE); END"
        )

    @staticmethod
    async def _assert_second_connection_can_write_immediately(db_path: Path) -> None:
        """A fresh connection must take a write reservation with no wait.

        ``busy_timeout=0`` makes a reservation still held by the first
        connection fail instantly instead of after a wait, so this either
        passes immediately or raises ``sqlite3.OperationalError`` (database
        is locked) -- there is no flakiness window either way.
        """
        second = await aiosqlite.connect(str(db_path))
        try:
            await second.execute("PRAGMA busy_timeout=0")
            await second.execute("BEGIN IMMEDIATE")
            await second.rollback()
        finally:
            await second.close()

    @pytest.mark.parametrize("foreign_keys_on", [True, False])
    async def test_hygiene_gc_ignore_veto_leaves_no_transaction_open(
        self,
        tmp_path: Path,
        foreign_keys_on: bool,
    ) -> None:
        policy = HygienePolicyConfig(
            enabled=True,
            eviction_threshold=0.0,
            auto_gc_enabled=True,
            gc_min_archive_age_cycles=0,
            gc_restore_window_seconds=0,
        )
        db_path = tmp_path / f"hygiene-gc-leak-{foreign_keys_on}.sqlite"
        db = await aiosqlite.connect(str(db_path))
        try:
            db.row_factory = aiosqlite.Row
            store = SqliteEngravaCore(
                db, embedding_provider=None, auto_embed=False, hygiene_policy=policy
            )
            await store.ensure_schema()
            if not foreign_keys_on:
                await db.execute("PRAGMA foreign_keys = OFF")

            await store.create_thought(_make_thought("t1"))
            await store.create_thought(_make_thought("t2"))
            await store.create_edge(_make_edge("e1", "t1", "t2"))
            await store.update_thought(
                "t1", lifecycle_status=LifecycleStatus.ARCHIVED, archived_at_cycle=0
            )
            await self._install_ignore_trigger(db)

            result = await store.run_hygiene(current_cycle=1000)

            assert result.gc_count == 0, "the veto must not be counted as a real deletion"
            assert result.archived_count == 0, (
                "archiving anything would make run_hygiene's own _maybe_commit "
                "mask the leak instead of exercising it"
            )
            assert db.in_transaction is False, (
                "nothing survived the veto, so the pass must not leave the "
                "transaction open -- run_hygiene has nothing to commit"
            )
            await self._assert_second_connection_can_write_immediately(db_path)
        finally:
            await db.close()

    @pytest.mark.parametrize("foreign_keys_on", [True, False])
    async def test_delete_thought_ignore_veto_leaves_no_transaction_open(
        self,
        tmp_path: Path,
        foreign_keys_on: bool,
    ) -> None:
        db_path = tmp_path / f"delete-thought-leak-{foreign_keys_on}.sqlite"
        db = await aiosqlite.connect(str(db_path))
        try:
            db.row_factory = aiosqlite.Row
            store = SqliteEngravaCore(db, embedding_provider=None, auto_embed=False)
            await store.ensure_schema()
            if not foreign_keys_on:
                await db.execute("PRAGMA foreign_keys = OFF")

            await store.create_thought(_make_thought("t1"))
            await store.create_thought(_make_thought("t2"))
            await store.create_edge(_make_edge("e1", "t1", "t2"))
            await self._install_ignore_trigger(db)

            deleted = await store.delete_thought("t1")

            assert deleted is False
            assert db.in_transaction is False, (
                "nothing was deleted, so delete_thought must not leave the transaction open"
            )
            await self._assert_second_connection_can_write_immediately(db_path)
        finally:
            await db.close()

    @pytest.mark.parametrize("foreign_keys_on", [True, False])
    async def test_ttl_delete_ignore_veto_leaves_no_transaction_open(
        self,
        tmp_path: Path,
        foreign_keys_on: bool,
    ) -> None:
        db_path = tmp_path / f"ttl-delete-leak-{foreign_keys_on}.sqlite"
        db = await aiosqlite.connect(str(db_path))
        try:
            db.row_factory = aiosqlite.Row
            store = SqliteEngravaCore(
                db, embedding_provider=None, auto_embed=False, ttl_strategy="delete"
            )
            await store.ensure_schema()
            if not foreign_keys_on:
                await db.execute("PRAGMA foreign_keys = OFF")

            past = "2026-01-01T00:00:00+00:00"
            now = "2026-06-01T00:00:00+00:00"
            await store.create_thought(_make_thought("t1", expires_at=past))
            await store.create_thought(_make_thought("t2"))
            await store.create_edge(_make_edge("e1", "t1", "t2"))
            await self._install_ignore_trigger(db)

            result = await store.cleanup_expired(now=now)

            assert result.expired_count == 1
            assert db.in_transaction is False, (
                "nothing was deleted in this batch, so cleanup_expired must "
                "not leave the transaction open"
            )
            await self._assert_second_connection_can_write_immediately(db_path)
        finally:
            await db.close()

    async def test_suspend_auto_commit_window_survives_a_vetoed_delete(
        self,
        tmp_path: Path,
    ) -> None:
        """Ownership case 1: the caller's own ``suspend_auto_commit`` window.

        An earlier write inside the same window opens the transaction before
        the vetoed delete runs, so ``_delete_thought_atomic`` must see
        ``opened_transaction`` as ``False`` and leave the window's
        transaction alone -- both the earlier write and the (unaffected,
        still-live) vetoed thought must come out the other side once the
        window itself closes.
        """
        db_path = tmp_path / "suspend-window-untouched.sqlite"
        db = await aiosqlite.connect(str(db_path))
        try:
            db.row_factory = aiosqlite.Row
            store = SqliteEngravaCore(db, embedding_provider=None, auto_embed=False)
            await store.ensure_schema()
            await store.create_thought(_make_thought("vetoed"))
            await self._install_ignore_trigger(db)

            async with store.suspend_auto_commit():
                await store.create_thought(_make_thought("kept"))
                deleted = await store.delete_thought("vetoed")
                assert deleted is False
                assert db.in_transaction is True, (
                    "the window's own transaction, opened by the earlier "
                    "create_thought, must still be open here"
                )

            assert db.in_transaction is False, "the window's own clean exit must commit"
            assert await store.get_thought("kept") is not None
            assert await store.get_thought("vetoed") is not None, (
                "the veto inside the window must not have been undone by "
                "anything touching the window's transaction"
            )
        finally:
            await db.close()

    async def test_bulk_store_batch_survives_a_vetoed_delete_in_the_same_window(
        self,
        tmp_path: Path,
    ) -> None:
        """Ownership case 2: a batched ``bulk_store`` row.

        ``bulk_store`` runs its insert loop inside its own (nested, but
        task-reentrant and same-transaction) ``suspend_auto_commit`` window.
        Calling a vetoed ``delete_thought`` in the same outer window, after
        ``bulk_store`` has already opened it with a real inserted row, must
        not touch that transaction -- the inserted rows must still commit
        together with the (unaffected) vetoed thought once the outer window
        closes.
        """
        db_path = tmp_path / "bulk-store-batch-untouched.sqlite"
        db = await aiosqlite.connect(str(db_path))
        try:
            db.row_factory = aiosqlite.Row
            store = SqliteEngravaCore(db, embedding_provider=None, auto_embed=False)
            await store.ensure_schema()
            await store.create_thought(_make_thought("vetoed"))
            await self._install_ignore_trigger(db)

            async with store.suspend_auto_commit():
                await store.bulk_store([_make_thought("b1"), _make_thought("b2")])
                deleted = await store.delete_thought("vetoed")
                assert deleted is False
                assert db.in_transaction is True, (
                    "the outer window's transaction, opened by bulk_store's "
                    "own inserts, must still be open here"
                )

            assert db.in_transaction is False, "the window's own clean exit must commit"
            assert await store.get_thought("b1") is not None
            assert await store.get_thought("b2") is not None
            assert await store.get_thought("vetoed") is not None
        finally:
            await db.close()

    async def test_explicit_begin_survives_a_vetoed_delete(
        self,
        tmp_path: Path,
    ) -> None:
        """Ownership case 3: a caller's own explicit ``BEGIN``.

        A caller that issues a raw ``BEGIN`` directly on the connection --
        bypassing ``suspend_auto_commit`` entirely, with nothing written yet
        -- owns that transaction. ``_delete_thought_atomic`` must see
        ``opened_transaction`` as ``False`` here too and leave it for the
        caller's own commit/rollback to decide.

        This calls ``_delete_thought_atomic`` directly rather than the public
        ``delete_thought`` so this test is the narrowest possible pin on the
        internal ownership mechanism, independent of whatever the public
        method's own call site does with it.
        ``delete_thought`` itself no longer needs a ``suspend_auto_commit``
        window to leave a raw, unmediated caller ``BEGIN`` alone on this
        path: it now gates its own ``_maybe_commit()`` on whether it actually
        deleted anything, exactly like this method gates its own transaction
        handling — see ``TestPubliclyVetoedWritesDoNotCommitACallersTransaction``
        for that behaviour exercised through the public method, with a real
        pending edit under the caller's ``BEGIN`` and a ``rollback()`` that
        must actually undo it.
        """
        db_path = tmp_path / "explicit-begin-untouched.sqlite"
        db = await aiosqlite.connect(str(db_path))
        try:
            db.row_factory = aiosqlite.Row
            store = SqliteEngravaCore(db, embedding_provider=None, auto_embed=False)
            await store.ensure_schema()
            await store.create_thought(_make_thought("vetoed"))
            await self._install_ignore_trigger(db)

            await db.execute("BEGIN")
            result = await store._delete_thought_atomic("vetoed")

            assert result.deleted is False
            assert result.wrote_anything is False
            assert db.in_transaction is True, (
                "the caller's own explicit BEGIN, opened before this call and "
                "with nothing written under it yet, must still be open here"
            )
            await db.rollback()
            assert await store.get_thought("vetoed") is not None
        finally:
            await db.close()

    @pytest.mark.parametrize("foreign_keys_on", [True, False])
    async def test_atomic_delete_vetoed_with_no_transaction_open_leaves_none_open(
        self,
        store: SqliteEngravaCore,
        foreign_keys_on: bool,
    ) -> None:
        """A vetoed ``_delete_thought_atomic`` called with no transaction open."""
        db = store._db
        if not foreign_keys_on:
            await db.execute("PRAGMA foreign_keys = OFF")
        await store.create_thought(_make_thought("t1"))
        await self._install_ignore_trigger(db)
        await db.commit()
        assert db.in_transaction is False, "the call starts with no transaction open"

        result = await store._delete_thought_atomic("t1")

        assert result.deleted is False
        assert db.in_transaction is False, (
            "the call began with no transaction open and deleted nothing, "
            "so no transaction may be open when it returns"
        )
        assert await store.get_thought("t1") is not None

    @pytest.mark.parametrize("foreign_keys_on", [True, False])
    async def test_atomic_delete_vetoed_inside_a_callers_transaction_leaves_it_open(
        self,
        store: SqliteEngravaCore,
        foreign_keys_on: bool,
    ) -> None:
        """A vetoed ``_delete_thought_atomic`` called inside a caller's ``BEGIN``."""
        db = store._db
        if not foreign_keys_on:
            await db.execute("PRAGMA foreign_keys = OFF")
        await store.create_thought(_make_thought("t1"))
        await store.create_thought(_make_thought("t2"))
        await self._install_ignore_trigger(db)
        await db.commit()

        await db.execute("BEGIN")
        await db.execute(
            "UPDATE thought SET essence = ? WHERE thought_id = ?",
            ("edited-by-caller", "t2"),
        )

        result = await store._delete_thought_atomic("t1")

        assert result.deleted is False
        assert db.in_transaction is True, (
            "the caller's own transaction, with their pending edit inside it, "
            "must still be open when the call returns"
        )
        cursor = await db.execute("SELECT essence FROM thought WHERE thought_id = 't2'")
        row = await cursor.fetchone()
        assert row is not None
        assert row["essence"] == "edited-by-caller", "the caller's pending edit must still be there"
        await db.rollback()
        cursor = await db.execute("SELECT essence FROM thought WHERE thought_id = 't2'")
        row = await cursor.fetchone()
        assert row is not None
        assert row["essence"] == "essence-t2", (
            "the edit was still pending, so the caller's own rollback must undo it"
        )


class TestPubliclyVetoedWritesDoNotCommitACallersTransaction:
    """A write-free outcome must not commit a caller's own pending edit.

    ``_delete_thought_atomic`` (exercised above) never touches a transaction
    it did not open. But before this fix, its two public callers,
    ``delete_thought`` and ``cleanup_expired``, undid that protection from the
    outside: both called ``_maybe_commit()`` unconditionally on return,
    regardless of whether anything was actually deleted. When a caller had
    opened its own transaction first (a raw ``BEGIN``, or a real write buried
    a few frames up the same task) and had a pending edit of its own sitting
    in it, a vetoed delete's unconditional commit durably applied that edit
    too — the caller's own later ``rollback()`` had nothing left to undo.

    Each test below reproduces exactly that: a caller-owned transaction, a
    real pending edit inside it, a write-free call, and a ``rollback()`` that
    must actually undo the edit. The controls alongside confirm the ordinary,
    writing path is unchanged: it keeps committing, caller-owned transaction
    or not, because "whoever writes, commits" was never in question — only
    the write-free branch was.
    """

    async def test_delete_thought_veto_does_not_commit_the_callers_pending_edit(
        self,
        tmp_path: Path,
    ) -> None:
        db_path = tmp_path / "delete-thought-veto-caller-txn.sqlite"
        db = await aiosqlite.connect(str(db_path))
        try:
            db.row_factory = aiosqlite.Row
            store = SqliteEngravaCore(db, embedding_provider=None, auto_embed=False)
            await store.ensure_schema()
            await store.create_thought(_make_thought("vetoed"))
            await store.create_thought(_make_thought("unrelated"))
            await db.execute(
                "CREATE TRIGGER thought_delete_ignore BEFORE DELETE ON thought "
                "BEGIN SELECT RAISE(IGNORE); END"
            )

            await db.execute("BEGIN")
            await db.execute(
                "UPDATE thought SET essence = ? WHERE thought_id = ?",
                ("edited-by-caller", "unrelated"),
            )

            deleted = await store.delete_thought("vetoed")

            assert deleted is False
            assert db.in_transaction is True, (
                "the caller's own transaction, with their pending edit still "
                "inside it, must still be open after a vetoed delete_thought()"
            )
            await db.rollback()

            row = await store.get_thought("unrelated")
            assert row is not None
            assert row.essence == "essence-unrelated", (
                "the caller's rollback must undo their own edit -- a vetoed "
                "delete_thought() must not have committed it on their behalf"
            )
        finally:
            await db.close()

    async def test_ordinary_delete_thought_still_commits(self, tmp_path: Path) -> None:
        """Control: a real delete, with no caller transaction, commits as before."""
        db_path = tmp_path / "delete-thought-ordinary-commit.sqlite"
        db = await aiosqlite.connect(str(db_path))
        try:
            db.row_factory = aiosqlite.Row
            store = SqliteEngravaCore(db, embedding_provider=None, auto_embed=False)
            await store.ensure_schema()
            await store.create_thought(_make_thought("t1"))

            deleted = await store.delete_thought("t1")

            assert deleted is True
            assert db.in_transaction is False, "a real delete must still commit"
            assert await store.get_thought("t1") is None
        finally:
            await db.close()

    async def test_cleanup_expired_all_vetoed_does_not_commit_the_callers_pending_edit(
        self,
        tmp_path: Path,
    ) -> None:
        db_path = tmp_path / "cleanup-expired-veto-caller-txn.sqlite"
        db = await aiosqlite.connect(str(db_path))
        try:
            db.row_factory = aiosqlite.Row
            store = SqliteEngravaCore(
                db, embedding_provider=None, auto_embed=False, ttl_strategy="delete"
            )
            await store.ensure_schema()
            past = "2026-01-01T00:00:00+00:00"
            now = "2026-06-01T00:00:00+00:00"
            await store.create_thought(_make_thought("vetoed", expires_at=past))
            await store.create_thought(_make_thought("unrelated"))
            await db.execute(
                "CREATE TRIGGER thought_delete_ignore BEFORE DELETE ON thought "
                "BEGIN SELECT RAISE(IGNORE); END"
            )

            await db.execute("BEGIN")
            await db.execute(
                "UPDATE thought SET essence = ? WHERE thought_id = ?",
                ("edited-by-caller", "unrelated"),
            )

            result = await store.cleanup_expired(now=now)

            assert result.expired_count == 1
            assert db.in_transaction is True, (
                "the caller's own transaction, with their pending edit still "
                "inside it, must still be open after a fully-vetoed batch"
            )
            await db.rollback()

            row = await store.get_thought("unrelated")
            assert row is not None
            assert row.essence == "essence-unrelated", (
                "the caller's rollback must undo their own edit -- a "
                "fully-vetoed cleanup_expired() must not have committed it"
            )
        finally:
            await db.close()

    async def test_ordinary_cleanup_expired_still_commits(self, tmp_path: Path) -> None:
        """Control: a real expiry delete, with no caller transaction, commits."""
        db_path = tmp_path / "cleanup-expired-ordinary-commit.sqlite"
        db = await aiosqlite.connect(str(db_path))
        try:
            db.row_factory = aiosqlite.Row
            store = SqliteEngravaCore(
                db, embedding_provider=None, auto_embed=False, ttl_strategy="delete"
            )
            await store.ensure_schema()
            past = "2026-01-01T00:00:00+00:00"
            now = "2026-06-01T00:00:00+00:00"
            await store.create_thought(_make_thought("t1", expires_at=past))

            result = await store.cleanup_expired(now=now)

            assert result.expired_count == 1
            assert db.in_transaction is False, "a real expiry delete must still commit"
            assert await store.get_thought("t1") is None
        finally:
            await db.close()


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
            assert version_row[0] == 21
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
            assert version_row[0] == 21
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
            assert version_row[0] == 21
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


# Failure-injection points spread across ``schema_core.sql``: the fourth
# statement, the FTS virtual table, a table declared late in the script, the
# middle of the hot-path index block, and the very last statement before the
# version stamp. A single injection point only ever pins the stamp as being
# *somewhere after* it — the last two entries are what make relocating the stamp
# above the trailing index block observable at runtime. Each entry is the prefix
# of the statement to fail at, paired with the object that statement creates.
_BOOTSTRAP_INJECTION_POINTS = [
    ("CREATE TABLE IF NOT EXISTS edge", "edge"),
    ("CREATE VIRTUAL TABLE IF NOT EXISTS thought_fts", "thought_fts"),
    ("CREATE TABLE IF NOT EXISTS extension_schema_versions", "extension_schema_versions"),
    ("CREATE INDEX IF NOT EXISTS idx_edge_to_thought", "idx_edge_to_thought"),
    ("CREATE INDEX IF NOT EXISTS idx_thought_prov_actor", "idx_thought_prov_actor"),
]
_BOOTSTRAP_INJECTION_IDS = [name for _statement, name in _BOOTSTRAP_INJECTION_POINTS]


class TestBootstrapAtomicity:
    """Fresh bootstrap stamps ``user_version`` only after the full schema applies."""

    @pytest.mark.parametrize(
        ("fail_at", "unreached_object"),
        _BOOTSTRAP_INJECTION_POINTS,
        ids=_BOOTSTRAP_INJECTION_IDS,
    )
    async def test_bootstrap_failure_leaves_version_unstamped_and_retryable(
        self,
        tmp_path: Path,
        fail_at: str,
        unreached_object: str,
    ) -> None:
        """A mid-bootstrap DDL failure leaves version 0; a retry reaches v20.

        The version stamp is the last statement of ``schema_core.sql``, so a DDL
        failure before it leaves ``user_version = 0`` (never a partial 20). The
        next ``ensure_schema`` then re-runs the idempotent bootstrap rather than
        skipping every migration against an incomplete schema.

        ``executescript`` is not atomic — whatever ran before the failure is
        durable — so this is asserted at several offsets, right up to the last
        statement before the stamp. A stamp moved even one statement earlier
        would durably mark an incomplete schema as current, and ``ensure_schema``
        never revisits a database that already reads as head. The structural
        counterpart (the stamp *is* the final statement, for every offset at
        once) lives in ``test_migration_upgrade_chains.py``.
        """
        db_path = tmp_path / "bootstrap.sqlite"
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            core = SqliteEngravaCore(db=db, embedding_provider=None, auto_embed=False)

            real_executescript = db.executescript

            async def _failing_executescript(script: str) -> object:
                # Run the script up to ``fail_at`` then fail — a genuine
                # mid-bootstrap DDL error that leaves a *partial* schema before
                # the final ``PRAGMA user_version`` stamp. The count assertion
                # keeps the split at the statement this case names: a second
                # textual occurrence would silently move the injection point and
                # the test would stop probing what it claims to.
                assert script.count(fail_at) == 1, f"expected one occurrence of {fail_at!r}"
                head, _sep, _tail = script.partition(fail_at)
                await real_executescript(head)
                msg = "injected bootstrap DDL failure"
                raise aiosqlite.OperationalError(msg)

            with (
                patch.object(db, "executescript", _failing_executescript),
                pytest.raises(aiosqlite.OperationalError) as bootstrap_error,
            ):
                await core.ensure_schema()

            # The failed bootstrap left a PARTIAL schema (everything before the
            # injection point) and did NOT durably stamp the version. Asserted
            # before the raised error is inspected, so a mismatched message
            # cannot shadow what actually happened to the database.
            version_row = await (await db.execute("PRAGMA user_version")).fetchone()
            assert version_row is not None
            assert version_row[0] == 0
            present = {
                str(row[0])
                for row in await (await db.execute("SELECT name FROM sqlite_master")).fetchall()
            }
            assert "thought" in present
            assert unreached_object not in present, (
                f"partial bootstrap should not have reached {unreached_object}"
            )
            assert "injected bootstrap" in str(bootstrap_error.value)

            # A retry with the real bootstrap converges on a complete head schema.
            await core.ensure_schema()
            version_row = await (await db.execute("PRAGMA user_version")).fetchone()
            assert version_row is not None
            assert version_row[0] == 21
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

    async def test_older_build_interrupted_bootstrap_refuses_instead_of_stamping_head(
        self,
        tmp_path: Path,
    ) -> None:
        """A pre-existing older-shape core table must not end up stamped head.

        This is the *cross-version* counterpart to
        ``test_bootstrap_failure_leaves_version_unstamped_and_retryable``
        above: that test interrupts **this build's own** bootstrap and shows a
        retry converges cleanly, because every table it half-created already
        has this build's full column set. Here the database instead comes
        from an **older build** whose own bootstrap died before its stamp —
        a real v20 schema (reconstructed the same way
        ``test_migration_upgrade_chains.py`` reconstructs every historical
        shape) with zero rows and ``user_version = 0``.

        ``schema_core.sql`` is pure ``CREATE ... IF NOT EXISTS``, so run
        against that file it leaves the pre-existing v20-shape ``thought`` /
        ``edge`` / ``action`` tables — missing the ``revision`` column —
        completely untouched, while its own unconditional trailing
        ``PRAGMA user_version = 21`` would otherwise stamp the database
        current anyway. ``ensure_schema`` must instead refuse, and leave the
        version off head so a later open keeps refusing rather than reading
        the database as fully migrated.
        """
        db_path = tmp_path / "interrupted-older-build.sqlite"
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            # A genuine v20 shape, stamped v20 by the reconstruction helper...
            await _bootstrap_core_at_version(db, 20)
            # ...then knocked back to 0: the stamp this database's own
            # (older) bootstrap never reached, because it died first.
            await db.execute("PRAGMA user_version = 0")
            await db.commit()

            cols_before = {
                row["name"]
                for row in await (await db.execute("PRAGMA table_info(thought)")).fetchall()
            }
            assert "revision" not in cols_before

            core = SqliteEngravaCore(db=db, embedding_provider=None, auto_embed=False)
            with pytest.raises(SchemaVersionError, match="older shape"):
                await core.ensure_schema()

            version_row = await (await db.execute("PRAGMA user_version")).fetchone()
            assert version_row is not None
            assert version_row[0] != 21, (
                "must not be left stamped head over a schema missing `revision`"
            )
            cols_after = {
                row["name"]
                for row in await (await db.execute("PRAGMA table_info(thought)")).fetchall()
            }
            assert "revision" not in cols_after

            # A second open refuses the same way -- it is not a one-shot
            # warning that then silently opens the file as current.
            with pytest.raises(SchemaVersionError):
                await core.ensure_schema()
            version_row = await (await db.execute("PRAGMA user_version")).fetchone()
            assert version_row is not None
            assert version_row[0] != 21


class TestBootstrapShapeCheckIsSelfUpdating:
    """The bootstrap postcondition is not pinned to one hard-coded column.

    A version of this check that looked only for the ``revision`` column
    caught today's cross-version bootstrap hazard, but would silently stop
    catching it the day a *next* core migration adds a column: an
    older-shape leftover table missing only that new column would still
    carry ``revision`` and pass. Deriving the check from a disposable
    reference database built off the exact ``schema_core.sql`` text means
    any missing column trips it, with nothing in this method to update when
    the script gains one. Demonstrated directly against a synthetic "one
    column further" schema standing in for a migration that has not been
    written yet, rather than waiting for a real one to land.
    """

    async def test_a_hypothetical_future_column_is_caught_without_a_code_change(
        self,
        tmp_path: Path,
    ) -> None:
        db_path = tmp_path / "future-migration-probe.sqlite"
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            store = SqliteEngravaCore(db=db, embedding_provider=None, auto_embed=False)
            await store.ensure_schema()

            real_schema_sql = (
                resources.files("engrava.infrastructure.sqlite")
                .joinpath("schema_core.sql")
                .read_text(encoding="utf-8")
            )
            marker = (
                "CREATE TABLE IF NOT EXISTS thought (\n    thought_id        TEXT    PRIMARY KEY,"
            )
            assert real_schema_sql.count(marker) == 1, "the fixture depends on this exact text"
            # A stand-in for "the next core migration": a column this build's
            # schema_core.sql does not declare anywhere.
            future_schema_sql = real_schema_sql.replace(
                marker, marker + "\n    probe_next_core_column TEXT,"
            )
            assert future_schema_sql != real_schema_sql

            # This store's tables already exist (bootstrapped above), so the
            # check compares them instead of short-circuiting on "nothing
            # exists yet". Against the schema it was actually bootstrapped
            # from, the check passes.
            assert await store._existing_core_tables_match_bootstrap_shape(real_schema_sql) is True
            # Against a schema one column further, it correctly reports the
            # gap -- no change to this method itself is needed to catch it.
            assert (
                await store._existing_core_tables_match_bootstrap_shape(future_schema_sql) is False
            )


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
            assert version_row[0] == 21
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


class TestRecreateFkCleanupDoesNotReplaceTheOriginalError:
    """A failing rollback or pragma-restore during migration cleanup must not
    replace the migration failure that triggered it.

    ``_recreate_child_tables_with_fk_atomically``'s outer cleanup used to be
    an unconditional ``finally: try: await self._db.rollback() finally: await
    self._db.execute("PRAGMA foreign_keys=ON")``. A failure in either
    statement there raised in front of whatever the recreate body was
    already failing with, so a user whose ``engrava migrate`` hit a mid-swap
    error and then hit a rollback or pragma failure on top of it saw only
    the second, purely-mechanical error -- never the actual reason their
    migration failed, which is the only thing they can act on.
    """

    async def test_rollback_failure_during_cleanup_does_not_replace_the_original_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Before this fix: a user would see ``OperationalError("injected
        rollback failure")`` -- the cleanup's own error -- when a mid-recreate
        failure's cleanup rollback also failed. After this fix: they see the
        real migration failure (``RuntimeError("injected mid-recreate
        failure")``), with the rollback failure only in the log.
        """
        db_path = tmp_path / "rollback-cleanup-failure.sqlite"
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

            async def _fail_body() -> None:
                msg = "injected mid-recreate failure"
                raise RuntimeError(msg)

            monkeypatch.setattr(core, "_recreate_embedding_with_fk", _fail_body)

            async def _failing_rollback() -> None:
                msg = "injected rollback failure"
                raise aiosqlite.OperationalError(msg)

            monkeypatch.setattr(db, "rollback", _failing_rollback)

            with (
                caplog.at_level(logging.WARNING),
                pytest.raises(RuntimeError, match="injected mid-recreate failure") as exc_info,
            ):
                await core.ensure_schema()

            assert "injected rollback failure" not in str(exc_info.value), (
                "the rollback's own failure must not replace the migration failure"
            )
            assert "rolling back the migration transaction" in caplog.text
            assert "injected rollback failure" in caplog.text

            # The pragma restore is still attempted (and, here, succeeds) even
            # though the rollback ahead of it failed.
            monkeypatch.undo()
            fk_row = await (await db.execute("PRAGMA foreign_keys")).fetchone()
            assert fk_row is not None
            assert fk_row[0] == 1, "FK enforcement must still be restored after a failed rollback"

    async def test_pragma_restore_failure_during_cleanup_does_not_replace_the_original_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The pragma-restore half of the same cleanup, isolated from the rollback."""
        db_path = tmp_path / "pragma-cleanup-failure.sqlite"
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

            async def _fail_body() -> None:
                msg = "injected mid-recreate failure"
                raise RuntimeError(msg)

            monkeypatch.setattr(core, "_recreate_embedding_with_fk", _fail_body)

            real_execute = db.execute

            async def _execute(sql: str, *args: object, **kwargs: object) -> object:
                if sql == "PRAGMA foreign_keys=ON":
                    msg = "injected pragma restore failure"
                    raise aiosqlite.OperationalError(msg)
                return await real_execute(sql, *args, **kwargs)

            monkeypatch.setattr(db, "execute", _execute)

            with (
                caplog.at_level(logging.WARNING),
                pytest.raises(RuntimeError, match="injected mid-recreate failure") as exc_info,
            ):
                await core.ensure_schema()

            assert "injected pragma restore failure" not in str(exc_info.value), (
                "the pragma restore's own failure must not replace the migration failure"
            )
            assert "restoring PRAGMA foreign_keys" in caplog.text
            assert "injected pragma restore failure" in caplog.text

            # The rollback ahead of it still ran (the transaction is closed)
            # even though the pragma restore after it failed.
            monkeypatch.undo()
            assert not db.in_transaction

    async def test_pragma_restore_failure_on_the_clean_success_path_still_propagates(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Control: with no original error, a pragma-restore failure IS the
        error and must propagate unchanged -- never logged and swallowed the
        way a cleanup failure alongside a real migration failure is. Mirrors
        ``_close_quietly``'s documented success-path contract for a close.
        """
        db_path = tmp_path / "pragma-clean-path-failure.sqlite"
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            await TestMigrationV11ToV12._bootstrap_v11_schema(db)
            await db.commit()
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            core = SqliteEngravaCore(db=db, embedding_provider=None, auto_embed=False)

            real_execute = db.execute

            async def _execute(sql: str, *args: object, **kwargs: object) -> object:
                if sql == "PRAGMA foreign_keys=ON":
                    msg = "injected pragma restore failure on the success path"
                    raise aiosqlite.OperationalError(msg)
                return await real_execute(sql, *args, **kwargs)

            monkeypatch.setattr(db, "execute", _execute)

            with pytest.raises(aiosqlite.OperationalError, match="success path"):
                await core.ensure_schema()

    async def test_rollback_failure_on_the_clean_success_path_still_restores_the_pragma(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The symmetric case: the cleanup rollback itself fails on the clean
        success path (no original migration error). Before this fix, the
        pragma restore below the rollback was never reached, silently
        leaving FK enforcement off for the rest of the connection's life.
        After this fix, the pragma restore still runs -- verified by reading
        it back from the connection, not from a mock's call list -- and the
        rollback's own failure still reaches the caller rather than being
        silently lost.
        """
        db_path = tmp_path / "rollback-clean-path-failure.sqlite"
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            await TestMigrationV11ToV12._bootstrap_v11_schema(db)
            await db.commit()
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            core = SqliteEngravaCore(db=db, embedding_provider=None, auto_embed=False)

            async def _failing_rollback() -> None:
                msg = "injected rollback failure on the success path"
                raise aiosqlite.OperationalError(msg)

            monkeypatch.setattr(db, "rollback", _failing_rollback)

            with pytest.raises(aiosqlite.OperationalError, match="success path"):
                await core.ensure_schema()

            monkeypatch.undo()
            fk_row = await (await db.execute("PRAGMA foreign_keys")).fetchone()
            assert fk_row is not None
            assert fk_row[0] == 1, (
                "FK enforcement must still be restored even though the "
                "cleanup rollback ahead of it failed on the success path"
            )

    async def test_rollback_raising_cancellederror_directly_still_restores_the_pragma(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The cleanup rollback raises ``asyncio.CancelledError`` itself --
        not because this coroutine was cancelled from the outside, but
        because the rollback call's own body raises it directly (a
        ``BaseException``, not an ``Exception``).

        ``_run_cleanup_step_quietly``'s shield-then-redraw re-awaits an
        already-finished task through an ``except Exception``, which does
        not catch a second ``CancelledError`` -- so before this fix, this
        exact case let the ``CancelledError`` escape the cleanup helper
        before the pragma restore below it ever ran. This must still reach
        the caller (never swallowed), and the pragma restore must still have
        run first.
        """
        db_path = tmp_path / "rollback-clean-path-cancellederror.sqlite"
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            await TestMigrationV11ToV12._bootstrap_v11_schema(db)
            await db.commit()
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            core = SqliteEngravaCore(db=db, embedding_provider=None, auto_embed=False)

            async def _cancelling_rollback() -> None:
                raise asyncio.CancelledError

            monkeypatch.setattr(db, "rollback", _cancelling_rollback)

            with pytest.raises(asyncio.CancelledError):
                await core.ensure_schema()

            monkeypatch.undo()
            fk_row = await (await db.execute("PRAGMA foreign_keys")).fetchone()
            assert fk_row is not None
            assert fk_row[0] == 1, (
                "FK enforcement must still be restored even though the "
                "cleanup rollback raised CancelledError directly"
            )

    async def test_rollback_raising_a_plain_baseexception_still_restores_the_pragma(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The general case behind the ``CancelledError`` case above: any
        ``BaseException`` that is not an ``Exception`` (a custom one here,
        standing in for ``SystemExit``/``KeyboardInterrupt``) raised by the
        cleanup rollback must still let the pragma restore run, and must
        still reach the caller unchanged.
        """

        class _InjectedBaseException(BaseException):
            pass

        db_path = tmp_path / "rollback-clean-path-baseexception.sqlite"
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            await TestMigrationV11ToV12._bootstrap_v11_schema(db)
            await db.commit()
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            core = SqliteEngravaCore(db=db, embedding_provider=None, auto_embed=False)

            async def _failing_rollback() -> None:
                msg = "injected non-Exception BaseException from rollback"
                raise _InjectedBaseException(msg)

            monkeypatch.setattr(db, "rollback", _failing_rollback)

            with pytest.raises(_InjectedBaseException, match="non-Exception"):
                await core.ensure_schema()

            monkeypatch.undo()
            fk_row = await (await db.execute("PRAGMA foreign_keys")).fetchone()
            assert fk_row is not None
            assert fk_row[0] == 1, (
                "FK enforcement must still be restored even though the "
                "cleanup rollback raised a plain BaseException"
            )

    async def test_cancellation_during_the_cleanup_rollback_is_not_swallowed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A cancellation delivered while the cleanup rollback is in flight
        must still reach the caller -- never converted into, or absorbed
        underneath, the migration failure it is cleaning up after -- and the
        pragma restore after it must still run.
        """
        db_path = tmp_path / "rollback-cleanup-cancel.sqlite"
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

            async def _fail_body() -> None:
                msg = "injected mid-recreate failure"
                raise RuntimeError(msg)

            monkeypatch.setattr(core, "_recreate_embedding_with_fk", _fail_body)

            real_rollback = db.rollback
            started = asyncio.Event()
            may_finish = asyncio.Event()
            finished = False

            async def _blocking_rollback() -> None:
                nonlocal finished
                started.set()
                await may_finish.wait()
                await real_rollback()
                finished = True

            monkeypatch.setattr(db, "rollback", _blocking_rollback)

            task = asyncio.create_task(core.ensure_schema())
            await started.wait()
            task.cancel()
            may_finish.set()

            with pytest.raises(asyncio.CancelledError):
                await task

            assert finished, (
                "the rollback never ran to completion under cancellation -- the "
                "exact leak this cleanup exists to prevent"
            )

            monkeypatch.undo()
            fk_row = await (await db.execute("PRAGMA foreign_keys")).fetchone()
            assert fk_row is not None
            assert fk_row[0] == 1, (
                "the pragma restore must still run even though a cancellation "
                "arrived during the rollback ahead of it"
            )

    async def test_cancellation_during_the_pragma_restore_is_not_swallowed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The symmetric case: cancellation while restoring the pragma itself
        (rollback already completed) must also reach the caller, not be
        absorbed by the ``except Exception`` half of the cleanup step.
        """
        db_path = tmp_path / "pragma-cleanup-cancel.sqlite"
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

            async def _fail_body() -> None:
                msg = "injected mid-recreate failure"
                raise RuntimeError(msg)

            monkeypatch.setattr(core, "_recreate_embedding_with_fk", _fail_body)

            real_execute = db.execute
            started = asyncio.Event()
            may_finish = asyncio.Event()

            async def _execute(sql: str, *args: object, **kwargs: object) -> object:
                if sql == "PRAGMA foreign_keys=ON":
                    started.set()
                    await may_finish.wait()
                return await real_execute(sql, *args, **kwargs)

            monkeypatch.setattr(db, "execute", _execute)

            task = asyncio.create_task(core.ensure_schema())
            await started.wait()
            task.cancel()
            may_finish.set()

            with pytest.raises(asyncio.CancelledError):
                await task


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


# ----------------------------------------------------------------------
# Deletion on a database that never ran the v11 → v12 migration
# ----------------------------------------------------------------------

#: The core ``user_version`` that first declares ``ON DELETE CASCADE`` on the
#: child tables (reached by ``_migrate_core_v11_to_v12``). Anything below it is
#: a "pre-cascade" database: the constraint is simply not in the schema.
_FIRST_CASCADING_CORE_VERSION = 12

_PRE_CASCADE_DIMENSION = 3
_PRE_CASCADE_MODEL = "unmigrated-probe"
_SURVIVOR_ID = "t-keep"
_DELETED_ID = "t-drop"
#: Chosen so the *deleted* thought is the nearest neighbour and the survivor the
#: second: a result set that dropped everything, or returned everything, is
#: distinguishable from one that returns exactly the phantom plus the survivor.
_QUERY_VECTOR = [0.9, 0.1, 0.0]


@dataclass(frozen=True)
class _PostDeleteObservation:
    """What one delete-then-reconcile run left behind, read back from storage."""

    core_version: int
    #: ``owner_id``s still present in the ``embedding`` table after the delete.
    embedding_owner_ids: list[str]
    #: Rows ``sync_embeddings`` considered valid backfill sources afterwards.
    backfilled: int
    #: Identifiers ``search_similar`` returns after the reconcile.
    search_similar_ids: list[str]
    #: The deleted id hydrated through the public read API.
    hydrated_deleted: ThoughtRecord | None
    #: ``delete_thought``'s own report — a self-report, asserted last.
    delete_reported: bool


async def _delete_then_reconcile(
    db_path: Path,
    *,
    pre_cascade: bool,
) -> _PostDeleteObservation:
    """Run the same deletion scenario against one schema version or the other.

    The **only** difference between the two arms is the schema the store is
    opened on: ``pre_cascade=True`` bootstraps the legacy core-11 tables (the
    shared :meth:`TestMigrationV11ToV12._bootstrap_v11_schema`, i.e. no FK
    clauses), ``pre_cascade=False`` runs ``ensure_schema`` up to the head
    version. Corpus, vectors, query, delete call and reconcile are identical.

    The reconcile step is a direct ``sync_embeddings`` call, which is what a
    fresh sqlite-vec-enabled open performs against an existing database.
    """
    db = await aiosqlite.connect(str(db_path))
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA foreign_keys=ON")
    store = SqliteEngravaCore(db, embedding_provider=None, auto_embed=False)
    store._owns_connection = True
    try:
        if pre_cascade:
            await TestMigrationV11ToV12._bootstrap_v11_schema(db)
        else:
            await store.ensure_schema()
        cursor = await db.execute("PRAGMA user_version")
        version_row = await cursor.fetchone()
        assert version_row is not None
        core_version = int(version_row[0])

        await store._configure_vector_backend(
            backend_name="sqlite-vec",
            embedding_dimension=_PRE_CASCADE_DIMENSION,
        )
        backend = store._vector_backend
        # Precondition, not a self-report: without a real vec0 backend the whole
        # scenario measures the numpy fallback instead.
        assert isinstance(backend, SqliteVecSearchBackend)

        for tid in (_SURVIVOR_ID, _DELETED_ID):
            await db.execute(
                "INSERT INTO thought (thought_id, thought_type, essence, content, priority) "
                "VALUES (?, 'OBSERVATION', ?, ?, 'P3')",
                (tid, tid, tid),
            )
        await db.commit()
        await store.store_embedding(
            thought_id=_SURVIVOR_ID,
            vector=[1.0, 0.0, 0.0],
            model_name=_PRE_CASCADE_MODEL,
        )
        await store.store_embedding(
            thought_id=_DELETED_ID,
            vector=list(_QUERY_VECTOR),
            model_name=_PRE_CASCADE_MODEL,
        )
        # Precondition: both ids are live search hits before anything is deleted,
        # so an "absent afterwards" reading cannot be an index that never worked.
        seeded = [r[0] for r in await store.search_similar(_QUERY_VECTOR, top_k=5)]
        assert seeded == [_DELETED_ID, _SURVIVOR_ID], seeded

        delete_reported = await store.delete_thought(_DELETED_ID)

        cursor = await db.execute("SELECT owner_id FROM embedding WHERE owner_type = 'THOUGHT'")
        embedding_owner_ids = [str(row["owner_id"]) for row in await cursor.fetchall()]
        backfilled = await backend.sync_embeddings(db)
        search_similar_ids = [r[0] for r in await store.search_similar(_QUERY_VECTOR, top_k=5)]
        hydrated_deleted = await store.get_thought(_DELETED_ID)
    finally:
        await store.close()

    return _PostDeleteObservation(
        core_version=core_version,
        embedding_owner_ids=embedding_owner_ids,
        backfilled=backfilled,
        search_similar_ids=search_similar_ids,
        hydrated_deleted=hydrated_deleted,
        delete_reported=delete_reported,
    )


@pytest.mark.skipif(
    importlib.util.find_spec("sqlite_vec") is None,
    reason="sqlite-vec package not installed",
)
class TestDeletionOnAPreCascadeSchema:
    """Deleting on a database below core-12 no longer leaves the identifier reachable.

    This fixes what this class used to pin as a **documented
    non-guarantee** (see ``docs/known-limitations.md`` — "Deletion on a
    database that has not been migrated" — and ``docs/data-lifecycle.md``,
    both updated alongside this test): a vector is now owned by a *live*
    thought rather than by the presence of an ``embedding`` row, enforced in
    ``delete_thought``'s own explicit child delete (no cascade required),
    ``sync_embeddings``'s reconciliation join, and the ``vec0`` search
    resolution. The first test below used to demonstrate the resurrection;
    it now demonstrates that it no longer happens — red on the pre-fix tree,
    green here.
    """

    async def test_pre_cascade_delete_does_not_resurrect_the_identifier(
        self,
        tmp_path: Path,
    ) -> None:
        """Core-11: no cascade exists, and the deleted id still does not come back.

        Chain, each link asserted here: ``delete_thought`` removes the
        ``embedding`` row itself rather than relying on a cascade that does
        not exist at this schema version, so ``sync_embeddings`` has no
        dangling row to treat as a valid backfill source, and
        ``search_similar`` never sees the deleted identifier again. The
        content was already gone before this fix and still is: hydrating the
        id yields ``None``.
        """
        observed = await _delete_then_reconcile(
            tmp_path / "pre-cascade.db",
            pre_cascade=True,
        )

        assert observed.core_version < _FIRST_CASCADING_CORE_VERSION
        # The mechanism: delete_thought's own explicit child delete removed
        # the embedding row without needing a cascade this schema lacks.
        assert observed.embedding_owner_ids == [_SURVIVOR_ID]
        assert observed.backfilled == 0
        # The fix: the deleted identifier does not return.
        assert _DELETED_ID not in observed.search_similar_ids
        # Second control — the un-deleted sibling is still returned, so this
        # cannot pass against a search path that returns nothing at all.
        assert _SURVIVOR_ID in observed.search_similar_ids
        assert observed.hydrated_deleted is None
        assert observed.delete_reported is True

    async def test_pre_cascade_reconcile_does_not_resurrect_a_pre_existing_dangling_row(
        self,
        tmp_path: Path,
    ) -> None:
        """A dangling row from *before* this fix still cannot come back.

        ``delete_thought``'s explicit child delete (proven above) stops *new*
        dangling rows, but a database that already accumulated one under the
        pre-fix engine needs the other half of the invariant:
        ``sync_embeddings`` itself must not treat a dangling ``embedding`` row
        as proof its thought is live, however that row came to exist. Bypasses
        ``delete_thought`` entirely — a raw ``DELETE FROM thought`` simulates
        exactly the historical damage — so this exercises the reconciliation
        join on its own, independent of the delete-path fix.
        """
        db_path = tmp_path / "pre-existing-dangling.db"
        db = await aiosqlite.connect(str(db_path))
        db.row_factory = aiosqlite.Row
        try:
            await TestMigrationV11ToV12._bootstrap_v11_schema(db)
            store = SqliteEngravaCore(db, embedding_provider=None, auto_embed=False)
            # store.close() below is a no-op on the connection unless the
            # store is told it owns it (see SqliteEngravaCore.close()'s own
            # docstring: "No-op on the connection when it is caller-managed").
            # Without this, `db`'s connection -- and its non-daemon aiosqlite
            # worker thread -- is never actually closed here, only relies on
            # `__del__` for cleanup; confirmed via a leaked-thread repro that
            # this is exactly what happened. `_delete_then_reconcile` above
            # already sets this for the same reason.
            store._owns_connection = True
            await store._configure_vector_backend(
                backend_name="sqlite-vec",
                embedding_dimension=_PRE_CASCADE_DIMENSION,
            )
            backend = store._vector_backend
            assert isinstance(backend, SqliteVecSearchBackend)

            for tid in (_SURVIVOR_ID, _DELETED_ID):
                await db.execute(
                    "INSERT INTO thought (thought_id, thought_type, essence, content, priority) "
                    "VALUES (?, 'OBSERVATION', ?, ?, 'P3')",
                    (tid, tid, tid),
                )
            await db.commit()
            await store.store_embedding(
                thought_id=_SURVIVOR_ID, vector=[1.0, 0.0, 0.0], model_name=_PRE_CASCADE_MODEL
            )
            await store.store_embedding(
                thought_id=_DELETED_ID, vector=list(_QUERY_VECTOR), model_name=_PRE_CASCADE_MODEL
            )

            # Simulate pre-fix damage directly: a bare parent delete, with no
            # cascade at this schema version and no explicit child delete
            # (this is deliberately *not* a call to delete_thought), leaving
            # the embedding row dangling exactly as an old engrava build
            # would have.
            await db.execute("DELETE FROM thought WHERE thought_id = ?", (_DELETED_ID,))
            await db.commit()
            dangling = [
                str(row["owner_id"])
                for row in await (
                    await db.execute("SELECT owner_id FROM embedding WHERE owner_type = 'THOUGHT'")
                ).fetchall()
            ]
            assert sorted(dangling) == [_DELETED_ID, _SURVIVOR_ID], (
                "fixture precondition failed: no dangling embedding row to reconcile against"
            )

            backfilled = await backend.sync_embeddings(db)
            search_similar_ids = [r[0] for r in await store.search_similar(_QUERY_VECTOR, top_k=5)]
        finally:
            await store.close()

        assert backfilled == 0, "the dangling row's owner is gone; it must not be backfilled"
        assert _DELETED_ID not in search_similar_ids
        assert _SURVIVOR_ID in search_similar_ids

    async def test_head_schema_delete_removes_the_identifier(
        self,
        tmp_path: Path,
    ) -> None:
        """Head schema: the same run leaves the id absent from the same query.

        The schema-version control for the test above. Same corpus, same
        query, same delete — only the schema differs, so a phantom that shows
        up there and not here is attributable to the missing cascade and to
        nothing else about the scenario.
        """
        observed = await _delete_then_reconcile(
            tmp_path / "head.db",
            pre_cascade=False,
        )

        assert observed.core_version >= _FIRST_CASCADING_CORE_VERSION
        assert observed.embedding_owner_ids == [_SURVIVOR_ID]
        assert observed.backfilled == 0
        assert _DELETED_ID not in observed.search_similar_ids
        assert _SURVIVOR_ID in observed.search_similar_ids
        assert observed.hydrated_deleted is None
        assert observed.delete_reported is True
