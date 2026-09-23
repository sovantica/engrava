"""``run_hygiene``'s archive and GC stages recover with their journal entry.

A sibling to ``test_create_delete_journal_append_atomicity.py``, which pins the
same failure-atomic-unit property for the create and delete paths: the
write(s) a journaled operation makes and the journal append describing them
must be one unit, so a failed or cancelled append unwinds the write too
instead of leaving it durable for a later, unrelated commit with no journal
entry to show for it.

This file covers the two remaining sites: ``_hygiene_archive`` and
``_hygiene_gc`` (including its nested ``retire_orphan_reflections`` call,
which writes through ``update_thought`` and whose own ``_maybe_commit()``
must not end ``run_hygiene``'s unit early).
"""

from __future__ import annotations

import asyncio
import datetime
import importlib.util
from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import (
    DefaultEngravaHooks,
    DeriveContext,
    DeriveGates,
    EdgeRecord,
    EdgeType,
    HygienePolicyConfig,
    KnowledgeSource,
    LifecycleStatus,
    Priority,
    SqliteEngravaCore,
    ThoughtRecord,
    ThoughtType,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from engrava.domain.protocols.derived_records import DerivedRecord

# A wall-clock instant every test pins so the minimum-inactivity-age gate and
# the GC restore window are both deterministic.
_NOW = datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.UTC)


# ---------------------------------------------------------------------------
# Domain factories
# ---------------------------------------------------------------------------


def _thought(
    thought_id: str,
    *,
    thought_type: ThoughtType = ThoughtType.OBSERVATION,
    lifecycle_status: LifecycleStatus = LifecycleStatus.ACTIVE,
    updated_cycle: int = 0,
    archived_at_cycle: int | None = None,
    archived_at: str | None = None,
    pinned: bool = False,
) -> ThoughtRecord:
    """A thought pre-aged so an enabled, zero-inactivity-gate policy can act on it."""
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=thought_type,
        essence=f"essence {thought_id}",
        content=f"content of {thought_id}",
        priority=Priority.P3,
        lifecycle_status=lifecycle_status,
        created_cycle=0,
        updated_cycle=updated_cycle,
        source="test",
        confidence=0.5,
        action_outcome_score=0.0,
        created_at="2000-01-01T00:00:00+00:00",
        updated_at="2000-01-01T00:00:00+00:00",
        archived_at_cycle=archived_at_cycle,
        archived_at=archived_at,
        pinned=pinned,
    )


def _gc_policy(**overrides: object) -> HygienePolicyConfig:
    """An enabled policy that archives nothing new and GCs eagerly.

    ``eviction_threshold=0.0`` means the archive stage's own eviction-score
    comparison (``eviction_score < eviction_threshold``) never fires, so
    stage 1 never interferes with a scenario built around pre-seeded,
    already-archived GC candidates. Both GC restore windows are open
    (``0``), so anything hygiene-archived is immediately GC-eligible.
    """
    params: dict[str, object] = {
        "enabled": True,
        "min_inactivity_age_seconds": 0,
        "eviction_threshold": 0.0,
        "auto_gc_enabled": True,
        "gc_min_archive_age_cycles": 0,
        "gc_restore_window_seconds": 0,
    }
    params.update(overrides)
    return HygienePolicyConfig(**params)  # type: ignore[arg-type]


def _archive_policy(**overrides: object) -> HygienePolicyConfig:
    """An enabled policy that archives everything ACTIVE/CREATED and never GCs."""
    params: dict[str, object] = {
        "enabled": True,
        "min_inactivity_age_seconds": 0,
        "eviction_threshold": 1.0,
        "auto_gc_enabled": False,
    }
    params.update(overrides)
    return HygienePolicyConfig(**params)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Store + connection helpers
# ---------------------------------------------------------------------------


async def _open_store(
    db_path: Path,
    *,
    policy: HygienePolicyConfig,
    derive_gates: DeriveGates | None = None,
    hooks: DefaultEngravaHooks | None = None,
) -> SqliteEngravaCore:
    """A fresh, journal-enabled store on its own real, on-disk connection."""
    conn = await aiosqlite.connect(str(db_path))
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode = WAL")
    await conn.execute("PRAGMA foreign_keys = ON")
    store = SqliteEngravaCore(
        conn,
        journal_enabled=True,
        hygiene_policy=policy,
        derive_gates=derive_gates,
        hooks=hooks,
        embedding_provider=None,
        auto_embed=False,
    )
    await store.ensure_schema()
    return store


# Skip the real-extension test when sqlite-vec is absent, but never let it
# silently pass when it is installed and broken. Duplicated from
# ``tests/test_sqlite_vec.py`` (rather than imported) so this file does not
# pull that module's own import graph into a strict mypy run alongside it --
# it is a one-line mark and two tiny raw-SQL reads, not worth the coupling.
sqlite_vec_required = pytest.mark.skipif(
    importlib.util.find_spec("sqlite_vec") is None,
    reason="sqlite-vec package not installed",
)


async def _vec_rowids(store: SqliteEngravaCore) -> set[int]:
    """Return the set of rowids currently present in ``embedding_vec``."""
    cursor = await store._db.execute("SELECT rowid FROM embedding_vec")
    return {int(row["rowid"]) for row in await cursor.fetchall()}


async def _embedding_rowid(store: SqliteEngravaCore, thought_id: str) -> int | None:
    """Return the ``embedding`` rowid for a thought, or ``None`` if absent."""
    cursor = await store._db.execute(
        "SELECT rowid FROM embedding WHERE owner_type = 'THOUGHT' AND owner_id = ?",
        (thought_id,),
    )
    row = await cursor.fetchone()
    return int(row["rowid"]) if row is not None else None


async def _open_vec_store(db_path: Path, *, policy: HygienePolicyConfig) -> SqliteEngravaCore:
    """A journal-enabled store with a real, loaded sqlite-vec backend.

    The numpy backend's ``_purge_orphan_vector`` is a documented no-op, so it
    cannot exercise "the purge writes, then the unit's append fails" -- only a
    real vector backend gives that write something to unwind.
    """
    conn = await aiosqlite.connect(str(db_path))
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode = WAL")
    await conn.execute("PRAGMA foreign_keys = ON")
    store = SqliteEngravaCore(conn, journal_enabled=True, hygiene_policy=policy)
    await store.ensure_schema()
    await store._configure_vector_backend(backend_name="sqlite-vec", embedding_dimension=3)
    return store


async def _reopen_store(db_path: Path) -> SqliteEngravaCore:
    """Reopen the same on-disk database on a brand-new connection.

    Used to prove durability (or its absence) purely from what actually made
    it to disk, independent of any in-process state the original connection
    might still be carrying.
    """
    conn = await aiosqlite.connect(str(db_path))
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA foreign_keys = ON")
    return SqliteEngravaCore(conn, embedding_provider=None, auto_embed=False)


async def _raw_row(db_path: Path, table: str, key: str, value: str) -> aiosqlite.Row | None:
    """Read a row straight from storage, on a brand-new independent connection.

    Unlike a read through the store's own connection -- which sees its own
    uncommitted writes -- this proves (or disproves) durability from a
    genuinely separate connection to the same on-disk file. Used mid-window,
    where a leaked early commit would otherwise be invisible to every
    same-connection assertion.
    """
    conn = await aiosqlite.connect(str(db_path))
    conn.row_factory = aiosqlite.Row
    try:
        cursor = await conn.execute(f"SELECT * FROM {table} WHERE {key} = ?", (value,))  # noqa: S608 -- test literals
        return await cursor.fetchone()
    finally:
        await conn.close()


async def _journal_entry_count(store: SqliteEngravaCore) -> int:
    cursor = await store._db.execute("SELECT COUNT(*) FROM journal_entry")
    row = await cursor.fetchone()
    assert row is not None
    return int(row[0])


async def _row_count(cursor: aiosqlite.Cursor) -> int:
    """Fetch a ``SELECT COUNT(*) AS n`` cursor's single row, narrowed before indexing."""
    row = await cursor.fetchone()
    assert row is not None
    return int(row["n"])


async def _raw_lifecycle(store: SqliteEngravaCore, thought_id: str) -> str | None:
    cursor = await store._db.execute(
        "SELECT lifecycle_status FROM thought WHERE thought_id = ?", (thought_id,)
    )
    row = await cursor.fetchone()
    return None if row is None else str(row["lifecycle_status"])


async def _write_raw_marker(store: SqliteEngravaCore, thought_id: str) -> None:
    """Insert a marker thought directly, bypassing the store's own commit."""
    thought = _thought(thought_id)
    await store._db.execute(store._CORE_INSERT_SQL, store._thought_to_core_params(thought))


def _fail_nth_journal_append(store: SqliteEngravaCore, exc: BaseException, *, n: int = 1) -> None:
    """Make the journal's own chain-tail read raise/cancel on its *n*-th call.

    Counts from the moment this is installed, not from the store's lifetime —
    identical helper to the one in ``test_create_delete_journal_append_atomicity.py``,
    duplicated here rather than imported since that file is explicitly off limits
    for this work item.
    """
    journal = store._journal
    assert journal is not None
    original = journal._get_latest_entry_state
    calls = {"count": 0}

    async def wrapper(*args: object, **kwargs: object) -> object:
        calls["count"] += 1
        if calls["count"] == n:
            raise exc
        return await original(*args, **kwargs)

    journal._get_latest_entry_state = wrapper  # type: ignore[method-assign,assignment]


def _new_runtime_error() -> BaseException:
    return RuntimeError("forced journal append failure")


def _new_cancelled_error() -> BaseException:
    return asyncio.CancelledError()


_INJECTIONS = [_new_runtime_error, _new_cancelled_error]
_INJECTION_IDS = ["exception", "cancellation"]


class _ListProducer(DefaultEngravaHooks):
    """A derived-records producer that records every dispatch it receives."""

    def __init__(self, records: Sequence[DerivedRecord] = ()) -> None:
        self._records = list(records)
        self.calls = 0
        self.source_ids: list[str] = []

    async def derive_records(
        self,
        thought: ThoughtRecord,
        ctx: DeriveContext,
    ) -> Sequence[DerivedRecord]:
        self.calls += 1
        self.source_ids.append(ctx.source_thought_id)
        return self._records


# ---------------------------------------------------------------------------
# Failure inside each stage, at least two candidates, failure on the second.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("exc_factory", _INJECTIONS, ids=_INJECTION_IDS)
class TestArchiveStageFailureUnwindsBothCandidates:
    async def test_standalone(
        self, exc_factory: Callable[[], BaseException], tmp_path: Path
    ) -> None:
        store = await _open_store(tmp_path / "archive-fail.db", policy=_archive_policy())
        try:
            await store.create_thought(_thought("t-1"))
            await store.create_thought(_thought("t-2"))
            journal_count_before = await _journal_entry_count(store)
            exc = exc_factory()
            _fail_nth_journal_append(store, exc, n=2)

            with pytest.raises(type(exc)) as excinfo:
                await store.run_hygiene(current_cycle=1000, now=_NOW)

            assert excinfo.value is exc
            assert await _raw_lifecycle(store, "t-1") == "ACTIVE"
            assert await _raw_lifecycle(store, "t-2") == "ACTIVE"
            assert store._db.in_transaction is False
            # The first candidate's own archive append had already succeeded
            # (and would otherwise be a real, distinct journal row) before the
            # second candidate's append failed; this shows it did not survive
            # either.
            assert await _journal_entry_count(store) == journal_count_before
        finally:
            await store._db.close()


@pytest.mark.parametrize("exc_factory", _INJECTIONS, ids=_INJECTION_IDS)
class TestGcStageFailureUnwindsBothCandidates:
    async def test_standalone(
        self, exc_factory: Callable[[], BaseException], tmp_path: Path
    ) -> None:
        store = await _open_store(tmp_path / "gc-fail.db", policy=_gc_policy())
        try:
            await store.create_thought(
                _thought("t-1", lifecycle_status=LifecycleStatus.ARCHIVED, archived_at_cycle=0)
            )
            await store.create_thought(
                _thought("t-2", lifecycle_status=LifecycleStatus.ARCHIVED, archived_at_cycle=0)
            )
            journal_count_before = await _journal_entry_count(store)
            exc = exc_factory()
            _fail_nth_journal_append(store, exc, n=2)

            with pytest.raises(type(exc)) as excinfo:
                await store.run_hygiene(current_cycle=1000, now=_NOW)

            assert excinfo.value is exc
            # Neither delete survived -- the first candidate's own delete had
            # already completed (and journaled) when the second's append
            # failed, but the pass is one unit, so its own success is undone
            # along with the second's failure.
            assert await _raw_lifecycle(store, "t-1") == "ARCHIVED"
            assert await _raw_lifecycle(store, "t-2") == "ARCHIVED"
            assert store._db.in_transaction is False
            # The first candidate's own DELETE_THOUGHT journal row had
            # already been appended before the second candidate's append
            # failed; this shows it did not survive either.
            assert await _journal_entry_count(store) == journal_count_before
        finally:
            await store._db.close()


# ---------------------------------------------------------------------------
# A nested retirement (its own commit suppressed) followed by a later GC
# failure: the retirement must not survive either.
# ---------------------------------------------------------------------------


class TestNestedRetirementThenLaterFailure:
    async def test_retirement_is_undone_with_the_failed_delete(self, tmp_path: Path) -> None:
        db_path = tmp_path / "nested-retirement.db"
        store = await _open_store(db_path, policy=_gc_policy())
        try:
            await store.create_thought(
                _thought("src", lifecycle_status=LifecycleStatus.ARCHIVED, archived_at_cycle=0)
            )
            await store.create_thought(_thought("refl", thought_type=ThoughtType.REFLECTION))
            await store.create_edge(
                EdgeRecord(
                    edge_id="e-1",
                    from_thought_id="refl",
                    to_thought_id="src",
                    edge_type=EdgeType.CONSOLIDATED_FROM,
                    weight=0.5,
                    created_cycle=0,
                    source=KnowledgeSource.EXPERIENCE,
                )
            )
            assert await _raw_lifecycle(store, "refl") == "ACTIVE"
            journal_count_before = await _journal_entry_count(store)

            exc = RuntimeError("forced journal append failure")
            # n=1 is the retirement's own UPDATE_THOUGHT append (must succeed
            # so the retirement genuinely happens); n=2 is the GC candidate's
            # own DELETE_THOUGHT append (the one that fails).
            _fail_nth_journal_append(store, exc, n=2)

            with pytest.raises(RuntimeError) as excinfo:
                await store.run_hygiene(current_cycle=1000, now=_NOW)

            assert excinfo.value is exc
            assert store._db.in_transaction is False
            # Same-connection check: the retirement must already be gone here,
            # not just after reopening -- a bare rollback would show this too.
            assert await _raw_lifecycle(store, "refl") == "ACTIVE"
            assert await _raw_lifecycle(store, "src") == "ARCHIVED"
            # The retirement's own UPDATE_THOUGHT append (n=1) had already
            # succeeded before the GC candidate's append (n=2) failed; this
            # shows that row did not survive either.
            assert await _journal_entry_count(store) == journal_count_before
        finally:
            await store._db.close()

        # Reopen on a brand-new connection: the retirement's lifecycle flip
        # must not have reached disk either.
        reopened = await _reopen_store(db_path)
        try:
            assert await _raw_lifecycle(reopened, "refl") == "ACTIVE"
            assert await _raw_lifecycle(reopened, "src") == "ARCHIVED"
            assert await _journal_entry_count(reopened) == journal_count_before
            cursor = await reopened._db.execute("SELECT COUNT(*) AS n FROM edge")
            assert await _row_count(cursor) == 1
        finally:
            await reopened._db.close()


# ---------------------------------------------------------------------------
# A retirement-only pass: archived_count == gc_count == 0, but the retirement
# itself is a surviving write and must be durable.
# ---------------------------------------------------------------------------


class TestRetirementOnlyPass:
    async def test_retirement_alone_commits_and_closes_the_transaction(
        self, tmp_path: Path
    ) -> None:
        db_path = tmp_path / "retirement-only.db"
        store = await _open_store(db_path, policy=_gc_policy())
        try:
            await store.create_thought(
                _thought("src", lifecycle_status=LifecycleStatus.ARCHIVED, archived_at_cycle=0)
            )
            await store.create_thought(_thought("refl", thought_type=ThoughtType.REFLECTION))
            await store.create_edge(
                EdgeRecord(
                    edge_id="e-1",
                    from_thought_id="refl",
                    to_thought_id="src",
                    edge_type=EdgeType.CONSOLIDATED_FROM,
                    weight=0.5,
                    created_cycle=0,
                    source=KnowledgeSource.EXPERIENCE,
                )
            )
            # Every GC candidate is vetoed -- gc_count stays 0 -- but the
            # eligible set is still non-empty, so retire_orphan_reflections
            # still runs and still retires "refl".
            await store._db.execute(
                "CREATE TRIGGER src_delete_ignore BEFORE DELETE ON thought "
                "WHEN OLD.thought_id = 'src' BEGIN SELECT RAISE(IGNORE); END"
            )

            result = await store.run_hygiene(current_cycle=1000, now=_NOW)

            assert result.archived_count == 0
            assert result.gc_count == 0
            assert store._db.in_transaction is False
        finally:
            await store._db.close()

        reopened = await _reopen_store(db_path)
        try:
            assert await _raw_lifecycle(reopened, "refl") == "ARCHIVED"
            assert await _raw_lifecycle(reopened, "src") == "ARCHIVED"
        finally:
            await reopened._db.close()


# ---------------------------------------------------------------------------
# A pass that writes nothing at all leaves no transaction open.
# ---------------------------------------------------------------------------


class TestNoWritePass:
    async def test_empty_store_leaves_no_transaction(self, tmp_path: Path) -> None:
        store = await _open_store(tmp_path / "no-write.db", policy=_gc_policy())
        try:
            result = await store.run_hygiene(current_cycle=1000, now=_NOW)
            assert result.archived_count == 0
            assert result.gc_count == 0
            assert store._db.in_transaction is False
        finally:
            await store._db.close()


# ---------------------------------------------------------------------------
# A pass whose only activity is undone, inside a caller-held raw transaction:
# a BEFORE DELETE trigger writes an audit row and then vetoes with
# RAISE(IGNORE). total_changes still advances (it is monotonic); wrote_anything
# must not.
# ---------------------------------------------------------------------------


class TestVetoedGcCandidateInsideRawTransaction:
    async def test_caller_transaction_and_audit_row_both_survive_untouched(
        self, tmp_path: Path
    ) -> None:
        db_path = tmp_path / "veto-raw.db"
        store = await _open_store(db_path, policy=_gc_policy())
        try:
            await store.create_thought(
                _thought("src", lifecycle_status=LifecycleStatus.ARCHIVED, archived_at_cycle=0)
            )
            await store._db.execute("CREATE TABLE audit_log (id TEXT NOT NULL)")
            await store._db.execute(
                "CREATE TRIGGER src_delete_audit_veto BEFORE DELETE ON thought "
                "WHEN OLD.thought_id = 'src' BEGIN "
                "INSERT INTO audit_log (id) VALUES (OLD.thought_id); "
                "SELECT RAISE(IGNORE); END"
            )

            await store._db.execute("BEGIN")
            await _write_raw_marker(store, "x-marker")

            result = await store.run_hygiene(current_cycle=1000, now=_NOW)

            assert result.gc_count == 0
            assert store._db.in_transaction is True
            assert await store._get_thought_row("x-marker") is not None
            assert await store._get_thought_row("src") is not None
            cursor = await store._db.execute("SELECT COUNT(*) AS n FROM audit_log")
            assert await _row_count(cursor) == 0

            await store._db.commit()

            # A false "wrote something" decision would have committed the
            # caller's transaction from *inside* run_hygiene already; this
            # re-check after the caller's own explicit commit additionally
            # confirms the audit row never became durable at all.
            cursor = await store._db.execute("SELECT COUNT(*) AS n FROM audit_log")
            assert await _row_count(cursor) == 0
        finally:
            await store._db.close()

        reopened = await _reopen_store(db_path)
        try:
            assert await reopened._get_thought_row("x-marker") is not None
            cursor = await reopened._db.execute("SELECT COUNT(*) AS n FROM audit_log")
            assert await _row_count(cursor) == 0
        finally:
            await reopened._db.close()


# ---------------------------------------------------------------------------
# Inside a caller-held suspend_auto_commit window: a failed pass undoes only
# its own writes, and the window's own commit at its outermost exit still
# publishes the caller's earlier write.
# ---------------------------------------------------------------------------


class TestFailureInsideSuspendAutoCommitWindow:
    async def test_caught_failure_leaves_only_the_hygiene_write_undone(
        self, tmp_path: Path
    ) -> None:
        db_path = tmp_path / "window-fail-caught.db"
        store = await _open_store(db_path, policy=_archive_policy())
        try:
            await store.create_thought(_thought("t-1"))

            async with store.suspend_auto_commit():
                # Pinned so this write itself is never picked up as its own
                # hygiene candidate -- it is here purely as the caller's own
                # "earlier write" (X), not as a second thing for the archive
                # stage to act on.
                await store.create_thought(_thought("marker", pinned=True))
                journal_count_before = await _journal_entry_count(store)
                exc = RuntimeError("forced journal append failure")
                _fail_nth_journal_append(store, exc, n=1)

                with pytest.raises(RuntimeError) as excinfo:
                    await store.run_hygiene(current_cycle=1000, now=_NOW)

                assert excinfo.value is exc
                assert await _raw_lifecycle(store, "t-1") == "ACTIVE"
                # The hygiene pass's own single append is the one that was
                # made to fail; nothing of it should be added to the journal,
                # not even transiently, on the same connection.
                assert await _journal_entry_count(store) == journal_count_before
                # Still inside the window: the transaction the window owns is
                # still open, and the marker -- durable only once the
                # window's own outermost commit runs -- must not be visible
                # from a second, independent connection to the same file
                # yet. A recovery path that committed early here would still
                # pass every assertion above (this connection sees its own
                # uncommitted writes) and only this one catches it.
                assert store._db.in_transaction is True
                assert await _raw_row(db_path, "thought", "thought_id", "marker") is None

            assert store._db.in_transaction is False
            assert await store._get_thought_row("marker") is not None
            assert await _raw_lifecycle(store, "t-1") == "ACTIVE"
            # Re-checked after the window's own outermost commit: publishing
            # the caller's earlier write (t-1's create + marker's create) must
            # not also publish anything from the failed hygiene pass.
            assert await _journal_entry_count(store) == journal_count_before
        finally:
            await store._db.close()

        # And now durable, seen from a fresh connection.
        assert await _raw_row(db_path, "thought", "thought_id", "marker") is not None

    async def test_uncaught_failure_rolls_back_the_whole_window(self, tmp_path: Path) -> None:
        """An exception nothing inside the window catches discards the caller's write too.

        ``suspend_auto_commit`` owns the transaction end-to-end: its own
        outermost rollback fires on *any* exception escaping the window, and
        undoes everything the window wrote -- the caller's own earlier
        "marker" create is not something the hygiene pass's own failure
        atomicity is responsible for preserving here; that is
        ``suspend_auto_commit``'s existing, unchanged contract, exercised
        with a hygiene-shaped failure that nothing catches.
        """
        db_path = tmp_path / "window-fail-uncaught.db"
        store = await _open_store(db_path, policy=_archive_policy())
        try:
            await store.create_thought(_thought("t-1"))
            exc = RuntimeError("forced journal append failure")

            async def _run_window_letting_the_failure_escape() -> None:
                async with store.suspend_auto_commit():
                    await store.create_thought(_thought("marker", pinned=True))
                    _fail_nth_journal_append(store, exc, n=1)
                    await store.run_hygiene(current_cycle=1000, now=_NOW)

            with pytest.raises(RuntimeError) as excinfo:
                await _run_window_letting_the_failure_escape()

            assert excinfo.value is exc
            assert store._db.in_transaction is False
            # The window's own outermost rollback discarded everything it
            # held -- the caller's own earlier write along with the hygiene
            # pass's, not only the latter.
            assert await store._get_thought_row("marker") is None
            assert await _raw_lifecycle(store, "t-1") == "ACTIVE"
        finally:
            await store._db.close()

        assert await _raw_row(db_path, "thought", "thought_id", "marker") is None


# ---------------------------------------------------------------------------
# The task-local suppression must not leak onto the instance-wide flag that
# _dispatch_derivation reads.
# ---------------------------------------------------------------------------


class TestSuppressionIsTaskLocal:
    async def test_instance_wide_flag_stays_false_and_a_second_dispatch_still_derives(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        producer = _ListProducer([])
        store = await _open_store(
            tmp_path / "suppression-task-local.db",
            policy=_gc_policy(),
            derive_gates=DeriveGates(enabled=False),
            hooks=producer,
        )
        try:
            await store.create_thought(
                _thought("src", lifecycle_status=LifecycleStatus.ARCHIVED, archived_at_cycle=0)
            )
            await store.create_thought(_thought("refl", thought_type=ThoughtType.REFLECTION))
            await store.create_edge(
                EdgeRecord(
                    edge_id="e-1",
                    from_thought_id="refl",
                    to_thought_id="src",
                    edge_type=EdgeType.CONSOLIDATED_FROM,
                    weight=0.5,
                    created_cycle=0,
                    source=KnowledgeSource.EXPERIENCE,
                )
            )
            # Already-committed source, created before the gate is enabled so
            # its own on-store dispatch never fired for it.
            already_committed = _thought("second-src")
            await store.create_thought(already_committed)

            # Enabling the gate now, after creation, simulates "a second task
            # that reaches _dispatch_derivation for an already-committed
            # source" without racing real OS-level concurrency: what matters
            # is the instance-wide flag's value at the moment of the call, not
            # which task object issues it.
            store._derive_gates = DeriveGates(enabled=True)

            recorded_skip_auto_commit: list[bool] = []
            original_retire = SqliteEngravaCore.retire_orphan_reflections
            hygiene_paused = asyncio.Event()
            hygiene_may_continue = asyncio.Event()

            async def _patched_retire(self: SqliteEngravaCore) -> int:
                recorded_skip_auto_commit.append(self._skip_auto_commit)
                hygiene_paused.set()
                await hygiene_may_continue.wait()
                return await original_retire(self)

            monkeypatch.setattr(SqliteEngravaCore, "retire_orphan_reflections", _patched_retire)

            hygiene_task = asyncio.ensure_future(store.run_hygiene(current_cycle=1000, now=_NOW))
            await hygiene_paused.wait()

            assert recorded_skip_auto_commit == [False]

            await store._dispatch_derivation(already_committed)
            assert producer.calls == 1
            assert producer.source_ids == ["second-src"]

            hygiene_may_continue.set()
            result = await hygiene_task
            assert result.gc_count == 1
        finally:
            await store._db.close()


# ---------------------------------------------------------------------------
# The vector purge is part of the GC unit: a real sqlite-vec vector a
# candidate's purge actually deletes must unwind along with the rest of the
# unit when a later candidate's own append fails. The numpy backend's
# `_purge_orphan_vector` is a documented no-op, so only a real vector backend
# gives the purge something to unwind -- see `_open_vec_store`.
# ---------------------------------------------------------------------------


@sqlite_vec_required
class TestGcVectorPurgeIsPartOfTheUnit:
    async def test_both_candidates_are_fully_restored_after_the_second_candidates_failure(
        self, tmp_path: Path
    ) -> None:
        db_path = tmp_path / "gc-vector-purge.db"
        store = await _open_vec_store(db_path, policy=_gc_policy())
        try:
            await store.create_thought(
                _thought("t-1", lifecycle_status=LifecycleStatus.ARCHIVED, archived_at_cycle=0)
            )
            await store.store_embedding(
                thought_id="t-1", vector=[1.0, 0.0, 0.0], model_name="test-fixture-model"
            )
            await store.create_thought(
                _thought("t-2", lifecycle_status=LifecycleStatus.ARCHIVED, archived_at_cycle=0)
            )
            await store.store_embedding(
                thought_id="t-2", vector=[0.0, 1.0, 0.0], model_name="test-fixture-model"
            )

            t1_embedding_rowid = await _embedding_rowid(store, "t-1")
            assert t1_embedding_rowid is not None
            assert t1_embedding_rowid in await _vec_rowids(store)
            t2_embedding_rowid = await _embedding_rowid(store, "t-2")
            assert t2_embedding_rowid is not None
            assert t2_embedding_rowid in await _vec_rowids(store)

            exc = RuntimeError("forced journal append failure")
            # Eligible order is archived_at_cycle ASC, thought_id ASC, so
            # "t-1" is the first GC candidate processed and "t-2" the second:
            # n=1 is t-1's own successful delete + purge + append; n=2 is
            # t-2's own append -- the one made to fail. By the time a
            # candidate's *own* append runs, that same candidate's own delete
            # and purge have already completed (append is the last step in
            # the loop body), so t-2's delete and purge ran too; only its
            # journal entry was ever going to be missing.
            _fail_nth_journal_append(store, exc, n=2)

            with pytest.raises(RuntimeError) as excinfo:
                await store.run_hygiene(current_cycle=1000, now=_NOW)

            assert excinfo.value is exc
            assert store._db.in_transaction is False
            # Neither candidate's delete, embedding cascade or vector purge
            # survived: the first candidate's had already succeeded, and the
            # second's had *also* already run (only its own append was left)
            # when the second's append failed and unwound the whole unit.
            assert await store._get_thought_row("t-1") is not None
            assert await _embedding_rowid(store, "t-1") == t1_embedding_rowid
            assert t1_embedding_rowid in await _vec_rowids(store)
            assert await store._get_thought_row("t-2") is not None
            assert await _embedding_rowid(store, "t-2") == t2_embedding_rowid
            assert t2_embedding_rowid in await _vec_rowids(store)
        finally:
            await store._db.close()
