"""Create, delete and action-recompute writes recover with their journal entry.

A sibling to ``test_journal_append_atomicity.py``, which pins the same property
for ``update_thought``, ``restore_thought``, ``update_edge`` and
``update_action``: the write(s) a journaled operation makes and the journal
append describing them must be one failure-atomic unit, so a failed or
cancelled append unwinds the write too instead of leaving it durable on a
later, unrelated commit with no journal entry to show for it.

This file covers the other journaled operations:
``create_thought`` (its plain-insert branch and both dedup entry points),
``delete_thought``, ``create_edge``, ``delete_edge``, ``create_action`` /
``update_action`` (including the action-outcome recompute), and
``cleanup_expired``.

Three transaction contexts are exercised for every row:

* **standalone** -- no transaction open when the call starts.
* **window** -- the call runs inside a caller-owned ``suspend_auto_commit()``
  window that catches the failure and exits cleanly.
* **raw transaction** -- the caller already holds an open transaction via a
  direct ``BEGIN`` on the connection (not mediated by this store), catches
  the failure, and commits afterwards.

Each is tried with an ordinary exception and with ``asyncio.CancelledError``,
since the two can take different paths through an ``except BaseException``
unwind.
"""

from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import (
    ActionRecord,
    ActionStatus,
    ActionType,
    CleanupResult,
    EdgeRecord,
    EdgeType,
    KnowledgeSource,
    LifecycleStatus,
    Priority,
    SqliteEngravaCore,
    ThoughtRecord,
    ThoughtType,
    VerificationStatus,
)
from tests.test_sqlite_vec import _embedding_rowid, _vec_rowids, sqlite_vec_required

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

# ---------------------------------------------------------------------------
# Domain factories
# ---------------------------------------------------------------------------


def _thought(thought_id: str = "t-1", *, essence: str = "essence") -> ThoughtRecord:
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
    )


def _thought_with_content(thought_id: str, *, content: str) -> ThoughtRecord:
    """A thought with an explicit, caller-chosen ``content`` (for dedup hits)."""
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.OBSERVATION,
        essence="essence",
        content=content,
        priority=Priority.P3,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=0,
        updated_cycle=0,
        source="test",
        confidence=0.75,
    )


def _expired_thought(thought_id: str = "t-1") -> ThoughtRecord:
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.OBSERVATION,
        essence="essence",
        content=f"content of {thought_id}",
        priority=Priority.P3,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=0,
        updated_cycle=0,
        source="test",
        confidence=0.75,
        expires_at="2020-01-01T00:00:00+00:00",
    )


def _edge(edge_id: str = "e-1", *, weight: float = 0.5) -> EdgeRecord:
    return EdgeRecord(
        edge_id=edge_id,
        from_thought_id="t-1",
        to_thought_id="t-2",
        edge_type=EdgeType.ASSOCIATED,
        weight=weight,
        created_cycle=0,
        source=KnowledgeSource.EXPERIENCE,
        decay_multiplier=1.0,
    )


def _action(action_id: str = "a-1", *, status: ActionStatus = ActionStatus.PLANNED) -> ActionRecord:
    return ActionRecord(
        action_id=action_id,
        source_thought_id="t-1",
        action_type=ActionType.CLI_OUTPUT,
        intent="do the thing",
        status=status,
        verification_status=VerificationStatus.PENDING,
    )


_FUTURE_NOW = "2030-01-01T00:00:00+00:00"

# ---------------------------------------------------------------------------
# Store + connection helpers
# ---------------------------------------------------------------------------


async def _open_store(db_path: Path, *, ttl_strategy: str = "archive") -> SqliteEngravaCore:
    """A fresh, journal-enabled store on its own real, on-disk connection."""
    conn = await aiosqlite.connect(str(db_path))
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode = WAL")
    await conn.execute("PRAGMA foreign_keys = ON")
    store = SqliteEngravaCore(conn, journal_enabled=True, ttl_strategy=ttl_strategy)
    await store.ensure_schema()
    return store


async def _raw_row(db_path: Path, table: str, key: str, value: str) -> aiosqlite.Row | None:
    """Read a row straight from storage, on a brand-new independent connection."""
    conn = await aiosqlite.connect(str(db_path))
    conn.row_factory = aiosqlite.Row
    try:
        cursor = await conn.execute(f"SELECT * FROM {table} WHERE {key} = ?", (value,))  # noqa: S608 -- test literals
        return await cursor.fetchone()
    finally:
        await conn.close()


async def _journal_entry_count(store: SqliteEngravaCore) -> int:
    """Count journal rows on the store's own connection, mid-transaction included.

    Used to prove a call that genuinely writes nothing left the journal
    untouched too -- a call that appends a spurious entry while reporting "no
    write" would still be visible here even before any commit, since this
    reads through the same connection the call itself used.
    """
    cursor = await store._db.execute("SELECT COUNT(*) FROM journal_entry")
    row = await cursor.fetchone()
    assert row is not None
    return int(row[0])


async def _write_raw_marker(store: SqliteEngravaCore, thought_id: str) -> None:
    """Insert a marker thought directly, bypassing the store's own commit.

    Used as the caller's own pending write ("X") inside a raw, caller-held
    transaction: calling one of the store's public write methods there would
    invoke its own ``_maybe_commit`` and end the caller's transaction early,
    which is exactly the scenario this helper avoids.
    """
    thought = _thought(thought_id)
    await store._db.execute(store._CORE_INSERT_SQL, store._thought_to_core_params(thought))


def _fail_nth_journal_append(store: SqliteEngravaCore, exc: BaseException, *, n: int = 1) -> None:
    """Make the journal's own chain-tail read raise/cancel on its *n*-th call.

    Generalises the single-shot fault used elsewhere in this suite: counting
    from the moment this is installed (not from the store's lifetime), the
    first ``n - 1`` calls succeed normally and the ``n``-th raises ``exc``.
    ``n=1`` (the default) reproduces the single-shot case. Needed wherever a
    guarded call makes more than one journal append internally and a specific
    one -- not necessarily the first -- is the one under test (e.g.
    ``update_action``'s own append succeeding while its outcome-recompute's
    append is the one that fails).
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


async def _install_thought_delete_ignore_trigger(store: SqliteEngravaCore) -> None:
    await store._db.execute(
        "CREATE TRIGGER thought_delete_ignore BEFORE DELETE ON thought "
        "BEGIN SELECT RAISE(IGNORE); END"
    )


# ---------------------------------------------------------------------------
# The row matrix: one Case per operation variant, reused across all three
# contexts and both injection kinds.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    id: str
    seed: Callable[[SqliteEngravaCore], Awaitable[None]]
    act: Callable[[SqliteEngravaCore], Awaitable[object]]
    assert_absent: Callable[[SqliteEngravaCore], Awaitable[None]]
    ttl_strategy: str = "archive"
    fail_nth: int = 1
    open_store: Callable[..., Awaitable[SqliteEngravaCore]] = _open_store


async def _seed_noop(store: SqliteEngravaCore) -> None:
    return


# -- Row 1: create_thought, plain-insert branch -----------------------------


async def _act_row1(store: SqliteEngravaCore) -> object:
    return await store.create_thought(_thought("row-t1"))


async def _assert_absent_row1(store: SqliteEngravaCore) -> None:
    assert await store._get_thought_row("row-t1") is None


# -- Row 2: create_thought, dedup entry point, miss -------------------------


async def _act_row2(store: SqliteEngravaCore) -> object:
    return await store.create_thought(_thought("row-t1", essence="row2"), deduplicate=True)


async def _assert_absent_row2(store: SqliteEngravaCore) -> None:
    assert await store._get_thought_row("row-t1") is None


# -- Row 3: create_thought, dedup entry point, hit (_increment_confirmation) --

_SHARED_CONTENT = "shared content for a dedup hit"


async def _seed_row3(store: SqliteEngravaCore) -> None:
    await store.create_thought(_thought_with_content("row-t1", content=_SHARED_CONTENT))


async def _act_row3(store: SqliteEngravaCore) -> object:
    return await store.create_thought(
        _thought_with_content("row-t2", content=_SHARED_CONTENT), deduplicate=True
    )


async def _assert_absent_row3(store: SqliteEngravaCore) -> None:
    row = await store._get_thought_row("row-t1")
    assert row is not None
    assert row["confirmation_count"] == 0
    assert await store._get_thought_row("row-t2") is None


# -- Row 4: delete_thought ----------------------------------------------------


async def _seed_row4(store: SqliteEngravaCore) -> None:
    await store.create_thought(_thought("row-t1"))


async def _act_row4(store: SqliteEngravaCore) -> object:
    return await store.delete_thought("row-t1")


async def _assert_absent_row4(store: SqliteEngravaCore) -> None:
    assert await store._get_thought_row("row-t1") is not None


# -- Row 5: create_edge -------------------------------------------------------


async def _seed_row5(store: SqliteEngravaCore) -> None:
    await store.create_thought(_thought("t-1"))
    await store.create_thought(_thought("t-2"))


async def _act_row5(store: SqliteEngravaCore) -> object:
    return await store.create_edge(_edge("row-e1"))


async def _assert_absent_row5(store: SqliteEngravaCore) -> None:
    assert await store._get_edge_row("row-e1") is None


# -- Row 6: delete_edge --------------------------------------------------------


async def _seed_row6(store: SqliteEngravaCore) -> None:
    await store.create_thought(_thought("t-1"))
    await store.create_thought(_thought("t-2"))
    await store.create_edge(_edge("row-e1"))


async def _act_row6(store: SqliteEngravaCore) -> object:
    return await store.delete_edge("row-e1")


async def _assert_absent_row6(store: SqliteEngravaCore) -> None:
    assert await store._get_edge_row("row-e1") is not None


# -- Row 7: update_action -> _recompute_action_outcome (the recompute's own
#    append is the one that fails; the action's own append -- made during
#    seed's PLANNED -> EXECUTING move, and again by the guarded call itself
#    before the recompute runs -- succeeds).


async def _seed_row7(store: SqliteEngravaCore) -> None:
    await store.create_thought(_thought("t-1"))
    await store.create_action(_action("row-a1"))
    await store.update_action("row-a1", status=ActionStatus.EXECUTING)


async def _act_row7(store: SqliteEngravaCore) -> object:
    return await store.update_action("row-a1", status=ActionStatus.CONFIRMED)


async def _assert_absent_row7(store: SqliteEngravaCore) -> None:
    row = await store._get_action_row("row-a1")
    assert row is not None
    assert row["status"] == "EXECUTING"
    thought_row = await store._get_thought_row("t-1")
    assert thought_row is not None
    assert thought_row["action_outcome_score"] is None


# -- Row 8: create_action -> _recompute_action_outcome (a terminal create) ---


async def _seed_row8(store: SqliteEngravaCore) -> None:
    await store.create_thought(_thought("t-1"))


async def _act_row8(store: SqliteEngravaCore) -> object:
    return await store.create_action(_action("row-a1", status=ActionStatus.CONFIRMED))


async def _assert_absent_row8(store: SqliteEngravaCore) -> None:
    assert await store._get_action_row("row-a1") is None
    thought_row = await store._get_thought_row("t-1")
    assert thought_row is not None
    assert thought_row["action_outcome_score"] is None


# -- Row 9: cleanup_expired, archive strategy --------------------------------


async def _seed_row9(store: SqliteEngravaCore) -> None:
    await store.create_thought(_expired_thought("row-t1"))


async def _act_row9(store: SqliteEngravaCore) -> object:
    return await store.cleanup_expired(now=_FUTURE_NOW)


async def _assert_absent_row9(store: SqliteEngravaCore) -> None:
    row = await store._get_thought_row("row-t1")
    assert row is not None
    assert row["lifecycle_status"] == "ACTIVE"
    assert row["expires_at"] is not None


# -- Row 10: cleanup_expired, delete strategy --------------------------------


async def _seed_row10(store: SqliteEngravaCore) -> None:
    await store.create_thought(_expired_thought("row-t1"))


async def _act_row10(store: SqliteEngravaCore) -> object:
    return await store.cleanup_expired(now=_FUTURE_NOW)


async def _assert_absent_row10(store: SqliteEngravaCore) -> None:
    assert await store._get_thought_row("row-t1") is not None


CASES: list[Case] = [
    Case("create_thought_plain_insert", _seed_noop, _act_row1, _assert_absent_row1),
    Case("create_thought_dedup_miss", _seed_noop, _act_row2, _assert_absent_row2),
    Case(
        "create_thought_dedup_hit_increment_confirmation",
        _seed_row3,
        _act_row3,
        _assert_absent_row3,
    ),
    Case("delete_thought", _seed_row4, _act_row4, _assert_absent_row4),
    Case("create_edge", _seed_row5, _act_row5, _assert_absent_row5),
    Case("delete_edge", _seed_row6, _act_row6, _assert_absent_row6),
    Case(
        "update_action_recompute",
        _seed_row7,
        _act_row7,
        _assert_absent_row7,
        fail_nth=2,
    ),
    Case("create_action_recompute", _seed_row8, _act_row8, _assert_absent_row8),
    Case(
        "cleanup_expired_archive",
        _seed_row9,
        _act_row9,
        _assert_absent_row9,
        ttl_strategy="archive",
    ),
    Case(
        "cleanup_expired_delete",
        _seed_row10,
        _act_row10,
        _assert_absent_row10,
        ttl_strategy="delete",
    ),
]

_CASE_IDS = [case.id for case in CASES]


def _new_runtime_error() -> BaseException:
    return RuntimeError("forced journal append failure")


def _new_cancelled_error() -> BaseException:
    return asyncio.CancelledError()


_INJECTIONS: list[Callable[[], BaseException]] = [_new_runtime_error, _new_cancelled_error]
_INJECTION_IDS = ["exception", "cancellation"]


# ---------------------------------------------------------------------------
# Context runners
# ---------------------------------------------------------------------------


async def _run_standalone(case: Case, exc: BaseException, tmp_path: Path) -> None:
    store = await case.open_store(
        tmp_path / f"{case.id}-standalone.db", ttl_strategy=case.ttl_strategy
    )
    try:
        await case.seed(store)
        _fail_nth_journal_append(store, exc, n=case.fail_nth)

        with pytest.raises(type(exc)) as excinfo:
            await case.act(store)

        assert excinfo.value is exc
        assert store._db.in_transaction is False
        await case.assert_absent(store)
    finally:
        await store._db.close()


async def _run_window(case: Case, exc: BaseException, tmp_path: Path) -> None:
    store = await case.open_store(tmp_path / f"{case.id}-window.db", ttl_strategy=case.ttl_strategy)
    try:
        await case.seed(store)

        async with store.suspend_auto_commit():
            await store.create_thought(_thought("x-marker"))
            _fail_nth_journal_append(store, exc, n=case.fail_nth)

            with pytest.raises(type(exc)) as excinfo:
                await case.act(store)

            assert excinfo.value is exc
            await case.assert_absent(store)

        # The window exits cleanly here -- its own outermost commit runs on
        # the way out. Re-checking now, not just inside the window, is what
        # would catch a write that survived the unit's own unwind but was
        # only actually flushed by this commit (a false journal record is
        # exactly that shape).
        assert store._db.in_transaction is False
        row = await store._get_thought_row("x-marker")
        assert row is not None
        await case.assert_absent(store)
    finally:
        await store._db.close()


async def _run_raw_transaction(case: Case, exc: BaseException, tmp_path: Path) -> None:
    db_path = tmp_path / f"{case.id}-raw.db"
    store = await case.open_store(db_path, ttl_strategy=case.ttl_strategy)
    try:
        await case.seed(store)

        await store._db.execute("BEGIN")
        await _write_raw_marker(store, "x-marker")
        _fail_nth_journal_append(store, exc, n=case.fail_nth)

        with pytest.raises(type(exc)) as excinfo:
            await case.act(store)

        assert excinfo.value is exc
        assert store._db.in_transaction is True
        row = await store._get_thought_row("x-marker")
        assert row is not None
        await case.assert_absent(store)

        await store._db.commit()

        # Re-check on the same connection now that the caller's own commit
        # has run -- this is where a write that only escaped the unit's own
        # unwind (rather than never having landed) would become durable.
        await case.assert_absent(store)
    finally:
        await store._db.close()

    durable_marker = await _raw_row(db_path, "thought", "thought_id", "x-marker")
    assert durable_marker is not None


@pytest.mark.parametrize("case", CASES, ids=_CASE_IDS)
@pytest.mark.parametrize("exc_factory", _INJECTIONS, ids=_INJECTION_IDS)
class TestFixRowMatrix:
    """Each row, in each of three transaction contexts, with each of two injections."""

    async def test_standalone(
        self,
        case: Case,
        exc_factory: Callable[[], BaseException],
        tmp_path: Path,
    ) -> None:
        await _run_standalone(case, exc_factory(), tmp_path)

    async def test_suspend_auto_commit_window(
        self,
        case: Case,
        exc_factory: Callable[[], BaseException],
        tmp_path: Path,
    ) -> None:
        await _run_window(case, exc_factory(), tmp_path)

    async def test_caller_held_raw_transaction(
        self,
        case: Case,
        exc_factory: Callable[[], BaseException],
        tmp_path: Path,
    ) -> None:
        await _run_raw_transaction(case, exc_factory(), tmp_path)


# ---------------------------------------------------------------------------
# The clean no-write path: a call that genuinely writes nothing.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NoWriteCase:
    id: str
    seed: Callable[[SqliteEngravaCore], Awaitable[None]]
    act: Callable[[SqliteEngravaCore], Awaitable[object]]
    ttl_strategy: str = "archive"
    expected_result: object = field(default=None)


async def _act_delete_thought_missing(store: SqliteEngravaCore) -> object:
    return await store.delete_thought("missing")


async def _seed_delete_thought_veto(store: SqliteEngravaCore) -> None:
    await store.create_thought(_thought("t-1"))
    await _install_thought_delete_ignore_trigger(store)


async def _act_delete_thought_veto(store: SqliteEngravaCore) -> object:
    return await store.delete_thought("t-1")


async def _act_delete_edge_missing(store: SqliteEngravaCore) -> object:
    return await store.delete_edge("missing")


async def _act_cleanup_expired_no_candidates(store: SqliteEngravaCore) -> object:
    return await store.cleanup_expired(now=_FUTURE_NOW)


async def _seed_cleanup_expired_all_vetoed(store: SqliteEngravaCore) -> None:
    await store.create_thought(_expired_thought("t-1"))
    await _install_thought_delete_ignore_trigger(store)


NO_WRITE_CASES: list[NoWriteCase] = [
    NoWriteCase(
        "delete_thought_missing_id",
        _seed_noop,
        _act_delete_thought_missing,
        expected_result=False,
    ),
    NoWriteCase(
        "delete_thought_veto",
        _seed_delete_thought_veto,
        _act_delete_thought_veto,
        expected_result=False,
    ),
    NoWriteCase(
        "delete_edge_missing_id",
        _seed_noop,
        _act_delete_edge_missing,
        expected_result=False,
    ),
    NoWriteCase(
        "cleanup_expired_no_candidates_archive",
        _seed_noop,
        _act_cleanup_expired_no_candidates,
        ttl_strategy="archive",
        expected_result=CleanupResult(
            expired_count=0, strategy_applied="archive", timestamp=_FUTURE_NOW
        ),
    ),
    NoWriteCase(
        "cleanup_expired_no_candidates_delete",
        _seed_noop,
        _act_cleanup_expired_no_candidates,
        ttl_strategy="delete",
        expected_result=CleanupResult(
            expired_count=0, strategy_applied="delete", timestamp=_FUTURE_NOW
        ),
    ),
    NoWriteCase(
        "cleanup_expired_all_vetoed_delete",
        _seed_cleanup_expired_all_vetoed,
        _act_cleanup_expired_no_candidates,
        ttl_strategy="delete",
        # expired_count counts candidates the expiry predicate matched, not
        # rows actually written: the one candidate here is found and then
        # vetoed, so the count is 1 -- not the 0 a "nothing happened" read
        # would suggest -- while still writing (and journaling) nothing.
        expected_result=CleanupResult(
            expired_count=1, strategy_applied="delete", timestamp=_FUTURE_NOW
        ),
    ),
]

_NO_WRITE_IDS = [case.id for case in NO_WRITE_CASES]


@pytest.mark.parametrize("case", NO_WRITE_CASES, ids=_NO_WRITE_IDS)
class TestCleanNoWritePath:
    async def test_standalone(self, case: NoWriteCase, tmp_path: Path) -> None:
        store = await _open_store(
            tmp_path / f"{case.id}-nowrite-standalone.db", ttl_strategy=case.ttl_strategy
        )
        try:
            await case.seed(store)
            journal_count_before = await _journal_entry_count(store)

            result = await case.act(store)

            if case.expected_result is not None:
                assert result == case.expected_result
            assert store._db.in_transaction is False
            assert await _journal_entry_count(store) == journal_count_before
            if case.id == "delete_thought_veto":
                assert await store._get_thought_row("t-1") is not None
        finally:
            await store._db.close()

    async def test_caller_held_raw_transaction(self, case: NoWriteCase, tmp_path: Path) -> None:
        db_path = tmp_path / f"{case.id}-nowrite-raw.db"
        store = await _open_store(db_path, ttl_strategy=case.ttl_strategy)
        try:
            await case.seed(store)
            journal_count_before = await _journal_entry_count(store)

            await store._db.execute("BEGIN")
            await _write_raw_marker(store, "x-marker")

            result = await case.act(store)

            if case.expected_result is not None:
                assert result == case.expected_result
            assert store._db.in_transaction is True
            row = await store._get_thought_row("x-marker")
            assert row is not None
            assert await _journal_entry_count(store) == journal_count_before
            if case.id == "delete_thought_veto":
                assert await store._get_thought_row("t-1") is not None

            await store._db.commit()

            # A false journal record would become durable right here -- the
            # commit above is exactly what a call misreporting "no write"
            # would otherwise smuggle through. Re-check on the same
            # connection now that it is.
            assert await _journal_entry_count(store) == journal_count_before
        finally:
            await store._db.close()

        durable_marker = await _raw_row(db_path, "thought", "thought_id", "x-marker")
        assert durable_marker is not None


# ---------------------------------------------------------------------------
# cleanup_expired with two candidates: the unit is the whole call, not a
# per-row savepoint -- a mid-batch append failure must unwind every candidate
# processed so far, not just the one it failed on.
# ---------------------------------------------------------------------------


async def _two_candidates_seed(store: SqliteEngravaCore) -> None:
    await store.create_thought(_expired_thought("t-1"))
    await store.create_thought(_expired_thought("t-2"))


async def _assert_neither_candidate_archived(store: SqliteEngravaCore) -> None:
    row1 = await store._get_thought_row("t-1")
    row2 = await store._get_thought_row("t-2")
    assert row1 is not None
    assert row1["lifecycle_status"] == "ACTIVE"
    assert row2 is not None
    assert row2["lifecycle_status"] == "ACTIVE"


async def _assert_neither_candidate_deleted(store: SqliteEngravaCore) -> None:
    assert await store._get_thought_row("t-1") is not None
    assert await store._get_thought_row("t-2") is not None


@dataclass(frozen=True)
class TwoCandidateCase:
    id: str
    ttl_strategy: str
    assert_neither_survived: Callable[[SqliteEngravaCore], Awaitable[None]]


TWO_CANDIDATE_CASES: list[TwoCandidateCase] = [
    TwoCandidateCase("archive", "archive", _assert_neither_candidate_archived),
    TwoCandidateCase("delete", "delete", _assert_neither_candidate_deleted),
]
_TWO_CANDIDATE_IDS = [case.id for case in TWO_CANDIDATE_CASES]


@pytest.mark.parametrize("case", TWO_CANDIDATE_CASES, ids=_TWO_CANDIDATE_IDS)
@pytest.mark.parametrize("exc_factory", _INJECTIONS, ids=_INJECTION_IDS)
class TestCleanupExpiredTwoCandidatesUnwindTogether:
    """The unit is the whole batch, not a per-row savepoint, in every context.

    A mid-batch append failure -- on the *second* candidate, after the first
    fully succeeded -- must unwind every candidate processed so far, not just
    the one it failed on. A per-row-savepoint implementation would let the
    first candidate's own row write and journal entry survive, since that
    row's own unit would already have released before the second row's
    append ever runs.
    """

    async def test_standalone(
        self,
        case: TwoCandidateCase,
        exc_factory: Callable[[], BaseException],
        tmp_path: Path,
    ) -> None:
        store = await _open_store(
            tmp_path / f"cleanup-two-candidates-{case.id}-standalone.db",
            ttl_strategy=case.ttl_strategy,
        )
        try:
            await _two_candidates_seed(store)
            exc = exc_factory()
            _fail_nth_journal_append(store, exc, n=2)

            with pytest.raises(type(exc)) as excinfo:
                await store.cleanup_expired(now=_FUTURE_NOW)

            assert excinfo.value is exc
            assert store._db.in_transaction is False
            await case.assert_neither_survived(store)
        finally:
            await store._db.close()

    async def test_suspend_auto_commit_window(
        self,
        case: TwoCandidateCase,
        exc_factory: Callable[[], BaseException],
        tmp_path: Path,
    ) -> None:
        store = await _open_store(
            tmp_path / f"cleanup-two-candidates-{case.id}-window.db",
            ttl_strategy=case.ttl_strategy,
        )
        try:
            await _two_candidates_seed(store)

            async with store.suspend_auto_commit():
                await store.create_thought(_thought("x-marker"))
                exc = exc_factory()
                _fail_nth_journal_append(store, exc, n=2)

                with pytest.raises(type(exc)) as excinfo:
                    await store.cleanup_expired(now=_FUTURE_NOW)

                assert excinfo.value is exc
                await case.assert_neither_survived(store)

            # The window's own outermost commit runs here, on the way out --
            # re-checking now (not just inside the window) is what would
            # catch a candidate that survived the batch's own unwind but was
            # only actually flushed by this commit.
            assert store._db.in_transaction is False
            assert await store._get_thought_row("x-marker") is not None
            await case.assert_neither_survived(store)
        finally:
            await store._db.close()

    async def test_caller_held_raw_transaction(
        self,
        case: TwoCandidateCase,
        exc_factory: Callable[[], BaseException],
        tmp_path: Path,
    ) -> None:
        db_path = tmp_path / f"cleanup-two-candidates-{case.id}-raw.db"
        store = await _open_store(db_path, ttl_strategy=case.ttl_strategy)
        try:
            await _two_candidates_seed(store)

            await store._db.execute("BEGIN")
            await _write_raw_marker(store, "x-marker")
            exc = exc_factory()
            _fail_nth_journal_append(store, exc, n=2)

            with pytest.raises(type(exc)) as excinfo:
                await store.cleanup_expired(now=_FUTURE_NOW)

            assert excinfo.value is exc
            assert store._db.in_transaction is True
            assert await store._get_thought_row("x-marker") is not None
            await case.assert_neither_survived(store)

            await store._db.commit()

            # Re-check on the same connection now that the caller's own
            # commit has run -- this is where a candidate that only escaped
            # the batch's own unwind (rather than never having landed) would
            # become durable.
            await case.assert_neither_survived(store)
        finally:
            await store._db.close()

        durable_marker = await _raw_row(db_path, "thought", "thought_id", "x-marker")
        assert durable_marker is not None


# ---------------------------------------------------------------------------
# The vector purge actually writes, then the append fails: rows 4 and 10's
# main-matrix cases create no embedding, so `_purge_orphan_vector` is a no-op
# there (the numpy backend does nothing either way -- see that method's
# docstring). Only a real vector backend gives the purge something to unwind.
# ---------------------------------------------------------------------------


async def _seed_delete_thought_with_vector(store: SqliteEngravaCore) -> None:
    await store.create_thought(_thought("row-t1"))
    await store.store_embedding(
        thought_id="row-t1", vector=[1.0, 0.0, 0.0], model_name="test-fixture-model"
    )


async def _assert_delete_thought_vector_survives(store: SqliteEngravaCore) -> None:
    await _assert_absent_row4(store)
    embedding_rowid = await _embedding_rowid(store, "row-t1")
    assert embedding_rowid is not None
    assert embedding_rowid in await _vec_rowids(store)


async def _seed_cleanup_expired_delete_with_vector(store: SqliteEngravaCore) -> None:
    await store.create_thought(_expired_thought("row-t1"))
    await store.store_embedding(
        thought_id="row-t1", vector=[1.0, 0.0, 0.0], model_name="test-fixture-model"
    )


async def _assert_cleanup_expired_vector_survives(store: SqliteEngravaCore) -> None:
    await _assert_absent_row10(store)
    embedding_rowid = await _embedding_rowid(store, "row-t1")
    assert embedding_rowid is not None
    assert embedding_rowid in await _vec_rowids(store)


async def _open_vec_store(db_path: Path, *, ttl_strategy: str = "archive") -> SqliteEngravaCore:
    """A journal-enabled store with a real, loaded sqlite-vec backend.

    The numpy backend's ``_purge_orphan_vector`` is a documented no-op, so it
    cannot exercise "the purge writes, then the unit's append fails" -- only a
    real vector backend gives that write something to unwind.
    """
    conn = await aiosqlite.connect(str(db_path))
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode = WAL")
    await conn.execute("PRAGMA foreign_keys = ON")
    store = SqliteEngravaCore(conn, journal_enabled=True, ttl_strategy=ttl_strategy)
    await store.ensure_schema()
    await store._configure_vector_backend(backend_name="sqlite-vec", embedding_dimension=3)
    return store


VECTOR_PURGE_CASES: list[Case] = [
    Case(
        "delete_thought_with_vector_purge",
        _seed_delete_thought_with_vector,
        _act_row4,
        _assert_delete_thought_vector_survives,
        open_store=_open_vec_store,
    ),
    Case(
        "cleanup_expired_delete_with_vector_purge",
        _seed_cleanup_expired_delete_with_vector,
        _act_row10,
        _assert_cleanup_expired_vector_survives,
        ttl_strategy="delete",
        open_store=_open_vec_store,
    ),
]
_VECTOR_PURGE_IDS = [case.id for case in VECTOR_PURGE_CASES]


@sqlite_vec_required
@pytest.mark.parametrize("case", VECTOR_PURGE_CASES, ids=_VECTOR_PURGE_IDS)
class TestVectorPurgeIsPartOfTheUnit:
    """A purge that actually wrote a vector unwinds along with the rest of the unit."""

    async def test_standalone(self, case: Case, tmp_path: Path) -> None:
        await _run_standalone(case, RuntimeError("forced journal append failure"), tmp_path)

    async def test_suspend_auto_commit_window(self, case: Case, tmp_path: Path) -> None:
        await _run_window(case, RuntimeError("forced journal append failure"), tmp_path)

    async def test_caller_held_raw_transaction(self, case: Case, tmp_path: Path) -> None:
        await _run_raw_transaction(case, RuntimeError("forced journal append failure"), tmp_path)


# ---------------------------------------------------------------------------
# Quarantine precedence: when a unit's own unwind fails and quarantines the
# connection, the frame that unit nests inside must re-raise the original
# error or cancellation unchanged -- never replace it with
# ConnectionQuarantinedError by touching the now-quarantined `self._db`.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("original_exc_factory", _INJECTIONS, ids=_INJECTION_IDS)
class TestQuarantinePrecedence:
    async def test_write_readback_savepoint_frame_around_delete_thought(
        self,
        monkeypatch: pytest.MonkeyPatch,
        original_exc_factory: Callable[[], BaseException],
    ) -> None:
        """The error must originate *inside* ``_delete_thought_atomic`` itself.

        Its own savepoint releases before a journal-append failure could ever
        reach it, so only a failure inside its protected span (here: the
        parent ``DELETE`` itself) exercises the outer
        ``_write_readback_savepoint`` frame this test targets.
        """
        db = await aiosqlite.connect(":memory:")
        db.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(
            db, embedding_provider=None, auto_embed=False, journal_enabled=True
        )
        await store.ensure_schema()
        await store.create_thought(_thought("t-1"))

        original_exc = original_exc_factory()
        unwind_exc = sqlite3.OperationalError("disk I/O error")
        real_execute = db.execute

        async def _wrapper(sql: str, *args: object, **kwargs: object) -> object:
            if sql == "DELETE FROM thought WHERE thought_id = ?":
                raise original_exc
            if sql == "ROLLBACK TO delete_thought_atomic":
                raise unwind_exc
            return await real_execute(sql, *args, **kwargs)

        monkeypatch.setattr(db, "execute", _wrapper)

        with pytest.raises(type(original_exc)) as excinfo:
            await store.delete_thought("t-1")

        assert excinfo.value is original_exc
        assert store._connection_quarantined is True

        if store._quarantine_close_task is not None:
            await store._quarantine_close_task

    async def test_serialize_dedup_probe_frame_around_a_dedup_write(
        self,
        monkeypatch: pytest.MonkeyPatch,
        original_exc_factory: Callable[[], BaseException],
    ) -> None:
        """A dedup-window write's own unit unwind fails, quarantining mid-window."""
        store = await _open_store_in_memory()

        original_exc = original_exc_factory()
        unwind_exc = sqlite3.OperationalError("disk I/O error")
        real_execute = store._db.execute

        async def _wrapper(sql: str, *args: object, **kwargs: object) -> object:
            if sql == "ROLLBACK TO insert_new_thought_row":
                raise unwind_exc
            return await real_execute(sql, *args, **kwargs)

        monkeypatch.setattr(store._db, "execute", _wrapper)
        _fail_nth_journal_append(store, original_exc, n=1)

        with pytest.raises(type(original_exc)) as excinfo:
            await store.create_thought(_thought("t-1"), deduplicate=True)

        assert excinfo.value is original_exc
        assert store._connection_quarantined is True

        if store._quarantine_close_task is not None:
            await store._quarantine_close_task

    async def test_suspend_auto_commit_frame_around_an_ordinary_unit(
        self,
        monkeypatch: pytest.MonkeyPatch,
        original_exc_factory: Callable[[], BaseException],
    ) -> None:
        """An update_thought unit's unwind fails inside a caller's window.

        ``update_thought`` runs inside ``_write_readback_savepoint`` and
        ``suspend_auto_commit`` is an outer frame around it. When the journal
        append fails and the savepoint's own unwind fails too, the original
        exception still propagates and the connection is quarantined.
        """
        db = await aiosqlite.connect(":memory:")
        db.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(
            db, embedding_provider=None, auto_embed=False, journal_enabled=True
        )
        await store.ensure_schema()
        await store.create_thought(_thought("t-1"))

        original_exc = original_exc_factory()
        unwind_exc = sqlite3.OperationalError("disk I/O error")
        real_execute = db.execute

        async def _wrapper(sql: str, *args: object, **kwargs: object) -> object:
            if sql == "ROLLBACK TO update_thought_readback":
                raise unwind_exc
            return await real_execute(sql, *args, **kwargs)

        monkeypatch.setattr(db, "execute", _wrapper)
        _fail_nth_journal_append(store, original_exc, n=1)

        with pytest.raises(type(original_exc)) as excinfo:
            async with store.suspend_auto_commit():
                await store.update_thought("t-1", essence="new essence")

        assert excinfo.value is original_exc
        assert store._connection_quarantined is True

        if store._quarantine_close_task is not None:
            await store._quarantine_close_task


async def _open_store_in_memory() -> SqliteEngravaCore:
    db = await aiosqlite.connect(":memory:")
    db.row_factory = aiosqlite.Row
    store = SqliteEngravaCore(db, embedding_provider=None, auto_embed=False, journal_enabled=True)
    await store.ensure_schema()
    return store
