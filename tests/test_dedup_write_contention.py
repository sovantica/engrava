"""Tests for the busy/contention boundary of the dedup probe-and-insert window.

``create_thought(deduplicate=True)``, ``get_or_create`` and ``upsert_by_hash``
open their probe-and-insert window with ``BEGIN IMMEDIATE`` (see
``SqliteEngravaCore._begin_dedup_write_lock``) so a second connection racing
the same window is turned away at transaction *start* instead of mid-write.
This module pins the busy path specifically: real lock contention from a
second real connection must surface as the typed ``WriteContentionError``,
never a raw ``sqlite3.OperationalError`` — and a transient busy that clears
before attempts are exhausted must succeed rather than fail.
"""

from __future__ import annotations

import asyncio
import sqlite3
from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import (
    CoreThoughtRecord,
    EngravaError,
    KnowledgeSource,
    LifecycleStatus,
    Priority,
    SqliteEngravaCore,
    ThoughtRecord,
    ThoughtType,
    ThoughtVisibility,
    WriteContentionError,
)
from engrava.infrastructure.sqlite.engrava_core import _is_busy_error

if TYPE_CHECKING:
    from pathlib import Path


def _thought(thought_id: str, *, content: str) -> CoreThoughtRecord:
    """Build a minimal, realistic ``CoreThoughtRecord`` for contention tests."""
    return CoreThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.OBSERVATION,
        essence="Contention probe",
        content=content,
        priority=Priority.P2,
        lifecycle_status=LifecycleStatus.CREATED,
        created_cycle=0,
        updated_cycle=0,
        source="test-suite",
        confidence=0.9,
        source_type=KnowledgeSource.EXPERIENCE,
        visibility=ThoughtVisibility.SELECTIVE,
    )


@pytest.fixture
async def db_path(tmp_path: Path) -> str:
    """A real on-disk database file, schema-bootstrapped, WAL-enabled.

    A file (not ``:memory:``) is required: the busy path this module tests is
    lock contention between two *separate* connections, which an in-memory
    database cannot host without shared-cache mode, itself a different
    concurrency regime than the file-backed one production stores use.
    """
    path = str(tmp_path / "contention.db")
    conn = await aiosqlite.connect(path)
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA foreign_keys=ON")
    bootstrap = SqliteEngravaCore(conn)
    await bootstrap.ensure_schema()
    await conn.close()
    return path


async def _open_store(
    db_path: str,
    *,
    busy_timeout_ms: int,
) -> tuple[aiosqlite.Connection, SqliteEngravaCore]:
    """Open a fresh connection + store against *db_path* with a given busy timeout.

    The manual ``SqliteEngravaCore(conn)`` constructor does not touch pragmas
    (see ``docs/concurrency.md``), so the timeout is set here explicitly —
    deliberately small in the contention tests, so a real busy failure surfaces
    in milliseconds instead of the production 5s default.
    """
    conn = await aiosqlite.connect(db_path)
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA foreign_keys=ON")
    await conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
    store = SqliteEngravaCore(conn)
    await store._probe_fts()
    return conn, store


# ---------------------------------------------------------------------------
# Real cross-connection contention -> typed exception
# ---------------------------------------------------------------------------


async def test_sustained_contention_raises_write_contention_error(db_path: str) -> None:
    """A write lock held for longer than every retry attempt raises the typed error.

    A second real connection opens ``BEGIN IMMEDIATE`` and never releases it —
    modelling a writer that is genuinely stuck, not merely between statements —
    so every attempt the store makes must observe ``SQLITE_BUSY``.
    """
    holder_conn = await aiosqlite.connect(db_path)
    await holder_conn.execute("PRAGMA busy_timeout=0")
    await holder_conn.execute("BEGIN IMMEDIATE")

    sut_conn, store = await _open_store(db_path, busy_timeout_ms=20)
    try:
        with pytest.raises(WriteContentionError) as excinfo:
            await store.get_or_create(_thought("t-busy", content="Contended content."))
        err = excinfo.value
        assert isinstance(err, EngravaError)
        assert err.operation == "get_or_create"
        assert err.attempts == 3
        assert "get_or_create" in str(err)
        assert "3 attempt" in str(err)
        # Never leaks the raw driver error type as the exception seen by the
        # caller; it is chained, not swallowed.
        assert isinstance(err.__cause__, sqlite3.OperationalError)

        # Nothing was ever admitted: the failed ``BEGIN IMMEDIATE`` opened no
        # transaction and inserted no row.
        assert not sut_conn.in_transaction
        cursor = await sut_conn.execute("SELECT COUNT(*) FROM thought")
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == 0
    finally:
        await holder_conn.rollback()
        await holder_conn.close()
        await sut_conn.close()


async def test_contention_clears_before_attempts_exhausted_succeeds(db_path: str) -> None:
    """A transient lock that clears mid-retry lets the call succeed.

    The holder releases its lock shortly after the first (immediately-failing)
    attempt, while the backoff before the second attempt is still elapsing, so
    the retry — not a fresh unlocked call — is what makes this pass.
    """
    holder_conn = await aiosqlite.connect(db_path)
    await holder_conn.execute("PRAGMA busy_timeout=0")
    await holder_conn.execute("BEGIN IMMEDIATE")

    async def _release_shortly() -> None:
        await asyncio.sleep(0.03)
        await holder_conn.rollback()

    # A short busy_timeout so the first attempt fails quickly (well before the
    # holder releases at 0.03s); the retry backoff (0.05s) then lands after
    # the release, so the second attempt is the one that succeeds.
    sut_conn, store = await _open_store(db_path, busy_timeout_ms=5)
    try:
        release_task = asyncio.create_task(_release_shortly())
        record, created = await store.get_or_create(_thought("t-retry", content="Retried content."))
        await release_task
        assert created is True
        assert record.thought_id == "t-retry"

        cursor = await sut_conn.execute("SELECT COUNT(*) FROM thought")
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == 1
    finally:
        await holder_conn.close()
        await sut_conn.close()


async def test_create_thought_dedup_true_raises_write_contention_error(db_path: str) -> None:
    """The same contention surfaces through ``create_thought(deduplicate=True)``."""
    holder_conn = await aiosqlite.connect(db_path)
    await holder_conn.execute("PRAGMA busy_timeout=0")
    await holder_conn.execute("BEGIN IMMEDIATE")

    sut_conn, store = await _open_store(db_path, busy_timeout_ms=20)
    try:
        with pytest.raises(WriteContentionError) as excinfo:
            await store.create_thought(
                _thought("t-busy-create", content="Contended create_thought."),
                deduplicate=True,
            )
        assert excinfo.value.operation == "create_thought"
    finally:
        await holder_conn.rollback()
        await holder_conn.close()
        await sut_conn.close()


async def test_upsert_by_hash_raises_write_contention_error(db_path: str) -> None:
    """The same contention surfaces through ``upsert_by_hash``."""
    holder_conn = await aiosqlite.connect(db_path)
    await holder_conn.execute("PRAGMA busy_timeout=0")
    await holder_conn.execute("BEGIN IMMEDIATE")

    sut_conn, store = await _open_store(db_path, busy_timeout_ms=20)
    try:
        with pytest.raises(WriteContentionError) as excinfo:
            await store.upsert_by_hash(_thought("t-busy-upsert", content="Contended upsert."))
        assert excinfo.value.operation == "upsert_by_hash"
    finally:
        await holder_conn.rollback()
        await holder_conn.close()
        await sut_conn.close()


# ---------------------------------------------------------------------------
# Write lock vs. auto-embed -- the binding design constraint
# ---------------------------------------------------------------------------


class _BlockingEmbeddingProvider:
    """An embedding provider whose ``embed`` call blocks until released.

    Lets a test observe, from the outside, whether the dedup write lock is
    still held while an embedding call is in flight -- rather than asserting
    on the internal call order, which a future refactor could preserve while
    still breaking the actual property.
    """

    model_name = "blocking-mock-embedder"

    def __init__(self) -> None:
        self.embed_started = asyncio.Event()
        self.release = asyncio.Event()

    @property
    def dimension(self) -> int:
        return 4

    async def embed(self, text: str) -> list[float]:
        self.embed_started.set()
        await self.release.wait()
        return [0.1, 0.2, 0.3, 0.4]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(text) for text in texts]


async def _open_store_with_embedding_provider(
    db_path: str,
    provider: _BlockingEmbeddingProvider,
) -> tuple[aiosqlite.Connection, SqliteEngravaCore]:
    """Open a fresh connection + store with auto-embed wired to *provider*."""
    conn = await aiosqlite.connect(db_path)
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA foreign_keys=ON")
    await conn.execute("PRAGMA busy_timeout=5000")
    store = SqliteEngravaCore(conn, embedding_provider=provider, auto_embed=True)
    await store._probe_fts()
    return conn, store


async def test_write_lock_is_released_before_auto_embed_runs(db_path: str) -> None:
    """The RESERVED write lock must not be held across the auto-embed call.

    Binding design constraint: the dedup
    probe-and-insert window's ``BEGIN IMMEDIATE`` must close *before* any
    ``await`` that does external I/O -- embedding chief among them -- or a
    slow or hanging provider call would stall every other writer on the file,
    trading the original data-loss bug for a store-wide outage instead of
    fixing it.

    This asserts the *observable* property directly rather than the internal
    call order: while a real ``get_or_create`` call's embedding is
    deliberately blocked mid-flight, a second, completely independent
    connection to the same file must still be able to open its own
    ``BEGIN IMMEDIATE`` -- which is only possible once the first call's write
    lock has actually been released at the SQLite level. Asserting "commit
    happens before auto-embed in the source" would keep passing even if a
    future change moved the release to the wrong place while preserving that
    textual ordering; only a real second connection can show the lock is
    actually gone.
    """
    provider = _BlockingEmbeddingProvider()
    conn, store = await _open_store_with_embedding_provider(db_path, provider)
    checker_conn = await aiosqlite.connect(db_path)
    try:
        await checker_conn.execute("PRAGMA busy_timeout=0")

        task = asyncio.create_task(
            store.get_or_create(_thought("t-embed-lock", content="Embedding-in-flight probe.")),
        )
        try:
            await asyncio.wait_for(provider.embed_started.wait(), timeout=5)

            # The embed call is now parked on ``provider.release``. With
            # ``busy_timeout=0`` this either succeeds immediately (lock is
            # free) or raises immediately (lock still held) -- no waiting, so
            # the assertion below is decisive rather than a timing guess.
            try:
                await checker_conn.execute("BEGIN IMMEDIATE")
            except sqlite3.OperationalError as exc:
                pytest.fail(
                    "write lock still held during auto-embed -- a second, "
                    f"independent connection could not open BEGIN IMMEDIATE: {exc}",
                )
            else:
                await checker_conn.rollback()
        finally:
            provider.release.set()

        record, created = await task
        assert created is True
        assert record.thought_id == "t-embed-lock"
    finally:
        await checker_conn.close()
        await conn.close()


# ---------------------------------------------------------------------------
# Whoever writes, commits -- the journal entry must land with the row
# ---------------------------------------------------------------------------


async def test_hit_path_journal_entry_is_committed_with_the_confirmation_bump(
    db_path: str,
) -> None:
    """The dedup guard commits nothing itself; the write path must commit all of it.

    ``_serialize_dedup_probe`` takes no commit responsibility of its own —
    whichever method does the write is the one that must make all of that
    write's parts durable, including its own journal append.
    ``_increment_confirmation`` (the ``get_or_create`` / ``create_thought
    (deduplicate=True)`` hit path) orders ``UPDATE -> read-back ->
    journal.append() -> _maybe_commit()`` specifically so the single commit at
    the end covers the journal entry too. Getting that order wrong (committing
    before the journal append, as an earlier version of this guard's fix did)
    leaves the append's own INSERT sitting in a fresh, uncommitted transaction
    that nothing then closes.

    Verified from the outside: a **second, independent connection** to the
    same file can see the journal row after the call returns. An uncommitted
    write is invisible across connections, so this is decisive rather than a
    same-connection read that an open transaction would also satisfy.
    """
    conn = await aiosqlite.connect(db_path)
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA foreign_keys=ON")
    store = SqliteEngravaCore(conn, journal_enabled=True)
    await store._probe_fts()
    try:
        await store.create_thought(_thought("t-hit-journal", content="journal probe"))

        record, created = await store.get_or_create(_thought("t-other-id", content="journal probe"))
        assert created is False
        assert record.thought_id == "t-hit-journal"
        assert record.confirmation_count == 1

        # Decisive: the guard's own window must have released the write lock
        # entirely -- not merely be quiescent on this connection.
        assert not conn.in_transaction

        checker_conn = sqlite3.connect(db_path)
        try:
            row = checker_conn.execute(
                "SELECT mutation_type, target_id FROM journal_entry "
                "WHERE target_id = ? ORDER BY sequence_number DESC LIMIT 1",
                ("t-hit-journal",),
            ).fetchone()
        finally:
            checker_conn.close()

        assert row is not None, (
            "journal entry for the confirmation bump is not visible to a second "
            "connection -- it was left in an uncommitted transaction"
        )
        assert row[0] == "UPDATE_THOUGHT"
        assert row[1] == "t-hit-journal"
    finally:
        await conn.close()


async def test_upsert_by_hash_no_op_match_releases_the_write_lock(db_path: str) -> None:
    """A matched row with no differing fields releases its opened lock.

    ``upsert_by_hash`` opens ``BEGIN IMMEDIATE`` *before* the probe runs, so it
    cannot yet know the match will turn out to need no change. When it does
    turn out that way (``changes`` computes empty), nothing is written on that
    branch at all -- no ``UPDATE``, no journal entry -- so it
    commits nothing (committing here would flush whatever *unrelated*,
    still-pending work the caller already had open on this connection -- see
    :meth:`SqliteEngravaCore.upsert_by_hash`'s no-op branch). But a branch that
    writes nothing has nothing of its own to preserve either, so it does not
    leave the window open: :meth:`SqliteEngravaCore._end_exploratory_probe`
    rolls back this call's own, still-empty ``BEGIN IMMEDIATE`` before
    returning -- never a caller's, since it only acts when this call's own
    probe is the one that opened the transaction (see that method's
    docstring for the ownership test). Decisive check: a **second,
    independent connection** *can* open its own ``BEGIN IMMEDIATE``
    immediately afterward, because the lock genuinely has been released, not
    merely because this connection happens to be quiescent.
    """
    conn = await aiosqlite.connect(db_path)
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA foreign_keys=ON")
    store = SqliteEngravaCore(conn)
    await store._probe_fts()
    try:
        await store.create_thought(_thought("t-noop-match", content="identical fields"))

        result = await store.upsert_by_hash(_thought("t-other-id", content="identical fields"))
        assert result.thought_id == "t-noop-match"
        assert not conn.in_transaction

        checker_conn = await aiosqlite.connect(db_path)
        try:
            await checker_conn.execute("PRAGMA busy_timeout=0")
            try:
                await checker_conn.execute("BEGIN IMMEDIATE")
            except sqlite3.OperationalError:
                pytest.fail(
                    "write lock was still held after a no-op upsert_by_hash match -- "
                    "a second, independent connection could not open its own "
                    "BEGIN IMMEDIATE, which means this call left its own, empty "
                    "transaction open instead of releasing the reservation it took.",
                )
            else:
                await checker_conn.rollback()
        finally:
            await checker_conn.close()
    finally:
        await conn.close()


async def test_upsert_by_hash_noop_first_in_suspend_window_releases_the_lock(
    db_path: str,
) -> None:
    """A no-op match as the *first* action in a suspend window still releases the lock.

    ``suspend_auto_commit()`` itself opens no transaction -- one is opened
    lazily by whichever write happens first inside the window. When that
    first action is a no-op ``upsert_by_hash()`` match, *this call's own*
    probe is the one that opens the ``BEGIN IMMEDIATE`` (nothing else in the
    window has touched the connection yet), so ``opened_transaction`` is
    ``True`` and :meth:`SqliteEngravaCore._end_exploratory_probe` must roll
    it back -- even though a ``suspend_auto_commit()`` window is nominally
    still open around it. This is the exact case an earlier, rejected version
    of that method got wrong (gating the rollback on
    ``not self._skip_auto_commit`` in addition to ``opened_transaction``): it
    would hold this reservation for the *entire remaining duration of the
    window*, not merely until the next write, because ``_skip_auto_commit``
    is ``True`` for the window's whole span regardless of which call opened
    the transaction. Decisive check: a **second, independent connection**
    can take the write lock immediately, while the ``suspend_auto_commit()``
    window is still open around the no-op call, proving the reservation was
    released rather than held for the window's remaining lifetime.
    """
    conn = await aiosqlite.connect(db_path)
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA foreign_keys=ON")
    store = SqliteEngravaCore(conn)
    await store._probe_fts()
    try:
        await store.create_thought(_thought("t-window-seed", content="identical fields"))

        async with store.suspend_auto_commit():
            result = await store.upsert_by_hash(
                _thought("t-window-other", content="identical fields"),
            )
            assert result.thought_id == "t-window-seed"

            checker_conn = await aiosqlite.connect(db_path)
            try:
                await checker_conn.execute("PRAGMA busy_timeout=0")
                try:
                    await checker_conn.execute("BEGIN IMMEDIATE")
                except sqlite3.OperationalError:
                    pytest.fail(
                        "write lock was still held after a no-op upsert_by_hash match "
                        "that was the first action inside a suspend_auto_commit() "
                        "window -- the reservation this call's own probe opened was "
                        "held for the rest of the window instead of being released.",
                    )
                else:
                    await checker_conn.rollback()
            finally:
                await checker_conn.close()
    finally:
        await conn.close()


async def test_cancellation_inside_the_probe_releases_the_write_lock(db_path: str) -> None:
    """Cancelling a task mid-probe rolls back and releases the RESERVED lock.

    ``_serialize_dedup_probe`` keeps a rollback on its exception path
    specifically so cancellation is covered: ``except BaseException``, not
    ``except Exception`` -- ``asyncio.CancelledError`` derives from
    ``BaseException``. This exercises that path directly against the guard
    itself (a sibling of ``TestSuspendAutoCommitIsStoreWide``'s cancellation
    test in ``test_concurrency_contract.py``, which covers
    ``suspend_auto_commit`` -- a different context manager with its own
    cleanup). Driven by an ``asyncio.Event``, not a sleep: the probe signals
    once it has started so the cancellation lands deterministically while the
    guard's ``BEGIN IMMEDIATE`` transaction is still open, before any row is
    written.
    """
    conn = await aiosqlite.connect(db_path)
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA foreign_keys=ON")
    store = SqliteEngravaCore(conn)
    await store._probe_fts()
    try:
        started = asyncio.Event()
        orig_probe = store._get_thought_by_content_hash

        async def slow_probe(content_hash: str) -> ThoughtRecord | None:
            started.set()
            await asyncio.sleep(10)  # cancelled long before this would return
            return await orig_probe(content_hash)

        store._get_thought_by_content_hash = slow_probe  # type: ignore[method-assign]

        task = asyncio.create_task(
            store.get_or_create(_thought("t-cancel-probe", content="cancel me")),
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert not conn.in_transaction, (
            "RESERVED lock left stranded after cancellation inside the dedup probe "
            f"(in_transaction={conn.in_transaction})"
        )

        # The store must still be usable -- a stranded lock would hang this or
        # raise "cannot start a transaction within a transaction".
        created = await store.create_thought(_thought("t-after-cancel", content="still works"))
        assert created.thought_id == "t-after-cancel"
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# Exception identity / export
# ---------------------------------------------------------------------------


def test_write_contention_error_is_exported_from_package_root() -> None:
    """``WriteContentionError`` is importable from ``engrava`` and typed correctly.

    Regression pin against the sibling defect noted in the workstream that
    introduced this error: three pre-existing exceptions are defined in
    ``engrava.domain.exceptions`` but never imported into ``engrava/__init__.py``,
    so they are unreachable via ``from engrava import ...``. This error must not
    repeat that mistake.
    """
    import engrava

    assert engrava.WriteContentionError is WriteContentionError
    assert "WriteContentionError" in engrava.__all__
    assert issubclass(WriteContentionError, EngravaError)


def test_write_contention_error_message_and_fields() -> None:
    """Constructing the error directly carries structured fields + a clear message."""
    err = WriteContentionError(operation="create_thought", attempts=3)
    assert err.operation == "create_thought"
    assert err.attempts == 3
    message = str(err)
    assert "create_thought" in message
    assert "3 attempt" in message


# ---------------------------------------------------------------------------
# Busy classification (module-private helper)
# ---------------------------------------------------------------------------


def test_is_busy_error_recognises_primary_and_extended_codes() -> None:
    """``_is_busy_error`` matches SQLITE_BUSY and its extended forms."""
    for code in (5, 261, 517, 773):
        exc = sqlite3.OperationalError("database is locked")
        exc.sqlite_errorcode = code
        assert _is_busy_error(exc) is True


def test_is_busy_error_rejects_unrelated_operational_errors() -> None:
    """A non-busy ``OperationalError`` (e.g. a locked schema) is not misclassified."""
    exc = sqlite3.OperationalError("database schema is locked")
    exc.sqlite_errorcode = 6  # SQLITE_LOCKED, not SQLITE_BUSY
    assert _is_busy_error(exc) is False


async def test_non_busy_operational_error_is_not_wrapped(db_path: str) -> None:
    """A non-busy failure opening the transaction propagates unchanged.

    Retrying could not fix a genuine (non-contention) operational failure, and
    mislabelling it as ``WriteContentionError`` would hide the real cause, so
    ``_begin_dedup_write_lock`` must let it through untouched.
    """
    _conn, store = await _open_store(db_path, busy_timeout_ms=20)
    try:

        async def _boom(*_args: object, **_kwargs: object) -> None:
            exc = sqlite3.OperationalError("disk I/O error")
            exc.sqlite_errorcode = 10  # SQLITE_IOERR, not a busy code
            raise exc

        store._db.execute = _boom  # type: ignore[assignment]

        with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
            await store._begin_dedup_write_lock(operation="create_thought")
    finally:
        await _conn.close()
