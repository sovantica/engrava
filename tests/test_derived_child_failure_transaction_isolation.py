"""A failed derived child undoes only itself, in every transaction context.

Covers the acceptance scenarios that ``tests/test_derived_records_seam.py``
does not: a per-child failure inside a caller's own ``suspend_auto_commit()``
window, and inside a caller-held raw ``BEGIN``. Outside any caller
transaction (per-child commits one at a time) is already covered by the
existing suite and is unchanged by this file.

Two variants below (``test_journal_append_failure_in_window_...`` and
``test_embedding_provider_failure_in_window_...``) reproduce the RED result
recorded by a probe run before this fix (not part of this public repo):
a caller's own earlier write in the same window was silently discarded when
a derived child failed under the default ``on_error="log"``.

**Every database here is file-backed, and every "durable after this point"
claim is proven by reopening a fresh, separate connection** (``_reopened``)
rather than reading back through the connection that wrote it: the writing
connection would still see its own uncommitted work, which is exactly what
would let a false "durable" claim slip through unnoticed.

**A raw ``BEGIN`` is not gated by ``_skip_auto_commit`` the way
``suspend_auto_commit()`` is.** ``suspend_auto_commit()`` sets that flag for
its whole duration, so every write's own ``_maybe_commit()`` becomes a no-op
until the window's single, final commit. A caller-issued raw ``BEGIN`` does
not set it at all: the *first* write to succeed still calls its own
``_maybe_commit()``, which -- since the flag is unset -- actually commits the
whole shared transaction, the caller's own pending write included. So a
raw-``BEGIN`` case where the child's row insert (step 1) itself is what fails
genuinely keeps the caller's write pending until its own explicit commit; a
case where a *later* step fails (the row already inserted successfully) sees
the caller's write committed early, as soon as that row's own insert commits
-- well before the later step ever runs. Each raw-``BEGIN`` test below
documents which of the two shapes applies to it. This also means a caller
holding a raw ``BEGIN`` does not uniformly "own" when its own writes become
durable the way a ``suspend_auto_commit()`` caller does -- see
:meth:`~engrava.infrastructure.sqlite.engrava_core.SqliteEngravaCore.derive_existing`
for the same point made at the source.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

import aiosqlite

from engrava import (
    CoreThoughtRecord,
    DefaultEngravaHooks,
    DerivedRecord,
    DeriveGates,
    LifecycleStatus,
    Priority,
    SqliteEngravaCore,
    ThoughtType,
)
from engrava.embeddings.callback import CallbackProvider
from engrava.infrastructure.sqlite.engrava_core import _derived_edge_id, _derived_thought_id
from tests.test_sqlite_vec import sqlite_vec_required

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    from pathlib import Path

    import pytest

    from engrava.domain.models.thought import ThoughtRecord
    from engrava.domain.protocols.derived_records import DeriveContext

_EMBEDDING_INSERT_SQL = (
    "INSERT INTO embedding "
    "(embedding_id, owner_type, owner_id, model_name, "
    "dimension, vector_blob, created_at) "
    "VALUES (?, ?, ?, ?, ?, ?, ?)"
)


@contextlib.asynccontextmanager
async def _connection(path: str) -> AsyncIterator[aiosqlite.Connection]:
    """A connection that is always closed, including on a failed assertion.

    An unclosed aiosqlite connection leaves a non-daemon worker thread behind
    that hangs the interpreter at process exit -- every test in this module
    opens its own connection directly (rather than a shared fixture, since
    several need a caller-held raw ``BEGIN`` spanning the whole test body) and
    so must guarantee the close itself. Always file-backed (never ``:memory:``)
    so a fresh, separate connection can reopen it afterward -- see
    ``_reopened``.
    """
    conn = await aiosqlite.connect(path)
    conn.row_factory = aiosqlite.Row
    try:
        yield conn
    finally:
        await conn.close()


@contextlib.asynccontextmanager
async def _reopened(path: str) -> AsyncIterator[SqliteEngravaCore]:
    """A fresh, separate connection + read-only store over the same on-disk database.

    Reading a "durable after this point" claim back through the connection
    that wrote it would still see that connection's own uncommitted work --
    proving durability requires a genuinely independent second connection
    that only ever sees what actually reached disk.
    """
    conn = await aiosqlite.connect(path)
    conn.row_factory = aiosqlite.Row
    store = SqliteEngravaCore(conn)
    try:
        yield store
    finally:
        await conn.close()


def _source(thought_id: str, content: str) -> CoreThoughtRecord:
    """Build a realistic source thought."""
    return CoreThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.NOTE,
        essence="source essence",
        content=content,
        priority=Priority.P2,
        lifecycle_status=LifecycleStatus.CREATED,
        created_cycle=0,
        updated_cycle=0,
        source="test-suite",
    )


def _child(content: str, *, attach_edge: bool = True) -> DerivedRecord:
    """Build a derived record with the given content."""
    return DerivedRecord(
        content=content,
        thought_type=ThoughtType.OBSERVATION,
        priority=Priority.P3,
        attach_provenance_edge=attach_edge,
    )


class ListProducer(DefaultEngravaHooks):
    """Return a fixed list of derived records for any source."""

    def __init__(self, records: list[DerivedRecord]) -> None:
        self._records = records

    async def derive_records(
        self,
        thought: ThoughtRecord,
        ctx: DeriveContext,
    ) -> Sequence[DerivedRecord]:
        return self._records


async def _open_store(
    conn: aiosqlite.Connection,
    producer: DefaultEngravaHooks,
    *,
    on_error: str,
    embedding_provider: CallbackProvider | None = None,
) -> SqliteEngravaCore:
    store = SqliteEngravaCore(
        conn,
        hooks=producer,
        journal_enabled=True,
        derive_gates=DeriveGates(enabled=False, on_error=on_error),  # type: ignore[arg-type]
        embedding_provider=embedding_provider,
        auto_embed=embedding_provider is not None,
    )
    await store.ensure_schema()
    return store


def _patch_journal_to_fail(
    store: SqliteEngravaCore,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mutation_type: str,
    target_id: str,
) -> None:
    """Make the journal writer raise for exactly one (mutation_type, target_id)."""
    assert store.journal is not None
    original = store.journal.append
    fail_mutation = mutation_type
    fail_target = target_id

    async def _flaky_append(
        mutation_type: str,
        target_id: str | None,
        delta: dict[str, object],
    ) -> object:
        if mutation_type == fail_mutation and target_id == fail_target:
            msg = f"injected journal failure for {target_id}"
            raise RuntimeError(msg)
        return await original(mutation_type=mutation_type, target_id=target_id, delta=delta)

    monkeypatch.setattr(store.journal, "append", _flaky_append)


def _failing_provider(poison_content: str) -> CallbackProvider:
    """An embedding provider that raises only when embedding ``poison_content``."""

    def _cb(text: str) -> list[float]:
        if poison_content in text:
            msg = "injected provider failure"
            raise RuntimeError(msg)
        return [float(len(text) % 7), 1.0, 0.5, 0.25]

    return CallbackProvider(_cb, dimension=4, model_name="test-4")


def _working_provider() -> CallbackProvider:
    """An embedding provider that always succeeds."""

    def _cb(text: str) -> list[float]:
        return [float(len(text) % 7), 1.0, 0.5, 0.25]

    return CallbackProvider(_cb, dimension=4, model_name="test-4")


def _distinctive_vector_provider(
    poison_content: str, poison_vector: list[float]
) -> CallbackProvider:
    """An embedding provider that returns a distinctive, easily-matched vector for one child."""

    def _cb(text: str) -> list[float]:
        if poison_content in text:
            return list(poison_vector)
        return [float(len(text) % 7), 1.0, 0.5, 0.25]

    return CallbackProvider(_cb, dimension=len(poison_vector), model_name="test-vec")


def _patch_embedding_insert_to_fail(
    store: SqliteEngravaCore,
    monkeypatch: pytest.MonkeyPatch,
    *,
    owner_id: str,
) -> None:
    """Make ``store_embedding``'s own base-row INSERT fail for one owner id.

    The provider succeeds; the failure is store_embedding's own unit's base
    ``embedding`` row write, injected *before* that ``INSERT`` ever runs --
    there is no journal append in this path to fail instead, unlike every
    other guarded write in this store. Contrast with
    ``_patch_vec_upsert_to_fail_for_vector``, which fails a *later* statement
    in the same unit, after this base row genuinely exists.
    """
    real_execute = store._db.execute

    async def _wrapper(sql: str, parameters: object = None) -> object:
        if (
            sql == _EMBEDDING_INSERT_SQL
            and isinstance(parameters, tuple)
            and parameters[2] == owner_id
        ):
            msg = f"injected embedding insert failure for {owner_id}"
            raise RuntimeError(msg)
        return await real_execute(sql, parameters)

    monkeypatch.setattr(store._db, "execute", _wrapper)


def _patch_vec_upsert_to_fail_for_vector(
    store: SqliteEngravaCore,
    monkeypatch: pytest.MonkeyPatch,
    poison_vector: list[float],
    captured_rowid: list[int],
) -> None:
    """Let the real sqlite-vec upsert run for one vector, then fail right after.

    Unlike ``_patch_embedding_insert_to_fail``, this calls the *real*
    ``upsert_embedding`` first -- so a genuine ``embedding_vec`` row is
    written for the matched vector -- and only then raises. By the time the
    failure hits, both the base ``embedding`` row's own ``INSERT`` (executed,
    not committed -- both are still inside ``store_embedding``'s savepoint
    unit and the caller's suspended-commit window) and the vec0 upsert have
    genuinely, physically run, so the unit's ``ROLLBACK TO`` has to undo a
    real row and a real vector, not merely an attempted one. The matched
    rowid is appended to ``captured_rowid`` so the caller can confirm no
    ``embedding_vec`` row survives at that id.
    """
    backend = store._vector_backend
    assert backend is not None
    real_upsert = backend.upsert_embedding

    async def _wrapper(db: aiosqlite.Connection, *, rowid: int, vector: list[float]) -> None:
        await real_upsert(db, rowid=rowid, vector=vector)
        if vector == poison_vector:
            captured_rowid.append(rowid)
            msg = "injected vec0 upsert failure"
            raise RuntimeError(msg)

    monkeypatch.setattr(backend, "upsert_embedding", _wrapper)


async def _embedding_row_exists(conn: aiosqlite.Connection, owner_id: str) -> bool:
    cursor = await conn.execute("SELECT 1 FROM embedding WHERE owner_id = ?", (owner_id,))
    return await cursor.fetchone() is not None


async def _has_embedding(store: SqliteEngravaCore, thought_id: str) -> bool:
    return await store.get_embedding(thought_id) is not None


# ---------------------------------------------------------------------------
# Inside a caller's suspend_auto_commit() window, on_error="log" (the default)
# ---------------------------------------------------------------------------


async def test_journal_append_failure_in_window_log_leaves_callers_write_durable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED -> GREEN: a child's row-journal failure, inside a window, under "log".

    Before the fix, the child's compensating rollback discarded the whole
    transaction, including the caller's own earlier write (``X``), silently,
    because ``on_error="log"`` never told the caller.
    """
    source_id = "src-1"
    doomed = "doomed child"
    db_path = tmp_path / "window.db"
    async with _connection(str(db_path)) as conn:
        producer = ListProducer([_child("first good"), _child(doomed), _child("second good")])
        store = await _open_store(conn, producer, on_error="log")
        await store.create_thought(_source(source_id, "Body."))
        doomed_id = _derived_thought_id(doomed)
        _patch_journal_to_fail(
            store, monkeypatch, mutation_type="INSERT_THOUGHT", target_id=doomed_id
        )

        async with store.suspend_auto_commit():
            await store.create_thought(_source("unrelated-src", "unrelated body"))  # X
            result = await store.derive_existing(source_id)

        # derive_existing returned normally -- no exception escaped under "log".
        assert result.created == 2
        assert result.skipped == 1
        assert store._db.in_transaction is False

    async with _reopened(str(db_path)) as fresh:
        # X, the caller's own earlier write, is durable after the clean exit.
        assert await fresh.get_thought("unrelated-src") is not None
        # The failed child (row and journal entry) is absent.
        assert await fresh.get_thought(doomed_id) is None
        # The other children are present.
        assert await fresh.get_thought(_derived_thought_id("first good")) is not None
        assert await fresh.get_thought(_derived_thought_id("second good")) is not None
        assert (await fresh.verify_journal()).valid


async def test_embedding_provider_failure_in_window_log_leaves_callers_write_durable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED -> GREEN: the embedding provider raises for one child, inside a window, "log".

    The provider raises before writing anything (``_auto_embed_thought``
    raises ahead of its locked section), so there is nothing of the embed
    step itself to unwind -- but the old full-transaction compensation still
    discarded ``X``.
    """
    source_id = "src-1"
    doomed = "doomed child"
    db_path = tmp_path / "window.db"
    async with _connection(str(db_path)) as conn:
        producer = ListProducer([_child("first good"), _child(doomed), _child("second good")])
        store = await _open_store(
            conn, producer, on_error="log", embedding_provider=_failing_provider(doomed)
        )
        await store.create_thought(_source(source_id, "Body."))
        doomed_id = _derived_thought_id(doomed)

        async with store.suspend_auto_commit():
            await store.create_thought(_source("unrelated-src", "unrelated body"))  # X
            result = await store.derive_existing(source_id)

        assert result.created == 2
        assert result.skipped == 1
        assert store._db.in_transaction is False

    async with _reopened(str(db_path)) as fresh:
        assert await fresh.get_thought("unrelated-src") is not None
        # That child's row is present without an embedding.
        assert await fresh.get_thought(doomed_id) is not None
        assert not await _has_embedding(fresh, doomed_id)
        assert await fresh.get_thought(_derived_thought_id("first good")) is not None
        assert await fresh.get_thought(_derived_thought_id("second good")) is not None
        assert (await fresh.verify_journal()).valid


async def test_store_embedding_failure_in_window_log_leaves_callers_write_durable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child's ``store_embedding`` fails before its base INSERT runs, inside a window, "log".

    Unlike a provider failure, the provider *does* return a vector here; the
    failure is injected on ``store_embedding``'s own base-row ``INSERT``
    itself, before it ever executes, and is unwound through that method's own
    ``_write_readback_savepoint`` unit -- not any compensation in
    ``_persist_derived_child`` itself. See
    ``test_vec0_upsert_fails_after_base_insert_in_window_log_leaves_callers_write_durable``
    below for the same unit failing *after* that row genuinely exists.
    """
    source_id = "src-1"
    doomed = "doomed child"
    db_path = tmp_path / "window.db"
    async with _connection(str(db_path)) as conn:
        producer = ListProducer([_child("first good"), _child(doomed), _child("second good")])
        store = await _open_store(
            conn, producer, on_error="log", embedding_provider=_working_provider()
        )
        await store.create_thought(_source(source_id, "Body."))
        doomed_id = _derived_thought_id(doomed)
        _patch_embedding_insert_to_fail(store, monkeypatch, owner_id=doomed_id)

        async with store.suspend_auto_commit():
            await store.create_thought(_source("unrelated-src", "unrelated body"))  # X
            result = await store.derive_existing(source_id)

        assert result.created == 2
        assert result.skipped == 1
        assert store._db.in_transaction is False

    async with _reopened(str(db_path)) as fresh:
        assert await fresh.get_thought("unrelated-src") is not None
        assert await fresh.get_thought(doomed_id) is not None
        assert not await _has_embedding(fresh, doomed_id)
        # No partial embedding row or vector remains.
        assert not await _embedding_row_exists(fresh._db, doomed_id)
        assert await fresh.get_thought(_derived_thought_id("first good")) is not None
        assert await fresh.get_thought(_derived_thought_id("second good")) is not None
        assert (await fresh.verify_journal()).valid


@sqlite_vec_required
async def test_vec0_upsert_fails_after_base_insert_in_window_log_leaves_callers_write_durable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``store_embedding``'s vec0 upsert genuinely runs, then fails, inside a window.

    Distinct from
    ``test_store_embedding_failure_in_window_log_leaves_callers_write_durable``,
    which fails before the base INSERT itself ever runs: here a real
    sqlite-vec backend is loaded, the base ``embedding`` row's own ``INSERT``
    executes (not commits -- both it and the vec0 upsert below are still
    inside ``store_embedding``'s own savepoint unit and the caller's
    suspended-commit window), the vec0 upsert statement itself then also runs
    for real, and only *after* that genuinely-written vector exists does the
    injected failure raise. So the unit's ``ROLLBACK TO`` has to undo a base
    row and a vec0 row that were both actually, physically written, not
    merely attempted.
    """
    source_id = "src-1"
    doomed = "doomed child"
    poison_vector = [-1.0, -1.0, -1.0, -1.0]
    db_path = tmp_path / "window-vec.db"
    captured_rowid: list[int] = []
    async with _connection(str(db_path)) as conn:
        # "doomed" is deliberately last: SQLite reuses a freed rowid for the
        # very next insert on the table, so if a later child's own embedding
        # were attempted after the rollback frees the doomed rowid, it could
        # legitimately land at that same numeric rowid and defeat the
        # rowid-based absence check below. Putting doomed last means nothing
        # in this batch inserts into `embedding` again afterward.
        producer = ListProducer([_child("first good"), _child("second good"), _child(doomed)])
        provider = _distinctive_vector_provider(doomed, poison_vector)
        store = await _open_store(conn, producer, on_error="log", embedding_provider=provider)
        await store._configure_vector_backend(backend_name="sqlite-vec", embedding_dimension=4)
        await store.create_thought(_source(source_id, "Body."))
        doomed_id = _derived_thought_id(doomed)
        _patch_vec_upsert_to_fail_for_vector(store, monkeypatch, poison_vector, captured_rowid)

        async with store.suspend_auto_commit():
            await store.create_thought(_source("unrelated-src", "unrelated body"))  # X
            result = await store.derive_existing(source_id)

        assert result.created == 2
        assert result.skipped == 1
        assert store._db.in_transaction is False

    # The real upsert ran (and wrote a real embedding_vec row) exactly once,
    # for the doomed child, before the injected failure raised.
    assert len(captured_rowid) == 1
    doomed_rowid = captured_rowid[0]

    async with _reopened(str(db_path)) as fresh:
        # Re-attach sqlite-vec on this fresh connection: the extension is
        # per-connection, even though the embedding_vec table's own schema
        # already exists on disk from the first connection's setup.
        await fresh._configure_vector_backend(backend_name="sqlite-vec", embedding_dimension=4)

        assert await fresh.get_thought("unrelated-src") is not None
        assert await fresh.get_thought(doomed_id) is not None
        assert not await _has_embedding(fresh, doomed_id)
        # No partial embedding row remains -- the unit undid the base row
        # too, even though that row genuinely existed at the moment the vec0
        # upsert failed.
        assert not await _embedding_row_exists(fresh._db, doomed_id)
        # Nor does the vec0 row the real upsert actually wrote, at the exact
        # rowid it was written under.
        vec_cursor = await fresh._db.execute(
            "SELECT 1 FROM embedding_vec WHERE rowid = ?", (doomed_rowid,)
        )
        assert await vec_cursor.fetchone() is None
        assert await fresh.get_thought(_derived_thought_id("first good")) is not None
        assert await fresh.get_thought(_derived_thought_id("second good")) is not None
        assert (await fresh.verify_journal()).valid


async def test_edge_journal_failure_in_window_log_leaves_callers_write_durable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child's edge-journal append fails, inside a window, under "log".

    Step 1 (the row insert) already succeeded and released its own savepoint
    cleanly before step 3 (the edge) ever runs -- but inside the window
    ``_maybe_commit()`` is a no-op (``_skip_auto_commit`` is set), so nothing
    has actually committed yet; only the window's own eventual exit decides
    that. Only the edge insert and its journal entry (step 3's own unit)
    unwind.
    """
    source_id = "src-1"
    doomed = "doomed child"
    db_path = tmp_path / "window.db"
    async with _connection(str(db_path)) as conn:
        producer = ListProducer([_child("first good"), _child(doomed), _child("second good")])
        store = await _open_store(conn, producer, on_error="log")
        await store.create_thought(_source(source_id, "Body."))
        doomed_id = _derived_thought_id(doomed)
        doomed_edge_id = _derived_edge_id(doomed_id, source_id)
        _patch_journal_to_fail(
            store, monkeypatch, mutation_type="INSERT_EDGE", target_id=doomed_edge_id
        )

        async with store.suspend_auto_commit():
            await store.create_thought(_source("unrelated-src", "unrelated body"))  # X
            result = await store.derive_existing(source_id)

        assert result.created == 2
        assert result.skipped == 1
        assert store._db.in_transaction is False

    async with _reopened(str(db_path)) as fresh:
        assert await fresh.get_thought("unrelated-src") is not None
        # That child's row is present; its edge and journal entry are absent.
        assert await fresh.get_thought(doomed_id) is not None
        in_edges = await fresh.get_edges(source_id, direction="IN")
        assert len(in_edges) == 2
        assert doomed_id not in {edge.from_thought_id for edge in in_edges}
        assert await fresh.get_thought(_derived_thought_id("first good")) is not None
        assert await fresh.get_thought(_derived_thought_id("second good")) is not None
        assert (await fresh.verify_journal()).valid


# ---------------------------------------------------------------------------
# Inside a caller-held raw BEGIN, on_error="log" -- the same four cases.
#
# Each uses a single-child producer (only the one under test): with more than
# one child, an *earlier* child's own successful commit (see the module
# docstring) would flush the caller's raw write before the case under test
# even runs, confounding what each case is trying to isolate.
# ---------------------------------------------------------------------------


async def _write_raw_marker(store: SqliteEngravaCore, thought_id: str) -> None:
    """Insert a marker thought directly, bypassing the store's own commit.

    Calling one of the store's public write methods here would invoke its own
    ``_maybe_commit`` and end the caller's raw transaction early -- exactly
    what this helper avoids, mirroring the source thought already durable
    before the window in the suspend_auto_commit() tests above.
    """
    thought = _source(thought_id, "unrelated body")
    await store._db.execute(store._CORE_INSERT_SQL, store._thought_to_core_params(thought))


async def test_journal_append_failure_under_raw_begin_log_leaves_callers_write_pending_then_durable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child's row-journal failure under a caller-held raw BEGIN, "log".

    Step 1 (the row insert) is what fails here, so nothing has succeeded yet
    to trigger an early commit: X genuinely stays pending until the caller's
    own explicit commit, and is durable only afterward. The pending check
    below necessarily reads through the writing connection itself (a separate
    connection cannot see uncommitted work at all); the post-commit durability
    check reopens a fresh one.
    """
    db_path = tmp_path / "raw-begin.db"
    async with _connection(str(db_path)) as conn:
        source_id = "src-1"
        doomed = "doomed child"
        producer = ListProducer([_child(doomed)])
        store = await _open_store(conn, producer, on_error="log")
        await store.create_thought(_source(source_id, "Body."))
        doomed_id = _derived_thought_id(doomed)
        _patch_journal_to_fail(
            store, monkeypatch, mutation_type="INSERT_THOUGHT", target_id=doomed_id
        )

        await store._db.execute("BEGIN")
        await _write_raw_marker(store, "unrelated-src")  # X, pending

        result = await store.derive_existing(source_id)
        assert result.skipped == 1

        # X is still pending -- the caller's own transaction has not committed.
        assert store._db.in_transaction is True
        assert await store._get_thought_row("unrelated-src") is not None
        assert await store._get_thought_row(doomed_id) is None

        await store._db.commit()
        assert store._db.in_transaction is False

    async with _reopened(str(db_path)) as fresh:
        assert await fresh.get_thought("unrelated-src") is not None
        assert await fresh.get_thought(doomed_id) is None
        assert (await fresh.verify_journal()).valid


async def test_embedding_provider_failure_under_raw_begin_log_commits_callers_write_early(
    tmp_path: Path,
) -> None:
    """The embedding provider raises for the one child, under a raw BEGIN, "log".

    Here step 1 (the row insert) succeeds *before* the provider ever runs, so
    its own ``_maybe_commit()`` already commits the caller's raw write (see
    the module docstring) well before the provider raises. X is durable from
    that point on, not merely "pending" -- the caller's own final commit is a
    no-op by the time it runs.
    """
    db_path = tmp_path / "raw-begin.db"
    async with _connection(str(db_path)) as conn:
        source_id = "src-1"
        doomed = "doomed child"
        producer = ListProducer([_child(doomed)])
        store = await _open_store(
            conn, producer, on_error="log", embedding_provider=_failing_provider(doomed)
        )
        await store.create_thought(_source(source_id, "Body."))
        doomed_id = _derived_thought_id(doomed)

        await store._db.execute("BEGIN")
        await _write_raw_marker(store, "unrelated-src")  # X

        result = await store.derive_existing(source_id)
        assert result.skipped == 1

        # The row's own commit already flushed X and closed the transaction.
        assert store._db.in_transaction is False

        await store._db.commit()  # no-op: nothing left pending

    async with _reopened(str(db_path)) as fresh:
        assert await fresh.get_thought("unrelated-src") is not None
        assert await fresh.get_thought(doomed_id) is not None
        assert not await _has_embedding(fresh, doomed_id)
        assert (await fresh.verify_journal()).valid


async def test_store_embedding_failure_under_raw_begin_log_commits_callers_write_early(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``store_embedding`` fails after the provider succeeded, under a raw BEGIN, "log".

    Step 1 (the row insert) succeeds and commits, flushing X, before the embed
    step even starts; ``store_embedding``'s own unit then opens and unwinds a
    second, later transaction of its own for the failed INSERT -- the earlier,
    already-committed row and X are unaffected either way.
    """
    db_path = tmp_path / "raw-begin.db"
    async with _connection(str(db_path)) as conn:
        source_id = "src-1"
        doomed = "doomed child"
        producer = ListProducer([_child(doomed)])
        store = await _open_store(
            conn, producer, on_error="log", embedding_provider=_working_provider()
        )
        await store.create_thought(_source(source_id, "Body."))
        doomed_id = _derived_thought_id(doomed)
        _patch_embedding_insert_to_fail(store, monkeypatch, owner_id=doomed_id)

        await store._db.execute("BEGIN")
        await _write_raw_marker(store, "unrelated-src")  # X

        result = await store.derive_existing(source_id)
        assert result.skipped == 1

        # The row's own commit already flushed X; store_embedding's own failed
        # attempt opened and closed a further transaction of its own.
        assert store._db.in_transaction is False

        await store._db.commit()  # no-op: nothing left pending

    async with _reopened(str(db_path)) as fresh:
        assert await fresh.get_thought("unrelated-src") is not None
        assert await fresh.get_thought(doomed_id) is not None
        assert not await _embedding_row_exists(fresh._db, doomed_id)
        assert (await fresh.verify_journal()).valid


async def test_edge_journal_failure_under_raw_begin_log_commits_callers_write_early(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child's edge-journal append fails, under a raw BEGIN, "log".

    Step 1 (the row insert) succeeds and commits, flushing X, before the edge
    step even starts; the edge insert's own unit then opens and unwinds a
    second, later transaction of its own.
    """
    db_path = tmp_path / "raw-begin.db"
    async with _connection(str(db_path)) as conn:
        source_id = "src-1"
        doomed = "doomed child"
        producer = ListProducer([_child(doomed)])
        store = await _open_store(conn, producer, on_error="log")
        await store.create_thought(_source(source_id, "Body."))
        doomed_id = _derived_thought_id(doomed)
        doomed_edge_id = _derived_edge_id(doomed_id, source_id)
        _patch_journal_to_fail(
            store, monkeypatch, mutation_type="INSERT_EDGE", target_id=doomed_edge_id
        )

        await store._db.execute("BEGIN")
        await _write_raw_marker(store, "unrelated-src")  # X

        result = await store.derive_existing(source_id)
        assert result.skipped == 1

        assert store._db.in_transaction is False

        await store._db.commit()  # no-op: nothing left pending

    async with _reopened(str(db_path)) as fresh:
        assert await fresh.get_thought("unrelated-src") is not None
        # The row survives; only the edge (and its entry) is undone.
        assert await fresh.get_thought(doomed_id) is not None
        assert await fresh.get_edges(source_id, direction="IN") == []
        assert (await fresh.verify_journal()).valid


# ---------------------------------------------------------------------------
# on_error="raise" inside a window, caught by the caller
# ---------------------------------------------------------------------------


async def test_raise_in_window_caught_by_caller_keeps_the_windows_other_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caught inside the window, "raise" undoes only the failed step, not the window.

    Contrast with the existing, unchanged
    ``test_backfill_raise_in_suspend_window_rolls_back_caller_writes_source_survives``:
    there the error escapes the window uncaught, so the window's own atomicity
    rolls everything back. Here the caller catches it *inside* the window, so
    the window sees a clean exit and commits -- including the caller's other
    write and the child that succeeded before the failure.

    Deliberately a step-1 (row-journal) failure, not an identity collision:
    a collision raises before any database work for that child even starts,
    so it never touches the compensating-rollback / unit machinery this work
    item changes, and would not tell old and new code apart.
    """
    source_id = "src-1"
    doomed = "doomed child"
    db_path = tmp_path / "window.db"
    async with _connection(str(db_path)) as conn:
        producer = ListProducer([_child("first good"), _child(doomed), _child("second good")])
        store = await _open_store(conn, producer, on_error="raise")
        await store.create_thought(_source(source_id, "Body."))
        doomed_id = _derived_thought_id(doomed)
        _patch_journal_to_fail(
            store, monkeypatch, mutation_type="INSERT_THOUGHT", target_id=doomed_id
        )

        caught: RuntimeError | None = None
        async with store.suspend_auto_commit():
            await store.create_thought(_source("unrelated-src", "unrelated body"))  # X
            try:
                await store.derive_existing(source_id)
            except RuntimeError as exc:
                caught = exc

        assert caught is not None
        assert store._db.in_transaction is False

    async with _reopened(str(db_path)) as fresh:
        # X survives after the window exits (caught, so the window committed).
        assert await fresh.get_thought("unrelated-src") is not None
        # The failed step is undone per the table: for a step-1 failure, the
        # child itself (row and journal entry) is absent. The earlier good
        # child that already succeeded stays; the remaining one was never
        # attempted ("raise" aborts the dispatch on the first failure).
        assert await fresh.get_thought(doomed_id) is None
        assert await fresh.get_thought(_derived_thought_id("first good")) is not None
        assert await fresh.get_thought(_derived_thought_id("second good")) is None
        assert (await fresh.verify_journal()).valid


# ---------------------------------------------------------------------------
# Cancellation delivered during a child's step 1, inside a window
# ---------------------------------------------------------------------------


async def test_cancellation_during_step1_in_window_undoes_only_that_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancellation on a child's own row-journal append, caught by the caller.

    ``CancelledError`` propagates out of ``derive_existing`` regardless of
    ``on_error`` (it is not an ``on_error`` case). Caught here inside the
    window -- exactly like the "raise" case above -- only that child is
    undone and the window still exits cleanly, keeping the caller's other
    write (X) intact.
    """
    source_id = "src-1"
    doomed = "doomed child"
    db_path = tmp_path / "window.db"
    async with _connection(str(db_path)) as conn:
        producer = ListProducer([_child("first good"), _child(doomed), _child("second good")])
        store = await _open_store(conn, producer, on_error="log")
        await store.create_thought(_source(source_id, "Body."))
        doomed_id = _derived_thought_id(doomed)
        assert store.journal is not None
        original_append = store.journal.append

        async def _cancel_append(
            mutation_type: str,
            target_id: str | None,
            delta: dict[str, object],
        ) -> object:
            if mutation_type == "INSERT_THOUGHT" and target_id == doomed_id:
                raise asyncio.CancelledError
            return await original_append(
                mutation_type=mutation_type, target_id=target_id, delta=delta
            )

        monkeypatch.setattr(store.journal, "append", _cancel_append)

        cancelled = False
        async with store.suspend_auto_commit():
            await store.create_thought(_source("unrelated-src", "unrelated body"))  # X
            try:
                await store.derive_existing(source_id)
            except asyncio.CancelledError:
                cancelled = True

        assert cancelled, "CancelledError must propagate out of derive_existing unchanged"
        assert store._db.in_transaction is False

    async with _reopened(str(db_path)) as fresh:
        # X is intact: the caller caught the cancellation, so the window committed.
        assert await fresh.get_thought("unrelated-src") is not None
        # Only the cancelled child is undone -- nothing about "first good"
        # (already inserted before the cancellation) is touched.
        assert await fresh.get_thought(doomed_id) is None
        assert await fresh.get_thought(_derived_thought_id("first good")) is not None
        assert (await fresh.verify_journal()).valid
