"""Tests for the pre-insert preparation seam.

``prepare_thought_for_insert`` is the restored, uniform override point for
validation/enrichment that runs before the decisive duplicate probe, or --
when ``deduplicate=False`` skips that probe -- before the unconditional write
instead; that choice hinges on ``deduplicate``, not on which entry point was
called, since ``create_thought``, ``bulk_store``, and ``remember`` each
default to ``deduplicate=False``. Not before every probe, either:
``get_or_create`` / ``upsert_by_hash`` run their own exploratory probe first
and skip this seam entirely on a stable hit.
This module pins:

* the exact per-entry-point invocation count: ``create_thought`` always once;
  ``get_or_create`` / ``upsert_by_hash`` zero on a stable hit, once on a
  miss, and still exactly once when a race turns the miss into a hit;
  ``remember`` once; ``bulk_store`` once per item, including a dedup hit
  inside the batch, run for the whole batch holding no lock this call
  itself acquires;
* that the returned record is revalidated and its ``content`` is decisive for
  the probe that follows;
* that a rejection leaves no row and no journal entry, on every direct path
  and mid-batch in ``bulk_store``;
* that the seam runs holding no lock this call itself acquires -- proven
  from the outside, for the SQLite-level transaction lock, with a second
  real connection; proven in-process, for ``_write_lock``, by checking the
  lock object directly and by confirming a different task's write is not
  blocked (``bulk_store``'s own seam phase) -- and with same-task /
  spawned-task recursive
  callbacks bounded by ``asyncio.wait_for`` -- including the
  otherwise-empty ``suspend_auto_commit()`` case, where an earlier ownership
  test on the exploratory probe's transaction left it open across the seam
  (a lock an *enclosing* caller's own task took is a different matter, not
  one this call itself acquired -- see ``docs/concurrency.md``'s nesting
  note); and
* the transaction-ownership design decision: the exploratory probe's own
  ``BEGIN IMMEDIATE`` is closed before the seam runs only when this call
  opened it -- regardless of whether it is nested inside a caller's own
  ``suspend_auto_commit`` window -- never a transaction the caller's own
  earlier write already opened; and
* that a same-task callback into a dedup entry point from *inside*
  ``update_thought`` -- reached from ``upsert_by_hash``'s hit branch while
  ``_write_lock``, ``_dedup_lock`` and the transaction the probe opened,
  when it opened one, are all three still held -- raises
  ``DedupLockReentryError`` instead of
  hanging forever on ``_dedup_lock``, on both the exploratory-probe-hit
  route and the decisive-probe-hit route the seam can newly reach by
  transforming a miss into a hit.

Every test here is expected to fail against the source before this seam
existed, where ``prepare_thought_for_insert`` is never called from any
entry point.
"""

from __future__ import annotations

import asyncio
import sqlite3
from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import (
    CoreThoughtRecord,
    DedupLockReentryError,
    KnowledgeSource,
    LifecycleStatus,
    Priority,
    SqliteEngravaCore,
    ThoughtType,
    ThoughtVisibility,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable
    from pathlib import Path

    from engrava.domain.models.thought import ThoughtRecord

    OnPrepare = Callable[["_SeamHookCore", ThoughtRecord], Awaitable[ThoughtRecord]]
    OnUpdate = Callable[["_SeamHookCore", str, dict[str, object]], Awaitable[None]]


def _thought(thought_id: str, *, content: str, **overrides: object) -> CoreThoughtRecord:
    """Build a realistic ``CoreThoughtRecord`` for seam tests."""
    fields: dict[str, object] = {
        "thought_id": thought_id,
        "thought_type": ThoughtType.OBSERVATION,
        "essence": content[:60],
        "content": content,
        "priority": Priority.P2,
        "lifecycle_status": LifecycleStatus.CREATED,
        "created_cycle": 0,
        "updated_cycle": 0,
        "source": "test-suite",
        "confidence": 0.9,
        "source_type": KnowledgeSource.EXPERIENCE,
        "visibility": ThoughtVisibility.SELECTIVE,
    }
    fields.update(overrides)
    return CoreThoughtRecord(**fields)  # type: ignore[arg-type]


class _SeamHookCore(SqliteEngravaCore):
    """Instrumented ``SqliteEngravaCore`` whose seam records calls and can be scripted.

    ``prepare_calls`` records every candidate this instance's seam was called
    with, in order -- the invocation-count assertions below read its length.
    An optional ``on_prepare`` async callback lets a test transform the
    record, inject a competing write, or recurse into another entry point,
    exactly like a real subclass override would. An optional ``on_update``
    async callback does the same for the separately-overridable
    ``update_thought`` -- the call ``upsert_by_hash``'s hit branch makes while
    still holding ``_write_lock``, ``_dedup_lock`` and the transaction the
    probe opened, when it opened one (see ``docs/extension-hooks.md`` §1B.3).
    """

    def __init__(
        self,
        *args: object,
        on_prepare: OnPrepare | None = None,
        on_update: OnUpdate | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.prepare_calls: list[ThoughtRecord] = []
        self._on_prepare = on_prepare
        self._on_update = on_update

    async def prepare_thought_for_insert(self, thought: ThoughtRecord) -> ThoughtRecord:
        self.prepare_calls.append(thought)
        if self._on_prepare is not None:
            return await self._on_prepare(self, thought)
        return thought

    async def update_thought(self, thought_id: str, **changes: object) -> ThoughtRecord:
        if self._on_update is not None:
            await self._on_update(self, thought_id, changes)
        return await super().update_thought(thought_id, **changes)


@pytest.fixture
async def db() -> AsyncIterator[aiosqlite.Connection]:
    """Fresh in-memory SQLite with the core schema bootstrapped."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode = WAL")
    await conn.execute("PRAGMA foreign_keys = ON")
    bootstrap = SqliteEngravaCore(conn)
    await bootstrap.ensure_schema()
    yield conn
    await conn.close()


async def _make_store(
    db: aiosqlite.Connection,
    *,
    on_prepare: OnPrepare | None = None,
    on_update: OnUpdate | None = None,
    journal_enabled: bool = False,
) -> _SeamHookCore:
    store = _SeamHookCore(
        db,
        on_prepare=on_prepare,
        on_update=on_update,
        journal_enabled=journal_enabled,
    )
    await store._probe_fts()
    return store


async def _count(db: aiosqlite.Connection, sql: str, *params: object) -> int:
    cursor = await db.execute(sql, params)
    row = await cursor.fetchone()
    assert row is not None
    return int(row[0])


@pytest.fixture
async def db_path(tmp_path: Path) -> str:
    """A real on-disk database file -- needed for real second-connection probes."""
    path = str(tmp_path / "seam.db")
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
    on_prepare: OnPrepare | None = None,
) -> tuple[aiosqlite.Connection, _SeamHookCore]:
    conn = await aiosqlite.connect(db_path)
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA foreign_keys=ON")
    store = _SeamHookCore(conn, on_prepare=on_prepare)
    await store._probe_fts()
    return conn, store


# ---------------------------------------------------------------------------
# create_thought -- always exactly once, hit or miss, either dedup mode
# ---------------------------------------------------------------------------


async def test_create_thought_plain_calls_seam_once_per_call(db: aiosqlite.Connection) -> None:
    store = await _make_store(db)
    for i in range(3):
        await store.create_thought(_thought(f"t-{i}", content=f"distinct {i}"))
    assert len(store.prepare_calls) == 3


async def test_create_thought_dedup_true_calls_seam_once_on_miss_and_once_on_hit(
    db: aiosqlite.Connection,
) -> None:
    """Unlike ``get_or_create`` / ``upsert_by_hash``, a dedup *hit* still costs one call.

    A direct ``create_thought()`` call cannot know in advance whether it will
    hit or miss -- restoring the pre-regression override-everywhere behaviour
    means the seam runs once per call regardless of which branch it resolves
    to.
    """
    store = await _make_store(db)
    content = "Same content, seen twice."
    await store.create_thought(_thought("t-1", content=content), deduplicate=True)
    await store.create_thought(_thought("t-2", content=content), deduplicate=True)
    assert len(store.prepare_calls) == 2


# ---------------------------------------------------------------------------
# get_or_create / upsert_by_hash -- zero on a stable hit, once on a miss
# ---------------------------------------------------------------------------


async def test_get_or_create_zero_seam_calls_on_stable_hit(db: aiosqlite.Connection) -> None:
    store = await _make_store(db)
    content = "Pre-existing content."
    await store.create_thought(_thought("t-existing", content=content))
    store.prepare_calls.clear()

    record, created = await store.get_or_create(_thought("t-other-id", content=content))

    assert created is False
    assert record.thought_id == "t-existing"
    assert store.prepare_calls == []


async def test_get_or_create_one_seam_call_on_miss(db: aiosqlite.Connection) -> None:
    store = await _make_store(db)
    record, created = await store.get_or_create(_thought("t-new", content="Never seen before."))
    assert created is True
    assert record.thought_id == "t-new"
    assert len(store.prepare_calls) == 1


async def test_upsert_by_hash_zero_seam_calls_on_stable_hit(db: aiosqlite.Connection) -> None:
    store = await _make_store(db)
    content = "Pre-existing content for upsert."
    await store.create_thought(_thought("t-existing", content=content))
    store.prepare_calls.clear()

    result = await store.upsert_by_hash(
        _thought("t-other-id", content=content, priority=Priority.P1),
    )

    assert result.thought_id == "t-existing"
    assert store.prepare_calls == []


async def test_upsert_by_hash_one_seam_call_on_miss(db: aiosqlite.Connection) -> None:
    store = await _make_store(db)
    result = await store.upsert_by_hash(_thought("t-new", content="Also never seen."))
    assert result.thought_id == "t-new"
    assert len(store.prepare_calls) == 1


# ---------------------------------------------------------------------------
# bulk_store / remember -- pinned invocation counts
# ---------------------------------------------------------------------------


async def test_remember_calls_seam_once(db: aiosqlite.Connection) -> None:
    store = await _make_store(db)
    record = await store.remember("plain text via remember()")
    assert record.content == "plain text via remember()"
    assert len(store.prepare_calls) == 1


async def test_bulk_store_calls_seam_once_per_item_including_a_dedup_hit(
    db: aiosqlite.Connection,
) -> None:
    """Once per item, not "once per genuinely-inserted row".

    A dedup hit inside the batch still costs one seam call -- the seam runs
    for the whole batch, item by item, before ``bulk_store`` knows which
    items will resolve to an insert and which will resolve to a
    confirmation bump, exactly like a direct ``create_thought(deduplicate=True)``
    call (see ``test_create_thought_dedup_true_calls_seam_once_on_miss_and_once_on_hit``).
    """
    store = await _make_store(db)
    content = "Repeated within the same batch."
    batch = [
        _thought("b-1", content=content),
        _thought("b-2", content=content),  # dedup-hits b-1 inside the batch
        _thought("b-3", content="Distinct."),
    ]
    persisted = await store.bulk_store(batch, deduplicate=True)

    assert [t.thought_id for t in persisted] == ["b-1", "b-1", "b-3"]
    assert len(store.prepare_calls) == 3
    assert await _count(db, "SELECT COUNT(*) FROM thought") == 2


async def test_bulk_store_seam_runs_for_the_whole_batch_before_any_lock(
    db: aiosqlite.Connection,
) -> None:
    """The seam call for every item completes before ``bulk_store`` inserts any of them.

    Proves the two-phase restructuring directly: pausing the seam on the
    *last* item and checking the table from inside that pause shows zero
    rows -- nothing from the earlier items has been inserted yet, because
    phase 1 (the seam, over the whole batch) runs entirely before phase 2
    (the locked insert loop) begins.
    """
    seam_entered_for_last = asyncio.Event()
    release_seam = asyncio.Event()

    async def pause_on_last(store: _SeamHookCore, thought: ThoughtRecord) -> ThoughtRecord:
        if thought.thought_id == "p-3":
            seam_entered_for_last.set()
            await release_seam.wait()
        return thought

    store = await _make_store(db, on_prepare=pause_on_last)
    batch = [_thought(f"p-{i}", content=f"phase-1 item {i}") for i in range(1, 4)]

    task = asyncio.create_task(store.bulk_store(batch))
    try:
        await asyncio.wait_for(seam_entered_for_last.wait(), timeout=5)
        assert len(store.prepare_calls) == 3
        assert await _count(db, "SELECT COUNT(*) FROM thought") == 0
    finally:
        release_seam.set()

    persisted = await asyncio.wait_for(task, timeout=5)
    assert [t.thought_id for t in persisted] == ["p-1", "p-2", "p-3"]
    assert await _count(db, "SELECT COUNT(*) FROM thought") == 3


async def test_bulk_store_seam_phase_holds_no_write_lock(db: aiosqlite.Connection) -> None:
    """A different task's own write is not blocked by bulk_store's seam phase.

    Before the two-phase restructuring, ``bulk_store`` took ``_write_lock``
    for its whole ``suspend_auto_commit()`` window and only then ran the
    seam per item, so a different task's guarded write blocked (and, with a
    short lock-acquire timeout, failed with ``WriteLockTimeoutError``) for as
    long as the seam paused on any item. With the seam run holding no lock
    this call itself acquires, before
    ``bulk_store`` ever takes that lock, a concurrent write from a different
    task must complete instead.
    """
    seam_entered_for_second = asyncio.Event()
    release_seam = asyncio.Event()

    async def pause_on_second(store: _SeamHookCore, thought: ThoughtRecord) -> ThoughtRecord:
        if thought.thought_id == "c-2":
            seam_entered_for_second.set()
            await release_seam.wait()
        return thought

    store = await _make_store(db, on_prepare=pause_on_second)
    store._write_lock._acquire_timeout_seconds = 2.0

    batch = [_thought(f"c-{i}", content=f"concurrent-write item {i}") for i in range(1, 4)]
    batch_task = asyncio.create_task(store.bulk_store(batch))
    try:
        await asyncio.wait_for(seam_entered_for_second.wait(), timeout=5)
        other = await asyncio.wait_for(
            store.create_thought(_thought("other", content="a different, concurrent write")),
            timeout=5,
        )
        assert other.thought_id == "other"
    finally:
        release_seam.set()

    persisted = await asyncio.wait_for(batch_task, timeout=5)
    assert [t.thought_id for t in persisted] == ["c-1", "c-2", "c-3"]
    assert {t.thought_id for t in store.prepare_calls} == {"c-1", "c-2", "c-3", "other"}
    assert len(store.prepare_calls) == 4


# ---------------------------------------------------------------------------
# Race: a second writer wins between the two probes -- seam still ran once
# ---------------------------------------------------------------------------


async def test_get_or_create_race_second_writer_wins_seam_runs_exactly_once(
    db: aiosqlite.Connection,
) -> None:
    """A competing insert lands between the exploratory and decisive probes.

    The seam simulates "another writer" by inserting the colliding content
    directly (bypassing the seam+dedup wrapper) the first time it is called --
    exactly what a genuinely concurrent second caller landing in that window
    would do. The decisive probe run after the seam must then see a hit,
    take the confirmation-bump branch instead of inserting, and the seam
    itself must not run a second time just because the outcome changed.
    """
    content = "Raced content."

    async def inject_competing_write(store: _SeamHookCore, thought: ThoughtRecord) -> ThoughtRecord:
        if len(store.prepare_calls) == 1:
            await store._insert_new_thought_row(
                _thought("t-competitor", content=content),
                expires_after_seconds=None,
            )
        return thought

    store = await _make_store(db, on_prepare=inject_competing_write)
    record, created = await store.get_or_create(_thought("t-mine", content=content))

    assert created is False
    assert record.thought_id == "t-competitor"
    assert record.confirmation_count == 1
    assert len(store.prepare_calls) == 1
    assert await _count(db, "SELECT COUNT(*) FROM thought") == 1


async def test_upsert_by_hash_race_second_writer_wins_seam_runs_exactly_once(
    db: aiosqlite.Connection,
) -> None:
    content = "Raced upsert content."

    async def inject_competing_write(store: _SeamHookCore, thought: ThoughtRecord) -> ThoughtRecord:
        if len(store.prepare_calls) == 1:
            await store._insert_new_thought_row(
                _thought("t-competitor", content=content, priority=Priority.P3),
                expires_after_seconds=None,
            )
        return thought

    store = await _make_store(db, on_prepare=inject_competing_write)
    result = await store.upsert_by_hash(
        _thought("t-mine", content=content, priority=Priority.P1),
    )

    assert result.thought_id == "t-competitor"
    assert result.priority == Priority.P1  # the race's mutable-field diff was still applied
    assert len(store.prepare_calls) == 1
    assert await _count(db, "SELECT COUNT(*) FROM thought") == 1


# ---------------------------------------------------------------------------
# The transformed record is revalidated and its content is decisive
# ---------------------------------------------------------------------------


async def test_seam_transformed_content_supplies_the_decisive_hash(
    db: aiosqlite.Connection,
) -> None:
    """Two calls with different original content collapse if the seam unifies them."""

    async def normalize(store: _SeamHookCore, thought: ThoughtRecord) -> ThoughtRecord:
        return thought.model_copy(update={"content": "normalized"})

    store = await _make_store(db, on_prepare=normalize)
    first, created_first = await store.get_or_create(_thought("t-a", content="original A"))
    second, created_second = await store.get_or_create(_thought("t-b", content="original B"))

    assert created_first is True
    assert created_second is False
    assert second.thought_id == first.thought_id
    assert await _count(db, "SELECT COUNT(*) FROM thought") == 1
    rows = list(
        await db.execute_fetchall(
            "SELECT content FROM thought WHERE thought_id = ?", (first.thought_id,)
        )
    )
    assert rows[0]["content"] == "normalized"


async def test_seam_returning_invalid_metadata_raises_before_insert(
    db: aiosqlite.Connection,
) -> None:
    """Constraint 2: the *returned* record is revalidated, not just the original."""

    async def poison_metadata(store: _SeamHookCore, thought: ThoughtRecord) -> ThoughtRecord:
        # A metadata value of an unsupported type violates `_validate_metadata`.
        return thought.model_copy(update={"metadata": {"bad": object()}})

    store = await _make_store(db, on_prepare=poison_metadata)
    with pytest.raises((TypeError, ValueError)):
        await store.get_or_create(_thought("t-poisoned", content="will be rejected"))

    assert await _count(db, "SELECT COUNT(*) FROM thought") == 0


async def test_seam_raising_leaves_no_row_and_no_journal_entry(
    db: aiosqlite.Connection,
) -> None:
    """Constraint 5: a pre-insert rejection leaves no trace, on every path."""

    async def reject(store: _SeamHookCore, thought: ThoughtRecord) -> ThoughtRecord:
        msg = "rejected by policy"
        raise ValueError(msg)

    store = await _make_store(db, on_prepare=reject, journal_enabled=True)

    with pytest.raises(ValueError, match="rejected by policy"):
        await store.create_thought(_thought("t-rejected", content="never lands"))
    with pytest.raises(ValueError, match="rejected by policy"):
        await store.get_or_create(_thought("t-rejected-2", content="never lands either"))
    with pytest.raises(ValueError, match="rejected by policy"):
        await store.upsert_by_hash(_thought("t-rejected-3", content="nor this"))

    assert await _count(db, "SELECT COUNT(*) FROM thought") == 0
    assert await _count(db, "SELECT COUNT(*) FROM journal_entry") == 0


async def test_seam_raising_mid_batch_leaves_no_row_and_no_journal_entry(
    db: aiosqlite.Connection,
) -> None:
    """Constraint 5 on ``bulk_store``: a rejection anywhere in the batch leaves no trace.

    The failing item is third of four -- the first two items' seam calls
    already succeeded (phase 1 has no way to know item 3 will fail until it
    tries), but because phase 1 runs entirely before phase 2 ever takes the
    lock or opens a transaction, nothing from items 1-2 is ever inserted:
    the same all-or-nothing outcome ``bulk_store`` has always produced when
    an item fails, preserved by construction rather than by a rollback.
    """

    async def reject_third(store: _SeamHookCore, thought: ThoughtRecord) -> ThoughtRecord:
        if thought.thought_id == "m-3":
            msg = "rejected by policy"
            raise ValueError(msg)
        return thought

    store = await _make_store(db, on_prepare=reject_third, journal_enabled=True)
    batch = [_thought(f"m-{i}", content=f"mid-batch item {i}") for i in range(1, 5)]

    with pytest.raises(ValueError, match="rejected by policy"):
        await store.bulk_store(batch)

    assert [t.thought_id for t in store.prepare_calls] == ["m-1", "m-2", "m-3"]
    assert await _count(db, "SELECT COUNT(*) FROM thought") == 0
    assert await _count(db, "SELECT COUNT(*) FROM journal_entry") == 0


# ---------------------------------------------------------------------------
# No lock this call itself acquires is held while the seam runs, at the top
# level -- the SQLite-level transaction lock, proven from the outside; the
# in-process _write_lock, proven in-process (a second connection cannot
# observe an asyncio.Lock in this process).
# ---------------------------------------------------------------------------


async def test_exploratory_probe_lock_is_released_before_seam_runs(db_path: str) -> None:
    """A second connection can open BEGIN IMMEDIATE, and ``_write_lock`` is unlocked, mid-seam.

    Two independent checks, because they observe different things: a second,
    independent connection can only see the SQLite-level transaction lock, not
    this process's own ``asyncio.Lock``, so it is decisive for the former and
    silent on the latter. An uncommitted / still-open transaction on *this*
    connection would not stop a second connection's own ``BEGIN IMMEDIATE``
    request from queueing (masking the bug); with ``busy_timeout=0`` on the
    checker it either succeeds immediately (lock genuinely free) or fails
    immediately (still held) -- no ambiguity. ``_write_lock`` itself is
    checked directly, in-process, alongside it.
    """
    seam_entered = asyncio.Event()
    release_seam = asyncio.Event()

    async def pause(store: _SeamHookCore, thought: ThoughtRecord) -> ThoughtRecord:
        seam_entered.set()
        await release_seam.wait()
        return thought

    conn, store = await _open_store(db_path, on_prepare=pause)
    checker_conn = await aiosqlite.connect(db_path)
    try:
        await checker_conn.execute("PRAGMA busy_timeout=0")

        task = asyncio.create_task(
            store.get_or_create(_thought("t-paused", content="seam is paused mid-flight")),
        )
        try:
            await asyncio.wait_for(seam_entered.wait(), timeout=5)
            assert store._write_lock._lock.locked() is False, (
                "_write_lock is still held in-process while the pre-insert seam is "
                "running, at the top level -- it should have been released before "
                "the seam was ever called"
            )
            try:
                await checker_conn.execute("BEGIN IMMEDIATE")
            except sqlite3.OperationalError as exc:
                pytest.fail(
                    "the SQLite-level transaction lock (BEGIN IMMEDIATE) is still "
                    f"held while the pre-insert seam is running: {exc}",
                )
            else:
                await checker_conn.rollback()
        finally:
            release_seam.set()

        record, created = await asyncio.wait_for(task, timeout=5)
        assert created is True
        assert record.thought_id == "t-paused"
    finally:
        await checker_conn.close()
        await conn.close()


async def test_otherwise_empty_suspend_auto_commit_window_releases_probe_lock_before_seam_runs(
    db_path: str,
) -> None:
    """Bug: ``_end_exploratory_probe`` used to also test ``not self._skip_auto_commit``.

    Nesting ``get_or_create()`` inside a caller's *own*, otherwise-empty
    ``suspend_auto_commit()`` window (nothing written yet) still opens the
    exploratory probe's own ``BEGIN IMMEDIATE`` -- ``opened_transaction`` is
    ``True`` because nothing was open before this probe began. The old
    ownership test also required ``not self._skip_auto_commit``, which is
    ``False`` here (we *are* nested), so it skipped the rollback and left
    that ``BEGIN IMMEDIATE`` -- a cross-connection write reservation -- open
    across the seam call. A second, independent connection's own
    ``BEGIN IMMEDIATE`` (``busy_timeout=0``, so it fails immediately rather
    than queueing and masking the bug) must succeed while the seam is
    paused, exactly as it does at the top level in the test above.
    """
    seam_entered = asyncio.Event()
    release_seam = asyncio.Event()

    async def pause(store: _SeamHookCore, thought: ThoughtRecord) -> ThoughtRecord:
        seam_entered.set()
        await release_seam.wait()
        return thought

    conn, store = await _open_store(db_path, on_prepare=pause)
    checker_conn = await aiosqlite.connect(db_path)
    try:
        await checker_conn.execute("PRAGMA busy_timeout=0")

        async def run_nested() -> tuple[ThoughtRecord, bool]:
            async with store.suspend_auto_commit():
                return await store.get_or_create(
                    _thought("t-nested", content="seam paused inside an empty window"),
                )

        task = asyncio.create_task(run_nested())
        try:
            await asyncio.wait_for(seam_entered.wait(), timeout=5)
            try:
                await checker_conn.execute("BEGIN IMMEDIATE")
            except sqlite3.OperationalError as exc:
                pytest.fail(
                    "the exploratory probe's own BEGIN IMMEDIATE is still held across "
                    "the seam inside an otherwise-empty suspend_auto_commit() window "
                    f"(opened_transaction was True but _skip_auto_commit suppressed "
                    f"the rollback): {exc}",
                )
            else:
                await checker_conn.rollback()
        finally:
            release_seam.set()

        record, created = await asyncio.wait_for(task, timeout=5)
        assert created is True
        assert record.thought_id == "t-nested"
    finally:
        await checker_conn.close()
        await conn.close()


# ---------------------------------------------------------------------------
# Deadlock probes -- same-task and spawned-task callbacks, both bounded
# ---------------------------------------------------------------------------


async def test_same_task_callback_into_dedup_entry_point_does_not_deadlock(
    db: aiosqlite.Connection,
) -> None:
    """A hook that calls back into another dedup entry point, on the same task.

    Before this fix, restoring a call to the public ``create_thought()`` from
    inside an already-locked dedup window ran the equivalent recursive call
    while still holding the plain, non-reentrant ``_dedup_lock`` -- a same-task
    callback would then wait forever on a lock it already holds. The seam
    never runs with ``_dedup_lock`` held, so this must complete quickly.
    """

    async def callback(store: _SeamHookCore, thought: ThoughtRecord) -> ThoughtRecord:
        if thought.thought_id == "t-outer":
            await store.get_or_create(_thought("t-inner", content="from the same task"))
        return thought

    store = await _make_store(db, on_prepare=callback)

    record, created = await asyncio.wait_for(
        store.get_or_create(_thought("t-outer", content="triggers a same-task callback")),
        timeout=5,
    )
    assert created is True
    assert record.thought_id == "t-outer"
    assert await _count(db, "SELECT COUNT(*) FROM thought") == 2


async def test_spawned_task_callback_into_dedup_entry_point_does_not_deadlock(
    db: aiosqlite.Connection,
) -> None:
    """A hook that spawns a task and awaits it, at the top level (no enclosing window).

    At the top level (this call did not nest inside a caller's own
    ``suspend_auto_commit``), ``_write_lock``'s real underlying lock is
    genuinely released before the seam runs, so a spawned, different task can
    acquire it too. The acquire timeout is shrunk so a regression fails fast
    (``WriteLockTimeoutError``) instead of hanging for the production default;
    ``asyncio.wait_for`` is a second, independent bound around the whole call.
    """
    store = await _make_store(db)
    store._write_lock._acquire_timeout_seconds = 2.0

    async def callback(inner_store: _SeamHookCore, thought: ThoughtRecord) -> ThoughtRecord:
        if thought.thought_id == "t-outer":
            child = asyncio.create_task(
                inner_store.get_or_create(_thought("t-child", content="from a spawned task")),
            )
            await child
        return thought

    store._on_prepare = callback

    record, created = await asyncio.wait_for(
        store.get_or_create(_thought("t-outer", content="spawns and awaits a child task")),
        timeout=10,
    )
    assert created is True
    assert record.thought_id == "t-outer"
    assert await _count(db, "SELECT COUNT(*) FROM thought") == 2


async def test_update_thought_same_task_callback_on_exploratory_hit_raises(
    db: aiosqlite.Connection,
) -> None:
    """``update_thought`` runs under ``_write_lock``, ``_dedup_lock`` and the
    transaction the probe opened, when it opened one, on
    ``upsert_by_hash``'s hit branch (see
    ``docs/extension-hooks.md`` §1B.3) -- this predates the seam, and is not
    something the seam introduces or could close. An ``update_thought``
    override that calls back into another dedup entry point on the *same*
    task tries to acquire ``_dedup_lock`` a second time; a plain
    ``asyncio.Lock`` would block that acquisition on itself forever, since
    only this same task's own outer call can ever release it.
    ``asyncio.wait_for`` bounds the probe so a regression hangs the test
    instead of the whole suite.
    """
    entered = asyncio.Event()

    async def on_update(
        inner_store: _SeamHookCore,
        thought_id: str,
        changes: dict[str, object],
    ) -> None:
        del thought_id, changes
        entered.set()
        await inner_store.get_or_create(
            _thought("u-inner", content="from an update_thought callback"),
        )

    store = await _make_store(db, on_update=on_update)
    await store.create_thought(
        _thought("u-existing", content="matched by hash", priority=Priority.P2),
    )

    with pytest.raises(DedupLockReentryError):
        await asyncio.wait_for(
            store.upsert_by_hash(
                _thought("u-new", content="matched by hash", priority=Priority.P1),
            ),
            timeout=5,
        )
    assert entered.is_set()


async def test_update_thought_same_task_callback_on_decisive_hit_raises(
    db: aiosqlite.Connection,
) -> None:
    """The decisive-hit route to ``update_thought`` is newly *reachable*, not new in shape.

    The seam can turn an exploratory *miss* into a decisive *hit* by
    transforming the candidate's ``content`` to match an existing row --
    deterministically, with no race required, unlike before this seam
    existed. That route reaches the same ``_upsert_matched_row`` ->
    ``update_thought`` call, under the same three guards
    (``_write_lock``, ``_dedup_lock``, the transaction the decisive probe
    opened, when it opened one), as the exploratory-hit case above -- pinned separately
    because the path to it is different.
    """
    entered = asyncio.Event()

    async def on_update(
        inner_store: _SeamHookCore,
        thought_id: str,
        changes: dict[str, object],
    ) -> None:
        del thought_id, changes
        entered.set()
        await inner_store.get_or_create(
            _thought("d-inner", content="from an update_thought callback"),
        )

    async def collide_on_decisive_probe(
        inner_store: _SeamHookCore,
        thought: ThoughtRecord,
    ) -> ThoughtRecord:
        if thought.thought_id == "d-outer":
            return thought.model_copy(update={"content": "matched only after the seam runs"})
        return thought

    store = await _make_store(db, on_prepare=collide_on_decisive_probe, on_update=on_update)
    await store.create_thought(
        _thought(
            "d-existing",
            content="matched only after the seam runs",
            priority=Priority.P2,
        ),
    )

    with pytest.raises(DedupLockReentryError):
        await asyncio.wait_for(
            store.upsert_by_hash(
                _thought("d-outer", content="does not match anything yet", priority=Priority.P1),
            ),
            timeout=5,
        )
    assert entered.is_set()


# ---------------------------------------------------------------------------
# Transaction ownership: never close what the caller's own window owns
# ---------------------------------------------------------------------------


async def test_nested_in_suspend_auto_commit_does_not_discard_the_callers_pending_write(
    db: aiosqlite.Connection,
) -> None:
    """The transaction-ownership decision most likely to be silently wrong.

    A plain ``create_thought`` inside a caller-held ``suspend_auto_commit()``
    window leaves its INSERT pending in an open transaction it does not
    commit (``_maybe_commit`` is a no-op there). A ``get_or_create`` miss
    issued afterward, on the *same* task, must not close that already-open
    transaction while releasing its own exploratory-probe window before
    running the seam -- doing so would silently discard the first write. Both
    rows must still be present once the window commits.
    """
    store = await _make_store(db)

    async with store.suspend_auto_commit():
        pending = await store.create_thought(_thought("t-pending", content="written first"))
        record, created = await store.get_or_create(
            _thought("t-second", content="written second, via the seam"),
        )

    assert created is True
    # One seam call for the plain create_thought("t-pending"), one for
    # get_or_create("t-second")'s miss -- neither wiped out by the other.
    assert [t.thought_id for t in store.prepare_calls] == ["t-pending", "t-second"]
    ids = {row["thought_id"] for row in await db.execute_fetchall("SELECT thought_id FROM thought")}
    assert ids == {pending.thought_id, record.thought_id} == {"t-pending", "t-second"}


async def test_upsert_by_hash_decisive_hit_leaves_the_callers_pending_writes_alone(
    db: aiosqlite.Connection,
) -> None:
    """A decisive no-change hit must not roll back writes it did not open.

    Inside the caller's own ``suspend_auto_commit()`` window nothing is open
    when the exploratory probe samples the connection, so that probe's
    ``opened_transaction`` is ``True``. The seam then leaves a transaction
    open with pending writes (a ``marker`` row of the caller's own, plus a
    ``same`` row whose content equals the candidate's, so the decisive probe
    hits with no field to update). The decisive probe finds that transaction
    already open, so *it* opened nothing and must end nothing: the first
    probe's flag describes the connection as it was then, not now. Both rows
    must survive the window's commit.
    """
    state = {"armed": False}

    async def leave_pending_writes(store: _SeamHookCore, thought: ThoughtRecord) -> ThoughtRecord:
        if state["armed"]:
            state["armed"] = False
            await SqliteEngravaCore.create_thought(
                store, _thought("marker", content="the caller's own pending write")
            )
            await SqliteEngravaCore.create_thought(store, _thought("same", content=thought.content))
        return thought

    store = await _make_store(db, on_prepare=leave_pending_writes)

    async with store.suspend_auto_commit():
        state["armed"] = True
        result = await store.upsert_by_hash(
            _thought("cand", content="candidate content the seam collides with"),
        )

    assert result.thought_id == "same"
    ids = {row["thought_id"] for row in await db.execute_fetchall("SELECT thought_id FROM thought")}
    assert ids == {"marker", "same"}


async def test_upsert_by_hash_decisive_hit_ends_the_transaction_its_own_probe_opened(
    db: aiosqlite.Connection,
) -> None:
    """The case the ownership flag exists for: the decisive probe opened the transaction.

    With no enclosing window, another writer lands the same content between
    the two probes and commits (no transaction is left open). The decisive
    probe then opens its own ``BEGIN IMMEDIATE``, matches, finds no mutable
    field to change and writes nothing, so it must end that transaction
    itself -- leaving it open would hold the cross-connection write
    reservation after the call returned.
    """
    content = "Raced content with nothing to update."

    async def inject_competing_write(store: _SeamHookCore, thought: ThoughtRecord) -> ThoughtRecord:
        if len(store.prepare_calls) == 1:
            await store._insert_new_thought_row(
                _thought("t-competitor", content=content),
                expires_after_seconds=None,
            )
        return thought

    store = await _make_store(db, on_prepare=inject_competing_write)
    result = await store.upsert_by_hash(_thought("t-mine", content=content))

    assert result.thought_id == "t-competitor"
    assert len(store.prepare_calls) == 1
    assert store._db.in_transaction is False
