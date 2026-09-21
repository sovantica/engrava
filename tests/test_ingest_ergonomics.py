"""Tests for the additive batch / get-or-create ingest primitives.

Covers three public write-API additions layered over the existing
content-hash deduplication of ``SqliteEngravaCore``:

* ``get_or_create`` — dedup with a ``(record, created)`` return that
  removes the caller's check-then-create round trip.
* ``upsert_by_hash`` — update-on-match semantics, distinct from
  ``create_thought(deduplicate=True)`` (which only bumps confirmation).
* ``bulk_store`` — transactional batch insert under a single commit,
  with a single batch-embed call when auto-embed is active.

The style mirrors ``test_ingest_deduplication.py``: one focused case per
behavioural axis so a regression localises cleanly. Behaviour-preservation
of the existing ``create_thought`` / ``deduplicate=True`` path is asserted
explicitly.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import struct
from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import (
    CoreThoughtRecord,
    DefaultEngravaHooks,
    EmbeddingGenerationError,
    KnowledgeSource,
    LifecycleStatus,
    Priority,
    SqliteEngravaCore,
    ThoughtType,
    ThoughtVisibility,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from engrava.domain.models.thought import ThoughtRecord


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


@pytest.fixture
async def db() -> AsyncIterator[aiosqlite.Connection]:
    """Fresh in-memory SQLite with the core schema bootstrapped."""
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
    """Reusable ``SqliteEngravaCore`` bound to the in-memory DB (no embed)."""
    s = SqliteEngravaCore(db)
    await s._probe_fts()
    return s


def _thought(
    thought_id: str,
    *,
    content: str = "The user prefers concise explanations over verbose ones.",
    thought_type: ThoughtType = ThoughtType.OBSERVATION,
    essence: str = "User preference for concision",
    priority: Priority = Priority.P2,
    metadata: dict[str, object] | None = None,
) -> CoreThoughtRecord:
    """Build a realistic ``CoreThoughtRecord`` for ingest tests."""
    return CoreThoughtRecord(
        thought_id=thought_id,
        thought_type=thought_type,
        essence=essence,
        content=content,
        priority=priority,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=0,
        updated_cycle=0,
        source="test-suite",
        confidence=0.9,
        source_type=KnowledgeSource.EXPERIENCE,
        visibility=ThoughtVisibility.SELECTIVE,
        metadata=metadata or {},
    )


async def _count(db: aiosqlite.Connection, sql: str, *params: object) -> int:
    cursor = await db.execute(sql, params)
    row = await cursor.fetchone()
    assert row is not None
    return int(row[0])


class _SpyProvider:
    """Deterministic embedding provider that counts batch vs single calls.

    Embeds each text to a fixed-dimension vector derived from its length so
    per-thought and batch encodings are byte-identical for the same input.
    Records how many times ``embed`` / ``embed_batch`` were invoked so a test
    can assert the bulk path issues exactly one batch call.
    """

    def __init__(self, *, dimension: int = 4, model_name: str = "spy-4") -> None:
        self._dimension = dimension
        self._model_name = model_name
        self.embed_calls = 0
        self.embed_batch_calls = 0
        self.batch_sizes: list[int] = []

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def _vec(self, text: str) -> list[float]:
        base = float(len(text) % 7) + 1.0
        return [base + i for i in range(self._dimension)]

    async def embed(self, text: str) -> list[float]:
        self.embed_calls += 1
        return self._vec(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.embed_batch_calls += 1
        self.batch_sizes.append(len(texts))
        return [self._vec(t) for t in texts]


class _RoleAwareSpyProvider(_SpyProvider):
    """Role-aware provider recording which document methods were used.

    Satisfies the full ``RoleAwareEmbeddingProvider`` capability so the store
    must dispatch to ``embed_document_batch`` on the bulk path (mirroring how
    the single-item path uses ``embed_document``). The document prefix is
    applied exactly like the real providers.
    """

    def __init__(self, *, document_prefix: str = "passage: ") -> None:
        super().__init__(dimension=4, model_name="role-spy-4")
        self._document_prefix = document_prefix
        self.embed_document_calls = 0
        self.embed_document_batch_calls = 0

    @property
    def query_prefix(self) -> str:
        return "query: "

    @property
    def document_prefix(self) -> str:
        return self._document_prefix

    async def embed_query(self, text: str) -> list[float]:
        return await self.embed("query: " + text)

    async def embed_document(self, text: str) -> list[float]:
        self.embed_document_calls += 1
        return await self.embed(self._document_prefix + text)

    async def embed_query_batch(self, texts: list[str]) -> list[list[float]]:
        return await self.embed_batch(["query: " + t for t in texts])

    async def embed_document_batch(self, texts: list[str]) -> list[list[float]]:
        self.embed_document_batch_calls += 1
        return await self.embed_batch([self._document_prefix + t for t in texts])


class _FailingProvider:
    """Embedding provider whose ``embed`` / ``embed_batch`` always raise."""

    dimension = 4
    model_name = "failing-4"

    async def embed(self, text: str) -> list[float]:
        msg = "provider offline"
        raise RuntimeError(msg)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        msg = "provider offline"
        raise RuntimeError(msg)


async def _embedding_store(
    conn: aiosqlite.Connection,
    provider: object,
    *,
    require_embedding: bool = False,
) -> SqliteEngravaCore:
    """Build an auto-embed store on the shared connection."""
    s = SqliteEngravaCore(
        conn,
        embedding_provider=provider,  # type: ignore[arg-type]
        auto_embed=True,
        require_embedding=require_embedding,
    )
    await s._probe_fts()
    return s


class _FlakyProvider:
    """Succeeds only when the embed text contains a chosen marker, else raises.

    Lets a test drive a sequence of embed attempts where some succeed (with a
    fixed, recognisable vector) and later ones fail, to check what a real
    on-disk connection reads back afterwards.
    """

    dimension = 4
    model_name = "flaky-4"

    def __init__(self, *, succeeds_on: str, vector: list[float]) -> None:
        self._succeeds_on = succeeds_on
        self._vector = vector

    async def embed(self, text: str) -> list[float]:
        if self._succeeds_on in text:
            return self._vector
        msg = f"provider exploded for: {text!r}"
        raise RuntimeError(msg)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


async def _open_file_store(
    db_path: str,
    provider: object,
    *,
    require_embedding: bool = False,
) -> tuple[SqliteEngravaCore, aiosqlite.Connection]:
    """Build an auto-embed store over a real on-disk file (not ``:memory:``).

    A second, independent connection can then read this file back to confirm
    what is actually durable, rather than trusting what the writer's own
    connection sees before it has necessarily flushed a commit to disk.
    """
    conn = await aiosqlite.connect(db_path)
    conn.row_factory = aiosqlite.Row
    store = SqliteEngravaCore(
        conn,
        embedding_provider=provider,  # type: ignore[arg-type]
        auto_embed=True,
        require_embedding=require_embedding,
    )
    await store.ensure_schema()
    return store, conn


async def _durable_thought_count(db_path: str, thought_id: str) -> int:
    """Read the durable row count for ``thought_id`` from a fresh connection."""
    conn = await aiosqlite.connect(db_path)
    try:
        return await _count(conn, "SELECT COUNT(*) FROM thought WHERE thought_id = ?", thought_id)
    finally:
        await conn.close()


async def _durable_thought_content(db_path: str, thought_id: str) -> str | None:
    """Read the durable ``content`` for ``thought_id`` from a fresh connection."""
    conn = await aiosqlite.connect(db_path)
    try:
        cursor = await conn.execute(
            "SELECT content FROM thought WHERE thought_id = ?", (thought_id,)
        )
        row = await cursor.fetchone()
        return None if row is None else str(row[0])
    finally:
        await conn.close()


async def _durable_embedding_vector(db_path: str, thought_id: str) -> list[float]:
    """Read the durable embedding vector for ``thought_id`` from a fresh connection."""
    conn = await aiosqlite.connect(db_path)
    try:
        cursor = await conn.execute(
            "SELECT vector_blob, dimension FROM embedding WHERE owner_id = ?", (thought_id,)
        )
        row = await cursor.fetchone()
        assert row is not None
        return list(struct.unpack(f"{row[1]}f", row[0]))
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# get_or_create
# ---------------------------------------------------------------------------


async def test_get_or_create_creates_on_first_call(
    store: SqliteEngravaCore,
    db: aiosqlite.Connection,
) -> None:
    """First call inserts a new row and reports ``created=True``."""
    record, created = await store.get_or_create(_thought("t-goc-1"))

    assert created is True
    assert record.thought_id == "t-goc-1"
    assert await _count(db, "SELECT COUNT(*) FROM thought") == 1


async def test_get_or_create_returns_existing_on_second_call(
    store: SqliteEngravaCore,
    db: aiosqlite.Connection,
) -> None:
    """Second call with identical content returns the existing row, no insert."""
    content = "A stable fact that should be reused across calls."
    first, first_created = await store.get_or_create(_thought("t-goc-a", content=content))
    second, second_created = await store.get_or_create(
        _thought("t-goc-b", content=content),
    )

    assert first_created is True
    assert second_created is False
    # No new row; the existing thought_id is returned (not the second id).
    assert await _count(db, "SELECT COUNT(*) FROM thought") == 1
    assert second.thought_id == first.thought_id == "t-goc-a"


async def test_get_or_create_confirmation_matches_deduplicate_true(
    db: aiosqlite.Connection,
) -> None:
    """``get_or_create`` bumps confirmation exactly like ``deduplicate=True``.

    Runs both APIs through the same repeated-content sequence on isolated
    stores and asserts the persisted ``confirmation_count`` converges.
    """
    content = "Repeated content whose confirmation count must match."

    goc_store = SqliteEngravaCore(db)
    await goc_store._probe_fts()
    for i in range(4):
        _, _created = await goc_store.get_or_create(_thought(f"goc-{i}", content=content))
    goc_count = await _count(
        db,
        "SELECT confirmation_count FROM thought WHERE content_hash IS NOT NULL",
    )

    # Fresh DB for the deduplicate=True reference run.
    conn2 = await aiosqlite.connect(":memory:")
    conn2.row_factory = aiosqlite.Row
    try:
        dedup_store = SqliteEngravaCore(conn2)
        await dedup_store.ensure_schema()
        for i in range(4):
            await dedup_store.create_thought(
                _thought(f"dd-{i}", content=content),
                deduplicate=True,
            )
        dedup_count = await _count(
            conn2,
            "SELECT confirmation_count FROM thought WHERE content_hash IS NOT NULL",
        )
    finally:
        await conn2.close()

    assert goc_count == dedup_count == 3


async def test_get_or_create_does_not_adopt_incoming_fields_on_hit(
    store: SqliteEngravaCore,
) -> None:
    """A hit returns the stored record unchanged (only confirmation bumped)."""
    content = "Content whose stored metadata must survive a get_or_create hit."
    first, _ = await store.get_or_create(
        _thought("t-goc-keep", content=content, priority=Priority.P1, metadata={"k": "original"}),
    )
    second, created = await store.get_or_create(
        _thought(
            "t-goc-keep-2",
            content=content,
            priority=Priority.P3,
            metadata={"k": "changed"},
        ),
    )

    assert created is False
    # Incoming P3 / changed metadata are ignored — stored values persist.
    assert second.priority is Priority.P1
    assert second.metadata == {"k": "original"}
    assert second.confirmation_count == first.confirmation_count + 1


async def test_get_or_create_validates_metadata_on_hit(
    store: SqliteEngravaCore,
) -> None:
    """Invalid metadata raises on a hit too, matching ``deduplicate=True``.

    ``create_thought`` validates metadata before it branches, so a dedup hit
    with oversized metadata still raises. ``get_or_create`` must be consistent.
    """
    content = "Content seeded to force a subsequent get_or_create hit."
    await store.get_or_create(_thought("t-goc-val", content=content))

    oversized = {"blob": "x" * 70_000}  # exceeds the 64 KiB store cap
    with pytest.raises(ValueError, match="metadata serialized size"):
        await store.get_or_create(
            _thought("t-goc-val-2", content=content, metadata=oversized),
        )


# ---------------------------------------------------------------------------
# upsert_by_hash
# ---------------------------------------------------------------------------


async def test_upsert_by_hash_inserts_on_miss(
    store: SqliteEngravaCore,
    db: aiosqlite.Connection,
) -> None:
    """No existing hash → a new row is inserted and returned."""
    record = await store.upsert_by_hash(_thought("t-up-1"))

    assert record.thought_id == "t-up-1"
    assert await _count(db, "SELECT COUNT(*) FROM thought") == 1


async def test_upsert_by_hash_updates_mutable_fields_on_match(
    store: SqliteEngravaCore,
    db: aiosqlite.Connection,
) -> None:
    """On a hash match the stored row's mutable fields are updated in place."""
    content = "Content that gets a newer version with different metadata/priority."
    first = await store.upsert_by_hash(
        _thought("t-up-a", content=content, priority=Priority.P3, metadata={"v": "1"}),
    )
    second = await store.upsert_by_hash(
        _thought(
            "t-up-b",
            content=content,
            priority=Priority.P1,
            essence="Revised essence",
            metadata={"v": "2", "extra": "added"},
        ),
    )

    # Same logical row (existing id kept), no new row inserted.
    assert second.thought_id == first.thought_id == "t-up-a"
    assert await _count(db, "SELECT COUNT(*) FROM thought") == 1
    # Mutable fields adopted the incoming record's values.
    assert second.priority is Priority.P1
    assert second.essence == "Revised essence"
    assert second.metadata == {"v": "2", "extra": "added"}
    # confirmation_count is NOT bumped (distinct from dedup semantics).
    assert second.confirmation_count == first.confirmation_count == 0
    # Content unchanged (it is the hash key).
    assert second.content == content


async def test_upsert_by_hash_persists_update_in_db(
    store: SqliteEngravaCore,
    db: aiosqlite.Connection,
) -> None:
    """The in-place update is durable in SQLite, not only in the returned model."""
    content = "Durability check for upsert_by_hash update-on-match."
    await store.upsert_by_hash(_thought("t-up-db", content=content, priority=Priority.P3))
    await store.upsert_by_hash(_thought("t-up-db-2", content=content, priority=Priority.P1))

    cursor = await db.execute(
        "SELECT priority, confirmation_count FROM thought WHERE content_hash IS NOT NULL",
    )
    row = await cursor.fetchone()
    assert row is not None
    assert row["priority"] == Priority.P1.value
    assert row["confirmation_count"] == 0


async def test_upsert_by_hash_differs_from_deduplicate_true(
    store: SqliteEngravaCore,
) -> None:
    """Contrast: ``deduplicate=True`` keeps stored fields; upsert overwrites them.

    Seeds a row via ``deduplicate=True`` (P3), then a same-content
    ``deduplicate=True`` call with P1 leaves P3 stored and bumps confirmation;
    an ``upsert_by_hash`` with P1 finally overwrites it.
    """
    content = "Content demonstrating the upsert vs dedup contrast."
    seeded = await store.create_thought(
        _thought("t-cmp-seed", content=content, priority=Priority.P3),
        deduplicate=True,
    )
    dedup_hit = await store.create_thought(
        _thought("t-cmp-dd", content=content, priority=Priority.P1),
        deduplicate=True,
    )
    # dedup keeps the stored P3, bumps confirmation.
    assert dedup_hit.priority is Priority.P3
    assert dedup_hit.confirmation_count == seeded.confirmation_count + 1

    upserted = await store.upsert_by_hash(
        _thought("t-cmp-up", content=content, priority=Priority.P1),
    )
    # upsert overwrites to P1 and does not further bump confirmation.
    assert upserted.priority is Priority.P1
    assert upserted.confirmation_count == dedup_hit.confirmation_count


async def test_upsert_by_hash_identical_fields_is_noop(
    store: SqliteEngravaCore,
    db: aiosqlite.Connection,
) -> None:
    """An upsert whose mutable fields already match returns the row untouched.

    In particular it must not re-assert the identical ``lifecycle_status`` (a
    same-state transition would otherwise raise) and must not bump the cycle.
    """
    content = "Content re-upserted with byte-identical mutable fields."
    first = await store.upsert_by_hash(_thought("t-noop", content=content))
    second = await store.upsert_by_hash(_thought("t-noop-2", content=content))

    assert second.thought_id == first.thought_id == "t-noop"
    # No update happened: OCC cycle is unchanged and no new row landed.
    assert second.updated_cycle == first.updated_cycle
    assert await _count(db, "SELECT COUNT(*) FROM thought") == 1


async def test_upsert_by_hash_validates_metadata_on_hit(
    store: SqliteEngravaCore,
) -> None:
    """Invalid metadata raises up front on a hit, before any in-place update."""
    content = "Content seeded to force a subsequent upsert hit."
    await store.upsert_by_hash(_thought("t-up-val", content=content))

    oversized = {"blob": "x" * 70_000}
    with pytest.raises(ValueError, match="metadata serialized size"):
        await store.upsert_by_hash(
            _thought("t-up-val-2", content=content, metadata=oversized),
        )


async def test_upsert_by_hash_applies_valid_lifecycle_transition(
    store: SqliteEngravaCore,
) -> None:
    """A differing, valid ``lifecycle_status`` on a match is applied in place."""
    content = "Content whose lifecycle advances on upsert."
    first = await store.upsert_by_hash(
        _thought("t-life", content=content),  # ACTIVE
    )
    assert first.lifecycle_status is LifecycleStatus.ACTIVE

    second = await store.upsert_by_hash(
        _thought("t-life-2", content=content).evolve(
            lifecycle_status=LifecycleStatus.ARCHIVED,
        ),
    )
    assert second.thought_id == first.thought_id
    assert second.lifecycle_status is LifecycleStatus.ARCHIVED


async def test_upsert_by_hash_noop_does_not_commit_callers_pending_work(
    store: SqliteEngravaCore,
    db: aiosqlite.Connection,
) -> None:
    """A no-op ``upsert_by_hash()`` must not commit the caller's own pending write.

    The unchanged-record branch writes nothing of its own -- no
    ``UPDATE``, no journal entry -- so it must not call ``_maybe_commit()``
    either. Regressed to calling it unconditionally, which committed whatever
    *unrelated* work the caller already had open on the same connection (a
    rejected journal insert is the motivating case), leaving a later
    ``rollback()`` with nothing left to undo.

    Reproduced here without any journal/rejection machinery: a raw pending
    ``INSERT`` left uncommitted on the shared connection stands in for "the
    caller's pending work", exactly the way the residual gap documented on
    :meth:`SqliteEngravaCore._serialize_dedup_probe` describes a transaction
    already open when this store's own guard is entered.
    """
    content = "Content re-upserted with byte-identical mutable fields."
    seeded = await store.upsert_by_hash(_thought("t-rb-noop-1", content=content))

    # The caller's own pending, uncommitted write on the same connection.
    await db.execute(
        "INSERT INTO thought (thought_id, thought_type, essence, content, priority) "
        "VALUES ('pending-row', 'OBSERVATION', 'e', 'pending content', 'P2')",
    )
    assert db.in_transaction

    # A no-op hit: same mutable fields as the seeded row, so nothing to write.
    result = await store.upsert_by_hash(_thought("t-rb-noop-2", content=content))
    assert result.thought_id == seeded.thought_id

    await db.rollback()

    assert await _count(db, "SELECT COUNT(*) FROM thought WHERE thought_id = 'pending-row'") == 0


async def test_upsert_by_hash_update_branch_still_commits_pending_work(
    store: SqliteEngravaCore,
    db: aiosqlite.Connection,
) -> None:
    """Control: the mutable-field-update branch keeps committing, unchanged.

    Unlike the no-op branch (see the sibling test above), this branch does
    write: it delegates to :meth:`SqliteEngravaCore.update_thought`, which
    calls ``_maybe_commit()`` itself after its own journal append -- the
    "whoever writes, commits" rule applied correctly. That commit is on the
    one shared connection, so it also makes durable whatever unrelated
    pending write the caller already had open; a later ``rollback()`` finds
    nothing left to undo. This must hold both before and after the fix to
    the no-op branch above, since that fix only removes the no-op branch's
    own, separate commit call.
    """
    content = "Content that receives a genuine mutable-field update."
    seeded = await store.upsert_by_hash(
        _thought("t-rb-upd-1", content=content, priority=Priority.P3),
    )

    await db.execute(
        "INSERT INTO thought (thought_id, thought_type, essence, content, priority) "
        "VALUES ('pending-row-ctrl', 'OBSERVATION', 'e', 'pending content 2', 'P2')",
    )
    assert db.in_transaction

    updated = await store.upsert_by_hash(
        _thought("t-rb-upd-2", content=content, priority=Priority.P1),
    )
    assert updated.thought_id == seeded.thought_id
    assert updated.priority is Priority.P1

    # update_thought already committed everything on this connection --
    # rollback() has nothing left to discard.
    await db.rollback()

    assert (
        await _count(db, "SELECT COUNT(*) FROM thought WHERE thought_id = 'pending-row-ctrl'") == 1
    )


async def test_upsert_by_hash_noop_inside_suspend_auto_commit_unaffected(
    store: SqliteEngravaCore,
    db: aiosqlite.Connection,
) -> None:
    """``suspend_auto_commit()`` already made ``_maybe_commit()`` a no-op there.

    Confirms the no-op branch's commit-call removal changes nothing
    observable inside a caller's own ``suspend_auto_commit()`` window: a
    no-op ``upsert_by_hash()`` call
    was, and remains, side-effect-free there, because ``_skip_auto_commit``
    already suppressed the branch's ``_maybe_commit()`` call before this fix
    removed the call outright. An outer rollback still discards every write
    made inside the window, exactly as before.
    """
    content = "Content re-upserted with byte-identical mutable fields, nested."

    async def _run() -> None:
        async with store.suspend_auto_commit():
            await store.upsert_by_hash(_thought("t-nest-1", content=content))
            # A no-op hit, still inside the same suspended window.
            await store.upsert_by_hash(_thought("t-nest-2", content=content))
            msg = "abort the whole window"
            raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="abort the whole window"):
        await _run()

    assert await _count(db, "SELECT COUNT(*) FROM thought") == 0


async def test_upsert_by_hash_noop_leaves_an_explicit_begin_untouched(
    store: SqliteEngravaCore,
    db: aiosqlite.Connection,
) -> None:
    """A no-op match must not touch a transaction opened by a raw ``BEGIN``.

    Distinct from the ``suspend_auto_commit()`` and ``bulk_store`` cases
    (siblings of this test): here nothing in this store's own API opened the
    transaction at all -- a caller issued ``BEGIN`` directly against the
    shared connection, exactly the "residual gap" documented on
    :meth:`SqliteEngravaCore._serialize_dedup_probe` ("a raw ``BEGIN`` issued
    directly against it, with nothing written yet"). Because
    ``self._db.in_transaction`` is already ``True`` when ``upsert_by_hash``
    is entered, ``opened_transaction`` is computed ``False`` and
    :meth:`SqliteEngravaCore._end_exploratory_probe` must take no action at
    all: neither committing the caller's pending write nor rolling it back.
    """
    seeded = await store.upsert_by_hash(_thought("t-begin-seed", content="explicit begin case"))

    await db.execute("BEGIN")
    await db.execute(
        "INSERT INTO thought (thought_id, thought_type, essence, content, priority) "
        "VALUES ('pending-explicit-begin', 'OBSERVATION', 'e', 'pending content', 'P2')",
    )
    assert db.in_transaction

    result = await store.upsert_by_hash(_thought("t-begin-other", content="explicit begin case"))
    assert result.thought_id == seeded.thought_id

    # Still open -- neither committed nor rolled back by the no-op call.
    assert db.in_transaction

    await db.rollback()
    assert (
        await _count(
            db,
            "SELECT COUNT(*) FROM thought WHERE thought_id = 'pending-explicit-begin'",
        )
        == 0
    )


class _ReentrantUpsertHooks(DefaultEngravaHooks):
    """``on_store`` hook that reenters ``upsert_by_hash`` with a no-op match.

    Models a plugin's ``on_store`` callback calling back into the store on
    the same task -- the reentrant shape ``_write_lock``/``_dedup_lock``'s
    task-reentrant design exists to support. ``store`` is wired in after
    construction since the hooks object must exist before the store that
    takes it.
    """

    def __init__(self, *, trigger_id: str, noop_content: str) -> None:
        super().__init__()
        self.store: SqliteEngravaCore | None = None
        self._trigger_id = trigger_id
        self._noop_content = noop_content
        self.reentrant_result: ThoughtRecord | None = None

    async def on_store(self, thought: ThoughtRecord) -> ThoughtRecord:
        """Reenter ``upsert_by_hash`` with a no-op match for the trigger row.

        Args:
            thought: The just-persisted thought passed to this hook.

        Returns:
            ``thought`` unchanged.

        """
        if thought.thought_id == self._trigger_id:
            assert self.store is not None
            self.reentrant_result = await self.store.upsert_by_hash(
                _thought("reentrant-probe", content=self._noop_content),
            )
        return thought


async def test_upsert_by_hash_noop_inside_bulk_store_batch_untouched(
    db: aiosqlite.Connection,
) -> None:
    """A no-op match reentered from inside a ``bulk_store`` batch touches nothing.

    ``bulk_store`` runs its whole insert loop -- including each row's
    ``on_store`` dispatch -- under one ``suspend_auto_commit()`` window (see
    :meth:`SqliteEngravaCore._bulk_store_inner`). If an ``on_store`` hook
    calls back into ``upsert_by_hash`` on the same task and that call is a
    no-op match, ``self._db.in_transaction`` is already ``True`` (the
    batch's own transaction), so ``opened_transaction`` is ``False`` and
    :meth:`SqliteEngravaCore._end_exploratory_probe` must leave the batch's
    transaction completely alone. Decisive check: every row the batch
    inserted -- including the one whose ``on_store`` triggered the reentrant
    call -- is still present and committed once the batch returns; a wrong
    implementation that rolled back on ``opened_transaction=False`` would
    discard the whole in-flight batch instead.
    """
    seed_content = "Content the reentrant no-op call matches unchanged."
    hooks = _ReentrantUpsertHooks(trigger_id="t-bulk-trigger", noop_content=seed_content)
    store = SqliteEngravaCore(db, hooks)
    hooks.store = store
    await store._probe_fts()

    seeded = await store.upsert_by_hash(_thought("t-bulk-seed", content=seed_content))

    persisted = await store.bulk_store(
        [
            _thought("t-bulk-other", content="unrelated batch content"),
            _thought("t-bulk-trigger", content="content distinct from the seed"),
        ],
    )

    assert [record.thought_id for record in persisted] == ["t-bulk-other", "t-bulk-trigger"]
    assert hooks.reentrant_result is not None
    assert hooks.reentrant_result.thought_id == seeded.thought_id
    assert not db.in_transaction

    for thought_id in ("t-bulk-seed", "t-bulk-other", "t-bulk-trigger"):
        count = await _count(db, "SELECT COUNT(*) FROM thought WHERE thought_id = ?", thought_id)
        assert count == 1


# ---------------------------------------------------------------------------
# bulk_store — transactional insert
# ---------------------------------------------------------------------------


async def test_bulk_store_empty_is_noop(store: SqliteEngravaCore) -> None:
    """An empty batch returns an empty list and touches nothing."""
    assert await store.bulk_store([]) == []


async def test_bulk_store_preserves_order(
    store: SqliteEngravaCore,
    db: aiosqlite.Connection,
) -> None:
    """Returned records are in input order and all rows land."""
    thoughts = [_thought(f"t-bulk-{i}", content=f"Bulk observation #{i}.") for i in range(6)]
    persisted = await store.bulk_store(thoughts)

    assert [p.thought_id for p in persisted] == [t.thought_id for t in thoughts]
    assert await _count(db, "SELECT COUNT(*) FROM thought") == 6


async def test_bulk_store_commits_once(
    store: SqliteEngravaCore,
    db: aiosqlite.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole batch commits exactly once, not once per row."""
    real_commit = db.commit
    commit_calls = 0

    async def _counting_commit() -> None:
        nonlocal commit_calls
        commit_calls += 1
        await real_commit()

    monkeypatch.setattr(db, "commit", _counting_commit)

    thoughts = [_thought(f"t-once-{i}", content=f"One-commit row #{i}.") for i in range(5)]
    await store.bulk_store(thoughts)

    assert commit_calls == 1
    assert await _count(db, "SELECT COUNT(*) FROM thought") == 5


async def test_bulk_store_rolls_back_on_mid_batch_failure(
    store: SqliteEngravaCore,
    db: aiosqlite.Connection,
) -> None:
    """A failure mid-batch rolls the whole transaction back — nothing persists."""
    # Duplicate thought_id in the batch: the second insert of ``dup`` raises
    # ValueError (thought already exists) inside the transaction.
    thoughts = [
        _thought("t-rb-1", content="first"),
        _thought("dup", content="second"),
        _thought("dup", content="third"),  # duplicate id -> raises
        _thought("t-rb-4", content="fourth"),
    ]

    with pytest.raises(ValueError, match="already exists"):
        await store.bulk_store(thoughts)

    # All-or-nothing: not even the rows before the failure survive.
    assert await _count(db, "SELECT COUNT(*) FROM thought") == 0


async def test_bulk_store_earlier_duplicate_id_outranks_a_later_ordinary_validation_error(
    store: SqliteEngravaCore,
    db: aiosqlite.Connection,
) -> None:
    """A later item's ordinary validation error must not pre-empt an earlier
    item's duplicate-id failure.

    A bare loop of ``create_thought()`` calls finishes item *n* -- including
    any insert-time failure -- before item *n + 1* is even looked at, so the
    duplicate id here (item 3) must be what raises, never the oversized
    metadata on item 4 that comes after it. This pins the two-phase
    ``bulk_store`` restructuring's failure-ordering fix: an earlier version
    validated every item, batch-wide, before any insert, which let a later
    item's ordinary (non-seam) validation error raise first instead.
    """
    oversized_metadata = {"blob": "x" * 70_000}  # exceeds the 64 KiB metadata cap
    thoughts = [
        _thought("t-rb-1", content="first"),
        _thought("dup", content="second"),
        _thought("dup", content="third"),  # duplicate id -> raises at insert time
        _thought("t-rb-4", content="fourth", metadata=oversized_metadata),
    ]

    with pytest.raises(ValueError, match="already exists"):
        await store.bulk_store(thoughts)

    # All-or-nothing either way: not even the rows before the failure survive.
    assert await _count(db, "SELECT COUNT(*) FROM thought") == 0


async def test_bulk_store_on_store_ordering_survives_a_later_ordinary_validation_error(
    db: aiosqlite.Connection,
) -> None:
    """A later item's ordinary validation error must not suppress an earlier
    item's ``on_store`` call.

    Matches the same base-ordering guarantee as the duplicate-id case above:
    item *n*'s ``on_store`` already ran -- item *n* is fully finished -- by
    the time item *n + 1* is even looked at, whether or not the whole batch
    later rolls back for being all-or-nothing.
    """
    on_store_calls: list[str] = []

    class _RecordingHooks(DefaultEngravaHooks):
        async def on_store(self, thought: ThoughtRecord) -> ThoughtRecord:
            on_store_calls.append(thought.thought_id)
            return thought

    store = SqliteEngravaCore(db, hooks=_RecordingHooks())
    await store._probe_fts()

    oversized_metadata = {"blob": "x" * 70_000}  # exceeds the 64 KiB metadata cap
    thoughts = [
        _thought("ok-1", content="first, valid"),
        _thought("ok-2", content="second, valid"),
        _thought("bad-3", content="third, invalid metadata", metadata=oversized_metadata),
    ]

    with pytest.raises(ValueError, match="metadata"):
        await store.bulk_store(thoughts)

    # ok-1 and ok-2's on_store already ran before bad-3's validation failed.
    assert on_store_calls == ["ok-1", "ok-2"]
    # All-or-nothing: the whole batch, including ok-1 / ok-2, still rolls back.
    assert await _count(db, "SELECT COUNT(*) FROM thought") == 0


async def test_bulk_store_honors_deduplicate_per_row(
    store: SqliteEngravaCore,
    db: aiosqlite.Connection,
) -> None:
    """``deduplicate=True`` collapses same-content rows within the batch."""
    thoughts = [
        _thought("t-dd-1", content="shared"),
        _thought("t-dd-2", content="shared"),
        _thought("t-dd-3", content="distinct"),
    ]
    persisted = await store.bulk_store(thoughts, deduplicate=True)

    # Two distinct logical thoughts; the second "shared" collapsed onto the first.
    assert await _count(db, "SELECT COUNT(*) FROM thought") == 2
    assert persisted[0].thought_id == persisted[1].thought_id == "t-dd-1"
    assert persisted[2].thought_id == "t-dd-3"


# ---------------------------------------------------------------------------
# bulk_store — batch embedding
# ---------------------------------------------------------------------------


async def test_bulk_store_issues_single_batch_embed(
    db: aiosqlite.Connection,
) -> None:
    """N thoughts under auto-embed trigger exactly one ``embed_batch`` call."""
    provider = _SpyProvider()
    store = await _embedding_store(db, provider)

    thoughts = [_thought(f"t-be-{i}", content=f"Batch embed row #{i}.") for i in range(5)]
    await store.bulk_store(thoughts)

    assert provider.embed_batch_calls == 1
    assert provider.embed_calls == 0
    assert provider.batch_sizes == [5]
    # Every thought got an embedding row.
    assert await _count(db, "SELECT COUNT(*) FROM embedding") == 5


async def test_bulk_store_batch_vectors_equal_per_thought(
    db: aiosqlite.Connection,
) -> None:
    """Vectors stored via the batch path equal per-thought embedding.

    Embeds the same thoughts twice: once via ``bulk_store`` (batch) and once
    via per-thought ``create_thought`` on a separate store/DB, then compares
    the persisted vectors byte-for-byte.
    """
    contents = [f"Vector-equality row #{i}." for i in range(4)]

    batch_provider = _SpyProvider()
    batch_store = await _embedding_store(db, batch_provider)
    await batch_store.bulk_store(
        [_thought(f"t-veq-{i}", content=c) for i, c in enumerate(contents)],
    )
    batch_vectors = {
        row["owner_id"]: bytes(row["vector_blob"])
        for row in await (
            await db.execute("SELECT owner_id, vector_blob FROM embedding")
        ).fetchall()
    }

    conn2 = await aiosqlite.connect(":memory:")
    conn2.row_factory = aiosqlite.Row
    try:
        single_store = SqliteEngravaCore(
            conn2,
            embedding_provider=_SpyProvider(),
            auto_embed=True,
        )
        await single_store.ensure_schema()
        for i, c in enumerate(contents):
            await single_store.create_thought(_thought(f"t-veq-{i}", content=c))
        single_vectors = {
            row["owner_id"]: bytes(row["vector_blob"])
            for row in await (
                await conn2.execute("SELECT owner_id, vector_blob FROM embedding")
            ).fetchall()
        }
    finally:
        await conn2.close()

    assert batch_vectors == single_vectors
    assert len(batch_vectors) == 4


async def test_bulk_store_dispatches_role_aware_document_batch(
    db: aiosqlite.Connection,
) -> None:
    """A role-aware provider is batched via ``embed_document_batch``, not ``embed_batch``.

    Mirrors the single-item path's dispatch to ``embed_document`` — the bulk
    path must use the document-role batch method (with its prefix), never the
    plain ``embed_batch``.
    """
    provider = _RoleAwareSpyProvider()
    store = await _embedding_store(db, provider)

    await store.bulk_store(
        [_thought(f"t-role-{i}", content=f"role batch #{i}") for i in range(3)],
    )

    assert provider.embed_document_batch_calls == 1
    assert provider.embed_document_calls == 0
    # Vectors reflect the document prefix (role-aware path was taken).
    assert await _count(db, "SELECT COUNT(*) FROM embedding") == 3


async def test_bulk_store_role_aware_vectors_equal_single_path(
    db: aiosqlite.Connection,
) -> None:
    """Role-aware batch vectors equal per-thought role-aware embedding."""
    contents = [f"role vector-equality #{i}" for i in range(3)]

    batch_store = await _embedding_store(db, _RoleAwareSpyProvider())
    await batch_store.bulk_store(
        [_thought(f"t-rveq-{i}", content=c) for i, c in enumerate(contents)],
    )
    batch_vectors = {
        row["owner_id"]: bytes(row["vector_blob"])
        for row in await (
            await db.execute("SELECT owner_id, vector_blob FROM embedding")
        ).fetchall()
    }

    conn2 = await aiosqlite.connect(":memory:")
    conn2.row_factory = aiosqlite.Row
    try:
        single = SqliteEngravaCore(
            conn2,
            embedding_provider=_RoleAwareSpyProvider(),
            auto_embed=True,
        )
        await single.ensure_schema()
        for i, c in enumerate(contents):
            await single.create_thought(_thought(f"t-rveq-{i}", content=c))
        single_vectors = {
            row["owner_id"]: bytes(row["vector_blob"])
            for row in await (
                await conn2.execute("SELECT owner_id, vector_blob FROM embedding")
            ).fetchall()
        }
    finally:
        await conn2.close()

    assert batch_vectors == single_vectors
    assert len(batch_vectors) == 3


async def test_bulk_store_skips_embedding_for_dedup_hits(
    db: aiosqlite.Connection,
) -> None:
    """A dedup hit within the batch is not re-embedded (only inserts are)."""
    provider = _SpyProvider()
    store = await _embedding_store(db, provider)

    thoughts = [
        _thought("t-skip-1", content="shared"),
        _thought("t-skip-2", content="shared"),  # dedup hit -> not embedded
        _thought("t-skip-3", content="unique"),
    ]
    await store.bulk_store(thoughts, deduplicate=True)

    # Two rows inserted -> two embeddings; the batch call embedded exactly 2 texts.
    assert provider.embed_batch_calls == 1
    assert provider.batch_sizes == [2]
    assert await _count(db, "SELECT COUNT(*) FROM embedding") == 2


async def test_bulk_store_dedup_hit_reusing_existing_id_not_reembedded(
    db: aiosqlite.Connection,
) -> None:
    """A dedup hit that reuses an existing row's id is classified by row existence.

    Regression guard: dedup-hit detection must key off whether the row already
    existed, not instance identity — otherwise a submitted thought whose id
    coincides with the matched row's id would be misread as a fresh insert and
    redundantly re-embedded.
    """
    provider = _SpyProvider()
    store = await _embedding_store(db, provider)

    # Seed one thought (id "shared-id", content C).
    await store.create_thought(_thought("shared-id", content="C"))
    assert provider.embed_calls == 1
    provider.embed_batch_calls = 0  # reset before the batch

    # Batch resubmits the SAME id with the SAME content under dedup -> a hit.
    await store.bulk_store([_thought("shared-id", content="C")], deduplicate=True)

    # No genuine insert -> no batch embed call, still exactly one embedding row.
    assert provider.embed_batch_calls == 0
    assert await _count(db, "SELECT COUNT(*) FROM embedding") == 1
    assert await _count(db, "SELECT COUNT(*) FROM thought") == 1


async def test_bulk_store_all_dedup_hits_issues_no_embed_call(
    db: aiosqlite.Connection,
) -> None:
    """A batch where every row is a dedup hit issues no batch-embed call."""
    provider = _SpyProvider()
    store = await _embedding_store(db, provider)

    # Seed the content first (one insert, one embed).
    await store.create_thought(_thought("t-seed", content="already here"))
    assert provider.embed_calls == 1

    # A batch of only-already-present content: all dedup hits, nothing to embed.
    await store.bulk_store(
        [
            _thought("t-allhit-1", content="already here"),
            _thought("t-allhit-2", content="already here"),
        ],
        deduplicate=True,
    )

    # No batch-embed call was made (to_embed was empty).
    assert provider.embed_batch_calls == 0
    assert await _count(db, "SELECT COUNT(*) FROM embedding") == 1


# ---------------------------------------------------------------------------
# No silent embedding skip (WARN + require_embedding)
# ---------------------------------------------------------------------------


def test_require_embedding_docstring_does_not_overclaim_top_level_durability() -> None:
    """The constructor's ``require_embedding`` entry must not say "committed either way".

    That phrase is only true for ``create_thought``/``update_thought``: a
    standalone ``bulk_store`` shares one transaction between its insert loop
    and the trailing batch-embed call, so a strict failure rolls the *whole
    batch* back instead (see ``test_bulk_store_strict_embed_failure_rolls_back``
    above) — the row this docstring tells an operator to repair may not exist.
    The entry must call that exception out by name rather than claim every
    top-level path commits regardless of which one raised.
    """
    doc = inspect.getdoc(SqliteEngravaCore) or ""
    start = doc.index("require_embedding:")
    end = doc.index("search_config:")
    section = doc[start:end]

    assert "committed either way" not in section
    assert "bulk_store" in section
    # A keyword check like this one can pass against a docstring that asserts
    # the opposite of the truth, as long as it contains "bulk_store" and
    # avoids one forbidden phrase — it does not verify behaviour. The
    # behaviour is instead asserted directly by
    # ``test_require_embedding_flag_flips_durability_through_a_typed_except``
    # and ``test_nested_update_rollback_can_leave_a_stale_embedding`` below.
    # This assertion only guards a specific absolute this file has already
    # gotten wrong once, so a regression back to it is caught immediately.
    assert "it never decides whether the thought row survives" not in section


async def test_require_embedding_flag_flips_durability_through_a_typed_except(
    tmp_path: Path,
) -> None:
    """Same provider failure, same caller code, opposite durable outcome.

    The constructor docstring must not claim ``require_embedding`` "never
    decides whether the thought row survives": it does, indirectly, because
    it decides the exception *type*, and an ordinary caller ``except
    EmbeddingGenerationError`` clause only catches the strict-mode error.
    Nested inside the caller's own ``suspend_auto_commit()`` window, that
    difference decides whether the window exits cleanly (commits) or lets
    the exception escape (rolls back). Verified from a second, independent
    connection onto the same on-disk file, per ``require_embedding`` value.
    """
    for require_embedding, expect_durable in ((True, True), (False, False)):
        db_path = str(tmp_path / f"flag-{require_embedding}.db")
        store, conn = await _open_file_store(
            db_path, _FailingProvider(), require_embedding=require_embedding
        )
        try:
            with contextlib.suppress(RuntimeError):
                # RuntimeError here is the default, untyped provider error
                # escaping the window uncaught.
                async with store.suspend_auto_commit():
                    with contextlib.suppress(EmbeddingGenerationError):
                        # Only the strict-mode error is caught here.
                        await store.create_thought(_thought("t-flag"))
        finally:
            await conn.close()

        durable = await _durable_thought_count(db_path, "t-flag")
        expected = 1 if expect_durable else 0
        assert durable == expected, (
            f"require_embedding={require_embedding}: expected durable rows="
            f"{expected}, got {durable}"
        )


async def test_nested_update_rollback_can_leave_a_stale_embedding(
    tmp_path: Path,
) -> None:
    """A rolled-back nested update does not guarantee a consistent embedding.

    Two-step sequence: a standalone update A -> B commits (the update commits
    before re-embedding) while its own re-embed fails, leaving content B with
    the embedding of A. A later nested update B -> C then also fails to
    re-embed and escapes uncaught, rolling the outer window back to content
    B — but the embedding was never touched by either failure, so it still
    represents A, not B. Verified by reading both columns back from a second,
    independent connection onto the same on-disk file.
    """
    vector_a = [1.0, 0.0, 0.0, 0.0]
    db_path = str(tmp_path / "stale-embed.db")
    store, conn = await _open_file_store(db_path, _FlakyProvider(succeeds_on="A", vector=vector_a))
    try:
        await store.create_thought(_thought("t-stale", essence="e", content="content-A"))

        # Standalone update A -> B commits; re-embed of B fails and propagates.
        with pytest.raises(RuntimeError):
            await store.update_thought("t-stale", content="content-B")

        # Nested update B -> C: re-embed of C also fails, escapes uncaught,
        # and the outer window rolls the content change back to B.
        with pytest.raises(RuntimeError):
            async with store.suspend_auto_commit():
                await store.update_thought("t-stale", content="content-C")
    finally:
        await conn.close()

    content = await _durable_thought_content(db_path, "t-stale")
    vector = await _durable_embedding_vector(db_path, "t-stale")
    assert content == "content-B", "rollback should restore the pre-nested-update content"
    assert vector == vector_a, (
        "the embedding is still the one from the first failure and was never "
        "repaired by the second rollback — content and embedding disagree"
    )


async def test_create_then_update_in_same_window_rollback_leaves_no_row(
    tmp_path: Path,
) -> None:
    """A rolled-back window can erase a row, not just revert it.

    Rollback does not unwind to "before this call" — it unwinds to whatever
    existed when the *outermost* ``suspend_auto_commit()`` window opened. If
    the thought was created earlier in that same window, a later update's
    re-embed failure that escapes the window rolls the create back too: the
    thought does not revert to some prior durable state, it stops existing
    at all. Verified from a second, independent connection onto the same
    on-disk file.
    """
    vector_a = [1.0, 0.0, 0.0, 0.0]
    db_path = str(tmp_path / "create-then-update-rollback.db")
    store, conn = await _open_file_store(db_path, _FlakyProvider(succeeds_on="A", vector=vector_a))

    async def _create_then_update() -> None:
        async with store.suspend_auto_commit():
            await store.create_thought(_thought("t-window", essence="e", content="content-A"))
            await store.update_thought("t-window", content="content-B")

    try:
        with pytest.raises(RuntimeError):
            await _create_then_update()
    finally:
        await conn.close()

    durable = await _durable_thought_count(db_path, "t-window")
    assert durable == 0, (
        "the create happened inside the same window as the failing update, "
        "so the whole window's rollback erases it — it does not revert to "
        "its pre-update state"
    )


async def test_auto_embed_failure_warns_and_propagates_by_default(
    db: aiosqlite.Connection,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Default (``require_embedding=False``): a WARN names the thought, error propagates.

    The thought is already committed (auto-embed runs after the commit), so it
    is persisted-but-unembedded and the original provider error surfaces.
    """
    store = await _embedding_store(db, _FailingProvider())

    with caplog.at_level(logging.WARNING), pytest.raises(RuntimeError, match="provider offline"):
        await store.create_thought(_thought("t-warn-1"))

    # The thought persisted despite the embed failure (existing torn-write behaviour).
    assert await _count(db, "SELECT COUNT(*) FROM thought") == 1
    assert await _count(db, "SELECT COUNT(*) FROM embedding") == 0
    # The WARN names the thought id so the missing embedding is never silent.
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("t-warn-1" in r.getMessage() for r in warnings)


async def test_auto_embed_failure_raises_typed_under_strict(
    db: aiosqlite.Connection,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``require_embedding=True``: the failure becomes a typed EmbeddingGenerationError."""
    store = await _embedding_store(db, _FailingProvider(), require_embedding=True)

    with (
        caplog.at_level(logging.WARNING),
        pytest.raises(EmbeddingGenerationError) as exc_info,
    ):
        await store.create_thought(_thought("t-strict-1"))

    assert "t-strict-1" in str(exc_info.value)
    assert exc_info.value.thought_id == "t-strict-1"
    # WARN still emitted alongside the typed raise.
    assert any(
        "t-strict-1" in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
    )


async def test_bulk_store_strict_embed_failure_rolls_back(
    db: aiosqlite.Connection,
) -> None:
    """A batch-embed failure under strict mode rolls the whole batch back."""
    store = await _embedding_store(db, _FailingProvider(), require_embedding=True)

    thoughts = [_thought(f"t-bstrict-{i}", content=f"row #{i}") for i in range(3)]
    with pytest.raises(EmbeddingGenerationError):
        await store.bulk_store(thoughts)

    # Whole transaction rolled back: no thoughts, no embeddings.
    assert await _count(db, "SELECT COUNT(*) FROM thought") == 0
    assert await _count(db, "SELECT COUNT(*) FROM embedding") == 0


async def test_strict_embed_failure_nested_in_suspend_auto_commit_is_not_durable(
    db: aiosqlite.Connection,
) -> None:
    """A single-item strict failure nested in the caller's own window is not durable.

    ``create_thought`` does not own the outermost transaction here — the
    caller's own ``suspend_auto_commit()`` window does — so raising
    ``EmbeddingGenerationError`` does not commit the row on its own. If the
    caller lets that outer window's exit see the exception (the case here),
    the whole window rolls back and the thought never persists, contrary to
    the single-item-is-always-committed intuition that holds only when this
    call owns its own transaction.
    """
    store = await _embedding_store(db, _FailingProvider(), require_embedding=True)

    with pytest.raises(EmbeddingGenerationError):
        async with store.suspend_auto_commit():
            await store.create_thought(_thought("t-nested-strict"))

    # The outer window rolled back: the row was never made durable.
    assert await _count(db, "SELECT COUNT(*) FROM thought") == 0
    assert await _count(db, "SELECT COUNT(*) FROM embedding") == 0


async def test_update_reembed_failure_warns_and_propagates_by_default(
    db: aiosqlite.Connection,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A re-embed failure on ``update_thought`` warns (naming the id) and propagates.

    Editing essence/content re-embeds. The create path's no-silent-skip
    guarantee must hold on the update path too — and the torn write is worse
    here (the row previously *had* a valid embedding, now left stale).
    """
    store = await _embedding_store(db, _SpyProvider())  # working provider first
    await store.create_thought(_thought("t-upd-warn"))
    assert await _count(db, "SELECT COUNT(*) FROM embedding") == 1

    store._embedding_provider = _FailingProvider()  # provider goes offline
    with (
        caplog.at_level(logging.WARNING),
        pytest.raises(RuntimeError, match="provider offline"),
    ):
        await store.update_thought("t-upd-warn", content="new content forces a re-embed")

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("t-upd-warn" in r.getMessage() for r in warnings)


async def test_update_reembed_failure_raises_typed_under_strict(
    db: aiosqlite.Connection,
) -> None:
    """``require_embedding=True``: a re-embed failure on update raises the typed error."""
    store = await _embedding_store(db, _SpyProvider(), require_embedding=True)
    await store.create_thought(_thought("t-upd-strict"))

    store._embedding_provider = _FailingProvider()
    with pytest.raises(EmbeddingGenerationError) as exc_info:
        await store.update_thought("t-upd-strict", content="new content forces a re-embed")
    assert exc_info.value.thought_id == "t-upd-strict"


# ---------------------------------------------------------------------------
# Additive / no-regression: existing behaviour unchanged
# ---------------------------------------------------------------------------


async def test_create_thought_default_still_creates_duplicates(
    store: SqliteEngravaCore,
    db: aiosqlite.Connection,
) -> None:
    """Existing ``create_thought`` default (``deduplicate=False``) is unchanged."""
    content = "Legacy caller content inserted repeatedly."
    for i in range(3):
        await store.create_thought(_thought(f"t-legacy-{i}", content=content))
    assert await _count(db, "SELECT COUNT(*) FROM thought") == 3


async def test_deduplicate_true_behaviour_unchanged(
    store: SqliteEngravaCore,
    db: aiosqlite.Connection,
) -> None:
    """``deduplicate=True`` still collapses to one row and bumps confirmation."""
    content = "Dedup path must remain byte-identical to before."
    records = [
        await store.create_thought(_thought(f"t-dedup-{i}", content=content), deduplicate=True)
        for i in range(5)
    ]
    assert await _count(db, "SELECT COUNT(*) FROM thought") == 1
    assert records[-1].confirmation_count == 4


async def test_single_create_still_embeds_when_not_bulk(
    db: aiosqlite.Connection,
) -> None:
    """A normal ``create_thought`` under auto-embed still uses the single path."""
    provider = _SpyProvider()
    store = await _embedding_store(db, provider)

    await store.create_thought(_thought("t-single", content="single embed path"))

    # Single-item path: one ``embed`` call, zero batch calls.
    assert provider.embed_calls == 1
    assert provider.embed_batch_calls == 0
    assert await _count(db, "SELECT COUNT(*) FROM embedding") == 1


# ---------------------------------------------------------------------------
# Protocol + ReadOnly wrapper contract parity
# ---------------------------------------------------------------------------


def test_protocol_exposes_new_ingest_methods() -> None:
    """``EngravaCoreProtocol`` declares the new ingest primitives."""
    from engrava.domain.protocols.engrava_core import EngravaCoreProtocol

    for name in ("get_or_create", "upsert_by_hash", "bulk_store"):
        assert hasattr(EngravaCoreProtocol, name)


async def test_readonly_blocks_new_ingest_writes() -> None:
    """The read-only wrapper accepts the new signatures and raises cleanly."""
    from engrava.domain.exceptions import ReadOnlyViolationError
    from engrava.infrastructure.read_only_store import ReadOnlyEngrava

    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    try:
        inner = SqliteEngravaCore(conn)
        await inner.ensure_schema()
        ro = ReadOnlyEngrava(inner)
        record = _thought("t-ro")

        with pytest.raises(ReadOnlyViolationError):
            await ro.get_or_create(record)
        with pytest.raises(ReadOnlyViolationError):
            await ro.upsert_by_hash(record, expires_after_seconds=5)
        with pytest.raises(ReadOnlyViolationError):
            await ro.bulk_store([record], deduplicate=True)
    finally:
        await conn.close()
