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

import logging
from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import (
    CoreThoughtRecord,
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
