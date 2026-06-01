"""Tests for opt-in content-hash deduplication on ``create_thought``.

Covers the ``deduplicate=True`` branch of
``SqliteEngravaCore.create_thought`` introduced for the deduplication fix.
The test file is large on purpose: every behavioural axis (default
preservation, hash determinism, unicode safety, concurrency, Pydantic
``frozen=True`` semantics, identity preservation, cross-thought-type
behaviour) gets its own focused case so a regression localises cleanly.
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import (
    CoreThoughtRecord,
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
    """Reusable ``SqliteEngravaCore`` bound to the in-memory DB."""
    s = SqliteEngravaCore(db)
    await s._probe_fts()
    return s


def _thought(
    thought_id: str,
    *,
    content: str = "The user prefers concise explanations over verbose ones.",
    thought_type: ThoughtType = ThoughtType.OBSERVATION,
    essence: str = "User preference for concision",
) -> CoreThoughtRecord:
    """Build a realistic ``CoreThoughtRecord`` for ingest tests."""
    return CoreThoughtRecord(
        thought_id=thought_id,
        thought_type=thought_type,
        essence=essence,
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


async def _count(db: aiosqlite.Connection, sql: str, *params: object) -> int:
    cursor = await db.execute(sql, params)
    row = await cursor.fetchone()
    assert row is not None
    return int(row[0])


# ---------------------------------------------------------------------------
# Hash determinism (1, 4, 5)
# ---------------------------------------------------------------------------


def test_content_hash_stable_across_runs() -> None:
    """``_compute_content_hash`` is deterministic for identical input."""
    from engrava.infrastructure.sqlite.engrava_core import _compute_content_hash

    content = "Recurring observation about a stable preference."
    a = _compute_content_hash(content)
    b = _compute_content_hash(content)

    assert a == b
    # Hex SHA-256 is 64 lowercase hex chars.
    assert len(a) == 64
    assert all(c in "0123456789abcdef" for c in a)


def test_content_hash_unicode_safe() -> None:
    """Hash handles emoji + CJK + RTL content deterministically."""
    from engrava.infrastructure.sqlite.engrava_core import _compute_content_hash

    cases = [
        "🌟 The user celebrated the milestone.",
        "用户偏好简洁解释而非冗长解释。",
        "המשתמש מעדיף הסברים תמציתיים.",
        "A normal ASCII observation.",
    ]
    digests = {c: _compute_content_hash(c) for c in cases}

    # All four are distinct (no collision on length-equal cases).
    assert len({*digests.values()}) == len(cases)
    # Each is a valid hex digest of expected length.
    for digest in digests.values():
        assert len(digest) == 64
    # Matches stdlib hashlib reference for at least one case.
    expected = hashlib.sha256(cases[0].encode("utf-8")).hexdigest()
    assert digests[cases[0]] == expected


# ---------------------------------------------------------------------------
# Default behaviour preserved (3, 11)
# ---------------------------------------------------------------------------


async def test_create_thought_default_deduplicate_false_creates_duplicates(
    store: SqliteEngravaCore,
    db: aiosqlite.Connection,
) -> None:
    """Default ``deduplicate=False`` creates a row per call (legacy behaviour)."""
    content = "An identical observation repeated by an old caller."
    for i in range(5):
        await store.create_thought(_thought(f"t-default-{i}", content=content))

    # Five rows, all sharing the same content_hash.
    total = await _count(db, "SELECT COUNT(*) FROM thought")
    distinct = await _count(db, "SELECT COUNT(DISTINCT content_hash) FROM thought")
    assert total == 5
    assert distinct == 1


async def test_dedup_disabled_creates_duplicates_explicit(
    store: SqliteEngravaCore,
    db: aiosqlite.Connection,
) -> None:
    """Explicit ``deduplicate=False`` is identical to the default."""
    content = "Explicitly-opt-out observation."
    for i in range(3):
        await store.create_thought(
            _thought(f"t-explicit-{i}", content=content),
            deduplicate=False,
        )
    assert await _count(db, "SELECT COUNT(*) FROM thought") == 3


# ---------------------------------------------------------------------------
# Dedup happy path (2, 7, 6)
# ---------------------------------------------------------------------------


async def test_dedup_creates_single_thought_for_identical_content(
    store: SqliteEngravaCore,
    db: aiosqlite.Connection,
) -> None:
    """12x ``deduplicate=True`` w identycznym content → 1 thought, count=12."""
    content = "Persona intro: Jordan Ellis, software architect, Berlin."
    persisted = [
        await store.create_thought(
            _thought(f"t-dedup-{i}", content=content),
            deduplicate=True,
        )
        for i in range(12)
    ]

    assert await _count(db, "SELECT COUNT(*) FROM thought") == 1
    # All returned references point at the same logical thought_id (the first one).
    assert all(p.thought_id == persisted[0].thought_id for p in persisted)
    # confirmation_count was bumped 11 times (initial insert at 0 + 11 dedup hits).
    assert persisted[-1].confirmation_count == 11
    db_count = await _count(
        db,
        "SELECT confirmation_count FROM thought WHERE thought_id = ?",
        persisted[0].thought_id,
    )
    assert db_count == 11


async def test_dedup_creates_separate_thoughts_for_different_content(
    store: SqliteEngravaCore,
    db: aiosqlite.Connection,
) -> None:
    """12x ``deduplicate=True`` z różnym content → 12 unique thoughts."""
    for i in range(12):
        await store.create_thought(
            _thought(f"t-distinct-{i}", content=f"Distinct observation #{i}."),
            deduplicate=True,
        )
    total = await _count(db, "SELECT COUNT(*) FROM thought")
    distinct_hashes = await _count(db, "SELECT COUNT(DISTINCT content_hash) FROM thought")
    assert total == 12
    assert distinct_hashes == 12


async def test_dedup_works_for_long_content(
    store: SqliteEngravaCore,
    db: aiosqlite.Connection,
) -> None:
    """Long content (~10kB) hashes + dedups correctly."""
    long_content = "Long observation. " * 600  # ~10kB ASCII
    await store.create_thought(_thought("t-long-1", content=long_content), deduplicate=True)
    await store.create_thought(_thought("t-long-2", content=long_content), deduplicate=True)

    assert await _count(db, "SELECT COUNT(*) FROM thought") == 1


# ---------------------------------------------------------------------------
# Cross thought_type semantics (7) + identity preservation (10)
# ---------------------------------------------------------------------------


async def test_dedup_collapses_across_thought_types(
    store: SqliteEngravaCore,
    db: aiosqlite.Connection,
) -> None:
    """content_hash collision across thought_types collapses to 1 thought.

    Content-hash deduplication discriminates by ``content`` alone (per
    spec §3.5: "exact same content").  When the same string is ingested
    once as OBSERVATION and once as REFLECTION with ``deduplicate=True``,
    the second call hits the existing row and bumps its
    ``confirmation_count`` rather than inserting a second one.  Callers
    that need type-discriminated semantics must use
    ``deduplicate=False`` (or scope their content text accordingly).
    """
    shared_content = "A statement that could legitimately exist in either layer."
    obs = await store.create_thought(
        _thought("t-obs", content=shared_content, thought_type=ThoughtType.OBSERVATION),
        deduplicate=True,
    )
    refl = await store.create_thought(
        _thought("t-refl", content=shared_content, thought_type=ThoughtType.REFLECTION),
        deduplicate=True,
    )

    assert obs.thought_id == refl.thought_id
    assert refl.confirmation_count == 1
    assert await _count(db, "SELECT COUNT(*) FROM thought") == 1


async def test_dedup_does_not_affect_thought_id(
    store: SqliteEngravaCore,
) -> None:
    """Dedup hit returns the existing thought_id (NIE generates new UUID)."""
    content = "Stable observation that should be reused."
    first = await store.create_thought(
        _thought("t-original", content=content),
        deduplicate=True,
    )
    again = await store.create_thought(
        _thought("t-totally-different-id", content=content),
        deduplicate=True,
    )
    assert again.thought_id == first.thought_id == "t-original"


# ---------------------------------------------------------------------------
# Pydantic frozen=True semantics (9)
# ---------------------------------------------------------------------------


async def test_existing_thought_via_dedup_returns_updated_pydantic_model(
    store: SqliteEngravaCore,
) -> None:
    """``model_copy(update=...)`` returns a new instance; original unchanged."""
    content = "Frozen-Pydantic semantic check."
    first = await store.create_thought(_thought("t-frozen", content=content), deduplicate=True)
    # Yield to the loop so the dedup-hit timestamp lands strictly after the
    # initial insert timestamp; without this nudge the two ``datetime.now(UTC)``
    # calls can land on the same microsecond on fast/warm Python runtimes
    # (Windows in particular has coarser default resolution under load).
    await asyncio.sleep(0.001)
    second = await store.create_thought(
        _thought("t-frozen-dup", content=content),
        deduplicate=True,
    )

    # Distinct instances, count bumped, updated_at refreshed forward.
    assert first is not second
    assert first.confirmation_count == 0
    assert second.confirmation_count == 1
    assert first.updated_at is not None
    assert second.updated_at is not None
    assert second.updated_at >= first.updated_at
    # Original thought is frozen — confirm we still cannot mutate it.
    with pytest.raises((TypeError, ValueError)):
        first.confirmation_count = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Concurrency safety (8)
# ---------------------------------------------------------------------------


async def test_dedup_concurrent_safety(
    store: SqliteEngravaCore,
    db: aiosqlite.Connection,
) -> None:
    """100 concurrent dedup calls collapse to 1 thought, count=100.

    Note: aiosqlite serialises SQL operations through a single
    background thread, so "concurrent" here exercises the asyncio
    scheduling layer rather than true OS-level concurrency.  The
    invariant we care about is that the check-then-update sequence
    converges deterministically under interleaving — which this test
    asserts: exactly one row, ``confirmation_count`` == fan-out - 1.
    """
    content = "Concurrently ingested identical content."
    fan_out = 100
    tasks = [
        store.create_thought(
            _thought(f"t-conc-{i}", content=content),
            deduplicate=True,
        )
        for i in range(fan_out)
    ]
    await asyncio.gather(*tasks)

    rows = await _count(db, "SELECT COUNT(*) FROM thought")
    assert rows == 1
    final_count = await _count(
        db,
        "SELECT confirmation_count FROM thought WHERE content_hash IS NOT NULL",
    )
    assert final_count == fan_out - 1


# ---------------------------------------------------------------------------
# content_hash NOT in Pydantic model (§3.4)
# ---------------------------------------------------------------------------


def test_content_hash_not_in_pydantic_model() -> None:
    """``content_hash`` is a SQL-only field — must not leak into ``ThoughtRecord``."""
    fields = set(CoreThoughtRecord.model_fields.keys())
    assert "content_hash" not in fields
    assert "content" in fields


# ---------------------------------------------------------------------------
# Protocol + ReadOnly wrapper contract parity (audit round 1)
# ---------------------------------------------------------------------------


def test_protocol_create_thought_accepts_deduplicate_kwarg() -> None:
    """``EngravaCoreProtocol.create_thought`` exposes the ``deduplicate`` kwarg.

    Without this, ingest-layer code typed against the public protocol
    cannot forward ``IngestConfig.deduplication_enabled`` to the store
    even though the concrete ``SqliteEngravaCore`` implementation
    accepts it.  Pinned as a regression test against
    audit-round-1 P2 finding.
    """
    import inspect

    from engrava.domain.protocols.engrava_core import EngravaCoreProtocol

    sig = inspect.signature(EngravaCoreProtocol.create_thought)
    assert "deduplicate" in sig.parameters
    param = sig.parameters["deduplicate"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is False
    # `from __future__ import annotations` keeps annotations as strings;
    # accept either the resolved type or its stringified form.
    assert param.annotation in {bool, "bool"}


async def test_readonly_create_thought_signature_parity_and_violation() -> None:
    """``ReadOnlyEngrava.create_thought`` accepts ``deduplicate`` and raises cleanly.

    Pinned as a regression test against audit-round-1 P2 finding —
    previously calling ``ro.create_thought(t, deduplicate=True)`` would
    surface a raw ``TypeError`` instead of the documented
    ``ReadOnlyViolationError``.
    """
    from engrava.domain.exceptions import ReadOnlyViolationError
    from engrava.infrastructure.read_only_store import ReadOnlyEngrava

    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    try:
        inner = SqliteEngravaCore(conn)
        await inner.ensure_schema()
        ro = ReadOnlyEngrava(inner)

        record = _thought("t-readonly")

        # Default kwargs path.
        with pytest.raises(ReadOnlyViolationError):
            await ro.create_thought(record)

        # New ``deduplicate`` kwarg must be accepted (signature parity)
        # AND still translate to ReadOnlyViolationError, not TypeError.
        with pytest.raises(ReadOnlyViolationError):
            await ro.create_thought(record, deduplicate=True)

        with pytest.raises(ReadOnlyViolationError):
            await ro.create_thought(record, deduplicate=False, expires_after_seconds=10)
    finally:
        await conn.close()
