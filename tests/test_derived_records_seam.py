"""Integration tests for the derived-records extension seam.

Exercises the core-controlled, per-child, source-first, deferred, non-atomic
persistence of an extension's derived records against a live
``SqliteEngravaCore``. Each required behaviour of the seam is covered here or
in ``tests/domain/test_derived_records_types.py``:

* no demo-consumer / extension import in the core seam.
* the seam's public types add zero third-party dependencies.
* ``on_error="log"`` is ordinary logging, no telemetry surface.
* disabled path is byte-identical (thoughts + edges + journal).
* hooks without ``derive_records`` run byte-identical (protocol compat).
* the deterministic structural-split demo consumer.
* the recursion guard across single / bulk / get-or-create, incl. an
  adversarial producer that performs a nested public write.
* fail-open, cancellation propagation, per-family continuation.
* first-classness (embed/retrieve), conflict-as-reuse, and bounds.

The explicit ``derive_existing()`` backfill trigger — the on-store seam's
retroactive counterpart — is covered in its own section at the end of this file
(convergence with the on-store path, idempotency, the recursion guard, fail-open
isolation, capability-present gating independent of the enabled master switch,
typed not-found vs clean skip, and a non-LLM structural-split demonstration).
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import hashlib
import logging
import sqlite3
import struct
import threading
import time
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING

import aiosqlite
import pytest

import engrava.infrastructure.sqlite.engrava_core as core_module
from engrava import (
    ConnectionQuarantinedError,
    CoreThoughtRecord,
    DefaultEngravaHooks,
    DeriveContext,
    DerivedRecord,
    DerivedRecordError,
    DeriveGates,
    DeriveResult,
    EdgeType,
    LifecycleStatus,
    Priority,
    SourceThoughtNotFoundError,
    SqliteEngravaCore,
    StructuralSplitProducer,
    ThoughtType,
)
from engrava.embeddings.callback import CallbackProvider
from engrava.infrastructure.sqlite.engrava_core import (
    _DERIVED_ESSENCE_MAX_CHARS,
    _build_embed_input,
    _derived_edge_id,
    _derived_thought_id,
    _essence_from_content,
    _is_unique_violation,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from engrava.domain.models.thought import ThoughtRecord


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


@pytest.fixture
async def db() -> AsyncIterator[aiosqlite.Connection]:
    """Fresh in-memory SQLite with the core schema bootstrapped."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA foreign_keys = ON")
    boot = SqliteEngravaCore(conn)
    await boot.ensure_schema()
    yield conn
    await conn.close()


def _source(
    thought_id: str = "src-1",
    *,
    content: str = "Only one paragraph here.",
    created_at: str | None = None,
) -> CoreThoughtRecord:
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
        created_at=created_at,
        updated_at=created_at,
    )


def _child(content: str, *, attach_edge: bool = True) -> DerivedRecord:
    """Build a derived record with the given content."""
    return DerivedRecord(
        content=content,
        thought_type=ThoughtType.OBSERVATION,
        priority=Priority.P3,
        attach_provenance_edge=attach_edge,
    )


async def _count(db: aiosqlite.Connection, sql: str, *params: object) -> int:
    cursor = await db.execute(sql, params)
    row = await cursor.fetchone()
    assert row is not None
    return int(row[0])


async def _thought_rows(db: aiosqlite.Connection) -> list[tuple[object, ...]]:
    cursor = await db.execute(
        "SELECT thought_id, content, essence, priority, lifecycle_status, "
        "created_cycle, source, created_at, updated_at FROM thought ORDER BY thought_id",
    )
    return [tuple(row) for row in await cursor.fetchall()]


# ---------------------------------------------------------------------------
# Test producers
# ---------------------------------------------------------------------------


class ListProducer(DefaultEngravaHooks):
    """Return a fixed list of derived records; record call count and context."""

    def __init__(self, records: list[DerivedRecord]) -> None:
        self._records = records
        self.calls = 0
        self.last_ctx: DeriveContext | None = None
        self.source_ids: list[str] = []
        self.on_store_calls = 0

    async def on_store(self, thought: ThoughtRecord) -> ThoughtRecord:
        self.on_store_calls += 1
        return thought

    async def derive_records(
        self,
        thought: ThoughtRecord,
        ctx: DeriveContext,
    ) -> Sequence[DerivedRecord]:
        self.calls += 1
        self.last_ctx = ctx
        self.source_ids.append(ctx.source_thought_id)
        return self._records


class RaisingProducer(DefaultEngravaHooks):
    """Raise a producer-internal error from ``derive_records``."""

    def __init__(self) -> None:
        self.calls = 0

    async def derive_records(
        self,
        thought: ThoughtRecord,
        ctx: DeriveContext,
    ) -> Sequence[DerivedRecord]:
        self.calls += 1
        msg = "producer boom"
        raise RuntimeError(msg)


class CancellingProducer(DefaultEngravaHooks):
    """Raise ``CancelledError`` from ``derive_records``."""

    async def derive_records(
        self,
        thought: ThoughtRecord,
        ctx: DeriveContext,
    ) -> Sequence[DerivedRecord]:
        raise asyncio.CancelledError


class NestedWriteProducer(DefaultEngravaHooks):
    """Adversarial, contract-violating producer that issues a nested write.

    It performs a prohibited nested public write from inside ``derive_records``
    purely to prove the recursion guard holds (the nested write must not
    re-dispatch derivation). It does not endorse the behaviour.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.store: SqliteEngravaCore | None = None

    async def derive_records(
        self,
        thought: ThoughtRecord,
        ctx: DeriveContext,
    ) -> Sequence[DerivedRecord]:
        self.calls += 1
        assert self.store is not None
        nested = _source(
            f"nested-{self.calls}",
            content=f"nested content {self.calls}",
        )
        await self.store.create_thought(nested)
        return [_child(f"child of {thought.thought_id}")]


class _CountingSequence:
    """A lazy sequence that records how many items were pulled via iteration."""

    def __init__(self, items: list[DerivedRecord]) -> None:
        self._items = items
        self.pulled = 0

    def __iter__(self) -> object:
        for item in self._items:
            self.pulled += 1
            yield item

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> DerivedRecord:
        return self._items[index]


def _make_store(
    db: aiosqlite.Connection,
    hooks: DefaultEngravaHooks,
    gates: DeriveGates,
    **kwargs: object,
) -> SqliteEngravaCore:
    return SqliteEngravaCore(db, hooks=hooks, derive_gates=gates, **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Deterministic structural-split demo consumer
# ---------------------------------------------------------------------------


async def test_structural_split_derives_one_child_per_paragraph(
    db: aiosqlite.Connection,
) -> None:
    """The demo producer splits paragraphs into linked derived thoughts."""
    store = _make_store(
        db,
        StructuralSplitProducer(),
        DeriveGates(enabled=True),
    )
    await store.create_thought(
        _source(content="First paragraph.\n\nSecond paragraph.\n\nThird."),
    )

    assert await _count(db, "SELECT COUNT(*) FROM thought") == 4
    assert (
        await _count(
            db,
            "SELECT COUNT(*) FROM edge WHERE edge_type = ?",
            EdgeType.DERIVED_FROM.value,
        )
        == 3
    )
    edges = await store.get_edges("src-1", direction="IN")
    assert len(edges) == 3
    assert all(e.edge_type == EdgeType.DERIVED_FROM for e in edges)


async def test_structural_split_single_paragraph_derives_nothing(
    db: aiosqlite.Connection,
) -> None:
    """A single-segment source yields no derived records."""
    store = _make_store(db, StructuralSplitProducer(), DeriveGates(enabled=True))
    await store.create_thought(_source(content="Just one paragraph, nothing to split."))
    assert await _count(db, "SELECT COUNT(*) FROM thought") == 1


async def test_structural_split_is_idempotent_across_reruns(
    db: aiosqlite.Connection,
) -> None:
    """Re-deriving identical content reuses the same child rows (idempotent)."""
    store = _make_store(db, StructuralSplitProducer(), DeriveGates(enabled=True))
    content = "Alpha para.\n\nBeta para."
    await store.create_thought(_source("src-a", content=content))
    thoughts_after_first = await _count(db, "SELECT COUNT(*) FROM thought")

    # A second, distinct source with identical content: children collapse onto
    # the same rows (content-level identity), only the source row is added.
    await store.create_thought(_source("src-b", content=content))
    assert await _count(db, "SELECT COUNT(*) FROM thought") == thoughts_after_first + 1
    # Each source has its own provenance edges to the shared children.
    assert (
        await _count(
            db,
            "SELECT COUNT(*) FROM edge WHERE edge_type = ?",
            EdgeType.DERIVED_FROM.value,
        )
        == 4
    )


# ---------------------------------------------------------------------------
# Disabled + protocol-compat byte-identical paths
# ---------------------------------------------------------------------------


async def _insert_and_dump(
    hooks: DefaultEngravaHooks,
    gates: DeriveGates,
) -> tuple[list[tuple[object, ...]], int, int, int]:
    """Insert one fixed source and return (thought rows, #thoughts, #edges, #journal)."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA foreign_keys = ON")
    store = SqliteEngravaCore(conn, hooks=hooks, journal_enabled=True, derive_gates=gates)
    await store.ensure_schema()
    await store.create_thought(
        _source(content="Para A.\n\nPara B.", created_at="2020-01-01T00:00:00+00:00"),
    )
    rows = await _thought_rows(conn)
    thoughts = await _count(conn, "SELECT COUNT(*) FROM thought")
    edges = await _count(conn, "SELECT COUNT(*) FROM edge")
    journal = await _count(conn, "SELECT COUNT(*) FROM journal_entry")
    await conn.close()
    return rows, thoughts, edges, journal


async def test_disabled_seam_is_byte_identical() -> None:
    """A producer with the seam disabled matches a store without any producer."""
    baseline = await _insert_and_dump(DefaultEngravaHooks(), DeriveGates(enabled=False))
    with_producer = await _insert_and_dump(
        StructuralSplitProducer(),
        DeriveGates(enabled=False),
    )
    assert with_producer == baseline
    # And specifically: no derived rows, no edges beyond the single source.
    assert with_producer[1] == 1
    assert with_producer[2] == 0


async def test_absent_capability_is_byte_identical_even_when_enabled() -> None:
    """Hooks lacking ``derive_records`` are inert even with the seam enabled."""
    baseline = await _insert_and_dump(DefaultEngravaHooks(), DeriveGates(enabled=False))
    enabled_no_producer = await _insert_and_dump(
        DefaultEngravaHooks(),
        DeriveGates(enabled=True),
    )
    assert enabled_no_producer == baseline
    assert enabled_no_producer[1] == 1
    assert enabled_no_producer[2] == 0


async def test_existing_hooks_still_receive_on_store(db: aiosqlite.Connection) -> None:
    """An enabled producer's ``on_store`` still runs for the source only."""
    producer = ListProducer([_child("derived one")])
    store = _make_store(db, producer, DeriveGates(enabled=True))
    await store.create_thought(_source())
    # on_store fires exactly once — for the source, never for the derived child.
    assert producer.on_store_calls == 1


# ---------------------------------------------------------------------------
# Recursion guard (single / bulk / get-or-create + adversarial nested)
# ---------------------------------------------------------------------------


async def test_recursion_guard_single_blocks_nested_dispatch(
    db: aiosqlite.Connection,
) -> None:
    """A producer's nested public write does NOT re-dispatch derivation.

    Regression guard for the ContextVar recursion guard: reverting the guard
    would let the nested ``create_thought`` re-enter derivation, so
    ``derive_records`` would be called more than once (unbounded recursion) and
    this assertion — exactly one call — would fail.
    """
    producer = NestedWriteProducer()
    store = _make_store(db, producer, DeriveGates(enabled=True))
    producer.store = store

    await store.create_thought(_source())

    assert producer.calls == 1
    # source + one nested write + one derived child, and nothing recursed.
    assert await _count(db, "SELECT COUNT(*) FROM thought") == 3
    # The nested write produced no derived children of its own.
    nested_edges = await store.get_edges("nested-1", direction="IN")
    assert nested_edges == []


async def test_recursion_guard_bulk_dispatches_once_per_record(
    db: aiosqlite.Connection,
) -> None:
    """``bulk_store`` dispatches derivation per record, each guarded."""
    producer = ListProducer([_child("shared child")])
    store = _make_store(db, producer, DeriveGates(enabled=True))
    await store.bulk_store([_source("a"), _source("b")])
    # Two sources, each dispatched once; the shared child collapses to one row.
    assert producer.calls == 2
    assert await _count(db, "SELECT COUNT(*) FROM thought") == 3
    assert producer.last_ctx is not None
    assert producer.last_ctx.origin == "bulk_store"


async def test_get_or_create_dispatches_on_create_not_on_hit(
    db: aiosqlite.Connection,
) -> None:
    """``get_or_create`` derives on an actual create, never on a hash hit."""
    producer = ListProducer([_child("child of create")])
    store = _make_store(db, producer, DeriveGates(enabled=True))

    _, created = await store.get_or_create(_source(content="unique body"))
    assert created is True
    assert producer.calls == 1
    assert producer.last_ctx is not None
    assert producer.last_ctx.origin == "get_or_create"

    # Second identical call is a hit — no new derivation.
    _, created_again = await store.get_or_create(_source("src-2", content="unique body"))
    assert created_again is False
    assert producer.calls == 1


# ---------------------------------------------------------------------------
# Fail-open, cancellation, continuation
# ---------------------------------------------------------------------------


async def test_producer_error_raise_keeps_source_durable(
    db: aiosqlite.Connection,
) -> None:
    """``on_error='raise'`` re-raises but the source stays durable."""
    producer = RaisingProducer()
    store = _make_store(db, producer, DeriveGates(enabled=True, on_error="raise"))
    with pytest.raises(RuntimeError, match="producer boom"):
        await store.create_thought(_source())
    # Durability != API success: the source persisted despite the raise.
    assert await store.get_thought("src-1") is not None


async def test_producer_error_log_swallows_and_logs(
    db: aiosqlite.Connection,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``on_error='log'`` swallows the failure with ordinary logging."""
    producer = RaisingProducer()
    store = _make_store(db, producer, DeriveGates(enabled=True, on_error="log"))
    with caplog.at_level(logging.WARNING, logger=core_module.__name__):
        result = await store.create_thought(_source())
    assert result.thought_id == "src-1"
    assert await store.get_thought("src-1") is not None
    assert any(record.levelno == logging.WARNING for record in caplog.records)


@pytest.mark.parametrize("on_error", ["raise", "log"])
async def test_cancellation_propagates_regardless_of_policy(
    db: aiosqlite.Connection,
    on_error: str,
) -> None:
    """A cancelled ``derive_records`` propagates ``CancelledError`` either way."""
    store = _make_store(
        db,
        CancellingProducer(),
        DeriveGates(enabled=True, on_error=on_error),  # type: ignore[arg-type]
    )
    with pytest.raises(asyncio.CancelledError):
        await store.create_thought(_source())
    # Source is durable and there is no torn transaction.
    assert await store.get_thought("src-1") is not None


async def test_child_failure_log_continues_remaining(
    db: aiosqlite.Connection,
) -> None:
    """A per-child failure under ``on_error='log'`` continues remaining children."""
    # Force the middle child to collide with the source identity (a rejected
    # child), so it fails deterministically without touching the others.
    collide_content = "poison content"
    colliding_source_id = _derived_thought_id(collide_content)
    producer = ListProducer(
        [_child("first good"), _child(collide_content), _child("second good")],
    )
    store = _make_store(db, producer, DeriveGates(enabled=True, on_error="log"))
    await store.create_thought(_source(colliding_source_id, content="Body."))
    # source + two good children (the colliding one skipped, logged).
    assert await _count(db, "SELECT COUNT(*) FROM thought") == 3


async def test_child_failure_raise_aborts_remaining(
    db: aiosqlite.Connection,
) -> None:
    """A per-child failure under ``on_error='raise'`` aborts remaining children."""
    collide_content = "poison content"
    colliding_source_id = _derived_thought_id(collide_content)
    producer = ListProducer(
        [_child("first good"), _child(collide_content), _child("third never")],
    )
    store = _make_store(db, producer, DeriveGates(enabled=True, on_error="raise"))
    with pytest.raises(DerivedRecordError):
        await store.create_thought(_source(colliding_source_id, content="Body."))
    # source + first good child committed; third child never reached.
    assert await store.get_thought(colliding_source_id) is not None
    assert await _count(db, "SELECT COUNT(*) FROM thought") == 2


# ---------------------------------------------------------------------------
# First-classness, conflict-as-reuse, bounds
# ---------------------------------------------------------------------------


def _hash_embed(text: str) -> list[float]:
    """Deterministic, collision-resistant 8-dim embedding from the text."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [float(digest[i]) for i in range(8)]


async def test_derived_children_are_embedded_and_retrievable(
    db: aiosqlite.Connection,
) -> None:
    """Derived children run the ordinary lifecycle: embedded + retrievable."""
    provider = CallbackProvider(_hash_embed, dimension=8, model_name="hash-8")
    store = _make_store(
        db,
        StructuralSplitProducer(),
        DeriveGates(enabled=True),
        embedding_provider=provider,
        auto_embed=True,
    )
    await store.create_thought(_source(content="Head para.\n\nTail para."))

    edges = await store.get_edges("src-1", direction="IN")
    assert len(edges) == 2
    for edge in edges:
        child = await store.get_thought(edge.from_thought_id)
        assert child is not None
        # The child ran the ordinary auto-embed lifecycle: a vector is stored.
        assert await store.get_embedding(edge.from_thought_id) is not None


async def test_child_colliding_with_preexisting_row_is_reused(
    db: aiosqlite.Connection,
) -> None:
    """A child whose identity matches a pre-existing row reuses it + links it."""
    child_content = "Second para."
    preexisting_id = _derived_thought_id(child_content)
    # Pre-create a normal thought that occupies the derived child's identity.
    seed_store = SqliteEngravaCore(db)
    await seed_store.create_thought(
        CoreThoughtRecord(
            thought_id=preexisting_id,
            thought_type=ThoughtType.NOTE,
            essence="preexisting",
            content=child_content,
            priority=Priority.P1,
            lifecycle_status=LifecycleStatus.CREATED,
            created_cycle=0,
            updated_cycle=0,
            source="seed",
        ),
    )
    thoughts_before = await _count(db, "SELECT COUNT(*) FROM thought")

    store = _make_store(db, StructuralSplitProducer(), DeriveGates(enabled=True))
    await store.create_thought(_source(content="First para.\n\nSecond para."))

    # The colliding child was reused (not duplicated): only the source and the
    # one genuinely-new child were added.
    assert await _count(db, "SELECT COUNT(*) FROM thought") == thoughts_before + 2
    # The provenance edge to the reused row still exists.
    in_edges = await store.get_edges(preexisting_id, direction="OUT")
    assert any(e.to_thought_id == "src-1" for e in in_edges)


async def test_over_cap_return_is_rejected_before_any_write_raise(
    db: aiosqlite.Connection,
) -> None:
    """An over-cap return is rejected before any child is written (raise)."""
    producer = ListProducer([_child(f"c{i}") for i in range(5)])
    store = _make_store(
        db,
        producer,
        DeriveGates(enabled=True, on_error="raise", max_derived_per_source=3),
    )
    with pytest.raises(DerivedRecordError, match="max_derived_per_source"):
        await store.create_thought(_source())
    # Rejected before any write: only the source row exists.
    assert await _count(db, "SELECT COUNT(*) FROM thought") == 1


async def test_over_cap_return_is_rejected_before_any_write_log(
    db: aiosqlite.Connection,
) -> None:
    """An over-cap return is skipped entirely under ``on_error='log'``."""
    producer = ListProducer([_child(f"c{i}") for i in range(5)])
    store = _make_store(
        db,
        producer,
        DeriveGates(enabled=True, on_error="log", max_derived_per_source=3),
    )
    await store.create_thought(_source())
    assert await _count(db, "SELECT COUNT(*) FROM thought") == 1


async def test_lazy_sequence_is_bounded_to_cap_plus_one(
    db: aiosqlite.Connection,
) -> None:
    """Core pulls at most ``max_derived_per_source + 1`` items from the return."""
    items = [_child(f"lazy-{i}") for i in range(100)]
    sequence = _CountingSequence(items)

    class _LazyProducer(DefaultEngravaHooks):
        async def derive_records(
            self,
            thought: ThoughtRecord,
            ctx: DeriveContext,
        ) -> Sequence[DerivedRecord]:
            return sequence  # type: ignore[return-value]

    store = _make_store(
        db,
        _LazyProducer(),
        DeriveGates(enabled=True, on_error="log", max_derived_per_source=4),
    )
    await store.create_thought(_source())
    assert sequence.pulled <= 5


# ---------------------------------------------------------------------------
# Surface hygiene (no extension import in core; zero new deps)
# ---------------------------------------------------------------------------

_ALLOWED_TOP_MODULES = frozenset(
    {"__future__", "engrava", "dataclasses", "typing", "collections", "re"},
)


def test_seam_types_import_no_third_party_packages() -> None:
    """The seam's public types module depends only on stdlib + engrava."""
    source = Path(core_module.__file__).parent.parent.parent
    module_path = source / "domain" / "protocols" / "derived_records.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            top = node.module.split(".")[0]
            assert top in _ALLOWED_TOP_MODULES, f"unexpected import: {node.module}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] in _ALLOWED_TOP_MODULES


def test_core_seam_does_not_import_demo_consumer() -> None:
    """The core does not import the demo (or any) derived-record producer."""
    core_source = Path(core_module.__file__).read_text(encoding="utf-8")
    assert "structural_split" not in core_source
    assert "StructuralSplitProducer" not in core_source


# ---------------------------------------------------------------------------
# Core-derived essence (combining-mark-safe truncation)
# ---------------------------------------------------------------------------


def test_essence_from_short_content_is_verbatim() -> None:
    """Content within the essence bound is used unchanged."""
    assert _essence_from_content("short body") == "short body"


def test_essence_from_long_content_truncates_to_bound() -> None:
    """Long content truncates to the essence bound."""
    essence = _essence_from_content("a" * 500)
    assert essence == "a" * _DERIVED_ESSENCE_MAX_CHARS
    assert len(essence) == _DERIVED_ESSENCE_MAX_CHARS


def test_essence_truncation_does_not_sever_combining_mark() -> None:
    """A base+combining-mark cluster straddling the cut is not severed."""
    # 'e' lands at the last kept index and its combining acute (U+0301) at the
    # first dropped index; the cut must back off past the whole cluster.
    content = "a" * (_DERIVED_ESSENCE_MAX_CHARS - 1) + "é" + "tail"
    essence = _essence_from_content(content)
    assert "́" not in essence
    assert not unicodedata.combining(essence[-1])
    assert len(essence) <= _DERIVED_ESSENCE_MAX_CHARS
    # The derived essence remains a valid ThoughtRecord essence.
    assert 1 <= len(essence) <= _DERIVED_ESSENCE_MAX_CHARS


# ---------------------------------------------------------------------------
# Producer-sequence iteration failures are fail-open
# ---------------------------------------------------------------------------


class _RaisingSequence:
    """A lazy sequence that yields one item, then raises on the next pull."""

    def __init__(self, first: DerivedRecord) -> None:
        self._first = first

    def __iter__(self) -> object:
        yield self._first
        msg = "iteration boom"
        raise RuntimeError(msg)

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> DerivedRecord:
        raise IndexError(index)


class _LazyRaiseProducer(DefaultEngravaHooks):
    """Return a sequence that raises while being consumed."""

    async def derive_records(
        self,
        thought: ThoughtRecord,
        ctx: DeriveContext,
    ) -> Sequence[DerivedRecord]:
        return _RaisingSequence(_child("first lazy"))  # type: ignore[return-value]


async def test_sequence_iteration_error_log_swallows_source_durable(
    db: aiosqlite.Connection,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An error while iterating the producer result is swallowed under log."""
    store = _make_store(db, _LazyRaiseProducer(), DeriveGates(enabled=True, on_error="log"))
    with caplog.at_level(logging.WARNING, logger=core_module.__name__):
        await store.create_thought(_source())
    assert await store.get_thought("src-1") is not None
    assert await _count(db, "SELECT COUNT(*) FROM thought") == 1
    assert any(record.levelno == logging.WARNING for record in caplog.records)


async def test_sequence_iteration_error_raise_reraises_source_durable(
    db: aiosqlite.Connection,
) -> None:
    """An error while iterating the producer result re-raises under raise."""
    store = _make_store(db, _LazyRaiseProducer(), DeriveGates(enabled=True, on_error="raise"))
    with pytest.raises(RuntimeError, match="iteration boom"):
        await store.create_thought(_source())
    assert await store.get_thought("src-1") is not None
    assert await _count(db, "SELECT COUNT(*) FROM thought") == 1


# ---------------------------------------------------------------------------
# bulk_store derivation never rolls a committed source/child back
# ---------------------------------------------------------------------------


class RaiseOnNthProducer(DefaultEngravaHooks):
    """Derive a child on every call except the ``n``-th, where it raises."""

    def __init__(self, n: int, child_prefix: str) -> None:
        self._n = n
        self._prefix = child_prefix
        self.calls = 0

    async def derive_records(
        self,
        thought: ThoughtRecord,
        ctx: DeriveContext,
    ) -> Sequence[DerivedRecord]:
        self.calls += 1
        if self.calls == self._n:
            msg = "nth-record boom"
            raise RuntimeError(msg)
        return [_child(f"{self._prefix}-{self.calls}")]


async def test_bulk_derivation_failure_never_rolls_back_committed_state(
    db: aiosqlite.Connection,
) -> None:
    """A derivation failure on the 2nd bulk record leaves all committed state.

    Derivation runs only after the batch commits, off the batch transaction, so
    under ``on_error='raise'`` a failure on record 2 leaves BOTH sources durable
    and record 1's derived child durable — nothing is rolled back.
    """
    producer = RaiseOnNthProducer(2, "child")
    store = _make_store(db, producer, DeriveGates(enabled=True, on_error="raise"))
    with pytest.raises(RuntimeError, match="nth-record boom"):
        await store.bulk_store([_source("a", content="A body"), _source("b", content="B body")])
    # Both sources committed with the batch, before any derivation ran.
    assert await store.get_thought("a") is not None
    assert await store.get_thought("b") is not None
    # Record 1's derived child is durable; record 2's derivation raised.
    assert await store.get_thought(_derived_thought_id("child-1")) is not None


# ---------------------------------------------------------------------------
# Durability + recoverability of a committed-yet-unenriched child
# ---------------------------------------------------------------------------


async def test_embedding_failure_after_commit_recovers_on_rerun(
    db: aiosqlite.Connection,
) -> None:
    """A child committed before a failed embed is enriched on a later re-run."""
    state = {"fail": True}

    def _cb(text: str) -> list[float]:
        # Fail only for the derived children — the source's own embed input
        # carries its distinctive essence marker and must succeed, so the source
        # is genuinely durable+embedded before a child embed fails.
        if state["fail"] and "source essence" not in text:
            msg = "embed down"
            raise RuntimeError(msg)
        return _hash_embed(text)

    provider = CallbackProvider(_cb, dimension=8, model_name="toggle")
    store = _make_store(
        db,
        StructuralSplitProducer(),
        DeriveGates(enabled=True, on_error="log"),
        embedding_provider=provider,
        auto_embed=True,
    )
    await store.create_thought(_source(content="Head para.\n\nTail para."))

    child_ids = [_derived_thought_id("Head para."), _derived_thought_id("Tail para.")]
    # Children committed, but unenriched: embed failed before the edge, so no
    # embedding and no provenance edge yet.
    for cid in child_ids:
        assert await store.get_thought(cid) is not None
        assert await store.get_embedding(cid) is None
    assert await store.get_edges("src-1", direction="IN") == []

    # Fix the provider and re-run derivation for the SAME committed source.
    state["fail"] = False
    committed = await store.get_thought("src-1")
    assert committed is not None
    await store._dispatch_derivation(committed)

    for cid in child_ids:
        assert await store.get_embedding(cid) is not None
    assert len(await store.get_edges("src-1", direction="IN")) == 2


async def test_edge_failure_after_commit_recovers_on_rerun(
    db: aiosqlite.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child committed before a failed edge gets its edge on a later re-run."""
    store = _make_store(db, StructuralSplitProducer(), DeriveGates(enabled=True, on_error="log"))

    async def _flaky_edge(from_id: str, to_id: str, cycle: int) -> None:
        msg = "edge down"
        raise RuntimeError(msg)

    monkeypatch.setattr(store, "_insert_derived_edge", _flaky_edge)
    await store.create_thought(_source(content="Alpha.\n\nBeta."))
    child_ids = [_derived_thought_id("Alpha."), _derived_thought_id("Beta.")]
    for cid in child_ids:
        assert await store.get_thought(cid) is not None
    assert await store.get_edges("src-1", direction="IN") == []

    # Restore edge creation and re-run derivation for the same committed source.
    monkeypatch.undo()
    committed = await store.get_thought("src-1")
    assert committed is not None
    await store._dispatch_derivation(committed)
    assert len(await store.get_edges("src-1", direction="IN")) == 2


async def test_cancellation_after_child_commit_propagates_and_recovers(
    db: aiosqlite.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation during edge creation propagates; committed child recovers."""
    store = _make_store(db, StructuralSplitProducer(), DeriveGates(enabled=True, on_error="log"))

    async def _cancel_edge(from_id: str, to_id: str, cycle: int) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(store, "_insert_derived_edge", _cancel_edge)
    with pytest.raises(asyncio.CancelledError):
        await store.create_thought(_source(content="One.\n\nTwo."))

    # Source durable; the first child committed before the cancelled edge.
    assert await store.get_thought("src-1") is not None
    assert await store.get_thought(_derived_thought_id("One.")) is not None

    # Recover: re-run derivation for the same committed source completes enrichment.
    monkeypatch.undo()
    committed = await store.get_thought("src-1")
    assert committed is not None
    await store._dispatch_derivation(committed)
    assert len(await store.get_edges("src-1", direction="IN")) == 2


# ---------------------------------------------------------------------------
# Recursion guard across all write entry points; re-materialization paths
# ---------------------------------------------------------------------------


class NestedOpProducer(DefaultEngravaHooks):
    """Adversarial producer that issues a nested public write via a chosen op."""

    def __init__(self, op: str) -> None:
        self.op = op
        self.calls = 0
        self.store: SqliteEngravaCore | None = None

    async def derive_records(
        self,
        thought: ThoughtRecord,
        ctx: DeriveContext,
    ) -> Sequence[DerivedRecord]:
        self.calls += 1
        assert self.store is not None
        nested = _source(f"nested-{self.op}", content=f"nested body {self.op}")
        if self.op == "bulk":
            await self.store.bulk_store([nested])
        else:
            await self.store.upsert_by_hash(nested)
        return [_child(f"child of {thought.thought_id}")]


@pytest.mark.parametrize("op", ["bulk", "upsert"])
async def test_recursion_guard_holds_for_nested_entry_points(
    db: aiosqlite.Connection,
    op: str,
) -> None:
    """A nested ``bulk_store`` / ``upsert_by_hash`` never re-dispatches derivation.

    Reverting the guard would let the nested write re-enter derivation
    (`derive_records` called more than once → unbounded recursion), so the
    exactly-one-call assertion fails.
    """
    producer = NestedOpProducer(op)
    store = _make_store(db, producer, DeriveGates(enabled=True))
    producer.store = store

    await store.create_thought(_source())

    assert producer.calls == 1
    # The nested write landed but produced no derived children of its own.
    assert await store.get_thought(f"nested-{op}") is not None
    assert await store.get_edges(f"nested-{op}", direction="IN") == []


async def test_restore_thought_never_dispatches_derivation(
    db: aiosqlite.Connection,
) -> None:
    """A re-materialization path (restore) never triggers derivation."""
    producer = ListProducer([_child("only-on-create")])
    store = _make_store(db, producer, DeriveGates(enabled=True))
    await store.create_thought(_source(content="Body."))  # derives once
    await store.update_thought("src-1", lifecycle_status=LifecycleStatus.ACTIVE)
    await store.update_thought("src-1", lifecycle_status=LifecycleStatus.ARCHIVED)
    calls_before = producer.calls
    await store.restore_thought("src-1", current_cycle=1)
    # Restore re-materialises an existing record — it must not derive.
    assert producer.calls == calls_before == 1


def test_dispatch_derivation_has_exactly_two_call_sites() -> None:
    """Derivation is dispatched only from create_thought and the bulk post-commit loop.

    Guards against a re-materialization path (import / restore / replay / journal
    recovery) accidentally gaining a derivation dispatch.
    """
    core_source = Path(core_module.__file__).read_text(encoding="utf-8")
    assert core_source.count("await self._dispatch_derivation(") == 2


# ---------------------------------------------------------------------------
# A dedup / hash hit never dispatches derivation, embeddings OFF
# ---------------------------------------------------------------------------


async def test_bulk_dedup_hit_never_derives_with_embeddings_off(
    db: aiosqlite.Connection,
) -> None:
    """A dedup hit in bulk_store never derives, even with auto-embed off.

    Eligibility is decided by the actual insert outcome (a dedup / hash hit
    returns before dispatch), not by an embed-only row snapshot, so derivation
    fires for the genuinely-new record only.
    """
    producer = ListProducer([_child("only-new-derives")])
    store = _make_store(db, producer, DeriveGates(enabled=True))  # no embedding provider

    # Seed a row so the same content is a dedup hit later. This create derives once.
    await store.create_thought(_source("x-pre", content="EXISTING body"))
    assert producer.source_ids == ["x-pre"]

    # Bulk with a dedup hit (same content as the seed) + a genuinely-new record.
    await store.bulk_store(
        [_source("x-dup", content="EXISTING body"), _source("y-new", content="BRAND new body")],
        deduplicate=True,
    )
    # Derivation fired for the new record ONLY — never for the dedup hit.
    assert producer.source_ids == ["x-pre", "y-new"]


# ---------------------------------------------------------------------------
# Documented contract: a create inside a caller-held transaction does not
# auto-derive; derivation is triggered by an explicit re-run / backfill.
# ---------------------------------------------------------------------------


async def test_create_inside_caller_suspend_does_not_auto_derive(
    db: aiosqlite.Connection,
) -> None:
    """A create in a caller-held transaction does not auto-derive; backfill does.

    Documented contract: derivation fires only on a durably
    auto-committed create. A caller that writes inside its own
    ``suspend_auto_commit`` window owns that transaction, so the source is not
    yet durable and derivation is not dispatched — the caller triggers it with an
    explicit re-run / backfill once the transaction has committed.
    """
    producer = ListProducer([_child("backfilled child")])
    store = _make_store(db, producer, DeriveGates(enabled=True))

    async with store.suspend_auto_commit():
        await store.create_thought(_source(content="Body."))

    # No auto-derivation for a create made inside the caller's transaction.
    assert producer.calls == 0
    assert await store.get_thought(_derived_thought_id("backfilled child")) is None

    # Explicit backfill (re-run) after the transaction committed derives it.
    committed = await store.get_thought("src-1")
    assert committed is not None
    await store._dispatch_derivation(committed)
    assert producer.calls == 1
    assert await store.get_thought(_derived_thought_id("backfilled child")) is not None


async def test_caller_suspend_rollback_does_not_derive(
    db: aiosqlite.Connection,
) -> None:
    """A rolled-back transaction never derives (the source never became durable)."""
    producer = ListProducer([_child("never derived")])
    store = _make_store(db, producer, DeriveGates(enabled=True))

    boom = "roll it back"

    async def _create_then_fail() -> None:
        async with store.suspend_auto_commit():
            await store.create_thought(_source(content="Body."))
            raise RuntimeError(boom)

    with pytest.raises(RuntimeError, match=boom):
        await _create_then_fail()

    assert producer.calls == 0
    assert await store.get_thought("src-1") is None
    assert await store.get_thought(_derived_thought_id("never derived")) is None


# ---------------------------------------------------------------------------
# The derivation gate is asked of the current task's own window only,
# never of another task's, or a window this task's marker no longer names.
# ---------------------------------------------------------------------------


class _PausingOnStoreProducer(DefaultEngravaHooks):
    """Pauses ``on_store`` for one thought id; signals when ``derive_records`` runs."""

    def __init__(self) -> None:
        self.pause_for: str | None = None
        self.paused = asyncio.Event()
        self.release = asyncio.Event()
        self.derive_called = asyncio.Event()
        self.calls = 0
        self.source_ids: list[str] = []

    async def on_store(self, thought: ThoughtRecord) -> ThoughtRecord:
        if thought.thought_id == self.pause_for:
            self.paused.set()
            await self.release.wait()
        return thought

    async def derive_records(
        self,
        thought: ThoughtRecord,
        ctx: DeriveContext,
    ) -> Sequence[DerivedRecord]:
        self.calls += 1
        self.source_ids.append(ctx.source_thought_id)
        self.derive_called.set()
        return [_child(f"derived from {thought.thought_id}")]


async def test_foreign_window_does_not_skip_derivation(db: aiosqlite.Connection) -> None:
    """Another task's open ``suspend_auto_commit`` window must not blind derivation.

    Task A's create commits and then pauses inside ``on_store``. While paused,
    task B opens its own ``suspend_auto_commit`` window on the SAME store and
    holds it open. A resumes and finishes ``on_store`` while B's window is
    still open. The derivation gate must consult only A's own task-local
    window marker (``None`` — A never opened a window of its own), not the
    store-wide fact that *some* window happens to be open, so A's producer is
    called even while B's window is still open.

    Persisting the derived child then legitimately blocks on the write lock
    B's window holds for its whole duration (a genuinely different task's
    guarded write), so this test observes the producer call BEFORE releasing
    B, then releases B, then awaits both tasks — the order the corrected
    behaviour requires: awaiting A first (or releasing B before observing the
    call) would either race the assertion or deadlock A's derivation against
    B's own window.
    """
    producer = _PausingOnStoreProducer()
    producer.pause_for = "src-a"
    store = _make_store(db, producer, DeriveGates(enabled=True))

    task_a = asyncio.create_task(store.create_thought(_source("src-a", content="Body for A.")))
    try:
        await asyncio.wait_for(producer.paused.wait(), timeout=5.0)

        entered = asyncio.Event()
        leave = asyncio.Event()

        async def _hold_foreign_window() -> None:
            async with store.suspend_auto_commit():
                entered.set()
                await leave.wait()

        task_b = asyncio.create_task(_hold_foreign_window())
        try:
            await asyncio.wait_for(entered.wait(), timeout=5.0)

            producer.release.set()
            # RED (today's code): the producer is never called while a
            # foreign window is open, so this times out -- that timeout IS
            # the failure, not a hang.
            await asyncio.wait_for(producer.derive_called.wait(), timeout=5.0)

            # Proves the call happened while B's window was still open, not
            # because B had already exited by the time we checked.
            assert not task_b.done()

            leave.set()
            await asyncio.wait_for(task_b, timeout=5.0)
        except BaseException:
            if not task_b.done():
                task_b.cancel()
            raise
        await asyncio.wait_for(task_a, timeout=5.0)
    except BaseException:
        if not task_a.done():
            task_a.cancel()
        raise

    assert producer.calls == 1
    assert producer.source_ids == ["src-a"]
    assert await store.get_thought(_derived_thought_id("derived from src-a")) is not None


async def test_nested_window_skips_until_outermost_closes(db: aiosqlite.Connection) -> None:
    """A create between the inner window's exit and the outer's still skips.

    Each nesting level's own identity is registered on entry and unregistered
    in its own ``finally``, and the task-local marker is restored to the
    enclosing level's identity on the inner exit (a plain ``ContextVar.reset``)
    -- so a create made after the inner window has exited, but before the
    outer one has, still sees an open window of its own task's and skips.
    Once both have closed, the marker is back to ``None`` and a create derives
    normally again.
    """
    producer = ListProducer([_child("nested-child")])
    store = _make_store(db, producer, DeriveGates(enabled=True))

    async with store.suspend_auto_commit():
        await store.create_thought(_source("inner-src", content="Inside the inner window."))
        async with store.suspend_auto_commit():
            await store.create_thought(_source("innermost-src", content="Inside both windows."))
        # The inner window has exited; the outer one is still open.
        await store.create_thought(_source("after-inner-src", content="After inner, in outer."))

    assert producer.calls == 0
    # Both identities were discarded on their own `finally`, not merely
    # shadowed by the marker reset.
    assert store._open_auto_commit_windows == set()

    # Both windows are closed now -- an ordinary create derives again.
    await store.create_thought(_source("after-both-src", content="After both windows."))
    assert producer.calls == 1
    assert producer.source_ids == ["after-both-src"]


async def test_two_stores_one_task_window_isolation() -> None:
    """A window on store B must not overwrite store A's own marker for this task.

    The task-local marker is a ``ContextVar`` created per store instance, not
    at module level, precisely so one task holding windows on two different
    stores keeps each store's identity independent. A create on A still skips
    while A's own window is open, regardless of B's window nested inside it; a
    create on B skips while B's own window is open; and a create on either
    store, once both windows have exited, derives normally.
    """
    conn_a = await aiosqlite.connect(":memory:")
    conn_a.row_factory = aiosqlite.Row
    conn_b = await aiosqlite.connect(":memory:")
    conn_b.row_factory = aiosqlite.Row
    try:
        producer_a = ListProducer([_child("child-of-a")])
        producer_b = ListProducer([_child("child-of-b")])
        store_a = _make_store(conn_a, producer_a, DeriveGates(enabled=True))
        store_b = _make_store(conn_b, producer_b, DeriveGates(enabled=True))
        await store_a.ensure_schema()
        await store_b.ensure_schema()

        async with store_a.suspend_auto_commit():
            await store_a.create_thought(_source("a-1", content="On A, in A's window."))
            async with store_b.suspend_auto_commit():
                await store_a.create_thought(_source("a-2", content="On A, B's window nested."))
                await store_b.create_thought(_source("b-1", content="On B, in B's window."))
            # B's window has exited; A's own window is still open.
            await store_a.create_thought(_source("a-3", content="On A, after B's window closed."))

        assert producer_a.calls == 0
        assert producer_b.calls == 0
        assert store_a._open_auto_commit_windows == set()
        assert store_b._open_auto_commit_windows == set()

        # Both stores' windows are closed now -- ordinary creates derive.
        await store_a.create_thought(_source("a-4", content="On A, after both windows closed."))
        await store_b.create_thought(_source("b-2", content="On B, after both windows closed."))
        assert producer_a.source_ids == ["a-4"]
        assert producer_b.source_ids == ["b-2"]
    finally:
        await conn_a.close()
        await conn_b.close()


async def test_spawned_task_create_after_window_closes_derives_normally(
    db: aiosqlite.Connection,
) -> None:
    """A task spawned inside a window derives once the window it copied has closed.

    ``asyncio.create_task`` copies the current ``contextvars.Context``, so a
    task spawned from inside a ``suspend_auto_commit`` window starts with that
    window's identity as its own marker too. But the window's ``_write_lock``
    hold is task-scoped, not context-scoped: the spawned task is a genuinely
    different task, so its own create blocks on that lock until the window
    closes. By the time it resumes, the window's identity is no longer in the
    open set, so the spawned task's create derives normally -- exactly like
    any ordinary create made after the window.
    """
    producer = ListProducer([_child("spawned-child")])
    store = _make_store(db, producer, DeriveGates(enabled=True))

    entered = asyncio.Event()
    leave = asyncio.Event()
    spawned: asyncio.Task[ThoughtRecord] | None = None

    async def _hold_window_and_spawn() -> None:
        nonlocal spawned
        async with store.suspend_auto_commit():
            spawned = asyncio.create_task(
                store.create_thought(_source("spawned-src", content="Spawned body.")),
            )
            entered.set()
            await leave.wait()

    holder = asyncio.create_task(_hold_window_and_spawn())
    await asyncio.wait_for(entered.wait(), timeout=5.0)
    assert spawned is not None
    # Let the spawned task actually start and block on the write lock.
    await asyncio.sleep(0)
    assert not spawned.done()

    leave.set()
    await asyncio.wait_for(holder, timeout=5.0)
    await asyncio.wait_for(spawned, timeout=5.0)

    assert producer.calls == 1
    assert producer.source_ids == ["spawned-src"]


async def test_cancellation_inside_window_unregisters_and_resets_marker(
    db: aiosqlite.Connection,
) -> None:
    """A real ``task.cancel()`` delivered inside a window still resets the marker.

    ``suspend_auto_commit`` already catches ``BaseException`` — which is why a
    cancellation lands in the same ``finally`` as any other exception — so the
    window's identity must be unregistered AND this task's marker reset there
    too, exactly like a clean exit. A faulty cleanup that unregisters the
    identity but leaves the marker stale would still make a later create look
    correct from a *different* task (the marker check requires the marker to
    equal a specific window id, and a different task's marker was never set in
    the first place) -- so this test checks the marker directly, and checks it
    from **inside** the cancelled task itself, where the reset actually
    happens.

    The window runs in its own spawned task, genuinely cancelled via
    ``task.cancel()`` while parked on an event that never fires -- not a
    ``CancelledError`` raised inline -- so the delivery is the real thing.
    ``asyncio.create_task`` copies the current ``contextvars.Context``, so the
    cancelled task's own marker is a value the *parent* task's context never
    shares: the parent's marker reads ``None`` throughout regardless of
    whether cleanup ran correctly. The cancelled task therefore records its
    own observations, in a ``finally`` around the window, and re-raises so the
    parent still sees the propagated ``CancelledError``.
    """
    producer = ListProducer([_child("post-cancel-child")])
    store = _make_store(db, producer, DeriveGates(enabled=True))

    entered = asyncio.Event()
    unobserved: object = object()
    observed_marker: object | None = unobserved
    observed_open_windows: frozenset[object] | None = None

    async def _window_cancelled_from_outside() -> None:
        nonlocal observed_marker, observed_open_windows
        try:
            async with store.suspend_auto_commit():
                await store.create_thought(_source("in-window", content="Body inside window."))
                entered.set()
                await asyncio.Event().wait()  # never set; only cancellation ends this
        finally:
            # Observed from INSIDE the cancelled task, after
            # suspend_auto_commit's own `finally` has already run (its
            # `async with` block has exited by the time control reaches
            # here) -- this task's own contextvars.Context, not the parent's
            # copy of it.
            observed_marker = store._current_auto_commit_window.get()
            observed_open_windows = frozenset(store._open_auto_commit_windows)

    task = asyncio.create_task(_window_cancelled_from_outside())
    await asyncio.wait_for(entered.wait(), timeout=5.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5.0)

    # What the cancelled task itself observed, right after its own window's
    # cleanup ran.
    assert observed_marker is None, "the cancelled task's own marker was left stale"
    assert observed_open_windows == frozenset(), (
        "the cancelled window's identity was not unregistered"
    )

    assert not db.in_transaction
    assert producer.calls == 0  # the in-window create rolled back with the cancellation
    assert store._open_auto_commit_windows == set()

    # The parent task's own marker -- trivially `None` even under a faulty
    # cleanup, since a child task's context is a copy, never the parent's own.
    # Kept because it documents that fact rather than because it discriminates
    # anything on its own.
    assert store._current_auto_commit_window.get() is None

    # Still the parent task: a later create must derive, proving the
    # (now-finished) cancelled task's window identity was actually
    # unregistered store-wide, not merely invisible to the parent's marker.
    await store.create_thought(_source("post-cancel", content="After cancellation."))
    assert producer.calls == 1
    assert producer.source_ids == ["post-cancel"]


# ---------------------------------------------------------------------------
# Conflict-as-reuse enrichment targets the STORED row, never producer content
# ---------------------------------------------------------------------------


async def test_reuse_never_attaches_producer_content_vector_to_foreign_row(
    db: aiosqlite.Connection,
) -> None:
    """A reused row whose stored content differs is embedded from ITS OWN content."""
    provider = CallbackProvider(_hash_embed, dimension=8, model_name="hash-8")
    child_content = "Second para."
    foreign_id = _derived_thought_id(child_content)
    foreign_essence = "foreign essence"
    foreign_content = "A completely different stored body of text."

    # Pre-create an unembedded row occupying the derived child's deterministic id
    # but with DIFFERENT content (no provider on this seed store).
    seed = SqliteEngravaCore(db)
    await seed.create_thought(
        CoreThoughtRecord(
            thought_id=foreign_id,
            thought_type=ThoughtType.NOTE,
            essence=foreign_essence,
            content=foreign_content,
            priority=Priority.P1,
            lifecycle_status=LifecycleStatus.CREATED,
            created_cycle=0,
            updated_cycle=0,
            source="seed",
        ),
    )

    store = _make_store(
        db,
        StructuralSplitProducer(),
        DeriveGates(enabled=True),
        embedding_provider=provider,
        auto_embed=True,
    )
    await store.create_thought(_source(content="First para.\n\nSecond para."))

    emb = await store.get_embedding(foreign_id)
    assert emb is not None
    stored_vec = list(struct.unpack("8f", emb.vector_blob))

    producer_vec = _hash_embed(_build_embed_input(child_content, child_content))
    own_content_vec = _hash_embed(_build_embed_input(foreign_essence, foreign_content))
    # The vector reflects the reused row's own content, never the producer's.
    assert stored_vec != producer_vec
    assert stored_vec == own_content_vec


# ---------------------------------------------------------------------------
# Combining-mark truncation degenerate cases
# ---------------------------------------------------------------------------


def test_essence_leading_combining_run_falls_back_to_raw_truncation() -> None:
    """A run of combining marks spanning the boundary falls back to raw truncation.

    No non-combining base exists to cut after, so the best-effort truncation
    yields a non-empty preview (never a single detached mark).
    """
    content = "́" * 250 + "abcdef"
    essence = _essence_from_content(content)
    assert len(essence) == _DERIVED_ESSENCE_MAX_CHARS
    assert len(essence) >= 1


def test_essence_all_combining_short_content_is_verbatim() -> None:
    """Short all-combining content is previewed verbatim (no truncation)."""
    content = "́" * 5
    assert _essence_from_content(content) == content
    assert len(_essence_from_content(content)) >= 1


# ---------------------------------------------------------------------------
# R4 — per-child transaction isolation: a post-insert failure leaves no orphan
# ---------------------------------------------------------------------------

_THREE_PARAS = "P1 alpha.\n\nP2 beta.\n\nP3 gamma."
_SEGMENTS = ("P1 alpha.", "P2 beta.", "P3 gamma.")


def _journaled_seam_store(
    db: aiosqlite.Connection,
    on_error: str,
) -> SqliteEngravaCore:
    """A journaling store with the structural-split seam enabled."""
    return SqliteEngravaCore(
        db,
        hooks=StructuralSplitProducer(),
        journal_enabled=True,
        derive_gates=DeriveGates(enabled=True, on_error=on_error),  # type: ignore[arg-type]
    )


async def _file_journaled_seam_store(
    db_path: Path,
    on_error: str,
) -> tuple[SqliteEngravaCore, aiosqlite.Connection]:
    """A file-backed journaling seam store (survives a quarantine close).

    A quarantine hard-closes the connection, which destroys an in-memory DB, so
    durability-after-quarantine must be verified on disk via a fresh connection
    (:func:`_reopen_and_verify_durable`).
    """
    conn = await aiosqlite.connect(str(db_path))
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA foreign_keys = ON")
    store = SqliteEngravaCore(
        conn,
        hooks=StructuralSplitProducer(),
        journal_enabled=True,
        derive_gates=DeriveGates(enabled=True, on_error=on_error),  # type: ignore[arg-type]
    )
    await store.ensure_schema()
    return store, conn


async def _reopen_and_verify_durable(
    db_path: Path,
    *,
    present: list[str],
    absent: list[str],
) -> None:
    """Open a fresh connection to the on-disk DB and check thought durability."""
    conn = await aiosqlite.connect(str(db_path))
    try:
        conn.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(conn)
        for tid in present:
            assert await store.get_thought(tid) is not None
        for tid in absent:
            assert await store.get_thought(tid) is None
        assert (await store.verify_journal()).valid
    finally:
        await conn.close()


def _patch_journal_to_fail(
    store: SqliteEngravaCore,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mutation_type: str,
    target_id: str,
) -> None:
    """Make the journal writer raise for exactly one (mutation_type, target_id)."""
    assert store._journal is not None
    original = store._journal.append
    fail_mutation = mutation_type
    fail_target = target_id

    async def _flaky_append(
        mutation_type: str,
        target_id: str | None,
        delta: dict[str, object],
    ) -> object:
        if mutation_type == fail_mutation and target_id == fail_target:
            msg = "journal down"
            raise RuntimeError(msg)
        return await original(mutation_type=mutation_type, target_id=target_id, delta=delta)

    monkeypatch.setattr(store._journal, "append", _flaky_append)


def _patch_unwind_to_fail(
    store: SqliteEngravaCore,
    monkeypatch: pytest.MonkeyPatch,
    savepoint_name: str,
    unwind_exc: BaseException,
) -> None:
    """Make a ``_write_readback_savepoint`` unit's own ``ROLLBACK TO`` fail.

    Patches the store's connection so the literal ``ROLLBACK TO
    <savepoint_name>`` statement raises ``unwind_exc`` instead of running --
    simulating an unwind whose own recovery cannot be trusted -- while every
    other statement (including the failing write that triggers the unwind in
    the first place) executes normally.
    """
    real_execute = store._db.execute
    target_sql = f"ROLLBACK TO {savepoint_name}"

    async def _wrapper(sql: str, *args: object, **kwargs: object) -> object:
        if sql == target_sql:
            raise unwind_exc
        return await real_execute(sql, *args, **kwargs)

    monkeypatch.setattr(store._db, "execute", _wrapper)


def _spy_on_insert_derived_row(
    store: SqliteEngravaCore,
    monkeypatch: pytest.MonkeyPatch,
) -> list[str]:
    """Record every child id actually passed to ``_insert_derived_row``.

    A row's own absence after a failure proves the write was undone, but not
    that a *later* child was never attempted at all (an attempt can fail
    before writing anything). Wrapping the store's own bound method observes
    every call this instance makes, in order, regardless of whether that call
    goes on to write, fail, or never even reach the database.
    """
    seen: list[str] = []
    original = store._insert_derived_row

    async def _spy(child: ThoughtRecord) -> bool:
        seen.append(child.thought_id)
        return await original(child)

    monkeypatch.setattr(store, "_insert_derived_row", _spy)
    return seen


async def test_child_insert_journal_failure_log_leaves_no_orphan(
    db: aiosqlite.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A journal failure on one child's insert leaves NO committed orphan row."""
    store = _journaled_seam_store(db, "log")
    poison = _derived_thought_id(_SEGMENTS[1])
    _patch_journal_to_fail(store, monkeypatch, mutation_type="INSERT_THOUGHT", target_id=poison)

    await store.create_thought(_source(content=_THREE_PARAS))

    # The failing child left no trace; the source and the other children persist.
    assert await store.get_thought("src-1") is not None
    assert await store.get_thought(_derived_thought_id(_SEGMENTS[0])) is not None
    assert await store.get_thought(poison) is None
    assert await store.get_thought(_derived_thought_id(_SEGMENTS[2])) is not None
    # Two children processed to completion (edges present); the poisoned one none.
    in_edges = await store.get_edges("src-1", direction="IN")
    assert len(in_edges) == 2
    assert poison not in {edge.from_thought_id for edge in in_edges}
    # No row/edge without its journal entry.
    assert (await store.verify_journal()).valid


async def test_child_edge_journal_failure_log_leaves_row_but_no_edge(
    db: aiosqlite.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A journal failure on one child's edge leaves the row, no orphan edge."""
    store = _journaled_seam_store(db, "log")
    poison_child = _derived_thought_id(_SEGMENTS[1])
    poison_edge = _derived_edge_id(poison_child, "src-1")
    _patch_journal_to_fail(store, monkeypatch, mutation_type="INSERT_EDGE", target_id=poison_edge)

    await store.create_thought(_source(content=_THREE_PARAS))

    # All three child rows are durable (the insert step succeeded for each).
    for segment in _SEGMENTS:
        assert await store.get_thought(_derived_thought_id(segment)) is not None
    # The poisoned child's edge was rolled back — no orphan edge.
    in_edges = await store.get_edges("src-1", direction="IN")
    assert len(in_edges) == 2
    assert poison_child not in {edge.from_thought_id for edge in in_edges}
    assert (await store.verify_journal()).valid


async def test_child_insert_journal_failure_raise_aborts_remaining_no_orphan(
    db: aiosqlite.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under raise, a child journal failure aborts the rest, leaves no orphan."""
    store = _journaled_seam_store(db, "raise")
    poison = _derived_thought_id(_SEGMENTS[1])
    _patch_journal_to_fail(store, monkeypatch, mutation_type="INSERT_THOUGHT", target_id=poison)

    with pytest.raises(RuntimeError, match="journal down"):
        await store.create_thought(_source(content=_THREE_PARAS))

    # Source + the earlier child are durable; the failing child left no orphan;
    # the remaining child was never processed.
    assert await store.get_thought("src-1") is not None
    assert await store.get_thought(_derived_thought_id(_SEGMENTS[0])) is not None
    assert await store.get_thought(poison) is None
    assert await store.get_thought(_derived_thought_id(_SEGMENTS[2])) is None
    assert (await store.verify_journal()).valid


async def test_bulk_child_journal_failure_log_leaves_no_orphan(
    db: aiosqlite.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bulk post-commit dispatch isolates a child journal failure per-child."""
    store = _journaled_seam_store(db, "log")
    # Two genuinely-new sources, each deriving two children; poison one child of
    # the second record's derivation.
    poison = _derived_thought_id("B two.")
    _patch_journal_to_fail(store, monkeypatch, mutation_type="INSERT_THOUGHT", target_id=poison)

    await store.bulk_store(
        [
            _source("bulk-a", content="A one.\n\nA two."),
            _source("bulk-b", content="B one.\n\nB two."),
        ],
    )

    # Both sources and every non-poisoned child are durable.
    assert await store.get_thought("bulk-a") is not None
    assert await store.get_thought("bulk-b") is not None
    for segment in ("A one.", "A two.", "B one."):
        assert await store.get_thought(_derived_thought_id(segment)) is not None
    # The poisoned child left no orphan row, and no edge to its source.
    assert await store.get_thought(poison) is None
    assert len(await store.get_edges("bulk-a", direction="IN")) == 2
    assert len(await store.get_edges("bulk-b", direction="IN")) == 1
    assert (await store.verify_journal()).valid


async def test_cancellation_during_pending_child_insert_rolls_back(
    db: aiosqlite.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CancelledError after a child's INSERT (before its commit) rolls it back."""
    store = _journaled_seam_store(db, "log")
    poison = _derived_thought_id(_SEGMENTS[1])
    assert store._journal is not None
    original = store._journal.append

    async def _cancel_append(
        mutation_type: str,
        target_id: str | None,
        delta: dict[str, object],
    ) -> object:
        # The insert has already executed; raising here leaves it pending until
        # the per-child rollback discards it.
        if mutation_type == "INSERT_THOUGHT" and target_id == poison:
            raise asyncio.CancelledError
        return await original(mutation_type=mutation_type, target_id=target_id, delta=delta)

    monkeypatch.setattr(store._journal, "append", _cancel_append)

    with pytest.raises(asyncio.CancelledError):
        await store.create_thought(_source(content=_THREE_PARAS))

    # Cancellation propagated; the failing child left no committed row; the
    # source and the earlier committed child stay durable; journal stays valid.
    assert await store.get_thought("src-1") is not None
    assert await store.get_thought(_derived_thought_id(_SEGMENTS[0])) is not None
    assert await store.get_thought(poison) is None
    assert (await store.verify_journal()).valid


async def test_failed_unwind_aborts_derivation_raise_propagates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under ``on_error="raise"`` a failed per-child unwind propagates + quarantines.

    ``_insert_derived_row``'s own ``_write_readback_savepoint`` unit unwinds a
    failed insert with ``ROLLBACK TO`` + ``RELEASE``; when that unwind itself
    fails, recovery cannot be proven, so the connection is quarantined and the
    child's *original* error (not a wrapper) propagates. The source + earlier
    child committed before the failure stay durable on disk (verified via a
    fresh connection).
    """
    db_path = tmp_path / "seam.db"
    store, conn = await _file_journaled_seam_store(db_path, "raise")
    poison = _derived_thought_id(_SEGMENTS[1])  # the 2nd of three children
    _patch_journal_to_fail(store, monkeypatch, mutation_type="INSERT_THOUGHT", target_id=poison)
    _patch_unwind_to_fail(store, monkeypatch, "insert_derived_row", RuntimeError("rollback down"))

    with pytest.raises(RuntimeError, match="journal down"):
        await store.create_thought(_source(content=_THREE_PARAS))

    # The store is quarantined and any subsequent op fails fast — reverting the
    # quarantine-on-failed-unwind fix leaves the store usable and this fails.
    assert store._connection_quarantined is True
    with pytest.raises(ConnectionQuarantinedError):
        await store.get_thought("src-1")

    # The source + earlier committed child stay durable on disk; the poison
    # child's pending insert (discarded on close) and the never-processed third
    # child are absent; the journal stays valid.
    await _reopen_and_verify_durable(
        db_path,
        present=["src-1", _derived_thought_id(_SEGMENTS[0])],
        absent=[poison, _derived_thought_id(_SEGMENTS[2])],
    )
    await conn.close()  # already closed by quarantine; double close is a no-op


async def test_failed_unwind_aborts_derivation_log_propagates_after_logging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Under ``on_error="log"`` a failed per-child unwind still propagates.

    A quarantined connection cannot be trusted for the remaining children under
    either policy, so this is the one case where the fail-open ``"log"`` policy
    does not swallow the failure: it logs at ``ERROR`` first — naming the
    source, so an operator can find which one was orphaned — then still raises.
    This replaces the old ``_DerivationRollbackError`` rule, which quarantined
    the same way but stopped silently under ``"log"`` instead of propagating.
    """
    db_path = tmp_path / "seam.db"
    store, conn = await _file_journaled_seam_store(db_path, "log")
    poison = _derived_thought_id(_SEGMENTS[1])  # the 2nd of three children
    _patch_journal_to_fail(store, monkeypatch, mutation_type="INSERT_THOUGHT", target_id=poison)
    _patch_unwind_to_fail(store, monkeypatch, "insert_derived_row", RuntimeError("rollback down"))

    with (
        caplog.at_level(logging.ERROR, logger=core_module.__name__),
        pytest.raises(RuntimeError, match="journal down"),
    ):
        await store.create_thought(_source(content=_THREE_PARAS))
    assert any(record.levelno == logging.ERROR for record in caplog.records)
    assert any("src-1" in record.getMessage() for record in caplog.records)

    # The store is quarantined and a subsequent op fails fast.
    assert store._connection_quarantined is True
    with pytest.raises(ConnectionQuarantinedError):
        await store.get_thought("src-1")

    await _reopen_and_verify_durable(
        db_path,
        present=["src-1", _derived_thought_id(_SEGMENTS[0])],
        absent=[poison, _derived_thought_id(_SEGMENTS[2])],
    )
    await conn.close()


async def test_cancelled_child_with_failed_unwind_still_propagates_cancelled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancelled child whose unwind also fails propagates CancelledError + quarantines.

    ``CancelledError`` always propagates. Even when the unit's own unwind
    raises an ordinary exception trying to recover from it, the cancellation —
    not the unwind's own error — is what escapes; and the failed unwind still
    quarantines the store.
    """
    db_path = tmp_path / "seam.db"
    store, conn = await _file_journaled_seam_store(db_path, "log")
    poison = _derived_thought_id(_SEGMENTS[1])
    assert store._journal is not None
    original_append = store._journal.append

    async def _cancel_append(
        mutation_type: str,
        target_id: str | None,
        delta: dict[str, object],
    ) -> object:
        # Raise at a genuinely-pending point: after the child's INSERT executed.
        if mutation_type == "INSERT_THOUGHT" and target_id == poison:
            raise asyncio.CancelledError
        return await original_append(mutation_type=mutation_type, target_id=target_id, delta=delta)

    monkeypatch.setattr(store._journal, "append", _cancel_append)
    _patch_unwind_to_fail(store, monkeypatch, "insert_derived_row", RuntimeError("rollback down"))
    attempted = _spy_on_insert_derived_row(store, monkeypatch)

    with pytest.raises(asyncio.CancelledError):
        await store.create_thought(_source(content=_THREE_PARAS))

    assert store._connection_quarantined is True
    with pytest.raises(ConnectionQuarantinedError):
        await store.get_thought("src-1")

    # The never-processed third child was never even attempted -- a row's own
    # absence would not by itself rule out a failed attempt at it.
    assert _derived_thought_id(_SEGMENTS[2]) not in attempted
    assert attempted == [_derived_thought_id(_SEGMENTS[0]), poison]

    # It is absent on disk too: a quarantined connection is non-continuable,
    # so the dispatch must never reach it.
    await _reopen_and_verify_durable(
        db_path,
        present=["src-1", _derived_thought_id(_SEGMENTS[0])],
        absent=[poison, _derived_thought_id(_SEGMENTS[2])],
    )
    await conn.close()


async def test_cancelled_unwind_itself_still_propagates_cancelled_and_quarantines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancellation landing during the unwind itself still quarantines + propagates.

    Distinct from the previous test: there the *child's own write* was
    cancelled and the unwind's *recovery attempt* failed with an ordinary
    error; here the child's own write fails with an ordinary error and it is
    the unwind's ``ROLLBACK TO`` itself that is cancelled. Either shape must
    quarantine the connection (recovery cannot be proven either way), let the
    ``CancelledError`` win, and never reach the never-processed third child —
    checked on disk via a fresh connection, since quarantine hard-closes this
    one (an in-memory database would simply lose its data at that point).
    """
    db_path = tmp_path / "seam.db"
    store, conn = await _file_journaled_seam_store(db_path, "log")
    poison = _derived_thought_id(_SEGMENTS[1])
    _patch_journal_to_fail(store, monkeypatch, mutation_type="INSERT_THOUGHT", target_id=poison)
    _patch_unwind_to_fail(store, monkeypatch, "insert_derived_row", asyncio.CancelledError())
    attempted = _spy_on_insert_derived_row(store, monkeypatch)

    with pytest.raises(asyncio.CancelledError):
        await store.create_thought(_source(content=_THREE_PARAS))

    assert store._connection_quarantined is True
    with pytest.raises(ConnectionQuarantinedError):
        await store.get_thought("src-1")
    with pytest.raises(ConnectionQuarantinedError):
        await store.create_thought(_source("src-2", content="unrelated body"))

    # The never-processed third child was never even attempted -- a row's own
    # absence would not by itself rule out a failed attempt at it.
    assert _derived_thought_id(_SEGMENTS[2]) not in attempted
    assert attempted == [_derived_thought_id(_SEGMENTS[0]), poison]

    await _reopen_and_verify_durable(
        db_path,
        present=["src-1", _derived_thought_id(_SEGMENTS[0])],
        absent=[poison, _derived_thought_id(_SEGMENTS[2])],
    )
    await conn.close()  # already closed by quarantine; double close is a no-op


async def test_quarantined_store_commit_backstop_refuses(
    db: aiosqlite.Connection,
) -> None:
    """A quarantined store fails fast on the flag-guarded entry points.

    Guarded public entry points and ``_maybe_commit`` fail fast with the typed
    ``ConnectionQuarantinedError`` (good UX). The hard driver-level backstop (the
    closed connection) is exercised by
    :func:`test_quarantine_hard_invalidates_connection`.
    """
    store = SqliteEngravaCore(db)
    await store.ensure_schema()
    await store._quarantine_connection("simulated indeterminate rollback")

    with pytest.raises(ConnectionQuarantinedError):
        await store._maybe_commit()
    with pytest.raises(ConnectionQuarantinedError):
        await store.get_thought("anything")
    with pytest.raises(ConnectionQuarantinedError):
        await store.create_thought(_source(content="body"))
    await _drain_quarantine_close(store)


async def _drain_quarantine_close(store: SqliteEngravaCore) -> None:
    """Await the store's detached best-effort quarantine-close to completion."""
    task = store._quarantine_close_task
    assert task is not None
    await asyncio.gather(task, return_exceptions=True)


async def test_quarantine_hard_invalidates_connection(
    db: aiosqlite.Connection,
) -> None:
    """Quarantine swaps in a proxy so even a bypassing commit fails hard.

    ``_maybe_commit`` is only the fast typed-error path; the invariant is made
    robust-by-construction by replacing ``self._db`` with the terminal proxy. A
    direct ``self._db.commit()`` (the ~20 sites that bypass ``_maybe_commit``)
    then raises ``ConnectionQuarantinedError`` — reverting the proxy swap lets
    the direct commit run and this fails.
    """
    store = SqliteEngravaCore(db)
    await store.ensure_schema()
    await store._quarantine_connection("simulated indeterminate rollback")

    # Hard failure on a path that never consults the flag guard.
    with pytest.raises(ConnectionQuarantinedError):
        await store._db.commit()
    with pytest.raises(ConnectionQuarantinedError):
        await store._db.execute("SELECT 1")
    # ...but close() on the proxy is an idempotent no-op (graceful shutdown).
    await store._db.close()
    await _drain_quarantine_close(store)


async def test_quarantine_is_terminal_even_when_close_raises(
    db: aiosqlite.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(a) Close-independence: a failing physical close never weakens quarantine.

    Correctness must not depend on ``close()`` succeeding. Here ``close()``
    RAISES, yet the terminal proxy already occupies ``self._db``, so a subsequent
    direct ``commit`` / ``execute`` still raises ``ConnectionQuarantinedError``.
    Reverting the proxy swap leaves the (still open) real connection on
    ``self._db`` and the direct commit succeeds — which this test catches.
    """
    store = SqliteEngravaCore(db)
    await store.ensure_schema()

    async def _failing_close() -> None:
        msg = "close boom"
        raise RuntimeError(msg)

    monkeypatch.setattr(store._db, "close", _failing_close)

    await store._quarantine_connection("indeterminate")
    assert store._connection_quarantined is True

    # Close FAILED, yet the store is terminal by construction (the proxy).
    with pytest.raises(ConnectionQuarantinedError):
        await store._db.commit()
    with pytest.raises(ConnectionQuarantinedError):
        await store._db.execute("SELECT 1")
    await _drain_quarantine_close(store)  # consume the failed close task


async def test_quarantine_revokes_connection_holders_independent_of_close(
    db: aiosqlite.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(a) The shared token revokes the JournalWriter regardless of physical close.

    The ``JournalWriter`` keeps its own reference to the real connection, so the
    ``_db`` proxy swap alone would not stop it. The shared revocation token,
    revoked synchronously by quarantine, makes every journal op fail hard — even
    though ``close()`` here FAILS (the real connection stays open). Reverting the
    token revoke lets ``append`` run on the still-open real connection.

    (The vector backend is NOT wired to the token: it retains no connection
    handle and is always handed ``self._db`` by the core, i.e. the proxy
    post-quarantine — see the proxy tests above. So it is already covered.)
    """
    store = _journaled_seam_store(db, "log")  # journaling ON → real JournalWriter
    await store.ensure_schema()
    assert store._journal is not None

    async def _failing_close() -> None:
        msg = "close boom"
        raise RuntimeError(msg)

    monkeypatch.setattr(store._db, "close", _failing_close)

    await store._quarantine_connection("indeterminate")

    # The JournalWriter holds the real connection, but the token is revoked, so
    # every connection-touching journal op fails hard independent of the close.
    with pytest.raises(ConnectionQuarantinedError):
        await store._journal.append(
            mutation_type="INSERT_THOUGHT",
            target_id="x",
            delta={"before": None, "after": {}},
        )
    with pytest.raises(ConnectionQuarantinedError):
        await store._journal.verify_integrity()
    with pytest.raises(ConnectionQuarantinedError):
        await store._journal.get_entries()
    await _drain_quarantine_close(store)


async def test_quarantine_returns_promptly_when_close_hangs(
    db: aiosqlite.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(b) Liveness: a hung close cannot block quarantine (detached cleanup).

    Because safety is guaranteed by the proxy + token, the best-effort close is
    scheduled detached. Even a ``close()`` that never returns must not stop
    ``_quarantine_connection`` from returning promptly, and the store is terminal
    at once. Reverting to awaiting the close to completion makes this time out.
    """
    store = SqliteEngravaCore(db)
    await store.ensure_schema()

    release = asyncio.Event()

    async def _hung_close() -> None:
        await release.wait()  # never returns until the test releases it

    monkeypatch.setattr(store._db, "close", _hung_close)

    # Must return promptly despite the hung close (base-fail: awaiting it hangs).
    await asyncio.wait_for(store._quarantine_connection("indeterminate"), timeout=2.0)
    assert store._connection_quarantined is True
    with pytest.raises(ConnectionQuarantinedError):
        await store._db.commit()

    # Clean up the still-pending detached close task.
    release.set()
    await _drain_quarantine_close(store)


async def test_quarantine_survives_self_cancelled_close(
    db: aiosqlite.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A self-cancelled detached close task stays terminal and logs no failure.

    A close task that cancels itself would make ``task.exception()`` raise
    ``CancelledError`` in the done-callback — the ``cancelled()`` guard skips it.
    The store is terminal regardless (proxy + token).
    """
    store = SqliteEngravaCore(db)
    await store.ensure_schema()

    async def _self_cancelled_close() -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(store._db, "close", _self_cancelled_close)

    await store._quarantine_connection("indeterminate")
    assert store._connection_quarantined is True
    with pytest.raises(ConnectionQuarantinedError):
        await store._db.commit()
    await _drain_quarantine_close(store)


async def test_close_cancellation_does_not_corpse_the_physical_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancellation of close()'s own caller must not corpse the physical close.

    The branch that *creates* the close task (no prior quarantine -- this is
    an ordinary close with nothing else in the picture) used to await it with
    a bare ``await``, which forwards this call's own cancellation into the
    task. A single cancellation could then cancel the physical close before
    ``_db.close()`` had meaningfully run at all, yet leave a ``done()``
    (cancelled) task sitting in the shared slot -- every later ``close()`` or
    quarantine drain would see a completed task and report success without
    the real connection ever having closed.

    Draining under ``_drain_shielded`` closes this off structurally rather
    than papering over it: it re-shields on every repeated cancellation and
    never returns until the task is genuinely done, so the physical close
    always runs to real completion. This is the same ``_drain_shielded``
    helper used elsewhere for the identical reason (see its own docstring):
    a single cancellation is easy for a coroutine to absorb by accident; a
    *repeated* one is what actually exercises whether the shield holds.
    """
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    store = SqliteEngravaCore(conn)
    store._owns_connection = True
    await store.ensure_schema()

    close_calls = {"n": 0}
    real_close = conn.close
    started = asyncio.Event()
    release = asyncio.Event()
    completed = {"done": False}

    async def _slow_close() -> None:
        close_calls["n"] += 1
        started.set()
        await release.wait()
        await real_close()
        completed["done"] = True

    monkeypatch.setattr(conn, "close", _slow_close)

    task = asyncio.ensure_future(store.close())
    await started.wait()  # the physical close has started (and is now gated)
    task.cancel()  # cancel the caller while it awaits the physical close
    await asyncio.sleep(0)
    task.cancel()  # a repeated cancellation must not abort the shielded close either
    await asyncio.sleep(0)
    release.set()  # let the physical close finish
    with pytest.raises(asyncio.CancelledError):
        await task

    assert completed["done"] is True, (
        "the physical close must run to real completion despite the caller's cancellation"
    )
    assert close_calls["n"] == 1
    assert store._quarantine_close_task is not None
    assert not store._quarantine_close_task.cancelled(), (
        "the close task itself must not be cancelled -- only this caller's await was"
    )


async def test_close_cancellation_during_flush_still_closes_the_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancellation during the access-buffer flush must not skip the close.

    ``close()``'s cancellation-safety -- the quarantine / ``_drain_shielded``
    machinery exercised above -- only protects the physical close itself.
    The access-buffer flush that runs *before* any of that was guarded only
    by ``except Exception``, which does not catch ``asyncio.CancelledError``.
    A cancellation landing there used to escape immediately, skipping the
    close entirely and leaking the connection's non-daemon worker thread --
    recreating the exact interpreter-shutdown hang this whole fix exists to
    prevent, despite the docstring's promise that "a flush failure never
    blocks the close."
    """
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    store = SqliteEngravaCore(conn, access_tracking_enabled=True)
    store._owns_connection = True
    await store.ensure_schema()

    started = asyncio.Event()

    async def _flush_stalls_forever() -> int:
        started.set()
        await asyncio.sleep(10)  # cancelled long before this would return
        return 0

    monkeypatch.setattr(store, "flush_access_buffer", _flush_stalls_forever)

    task = asyncio.ensure_future(store.close())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    try:
        # close() awaits the worker's own stop future, so by the time the
        # physical close completes the thread should already be done; poll
        # briefly for the rare case where the OS thread's actual exit lags
        # a hair behind.
        deadline = time.monotonic() + 2.0
        while conn._thread.is_alive() and time.monotonic() < deadline:  # noqa: ASYNC110
            await asyncio.sleep(0.01)

        assert not conn._thread.is_alive(), (
            "the connection's non-daemon worker thread survived a "
            "cancellation during the access-buffer flush -- the close "
            "below it never ran"
        )
    finally:
        # However the assertion above turns out, never leave a leaked
        # worker thread running past this test -- it is not a daemon, so
        # it would otherwise block interpreter shutdown for the entire
        # suite.
        if conn._thread.is_alive():
            stopped = conn.stop()
            if stopped is not None:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(stopped, timeout=5)
            conn._thread.join(timeout=5)


async def test_close_cancellation_during_flush_outranks_a_failing_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deferred cancellation must win over the physical close's own failure.

    The sibling test above shows a cancelled flush still lets the physical
    close run. But ``self._quarantine_close_task.result()`` used to raise
    the close's own exception *before* the line that re-raises the
    deferred cancellation -- so a cancelled flush followed by a close that
    itself fails surfaced the close's ``RuntimeError``, not the caller's
    ``CancelledError``, contradicting this method's own documented promise
    that a cancellation of the caller's await always propagates ahead of
    whatever the close task resolved to. The rule
    ``_log_close_failure_over_pending_cancellation`` documents applies
    here: the caller's own cancellation outranks anything the cleanup
    discovers about itself.
    """
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    store = SqliteEngravaCore(conn, access_tracking_enabled=True)
    store._owns_connection = True
    await store.ensure_schema()

    started = asyncio.Event()

    async def _flush_stalls_forever() -> int:
        started.set()
        await asyncio.sleep(10)  # cancelled long before this would return
        return 0

    monkeypatch.setattr(store, "flush_access_buffer", _flush_stalls_forever)

    real_close = conn.close

    async def _close_blows_up() -> None:
        # Still performs the real close -- the point of this test is that
        # its own failure report afterward must not outrank the
        # already-pending cancellation.
        await real_close()
        msg = "close blew up after a cancellation was already pending"
        raise RuntimeError(msg)

    monkeypatch.setattr(conn, "close", _close_blows_up)

    task = asyncio.ensure_future(store.close())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    try:
        deadline = time.monotonic() + 2.0
        while conn._thread.is_alive() and time.monotonic() < deadline:  # noqa: ASYNC110
            await asyncio.sleep(0.01)

        assert not conn._thread.is_alive(), (
            "the connection's worker thread survived -- the close must "
            "still run to completion even when a cancellation is already "
            "pending"
        )
    finally:
        if conn._thread.is_alive():
            stopped = conn.stop()
            if stopped is not None:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(stopped, timeout=5)
            conn._thread.join(timeout=5)


async def test_close_after_quarantine_shares_the_close_task_and_returns_promptly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """close() after quarantine joins the same physical close, exactly once.

    Builds its own connection rather than using the shared ``db`` fixture:
    that fixture's own teardown does a bare ``await conn.close()`` on the
    real connection, which is a *third*, uncoordinated close attempt this
    test does not want in the picture.

    Quarantine schedules its own detached close of the real connection
    first. ``close()`` explicitly awaits quarantine's own close task, so it
    must not return before the physical close is actually finished.

    Asserted by **gating the physical close and checking that ``close()``
    is still pending**, not by counting how many times it ran: a counter
    can be satisfied by quarantine's own detached task completing on its
    own schedule, regardless of whether ``close()`` ever joined it, which
    would be a scheduling-dependent false pass. Checking that ``close()``
    remains pending until the gate is explicitly released tests the
    property directly.
    """
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    store = SqliteEngravaCore(conn)
    store._owns_connection = True
    await store.ensure_schema()

    close_calls = {"n": 0}
    real_close = conn.close
    close_started = asyncio.Event()
    close_may_finish = asyncio.Event()

    async def _gated_close() -> None:
        close_calls["n"] += 1
        close_started.set()
        await close_may_finish.wait()
        await real_close()

    monkeypatch.setattr(conn, "close", _gated_close)

    await store._quarantine_connection("indeterminate")
    assert store._connection_quarantined is True

    close_task = asyncio.ensure_future(store.close())
    await asyncio.wait_for(close_started.wait(), timeout=5.0)
    assert not close_task.done(), (
        "close() must wait for the physical close to finish, not return once it has merely started"
    )

    close_may_finish.set()
    await asyncio.wait_for(close_task, timeout=5.0)

    assert close_calls["n"] == 1, "the real connection must be physically closed exactly once"


async def test_quarantine_during_an_in_flight_close_does_not_double_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A quarantine racing an in-flight close() must not enter the real close twice.

    Reproduces the exact shape confirmed on a real database: ``close()`` has
    already entered the real connection's close (published as the shared
    ``_quarantine_close_task`` before awaiting it), and quarantine runs while
    that is still in flight. On the pinned aiosqlite version two concurrent
    physical closes on the same connection each enqueue their own stop
    sentinel to the worker thread, which exits on the first and can leave
    the other caller's future unresolved forever -- a hang. The final await
    is timeout-guarded so a regression here fails this test instead of
    stalling the suite.

    Builds its own connection rather than using the shared ``db`` fixture —
    see the previous test's docstring for why.
    """
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    store = SqliteEngravaCore(conn)
    store._owns_connection = True
    await store.ensure_schema()

    close_calls = {"n": 0}
    real_close = conn.close
    close_started = asyncio.Event()
    close_may_finish = asyncio.Event()

    async def _gated_close() -> None:
        close_calls["n"] += 1
        close_started.set()
        await close_may_finish.wait()
        await real_close()

    monkeypatch.setattr(conn, "close", _gated_close)

    close_task = asyncio.ensure_future(store.close())
    await asyncio.wait_for(close_started.wait(), timeout=5.0)
    assert store._quarantine_close_task is not None, (
        "close() must publish the shared close task before awaiting the physical close"
    )

    # Quarantine runs while close() is still mid-flight, gated on the same
    # real close -- it must not start a second one.
    await asyncio.wait_for(store._quarantine_connection("indeterminate"), timeout=5.0)
    assert store._connection_quarantined is True

    close_may_finish.set()
    await asyncio.wait_for(close_task, timeout=5.0)

    assert close_calls["n"] == 1, "the real connection must be physically closed exactly once"


async def _wedge_the_worker_thread(
    conn: aiosqlite.Connection,
) -> tuple[asyncio.Future, threading.Event]:
    """Genuinely block the aiosqlite worker thread, not merely slow it down.

    A monkeypatched ``async def`` "slow close" (used by the tests above)
    only ever blocks at the ``asyncio`` layer -- the underlying worker
    thread is idle the whole time and would pick up a queued stop sentinel
    instantly. That is enough to test bounded *observation*, but the bound
    this exercises is specifically meant to survive a worker that will
    never answer at all, and the two are not the same failure to construct.

    This registers a real SQLite user-defined function that, once invoked,
    blocks the calling thread on a ``threading.Event`` -- a synchronous,
    OS-level block with no ``await`` anywhere nothing in ``asyncio`` can
    reach or cancel. Firing it via ``conn.execute(...)`` without awaiting
    the result queues that call onto aiosqlite's single worker thread
    exactly like any other statement; once the thread picks it up, it is
    genuinely stuck there -- every later item on the same queue, including
    ``close()``'s own stop request, waits behind it with no supported way
    to interrupt it, matching the shape a truly wedged worker takes in
    production far more closely than an ``asyncio``-level mock does.

    Returns:
        The in-flight ``execute()`` future (still pending) and the
        ``threading.Event`` the caller must ``.set()`` to unwedge the
        worker and let this test clean up after itself.

    """
    entered = threading.Event()
    release = threading.Event()

    def _wedge() -> int:
        entered.set()
        release.wait()  # blocks the real OS thread until the test releases it
        return 0

    await conn.create_function("wedge_worker_thread", 0, _wedge)
    wedge_future = asyncio.ensure_future(conn.execute("SELECT wedge_worker_thread()"))
    # threading.Event.wait() is a blocking call -- polling .is_set() keeps
    # this coroutine, and the event loop it runs on, from blocking too. Not
    # an asyncio.Event: the signal crosses from the worker's real OS thread,
    # which cannot set an asyncio primitive directly.
    while not entered.is_set():  # noqa: ASYNC110
        await asyncio.sleep(0.001)
    return wedge_future, release


async def test_close_bound_expires_on_a_genuinely_unresponsive_worker() -> None:
    """The first close() on a genuinely wedged worker returns bounded, not never.

    Before this bound existed, ``close()`` awaited the physical close with no
    limit at all: queued behind a worker that will never answer, it would
    never return -- the exact failure this test constructs for real via
    :func:`_wedge_the_worker_thread`, rather than a finite, merely-slow mock.
    A bound that only proved itself against a mock that always eventually
    returns would not actually prove anything about the unbounded case.
    """
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    store = SqliteEngravaCore(conn, close_timeout_seconds=0.2)
    store._owns_connection = True
    await store.ensure_schema()

    wedge_future, release = await _wedge_the_worker_thread(conn)
    try:
        started = time.monotonic()
        with pytest.raises(ConnectionQuarantinedError):
            await asyncio.wait_for(store.close(), timeout=5.0)
        elapsed = time.monotonic() - started

        assert elapsed < 2.0, (
            "close() must return within its own bound, not wait out the "
            "unresponsive worker -- the outer wait_for(timeout=5.0) is only "
            "a suite-safety net, not the property under test"
        )
        assert store._connection_quarantined is True
        with pytest.raises(ConnectionQuarantinedError):
            await store._db.commit()
    finally:
        # Release the wedged thread and drain everything queued behind it
        # (the wedge call itself, and close()'s own stop request) so
        # nothing outlives this test -- a genuinely stuck worker that is
        # never released leaks a live, non-daemon thread for as long as the
        # process runs.
        release.set()
        with contextlib.suppress(BaseException):
            await asyncio.wait_for(wedge_future, timeout=5.0)
        close_task = store._quarantine_close_task
        assert close_task is not None
        with contextlib.suppress(BaseException):
            await asyncio.wait_for(close_task, timeout=5.0)
        deadline = time.monotonic() + 5.0
        while conn._thread.is_alive() and time.monotonic() < deadline:  # noqa: ASYNC110
            await asyncio.sleep(0.01)
        assert not conn._thread.is_alive(), (
            "the aiosqlite worker thread survived past the test -- it is "
            "not a daemon and would otherwise block interpreter shutdown "
            "for the rest of the suite"
        )


async def test_close_bound_expires_again_on_a_second_call_without_a_second_physical_close() -> None:
    """A second close() on the same wedge is bounded too, and starts nothing new.

    The first close() expiring is the easy half. This covers the amendment's
    harder requirement: a caller that calls close() again after the first
    one gave up must not trigger a second physical close on the pinned
    connection (two concurrent closes can each enqueue their own stop
    sentinel, and the worker only ever answers the first -- see close()'s
    own docstring), and must still get its own call bounded rather than
    inheriting an unbounded wait on whatever the first call started.
    """
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    store = SqliteEngravaCore(conn, close_timeout_seconds=0.2)
    store._owns_connection = True
    await store.ensure_schema()

    wedge_future, release = await _wedge_the_worker_thread(conn)
    try:
        with pytest.raises(ConnectionQuarantinedError):
            await asyncio.wait_for(store.close(), timeout=5.0)
        first_close_task = store._quarantine_close_task
        assert first_close_task is not None

        started = time.monotonic()
        with pytest.raises(ConnectionQuarantinedError):
            await asyncio.wait_for(store.close(), timeout=5.0)
        elapsed = time.monotonic() - started

        assert elapsed < 2.0, "the second close() must also be bounded, not inherit an open wait"
        assert store._quarantine_close_task is first_close_task, (
            "a second close() must piggyback on the same physical-close task "
            "-- a different task here would mean a second, independent "
            "close() was issued against the same underlying connection"
        )
    finally:
        release.set()
        with contextlib.suppress(BaseException):
            await asyncio.wait_for(wedge_future, timeout=5.0)
        close_task = store._quarantine_close_task
        assert close_task is not None
        with contextlib.suppress(BaseException):
            await asyncio.wait_for(close_task, timeout=5.0)
        deadline = time.monotonic() + 5.0
        while conn._thread.is_alive() and time.monotonic() < deadline:  # noqa: ASYNC110
            await asyncio.sleep(0.01)
        assert not conn._thread.is_alive(), (
            "the aiosqlite worker thread survived past the test -- it is "
            "not a daemon and would otherwise block interpreter shutdown "
            "for the rest of the suite"
        )


async def test_close_cancelled_while_draining_a_wedged_worker_still_quarantines() -> None:
    """A caller cancelled while draining the task must still see the store quarantined.

    ``_drain_shielded`` does not return early when *our* await of it is
    cancelled: it re-shields and keeps waiting until its own bounded
    ``asyncio.wait(...)`` completes. Against a genuinely wedged worker, that
    bound then expires with the task still not ``done()`` -- so this call to
    ``close()`` can end up with a non-``None`` cancellation *and* a genuine,
    unrelated expiry at the same time. The expiry must still reach
    ``_abandon_expired_close`` (quarantine + log) regardless: cancellation
    precedence decides which exception propagates out of ``close()``, never
    whether the store gets quarantined.
    """
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    store = SqliteEngravaCore(conn, close_timeout_seconds=0.2)
    store._owns_connection = True
    await store.ensure_schema()

    wedge_future, release = await _wedge_the_worker_thread(conn)
    try:
        close_task = asyncio.ensure_future(store.close())
        await asyncio.sleep(0.01)  # let close() start draining the physical-close task
        close_task.cancel()

        started = time.monotonic()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(close_task, timeout=5.0)
        elapsed = time.monotonic() - started

        assert elapsed < 2.0, (
            "the cancelled close() must still return once its own bound "
            "expires, not wait out the unresponsive worker indefinitely"
        )
        assert store._connection_quarantined is True, (
            "an expired bound must quarantine the store even when this "
            "call's own await was cancelled while draining the task"
        )
        with pytest.raises(ConnectionQuarantinedError):
            await store._db.commit()
    finally:
        release.set()
        with contextlib.suppress(BaseException):
            await asyncio.wait_for(wedge_future, timeout=5.0)
        close_task = store._quarantine_close_task
        assert close_task is not None
        with contextlib.suppress(BaseException):
            await asyncio.wait_for(close_task, timeout=5.0)
        deadline = time.monotonic() + 5.0
        while conn._thread.is_alive() and time.monotonic() < deadline:  # noqa: ASYNC110
            await asyncio.sleep(0.01)
        assert not conn._thread.is_alive(), (
            "the aiosqlite worker thread survived past the test -- it is "
            "not a daemon and would otherwise block interpreter shutdown "
            "for the rest of the suite"
        )


async def test_close_bound_covers_the_access_buffer_flush_too() -> None:
    """A wedged worker cannot be reached only through the flush step either.

    ``close()`` flushes the access buffer *before* it ever reaches the
    bounded physical-close drain -- and that flush awaits the same worker
    directly (``_write_lock`` + a raw ``executemany``), with no bound of its
    own until this test's own fix. Access tracking defaults to ``False`` on
    the manual constructor, but ``from_config`` turns it on whenever dreaming
    is enabled, and ``DreamingConfig.access_tracking_enabled`` itself
    defaults to ``True`` -- so a store built the way most deployments build
    one (dreaming on, nothing said about access tracking) would have hung
    here even with the physical-close bound in place, exactly the case
    docs/deployment.md promises is covered. Executed with a genuinely wedged
    worker, not reasoned about: a buffer entry is seeded directly so the
    flush actually reaches ``executemany`` instead of returning early on an
    empty buffer.
    """
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    store = SqliteEngravaCore(conn, access_tracking_enabled=True, close_timeout_seconds=0.2)
    store._owns_connection = True
    await store.ensure_schema()
    store._access_buffer.record("some-thought-id", now="2026-01-01T00:00:00+00:00")
    assert len(store._access_buffer) == 1

    wedge_future, release = await _wedge_the_worker_thread(conn)
    try:
        started = time.monotonic()
        with pytest.raises(ConnectionQuarantinedError):
            # The outer wait_for is only a suite-safety net (per the sibling
            # tests above) -- without the flush bound, this would need it to
            # actually fire, which is exactly the regression this guards.
            await asyncio.wait_for(store.close(), timeout=5.0)
        elapsed = time.monotonic() - started

        # Worst case here is the flush's own bound plus the physical-close
        # bound run back to back (~0.4s at this test's 0.2s setting) --
        # comfortably under any margin that would indicate an unbounded
        # wait slipped through.
        assert elapsed < 2.0, (
            "close() must not hang in the access-buffer flush before it "
            "ever reaches the bounded physical close"
        )
        assert store._connection_quarantined is True
    finally:
        release.set()
        with contextlib.suppress(BaseException):
            await asyncio.wait_for(wedge_future, timeout=5.0)
        close_task = store._quarantine_close_task
        if close_task is not None:
            with contextlib.suppress(BaseException):
                await asyncio.wait_for(close_task, timeout=5.0)
        deadline = time.monotonic() + 5.0
        while conn._thread.is_alive() and time.monotonic() < deadline:  # noqa: ASYNC110
            await asyncio.sleep(0.01)
        assert not conn._thread.is_alive(), (
            "the aiosqlite worker thread survived past the test -- it is "
            "not a daemon and would otherwise block interpreter shutdown "
            "for the rest of the suite"
        )


async def test_cancelled_unwinds_own_closing_rollback_quarantines_and_propagates(
    db: aiosqlite.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancellation on the unwind's own closing ``rollback()`` quarantines + propagates.

    Distinct from ``test_cancelled_unwind_itself_still_propagates_cancelled_and_quarantines``:
    that one cancels the unwind's ``ROLLBACK TO`` (the savepoint step); this one
    cancels the plain ``rollback()`` the unit calls afterward to close the
    transaction it opened for the failing insert. Both are statements inside
    the same unwind ``try`` block in ``_write_readback_savepoint``, and either
    one raising -- cancellation or not -- must quarantine the connection and
    let the ``CancelledError`` win.
    """
    store = _journaled_seam_store(db, "log")
    poison = _derived_thought_id(_SEGMENTS[1])
    # The child fails with an ordinary error so it enters the unwind path...
    _patch_journal_to_fail(store, monkeypatch, mutation_type="INSERT_THOUGHT", target_id=poison)

    async def _cancelled_rollback() -> None:
        # ...and the unwind's own closing rollback() is what gets cancelled.
        raise asyncio.CancelledError

    monkeypatch.setattr(store._db, "rollback", _cancelled_rollback)

    with pytest.raises(asyncio.CancelledError):
        await store.create_thought(_source(content=_THREE_PARAS))

    assert store._connection_quarantined is True
    with pytest.raises(ConnectionQuarantinedError):
        await store.get_thought("src-1")
    await _drain_quarantine_close(store)


# ---------------------------------------------------------------------------
# Provenance integrity — a foreign-id conflict-as-reuse must NOT claim provenance
# ---------------------------------------------------------------------------


async def _seed_foreign_row(
    db: aiosqlite.Connection,
    *,
    at_content: str,
    stored_content: str,
) -> str:
    """Pre-create a thought at ``uuid5(at_content)`` but with ``stored_content``.

    Returns the deterministic id occupied by the seeded (foreign-content) row.
    """
    foreign_id = _derived_thought_id(at_content)
    seed = SqliteEngravaCore(db)
    await seed.create_thought(
        CoreThoughtRecord(
            thought_id=foreign_id,
            thought_type=ThoughtType.NOTE,
            essence="foreign essence",
            content=stored_content,
            priority=Priority.P1,
            lifecycle_status=LifecycleStatus.CREATED,
            created_cycle=0,
            updated_cycle=0,
            source="seed",
        ),
    )
    return foreign_id


async def test_foreign_id_reuse_does_not_attach_false_provenance_log(
    db: aiosqlite.Connection,
) -> None:
    """A conflict-as-reuse onto a foreign-content row attaches NO provenance edge.

    A caller pre-creates a thought whose id equals ``uuid5(X)`` but with
    different content ``Y``. A derived child with content ``X`` would reuse row
    ``Y``; attaching a ``DERIVED_FROM`` edge would falsely assert "Y was derived
    from source". Under ``on_error="log"`` the collision is logged and skipped:
    no edge is attached (and the foreign row's own content is untouched).
    Reverting the fix attaches the edge and this fails.
    """
    child_content = "Derived body X."
    foreign_id = await _seed_foreign_row(
        db,
        at_content=child_content,
        stored_content="A completely unrelated stored body Y.",
    )

    producer = ListProducer([_child(child_content)])
    store = _make_store(db, producer, DeriveGates(enabled=True, on_error="log"))
    await store.create_thought(_source(content="Source body."))

    # No DERIVED_FROM provenance edge was attached to the foreign row.
    assert (
        await _count(
            db,
            "SELECT COUNT(*) FROM edge WHERE edge_type = ? AND from_thought_id = ?",
            EdgeType.DERIVED_FROM.value,
            foreign_id,
        )
        == 0
    )
    assert (
        await _count(
            db,
            "SELECT COUNT(*) FROM edge WHERE edge_type = ?",
            EdgeType.DERIVED_FROM.value,
        )
        == 0
    )
    # The source stays durable; the foreign row keeps its own content.
    assert await store.get_thought("src-1") is not None
    foreign = await store.get_thought(foreign_id)
    assert foreign is not None
    assert foreign.content == "A completely unrelated stored body Y."


async def test_foreign_id_reuse_raises_under_raise_policy(
    db: aiosqlite.Connection,
) -> None:
    """The foreign-content collision surfaces as ``DerivedRecordError`` under raise."""
    child_content = "Derived body X."
    foreign_id = await _seed_foreign_row(
        db,
        at_content=child_content,
        stored_content="Unrelated stored body Y.",
    )

    producer = ListProducer([_child(child_content)])
    store = _make_store(db, producer, DeriveGates(enabled=True, on_error="raise"))
    with pytest.raises(DerivedRecordError):
        await store.create_thought(_source(content="Source body."))

    # Still no false provenance edge, and the source stays durable.
    assert (
        await _count(
            db,
            "SELECT COUNT(*) FROM edge WHERE edge_type = ? AND from_thought_id = ?",
            EdgeType.DERIVED_FROM.value,
            foreign_id,
        )
        == 0
    )
    assert await store.get_thought("src-1") is not None


async def test_matching_content_reuse_still_attaches_provenance(
    db: aiosqlite.Connection,
) -> None:
    """A conflict-as-reuse onto a SAME-content row still attaches the edge.

    The provenance guard rejects only foreign content; a legitimate reuse (the
    stored row's content matches the derived record) must keep attaching the
    ``DERIVED_FROM`` edge.
    """
    child_content = "Shared derived body."
    same_id = await _seed_foreign_row(
        db,
        at_content=child_content,
        stored_content=child_content,  # SAME content — a legitimate reuse.
    )

    producer = ListProducer([_child(child_content)])
    store = _make_store(db, producer, DeriveGates(enabled=True, on_error="raise"))
    await store.create_thought(_source(content="Source body."))

    # The provenance edge to the reused (matching-content) row exists.
    assert (
        await _count(
            db,
            "SELECT COUNT(*) FROM edge WHERE edge_type = ? AND from_thought_id = ? "
            "AND to_thought_id = ?",
            EdgeType.DERIVED_FROM.value,
            same_id,
            "src-1",
        )
        == 1
    )


# ---------------------------------------------------------------------------
# UNIQUE-violation classification is structural (extended error code), not text
# ---------------------------------------------------------------------------


def _integrity_error(sql: str) -> sqlite3.IntegrityError:
    """Run *sql* against a fixture DB and return the raised IntegrityError."""
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            "CREATE TABLE parent (id INTEGER PRIMARY KEY, v UNIQUE, "
            "CONSTRAINT chk_unique_flag CHECK (v <> 99))",
        )
        conn.execute(
            "CREATE TABLE child (id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES parent(id))",
        )
        conn.execute("INSERT INTO parent (id, v) VALUES (1, 'a')")
        try:
            conn.execute(sql)
        except sqlite3.IntegrityError as exc:
            return exc
        msg = f"expected IntegrityError for: {sql}"
        raise AssertionError(msg)
    finally:
        conn.close()


def test_is_unique_violation_true_for_unique_and_primary_key() -> None:
    """A UNIQUE and a PRIMARY KEY violation both classify as a unique violation."""
    unique_exc = _integrity_error("INSERT INTO parent (id, v) VALUES (2, 'a')")
    pk_exc = _integrity_error("INSERT INTO parent (id, v) VALUES (1, 'b')")
    assert unique_exc.sqlite_errorcode == sqlite3.SQLITE_CONSTRAINT_UNIQUE
    assert pk_exc.sqlite_errorcode == sqlite3.SQLITE_CONSTRAINT_PRIMARYKEY
    assert _is_unique_violation(unique_exc) is True
    assert _is_unique_violation(pk_exc) is True


def test_is_unique_violation_false_for_foreign_key() -> None:
    """A FOREIGN KEY violation is NOT a unique violation (must re-raise upstream)."""
    fk_exc = _integrity_error("INSERT INTO child (id, parent_id) VALUES (1, 999)")
    assert fk_exc.sqlite_errorcode == sqlite3.SQLITE_CONSTRAINT_FOREIGNKEY
    assert _is_unique_violation(fk_exc) is False


def test_is_unique_violation_false_for_check_named_unique() -> None:
    """A CHECK failure whose name contains ``"unique"`` is NOT a unique violation.

    This is the fragile case: its message ("CHECK constraint failed:
    chk_unique_flag") contains "UNIQUE", so the old text-based classifier
    misclassifies it as a unique violation. The structural (extended error code)
    check correctly returns ``False``. Reverting the fix fails this test.
    """
    check_exc = _integrity_error("INSERT INTO parent (id, v) VALUES (2, 99)")
    assert check_exc.sqlite_errorcode == sqlite3.SQLITE_CONSTRAINT_CHECK
    assert "UNIQUE" in str(check_exc).upper()  # the text-fragility trap
    assert _is_unique_violation(check_exc) is False


# ===========================================================================
# derive_existing() — explicit backfill trigger (the on-store seam's
# retroactive counterpart). Convergence, idempotency, recursion guard,
# fail-open isolation, capability-present gating (independent of enabled),
# typed not-found vs clean skip, and a non-LLM demo.
# ===========================================================================


async def _edge_rows(db: aiosqlite.Connection) -> list[tuple[object, ...]]:
    """Dump every edge's stable identity columns, ordered for byte comparison."""
    cursor = await db.execute(
        "SELECT edge_id, from_thought_id, to_thought_id, edge_type, created_cycle "
        "FROM edge ORDER BY edge_id",
    )
    return [tuple(row) for row in await cursor.fetchall()]


async def _derived_identity_dump(
    conn: aiosqlite.Connection,
) -> tuple[list[tuple[object, ...]], list[tuple[object, ...]]]:
    """Dump the deterministic (wall-clock-free) thought + edge fields.

    Used for a cross-store convergence comparison: a derived child's
    ``created_at`` / ``updated_at`` are wall-clock stamps assigned at persist
    time, so two independent runs differ there; every *identity-bearing* field
    (id, content, essence, type, priority, status, cycle, source) plus the whole
    edge is deterministic and must match byte-for-byte.
    """
    tcur = await conn.execute(
        "SELECT thought_id, content, essence, thought_type, priority, "
        "lifecycle_status, created_cycle, updated_cycle, source "
        "FROM thought ORDER BY thought_id",
    )
    thoughts = [tuple(row) for row in await tcur.fetchall()]
    return thoughts, await _edge_rows(conn)


async def _fresh_conn() -> aiosqlite.Connection:
    """A fresh in-memory connection with the row factory + FK pragma set."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA foreign_keys = ON")
    return conn


class EchoDeriveProducer(DefaultEngravaHooks):
    """Derive exactly one child: the source content plus a distinct marker.

    Because the derived content differs from the source content, a *grandchild*
    (were the child ever re-derived) would have a distinct identity — which lets
    a test prove that a derived record is never re-derived.
    """

    def __init__(self) -> None:
        self.derived_from: list[str] = []

    async def derive_records(
        self,
        thought: ThoughtRecord,
        ctx: DeriveContext,
    ) -> Sequence[DerivedRecord]:
        self.derived_from.append(thought.thought_id)
        return [_child(f"{thought.content} [d]")]


class BackfillReentrantProducer(DefaultEngravaHooks):
    """Adversarial: calls ``derive_existing`` from inside ``derive_records``.

    A contract-violating re-entrant backfill exercised solely to prove the
    recursion guard holds — a ``derive_existing`` invoked from within an active
    derivation must be a no-op (it does not re-invoke the producer). It does not
    endorse the behaviour.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.store: SqliteEngravaCore | None = None
        self.nested_results: list[DeriveResult] = []

    async def derive_records(
        self,
        thought: ThoughtRecord,
        ctx: DeriveContext,
    ) -> Sequence[DerivedRecord]:
        self.calls += 1
        assert self.store is not None
        self.nested_results.append(await self.store.derive_existing(thought.thought_id))
        return [_child("the only child")]


# --- Convergence with the on-store path + idempotency -----------------


async def test_backfill_of_on_store_source_is_byte_identical_noop(
    db: aiosqlite.Connection,
) -> None:
    """Backfilling an already-derived source is a byte-identical
    no-op — every child + edge is reused, nothing is created, and a second
    backfill is identical (idempotent)."""
    store = _make_store(db, StructuralSplitProducer(), DeriveGates(enabled=True))
    await store.create_thought(_source(content="Alpha para.\n\nBeta para."))
    thoughts_before = await _thought_rows(db)
    edges_before = await _edge_rows(db)

    result = await store.derive_existing("src-1")
    assert result == DeriveResult(thought_id="src-1", created=0, reused=2, skipped=0)
    # The children + edges the on-store path produced are reused byte-for-byte —
    # no new rows, no new edges (reuse writes nothing).
    assert await _thought_rows(db) == thoughts_before
    assert await _edge_rows(db) == edges_before

    # A second backfill is the identical no-op.
    assert await store.derive_existing("src-1") == DeriveResult("src-1", 0, 2, 0)
    assert await _thought_rows(db) == thoughts_before
    assert await _edge_rows(db) == edges_before


async def test_backfill_children_and_edges_match_on_store_from_scratch() -> None:
    """Children + edges created by a from-scratch backfill are byte-
    identical (deterministic fields) to those an on-store write would produce for
    the same content — proving convergence, not merely reuse."""
    content = "Alpha para.\n\nBeta para."

    # Store A: automatic on-store derivation.
    conn_auto = await _fresh_conn()
    auto_store = SqliteEngravaCore(
        conn_auto,
        hooks=StructuralSplitProducer(),
        derive_gates=DeriveGates(enabled=True),
    )
    await auto_store.ensure_schema()
    await auto_store.create_thought(_source(content=content))
    auto = await _derived_identity_dump(conn_auto)
    await conn_auto.close()

    # Store B: seam disabled at store time, then explicit backfill.
    conn_back = await _fresh_conn()
    backfill_store = SqliteEngravaCore(
        conn_back,
        hooks=StructuralSplitProducer(),
        derive_gates=DeriveGates(enabled=False),
    )
    await backfill_store.ensure_schema()
    await backfill_store.create_thought(_source(content=content))
    assert await _count(conn_back, "SELECT COUNT(*) FROM thought") == 1  # no on-store
    result = await backfill_store.derive_existing("src-1")
    assert result == DeriveResult(thought_id="src-1", created=2, reused=0, skipped=0)
    backfilled = await _derived_identity_dump(conn_back)
    await conn_back.close()

    assert backfilled == auto


async def test_backfill_is_idempotent_across_reruns(db: aiosqlite.Connection) -> None:
    """The first backfill creates every child; a re-run reuses them all and
    leaves the store unchanged."""
    store = _make_store(db, StructuralSplitProducer(), DeriveGates(enabled=False))
    await store.create_thought(_source(content="One.\n\nTwo.\n\nThree."))

    first = await store.derive_existing("src-1")
    assert first == DeriveResult(thought_id="src-1", created=3, reused=0, skipped=0)
    thoughts_after_first = await _thought_rows(db)
    edges_after_first = await _edge_rows(db)

    second = await store.derive_existing("src-1")
    assert second == DeriveResult(thought_id="src-1", created=0, reused=3, skipped=0)
    assert await _thought_rows(db) == thoughts_after_first
    assert await _edge_rows(db) == edges_after_first


async def test_backfill_reuses_preexisting_child_in_counts(
    db: aiosqlite.Connection,
) -> None:
    """A child colliding with a pre-existing row is reused (not re-created)
    and reported as ``reused`` in the result counts."""
    child_content = "Second para."
    preexisting_id = _derived_thought_id(child_content)
    seed = SqliteEngravaCore(db)
    await seed.create_thought(
        CoreThoughtRecord(
            thought_id=preexisting_id,
            thought_type=ThoughtType.NOTE,
            essence="preexisting",
            content=child_content,
            priority=Priority.P1,
            lifecycle_status=LifecycleStatus.CREATED,
            created_cycle=0,
            updated_cycle=0,
            source="seed",
        ),
    )
    store = _make_store(db, StructuralSplitProducer(), DeriveGates(enabled=False))
    await store.create_thought(_source(content="First para.\n\nSecond para."))

    result = await store.derive_existing("src-1")
    # "First para." is newly created; "Second para." reuses the pre-existing row.
    assert result == DeriveResult(thought_id="src-1", created=1, reused=1, skipped=0)


# --- Capability-present gating, independent of enabled -----------------


async def test_backfill_runs_with_seam_disabled(db: aiosqlite.Connection) -> None:
    """Backfill runs on capability-present alone — the on-store trigger is
    off (``enabled=False``) so only the explicit call derives."""
    store = _make_store(db, StructuralSplitProducer(), DeriveGates(enabled=False))
    await store.create_thought(_source(content="A.\n\nB."))
    assert await _count(db, "SELECT COUNT(*) FROM thought") == 1  # enabled=False ⇒ no on-store

    result = await store.derive_existing("src-1")
    assert result == DeriveResult(thought_id="src-1", created=2, reused=0, skipped=0)
    assert await _count(db, "SELECT COUNT(*) FROM thought") == 3


async def test_backfill_without_producer_is_clean_noop(
    db: aiosqlite.Connection,
) -> None:
    """With no producer capability registered, backfill is a clean no-op."""
    store = _make_store(db, DefaultEngravaHooks(), DeriveGates(enabled=True))
    await store.create_thought(_source(content="A.\n\nB."))
    result = await store.derive_existing("src-1")
    assert result == DeriveResult(thought_id="src-1", created=0, reused=0, skipped=0)
    assert await _count(db, "SELECT COUNT(*) FROM thought") == 1


async def test_backfill_honors_cap_under_raise(db: aiosqlite.Connection) -> None:
    """Backfill honours ``max_derived_per_source`` — an over-cap return is
    rejected before any child write."""
    producer = ListProducer([_child(f"c{i}") for i in range(5)])
    store = _make_store(
        db,
        producer,
        DeriveGates(enabled=False, on_error="raise", max_derived_per_source=3),
    )
    await store.create_thought(_source())
    with pytest.raises(DerivedRecordError, match="max_derived_per_source"):
        await store.derive_existing("src-1")
    assert await _count(db, "SELECT COUNT(*) FROM thought") == 1


async def test_backfill_within_suspended_commit_joins_caller_transaction(
    db: aiosqlite.Connection,
) -> None:
    """Backfill does not early-return inside a ``suspend_auto_commit`` window (its
    source is already durable); the children join the caller's transaction and
    become durable when it commits."""
    store = _make_store(db, StructuralSplitProducer(), DeriveGates(enabled=False))
    await store.create_thought(_source(content="Alpha.\n\nBeta."))
    async with store.suspend_auto_commit():
        result = await store.derive_existing("src-1")
    assert result == DeriveResult(thought_id="src-1", created=2, reused=0, skipped=0)
    # After the caller's transaction commits (context exit), children are durable.
    assert await _count(db, "SELECT COUNT(*) FROM thought") == 3


# --- Not-found (typed error) vs ineligible (clean skip) ----------------


async def test_backfill_missing_source_raises_typed_error(
    db: aiosqlite.Connection,
) -> None:
    """A missing source id raises the typed error, never a silent no-op."""
    store = _make_store(db, StructuralSplitProducer(), DeriveGates(enabled=False))
    with pytest.raises(SourceThoughtNotFoundError):
        await store.derive_existing("nonexistent-id")


async def test_backfill_does_not_re_derive_a_derived_record(
    db: aiosqlite.Connection,
) -> None:
    """A source that is itself a derived record is a clean skip — the
    producer is never invoked on it and no grandchild is created."""
    producer = EchoDeriveProducer()
    store = _make_store(db, producer, DeriveGates(enabled=True))
    await store.create_thought(_source(content="X"))
    child_id = _derived_thought_id("X [d]")
    assert await store.get_thought(child_id) is not None
    assert producer.derived_from == ["src-1"]  # derived once, from the source

    result = await store.derive_existing(child_id)
    assert result == DeriveResult(thought_id=child_id, created=0, reused=0, skipped=0)
    # The producer was NOT invoked on the derived child (guard-marker skip)...
    assert producer.derived_from == ["src-1"]
    # ...so no grandchild ("X [d] [d]") was ever created.
    assert await store.get_thought(_derived_thought_id("X [d] [d]")) is None


# --- Recursion guard (depth <= 1, nested writes, nested backfill) ------


async def test_backfill_recursion_guard_blocks_nested_write(
    db: aiosqlite.Connection,
) -> None:
    """A nested public write a producer issues *during backfill* does not
    re-dispatch — depth stays at most one (no runaway recursion)."""
    producer = NestedWriteProducer()
    store = _make_store(db, producer, DeriveGates(enabled=True))
    producer.store = store
    await store.create_thought(_source(content="single"))
    assert producer.calls == 1  # on-store derivation ran once; nested write guarded

    result = await store.derive_existing("src-1")
    # Exactly one *more* dispatch: the nested create_thought during the backfill
    # did not re-dispatch (revert the guard ⇒ unbounded recursion ⇒ calls > 2).
    assert producer.calls == 2
    assert result == DeriveResult(thought_id="src-1", created=0, reused=1, skipped=0)
    # The nested write produced no derived children of its own.
    assert await store.get_edges("nested-2", direction="IN") == []


async def test_backfill_nested_derive_existing_is_a_noop(
    db: aiosqlite.Connection,
) -> None:
    """The strongest test of the recursion guard: a ``derive_existing`` invoked
    from within a derivation is a no-op — it ignores ``enabled``, so *only*
    the recursion guard stops it."""
    producer = BackfillReentrantProducer()
    store = _make_store(db, producer, DeriveGates(enabled=False))
    producer.store = store
    await store.create_thought(_source(content="body"))
    assert producer.calls == 0  # enabled=False ⇒ no on-store derivation

    result = await store.derive_existing("src-1")
    # derive_records ran exactly once; the re-entrant backfill did NOT re-run it.
    assert producer.calls == 1
    assert producer.nested_results == [DeriveResult(thought_id="src-1")]
    assert result == DeriveResult(thought_id="src-1", created=1, reused=0, skipped=0)


# --- Fail-open, per-child isolation, cancellation ---------------------


async def test_backfill_producer_error_raise_keeps_source_durable(
    db: aiosqlite.Connection,
) -> None:
    """``on_error='raise'`` re-raises, but the source stays durable."""
    producer = RaisingProducer()
    store = _make_store(db, producer, DeriveGates(enabled=False, on_error="raise"))
    await store.create_thought(_source())
    with pytest.raises(RuntimeError, match="producer boom"):
        await store.derive_existing("src-1")
    assert await store.get_thought("src-1") is not None


async def test_backfill_producer_error_log_swallows(
    db: aiosqlite.Connection,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``on_error='log'`` swallows the producer failure with ordinary logging."""
    producer = RaisingProducer()
    store = _make_store(db, producer, DeriveGates(enabled=False, on_error="log"))
    await store.create_thought(_source())
    with caplog.at_level(logging.WARNING, logger=core_module.__name__):
        result = await store.derive_existing("src-1")
    assert result == DeriveResult(thought_id="src-1", created=0, reused=0, skipped=0)
    assert await store.get_thought("src-1") is not None
    assert any(record.levelno == logging.WARNING for record in caplog.records)


@pytest.mark.parametrize("on_error", ["raise", "log"])
async def test_backfill_cancellation_propagates(
    db: aiosqlite.Connection,
    on_error: str,
) -> None:
    """A cancelled ``derive_records`` propagates ``CancelledError`` either way,
    leaving the source durable."""
    store = _make_store(
        db,
        CancellingProducer(),
        DeriveGates(enabled=False, on_error=on_error),  # type: ignore[arg-type]
    )
    await store.create_thought(_source())
    with pytest.raises(asyncio.CancelledError):
        await store.derive_existing("src-1")
    assert await store.get_thought("src-1") is not None


async def test_backfill_child_failure_is_isolated_and_journal_valid() -> None:
    """A per-child failure under ``on_error='log'`` is isolated — it is
    counted as skipped, the other children commit, the source stays durable, and
    the journal hash-chain remains valid (no orphan / torn transaction)."""
    collide = "poison content"
    # Make the source id equal a middle child's content-addressed derived id, so
    # that child deterministically self-collides (rejected) without touching the
    # others.
    colliding_source_id = _derived_thought_id(collide)
    producer = ListProducer(
        [_child("first good"), _child(collide), _child("second good")],
    )
    conn = await _fresh_conn()
    store = SqliteEngravaCore(
        conn,
        hooks=producer,
        derive_gates=DeriveGates(enabled=False, on_error="log"),
        journal_enabled=True,
    )
    await store.ensure_schema()
    await store.create_thought(_source(colliding_source_id, content="Body."))

    result = await store.derive_existing(colliding_source_id)
    assert result == DeriveResult(
        thought_id=colliding_source_id,
        created=2,
        reused=0,
        skipped=1,
    )
    # Source durable; the two good children committed; journal chain intact.
    assert await store.get_thought(colliding_source_id) is not None
    assert await _count(conn, "SELECT COUNT(*) FROM thought") == 3
    integrity = await store.verify_journal()
    assert integrity.valid
    await conn.close()


async def test_backfill_raise_in_suspend_window_rolls_back_caller_writes_source_survives() -> None:
    """A raising backfill inside a caller transaction rolls the whole window
    back — the caller's unrelated write included — while the source stays durable.

    Under a caller-held ``suspend_auto_commit`` window with ``on_error="raise"`` a
    derived-child failure propagates out of the window, so ``suspend_auto_commit``
    rolls the entire transaction back. That is the window owner's normal atomicity,
    not behaviour unique to backfill; ``derive_existing`` is merely the path that
    runs derivation *inside* such a window (the on-store trigger defers instead).
    The source thought, committed **before** the window, is unaffected.

    Regression-sensitive by construction: the first produced child is persisted
    (uncommitted) into the window before the second child collides and aborts, so
    if the window did NOT roll back on the raise the unrelated write and that
    first child's row + ``DERIVED_FROM`` edge would survive — assertions (a)/(c)
    would fail. If the already-committed source were swept into the rollback,
    assertion (b) would fail.
    """
    collide = "poison content"
    # Make the source id equal a produced child's content-addressed derived id, so
    # that child deterministically collides with its own source and fails to
    # persist (identity collision) — surfaced as a raise under on_error="raise".
    colliding_source_id = _derived_thought_id(collide)
    producer = ListProducer(
        [_child("first good"), _child(collide), _child("second good")],
    )
    conn = await _fresh_conn()
    store = SqliteEngravaCore(
        conn,
        hooks=producer,
        derive_gates=DeriveGates(enabled=False, on_error="raise"),
        journal_enabled=True,
    )
    await store.ensure_schema()
    # The source commits durably BEFORE the window (enabled=False ⇒ no on-store
    # derivation), so it is not part of the caller transaction opened below.
    await store.create_thought(_source(colliding_source_id, content="Body."))

    # One caller-held transaction: an unrelated write, then a backfill whose second
    # child collides and (under raise) aborts — the error leaves the window, which
    # rolls the whole transaction back.
    async def _unrelated_write_then_failing_backfill() -> None:
        async with store.suspend_auto_commit():
            await store.create_thought(_source("unrelated-src", content="unrelated body"))
            await store.derive_existing(colliding_source_id)

    with pytest.raises(DerivedRecordError):
        await _unrelated_write_then_failing_backfill()

    # (a) the caller's unrelated write was rolled back with the failed derivation.
    assert await store.get_thought("unrelated-src") is None
    # (b) the source, committed before the window, is unaffected.
    assert await store.get_thought(colliding_source_id) is not None
    # (c) no orphan child rows or edges: only the source row survives, and the
    # first child's row + its DERIVED_FROM edge (persisted but uncommitted in the
    # window) were rolled back too.
    assert await _count(conn, "SELECT COUNT(*) FROM thought") == 1
    assert await _edge_rows(conn) == []
    # (d) the journal hash-chain remains valid (no torn / half-written entry).
    integrity = await store.verify_journal()
    assert integrity.valid
    await conn.close()


# --- Non-LLM demo + first-classness ----------------------------


async def test_structural_split_backfill_is_non_llm_demo(
    db: aiosqlite.Connection,
) -> None:
    """A deterministic structural-split producer backfilled via
    ``derive_existing`` — one linked child per paragraph, no LLM."""
    store = _make_store(db, StructuralSplitProducer(), DeriveGates(enabled=False))
    await store.create_thought(_source(content="One.\n\nTwo.\n\nThree."))
    assert await _count(db, "SELECT COUNT(*) FROM thought") == 1

    result = await store.derive_existing("src-1")
    assert result == DeriveResult(thought_id="src-1", created=3, reused=0, skipped=0)
    edges = await store.get_edges("src-1", direction="IN")
    assert len(edges) == 3
    assert all(e.edge_type == EdgeType.DERIVED_FROM for e in edges)


async def test_backfilled_children_are_embedded_and_retrievable(
    db: aiosqlite.Connection,
) -> None:
    """Backfilled children run the ordinary lifecycle — embedded + linked."""
    provider = CallbackProvider(_hash_embed, dimension=8, model_name="hash-8")
    store = _make_store(
        db,
        StructuralSplitProducer(),
        DeriveGates(enabled=False),
        embedding_provider=provider,
        auto_embed=True,
    )
    await store.create_thought(_source(content="Head para.\n\nTail para."))

    result = await store.derive_existing("src-1")
    assert result.created == 2
    for edge in await store.get_edges("src-1", direction="IN"):
        assert await store.get_embedding(edge.from_thought_id) is not None
