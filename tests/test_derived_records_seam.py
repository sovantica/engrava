"""Integration tests for the derived-records extension seam.

Exercises the core-controlled, per-child, source-first, deferred, non-atomic
persistence of an extension's derived records against a live
``SqliteEngravaCore``. Each acceptance criterion of the seam is covered here or
in ``tests/domain/test_derived_records_types.py``:

* AC-2  — no demo-consumer / extension import in the core seam.
* AC-3  — the seam's public types add zero third-party dependencies.
* AC-4  — ``on_error="log"`` is ordinary logging, no telemetry surface.
* AC-5  — disabled path is byte-identical (thoughts + edges + journal).
* AC-6  — hooks without ``derive_records`` run byte-identical (protocol compat).
* AC-8  — the deterministic structural-split demo consumer.
* AC-9  — the recursion guard across single / bulk / get-or-create, incl. an
          adversarial producer that performs a nested public write.
* AC-10 — fail-open, cancellation propagation, per-family continuation.
* AC-11 — first-classness (embed/retrieve), conflict-as-reuse, and bounds.

The explicit ``derive_existing()`` backfill trigger — the on-store seam's
retroactive counterpart — is covered in its own section at the end of this file
(convergence with the on-store path, idempotency, the recursion guard, fail-open
isolation, capability-present gating independent of the enabled master switch,
typed not-found vs clean skip, and a non-LLM structural-split demonstration).
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import logging
import sqlite3
import struct
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
    _DerivationRollbackError,
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
# AC-8 — deterministic structural-split demo consumer
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
# AC-5 / AC-6 — disabled + protocol-compat byte-identical paths
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
# AC-9 — recursion guard (single / bulk / get-or-create + adversarial nested)
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
# AC-10 — fail-open, cancellation, continuation
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
# AC-11 — first-classness, conflict-as-reuse, bounds
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
# AC-2 / AC-3 — surface hygiene (no extension import in core; zero new deps)
# ---------------------------------------------------------------------------

_ALLOWED_TOP_MODULES = frozenset(
    {"__future__", "engrava", "dataclasses", "typing", "collections", "re"},
)


def test_seam_types_import_no_third_party_packages() -> None:
    """The seam's public types module depends only on stdlib + engrava (AC-3)."""
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
    """The core does not import the demo (or any) derived-record producer (AC-2)."""
    core_source = Path(core_module.__file__).read_text(encoding="utf-8")
    assert "structural_split" not in core_source
    assert "StructuralSplitProducer" not in core_source


# ---------------------------------------------------------------------------
# F4 — core-derived essence (combining-mark-safe truncation)
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
# F2 — producer-sequence iteration failures are fail-open
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
# F1 / F3 — bulk_store derivation never rolls a committed source/child back
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
# F6 — durability + recoverability of a committed-yet-unenriched child
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
# F7 — recursion guard across all write entry points; re-materialization paths
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
# R2-1 — a dedup / hash hit never dispatches derivation (D5), embeddings OFF
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
# R2-2 — documented contract: a create inside a caller-held transaction does not
# auto-derive; derivation is triggered by an explicit re-run / backfill.
# ---------------------------------------------------------------------------


async def test_create_inside_caller_suspend_does_not_auto_derive(
    db: aiosqlite.Connection,
) -> None:
    """A create in a caller-held transaction does not auto-derive; backfill does.

    Documented contract (ADR D8): derivation fires only on a durably
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
# R2-3 — conflict-as-reuse enrichment targets the STORED row, never producer content
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
# R2-4 — combining-mark truncation degenerate cases
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


async def test_failed_rollback_aborts_derivation_raise_propagates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under ``on_error="raise"`` a failed per-child rollback propagates + quarantines.

    A rollback failure leaves the transaction indeterminate, so the dispatch
    aborts (``_DerivationRollbackError`` under the raise policy) AND — (#1) any
    non-clean compensating rollback quarantines the store, whether or not a
    cancellation was involved — the connection is hard-invalidated so a later
    op can never flush the orphan. The source + earlier child committed before
    the failure stay durable on disk (verified via a fresh connection).
    """
    db_path = tmp_path / "seam.db"
    store, conn = await _file_journaled_seam_store(db_path, "raise")
    poison = _derived_thought_id(_SEGMENTS[1])  # the 2nd of three children
    _patch_journal_to_fail(store, monkeypatch, mutation_type="INSERT_THOUGHT", target_id=poison)

    async def _bad_rollback() -> None:
        msg = "rollback down"
        raise RuntimeError(msg)

    monkeypatch.setattr(store._db, "rollback", _bad_rollback)

    with pytest.raises(_DerivationRollbackError):
        await store.create_thought(_source(content=_THREE_PARAS))

    # (#1) The store is quarantined and any subsequent op fails fast — reverting
    # the quarantine-on-non-cancel fix leaves the store usable and this fails.
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


async def test_failed_rollback_aborts_derivation_log_does_not_raise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under ``on_error="log"`` a failed per-child rollback aborts WITHOUT raising.

    The source is already durably committed, so a ``_DerivationRollbackError``
    must never escape a ``"log"`` policy as a caller-visible raise (fail-open,
    ADR D10). But a rollback failure is indeterminate, so (#1) the store is still
    quarantined even under ``"log"`` — precisely to stop the log-and-continue
    caller from later flushing the orphan on the same store.
    """
    db_path = tmp_path / "seam.db"
    store, conn = await _file_journaled_seam_store(db_path, "log")
    poison = _derived_thought_id(_SEGMENTS[1])  # the 2nd of three children
    _patch_journal_to_fail(store, monkeypatch, mutation_type="INSERT_THOUGHT", target_id=poison)

    async def _bad_rollback() -> None:
        msg = "rollback down"
        raise RuntimeError(msg)

    monkeypatch.setattr(store._db, "rollback", _bad_rollback)

    # No exception escapes to the caller under the log policy (F2 preserved).
    await store.create_thought(_source(content=_THREE_PARAS))

    # (#1) But the store is quarantined and a subsequent op fails fast.
    assert store._connection_quarantined is True
    with pytest.raises(ConnectionQuarantinedError):
        await store.get_thought("src-1")

    await _reopen_and_verify_durable(
        db_path,
        present=["src-1", _derived_thought_id(_SEGMENTS[0])],
        absent=[poison, _derived_thought_id(_SEGMENTS[2])],
    )
    await conn.close()


async def test_cancellation_with_failed_rollback_still_propagates_cancelled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancelled child whose rollback also fails propagates CancelledError + quarantines.

    D10 requires ``CancelledError`` to always propagate. Even when the
    compensating rollback itself raises an ordinary exception, the cancellation —
    not the rollback error — is what escapes; and (#1) the failed rollback still
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

    async def _bad_rollback() -> None:
        msg = "rollback down"
        raise RuntimeError(msg)

    monkeypatch.setattr(store._journal, "append", _cancel_append)
    monkeypatch.setattr(store._db, "rollback", _bad_rollback)

    with pytest.raises(asyncio.CancelledError):
        await store.create_thought(_source(content=_THREE_PARAS))

    assert store._connection_quarantined is True
    with pytest.raises(ConnectionQuarantinedError):
        await store.get_thought("src-1")

    await _reopen_and_verify_durable(
        db_path,
        present=["src-1", _derived_thought_id(_SEGMENTS[0])],
        absent=[poison],
    )
    await conn.close()


async def test_repeated_cancellation_during_rollback_still_completes_it(
    db: aiosqlite.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shielded compensating rollback completes despite repeated cancellation.

    A child fails (ordinary error) and enters the compensating rollback, which
    runs as a shielded task. The caller is then cancelled *twice* while awaiting
    it. Because the rollback task is shielded, neither cancellation aborts it: it
    runs to completion, so ``in_transaction`` is ``False`` and the poison child's
    pending insert is discarded — while ``CancelledError`` still propagates.

    Reverting the hardening (the old ``suppress`` + unshielded retry) lets the
    repeated cancellation abort the rollback, leaving the connection
    mid-transaction (``in_transaction`` stays ``True``), which this test catches.
    """
    store = _journaled_seam_store(db, "log")
    poison = _derived_thought_id(_SEGMENTS[1])  # the 2nd of three children
    # Make the poison child's INSERT_THOUGHT journal append fail (ordinary error)
    # so the child enters the compensating-rollback path.
    _patch_journal_to_fail(store, monkeypatch, mutation_type="INSERT_THOUGHT", target_id=poison)

    real_rollback = store._db.rollback
    started = asyncio.Event()
    release = asyncio.Event()
    completed = {"done": False}

    async def _slow_rollback() -> None:
        # Signal it has started, block until released (so the caller can be
        # cancelled mid-rollback), then run the real rollback to completion.
        started.set()
        await release.wait()
        await real_rollback()
        completed["done"] = True

    monkeypatch.setattr(store._db, "rollback", _slow_rollback)

    task = asyncio.create_task(store.create_thought(_source(content=_THREE_PARAS)))
    await started.wait()  # the shielded rollback task is running
    task.cancel()  # cancel the caller while it awaits the rollback
    await asyncio.sleep(0)
    task.cancel()  # a repeated cancellation must not abort the shielded rollback
    await asyncio.sleep(0)
    release.set()  # let the shielded rollback finish
    with pytest.raises(asyncio.CancelledError):
        await task

    # The shielded rollback ran to completion despite the repeated cancellation:
    # no open transaction leaks and the poison child's pending insert is gone.
    assert completed["done"] is True
    assert store._db.in_transaction is False
    assert store._connection_quarantined is False  # a clean rollback never quarantines
    assert await store.get_thought(poison) is None
    assert await store.get_thought("src-1") is not None
    assert await store.get_thought(_derived_thought_id(_SEGMENTS[0])) is not None
    assert (await store.verify_journal()).valid


async def test_rollback_failure_during_cancellation_quarantines_store(
    db: aiosqlite.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rollback that ultimately fails under cancellation quarantines the store.

    When a cancellation is in flight and the compensating rollback ultimately
    *fails*, the long-lived connection may still hold an open transaction. The
    store must be quarantined so a subsequent public operation fails fast with a
    typed ``ConnectionQuarantinedError`` rather than silently running on /able
    to flush the indeterminate transaction. The cancellation still propagates.

    Reverting the quarantine leaves the store usable, so the follow-up operation
    would run on the open transaction instead of failing — which this test
    catches.
    """
    store = _journaled_seam_store(db, "log")
    poison = _derived_thought_id(_SEGMENTS[1])
    _patch_journal_to_fail(store, monkeypatch, mutation_type="INSERT_THOUGHT", target_id=poison)

    started = asyncio.Event()
    release = asyncio.Event()

    async def _slow_failing_rollback() -> None:
        started.set()
        await release.wait()
        msg = "rollback truly failed"
        raise RuntimeError(msg)

    monkeypatch.setattr(store._db, "rollback", _slow_failing_rollback)

    task = asyncio.create_task(store.create_thought(_source(content=_THREE_PARAS)))
    await started.wait()
    task.cancel()  # cancel the caller while it awaits the rollback
    await asyncio.sleep(0)
    release.set()  # the shielded rollback now runs to completion — and fails
    with pytest.raises(asyncio.CancelledError):
        await task

    # The rollback ultimately failed while a cancellation was in flight, so the
    # connection may still hold an open transaction: the store is quarantined and
    # every subsequent public operation fails fast rather than running on it.
    assert store._connection_quarantined is True
    with pytest.raises(ConnectionQuarantinedError):
        await store.get_thought("src-1")
    with pytest.raises(ConnectionQuarantinedError):
        await store.create_thought(_source("src-2", content="unrelated body"))
    await _drain_quarantine_close(store)


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


async def test_cancelled_rollback_task_quarantines_and_propagates_cancelled(
    db: aiosqlite.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(#2) A cancelled compensating-rollback task quarantines + propagates cancel.

    If the rollback task is cancelled, ``rollback_task.exception()`` would raise
    ``CancelledError`` — which, unguarded, escapes *before* the quarantine runs.
    Checking ``cancelled()`` first treats it as non-clean completion: the store
    is quarantined and a ``CancelledError`` still propagates. Reverting the
    ``cancelled()`` guard skips the quarantine, so the flag assertion fails.
    """
    store = _journaled_seam_store(db, "log")
    poison = _derived_thought_id(_SEGMENTS[1])
    # The child fails with an ordinary error so it enters the rollback path...
    _patch_journal_to_fail(store, monkeypatch, mutation_type="INSERT_THOUGHT", target_id=poison)

    async def _cancelled_rollback() -> None:
        # ...and the compensating rollback itself is cancelled (its task raises
        # CancelledError → the task ends in the cancelled state).
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


# --- AC-4: convergence with the on-store path + idempotency -----------------


async def test_backfill_of_on_store_source_is_byte_identical_noop(
    db: aiosqlite.Connection,
) -> None:
    """AC-4 (primary): backfilling an already-derived source is a byte-identical
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
    """AC-4: children + edges created by a from-scratch backfill are byte-
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
    """AC-4: the first backfill creates every child; a re-run reuses them all and
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
    """AC-4: a child colliding with a pre-existing row is reused (not re-created)
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


# --- AC-7: capability-present gating, independent of enabled -----------------


async def test_backfill_runs_with_seam_disabled(db: aiosqlite.Connection) -> None:
    """AC-7: backfill runs on capability-present alone — the on-store trigger is
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
    """AC-7: with no producer capability registered, backfill is a clean no-op."""
    store = _make_store(db, DefaultEngravaHooks(), DeriveGates(enabled=True))
    await store.create_thought(_source(content="A.\n\nB."))
    result = await store.derive_existing("src-1")
    assert result == DeriveResult(thought_id="src-1", created=0, reused=0, skipped=0)
    assert await _count(db, "SELECT COUNT(*) FROM thought") == 1


async def test_backfill_honors_cap_under_raise(db: aiosqlite.Connection) -> None:
    """AC-7: backfill honours ``max_derived_per_source`` — an over-cap return is
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


# --- AC-8: not-found (typed error) vs ineligible (clean skip) ----------------


async def test_backfill_missing_source_raises_typed_error(
    db: aiosqlite.Connection,
) -> None:
    """AC-8: a missing source id raises the typed error, never a silent no-op."""
    store = _make_store(db, StructuralSplitProducer(), DeriveGates(enabled=False))
    with pytest.raises(SourceThoughtNotFoundError):
        await store.derive_existing("nonexistent-id")


async def test_backfill_does_not_re_derive_a_derived_record(
    db: aiosqlite.Connection,
) -> None:
    """AC-8/AC-5: a source that is itself a derived record is a clean skip — the
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


# --- AC-5: recursion guard (depth <= 1, nested writes, nested backfill) ------


async def test_backfill_recursion_guard_blocks_nested_write(
    db: aiosqlite.Connection,
) -> None:
    """AC-5: a nested public write a producer issues *during backfill* does not
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
    """AC-5 (strongest): a ``derive_existing`` invoked from within a derivation is
    a no-op — it ignores ``enabled``, so *only* the recursion guard stops it."""
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


# --- AC-6: fail-open, per-child isolation, cancellation ---------------------


async def test_backfill_producer_error_raise_keeps_source_durable(
    db: aiosqlite.Connection,
) -> None:
    """AC-6: ``on_error='raise'`` re-raises, but the source stays durable."""
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
    """AC-6: ``on_error='log'`` swallows the producer failure with ordinary logging."""
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
    """AC-6: a cancelled ``derive_records`` propagates ``CancelledError`` either way,
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
    """AC-6: a per-child failure under ``on_error='log'`` is isolated — it is
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
    """AC-6: a raising backfill inside a caller transaction rolls the whole window
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


# --- AC-10/AC-11: non-LLM demo + first-classness ----------------------------


async def test_structural_split_backfill_is_non_llm_demo(
    db: aiosqlite.Connection,
) -> None:
    """AC-10: a deterministic structural-split producer backfilled via
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
    """AC-11: backfilled children run the ordinary lifecycle — embedded + linked."""
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
