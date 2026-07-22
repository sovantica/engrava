"""Integration tests for ReadOnlyEngrava.

Verifies that:
1. The adapter satisfies its declared runtime-checkable read protocol.
2. All read operations are delegated to the inner Engrava store.
3. Every write operation raises ReadOnlyViolationError unconditionally.
4. Reads with hidden access-tracking writes are delegated with the wrapped
   store's access tracking suppressed (no read stages an access-count write).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import (
    ActionRecord,
    ActionStatus,
    ActionType,
    CoreThoughtRecord,
    EdgeRecord,
    EdgeType,
    EngravaCoreProtocol,
    EngravaReadProtocol,
    FieldOp,
    FieldPredicate,
    KnowledgeSource,
    LifecycleStatus,
    MetadataFilter,
    Priority,
    ReadOnlyEngrava,
    ReadOnlyViolationError,
    SqliteEngravaCore,
    ThoughtType,
    ThoughtVisibility,
    VerificationStatus,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def db() -> AsyncIterator[aiosqlite.Connection]:
    """In-memory SQLite database with core schema."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode = WAL")
    await conn.execute("PRAGMA foreign_keys = ON")
    store = SqliteEngravaCore(conn)
    await store.ensure_schema()
    yield conn
    await conn.close()


@pytest.fixture
async def inner_store(db: aiosqlite.Connection) -> SqliteEngravaCore:
    """SqliteEngravaCore backed by the test database."""
    s = SqliteEngravaCore(db)
    await s._probe_fts()
    return s


@pytest.fixture
def ro_store(inner_store: SqliteEngravaCore) -> ReadOnlyEngrava:
    """ReadOnlyEngrava wrapping the inner store."""
    return ReadOnlyEngrava(inner_store)


@pytest.fixture
async def tracking_store(db: aiosqlite.Connection) -> SqliteEngravaCore:
    """SqliteEngravaCore with access tracking enabled (reads buffer accesses)."""
    s = SqliteEngravaCore(db, access_tracking_enabled=True)
    await s._probe_fts()
    return s


def _make_thought(
    thought_id: str = "t-001",
    *,
    lifecycle_status: LifecycleStatus = LifecycleStatus.ACTIVE,
    priority: Priority = Priority.P2,
    visibility: ThoughtVisibility = ThoughtVisibility.SELECTIVE,
) -> CoreThoughtRecord:
    return CoreThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.TASK,
        essence="Test thought",
        content="Test content",
        priority=priority,
        lifecycle_status=lifecycle_status,
        created_cycle=0,
        updated_cycle=0,
        source="test",
        confidence=0.8,
        source_type=KnowledgeSource.EXPERIENCE,
        visibility=visibility,
    )


def _make_edge(
    edge_id: str = "e-001",
    from_id: str = "t-a",
    to_id: str = "t-b",
) -> EdgeRecord:
    return EdgeRecord(
        edge_id=edge_id,
        from_thought_id=from_id,
        to_thought_id=to_id,
        edge_type=EdgeType.ASSOCIATED,
        weight=1.0,
        created_cycle=0,
    )


def _make_action(action_id: str = "a-001", thought_id: str = "t-001") -> ActionRecord:
    return ActionRecord(
        action_id=action_id,
        source_thought_id=thought_id,
        action_type=ActionType.TOOL_CALL,
        intent="do the thing",
        status=ActionStatus.PLANNED,
        verification_status=VerificationStatus.PENDING,
    )


# Every writable capability the adapter must reject, paired with a call that
# exercises its real signature (parity check + rejection check in one).
_WRITE_CASES: list[tuple[str, Callable[[ReadOnlyEngrava], Awaitable[object]]]] = [
    ("create_thought", lambda ro: ro.create_thought(_make_thought())),
    ("get_or_create", lambda ro: ro.get_or_create(_make_thought())),
    ("upsert_by_hash", lambda ro: ro.upsert_by_hash(_make_thought())),
    ("bulk_store", lambda ro: ro.bulk_store([_make_thought()])),
    ("remember", lambda ro: ro.remember("remember me")),
    ("update_thought", lambda ro: ro.update_thought("t-001", essence="x")),
    ("restore_thought", lambda ro: ro.restore_thought("t-001")),
    ("cleanup_expired", lambda ro: ro.cleanup_expired()),
    ("delete_thought", lambda ro: ro.delete_thought("t-001")),
    ("record_access", lambda ro: ro.record_access("t-001")),
    ("create_edge", lambda ro: ro.create_edge(_make_edge())),
    ("update_edge", lambda ro: ro.update_edge("e-001", weight=0.5)),
    ("delete_edge", lambda ro: ro.delete_edge("e-001")),
    ("store_embedding", lambda ro: ro.store_embedding("t-001", [0.1, 0.2, 0.3])),
    ("create_action", lambda ro: ro.create_action(_make_action())),
    ("update_action", lambda ro: ro.update_action("a-001", status=ActionStatus.EXECUTING)),
    ("derive_existing", lambda ro: ro.derive_existing("t-001")),
]

_WRITE_IDS = [name for name, _ in _WRITE_CASES]


# The same writes invoked with KEYWORD arguments under the wrapped store's real
# parameter names.  These bind only if the blockers keep the protocol parameter
# names (no leading underscore) — otherwise the call fails with ``TypeError`` at
# argument binding before ``ReadOnlyViolationError`` is ever raised.
_WRITE_KWARG_CASES: list[tuple[str, Callable[[ReadOnlyEngrava], Awaitable[object]]]] = [
    ("create_thought", lambda ro: ro.create_thought(thought=_make_thought())),
    ("get_or_create", lambda ro: ro.get_or_create(thought=_make_thought())),
    ("upsert_by_hash", lambda ro: ro.upsert_by_hash(thought=_make_thought())),
    ("bulk_store", lambda ro: ro.bulk_store(thoughts=[_make_thought()])),
    ("remember", lambda ro: ro.remember(text="remember me")),
    ("update_thought", lambda ro: ro.update_thought(thought_id="t-001", essence="x")),
    ("restore_thought", lambda ro: ro.restore_thought(thought_id="t-001")),
    ("cleanup_expired", lambda ro: ro.cleanup_expired(exclude_id="t-001")),
    ("delete_thought", lambda ro: ro.delete_thought(thought_id="t-001")),
    ("record_access", lambda ro: ro.record_access(thought_id="t-001")),
    ("create_edge", lambda ro: ro.create_edge(edge=_make_edge())),
    ("update_edge", lambda ro: ro.update_edge(edge_id="e-001", weight=0.5)),
    ("delete_edge", lambda ro: ro.delete_edge(edge_id="e-001")),
    ("store_embedding", lambda ro: ro.store_embedding(thought_id="t-001", vector=[0.1, 0.2])),
    ("create_action", lambda ro: ro.create_action(action=_make_action())),
    (
        "update_action",
        lambda ro: ro.update_action(action_id="a-001", status=ActionStatus.EXECUTING),
    ),
    ("derive_existing", lambda ro: ro.derive_existing(thought_id="t-001")),
]

_WRITE_KWARG_IDS = [name for name, _ in _WRITE_KWARG_CASES]


# Reads whose concrete backend stages a deferred access-frequency write.  Each
# is invoked identically on the raw store and on the read-only view.
_BUFFERING_READS: list[
    tuple[str, Callable[[SqliteEngravaCore | ReadOnlyEngrava], Awaitable[object]]]
] = [
    ("get_thought", lambda store: store.get_thought("t-1")),
    (
        "search_hybrid",
        lambda store: store.search_hybrid("hello", query_vector=[1.0, 0.0, 0.0], top_k=5),
    ),
]

_BUFFERING_READ_IDS = [name for name, _ in _BUFFERING_READS]


# ---------------------------------------------------------------------------
# Protocol parity — the adapter satisfies its declared read protocol
# ---------------------------------------------------------------------------


class TestReadOnlyEngravaProtocolParity:
    """ReadOnlyEngrava is substitutable for EngravaReadProtocol."""

    def test_satisfies_read_protocol_isinstance(self, ro_store: ReadOnlyEngrava) -> None:
        assert isinstance(ro_store, EngravaReadProtocol)

    def test_every_read_protocol_method_is_present(self, ro_store: ReadOnlyEngrava) -> None:
        """The adapter exposes every method declared on EngravaReadProtocol."""
        read_methods = [name for name in vars(EngravaReadProtocol) if not name.startswith("_")]
        assert read_methods  # guard against an empty (mis-imported) protocol
        for name in read_methods:
            assert callable(getattr(ro_store, name)), name

    def test_usable_where_read_protocol_expected(self, ro_store: ReadOnlyEngrava) -> None:
        """A read-protocol-typed consumer accepts the view unchanged."""

        def _accepts(store: EngravaReadProtocol) -> EngravaReadProtocol:
            return store

        assert _accepts(ro_store) is ro_store

    async def test_readonly_guarantee_is_behavioural_not_structural(
        self, ro_store: ReadOnlyEngrava
    ) -> None:
        """The write blockers exist as real members, so the view structurally
        matches the full core protocol too — the read-only guarantee is that
        every write *raises*, not that the write names are absent.
        """
        # ``runtime_checkable`` matches on member names, so the blockers make the
        # view an instance of the full core protocol as well.
        assert isinstance(ro_store, EngravaCoreProtocol)
        # ...but the behaviour is what enforces read-only: a write raises.
        with pytest.raises(ReadOnlyViolationError):
            await ro_store.create_thought(_make_thought())


# ---------------------------------------------------------------------------
# Read operations — must delegate to inner store
# ---------------------------------------------------------------------------


class TestReadOnlyEngravaRead:
    """Read operations are properly delegated to the inner store."""

    async def test_get_thought_existing(
        self,
        inner_store: SqliteEngravaCore,
        ro_store: ReadOnlyEngrava,
    ) -> None:
        await inner_store.create_thought(_make_thought("t-001"))
        result = await ro_store.get_thought("t-001")
        assert result is not None
        assert result.thought_id == "t-001"
        assert result.essence == "Test thought"

    async def test_get_thought_missing_returns_none(self, ro_store: ReadOnlyEngrava) -> None:
        result = await ro_store.get_thought("nonexistent")
        assert result is None

    async def test_list_thoughts_returns_all(
        self,
        inner_store: SqliteEngravaCore,
        ro_store: ReadOnlyEngrava,
    ) -> None:
        await inner_store.create_thought(_make_thought("t-001"))
        await inner_store.create_thought(_make_thought("t-002"))
        results = await ro_store.list_thoughts(limit=10)
        assert len(results) == 2

    async def test_list_thoughts_empty_store(self, ro_store: ReadOnlyEngrava) -> None:
        results = await ro_store.list_thoughts()
        assert results == []

    async def test_list_thoughts_filter_priority(
        self,
        inner_store: SqliteEngravaCore,
        ro_store: ReadOnlyEngrava,
    ) -> None:
        await inner_store.create_thought(_make_thought("t-p2", priority=Priority.P2))
        await inner_store.create_thought(_make_thought("t-p1", priority=Priority.P1))
        p1_results = await ro_store.list_thoughts(priority="P1")
        assert len(p1_results) == 1
        assert p1_results[0].thought_id == "t-p1"

    async def test_list_thoughts_filter_lifecycle_status(
        self,
        inner_store: SqliteEngravaCore,
        ro_store: ReadOnlyEngrava,
    ) -> None:
        await inner_store.create_thought(
            _make_thought("t-active", lifecycle_status=LifecycleStatus.ACTIVE)
        )
        await inner_store.create_thought(
            _make_thought("t-created", lifecycle_status=LifecycleStatus.CREATED)
        )
        active = await ro_store.list_thoughts(lifecycle_status="ACTIVE")
        assert len(active) == 1
        assert active[0].thought_id == "t-active"

    async def test_list_thoughts_filter_visibility(
        self,
        inner_store: SqliteEngravaCore,
        ro_store: ReadOnlyEngrava,
    ) -> None:
        await inner_store.create_thought(
            _make_thought("t-public", visibility=ThoughtVisibility.PUBLIC)
        )
        await inner_store.create_thought(
            _make_thought("t-private", visibility=ThoughtVisibility.PRIVATE)
        )
        public = await ro_store.list_thoughts(visibility="public")
        assert len(public) == 1
        assert public[0].thought_id == "t-public"

    async def test_list_thoughts_limit_and_offset(
        self,
        inner_store: SqliteEngravaCore,
        ro_store: ReadOnlyEngrava,
    ) -> None:
        for i in range(5):
            await inner_store.create_thought(_make_thought(f"t-{i:03d}"))
        page_1 = await ro_store.list_thoughts(limit=2, offset=0)
        page_2 = await ro_store.list_thoughts(limit=2, offset=2)
        assert len(page_1) == 2
        assert len(page_2) == 2
        assert {t.thought_id for t in page_1}.isdisjoint({t.thought_id for t in page_2})

    async def test_search_similar_delegates(
        self,
        inner_store: SqliteEngravaCore,
        ro_store: ReadOnlyEngrava,
    ) -> None:
        await inner_store.create_thought(_make_thought("t-001"))
        await inner_store.store_embedding("t-001", [1.0, 0.0, 0.0])
        results = await ro_store.search_similar([1.0, 0.0, 0.0], top_k=5)
        assert len(results) == 1
        assert results[0][0] == "t-001"
        assert results[0][1] > 0.99

    async def test_search_similar_empty_store(self, ro_store: ReadOnlyEngrava) -> None:
        results = await ro_store.search_similar([1.0, 0.0, 0.0])
        assert results == []

    async def test_search_similar_with_threshold_filters(
        self,
        inner_store: SqliteEngravaCore,
        ro_store: ReadOnlyEngrava,
    ) -> None:
        await inner_store.create_thought(_make_thought("t-001"))
        await inner_store.store_embedding("t-001", [1.0, 0.0, 0.0])
        # Orthogonal vector → similarity ~0.0, below threshold
        results = await ro_store.search_similar([0.0, 1.0, 0.0], threshold=0.99)
        assert results == []

    async def test_search_similar_top_k(
        self,
        inner_store: SqliteEngravaCore,
        ro_store: ReadOnlyEngrava,
    ) -> None:
        for i in range(5):
            await inner_store.create_thought(_make_thought(f"t-{i:03d}"))
            await inner_store.store_embedding(f"t-{i:03d}", [1.0, float(i) * 0.1, 0.0])
        results = await ro_store.search_similar([1.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2

    async def test_get_edges_delegates(
        self,
        inner_store: SqliteEngravaCore,
        ro_store: ReadOnlyEngrava,
    ) -> None:
        await inner_store.create_thought(_make_thought("t-a"))
        await inner_store.create_thought(_make_thought("t-b"))
        await inner_store.create_edge(_make_edge("e-001", "t-a", "t-b"))
        edges = await ro_store.get_edges("t-a")
        assert len(edges) == 1
        assert edges[0].edge_id == "e-001"

    async def test_get_edges_no_edges(
        self,
        inner_store: SqliteEngravaCore,
        ro_store: ReadOnlyEngrava,
    ) -> None:
        await inner_store.create_thought(_make_thought("t-001"))
        edges = await ro_store.get_edges("t-001")
        assert edges == []

    async def test_get_edges_direction_in(
        self,
        inner_store: SqliteEngravaCore,
        ro_store: ReadOnlyEngrava,
    ) -> None:
        await inner_store.create_thought(_make_thought("t-a"))
        await inner_store.create_thought(_make_thought("t-b"))
        edge = EdgeRecord(
            edge_id="e-001",
            from_thought_id="t-a",
            to_thought_id="t-b",
            edge_type=EdgeType.DEPENDS_ON,
            weight=1.0,
            created_cycle=0,
        )
        await inner_store.create_edge(edge)
        # t-b is the target, so direction="IN" from t-b's perspective
        edges = await ro_store.get_edges("t-b", direction="IN")
        assert len(edges) == 1
        assert edges[0].from_thought_id == "t-a"

    async def test_get_embedding_delegates(
        self,
        inner_store: SqliteEngravaCore,
        ro_store: ReadOnlyEngrava,
    ) -> None:
        await inner_store.create_thought(_make_thought("t-001"))
        await inner_store.store_embedding("t-001", [0.1, 0.2, 0.3])
        emb = await ro_store.get_embedding("t-001")
        assert emb is not None
        assert emb.dimension == 3
        assert emb.owner_id == "t-001"

    async def test_get_embedding_missing_returns_none(self, ro_store: ReadOnlyEngrava) -> None:
        result = await ro_store.get_embedding("nonexistent")
        assert result is None

    async def test_get_actions_delegates(
        self,
        inner_store: SqliteEngravaCore,
        ro_store: ReadOnlyEngrava,
    ) -> None:
        await inner_store.create_thought(_make_thought("t-001"))
        await inner_store.create_action(_make_action("a-001", "t-001"))
        actions = await ro_store.get_actions("t-001")
        assert len(actions) == 1
        assert actions[0].action_id == "a-001"

    async def test_list_edges_delegates(
        self,
        inner_store: SqliteEngravaCore,
        ro_store: ReadOnlyEngrava,
    ) -> None:
        await inner_store.create_thought(_make_thought("t-a"))
        await inner_store.create_thought(_make_thought("t-b"))
        await inner_store.create_edge(_make_edge("e-001", "t-a", "t-b"))
        edges = await ro_store.list_edges()
        assert {e.edge_id for e in edges} == {"e-001"}

    async def test_metrics_delegates(
        self,
        inner_store: SqliteEngravaCore,
        ro_store: ReadOnlyEngrava,
    ) -> None:
        await inner_store.create_thought(_make_thought("t-001"))
        ro_metrics = await ro_store.metrics()
        inner_metrics = await inner_store.metrics()
        assert ro_metrics.thoughts.total == inner_metrics.thoughts.total == 1

    async def test_max_cycle_delegates(
        self,
        inner_store: SqliteEngravaCore,
        ro_store: ReadOnlyEngrava,
    ) -> None:
        await inner_store.create_thought(_make_thought("t-001"))
        assert await ro_store.max_cycle() == await inner_store.max_cycle()


# ---------------------------------------------------------------------------
# Newly-forwarded reads — recall / search_fts / search_hybrid + provenance
# ---------------------------------------------------------------------------


class TestReadOnlyEngravaNewReads:
    """Reads added to the protocol are forwarded, not raised as AttributeError."""

    async def test_search_fts_delegates(
        self,
        inner_store: SqliteEngravaCore,
        ro_store: ReadOnlyEngrava,
    ) -> None:
        await inner_store.create_thought(_make_thought("t-001"))
        assert await ro_store.search_fts("Test") == await inner_store.search_fts("Test")

    async def test_search_hybrid_delegates(
        self,
        inner_store: SqliteEngravaCore,
        ro_store: ReadOnlyEngrava,
    ) -> None:
        await inner_store.create_thought(_make_thought("t-001"))
        await inner_store.store_embedding("t-001", [1.0, 0.0, 0.0])
        ro_res = await ro_store.search_hybrid("Test", query_vector=[1.0, 0.0, 0.0], top_k=5)
        inner_res = await inner_store.search_hybrid("Test", query_vector=[1.0, 0.0, 0.0], top_k=5)
        assert [tid for tid, _ in ro_res.results] == [tid for tid, _ in inner_res.results]
        assert "t-001" in [tid for tid, _ in ro_res.results]

    async def test_recall_delegates(
        self,
        inner_store: SqliteEngravaCore,
        ro_store: ReadOnlyEngrava,
    ) -> None:
        await inner_store.create_thought(_make_thought("t-001"))
        await inner_store.store_embedding("t-001", [1.0, 0.0, 0.0])
        ro_res = await ro_store.recall("Test", top_k=5)
        inner_res = await inner_store.recall("Test", top_k=5)
        assert [tid for tid, _ in ro_res.results] == [tid for tid, _ in inner_res.results]

    async def test_list_thoughts_provenance_filter_forwarded(
        self,
        inner_store: SqliteEngravaCore,
        ro_store: ReadOnlyEngrava,
    ) -> None:
        """The ``provenance_filter`` keyword reaches the wrapped store (no TypeError)."""
        await inner_store.create_thought(_make_thought("t-001"))
        prov_filter = MetadataFilter([FieldPredicate("$.session_id", FieldOp.EQ, "sess-A")])
        ro_result = await ro_store.list_thoughts(provenance_filter=prov_filter)
        inner_result = await inner_store.list_thoughts(provenance_filter=prov_filter)
        assert [t.thought_id for t in ro_result] == [t.thought_id for t in inner_result]


# ---------------------------------------------------------------------------
# Write operations — must raise ReadOnlyViolationError
# ---------------------------------------------------------------------------


class TestReadOnlyEngravaWriteBlocked:
    """Every write capability raises ReadOnlyViolationError naming the method."""

    @pytest.mark.parametrize(("name", "invoke"), _WRITE_CASES, ids=_WRITE_IDS)
    async def test_write_raises_and_names_method(
        self,
        ro_store: ReadOnlyEngrava,
        name: str,
        invoke: Callable[[ReadOnlyEngrava], Awaitable[object]],
    ) -> None:
        with pytest.raises(ReadOnlyViolationError, match=name) as exc_info:
            await invoke(ro_store)
        assert exc_info.value.operation == name

    @pytest.mark.parametrize(("name", "invoke"), _WRITE_KWARG_CASES, ids=_WRITE_KWARG_IDS)
    async def test_write_keyword_form_raises_not_typeerror(
        self,
        ro_store: ReadOnlyEngrava,
        name: str,
        invoke: Callable[[ReadOnlyEngrava], Awaitable[object]],
    ) -> None:
        """Keyword-form writes must bind and raise ReadOnlyViolationError.

        Regression guard: if a blocker renamed a protocol parameter (e.g. to a
        leading-underscore name to dodge unused-arg lint), a keyword call would
        fail at argument binding with ``TypeError`` before the body runs.
        """
        with pytest.raises(ReadOnlyViolationError, match=name) as exc_info:
            await invoke(ro_store)
        assert exc_info.value.operation == name

    async def test_store_embedding_model_kwarg_still_raises(
        self, ro_store: ReadOnlyEngrava
    ) -> None:
        """Keyword arguments are accepted but the error is still raised."""
        with pytest.raises(ReadOnlyViolationError):
            await ro_store.store_embedding("t-001", [0.1, 0.2], model_name="custom-model")

    async def test_no_write_touched_the_store(
        self,
        inner_store: SqliteEngravaCore,
        ro_store: ReadOnlyEngrava,
    ) -> None:
        """A rejected write never reaches the wrapped store."""
        await inner_store.create_thought(_make_thought("t-001"))
        for _name, invoke in _WRITE_CASES:
            with pytest.raises(ReadOnlyViolationError):
                await invoke(ro_store)
        # The one pre-existing thought is intact; nothing was created or deleted.
        remaining = await inner_store.list_thoughts(limit=100)
        assert {t.thought_id for t in remaining} == {"t-001"}


# ---------------------------------------------------------------------------
# Access-tracking policy — delegated reads never stage an access write
# ---------------------------------------------------------------------------


class TestReadOnlyEngravaAccessTracking:
    """Reads through the view suppress the wrapped store's access tracking."""

    @pytest.mark.parametrize(("name", "invoke"), _BUFFERING_READS, ids=_BUFFERING_READ_IDS)
    async def test_read_through_view_does_not_feed_access_tracking(
        self,
        tracking_store: SqliteEngravaCore,
        name: str,
        invoke: Callable[[SqliteEngravaCore | ReadOnlyEngrava], Awaitable[object]],
    ) -> None:
        ro = ReadOnlyEngrava(tracking_store)
        await tracking_store.create_thought(
            _make_thought("t-1", visibility=ThoughtVisibility.PUBLIC)
        )
        await tracking_store.store_embedding("t-1", [1.0, 0.0, 0.0])
        # Drain any setup noise so the buffer starts empty.
        await tracking_store.flush_access_buffer()

        # Control: the same read issued directly on the store buffers an access.
        await invoke(tracking_store)
        control = await tracking_store.flush_access_buffer()
        assert control >= 1, f"precondition: direct {name} must feed access tracking"

        # A read through the read-only view stages no access write.
        await invoke(ro)
        assert await tracking_store.flush_access_buffer() == 0

    async def test_record_access_still_blocked_under_tracking(
        self,
        tracking_store: SqliteEngravaCore,
    ) -> None:
        """The explicit access-recording write stays blocked on the view."""
        ro = ReadOnlyEngrava(tracking_store)
        await tracking_store.create_thought(_make_thought("t-1"))
        with pytest.raises(ReadOnlyViolationError, match="record_access"):
            await ro.record_access("t-1")

    async def test_overlapping_view_reads_never_record_access(
        self,
        tracking_store: SqliteEngravaCore,
    ) -> None:
        """Concurrent suppressed reads must not clobber each other's suppression.

        The suppression flag is task-local, so two overlapping reads through the
        view stay suppressed regardless of how their enter/exit interleave — an
        instance-wide save/restore boolean would let one read's exit re-enable
        buffering while the other is still in flight.
        """
        ro = ReadOnlyEngrava(tracking_store)
        for i in range(4):
            await tracking_store.create_thought(_make_thought(f"t-{i}"))
        await tracking_store.flush_access_buffer()  # clean start

        # Run several overlapping view reads as distinct tasks so their internal
        # DB awaits interleave.
        await asyncio.gather(*(ro.get_thought(f"t-{i}") for i in range(4)))
        assert await tracking_store.flush_access_buffer() == 0

    async def test_overlapping_direct_read_still_tracked(
        self,
        tracking_store: SqliteEngravaCore,
    ) -> None:
        """A direct read keeps its own access tracking while a view read runs.

        Task-local suppression means the view read suppresses only its own task;
        a concurrent direct read on the same store still feeds the signal.
        """
        ro = ReadOnlyEngrava(tracking_store)
        await tracking_store.create_thought(_make_thought("t-1"))
        await tracking_store.create_thought(_make_thought("t-2"))
        await tracking_store.flush_access_buffer()  # clean start

        await asyncio.gather(ro.get_thought("t-1"), tracking_store.get_thought("t-2"))
        # Only the direct read buffered; the view read stayed suppressed.
        assert await tracking_store.flush_access_buffer() == 1


class TestReadOnlyEngravaConstructor:
    """The read-only guarantee requires a suppression-capable backend."""

    def test_wrapping_non_suppressible_store_raises(self) -> None:
        """A store without suppress_access_tracking cannot be wrapped read-only."""

        class _NotSuppressible:
            """A stand-in that lacks the side-effect-free-read capability."""

        with pytest.raises(TypeError, match="suppress_access_tracking"):
            ReadOnlyEngrava(_NotSuppressible())  # type: ignore[arg-type]

    def test_wrapping_core_store_succeeds(self, inner_store: SqliteEngravaCore) -> None:
        """A full core store satisfies the suppression requirement."""
        assert isinstance(ReadOnlyEngrava(inner_store), ReadOnlyEngrava)


# ---------------------------------------------------------------------------
# Wrapping behaviour — inner store isolation
# ---------------------------------------------------------------------------


class TestReadOnlyEngravaIsolation:
    """ReadOnlyEngrava is a thin wrapper; mutations via inner store are visible."""

    async def test_mutation_via_inner_visible_through_ro(
        self,
        inner_store: SqliteEngravaCore,
        ro_store: ReadOnlyEngrava,
    ) -> None:
        await inner_store.create_thought(_make_thought("t-001"))
        await inner_store.update_thought("t-001", essence="Updated essence")
        result = await ro_store.get_thought("t-001")
        assert result is not None
        assert result.essence == "Updated essence"

    async def test_deletion_via_inner_visible_through_ro(
        self,
        inner_store: SqliteEngravaCore,
        ro_store: ReadOnlyEngrava,
    ) -> None:
        await inner_store.create_thought(_make_thought("t-001"))
        assert await ro_store.get_thought("t-001") is not None
        await inner_store.delete_thought("t-001")
        assert await ro_store.get_thought("t-001") is None

    async def test_multiple_ro_wrappers_share_same_inner(
        self,
        inner_store: SqliteEngravaCore,
    ) -> None:
        ro1 = ReadOnlyEngrava(inner_store)
        ro2 = ReadOnlyEngrava(inner_store)
        await inner_store.create_thought(_make_thought("t-001"))
        assert await ro1.get_thought("t-001") is not None
        assert await ro2.get_thought("t-001") is not None
