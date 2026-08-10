"""Integration tests for ReadOnlyEngrava.

Verifies that:
1. The adapter satisfies its declared runtime-checkable read protocol.
2. All read operations are delegated to the inner Engrava store.
3. Every write operation raises ReadOnlyViolationError unconditionally.
4. Reads with hidden access-tracking writes are delegated with the wrapped
   store's access tracking suppressed (no read stages an access-count write).

The read and write capability sets are **derived from the protocols**, never
restated here: the core protocol extends the read protocol with exactly the
writable capabilities, so a member the read half does not declare is a write.
A capability added to the contract therefore joins these suites by
construction — a hand-listed set would keep passing while quietly covering less
of the surface it exists to protect.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import inspect
from typing import TYPE_CHECKING, NamedTuple, Protocol

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


async def _seed_readable_store(store: SqliteEngravaCore) -> None:
    """Give *store* one row of every kind the read surface can return.

    Every read case below runs against this state, so no read is a no-op that
    would stage nothing simply because the store is empty.
    """
    await store.create_thought(_make_thought("t-1", visibility=ThoughtVisibility.PUBLIC))
    await store.create_thought(_make_thought("t-2", visibility=ThoughtVisibility.PUBLIC))
    await store.store_embedding("t-1", [1.0, 0.0, 0.0])
    await store.create_edge(_make_edge("e-1", "t-1", "t-2"))
    await store.create_action(_make_action("a-1", "t-1"))
    await store.flush_access_buffer()  # start every measurement from an empty buffer


async def _store_snapshot(store: SqliteEngravaCore) -> tuple[object, ...]:
    """Return every record the store holds, for a whole-state before/after check.

    The records are value objects, so comparing two snapshots detects a changed
    field just as well as a deleted row — an identifier set would not.
    """
    return (
        await store.list_thoughts(limit=1000, include_expired=True),
        await store.list_edges(),
        await store.get_actions("t-1"),
        await store.get_embedding("t-1"),
        await store.max_cycle(),
    )


# ---------------------------------------------------------------------------
# Derived capability sets — the protocol split is the source of truth
# ---------------------------------------------------------------------------


def _public_members(cls: type) -> frozenset[str]:
    """Return the public member names *cls* declares, across its own hierarchy.

    Args:
        cls: A protocol or class whose declared capability surface is wanted.

    Returns:
        Every non-underscore name in the class body of *cls* or of any base.

    """
    return frozenset(
        name for klass in cls.__mro__ for name in vars(klass) if not name.startswith("_")
    )


# ``EngravaCoreProtocol`` extends ``EngravaReadProtocol`` with exactly the
# writable capabilities, so set subtraction *is* the read/write rule: a core
# member the read protocol does not declare is a write, and one it does declare
# is a read.  Nothing is excluded by name — adding a read to the read protocol
# keeps it out of the write suite, and adding a write to the core protocol puts
# it in, with no edit here.
_READ_CAPABILITIES = _public_members(EngravaReadProtocol)
_WRITE_CAPABILITIES = _public_members(EngravaCoreProtocol) - _READ_CAPABILITIES
_READ_IDS = sorted(_READ_CAPABILITIES)
_WRITE_IDS = sorted(_WRITE_CAPABILITIES)


def _call_shape(store: object, name: str) -> tuple[tuple[str, str, bool], ...]:
    """Return the caller-visible shape of ``store.name``.

    Taken from the **bound** method, so Python itself drops the receiver — no
    rule here has to know what the implementation named it.  Each parameter is
    reduced to ``(name, kind, optional)``, and variadics are kept: a blocker
    that grows a ``**kwargs`` catch-all, flips a parameter between positional
    and keyword-only, or turns an optional parameter into a required one all
    change this shape, and each of those is a real behavioural difference from
    the store the caller thinks they are talking to.

    Args:
        store: An instance exposing the capability.
        name: The capability to inspect.

    Returns:
        One ``(parameter name, parameter kind, has a default)`` triple per
        parameter, in declaration order.

    """
    return tuple(
        (parameter.name, parameter.kind.name, parameter.default is not inspect.Parameter.empty)
        for parameter in inspect.signature(getattr(store, name)).parameters.values()
    )


def _unclassified_members(view: type) -> frozenset[str]:
    """Return the public members of *view* that the core contract does not declare.

    Args:
        view: A read-only view class over an Engrava store.

    Returns:
        Names that are neither a declared read nor a declared write — members no
        delegation or rejection assertion covers.

    """
    return _public_members(view) - _READ_CAPABILITIES - _WRITE_CAPABILITIES


# A blocker must reject *whatever* it is handed, so every call form is issued
# with more than one placeholder value: one that a naive ``if x is None`` guard
# would catch, and one it would not.
_PLACEHOLDERS: tuple[object, ...] = (None, "a real-looking argument")


class _CallForm(NamedTuple):
    """One way a caller can invoke a capability, and where its arguments go."""

    label: str
    positional_count: int
    keyword_names: tuple[str, ...]

    def filled_args(self, placeholder: object) -> tuple[object, ...]:
        """Return the positional arguments, every slot holding *placeholder*."""
        return (placeholder,) * self.positional_count

    def filled_kwargs(self, placeholder: object) -> dict[str, object]:
        """Return the keyword arguments, every value holding *placeholder*."""
        return dict.fromkeys(self.keyword_names, placeholder)


def _blocker_call_forms(store: object, name: str) -> tuple[_CallForm, ...]:
    """Return the ways a caller can invoke the write *name* on the wrapped store.

    Built from the live bound signature rather than written out, so the view's
    blocker is exercised against every parameter the store really accepts —
    including any added later.  Three forms, because each catches a different
    way a blocker can drift from the store it stands in for:

    * **positional** — everything supplied the way a positional caller would;
    * **keyword** — everything supplied by name, which fails with ``TypeError``
      if a blocker renamed a parameter (e.g. to a leading underscore to dodge
      unused-argument lint);
    * **required-only** — every optional parameter omitted, which fails if a
      blocker made one of them required.

    A blocker raises before it reads its arguments, so the values are irrelevant
    and only the *binding* is under test.

    Args:
        store: An instance whose signature defines the caller-visible shape.
        name: A derived write capability.

    Returns:
        One :class:`_CallForm` per calling style, in the order described above.

    """
    kind = inspect.Parameter
    parameters = [
        parameter
        for parameter in inspect.signature(getattr(store, name)).parameters.values()
        if parameter.kind not in (kind.VAR_POSITIONAL, kind.VAR_KEYWORD)
    ]
    positional_kinds = (kind.POSITIONAL_ONLY, kind.POSITIONAL_OR_KEYWORD)

    def _mostly_positional(label: str, chosen: list[inspect.Parameter]) -> _CallForm:
        return _CallForm(
            label,
            sum(1 for parameter in chosen if parameter.kind in positional_kinds),
            tuple(parameter.name for parameter in chosen if parameter.kind is kind.KEYWORD_ONLY),
        )

    by_name = _CallForm(
        "keyword",
        sum(1 for parameter in parameters if parameter.kind is kind.POSITIONAL_ONLY),
        tuple(
            parameter.name for parameter in parameters if parameter.kind is not kind.POSITIONAL_ONLY
        ),
    )
    required = [
        parameter for parameter in parameters if parameter.default is inspect.Parameter.empty
    ]
    return (
        _mostly_positional("positional", parameters),
        by_name,
        _mostly_positional("required-only", required),
    )


# Reads need real arguments to reach the wrapped store's retrieval paths, so each
# call is spelled out — but the *set* is pinned against the read protocol by
# ``test_every_read_capability_has_an_invocation``, so a read added to the
# contract fails this suite until it has a case.  The mapping is compared to its
# source of truth; it is not a second hand-maintained list of what to cover.
_READ_INVOKERS: dict[str, Callable[[SqliteEngravaCore | ReadOnlyEngrava], Awaitable[object]]] = {
    "get_actions": lambda store: store.get_actions("t-1"),
    "get_edges": lambda store: store.get_edges("t-1"),
    "get_embedding": lambda store: store.get_embedding("t-1"),
    "get_thought": lambda store: store.get_thought("t-1"),
    "list_edges": lambda store: store.list_edges(),
    "list_thoughts": lambda store: store.list_thoughts(limit=10),
    "max_cycle": lambda store: store.max_cycle(),
    "metrics": lambda store: store.metrics(),
    "recall": lambda store: store.recall("Test", top_k=5),
    "search_fts": lambda store: store.search_fts("Test", 5),
    "search_hybrid": lambda store: store.search_hybrid(
        "Test", query_vector=[1.0, 0.0, 0.0], top_k=5
    ),
    "search_similar": lambda store: store.search_similar([1.0, 0.0, 0.0], top_k=5),
}


class _Reached(NamedTuple):
    """One call that actually arrived at the wrapped store."""

    capability: str
    suppressed: bool


class _StoreSpy:
    """A store stand-in that records every call the view forwards to it.

    Sits between ``ReadOnlyEngrava`` and a real store and logs each declared
    capability that reaches it, together with whether the caller was inside
    ``suppress_access_tracking`` at that moment.  That gives two things a
    state-based assertion cannot:

    * every read carries a discriminating case, including the nine whose backend
      stages nothing and for which "no access write was staged" is vacuous;
    * "no write reached the store" becomes a direct observation rather than an
      inference from unchanged state, which a write that stores an identical
      value would survive.

    Suppression depth is a :class:`contextvars.ContextVar`, mirroring the real
    store: an instance-wide counter would report a concurrent *direct* call as
    suppressed merely because some other task happened to be inside a block.

    Attributes:
        inner: The real store the spy forwards to.
        reached: Every call that arrived, in order.

    """

    def __init__(self, inner: SqliteEngravaCore) -> None:
        self.inner = inner
        self.reached: list[_Reached] = []
        self.hold_before_delegating: asyncio.Event | None = None
        self.recorded = asyncio.Event()
        self._depth: contextvars.ContextVar[int] = contextvars.ContextVar(
            "read_only_spy_depth", default=0
        )

    def record(self, name: str) -> None:
        """Log that *name* reached the store, and under what suppression state."""
        self.reached.append(_Reached(name, self._depth.get() > 0))
        self.recorded.set()

    async def checkpoint(self) -> None:
        """Park a forwarded call so a test can interleave another one."""
        if self.hold_before_delegating is not None:
            await self.hold_before_delegating.wait()

    @contextlib.asynccontextmanager
    async def suppress_access_tracking(self) -> AsyncIterator[None]:
        """Enter the wrapped store's suppression block, task-locally."""
        token = self._depth.set(self._depth.get() + 1)
        try:
            async with self.inner.suppress_access_tracking():
                yield
        finally:
            self._depth.reset(token)


# Captured before the forwarders are installed, so a capability that collided
# with the spy's own instrumentation would be visible rather than silently
# replacing it.
_SPY_INSTRUMENTATION = frozenset(vars(_StoreSpy))


def _spy_forwarder(name: str) -> Callable[..., Awaitable[object]]:
    """Build the spy's forwarding method for one declared capability."""

    async def _recording(spy: _StoreSpy, *args: object, **kwargs: object) -> object:
        spy.record(name)
        await spy.checkpoint()
        return await getattr(spy.inner, name)(*args, **kwargs)

    _recording.__name__ = name
    _recording.__qualname__ = f"_StoreSpy.{name}"
    _recording.__doc__ = f"Forward ``{name}`` to the wrapped store, recording the call."
    return _recording


# Installed from the derived capability sets, so the spy presents exactly the
# surface the contract declares — a hand-written set of forwarders here would be
# the same defect this file exists to remove, and would additionally make the spy
# fail the view's structural check the moment the contract grew.
for _capability in (*_READ_IDS, *_WRITE_IDS):
    setattr(_StoreSpy, _capability, _spy_forwarder(_capability))


async def _await_calls(spy: _StoreSpy, count: int) -> None:
    """Wait until *spy* has recorded *count* calls, or give up rather than hang."""
    async with asyncio.timeout(5):
        while len(spy.reached) < count:
            spy.recorded.clear()
            await spy.recorded.wait()


# ---------------------------------------------------------------------------
# The derivation rule itself — it must select writes, not everything
# ---------------------------------------------------------------------------


class TestCapabilityDerivation:
    """The read/write split is computed from the protocols, not enumerated."""

    def test_the_core_protocol_really_extends_the_read_protocol(self) -> None:
        """The subtraction is only meaningful while the extension relation holds.

        If ``EngravaCoreProtocol`` stopped inheriting the read half, the write
        set would silently swell to include every read as well, and the whole
        suite would keep passing while asserting the wrong thing.
        """
        assert _READ_CAPABILITIES, "read protocol declared no members (mis-imported?)"
        assert _WRITE_CAPABILITIES, "core protocol added no writable capability (mis-imported?)"
        assert _public_members(EngravaCoreProtocol) > _READ_CAPABILITIES

    def test_derivation_keeps_writes_and_drops_reads(self) -> None:
        """The rule selects only what the extending protocol adds, not everything.

        The counter-case to the rule as arithmetic: a member the read half
        declares is *not* a write, even when the extending protocol re-declares
        it, and a member only the extending protocol declares *is*.  It pins the
        selection behaviour, not the meaning of the real protocol split — that
        meaning is the protocols' own layering, asserted above.
        """

        class _Reads(Protocol):
            def look(self) -> None: ...

        class _ReadsAndWrites(_Reads, Protocol):
            def look(self) -> None: ...  # re-declared read stays a read

            def mutate(self) -> None: ...

        assert _public_members(_ReadsAndWrites) - _public_members(_Reads) == {"mutate"}

    def test_no_capability_shadows_the_spy_instrumentation(
        self, inner_store: SqliteEngravaCore
    ) -> None:
        """A declared capability must not collide with the spy's own members.

        The spy's forwarders are installed by name from the derived sets, so a
        capability called ``record`` or ``reached`` would silently replace the
        instrumentation and every observation made through it would be junk.
        """
        spy = _StoreSpy(inner_store)
        instrumentation = _SPY_INSTRUMENTATION | frozenset(vars(spy))
        assert instrumentation.isdisjoint(_READ_CAPABILITIES | _WRITE_CAPABILITIES)

    def test_every_write_capability_is_a_method_on_the_core_store(self) -> None:
        """The concrete store implements every derived write as a callable.

        The derived call forms come from the store's own signature, so a write
        the store does not provide — or provides as a plain attribute — would
        leave the rejection cases with nothing to inspect.
        """
        not_a_method = [
            name for name in _WRITE_IDS if not callable(getattr(SqliteEngravaCore, name, None))
        ]
        assert not not_a_method, f"core store does not implement declared writes: {not_a_method}"


# ---------------------------------------------------------------------------
# Protocol parity — the adapter satisfies its declared read protocol
# ---------------------------------------------------------------------------


class TestReadOnlyEngravaProtocolParity:
    """ReadOnlyEngrava is substitutable for EngravaReadProtocol."""

    def test_satisfies_read_protocol_isinstance(self, ro_store: ReadOnlyEngrava) -> None:
        assert isinstance(ro_store, EngravaReadProtocol)

    def test_every_read_protocol_method_is_present(self, ro_store: ReadOnlyEngrava) -> None:
        """The adapter exposes every method declared on EngravaReadProtocol."""
        assert _READ_CAPABILITIES  # guard against an empty (mis-imported) protocol
        for name in _READ_IDS:
            assert callable(getattr(ro_store, name)), name

    def test_every_public_member_is_a_declared_read_or_write(self) -> None:
        """The view exposes nothing the core contract does not declare.

        This is what stops a *new* method quietly becoming a write path through
        a store the caller believes cannot write: a member that is neither a
        declared read (delegated) nor a declared write (rejected) is
        unclassified, and an unclassified member has been asserted about by
        nothing at all.
        """
        assert _unclassified_members(ReadOnlyEngrava) == frozenset()

    def test_a_member_outside_the_contract_is_reported_unclassified(self) -> None:
        """The counter-case: an added member the contract does not declare is caught."""

        class _ViewWithAnExtraMethod(ReadOnlyEngrava):
            async def purge_everything(self) -> None:
                """A capability the core contract never declared."""

        assert _unclassified_members(_ViewWithAnExtraMethod) == {"purge_everything"}

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

    @pytest.mark.parametrize("name", _WRITE_IDS)
    async def test_write_raises_and_names_method(
        self,
        inner_store: SqliteEngravaCore,
        ro_store: ReadOnlyEngrava,
        name: str,
    ) -> None:
        """Every way of calling a write raises, and the error names the method.

        Each derived calling style is exercised, with each placeholder value: a
        keyword-form call that fails to bind would raise ``TypeError`` before the
        blocker body ever runs, an omitted optional argument would do the same,
        and a blocker that rejected only ``None`` would delegate the rest.
        """
        for form in _blocker_call_forms(inner_store, name):
            for placeholder in _PLACEHOLDERS:
                with pytest.raises(ReadOnlyViolationError, match=name) as exc_info:
                    await getattr(ro_store, name)(
                        *form.filled_args(placeholder), **form.filled_kwargs(placeholder)
                    )
                assert exc_info.value.operation == name, form.label

    @pytest.mark.parametrize("name", _WRITE_IDS)
    def test_blocker_accepts_what_the_wrapped_store_accepts(
        self,
        inner_store: SqliteEngravaCore,
        ro_store: ReadOnlyEngrava,
        name: str,
    ) -> None:
        """A blocker presents the wrapped store's exact caller-visible shape.

        The view documents that a caller may pass a write through it exactly as
        they would to the real store and still get a coherent
        ``ReadOnlyViolationError`` rather than an opaque ``TypeError``.  That
        holds only while the shapes match — names, kinds and optionality alike —
        so the match is asserted rather than assumed.
        """
        assert _call_shape(ro_store, name) == _call_shape(inner_store, name)

    async def test_an_undeclared_keyword_does_not_bind_and_writes_nothing(
        self,
        inner_store: SqliteEngravaCore,
        ro_store: ReadOnlyEngrava,
    ) -> None:
        """A keyword outside the wrapped store's signature is refused, not absorbed.

        The counter-case to the signature-parity assertion above: the blockers
        mirror the store's parameters *exactly*, so a keyword neither declares
        fails to bind rather than being swallowed by a catch-all — and nothing
        reaches the store on that path either.
        """
        await inner_store.create_thought(_make_thought("t-001"))
        with pytest.raises(TypeError, match="not_a_real_keyword"):
            await ro_store.store_embedding("t-001", [0.1, 0.2], not_a_real_keyword=1)
        remaining = await inner_store.list_thoughts(limit=100)
        assert {t.thought_id for t in remaining} == {"t-001"}
        assert await inner_store.get_embedding("t-001") is None

    async def test_store_embedding_model_kwarg_still_raises(
        self, ro_store: ReadOnlyEngrava
    ) -> None:
        """Keyword arguments are accepted but the error is still raised."""
        with pytest.raises(ReadOnlyViolationError):
            await ro_store.store_embedding("t-001", [0.1, 0.2], model_name="custom-model")

    async def test_no_write_reached_the_wrapped_store(
        self,
        inner_store: SqliteEngravaCore,
    ) -> None:
        """A rejected write never reaches the wrapped store at all.

        Observed at the boundary rather than inferred from state: a delegated
        write that stored an identical value, or wrote and rolled back, would
        leave a snapshot unchanged and still be a write path through a store the
        caller believes cannot write.  The state check follows as corroboration.
        """
        spy = _StoreSpy(inner_store)
        view = ReadOnlyEngrava(spy)
        await _seed_readable_store(inner_store)
        before = await _store_snapshot(inner_store)

        for name in _WRITE_IDS:
            for form in _blocker_call_forms(inner_store, name):
                for placeholder in _PLACEHOLDERS:
                    with pytest.raises(ReadOnlyViolationError):
                        await getattr(view, name)(
                            *form.filled_args(placeholder), **form.filled_kwargs(placeholder)
                        )

        assert spy.reached == []
        assert await _store_snapshot(inner_store) == before

    async def test_the_spy_does_see_a_call_that_reaches_the_store(
        self,
        inner_store: SqliteEngravaCore,
    ) -> None:
        """The counter-case: an empty call log means nothing if nothing is logged.

        A delegated *read* through the same view is recorded, so the empty write
        log above is evidence that writes were blocked rather than evidence that
        the spy never records anything.
        """
        spy = _StoreSpy(inner_store)
        view = ReadOnlyEngrava(spy)
        await _seed_readable_store(inner_store)

        await view.get_thought("t-1")

        assert [call.capability for call in spy.reached] == ["get_thought"]


# ---------------------------------------------------------------------------
# Access-tracking policy — delegated reads never stage an access write
# ---------------------------------------------------------------------------


class TestReadOnlyEngravaAccessTracking:
    """Reads through the view suppress the wrapped store's access tracking."""

    def test_every_read_capability_has_an_invocation(self) -> None:
        """Every declared read is exercised below — no read is silently uncovered.

        The invocation mapping is compared against the read protocol rather than
        trusted: a read added to the contract with no case here fails *this*
        assertion instead of quietly leaving a delegated read unasserted.
        """
        assert frozenset(_READ_INVOKERS) == _READ_CAPABILITIES, (
            f"read capabilities without a case: "
            f"{sorted(_READ_CAPABILITIES - frozenset(_READ_INVOKERS))}; "
            f"cases for capabilities the protocol no longer declares: "
            f"{sorted(frozenset(_READ_INVOKERS) - _READ_CAPABILITIES)}"
        )

    @pytest.mark.parametrize("name", _READ_IDS)
    async def test_delegated_read_runs_under_suppression(
        self,
        tracking_store: SqliteEngravaCore,
        name: str,
    ) -> None:
        """Every delegated read reaches the wrapped store with suppression active.

        This is the guarantee as the view states it, and it is checkable for
        *every* read: only three of them stage an access write today, so an
        "it staged nothing" assertion is evidence for three cases and vacuous
        for the other nine.  A read that starts buffering later is already
        covered here, before anyone notices it changed.  Every recorded call is
        checked, not the last one and not only the calls that carry the expected
        name, so neither a repeated read nor a stray unsuppressed call to some
        *other* read can hide behind a filtered assertion.
        """
        spy = _StoreSpy(tracking_store)
        await _seed_readable_store(tracking_store)

        await _READ_INVOKERS[name](ReadOnlyEngrava(spy))

        assert spy.reached == [_Reached(name, True)]

    @pytest.mark.parametrize("name", _READ_IDS)
    async def test_the_same_read_called_directly_is_not_suppressed(
        self,
        tracking_store: SqliteEngravaCore,
        name: str,
    ) -> None:
        """The counter-case: suppression comes from the view, not from the spy.

        Without this, every call recording ``True`` would be indistinguishable
        from a spy that reports ``True`` unconditionally.
        """
        spy = _StoreSpy(tracking_store)
        await _seed_readable_store(tracking_store)

        await _READ_INVOKERS[name](spy)

        assert spy.reached == [_Reached(name, False)]

    async def test_a_concurrent_direct_read_is_not_counted_as_suppressed(
        self,
        tracking_store: SqliteEngravaCore,
    ) -> None:
        """Suppression is per task, so an overlapping direct read records ``False``.

        An instance-wide counter would report the direct read as suppressed
        merely because the view's read was in flight in another task, and every
        suppression case above would then be measuring task interleaving rather
        than the view.  The overlap is forced rather than hoped for: the view's
        call is parked inside its suppression block until the direct call has
        been recorded.
        """
        spy = _StoreSpy(tracking_store)
        view = ReadOnlyEngrava(spy)
        await _seed_readable_store(tracking_store)
        spy.hold_before_delegating = asyncio.Event()

        through_view = asyncio.create_task(view.get_thought("t-1"))
        await _await_calls(spy, 1)  # the view is now parked, still inside its block
        directly = asyncio.create_task(spy.get_thought("t-2"))  # type: ignore[attr-defined]
        await _await_calls(spy, 2)
        spy.hold_before_delegating.set()
        await asyncio.gather(through_view, directly)

        assert spy.reached == [
            _Reached("get_thought", True),
            _Reached("get_thought", False),
        ]

    @pytest.mark.parametrize("name", _READ_IDS)
    async def test_read_through_view_does_not_feed_access_tracking(
        self,
        tracking_store: SqliteEngravaCore,
        name: str,
    ) -> None:
        """No delegated read stages an access write, whichever read it is.

        The observable half of the guarantee above: suppression is only worth
        anything if nothing is actually buffered.
        """
        ro = ReadOnlyEngrava(tracking_store)
        await _seed_readable_store(tracking_store)

        await _READ_INVOKERS[name](ro)

        assert await tracking_store.flush_access_buffer() == 0

    async def test_the_same_reads_stage_writes_without_the_view(
        self,
        tracking_store: SqliteEngravaCore,
    ) -> None:
        """Control for the suppression cases: the measurement can see a write.

        Each case above asserts that a read through the view staged nothing,
        which is only evidence while a read is *able* to stage something.  The
        same reads issued directly on the wrapped store must therefore stage at
        least one access write — otherwise the suppression suite would stay
        green because access tracking stopped working, not because the view
        suppresses it.
        """
        await _seed_readable_store(tracking_store)

        staged_directly: set[str] = set()
        for name in _READ_IDS:
            await _READ_INVOKERS[name](tracking_store)
            if await tracking_store.flush_access_buffer() >= 1:
                staged_directly.add(name)

        assert staged_directly, (
            "no read staged an access write when called directly on the store — "
            "the suppression cases above cannot distinguish a suppressed read "
            "from a store that no longer tracks accesses at all"
        )

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
