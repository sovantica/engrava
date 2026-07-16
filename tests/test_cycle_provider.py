"""Tests for the opt-in cycle-provider seam.

Covers the ``CycleProvider`` protocol, the keyword-only / runtime-only
``cycle_provider`` argument, the ``is not None`` resolution rule (explicit
``0`` wins over a provider), the pulled-value validation trust boundary
(rejecting ``bool`` / negative / non-``int``), the read-only ``max_cycle()``
high-water accessor (including the edge-cycle-exceeds-thoughts case and
reachability through ``from_config``), the read-time-only guarantee (a provider
never stamps writes), the three reference providers, and per-consuming-path
resolution for ``search_hybrid`` / ``recall`` / ``consolidate`` / ``run_hygiene``.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import (
    CallableCycleProvider,
    CycleProvider,
    CycleProviderError,
    DreamingConfig,
    EdgeRecord,
    EdgeType,
    EngravaConfig,
    HygienePolicyConfig,
    KnowledgeSource,
    LifecycleStatus,
    MaxCycleProvider,
    Priority,
    ReadOnlyEngrava,
    SearchConfig,
    SqliteEngravaCore,
    StaticCycleProvider,
    ThoughtRecord,
    ThoughtType,
)
from engrava.infrastructure.sqlite.engrava_core import _validate_provider_cycle

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

_CORE_LOGGER = "engrava.infrastructure.sqlite.engrava_core"


# ---------------------------------------------------------------------------
# Test doubles + helpers
# ---------------------------------------------------------------------------


class _SpyProvider:
    """A valid provider that records how many times it was pulled."""

    def __init__(self, value: int) -> None:
        self._value = value
        self.calls = 0

    def current_cycle(self) -> int:
        self.calls += 1
        return self._value


class _FixedProvider:
    """A provider that returns exactly what it was handed, unchecked.

    Unlike :class:`StaticCycleProvider` it does not constrain the value's type,
    so it can feed the store an intentionally-invalid value and exercise the
    store-side value-validation trust boundary.
    """

    def __init__(self, value: object) -> None:
        self._value = value

    def current_cycle(self) -> int:
        # Intentionally unchecked: the whole point is to hand the store a value
        # the type system forbids so its runtime validation is exercised.
        return self._value  # type: ignore[return-value]


#: A search config with a non-zero recency weight, so that a resolved cycle
#: (explicit or provider-supplied) actually activates the recency backend — the
#: default store leaves recency un-weighted, which would mask the seam's effect.
_RECENCY_ON = SearchConfig(default_recency_weight=0.5)


async def _make_store(
    *,
    cycle_provider: CycleProvider | None = None,
    hygiene_policy: HygienePolicyConfig | None = None,
    search_config: SearchConfig | None = None,
    dreaming: bool = False,
) -> SqliteEngravaCore:
    """Build a bootstrapped in-memory store with the given wiring."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA foreign_keys = ON")
    store = SqliteEngravaCore(
        conn,
        hygiene_policy=hygiene_policy,
        search_config=search_config,
        cycle_provider=cycle_provider,
    )
    if dreaming:
        from engrava.extensions.dreaming import DreamingExtension

        store._dreaming_extension = DreamingExtension(config=DreamingConfig(enabled=True))
    await store.ensure_schema()
    return store


def _thought(
    thought_id: str,
    *,
    essence: str = "alpha topic",
    content: str = "alpha topic content body",
    created_cycle: int = 0,
    updated_cycle: int = 0,
    lifecycle_status: LifecycleStatus = LifecycleStatus.ACTIVE,
) -> ThoughtRecord:
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.OBSERVATION,
        essence=essence,
        content=content,
        priority=Priority.P3,
        lifecycle_status=lifecycle_status,
        created_cycle=created_cycle,
        updated_cycle=updated_cycle,
        source="test",
    )


async def _raw_cycles(store: SqliteEngravaCore, thought_id: str) -> tuple[int, int]:
    cursor = await store._db.execute(
        "SELECT created_cycle, updated_cycle FROM thought WHERE thought_id = ?",
        (thought_id,),
    )
    row = await cursor.fetchone()
    assert row is not None
    return int(row["created_cycle"]), int(row["updated_cycle"])


async def _raw_edge_created_cycle(store: SqliteEngravaCore, edge_id: str) -> int:
    cursor = await store._db.execute(
        "SELECT created_cycle FROM edge WHERE edge_id = ?",
        (edge_id,),
    )
    row = await cursor.fetchone()
    assert row is not None
    return int(row["created_cycle"])


async def _edge(store: SqliteEngravaCore, edge_id: str, *, created_cycle: int) -> None:
    await store.create_edge(
        EdgeRecord(
            edge_id=edge_id,
            from_thought_id="t1",
            to_thought_id="t2",
            edge_type=EdgeType.ASSOCIATED,
            weight=0.5,
            created_cycle=created_cycle,
            source=KnowledgeSource.EXPERIENCE,
        )
    )


@pytest.fixture
async def store() -> AsyncIterator[SqliteEngravaCore]:
    s = await _make_store()
    yield s
    await s._db.close()


# ---------------------------------------------------------------------------
# Reference providers
# ---------------------------------------------------------------------------


class TestReferenceProviders:
    def test_static_returns_fixed_value(self) -> None:
        assert StaticCycleProvider(7).current_cycle() == 7

    def test_static_returns_zero(self) -> None:
        assert StaticCycleProvider(0).current_cycle() == 0

    def test_callable_delegates_and_reflects_mutation(self) -> None:
        counter = {"cycle": 3}
        provider = CallableCycleProvider(lambda: counter["cycle"])
        assert provider.current_cycle() == 3
        counter["cycle"] = 4
        assert provider.current_cycle() == 4

    def test_all_reference_providers_satisfy_protocol(self) -> None:
        # runtime_checkable structural conformance (no explicit inheritance).
        assert isinstance(StaticCycleProvider(1), CycleProvider)
        assert isinstance(CallableCycleProvider(lambda: 1), CycleProvider)


# ---------------------------------------------------------------------------
# Pulled-value validation (the trust boundary)
# ---------------------------------------------------------------------------


class TestValidateProviderCycle:
    @pytest.mark.parametrize("value", [0, 1, 5, 1_000_000])
    def test_accepts_non_negative_int(self, value: int) -> None:
        assert _validate_provider_cycle(value) == value

    def test_rejects_bool_true(self) -> None:
        # ``bool`` subclasses ``int`` but must be rejected (type identity check).
        with pytest.raises(CycleProviderError, match="expected int, got bool"):
            _validate_provider_cycle(True)

    def test_rejects_bool_false(self) -> None:
        with pytest.raises(CycleProviderError, match="expected int, got bool"):
            _validate_provider_cycle(False)

    def test_rejects_negative(self) -> None:
        with pytest.raises(CycleProviderError, match="non-negative"):
            _validate_provider_cycle(-1)

    @pytest.mark.parametrize("value", ["3", 3.0, None, [3]])
    def test_rejects_non_int(self, value: object) -> None:
        with pytest.raises(CycleProviderError, match="expected int"):
            _validate_provider_cycle(value)


# ---------------------------------------------------------------------------
# Resolution rule (_resolve_current_cycle) — explicit-wins, provider, none
# ---------------------------------------------------------------------------


class TestResolveCurrentCycle:
    async def test_none_without_provider_stays_none(self, store: SqliteEngravaCore) -> None:
        assert store._resolve_current_cycle(None) is None

    async def test_explicit_without_provider_passthrough(self, store: SqliteEngravaCore) -> None:
        assert store._resolve_current_cycle(0) == 0
        assert store._resolve_current_cycle(5) == 5

    async def test_pulls_provider_when_none(self) -> None:
        spy = _SpyProvider(7)
        s = await _make_store(cycle_provider=spy)
        try:
            assert s._resolve_current_cycle(None) == 7
            assert spy.calls == 1
        finally:
            await s._db.close()

    async def test_explicit_value_wins_over_provider(self) -> None:
        spy = _SpyProvider(99)
        s = await _make_store(cycle_provider=spy)
        try:
            assert s._resolve_current_cycle(5) == 5
            assert spy.calls == 0  # provider never consulted when caller passed a value
        finally:
            await s._db.close()

    async def test_explicit_zero_wins_not_truthiness(self) -> None:
        # The crux: resolution uses ``is not None``, never truthiness, so an
        # explicit 0 wins and the provider is never pulled.
        spy = _SpyProvider(99)
        s = await _make_store(cycle_provider=spy)
        try:
            assert s._resolve_current_cycle(0) == 0
            assert spy.calls == 0
        finally:
            await s._db.close()

    async def test_invalid_provider_value_raises(self) -> None:
        s = await _make_store(cycle_provider=_FixedProvider(-1))
        try:
            with pytest.raises(CycleProviderError):
                s._resolve_current_cycle(None)
        finally:
            await s._db.close()


# ---------------------------------------------------------------------------
# max_cycle() high-water accessor (D3)
# ---------------------------------------------------------------------------


class TestMaxCycle:
    async def test_empty_store_is_zero(self, store: SqliteEngravaCore) -> None:
        assert await store.max_cycle() == 0

    async def test_advances_with_thoughts(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_thought("t1", updated_cycle=3))
        await store.create_thought(_thought("t2", updated_cycle=1))
        assert await store.max_cycle() == 3

    async def test_all_cycle_zero_store_is_zero(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_thought("t1", updated_cycle=0))
        await store.create_thought(_thought("t2", updated_cycle=0))
        assert await store.max_cycle() == 0

    async def test_edge_cycle_exceeds_thoughts(self, store: SqliteEngravaCore) -> None:
        # An edge created at a higher cycle than any thought must lift the mark:
        # thought-only would under-report it.
        await store.create_thought(_thought("t1", updated_cycle=3))
        await store.create_thought(_thought("t2", updated_cycle=2))
        await store.create_edge(
            EdgeRecord(
                edge_id="e1",
                from_thought_id="t1",
                to_thought_id="t2",
                edge_type=EdgeType.ASSOCIATED,
                weight=0.5,
                created_cycle=7,
                source=KnowledgeSource.EXPERIENCE,
            )
        )
        assert await store.max_cycle() == 7

    async def test_thought_cycle_exceeds_edges(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_thought("t1", updated_cycle=9))
        await store.create_thought(_thought("t2", updated_cycle=2))
        await store.create_edge(
            EdgeRecord(
                edge_id="e1",
                from_thought_id="t1",
                to_thought_id="t2",
                edge_type=EdgeType.ASSOCIATED,
                weight=0.5,
                created_cycle=4,
                source=KnowledgeSource.EXPERIENCE,
            )
        )
        assert await store.max_cycle() == 9

    async def test_reachable_and_read_only_via_read_only_wrapper(
        self, store: SqliteEngravaCore
    ) -> None:
        await store.create_thought(_thought("t1", updated_cycle=6))
        ro = ReadOnlyEngrava(store)
        assert await ro.max_cycle() == 6


# ---------------------------------------------------------------------------
# Provider is READ-TIME only — it never stamps writes (D2)
# ---------------------------------------------------------------------------


class TestProviderNeverStampsWrites:
    async def test_remember_keeps_cycle_zero(self) -> None:
        spy = _SpyProvider(42)
        s = await _make_store(cycle_provider=spy)
        try:
            t = await s.remember("a fresh memory")
            assert t.created_cycle == 0
            assert t.updated_cycle == 0
            assert await _raw_cycles(s, t.thought_id) == (0, 0)
            assert spy.calls == 0  # the write path never consults the provider
        finally:
            await s._db.close()

    async def test_create_thought_preserves_explicit_cycle(self) -> None:
        spy = _SpyProvider(42)
        s = await _make_store(cycle_provider=spy)
        try:
            await s.create_thought(_thought("t1", created_cycle=2, updated_cycle=4))
            assert await _raw_cycles(s, "t1") == (2, 4)
            assert spy.calls == 0
        finally:
            await s._db.close()

    async def test_bulk_store_keeps_cycles_and_ignores_provider(self) -> None:
        spy = _SpyProvider(42)
        s = await _make_store(cycle_provider=spy)
        try:
            await s.bulk_store(
                [
                    _thought("t1", created_cycle=0, updated_cycle=1),
                    _thought("t2", created_cycle=0, updated_cycle=2),
                ]
            )
            assert await _raw_cycles(s, "t1") == (0, 1)
            assert await _raw_cycles(s, "t2") == (0, 2)
            assert spy.calls == 0
        finally:
            await s._db.close()

    async def test_restore_thought_does_not_stamp_provider_cycle(self) -> None:
        # restore_thought is the only write path that even accepts current_cycle;
        # with it omitted, updated_cycle must stay unchanged even when a provider
        # is configured — the provider is read-time only, never a write stamp.
        spy = _SpyProvider(42)
        s = await _make_store(cycle_provider=spy)
        try:
            await s.create_thought(_thought("t1", updated_cycle=5))
            await s._db.execute(
                "UPDATE thought SET lifecycle_status = 'ARCHIVED' WHERE thought_id = 't1'"
            )
            await s._db.commit()
            restored = await s.restore_thought("t1")  # no explicit cycle
            assert restored.updated_cycle == 5  # not the provider's 42
            assert await _raw_cycles(s, "t1") == (0, 5)
            assert spy.calls == 0
        finally:
            await s._db.close()

    async def test_update_thought_never_consults_provider(self) -> None:
        # The read-time-only rule covers update_thought: it must preserve the
        # record's explicit cycle and make zero provider calls.
        spy = _SpyProvider(42)
        s = await _make_store(cycle_provider=spy)
        try:
            await s.create_thought(_thought("t1", updated_cycle=5))
            updated = await s.update_thought("t1", essence="a changed essence")
            assert updated.essence == "a changed essence"
            assert updated.updated_cycle == 5  # unchanged, never the provider's 42
            assert await _raw_cycles(s, "t1") == (0, 5)
            assert spy.calls == 0
        finally:
            await s._db.close()

    async def test_update_edge_never_consults_provider(self) -> None:
        # update_edge likewise preserves the edge's explicit created_cycle and
        # never consults the provider.
        spy = _SpyProvider(42)
        s = await _make_store(cycle_provider=spy)
        try:
            await s.create_thought(_thought("t1", updated_cycle=0))
            await s.create_thought(_thought("t2", updated_cycle=0))
            await _edge(s, "e1", created_cycle=7)
            updated = await s.update_edge("e1", weight=0.9)
            assert updated.weight == pytest.approx(0.9)
            assert updated.created_cycle == 7  # unchanged, never the provider's 42
            assert await _raw_edge_created_cycle(s, "e1") == 7
            assert spy.calls == 0
        finally:
            await s._db.close()


# ---------------------------------------------------------------------------
# search_hybrid / recall resolution (per consuming path)
# ---------------------------------------------------------------------------


class TestSearchHybridResolution:
    async def test_no_provider_none_cycle_recency_off(self) -> None:
        # Even with recency weighted, current_cycle=None and no provider keeps
        # recency inactive — today's behaviour, byte-for-byte.
        s = await _make_store(search_config=_RECENCY_ON)
        try:
            await s.create_thought(_thought("t1"))
            result = await s.search_hybrid("alpha", current_cycle=None)
            assert "recency" not in result.backends_used
        finally:
            await s._db.close()

    async def test_provider_activates_recency(self) -> None:
        s = await _make_store(cycle_provider=StaticCycleProvider(5), search_config=_RECENCY_ON)
        try:
            await s.create_thought(_thought("t1"))
            result = await s.search_hybrid("alpha", current_cycle=None)
            assert "recency" in result.backends_used
        finally:
            await s._db.close()

    async def test_provider_value_equals_explicit_value(self) -> None:
        # Byte-for-byte: a provider-supplied cycle produces the identical ranking
        # to passing the same cycle explicitly (recency weighted so it matters).
        s_provider = await _make_store(
            cycle_provider=StaticCycleProvider(3), search_config=_RECENCY_ON
        )
        s_explicit = await _make_store(search_config=_RECENCY_ON)
        try:
            for tid, uc in (("t1", 3), ("t2", 1), ("t3", 0)):
                await s_provider.create_thought(_thought(tid, updated_cycle=uc))
                await s_explicit.create_thought(_thought(tid, updated_cycle=uc))
            res_provider = await s_provider.search_hybrid("alpha", current_cycle=None)
            res_explicit = await s_explicit.search_hybrid("alpha", current_cycle=3)
            assert res_provider.results == res_explicit.results
            assert res_provider.backends_used == res_explicit.backends_used
            assert "recency" in res_provider.backends_used
        finally:
            await s_provider._db.close()
            await s_explicit._db.close()

    async def test_explicit_zero_wins_and_is_used(self) -> None:
        # 0 is a valid cycle: recency stays ACTIVE (it did not fall through to
        # None), and the provider is never pulled.
        spy = _SpyProvider(99)
        s = await _make_store(cycle_provider=spy, search_config=_RECENCY_ON)
        try:
            await s.create_thought(_thought("t1"))
            result = await s.search_hybrid("alpha", current_cycle=0)
            assert "recency" in result.backends_used
            assert spy.calls == 0
        finally:
            await s._db.close()

    async def test_invalid_provider_value_raises_on_search(self) -> None:
        s = await _make_store(cycle_provider=_FixedProvider(True))
        try:
            await s.create_thought(_thought("t1"))
            with pytest.raises(CycleProviderError):
                await s.search_hybrid("alpha", current_cycle=None)
        finally:
            await s._db.close()


class TestRecallResolution:
    async def test_recall_provider_activates_recency(self) -> None:
        # recall() takes no per-call recency weight, so the store carries one.
        s = await _make_store(cycle_provider=StaticCycleProvider(5), search_config=_RECENCY_ON)
        try:
            await s.create_thought(_thought("t1"))
            result = await s.recall("alpha", current_cycle=None)
            assert "recency" in result.backends_used
        finally:
            await s._db.close()

    async def test_recall_nudge_fires_without_provider(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        s = await _make_store()
        try:
            for i in range(30):  # exceed _RECENCY_NUDGE_THRESHOLD (25)
                await s.create_thought(_thought(f"t{i}"))
            with caplog.at_level(logging.DEBUG, logger=_CORE_LOGGER):
                await s.recall("alpha", current_cycle=None)
            assert any("without current_cycle" in rec.message for rec in caplog.records)
        finally:
            await s._db.close()

    async def test_recall_nudge_suppressed_with_provider(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        s = await _make_store(cycle_provider=StaticCycleProvider(1))
        try:
            for i in range(30):
                await s.create_thought(_thought(f"t{i}"))
            with caplog.at_level(logging.DEBUG, logger=_CORE_LOGGER):
                await s.recall("alpha", current_cycle=None)
            assert not any("without current_cycle" in rec.message for rec in caplog.records)
        finally:
            await s._db.close()


# ---------------------------------------------------------------------------
# run_hygiene resolution
# ---------------------------------------------------------------------------


class TestRunHygieneResolution:
    async def test_provider_supplies_cycle_and_stamps_archive(self) -> None:
        policy = HygienePolicyConfig(enabled=True, eviction_threshold=1.0, check_every_n_cycles=1)
        s = await _make_store(cycle_provider=StaticCycleProvider(1000), hygiene_policy=policy)
        try:
            await s.create_thought(_thought("cold", updated_cycle=0))
            result = await s.run_hygiene()  # no explicit cycle → pulled from provider
            assert result.archived_count == 1
            # The provider's cycle (1000) is what hygiene stamped as archived_at_cycle.
            cursor = await s._db.execute(
                "SELECT archived_at_cycle FROM thought WHERE thought_id = 'cold'"
            )
            row = await cursor.fetchone()
            assert row is not None
            assert int(row["archived_at_cycle"]) == 1000
        finally:
            await s._db.close()

    async def test_explicit_cycle_wins_over_provider(self) -> None:
        policy = HygienePolicyConfig(enabled=True, eviction_threshold=1.0, check_every_n_cycles=1)
        spy = _SpyProvider(1000)
        s = await _make_store(cycle_provider=spy, hygiene_policy=policy)
        try:
            await s.create_thought(_thought("cold", updated_cycle=0))
            await s.run_hygiene(current_cycle=50)
            assert spy.calls == 0  # explicit value wins; provider not pulled
            cursor = await s._db.execute(
                "SELECT archived_at_cycle FROM thought WHERE thought_id = 'cold'"
            )
            row = await cursor.fetchone()
            assert row is not None
            assert int(row["archived_at_cycle"]) == 50
        finally:
            await s._db.close()

    async def test_no_cycle_no_provider_raises(self) -> None:
        policy = HygienePolicyConfig(enabled=True, eviction_threshold=1.0, check_every_n_cycles=1)
        s = await _make_store(hygiene_policy=policy)
        try:
            await s.create_thought(_thought("cold", updated_cycle=0))
            with pytest.raises(ValueError, match="requires a cognitive cycle"):
                await s.run_hygiene()
        finally:
            await s._db.close()

    async def test_disabled_policy_no_cycle_is_noop_not_error(self) -> None:
        # A disabled policy needs no cycle: it must no-op, never raise for a
        # missing cycle (resolution happens after the no-op guard).
        policy = HygienePolicyConfig(enabled=False)
        s = await _make_store(hygiene_policy=policy)
        try:
            result = await s.run_hygiene()
            assert result.archived_count == 0
            assert result.candidates_evaluated == 0
        finally:
            await s._db.close()


# ---------------------------------------------------------------------------
# consolidate resolution
# ---------------------------------------------------------------------------


class TestConsolidateResolution:
    async def test_provider_supplies_cycle(self) -> None:
        policy = HygienePolicyConfig(enabled=True, eviction_threshold=1.0, check_every_n_cycles=1)
        s = await _make_store(
            cycle_provider=StaticCycleProvider(1000), hygiene_policy=policy, dreaming=True
        )
        try:
            await s.create_thought(_thought("cold", updated_cycle=0))
            await s.consolidate()  # no explicit cycle → pulled from provider
            cursor = await s._db.execute(
                "SELECT lifecycle_status FROM thought WHERE thought_id = 'cold'"
            )
            row = await cursor.fetchone()
            assert row is not None
            assert str(row["lifecycle_status"]) == "ARCHIVED"
        finally:
            await s._db.close()

    async def test_no_cycle_no_provider_raises(self) -> None:
        s = await _make_store(dreaming=True)
        try:
            with pytest.raises(ValueError, match="requires a cognitive cycle"):
                await s.consolidate()
        finally:
            await s._db.close()

    async def test_explicit_cycle_wins_over_provider(self) -> None:
        spy = _SpyProvider(1000)
        s = await _make_store(cycle_provider=spy, dreaming=True)
        try:
            await s.consolidate(current_cycle=3)
            assert spy.calls == 0
        finally:
            await s._db.close()


# ---------------------------------------------------------------------------
# MaxCycleProvider — create / staleness / refresh
# ---------------------------------------------------------------------------


class TestMaxCycleProvider:
    async def test_create_snapshots_high_water_mark(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_thought("t1", updated_cycle=4))
        provider = await MaxCycleProvider.create(store)
        assert provider.current_cycle() == 4

    async def test_create_on_empty_store_is_zero(self, store: SqliteEngravaCore) -> None:
        provider = await MaxCycleProvider.create(store)
        assert provider.current_cycle() == 0

    async def test_snapshot_is_stale_until_refresh(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_thought("t1", updated_cycle=4))
        provider = await MaxCycleProvider.create(store)
        # Advance the store past the snapshot.
        await store.create_thought(_thought("t2", updated_cycle=9))
        # current_cycle() still returns the stale snapshot, never a live read.
        assert provider.current_cycle() == 4
        # refresh() re-reads and returns (and caches) the new high-water mark.
        assert await provider.refresh() == 9
        assert provider.current_cycle() == 9

    async def test_satisfies_protocol_and_wires_into_store(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_thought("t1", updated_cycle=2))
        provider = await MaxCycleProvider.create(store)
        assert isinstance(provider, CycleProvider)
        # A store configured with it resolves the snapshot value.
        consumer = await _make_store(cycle_provider=provider)
        try:
            assert consumer._resolve_current_cycle(None) == 2
        finally:
            await consumer._db.close()


# ---------------------------------------------------------------------------
# Golden regressions: NO provider + explicit cycle == the pre-seam path
# ---------------------------------------------------------------------------


class TestGoldenDefaultUnchanged:
    """Exact/golden regressions on the default (no-provider) path.

    With no ``cycle_provider`` configured and an explicit ``current_cycle``, the
    resolution is a pure no-op, so each cycle-consuming path must produce output
    byte-for-byte identical to the pre-seam implementation. These freeze the
    exact fused order + scores (and the exact eligibility outcome) so a future
    change to the resolution/fusion that perturbs the default path fails loudly.
    """

    async def _seed_search_corpus(self, s: SqliteEngravaCore) -> None:
        await s.create_thought(
            _thought("t1", essence="alpha signal", content="alpha signal body one", updated_cycle=2)
        )
        await s.create_thought(
            _thought("t2", essence="alpha signal", content="alpha signal body two", updated_cycle=8)
        )
        await s.create_thought(
            _thought("t3", essence="alpha noise", content="alpha noise body three", updated_cycle=5)
        )

    async def test_golden_search_hybrid(self) -> None:
        s = await _make_store()  # no provider
        try:
            await self._seed_search_corpus(s)
            result = await s.search_hybrid(
                "alpha",
                current_cycle=10,
                recency_weight=0.5,
                recency_half_life=10,
                fts_weight=0.5,
                vector_weight=0.0,
                priority_weight=0.0,
                graph_weight=0.0,
            )
            assert [tid for tid, _ in result.results] == ["t2", "t3", "t1"]
            assert [score for _, score in result.results] == pytest.approx(
                [0.6852752816480621, 0.6035533905932737, 0.5371745887492587], abs=1e-9
            )
            assert result.backends_used == frozenset({"fts5", "recency"})
        finally:
            await s._db.close()

    async def test_golden_recall(self) -> None:
        # recall() carries no per-call weights, so the store config supplies them.
        s = await _make_store(
            search_config=SearchConfig(
                default_recency_weight=0.5,
                default_priority_weight=0.0,
                default_graph_weight=0.0,
                default_vector_weight=0.0,
            )
        )
        try:
            await self._seed_search_corpus(s)
            result = await s.recall("alpha", current_cycle=10)
            assert [tid for tid, _ in result.results] == ["t2", "t3", "t1"]
            assert [score for _, score in result.results] == pytest.approx(
                [0.7954093421326784, 0.7706456197105046, 0.7468906693299828], abs=1e-9
            )
            assert result.backends_used == frozenset({"fts5", "recency"})
        finally:
            await s._db.close()

    async def test_golden_run_hygiene(self) -> None:
        policy = HygienePolicyConfig(enabled=True, eviction_threshold=1.0, check_every_n_cycles=1)
        s = await _make_store(hygiene_policy=policy)  # no provider
        try:
            await s.create_thought(_thought("cold", updated_cycle=0))
            result = await s.run_hygiene(current_cycle=1000)
            assert result.archived_count == 1
            assert result.candidates_evaluated == 1
            assert result.dry_run is False
            cursor = await s._db.execute(
                "SELECT lifecycle_status, archived_at_cycle FROM thought WHERE thought_id = 'cold'"
            )
            row = await cursor.fetchone()
            assert row is not None
            assert str(row["lifecycle_status"]) == "ARCHIVED"
            assert int(row["archived_at_cycle"]) == 1000
        finally:
            await s._db.close()

    async def test_golden_consolidate(self) -> None:
        policy = HygienePolicyConfig(enabled=True, eviction_threshold=1.0, check_every_n_cycles=1)
        s = await _make_store(hygiene_policy=policy, dreaming=True)  # no provider
        try:
            await s.create_thought(_thought("cold", updated_cycle=0))
            result = await s.consolidate(current_cycle=1000)
            assert result.promoted_count == 0
            cursor = await s._db.execute(
                "SELECT lifecycle_status FROM thought WHERE thought_id = 'cold'"
            )
            row = await cursor.fetchone()
            assert row is not None
            assert str(row["lifecycle_status"]) == "ARCHIVED"
        finally:
            await s._db.close()


# ---------------------------------------------------------------------------
# Keyword-only + from_config compat, and provider-not-serialized
# ---------------------------------------------------------------------------


class TestConstructorAndConfigCompat:
    async def test_cycle_provider_is_keyword_only(self) -> None:
        import inspect

        param = inspect.signature(SqliteEngravaCore.__init__).parameters["cycle_provider"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
        assert param.default is None

    async def test_positional_constructor_call_unchanged(self) -> None:
        # The historical positional shape ``SqliteEngravaCore(db[, hooks])`` is
        # unaffected: the second positional still binds ``hooks``, not the new arg.
        conn = await aiosqlite.connect(":memory:")
        conn.row_factory = aiosqlite.Row
        try:
            s = SqliteEngravaCore(conn)
            await s.ensure_schema()
            assert s._cycle_provider is None
        finally:
            await conn.close()

    async def test_from_config_without_provider_is_none(self, tmp_path: Path) -> None:
        cfg = tmp_path / "engrava.yaml"
        cfg.write_text(f"database:\n  path: {tmp_path / 'db.sqlite'}\n", encoding="utf-8")
        async with await SqliteEngravaCore.from_config(cfg) as s:
            assert s._cycle_provider is None

    async def test_from_config_accepts_runtime_provider_and_max_cycle(self, tmp_path: Path) -> None:
        # The OCE-validation ask: both max_cycle() and cycle_provider are
        # reachable on the store returned by from_config.
        cfg = tmp_path / "engrava.yaml"
        cfg.write_text(f"database:\n  path: {tmp_path / 'db.sqlite'}\n", encoding="utf-8")
        provider = StaticCycleProvider(9)
        async with await SqliteEngravaCore.from_config(cfg, cycle_provider=provider) as s:
            await s.create_thought(_thought("t1", updated_cycle=5))
            assert await s.max_cycle() == 5
            assert s._cycle_provider is provider
            assert s._resolve_current_cycle(None) == 9

    def test_provider_is_not_a_serialized_config_field(self) -> None:
        # A provider is a live runtime object, supplied only as a keyword to the
        # constructor / from_config. It is deliberately absent from the serialized
        # config model, so it can be neither parsed from nor written to a config
        # file: the EngravaConfig dataclass declares no cycle_provider field.
        field_names = {f.name for f in dataclasses.fields(EngravaConfig)}
        assert "cycle_provider" not in field_names
