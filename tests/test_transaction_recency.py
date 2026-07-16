"""Tests for the transaction-time recency axis.

Covers the second, separately-typed recency axis: ranking by ``updated_at``
(falling back to ``created_at``) against a caller-supplied ``recency_now``
instant, in wall-clock seconds, alongside (never conflated with) the existing
cognitive-cycle recency.

Verifies: store-time recency favours more-recently-written rows on a standalone
(all-cycle-0) store and on a mixed store; determinism given a fixed ``now``; an
explicit ``recency_now`` takes precedence over a passive ``cycle_provider`` (the
provider is not consulted); transaction recency is off without ``recency_now``;
an **explicit** ``current_cycle`` + ``recency_now`` ⇒ ``RecencyModeConflictError``;
the default (no transaction args) cycle path is byte-for-byte unchanged; a
malformed ``recency_now`` / non-positive half-life ⇒ ``InvalidRecencyArgumentError``;
a missing / malformed row timestamp ⇒ the deterministic minimum score;
same-instant and future-dated tie-break determinism; weight-0 recency is inert
on both the fused and fallback paths; and the recency ranking reads no wall clock
end-to-end (both fused and fallback).
"""

from __future__ import annotations

import datetime
import inspect
import math
from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import (
    ConfigError,
    InvalidRecencyArgumentError,
    LifecycleStatus,
    Priority,
    RecencyModeConflictError,
    SearchConfig,
    SqliteEngravaCore,
    ThoughtRecord,
    ThoughtType,
)
from engrava.config import _parse_search
from engrava.infrastructure.sqlite import engrava_core as _core
from engrava.infrastructure.sqlite.engrava_core import (
    _DEFAULT_RECENCY_NOW_HALF_LIFE_SECONDS,
    _MIN_RECENCY_SCORE,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class _CallCountingProvider:
    """A cycle provider that records how many times it is consulted."""

    def __init__(self, value: int) -> None:
        self._value = value
        self.calls = 0

    def current_cycle(self) -> int:
        self.calls += 1
        return self._value


_REAL_DATETIME = datetime.datetime


class _ClockReadError(AssertionError):
    """Raised by the patched clock to flag a forbidden wall-clock read."""


class _RaisingClock(_REAL_DATETIME):
    """A ``datetime`` whose wall-clock readers raise (everything else works)."""

    @classmethod
    def now(cls, tz: datetime.tzinfo | None = None) -> _REAL_DATETIME:  # noqa: ARG003
        msg = "wall-clock now() read"
        raise _ClockReadError(msg)

    @classmethod
    def utcnow(cls) -> _REAL_DATETIME:
        msg = "wall-clock utcnow() read"
        raise _ClockReadError(msg)


class _CountingClock(_REAL_DATETIME):
    """A ``datetime`` that counts wall-clock reads and returns a fixed instant."""

    calls = 0

    @classmethod
    def now(cls, tz: datetime.tzinfo | None = None) -> _REAL_DATETIME:
        _CountingClock.calls += 1
        return _REAL_DATETIME(2026, 7, 16, 12, 0, 0, tzinfo=tz)

    @classmethod
    def utcnow(cls) -> _REAL_DATETIME:
        _CountingClock.calls += 1
        return _REAL_DATETIME(2026, 7, 16, 12, 0, 0)


# A fixed caller "now" — every transaction-recency assertion is anchored to this
# instant, never the host clock, so the tests are fully replayable.
_NOW = "2026-07-16T12:00:00+00:00"
_NOW_DT = datetime.datetime(2026, 7, 16, 12, 0, 0, tzinfo=datetime.UTC)


async def _make_store(*, search_config: SearchConfig | None = None) -> SqliteEngravaCore:
    """Build a bootstrapped in-memory store (no embedding provider)."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA foreign_keys = ON")
    store = SqliteEngravaCore(conn, search_config=search_config)
    await store.ensure_schema()
    return store


def _thought(
    thought_id: str,
    *,
    essence: str = "alpha topic",
    content: str = "alpha topic content body",
    updated_cycle: int = 0,
    created_at: str | None = None,
    updated_at: str | None = None,
) -> ThoughtRecord:
    """A thought with caller-controlled transaction timestamps.

    ``create_thought`` only auto-stamps ``created_at`` / ``updated_at`` when they
    are ``None``, so passing them explicitly gives the test byte-exact control of
    the transaction-time axis.
    """
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.OBSERVATION,
        essence=essence,
        content=content,
        priority=Priority.P3,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=0,
        updated_cycle=updated_cycle,
        source="test",
        created_at=created_at,
        updated_at=updated_at,
    )


def _expected_transaction_score(ts_iso: str, *, half_life_seconds: float) -> float:
    """Recompute the transaction recency score from first principles.

    ``ts_iso`` is always a UTC-aware ISO-8601 string in these tests.
    """
    ts = datetime.datetime.fromisoformat(ts_iso).astimezone(datetime.UTC)
    age = max((_NOW_DT - ts).total_seconds(), 0.0)
    return math.exp(-math.log(2) * age / half_life_seconds)


@pytest.fixture
async def store() -> AsyncIterator[SqliteEngravaCore]:
    s = await _make_store()
    yield s
    await s._db.close()


# ---------------------------------------------------------------------------
# Store-time recency favours more-recently-written rows
# ---------------------------------------------------------------------------


class TestStoreTimeRecencyRanking:
    async def test_standalone_all_cycle_zero_ranks_recent_first(
        self, store: SqliteEngravaCore
    ) -> None:
        # The MCP case: every row is cycle-0, so cycle recency is degenerate.
        # Transaction recency ranks by write time — most-recently-written first.
        await store.create_thought(_thought("old", updated_at="2026-01-01T00:00:00+00:00"))
        await store.create_thought(_thought("mid", updated_at="2026-06-01T00:00:00+00:00"))
        await store.create_thought(_thought("new", updated_at="2026-07-16T11:59:00+00:00"))

        result = await store.search_hybrid(
            "alpha",
            recency_now=_NOW,
            recency_weight=1.0,
            recency_now_half_life=86400,  # 1-day half-life sharpens the ordering
            priority_weight=0.0,
        )
        ranked_ids = [tid for tid, _ in result.results]
        assert ranked_ids == ["new", "mid", "old"]
        assert "recency" in result.backends_used

    async def test_mixed_store_fresh_cycle_zero_write_not_ranked_oldest(
        self, store: SqliteEngravaCore
    ) -> None:
        # A mixed store: an agent's fresh cycle-0 write (recent updated_at) must
        # NOT rank oldest, even though a high-cycle row exists — the cycle-0
        # killer bug is structurally impossible on the transaction axis.
        await store.create_thought(
            _thought("stale_high_cycle", updated_cycle=100, updated_at="2026-01-01T00:00:00+00:00")
        )
        await store.create_thought(
            _thought("fresh_cycle_zero", updated_cycle=0, updated_at="2026-07-16T11:59:00+00:00")
        )

        result = await store.search_hybrid(
            "alpha",
            recency_now=_NOW,
            recency_weight=1.0,
            recency_now_half_life=86400,
            priority_weight=0.0,
        )
        ranked_ids = [tid for tid, _ in result.results]
        assert ranked_ids[0] == "fresh_cycle_zero"
        assert ranked_ids == ["fresh_cycle_zero", "stale_high_cycle"]

    async def test_updated_at_primary_created_at_fallback(self, store: SqliteEngravaCore) -> None:
        # updated_at is primary; created_at is used only when updated_at is NULL.
        # Row A: recent updated_at, ancient created_at -> scored by updated_at.
        await store.create_thought(
            _thought(
                "a",
                created_at="2020-01-01T00:00:00+00:00",
                updated_at="2026-07-16T11:59:00+00:00",
            )
        )
        # Row B: NULL updated_at, recent created_at -> falls back to created_at.
        await store.create_thought(
            _thought("b", created_at="2026-07-16T11:58:00+00:00", updated_at=None)
        )
        # Row C: NULL updated_at, ancient created_at -> old.
        await store.create_thought(
            _thought("c", created_at="2020-01-01T00:00:00+00:00", updated_at=None)
        )
        # Force B's updated_at back to NULL (create_thought auto-stamps a NULL
        # updated_at at write time, so re-null it to exercise the fallback).
        await store._db.execute(
            "UPDATE thought SET updated_at = NULL WHERE thought_id IN ('b','c')"
        )

        scores = await store._load_transaction_recency_scores(
            thought_ids={"a", "b", "c"},
            now=_NOW_DT,
            half_life_seconds=86400.0,
        )
        # a (updated_at 1 min old) and b (created_at 2 min old) are both fresh;
        # c (created_at 6 years old) is ~0.
        assert scores["a"] > 0.99
        assert scores["b"] > 0.99
        assert scores["a"] > scores["b"]  # updated_at (1 min) fresher than created_at (2 min)
        assert scores["c"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Determinism given a fixed now (no clock read)
# ---------------------------------------------------------------------------


class TestDeterminismAndNoClockRead:
    async def test_fixed_now_is_deterministic(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_thought("a", updated_at="2026-07-16T11:00:00+00:00"))
        await store.create_thought(_thought("b", updated_at="2026-07-16T10:00:00+00:00"))
        first = await store.search_hybrid("alpha", recency_now=_NOW, recency_weight=1.0)
        second = await store.search_hybrid("alpha", recency_now=_NOW, recency_weight=1.0)
        assert first.results == second.results

    async def test_score_derives_from_recency_now_not_host_clock(
        self, store: SqliteEngravaCore
    ) -> None:
        # The row is ~1 minute old *relative to recency_now* (which is in the
        # test's fixed past). If the core secretly read the host clock, the age
        # would be years and the score ~0. Asserting the exact recency_now-based
        # decay proves the ranking uses the caller instant, never a host clock.
        await store.create_thought(_thought("a", updated_at="2026-07-16T11:59:00+00:00"))
        scores = await store._load_transaction_recency_scores(
            thought_ids={"a"},
            now=_NOW_DT,
            half_life_seconds=604800.0,
        )
        assert scores["a"] == pytest.approx(
            _expected_transaction_score("2026-07-16T11:59:00+00:00", half_life_seconds=604800.0)
        )

    async def test_no_recency_now_leaves_transaction_axis_off(
        self, store: SqliteEngravaCore
    ) -> None:
        # No cycle reference and no recency_now => recency simply off (no clock
        # invented). A recency_now_half_life alone does not activate the axis.
        await store.create_thought(_thought("a", updated_at="2026-07-16T11:59:00+00:00"))
        result = await store.search_hybrid("alpha", recency_weight=1.0, recency_now_half_life=86400)
        assert "recency" not in result.backends_used

    def test_transaction_scorer_reads_no_clock(self) -> None:
        # Source-level guard: the transaction recency scorer must derive age from
        # the caller's ``now`` only — never a host-clock read.
        src = inspect.getsource(SqliteEngravaCore._load_transaction_recency_scores)
        assert "datetime.now" not in src
        assert "utcnow" not in src
        assert ".now(" not in src


# ---------------------------------------------------------------------------
# Mutual exclusion — both references raise
# ---------------------------------------------------------------------------


class TestRecencyModeConflict:
    async def test_explicit_cycle_and_recency_now_conflict(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_thought("a", updated_at=_NOW))
        with pytest.raises(RecencyModeConflictError):
            await store.search_hybrid("alpha", current_cycle=5, recency_now=_NOW)

    async def test_explicit_recency_now_overrides_passive_provider(self) -> None:
        # Precedence: an explicit recency_now selects transaction recency and a
        # configured (passive) cycle_provider is NOT consulted — no conflict.
        spy = _CallCountingProvider(3)
        s = await _make_store()
        s._cycle_provider = spy
        try:
            await s.create_thought(_thought("old", updated_at="2026-01-01T00:00:00+00:00"))
            await s.create_thought(_thought("new", updated_at="2026-07-16T11:59:00+00:00"))
            result = await s.search_hybrid(
                "alpha",
                recency_now=_NOW,
                recency_weight=1.0,
                recency_now_half_life=86400,
                priority_weight=0.0,
            )
            # transaction recency ranks recent-first, and the provider was never
            # consulted (the conflict check + resolution are on the raw args).
            assert [tid for tid, _ in result.results] == ["new", "old"]
            assert "recency" in result.backends_used
            assert spy.calls == 0
        finally:
            await s._db.close()

    async def test_provider_supplies_cycle_when_no_explicit_recency_ref(self) -> None:
        # Control: with NO recency_now, the provider still supplies the cycle
        # (cycle recency) — the passive provider is only bypassed by recency_now.
        spy = _CallCountingProvider(5)
        s = await _make_store(search_config=SearchConfig(default_recency_weight=1.0))
        s._cycle_provider = spy
        try:
            await s.create_thought(_thought("a", updated_cycle=5))
            result = await s.search_hybrid("alpha", priority_weight=0.0)
            assert "recency" in result.backends_used
            assert spy.calls >= 1
        finally:
            await s._db.close()

    async def test_conflict_raised_even_with_zero_weight(self, store: SqliteEngravaCore) -> None:
        # Supplying both references is an API misuse rejected regardless of the
        # weight — the axes are never silently combined.
        await store.create_thought(_thought("a", updated_at=_NOW))
        with pytest.raises(RecencyModeConflictError):
            await store.search_hybrid(
                "alpha", current_cycle=5, recency_now=_NOW, recency_weight=0.0
            )

    async def test_recall_conflict(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_thought("a", updated_at=_NOW))
        with pytest.raises(RecencyModeConflictError):
            await store.recall("alpha", current_cycle=5, recency_now=_NOW)


# ---------------------------------------------------------------------------
# Default-unchanged golden — COMPLETE result equality
# ---------------------------------------------------------------------------


class TestDefaultUnchangedGolden:
    #: Golden captured on the pre-change base (release/v0.6.0) for the cycle-
    #: recency path with NO transaction args, and re-verified byte-for-byte
    #: identical on this branch. The store is embedding-free with priority and
    #: graph off, so the fused path is FTS (all tied -> 0.5) + cycle recency:
    #: eff_fts = 0.3/1.3, eff_rec = 1.0/1.3; cycle ages are (5-uc). Locks the
    #: existing cycle path against any perturbation from the transaction axis.
    _GOLDEN: tuple[tuple[str, float], ...] = (
        ("t_new", 0.8846153846153846),
        ("t_mid", 0.6983525255809224),
        ("t_old", 0.49999999999999994),
    )

    async def _seed(self, s: SqliteEngravaCore) -> None:
        await s.create_thought(_thought("t_old", updated_cycle=0))
        await s.create_thought(_thought("t_mid", updated_cycle=3))
        await s.create_thought(_thought("t_new", updated_cycle=5))

    async def test_cycle_path_matches_golden_completely(self, store: SqliteEngravaCore) -> None:
        await self._seed(store)
        result = await store.search_hybrid(
            "alpha",
            current_cycle=5,
            recency_weight=1.0,
            recency_half_life=5,
            priority_weight=0.0,
        )
        # COMPLETE equality: exact fused score list, exact backend set, and the
        # reflection-eviction count — not selected fields, not an approximation.
        assert result.results == list(self._GOLDEN)
        assert result.backends_used == frozenset({"fts5", "recency"})
        assert result.reflections_evicted == 0

    async def test_transaction_defaults_are_inert(self, store: SqliteEngravaCore) -> None:
        # Passing the new params at their defaults (None) must be byte-for-byte
        # identical to omitting them entirely — the additive params are inert.
        await self._seed(store)
        without = await store.search_hybrid(
            "alpha", current_cycle=5, recency_weight=1.0, recency_half_life=5, priority_weight=0.0
        )
        with_defaults = await store.search_hybrid(
            "alpha",
            current_cycle=5,
            recency_weight=1.0,
            recency_half_life=5,
            priority_weight=0.0,
            recency_now=None,
            recency_now_half_life=None,
        )
        assert without.results == with_defaults.results
        assert without.backends_used == with_defaults.backends_used
        assert without.results == list(self._GOLDEN)


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


class TestFailurePaths:
    @pytest.mark.parametrize(
        "bad_now",
        ["not-a-date", "2026-13-01T00:00:00", "", "16/07/2026"],
    )
    async def test_malformed_recency_now_raises_typed_error(
        self, store: SqliteEngravaCore, bad_now: str
    ) -> None:
        await store.create_thought(_thought("a", updated_at=_NOW))
        with pytest.raises(
            InvalidRecencyArgumentError, match="recency_now must be an ISO-8601 timestamp"
        ) as exc_info:
            await store.search_hybrid("alpha", recency_now=bad_now, recency_weight=1.0)
        # The exact typed class (not a bare ValueError) and the chained cause.
        assert type(exc_info.value) is InvalidRecencyArgumentError
        assert not isinstance(exc_info.value, ValueError)
        assert exc_info.value.__cause__ is not None

    async def test_naive_recency_now_is_accepted_as_utc(self, store: SqliteEngravaCore) -> None:
        # A naive value is not malformed: it is interpreted as UTC (host tz never
        # consulted), matching the shared temporal helper.
        await store.create_thought(_thought("a", updated_at="2026-07-16T11:59:00+00:00"))
        result = await store.search_hybrid(
            "alpha", recency_now="2026-07-16T12:00:00", recency_weight=1.0
        )
        assert "recency" in result.backends_used
        assert [tid for tid, _ in result.results] == ["a"]

    async def test_missing_row_timestamp_scores_minimum(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_thought("live", updated_at="2026-07-16T11:59:00+00:00"))
        await store.create_thought(_thought("ghost", updated_at=_NOW))
        # Null out both transaction columns for the ghost row (legacy/imported).
        await store._db.execute(
            "UPDATE thought SET updated_at = NULL, created_at = NULL WHERE thought_id = 'ghost'"
        )
        scores = await store._load_transaction_recency_scores(
            thought_ids={"live", "ghost"}, now=_NOW_DT, half_life_seconds=604800.0
        )
        assert scores["ghost"] == _MIN_RECENCY_SCORE
        assert scores["live"] > 0.99

    async def test_malformed_row_timestamp_scores_minimum(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_thought("garbled", updated_at=_NOW))
        # Write a malformed timestamp directly (bypassing model validation) to
        # exercise the legacy/imported bad-data path.
        await store._db.execute(
            "UPDATE thought SET updated_at = 'not-a-timestamp', created_at = 'also-bad' "
            "WHERE thought_id = 'garbled'"
        )
        scores = await store._load_transaction_recency_scores(
            thought_ids={"garbled"}, now=_NOW_DT, half_life_seconds=604800.0
        )
        assert scores["garbled"] == _MIN_RECENCY_SCORE

    @pytest.mark.parametrize("bad_half_life", [0, -1, -604800])
    async def test_non_positive_half_life_rejected(
        self, store: SqliteEngravaCore, bad_half_life: int
    ) -> None:
        await store.create_thought(_thought("a", updated_at=_NOW))
        with pytest.raises(
            InvalidRecencyArgumentError, match="recency_now_half_life must be a positive"
        ) as exc_info:
            await store.search_hybrid(
                "alpha", recency_now=_NOW, recency_weight=1.0, recency_now_half_life=bad_half_life
            )
        assert type(exc_info.value) is InvalidRecencyArgumentError

    async def test_half_life_without_reference_is_ignored(self, store: SqliteEngravaCore) -> None:
        # A half-life without a recency reference is inert — no error, recency off.
        await store.create_thought(_thought("a", updated_at=_NOW))
        result = await store.search_hybrid("alpha", recency_now_half_life=-5)
        assert "recency" not in result.backends_used

    async def test_zero_weight_reference_inert_no_error(self, store: SqliteEngravaCore) -> None:
        # A transaction reference with weight 0 is accepted but inert (the weight
        # gates recency) — no conflict, no error, recency simply off.
        await store.create_thought(_thought("a", updated_at=_NOW))
        result = await store.search_hybrid("alpha", recency_now=_NOW, recency_weight=0.0)
        assert "recency" not in result.backends_used

    async def test_zero_weight_transaction_inert_on_fallback_path(
        self, store: SqliteEngravaCore
    ) -> None:
        # DEFECT regression: weight-0 transaction recency must be inert on the
        # query-less FALLBACK path too (no FTS, no vector) — byte-identical to a
        # fallback with no recency reference, never ordered/scored by transaction
        # time. ``old`` has the higher updated_cycle, so the neutral fallback
        # orders it first; transaction time would order ``new`` first.
        await store.create_thought(
            _thought("old", updated_cycle=2, updated_at="2026-01-01T00:00:00+00:00")
        )
        await store.create_thought(
            _thought("new", updated_cycle=1, updated_at="2026-07-16T11:59:00+00:00")
        )
        # priority_weight=0.0 removes the default priority boost so the neutral
        # fallback scores are flat 0.0 (the pure no-recency baseline).
        baseline = await store.search_hybrid("", priority_weight=0.0)
        weight0 = await store.search_hybrid(
            "", recency_now=_NOW, recency_weight=0.0, priority_weight=0.0
        )
        # COMPLETE result-object equality with the neutral (no-recency) fallback —
        # every field, including ``reflections_evicted`` (HybridSearchResult is a
        # frozen dataclass, so ``==`` compares all fields).
        assert weight0 == baseline
        assert "recency" not in weight0.backends_used
        assert [tid for tid, _ in weight0.results] == ["old", "new"]  # updated_cycle DESC
        assert all(score == 0.0 for _, score in weight0.results)  # flat, not txn-scored
        # With weight > 0 the fallback DOES order by transaction time (new first),
        # confirming weight-0 genuinely suppressed the axis (not a coincidence).
        active = await store.search_hybrid(
            "",
            recency_now=_NOW,
            recency_weight=1.0,
            recency_now_half_life=86400,
            priority_weight=0.0,
        )
        assert [tid for tid, _ in active.results] == ["new", "old"]


# ---------------------------------------------------------------------------
# Tie-break determinism (same-instant + future-dated)
# ---------------------------------------------------------------------------


class TestTieBreakDeterminism:
    async def test_same_instant_ties_break_by_thought_id(self, store: SqliteEngravaCore) -> None:
        # Identical timestamps => identical recency scores => the fused result's
        # deterministic secondary order (thought_id ascending) decides.
        shared = "2026-07-16T11:00:00+00:00"
        for tid in ("t_c", "t_a", "t_b"):
            await store.create_thought(_thought(tid, updated_at=shared))
        result = await store.search_hybrid(
            "alpha", recency_now=_NOW, recency_weight=1.0, priority_weight=0.0
        )
        scores = dict(result.results)
        assert len(set(scores.values())) == 1  # all equal
        assert [tid for tid, _ in result.results] == ["t_a", "t_b", "t_c"]

    async def test_future_dated_rows_clamp_to_age_zero_and_tie(
        self, store: SqliteEngravaCore
    ) -> None:
        # Future-dated rows (updated_at after recency_now) clamp to age 0 -> the
        # maximum score 1.0; two future rows tie and break by thought_id.
        await store.create_thought(_thought("f_b", updated_at="2027-01-01T00:00:00+00:00"))
        await store.create_thought(_thought("f_a", updated_at="2030-01-01T00:00:00+00:00"))
        scores = await store._load_transaction_recency_scores(
            thought_ids={"f_a", "f_b"}, now=_NOW_DT, half_life_seconds=604800.0
        )
        assert scores["f_a"] == pytest.approx(1.0)
        assert scores["f_b"] == pytest.approx(1.0)
        result = await store.search_hybrid(
            "alpha", recency_now=_NOW, recency_weight=1.0, priority_weight=0.0
        )
        assert [tid for tid, _ in result.results] == ["f_a", "f_b"]


# ---------------------------------------------------------------------------
# Fallback path (no FTS, no vector) still ranks by transaction time
# ---------------------------------------------------------------------------


class TestFallbackPath:
    async def test_fallback_ranks_by_transaction_time(self, store: SqliteEngravaCore) -> None:
        # An empty query text disables FTS; with no embeddings the vector arm is
        # off too, so the query-less fallback path runs — and must still rank by
        # transaction recency.
        await store.create_thought(_thought("old", updated_at="2026-01-01T00:00:00+00:00"))
        await store.create_thought(_thought("new", updated_at="2026-07-16T11:59:00+00:00"))
        result = await store.search_hybrid(
            "", recency_now=_NOW, recency_weight=1.0, recency_now_half_life=86400
        )
        assert "recency" in result.backends_used
        assert [tid for tid, _ in result.results] == ["new", "old"]


# ---------------------------------------------------------------------------
# End-to-end no-clock guard — recency ranking reads no wall clock
# ---------------------------------------------------------------------------


class TestNoClockRead:
    async def test_recency_scorers_complete_when_clock_raises(
        self, store: SqliteEngravaCore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Patch the wall-clock readers (now / utcnow) to RAISE, then run BOTH
        # recency scorers directly. Each derives age from caller-supplied inputs
        # only, so it completes without touching the clock (a clock read would
        # raise). Thoughts are created before the patch (writes do stamp a clock).
        await store.create_thought(
            _thought("a", updated_cycle=3, updated_at="2026-07-16T11:59:00+00:00")
        )
        monkeypatch.setattr(_core.datetime, "datetime", _RaisingClock)
        # Sanity: the patch really intercepts BOTH wall-clock readers, so the
        # scorers succeeding below genuinely proves clock-freeness (not a no-op).
        with pytest.raises(_ClockReadError):
            _core.datetime.datetime.now(datetime.UTC)
        with pytest.raises(_ClockReadError):
            _core.datetime.datetime.utcnow()
        txn = await store._load_transaction_recency_scores(
            thought_ids={"a"}, now=_NOW_DT, half_life_seconds=604800.0
        )
        assert txn["a"] > 0.99
        cyc = await store._load_recency_scores(
            thought_ids={"a"}, current_cycle=5, recency_half_life=5
        )
        assert cyc["a"] > 0.0

    async def test_recency_adds_no_clock_read_end_to_end(
        self, store: SqliteEngravaCore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # End-to-end guard. The search path's ONLY wall-clock reads are the
        # pre-existing expiry-liveness filter, never recency. Adding a
        # recency reference (transaction OR cycle) to a fused OR fallback query
        # makes the SAME number of clock reads as the no-recency baseline — so
        # recency itself reads no clock. A fixed instant is returned so the
        # liveness filter still functions; the non-expiring rows are unaffected.
        await store.create_thought(
            _thought("old", updated_cycle=2, updated_at="2026-01-01T00:00:00+00:00")
        )
        await store.create_thought(
            _thought("new", updated_cycle=5, updated_at="2026-07-16T11:59:00+00:00")
        )
        monkeypatch.setattr(_core.datetime, "datetime", _CountingClock)
        # Sanity: the counting patch intercepts both wall-clock readers.
        _CountingClock.calls = 0
        _core.datetime.datetime.now(datetime.UTC)
        _core.datetime.datetime.utcnow()
        assert _CountingClock.calls == 2

        # --- fused path (FTS active) ---
        _CountingClock.calls = 0
        await store.search_hybrid("alpha", priority_weight=0.0)
        base = _CountingClock.calls
        assert base > 0  # the liveness filter does read the clock (baseline)

        _CountingClock.calls = 0
        txn = await store.search_hybrid(
            "alpha",
            recency_now=_NOW,
            recency_weight=1.0,
            recency_now_half_life=86400,
            priority_weight=0.0,
        )
        assert _CountingClock.calls == base  # transaction recency read no clock
        assert [tid for tid, _ in txn.results] == ["new", "old"]  # it actually ran

        _CountingClock.calls = 0
        await store.search_hybrid("alpha", current_cycle=5, recency_weight=1.0, priority_weight=0.0)
        assert _CountingClock.calls == base  # cycle recency read no clock

        # --- fallback path (no FTS, no vector) ---
        _CountingClock.calls = 0
        await store.search_hybrid("")
        base_fb = _CountingClock.calls

        _CountingClock.calls = 0
        txn_fb = await store.search_hybrid(
            "", recency_now=_NOW, recency_weight=1.0, recency_now_half_life=86400
        )
        assert _CountingClock.calls == base_fb  # no clock read in the fallback recency
        assert [tid for tid, _ in txn_fb.results] == ["new", "old"]  # it actually ran


# ---------------------------------------------------------------------------
# Config default for the transaction half-life
# ---------------------------------------------------------------------------


class TestHalfLifeConfig:
    def test_default_config_half_life_is_seven_days(self) -> None:
        assert SearchConfig().recency_now_half_life_seconds == 604800
        assert _DEFAULT_RECENCY_NOW_HALF_LIFE_SECONDS == 604800

    def test_config_parse_default(self) -> None:
        assert _parse_search({}).recency_now_half_life_seconds == 604800

    def test_config_parse_override(self) -> None:
        assert (
            _parse_search({"recency_now_half_life_seconds": 3600}).recency_now_half_life_seconds
            == 3600
        )

    @pytest.mark.parametrize("bad", [0, -1, 3.5, "week", True])
    def test_config_parse_rejects_non_positive_int(self, bad: object) -> None:
        with pytest.raises(ConfigError, match="recency_now_half_life_seconds"):
            _parse_search({"recency_now_half_life_seconds": bad})

    async def test_config_half_life_drives_ranking_when_not_overridden(self) -> None:
        # Integration: a store-configured ``recency_now_half_life_seconds`` is
        # consulted by ``search_hybrid`` when the call omits
        # ``recency_now_half_life``, and it demonstrably drives the fused score.
        # The store's half-life is 86400 s (1 day); the single candidate is
        # exactly 1 day old, so its recency score is 0.5 and — with the lone fts
        # candidate normalising to 0.5 and only fts+recency weighted — the fused
        # score is exactly 0.5. Under the default 604800 s half-life the same row
        # would score markedly fresher, so a differing score proves the
        # CONFIGURED value (not the default) drove the ranking.
        s = await _make_store(search_config=SearchConfig(recency_now_half_life_seconds=86400))
        try:
            await s.create_thought(_thought("a", updated_at="2026-07-15T12:00:00+00:00"))
            configured = await s.search_hybrid(
                "alpha", recency_now=_NOW, recency_weight=1.0, priority_weight=0.0
            )
            overridden = await s.search_hybrid(
                "alpha",
                recency_now=_NOW,
                recency_weight=1.0,
                priority_weight=0.0,
                recency_now_half_life=604800,  # explicit default, for contrast
            )
            assert configured.results[0][1] == pytest.approx(0.5)  # age == config half-life
            # The default 7-day half-life yields a distinctly fresher fused score,
            # confirming the configured 86400 s (not the default) drove the first.
            assert overridden.results[0][1] > 0.75
            assert configured.results[0][1] != pytest.approx(overridden.results[0][1])
        finally:
            await s._db.close()
