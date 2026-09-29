"""Dreaming activation correctness — reachable scoring + live access substrate.

Covers active-signal weight redistribution (promotion is arithmetically
reachable under defaults), the batched access substrate
(feeds the ``frequency`` signal without per-read writes), config wiring
(``from_config`` builds + runs dreaming, partial ``signals`` merges), and the
default-off byte-identity guarantee.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import SqliteEngravaCore
from engrava.config import DreamingConfig, DreamingGates, _parse_dreaming
from engrava.domain.enums import LifecycleStatus, Priority, ThoughtType
from engrava.domain.exceptions import WriteLockTimeoutError
from engrava.domain.models.thought import ThoughtRecord
from engrava.extensions.dreaming import DreamingExtension
from engrava.infrastructure.sqlite.engrava_core import _AccessBuffer

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
    from pathlib import Path

# Cycle at which recency == staleness == 1.0: a thought created at cycle 0 and
# last updated at cycle 100 has age 0 (recency 1.0) and span 100 (staleness 1.0)
# when scored at cycle 100.
_CYCLE = 100


def _obs(
    thought_id: str,
    *,
    confirmation_count: int = 0,
    confidence: float | None = None,
    access_count: int = 0,
    created_cycle: int = 0,
    updated_cycle: int = _CYCLE,
    action_outcome_score: float | None = None,
) -> ThoughtRecord:
    """A recent + mature ACTIVE OBSERVATION (recency = staleness = 1.0 at _CYCLE)."""
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.OBSERVATION,
        essence=f"Essence {thought_id}",
        content=f"Content of {thought_id} about apples and oranges and pears.",
        priority=Priority.P3,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=created_cycle,
        updated_cycle=updated_cycle,
        source="test",
        confirmation_count=confirmation_count,
        confidence=confidence,
        access_count=access_count,
        action_outcome_score=action_outcome_score,
    )


def _activation_cfg(
    *,
    promote_threshold: float = 0.7,
    access_tracking_enabled: bool = True,
    max_p1_fraction: float = 0.5,
) -> DreamingConfig:
    """Dreaming config with the real 0.7 threshold and open promotion gates.

    The promotion *gates* (age / confirmation) are opened so the test isolates
    the *score* reachability; the score threshold stays at its shipped 0.7.
    """
    return DreamingConfig(
        enabled=True,
        promote_threshold=promote_threshold,
        max_p1_fraction=max_p1_fraction,
        access_tracking_enabled=access_tracking_enabled,
        gates=DreamingGates(
            min_age_cycles=0,
            allow_zero_confirmation=True,
            enable_reflections=False,  # isolate promotion from clustering
        ),
    )


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[SqliteEngravaCore]:
    db = await aiosqlite.connect(str(tmp_path / "activation.db"))
    db.row_factory = aiosqlite.Row
    s = SqliteEngravaCore(db=db)
    await s.ensure_schema()
    yield s
    await db.close()


# ---------------------------------------------------------------------------
# _compute_active_weights — redistribution unit (the reachability mechanism)
# ---------------------------------------------------------------------------


class TestActiveWeightRedistribution:
    """The score is a weighted average over the signals active for the run."""

    def test_all_default_signals_flat_redistribute_to_recency_staleness(self) -> None:
        ext = DreamingExtension(config=_activation_cfg(access_tracking_enabled=False))
        candidates = [_obs("a"), _obs("b")]  # unconfirmed, no confidence, no access
        weights, flat = ext._compute_active_weights(candidates, current_cycle=_CYCLE)

        # confirmation / confidence / frequency / action_outcome carry no
        # data → flat (no candidate has an action outcome either).
        assert set(flat) == {"confirmation", "confidence", "frequency", "action_outcome"}
        # Their weight redistributes onto recency (.25) + staleness (.20).
        assert weights["recency"] == pytest.approx(0.25 / 0.45)
        assert weights["staleness"] == pytest.approx(0.20 / 0.45)
        assert weights["confirmation"] == 0.0
        assert weights["frequency"] == 0.0
        assert weights["action_outcome"] == 0.0
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_degenerate_no_active_signal_yields_zero_weights(self) -> None:
        ext = DreamingExtension(config=_activation_cfg(access_tracking_enabled=False))
        # current_cycle None → recency + staleness inactive; nothing else has data.
        weights, flat = ext._compute_active_weights([_obs("a")], current_cycle=None)  # type: ignore[arg-type]
        assert all(w == 0.0 for w in weights.values())
        assert "recency" in flat
        assert "staleness" in flat

    def test_confirmation_becomes_active_when_a_candidate_is_confirmed(self) -> None:
        """Candidate-set dependence (intended): a confirmed peer activates the signal."""
        ext = DreamingExtension(config=_activation_cfg(access_tracking_enabled=False))
        weights, flat = ext._compute_active_weights(
            [_obs("a"), _obs("b", confirmation_count=5)],
            current_cycle=_CYCLE,
        )
        assert "confirmation" not in flat  # now active (one candidate has data)
        assert weights["confirmation"] == pytest.approx(0.20 / 0.65)


# ---------------------------------------------------------------------------
# Reachability — promotion is possible under the default threshold
# ---------------------------------------------------------------------------


class TestPromotionReachable:
    async def test_recent_mature_observations_promote_under_default_threshold(
        self, store: SqliteEngravaCore
    ) -> None:
        """A recent + mature OBSERVATION promotes at the shipped 0.7 threshold.

        With frequency and confirmation flat (no data in the pool), the
        active-signal weights are redistributed onto the remaining signals, so
        recency + staleness alone clear the 0.7 threshold.
        """
        ext = DreamingExtension(config=_activation_cfg())
        for i in range(4):
            await store.create_thought(_obs(f"obs-{i}"))

        result = await ext.run_consolidation(store, current_cycle=_CYCLE)

        assert result.promoted_count >= 1
        # Redistribution recorded on the result; the dead signals are flat.
        assert result.active_signal_weights["recency"] > 0.0
        assert "frequency" in result.flat_signals

    async def test_candidate_set_dependence_end_to_end(self, store: SqliteEngravaCore) -> None:
        """A confirmed peer can demote an unconfirmed thought below the gate.

        With confirmation flat, an unconfirmed recent thought scores ~1.0 and
        promotes. Add a confirmed peer → confirmation activates → the same
        unconfirmed thought scores ~0.692 < 0.70 and no longer promotes. This
        pins the intended pool-relative behaviour.
        """
        ext = DreamingExtension(config=_activation_cfg())
        await store.create_thought(_obs("solo"))
        alone = await ext.run_consolidation(store, current_cycle=_CYCLE)
        assert "solo" in alone.promoted_ids

        # Fresh store with the same unconfirmed thought + a confirmed peer.
        db2 = await aiosqlite.connect(":memory:")
        db2.row_factory = aiosqlite.Row
        store2 = SqliteEngravaCore(db=db2)
        await store2.ensure_schema()
        await store2.create_thought(_obs("unconf"))
        await store2.create_thought(_obs("conf", confirmation_count=5))
        mixed = await ext.run_consolidation(store2, current_cycle=_CYCLE)
        assert "unconf" not in mixed.promoted_ids  # confirmation now discriminates
        assert "confirmation" not in mixed.flat_signals
        await db2.close()

    async def test_action_outcome_signal_activates_end_to_end(
        self, store: SqliteEngravaCore
    ) -> None:
        """A candidate carrying an action outcome activates the 6th signal.

        With no action outcomes in the pool the signal is flat (zero weight).
        Introduce one candidate whose action_outcome_score is set and the signal
        must activate end-to-end through run_consolidation — pinning
        default_signal_active's action_outcome branch, which is otherwise
        revert-safe to always-inactive (unit-flat coverage alone misses it).
        """
        ext = DreamingExtension(config=_activation_cfg())
        await store.create_thought(_obs("no-outcome"))
        solo = await ext.run_consolidation(store, current_cycle=_CYCLE)
        assert "action_outcome" in solo.flat_signals  # flat: nobody has an outcome
        assert solo.active_signal_weights["action_outcome"] == 0.0

        db2 = await aiosqlite.connect(":memory:")
        db2.row_factory = aiosqlite.Row
        store2 = SqliteEngravaCore(db=db2)
        await store2.ensure_schema()
        await store2.create_thought(_obs("plain"))
        await store2.create_thought(_obs("has-outcome", action_outcome_score=0.9))
        mixed = await ext.run_consolidation(store2, current_cycle=_CYCLE)
        assert "action_outcome" not in mixed.flat_signals  # activated by the peer
        assert mixed.active_signal_weights["action_outcome"] > 0.0
        await db2.close()

    async def test_population_p1_cap_bounds_repeated_cycles(self, store: SqliteEngravaCore) -> None:
        """Repeated cycles do not push the P1 population past max_p1_fraction."""
        ext = DreamingExtension(config=_activation_cfg(max_p1_fraction=0.25))
        for i in range(8):
            await store.create_thought(_obs(f"obs-{i}"))
        await ext.run_consolidation(store, current_cycle=_CYCLE)
        await ext.run_consolidation(store, current_cycle=_CYCLE + 1)
        await ext.run_consolidation(store, current_cycle=_CYCLE + 2)
        p1 = await store.count_thoughts(priority="P1")
        total = await store.count_thoughts()
        assert p1 <= max(1, int(total * 0.25))


# ---------------------------------------------------------------------------
# Access substrate — batched, no per-read write, feeds frequency
# ---------------------------------------------------------------------------


class TestAccessSubstrate:
    async def test_retrieval_buffers_then_flush_increments_no_hot_write(
        self, store: SqliteEngravaCore
    ) -> None:
        store._access_tracking_enabled = True
        await store.create_thought(_obs("t"))

        await store.get_thought("t")
        await store.get_thought("t")
        # No DB write on the read path yet — access_count still 0.
        row = await store.get_thought("t")  # a third buffered access
        assert row is not None
        assert row.access_count == 0
        assert len(store._access_buffer) == 1

        updated = await store.flush_access_buffer()
        assert updated == 1
        after = await store.get_thought("t")  # buffers again, but DB already flushed
        assert after is not None
        assert after.access_count == 3

    async def test_default_off_no_access_tracking_byte_identical(
        self, store: SqliteEngravaCore
    ) -> None:
        """Default store (tracking off) never buffers or writes access counts."""
        assert store._access_tracking_enabled is False
        await store.create_thought(_obs("t"))
        await store.get_thought("t")
        assert len(store._access_buffer) == 0
        assert await store.flush_access_buffer() == 0
        row = await store.get_thought("t")
        assert row is not None
        assert row.access_count == 0

    async def test_flush_of_all_stale_entries_does_not_commit_the_callers_pending_edit(
        self, store: SqliteEngravaCore
    ) -> None:
        """A flush whose whole batch is stale writes nothing and commits nothing.

        Mirrors the ``delete_thought`` case pinned in
        ``tests/test_referential_integrity.py``: every buffered id's thought
        was deleted before the flush runs, so the batched ``UPDATE`` matches
        zero rows across the board — this call has nothing of its own to make
        durable, and must not commit a caller's own pending transaction.
        """
        store._access_tracking_enabled = True
        await store.create_thought(_obs("stale"))
        await store.create_thought(_obs("unrelated"))
        await store.get_thought("stale")  # buffers one access
        assert len(store._access_buffer) == 1

        await store.delete_thought("stale")  # the buffered id no longer exists

        await store._db.execute("BEGIN")
        await store._db.execute(
            "UPDATE thought SET essence = ? WHERE thought_id = ?",
            ("edited-by-caller", "unrelated"),
        )

        flushed = await store.flush_access_buffer()

        assert flushed == 1  # the stale entry is still drained from the buffer
        assert store._db.in_transaction is True, (
            "the caller's own transaction, with their pending edit still "
            "inside it, must still be open after an all-stale flush"
        )
        await store._db.rollback()

        row = await store.get_thought("unrelated")
        assert row is not None
        assert row.essence == "Essence unrelated", (
            "the caller's rollback must undo their own edit -- an all-stale "
            "flush_access_buffer() must not have committed it on their behalf"
        )

    async def test_ordinary_flush_still_commits(self, store: SqliteEngravaCore) -> None:
        """Control: a flush with at least one live match commits as before."""
        store._access_tracking_enabled = True
        await store.create_thought(_obs("t"))
        await store.get_thought("t")

        flushed = await store.flush_access_buffer()

        assert flushed == 1
        assert store._db.in_transaction is False, "a real access-count write must still commit"
        after = await store.get_thought("t")
        assert after is not None
        assert after.access_count == 1

    async def test_access_flush_is_not_journaled(self) -> None:
        """The batched access flush writes no journal entry and keeps the chain valid.

        Access counts are high-volume regenerable telemetry, deliberately kept
        out of the hash chain. A flush must neither append a journal entry nor
        break verification — a future edit routing the flush through the
        journaled path would otherwise pass silently.
        """
        db = await aiosqlite.connect(":memory:")
        db.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(db=db, journal_enabled=True)
        await store.ensure_schema()
        store._access_tracking_enabled = True
        await store.create_thought(_obs("t"))

        async def _jlen() -> int:
            cur = await store._db.execute("SELECT COUNT(*) FROM journal_entry")
            row = await cur.fetchone()
            assert row is not None
            return int(row[0])

        for _ in range(3):
            await store.get_thought("t")
        before = await _jlen()

        flushed = await store.flush_access_buffer()

        assert flushed == 1  # one distinct id drained
        assert await _jlen() == before  # the batched UPDATE is NOT journaled
        result = await store.verify_journal()
        assert result.valid is True
        await db.close()


class TestFlushOnAContendedWriteLock:
    """A flush that cannot take the write lock does not drain the buffer.

    The flush needs the store's write lock to apply the buffered events, and
    taking that lock is bounded (``WriteLockTimeoutError``). Losing events
    because the database write failed is the documented best-effort behaviour
    of this telemetry; losing them because the lock could not be *taken* is
    not a database failure, so a flush that times out on the lock must not
    drain the buffer.
    """

    @staticmethod
    @contextlib.asynccontextmanager
    async def _tracking_store(
        tmp_path: Path, *, lock_timeout_seconds: float
    ) -> AsyncIterator[SqliteEngravaCore]:
        db = await aiosqlite.connect(str(tmp_path / "contended.db"))
        db.row_factory = aiosqlite.Row
        try:
            s = SqliteEngravaCore(
                db=db,
                access_tracking_enabled=True,
                write_lock_acquire_timeout_seconds=lock_timeout_seconds,
            )
            await s.ensure_schema()
            yield s
        finally:
            await db.close()

    @staticmethod
    @contextlib.asynccontextmanager
    async def _write_lock_held_by_another_task(
        store: SqliteEngravaCore,
    ) -> AsyncIterator[Callable[[], None]]:
        """Hold the write lock in a separate task; yield the callable that frees it."""
        held = asyncio.Event()
        release = asyncio.Event()

        async def _holder() -> None:
            async with store._write_lock:
                held.set()
                await release.wait()

        task = asyncio.create_task(_holder())
        try:
            await held.wait()
            yield release.set
        finally:
            release.set()
            await task

    @staticmethod
    def _record_connection_calls(
        monkeypatch: pytest.MonkeyPatch, store: SqliteEngravaCore
    ) -> list[str]:
        """Return a list that gains the name of every statement or transaction call made."""
        calls: list[str] = []

        def recorder(name: str, real: Callable[..., object]) -> Callable[..., object]:
            def _record(*args: object, **kwargs: object) -> object:
                calls.append(name)
                return real(*args, **kwargs)

            return _record

        for name in (
            "execute",
            "executemany",
            "executescript",
            "execute_insert",
            "execute_fetchall",
            "commit",
            "rollback",
        ):
            monkeypatch.setattr(store._db, name, recorder(name, getattr(store._db, name)))
        return calls

    @staticmethod
    def _signal_when_lock_requested(
        monkeypatch: pytest.MonkeyPatch, store: SqliteEngravaCore, *, times: int
    ) -> asyncio.Event:
        """Return an event set once the write lock has been requested ``times`` times."""
        requested = asyncio.Event()
        calls = 0
        real_acquire = store._write_lock.acquire

        async def _acquire() -> None:
            nonlocal calls
            calls += 1
            if calls >= times:
                requested.set()
            await real_acquire()

        monkeypatch.setattr(store._write_lock, "acquire", _acquire)
        return requested

    @staticmethod
    async def _access_count(store: SqliteEngravaCore, thought_id: str) -> int:
        cursor = await store._db.execute(
            "SELECT access_count FROM thought WHERE thought_id = ?", (thought_id,)
        )
        row = await cursor.fetchone()
        assert row is not None
        return int(row[0])

    async def test_a_timeout_on_the_write_lock_keeps_the_buffered_events(
        self, tmp_path: Path
    ) -> None:
        async with self._tracking_store(tmp_path, lock_timeout_seconds=0.05) as store:
            await store.create_thought(_obs("t"))
            await store.get_thought("t")
            assert len(store._access_buffer) == 1

            async with self._write_lock_held_by_another_task(store) as release:
                with pytest.raises(WriteLockTimeoutError):
                    await store.flush_access_buffer()
                assert len(store._access_buffer) == 1, (
                    "a flush that could not take the write lock must not have emptied the buffer"
                )
                release()

            assert await store.flush_access_buffer() == 1
            assert len(store._access_buffer) == 0
            assert await self._access_count(store, "t") == 1

    async def test_a_flush_cancelled_while_it_waits_keeps_the_buffered_events(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async with self._tracking_store(tmp_path, lock_timeout_seconds=30.0) as store:
            await store.create_thought(_obs("t"))
            await store.get_thought("t")

            async with self._write_lock_held_by_another_task(store) as release:
                waiting = self._signal_when_lock_requested(monkeypatch, store, times=1)
                flush = asyncio.create_task(store.flush_access_buffer())
                try:
                    await waiting.wait()  # the flush has started and is waiting for the lock

                    flush.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await flush
                    assert len(store._access_buffer) == 1, (
                        "a flush cancelled while it waited for the write lock must not "
                        "have emptied the buffer"
                    )
                finally:
                    release()
                    await asyncio.gather(flush, return_exceptions=True)

            assert await store.flush_access_buffer() == 1
            assert await self._access_count(store, "t") == 1

    async def test_an_access_recorded_while_the_flush_waits_is_included(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async with self._tracking_store(tmp_path, lock_timeout_seconds=30.0) as store:
            await store.create_thought(_obs("t"))
            await store.get_thought("t")

            async with self._write_lock_held_by_another_task(store) as release:
                waiting = self._signal_when_lock_requested(monkeypatch, store, times=1)
                flush = asyncio.create_task(store.flush_access_buffer())
                try:
                    await waiting.wait()  # the flush has started and is waiting for the lock

                    await store.get_thought("t")  # a second access of the same id while it waits
                    release()
                    flushed = await flush
                finally:
                    release()
                    await asyncio.gather(flush, return_exceptions=True)

            assert flushed == 1  # one distinct id
            assert len(store._access_buffer) == 0
            assert await self._access_count(store, "t") == 2

    async def test_an_empty_flush_does_not_wait_for_the_write_lock(self, tmp_path: Path) -> None:
        async with self._tracking_store(tmp_path, lock_timeout_seconds=0.05) as store:
            assert len(store._access_buffer) == 0

            async with self._write_lock_held_by_another_task(store):
                assert await store.flush_access_buffer() == 0

    async def test_the_flush_that_lost_the_race_does_not_touch_the_database(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async with self._tracking_store(tmp_path, lock_timeout_seconds=30.0) as store:
            await store.create_thought(_obs("t"))
            await store.get_thought("t")
            calls = self._record_connection_calls(monkeypatch, store)
            calls.clear()

            assert await store.flush_access_buffer() == 1
            a_lone_flush = list(calls)
            assert "executemany" in a_lone_flush

            await store.get_thought("t")
            calls.clear()

            async with self._write_lock_held_by_another_task(store) as release:
                both_waiting = self._signal_when_lock_requested(monkeypatch, store, times=2)
                first = asyncio.create_task(store.flush_access_buffer())
                second = asyncio.create_task(store.flush_access_buffer())
                try:
                    await asyncio.wait_for(both_waiting.wait(), timeout=5)
                    release()
                    results = await asyncio.gather(first, second)
                finally:
                    release()
                    await asyncio.gather(first, second, return_exceptions=True)

            assert sorted(results) == [0, 1]
            assert calls == a_lone_flush, (
                "the flush that found nothing to write must not call the connection at all"
            )
            assert await self._access_count(store, "t") == 2


class TestAccessBuffer:
    """Unit invariants of the bounded FIFO access buffer.

    ``TestAccessSubstrate`` covers the store-level batching; these pin the
    buffer's own load-bearing guarantees — bounded memory via oldest-inserted
    eviction and lossless coalescing — that a revert to unbounded growth or
    wrong-order eviction would otherwise leave green.
    """

    def test_coalesce_increments_delta_without_growing(self) -> None:
        """Repeated accesses of one id coalesce into a single counted entry."""
        buf = _AccessBuffer(cap=8)
        for _ in range(3):
            buf.record("a", now="t")
        assert len(buf) == 1
        assert buf.drain() == [("a", 3, "t")]

    def test_drain_clears_the_buffer(self) -> None:
        """drain returns the pending deltas once, then empties."""
        buf = _AccessBuffer(cap=8)
        buf.record("a", now="t0")
        assert buf.drain() == [("a", 1, "t0")]
        assert len(buf) == 0
        assert buf.drain() == []

    def test_fifo_evicts_oldest_inserted_at_cap(self) -> None:
        """A new id beyond the cap evicts the oldest-inserted id (FIFO)."""
        buf = _AccessBuffer(cap=3)
        for tid in ("a", "b", "c"):
            buf.record(tid, now="t")
        buf.record("d", now="t")  # exceeds cap -> evict oldest-inserted ("a")
        assert len(buf) == 3
        assert {tid for tid, _, _ in buf.drain()} == {"b", "c", "d"}

    def test_eviction_is_counted(self) -> None:
        """Each eviction increments the running evicted-total."""
        buf = _AccessBuffer(cap=1)
        buf.record("a", now="t")
        buf.record("b", now="t")  # evicts "a"
        assert buf._evicted_total == 1
        assert {tid for tid, _, _ in buf.drain()} == {"b"}

    def test_coalesce_at_cap_never_evicts(self) -> None:
        """Coalescing into an existing id at the cap must not evict a peer."""
        buf = _AccessBuffer(cap=2)
        buf.record("a", now="t")
        buf.record("b", now="t")  # full
        buf.record("a", now="t2")  # coalesce -> must NOT evict "b"
        assert len(buf) == 2
        assert buf._evicted_total == 0
        assert {tid: delta for tid, delta, _ in buf.drain()} == {"a": 2, "b": 1}

    def test_cap_floored_at_one(self) -> None:
        """A non-positive cap is floored to 1, so the buffer stays bounded."""
        buf = _AccessBuffer(cap=0)  # floored to 1
        buf.record("a", now="t")
        buf.record("b", now="t")  # evicts "a"
        assert len(buf) == 1
        assert {tid for tid, _, _ in buf.drain()} == {"b"}

    def test_eviction_logs_warning_naming_the_id(self, caplog: pytest.LogCaptureFixture) -> None:
        """An eviction emits a WARNING naming the dropped thought id."""
        buf = _AccessBuffer(cap=1)
        buf.record("keep-me-first", now="t")
        with caplog.at_level(logging.WARNING, logger="engrava.infrastructure.sqlite.engrava_core"):
            buf.record("second", now="t")
        messages = [r.getMessage() for r in caplog.records]
        assert any("access buffer full" in m for m in messages)
        assert any("keep-me-first" in m for m in messages)


# ---------------------------------------------------------------------------
# Config wiring — from_config activation + signals merge + new fields
# ---------------------------------------------------------------------------


class TestConfigActivation:
    def test_partial_signals_merge_keeps_other_defaults(self) -> None:
        """Overriding one signal weight must not zero the other five."""
        cfg = _parse_dreaming({"enabled": True, "signals": {"recency": 0.5}})
        assert cfg is not None
        assert cfg.signals["recency"] == 0.5
        # The other five keep their defaults, not dropped.
        assert set(cfg.signals) == {
            "recency",
            "staleness",
            "confirmation",
            "confidence",
            "frequency",
            "action_outcome",
        }
        assert cfg.signals["staleness"] == 0.20

    def test_new_dreaming_fields_parse_from_yaml(self) -> None:
        cfg = _parse_dreaming(
            {
                "enabled": True,
                "access_tracking_enabled": False,
                "self_filter_mode": "self_only",
                "min_source_confidence": "high",
                "boilerplate_threshold": 0.5,
                "eligible_content_types": ["note", "fact"],
            }
        )
        assert cfg is not None
        assert cfg.access_tracking_enabled is False
        assert cfg.self_filter_mode == "self_only"
        assert cfg.min_source_confidence == "high"
        assert cfg.boilerplate_threshold == 0.5
        assert cfg.eligible_content_types == frozenset({"note", "fact"})

    def test_access_tracking_rejects_non_bool(self) -> None:
        from engrava.config import ConfigError

        with pytest.raises(ConfigError):
            _parse_dreaming({"enabled": True, "access_tracking_enabled": "yes"})

    async def test_from_config_wires_and_runs_dreaming(self, tmp_path: Path) -> None:
        """A YAML-only user activates dreaming end-to-end via store.consolidate."""
        db_path = tmp_path / "yaml.db"
        cfg_file = tmp_path / "engrava.yaml"
        cfg_file.write_text(
            "database:\n"
            f"  path: {db_path}\n"
            "extensions:\n"
            "  dreaming:\n"
            "    enabled: true\n"
            "    gates:\n"
            "      enable_reflections: false\n",
            encoding="utf-8",
        )
        store = await SqliteEngravaCore.from_config(cfg_file)
        try:
            assert store._dreaming_extension is not None
            # The documented gates key flows through the YAML path (it is not
            # silently dropped): reflections are disabled as configured.
            assert store._dreaming_extension.config.gates.enable_reflections is False
            assert store._access_tracking_enabled is True
            for i in range(4):
                await store.create_thought(_obs(f"obs-{i}"))
            result = await store.consolidate(current_cycle=_CYCLE)
            assert result.promoted_count >= 1
        finally:
            await store.close()

    async def test_consolidate_without_dreaming_raises(self, store: SqliteEngravaCore) -> None:
        """A manually-built store has no wired extension → consolidate() raises."""
        with pytest.raises(RuntimeError, match="dreaming"):
            await store.consolidate(current_cycle=1)


# ---------------------------------------------------------------------------
# attach_dreaming_extension — the public seam beside the private write
# ---------------------------------------------------------------------------


async def _seeded_store(
    tmp_path: Path, name: str
) -> tuple[aiosqlite.Connection, SqliteEngravaCore]:
    """A manually-constructed store seeded with promotable candidates.

    Returns the raw connection alongside the store: a manual constructor
    never owns its connection (see ``SqliteEngravaCore.close``), so the
    caller — not ``store.close()`` — is responsible for closing it, exactly
    as the module-level ``store`` fixture above does.
    """
    db = await aiosqlite.connect(str(tmp_path / name))
    db.row_factory = aiosqlite.Row
    s = SqliteEngravaCore(db=db)
    await s.ensure_schema()
    for i in range(4):
        await s.create_thought(_obs(f"obs-{i}"))
    return db, s


class TestAttachDreamingExtension:
    """The public seam for wiring a consolidator, beside the private write."""

    async def test_attach_matches_private_write_behaviour(self, tmp_path: Path) -> None:
        """Attaching through the seam runs identically to the private write.

        Two identically-seeded stores, one wired through
        ``attach_dreaming_extension`` and one through the private attribute
        write, must produce the same ``ConsolidationResult`` for the same
        input — proving the seam does not just set a flag but drives the same
        code path that ``consolidate()`` reads.
        """
        cfg = _activation_cfg()
        seam_db, via_seam = await _seeded_store(tmp_path, "via_seam.db")
        private_db, via_private = await _seeded_store(tmp_path, "via_private.db")
        try:
            via_seam.attach_dreaming_extension(DreamingExtension(config=cfg))
            via_private._dreaming_extension = DreamingExtension(config=cfg)

            seam_result = await via_seam.consolidate(current_cycle=_CYCLE)
            private_result = await via_private.consolidate(current_cycle=_CYCLE)

            assert seam_result == private_result
            assert seam_result.promoted_count >= 1
        finally:
            await seam_db.close()
            await private_db.close()

    async def test_second_attach_replaces_the_first(self, store: SqliteEngravaCore) -> None:
        """Attaching again replaces whatever was attached before.

        There is no "already attached" refusal: the second call simply wins,
        exactly as a second private-attribute write would. This is
        demonstrated by identity (the store now points at the second
        extension), not merely by the absence of an exception.
        """
        first = DreamingExtension(config=_activation_cfg())
        second = DreamingExtension(config=_activation_cfg())

        store.attach_dreaming_extension(first)
        assert store._dreaming_extension is first

        store.attach_dreaming_extension(second)
        assert store._dreaming_extension is second
        assert store._dreaming_extension is not first

    async def test_private_write_still_works_and_agrees_with_the_seam(self, tmp_path: Path) -> None:
        """The private path is untouched: it still wires and runs dreaming.

        Both doors set the same single attribute, so a store wired through
        one and then re-wired through the other ends up in exactly the state
        the second call describes — they cannot disagree about what is
        attached because there is only one slot.
        """
        db, s = await _seeded_store(tmp_path, "private_then_seam.db")
        try:
            ext_a = DreamingExtension(config=_activation_cfg())
            s._dreaming_extension = ext_a
            assert s._dreaming_extension is ext_a

            ext_b = DreamingExtension(config=_activation_cfg())
            s.attach_dreaming_extension(ext_b)
            assert s._dreaming_extension is ext_b

            result = await s.consolidate(current_cycle=_CYCLE)
            assert result.promoted_count >= 1
        finally:
            await db.close()

    async def test_attach_accepts_a_conforming_extension(self, store: SqliteEngravaCore) -> None:
        """An object implementing ``run_consolidation`` is accepted."""
        ext = DreamingExtension(config=_activation_cfg())
        store.attach_dreaming_extension(ext)
        assert store._dreaming_extension is ext

    async def test_attach_rejects_a_non_conforming_object(self, store: SqliteEngravaCore) -> None:
        """An object without ``run_consolidation`` is refused at the door.

        A seam that accepted anything here would only fail later, deep inside
        a consolidation cycle; this proves it refuses immediately instead.
        """

        class NotAnExtension:
            pass

        with pytest.raises(TypeError, match="DreamingConsolidatorProtocol"):
            store.attach_dreaming_extension(NotAnExtension())  # type: ignore[arg-type]

        assert store._dreaming_extension is None
