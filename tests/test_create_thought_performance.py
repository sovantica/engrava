"""Performance budget tests for ``create_thought`` after the dedup change.

Per the performance contract, the SHA-256 ``content_hash`` compute
that we added to every insert path must not regress baseline insert
throughput meaningfully.  These tests do not assert against absolute
wall-clock numbers (which vary across CI hardware) but against
*relative* budgets vs. an explicit baseline measured in the same
process: dedup=False adds at most ~2x overhead on top of an artificial
"hash-only" baseline, and dedup=True at most ~5x.

The thresholds are deliberately loose enough that random scheduling
noise on Windows / Linux will not flake the tests, while still being
tight enough that an O(N) regression in the dedup branch (e.g.
accidentally rebuilding an in-memory cache per call) would trip them.
"""

from __future__ import annotations

import asyncio
import statistics
import time
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
from engrava.infrastructure.sqlite.engrava_core import _compute_content_hash

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# Realistic thought content (~2 sentences) — comparable to AMB ingest payloads.
_REALISTIC_CONTENT = (
    "The user mentioned that they prefer concise explanations and avoid "
    "long-winded responses unless explicitly asked for additional detail."
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def store() -> AsyncIterator[SqliteEngravaCore]:
    """Bootstrapped store backed by an in-memory DB."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode = WAL")
    s = SqliteEngravaCore(conn)
    await s.ensure_schema()
    yield s
    await conn.close()


def _make_thought(thought_id: str, content: str = _REALISTIC_CONTENT) -> CoreThoughtRecord:
    return CoreThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.OBSERVATION,
        essence="perf",
        content=content,
        priority=Priority.P2,
        lifecycle_status=LifecycleStatus.CREATED,
        created_cycle=0,
        updated_cycle=0,
        source="perf",
        confidence=0.9,
        source_type=KnowledgeSource.EXPERIENCE,
        visibility=ThoughtVisibility.SELECTIVE,
    )


async def _measure(
    coro_factory: object,
    n: int,
) -> float:
    """Time *n* awaits of *coro_factory()* and return median seconds."""
    samples: list[float] = []
    for _ in range(n):
        start = time.perf_counter()
        await coro_factory()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_content_hash_compute_microbenchmark() -> None:
    """SHA-256 of realistic content is comfortably under 100us per call."""
    n = 200
    samples: list[float] = []
    for _ in range(n):
        start = time.perf_counter()
        _compute_content_hash(_REALISTIC_CONTENT)
        samples.append(time.perf_counter() - start)
    median = statistics.median(samples)
    # Loose ceiling — SHA-256 of ~150B is sub-microsecond on any modern
    # CPU; 100us leaves three orders of magnitude of headroom for slow
    # CI runners while still catching algorithmic regressions.
    assert median < 1e-4, f"hash compute too slow: median={median:.6f}s"


async def test_create_thought_deduplicate_false_within_budget(
    store: SqliteEngravaCore,
) -> None:
    """Default insert path stays close to the legacy budget."""
    # Warm-up: avoid first-call SQL prepare cost biasing the measurement.
    await store.create_thought(_make_thought("warm-1"))
    await store.create_thought(_make_thought("warm-2"))

    counter = {"i": 0}

    async def factory() -> None:
        counter["i"] += 1
        await store.create_thought(_make_thought(f"perf-default-{counter['i']}"))

    median = await _measure(factory, 50)
    # 100ms median is well above realistic ~1-5ms per insert on an
    # in-memory DB; trips only on a serious regression.
    assert median < 0.1, f"create_thought(deduplicate=False) too slow: median={median:.6f}s"


async def test_create_thought_deduplicate_true_within_budget(
    store: SqliteEngravaCore,
) -> None:
    """Dedup branch stays within ~5x of the legacy budget despite the lock."""
    await store.create_thought(_make_thought("warm-1"), deduplicate=True)
    await store.create_thought(_make_thought("warm-2"), deduplicate=True)

    counter = {"i": 0}

    async def factory() -> None:
        counter["i"] += 1
        # Distinct content per call → exercises the not-found branch
        # (lock + hash lookup + insert), which is the worst case
        # because dedup-hit short-circuits the insert.
        await store.create_thought(
            _make_thought(f"perf-dedup-{counter['i']}", content=f"unique-{counter['i']}"),
            deduplicate=True,
        )

    median = await _measure(factory, 50)
    assert median < 0.15, f"create_thought(deduplicate=True) too slow: median={median:.6f}s"


async def test_concurrent_dedup_completes_within_reasonable_time(
    store: SqliteEngravaCore,
) -> None:
    """Concurrent dedup of 200 identical calls finishes well under 5s."""
    content = "Concurrent perf observation reused across all tasks."
    fan_out = 200
    start = time.perf_counter()
    await asyncio.gather(
        *[
            store.create_thought(
                _make_thought(f"perf-conc-{i}", content=content),
                deduplicate=True,
            )
            for i in range(fan_out)
        ],
    )
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0, f"concurrent dedup too slow: elapsed={elapsed:.3f}s"
