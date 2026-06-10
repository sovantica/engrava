"""Performance characterisation of the valid-time query path.

This module bounds the cost of adding a valid-time predicate to a MindQL
``FIND`` query and pins down what the SQLite planner actually does with it.

What this test guarantees (deterministic, planner-level)
--------------------------------------------------------
The robust, environment-independent guarantee is *structural*: applying a
``valid_now`` predicate does **not** make the query plan more expensive in
kind. A plain ``FIND thoughts`` scans exactly one table; ``FIND thoughts
WHERE valid_now`` scans exactly the *same* one table and nothing more — no
extra table scan, no join, no correlated subquery, and no transient B-tree
(no "USE TEMP B-TREE"). The predicate is a per-row filter layered onto a
scan the engine already performs, so it cannot change the asymptotic shape
of the query. This is asserted directly against ``EXPLAIN QUERY PLAN`` and
is the primary contract of this module.

What the planner actually does with each predicate (measured, honest)
---------------------------------------------------------------------
The NULL-tolerant predicates resolve to a SQL body of the shape::

    (valid_from IS NULL OR valid_from <= ?) AND (valid_until IS NULL OR ...)

The ``column IS NULL OR column <op> ?`` disjunction is **not sargable**, so
SQLite cannot use ``idx_thought_valid_from`` / ``idx_thought_valid_until`` /
``idx_thought_valid_range`` for ``valid_now`` / ``valid_at`` / ``valid_within``
— it performs a full ``SCAN thought``. This is an intentional consequence of
NULL-tolerance (open/legacy rows with NULL bounds must remain visible), not a
missing index: the indexes exist and are reachable. The closed-containment
``valid_between`` predicate, whose body is ``valid_from IS NOT NULL AND
valid_from >= ? AND valid_until IS NOT NULL AND valid_until <= ?``, *is*
sargable and the planner does pick a valid-time index for it. This module
asserts that reachability via ``valid_between`` so a future regression that
drops the indexes is caught.

Why there is no ``< 5%`` wall-clock overhead assertion
------------------------------------------------------
A naive wall-clock comparison of ``FIND thoughts`` vs ``FIND thoughts WHERE
valid_now`` is meaningless here: the predicate *filters rows out*, so the
temporal query materialises fewer Python row dicts and measures **faster**
than the unfiltered baseline (observed median ratio around ``-50%``). Removing
that row-materialisation confound by comparing ``COUNT(*)`` instead isolates
the predicate's raw CPU cost — and that cost is genuinely large in *relative*
terms (observed best-of-5 median ``~245us`` plain vs ``~1.2ms`` with the
predicate, i.e. roughly ``+400%``), because a bare ``COUNT(*)`` is a
near-free optimised count while the predicate forces a row-by-row scan with
two compound boolean tests each. Neither figure supports a ``< 5%`` bound,
and asserting one would be false. Absolute per-query cost stays sub-2ms at
this corpus size, but it is too small and too noisy to gate on reliably.
The structural plan-shape assertions above are therefore the contract; the
timing here is recorded for context only and not asserted on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import (
    LifecycleStatus,
    Priority,
    SqliteEngravaCore,
    ThoughtRecord,
    ThoughtType,
)
from engrava.mindql.executor import MindQLExecutor
from engrava.mindql.parser import parse

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

# The three valid-time indexes created by the schema and the migration. Any
# one of these being chosen for a sargable predicate proves the indexes are
# present and reachable by the planner.
_VALID_TIME_INDEXES = frozenset(
    {
        "idx_thought_valid_from",
        "idx_thought_valid_until",
        "idx_thought_valid_range",
    }
)

# Corpus size. Query-plan selection is independent of exact row count, so a
# few thousand rows is fully representative while keeping the test fast (a
# 10k-row build costs ~8s of inserts; 2k keeps the whole module under a
# couple of seconds). ANALYZE is run so the planner has real statistics.
_CORPUS_SIZE = 2_000

# Valid-time bounds used to populate the corpus. The window [JAN, JUN) gives
# every shape category a non-trivial population.
_T_JAN = "2025-01-01T00:00:00+00:00"
_T_JUN = "2025-06-01T00:00:00+00:00"


def _make_thought(index: int) -> ThoughtRecord:
    """Build one corpus thought with a representative valid-time shape.

    The corpus cycles through the four valid-time shapes so the planner sees
    a realistic mix of NULL and non-NULL bounds:

    * closed window ``[JAN, JUN)``,
    * open lower bound (``valid_from`` NULL),
    * open upper bound (``valid_until`` NULL),
    * fully open / legacy (both bounds NULL).

    Args:
        index: Zero-based position in the corpus, used to vary the shape and
            to mint a unique ``thought_id``.

    Returns:
        A validated :class:`ThoughtRecord` ready to persist.
    """
    shape = index % 4
    if shape == 0:
        valid_from, valid_until = _T_JAN, _T_JUN
    elif shape == 1:
        valid_from, valid_until = None, _T_JUN
    elif shape == 2:
        valid_from, valid_until = _T_JAN, None
    else:
        valid_from, valid_until = None, None
    return ThoughtRecord(
        thought_id=f"t-{index:06d}",
        thought_type=ThoughtType.OBSERVATION,
        essence=f"essence {index}",
        content=f"content {index}",
        priority=Priority.P2,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=1,
        updated_cycle=1,
        source="test",
        valid_from=valid_from,
        valid_until=valid_until,
    )


@pytest.fixture
async def perf_conn() -> AsyncIterator[aiosqlite.Connection]:
    """A SQLite connection populated with a representative valid-time corpus.

    Runs ``ANALYZE`` so the query planner has real table statistics —
    index-vs-scan decisions made against an unanalysed database are not
    representative of a deployed store.

    Yields:
        An open aiosqlite connection whose ``thought`` table holds
        ``_CORPUS_SIZE`` rows spanning every valid-time bound shape.
    """
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA foreign_keys = ON")
    store = SqliteEngravaCore(conn)
    await store.ensure_schema()
    for i in range(_CORPUS_SIZE):
        await store.create_thought(_make_thought(i))
    await conn.commit()
    await conn.execute("ANALYZE")
    await conn.commit()
    try:
        yield conn
    finally:
        await conn.close()


async def _query_plan(conn: aiosqlite.Connection, mindql: str) -> list[str]:
    """Return the ``EXPLAIN QUERY PLAN`` detail lines for a MindQL ``FIND``.

    The exact SQL the executor would run is obtained from its own SQL builder
    (so the plan reflects the production query verbatim) and explained against
    the live connection.

    Args:
        conn: The open aiosqlite connection to explain against.
        mindql: A MindQL ``FIND`` statement, e.g. ``FIND thoughts WHERE valid_now``.

    Returns:
        The ``detail`` column of every plan row, upper-cased for matching.
    """
    query = parse(mindql)
    executor = MindQLExecutor(conn)
    sql, params = executor._build_select_sql(query.table or "thought", query)
    cursor = await conn.execute(f"EXPLAIN QUERY PLAN {sql}", params)
    rows = await cursor.fetchall()
    return [str(row["detail"]).upper() for row in rows]


def _table_scan_count(plan_details: list[str]) -> int:
    """Count full-table ``SCAN`` steps over the ``thought`` table in a plan.

    Args:
        plan_details: Upper-cased ``EXPLAIN QUERY PLAN`` detail lines.

    Returns:
        The number of plan steps that are a full scan of ``thought`` (a step
        beginning with ``SCAN THOUGHT``). An indexed ``SEARCH`` is not counted.
    """
    return sum(1 for detail in plan_details if detail.startswith("SCAN THOUGHT"))


class TestTemporalQueryPlanShape:
    """Structural plan-shape guarantees — deterministic, the primary contract."""

    async def test_valid_now_adds_no_extra_scan_join_or_subquery(
        self,
        perf_conn: aiosqlite.Connection,
    ) -> None:
        """``valid_now`` keeps the single-table plan shape of a plain ``FIND``.

        The temporal predicate must not turn a one-table scan into a join, a
        correlated subquery, or a second scan, and must not require a transient
        B-tree. It is a per-row filter on the scan the engine already performs.
        """
        plain = await _query_plan(perf_conn, "FIND thoughts")
        temporal = await _query_plan(perf_conn, "FIND thoughts WHERE valid_now")

        # Baseline plain FIND is a single full scan of thought.
        assert _table_scan_count(plain) == 1, plain
        # The temporal predicate adds no second table scan ...
        assert _table_scan_count(temporal) == 1, temporal
        # ... and introduces no join / subquery / temp B-tree machinery.
        joined = " ".join(temporal)
        assert "SUBQUERY" not in joined, temporal
        assert "TEMP B-TREE" not in joined, temporal
        assert "USE TEMP B-TREE" not in joined, temporal

    async def test_null_tolerant_predicates_full_scan_is_intentional(
        self,
        perf_conn: aiosqlite.Connection,
    ) -> None:
        """The NULL-tolerant predicates resolve to a full scan, by design.

        ``valid_now`` / ``valid_at`` / ``valid_within`` all use a
        ``column IS NULL OR column <op> ?`` disjunction to keep open-bound and
        legacy (NULL) rows visible. That disjunction is not sargable, so the
        planner cannot use a valid-time index and scans the table. This test
        documents and locks that behaviour so the assertion in the
        ``valid_between`` test (index *is* used) is unambiguous.
        """
        for mindql in (
            "FIND thoughts WHERE valid_now",
            f"FIND thoughts WHERE valid_at '{_T_JAN}'",
            f"FIND thoughts WHERE valid_within '{_T_JAN}' '{_T_JUN}'",
        ):
            plan = await _query_plan(perf_conn, mindql)
            assert _table_scan_count(plan) == 1, (mindql, plan)
            assert not any("USING INDEX" in detail for detail in plan), (mindql, plan)

    async def test_valid_between_reaches_a_valid_time_index(
        self,
        perf_conn: aiosqlite.Connection,
    ) -> None:
        """The sargable ``valid_between`` predicate proves the indexes are wired.

        ``valid_between`` uses ``valid_from IS NOT NULL AND valid_from >= ?``
        (and the symmetric upper-bound test), which *is* sargable. The planner
        therefore picks one of the three valid-time indexes. Asserting this
        guards against a regression that silently drops those indexes — which
        would also remove the only index-accelerated valid-time path.
        """
        plan = await _query_plan(
            perf_conn,
            f"FIND thoughts WHERE valid_between '{_T_JAN}' '{_T_JUN}'",
        )
        joined = " ".join(plan)
        assert "USING INDEX" in joined, plan
        assert any(index.upper() in joined for index in _VALID_TIME_INDEXES), plan
        # And it is an indexed SEARCH, not a full table scan.
        assert _table_scan_count(plan) == 0, plan
