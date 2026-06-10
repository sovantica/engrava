"""Tests for MindQL parser and executor.

Covers parsing of FIND, COUNT, SELECT, and extension commands,
as well as executor integration with aiosqlite.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import (
    EdgeRecord,
    EdgeType,
    LifecycleStatus,
    Priority,
    SqliteEngravaCore,
    ThoughtRecord,
    ThoughtType,
)
from engrava.domain.protocols.hooks import MindQLExtension
from engrava.mindql import executor as executor_module
from engrava.mindql.executor import MindQLExecutor
from engrava.mindql.parser import (
    Condition,
    MindQLCommand,
    MindQLOperator,
    MindQLParseError,
    MindQLQuery,
    TemporalPredicate,
    TemporalPredicateKind,
    parse,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def db() -> AsyncIterator[aiosqlite.Connection]:
    """Create an in-memory SQLite database with core schema."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode = WAL")
    await conn.execute("PRAGMA foreign_keys = ON")
    store = SqliteEngravaCore(conn)
    await store.ensure_schema()
    yield conn
    await conn.close()


@pytest.fixture
async def populated_db(db: aiosqlite.Connection) -> aiosqlite.Connection:
    """Database with sample data for query testing."""
    store = SqliteEngravaCore(db)
    for i in range(5):
        t = ThoughtRecord(
            thought_id=f"t-{i:03d}",
            thought_type=ThoughtType.OBSERVATION,
            essence=f"Thought number {i}",
            content=f"Full content of thought {i}",
            priority=Priority.P1 if i < 2 else Priority.P2,
            lifecycle_status=LifecycleStatus.ACTIVE if i < 4 else LifecycleStatus.ARCHIVED,
            created_cycle=i + 1,
            updated_cycle=i + 1,
            source="test",
            confidence=0.8,
        )
        await store.create_thought(t)
    return db


# Fixed ISO-8601 instants used across the temporal-predicate tests. The window
# of the "bounded" rows is [JAN, JUN); MID falls inside it, BEFORE/AFTER do not.
_T_JAN = "2025-01-01T00:00:00+00:00"
_T_FEB = "2025-02-01T00:00:00+00:00"
_T_MID = "2025-03-01T00:00:00+00:00"
_T_JUN = "2025-06-01T00:00:00+00:00"
_T_DEC = "2025-12-31T00:00:00+00:00"
_T_BEFORE = "2024-01-01T00:00:00+00:00"
_T_AFTER = "2026-01-01T00:00:00+00:00"


@pytest.fixture
def pinned_now() -> Iterator[None]:
    """Pin ``valid_now`` to a deterministic instant for the duration of a test.

    Resets the context variable afterwards so tests stay isolated.
    """
    token = executor_module.mindql_now.set(_T_MID)
    try:
        yield
    finally:
        executor_module.mindql_now.reset(token)


@pytest.fixture
async def temporal_db(db: aiosqlite.Connection) -> aiosqlite.Connection:
    """Database with thoughts and edges spanning a range of valid-time bounds.

    Layout (both tables share the same shape):

    * ``*-bounded``  — closed window ``[JAN, JUN)``.
    * ``*-open-from``— ``valid_from`` NULL, ``valid_until`` JUN.
    * ``*-open-until``— ``valid_from`` JAN, ``valid_until`` NULL.
    * ``*-legacy``   — both bounds NULL (pre-feature / un-backfilled rows).
    * ``*-future``   — closed window ``[AFTER, +inf)`` (begins after MID).
    """
    store = SqliteEngravaCore(db)

    def mk_thought(suffix: str, vf: str | None, vu: str | None) -> ThoughtRecord:
        return ThoughtRecord(
            thought_id=f"t-{suffix}",
            thought_type=ThoughtType.OBSERVATION,
            essence=f"essence {suffix}",
            content=f"content {suffix}",
            priority=Priority.P1,
            lifecycle_status=LifecycleStatus.ACTIVE,
            created_cycle=1,
            updated_cycle=1,
            source="test",
            valid_from=vf,
            valid_until=vu,
        )

    rows: list[tuple[str, str | None, str | None]] = [
        ("bounded", _T_JAN, _T_JUN),
        ("open-from", None, _T_JUN),
        ("open-until", _T_JAN, None),
        ("legacy", None, None),
        ("future", _T_AFTER, None),
    ]
    for suffix, vf, vu in rows:
        await store.create_thought(mk_thought(suffix, vf, vu))

    # A shared source anchor plus one distinct target per edge so every edge
    # has unique (from, to, type) — the edge table enforces that as UNIQUE.
    # These wiring thoughts use a ``wire-`` id namespace so they never collide
    # with the ``t-<shape>`` rows the thought-table assertions reason about.
    await store.create_thought(mk_thought("wire-anchor", _T_JAN, None))
    for suffix, vf, vu in rows:
        await store.create_thought(mk_thought(f"wire-target-{suffix}", _T_JAN, None))
        await store.create_edge(
            EdgeRecord(
                edge_id=f"e-{suffix}",
                from_thought_id="t-wire-anchor",
                to_thought_id=f"t-wire-target-{suffix}",
                edge_type=EdgeType.ASSOCIATED,
                weight=0.5,
                created_cycle=1,
                valid_from=vf,
                valid_until=vu,
            )
        )
    await db.commit()
    return db


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


class TestParserFind:
    """Test FIND query parsing."""

    def test_find_thoughts_no_where(self) -> None:
        q = parse("FIND thoughts")
        assert q.command == MindQLCommand.FIND
        assert q.table == "thought"
        assert q.conditions == []
        assert q.limit is None

    def test_find_thoughts_with_where(self) -> None:
        q = parse("FIND thoughts WHERE lifecycle_status = 'ACTIVE'")
        assert q.command == MindQLCommand.FIND
        assert q.table == "thought"
        assert len(q.conditions) == 1
        assert q.conditions[0].field == "lifecycle_status"
        assert q.conditions[0].operator == MindQLOperator.EQ
        assert q.conditions[0].value == "ACTIVE"

    def test_find_with_multiple_conditions(self) -> None:
        q = parse("FIND thoughts WHERE priority = 'P1' AND lifecycle_status = 'ACTIVE'")
        assert len(q.conditions) == 2

    def test_find_with_limit(self) -> None:
        q = parse("FIND thoughts WHERE priority = 'P1' LIMIT 10")
        assert q.limit == 10

    def test_find_limit_only(self) -> None:
        q = parse("FIND edges LIMIT 5")
        assert q.table == "edge"
        assert q.limit == 5

    def test_find_numeric_comparison(self) -> None:
        q = parse("FIND thoughts WHERE created_cycle > 5")
        assert q.conditions[0].operator == MindQLOperator.GT
        assert q.conditions[0].value == 5

    def test_find_ne_operator(self) -> None:
        q = parse("FIND thoughts WHERE priority != 'P4'")
        assert q.conditions[0].operator == MindQLOperator.NE

    def test_find_ge_operator(self) -> None:
        q = parse("FIND thoughts WHERE updated_cycle >= 3")
        assert q.conditions[0].operator == MindQLOperator.GE
        assert q.conditions[0].value == 3

    def test_find_case_insensitive_command(self) -> None:
        q = parse("find thoughts")
        assert q.command == MindQLCommand.FIND

    def test_find_plural_and_singular_table(self) -> None:
        q1 = parse("FIND thoughts")
        q2 = parse("FIND thought")
        assert q1.table == q2.table == "thought"

    def test_find_all_tables(self) -> None:
        for name in ("thoughts", "edges", "embeddings", "actions"):
            q = parse(f"FIND {name}")
            assert q.table is not None


class TestParserCount:
    """Test COUNT query parsing."""

    def test_count_simple(self) -> None:
        q = parse("COUNT thoughts")
        assert q.command == MindQLCommand.COUNT
        assert q.table == "thought"
        assert q.conditions == []

    def test_count_with_where(self) -> None:
        q = parse("COUNT thoughts WHERE priority = 'P1'")
        assert len(q.conditions) == 1


class TestParserSelect:
    """Test SELECT passthrough parsing."""

    def test_select_passthrough(self) -> None:
        sql = "SELECT thought_id FROM thought WHERE lifecycle_status = 'ACTIVE'"
        q = parse(sql)
        assert q.command == MindQLCommand.SELECT
        assert q.raw_sql == sql


class TestParserExtension:
    """Test extension command parsing."""

    def test_extension_command(self) -> None:
        q = parse("LAYER 3", known_extensions={"LAYER"})
        assert q.command == MindQLCommand.EXTENSION
        assert q.extension_name == "LAYER"
        assert q.extension_args == ["3"]

    def test_extension_multiple_args(self) -> None:
        q = parse("CUSTOM foo bar baz", known_extensions={"CUSTOM"})
        assert q.extension_args == ["foo", "bar", "baz"]


class TestParserErrors:
    """Test parser error handling."""

    def test_empty_query(self) -> None:
        with pytest.raises(MindQLParseError, match="Empty"):
            parse("")

    def test_unknown_command(self) -> None:
        with pytest.raises(MindQLParseError, match="Unknown command"):
            parse("DROP thoughts")

    def test_unknown_table(self) -> None:
        with pytest.raises(MindQLParseError, match="Unknown table"):
            parse("FIND users")

    def test_find_missing_table(self) -> None:
        with pytest.raises(MindQLParseError, match="requires a table"):
            parse("FIND")


# ---------------------------------------------------------------------------
# Executor tests
# ---------------------------------------------------------------------------


class TestExecutorFind:
    """Test FIND execution."""

    async def test_find_all(self, populated_db: aiosqlite.Connection) -> None:
        executor = MindQLExecutor(populated_db)
        q = parse("FIND thoughts")
        result = await executor.execute(q)
        assert len(result.rows) == 5
        assert result.command == "FIND"

    async def test_find_with_where(self, populated_db: aiosqlite.Connection) -> None:
        executor = MindQLExecutor(populated_db)
        q = parse("FIND thoughts WHERE lifecycle_status = 'ACTIVE'")
        result = await executor.execute(q)
        assert len(result.rows) == 4

    async def test_find_with_limit(self, populated_db: aiosqlite.Connection) -> None:
        executor = MindQLExecutor(populated_db)
        q = parse("FIND thoughts LIMIT 2")
        result = await executor.execute(q)
        assert len(result.rows) == 2

    async def test_find_priority_filter(self, populated_db: aiosqlite.Connection) -> None:
        executor = MindQLExecutor(populated_db)
        q = parse("FIND thoughts WHERE priority = 'P1'")
        result = await executor.execute(q)
        assert len(result.rows) == 2


class TestExecutorCount:
    """Test COUNT execution."""

    async def test_count_all(self, populated_db: aiosqlite.Connection) -> None:
        executor = MindQLExecutor(populated_db)
        q = parse("COUNT thoughts")
        result = await executor.execute(q)
        assert result.count == 5


# ---------------------------------------------------------------------------
# SqliteEngravaCore.execute_mindql — store-level execution entry point
# ---------------------------------------------------------------------------


class TestStoreExecuteMindql:
    """Test the store-level ``execute_mindql`` entry point.

    The method is deliberately **policy-free**: it executes whatever command
    the parsed query carries (FIND / COUNT / SELECT / extension). Command-set
    restriction (e.g. FIND-only) is a consumer concern, not the core's.
    """

    async def test_find_executes(self, populated_db: aiosqlite.Connection) -> None:
        store = SqliteEngravaCore(populated_db)
        result = await store.execute_mindql(parse("FIND thoughts WHERE priority = 'P1'"))
        assert result.command == MindQLCommand.FIND
        assert len(result.rows) == 2

    async def test_count_executes_not_rejected(
        self,
        populated_db: aiosqlite.Connection,
    ) -> None:
        # Proves the method does NOT enforce FIND-only: COUNT runs.
        store = SqliteEngravaCore(populated_db)
        result = await store.execute_mindql(parse("COUNT thoughts"))
        assert result.command == MindQLCommand.COUNT
        assert result.count == 5

    async def test_select_executes_not_rejected(
        self,
        populated_db: aiosqlite.Connection,
    ) -> None:
        # Proves neutrality further: raw-SQL SELECT passthrough runs.
        store = SqliteEngravaCore(populated_db)
        result = await store.execute_mindql(
            parse("SELECT thought_id FROM thought WHERE lifecycle_status = 'ACTIVE'"),
        )
        assert result.command == MindQLCommand.SELECT
        assert len(result.rows) == 4

    async def test_extensions_param_routes_command(
        self,
        populated_db: aiosqlite.Connection,
    ) -> None:
        # The optional extensions map is wired through to the executor.
        received: dict[str, object] = {}

        async def _handler(
            conn: aiosqlite.Connection,
            args: object,
        ) -> list[dict[str, object]]:
            received["args"] = args
            return [{"ok": True}]

        extensions = {
            "PING": MindQLExtension(
                command_name="PING",
                handler=_handler,
                description="echo",
            ),
        }
        store = SqliteEngravaCore(populated_db)
        parsed = parse("PING hello", known_extensions={"PING"})
        result = await store.execute_mindql(parsed, extensions=extensions)
        # The extensions map was wired through and the registered handler ran.
        assert result.command == "PING"
        assert result.rows == [{"ok": True}]
        assert received["args"] is not None

    async def test_invalid_find_column_raises(
        self,
        populated_db: aiosqlite.Connection,
    ) -> None:
        store = SqliteEngravaCore(populated_db)
        q = MindQLQuery(
            command=MindQLCommand.FIND,
            table="thought",
            conditions=[
                Condition(field="nonexistent_col", operator=MindQLOperator.EQ, value="x"),
            ],
        )
        with pytest.raises(MindQLParseError, match="not allowed"):
            await store.execute_mindql(q)

    async def test_count_with_filter(self, populated_db: aiosqlite.Connection) -> None:
        executor = MindQLExecutor(populated_db)
        q = parse("COUNT thoughts WHERE priority = 'P1'")
        result = await executor.execute(q)
        assert result.count == 2


class TestExecutorSelect:
    """Test SELECT passthrough execution."""

    async def test_select(self, populated_db: aiosqlite.Connection) -> None:
        executor = MindQLExecutor(populated_db)
        q = parse("SELECT thought_id FROM thought WHERE lifecycle_status = 'ACTIVE'")
        result = await executor.execute(q)
        assert len(result.rows) == 4
        assert "thought_id" in result.columns

    async def test_select_reject_non_select(self, populated_db: aiosqlite.Connection) -> None:
        executor = MindQLExecutor(populated_db)
        q = MindQLQuery(command=MindQLCommand.SELECT, raw_sql="DELETE FROM thought")
        with pytest.raises(MindQLParseError, match="SELECT"):
            await executor.execute(q)


class TestExecutorExtension:
    """Test extension command execution."""

    async def test_extension_handler(self, populated_db: aiosqlite.Connection) -> None:
        async def _echo_handler(
            db: object,
            args: list[str],
        ) -> list[dict[str, object]]:
            return [{"echo": " ".join(args)}]

        ext = MindQLExtension(
            command_name="ECHO",
            handler=_echo_handler,
            description="Echo args",
        )
        executor = MindQLExecutor(populated_db, extensions={"ECHO": ext})
        q = parse("ECHO hello world", known_extensions={"ECHO"})
        result = await executor.execute(q)
        assert result.rows == [{"echo": "hello world"}]

    async def test_unknown_extension(self, populated_db: aiosqlite.Connection) -> None:
        executor = MindQLExecutor(populated_db)
        q = MindQLQuery(
            command=MindQLCommand.EXTENSION,
            extension_name="NOPE",
        )
        with pytest.raises(MindQLParseError, match="Unknown extension"):
            await executor.execute(q)


class TestExecutorColumnValidation:
    """Test that invalid columns are rejected."""

    async def test_disallowed_column(self, populated_db: aiosqlite.Connection) -> None:
        executor = MindQLExecutor(populated_db)
        q = MindQLQuery(
            command=MindQLCommand.FIND,
            table="thought",
            conditions=[
                Condition(field="nonexistent_col", operator=MindQLOperator.EQ, value="x"),
            ],
        )
        with pytest.raises(MindQLParseError, match="not allowed"):
            await executor.execute(q)


# ---------------------------------------------------------------------------
# Temporal-predicate parsing
# ---------------------------------------------------------------------------


class TestParserTemporalPredicates:
    """Parsing of the four valid-time WHERE predicates."""

    def test_valid_now_no_args(self) -> None:
        q = parse("FIND thoughts WHERE valid_now")
        assert q.conditions == []
        assert len(q.temporal_predicates) == 1
        pred = q.temporal_predicates[0]
        assert pred.kind == TemporalPredicateKind.VALID_NOW
        assert pred.start is None
        assert pred.end is None

    def test_valid_at_one_arg(self) -> None:
        q = parse(f"FIND thoughts WHERE valid_at '{_T_JAN}'")
        pred = q.temporal_predicates[0]
        assert pred.kind == TemporalPredicateKind.VALID_AT
        assert pred.start == _T_JAN
        assert pred.end is None

    def test_valid_within_two_args(self) -> None:
        q = parse(f"FIND edges WHERE valid_within '{_T_JAN}' '{_T_JUN}'")
        pred = q.temporal_predicates[0]
        assert pred.kind == TemporalPredicateKind.VALID_WITHIN
        assert pred.start == _T_JAN
        assert pred.end == _T_JUN

    def test_valid_between_two_args(self) -> None:
        q = parse(f"FIND edges WHERE valid_between '{_T_JAN}' '{_T_DEC}'")
        pred = q.temporal_predicates[0]
        assert pred.kind == TemporalPredicateKind.VALID_BETWEEN
        assert pred.start == _T_JAN
        assert pred.end == _T_DEC

    def test_bare_unquoted_timestamp_arg(self) -> None:
        # Whitespace-free ISO-8601 tokens are accepted without quotes.
        q = parse(f"FIND thoughts WHERE valid_at {_T_JAN}")
        assert q.temporal_predicates[0].start == _T_JAN

    def test_case_insensitive_keyword(self) -> None:
        q = parse("FIND thoughts WHERE VALID_NOW")
        assert q.temporal_predicates[0].kind == TemporalPredicateKind.VALID_NOW

    def test_composable_with_condition(self) -> None:
        q = parse("FIND thoughts WHERE priority = 'P1' AND valid_now")
        assert len(q.conditions) == 1
        assert q.conditions[0].field == "priority"
        assert len(q.temporal_predicates) == 1
        assert q.temporal_predicates[0].kind == TemporalPredicateKind.VALID_NOW

    def test_condition_after_temporal(self) -> None:
        q = parse(f"FIND thoughts WHERE valid_at '{_T_JAN}' AND priority = 'P1'")
        assert len(q.conditions) == 1
        assert len(q.temporal_predicates) == 1

    def test_two_temporal_predicates(self) -> None:
        q = parse(f"FIND thoughts WHERE valid_now AND valid_at '{_T_JAN}'")
        assert len(q.temporal_predicates) == 2

    def test_timezone_normalised_to_utc(self) -> None:
        # A +02:00 offset is normalised to the equivalent UTC instant so
        # that lexicographic TEXT comparison in SQLite stays correct.
        q = parse("FIND thoughts WHERE valid_at '2025-01-01T02:00:00+02:00'")
        assert q.temporal_predicates[0].start == "2025-01-01T00:00:00+00:00"

    def test_normal_query_has_no_temporal_predicates(self) -> None:
        q = parse("FIND thoughts WHERE priority = 'P1'")
        assert q.temporal_predicates == []


class TestParserTemporalPredicateErrors:
    """Malformed temporal predicates raise a generic, public-safe error."""

    def test_valid_at_missing_arg(self) -> None:
        with pytest.raises(MindQLParseError, match="Malformed temporal predicate"):
            parse("FIND thoughts WHERE valid_at")

    def test_valid_within_missing_second_arg(self) -> None:
        with pytest.raises(MindQLParseError, match="Malformed temporal predicate"):
            parse(f"FIND thoughts WHERE valid_within '{_T_JAN}'")

    def test_valid_between_missing_args(self) -> None:
        with pytest.raises(MindQLParseError, match="Malformed temporal predicate"):
            parse("FIND thoughts WHERE valid_between")

    def test_valid_now_with_surplus_arg(self) -> None:
        with pytest.raises(MindQLParseError, match="Malformed temporal predicate"):
            parse(f"FIND thoughts WHERE valid_now '{_T_JAN}'")

    def test_valid_at_surplus_arg(self) -> None:
        with pytest.raises(MindQLParseError, match="Malformed temporal predicate"):
            parse(f"FIND thoughts WHERE valid_at '{_T_JAN}' '{_T_JUN}'")

    def test_non_iso_timestamp_rejected(self) -> None:
        with pytest.raises(MindQLParseError, match="Malformed temporal predicate"):
            parse("FIND thoughts WHERE valid_at 'not-a-date'")

    def test_empty_quoted_arg_rejected(self) -> None:
        with pytest.raises(MindQLParseError, match="Malformed temporal predicate"):
            parse("FIND thoughts WHERE valid_at ''")

    def test_error_message_does_not_leak_keyword_list(self) -> None:
        # The public-safe message must not enumerate the supported keywords.
        with pytest.raises(MindQLParseError) as exc_info:
            parse("FIND thoughts WHERE valid_at")
        message = str(exc_info.value)
        for keyword in ("valid_now", "valid_at", "valid_within", "valid_between"):
            assert keyword not in message


# ---------------------------------------------------------------------------
# Temporal-predicate execution (both thought and edge tables)
# ---------------------------------------------------------------------------


async def _find_ids(
    store: SqliteEngravaCore,
    table: str,
    where: str,
    id_column: str,
) -> set[str]:
    """Run a FIND on ``table`` with ``where`` and collect the id column.

    The ``wire-`` namespace (anchor / target thoughts used only to satisfy
    edge referential integrity) is filtered out so thought-table assertions
    reason solely about the five valid-time shape rows.
    """
    result = await store.execute_mindql(parse(f"FIND {table} WHERE {where}"))
    return {row[id_column] for row in result.rows if "wire-" not in row[id_column]}


# Each table is exercised through the same matrix: (table, id column, id prefix).
_TEMPORAL_TABLE_CASES = [
    ("thoughts", "thought_id", "t-"),
    ("edges", "edge_id", "e-"),
]


class TestExecutorTemporalThoughtAndEdge:
    """Temporal predicates filter both the thought and edge tables."""

    @pytest.mark.parametrize(("table", "id_column", "prefix"), _TEMPORAL_TABLE_CASES)
    async def test_valid_at_selects_in_window(
        self,
        temporal_db: aiosqlite.Connection,
        table: str,
        id_column: str,
        prefix: str,
    ) -> None:
        store = SqliteEngravaCore(temporal_db)
        ids = await _find_ids(store, table, f"valid_at '{_T_MID}'", id_column)
        # MID is inside [JAN, JUN), so bounded + both open variants match;
        # the open-bound rows (open-from, open-until, legacy) are NULL-tolerant;
        # the future row (begins AFTER) is excluded.
        assert ids == {
            f"{prefix}bounded",
            f"{prefix}open-from",
            f"{prefix}open-until",
            f"{prefix}legacy",
        }

    @pytest.mark.parametrize(("table", "id_column", "prefix"), _TEMPORAL_TABLE_CASES)
    async def test_valid_at_before_window_excludes_bounded(
        self,
        temporal_db: aiosqlite.Connection,
        table: str,
        id_column: str,
        prefix: str,
    ) -> None:
        store = SqliteEngravaCore(temporal_db)
        ids = await _find_ids(store, table, f"valid_at '{_T_BEFORE}'", id_column)
        # BEFORE JAN: the bounded and open-until rows (valid_from JAN) start
        # later and are excluded; open lower bounds (open-from, legacy) match.
        assert ids == {f"{prefix}open-from", f"{prefix}legacy"}

    @pytest.mark.parametrize(("table", "id_column", "prefix"), _TEMPORAL_TABLE_CASES)
    @pytest.mark.usefixtures("pinned_now")
    async def test_valid_now_excludes_future_and_expired(
        self,
        temporal_db: aiosqlite.Connection,
        table: str,
        id_column: str,
        prefix: str,
    ) -> None:
        # pinned_now fixes "now" to MID (inside [JAN, JUN)).
        store = SqliteEngravaCore(temporal_db)
        ids = await _find_ids(store, table, "valid_now", id_column)
        # Future row (valid_from AFTER) is future-valid → excluded; nothing has
        # expired before MID, so every in/open-window row is returned.
        assert ids == {
            f"{prefix}bounded",
            f"{prefix}open-from",
            f"{prefix}open-until",
            f"{prefix}legacy",
        }
        assert f"{prefix}future" not in ids

    @pytest.mark.parametrize(("table", "id_column", "prefix"), _TEMPORAL_TABLE_CASES)
    async def test_valid_now_excludes_expired_validity(
        self,
        temporal_db: aiosqlite.Connection,
        table: str,
        id_column: str,
        prefix: str,
    ) -> None:
        # Pin "now" to AFTER: the bounded and open-from rows ended at JUN and
        # are now expired; open-until / legacy / future stay valid.
        token = executor_module.mindql_now.set(_T_AFTER)
        try:
            store = SqliteEngravaCore(temporal_db)
            ids = await _find_ids(store, table, "valid_now", id_column)
        finally:
            executor_module.mindql_now.reset(token)
        assert ids == {
            f"{prefix}open-until",
            f"{prefix}legacy",
            f"{prefix}future",
        }

    @pytest.mark.parametrize(("table", "id_column", "prefix"), _TEMPORAL_TABLE_CASES)
    async def test_valid_within_overlap(
        self,
        temporal_db: aiosqlite.Connection,
        table: str,
        id_column: str,
        prefix: str,
    ) -> None:
        store = SqliteEngravaCore(temporal_db)
        # Probe window [FEB, MID) overlaps every row whose interval intersects
        # it; the future row begins at AFTER and does not overlap.
        ids = await _find_ids(
            store,
            table,
            f"valid_within '{_T_FEB}' '{_T_MID}'",
            id_column,
        )
        assert ids == {
            f"{prefix}bounded",
            f"{prefix}open-from",
            f"{prefix}open-until",
            f"{prefix}legacy",
        }
        assert f"{prefix}future" not in ids

    @pytest.mark.parametrize(("table", "id_column", "prefix"), _TEMPORAL_TABLE_CASES)
    async def test_valid_between_containment_requires_real_bounds(
        self,
        temporal_db: aiosqlite.Connection,
        table: str,
        id_column: str,
        prefix: str,
    ) -> None:
        store = SqliteEngravaCore(temporal_db)
        # Containment window [JAN, DEC] fully contains the bounded row only;
        # every open-bounded row (NULL on either end) is correctly excluded.
        ids = await _find_ids(
            store,
            table,
            f"valid_between '{_T_JAN}' '{_T_DEC}'",
            id_column,
        )
        assert ids == {f"{prefix}bounded"}

    @pytest.mark.parametrize(("table", "id_column", "prefix"), _TEMPORAL_TABLE_CASES)
    async def test_null_tolerance_open_and_legacy_rows(
        self,
        temporal_db: aiosqlite.Connection,
        table: str,
        id_column: str,
        prefix: str,
    ) -> None:
        # The key NULL-tolerance guarantee: a row with NULL valid_from (legacy
        # thought / existing edge) is RETURNED by valid_at / valid_now /
        # valid_within and EXCLUDED by valid_between.
        store = SqliteEngravaCore(temporal_db)
        legacy = f"{prefix}legacy"

        at_ids = await _find_ids(store, table, f"valid_at '{_T_MID}'", id_column)
        assert legacy in at_ids

        token = executor_module.mindql_now.set(_T_MID)
        try:
            now_ids = await _find_ids(store, table, "valid_now", id_column)
        finally:
            executor_module.mindql_now.reset(token)
        assert legacy in now_ids

        within_ids = await _find_ids(
            store,
            table,
            f"valid_within '{_T_FEB}' '{_T_MID}'",
            id_column,
        )
        assert legacy in within_ids

        between_ids = await _find_ids(
            store,
            table,
            f"valid_between '{_T_JAN}' '{_T_DEC}'",
            id_column,
        )
        assert legacy not in between_ids

    @pytest.mark.parametrize(("table", "id_column", "prefix"), _TEMPORAL_TABLE_CASES)
    async def test_temporal_composes_with_column_condition(
        self,
        temporal_db: aiosqlite.Connection,
        table: str,
        id_column: str,
        prefix: str,
    ) -> None:
        store = SqliteEngravaCore(temporal_db)
        # A column condition narrows the temporal result. All seeded rows are
        # created_cycle = 1, so the AND-ed condition keeps the same set.
        ids = await _find_ids(
            store,
            table,
            f"created_cycle = 1 AND valid_at '{_T_MID}'",
            id_column,
        )
        assert ids == {
            f"{prefix}bounded",
            f"{prefix}open-from",
            f"{prefix}open-until",
            f"{prefix}legacy",
        }


class TestExecutorTemporalCount:
    """COUNT honours temporal predicates through the shared WHERE builder."""

    async def test_count_valid_between(self, temporal_db: aiosqlite.Connection) -> None:
        store = SqliteEngravaCore(temporal_db)
        result = await store.execute_mindql(
            parse(f"COUNT thoughts WHERE valid_between '{_T_JAN}' '{_T_DEC}'"),
        )
        # Only the single fully-bounded thought is contained.
        assert result.count == 1

    @pytest.mark.usefixtures("pinned_now")
    async def test_count_valid_now(self, temporal_db: aiosqlite.Connection) -> None:
        store = SqliteEngravaCore(temporal_db)
        result = await store.execute_mindql(parse("COUNT edges WHERE valid_now"))
        # bounded + open-from + open-until + legacy = 4 (future excluded).
        assert result.count == 4


class TestExecutorTemporalClockInjection:
    """The injectable clock makes ``valid_now`` deterministic."""

    async def test_pinned_instant_is_used(
        self,
        temporal_db: aiosqlite.Connection,
    ) -> None:
        store = SqliteEngravaCore(temporal_db)
        # Pin before the bounded window opens: the bounded row is not yet valid.
        token = executor_module.mindql_now.set(_T_BEFORE)
        try:
            ids = await _find_ids(store, "thoughts", "valid_now", "thought_id")
        finally:
            executor_module.mindql_now.reset(token)
        assert "t-bounded" not in ids
        # Open lower bounds remain valid even before the bounded window.
        assert "t-open-from" in ids
        assert "t-legacy" in ids

    async def test_default_clock_uses_server_now(
        self,
        temporal_db: aiosqlite.Connection,
    ) -> None:
        # With no pinned instant, valid_now resolves against the real clock.
        # Every non-future row's window covers "now" (all created this run),
        # except the bounded / open-from rows which already ended at JUN 2025.
        assert executor_module.mindql_now.get() is None
        store = SqliteEngravaCore(temporal_db)
        ids = await _find_ids(store, "thoughts", "valid_now", "thought_id")
        # The future row begins in 2026 — whether it is valid depends on the
        # real date — so only assert the open-ended rows are present and the
        # already-expired bounded window is absent.
        assert "t-open-until" in ids
        assert "t-legacy" in ids
        assert "t-bounded" not in ids


class TestExecutorTemporalTableGuard:
    """Temporal predicates are rejected on tables without valid-time columns."""

    async def test_rejected_on_action_table(
        self,
        populated_db: aiosqlite.Connection,
    ) -> None:
        store = SqliteEngravaCore(populated_db)
        query = MindQLQuery(
            command=MindQLCommand.FIND,
            table="action",
            temporal_predicates=[TemporalPredicate(kind=TemporalPredicateKind.VALID_NOW)],
        )
        with pytest.raises(MindQLParseError, match="not supported for table"):
            await store.execute_mindql(query)


class TestBackwardCompatibility:
    """A query without a temporal predicate behaves exactly as before."""

    async def test_find_without_temporal_unchanged(
        self,
        populated_db: aiosqlite.Connection,
    ) -> None:
        store = SqliteEngravaCore(populated_db)
        before = await store.execute_mindql(parse("FIND thoughts WHERE priority = 'P1'"))
        # No temporal predicate → the historical row set is returned verbatim.
        assert {row["thought_id"] for row in before.rows} == {"t-000", "t-001"}

    async def test_find_all_without_temporal_unchanged(
        self,
        populated_db: aiosqlite.Connection,
    ) -> None:
        store = SqliteEngravaCore(populated_db)
        result = await store.execute_mindql(parse("FIND thoughts"))
        assert len(result.rows) == 5
