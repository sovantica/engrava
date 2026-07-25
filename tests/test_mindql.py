"""Tests for MindQL parser and executor.

Covers parsing of FIND, COUNT, SELECT, and extension commands,
as well as executor integration with aiosqlite.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, SupportsIndex, cast

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
from engrava.mindql import parser as parser_module
from engrava.mindql.executor import MindQLExecutor
from engrava.mindql.parser import (
    BoolExpr,
    Comparison,
    Condition,
    InCondition,
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


# ---------------------------------------------------------------------------
# Quoted-value typing: a single-quoted literal stays a string verbatim
# ---------------------------------------------------------------------------


class TestQuotedValueStaysString:
    """A single-quoted WHERE value must keep its string type.

    Only an *unquoted* bare value is coerced to ``int`` / ``float``; a value
    the user wrapped in single quotes is taken verbatim as a string, so a
    zero-padded identifier such as ``'007'`` is never silently turned into the
    integer ``7``.
    """

    def test_quoted_numeric_value_is_string(self) -> None:
        q = parse("FIND thoughts WHERE source = '12'")
        assert q.conditions[0].value == "12"
        assert isinstance(q.conditions[0].value, str)

    def test_unquoted_numeric_value_still_coerces(self) -> None:
        q = parse("FIND thoughts WHERE source = 12")
        assert q.conditions[0].value == 12
        assert isinstance(q.conditions[0].value, int)

    def test_quoted_zero_padded_value_is_string(self) -> None:
        q = parse("FIND thoughts WHERE source = '007'")
        assert q.conditions[0].value == "007"
        assert isinstance(q.conditions[0].value, str)

    async def test_quoted_zero_padded_value_matches_stored_string(
        self,
        db: aiosqlite.Connection,
    ) -> None:
        store = SqliteEngravaCore(db)
        await store.create_thought(
            ThoughtRecord(
                thought_id="t-zero-pad",
                thought_type=ThoughtType.OBSERVATION,
                essence="zero padded source",
                content="zero padded source",
                priority=Priority.P1,
                lifecycle_status=LifecycleStatus.ACTIVE,
                created_cycle=1,
                updated_cycle=1,
                source="007",
            )
        )
        result = await store.execute_mindql(parse("FIND thoughts WHERE source = '007'"))
        # Pre-fix: '007' was coerced to int 7 and never matched the stored
        # string '007', so this returned no rows.
        assert {row["thought_id"] for row in result.rows} == {"t-zero-pad"}


# ---------------------------------------------------------------------------
# Strict condition matching: a fragment with trailing content is rejected
# ---------------------------------------------------------------------------


class TestConditionFullMatch:
    """A WHERE fragment must match an operand grammar in full.

    A prefix match used to silently discard any trailing content after the
    first ``field op value`` token. On the flat (pure-``AND``) path each
    fragment is still matched in full. ``OR`` is now a first-class operator, so
    ``priority = 'P1' OR 1=1`` parses as a boolean tree — but the injected
    ``1=1`` operand names the non-column ``1``, which the per-table allowlist
    rejects when the query runs, so the surplus can never quietly change the
    result set.
    """

    def test_trailing_injection_operand_rejected_at_execution(self) -> None:
        # ``OR 1=1`` is valid grammar now, but ``1`` is not an allowlisted
        # column, so execution rejects it rather than widening the result set.
        q = parse("FIND thoughts WHERE priority = 'P1' OR 1=1")
        assert isinstance(q.where, BoolExpr)
        assert q.where.op == "OR"

    async def test_trailing_injection_operand_rejected_when_run(
        self,
        populated_db: aiosqlite.Connection,
    ) -> None:
        executor = MindQLExecutor(populated_db)
        q = parse("FIND thoughts WHERE priority = 'P1' OR 1=1")
        with pytest.raises(MindQLParseError, match="not allowed"):
            await executor.execute(q)

    def test_clean_single_condition_still_parses(self) -> None:
        q = parse("FIND thoughts WHERE priority = 'P1'")
        assert len(q.conditions) == 1
        assert q.conditions[0].field == "priority"
        assert q.conditions[0].value == "P1"

    def test_clean_and_split_conditions_still_parse(self) -> None:
        q = parse("FIND thoughts WHERE source = 'x' AND priority = 'P1'")
        assert len(q.conditions) == 2
        assert q.conditions[0].field == "source"
        assert q.conditions[0].value == "x"
        assert q.conditions[1].field == "priority"
        assert q.conditions[1].value == "P1"


# ---------------------------------------------------------------------------
# Default LIMIT: an unbounded FIND is capped; an explicit LIMIT still wins
# ---------------------------------------------------------------------------


class TestDefaultFindLimit:
    """A FIND with no LIMIT is capped at the default; COUNT is unaffected."""

    async def test_find_without_limit_capped_at_default(
        self,
        db: aiosqlite.Connection,
    ) -> None:
        default = executor_module.DEFAULT_FIND_LIMIT
        store = SqliteEngravaCore(db)
        for i in range(default + 10):
            await store.create_thought(
                ThoughtRecord(
                    thought_id=f"t-cap-{i:04d}",
                    thought_type=ThoughtType.OBSERVATION,
                    essence=f"essence {i}",
                    content=f"content {i}",
                    priority=Priority.P1,
                    lifecycle_status=LifecycleStatus.ACTIVE,
                    created_cycle=1,
                    updated_cycle=1,
                    source="cap",
                )
            )
        result = await store.execute_mindql(parse("FIND thoughts"))
        # Pre-fix: the query was unbounded and returned every stored row.
        assert len(result.rows) == default

    async def test_explicit_limit_still_wins(
        self,
        db: aiosqlite.Connection,
    ) -> None:
        store = SqliteEngravaCore(db)
        for i in range(20):
            await store.create_thought(
                ThoughtRecord(
                    thought_id=f"t-lim-{i:04d}",
                    thought_type=ThoughtType.OBSERVATION,
                    essence=f"essence {i}",
                    content=f"content {i}",
                    priority=Priority.P1,
                    lifecycle_status=LifecycleStatus.ACTIVE,
                    created_cycle=1,
                    updated_cycle=1,
                    source="lim",
                )
            )
        result = await store.execute_mindql(parse("FIND thoughts LIMIT 5"))
        assert len(result.rows) == 5

    async def test_count_unaffected_by_default_limit(
        self,
        db: aiosqlite.Connection,
    ) -> None:
        default = executor_module.DEFAULT_FIND_LIMIT
        store = SqliteEngravaCore(db)
        total = default + 10
        for i in range(total):
            await store.create_thought(
                ThoughtRecord(
                    thought_id=f"t-cnt-{i:04d}",
                    thought_type=ThoughtType.OBSERVATION,
                    essence=f"essence {i}",
                    content=f"content {i}",
                    priority=Priority.P1,
                    lifecycle_status=LifecycleStatus.ACTIVE,
                    created_cycle=1,
                    updated_cycle=1,
                    source="cnt",
                )
            )
        result = await store.execute_mindql(parse("COUNT thoughts"))
        # COUNT does not apply the FIND default cap.
        assert result.count == total


# ---------------------------------------------------------------------------
# IN operator
# ---------------------------------------------------------------------------


class TestParserIn:
    """Parsing of the ``field IN (v1, v2, …)`` operator."""

    def test_in_quoted_values(self) -> None:
        q = parse("FIND thoughts WHERE thought_type IN ('BELIEF', 'OBSERVATION')")
        assert isinstance(q.where, InCondition)
        assert q.where.field == "thought_type"
        assert q.where.values == ("BELIEF", "OBSERVATION")
        # The flat lists stay empty on the tree path.
        assert q.conditions == []
        assert q.temporal_predicates == []

    def test_in_unquoted_values_coerced(self) -> None:
        q = parse("FIND thoughts WHERE created_cycle IN (1, 2, 3)")
        assert isinstance(q.where, InCondition)
        assert q.where.values == (1, 2, 3)
        assert all(isinstance(v, int) for v in q.where.values)

    def test_in_quoted_value_stays_string(self) -> None:
        q = parse("FIND thoughts WHERE source IN ('007')")
        assert isinstance(q.where, InCondition)
        assert q.where.values == ("007",)
        assert isinstance(q.where.values[0], str)

    def test_in_single_value(self) -> None:
        q = parse("FIND thoughts WHERE priority IN ('P1')")
        assert isinstance(q.where, InCondition)
        assert q.where.values == ("P1",)

    def test_empty_in_rejected(self) -> None:
        with pytest.raises(MindQLParseError, match="IN requires at least one value"):
            parse("FIND thoughts WHERE priority IN ()")

    def test_in_case_insensitive_keyword(self) -> None:
        q = parse("FIND thoughts WHERE priority in ('P1', 'P2')")
        assert isinstance(q.where, InCondition)
        assert q.where.values == ("P1", "P2")


class TestExecutorIn:
    """Execution of the ``IN`` operator against a real database."""

    async def test_in_selects_matching_rows(
        self,
        populated_db: aiosqlite.Connection,
    ) -> None:
        executor = MindQLExecutor(populated_db)
        q = parse("FIND thoughts WHERE thought_id IN ('t-000', 't-002')")
        result = await executor.execute(q)
        assert {row["thought_id"] for row in result.rows} == {"t-000", "t-002"}

    async def test_in_binds_every_value(
        self,
        populated_db: aiosqlite.Connection,
    ) -> None:
        # EXPLAIN exposes the compiled SQL: every IN value must be a bound ``?``.
        executor = MindQLExecutor(populated_db)
        q = parse("EXPLAIN FIND thoughts WHERE thought_id IN ('t-000', 't-002')")
        result = await executor.execute(q)
        sql = result.rows[0]["sql"]
        assert "thought_id IN (?, ?)" in sql
        assert result.rows[0]["params"][:2] == ["t-000", "t-002"]

    async def test_in_disallowed_column_rejected(
        self,
        populated_db: aiosqlite.Connection,
    ) -> None:
        executor = MindQLExecutor(populated_db)
        q = parse("FIND thoughts WHERE bogus_col IN ('x')")
        with pytest.raises(MindQLParseError, match="not allowed"):
            await executor.execute(q)


# ---------------------------------------------------------------------------
# OR / AND precedence and parentheses
# ---------------------------------------------------------------------------


class TestParserBooleanExpression:
    """Parsing of OR / AND precedence and parenthesised grouping."""

    def test_and_binds_tighter_than_or(self) -> None:
        # ``a = 1 OR b = 2 AND c = 3`` groups as ``a=1 OR (b=2 AND c=3)``.
        q = parse(
            "FIND thoughts WHERE priority = 'P1' OR lifecycle_status = 'ACTIVE' AND source = 'x'"
        )
        assert isinstance(q.where, BoolExpr)
        assert q.where.op == "OR"
        assert len(q.where.operands) == 2
        left, right = q.where.operands
        assert isinstance(left, Comparison)
        assert left.field == "priority"
        assert isinstance(right, BoolExpr)
        assert right.op == "AND"
        assert [op.field for op in right.operands if isinstance(op, Comparison)] == [
            "lifecycle_status",
            "source",
        ]

    def test_parentheses_override_precedence(self) -> None:
        # Explicit parentheses group the OR before the AND.
        q = parse("FIND thoughts WHERE (priority = 'P1' OR priority = 'P2') AND source = 'x'")
        assert isinstance(q.where, BoolExpr)
        assert q.where.op == "AND"
        left, right = q.where.operands
        assert isinstance(left, BoolExpr)
        assert left.op == "OR"
        assert isinstance(right, Comparison)
        assert right.field == "source"

    def test_temporal_predicate_is_an_operand(self) -> None:
        q = parse(f"FIND thoughts WHERE priority = 'P1' OR valid_at '{_T_JAN}'")
        assert isinstance(q.where, BoolExpr)
        assert q.where.op == "OR"
        _, right = q.where.operands
        assert isinstance(right, TemporalPredicate)
        assert right.kind == TemporalPredicateKind.VALID_AT

    def test_in_is_an_operand(self) -> None:
        q = parse(
            "FIND thoughts WHERE priority = 'P1' OR thought_type IN ('BELIEF', 'OBSERVATION')"
        )
        assert isinstance(q.where, BoolExpr)
        _, right = q.where.operands
        assert isinstance(right, InCondition)

    def test_nested_parentheses(self) -> None:
        q = parse("FIND thoughts WHERE ((priority = 'P1'))")
        assert isinstance(q.where, Comparison)
        assert q.where.field == "priority"

    def test_unbalanced_parentheses_rejected(self) -> None:
        with pytest.raises(MindQLParseError, match=r"[Pp]arenthes"):
            parse("FIND thoughts WHERE (priority = 'P1'")

    def test_dangling_or_rejected(self) -> None:
        with pytest.raises(MindQLParseError):
            parse("FIND thoughts WHERE priority = 'P1' OR")


class TestExecutorBooleanExpression:
    """Execution of OR / AND / parenthesised WHERE trees."""

    async def test_or_widens_result(
        self,
        populated_db: aiosqlite.Connection,
    ) -> None:
        executor = MindQLExecutor(populated_db)
        q = parse("FIND thoughts WHERE thought_id = 't-000' OR thought_id = 't-004'")
        result = await executor.execute(q)
        assert {row["thought_id"] for row in result.rows} == {"t-000", "t-004"}

    async def test_precedence_matches_sqlite(
        self,
        populated_db: aiosqlite.Connection,
    ) -> None:
        # priority=P1 rows are t-000,t-001 (both ACTIVE). t-004 is ARCHIVED.
        # ``priority='P1' OR lifecycle_status='ARCHIVED' AND priority='P2'``
        # groups as P1 OR (ARCHIVED AND P2) → {t-000,t-001,t-004}.
        executor = MindQLExecutor(populated_db)
        q = parse(
            "FIND thoughts WHERE priority = 'P1' "
            "OR lifecycle_status = 'ARCHIVED' AND priority = 'P2'"
        )
        result = await executor.execute(q)
        assert {row["thought_id"] for row in result.rows} == {"t-000", "t-001", "t-004"}

    async def test_parentheses_change_result(
        self,
        populated_db: aiosqlite.Connection,
    ) -> None:
        # (P1 OR P2) AND ACTIVE → active rows only (t-000..t-003).
        executor = MindQLExecutor(populated_db)
        q = parse(
            "FIND thoughts WHERE (priority = 'P1' OR priority = 'P2') "
            "AND lifecycle_status = 'ACTIVE'"
        )
        result = await executor.execute(q)
        assert {row["thought_id"] for row in result.rows} == {
            "t-000",
            "t-001",
            "t-002",
            "t-003",
        }

    async def test_tree_temporal_predicate_composes(
        self,
        temporal_db: aiosqlite.Connection,
    ) -> None:
        store = SqliteEngravaCore(temporal_db)
        # ``created_cycle = 1 OR valid_at MID`` — all rows are created_cycle 1,
        # so the OR keeps every row (including the future one, via the left arm).
        result = await store.execute_mindql(
            parse(f"FIND thoughts WHERE created_cycle = 1 OR valid_at '{_T_MID}'")
        )
        ids = {row["thought_id"] for row in result.rows if "wire-" not in row["thought_id"]}
        assert "t-future" in ids


# ---------------------------------------------------------------------------
# ORDER BY
# ---------------------------------------------------------------------------


class TestParserOrderBy:
    """Parsing of the ORDER BY clause (FIND only)."""

    def test_single_field_default_asc(self) -> None:
        q = parse("FIND thoughts ORDER BY created_cycle")
        assert q.order_by == (("created_cycle", "ASC"),)

    def test_single_field_desc(self) -> None:
        q = parse("FIND thoughts ORDER BY created_cycle DESC")
        assert q.order_by == (("created_cycle", "DESC"),)

    def test_multi_field(self) -> None:
        q = parse("FIND thoughts ORDER BY priority ASC, created_cycle DESC")
        assert q.order_by == (("priority", "ASC"), ("created_cycle", "DESC"))

    def test_direction_case_insensitive(self) -> None:
        q = parse("FIND thoughts ORDER BY created_cycle desc")
        assert q.order_by == (("created_cycle", "DESC"),)

    def test_order_by_with_where_and_limit(self) -> None:
        q = parse("FIND thoughts WHERE priority = 'P1' ORDER BY created_cycle DESC LIMIT 3")
        assert q.order_by == (("created_cycle", "DESC"),)
        assert q.limit == 3
        assert len(q.conditions) == 1

    def test_order_by_rejected_on_count(self) -> None:
        with pytest.raises(MindQLParseError, match="ORDER BY is only supported for FIND"):
            parse("COUNT thoughts ORDER BY created_cycle")

    def test_invalid_direction_rejected(self) -> None:
        with pytest.raises(MindQLParseError, match="Invalid ORDER BY direction"):
            parse("FIND thoughts ORDER BY created_cycle SIDEWAYS")


class TestExecutorOrderBy:
    """Execution of ORDER BY, including allowlist enforcement on sort fields."""

    async def test_order_by_desc_sorts_rows(
        self,
        populated_db: aiosqlite.Connection,
    ) -> None:
        executor = MindQLExecutor(populated_db)
        q = parse("FIND thoughts ORDER BY created_cycle DESC")
        result = await executor.execute(q)
        cycles = [row["created_cycle"] for row in result.rows]
        assert cycles == sorted(cycles, reverse=True)

    async def test_order_by_multi_field(
        self,
        populated_db: aiosqlite.Connection,
    ) -> None:
        executor = MindQLExecutor(populated_db)
        q = parse("FIND thoughts ORDER BY priority ASC, created_cycle DESC")
        result = await executor.execute(q)
        # Sort key emitted before LIMIT/OFFSET.
        explain = await executor.execute(
            parse("EXPLAIN FIND thoughts ORDER BY priority ASC, created_cycle DESC")
        )
        sql = explain.rows[0]["sql"]
        assert "ORDER BY priority ASC, created_cycle DESC" in sql
        assert sql.index("ORDER BY") < sql.index("LIMIT")
        assert len(result.rows) == 5

    async def test_order_by_non_allowlisted_field_rejected(
        self,
        populated_db: aiosqlite.Connection,
    ) -> None:
        executor = MindQLExecutor(populated_db)
        q = parse("FIND thoughts ORDER BY bogus_col")
        with pytest.raises(MindQLParseError, match="not allowed"):
            await executor.execute(q)


# ---------------------------------------------------------------------------
# OFFSET
# ---------------------------------------------------------------------------


class TestParserOffset:
    """Parsing of the OFFSET clause (FIND only)."""

    def test_offset_with_limit(self) -> None:
        q = parse("FIND thoughts LIMIT 2 OFFSET 1")
        assert q.limit == 2
        assert q.offset == 1

    def test_offset_without_limit(self) -> None:
        q = parse("FIND thoughts OFFSET 3")
        assert q.offset == 3
        assert q.limit is None

    def test_offset_rejected_on_count(self) -> None:
        with pytest.raises(MindQLParseError, match="OFFSET is only supported for FIND"):
            parse("COUNT thoughts OFFSET 1")


class TestExecutorOffset:
    """Execution of OFFSET, including the default-LIMIT fallback."""

    async def test_offset_skips_rows(
        self,
        populated_db: aiosqlite.Connection,
    ) -> None:
        executor = MindQLExecutor(populated_db)
        full = await executor.execute(parse("FIND thoughts ORDER BY thought_id ASC"))
        paged = await executor.execute(
            parse("FIND thoughts ORDER BY thought_id ASC LIMIT 2 OFFSET 2")
        )
        assert [row["thought_id"] for row in paged.rows] == [
            full.rows[2]["thought_id"],
            full.rows[3]["thought_id"],
        ]

    async def test_offset_without_limit_uses_default_cap(
        self,
        populated_db: aiosqlite.Connection,
    ) -> None:
        # SQLite requires LIMIT for OFFSET → default cap is applied as LIMIT.
        executor = MindQLExecutor(populated_db)
        explain = await executor.execute(parse("EXPLAIN FIND thoughts OFFSET 1"))
        sql = explain.rows[0]["sql"]
        assert f"LIMIT {executor_module.DEFAULT_FIND_LIMIT} OFFSET 1" in sql
        result = await executor.execute(parse("FIND thoughts OFFSET 1"))
        assert len(result.rows) == 4


# ---------------------------------------------------------------------------
# EXPLAIN prefix — compiles the plan and never executes it
# ---------------------------------------------------------------------------


class TestParserExplain:
    """Parsing of the EXPLAIN prefix."""

    def test_explain_find(self) -> None:
        q = parse("EXPLAIN FIND thoughts WHERE priority = 'P1'")
        assert q.explain is True
        assert q.command == MindQLCommand.FIND

    def test_explain_count(self) -> None:
        q = parse("EXPLAIN COUNT thoughts")
        assert q.explain is True
        assert q.command == MindQLCommand.COUNT

    def test_explain_select(self) -> None:
        q = parse("EXPLAIN SELECT thought_id FROM thought")
        assert q.explain is True
        assert q.command == MindQLCommand.SELECT

    def test_explain_case_insensitive(self) -> None:
        q = parse("explain find thoughts")
        assert q.explain is True

    def test_explain_alone_rejected(self) -> None:
        with pytest.raises(MindQLParseError, match="EXPLAIN requires a query"):
            parse("EXPLAIN")

    def test_explain_extension_rejected(self) -> None:
        # A plain extension command parses fine; only EXPLAIN + extension fails.
        parse("PING x", known_extensions={"PING"})
        with pytest.raises(MindQLParseError, match="EXPLAIN is only supported"):
            parse("EXPLAIN PING x", known_extensions={"PING"})


class TestExecutorExplain:
    """EXPLAIN returns the compiled plan and NEVER executes."""

    async def test_explain_find_returns_plan(
        self,
        populated_db: aiosqlite.Connection,
    ) -> None:
        executor = MindQLExecutor(populated_db)
        result = await executor.execute(parse("EXPLAIN FIND thoughts WHERE priority = 'P1'"))
        assert result.command == "EXPLAIN"
        assert result.columns == ["sql", "params"]
        row = result.rows[0]
        assert row["sql"].startswith("SELECT * FROM thought WHERE priority = ?")
        assert row["params"] == ["P1"]

    async def test_explain_count_returns_plan(
        self,
        populated_db: aiosqlite.Connection,
    ) -> None:
        executor = MindQLExecutor(populated_db)
        result = await executor.execute(parse("EXPLAIN COUNT thoughts WHERE priority = 'P1'"))
        row = result.rows[0]
        assert "SELECT COUNT(*)" in row["sql"]
        assert row["params"] == ["P1"]

    async def test_explain_select_returns_plan(
        self,
        populated_db: aiosqlite.Connection,
    ) -> None:
        executor = MindQLExecutor(populated_db)
        result = await executor.execute(parse("EXPLAIN SELECT thought_id FROM thought"))
        row = result.rows[0]
        assert row["sql"] == "SELECT thought_id FROM thought"
        assert row["params"] == []

    async def test_explain_never_executes_side_effecting_select(
        self,
        populated_db: aiosqlite.Connection,
    ) -> None:
        # The load-bearing safety invariant: EXPLAIN compiles and returns the
        # plan WITHOUT touching the DB. An invalid / side-effecting SQL string
        # would raise or mutate if executed — under EXPLAIN it does neither.
        executor = MindQLExecutor(populated_db)
        q = MindQLQuery(
            command=MindQLCommand.SELECT,
            raw_sql="SELECT this is not valid sql at all",
            explain=True,
        )
        result = await executor.execute(q)
        # No execution → no error, plan returned verbatim.
        assert result.command == "EXPLAIN"
        assert result.rows[0]["sql"] == "SELECT this is not valid sql at all"

    async def test_explain_does_not_call_db_execute(
        self,
        populated_db: aiosqlite.Connection,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Spy: db.execute must never be called on an EXPLAIN path.
        executor = MindQLExecutor(populated_db)
        called = False
        original_execute = populated_db.execute

        async def _spy(*args: object, **kwargs: object) -> object:
            nonlocal called
            called = True
            return await original_execute(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(populated_db, "execute", _spy)
        await executor.execute(parse("EXPLAIN FIND thoughts WHERE priority = 'P1'"))
        await executor.execute(parse("EXPLAIN COUNT thoughts"))
        await executor.execute(parse("EXPLAIN SELECT thought_id FROM thought"))
        assert called is False

    async def test_explain_still_validates_columns(
        self,
        populated_db: aiosqlite.Connection,
    ) -> None:
        # Compilation still enforces the allowlist, so an EXPLAIN of a bad
        # column raises at compile time (without executing).
        executor = MindQLExecutor(populated_db)
        q = MindQLQuery(
            command=MindQLCommand.FIND,
            table="thought",
            explain=True,
            conditions=[
                Condition(field="bogus", operator=MindQLOperator.EQ, value="x"),
            ],
        )
        with pytest.raises(MindQLParseError, match="not allowed"):
            await executor.execute(q)


# ---------------------------------------------------------------------------
# Parametrized SELECT passthrough + multi-statement hardening
# ---------------------------------------------------------------------------


class TestExecutorSelectParams:
    """Bound parameters and multi-statement hardening on SELECT passthrough."""

    async def test_select_with_bound_params(
        self,
        populated_db: aiosqlite.Connection,
    ) -> None:
        executor = MindQLExecutor(populated_db)
        q = MindQLQuery(
            command=MindQLCommand.SELECT,
            raw_sql="SELECT thought_id FROM thought WHERE priority = ?",
            select_params=("P1",),
        )
        result = await executor.execute(q)
        assert {row["thought_id"] for row in result.rows} == {"t-000", "t-001"}

    async def test_select_without_params_unchanged(
        self,
        populated_db: aiosqlite.Connection,
    ) -> None:
        # select_params=None behaves exactly as before (no bound params).
        executor = MindQLExecutor(populated_db)
        q = parse("SELECT thought_id FROM thought WHERE lifecycle_status = 'ACTIVE'")
        assert q.select_params is None
        result = await executor.execute(q)
        assert len(result.rows) == 4

    async def test_trailing_semicolon_tolerated(
        self,
        populated_db: aiosqlite.Connection,
    ) -> None:
        executor = MindQLExecutor(populated_db)
        q = MindQLQuery(
            command=MindQLCommand.SELECT,
            raw_sql="SELECT thought_id FROM thought;",
        )
        result = await executor.execute(q)
        assert len(result.rows) == 5

    async def test_multi_statement_rejected(
        self,
        populated_db: aiosqlite.Connection,
    ) -> None:
        executor = MindQLExecutor(populated_db)
        q = MindQLQuery(
            command=MindQLCommand.SELECT,
            raw_sql="SELECT thought_id FROM thought; DROP TABLE thought",
        )
        with pytest.raises(MindQLParseError, match="single SELECT statement"):
            await executor.execute(q)

    async def test_non_select_still_rejected(
        self,
        populated_db: aiosqlite.Connection,
    ) -> None:
        executor = MindQLExecutor(populated_db)
        q = MindQLQuery(command=MindQLCommand.SELECT, raw_sql="DELETE FROM thought")
        with pytest.raises(MindQLParseError, match="Only SELECT"):
            await executor.execute(q)


# ---------------------------------------------------------------------------
# Zero-regression: the pure-AND path leaves where=None and compiles identically
# ---------------------------------------------------------------------------


class TestZeroRegressionFlatPath:
    """A pure-AND WHERE must keep the historical flat path (``where`` is None)."""

    def test_simple_comparison_leaves_where_none(self) -> None:
        q = parse("FIND thoughts WHERE priority = 'P1'")
        assert q.where is None
        assert len(q.conditions) == 1

    def test_pure_and_leaves_where_none(self) -> None:
        q = parse("FIND thoughts WHERE priority = 'P1' AND source = 'x'")
        assert q.where is None
        assert len(q.conditions) == 2

    def test_temporal_pure_and_leaves_where_none(self) -> None:
        q = parse(f"FIND thoughts WHERE priority = 'P1' AND valid_at '{_T_JAN}'")
        assert q.where is None
        assert len(q.conditions) == 1
        assert len(q.temporal_predicates) == 1

    async def test_flat_temporal_sql_identical_to_pre_feature(
        self,
        populated_db: aiosqlite.Connection,
    ) -> None:
        # The compiled SQL for a pure-AND temporal query is byte-identical to
        # what the historical flat builder produced (conditions then temporal).
        executor = MindQLExecutor(populated_db)
        explain = await executor.execute(
            parse(f"EXPLAIN FIND thoughts WHERE priority = 'P1' AND valid_at '{_T_JAN}'")
        )
        sql = explain.rows[0]["sql"]
        # A literal expected-SQL string for an equality assertion, not query
        # construction — the value binds are ``?`` placeholders.
        expected_sql = (
            "SELECT * FROM thought WHERE priority = ? "  # noqa: S608
            "AND (valid_from IS NULL OR valid_from <= ?) "
            "AND (valid_until IS NULL OR valid_until > ?) "
            f"LIMIT {executor_module.DEFAULT_FIND_LIMIT}"
        )
        assert sql == expected_sql
        assert explain.rows[0]["params"] == ["P1", _T_JAN, _T_JAN]


# ---------------------------------------------------------------------------
# Parser edge cases (error paths and boundary tokenisation)
# ---------------------------------------------------------------------------


class TestParserEdgeCases:
    """Boundary and error paths in the read-surface grammar."""

    def test_empty_where_rejected(self) -> None:
        with pytest.raises(MindQLParseError, match="WHERE requires at least one"):
            parse("FIND thoughts WHERE")

    def test_empty_where_before_limit_rejected(self) -> None:
        with pytest.raises(MindQLParseError, match="WHERE requires at least one"):
            parse("FIND thoughts WHERE LIMIT 5")

    def test_garbage_tail_rejected(self) -> None:
        with pytest.raises(MindQLParseError, match="Expected WHERE, ORDER BY"):
            parse("FIND thoughts BOGUS")

    def test_trailing_and_ignored_on_flat_path(self) -> None:
        # A trailing AND produces an empty fragment that is skipped.
        q = parse("FIND thoughts WHERE priority = 'P1' AND ")
        assert q.where is None
        assert len(q.conditions) == 1

    def test_invalid_order_by_field_rejected(self) -> None:
        with pytest.raises(MindQLParseError, match="Invalid ORDER BY field"):
            parse("FIND thoughts ORDER BY na-me")

    def test_empty_order_by_item_rejected(self) -> None:
        with pytest.raises(MindQLParseError, match="Malformed ORDER BY"):
            parse("FIND thoughts ORDER BY priority, ")

    def test_too_many_order_by_tokens_rejected(self) -> None:
        with pytest.raises(MindQLParseError, match="Malformed ORDER BY item"):
            parse("FIND thoughts ORDER BY priority ASC EXTRA")

    def test_invalid_condition_in_tree_rejected(self) -> None:
        with pytest.raises(MindQLParseError, match="Invalid condition"):
            parse("FIND thoughts WHERE (garbage fragment) OR priority = 'P1'")

    def test_quoted_value_with_boolean_words_not_split(self) -> None:
        # 'A AND OR (B)' inside quotes stays one value on the tree path.
        q = parse("FIND thoughts WHERE essence = 'A AND OR (B)' OR priority = 'P1'")
        assert isinstance(q.where, BoolExpr)
        left = q.where.operands[0]
        assert isinstance(left, Comparison)
        assert left.value == "A AND OR (B)"

    def test_unterminated_string_in_where_rejected(self) -> None:
        with pytest.raises(MindQLParseError, match="Unterminated string"):
            parse("FIND thoughts WHERE essence = 'oops OR priority = 'P1'")

    def test_unterminated_in_list_rejected(self) -> None:
        with pytest.raises(MindQLParseError, match="Unterminated"):
            parse("FIND thoughts WHERE priority IN ('P1'")

    def test_in_with_trailing_comma_rejected(self) -> None:
        with pytest.raises(MindQLParseError, match="Malformed IN value list"):
            parse("FIND thoughts WHERE priority IN ('P1',)")

    def test_in_with_leading_comma_rejected(self) -> None:
        with pytest.raises(MindQLParseError, match="Malformed IN value list"):
            parse("FIND thoughts WHERE priority IN (,'P1')")

    def test_in_with_double_comma_rejected(self) -> None:
        with pytest.raises(MindQLParseError, match="Malformed IN value list"):
            parse("FIND thoughts WHERE priority IN ('P1',,'P2')")

    def test_in_missing_comma_between_values_rejected(self) -> None:
        with pytest.raises(MindQLParseError, match="Malformed IN value list"):
            parse("FIND thoughts WHERE priority IN ('P1' 'P2')")

    def test_in_unmatchable_value_rejected(self) -> None:
        # A stray ``(`` where a value is expected matches no value token.
        with pytest.raises(MindQLParseError, match="Malformed IN value list"):
            parse("FIND thoughts WHERE priority IN ('P1', ()")

    def test_stray_close_paren_rejected(self) -> None:
        with pytest.raises(MindQLParseError, match="Unexpected token"):
            parse("FIND thoughts WHERE priority = 'P1' OR )")

    def test_float_value_coercion(self) -> None:
        q = parse("FIND thoughts WHERE confidence > 0.5")
        assert q.conditions[0].value == 0.5
        assert isinstance(q.conditions[0].value, float)

    def test_in_unquoted_non_numeric_stays_string(self) -> None:
        # A bare, non-numeric IN value coerces to itself (a string).
        q = parse("FIND thoughts WHERE lifecycle_status IN (ACTIVE, ARCHIVED)")
        assert isinstance(q.where, InCondition)
        assert q.where.values == ("ACTIVE", "ARCHIVED")
        assert all(isinstance(v, str) for v in q.where.values)

    def test_in_unterminated_quote_inside_list_rejected(self) -> None:
        with pytest.raises(MindQLParseError, match="Unterminated string"):
            parse("FIND thoughts WHERE priority IN ('P1) OR essence = 'x'")

    def test_bare_quote_operand_tokenised_atomically(self) -> None:
        # An operand beginning with a quote on the tree path must tokenise the
        # quoted run atomically (never splitting on the OR inside it). The
        # resulting fragment is not a valid comparison, so it is rejected as a
        # condition rather than silently mis-split.
        with pytest.raises(MindQLParseError, match="Invalid condition"):
            parse("FIND thoughts WHERE 'a OR b' OR priority = 'P1'")


class TestLexicalScanQuoteAwareness:
    """Grammar tokens inside a string literal must not be mis-scanned.

    ``OR`` / ``IN`` / parentheses inside a value literal must NOT force the
    boolean-tree path (the query is still a single simple comparison → flat
    path, ``where=None``), and a ``;`` inside a SELECT-passthrough literal must
    NOT trip the single-statement guard.
    """

    def test_or_inside_value_literal_stays_flat_path(self) -> None:
        q = parse("FIND thoughts WHERE content = 'A OR B'")
        assert q.where is None
        assert len(q.conditions) == 1
        assert q.conditions[0].value == "A OR B"

    def test_in_and_parens_inside_value_literal_stay_flat_path(self) -> None:
        q = parse("FIND thoughts WHERE content = 'x IN (y)'")
        assert q.where is None
        assert q.conditions[0].value == "x IN (y)"

    async def test_or_inside_literal_matches_correctly(
        self,
        db: aiosqlite.Connection,
    ) -> None:
        store = SqliteEngravaCore(db)
        await store.create_thought(
            ThoughtRecord(
                thought_id="lit-or",
                thought_type=ThoughtType.OBSERVATION,
                essence="e",
                content="A OR B",
                priority=Priority.P1,
                lifecycle_status=LifecycleStatus.ACTIVE,
                created_cycle=1,
                updated_cycle=1,
                source="test",
                confidence=0.5,
            )
        )
        executor = MindQLExecutor(db)
        result = await executor.execute(parse("FIND thoughts WHERE content = 'A OR B'"))
        assert [row["thought_id"] for row in result.rows] == ["lit-or"]

    async def test_semicolon_inside_select_literal_allowed(
        self,
        populated_db: aiosqlite.Connection,
    ) -> None:
        # A ';' inside a string literal is not a statement separator.
        executor = MindQLExecutor(populated_db)
        result = await executor.execute(parse("SELECT ';' AS s"))
        assert result.rows == [{"s": ";"}]

    async def test_real_second_statement_still_rejected(
        self,
        populated_db: aiosqlite.Connection,
    ) -> None:
        executor = MindQLExecutor(populated_db)
        with pytest.raises(MindQLParseError, match="single SELECT"):
            await executor.execute(parse("SELECT 1; DROP TABLE thought"))

    def test_bare_unterminated_literal_rejected(self) -> None:
        # An unterminated literal with no OR/IN/paren outside it must still be
        # rejected precisely (routed to the tree tokeniser), not silently
        # accepted by the flat path as a value beginning with a quote.
        with pytest.raises(MindQLParseError, match="Unterminated string"):
            parse("FIND thoughts WHERE content = 'unterminated")

    def test_doubled_quote_escape_not_mis_scanned(self) -> None:
        # A doubled ``''`` inside a literal is skipped by the lexical scan, so
        # the ``OR`` inside the literal is not mistaken for grammar and the
        # literal is seen as balanced (not unterminated). MindQL values do not
        # support the ``''`` escape, so the fragment rejects cleanly rather
        # than being mis-routed or silently accepted.
        with pytest.raises(MindQLParseError):
            parse("FIND thoughts WHERE content = 'it''s OR mine'")


# ---------------------------------------------------------------------------
# Executor-side query validation (constructed queries — parser bypassed)
# ---------------------------------------------------------------------------


# ``MindQLQuery`` is a public export and ``execute_mindql`` accepts a
# constructed one, so ``parse()`` is not on the path these tests exercise:
# every query below is built by hand, exactly as a programmatic caller can.
# The FIND/COUNT compiler interpolates rather than binds four things: the table
# and column identifiers, the ORDER BY direction and boolean-joiner keywords,
# and the LIMIT/OFFSET literals. Each is checked here at that boundary. This is
# the compiler's boundary only — the raw SELECT passthrough is a separate entry
# point with its own guard, and it is deliberately not restricted to these
# tables.

_OFF_LIMITS_TABLE = "offlimits"
_OFF_LIMITS_VALUE = "off-limits-value"

# A table-position payload that reads the off-limits table through a derived
# table, so it leaks whatever the target table's column count happens to be
# (a UNION payload would have to match that count to return anything).
_TABLE_SUBSELECT_PAYLOAD = "(SELECT secret AS thought_id FROM offlimits)"

# Thought ids created by the ``populated_db`` fixture, in id order.
_POPULATED_IDS = ["t-000", "t-001", "t-002", "t-003", "t-004"]

# The queryable surface, restated here on purpose. The executor's per-table
# allowlist is hand-maintained and has no derivable source of truth — which
# columns are safe to expose is a policy choice — so this second statement is
# what makes a silent collapse or widening of it fail a test. The ``thought``
# entry is every column the public MindQL guide documents as filterable, plus
# ``source_type``. Changing the queryable surface means changing both.
_EXPECTED_FILTERABLE_COLUMNS: dict[str, list[str]] = {
    "thought": [
        "thought_id",
        "thought_type",
        "lifecycle_status",
        "priority",
        "essence",
        "content",
        "source",
        "source_type",
        "confidence",
        "visibility",
        "confirmation_count",
        "created_cycle",
        "updated_cycle",
    ],
    "edge": [
        "edge_id",
        "from_thought_id",
        "to_thought_id",
        "edge_type",
        "weight",
        "created_cycle",
        "source",
        "decay_multiplier",
    ],
    "embedding": [
        "embedding_id",
        "owner_type",
        "owner_id",
        "model_name",
        "dimension",
    ],
    "action": [
        "action_id",
        "source_thought_id",
        "action_type",
        "intent",
        "status",
        "verification_status",
    ],
}


class _FormatHijackingInt(int):
    """An ``int`` that emits text of its choosing when interpolated.

    Passing an ``isinstance`` check is not enough: a value is only safe once
    the executor stops interpolating the caller's object.
    """

    __slots__ = ()

    def __format__(self, format_spec: str) -> str:
        return "1 -- "


class _FormatHijackingStr(str):
    """A ``str`` that compares equal to one table but emits another."""

    __slots__ = ()

    def __format__(self, format_spec: str) -> str:
        return _OFF_LIMITS_TABLE


class _FormatHijackingColumn(str):
    """A ``str`` equal to an allowlisted column that emits a predicate."""

    __slots__ = ()

    def __format__(self, format_spec: str) -> str:
        return "priority = 'P1' OR 1=1 OR 'x'"


@pytest.fixture
async def guarded_db(populated_db: aiosqlite.Connection) -> aiosqlite.Connection:
    """Populated store plus a table outside the FIND/COUNT allowlist."""
    await populated_db.execute("CREATE TABLE offlimits (secret TEXT)")
    await populated_db.execute("INSERT INTO offlimits VALUES (?)", (_OFF_LIMITS_VALUE,))
    await populated_db.commit()
    return populated_db


async def _schema_snapshot(conn: aiosqlite.Connection) -> list[tuple[object, ...]]:
    """Return the full ``sqlite_master`` contents as a comparable snapshot."""
    cursor = await conn.execute(
        "SELECT type, name, tbl_name, rootpage, sql FROM sqlite_master ORDER BY type, name",
    )
    return [tuple(row) for row in await cursor.fetchall()]


async def _off_limits_rows(conn: aiosqlite.Connection) -> list[str]:
    """Read the off-limits table directly, bypassing the FIND/COUNT compiler."""
    cursor = await conn.execute("SELECT secret FROM offlimits")
    return [str(row[0]) for row in await cursor.fetchall()]


async def _thought_ids(conn: aiosqlite.Connection) -> list[str]:
    """Read every stored thought id directly, bypassing the compiler."""
    cursor = await conn.execute("SELECT thought_id FROM thought ORDER BY thought_id")
    return [str(row[0]) for row in await cursor.fetchall()]


class TestExecutorTableIdentifierValidation:
    """FIND/COUNT validate the target table themselves, not only via the parser."""

    async def test_off_limits_table_is_not_readable_through_find(
        self,
        guarded_db: aiosqlite.Connection,
    ) -> None:
        store = SqliteEngravaCore(guarded_db)
        schema_before = await _schema_snapshot(guarded_db)
        query = MindQLQuery(command=MindQLCommand.FIND, table=_OFF_LIMITS_TABLE)

        with pytest.raises(MindQLParseError, match="Table 'offlimits' not allowed"):
            await store.execute_mindql(query)

        # Not merely "raises": the store is untouched. That a rejected query
        # never runs at all is pinned separately, by the driver-spy test.
        assert await _schema_snapshot(guarded_db) == schema_before
        assert await _off_limits_rows(guarded_db) == [_OFF_LIMITS_VALUE]
        assert await _thought_ids(guarded_db) == _POPULATED_IDS

    async def test_derived_table_payload_cannot_exfiltrate_through_find(
        self,
        guarded_db: aiosqlite.Connection,
    ) -> None:
        store = SqliteEngravaCore(guarded_db)
        schema_before = await _schema_snapshot(guarded_db)
        query = MindQLQuery(command=MindQLCommand.FIND, table=_TABLE_SUBSELECT_PAYLOAD)

        with pytest.raises(MindQLParseError, match="not allowed"):
            await store.execute_mindql(query)

        assert await _schema_snapshot(guarded_db) == schema_before
        assert await _off_limits_rows(guarded_db) == [_OFF_LIMITS_VALUE]

    async def test_off_limits_table_is_not_countable(
        self,
        guarded_db: aiosqlite.Connection,
    ) -> None:
        # COUNT builds its own SQL, so it is a second interpolation site.
        store = SqliteEngravaCore(guarded_db)
        schema_before = await _schema_snapshot(guarded_db)
        query = MindQLQuery(command=MindQLCommand.COUNT, table=_OFF_LIMITS_TABLE)

        with pytest.raises(MindQLParseError, match="Table 'offlimits' not allowed"):
            await store.execute_mindql(query)

        assert await _schema_snapshot(guarded_db) == schema_before
        assert await _thought_ids(guarded_db) == _POPULATED_IDS

    async def test_schema_table_is_not_readable_through_find(
        self,
        guarded_db: aiosqlite.Connection,
    ) -> None:
        # ``sqlite_master`` is the whole database's DDL, including every
        # extension table a MindQL caller was never granted.
        store = SqliteEngravaCore(guarded_db)
        query = MindQLQuery(command=MindQLCommand.FIND, table="sqlite_master")

        with pytest.raises(MindQLParseError, match="Table 'sqlite_master' not allowed"):
            await store.execute_mindql(query)

    async def test_explain_does_not_compile_an_off_limits_table(
        self,
        guarded_db: aiosqlite.Connection,
    ) -> None:
        # EXPLAIN returns the compiled statement as data. An unvalidated table
        # there hands the caller a ready-to-run statement over that table.
        store = SqliteEngravaCore(guarded_db)
        find = MindQLQuery(
            command=MindQLCommand.FIND,
            table=_OFF_LIMITS_TABLE,
            explain=True,
        )
        count = MindQLQuery(
            command=MindQLCommand.COUNT,
            table=_OFF_LIMITS_TABLE,
            explain=True,
        )

        with pytest.raises(MindQLParseError, match="Table 'offlimits' not allowed"):
            await store.execute_mindql(find)
        with pytest.raises(MindQLParseError, match="Table 'offlimits' not allowed"):
            await store.execute_mindql(count)

    async def test_every_canonical_table_still_executes(
        self,
        guarded_db: aiosqlite.Connection,
    ) -> None:
        # The guard must not narrow the surface: every table the grammar
        # accepts still runs, on both FIND and COUNT.
        store = SqliteEngravaCore(guarded_db)
        for table in sorted(parser_module._TABLE_MAP.values()):
            found = await store.execute_mindql(
                MindQLQuery(command=MindQLCommand.FIND, table=table, explain=True),
            )
            # Not just "no exception": the plan must name the table asked for.
            find_sql = f"SELECT * FROM {table} LIMIT 100"  # noqa: S608 -- expectation
            assert found.rows[0]["sql"] == find_sql
            ran_find = await store.execute_mindql(
                MindQLQuery(command=MindQLCommand.FIND, table=table),
            )
            assert ran_find.command == "FIND"
            counted = await store.execute_mindql(
                MindQLQuery(command=MindQLCommand.COUNT, table=table, explain=True),
            )
            count_sql = f"SELECT COUNT(*) AS cnt FROM {table}"  # noqa: S608 -- expectation
            assert counted.rows[0]["sql"] == count_sql
            ran = await store.execute_mindql(
                MindQLQuery(command=MindQLCommand.COUNT, table=table),
            )
            assert ran.count is not None

    async def test_unset_table_still_defaults_to_thought(
        self,
        guarded_db: aiosqlite.Connection,
    ) -> None:
        store = SqliteEngravaCore(guarded_db)
        result = await store.execute_mindql(MindQLQuery(command=MindQLCommand.FIND))
        assert sorted(str(row["thought_id"]) for row in result.rows) == _POPULATED_IDS

    async def test_str_subclass_cannot_redirect_the_query(
        self,
        guarded_db: aiosqlite.Connection,
    ) -> None:
        # The allowlist check passes on equality, but the emitted statement
        # must carry the canonical name, not whatever the object formats to.
        store = SqliteEngravaCore(guarded_db)
        query = MindQLQuery(
            command=MindQLCommand.FIND,
            table=_FormatHijackingStr("thought"),
        )

        result = await store.execute_mindql(query)
        assert sorted(str(row["thought_id"]) for row in result.rows) == _POPULATED_IDS
        assert all(_OFF_LIMITS_VALUE not in row.values() for row in result.rows)

    async def test_non_string_table_is_rejected_with_a_typed_error(
        self,
        guarded_db: aiosqlite.Connection,
    ) -> None:
        # A falsy non-string must not slip through the ``or`` default, and an
        # unhashable one must not surface as an untyped lookup failure.
        store = SqliteEngravaCore(guarded_db)
        for bad_table in (cast("str", []), cast("str", ["thought"]), cast("str", 0)):
            with pytest.raises(MindQLParseError, match="Table name must be a string"):
                await store.execute_mindql(
                    MindQLQuery(command=MindQLCommand.FIND, table=bad_table),
                )


class TestExecutorSortDirectionValidation:
    """The ORDER BY direction is an interpolated keyword, and is validated."""

    async def test_direction_payload_cannot_hijack_the_row_cap(
        self,
        guarded_db: aiosqlite.Connection,
    ) -> None:
        # " ASC LIMIT 1 --" comments out the LIMIT the executor appends, so the
        # emitted statement runs under the caller's cap instead of the store's.
        store = SqliteEngravaCore(guarded_db)
        query = MindQLQuery(
            command=MindQLCommand.FIND,
            table="thought",
            order_by=(("created_cycle", "ASC LIMIT 1 --"),),
        )

        with pytest.raises(MindQLParseError, match="Sort direction"):
            await store.execute_mindql(query)

        assert await _thought_ids(guarded_db) == _POPULATED_IDS

    async def test_unknown_direction_is_rejected_with_a_typed_error(
        self,
        guarded_db: aiosqlite.Connection,
    ) -> None:
        store = SqliteEngravaCore(guarded_db)
        query = MindQLQuery(
            command=MindQLCommand.FIND,
            table="thought",
            order_by=(("created_cycle", "DESCENDING"),),
        )

        with pytest.raises(MindQLParseError, match="Sort direction 'DESCENDING' not allowed"):
            await store.execute_mindql(query)

    async def test_lowercase_direction_still_sorts(
        self,
        guarded_db: aiosqlite.Connection,
    ) -> None:
        # A constructed query carrying a lowercase direction executes today;
        # the guard normalises the case instead of rejecting the shape.
        store = SqliteEngravaCore(guarded_db)
        query = MindQLQuery(
            command=MindQLCommand.FIND,
            table="thought",
            order_by=(("created_cycle", "desc"),),
        )

        result = await store.execute_mindql(query)
        assert [row["thought_id"] for row in result.rows] == list(reversed(_POPULATED_IDS))

    async def test_non_string_direction_is_rejected(
        self,
        guarded_db: aiosqlite.Connection,
    ) -> None:
        # The declared ``str`` is not enforced at runtime, so the guard must
        # not assume it: a non-string must not reach ``.upper()`` and surface
        # as an untyped ``AttributeError``.
        store = SqliteEngravaCore(guarded_db)
        query = MindQLQuery(
            command=MindQLCommand.FIND,
            table="thought",
            order_by=(("created_cycle", cast("str", 5)),),
        )

        with pytest.raises(MindQLParseError, match="Sort direction 5 not allowed"):
            await store.execute_mindql(query)

        assert await _thought_ids(guarded_db) == _POPULATED_IDS

    async def test_str_subclass_direction_cannot_reach_the_statement(
        self,
        guarded_db: aiosqlite.Connection,
    ) -> None:
        # The keyword guard matches on equality, so what it emits must be its
        # own literal rather than the object that matched.
        store = SqliteEngravaCore(guarded_db)
        plan = await store.execute_mindql(
            MindQLQuery(
                command=MindQLCommand.FIND,
                table="thought",
                order_by=(("created_cycle", cast("str", _FormatHijackingStr("DESC"))),),
                explain=True,
            ),
        )
        assert plan.rows[0]["sql"] == (
            "SELECT * FROM thought ORDER BY created_cycle DESC LIMIT 100"
        )

    async def test_parser_directions_still_sort_both_ways(
        self,
        guarded_db: aiosqlite.Connection,
    ) -> None:
        store = SqliteEngravaCore(guarded_db)
        ascending = await store.execute_mindql(parse("FIND thoughts ORDER BY created_cycle ASC"))
        descending = await store.execute_mindql(parse("FIND thoughts ORDER BY created_cycle DESC"))
        assert [row["thought_id"] for row in ascending.rows] == _POPULATED_IDS
        assert [row["thought_id"] for row in descending.rows] == list(reversed(_POPULATED_IDS))


class TestExecutorRowBoundValidation:
    """LIMIT and OFFSET are interpolated literals, so both are validated."""

    async def test_boolean_limit_is_rejected(
        self,
        guarded_db: aiosqlite.Connection,
    ) -> None:
        # ``bool`` is an ``int`` subclass, so this passes a type checker and
        # SQLite reads ``LIMIT True`` as ``LIMIT 1``.
        store = SqliteEngravaCore(guarded_db)
        query = MindQLQuery(command=MindQLCommand.FIND, table="thought", limit=True)

        with pytest.raises(MindQLParseError, match="LIMIT must be a non-negative integer"):
            await store.execute_mindql(query)

    async def test_boolean_offset_is_rejected(
        self,
        guarded_db: aiosqlite.Connection,
    ) -> None:
        store = SqliteEngravaCore(guarded_db)
        query = MindQLQuery(command=MindQLCommand.FIND, table="thought", offset=True)

        with pytest.raises(MindQLParseError, match="OFFSET must be a non-negative integer"):
            await store.execute_mindql(query)

    async def test_limit_payload_cannot_comment_out_the_offset(
        self,
        guarded_db: aiosqlite.Connection,
    ) -> None:
        store = SqliteEngravaCore(guarded_db)
        query = MindQLQuery(
            command=MindQLCommand.FIND,
            table="thought",
            limit=cast("int", "1 -- "),
            offset=3,
        )

        with pytest.raises(MindQLParseError, match="LIMIT must be a non-negative integer"):
            await store.execute_mindql(query)

        assert await _thought_ids(guarded_db) == _POPULATED_IDS

    async def test_negative_offset_is_rejected(
        self,
        guarded_db: aiosqlite.Connection,
    ) -> None:
        # SQLite silently treats a negative OFFSET as 0, so this is a
        # contract violation the database will never report.
        store = SqliteEngravaCore(guarded_db)
        query = MindQLQuery(command=MindQLCommand.FIND, table="thought", offset=-5)

        with pytest.raises(MindQLParseError, match="OFFSET must be a non-negative integer"):
            await store.execute_mindql(query)

    async def test_negative_limit_cannot_bypass_the_default_row_cap(
        self,
        db: aiosqlite.Connection,
    ) -> None:
        # SQLite reads ``LIMIT -1`` as "no limit", which turns the store's
        # unqualified-FIND cap off entirely.
        default = executor_module.DEFAULT_FIND_LIMIT
        store = SqliteEngravaCore(db)
        for i in range(default + 5):
            await store.create_thought(
                ThoughtRecord(
                    thought_id=f"t-cap-{i:04d}",
                    thought_type=ThoughtType.OBSERVATION,
                    essence=f"essence {i}",
                    content=f"content {i}",
                    priority=Priority.P1,
                    lifecycle_status=LifecycleStatus.ACTIVE,
                    created_cycle=1,
                    updated_cycle=1,
                    source="cap",
                )
            )
        query = MindQLQuery(command=MindQLCommand.FIND, table="thought", limit=-1)

        with pytest.raises(MindQLParseError, match="LIMIT must be a non-negative integer"):
            await store.execute_mindql(query)

        # The cap is still the only bound on an unqualified FIND.
        uncapped = await store.execute_mindql(MindQLQuery(command=MindQLCommand.FIND))
        assert len(uncapped.rows) == default

    async def test_int_subclass_cannot_control_the_emitted_bound(
        self,
        guarded_db: aiosqlite.Connection,
    ) -> None:
        # The value is a valid bound, so the query is legitimate — but the
        # emitted text must be the integer, not what the subclass formats to.
        store = SqliteEngravaCore(guarded_db)
        query = MindQLQuery(
            command=MindQLCommand.FIND,
            table="thought",
            order_by=(("thought_id", "ASC"),),
            limit=_FormatHijackingInt(1),
            offset=3,
        )

        result = await store.execute_mindql(query)
        # Pre-guard the subclass commented the OFFSET out and returned t-000.
        assert [row["thought_id"] for row in result.rows] == ["t-003"]

        explained = await store.execute_mindql(
            MindQLQuery(
                command=query.command,
                table=query.table,
                order_by=query.order_by,
                limit=query.limit,
                offset=query.offset,
                explain=True,
            ),
        )
        assert explained.rows[0]["sql"] == (
            "SELECT * FROM thought ORDER BY thought_id ASC LIMIT 1 OFFSET 3"
        )

    async def test_zero_limit_and_zero_offset_still_execute(
        self,
        guarded_db: aiosqlite.Connection,
    ) -> None:
        # Zero is a legal bound on both clauses and must stay legal.
        store = SqliteEngravaCore(guarded_db)
        empty = await store.execute_mindql(
            MindQLQuery(command=MindQLCommand.FIND, table="thought", limit=0),
        )
        assert empty.rows == []
        full = await store.execute_mindql(
            MindQLQuery(command=MindQLCommand.FIND, table="thought", offset=0),
        )
        assert sorted(str(row["thought_id"]) for row in full.rows) == _POPULATED_IDS

    async def test_parser_bounds_still_page(
        self,
        guarded_db: aiosqlite.Connection,
    ) -> None:
        store = SqliteEngravaCore(guarded_db)
        result = await store.execute_mindql(
            parse("FIND thoughts ORDER BY thought_id ASC LIMIT 2 OFFSET 2"),
        )
        assert [row["thought_id"] for row in result.rows] == ["t-002", "t-003"]


class TestExecutorBooleanJoinerValidation:
    """The WHERE tree's boolean joiner is interpolated, so it is validated."""

    @staticmethod
    def _tree_query(joiner: str) -> MindQLQuery:
        return MindQLQuery(
            command=MindQLCommand.FIND,
            table="thought",
            where=BoolExpr(
                op=cast("Literal['AND', 'OR']", joiner),
                operands=(
                    Comparison(field="priority", operator=MindQLOperator.EQ, value="P1"),
                    Comparison(field="priority", operator=MindQLOperator.EQ, value="P2"),
                ),
            ),
        )

    async def test_joiner_payload_is_rejected(
        self,
        guarded_db: aiosqlite.Connection,
    ) -> None:
        # The joiner sits between two already-parameterised operands, so a
        # payload here rewrites the predicate without touching a bound value.
        store = SqliteEngravaCore(guarded_db)

        with pytest.raises(MindQLParseError, match="Boolean operator"):
            await store.execute_mindql(self._tree_query("OR 1=1 OR"))

        assert await _thought_ids(guarded_db) == _POPULATED_IDS

    async def test_lowercase_joiner_still_compiles(
        self,
        guarded_db: aiosqlite.Connection,
    ) -> None:
        # Case is normalised rather than rejected, as it is for sort direction.
        store = SqliteEngravaCore(guarded_db)
        result = await store.execute_mindql(self._tree_query("or"))
        assert sorted(str(row["thought_id"]) for row in result.rows) == _POPULATED_IDS

    async def test_str_subclass_joiner_cannot_reach_the_statement(
        self,
        guarded_db: aiosqlite.Connection,
    ) -> None:
        store = SqliteEngravaCore(guarded_db)
        query = self._tree_query(cast("str", _FormatHijackingStr("OR")))
        plan = await store.execute_mindql(
            MindQLQuery(
                command=query.command,
                table=query.table,
                where=query.where,
                explain=True,
            ),
        )
        assert plan.rows[0]["sql"] == (
            "SELECT * FROM thought WHERE (priority = ? OR priority = ?) LIMIT 100"
        )

    async def test_parser_joiners_still_compile(
        self,
        guarded_db: aiosqlite.Connection,
    ) -> None:
        store = SqliteEngravaCore(guarded_db)
        both = await store.execute_mindql(
            parse("FIND thoughts WHERE priority = 'P1' OR priority = 'P2'"),
        )
        assert sorted(str(row["thought_id"]) for row in both.rows) == _POPULATED_IDS
        neither = await store.execute_mindql(
            parse("FIND thoughts WHERE priority = 'P1' AND priority = 'P2'"),
        )
        assert neither.rows == []


class TestRejectedIdentifiersNeverExecute:
    """One rejected payload per guard, none of which reaches SQLite at all."""

    async def test_no_rejected_query_reaches_the_driver(
        self,
        guarded_db: aiosqlite.Connection,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        store = SqliteEngravaCore(guarded_db)
        executed: list[str] = []
        original_execute = guarded_db.execute

        async def _spy(sql: object, *args: object, **kwargs: object) -> object:
            executed.append(str(sql))
            return await original_execute(sql, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(guarded_db, "execute", _spy)

        hostile = [
            MindQLQuery(command=MindQLCommand.FIND, table=_OFF_LIMITS_TABLE),
            MindQLQuery(command=MindQLCommand.COUNT, table=_OFF_LIMITS_TABLE),
            MindQLQuery(command=MindQLCommand.FIND, table=_TABLE_SUBSELECT_PAYLOAD),
            MindQLQuery(command=MindQLCommand.FIND, table="sqlite_master"),
            MindQLQuery(
                command=MindQLCommand.FIND,
                table="thought",
                order_by=(("created_cycle", "ASC LIMIT 1 --"),),
            ),
            MindQLQuery(
                command=MindQLCommand.FIND,
                table="thought",
                order_by=(("created_cycle", cast("str", 5)),),
            ),
            MindQLQuery(command=MindQLCommand.FIND, table="thought", limit=True),
            MindQLQuery(command=MindQLCommand.FIND, table="thought", offset=True),
            MindQLQuery(command=MindQLCommand.FIND, table="thought", limit=-1),
            MindQLQuery(
                command=MindQLCommand.FIND,
                table="thought",
                limit=cast("int", "1 -- "),
            ),
            MindQLQuery(command=MindQLCommand.FIND, table="thought", offset=-5),
            MindQLQuery(command=MindQLCommand.FIND, table=cast("str", ["thought"])),
            MindQLQuery(
                command=MindQLCommand.FIND,
                table="thought",
                conditions=[
                    Condition(field="valid_from", operator=MindQLOperator.EQ, value="x"),
                ],
            ),
            TestExecutorBooleanJoinerValidation._tree_query("OR 1=1 OR"),
        ]
        for query in hostile:
            with pytest.raises(MindQLParseError):
                await store.execute_mindql(query)

        assert executed == []


class TestAllowedColumnsAllowlist:
    """The per-table column allowlist is pinned to what it actually scopes."""

    async def test_columns_are_scoped_to_their_own_table(
        self,
        populated_db: aiosqlite.Connection,
    ) -> None:
        # ``weight`` exists on edge only, ``essence`` on thought only: each is
        # queryable on its own table and refused on the other.
        store = SqliteEngravaCore(populated_db)
        await store.create_edge(
            EdgeRecord(
                edge_id="e-scope",
                from_thought_id="t-000",
                to_thought_id="t-001",
                edge_type=EdgeType.ASSOCIATED,
                weight=0.5,
                created_cycle=1,
            )
        )
        await populated_db.commit()

        edge_hit = await store.execute_mindql(
            MindQLQuery(
                command=MindQLCommand.FIND,
                table="edge",
                conditions=[Condition(field="weight", operator=MindQLOperator.EQ, value=0.5)],
            ),
        )
        assert [row["edge_id"] for row in edge_hit.rows] == ["e-scope"]

        thought_hit = await store.execute_mindql(
            MindQLQuery(
                command=MindQLCommand.FIND,
                table="thought",
                conditions=[
                    Condition(
                        field="essence",
                        operator=MindQLOperator.EQ,
                        value="Thought number 1",
                    ),
                ],
            ),
        )
        assert [row["thought_id"] for row in thought_hit.rows] == ["t-001"]

        with pytest.raises(MindQLParseError, match="'weight' not allowed for table 'thought'"):
            await store.execute_mindql(
                MindQLQuery(
                    command=MindQLCommand.FIND,
                    table="thought",
                    conditions=[
                        Condition(field="weight", operator=MindQLOperator.EQ, value=0.5),
                    ],
                ),
            )
        with pytest.raises(MindQLParseError, match="'essence' not allowed for table 'edge'"):
            await store.execute_mindql(
                MindQLQuery(
                    command=MindQLCommand.FIND,
                    table="edge",
                    conditions=[
                        Condition(field="essence", operator=MindQLOperator.EQ, value="x"),
                    ],
                ),
            )

    async def test_str_subclass_column_cannot_rewrite_the_predicate(
        self,
        populated_db: aiosqlite.Connection,
    ) -> None:
        # The allowlist check passes on equality, so the emitted statement
        # must carry the allowlist's own column name at every site that
        # interpolates one: WHERE, IN, and ORDER BY.
        store = SqliteEngravaCore(populated_db)
        hijacked = cast("str", _FormatHijackingColumn("priority"))

        filtered = await store.execute_mindql(
            MindQLQuery(
                command=MindQLCommand.FIND,
                table="thought",
                conditions=[
                    Condition(field=hijacked, operator=MindQLOperator.EQ, value="P1"),
                ],
            ),
        )
        # An "OR 1=1" payload would widen this to all five rows.
        assert sorted(str(row["thought_id"]) for row in filtered.rows) == ["t-000", "t-001"]

        in_filtered = await store.execute_mindql(
            MindQLQuery(
                command=MindQLCommand.FIND,
                table="thought",
                where=InCondition(field=hijacked, values=("P1",)),
            ),
        )
        assert sorted(str(row["thought_id"]) for row in in_filtered.rows) == ["t-000", "t-001"]

        tree_filtered = await store.execute_mindql(
            MindQLQuery(
                command=MindQLCommand.FIND,
                table="thought",
                where=Comparison(
                    field=hijacked,
                    operator=MindQLOperator.EQ,
                    value="P1",
                ),
            ),
        )
        assert sorted(str(row["thought_id"]) for row in tree_filtered.rows) == ["t-000", "t-001"]

        sorted_plan = await store.execute_mindql(
            MindQLQuery(
                command=MindQLCommand.FIND,
                table="thought",
                order_by=((hijacked, "ASC"),),
                explain=True,
            ),
        )
        assert sorted_plan.rows[0]["sql"] == (
            "SELECT * FROM thought ORDER BY priority ASC LIMIT 100"
        )

    def test_allowlist_covers_exactly_the_grammar_tables(self) -> None:
        # The executor's table allowlist is derived from the keys of the
        # column allowlist, so it must cover every table the grammar accepts —
        # otherwise a parsed query would be refused at execution.
        assert set(executor_module._ALLOWED_COLUMNS) == set(parser_module._TABLE_MAP.values())

    async def test_the_queryable_surface_is_exactly_what_is_expected(
        self,
        populated_db: aiosqlite.Connection,
    ) -> None:
        # Two directions at once: every expected column really compiles and
        # runs against the live schema (a collapse, or an allowlisted column
        # the schema no longer has, fails here), and the allowlist carries
        # nothing beyond them (a silent widening fails here).
        store = SqliteEngravaCore(populated_db)
        for table, columns in _EXPECTED_FILTERABLE_COLUMNS.items():
            for column in columns:
                query = MindQLQuery(
                    command=MindQLCommand.COUNT,
                    table=table,
                    conditions=[
                        Condition(field=column, operator=MindQLOperator.NE, value="\x00"),
                    ],
                )
                plan = await store.execute_mindql(
                    MindQLQuery(
                        command=query.command,
                        table=query.table,
                        conditions=query.conditions,
                        explain=True,
                    ),
                )
                # The column asked for is the column compiled, not merely
                # some accepted column.
                head = f"SELECT COUNT(*) AS cnt FROM {table}"  # noqa: S608 -- expectation
                assert plan.rows[0]["sql"] == f"{head} WHERE {column} != ?"
                result = await store.execute_mindql(query)
                assert result.count is not None, f"{table}.{column} no longer filters"
            assert set(executor_module._ALLOWED_COLUMNS[table]) == set(columns)

    async def test_valid_time_columns_are_not_filterable(
        self,
        populated_db: aiosqlite.Connection,
    ) -> None:
        # A documented non-guarantee: valid time is reachable only through the
        # temporal predicates, never as an ordinary column comparison.
        store = SqliteEngravaCore(populated_db)
        for table in ("thought", "edge"):
            for column in ("valid_from", "valid_until"):
                flat = MindQLQuery(
                    command=MindQLCommand.FIND,
                    table=table,
                    conditions=[
                        Condition(field=column, operator=MindQLOperator.EQ, value="x"),
                    ],
                )
                tree = MindQLQuery(
                    command=MindQLCommand.COUNT,
                    table=table,
                    where=Comparison(
                        field=column,
                        operator=MindQLOperator.EQ,
                        value="x",
                    ),
                )
                sort = MindQLQuery(
                    command=MindQLCommand.FIND,
                    table=table,
                    order_by=((column, "ASC"),),
                )
                for query in (flat, tree, sort):
                    with pytest.raises(MindQLParseError, match=f"not allowed for table '{table}'"):
                        await store.execute_mindql(query)

    async def test_every_allowlisted_column_exists_in_the_schema(
        self,
        db: aiosqlite.Connection,
    ) -> None:
        # The allowlist is hand-maintained against the real schema; a renamed
        # or dropped column would otherwise leave a stale entry that fails as
        # an untyped SQLite error at query time.
        for table, columns in executor_module._ALLOWED_COLUMNS.items():
            cursor = await db.execute("SELECT name FROM pragma_table_info(?)", (table,))
            actual = {str(row[0]) for row in await cursor.fetchall()}
            assert actual, f"table {table!r} is absent from the core schema"
            missing = sorted(columns - actual)
            assert not missing, f"{table}: allowlisted columns absent from the schema: {missing}"


# ---------------------------------------------------------------------------
# SELECT passthrough: read-only-ness decided on a value the module owns
# ---------------------------------------------------------------------------


# The passthrough is documented as read-only, and that guarantee is established
# by inspecting the statement the caller hands in. ``str`` is subclassable and
# every method such an inspection reaches for — ``strip``, ``upper``,
# ``startswith``, ``endswith``, ``__len__``, ``__str__`` — is overridable, while
# SQLite still reads the object's real text. A guard built out of the caller's
# own methods can therefore be told it is running a SELECT and hand a DELETE to
# the database.
#
# Every refusal below is judged against the store rather than against the
# exception: the rows are read back — or the driver is watched — *before*
# anything about the raised error is asserted. "It raised" is the weaker claim
# and would be satisfied by a guard that raised only after the write had
# already run; asserting it first would abort the test at the wrong place and
# hide what actually happened.

_WRITE_PAYLOAD = "DELETE FROM thought"
_READ_STATEMENT = "SELECT thought_id FROM thought"
# Two statements in one string: the second one is what a single-statement guard
# exists to keep out.
_SMUGGLED_PAYLOAD = "SELECT 1; DELETE FROM thought"


class _LyingSelectSql(str):
    """Answers ``strip`` with itself and ``upper`` with a SELECT it is not.

    The reported vector. An inspection built from these two methods sees a
    read; the buffer SQLite receives is a write.
    """

    __slots__ = ()

    def strip(self, chars: str | None = None, /) -> _LyingSelectSql:
        return self

    def upper(self) -> str:
        return "SELECT"


class _LyingStartswithSql(str):
    """Stays in play through ``strip``/``upper``, then lies in ``startswith``."""

    __slots__ = ()

    def strip(self, chars: str | None = None, /) -> _LyingStartswithSql:
        return self

    def upper(self) -> _LyingStartswithSql:
        return self

    def startswith(
        self,
        prefix: str | tuple[str, ...],
        start: SupportsIndex | None = None,
        end: SupportsIndex | None = None,
        /,
    ) -> bool:
        return True


class _LyingSeparatorSql(str):
    """Inverts every answer about its own trailing statement separator.

    Aimed at the other half of the guard: a statement that really ends in
    ``;`` claims it does not, and one that does not claims it does.
    """

    __slots__ = ()

    def strip(self, chars: str | None = None, /) -> _LyingSeparatorSql:
        return self

    def endswith(
        self,
        suffix: str | tuple[str, ...],
        start: SupportsIndex | None = None,
        end: SupportsIndex | None = None,
        /,
    ) -> bool:
        return not str.endswith(self, suffix)


class _LyingEqualitySql(str):
    """Compares equal to anything, while carrying a write."""

    __slots__ = ()

    def strip(self, chars: str | None = None, /) -> _LyingEqualitySql:
        return self

    def __eq__(self, other: object) -> bool:
        return True

    def __hash__(self) -> int:
        return str.__hash__(self)


class _HiddenSeparatorSql(str):
    """Reports its length as everything before the first ``;``.

    A lexical scan that walks the caller's object never reaches the second
    statement, so the single-statement check passes on a string that carries
    two.
    """

    __slots__ = ()

    def strip(self, chars: str | None = None, /) -> _HiddenSeparatorSql:
        return self

    def __len__(self) -> int:
        return str.find(self, ";")


class _MaskedWriteSql(str):
    """Carries a write while ``__str__`` shows a read."""

    __slots__ = ()

    def __str__(self) -> str:
        return _READ_STATEMENT


class _MaskedReadSql(str):
    """Carries a read while ``__str__`` shows a write."""

    __slots__ = ()

    def __str__(self) -> str:
        return _WRITE_PAYLOAD


class _PassthroughStripSql(str):
    """A legitimate statement whose ``strip`` keeps the object itself in play."""

    __slots__ = ()

    def strip(self, chars: str | None = None, /) -> _PassthroughStripSql:
        return self


class _ShiftingSql(str):  # noqa: SLOT000 -- carries state; str takes no non-empty __slots__
    """Answers the guard's first question one way and tells the truth after.

    Time-of-check/time-of-use: the statement the guard inspects is not the
    statement a later reader — including SQLite — would see.
    """

    def __init__(self, _value: str) -> None:
        super().__init__()
        self._answered = False

    def strip(self, chars: str | None = None, /) -> _ShiftingSql:
        return self

    def upper(self) -> str:
        if self._answered:
            return str.upper(self)
        self._answered = True
        return "SELECT"


class _ExplodingRepr:
    """A non-string that raises when anything tries to represent it.

    The refusal message is the last place on the guard's path where a caller's
    code could still run. This value makes that visible: formatting it into an
    error would replace the module's own exception with whatever ``__repr__``
    chose to raise.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        msg = "__repr__ must not run inside a guard"
        raise RuntimeError(msg)


async def _refusal(store: SqliteEngravaCore, query: MindQLQuery) -> Exception | None:
    """Run a query that must be refused and return what it raised, if anything.

    The exception is returned rather than asserted so the caller can read the
    stored rows first. A refusal that arrives *after* the write has run is not
    a refusal, and asserting on the exception ahead of the state would hide
    exactly that.
    """
    try:
        await store.execute_mindql(query)
    except Exception as exc:  # noqa: BLE001 -- returned so state is asserted first
        return exc
    return None


class TestSelectPassthroughGuardOwnsItsValue:
    """The read-only guard cannot be talked out of its decision."""

    async def test_reported_bypass_destroys_nothing(
        self,
        populated_db: aiosqlite.Connection,
    ) -> None:
        store = SqliteEngravaCore(populated_db)
        before = await _thought_ids(populated_db)

        raised = await _refusal(
            store,
            MindQLQuery(
                command=MindQLCommand.SELECT,
                raw_sql=_LyingSelectSql(_WRITE_PAYLOAD),
            ),
        )

        after = await _thought_ids(populated_db)
        assert after == before == _POPULATED_IDS
        assert isinstance(raised, MindQLParseError)
        assert "Only SELECT statements are allowed" in str(raised)

    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            pytest.param(
                _LyingStartswithSql(_WRITE_PAYLOAD),
                "Only SELECT statements are allowed",
                id="lying-startswith",
            ),
            pytest.param(
                _LyingEqualitySql(_WRITE_PAYLOAD),
                "Only SELECT statements are allowed",
                id="lying-eq",
            ),
            pytest.param(
                _MaskedWriteSql(_WRITE_PAYLOAD),
                "Only SELECT statements are allowed",
                id="masked-str-dunder",
            ),
            pytest.param(
                _HiddenSeparatorSql(_SMUGGLED_PAYLOAD),
                "Only a single SELECT statement is allowed",
                id="hidden-length",
            ),
        ],
    )
    async def test_adversarial_statement_writes_nothing(
        self,
        populated_db: aiosqlite.Connection,
        payload: str,
        expected: str,
    ) -> None:
        store = SqliteEngravaCore(populated_db)
        before = await _thought_ids(populated_db)

        raised = await _refusal(
            store,
            MindQLQuery(command=MindQLCommand.SELECT, raw_sql=payload),
        )

        after = await _thought_ids(populated_db)
        assert after == before == _POPULATED_IDS
        assert isinstance(raised, MindQLParseError)
        assert expected in str(raised)

    async def test_statement_that_changes_its_answer_writes_nothing(
        self,
        populated_db: aiosqlite.Connection,
    ) -> None:
        # Built fresh: this one keeps state, and its lie is spent on first use.
        store = SqliteEngravaCore(populated_db)
        before = await _thought_ids(populated_db)

        raised = await _refusal(
            store,
            MindQLQuery(
                command=MindQLCommand.SELECT,
                raw_sql=_ShiftingSql(_WRITE_PAYLOAD),
            ),
        )

        after = await _thought_ids(populated_db)
        assert after == before == _POPULATED_IDS
        assert isinstance(raised, MindQLParseError)
        assert "Only SELECT statements are allowed" in str(raised)

    @pytest.mark.parametrize(
        ("value", "value_id"),
        [
            pytest.param(None, "none", id="none"),
            pytest.param(5, "int", id="int"),
            pytest.param(b"SELECT thought_id FROM thought", "bytes", id="bytes"),
            pytest.param(["SELECT thought_id FROM thought"], "list", id="list"),
        ],
    )
    async def test_non_string_sql_is_refused_with_a_typed_error(
        self,
        populated_db: aiosqlite.Connection,
        value: object,
        value_id: str,
    ) -> None:
        # A constructed query is not type-checked at runtime, so the guard owns
        # the type as well as the value — and refuses with its own exception
        # rather than whatever the value's methods happen to raise.
        store = SqliteEngravaCore(populated_db)
        before = await _thought_ids(populated_db)

        raised = await _refusal(
            store,
            MindQLQuery(command=MindQLCommand.SELECT, raw_sql=cast("str", value)),
        )

        after = await _thought_ids(populated_db)
        assert after == before == _POPULATED_IDS
        assert isinstance(raised, MindQLParseError), f"{value_id}: {raised!r}"
        assert "must be a string" in str(raised)

    async def test_a_hostile_repr_cannot_change_the_error_type(
        self,
        populated_db: aiosqlite.Connection,
    ) -> None:
        # The refusal owes a typed domain error for *every* non-string, which
        # means its message may not be built out of the value being refused.
        # Formatting the rejected value would hand the last word on the guard's
        # path back to the caller.
        store = SqliteEngravaCore(populated_db)
        before = await _thought_ids(populated_db)

        raised = await _refusal(
            store,
            MindQLQuery(
                command=MindQLCommand.SELECT,
                raw_sql=cast("str", _ExplodingRepr()),
            ),
        )

        after = await _thought_ids(populated_db)
        assert after == before == _POPULATED_IDS
        assert isinstance(raised, MindQLParseError)
        assert "must be a string" in str(raised)

    async def test_no_refused_statement_reaches_the_driver(
        self,
        populated_db: aiosqlite.Connection,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        store = SqliteEngravaCore(populated_db)
        executed: list[str] = []
        original_execute = populated_db.execute

        async def _spy(sql: str, *args: object, **kwargs: object) -> object:
            executed.append(str(sql))
            return await original_execute(sql, *args, **kwargs)

        monkeypatch.setattr(populated_db, "execute", _spy)

        hostile: list[str] = [
            _LyingSelectSql(_WRITE_PAYLOAD),
            _LyingStartswithSql(_WRITE_PAYLOAD),
            _LyingEqualitySql(_WRITE_PAYLOAD),
            _MaskedWriteSql(_WRITE_PAYLOAD),
            _HiddenSeparatorSql(_SMUGGLED_PAYLOAD),
            _ShiftingSql(_WRITE_PAYLOAD),
        ]
        raised = [
            await _refusal(
                store,
                MindQLQuery(command=MindQLCommand.SELECT, raw_sql=payload),
            )
            for payload in hostile
        ]

        # "Nothing reached SQLite" is the stronger claim and goes first: a
        # guard that ran the statement and then raised would satisfy every
        # assertion about the exception while the write had already happened.
        assert executed == []
        assert [type(exc) for exc in raised] == [MindQLParseError] * len(hostile)

    async def test_the_guard_returns_the_modules_own_string(
        self,
        populated_db: aiosqlite.Connection,
    ) -> None:
        # The statement that was validated is the statement that runs: the
        # guard hands back a plain ``str`` it built, never the caller's object,
        # so nothing the object overrides can act between check and use.
        store = SqliteEngravaCore(populated_db)
        plan = await store.execute_mindql(
            MindQLQuery(
                command=MindQLCommand.SELECT,
                raw_sql=_PassthroughStripSql(_READ_STATEMENT),
                explain=True,
            ),
        )

        compiled = plan.rows[0]["sql"]
        assert type(compiled) is str
        assert compiled == _READ_STATEMENT

    async def test_str_dunder_cannot_redirect_execution(
        self,
        populated_db: aiosqlite.Connection,
    ) -> None:
        # The mirror of the masked-write case: a legitimate SELECT whose
        # ``__str__`` reports a DELETE must run the SELECT it actually is.
        # Normalising through ``str()`` rather than the buffer would validate,
        # and return, a statement that is not the one handed in — which is the
        # property that matters, whatever that other statement happens to do.
        store = SqliteEngravaCore(populated_db)
        before = await _thought_ids(populated_db)

        result = await store.execute_mindql(
            MindQLQuery(
                command=MindQLCommand.SELECT,
                raw_sql=_MaskedReadSql(_READ_STATEMENT),
            ),
        )

        assert await _thought_ids(populated_db) == before == _POPULATED_IDS
        assert [str(row["thought_id"]) for row in result.rows] == _POPULATED_IDS


class TestSelectPassthroughLegitimateShapes:
    """Owning the value must not cost the passthrough anything it accepted."""

    @pytest.mark.parametrize(
        "raw_sql",
        [
            pytest.param(_READ_STATEMENT, id="plain"),
            pytest.param(f"   {_READ_STATEMENT}   ", id="surrounding-whitespace"),
            pytest.param(f"{_READ_STATEMENT};", id="trailing-separator"),
            pytest.param(f"{_READ_STATEMENT} ;  ", id="spaced-trailing-separator"),
            pytest.param("select thought_id from thought", id="lower-case"),
            pytest.param(f"\n{_READ_STATEMENT}\n", id="newlines"),
            pytest.param(_PassthroughStripSql(_READ_STATEMENT), id="str-subclass"),
            pytest.param(_LyingSeparatorSql(_READ_STATEMENT), id="lying-separator-absent"),
            pytest.param(_LyingSeparatorSql(f"{_READ_STATEMENT};"), id="lying-separator-present"),
        ],
    )
    async def test_legitimate_statement_still_executes(
        self,
        populated_db: aiosqlite.Connection,
        raw_sql: str,
    ) -> None:
        store = SqliteEngravaCore(populated_db)
        result = await store.execute_mindql(
            MindQLQuery(command=MindQLCommand.SELECT, raw_sql=raw_sql),
        )
        assert [str(row["thought_id"]) for row in result.rows] == _POPULATED_IDS

    async def test_bound_parameters_survive_a_str_subclass(
        self,
        populated_db: aiosqlite.Connection,
    ) -> None:
        store = SqliteEngravaCore(populated_db)
        result = await store.execute_mindql(
            MindQLQuery(
                command=MindQLCommand.SELECT,
                raw_sql=_PassthroughStripSql(
                    "SELECT thought_id FROM thought WHERE priority = ?",
                ),
                select_params=("P1",),
            ),
        )
        assert [str(row["thought_id"]) for row in result.rows] == ["t-000", "t-001"]

    async def test_quoted_separator_is_still_not_a_second_statement(
        self,
        populated_db: aiosqlite.Connection,
    ) -> None:
        # The single-statement scan runs on the guard's own copy; a ``;`` inside
        # a string literal is still data, not a separator.
        store = SqliteEngravaCore(populated_db)
        result = await store.execute_mindql(
            MindQLQuery(command=MindQLCommand.SELECT, raw_sql="SELECT ';' AS s"),
        )
        assert result.rows == [{"s": ";"}]
