"""Tests for MindQL parser and executor.

Covers parsing of FIND, COUNT, SELECT, and extension commands,
as well as executor integration with aiosqlite.
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
from engrava.domain.protocols.hooks import MindQLExtension
from engrava.mindql.executor import MindQLExecutor
from engrava.mindql.parser import (
    Condition,
    MindQLCommand,
    MindQLOperator,
    MindQLParseError,
    MindQLQuery,
    parse,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


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
