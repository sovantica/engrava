"""MindQL executor — run parsed MindQL queries against an aiosqlite database.

The executor translates ``MindQLQuery`` plans into SQL and runs them,
or routes extension commands to registered handlers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from engrava.mindql.parser import MindQLCommand, MindQLOperator, MindQLParseError

if TYPE_CHECKING:
    import aiosqlite

    from engrava.domain.protocols.hooks import MindQLExtension
    from engrava.mindql.parser import MindQLQuery


# Columns that are safe to filter on (allowlist per table).
_ALLOWED_COLUMNS: dict[str, frozenset[str]] = {
    "thought": frozenset(
        {
            "thought_id",
            "thought_type",
            "essence",
            "content",
            "priority",
            "lifecycle_status",
            "created_cycle",
            "updated_cycle",
            "source",
            "confidence",
            "source_type",
            "visibility",
            "confirmation_count",
        }
    ),
    "edge": frozenset(
        {
            "edge_id",
            "from_thought_id",
            "to_thought_id",
            "edge_type",
            "weight",
            "created_cycle",
            "source",
            "decay_multiplier",
        }
    ),
    "embedding": frozenset(
        {
            "embedding_id",
            "owner_type",
            "owner_id",
            "model_name",
            "dimension",
        }
    ),
    "action": frozenset(
        {
            "action_id",
            "source_thought_id",
            "action_type",
            "intent",
            "status",
            "verification_status",
        }
    ),
}

_OP_SQL: dict[MindQLOperator, str] = {
    MindQLOperator.EQ: "=",
    MindQLOperator.NE: "!=",
    MindQLOperator.GT: ">",
    MindQLOperator.LT: "<",
    MindQLOperator.GE: ">=",
    MindQLOperator.LE: "<=",
}


@dataclass(frozen=True)
class MindQLResult:
    """Result of a MindQL query execution.

    Attributes:
        columns: Column names in result order.
        rows: List of row dicts.
        count: For COUNT queries, the count value.
        command: The command that was executed.

    """

    columns: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    count: int | None = None
    command: str = ""


class MindQLExecutor:
    """Execute MindQL queries against an aiosqlite connection.

    Args:
        db: Open aiosqlite connection with row_factory set.
        extensions: Registered MindQL extension commands.

    """

    def __init__(
        self,
        db: aiosqlite.Connection,
        extensions: dict[str, MindQLExtension] | None = None,
    ) -> None:
        self._db = db
        self._extensions: dict[str, MindQLExtension] = extensions or {}

    async def execute(self, query: MindQLQuery) -> MindQLResult:
        """Execute a parsed MindQL query.

        Args:
            query: A parsed ``MindQLQuery`` plan.

        Returns:
            A ``MindQLResult`` with columns, rows, and/or count.

        Raises:
            MindQLParseError: If the query references invalid columns.

        """
        if query.command == MindQLCommand.SELECT:
            return await self._execute_select(query)
        if query.command == MindQLCommand.FIND:
            return await self._execute_find(query)
        if query.command == MindQLCommand.COUNT:
            return await self._execute_count(query)
        if query.command == MindQLCommand.EXTENSION:
            return await self._execute_extension(query)

        msg = f"Unsupported command: {query.command}"
        raise MindQLParseError(msg)

    async def _execute_select(self, query: MindQLQuery) -> MindQLResult:
        """Execute a SELECT passthrough query (read-only).

        Args:
            query: Parsed query with ``raw_sql`` set.

        Returns:
            Query result.

        Raises:
            MindQLParseError: If the SQL is not a SELECT statement.

        """
        sql = query.raw_sql or ""
        normalized = sql.strip().upper()
        if not normalized.startswith("SELECT"):
            msg = "Only SELECT statements are allowed"
            raise MindQLParseError(msg)

        cursor = await self._db.execute(sql)
        raw_rows = await cursor.fetchall()
        keys = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = [dict(zip(keys, row, strict=True)) for row in raw_rows]
        return MindQLResult(columns=keys, rows=rows, command="SELECT")

    async def _execute_find(self, query: MindQLQuery) -> MindQLResult:
        """Execute a FIND query.

        Args:
            query: Parsed FIND query.

        Returns:
            Query result with matching rows.

        """
        table = query.table or "thought"
        sql, params = self._build_select_sql(table, query)

        cursor = await self._db.execute(sql, params)
        raw_rows = await cursor.fetchall()
        keys = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = [dict(zip(keys, row, strict=True)) for row in raw_rows]
        return MindQLResult(columns=keys, rows=rows, command="FIND")

    async def _execute_count(self, query: MindQLQuery) -> MindQLResult:
        """Execute a COUNT query.

        Args:
            query: Parsed COUNT query.

        Returns:
            Query result with count value.

        """
        table = query.table or "thought"
        clauses, params = self._build_where(table, query)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT COUNT(*) AS cnt FROM {table}{where}"  # noqa: S608

        cursor = await self._db.execute(sql, params)
        row = await cursor.fetchone()
        count = int(row[0]) if row else 0
        return MindQLResult(
            columns=["count"],
            rows=[{"count": count}],
            count=count,
            command="COUNT",
        )

    async def _execute_extension(self, query: MindQLQuery) -> MindQLResult:
        """Execute an extension command.

        Args:
            query: Parsed extension query.

        Returns:
            Query result from the extension handler.

        Raises:
            MindQLParseError: If the extension command is not registered.

        """
        name = query.extension_name or ""
        ext = self._extensions.get(name)
        if ext is None:
            msg = f"Unknown extension command: {name!r}"
            raise MindQLParseError(msg)

        result = await ext.handler(self._db, query.extension_args)
        rows = result if isinstance(result, list) else []
        keys = list(rows[0].keys()) if rows else []
        return MindQLResult(columns=keys, rows=rows, command=name)

    def _build_select_sql(
        self,
        table: str,
        query: MindQLQuery,
    ) -> tuple[str, list[object]]:
        """Build a parameterized SELECT SQL from a FIND query.

        Args:
            table: Target table name.
            query: Parsed FIND query.

        Returns:
            Tuple of (SQL string, parameter list).

        """
        clauses, params = self._build_where(table, query)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        limit = f" LIMIT {query.limit}" if query.limit is not None else ""
        sql = f"SELECT * FROM {table}{where}{limit}"  # noqa: S608
        return sql, params

    def _build_where(
        self,
        table: str,
        query: MindQLQuery,
    ) -> tuple[list[str], list[object]]:
        """Build WHERE clauses and params, validating columns.

        Args:
            table: Target table name.
            query: Parsed query.

        Returns:
            Tuple of (clause strings, parameter values).

        Raises:
            MindQLParseError: If a condition references a disallowed column.

        """
        allowed = _ALLOWED_COLUMNS.get(table, frozenset())
        clauses: list[str] = []
        params: list[object] = []

        for cond in query.conditions:
            if cond.field not in allowed:
                msg = f"Column {cond.field!r} not allowed for table {table!r}"
                raise MindQLParseError(msg)
            op_sql = _OP_SQL[cond.operator]
            clauses.append(f"{cond.field} {op_sql} ?")
            params.append(cond.value)

        return clauses, params
