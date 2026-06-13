"""MindQL executor — run parsed MindQL queries against an aiosqlite database.

The executor translates ``MindQLQuery`` plans into SQL and runs them,
or routes extension commands to registered handlers.
"""

from __future__ import annotations

import contextvars
import datetime
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from engrava.mindql.parser import (
    MindQLCommand,
    MindQLOperator,
    MindQLParseError,
    TemporalPredicateKind,
)

if TYPE_CHECKING:
    import aiosqlite

    from engrava.domain.protocols.hooks import MindQLExtension
    from engrava.mindql.parser import MindQLQuery, TemporalPredicate


# Optional pinned "now" for ``valid_now`` resolution. When unset (the
# default), ``valid_now`` resolves against the server clock at execution
# time. Tests pin a deterministic instant via :func:`mindql_now.set`.
mindql_now: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "mindql_now",
    default=None,
)


def _resolve_now() -> str:
    """Resolve the instant ``valid_now`` evaluates against.

    Returns:
        The pinned instant from the ``mindql_now`` context variable when set,
        otherwise the current UTC time as an ISO-8601 string.

    """
    pinned = mindql_now.get()
    if pinned is not None:
        return pinned
    return datetime.datetime.now(datetime.UTC).isoformat()


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

# Tables carrying the bi-temporal valid-time columns. Temporal predicates are
# only meaningful against these; applying one to any other table is rejected.
_TEMPORAL_TABLES: frozenset[str] = frozenset({"thought", "edge"})

# Default row cap applied to a FIND query that carries no explicit LIMIT, so an
# unqualified ``FIND thoughts`` cannot run an unbounded scan. An explicit
# ``LIMIT`` in the query always overrides this. COUNT queries are unaffected —
# they aggregate and never materialise the row set.
DEFAULT_FIND_LIMIT = 100


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

        A ``LIMIT`` is always emitted: the query's own limit when it has one,
        otherwise :data:`DEFAULT_FIND_LIMIT` so an unqualified FIND cannot run
        an unbounded scan.

        Args:
            table: Target table name.
            query: Parsed FIND query.

        Returns:
            Tuple of (SQL string, parameter list).

        """
        clauses, params = self._build_where(table, query)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        # An explicit LIMIT wins; otherwise cap the scan at DEFAULT_FIND_LIMIT
        # so an unqualified FIND cannot run an unbounded query.
        effective_limit = query.limit if query.limit is not None else DEFAULT_FIND_LIMIT
        limit = f" LIMIT {effective_limit}"
        sql = f"SELECT * FROM {table}{where}{limit}"  # noqa: S608
        return sql, params

    def _build_where(
        self,
        table: str,
        query: MindQLQuery,
    ) -> tuple[list[str], list[object]]:
        """Build WHERE clauses and params, validating columns.

        Ordinary ``field op value`` conditions and opt-in valid-time
        temporal predicates are both emitted here, so every code path that
        builds a query body (FIND and COUNT) gets temporal filtering for
        free. The clauses are returned in source order: conditions first,
        then temporal predicates.

        Args:
            table: Target table name.
            query: Parsed query.

        Returns:
            Tuple of (clause strings, parameter values).

        Raises:
            MindQLParseError: If a condition references a disallowed column,
                or a temporal predicate targets a table without valid-time
                columns.

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

        for predicate in query.temporal_predicates:
            fragment, frag_params = self._build_temporal_clause(table, predicate)
            clauses.append(fragment)
            params.extend(frag_params)

        return clauses, params

    @staticmethod
    def _build_temporal_clause(
        table: str,
        predicate: TemporalPredicate,
    ) -> tuple[str, list[object]]:
        """Build the NULL-tolerant SQL fragment for one temporal predicate.

        NULL ``valid_from`` is an open lower bound (negative infinity) and
        NULL ``valid_until`` is an open upper bound (positive infinity).
        ``valid_at`` / ``valid_now`` / ``valid_within`` are NULL-tolerant so
        rows with an open bound stay visible; ``valid_between`` requires real
        bounds on both ends and therefore excludes open-bound rows.

        Args:
            table: Target table name (must carry valid-time columns).
            predicate: The parsed temporal predicate.

        Returns:
            Tuple of (SQL fragment, ordered parameter values).

        Raises:
            MindQLParseError: If ``table`` has no valid-time columns.

        """
        if table not in _TEMPORAL_TABLES:
            msg = f"Temporal predicate not supported for table {table!r}"
            raise MindQLParseError(msg)

        kind = predicate.kind
        if kind == TemporalPredicateKind.VALID_NOW:
            now = _resolve_now()
            return (
                "(valid_from IS NULL OR valid_from <= ?) "
                "AND (valid_until IS NULL OR valid_until > ?)",
                [now, now],
            )
        if kind == TemporalPredicateKind.VALID_AT:
            instant = predicate.start
            return (
                "(valid_from IS NULL OR valid_from <= ?) "
                "AND (valid_until IS NULL OR valid_until > ?)",
                [instant, instant],
            )
        if kind == TemporalPredicateKind.VALID_WITHIN:
            return (
                "(valid_from IS NULL OR valid_from < ?) "
                "AND (valid_until IS NULL OR valid_until > ?)",
                [predicate.end, predicate.start],
            )
        # VALID_BETWEEN — closed containment requiring real bounds on both ends.
        return (
            "valid_from IS NOT NULL AND valid_from >= ? "
            "AND valid_until IS NOT NULL AND valid_until <= ?",
            [predicate.start, predicate.end],
        )
