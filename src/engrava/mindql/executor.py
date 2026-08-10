"""MindQL executor — run parsed MindQL queries against an aiosqlite database.

The executor translates ``MindQLQuery`` plans into SQL and runs them,
or routes extension commands to registered handlers.
"""

from __future__ import annotations

import contextvars
import datetime
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeAlias

from engrava.mindql.parser import (
    BoolExpr,
    Comparison,
    InCondition,
    MindQLCommand,
    MindQLOperator,
    MindQLParseError,
    TemporalPredicate,
    TemporalPredicateKind,
    strip_string_literals,
)

if TYPE_CHECKING:
    import aiosqlite

    from engrava.domain.protocols.hooks import MindQLExtension
    from engrava.mindql.parser import MindQLQuery, WhereNode


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

# Tables a query may name, derived from the per-table column allowlist above: a
# table the executor cannot validate columns for is a table it will not
# interpolate into SQL. Every table the grammar accepts must appear here. The
# mapping is name-to-itself so a lookup yields the module's own string rather
# than the caller's object (see :func:`_guard_table`).
_ALLOWED_TABLES: dict[str, str] = {name: name for name in _ALLOWED_COLUMNS}

# Target table for a query that names none.
_DEFAULT_TABLE = "thought"

# The only sort directions that may be interpolated into an ORDER BY clause.
_SORT_DIRECTIONS: tuple[str, ...] = ("ASC", "DESC")

# The only boolean joiners that may be interpolated into a WHERE fragment.
_BOOL_JOINERS: tuple[str, ...] = ("AND", "OR")

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


def _guard_table(table: object) -> str:
    """Resolve the target table of a query and validate the identifier.

    A table name is an *identifier*: it cannot be bound as a ``?`` parameter,
    so it is interpolated into the statement and must therefore be checked
    against a closed set before it gets there. ``parse()`` already restricts
    the tables its grammar accepts, but a ``MindQLQuery`` can also be
    constructed directly and handed to the executor, so this boundary owns
    the check independently.

    The value returned is the module's own string, never the caller's object:
    a ``str`` subclass can pass an equality check and still emit arbitrary
    text from ``__format__`` when it is interpolated.

    Args:
        table: The query's target table, or ``None``/empty when it names none.

    Returns:
        The canonical table name, defaulting to ``thought``.

    Raises:
        MindQLParseError: If the value is not a string, or names a table
            outside the allowlist.

    """
    if table is not None and not isinstance(table, str):
        msg = f"Table name must be a string, got {table!r}"
        raise MindQLParseError(msg)
    resolved = table or _DEFAULT_TABLE
    canonical = _ALLOWED_TABLES.get(resolved)
    if canonical is None:
        allowed = ", ".join(sorted(_ALLOWED_TABLES))
        msg = f"Table {resolved!r} not allowed. Expected one of: {allowed}"
        raise MindQLParseError(msg)
    return canonical


def _guard_column(table: str, field: object) -> str:
    """Validate a column identifier against the table's allowlist.

    Every column name the executor emits — in a WHERE comparison, an ``IN``
    list, or an ORDER BY key — passes through here. As with the table name,
    the string returned is the allowlist's own, never the caller's object, so
    a ``str`` subclass cannot pass the membership check and then emit
    something else from ``__format__``.

    The scan is linear rather than a set lookup because it must return the
    matching member, not just a yes/no: the allowlists hold at most a dozen
    entries each, and the cost is invisible beside the query it builds.

    Args:
        table: Target table name, already validated by :func:`_guard_table`.
        field: The column the query wants to filter or sort on.

    Returns:
        The matching column name from the table's allowlist.

    Raises:
        MindQLParseError: If the value is not a string, or names a column not
            allowed for the table.

    """
    if not isinstance(field, str):
        msg = f"Column name must be a string, got {field!r}"
        raise MindQLParseError(msg)
    for column in _ALLOWED_COLUMNS.get(table, frozenset()):
        if field == column:
            return column
    msg = f"Column {field!r} not allowed for table {table!r}"
    raise MindQLParseError(msg)


def _guard_keyword(value: object, allowed: tuple[str, ...], what: str) -> str:
    """Validate an interpolated SQL keyword against a closed set.

    Sort directions and boolean joiners are keywords, not values: they cannot
    be bound as ``?`` parameters, so they are interpolated and must be checked
    first. Case is normalised, so a directly constructed query may spell a
    keyword either way, but the string returned is always the module's own
    literal — the caller's object never reaches the statement, which also
    disarms a ``str`` subclass that overrides ``__format__``.

    Args:
        value: The keyword the query supplied.
        allowed: The closed set of accepted keywords, in message order.
        what: What the keyword is, for the error message.

    Returns:
        The matching keyword from ``allowed``.

    Raises:
        MindQLParseError: If the value matches no accepted keyword.

    """
    if isinstance(value, str):
        canonical = value.upper()
        for keyword in allowed:
            if canonical == keyword:
                return keyword
    msg = f"{what} {value!r} not allowed. Expected one of: {', '.join(allowed)}"
    raise MindQLParseError(msg)


def _guard_row_bound(value: object, clause: str) -> int:
    """Validate an interpolated ``LIMIT`` / ``OFFSET`` value.

    ``MindQLQuery`` declares both as ``int``, but it is a plain dataclass and
    a directly constructed query is not type-checked at runtime, so the value
    is verified where it is interpolated. ``bool`` is rejected explicitly: it
    subclasses ``int``, passes a type checker, and reaches SQLite as the
    keyword ``True``/``False``. Negative values are rejected because SQLite
    reads ``LIMIT -1`` as "no limit" — silently removing the row cap — and a
    negative ``OFFSET`` as zero. The bound is rebuilt as a plain ``int`` so
    that an ``int`` subclass cannot emit arbitrary text from ``__format__``.

    Args:
        value: The requested bound.
        clause: ``"LIMIT"`` or ``"OFFSET"``, for the message.

    Returns:
        The validated bound, as a plain ``int``.

    Raises:
        MindQLParseError: If the value is not a non-negative, non-boolean int.

    """
    bound = int(value) if isinstance(value, int) and not isinstance(value, bool) else None
    if bound is None or bound < 0:
        msg = f"{clause} must be a non-negative integer, got {value!r}"
        raise MindQLParseError(msg)
    return bound


def _guard_select_sql(sql: object) -> str:
    """Validate a SELECT-passthrough statement and return it trimmed.

    Belt-and-suspenders read-only guard: the statement must begin with
    ``SELECT``, and it must be a *single* statement. A single trailing ``;`` is
    tolerated (and stripped); any ``;`` remaining mid-string is rejected so a
    second statement can never be smuggled in.

    Read-only-ness is decided on a value this module owns. ``str`` is
    subclassable and every method an inspection would reach for — ``strip``,
    ``upper``, ``startswith``, ``endswith``, ``__len__`` — is overridable,
    while SQLite still reads the object's real text: a guard built out of the
    caller's own methods can be told it is running a ``SELECT`` and hand a
    ``DELETE`` to the database. So the statement is normalised once through the
    unbound ``str.strip``, which reads that real text and yields a plain
    ``str``; every later check runs on that copy, and the copy is what the
    caller gets back to execute. ``isinstance`` alone would not close this —
    the subclass *is* a ``str`` — but it is still required, because
    ``str.strip`` applies to nothing else.

    Args:
        sql: The raw SQL from a SELECT passthrough query.

    Returns:
        The validated, trimmed SQL statement (without a trailing ``;``), as a
        plain ``str``.

    Raises:
        MindQLParseError: If the value is not a string, or is not a single
            SELECT statement.

    """
    if not isinstance(sql, str):
        # The message names no part of the rejected value. Representing it
        # would run the caller's ``__repr__`` on the guard's own path, and a
        # hostile one raises in place of the error this guard promises; what
        # failed and what was expected is the whole of what a caller needs.
        msg = "SELECT passthrough SQL must be a string"
        raise MindQLParseError(msg)
    # ``str.strip(sql)``, not ``sql.strip()``: the unbound built-in reads the
    # object's real text and returns a plain ``str``, so no override decides
    # what is inspected below — nor what is executed, since this is the value
    # every branch returns.
    statement = str.strip(sql)
    if not statement.upper().startswith("SELECT"):
        msg = "Only SELECT statements are allowed"
        raise MindQLParseError(msg)
    # Strip a single real trailing separator, then reject any that remain
    # mid-statement. A trailing ``;`` is a real separator (a literal-closing
    # quote would be the final char instead). The mid-statement check ignores
    # ``;`` inside quoted string literals, so a valid ``SELECT ';' AS s`` is
    # not falsely rejected.
    without_trailing = statement[:-1].rstrip() if statement.endswith(";") else statement
    if ";" in strip_string_literals(without_trailing):
        msg = "Only a single SELECT statement is allowed"
        raise MindQLParseError(msg)
    return without_trailing


# A single MindQL result row: an ordered ``{column: value}`` mapping.
#
# Cell values are heterogeneous across command types, so ``object`` (the sound,
# checked top type) is the honest upper bound rather than a scalar-only union:
#   * FIND / SELECT rows carry SQLite storage-class scalars
#     (``str | int | float | bytes | None``);
#   * COUNT carries a single ``int``;
#   * EXPLAIN carries the compiled SQL ``str`` plus its bound-parameter ``list``;
#   * extension commands may place any object their handler returns -- the
#     ``MindQLExtension.handler`` contract is already ``list[dict[str, object]]``.
# ``object`` forces consumers to narrow a cell before use, whereas ``Any`` would
# silently disable that check.
MindQLRow: TypeAlias = dict[str, object]


@dataclass(frozen=True)
class MindQLResult:
    """Result of a MindQL query execution.

    Attributes:
        columns: Column names in result order.
        rows: Result rows, each an ordered ``{column: value}`` mapping
            (:data:`MindQLRow`).
        count: For COUNT queries, the count value.
        command: The command that was executed.

    """

    columns: list[str] = field(default_factory=list)
    rows: list[MindQLRow] = field(default_factory=list)
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
            MindQLParseError: If the query carries something the executor will
                not interpolate — an unknown table or column, a sort direction
                or boolean joiner outside its keyword set, or a ``LIMIT`` /
                ``OFFSET`` that is not a non-negative integer. Also for a
                temporal predicate on a table without valid-time columns, a
                SELECT passthrough whose raw SQL is not a string or not a
                single SELECT statement, an unregistered extension command,
                and an unsupported command.
                ``MindQLQuery`` can be constructed directly, so all of this is
                validated here and not only by
                :func:`~engrava.mindql.parser.parse`.

        """
        if query.explain:
            return self._explain(query)
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

        When ``query.select_params`` is set, the statement runs with those
        bound parameters; when it is ``None`` the statement runs with no bound
        parameters, exactly as before this argument existed.

        Args:
            query: Parsed query with ``raw_sql`` set.

        Returns:
            Query result.

        Raises:
            MindQLParseError: If the raw SQL is not a string, or is not a
                single SELECT statement.

        """
        sql = _guard_select_sql(query.raw_sql)
        if query.select_params is not None:
            cursor = await self._db.execute(sql, query.select_params)
        else:
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

        Raises:
            MindQLParseError: If the query names a table, column, sort
                direction, boolean joiner or row bound the executor will not
                interpolate, or applies a temporal predicate to a table
                without valid-time columns.

        """
        table = _guard_table(query.table)
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

        Raises:
            MindQLParseError: If the query names a table, column or boolean
                joiner the executor will not interpolate, or applies a
                temporal predicate to a table without valid-time columns.

        """
        table = _guard_table(query.table)
        clauses, params = self._build_where(table, query)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT COUNT(*) AS cnt FROM {table}{where}"  # noqa: S608 -- table guarded above

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
        an unbounded scan. When an OFFSET is present without an explicit LIMIT,
        the default cap is applied as the LIMIT (SQLite requires a LIMIT for
        OFFSET). ORDER BY, when present, is emitted before LIMIT/OFFSET.

        Both bounds are interpolated rather than bound as parameters, so each
        is validated as a non-negative integer first.

        Args:
            table: Target table name, already validated by :func:`_guard_table`.
            query: Parsed FIND query.

        Returns:
            Tuple of (SQL string, parameter list).

        Raises:
            MindQLParseError: If ``limit`` or ``offset`` is not a non-negative
                integer, the WHERE / ORDER BY clauses reject an identifier or
                keyword, or a temporal predicate targets a table without
                valid-time columns.

        """
        clauses, params = self._build_where(table, query)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        order_by = self._build_order_by(table, query)
        # An explicit LIMIT wins; otherwise cap the scan at DEFAULT_FIND_LIMIT
        # so an unqualified FIND cannot run an unbounded query. OFFSET requires
        # a LIMIT in SQLite, so a query with OFFSET but no explicit LIMIT also
        # falls back to the default cap.
        if query.limit is None:
            effective_limit = DEFAULT_FIND_LIMIT
        else:
            effective_limit = _guard_row_bound(query.limit, "LIMIT")
        limit = f" LIMIT {effective_limit}"
        offset = ""
        if query.offset is not None:
            offset = f" OFFSET {_guard_row_bound(query.offset, 'OFFSET')}"
        tail = f"{where}{order_by}{limit}{offset}"
        sql = f"SELECT * FROM {table}{tail}"  # noqa: S608 -- all interpolations guarded
        return sql, params

    @staticmethod
    def _build_order_by(table: str, query: MindQLQuery) -> str:
        """Build the ORDER BY SQL fragment, validating every sort field.

        Neither part of a sort key can be bound as a ``?`` parameter, so both
        are interpolated and both are validated first: the field is a column
        identifier checked against the table allowlist, and the direction is a
        keyword checked against ``ASC``/``DESC``.

        Args:
            table: Target table name, already validated by :func:`_guard_table`.
            query: Parsed FIND query.

        Returns:
            The ``" ORDER BY …"`` fragment, or an empty string when the query
            carries no ORDER BY.

        Raises:
            MindQLParseError: If a sort field is not in the table allowlist, or
                a sort direction is neither ``ASC`` nor ``DESC``.

        """
        if not query.order_by:
            return ""
        items: list[str] = []
        for field_name, direction in query.order_by:
            column = _guard_column(table, field_name)
            keyword = _guard_keyword(direction, _SORT_DIRECTIONS, "Sort direction")
            items.append(f"{column} {keyword}")
        return f" ORDER BY {', '.join(items)}"

    def _build_where(
        self,
        table: str,
        query: MindQLQuery,
    ) -> tuple[list[str], list[object]]:
        """Build WHERE clauses and params, validating columns.

        When ``query.where`` is set (the query uses ``OR``, parentheses, or
        ``IN``) the boolean-expression tree is compiled to a single
        parenthesised clause with bound parameters. Otherwise the historical
        flat path runs byte-for-byte: ordinary ``field op value`` conditions
        and opt-in valid-time temporal predicates are emitted in source order
        (conditions first, then temporal predicates), so every code path that
        builds a query body (FIND and COUNT) gets temporal filtering for free.

        Args:
            table: Target table name.
            query: Parsed query.

        Returns:
            Tuple of (clause strings, parameter values).

        Raises:
            MindQLParseError: If a condition references a disallowed column,
                a boolean joiner is neither ``AND`` nor ``OR``, or a temporal
                predicate targets a table without valid-time columns.

        """
        if query.where is not None:
            fragment, params = self._compile_where_node(table, query.where)
            return [fragment], params

        clauses: list[str] = []
        params = []

        for cond in query.conditions:
            column = _guard_column(table, cond.field)
            op_sql = _OP_SQL[cond.operator]
            clauses.append(f"{column} {op_sql} ?")
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

    def _compile_where_node(
        self,
        table: str,
        node: WhereNode,
    ) -> tuple[str, list[object]]:
        """Compile a WHERE boolean-expression node to SQL and bound params.

        Recurses over the tree, emitting a parenthesised SQL fragment with every
        value bound as a ``?`` parameter (never interpolated). The column
        allowlist is enforced on every comparison and ``IN`` field, the
        temporal-table guard on every temporal predicate, and the boolean
        joiner — the one keyword this fragment interpolates — on every node.

        Args:
            table: Target table name, already validated by :func:`_guard_table`.
            node: The WHERE tree node to compile.

        Returns:
            Tuple of (SQL fragment, ordered parameter values).

        Raises:
            MindQLParseError: If a comparison / ``IN`` field is not allowed for
                the table, a boolean joiner is neither ``AND`` nor ``OR``, or a
                temporal predicate targets a table without valid-time columns.

        """
        if isinstance(node, BoolExpr):
            fragments: list[str] = []
            params: list[object] = []
            for operand in node.operands:
                frag, frag_params = self._compile_where_node(table, operand)
                fragments.append(frag)
                params.extend(frag_params)
            joiner = f" {_guard_keyword(node.op, _BOOL_JOINERS, 'Boolean operator')} "
            return f"({joiner.join(fragments)})", params
        if isinstance(node, Comparison):
            column = _guard_column(table, node.field)
            op_sql = _OP_SQL[node.operator]
            return f"{column} {op_sql} ?", [node.value]
        if isinstance(node, InCondition):
            column = _guard_column(table, node.field)
            placeholders = ", ".join("?" for _ in node.values)
            return f"{column} IN ({placeholders})", list(node.values)
        # TemporalPredicate — reuse the shared NULL-tolerant builder.
        fragment, temporal_params = self._build_temporal_clause(table, node)
        return f"({fragment})", temporal_params

    def _explain(self, query: MindQLQuery) -> MindQLResult:
        """Compile a query to its SQL and bound params WITHOUT executing it.

        This is a pure compile-and-return path: it never touches the database,
        never runs ``EXPLAIN QUERY PLAN``, and never executes the compiled or
        raw SQL. For FIND / COUNT it returns the parameterised SQL that would
        run; for SELECT it returns the guarded raw SQL and its declared params
        (if any).

        Args:
            query: Parsed query with ``explain`` set.

        Returns:
            A result carrying one row ``{"sql": …, "params": …}`` under the
            ``["sql", "params"]`` columns, with ``command="EXPLAIN"``.

        Raises:
            MindQLParseError: If the query cannot be compiled (for example an
                invalid table, column, row bound, or sort direction, or a
                non-SELECT SELECT passthrough).

        """
        if query.command == MindQLCommand.SELECT:
            sql = _guard_select_sql(query.raw_sql)
            params: list[object] = (
                list(query.select_params) if query.select_params is not None else []
            )
        elif query.command == MindQLCommand.FIND:
            table = _guard_table(query.table)
            sql, params = self._build_select_sql(table, query)
        elif query.command == MindQLCommand.COUNT:
            table = _guard_table(query.table)
            clauses, params = self._build_where(table, query)
            where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
            sql = f"SELECT COUNT(*) AS cnt FROM {table}{where}"  # noqa: S608 -- table guarded
        else:  # pragma: no cover - defensive: EXTENSION+EXPLAIN rejected at parse
            msg = "EXPLAIN is only supported for FIND, COUNT, and SELECT queries"
            raise MindQLParseError(msg)

        return MindQLResult(
            columns=["sql", "params"],
            rows=[{"sql": sql, "params": params}],
            command="EXPLAIN",
        )
