"""MindQL parser — parse MindQL queries into an executable plan.

Supports three query forms:

1. **FIND** — retrieve rows::

       FIND thoughts WHERE lifecycle_status = 'ACTIVE' LIMIT 10
       FIND edges WHERE from_thought_id = 'abc'

2. **COUNT** — aggregate count::

       COUNT thoughts WHERE priority = 'P1'

3. **SELECT** — passthrough SQL (read-only)::

       SELECT thought_id, essence FROM thought WHERE lifecycle_status = 'ACTIVE'

4. **Extension commands** — routed to registered handlers::

       CUSTOM_CMD arg

The WHERE clause also accepts opt-in **temporal predicates** that filter rows
by their valid-time interval (the ``valid_from`` / ``valid_until`` columns on
the ``thought`` and ``edge`` tables). They are NULL-tolerant: a row with an
open lower or upper bound stays visible::

       FIND thoughts WHERE valid_now
       FIND edges WHERE valid_at '2025-01-01T00:00:00+00:00'
       FIND thoughts WHERE priority = 'P1' AND valid_within '2025-01-01' '2025-02-01'
       FIND edges WHERE valid_between '2025-01-01' '2025-12-31'

A WHERE clause may combine operands with ``AND``, ``OR``, and parenthesised
grouping, following SQLite-standard precedence (``AND`` binds tighter than
``OR``)::

       FIND thoughts WHERE priority = 'P1' OR lifecycle_status = 'ACTIVE'
       FIND thoughts WHERE (priority = 'P1' OR priority = 'P2') AND source = 'x'
       FIND thoughts WHERE thought_type IN ('BELIEF', 'OBSERVATION')

A ``FIND`` may additionally carry ``ORDER BY`` and ``OFFSET`` clauses, and any
read query may be prefixed with ``EXPLAIN`` to return the compiled SQL and
bound parameters *without executing it*::

       FIND thoughts WHERE priority = 'P1' ORDER BY created_cycle DESC LIMIT 10 OFFSET 20
       EXPLAIN FIND thoughts WHERE thought_type IN ('BELIEF')

Grammar (simplified BNF)::

    query          := [ "EXPLAIN" ] ( find_query | count_query
                    | select_query ) | extension_query
    find_query     := "FIND" table_name [where_clause] [order_by_clause]
                      [limit_clause] [offset_clause]
    count_query    := "COUNT" table_name [where_clause]
    select_query   := "SELECT" <raw SQL>
    extension_query:= COMMAND_NAME [args...]

    table_name     := "thoughts" | "thought" | "edges" | "edge"
                    | "embeddings" | "embedding" | "actions" | "action"
    where_clause   := "WHERE" bool_expr
    bool_expr      := or_expr
    or_expr        := and_expr ("OR" and_expr)*
    and_expr       := operand ("AND" operand)*
    operand        := "(" bool_expr ")" | condition | in_condition
                    | temporal_predicate
    condition      := field_name operator value
    in_condition   := field_name "IN" "(" value ("," value)* ")"
    temporal_pred  := "valid_now"
                    | "valid_at" timestamp
                    | "valid_within" timestamp timestamp
                    | "valid_between" timestamp timestamp
    operator       := "=" | "!=" | ">" | "<" | ">=" | "<="
    value          := quoted_string | number
    timestamp      := quoted_string | bare_iso8601_token
    quoted_string  := "'" <chars> "'"
    order_by_clause:= "ORDER" "BY" sort_item ("," sort_item)*
    sort_item      := field_name [ "ASC" | "DESC" ]
    limit_clause   := "LIMIT" integer
    offset_clause  := "OFFSET" integer

A WHERE that uses **only** simple comparisons and/or temporal predicates joined
by ``AND`` (no ``OR``, no parentheses, no ``IN``) populates the flat
``conditions`` + ``temporal_predicates`` lists and leaves ``where`` as ``None``,
so execution follows the historical path byte-for-byte. Any use of ``OR``,
parentheses, or ``IN`` instead populates the ``where`` boolean-expression tree.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

from engrava.domain.models._temporal import validate_iso8601_nullable


class MindQLCommand(StrEnum):
    """Supported MindQL command verbs."""

    FIND = "FIND"
    COUNT = "COUNT"
    SELECT = "SELECT"
    EXTENSION = "EXTENSION"


class MindQLOperator(StrEnum):
    """Comparison operators for WHERE conditions."""

    EQ = "="
    NE = "!="
    GT = ">"
    LT = "<"
    GE = ">="
    LE = "<="


# Canonical table names (plural → actual SQLite table)
_TABLE_MAP: dict[str, str] = {
    "thoughts": "thought",
    "thought": "thought",
    "edges": "edge",
    "edge": "edge",
    "embeddings": "embedding",
    "embedding": "embedding",
    "actions": "action",
    "action": "action",
}


@dataclass(frozen=True)
class Condition:
    """A single WHERE condition.

    Attributes:
        field: Column name.
        operator: Comparison operator.
        value: Comparison value (string, int, or float).

    """

    field: str
    operator: MindQLOperator
    value: str | int | float


class TemporalPredicateKind(StrEnum):
    """Kinds of valid-time predicate supported in a WHERE clause.

    Each kind filters rows by their valid-time interval — the
    ``valid_from`` / ``valid_until`` columns present on the ``thought`` and
    ``edge`` tables — without referencing those columns as ordinary
    filterable values.
    """

    VALID_NOW = "valid_now"
    VALID_AT = "valid_at"
    VALID_WITHIN = "valid_within"
    VALID_BETWEEN = "valid_between"


@dataclass(frozen=True)
class TemporalPredicate:
    """A single valid-time predicate in a WHERE clause.

    Attributes:
        kind: Which temporal predicate this is.
        start: First ISO-8601 timestamp argument, or ``None`` for
            ``valid_now`` (which carries no argument and resolves against
            the current instant at execution time).
        end: Second ISO-8601 timestamp argument, present only for the
            two-argument predicates (``valid_within`` / ``valid_between``).

    """

    kind: TemporalPredicateKind
    start: str | None = None
    end: str | None = None


@dataclass(frozen=True)
class Comparison:
    """A ``field op value`` operand in a WHERE boolean-expression tree.

    This mirrors :class:`Condition` but is a distinct node type so the flat
    ``conditions`` list (historical path) and the ``where`` tree (new path)
    never share mutable state.

    Attributes:
        field: Column name (allowlist-checked at execution time).
        operator: Comparison operator.
        value: Comparison value (string, int, or float).

    """

    field: str
    operator: MindQLOperator
    value: str | int | float


@dataclass(frozen=True)
class InCondition:
    """A ``field IN (v1, v2, …)`` operand in a WHERE boolean-expression tree.

    Attributes:
        field: Column name (allowlist-checked at execution time).
        values: The tuple of membership values. Each follows the same quoting
            rules as a comparison value: single-quoted stays a verbatim string,
            unquoted is coerced to ``int`` / ``float`` / ``str``. Never empty
            (an empty ``IN ()`` is a parse error).

    """

    field: str
    values: tuple[str | int | float, ...]


@dataclass(frozen=True)
class BoolExpr:
    """A boolean combination of WHERE operands.

    Grouping (parentheses) in the source is represented by nesting: a
    parenthesised sub-expression becomes a nested :class:`BoolExpr`. SQLite
    precedence (``AND`` binds tighter than ``OR``) is encoded structurally —
    an ``OR`` node's operands are the ``AND`` groups.

    Attributes:
        op: The boolean operator joining the operands (``"AND"`` or ``"OR"``).
        operands: The joined operands, in source order. Always at least two.

    """

    op: Literal["AND", "OR"]
    operands: tuple[WhereNode, ...]


# A node in the WHERE boolean-expression tree. Leaves are comparisons,
# ``IN`` conditions, and temporal predicates; interior nodes are ``BoolExpr``.
WhereNode = Comparison | InCondition | TemporalPredicate | BoolExpr


@dataclass(frozen=True)
class MindQLQuery:
    """Parsed MindQL query plan.

    Attributes:
        command: The MindQL command verb.
        table: Target table (canonical name, e.g. ``"thought"``).
        conditions: Flat WHERE conditions. Populated only on the historical
            pure-``AND`` path (``where`` is ``None``); empty when the query
            uses the boolean-expression tree.
        temporal_predicates: Valid-time predicates parsed from the WHERE
            clause. Populated only on the historical pure-``AND`` path
            (``where`` is ``None``); empty when the query uses the tree.
        where: The WHERE boolean-expression tree, or ``None``. It is ``None``
            whenever the WHERE uses only simple comparisons and/or temporal
            predicates joined by ``AND`` (the historical grammar), in which
            case ``conditions`` + ``temporal_predicates`` are populated instead
            and execution follows the byte-identical historical path. It is a
            tree only when the WHERE contains ``OR``, parentheses, or ``IN``.
        order_by: Ordered ``(field, direction)`` sort keys where direction is
            ``"ASC"`` or ``"DESC"``. FIND only; empty when absent.
        limit: Optional LIMIT clause.
        offset: Optional OFFSET clause (FIND only). ``None`` when absent.
        explain: When ``True`` the query is compiled to its SQL and bound
            parameters and returned as data *without being executed*.
        raw_sql: Original SQL for SELECT passthrough.
        select_params: Optional bound parameters for a SELECT passthrough. Set
            programmatically by the caller (never parsed from MQL text). When
            ``None`` the SELECT runs with no bound parameters, exactly as
            before this field existed.
        extension_name: Extension command name (for EXTENSION type).
        extension_args: Extension command arguments.

    """

    command: MindQLCommand
    table: str | None = None
    conditions: list[Condition] = field(default_factory=list)
    temporal_predicates: list[TemporalPredicate] = field(default_factory=list)
    where: WhereNode | None = None
    order_by: tuple[tuple[str, str], ...] = ()
    limit: int | None = None
    offset: int | None = None
    explain: bool = False
    raw_sql: str | None = None
    select_params: tuple[object, ...] | None = None
    extension_name: str | None = None
    extension_args: list[str] = field(default_factory=list)


class MindQLParseError(Exception):
    """Raised when a MindQL query cannot be parsed.

    Args:
        message: Human-readable description of the parse error.

    """


# Regex for tokenizing conditions: field op value
_CONDITION_RE = re.compile(
    r"(\w+)\s*(!=|>=|<=|=|>|<)\s*(?:'([^']*)'|(\S+))",
)

# Number of timestamp arguments each temporal predicate carries.
_TEMPORAL_ARITY: dict[TemporalPredicateKind, int] = {
    TemporalPredicateKind.VALID_NOW: 0,
    TemporalPredicateKind.VALID_AT: 1,
    TemporalPredicateKind.VALID_WITHIN: 2,
    TemporalPredicateKind.VALID_BETWEEN: 2,
}

# A timestamp argument: a single-quoted string OR a bare whitespace-free token.
_TIMESTAMP_ARG_RE = re.compile(r"'([^']*)'|(\S+)")

_OPERATOR_MAP: dict[str, MindQLOperator] = {
    "=": MindQLOperator.EQ,
    "!=": MindQLOperator.NE,
    ">": MindQLOperator.GT,
    "<": MindQLOperator.LT,
    ">=": MindQLOperator.GE,
    "<=": MindQLOperator.LE,
}


def parse(
    mql: str,
    *,
    known_extensions: set[str] | None = None,
) -> MindQLQuery:
    """Parse a MindQL query string into a ``MindQLQuery`` plan.

    Args:
        mql: The MindQL query string.
        known_extensions: Set of registered extension command names
            (uppercase). Used to detect extension queries.

    Returns:
        A parsed ``MindQLQuery`` ready for execution.

    Raises:
        MindQLParseError: If the query is malformed.

    """
    stripped = mql.strip()
    if not stripped:
        msg = "Empty query"
        raise MindQLParseError(msg)

    # --- Optional EXPLAIN prefix (non-executing) ---
    explain = False
    explain_match = re.match(r"EXPLAIN(\s+|$)", stripped, re.IGNORECASE)
    if explain_match:
        explain = True
        stripped = stripped[explain_match.end() :].strip()
        if not stripped:
            msg = "EXPLAIN requires a query"
            raise MindQLParseError(msg)

    upper = stripped.upper()

    # --- SELECT passthrough ---
    if upper.startswith("SELECT"):
        return MindQLQuery(
            command=MindQLCommand.SELECT,
            raw_sql=stripped,
            explain=explain,
        )

    tokens = stripped.split()
    verb = tokens[0].upper()

    # --- Extension command ---
    extensions = known_extensions or set()
    if verb in extensions:
        if explain:
            msg = "EXPLAIN is only supported for FIND, COUNT, and SELECT queries"
            raise MindQLParseError(msg)
        return MindQLQuery(
            command=MindQLCommand.EXTENSION,
            extension_name=verb,
            extension_args=tokens[1:],
        )

    # --- FIND / COUNT ---
    if verb not in ("FIND", "COUNT"):
        msg = f"Unknown command: {tokens[0]!r}. Expected FIND, COUNT, SELECT, or extension command."
        raise MindQLParseError(msg)

    command = MindQLCommand(verb)

    if len(tokens) < 2:  # noqa: PLR2004
        msg = f"{verb} requires a table name"
        raise MindQLParseError(msg)

    table_raw = tokens[1].lower()
    if table_raw not in _TABLE_MAP:
        msg = f"Unknown table: {tokens[1]!r}. Expected: {', '.join(sorted(_TABLE_MAP))}"
        raise MindQLParseError(msg)
    table = _TABLE_MAP[table_raw]

    # Parse remainder for WHERE / ORDER BY / LIMIT / OFFSET.
    remainder = " ".join(tokens[2:])
    parsed = _parse_clauses(remainder, command=command)

    return MindQLQuery(
        command=command,
        table=table,
        conditions=parsed.conditions,
        temporal_predicates=parsed.temporal_predicates,
        where=parsed.where,
        order_by=parsed.order_by,
        limit=parsed.limit,
        offset=parsed.offset,
        explain=explain,
    )


@dataclass(frozen=True)
class _ParsedClauses:
    """The parsed WHERE / ORDER BY / LIMIT / OFFSET tail of a FIND/COUNT query.

    Attributes:
        conditions: Flat conditions (populated only on the pure-``AND`` path).
        temporal_predicates: Flat temporal predicates (pure-``AND`` path only).
        where: The boolean-expression tree, or ``None`` on the pure-``AND``
            path.
        order_by: Ordered sort keys, empty when no ORDER BY was present.
        limit: The LIMIT value, or ``None``.
        offset: The OFFSET value, or ``None``.

    """

    conditions: list[Condition]
    temporal_predicates: list[TemporalPredicate]
    where: WhereNode | None
    order_by: tuple[tuple[str, str], ...]
    limit: int | None
    offset: int | None


def _parse_clauses(text: str, *, command: MindQLCommand) -> _ParsedClauses:
    """Parse the WHERE / ORDER BY / LIMIT / OFFSET tail of a FIND/COUNT query.

    ORDER BY and OFFSET are FIND-only; COUNT returns a scalar and rejects both.
    The WHERE clause takes the historical flat path (populating ``conditions`` +
    ``temporal_predicates`` with ``where`` left ``None``) when it uses only
    simple comparisons and/or temporal predicates joined by ``AND``; any use of
    ``OR``, parentheses, or ``IN`` instead produces a boolean-expression
    ``where`` tree and leaves the flat lists empty.

    Args:
        text: Everything after the table name.
        command: The owning command (``FIND`` or ``COUNT``), used to gate the
            FIND-only ORDER BY and OFFSET clauses.

    Returns:
        The parsed clause bundle.

    Raises:
        MindQLParseError: If a clause is malformed, or an ORDER BY / OFFSET is
            used with COUNT.

    """
    if not text.strip():
        return _ParsedClauses(
            conditions=[],
            temporal_predicates=[],
            where=None,
            order_by=(),
            limit=None,
            offset=None,
        )

    where_text, order_by, limit, offset = _strip_trailing_clauses(text, command)

    conditions: list[Condition] = []
    temporal_predicates: list[TemporalPredicate] = []
    where: WhereNode | None = None

    upper = where_text.upper()
    if upper.startswith("WHERE"):
        body = where_text[5:].strip()  # skip "WHERE"
        if not body:
            msg = "WHERE requires at least one condition"
            raise MindQLParseError(msg)
        if _needs_tree(body):
            where = _parse_bool_expr(body)
        else:
            conditions, temporal_predicates = _parse_flat_where(body)
    elif where_text.strip():
        msg = f"Expected WHERE, ORDER BY, LIMIT, or OFFSET, got: {where_text.strip()!r}"
        raise MindQLParseError(msg)

    return _ParsedClauses(
        conditions=conditions,
        temporal_predicates=temporal_predicates,
        where=where,
        order_by=order_by,
        limit=limit,
        offset=offset,
    )


def _strip_trailing_clauses(
    text: str,
    command: MindQLCommand,
) -> tuple[str, tuple[tuple[str, str], ...], int | None, int | None]:
    """Peel the trailing OFFSET / LIMIT / ORDER BY clauses off a FIND/COUNT tail.

    Clauses are stripped from the end in reverse source order (OFFSET, then
    LIMIT, then ORDER BY) so what remains is exactly the WHERE clause. ORDER BY
    and OFFSET are FIND-only.

    Args:
        text: Everything after the table name.
        command: The owning command, used to gate the FIND-only clauses.

    Returns:
        Tuple of (remaining WHERE text, order_by pairs, limit, offset).

    Raises:
        MindQLParseError: If ORDER BY / OFFSET is used with COUNT.

    """
    offset: int | None = None
    offset_match = re.search(r"\bOFFSET\s+(\d+)\s*$", text, re.IGNORECASE)
    if offset_match:
        if command is not MindQLCommand.FIND:
            msg = "OFFSET is only supported for FIND queries"
            raise MindQLParseError(msg)
        offset = int(offset_match.group(1))
        text = text[: offset_match.start()].strip()

    limit: int | None = None
    limit_match = re.search(r"\bLIMIT\s+(\d+)\s*$", text, re.IGNORECASE)
    if limit_match:
        limit = int(limit_match.group(1))
        text = text[: limit_match.start()].strip()

    order_by: tuple[tuple[str, str], ...] = ()
    order_match = re.search(r"\bORDER\s+BY\s+(.+?)\s*$", text, re.IGNORECASE)
    if order_match:
        if command is not MindQLCommand.FIND:
            msg = "ORDER BY is only supported for FIND queries"
            raise MindQLParseError(msg)
        order_by = _parse_order_by(order_match.group(1))
        text = text[: order_match.start()].strip()

    return text, order_by, limit, offset


def _parse_flat_where(
    where_text: str,
) -> tuple[list[Condition], list[TemporalPredicate]]:
    """Parse a pure-``AND`` WHERE into flat condition + temporal-predicate lists.

    This is the historical path, byte-for-byte unchanged: each ``AND``-separated
    part is tried as a temporal predicate first, then as an ordinary condition.
    Reached only when the WHERE contains no ``OR``, parentheses, or ``IN``.

    Args:
        where_text: The WHERE clause body (``WHERE`` keyword already stripped).

    Returns:
        Tuple of (conditions, temporal_predicates) in source order.

    Raises:
        MindQLParseError: If a fragment is neither a temporal predicate nor a
            valid condition.

    """
    conditions: list[Condition] = []
    temporal_predicates: list[TemporalPredicate] = []
    parts = re.split(r"\bAND\b", where_text, flags=re.IGNORECASE)
    for part in parts:
        stripped_part = part.strip()
        if not stripped_part:
            continue
        # Recognise temporal predicates before the ordinary condition
        # grammar — they carry keyword + bare timestamp args (no operator)
        # and so never match ``_CONDITION_RE``.
        temporal = _try_parse_temporal_predicate(stripped_part)
        if temporal is not None:
            temporal_predicates.append(temporal)
            continue
        conditions.append(_parse_condition(stripped_part))
    return conditions, temporal_predicates


def strip_string_literals(text: str) -> str:
    """Return ``text`` with single-quoted string literals blanked out.

    Used by lexical scans — tree-path detection and the SELECT single-statement
    guard — so that grammar keywords or statement separators appearing *inside*
    a quoted literal (for example ``'A OR B'`` or ``';'``) are never mistaken
    for real grammar. The SQL doubled-quote escape (``''`` inside a literal) is
    handled: it stays part of the string and does not close it.

    Args:
        text: The raw MindQL / SQL fragment.

    Returns:
        ``text`` with the contents of every single-quoted literal (and the
        surrounding quotes) removed, so only the non-string "skeleton" remains.

    """
    out: list[str] = []
    i = 0
    n = len(text)
    in_str = False
    while i < n:
        ch = text[i]
        if in_str:
            if ch == "'":
                if i + 1 < n and text[i + 1] == "'":
                    # Doubled '' — an escaped quote; stays inside the literal.
                    i += 2
                    continue
                in_str = False
            i += 1
            continue
        if ch == "'":
            in_str = True
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _has_unterminated_string(text: str) -> bool:
    """Report whether ``text`` ends with a single-quoted literal left open.

    Walks the same way as :func:`strip_string_literals` (honouring the ``''``
    escape) and reports whether a string literal was opened but never closed.
    An unterminated literal must take the boolean-tree path so the tokeniser
    raises the precise ``Unterminated string`` error rather than the lenient
    flat path silently accepting the dangling quote as part of a value.

    Args:
        text: The raw MindQL / SQL fragment.

    Returns:
        ``True`` when a string literal is left unterminated, otherwise
        ``False``.

    """
    i = 0
    n = len(text)
    in_str = False
    while i < n:
        ch = text[i]
        if in_str:
            if ch == "'":
                if i + 1 < n and text[i + 1] == "'":
                    i += 2
                    continue
                in_str = False
        elif ch == "'":
            in_str = True
        i += 1
    return in_str


def _needs_tree(where_text: str) -> bool:
    """Report whether a WHERE clause needs the boolean-expression tree path.

    The tree path is required whenever the WHERE contains ``OR``, a parenthesis,
    or an ``IN`` operator (word-boundary matched, case-insensitive) **outside of
    a quoted string literal**, or contains an **unterminated** string literal
    (which the tree tokeniser rejects with a precise error). A WHERE that uses
    only simple comparisons and/or temporal predicates joined by ``AND`` takes
    the historical flat path instead — including when a string value happens to
    contain the text ``OR`` / ``IN`` / parentheses (those live inside a literal
    and are ignored).

    Args:
        where_text: The WHERE clause body.

    Returns:
        ``True`` when the tree path is required, otherwise ``False``.

    """
    if _has_unterminated_string(where_text):
        return True
    skeleton = strip_string_literals(where_text)
    if "(" in skeleton or ")" in skeleton:
        return True
    return re.search(r"\b(OR|IN)\b", skeleton, re.IGNORECASE) is not None


def _parse_order_by(text: str) -> tuple[tuple[str, str], ...]:
    """Parse the body of an ORDER BY clause into ``(field, direction)`` pairs.

    Args:
        text: Everything after ``ORDER BY`` (comma-separated sort items).

    Returns:
        Ordered ``(field, direction)`` pairs; direction is ``"ASC"`` or
        ``"DESC"`` (default ``"ASC"``). Field identifiers are validated against
        the table allowlist at execution time, never bound as parameters.

    Raises:
        MindQLParseError: If a sort item is empty or malformed.

    """
    items: list[tuple[str, str]] = []
    for raw in text.split(","):
        tokens = raw.split()
        if not tokens:
            msg = "Malformed ORDER BY clause"
            raise MindQLParseError(msg)
        field_name = tokens[0]
        if not re.fullmatch(r"\w+", field_name):
            msg = f"Invalid ORDER BY field: {field_name!r}"
            raise MindQLParseError(msg)
        direction = "ASC"
        if len(tokens) == 2:  # noqa: PLR2004
            direction_token = tokens[1].upper()
            if direction_token not in ("ASC", "DESC"):
                msg = f"Invalid ORDER BY direction: {tokens[1]!r}"
                raise MindQLParseError(msg)
            direction = direction_token
        elif len(tokens) > 2:  # noqa: PLR2004
            msg = f"Malformed ORDER BY item: {raw.strip()!r}"
            raise MindQLParseError(msg)
        items.append((field_name, direction))
    return tuple(items)


def _parse_condition(part: str) -> Condition:
    """Parse a single ``field op value`` WHERE condition.

    Args:
        part: One ``AND``-separated WHERE fragment.

    Returns:
        The parsed condition.

    Raises:
        MindQLParseError: If the fragment is not a valid condition.

    """
    match = _CONDITION_RE.fullmatch(part)
    if not match:
        msg = f"Invalid condition: {part!r}"
        raise MindQLParseError(msg)
    field_name = match.group(1)
    op_str = match.group(2)
    # group 3 = quoted value, group 4 = unquoted value. A single-quoted literal
    # is taken verbatim as a string; only the unquoted bare value is coerced to
    # int/float, so e.g. ``'007'`` stays the string ``"007"`` instead of int 7.
    quoted_value = match.group(3)
    value: str | int | float = (
        quoted_value if quoted_value is not None else _coerce_value(match.group(4))
    )
    return Condition(
        field=field_name,
        operator=_OPERATOR_MAP[op_str],
        value=value,
    )


# --- Boolean-expression (tree) WHERE parser --------------------------------
#
# Used only when the WHERE clause contains ``OR``, parentheses, or ``IN``.
# The clause is tokenised into structural tokens (``(``, ``)``, ``AND``, ``OR``)
# and operand fragments, then parsed by recursive descent as OR-of-ANDs with
# parenthesised grouping (SQLite precedence: AND binds tighter than OR).

# Matches ``field IN (`` at the start of an operand, so the parenthesised value
# list is consumed as part of the operand rather than mistaken for grouping.
_IN_HEAD_RE = re.compile(r"\w+\s+IN\s*\(", re.IGNORECASE)

# A single value inside an ``IN (...)`` list: single-quoted string or bare token.
_IN_VALUE_RE = re.compile(r"\s*(?:'([^']*)'|([^,()'\s]+))\s*")


# A structural boolean keyword (``AND`` / ``OR``) at the current scan position.
_BOOL_KEYWORD_RE = re.compile(r"(AND|OR)\b", re.IGNORECASE)


class _WhereTokenizer:
    """Split a WHERE clause into structural tokens and operand fragments.

    The output stream contains the literal tokens ``"("``, ``")"``, ``"AND"``,
    and ``"OR"`` (uppercased) plus operand fragments (comparison / ``IN`` /
    temporal-predicate text) as raw substrings. Single-quoted strings are
    scanned atomically so a quoted value containing a parenthesis or the words
    ``AND`` / ``OR`` is never split. An ``IN (...)`` value list is captured
    whole as part of its operand fragment, so its parentheses are not confused
    with grouping.

    Args:
        text: The WHERE clause body.

    """

    def __init__(self, text: str) -> None:
        self._text = text
        self._pos = 0
        self._tokens: list[str] = []
        self._operand_start: int | None = None

    def _flush(self, end: int) -> None:
        """Emit any pending operand fragment ending at ``end``."""
        if self._operand_start is not None:
            fragment = self._text[self._operand_start : end].strip()
            if fragment:
                self._tokens.append(fragment)
            self._operand_start = None

    def _skip_quoted(self) -> None:
        """Advance past a single-quoted string, which belongs to an operand."""
        if self._operand_start is None:
            self._operand_start = self._pos
        end_quote = self._text.find("'", self._pos + 1)
        if end_quote == -1:
            msg = "Unterminated string literal in WHERE clause"
            raise MindQLParseError(msg)
        self._pos = end_quote + 1

    def _try_in_head(self) -> bool:
        """Consume an ``IN (...)`` operand head atomically. Return whether one hit."""
        if self._operand_start is not None:
            return False
        in_match = _IN_HEAD_RE.match(self._text, self._pos)
        if not in_match:
            return False
        self._operand_start = self._pos
        self._pos = _consume_in_list(self._text, in_match.end())
        return True

    def _try_paren(self, char: str) -> bool:
        """Emit a grouping parenthesis token. Return whether one was emitted."""
        if char not in "()":
            return False
        self._flush(self._pos)
        self._tokens.append(char)
        self._pos += 1
        return True

    def _try_keyword(self) -> bool:
        """Emit a boolean keyword token. Return whether one was emitted."""
        keyword_match = _BOOL_KEYWORD_RE.match(self._text, self._pos)
        if not keyword_match:
            return False
        # A keyword only breaks the stream when not inside an operand fragment.
        self._flush(self._pos)
        self._tokens.append(keyword_match.group(1).upper())
        self._pos = keyword_match.end()
        return True

    def tokenize(self) -> list[str]:
        """Run the scan and return the token stream.

        Returns:
            The token stream.

        Raises:
            MindQLParseError: If a single-quoted string is unterminated.

        """
        length = len(self._text)
        while self._pos < length:
            char = self._text[self._pos]
            if char.isspace():
                self._pos += 1
                continue
            if self._try_in_head():
                continue
            if self._try_paren(char):
                continue
            if char == "'":
                self._skip_quoted()
                continue
            if self._try_keyword():
                continue
            if self._operand_start is None:
                self._operand_start = self._pos
            self._pos += 1
        self._flush(length)
        return self._tokens


def _tokenize_where(text: str) -> list[str]:
    """Tokenise a WHERE clause into structural tokens and operand fragments.

    Args:
        text: The WHERE clause body.

    Returns:
        The token stream (see :class:`_WhereTokenizer`).

    Raises:
        MindQLParseError: If a single-quoted string is unterminated.

    """
    return _WhereTokenizer(text).tokenize()


def _consume_in_list(text: str, start: int) -> int:
    """Return the index just past the closing ``)`` of an ``IN (...)`` list.

    Args:
        text: The full WHERE clause body.
        start: Index just after the opening ``(`` of the value list.

    Returns:
        Index immediately after the matching closing ``)``.

    Raises:
        MindQLParseError: If the list is unterminated or a quoted value is
            unterminated.

    """
    pos = start
    length = len(text)
    while pos < length:
        char = text[pos]
        if char == "'":
            end_quote = text.find("'", pos + 1)
            if end_quote == -1:
                msg = "Unterminated string literal in WHERE clause"
                raise MindQLParseError(msg)
            pos = end_quote + 1
            continue
        if char == ")":
            return pos + 1
        pos += 1
    msg = "Unterminated IN clause"
    raise MindQLParseError(msg)


def _parse_bool_expr(where_text: str) -> WhereNode:
    """Parse a WHERE clause into a boolean-expression tree.

    Implements a small recursive-descent parser over the token stream from
    :func:`_tokenize_where`, honouring SQLite precedence (``AND`` binds tighter
    than ``OR``) and parenthesised grouping. Operands are parsed by the shared
    condition / ``IN`` / temporal-predicate grammar.

    Args:
        where_text: The WHERE clause body (``WHERE`` keyword already stripped).

    Returns:
        The root :class:`WhereNode`.

    Raises:
        MindQLParseError: If the expression is malformed (empty, unbalanced
            parentheses, dangling operators, or an invalid operand).

    """
    tokens = _tokenize_where(where_text)
    if not tokens:  # pragma: no cover - defensive: reached only via tree path
        msg = "WHERE requires at least one condition"
        raise MindQLParseError(msg)
    parser = _BoolExprParser(tokens)
    node = parser.parse_or()
    if not parser.at_end():  # pragma: no cover - defensive: parse_or drains tokens
        remaining = parser.peek()
        msg = f"Unexpected token in WHERE clause: {remaining!r}"
        raise MindQLParseError(msg)
    return node


class _BoolExprParser:
    """Recursive-descent parser for the WHERE boolean-expression grammar.

    Args:
        tokens: The token stream from :func:`_tokenize_where`.

    """

    def __init__(self, tokens: list[str]) -> None:
        self._tokens = tokens
        self._pos = 0

    def at_end(self) -> bool:
        """Return whether all tokens have been consumed."""
        return self._pos >= len(self._tokens)

    def peek(self) -> str | None:
        """Return the current token without consuming it, or ``None`` at end."""
        if self.at_end():
            return None
        return self._tokens[self._pos]

    def _advance(self) -> str:
        current = self._tokens[self._pos]
        self._pos += 1
        return current

    def parse_or(self) -> WhereNode:
        """Parse an OR-expression (lowest precedence)."""
        operands = [self.parse_and()]
        while self.peek() == "OR":
            self._advance()
            operands.append(self.parse_and())
        if len(operands) == 1:
            return operands[0]
        return BoolExpr(op="OR", operands=tuple(operands))

    def parse_and(self) -> WhereNode:
        """Parse an AND-expression (binds tighter than OR)."""
        operands = [self.parse_operand()]
        while self.peek() == "AND":
            self._advance()
            operands.append(self.parse_operand())
        if len(operands) == 1:
            return operands[0]
        return BoolExpr(op="AND", operands=tuple(operands))

    def parse_operand(self) -> WhereNode:
        """Parse a single operand: a parenthesised group or a leaf fragment."""
        current = self.peek()
        if current is None:
            msg = "Expected a condition in WHERE clause"
            raise MindQLParseError(msg)
        if current == "(":
            self._advance()
            node = self.parse_or()
            if self.peek() != ")":
                msg = "Unbalanced parentheses in WHERE clause"
                raise MindQLParseError(msg)
            self._advance()
            return node
        if current in ("AND", "OR", ")"):
            msg = f"Unexpected token in WHERE clause: {current!r}"
            raise MindQLParseError(msg)
        self._advance()
        return _parse_operand_fragment(current)


def _parse_operand_fragment(fragment: str) -> WhereNode:
    """Parse a single WHERE operand fragment into a leaf node.

    Tries the operand grammars in order: temporal predicate, ``IN`` condition,
    then ordinary comparison.

    Args:
        fragment: One operand substring from the tokeniser.

    Returns:
        The parsed leaf node.

    Raises:
        MindQLParseError: If the fragment matches none of the operand grammars.

    """
    temporal = _try_parse_temporal_predicate(fragment)
    if temporal is not None:
        return temporal
    in_condition = _try_parse_in_condition(fragment)
    if in_condition is not None:
        return in_condition
    condition = _parse_condition(fragment)
    return Comparison(
        field=condition.field,
        operator=condition.operator,
        value=condition.value,
    )


def _try_parse_in_condition(fragment: str) -> InCondition | None:
    """Recognise and parse a ``field IN (v1, v2, …)`` operand.

    Args:
        fragment: One operand substring.

    Returns:
        The parsed :class:`InCondition` when ``fragment`` is an ``IN`` operand,
        otherwise ``None`` so the caller can fall through.

    Raises:
        MindQLParseError: If ``fragment`` is an ``IN`` operand whose value list
            is empty or malformed.

    """
    match = re.match(r"(\w+)\s+IN\s*\((.*)\)\s*$", fragment, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    field_name = match.group(1)
    body = match.group(2)
    values = _parse_in_values(body)
    if not values:
        msg = "IN requires at least one value"
        raise MindQLParseError(msg)
    return InCondition(field=field_name, values=values)


def _parse_in_values(body: str) -> tuple[str | int | float, ...]:
    """Parse the comma-separated value list inside an ``IN (...)`` clause.

    Each value follows the same quoting rules as a comparison value: a
    single-quoted literal is kept verbatim as a string, a bare token is coerced
    to ``int`` / ``float`` / ``str``.

    Args:
        body: The text between the ``IN`` parentheses.

    Returns:
        The parsed values in source order (possibly empty).

    Raises:
        MindQLParseError: If the value list is malformed.

    """
    values: list[str | int | float] = []
    pos = 0
    length = len(body)
    expect_value = True
    while pos < length:
        if body[pos].isspace():
            pos += 1
            continue
        if body[pos] == ",":
            if expect_value:
                msg = "Malformed IN value list"
                raise MindQLParseError(msg)
            expect_value = True
            pos += 1
            continue
        if not expect_value:
            msg = "Malformed IN value list"
            raise MindQLParseError(msg)
        value_match = _IN_VALUE_RE.match(body, pos)
        if not value_match or value_match.end() == pos:
            msg = "Malformed IN value list"
            raise MindQLParseError(msg)
        quoted = value_match.group(1)
        bare = value_match.group(2)
        if quoted is not None:
            values.append(quoted)
        else:
            values.append(_coerce_value(bare))
        pos = value_match.end()
        expect_value = False
    if expect_value and values:
        msg = "Malformed IN value list"
        raise MindQLParseError(msg)
    return tuple(values)


def _try_parse_temporal_predicate(part: str) -> TemporalPredicate | None:
    """Recognise and parse a temporal predicate WHERE fragment.

    Args:
        part: One ``AND``-separated WHERE fragment.

    Returns:
        The parsed :class:`TemporalPredicate` when ``part`` begins with a
        temporal keyword, otherwise ``None`` so the caller can fall through
        to the ordinary condition grammar.

    Raises:
        MindQLParseError: If ``part`` names a temporal keyword but its
            timestamp arguments are missing, surplus, or not ISO-8601. The
            message is intentionally generic and never echoes the keyword
            list.

    """
    tokens = part.split(None, 1)
    keyword = tokens[0].lower()
    try:
        kind = TemporalPredicateKind(keyword)
    except ValueError:
        return None

    rest = tokens[1] if len(tokens) > 1 else ""
    raw_args = _TIMESTAMP_ARG_RE.findall(rest)
    # Each match is a ``(quoted, bare)`` pair; exactly one group is non-empty.
    args = [quoted or bare for quoted, bare in raw_args]

    expected = _TEMPORAL_ARITY[kind]
    if len(args) != expected:
        msg = "Malformed temporal predicate"
        raise MindQLParseError(msg)

    validated = [_require_iso8601(arg) for arg in args]
    start = validated[0] if expected >= 1 else None
    end = validated[1] if expected >= 2 else None  # noqa: PLR2004
    return TemporalPredicate(kind=kind, start=start, end=end)


def _require_iso8601(arg: str) -> str:
    """Validate that a temporal-predicate argument is a non-empty ISO-8601 value.

    Unlike :func:`validate_iso8601_nullable`, a missing or empty value is an
    error here: temporal predicates that take an argument require a real
    timestamp.

    Args:
        arg: The raw timestamp token (quotes already stripped).

    Returns:
        The validated, UTC-normalised timestamp string.

    Raises:
        MindQLParseError: If the argument is empty or not valid ISO-8601. The
            message is generic and public-safe.

    """
    if not arg.strip():
        msg = "Malformed temporal predicate"
        raise MindQLParseError(msg)
    try:
        validated = validate_iso8601_nullable(arg)
    except ValueError as exc:
        msg = "Malformed temporal predicate"
        raise MindQLParseError(msg) from exc
    # ``validate_iso8601_nullable`` only returns ``None`` for a ``None`` input,
    # which cannot happen here, but narrow the type for the caller.
    if validated is None:  # pragma: no cover - defensive, unreachable
        msg = "Malformed temporal predicate"
        raise MindQLParseError(msg)
    return validated


def _coerce_value(raw: str) -> str | int | float:
    """Coerce a raw string value to int, float, or leave as string.

    Args:
        raw: The raw token value.

    Returns:
        Coerced value.

    """
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw
