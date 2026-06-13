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

Grammar (simplified BNF)::

    query          := find_query | count_query | select_query | extension_query
    find_query     := "FIND" table_name [where_clause] [limit_clause]
    count_query    := "COUNT" table_name [where_clause]
    select_query   := "SELECT" <raw SQL>
    extension_query:= COMMAND_NAME [args...]

    table_name     := "thoughts" | "thought" | "edges" | "edge"
                    | "embeddings" | "embedding" | "actions" | "action"
    where_clause   := "WHERE" clause ("AND" clause)*
    clause         := condition | temporal_predicate
    condition      := field_name operator value
    temporal_pred  := "valid_now"
                    | "valid_at" timestamp
                    | "valid_within" timestamp timestamp
                    | "valid_between" timestamp timestamp
    operator       := "=" | "!=" | ">" | "<" | ">=" | "<="
    value          := quoted_string | number
    timestamp      := quoted_string | bare_iso8601_token
    quoted_string  := "'" <chars> "'"
    limit_clause   := "LIMIT" integer
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

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
class MindQLQuery:
    """Parsed MindQL query plan.

    Attributes:
        command: The MindQL command verb.
        table: Target table (canonical name, e.g. ``"thought"``).
        conditions: WHERE conditions.
        temporal_predicates: Valid-time predicates parsed from the WHERE
            clause. Empty when the query carries no temporal predicate, in
            which case execution behaves exactly as before this feature.
        limit: Optional LIMIT clause.
        raw_sql: Original SQL for SELECT passthrough.
        extension_name: Extension command name (for EXTENSION type).
        extension_args: Extension command arguments.

    """

    command: MindQLCommand
    table: str | None = None
    conditions: list[Condition] = field(default_factory=list)
    temporal_predicates: list[TemporalPredicate] = field(default_factory=list)
    limit: int | None = None
    raw_sql: str | None = None
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

    upper = stripped.upper()

    # --- SELECT passthrough ---
    if upper.startswith("SELECT"):
        return MindQLQuery(command=MindQLCommand.SELECT, raw_sql=stripped)

    tokens = stripped.split()
    verb = tokens[0].upper()

    # --- Extension command ---
    extensions = known_extensions or set()
    if verb in extensions:
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

    # Parse remainder for WHERE and LIMIT
    remainder = " ".join(tokens[2:])
    conditions, temporal_predicates, limit = _parse_clauses(remainder)

    return MindQLQuery(
        command=command,
        table=table,
        conditions=conditions,
        temporal_predicates=temporal_predicates,
        limit=limit,
    )


def _parse_clauses(
    text: str,
) -> tuple[list[Condition], list[TemporalPredicate], int | None]:
    """Parse WHERE and LIMIT clauses from the remainder of a FIND/COUNT query.

    Each ``AND``-separated WHERE part is first checked against the temporal
    predicate keywords (``valid_now`` / ``valid_at`` / ``valid_within`` /
    ``valid_between``); only when a part is not a temporal predicate does it
    fall through to the ordinary ``field op value`` condition grammar.

    Args:
        text: Everything after the table name.

    Returns:
        Tuple of (conditions, temporal_predicates, limit).

    Raises:
        MindQLParseError: If clauses are malformed.

    """
    conditions: list[Condition] = []
    temporal_predicates: list[TemporalPredicate] = []
    limit: int | None = None

    if not text.strip():
        return conditions, temporal_predicates, limit

    upper = text.upper()

    # Extract LIMIT
    limit_match = re.search(r"\bLIMIT\s+(\d+)\s*$", text, re.IGNORECASE)
    if limit_match:
        limit = int(limit_match.group(1))
        text = text[: limit_match.start()].strip()
        upper = text.upper()

    # Extract WHERE conditions
    if upper.startswith("WHERE"):
        where_text = text[5:].strip()  # skip "WHERE"
        # Split on AND (case-insensitive)
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
    elif text.strip():
        msg = f"Expected WHERE or LIMIT, got: {text.strip()!r}"
        raise MindQLParseError(msg)

    return conditions, temporal_predicates, limit


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
