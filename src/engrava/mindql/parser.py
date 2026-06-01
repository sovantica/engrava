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

Grammar (simplified BNF)::

    query          := find_query | count_query | select_query | extension_query
    find_query     := "FIND" table_name [where_clause] [limit_clause]
    count_query    := "COUNT" table_name [where_clause]
    select_query   := "SELECT" <raw SQL>
    extension_query:= COMMAND_NAME [args...]

    table_name     := "thoughts" | "thought" | "edges" | "edge"
                    | "embeddings" | "embedding" | "actions" | "action"
    where_clause   := "WHERE" condition ("AND" condition)*
    condition      := field_name operator value
    operator       := "=" | "!=" | ">" | "<" | ">=" | "<="
    value          := quoted_string | number
    quoted_string  := "'" <chars> "'"
    limit_clause   := "LIMIT" integer
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum


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


@dataclass(frozen=True)
class MindQLQuery:
    """Parsed MindQL query plan.

    Attributes:
        command: The MindQL command verb.
        table: Target table (canonical name, e.g. ``"thought"``).
        conditions: WHERE conditions.
        limit: Optional LIMIT clause.
        raw_sql: Original SQL for SELECT passthrough.
        extension_name: Extension command name (for EXTENSION type).
        extension_args: Extension command arguments.

    """

    command: MindQLCommand
    table: str | None = None
    conditions: list[Condition] = field(default_factory=list)
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
    conditions, limit = _parse_clauses(remainder)

    return MindQLQuery(
        command=command,
        table=table,
        conditions=conditions,
        limit=limit,
    )


def _parse_clauses(text: str) -> tuple[list[Condition], int | None]:
    """Parse WHERE and LIMIT clauses from the remainder of a FIND/COUNT query.

    Args:
        text: Everything after the table name.

    Returns:
        Tuple of (conditions, limit).

    Raises:
        MindQLParseError: If clauses are malformed.

    """
    conditions: list[Condition] = []
    limit: int | None = None

    if not text.strip():
        return conditions, limit

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
            match = _CONDITION_RE.match(stripped_part)
            if not match:
                msg = f"Invalid condition: {stripped_part!r}"
                raise MindQLParseError(msg)
            field_name = match.group(1)
            op_str = match.group(2)
            # group 3 = quoted value, group 4 = unquoted value
            raw_value: str = match.group(3) if match.group(3) is not None else match.group(4)
            value: str | int | float = _coerce_value(raw_value)
            conditions.append(
                Condition(
                    field=field_name,
                    operator=_OPERATOR_MAP[op_str],
                    value=value,
                )
            )
    elif text.strip():
        msg = f"Expected WHERE or LIMIT, got: {text.strip()!r}"
        raise MindQLParseError(msg)

    return conditions, limit


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
