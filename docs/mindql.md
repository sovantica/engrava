# MindQL

MindQL is a small declarative query language for the engrava thought-graph.
It provides a human-readable syntax for retrieving and counting rows, plus a
read-only SQL passthrough.

## Syntax

```
FIND  <table> [WHERE <condition> [AND <condition> ...]] [LIMIT <n>]
COUNT <table> [WHERE <condition> [AND <condition> ...]]
SELECT <raw read-only SQL>
```

- The command verb (`FIND`, `COUNT`, `SELECT`) is case-insensitive.
- `FIND` and `COUNT` **require a table name** as the second token.
- A `WHERE` clause is `field operator value`; string values must be
  single-quoted, bare numbers are coerced to `int`/`float`.
- Operators: `=`, `!=`, `>`, `<`, `>=`, `<=`. Conditions chain with `AND`.

### Queryable tables

| Token(s) | Table |
|----------|-------|
| `thoughts`, `thought` | `thought` |
| `edges`, `edge` | `edge` |
| `embeddings`, `embedding` | `embedding` |
| `actions`, `action` | `action` |

## Commands

### FIND

Retrieve rows from a table.

```
FIND thoughts WHERE thought_type = 'OBSERVATION' LIMIT 10
FIND thoughts WHERE lifecycle_status = 'ACTIVE' AND priority = 'P1'
FIND edges WHERE edge_type = 'ASSOCIATED' LIMIT 5
```

Filterable `thought` columns include `thought_type`, `lifecycle_status`,
`priority`, `essence`, `content`, `source`, `confidence`, `visibility`,
`confirmation_count`, `created_cycle`, `updated_cycle`, and `thought_id`.
A column outside the per-table allowlist raises `MindQLParseError`.

**Returns:** matching rows as dicts.

### COUNT

Count rows matching the filters. `COUNT` does **not** accept a `LIMIT`.

```
COUNT thoughts WHERE lifecycle_status = 'ACTIVE'
COUNT thoughts WHERE thought_type = 'OBSERVATION'
COUNT edges
```

**Returns:** the count is exposed on `MindQLResult.count`.

### SELECT

`SELECT` is a **read-only SQL passthrough** — it runs the statement verbatim,
so it needs a full `FROM` clause and standard SQL syntax. The underlying
tables are `thought`, `edge`, `embedding`, and `action`.

```
SELECT thought_id, essence FROM thought WHERE lifecycle_status = 'ACTIVE'
SELECT thought_id, priority, essence FROM thought WHERE thought_type = 'BELIEF' LIMIT 20
```

Only statements that begin with `SELECT` are permitted; anything else is
rejected.

## Valid-time predicates

`FIND` and `COUNT` against the `thoughts` and `edges` tables accept four opt-in
**valid-time** predicates in the `WHERE` clause, for querying *when a fact was
true in the world* (the second time axis — see
[The Bi-temporal Model](bitemporal.md) for the full semantics):

```
FIND thoughts WHERE valid_now
FIND edges WHERE valid_at '2026-01-01T00:00:00+00:00'
FIND thoughts WHERE priority = 'P1' AND valid_within '2026-01-01T00:00:00+00:00' '2026-02-01T00:00:00+00:00'
FIND thoughts WHERE valid_between '2026-01-01T00:00:00+00:00' '2026-12-31T00:00:00+00:00'
```

- `valid_now` takes no argument; `valid_at` takes one ISO-8601 timestamp;
  `valid_within` and `valid_between` take two.
- They combine with ordinary conditions via `AND`.
- `valid_now` / `valid_at` / `valid_within` are **NULL-tolerant** (a record with
  an open `valid_from`/`valid_until` bound stays in the result); `valid_between`
  requires real bounds on both ends and therefore excludes open-bound rows.
- A query that uses **no** temporal predicate behaves exactly as before.

The semantics, the open-interval (`NULL` = ±∞) rule, and `invalidate` are
documented in full on [The Bi-temporal Model](bitemporal.md).

## Extension Commands

Custom MindQL verbs are provided through an extension's
`ExtensionManifest.mindql_extensions` and reach the executor via the
`extensions=` argument (entry-point discovery wires this up automatically).
See [Extensions](extensions.md) for the registration flow.

## Python API

### Parsing

`parse()` returns a `MindQLQuery` plan. Its fields are `command`, `table`,
`conditions`, `limit`, `raw_sql`, `extension_name`, and `extension_args`.

```python
from engrava import parse, MindQLParseError

try:
    query = parse("FIND thoughts WHERE thought_type = 'OBSERVATION' LIMIT 5")
    print(query.command)     # MindQLCommand.FIND
    print(query.table)       # "thought"
    print(query.conditions)  # [Condition(field='thought_type', operator=..., value='OBSERVATION')]
    print(query.limit)       # 5
except MindQLParseError as exc:
    print(f"Parse error: {exc}")
```

### Execution

`MindQLExecutor` runs against an open `aiosqlite.Connection`, and `execute()`
takes a **parsed** `MindQLQuery` — parse the string first. `MindQLResult`
exposes `columns`, `rows`, `count`, and `command`.

```python
from engrava import MindQLExecutor, parse

executor = MindQLExecutor(conn)  # conn is an aiosqlite.Connection
result = await executor.execute(
    parse("FIND thoughts WHERE thought_type = 'OBSERVATION' LIMIT 10")
)
for row in result.rows:
    print(row["essence"])

count_result = await executor.execute(
    parse("COUNT thoughts WHERE lifecycle_status = 'ACTIVE'")
)
print(f"Active thoughts: {count_result.count}")
```

### Error Handling

- `MindQLParseError` — raised for syntax errors, unknown tables, or columns
  outside a table's allowlist.
- Unknown command verbs raise `MindQLParseError` unless registered as an
  extension command.

## CLI Usage

```bash
# Find observations
engrava --db my.db query "FIND thoughts WHERE thought_type = 'OBSERVATION' LIMIT 5"

# Count active thoughts
engrava --db my.db query "COUNT thoughts WHERE lifecycle_status = 'ACTIVE'"

# Select specific fields (raw SQL passthrough)
engrava --db my.db query "SELECT thought_id, essence FROM thought LIMIT 5"
```

Output formats: `--format table` (default), `--format json`, `--format csv`.
