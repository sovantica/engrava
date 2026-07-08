# MindQL

MindQL is a small declarative query language for the engrava thought-graph.
It provides a human-readable syntax for retrieving and counting rows, plus a
read-only SQL passthrough.

## Syntax

```
[EXPLAIN] FIND  <table> [WHERE <bool-expr>] [ORDER BY <sort> ...] [LIMIT <n>] [OFFSET <n>]
[EXPLAIN] COUNT <table> [WHERE <bool-expr>]
[EXPLAIN] SELECT <raw read-only SQL>
```

- The command verb (`FIND`, `COUNT`, `SELECT`) is case-insensitive.
- `FIND` and `COUNT` **require a table name** as the second token.
- A simple `WHERE` condition is `field operator value`; string values must be
  single-quoted, bare numbers are coerced to `int`/`float`. The quoting decides
  the type: a single-quoted value is kept **verbatim as a string**, so a
  zero-padded identifier like `source = '007'` matches the stored string `'007'`,
  whereas an unquoted `created_cycle = 7` is coerced to the integer `7`.
- Operators: `=`, `!=`, `>`, `<`, `>=`, `<=`, and `IN (...)`.
- Conditions combine with `AND`, `OR`, and parentheses (see
  [Boolean expressions](#boolean-expressions-and-or-parentheses)).
- `FIND` additionally supports `ORDER BY` and `OFFSET`; any read query may be
  prefixed with `EXPLAIN` to see the compiled SQL without running it.

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
FIND thoughts WHERE priority IN ('P1', 'P2') ORDER BY created_cycle DESC LIMIT 20
FIND thoughts ORDER BY created_cycle DESC LIMIT 10 OFFSET 20
FIND edges WHERE edge_type = 'ASSOCIATED' LIMIT 5
```

Filterable `thought` columns include `thought_type`, `lifecycle_status`,
`priority`, `essence`, `content`, `source`, `confidence`, `visibility`,
`confirmation_count`, `created_cycle`, `updated_cycle`, and `thought_id`.
A column outside the per-table allowlist raises `MindQLParseError`.

**Default row cap.** A `FIND` with no `LIMIT` clause is capped at **100 rows**
when it runs, so an unqualified `FIND thoughts` can never trigger an unbounded
scan. The cap is applied at execution, not at parse time — `parse("FIND thoughts")`
leaves `query.limit` as `None`, and the executor substitutes the default only if
no explicit `LIMIT` is present. An explicit `LIMIT` always overrides the default;
`COUNT` queries are unaffected (they aggregate and never materialise the rows).

**Returns:** matching rows as dicts.

#### ORDER BY

`FIND` accepts an `ORDER BY` clause with one or more sort keys, each with an
optional direction (`ASC` default, or `DESC`, case-insensitive):

```
FIND thoughts ORDER BY created_cycle DESC
FIND thoughts WHERE priority = 'P1' ORDER BY priority ASC, created_cycle DESC LIMIT 10
```

Every sort field must be in the per-table column allowlist. Sort fields are
column **identifiers**, not values — they are validated against the allowlist
and never bound as parameters, so an unknown field raises `MindQLParseError`.
`ORDER BY` is emitted before `LIMIT`/`OFFSET`. It is rejected on `COUNT` (which
returns a scalar) and is not used with the raw `SELECT` passthrough.

#### OFFSET / pagination

`FIND` accepts an `OFFSET <n>` clause (`n >= 0`) for pagination, combined with
`LIMIT` and `ORDER BY`:

```
FIND thoughts ORDER BY created_cycle DESC LIMIT 20 OFFSET 40
```

SQLite requires a `LIMIT` for `OFFSET`, so an `OFFSET` with no explicit `LIMIT`
applies the [default row cap](#find) as the limit. `OFFSET` is rejected on
`COUNT` and the `SELECT` passthrough. For stable pages, pair `OFFSET` with an
`ORDER BY` so the row order is deterministic.

### WHERE operators

#### IN

The `IN` operator matches a column against a set of values:

```
FIND thoughts WHERE thought_type IN ('BELIEF', 'OBSERVATION')
FIND thoughts WHERE created_cycle IN (1, 2, 3)
```

Each value follows the same quoting rule as a comparison value: single-quoted
stays a **verbatim string**, unquoted is coerced to `int`/`float`. An empty
`IN ()` is a parse error. Every value is bound as a `?` placeholder (never
interpolated), and the field must be in the per-table allowlist.

#### Boolean expressions: AND, OR, parentheses

Conditions (simple comparisons, `IN`, and valid-time predicates) combine with
`AND`, `OR`, and parenthesised grouping, using **SQLite-standard precedence:
`AND` binds tighter than `OR`**:

```
FIND thoughts WHERE priority = 'P1' OR lifecycle_status = 'ACTIVE'
FIND thoughts WHERE priority = 'P1' OR priority = 'P2' AND source = 'x'
FIND thoughts WHERE (priority = 'P1' OR priority = 'P2') AND lifecycle_status = 'ACTIVE'
```

Because `AND` binds tighter, the second example groups as
`priority = 'P1' OR (priority = 'P2' AND source = 'x')`. Use parentheses (as in
the third example) to group an `OR` before an `AND`. A `WHERE` that uses only
simple comparisons and/or valid-time predicates joined by `AND` (no `OR`, no
parentheses, no `IN`) behaves exactly as it always has.

### COUNT

Count rows matching the filters. `COUNT` does **not** accept `LIMIT`,
`ORDER BY`, or `OFFSET`.

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
rejected. The passthrough is also restricted to a **single** statement: a
single trailing `;` is tolerated, but any `;` remaining mid-string is rejected
(`MindQLParseError: Only a single SELECT statement is allowed`) so a second
statement can never be smuggled in.

**Bound parameters.** The passthrough can carry bound parameters, set
programmatically on the query object (never parsed from the MQL text). When
`MindQLQuery.select_params` is set, the statement runs with those values bound
to its `?` placeholders — the preferred way to pass untrusted values into a
passthrough:

```python
from engrava import MindQLQuery, MindQLCommand, MindQLExecutor

query = MindQLQuery(
    command=MindQLCommand.SELECT,
    raw_sql="SELECT thought_id FROM thought WHERE priority = ?",
    select_params=("P1",),
)
result = await MindQLExecutor(conn).execute(query)
```

When `select_params` is `None` (the default, and what `parse()` produces) the
statement runs with no bound parameters, exactly as before.

## EXPLAIN

Any read query may be prefixed with `EXPLAIN` to see the compiled SQL and its
bound parameters **without executing it**:

```
EXPLAIN FIND thoughts WHERE priority = 'P1' ORDER BY created_cycle DESC
EXPLAIN COUNT thoughts WHERE thought_type IN ('BELIEF', 'OBSERVATION')
EXPLAIN SELECT thought_id FROM thought WHERE lifecycle_status = 'ACTIVE'
```

`EXPLAIN` is a pure compile-and-return: it never touches the database, never
runs SQLite's `EXPLAIN QUERY PLAN`, and never executes the compiled or raw SQL.
The result carries the plan as a single row under the columns `["sql", "params"]`
with `command="EXPLAIN"`:

```python
result = await MindQLExecutor(conn).execute(
    parse("EXPLAIN FIND thoughts WHERE priority = 'P1'")
)
plan = result.rows[0]
print(plan["sql"])     # SELECT * FROM thought WHERE priority = ? LIMIT 100
print(plan["params"])  # ['P1']
```

For `FIND`/`COUNT` the plan is the parameterised SQL that would run; for
`SELECT` it is the guarded raw SQL and its declared `select_params` (if any).
`EXPLAIN` still validates columns, so an `EXPLAIN` of a query that references a
disallowed column raises `MindQLParseError` at compile time. `EXPLAIN` is only
supported for `FIND`, `COUNT`, and `SELECT` (not extension commands).

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

> **Valid time is predicate-only, not a filterable column.** Query valid time
> *only* through the four predicates above. `valid_from` and `valid_until` are not
> in the per-table column allowlist, so an ordinary comparison such as
> `WHERE valid_from = '2026-01-01T00:00:00+00:00'` is **rejected when the query
> runs** (`MindQLParseError: Column 'valid_from' not allowed for table 'thought'`)
> — use `valid_at` / `valid_within` / `valid_between` instead.

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
`conditions`, `temporal_predicates` (the parsed valid-time predicates, empty when
none are used), `where` (the boolean-expression tree, or `None` on the pure-`AND`
path), `order_by`, `limit`, `offset`, `explain`, `raw_sql`, `select_params`,
`extension_name`, and `extension_args`.

`where` is `None` whenever the `WHERE` uses only simple comparisons and/or
valid-time predicates joined by `AND` — in that case `conditions` +
`temporal_predicates` are populated and execution follows the historical path
unchanged. It is a tree (`Comparison`, `InCondition`, `TemporalPredicate`, and
`BoolExpr` nodes, all in `engrava.mindql.parser`) only when the `WHERE` contains
`OR`, parentheses, or `IN`.

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
- A `WHERE` fragment must match the `field operator value` grammar **in full**.
  Trailing content after a condition (for example `WHERE priority = 'P1' OR 1=1`)
  is rejected with a `MindQLParseError` rather than silently parsing only the
  leading `priority = 'P1'` and discarding the rest — so a malformed condition
  can never quietly change the result set.

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
