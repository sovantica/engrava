# MindQL

MindQL is a declarative query language for the engrava thought-graph.
It provides a human-readable syntax for common operations.

## Syntax

```
COMMAND [filters] [LIMIT n]
```

Commands are case-insensitive. Filters use `key=value` pairs.

## Commands

### FIND

Search for thoughts matching the given filters.

```
FIND type=OBSERVATION LIMIT 10
FIND status=ACTIVE priority=P1
FIND type=INSIGHT LIMIT 5
```

**Filters:**

| Key | Description | Values |
|-----|-------------|--------|
| `type` | Thought type | `OBSERVATION`, `INSIGHT`, `BELIEF`, `GOAL`, `PLAN`, `MEMORY`, `HYPOTHESIS`, `EMOTION` |
| `status` | Lifecycle status | `ACTIVE`, `COMPLETED`, `ARCHIVED`, `DORMANT` |
| `priority` | Priority level | `P1`, `P2`, `P3`, `P4` |

**Returns:** List of matching `ThoughtRecord` objects (as dicts).

### COUNT

Count thoughts matching the given filters.

```
COUNT status=ACTIVE
COUNT type=OBSERVATION
COUNT
```

**Returns:** `[{"count": N}]`

### SELECT

Select specific fields from matching thoughts.

```
SELECT thought_id, essence WHERE type=OBSERVATION
SELECT thought_id, priority, essence WHERE status=ACTIVE LIMIT 20
```

**Returns:** List of dicts with only the requested fields.

## Extension Commands

Custom commands can be registered via `EngravaHooksProtocol.mindql_extension_registry()`.
See [Extensions](extensions.md) for details.

## Python API

### Parsing

```python
from engrava import parse, MindQLCommand, MindQLParseError

try:
    cmd: MindQLCommand = parse("FIND type=INSIGHT LIMIT 5")
    print(cmd.verb)     # "FIND"
    print(cmd.filters)  # {"type": "INSIGHT"}
    print(cmd.limit)    # 5
except MindQLParseError as e:
    print(f"Parse error: {e}")
```

### Execution

```python
from engrava import MindQLExecutor, MindQLResult

executor = MindQLExecutor(store)
result: MindQLResult = await executor.execute("FIND type=OBSERVATION LIMIT 10")

for row in result.rows:
    print(row["essence"])

print(f"Total: {result.row_count} rows")
```

### Error Handling

- `MindQLParseError` — raised for syntax errors
- Unknown commands raise `MindQLParseError` (unless registered as an extension)

## CLI Usage

```bash
# Find observations
engrava --db my.db query "FIND type=OBSERVATION LIMIT 5"

# Count active thoughts
engrava --db my.db query "COUNT status=ACTIVE"

# Select specific fields
engrava --db my.db query "SELECT thought_id, essence WHERE type=INSIGHT"
```

Output formats: `--format table` (default), `--format json`, `--format csv`.
