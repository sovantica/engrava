# Metrics & Snapshot API

engrava exposes a snapshot metrics API via `await store.metrics()`. The
returned `EngravaMetrics` dataclass aggregates thought/edge counts,
storage footprint, and a rolling-window search-latency histogram.

`store.metrics()` returns a stable `EngravaMetrics` dataclass with:

- `thoughts` — counts by type and lifecycle status
- `edges` — counts by edge type
- `storage` — on-disk footprint for the main SQLite database and WAL
- `search_latency` — rolling-window p50/p95/p99 search latency

## Quick Example

```python
from engrava import SqliteEngravaCore
import aiosqlite

async def main() -> None:
    conn = await aiosqlite.connect("engrava.db")
    conn.row_factory = aiosqlite.Row
    store = SqliteEngravaCore(conn)
    try:
        metrics = await store.metrics()
        print(metrics.thoughts.total)
        print(metrics.edges.by_type)
        print(metrics.search_latency.p95_ms)
    finally:
        await conn.close()
```

## Configuration

```yaml
metrics:
  enabled: true
  window_size: 1000
```

When `enabled: false`, `store.metrics()` returns a zero-filled snapshot and does
not issue SQL queries.

## CLI

`engrava info` now renders the same snapshot contract used by the Python API.

```bash
engrava --db mydata.db info
engrava --db mydata.db --format json info
```

## Notes

- The latency histogram tracks completed public search calls.
- Nested calls inside `search_hybrid()` are suppressed, so one hybrid search
  contributes one latency sample.
- This snapshot API tracks only aggregate counts and search latency — not individual events.
