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

## Production monitoring

`store.metrics()` is a **pull** snapshot — there is no built-in exporter. To
monitor a deployment, scrape the snapshot on an interval and feed the fields into
your metrics system (Prometheus, OpenTelemetry, StatsD, …).

### Exporting the snapshot

The snapshot is a plain dataclass, so mapping it to any client is
straightforward. A Prometheus example:

```python
from prometheus_client import Gauge

THOUGHTS = Gauge("engrava_thoughts_total", "Total thoughts")
DB_BYTES = Gauge("engrava_db_bytes", "Main database size in bytes")
WAL_BYTES = Gauge("engrava_wal_bytes", "WAL size in bytes")
SEARCH_P95 = Gauge("engrava_search_p95_ms", "Search p95 latency (ms)")
SEARCH_P99 = Gauge("engrava_search_p99_ms", "Search p99 latency (ms)")


async def collect(store) -> None:
    m = await store.metrics()
    THOUGHTS.set(m.thoughts.total)
    DB_BYTES.set(m.storage.db_bytes)
    WAL_BYTES.set(m.storage.wal_bytes)
    SEARCH_P95.set(m.search_latency.p95_ms)
    SEARCH_P99.set(m.search_latency.p99_ms)
```

The fields available to export are exactly those on `EngravaMetrics`:
`thoughts` (`total`, `by_type`, `by_status`), `edges` (`total`, `by_type`),
`storage` (`db_bytes`, `wal_bytes`, `vec_index_bytes`, `total_bytes`), and
`search_latency` (`sample_count`, `p50_ms`, `p95_ms`, `p99_ms`, `min_ms`,
`max_ms`, `mean_ms`).

### Scrape cadence

Treat `metrics()` like any pull endpoint: a **30–60 s** scrape interval is
typically plenty. Counts and storage change slowly; the latency histogram is a
rolling window (`metrics.window_size`, default 1000 samples), so it already
smooths short spikes. Avoid sub-second scrapes — each call runs a few aggregate
SQL queries.

### What to alert on

| Signal | Source field | Alert when… |
|---|---|---|
| Storage growth | `storage.db_bytes`, `storage.total_bytes` | size approaches your disk budget, or grows unexpectedly fast |
| WAL not checkpointing | `storage.wal_bytes` | the WAL keeps growing and never shrinks (checkpoints not happening) |
| Search latency | `search_latency.p95_ms` / `p99_ms` | p95/p99 exceeds your budget — often the sign you've passed the brute-force vector ceiling (see [Performance](performance.md)) |
| Expired backlog | `count_thoughts(include_expired=True)` − `count_thoughts()` | the number of expired-but-not-cleaned thoughts grows (run `engrava gc --expired`) — see [Data Lifecycle](data-lifecycle.md) |
| Audit integrity | `await store.journal.verify_integrity()` | the chain fails verification (tampering or corruption) — see [Audit Trail](audit-trail.md) |

The expired-backlog and audit-integrity signals are **not** in the metrics
snapshot — compute them from the calls shown above on your own cadence.

### Health check

A cheap liveness/readiness probe is a metrics call (which also confirms the
database is reachable and the schema is readable):

```python
async def healthcheck(store) -> bool:
    try:
        await store.metrics()
    except Exception:
        return False
    return True
```

### Logging

The library logs through the standard `logging` module under the **`engrava.*`**
namespace (each module uses `logging.getLogger(__name__)`, e.g.
`engrava.extensions.dreaming`, `engrava.extensions.vector_sqlite_vec`,
`engrava.config`). It logs at **`WARNING`** (degraded conditions, e.g. sqlite-vec
unavailable → numpy fallback), **`INFO`** (dreaming progress), and **`DEBUG`**
(detailed internals) — it does **not** log at `ERROR`/`CRITICAL`; failures are
raised as typed exceptions for the caller to handle. Configure it like any
library logger:

```python
import logging

logging.getLogger("engrava").setLevel(logging.WARNING)  # quiet, production default
# logging.getLogger("engrava").setLevel(logging.INFO)   # see dreaming activity
```

### Out of scope

The snapshot is deliberately small. It does **not** include:

- **write / mutation counters** or **error counters** — track those at your
  application layer (Engrava raises typed exceptions you can count there);
- **dreaming metrics** — `run_consolidation()` returns a `ConsolidationResult`
  (promoted / edges / reflections counts) per run; consume that directly;
- **journal size or per-event audit metrics** — the audit history lives in the
  [journal](audit-trail.md) itself, which you query and verify directly, not via
  the metrics snapshot.
