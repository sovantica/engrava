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

## Observability signals

Beyond the aggregate snapshot above, the store exposes two **read-only, monotonic
counters** as plain properties — `store.fts_match_failure_count` and
`store.vector_arm_degradation_count` (no `await`, no SQL; they are **not** part of
the `metrics()` snapshot). Each starts at `0`, only ever increases over the life of
a store instance, and resets only when you construct a new store. They surface
silent, self-healing search-arm degradations so an operator can detect a
systematic problem; reading them never changes behaviour.

| Counter | Increments when… | What it means |
|---|---|---|
| `fts_match_failure_count` | a normalized FTS5 `MATCH` raises **before** the sanitizing retry runs | some queries are taking the bare-mode fallback path instead of matching as written. The fallback runs a sanitized query that is a valid MATCH and returns its matches. |
| `vector_arm_degradation_count` | a **degenerate** query vector (empty, all-zero, or non-finite) makes the vector arm return nothing | some queries are producing bad embeddings, not that the corpus is empty. A **wrong-dimension** query vector is **not** counted here — it raises `VectorDimensionMismatchError` instead (see [Known Limitations](known-limitations.md#query-vector-dimension-mismatch)). |

A steadily-growing `fts_match_failure_count` points at a query-construction issue
in the caller (queries that keep tripping FTS5 syntax); a growing
`vector_arm_degradation_count` points at an embedding provider returning empty or
degenerate vectors. Neither is fatal — both are self-healing — but a rising trend
is worth an alert.

**FTS query robustness.** The normalizer is built so ordinary text and **valid
expert syntax** (quoted phrases, `*` prefixes, uppercase `AND`/`OR`/`NOT`,
`essence:` / `content:` column filters) keep their BM25 ranking and never trip the
fallback — only a genuinely malformed expression does, and even then the query is
retried once through the always-valid bare normalization. See
[Search → Keyword query syntax](search.md#keyword-query-syntax-fts).

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

The main metric groups on `EngravaMetrics` are `thoughts` (`total`, `by_type`,
`by_status`), `edges` (`total`, `by_type`), `storage` (`db_bytes`, `wal_bytes`,
`vec_index_bytes`, `total_bytes`), and `search_latency` (`sample_count`,
`p50_ms`, `p95_ms`, `p99_ms`, `min_ms`, `max_ms`, `mean_ms`). The snapshot also
carries `schema_version` and `snapshot_timestamp` for the snapshot itself.

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
| Audit integrity | `store.journal.verify_integrity()` (journaling only) | the chain fails verification (tampering or corruption) — see [Audit Trail](audit-trail.md) |

The expired-backlog and audit-integrity signals are **not** in the metrics
snapshot — compute them from the calls shown above on your own cadence.

The audit-integrity check applies **only when journaling is enabled**
(`journal.enabled: true`). With journaling off, `store.journal` is `None`, so
guard the call:

```python
async def journal_ok(store) -> bool:
    if store.journal is None:
        return True  # journaling disabled — nothing to verify
    result = await store.journal.verify_integrity()
    return result.valid
```

### Health check

For a readiness probe you want a call that actually touches the database. Note
that `metrics()` is **not** reliable for this when metrics are disabled: with
`metrics.enabled: false`, `store.metrics()` returns a zero-filled snapshot
**without issuing any SQL**, so it would report healthy even if the database were
unreadable. Use a lightweight real read instead — `count_thoughts()` always
queries the database (independent of the metrics setting):

```python
async def healthcheck(store) -> bool:
    try:
        await store.count_thoughts()  # issues SQL — confirms DB + schema are readable
    except Exception:
        return False
    return True
```

(If you know metrics are enabled in your deployment, `await store.metrics()`
works too and additionally returns the live counts.)

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
  application layer (Engrava raises typed exceptions you can count there). The two
  search-arm health counters live as separate store properties, not in the
  snapshot — see [Observability signals](#observability-signals);
- **dreaming metrics** — `run_consolidation()` returns a `ConsolidationResult`
  (promoted / edges / reflections counts) per run; consume that directly;
- **journal size or per-event audit metrics** — the audit history lives in the
  [journal](audit-trail.md) itself, which you query and verify directly, not via
  the metrics snapshot.
