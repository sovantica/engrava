# Metrics & Snapshot API

engrava exposes a snapshot metrics API via `await store.metrics()`. The
returned `EngravaMetrics` dataclass aggregates thought/edge counts,
storage footprint, and a rolling-window search-latency histogram.

`store.metrics()` returns a stable `EngravaMetrics` dataclass with:

- `thoughts` — counts by type and lifecycle status
- `edges` — counts by edge type
- `storage` — on-disk footprint for the main SQLite database and WAL
- `search_latency` — rolling-window p50/p95/p99 search latency
- `measured` — `True` only when this snapshot came from actually querying the
  store; `False` for the zero-filled placeholder returned when `metrics.enabled`
  is `False`. Check this before trusting a zero — an unmeasured snapshot and an
  honestly empty store both read `thoughts.total == 0`.

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
not issue SQL queries. That snapshot has `measured=False`. Nothing else in the
snapshot states whether a measurement happened. Every field a caller might read
instead is a proxy that fails in at least one direction — `storage.db_bytes`
reads `0` for a disabled store and for a measured in-memory one (a freshly
created, empty file-backed store instead measures a real, nonzero
`storage.db_bytes` of `180224` bytes as soon as `ensure_schema()` has run, but
that same field reads `0` again on a measured store whose file later goes
missing under an open connection); `search_latency.sample_count` reads `0` for
a disabled store and for a measured store that has served no searches.
`measured` states it directly.

## CLI

`engrava info` renders the same snapshot the Python API returns, with one
deliberate difference: its own `schema_version` is exposed as
`metrics_schema_version`, alongside a `database_schema_version` field the
Python snapshot does not carry (the database's `PRAGMA user_version`, read
separately) — see [Upgrade Guide → 0.6 -> 0.7](upgrade.md#06---07) for why the
CLI output and the `EngravaMetrics` object are no longer key-for-key
identical.

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
straightforward — but a plain `Gauge` is the wrong tool: it exposes a sample
(`0.0` by default) from the moment it is *registered*, regardless of whether
it has ever been `.set()`, so any "register once, update on each scrape"
scheme has a window where a scrape reads a value nobody measured. Use a
[custom collector](https://prometheus.github.io/client_python/collector/custom/)
instead — it queries the store fresh on every scrape and yields nothing at
all when `measured` is false, so there is no registration window, no stale
value between ticks, and no separate setup step to get wrong.

**The pattern below is written for a WSGI-style exposition**
(`start_http_server()`), where `collect()` runs on a thread with no event loop
of its own. It is the wrong shape for an ASGI application (e.g. one mounting
`prometheus_client.make_asgi_app()`): there, `collect()` runs on the app's own
running loop, and this pattern's internal `asyncio.run()` call raises. For an
ASGI deployment, do the async store read once in your own request handler and
register a collector over the already-computed metric families instead — its
`collect()` then returns that precomputed list synchronously, with nothing
left for it to await. We do not carry a tested ASGI example here; see the
comment inside `collect()` below for exactly where and why this shape fails
under one.

```python
import asyncio

from prometheus_client import REGISTRY
from prometheus_client.core import GaugeMetricFamily

_METRICS = (
    ("engrava_thoughts_total", "Total thoughts"),
    ("engrava_db_bytes", "Main database size in bytes"),
    ("engrava_wal_bytes", "WAL size in bytes"),
    ("engrava_search_p95_ms", "Search p95 latency (ms)"),
    ("engrava_search_p99_ms", "Search p99 latency (ms)"),
)


class EngravaCollector:
    """Queries the store fresh on every scrape; yields nothing when unmeasured."""

    def __init__(self, store):
        self._store = store

    def describe(self):
        # Sample-free descriptors, so the registry learns these names and
        # help text at registration time -- without this, `register()` calls
        # `collect()` itself to learn them, which would need to query the
        # store from the registering thread. Returning `[]` here instead
        # would also work for the running-loop problem, but it disables the
        # registry's own collision detection and `restricted_registry()`
        # support along with it, so return the (sampleless) families instead.
        return [GaugeMetricFamily(name, help_text) for name, help_text in _METRICS]

    def collect(self):
        # `asyncio.run()` requires no event loop already running on this
        # thread -- true for a WSGI worker (`start_http_server()`), which is
        # what this example is written for, but not for an ASGI exposition
        # app (e.g. `prometheus_client.make_asgi_app()`): there, `collect()`
        # itself runs on the app's own running loop regardless of which
        # thread or loop the store's connection was created on, and this
        # raises "asyncio.run() cannot be called from a running event loop".
        # This example does not support that shape; the alternative for it is
        # a periodic background task that refreshes a plain value collect()
        # reads, rather than querying the store here.
        m = asyncio.run(self._store.metrics())
        if not m.measured:
            return
        values = (m.thoughts.total, m.storage.db_bytes, m.storage.wal_bytes,
                  m.search_latency.p95_ms, m.search_latency.p99_ms)
        for (name, help_text), value in zip(_METRICS, values, strict=True):
            yield GaugeMetricFamily(name, help_text, value=value)


REGISTRY.register(EngravaCollector(store))
```

`collect()` runs synchronously on the scrape thread, hence `asyncio.run(...)`;
if your process already runs its own event loop, drive the same coroutine
through that loop instead (e.g. a thread-safe future) rather than nesting
`asyncio.run()` inside a running one.

The main metric groups on `EngravaMetrics` are `thoughts` (`total`, `by_type`,
`by_status`), `edges` (`total`, `by_type`), `storage` (`db_bytes`, `wal_bytes`,
`vec_index_bytes`, `total_bytes`), and `search_latency` (`sample_count`,
`p50_ms`, `p95_ms`, `p99_ms`, `min_ms`, `max_ms`, `mean_ms`). The snapshot also
carries `schema_version`, `snapshot_timestamp`, and `measured` for the
snapshot itself — always check `measured` before trusting the rest.

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

`store.journal.verify_integrity()` needs an active writer: with
`journal.enabled: false`, `store.journal` is `None`. Do **not** report that case
as healthy — entries written in an earlier session stay in `journal_entry`, so a
store reopened with journaling off still has a chain that can be tampered with
and still needs verifying. Monitor through `store.verify_journal()`, which audits
whatever chain is on disk **independent of the current `journal.enabled` state**
and returns `valid=True` with `entries_checked=0` when there is no chain at all:

```python
async def journal_ok(store) -> bool:
    result = await store.verify_journal()
    return result.valid
```

See [Audit Trail → verifying integrity](audit-trail.md#verifying-integrity).

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
`engrava.extensions.dreaming`, `engrava.infrastructure.sqlite.vector_sqlite_vec`,
`engrava.config`). As a general rule it logs at **`WARNING`** (degraded
conditions, e.g. sqlite-vec unavailable → numpy fallback), **`INFO`**
(dreaming progress), and **`DEBUG`** (detailed internals), and raises failures
as typed exceptions for the caller to handle rather than logging them.
`CRITICAL` is never used.

The one departure: when the derived-records extension is enabled
(`derive.enabled=True`, off by default, and only reachable with a producer
capability configured) and a per-child rollback fails after its source has
already committed, that failure logs at `ERROR` (`derive.on_error="log"`, the
default for that setting) and the remaining children are abandoned without
raising — the source is already durable, so under the default policy this must
not escape as a caller-visible exception. This is the library's only `ERROR`
call site. Configure it like any library logger:

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
