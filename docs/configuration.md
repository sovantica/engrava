# Configuration

engrava supports YAML-based configuration for production deployments.
This document covers all configuration options.

## Configuration File

Create a `engrava.yaml` file:

```yaml
database:
  path: "./engrava.db"
  wal_mode: true

search:
  default_fts_weight: 0.30
  default_vector_weight: 0.55
  default_recency_weight: 0.10
  default_priority_weight: 0.05
  default_graph_weight: 0.00       # opt-in graph signal (0.0 = OFF by default)
  recency_half_life: 50
  priority_boost_p1: 1.0
  priority_boost_p2: 0.6
  priority_boost_p3: 0.3
  priority_boost_p4: 0.0
  graph_edge_decay: 0.5            # 1-hop distance penalty
  max_neighbors_per_candidate: 5   # safety cap
  reflection_boost: 1.0            # REFLECTION score multiplier (1.0 = neutral)
  reflection_topk_cap: 0.3         # max fraction of top-K that may be REFLECTIONs
  collapse_pool_factor: 4          # arm-budget widening when collapse_key is set
  vec0_overfetch_factor: 4         # sqlite-vec over-fetch before the live-row trim

extensions:
  vector:
    backend: numpy
    dimension: 384

  dreaming:
    enabled: true
    schedule_every_n_cycles: 100
    promote_threshold: 0.7
    candidates_limit: 200
    gates:
      min_confirmations: 2
      min_age_cycles: 1
      max_promoted_per_run: 20
      allow_zero_confirmation: true
```

## Loading Configuration

```python
from engrava import load_config, SqliteEngravaCore

config = load_config("engrava.yaml")

async with await SqliteEngravaCore.from_config("engrava.yaml") as store:
    thought = await store.get_thought("abc")
```

### Full Factory Method

```python
from engrava.config import load_config, resolve_embedding_provider

config = load_config("engrava.yaml")
# resolve_embedding_provider takes the EmbeddingConfig, i.e. config.embeddings
provider = resolve_embedding_provider(config.embeddings)
```

## Configuration Reference

### `database`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `database.path` | `str` | **required** | Path to the SQLite database file (no default — omitting it raises `ConfigError`) |
| `database.wal_mode` | `bool` | `true` | Enable WAL journal mode for concurrent reads |

### `search`

Controls hybrid search behavior (FTS5 + vector + recency + priority).

All 19 `SearchConfig` fields are settable here; every one has a default, so the
whole section is optional.

**Signal weights and per-priority boosts:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `default_fts_weight` | `float` | `0.30` | Weight for FTS5/BM25 text score |
| `default_vector_weight` | `float` | `0.55` | Weight for vector similarity score |
| `default_recency_weight` | `float` | `0.10` | Weight for recency-based score |
| `default_priority_weight` | `float` | `0.05` | Weight for priority signal |
| `default_graph_weight` | `float` | `0.0` | Weight for 1-hop graph signal. **`0.0` ⇒ the graph signal is OFF by default** — no graph queries run and there is zero overhead. Raise it (or pass `graph_weight=` per call) to opt in. |
| `recency_half_life` | `int` | `50` | Cycles for recency score to halve |
| `priority_boost_p1` | `float` | `1.0` | Score multiplier for P1 thoughts |
| `priority_boost_p2` | `float` | `0.6` | Score multiplier for P2 thoughts |
| `priority_boost_p3` | `float` | `0.3` | Score multiplier for P3 thoughts |
| `priority_boost_p4` | `float` | `0.0` | Score multiplier for P4 thoughts |
| `graph_edge_decay` | `float` | `0.5` | Decay factor for the 1-hop neighbour boost |
| `max_neighbors_per_candidate` | `int` | `5` | Max neighbours considered per candidate |

**Reflection handling:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `reflection_boost` | `float` | `1.0` | Score multiplier applied to `REFLECTION` thoughts retrieved by `search_hybrid()`. `1.0` is neutral (reflections compete on equal footing); above `1.0` upranks them. Overridable per call with `reflection_boost=`. |
| `reflection_topk_cap` | `float` | `0.3` | Maximum fraction of the final top-K that may be `REFLECTION` thoughts. After all signals, excess low-scoring reflections in the top-K window are evicted and backfilled with the highest-scoring off-list non-reflections. `1.0` disables the cap. When the cap evicts, an `INFO` line is logged and `HybridSearchResult.reflections_evicted` reports the count. |

**Graph-expansion (candidate-pool widening via `CONSOLIDATED_FROM` edges):**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `graph_expansion_enabled` | `bool` | `true` | When `true`, the candidate pool is expanded by traversing `CONSOLIDATED_FROM` edges from the top-N reflections; pulled source observations get a propagated score. |
| `graph_expansion_top_n` | `int` | `5` | Number of top-ranked reflections used as expansion seeds per query |
| `graph_expansion_propagation_factor` | `float` | `0.7` | Multiplier on the parent reflection score when computing the propagated source score (below `1.0` so sources never outrank their reflection) |
| `graph_expansion_max_sources_per_reflection` | `int` | `20` | Cap on source observations pulled per reflection (highest `edge.weight` first) |
| `graph_expansion_reflection_source_ceiling` | `int` | `50` | Reflections with more than this many `CONSOLIDATED_FROM` sources are skipped during expansion (guards against giant-cluster noise flooding the pool) |

**Bounded pool multipliers:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `collapse_pool_factor` | `int` | `4` | Bounded multiplier applied to each arm's candidate budget **only when** a `collapse_key` is passed to `search_hybrid()` / `recall()`. Gives de-fragmentation backfill a deeper distinct-unit pool. Must be `>= 1`. No effect on the `collapse_key=None` path. |
| `vec0_overfetch_factor` | `int` | `4` | Bounded multiplier applied to `top_k` when the sqlite-vec (`vec0`) backend serves `search_similar()`. `vec0` applies its `LIMIT` before expired/retired rows are filtered, so the arm over-fetches then trims to `top_k`. Must be `>= 1`. No effect on the numpy backend. |

Weights are redistributed proportionally when a signal is unavailable
(e.g. no `current_cycle` → recency skipped). Set any weight to `0.0`
to disable that signal entirely.

> **The graph signal is off by default.** `default_graph_weight` is `0.0`, so a
> default store runs no graph queries at all. This is separate from
> `graph_expansion_enabled` (default `true`), which controls candidate-pool
> widening over `CONSOLIDATED_FROM` edges — the *ranking* graph signal stays
> off until you give `default_graph_weight` (or a per-call `graph_weight`) a
> non-zero value.

See [search.md](search.md) for the full 5-signal ranking model.

### Silent-behaviour footguns

A few defaults keep a signal or a code path quiet unless you opt in. None of
these raises — they just do less than you might expect — so they are worth
knowing before you rely on the behaviour.

- **No `current_cycle` ⇒ recency is silently inactive.** `search_hybrid()` /
  `recall()` only blend the recency signal when you pass `current_cycle=`.
  Omit it and recency is skipped entirely (its weight is redistributed to the
  other signals) — recent thoughts get no ranking advantage. There is no error;
  a store past an internal size threshold emits a single one-time `DEBUG`
  breadcrumb, nothing more. Pass the current cognitive cycle to activate it.

- **`default_graph_weight=0.0` ⇒ the graph signal is off.** As noted above, a
  default store runs no graph ranking queries. Give the weight (or a per-call
  `graph_weight=`) a non-zero value to opt in.

- **Manual `SqliteEngravaCore(...)` construction requires `ensure_schema()`.**
  The `from_config()` / `EngravaManager` factories create tables, indexes, the
  FTS5 virtual table and enable foreign keys for you. If you instead construct
  `SqliteEngravaCore(conn, ...)` directly against a raw connection, you **must**
  `await store.ensure_schema()` once before use — otherwise the tables (and FK
  enforcement) are absent and the first query fails. `ensure_schema()` is
  idempotent, so calling it on an already-migrated database is a no-op.

- **sqlite-vec falls back to numpy silently.** With
  `extensions.vector.backend: sqlite-vec`, the `vec0` backend is used **only if
  the optional `sqlite-vec` package is importable**. If it is absent (or fails
  to load), the store logs a `WARNING` and **transparently falls back to the
  brute-force numpy cosine arm** — results are identical, but you do not get the
  `vec0` index. Searches keep working, so a missing extension is easy to miss;
  install the `sqlite-vec` extra (and check for the load warning) if you intend
  to run on the native index.

### `embeddings`

Embedding provider configuration. (The YAML key is `embeddings`, plural.) The
vector dimension lives under `extensions.vector.dimension`, not here.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `provider` | `str` | `null` | Provider type: `"sentence-transformer"`, `"openai-compatible"`, `"ollama"`, `"huggingface"` |
| `model` | `str` | `null` | Model name or identifier |
| `auto_embed` | `bool` | `false` | Auto-embed on `create_thought` / `update_thought` |
| `require_embedding` | `bool` | `false` | Turn an auto-embed provider failure into a hard error. With the default `false`, a failure logs a `WARNING` naming the thought and re-raises the provider's own error; the thought is *already committed*, so it stays persisted without an embedding (invisible to vector search). Set `true` to instead raise a typed `EmbeddingGenerationError`, the explicit fail-fast an operator opts into. No effect unless `auto_embed` is on |
| `device` | `str` | `"cpu"` | Compute device for local providers (`"cpu"`, `"cuda"`) |
| `batch_size` | `int` | `32` | Batch encoding size for local providers |
| `base_url` | `str` | `null` | Base URL for remote providers |
| `api_key` | `str` | `null` | API key for remote providers (supports `${ENV_VAR}`) |
| `query_prefix` | `str` | `null` | Instruction prefix prepended to a search query before embedding (e.g. `"query: "`). Applies only to `sentence-transformer` / `ollama` / `huggingface`; `openai-compatible` ignores it. Empty/`null` is a literal passthrough — byte-identical to no prefixing |
| `document_prefix` | `str` | `null` | Instruction prefix prepended to a stored thought before embedding (e.g. `"passage: "`). Same provider scope and passthrough guarantee as `query_prefix`. Changing this on an existing store changes every stored vector and requires a deliberate re-embed (the store raises rather than silently re-embedding) |

> **Asymmetric prefixes are opt-in and for instruction-tuned models only.** Models
> like E5, BGE, GTE, and Ollama's `nomic-embed-text` are trained with mandatory
> role instructions (`"query: "` on the query, `"passage: "` on the document) and
> retrieve worse when run without them. Leave both prefixes empty (the default) for
> OpenAI and other symmetric models — the empty path is byte-identical to prior
> behaviour, and no existing store needs migrating. See
> [the embeddings guide](guides/embeddings.md#asymmetric-prefixes-for-instruction-tuned-models)
> for the full re-embed policy.

### `dreaming`

Memory consolidation configuration.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | `bool` | `false` | Enable dreaming consolidation |
| `schedule_every_n_cycles` | `int` | `100` | Consolidation cadence (every N cycles) |
| `promote_threshold` | `float` | `0.7` | Weighted-score cutoff for promotion |
| `candidates_limit` | `int` | `200` | Max thoughts to evaluate per pass |

#### `dreaming.gates`

Gate thresholds — a thought must pass all active gates to be scored.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `min_confirmations` | `int` | `2` | Minimum confirmation count. **Bypassed** when `allow_zero_confirmation` is `true`. |
| `min_age_cycles` | `int` | `1` | Minimum `current_cycle - created_cycle`. Always enforced. |
| `max_promoted_per_run` | `int` | `20` | Cap on promotions per consolidation run |
| `allow_zero_confirmation` | `bool` | `true` | Bypass the confirmation gate for single-write batches. Set to `false` only when your application explicitly tracks confirmations. |

#### `dreaming.edges`

Edge creation from dreaming. Promoted thoughts create
`ASSOCIATED` edges to their nearest neighbours.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | `bool` | `true` | Create edges on promotion |
| `top_k` | `int` | `1` | Max neighbours to link per promoted thought |
| `min_similarity` | `float` | `0.7` | Cosine threshold for edge creation |
| `edge_weight_factor` | `float` | `0.5` | `edge.weight = factor * similarity` |

See [dreaming.md](dreaming.md) for details.

### `services`

Multi-service isolation (one database file per named service, stored under a
shared `data_dir` as `<name>.db`).

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `data_dir` | `str` | **required** | Directory holding the per-service `<name>.db` files |
| `default_service` | `str` | `"main"` | Default service name when `--service` is omitted |
| `configs` | `dict` | `{}` | Map of service name → per-service config |

Each service entry under `configs` supports a single optional override (there
is no per-service `db_path` — the file is derived as `<data_dir>/<name>.db`):

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `embeddings` | `dict` | — | Per-service embedding-provider override (same shape as the top-level `embeddings` section) |

### `journal`

The hash-chain audit trail. Off by default. See [Audit Trail](audit-trail.md).

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | `bool` | `false` | Record every thought/edge mutation as a hash-linked journal entry |
| `verify_on_open` | `bool` | `false` | Re-walk the persisted hash chain when opening via `from_config` and raise `JournalIntegrityError` if it does not verify. Independent of `enabled`; adds an `O(entries)` cost per open. See [Verifying automatically on open](audit-trail.md#verifying-automatically-on-open). |

```yaml
journal:
  enabled: true
  verify_on_open: true
```

### `ttl`

Time-to-live / auto-expiry of thoughts. See the
[data-lifecycle recipes](recipes/index.md).

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `strategy` | `str` | `"archive"` | What `cleanup_expired` does to expired thoughts: `"archive"` (soft, marks `ARCHIVED`) or `"delete"` (hard) |
| `check_every_n_operations` | `int` | `0` | Run auto-cleanup every *N* store operations (`0` = manual only, via `cleanup_expired()` / `engrava gc --expired`) |
| `default_ttl_seconds` | `int \| null` | `null` | Default TTL applied to new thoughts with no explicit `expires_at` (`null` = no default) |

```yaml
ttl:
  strategy: archive          # or "delete"
  check_every_n_operations: 100
  default_ttl_seconds: 2592000   # 30 days
```

### `hygiene_policy`

The deterministic [Memory Hygiene](memory-hygiene.md) forgetting loop — a no-LLM
pass that archives cold, low-value thoughts and, separately opt-in,
garbage-collects them after a restore window. **Absent or `enabled: false` (the
default) ⇒ the loop never runs and no read/write path changes.** Distinct from
[`ttl`](#ttl): TTL expires by wall-clock `expires_at`; hygiene forgets by a
signal-derived keep-score.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | `bool` | `false` | Master switch. When `false`, the forgetting loop is entirely inert. |
| `eviction_threshold` | `float` | `0.20` | Archive a thought when its eviction-score (keep-score × decay) is below this. A deliberately low bar. |
| `protected_priorities` | `list[str]` | `["P1"]` | Priorities never auto-archived or auto-GC'd. Set to `[]` for more aggressive hygiene. (Pinning is the hard never-forget marker.) |
| `signal_weights` | `map[str, float]` | see below | Keep-score weights over the reusable signals. A partial map merges onto the defaults. |
| `check_every_n_cycles` | `int` | `1` | Cadence for the convenience pass from `consolidate()` only — an explicit `run_hygiene` bypasses it. |
| `max_evictions_per_run` | `int` | `100` | Caps **each** stage per run (≤ N archived and ≤ N GC'd). |
| `auto_gc_enabled` | `bool` | `false` | Whether Stage 2 (physical delete) runs. Enabling hygiene never implicitly enables deletion. |
| `gc_min_archive_age_cycles` | `int` | `10` | Restore window: a hygiene-archived thought is GC-eligible only after this many cycles. |
| `dry_run` | `bool` | `false` | Preview mode — compute the would-archive set (returned with reasons) without mutating or journaling. |

Default `signal_weights`: `recency 0.30`, `frequency 0.25`, `confirmation 0.20`,
`confidence 0.15`, `staleness 0.10`. (`confidence` contributes to the keep-score
but is **not** protection.)

```yaml
hygiene_policy:
  enabled: false                 # OFF by default — the whole loop is opt-in
  eviction_threshold: 0.20
  protected_priorities: ["P1"]
  signal_weights:
    recency: 0.30
    frequency: 0.25
    confirmation: 0.20
    confidence: 0.15
    staleness: 0.10
  check_every_n_cycles: 1
  max_evictions_per_run: 100
  auto_gc_enabled: false         # Stage 2 physical delete is separately opt-in
  gc_min_archive_age_cycles: 10  # restore window before a GC is eligible
  dry_run: false                 # set true to preview without mutating
```

> Garbage collection here is cognitive hygiene, not compliance deletion — it is
> best-effort and offers no deletion guarantee. See
> [Memory Hygiene](memory-hygiene.md) and
> [Data lifecycle](data-lifecycle.md) for the full mechanics.

### `ingest`

Ingest-layer behaviour (content-hash deduplication).

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `deduplication_enabled` | `bool` | `true` | Whether ingest pipelines should pass `deduplicate=True` so identical `content` collapses into one thought (bumping `confirmation_count`) instead of a duplicate row |

> Note: this flag advises ingest-layer callers; the persistence-layer
> `create_thought` still defaults to `deduplicate=False` — see
> [Recipes → Deduplicate repeated facts](recipes/index.md).

### `hooks`

Wire a custom `EngravaHooksProtocol` implementation by dotted path. See
[Extensions](extensions.md).

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `class` | `str \| null` | `null` | Dotted import path to a hooks class, last segment is the class name (e.g. `"my_package.hooks.MyHooks"`), instantiated and used by `from_config` |

```yaml
hooks:
  class: "my_package.hooks.MyHooks"
```

The path is split on the final dot (`module.path` + `ClassName`) — this is a
plain dotted path, **not** the `module.path:ATTRIBUTE` colon form used by
[`manifests.paths`](#manifests) below.

### `manifests`

Load extension manifests (their hooks + schema migrations). Accepts a plain
list of dotted paths, or a mapping with `discover` / `paths`. See
[Extensions](extensions.md).

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `paths` | `list[str]` | `[]` | Dotted `module.path:ATTRIBUTE` references to `ExtensionManifest` objects |
| `discover` | `bool` | `false` | Also scan the `engrava.extensions` entry-point group for manifests |

```yaml
# list form
manifests:
  - "my_plugin.manifest:MANIFEST"

# or mapping form
manifests:
  discover: true
  paths:
    - "my_plugin.manifest:MANIFEST"
```

> The `metrics:` section (latency window size, enable/disable) is documented in
> [Observability](observability.md).

## Environment Variables

Both are read by the **`engrava` CLI** only (library callers pass paths
explicitly to `load_config` / `SqliteEngravaCore`).

| Variable | Description |
|----------|-------------|
| `ENGRAVA_CONFIG` | Fallback path to the YAML configuration file when `--config` is omitted (`--config` > `ENGRAVA_CONFIG` > none) |
| `ENGRAVA_DB` | Fallback database-file path when `--db` is omitted (`--db` > `ENGRAVA_DB` > `./engrava.db`) |

## Multi-Service Usage

```python
from engrava import EngravaManager, load_config

config = load_config("engrava.yaml")

async with EngravaManager.from_config(config.services) as mgr:
    store = await mgr.get_store("main")
    # Use store normally...
```

See the CLI `--service` flag for command-line multi-service access.
