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
  default_graph_weight: 0.00       # opt-in graph signal
  recency_half_life: 50
  priority_boost_p1: 1.0
  priority_boost_p2: 0.6
  priority_boost_p3: 0.3
  priority_boost_p4: 0.0
  graph_edge_decay: 0.5            # 1-hop distance penalty
  max_neighbors_per_candidate: 5   # safety cap

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

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `default_fts_weight` | `float` | `0.30` | Weight for FTS5/BM25 text score |
| `default_vector_weight` | `float` | `0.55` | Weight for vector similarity score |
| `default_recency_weight` | `float` | `0.10` | Weight for recency-based score |
| `default_priority_weight` | `float` | `0.05` | Weight for priority signal |
| `default_graph_weight` | `float` | `0.0` | Weight for 1-hop graph signal (opt-in) |
| `recency_half_life` | `int` | `50` | Cycles for recency score to halve |
| `priority_boost_p1` | `float` | `1.0` | Score multiplier for P1 thoughts |
| `priority_boost_p2` | `float` | `0.6` | Score multiplier for P2 thoughts |
| `priority_boost_p3` | `float` | `0.3` | Score multiplier for P3 thoughts |
| `priority_boost_p4` | `float` | `0.0` | Score multiplier for P4 thoughts |
| `graph_edge_decay` | `float` | `0.5` | Decay factor for 1-hop neighbour boost |
| `max_neighbors_per_candidate` | `int` | `5` | Max neighbours considered per candidate |

Weights are redistributed proportionally when a signal is unavailable
(e.g. no `current_cycle` → recency skipped). Set any weight to `0.0`
to disable that signal entirely.

See [search.md](search.md) for the full 5-signal ranking model.

### `embeddings`

Embedding provider configuration. (The YAML key is `embeddings`, plural.) The
vector dimension lives under `extensions.vector.dimension`, not here.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `provider` | `str` | `null` | Provider type: `"sentence-transformer"`, `"openai-compatible"`, `"ollama"`, `"huggingface"` |
| `model` | `str` | `null` | Model name or identifier |
| `auto_embed` | `bool` | `false` | Auto-embed on `create_thought` / `update_thought` |
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

```yaml
journal:
  enabled: true
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
