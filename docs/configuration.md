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
    promote_threshold: 0.55
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
provider = resolve_embedding_provider(config)
```

## Configuration Reference

### Top-Level

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `db_path` | `str` | `"engrava.db"` | Path to the SQLite database file |

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

### `embedding`

Embedding provider configuration.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `provider` | `str` | `null` | Provider type: `"sentence-transformer"`, `"openai"`, `"ollama"`, `"huggingface"`, `"callback"` |
| `model` | `str` | — | Model name or identifier |
| `dimension` | `int` | — | Vector dimensionality |
| `api_base` | `str` | — | API base URL (for `openai`, `ollama`) |
| `api_key` | `str` | — | API key (for `openai`, `huggingface`) |

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

Multi-service isolation (one database per named service).

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `default_service` | `str` | — | Default service name when `--service` is omitted |
| `entries` | `dict` | `{}` | Map of service name → service config |

Each service entry supports:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `db_path` | `str` | — | Path to this service's SQLite database |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ENGRAVA_CONFIG` | Path to the YAML configuration file |
| `ENGRAVA_DB` | Override `db_path` from configuration |

## Multi-Service Usage

```python
from engrava import EngravaManager, load_config

config = load_config("engrava.yaml")

async with EngravaManager.from_config(config.services) as mgr:
    store = await mgr.get_store("main")
    # Use store normally...
```

See the CLI `--service` flag for command-line multi-service access.
