# Embeddings

Engrava's semantic (meaning-based) search is powered by **embeddings** — vector
representations of your thoughts. This guide shows how to wire a real embedding
provider so retrieval actually understands meaning, and how the query side
works.

> **Embeddings are optional.** With no provider configured, search still works
> using the bundled lexical FTS5/BM25 index — the vector signal is simply
> skipped (`HybridSearchResult.backends_used` will not contain `"vector"`). Add
> a provider to get semantic retrieval.

## Two things a provider gives you

1. **Ingest-time embedding** — with `auto_embed=True`, every thought is embedded
   on write, so it becomes findable by meaning. When the `essence` is just the
   leading prefix of `content` (a common convention, e.g. `essence = content[:200]`),
   only `content` is embedded — the redundant prefix is dropped so it can't dominate
   the vector. A genuinely distinct `essence` is still embedded alongside `content`.
2. **Query-time embedding** — at search time the query must also be a vector.
   `search_hybrid` takes the query *text* and, when a provider is configured,
   embeds it **for you** (unless you pass an explicit `query_vector`).
   `search_similar` takes a *vector* directly, so you embed the query yourself
   first. See [The query side](#the-query-side) for both.

The corpus and the query must use **the same model / dimension** — once a store
has embeddings for one model, writing with a different model raises
`EmbeddingModelMismatchError`.

## Wiring a provider

Pass the provider to the store constructor (and set `auto_embed=True`):

```python
import aiosqlite
from engrava import SqliteEngravaCore, SentenceTransformerProvider

provider = SentenceTransformerProvider(model_name="all-MiniLM-L6-v2")
async with aiosqlite.connect("engrava.db") as conn:
    conn.row_factory = aiosqlite.Row
    store = SqliteEngravaCore(conn, embedding_provider=provider, auto_embed=True)
    await store.ensure_schema()
```

Or declare it in `engrava.yaml` and let `from_config` build it (see the
[`embeddings` section](../configuration.md)):

```yaml
embeddings:
  provider: sentence-transformer
  model: all-MiniLM-L6-v2
  auto_embed: true
```

```python
from engrava import SqliteEngravaCore

async with await SqliteEngravaCore.from_config("engrava.yaml") as store:
    ...   # provider wired from config, auto_embed honoured
```

## Providers

Every provider implements the same async interface — `await provider.embed(text)`
returns a `list[float]` — so they're interchangeable. Pick by where you want the
model to run.

### `SentenceTransformerProvider` — local model (no API, no network)

Runs a sentence-transformers model on your machine. Requires the
`embeddings-local` extra (pulls `sentence-transformers` + `torch`).

```bash
pip install "engrava[embeddings-local]"
```

```python
from engrava import SentenceTransformerProvider

provider = SentenceTransformerProvider(
    model_name="all-MiniLM-L6-v2",   # default: all-MiniLM-L12-v2
    device="cpu",                    # or "cuda"
    batch_size=32,
)
```

No API key, no network after the first model download. Best default for
self-hosting.

On load, this provider raises the model's `max_seq_length` to the architecture's
true maximum when the shipped checkpoint reports a conservatively-low value — the
bundled `all-MiniLM-L12-v2`, for instance, ships `128` while its backbone supports
`512`. Without this, the tail of any longer thought would be silently truncated
before encoding. The value is read from the model, not hard-coded, so a model that
already reports its real maximum is left unchanged.

### `OpenAICompatibleProvider` — OpenAI or any OpenAI-compatible API

Calls an OpenAI-style `/embeddings` endpoint. Requires the `embeddings-openai`
extra (pulls `httpx`).

```bash
pip install "engrava[embeddings-openai]"
```

```python
import os
from engrava import OpenAICompatibleProvider

provider = OpenAICompatibleProvider(
    model_name="text-embedding-3-small",   # this is the default
    base_url="https://api.openai.com/v1",  # default; point at any compatible API
    api_key=os.environ["OPENAI_API_KEY"],  # or omit — falls back to $OPENAI_API_KEY
)
```

`api_key` defaults to the `OPENAI_API_KEY` environment variable when omitted.
Set `base_url` to target a compatible gateway (Azure OpenAI, a local proxy, etc.).

**Automatic retry on transient failures.** This provider retries a request with
bounded exponential backoff when the endpoint reports a transient failure — a read
timeout or network blip, or a transient HTTP status (`408`, `409`, `425`, `429`,
`500`, `502`, `503`, `504`) — so a short outage is absorbed instead of failing your
ingest. Non-transient statuses (`400`, `401`, `403`, `404`) surface immediately
with no retry, and a transient failure that persists across every attempt is still
raised (the call never loops forever). Two keyword-only knobs tune it:
`max_attempts` (default `3`) and `base_retry_delay_s` (default `1.0`); the defaults
leave the success path at a single request, so existing callers see no change. This
applies to `OpenAICompatibleProvider` only — `OllamaProvider` and
`HuggingFaceProvider` do not retry.

```python
provider = OpenAICompatibleProvider(
    model_name="text-embedding-3-small",
    max_attempts=5,           # up to 5 tries on transient failures
    base_retry_delay_s=0.5,   # exponential backoff starting at 0.5s
)
```

### `OllamaProvider` — local Ollama server

Calls a running [Ollama](https://ollama.com) instance. Requires the
`embeddings-ollama` extra (pulls `httpx`); no API key.

```bash
pip install "engrava[embeddings-ollama]"
```

```python
from engrava import OllamaProvider

provider = OllamaProvider(
    model_name="nomic-embed-text",          # default
    base_url="http://localhost:11434",      # default Ollama address
)
```

### `HuggingFaceProvider` — HuggingFace Inference API

Calls the HuggingFace Inference API. Requires the `embeddings-hf` extra (pulls
`huggingface_hub`).

```bash
pip install "engrava[embeddings-hf]"
```

```python
import os
from engrava import HuggingFaceProvider

provider = HuggingFaceProvider(
    model_name="sentence-transformers/all-MiniLM-L12-v2",  # default
    api_key=os.environ["HF_TOKEN"],   # or omit — falls back to $HF_TOKEN
)
```

`api_key` defaults to the `HF_TOKEN` environment variable when omitted.

### `CallbackProvider` — bring your own embedding function

Wrap any function `str -> list[float]`. Built-in (no extra). Use it for a custom
model, a cached lookup, or testing.

```python
from engrava import CallbackProvider

provider = CallbackProvider(
    callback=my_embed_fn,   # str -> list[float]
    dimension=384,          # the length your callback returns
    model_name="my-model",
)
```

> Do **not** ship a placeholder like `lambda text: [0.1] * 384` — a constant
> vector makes every thought identical, so similarity is meaningless. Use a real
> model (the providers above) or a genuine embedding function.

## The query side

The two search methods handle the query vector differently — `search_hybrid`
takes the query **text**, `search_similar` takes a query **vector**.

**`search_hybrid(query_text, query_vector=None, ...)`** — pass the query text.
When an embedding provider is configured, Engrava embeds that text for you if
you don't supply a `query_vector`; pass one explicitly only to override:

```python
# Provider configured → the query text is embedded for you:
result = await store.search_hybrid("trips to Japan", top_k=5, current_cycle=cycle)

# Or override with a vector you already have:
query_vec = await provider.embed("trips to Japan")
result = await store.search_hybrid("trips to Japan", query_vector=query_vec, top_k=5)
```

If **no** provider is configured **and** you pass no `query_vector`,
`search_hybrid` skips the vector signal and falls back to the lexical (FTS5/BM25)
signal — still useful, just keyword-based rather than semantic.

**`search_similar(query_vector, ...)`** — takes a ready vector as its first,
required argument. It does not accept query text, so there is nothing for it to
auto-embed: you must embed the query yourself first.

```python
query_vec = await provider.embed("trips to Japan")   # required — no auto-embed here
result = await store.search_similar(query_vec, top_k=5)
```

## Choosing a model and dimension

- **Keep one model per store.** The query and corpus vectors must come from the
  same model; switching models on an existing store requires re-embedding (see
  `engrava restore --re-embed`).
- **Dimension follows the model.** Local/HF providers infer it from the model;
  `CallbackProvider` requires you to declare `dimension` to match what your
  callback returns. For the `sqlite-vec` vector backend, set
  `extensions.vector.dimension` in config to match.

## Config-driven equivalents

Each provider has a `provider:` name for `engrava.yaml`, resolved by
`resolve_embedding_provider(config.embeddings)`:

| `provider:` value | Class | Extra |
|---|---|---|
| `sentence-transformer` | `SentenceTransformerProvider` | `embeddings-local` |
| `openai-compatible` | `OpenAICompatibleProvider` | `embeddings-openai` |
| `ollama` | `OllamaProvider` | `embeddings-ollama` |
| `huggingface` | `HuggingFaceProvider` | `embeddings-hf` |

`CallbackProvider` takes a Python callable, so it's wired in code (via the
`embedding_provider=` constructor argument), not YAML.

## Next

- [Configuration](../configuration.md) — the `embeddings` YAML section.
- [Hybrid Search](../search.md) — how the vector signal fuses with the others.
- [Building a memory-backed agent](agent-memory.md) — embeddings in the agent loop.
