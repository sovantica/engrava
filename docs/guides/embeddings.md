# Embeddings

Engrava's semantic (meaning-based) search is powered by **embeddings** — vector
representations of your thoughts. This guide shows how to wire a real embedding
provider so retrieval actually understands meaning, and how the query side
works.

> **Embeddings are optional.** With no provider configured, search still works
> using the bundled lexical FTS5/BM25 index — the vector signal is skipped
> (`HybridSearchResult.backends_used` will not contain `"vector"`) unless you
> pass your own `query_vector` to `search_hybrid` for thoughts whose vectors you
> stored with `store_embedding`. A provider plus `auto_embed=True` (new
> thoughts are embedded on write) gives semantic retrieval without supplying
> vectors yourself; otherwise store vectors yourself with `store_embedding`.

## Two things a provider gives you

1. **Ingest-time embedding** — with `auto_embed=True`, every thought is embedded
   on write, so it becomes findable by meaning. When the `essence` is just the
   leading prefix of `content` (a common convention, e.g. `essence = content[:200]`),
   only `content` is embedded — the redundant prefix is dropped so it can't dominate
   the vector. A genuinely distinct `essence` is still embedded alongside `content`.
   **`restore --re-embed` does not apply this dedup** — it always embeds
   `essence + "\n" + content`, unconditionally. For a corpus using the
   `essence = content[:200]` convention, a re-embedded restore is therefore
   *not* vector-identical to the corpus it replaces.
2. **Query-time embedding** — at search time the query must also be a vector.
   `search_hybrid` takes the query *text* and, when a provider is configured,
   embeds it **for you** (unless you pass an explicit `query_vector`).
   `search_similar` takes a *vector* directly, so you embed the query yourself
   first. See [The query side](#the-query-side) for both.

The corpus and the query must use **the same model / dimension** — once a store
has embeddings for one model, writing with a different model raises
`EmbeddingModelMismatchError`. For instruction-tuned models the corpus also pins
the `document_prefix` it was built with (see
[Asymmetric prefixes](#asymmetric-prefixes-for-instruction-tuned-models)).

## Wiring a provider

Pass the provider to the store constructor (and set `auto_embed=True`):

```python
import aiosqlite
from engrava import SqliteEngravaCore, SentenceTransformerProvider

provider = SentenceTransformerProvider(model_name="all-MiniLM-L6-v2")
# Abbreviated for brevity: a bare async-with here can let a close failure
# mask a real error. See "Graceful shutdown" in deployment.md for the safe
# explicit-close shape.
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

### When auto-embed fails (the honest boundary)

`create_thought` runs auto-embed **after** inserting the thought row — but
whether that row is durable yet depends on whether this call owns the
outermost transaction. On its own (the common case, no enclosing
`suspend_auto_commit()` window), the insert has already committed, so if the
embedding provider fails (network blip, rate limit, a crashing local model), the
thought is already persisted — it just has no embedding, which means it is
**invisible to vector search** until re-embedded. Nested inside the caller's
own `suspend_auto_commit()` window, nothing is durable yet: the outermost
window's exit decides — caught and exited cleanly, the row commits; uncaught,
it rolls back with the rest of that window. This is surfaced two ways:

- An `Exception` that escapes the guarded embed call (`provider.embed()` /
  `embed_batch()`, or their role-aware equivalents) is never silent: a
  `WARNING` naming the thought id and the provider error is always logged,
  then the provider's own exception propagates (unchanged default
  behaviour). That guarantee is scoped to that one call, not to provider
  failures in general — a provider whose `model_name` property raises (read
  right after a successful embed, to store alongside the vector) skips the
  logging and typing entirely, regardless of `require_embedding` or
  exception type. A provider that raises `asyncio.CancelledError` (or any
  other `BaseException` that is not an `Exception`) from inside the guarded
  embed call also skips it — it propagates directly, with no `WARNING` and
  without ever becoming `EmbeddingGenerationError`.
- Set `require_embedding: true` (config) or `require_embedding=True` (constructor)
  to turn that failure into a typed `EmbeddingGenerationError` — the explicit
  fail-fast for operators who would rather the write raise loudly than leave an
  unembedded thought behind. The flag decides the exception *type*, not the
  durability outcome above by itself — but that type *can* affect durability
  when the failure happens nested inside a caller's own
  `suspend_auto_commit()` window, if the caller's own exception handling
  distinguishes the two types: an `except EmbeddingGenerationError` clause
  around the call catches the strict-mode error and lets that window exit
  cleanly (so it commits), while the same clause does not catch the untyped
  provider exception raised by default, which escapes the window and rolls
  it back. A caller that instead catches both types, or neither, sees the
  same outcome regardless of this flag.

```python
import aiosqlite
from engrava import SqliteEngravaCore, EmbeddingGenerationError

async def strict_ingest(provider: object, text: str) -> None:
    # Abbreviated for brevity: a bare async-with here can let a close
    # failure mask a real error. See "Graceful shutdown" in deployment.md
    # for the safe explicit-close shape.
    async with aiosqlite.connect("engrava.db") as conn:
        conn.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(
            conn,
            embedding_provider=provider,  # type: ignore[arg-type]
            auto_embed=True,
            require_embedding=True,  # embed failure -> EmbeddingGenerationError
        )
        await store.ensure_schema()
        try:
            await store.remember(text)
        except EmbeddingGenerationError as exc:
            # The thought exists but is unembedded; decide how to recover.
            print(f"embed failed for {exc.thought_id}")
```

### Batch ingest embeds in one call

`bulk_store` persists many thoughts in a single transaction that is
all-or-nothing when it owns that transaction outright — nested inside a
caller's own `suspend_auto_commit()`, a caught failure can still commit the
batch's already-inserted rows (see the [API reference](../api-reference.md))
— and, when `auto_embed` is on, embeds them all in **one** batch provider call (using
the role-aware `embed_document_batch` when the provider exposes it, else
`embed_batch`). Whether that call saves round trips depends on the provider:
`OpenAICompatibleProvider` and `OllamaProvider` send the whole batch in one
request per attempt (`OpenAICompatibleProvider` sends it again when it retries a
transient failure; `OllamaProvider` does not retry), and
`SentenceTransformerProvider` encodes it in a single model call, but
`HuggingFaceProvider` and `CallbackProvider` implement the batch as a loop that
still calls the underlying API or callback once per text. A custom provider
can implement a genuine batch. The stored vectors are identical to embedding
each thought individually. `get_or_create` and
`upsert_by_hash` are content-hash convenience writes over the same
deduplication — see [the write-API guide](agent-memory.md) and the
[API reference](../api-reference.md).

```python
import aiosqlite
from engrava import SqliteEngravaCore, ThoughtRecord, ThoughtType, Priority, LifecycleStatus

async def batch_ingest(provider: object, texts: list[str]) -> None:
    # Abbreviated for brevity: a bare async-with here can let a close
    # failure mask a real error. See "Graceful shutdown" in deployment.md
    # for the safe explicit-close shape.
    async with aiosqlite.connect("engrava.db") as conn:
        conn.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(
            conn,
            embedding_provider=provider,  # type: ignore[arg-type]
            auto_embed=True,
        )
        await store.ensure_schema()
        thoughts = [
            ThoughtRecord(
                thought_id=f"t-{i}",
                thought_type=ThoughtType.OBSERVATION,
                essence=t[:200],
                content=t,
                priority=Priority.P3,
                lifecycle_status=LifecycleStatus.ACTIVE,
                source="batch",
            )
            for i, t in enumerate(texts)
        ]
        # One transaction, one commit, one embed_batch call for the whole list.
        await store.bulk_store(thoughts)
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
linear backoff (`base_retry_delay_s * attempt` — not exponential, despite how
this is sometimes described) when the endpoint reports a transient failure — a read
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
    base_retry_delay_s=0.5,   # linear backoff starting at 0.5s (attempt 2 waits 1.0s, ...)
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

## Asymmetric prefixes for instruction-tuned models

Some embedding models are **instruction-tuned**: they were trained with a
mandatory role instruction on every input — typically `"query: "` on a search
query and `"passage: "` (or `"search_document: "`) on a stored document. Running
such a model without those instructions leaves noticeable retrieval quality on
the table. This family includes E5, BGE, GTE, and Ollama's `nomic-embed-text`.

`SentenceTransformerProvider`, `OllamaProvider`, and `HuggingFaceProvider` accept
an optional, keyword-only `query_prefix` / `document_prefix` for exactly this.
The document prefix is applied on every document-embed path (ingest auto-embed
and configured `restore --re-embed`, in direct or service mode); the query
prefix on every query-embed path (`search_hybrid`, `recall`, and reflection
search).

```python
provider = SentenceTransformerProvider(
    model_name="intfloat/e5-base-v2",
    query_prefix="query: ",
    document_prefix="passage: ",
)
```

Or in `engrava.yaml`:

```yaml
embeddings:
  provider: sentence-transformer
  model: intfloat/e5-base-v2
  query_prefix: "query: "
  document_prefix: "passage: "
  auto_embed: true
```

A few rules make this safe to adopt incrementally:

- **Opt-in and default-off.** Both prefixes default to empty. An empty prefix is a
  literal passthrough — the text is embedded exactly as before, with no separator
  or whitespace added — so a store with no prefixes configured (and every symmetric
  model, including OpenAI) behaves byte-identically to prior versions. Nothing to
  migrate.
- **The symmetric OpenAI provider ignores prefixes entirely.** `OpenAICompatibleProvider`
  has no prefix parameters; a `query_prefix` / `document_prefix` in config is simply
  not passed to it.
- **Changing `document_prefix` requires a deliberate re-embed.** The document prefix
  is part of the corpus identity: change it and every stored vector would change.
  Turning it on (or changing it) on a store that already holds vectors raises
  `EmbeddingModelMismatchError` — Engrava never silently re-embeds. Re-embed on
  purpose by restoring a snapshot with `--re-embed` and `--config` (which applies
  the top-level provider in direct mode, or a per-service override before the
  top-level fallback), or start a fresh store. Restore replaces the model,
  dimension, document-prefix fingerprint, and query-prefix pairing in the same
  transaction as the new vectors. Use a fresh target or `--clear` when the
  target already contains embeddings. If the database has a persisted
  sqlite-vec index, restore drops it transactionally and the next configured
  open rebuilds it from `embedding`; keep the `engrava[vec]` extra installed for
  that reset.
- **Changing `query_prefix` alone does *not* re-embed anything** — it does not touch
  stored vectors. But the corpus records the query prefix it was built to pair with,
  so querying with a *divergent* query prefix raises `EmbeddingQueryPrefixMismatchError`
  loudly at search time rather than silently degrading ranking. The fix is to restore
  the matching query prefix (or deliberately re-embed with a new document prefix).
- **Prefixes count toward the context window.** A configured prefix is prepended
  before encoding, so a long document whose text *plus* prefix exceeds the model's
  `max_seq_length` truncates exactly as an unprefixed over-length input would — there
  is no special reservation for the prefix.

## Choosing a model and dimension

- **Keep one model per store.** The query and corpus vectors must come from the
  same model; switching models on an existing store requires a deliberate
  migration. The CLI can re-embed while restoring a snapshot with a configured
  top-level or per-service provider; see [CLI restore](../cli.md#restore).
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
