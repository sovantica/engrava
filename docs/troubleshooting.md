# Troubleshooting

Common symptoms, their cause, and the fix. Each entry shows the error (or the
surprising behaviour) you actually see, then what to change.

If your problem is a platform constraint rather than a mistake (macOS extension
loading, the ~100k brute-force ceiling, FTS5 availability), see
[Known Limitations](known-limitations.md) instead.

## `AttributeError: 'tuple' object has no attribute 'keys'` on read

**Symptom.** Writes succeed, but the first `get_thought` / search call raises:

```
AttributeError: 'tuple' object has no attribute 'keys'
```

**Cause.** The aiosqlite connection has no row factory, so rows come back as
plain tuples. Engrava maps rows to records by column name and needs
`aiosqlite.Row`. The failure surfaces on **read**, not on connect or write,
which makes it look unrelated to setup.

**Fix.** Set the row factory immediately after connecting:

```python
import aiosqlite

conn = await aiosqlite.connect("engrava.db")
conn.row_factory = aiosqlite.Row  # required
```

`SqliteEngravaCore.from_config(...)` opens the connection for you and sets this
correctly — the manual snippet above only applies when you construct the store
from your own connection.

## `ValueError: '...' is not a valid ThoughtType` (or `Priority`, `EdgeType`, …)

**Symptom.**

```
ValueError: 'INSIGHT' is not a valid ThoughtType
```

**Cause.** A string was passed that is not a member of the enum. The valid
`ThoughtType` members are `TASK`, `OBSERVATION`, `BELIEF`, `REFLECTION`,
`OUTPUT_DRAFT`, and `NOTE` — there is no `INSIGHT`. The same applies to
`Priority` (`P1`–`P4`), `EdgeType`, `LifecycleStatus`, etc.

**Fix.** Use a real enum member, ideally the symbol rather than a string literal:

```python
from engrava import ThoughtType

ThoughtType.BELIEF  # preferred
ThoughtType("BELIEF")  # also valid — must match a real member
```

See [Core Concepts](concepts.md) for the full taxonomy and when to use each type.

## Search returns nothing (or fewer results than expected)

**Symptom.** `search_hybrid` / `search_fts` returns an empty or short result
list even though matching thoughts exist.

**Cause.** A signal you assumed was active was **silently skipped**, so the query
ran on fewer signals than you expected. Engrava skips a signal rather than
erroring when its prerequisite is missing. Work through this checklist:

| If… | then… |
|---|---|
| No `embedding_provider` is configured | the **vector** signal is skipped — only FTS/priority run. A purely semantic query with no shared keywords may find nothing. |
| You pass `query_text` but no provider and no `query_vector` | same as above — there is no vector to compare against. |
| `current_cycle` is `None` | the **recency** signal is skipped (it cannot compute an age). |
| `recency_weight` is `0.0` | recency is disabled even if `current_cycle` is set. |
| The query shares no FTS tokens with any thought | FTS legitimately returns nothing — this is a real miss, not a bug. Note a *bare* query is `OR`-matched (any shared word hits), so this is rarer than it looks; if you instead get **too many** hits, you may want strict matching — see below. |
| You used lowercase `and` / `or` between words | These are **not** FTS5 operators — they are matched as ordinary words (and `OR`-joined like any bare query). Booleans must be **uppercase** (`AND`, `OR`, `NOT`). |

Inspect which signals actually ran via `HybridSearchResult.backends_used`:

```python
result = await store.search_hybrid("python async", top_k=5, current_cycle=10)
print(sorted(result.backends_used))  # e.g. ['fts5', 'priority', 'recency']
```

If `'vector'` is missing and you expected semantic matching, configure an
embedding provider (see the [Embeddings guide](guides/embeddings.md)). If
`'recency'` is missing, pass a non-`None` `current_cycle` **and** a
`recency_weight > 0`.

## Keyword search returns too many results (I wanted all words to match)

**Symptom.** A multi-word `search_fts` / `search_keywords` query returns documents
that contain only *some* of the words, not all of them.

**Cause.** A **bare** keyword query is matched with `OR`, by design — a document
matches when it shares *any* word, and BM25 ranks the ones sharing the most
distinctive words first (see [Keyword query syntax](search.md#keyword-query-syntax-fts)).
This is what lets natural-language questions find relevant answers; it is not a bug.

**Fix.** When you genuinely need every term, use FTS5 expert syntax explicitly —
**uppercase** `AND` between the words, or a quoted phrase for an exact sequence:

```python
# require both words
await store.search_fts("python AND asyncio", top_k=10)
# require an exact phrase
await store.search_fts('"event loop"', top_k=10)
```

Lowercase `and`/`or` will **not** work — they are matched as ordinary words.

## Pasting a URL, path, or timestamp into search

**Symptom.** You expect a query containing `http://…`, `12:30`, or a Windows path
to error or to be interpreted as an FTS column filter.

**Cause / behaviour.** It does neither. Only the real `essence:` and `content:`
column filters are honoured; any other `token:token` (a URL scheme, a clock time)
is split into ordinary search terms, so the query is safe to run and a genuinely
malformed FTS expression degrades to zero FTS hits rather than raising. No action
needed — this is the intended robustness.

## Dreaming promotes nothing (consolidation is inert)

**Symptom.** `run_consolidation(...)` returns `promoted_count == 0` every time.

**Cause.** Promotion requires a candidate to clear **two independent bars**, and
either one alone keeps the count at zero:

1. **The age gate.** A thought is eligible only when
   `current_cycle - created_cycle >= min_age_cycles` (default `1`). If you never
   advance your cycle counter — every thought stays at the same `current_cycle`
   you created it in — `0 >= 1` is false and nothing is ever eligible. This is
   the most common cause. See [Core Concepts → Cycle](concepts.md).
2. **The promotion threshold.** Even after the gate passes, a candidate's
   weighted signal score must reach `promote_threshold`. Brand-new, unconfirmed,
   never-accessed thoughts score low, so a high threshold promotes nothing.

**Fix.**

```python
from engrava.config import DreamingConfig, DreamingGates
from engrava.extensions.dreaming import DreamingExtension

config = DreamingConfig(
    enabled=True,
    promote_threshold=0.4,  # lower it if nothing clears the bar
    gates=DreamingGates(
        allow_zero_confirmation=True,  # essential for single-write ingest
        min_age_cycles=1,
    ),
)
ext = DreamingExtension(config=config)

# Advance current_cycle past the thoughts' created_cycle so the age gate passes:
result = await ext.run_consolidation(store, current_cycle=10)
print(result.promoted_count)
```

See [Dreaming](dreaming.md) for the full gate-and-signal model.

## Embedding ingest fails against an OpenAI-compatible endpoint

**Symptom.** Writes that auto-embed (or explicit embed calls) raise after a pause,
either immediately or after a few seconds of retrying.

**Cause / what to expect.** `OpenAICompatibleProvider` retries a request with
bounded exponential backoff on a *transient* failure — a read timeout or network
blip, or a transient HTTP status (`408`, `409`, `425`, `429`, `500`, `502`, `503`,
`504`). Two outcomes:

- **A transient failure that persists across every attempt** is raised as a
  `RuntimeError` once `max_attempts` is exhausted (it never loops forever). If you
  see this under sustained `429`s, you are being rate-limited — raise
  `base_retry_delay_s`, lower your ingest concurrency, or batch more slowly.
- **A non-transient status** (`400`, `401`, `403`, `404`) is surfaced
  **immediately with no retry** — it indicates a request/auth/model error, not a
  blip. Check your `api_key`, `base_url`, and `model_name`.

**Fix.** Tune the retry budget on the provider (`max_attempts`, default `3`;
`base_retry_delay_s`, default `1.0`), or address the underlying cause above. Only
`OpenAICompatibleProvider` retries — `OllamaProvider` / `HuggingFaceProvider` do
not. See the [Embeddings guide](guides/embeddings.md#openaicompatibleprovider--openai-or-any-openai-compatible-api).

## `EmbeddingModelMismatchError` when opening an existing database

**Symptom.** A store that worked before now raises `EmbeddingModelMismatchError`
on startup or first embed.

**Cause.** Engrava records the embedding **model name and dimension** in the
database the first time it embeds. If you later open that same database with a
different model name or a different dimension, the stored vectors are
incompatible with new ones, so it refuses rather than silently mixing
dimensions (which would corrupt similarity results).

**Fix.** Use the same embedding model the database was created with, or
re-embed the corpus under the new model. The CLI does this safely:

```bash
engrava restore --re-embed   # validates model consistency, re-embeds
```

See [Known Limitations → Embedding Dimension Consistency](known-limitations.md#embedding-dimension-consistency).

## `ReferentialIntegrityError` — and you can't import it from `engrava`

**Symptom.** Creating an edge to a thought that doesn't exist raises:

```
referential integrity violation: edge.to_thought_id='...' does not reference an existing thought
```

…and the obvious import fails:

```python
from engrava import ReferentialIntegrityError  # ImportError!
```

**Cause (two parts).**

1. **The error itself** means one endpoint of an edge (`from_thought_id` or
   `to_thought_id`) is not a real thought id. Create both thoughts before the
   edge that links them.
2. **The import:** `ReferentialIntegrityError` is **not** re-exported from the
   top-level `engrava` package. It lives in `engrava.domain.exceptions`.

**Fix.** Import it from its real module, and ensure both endpoints exist first:

```python
from engrava.domain.exceptions import ReferentialIntegrityError

try:
    await store.create_edge(edge)
except ReferentialIntegrityError:
    ...  # one endpoint is missing — create the thought, then retry
```

The exceptions that *are* re-exported at the top level are `EngravaError` (the
base), `ConfigError`, `EmbeddingModelMismatchError`, `ExtensionMigrationError`,
`InvalidTransitionError`, `MindQLParseError`, `ReadOnlyViolationError`,
`StaleDataError`, and `ThoughtNotFoundError`. Anything else lives under
`engrava.domain.exceptions`.

## Still stuck?

- Re-read the relevant guide: [Core Concepts](concepts.md),
  [Search](search.md), [Embeddings](guides/embeddings.md), [Dreaming](dreaming.md).
- Check the [FAQ](faq.md) for "is this supposed to work this way?" questions.
- Confirm it isn't a documented constraint in [Known Limitations](known-limitations.md).
- Open an issue with a minimal reproduction.
