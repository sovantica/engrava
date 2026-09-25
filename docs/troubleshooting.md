# Troubleshooting

Common symptoms, their cause, and the fix. Each entry shows the error (or the
surprising behaviour) you actually see, then what to change.

If your problem is a platform constraint rather than a mistake (macOS extension
loading, the NumPy brute-force vector fallback, FTS5 availability), see
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
| No `embedding_provider` is configured (and you pass no `query_vector`) | the **vector** signal is skipped — only FTS/priority run. A purely semantic query with no shared keywords may find nothing. Passing your own `query_vector` re-enables the vector signal without a provider, for thoughts whose vectors you stored with `store_embedding`. |
| You pass `query_text` but no provider and no `query_vector` | same as above — there is no vector to compare against. |
| No explicit `current_cycle` or `recency_now`, and no configured `cycle_provider` | the **recency** signal is skipped because no recency reference is available. |
| `recency_weight` is `0.0` | recency is disabled even when a cycle or transaction-time reference is available. |
| The query shares no FTS tokens with any thought | FTS legitimately returns nothing — this is a real miss, not a bug. Note a *bare* query is `OR`-matched (any shared word hits), so this is rarer than it looks; if you instead get **too many** hits, you may want strict matching — see below. |
| You used lowercase `and` / `or` between words | These are **not** FTS5 operators — they are matched as ordinary words (and `OR`-joined like any bare query). Booleans must be **uppercase** (`AND`, `OR`, `NOT`). |

Inspect which signals actually ran via `HybridSearchResult.backends_used`:

```python
result = await store.search_hybrid("python async", top_k=5, current_cycle=10)
print(sorted(result.backends_used))  # e.g. ['fts5', 'priority', 'recency']
```

If `'vector'` is missing and you expected semantic matching, configure an
embedding provider (see the [Embeddings guide](guides/embeddings.md)). If
`'recency'` is missing, use a positive `recency_weight` and provide exactly one
recency mode: pass `current_cycle`, configure a `cycle_provider`, or pass
`recency_now` for transaction-time recency.

## Keyword search returns too many results (I wanted all words to match)

**Symptom.** A multi-word `search_fts` query returns documents that contain only
*some* of the words, not all of them.

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
is split into ordinary search terms, so the query is safe to run. When a
normalized full-text expression is a genuinely malformed FTS5 query, engrava logs
a warning, increments the read-only `fts_match_failure_count` counter, and
**retries once** through the bare normalization (unsafe characters dropped,
wildcards collapsed to legal prefixes, any exposed `AND`/`OR`/`NOT` phrase-quoted
so FTS5 cannot read it as an operator), which is always a valid MATCH; the FTS arm
returns that query's matches (an empty set when the sanitized query matches
nothing). It does not raise. No action needed — this is the intended robustness —
though a rising `fts_match_failure_count` is how you see it happening. See
[Keyword query syntax (FTS)](search.md#keyword-query-syntax-fts) and
[Observability signals](observability.md#observability-signals).

## Keyword search returns a thought after restore that does not contain the word

**Symptom.** After `restore` (a plain merge or `--clear`), `search_fts` /
`recall` returns a thought whose essence and content plainly do not contain
the word you searched for.

**Cause.** `restore` inserts every record with `INSERT OR REPLACE`. Before
this was fixed, a record that collided with an existing row on a primary key
or `UNIQUE` constraint made SQLite delete the old row and re-insert it
internally to resolve the conflict — and because nothing in engrava sets
`PRAGMA recursive_triggers` (SQLite's default is off), the FTS delete trigger
never fires for a row removed this way, only the insert trigger for its
replacement. The old, stale index entry survived, pointing at a rowid the
`thought` table could later hand to a completely unrelated row (a later
`--clear` restore, for instance) — at which point a keyword search for the
original word resolved to that unrelated thought instead.

**Fix.** `restore` now rebuilds the full-text index unconditionally, inside
its own transaction, after every merge or `--clear`, so a build carrying this
fix cannot leave a stale entry behind this way.

**Repair a database an older build already restored into.** Rebuild its
index directly with the SQLite CLI:

```bash
sqlite3 engrava.db "INSERT INTO thought_fts(thought_fts) VALUES('rebuild');"
```

This is FTS5's own index-rebuild command: it reconstructs `thought_fts` from
the `thought` table's current rows, using the table's already-configured
tokenizer. Verified directly against a database carrying stale entries built
this way: afterward, a `MATCH` for each of that database's thoughts named only
rows that actually contain the term, every column of every `thought` row came
back unchanged (not just which ids survived), and the `journal_entry` row
count was unchanged, and `engrava verify` afterward reported the journal valid
with that same number of entries.

## Dreaming promotes nothing (consolidation is inert)

**Symptom.** `run_consolidation(...)` returns `promoted_count == 0` every time.

**Cause.** A candidate has to pass every stage below, and failing any one keeps
the count at zero. Check them in this order:

1. **The age gate.** A thought is eligible only when
   `current_cycle - created_cycle >= min_age_cycles` (default `1`). If you never
   advance your cycle counter — every thought stays at the same `current_cycle`
   you created it in — `0 >= 1` is false and nothing is ever eligible. This is
   the most common cause. See [Core Concepts → Cycle](concepts.md).
2. **The confirmation gate.** Unless `allow_zero_confirmation` is `True` (the
   default), `confirmation_count` must be at least `min_confirmations` (default
   `2`). With the flag turned off and no confirmations recorded, nothing passes.
3. **Eligibility filters.** By default only `OBSERVATION` thoughts are promoted
   (`promote_targets`), and the metadata filters can reject a thought, for
   example the default `excluded_content_types` entry `code`. See
   [Dreaming → Eligibility filters and corpus caps](dreaming.md#eligibility-filters-and-corpus-caps).
4. **The promotion threshold.** Even after the stages above pass, a candidate's
   weighted signal score must be **strictly greater than** `promote_threshold`; a
   score exactly equal to it does not promote. Brand-new, unconfirmed,
   never-accessed thoughts score low, so a high threshold promotes nothing.
5. **The caps.** A run promotes at most `max_promoted_per_run` thoughts (default
   `20`), and it needs a free P1 slot. The number of free P1 slots for a run is
   `max_p1_fraction` of the store (default `0.05`), rounded down but never below
   one thought, minus the thoughts that are already P1, so a small store has
   only a few slots and none once its P1 count has reached that number.

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
bounded linear backoff (`base_retry_delay_s * attempt_number`) on a
*transient* failure — a read timeout or network blip, or a transient HTTP
status (`408`, `409`, `425`, `429`, `500`, `502`, `503`, `504`). Two outcomes:

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

The same error is raised when only the provider's `document_prefix` differs from
the one the corpus was embedded with: adding one to a store built without,
changing it, or removing it. A non-empty prefix is recorded as a fingerprint
next to the model name; a store built without one records none. Adding,
changing or removing a prefix is therefore a change to the corpus identity, and
the vectors already stored were produced under the old one. A change to the
`query_prefix` alone does not raise this error; it raises
`EmbeddingQueryPrefixMismatchError` at search time instead. See
[Embeddings guide → Asymmetric prefixes](guides/embeddings.md#asymmetric-prefixes-for-instruction-tuned-models).

**Fix.** Use the same embedding model and `document_prefix` the database was
created with, or restore a trusted snapshot with a configured provider and
deliberately re-embed the corpus. Direct mode uses top-level `embeddings`:

```bash
engrava --db restored.db --config engrava.yaml restore \
  -i backup.jsonl --clear --re-embed
```

Service mode prefers `services.configs.<name>.embeddings` and falls back to the
top-level provider:

```bash
engrava --config engrava.yaml restore \
  --service main -i backup.jsonl --clear --re-embed
```

`--re-embed` requires `--config` with a provider at one of those levels. If no
provider is configured, import with `--skip-embeddings` or keep the source
embeddings unchanged. If the target already contains embeddings, restore also
requires `--clear`; this prevents old vectors from surviving under the new
model/dimension/prefix identity.

See [Known Limitations → Embedding Dimension Consistency](known-limitations.md#embedding-dimension-consistency).

## `ReferentialIntegrityError` when creating an edge

**Symptom.** Creating an edge to a thought that doesn't exist raises:

```
referential integrity violation: edge.to_thought_id='...' does not reference an existing thought
```

**Cause.** One endpoint of the edge (`from_thought_id` or `to_thought_id`) is
not a real thought id. Create both thoughts before the edge that links them.

**Fix.**

```python
from engrava import ReferentialIntegrityError

try:
    await store.create_edge(edge)
except ReferentialIntegrityError:
    ...  # one endpoint is missing — create the thought, then retry
```

## Still stuck?

- Use [Error handling and recovery](error-handling.md) to decide whether to
  retry, repair the input, or replace the store.
- Re-read the relevant guide: [Core Concepts](concepts.md),
  [Search](search.md), [Embeddings](guides/embeddings.md), [Dreaming](dreaming.md).
- Check the [FAQ](faq.md) for "is this supposed to work this way?" questions.
- Confirm it isn't a documented constraint in [Known Limitations](known-limitations.md).
- Open an issue with a minimal reproduction.
