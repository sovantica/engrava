# Changelog

All notable changes to engrava will be documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

## <small>0.3.1 (2026-06-02)</small>

* fix(vector): load sqlite-vec extension on the connection's worker thread ([457d2f7](https://github.com/sovantica/engrava/commit/457d2f7))
* fix(vector): re-disable extension loading in finally after load attempt ([e9ab267](https://github.com/sovantica/engrava/commit/e9ab267))
* docs: point documentation url to engrava.ai/docs ([aeeb3cb](https://github.com/sovantica/engrava/commit/aeeb3cb))

## [Unreleased]

### Added

- **`remember` and `recall`: store and retrieve in one call.** Two ergonomic
  convenience methods on the store let you persist a string and get relevant
  strings back without hand-building a `ThoughtRecord` or wiring up
  `search_hybrid`. `remember(text, *, metadata=None, deduplicate=False)` stores
  a string as a thought (deriving the essence from its opening) and honours
  opt-in content deduplication; `recall(query, *, top_k=10, current_cycle=None)`
  returns the ranked matches. Passing `current_cycle` to `recall` blends in the
  recency signal; on a large store recalled without it, a single DEBUG log line
  points out that a cycle would let recent thoughts rank higher. `ThoughtRecord`
  now defaults `created_cycle` and `updated_cycle` to `0`, so callers that do
  not track cognitive cycles can omit them.

- **Bi-temporal model: track when a fact is *true*, not just when you stored
  it.** `ThoughtRecord` and `EdgeRecord` gain two optional, nullable ISO-8601
  fields — `valid_from` and `valid_until` — describing the half-open real-world
  interval during which a fact holds (the upper bound is exclusive; a `None`
  bound is treated as ±∞, so facts you never annotate keep matching every query).
  Four opt-in MindQL `WHERE` predicates query this *valid time* on the `thoughts`
  and `edges` tables: `valid_now`, `valid_at <ts>`, `valid_within <start> <end>`
  (interval overlap), and `valid_between <start> <end>` (fully contained — the one
  predicate that excludes open-bounded rows). Two new store primitives,
  `invalidate_thought(id, valid_until)` and `invalidate_edge(id, valid_until)`,
  retire a fact by closing its interval instead of deleting it — deterministic,
  idempotent, non-cascading, and fully auditable (the row stays on file and a
  point-in-time query before the cut-off still finds it). Reflections built by
  dreaming inherit their members' valid-time extent (open-on-either-side is
  contagious). A query that uses no temporal predicate behaves exactly as before.
  See the [Bi-temporal Model](docs/bitemporal.md) guide.

- **MCP server: connect any MCP client to an engrava store.** A new optional
  `mcp` extra (`pip install "engrava[mcp]"`) ships a Model Context Protocol
  server that exposes a store over stdio to Claude Desktop, Claude Code, Cursor,
  Windsurf, VS Code, and other MCP clients. Two entry points, `engrava-mcp` and
  `python -m engrava.mcp`, build the same server. It registers eleven tools (six
  read, five write), two static `engrava://` resources plus an
  `engrava://thought/{thought_id}` resource template, and three prompt templates,
  resolving its store from `ENGRAVA_MCP_CONFIG` (an `engrava.yaml`) or
  `ENGRAVA_DB_PATH` (a bare database file). A read-only mode
  (`ENGRAVA_MCP_READ_ONLY`) drops the five write tools entirely, leaving a
  retrieval-only surface. The server is a pure API consumer — plain
  `pip install engrava` is unaffected and stays dependency-light. See the
  [MCP server](docs/guides/mcp.md) guide.

- **`execute_mindql` on the store.** `SqliteEngravaCore.execute_mindql(query, *,
  extensions=None)` runs a parsed `MindQLQuery` directly against the store's own
  connection, returning a `MindQLResult` — a convenience over constructing a
  `MindQLExecutor` by hand. See the [API Reference](docs/api-reference.md#mindql).

### Performance

- **Hot-path indexes and tuned SQLite PRAGMAs make the common reads faster.**
  Four indexes now back the equality filters and the sort column hit on every
  common read — looking up edges by their target thought, fetching a thought's
  embedding, listing thoughts in recency order, and filtering thoughts by type
  — turning what were full table scans into index lookups. The connection is
  also opened with `synchronous=NORMAL` (the documented-safe companion to WAL:
  durable across an application crash, only at risk of losing the most recent
  transactions on an OS crash or power loss) and `busy_timeout=5000`, so a
  second connection waits briefly for a lock instead of failing immediately
  with a "database is locked" error. The index changes are an additive schema
  migration that runs automatically on first open with zero data loss; see the
  [upgrade guide](docs/upgrade.md#03---04).

### Fixed

- **MindQL no longer mistypes quoted values or silently ignores malformed
  conditions.** A single-quoted WHERE value is now kept verbatim as a string,
  so a zero-padded identifier such as `WHERE source = '007'` matches the stored
  string `'007'` instead of being coerced to the integer `7` and matching
  nothing; an *unquoted* bare value (for example `WHERE created_cycle = 12`) is
  still coerced to a number as before. A WHERE fragment must now match the
  `field op value` grammar in full: trailing content after a condition (such as
  `WHERE priority = 'P1' OR 1=1`) previously matched only the leading prefix and
  silently discarded the rest, which could change the result set unnoticed — it
  now raises a parse error. Finally, a `FIND` query with no `LIMIT` clause is
  capped at a sane default (100 rows) rather than running an unbounded scan; an
  explicit `LIMIT` always overrides the default, and `COUNT` queries are
  unaffected.

- **Long memories are now embedded in full, and a thought's opening is no
  longer double-counted.** Two silent recall killers in the vector arm of
  search are fixed. First, when a thought's `essence` is just the opening of
  its `content` (a common convention, e.g. `essence = content[:200]`),
  auto-embed used to concatenate the two and encode that opening twice, letting
  it dominate the vector and dilute the discriminative tail; the redundant
  prefix is now dropped and `content` is embedded alone, while a genuinely
  distinct `essence` is still encoded alongside the content as before. Second,
  the local `sentence-transformers` provider now raises `max_seq_length` to the
  model's true architecture maximum after loading (derived from the model, not
  hard-coded), instead of accepting a conservative shipped default — the bundled
  `all-MiniLM-L12-v2` reported `128` while its backbone supports `512`, so the
  tail of any longer thought was silently truncated away before encoding.
  Existing stored embeddings are unaffected until a thought is re-written
  (re-create or an `essence`/`content` update), at which point it is re-embedded
  with the corrected input.

- **Natural-language queries now reach the full-text index.** `search_fts`
  previously joined the words of a bare query with FTS5's implicit `AND`, so a
  question only matched documents that contained *every* word — including
  function words like "what", "was" and "my" — and a relevantly-phrased answer
  was missed. Bare queries are now matched with `OR`: a document is returned
  when it shares any content word, and BM25's IDF weighting ranks the documents
  that share the most distinctive words first, so no stopword list or stemmer is
  needed in any language. Contractions and clitics no longer silently miss
  (`sister's` matches a stored `sister's dog`; `l'école` matches `l'école
  française`) because unsafe characters now split a token into separate terms
  instead of being deleted into an unindexed word. Pasting a URL or a timestamp
  into search no longer raises: only the real `essence:` and `content:` column
  filters are honoured, while tokens such as `http://example.com` or `12:30` are
  treated as ordinary search terms. Expert syntax is unchanged — quoted phrases,
  uppercase `AND`/`OR`/`NOT`, hyphenated identifiers and the `essence:`/
  `content:` column filters all keep their existing behaviour. As a final
  safeguard, a malformed full-text expression is now logged and degraded to no
  full-text hits instead of propagating, so the rest of a hybrid search still
  returns results.

- **Transient errors from an OpenAI-compatible embeddings endpoint no
  longer abort the whole call.** `OpenAICompatibleProvider` now retries a
  single embeddings request with bounded exponential backoff when the
  endpoint reports a transient failure — a read timeout or network blip,
  or a transient HTTP status (`408`, `409`, `425`, `429`, `500`, `502`,
  `503`, `504`). A short outage is absorbed instead of failing the
  caller's ingest. Non-transient errors (such as `400`, `401`, `403`,
  `404`) still surface immediately with no retry, and a transient failure
  that persists across every attempt is still raised, so the call never
  loops forever. The behaviour is tunable through two new keyword-only
  constructor arguments, `max_attempts` (default `3`) and
  `base_retry_delay_s` (default `1.0`); the defaults leave the success
  path at a single request, so existing callers see no change. The other
  embedding providers are unaffected.

- **sqlite-vec backend no longer crashes on configuration.** Selecting the
  `sqlite-vec` vector backend (via `engrava[vec]`) previously raised a
  thread error when building a store from configuration, because the
  extension was loaded on the calling thread instead of the connection's
  own worker thread. The extension now loads on the worker thread that owns
  the connection, so configuration succeeds; if the extension still cannot
  be loaded for any reason, the store falls back to the built-in search
  backend with a warning instead of raising. Extension loading is also
  always re-disabled after the load attempt, even when the load fails, so a
  connection is never left with extension loading enabled.

### Changed

- **CI hardening (maintenance, no runtime impact).** Continuous integration
  now runs a generic secret scan (gitleaks, built-in rules) and a dependency
  vulnerability audit (`pip-audit --strict`) on every push and pull request,
  and the release pipeline verifies that the built wheel bundles its required
  package data before publishing to PyPI. These are tooling/CI changes only —
  no change to the installed package or its behaviour.

## 0.3.0 (2026-06-02)

* ci: add on-demand smoke-gate workflow (#10) ([50e2bf2](https://github.com/sovantica/engrava/commit/50e2bf2)), closes [#10](https://github.com/sovantica/engrava/issues/10)
* ci: automated release pipeline ([18c1d68](https://github.com/sovantica/engrava/commit/18c1d68))
* ci: quote semantic_version input to fix release workflow startup ([2f3c9c3](https://github.com/sovantica/engrava/commit/2f3c9c3))
* ci: skip branch-name guard for automated dependency PRs (#8) ([b14bb0a](https://github.com/sovantica/engrava/commit/b14bb0a)), closes [#8](https://github.com/sovantica/engrava/issues/8)
* ci: skip upgrade smoke test when the baseline is not yet published (#9) ([cc05ed7](https://github.com/sovantica/engrava/commit/cc05ed7)), closes [#9](https://github.com/sovantica/engrava/issues/9)
* ci: use semantic-release CLI directly to satisfy actions allowlist ([0ebf9f1](https://github.com/sovantica/engrava/commit/0ebf9f1))
* release: merge dev into release/v0.3.0 (automated release pipeline) ([02d3f05](https://github.com/sovantica/engrava/commit/02d3f05))
* release: v0.3.0 — first public release ([8bd705e](https://github.com/sovantica/engrava/commit/8bd705e))
* feat: graph memory database — dreaming consolidation, hybrid search, audit trail ([ed82259](https://github.com/sovantica/engrava/commit/ed82259))
* test: refresh stale FTS-upgrade fixture and public-export baseline (#11) ([46cab01](https://github.com/sovantica/engrava/commit/46cab01)), closes [#11](https://github.com/sovantica/engrava/issues/11)
* chore(deps): bump actions/checkout from 4 to 6 (#3) ([840ab88](https://github.com/sovantica/engrava/commit/840ab88)), closes [#3](https://github.com/sovantica/engrava/issues/3)
* chore(deps): bump actions/download-artifact from 4 to 8 (#7) ([c7ca7cf](https://github.com/sovantica/engrava/commit/c7ca7cf)), closes [#7](https://github.com/sovantica/engrava/issues/7)
* chore(deps): bump actions/setup-node from 4 to 6 (#6) ([99acddc](https://github.com/sovantica/engrava/commit/99acddc)), closes [#6](https://github.com/sovantica/engrava/issues/6)
* chore(deps): bump actions/upload-artifact from 4 to 7 (#5) ([7ad3f2f](https://github.com/sovantica/engrava/commit/7ad3f2f)), closes [#5](https://github.com/sovantica/engrava/issues/5)
* chore(deps): bump softprops/action-gh-release from 2 to 3 (#4) ([9eb3729](https://github.com/sovantica/engrava/commit/9eb3729)), closes [softprops/action-#release](https://github.com/softprops/action-/issues/release) [#4](https://github.com/sovantica/engrava/issues/4)
* Bump idna from 3.13 to 3.15 (#1) ([f402fd2](https://github.com/sovantica/engrava/commit/f402fd2)), closes [#1](https://github.com/sovantica/engrava/issues/1)
* docs: align metadata and docs with product tagline; add dependabot and issue config (#2) ([b3bb62d](https://github.com/sovantica/engrava/commit/b3bb62d)), closes [#2](https://github.com/sovantica/engrava/issues/2)

# Changelog

All notable changes to engrava will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0]

### Documentation

- **docs:** align package description and documentation with the product
  tagline ("The memory database for AI agents") and the public contact
  address.

### Maintenance

- **Pre-publish smoke gate.** The publish workflow now runs a hard-fail
  quality gate before any wheel or sdist is built: the bundled
  synthetic benchmark is driven in its binding acceptance-criterion
  mode and the committed floors (synthesis coverage, direct-query
  neutrality, sanity-scenario neutrality, sanity with reflection-boost)
  are enforced from source. A breach blocks publish. A standalone
  ``scripts/check_smoke_gate.py`` exposes the same gate for local runs
  and is wired into ``make smoke-gate``. An optional LongMemEval
  recall@5 probe enforces a calibrated absolute floor when invoked via
  ``--include-longmemeval``; the probe stays off in CI by default
  because the LongMemEval dataset is user-download and the run takes
  several minutes, but maintainers invoke it manually before tagging a
  release.

### Added

- **Bundled walkthrough example + self-anchored metadata helpers.** A
  single-file script a Free-tier user can run directly from the repo
  to see the engine work in under a minute:

    * `examples/quickstart.py` — boots an in-memory store with a local
      embedding encoder, ingests a handful of percepts and utterances
      built via the new `engrava.metadata` helpers, runs one dreaming
      consolidation cycle, and queries via hybrid search.

  The script pre-flights `sentence_transformers` with a clean
  actionable message and `sys.exit(2)` when the `[embeddings-local]`
  extra is missing.

  Adds three pure-function helpers in `engrava.metadata`, re-exported
  from the top-level package: `percept(...)` for input arriving at
  the agent, `utterance(...)` for the agent's own outgoing content,
  and `thought(...)` for the agent's internal cognition. They build
  the structured `metadata` dictionary the persistence layer
  recognises, anchoring every stored thought to a perspective and a
  source. Same arguments always return an equal dictionary.

  `docs/quickstart.md` gains a "Run the bundled walkthrough" section
  with the dreaming and self-anchored-identity narrative; new
  `examples/README.md` indexes the scripts.

- **LongMemEval public benchmark harness** — engrava-side runner for
  the published LongMemEval memory-evaluation dataset (Wu et al., ICLR
  2025, arXiv:2410.10813).

    * `python -m engrava.benchmarks.longmemeval` ingests each
      question's haystack into a fresh engrava store, optionally runs
      one dreaming consolidation cycle, queries via `search_hybrid`,
      and scores the retrieved chunks.
    * Three evaluation modes: deterministic substring containment
      (default), deterministic cosine-similarity over a configurable
      threshold, and an opt-in LLM judge wired via a thin
      `LLMJudgeClient` protocol so callers can plug in any provider.
    * The dataset is **not bundled** — the loader downloads the
      requested variant from the upstream HuggingFace distribution on
      first use and caches it under `~/.engrava/benchmarks/longmemeval/`.
    * Self-anchored thought metadata follows the same shape as the
      synthetic benchmark (`perspective`, `source.is_self`,
      `session_id`, `turn_index`), so dreaming filters see the same
      inputs they do in production.
    * See `src/engrava/benchmarks/longmemeval/README.md` for
      attribution, license, and download instructions.

- **Synthetic benchmark suite** — reproducible dreaming evidence
  runnable on any laptop without API keys or network access.

    * `python -m engrava.benchmarks.synthetic` runs binding acceptance
      measurements (~5 minutes): synthesis coverage, direct retrieval
      neutrality, sanity tolerance.
    * `python -m engrava.benchmarks.synthetic --with-reproducibility`
      adds full per-scenario texture from the bundled
      `synthetic-v1.json` dataset (~10 minutes total).
    * Measures three properties: synthesis coverage (dreaming produces
      REFLECTIONs that consolidate related facts), direct retrieval
      neutrality (dreaming does not degrade baseline competence), and
      sanity tolerance (small, non-pathological influence on neutral
      queries).
    * See `docs/benchmarks.md` for the interpretation guide and the
      v0.4.0 roadmap (tighter neutrality ceilings + recall-lift
      evidence).

- **Cluster quality gates for the dreaming consolidation loop.** Seven
  deterministic gates run on every candidate cluster before a
  REFLECTION is materialised, dropping clusters that would produce
  low-signal or actively misleading memories. Most gates are
  language-agnostic; the contradiction gate (Gate 3) ships with an
  English-only sentiment-token lexicon and silently passes clusters
  in other languages. Gates fire in two phases:

  *Pre-build (cheap rejections before content assembly):*

    * **Gate 1 — duplicate content members** rejects clusters that
      contain byte-identical member content within the same cluster
      (a dedup escape that would inflate one statement into a
      pseudo-cluster).
    * **Gate 2 — persona-only** rejects clusters where the share of
      members carrying a persona/identity marker (and no conversation
      marker) exceeds the configured threshold.
    * **Gate 3 — contradictory** rejects clusters with member pairs
      asserting documented opposite predicates (English lexicon).
    * **Gate 4 — low cohesion** rejects clusters whose mean pairwise
      cosine similarity (over L2-normalised embeddings) falls below
      the configured threshold.
    * **Gate 5 — external-source homogeneity** rejects clusters whose
      share of members with `metadata["source"]["is_self"] != True`
      falls below the configured minimum-external fraction; missing
      or malformed `source` is treated as external under safe
      fallback. Belt-and-suspenders over the upstream eligibility
      filter.
    * **Gate 6 — named-entity consistency** rejects clusters where the
      fraction of members whose per-member named-entity set
      intersects the first member's set falls below the configured
      threshold; single-member and empty-NE-on-anchor clusters pass
      vacuously.

  *Post-build (after `build_reflection_content_v2`):*

    * **Gate 8 — meaningful keyphrases** rejects REFLECTIONs whose final
      `top_keyphrases` list is empty or composed solely of generic
      filler bigrams.

  All gates are pure functions in
  `engrava.extensions.dreaming_cluster_quality`. They are wired into
  the consolidation loop with per-gate rejection counters surfaced via
  a single `INFO` summary log per pass.

  Six new `DreamingGates` fields control the behaviour, all validated
  in `__post_init__` so direct construction cannot bypass the
  `[0.0, 1.0]` contract:

    * `cluster_quality_gating_enabled: bool = True` — master switch.
    * `cluster_quality_persona_threshold: float = 0.75` — share of
      persona-marked members above which a cluster is persona-only.
    * `cluster_quality_cohesion_threshold: float = 0.40` — minimum mean
      pairwise cosine for the cluster to be considered coherent.
    * `cluster_quality_external_homogeneity_threshold: float = 0.95` —
      minimum fraction of external-source members for the gate to
      pass (under safe fallback, missing/malformed `metadata.source`
      counts as external).
    * `cluster_quality_ne_consistency_threshold: float = 0.60` —
      minimum fraction of members whose named-entity set intersects
      the anchor (first) member's set.
    * `cluster_quality_require_meaningful_keyphrases: bool = True` —
      reject post-build REFLECTIONs without informative keyphrases.

  Gating is on by default; existing fixtures that intentionally pass
  synthetic, low-signal clusters opt out via
  `cluster_quality_gating_enabled=False`. Legacy thought metadata
  without `source` keys is treated as external under safe fallback,
  preserving backward compatibility.

- **Statistical cross-cluster boilerplate filter for REFLECTION
  keyphrases.**  Two language-agnostic helpers ship in
  `engrava.extensions.dreaming_keyphrases` —
  `compute_cluster_phrase_frequency` (cluster-count map over the
  per-cluster top-N keyphrase lists) and `is_boilerplate_phrase`
  (case-insensitive threshold check with a small-corpus bypass).
  `build_reflection_content_v2` accepts two new optional kwargs,
  `cluster_phrase_df` and `total_clusters`, and uses them to drop
  phrases that exceed the configured share of clusters; both kwargs
  default to `None`, so existing callers see no behaviour change.
  The dreaming consolidation loop now runs a lightweight pre-pass
  that builds the document-frequency map ahead of REFLECTION
  creation and forwards the kwargs into the content builder.

  Three new `DreamingConfig` knobs control the filter:

    * `boilerplate_threshold: float = 0.30` — phrases appearing in
      more than 30 % of clusters are dropped.  Set to `1.0` to
      disable the filter.
    * `boilerplate_min_corpus_size: int = 5` — minimum cluster count
      before the filter engages; smaller runs preserve every
      keyphrase regardless of frequency.
    * `boilerplate_min_keyphrases_per_refl: int = 1` — fallback
      guard.  If filtering would shrink a REFLECTION's keyphrase
      list below this size the raw list is kept, so REFLECTIONs
      never end up with an empty `top_keyphrases` field.

  Because the filter is statistical and operates on lowercased
  phrases, it learns "boilerplate" from the live deployment without
  any hardcoded blocklist and works equally well across English,
  Polish, Japanese, French, German and every other language that
  the TF-IDF tokeniser already supports.

- **Metadata-aware dreaming filter** on `DreamingConfig`.  Five new
  fields gate which thoughts the dreaming pipeline considers eligible
  for promotion and REFLECTION clustering:

    * `eligible_perspectives: frozenset[Literal["percept", "utterance",
      "thought"]] | None` — positive filter on
      `metadata["perspective"]`.
    * `self_filter_mode: Literal["any", "self_only", "external_only"]`
      — filter on `metadata["source"]["is_self"]`.
    * `min_source_confidence: Literal["high", "medium", "low"]` —
      minimum required `metadata["source"]["confidence"]`, ranked
      `low < medium < high`.
    * `excluded_content_types: frozenset[str]` — negative filter on
      `metadata["content_type"]`; defaults to `frozenset({"code"})`
      because code fragments cluster poorly under cosine similarity.
    * `eligible_content_types: frozenset[str] | None` — optional
      positive filter on `metadata["content_type"]`.

  The filter is applied during promotion (filtered candidates skip the
  P1-promotion decision) and during REFLECTION creation (only eligible
  cluster members feed the cluster hash, the centroid embedding, the
  structured content payload and the `CONSOLIDATED_FROM` lineage edges
  — a cluster whose eligible subset falls below
  `DreamingGates.min_cluster_size` is dropped entirely).  Defaults
  preserve the pre-existing dreaming behaviour: thoughts without any
  structured metadata pass unconditionally, and a freshly-constructed
  `DreamingConfig()` leaves every axis disabled.  Promotion-side
  rejections are reported at `DEBUG` level (`dreaming filter: N/M
  candidates rejected by metadata filter (cycle X)`).

- **`ThoughtRecord.metadata` accepts nested dict values** for structured
  namespaces.  The `MetadataValue` type alias now resolves recursively to
  `str | int | float | bool | None | dict[str, MetadataValue]`, so callers
  can express grouped attributes such as
  `metadata["source"] = {"is_self": True, "confidence": "high", ...}`
  directly instead of flattening to dot-prefixed keys.  Leaf values are
  still restricted to JSON scalars; lists, tuples, sets and custom
  containers remain rejected at every depth.  The persistence-layer
  validator walks nested dicts recursively and produces dotted key-path
  error messages (e.g. `metadata value at source.tags type list not
  allowed`).  Backward-compatible: flat-only callers see no behaviour
  change; the SQLite `metadata_json TEXT` column already stored the
  serialised JSON faithfully, so no schema migration is required.

### Fixed

- **REFLECTION freshness (lifecycle-aware consolidation).** A REFLECTION is
  a synthesis of a live cluster of thoughts, but until now it was frozen at
  creation and never re-bound to the current state of that cluster. Three
  related freshness gaps are closed so dreaming improves recall over a
  long-running agent's lifetime instead of slowly polluting it:

    * **Orphan retire.** The consolidation pass now sweeps existing
      REFLECTIONs and retires (ACTIVE → ARCHIVED) any whose every consolidated
      source thought has left the active set, so an ordinary `gc` reclaims it
      (cascading its centroid embedding and consolidation edges). The sweep
      fires only when *all* sources are gone and the REFLECTION has at least
      one source — a partially-archived cluster keeps its synthesis, and a
      source-less REFLECTION is never retired.

    * **Centroid re-bind on evolve.** When a source thought's essence or
      content changes and the thought is re-embedded, every REFLECTION that
      was consolidated from it has its centroid recomputed from the current
      member vectors (the same deterministic L2-normalized mean used at
      creation, overwritten in place — no schema change). Metadata- or
      priority-only edits do not re-embed the source and therefore leave
      dependent REFLECTION centroids untouched.

    * **Recall freshness floor.** Similarity, hybrid, and reflection-only
      search no longer surface a retired REFLECTION in the window between
      archival and physical collection, so a stale synthesis can no longer
      out-rank fresh relevant thoughts. The floor is a lifecycle check at the
      data layer; existing reflection ranking knobs are unchanged.

  Every mechanism is deterministic (vector mean / cosine / SQL); no model is
  invoked. Pre-publish recall-quality fix.

- **Referential integrity (cascade delete + edge FK).** The schema now
  declares foreign keys with `ON DELETE CASCADE` for the three child
  tables that reference `thought` (edges on both endpoints, embeddings,
  and action records). Deleting a thought now actually removes the
  related rows instead of leaving them behind as orphans, and inserting
  an edge to a non-existent thought is rejected with a typed domain
  exception (`ReferentialIntegrityError`) instead of the raw SQLite
  integrity error. The change ships as schema version 12; existing
  databases migrate in place via a recreate-table step that purges
  any pre-existing orphan rows before enabling the constraints. The
  migration is idempotent and recovers cleanly when a previous pass
  was interrupted between tables. The connection-level pragma that
  enables FK enforcement is now issued automatically from
  `ensure_schema`, so callers that construct the store directly (not
  via `from_config`) still get enforcement.

- **PyPI wheel and sdist now include `schema_core.sql`.**
  `SqliteEngravaCore.ensure_schema` loads the core SQL schema via
  `importlib.resources`, but `pyproject.toml` had no
  `[tool.setuptools.package-data]` entry and there was no
  `MANIFEST.in`, so neither the wheel nor the sdist actually bundled
  the file. A fresh `pip install engrava` therefore crashed on the
  first `EngravaCore` initialisation with `FileNotFoundError`. The
  fix adds an explicit narrow package-data pattern
  (`engrava = ["infrastructure/sqlite/*.sql"]`) plus a minimal
  `MANIFEST.in` for the sdist side, so installed distributions now
  contain the schema file and `EngravaCore` initialises cleanly out
  of the box. Critical pre-publish fix.

### Maintenance

- **Build artifacts removed from tracked state.** The generated
  `coverage.json` is no longer checked into the repository — it is
  regenerated on every CI run. `.gitignore` gains `.coverage.*`,
  `coverage.json`, `coverage.xml`, and a defensive bare-`.env` /
  `.env.*` pair (with a `!.env.example` exception for template
  files), so future coverage and environment files cannot
  accidentally re-enter the public surface. No behavioural change —
  repository hygiene only.

- **Packaging guard script + broader package-data coverage.** Adds
  `scripts/verify_wheel_data.py`, a standalone build-and-inspect step
  that rebuilds wheel + sdist and asserts the critical data files
  (`schema_core.sql`, `synthetic-v1.json`) are bundled in both
  artifacts; non-zero exit on missing files. `MANIFEST.in` adds
  `recursive-include src/engrava *.json` so sdist coverage tracks the
  wheel's `[tool.setuptools.package-data]` entry. Closes the
  fresh-install regression class previously fixed only for
  `schema_core.sql` (see Fixed); cumulative outcome with prior
  contributor-instructions and benchmark refreshes.

- **Contributor instructions refreshed.** Per-repo agent guidance files
  (`CLAUDE.md`, `AGENTS.md`, `.cursorrules`, `.github/copilot-instructions.md`,
  `BRANCHING.md`) regenerated from canonical templates. Generic OSS-friendly
  guidance on Conventional Commits, branch naming, code quality bar, and
  contribution flow. Internal workflow references removed from the public
  surface.

- **Repository metadata aligned with organization branding.**  The
  `LICENSE` copyright holder, the `pyproject.toml` `[project] authors`
  field, and the `[project.urls]` GitHub references now reflect the
  Sovantica organization identity instead of an individual maintainer.
  A `hello@sovantica.ai` contact address replaces the previous
  unauthored entry, and the `Documentation` URL points at
  `https://docs.engrava.ai` so PyPI listings link to the product
  documentation rather than the in-repo `docs/` tree.  No behavioural
  changes — packaging metadata only.

### Behavior Changes

- **`ThoughtRecord.metadata` field for caller-supplied structured attributes.**
  Each thought now carries an extensible `dict[str, MetadataValue]`
  (`MetadataValue = str | int | float | bool | None`) for flat scalar
  attributes such as conversation role, source language, content type,
  external session identifier, turn index or speaker name.  The field
  defaults to an empty dict, so existing callers compile and run
  unchanged — no code change is required to upgrade.

  Persistence: a new `metadata_json` column (`TEXT NOT NULL DEFAULT '{}'`)
  is added to the `thought` table; the core schema bumps from
  `user_version = 10` to `11` and `_migrate_core_v10_to_v11` performs
  the additive migration with duplicate-column tolerance for safe
  re-runs.  Pre-existing rows receive the empty-dict default
  automatically.  JSON serialization uses `ensure_ascii=False` so
  non-ASCII attribute values round-trip byte-exact.

  Validation: caller-supplied metadata is checked at both API entries
  (`create_thought` and `update_thought`).  Nested or list values raise
  `ValueError` with a per-key message.  Serialized payloads above
  ~4 KiB emit a `WARNING` log; payloads above ~64 KiB are rejected
  outright with `ValueError`.

  No new public API surface beyond the field itself — downstream
  filtering / dispatching consumers are tracked separately.

- **`DreamingConfig.max_p1_fraction` defaults to 5 % (was effectively unlimited).**
  Dreaming consolidation now caps the fraction of corpus thoughts at priority P1
  to `max_p1_fraction` (default `0.05`).  Once the cap is reached, further
  promotions are silently skipped for that run and `ConsolidationResult.promotion_capped`
  is set to `True`.  Motivation: empirical analysis found that
  unrestricted promotion accumulated 29.9 % P1 thoughts, giving those entries a
  systematic 67 % ranking boost in hybrid-search fusion.

  **Upgrade impact:** existing databases with >5 % P1 will stop receiving new
  P1 promotions until the fraction drops below the cap (via normal thought
  expiry / lifecycle transitions).  Run `python -m scripts.rebalance_p1
  --db-path <PATH>` to immediately demote excess P1 thoughts to P2.
  Set `extensions.dreaming.max_p1_fraction: 1.0` in YAML to restore legacy
  unlimited behaviour.

- **`DreamingConfig.promote_targets` defaults to `"OBS_ONLY"`.**  Only
  `OBSERVATION` thoughts are now eligible for P1 promotion by default.
  `REFLECTION` thoughts can be included by setting
  `extensions.dreaming.promote_targets: ALL` or `REFL_ONLY`.

- **`DreamingConfig.reflection_default_priority` defaults to `"P2"`.**
  Newly-created `REFLECTION` thoughts now start at P2 instead of inheriting
  the highest priority of their cluster members (which was effectively P1 in
  most stores).  Configure via `extensions.dreaming.reflection_default_priority`.

- **`ConsolidationResult` gains two new fields** (zero-impact on existing
  consumers — both default to `False` / `0.0`):
  - `promotion_capped: bool` — True when the P1 fraction cap stopped promotion.
  - `p1_fraction_after: float` — fraction of total corpus at P1 after the run.

- **`SqliteEngravaCore.count_thoughts()` gains a `priority` keyword filter.**
  Allows callers to count thoughts at a specific priority level without
  fetching full records (e.g. `await store.count_thoughts(priority="P1")`).

- **New utility `python -m scripts.rebalance_p1`** — demotes excess P1 thoughts
  to P2 to meet the `max_p1_fraction` cap on existing databases.  Idempotent,
  supports `--dry-run`, and `--max-p1-fraction` to override the target fraction.

- **Extended `SENTENCE_STARTER_BLOCKLIST` with 11 empirical entries** from a
  short07 NE top-15 audit (2026-05-04).  11 of the 15 most frequent
  `named_entities` tokens in 47 REFLECTION thoughts were sentence-starter words
  absent from the prior blocklist: `Also`, `Did`, `Embracing`, `For`,
  `Have`, `How`, `Instead`, `Lastly`, `Not`, `Reflecting`, `Ultimately`.  An
  additional 15 common gerund/participle starters (`Conducting`, `Connecting`,
  `Continuing`, …) are added as preventive coverage.  Additive-only — no
  existing entry removed; real proper nouns (`Alex`, `Cornell`, `1974`, …)
  remain unblocked.

- **Structural REFLECTION content schema v2.** The dreaming extension
  now emits a richer structural JSON when it creates a REFLECTION
  thought from a cluster.  The legacy three-field layout
  (`member_ids`, `keywords`, `cluster_hash`) is preserved verbatim
  for backward compatibility, and the dict gains nine additional
  fields:

  - `type` / `version` — schema-dispatch markers
    (`"reflection"` / `2`).  Legacy v1 emissions never carried these
    keys; consumers can detect a legacy row by their absence.
  - `member_count` / `cluster_algorithm` / `created_at` — fields
    mandated by the cognitive-boundary REFLECTION spec but missing
    from the previous emitter.
  - `top_keyphrases` — TF-IDF-scored 2-3 word n-grams over the
    cluster, with the corpus baseline supplied by the caller.
  - `member_excerpts` — top-N members by priority + recency, each
    truncated at the word boundary to ~80 characters.
  - `temporal_span` — `min_created_at` / `max_created_at` plus the
    span in days across the cluster.
  - `named_entities` — regex-based capitalised tokens plus year /
    measurement matches.

  All enrichment is deterministic and LLM-free; the cognitive-
  boundary CI guard test covers the new sibling modules
  (`engrava.extensions.dreaming_keyphrases`,
  `engrava.extensions.dreaming_reflection_content`) so an LLM SDK
  cannot sneak in via either of them.

  Two new `DreamingConfig` fields tune the structural output:
  `top_keyphrases_count` (default 3) and `top_member_excerpts_count`
  (default 5).  Both default to safe values, are additive on the
  dataclass, and can be overridden via the standard
  `extensions.dreaming` YAML section.

  Existing REFLECTION rows in production databases continue to read
  correctly — the dispatch parser detects the legacy schema by the
  absence of the `version` field.  A new opt-in utility
  `python -m scripts.reenrich_reflections_to_v2 --db-path PATH`
  walks the database in batches and rewrites legacy v1 content to
  v2 in place; idempotent (re-running on an already-migrated DB is
  a no-op) with a `--dry-run` mode for previewing.

  Empirical motivation: AMB PersonaMem MCQ judges score parsable
  prose-like surface higher than terse JSON; the structural
  enrichment closes most of that gap without crossing the no-LLM
  cognitive boundary.

- **Structural REFLECTION content quality amendment.** The v2
  builder gains three deterministic quality fixes layered on top of
  the existing schema (no key changes — only field VALUES improve):

  - **Sentence-starter blocklist for `named_entities`.** Common
    capitalised non-entity words (`Absolutely`, `However`,
    `Therefore`, `Furthermore`, `User`, `Assistant`, `System`, …)
    no longer pollute the entity list.  The blocklist is exposed
    publicly as `SENTENCE_STARTER_BLOCKLIST` from
    `engrava.extensions.dreaming_keyphrases` for downstream
    consumers that want the same filter.  Real proper nouns
    (`Cornell`, `Berlin`, `Anthropic`, …) are unaffected.
  - **Role-marker stripping before keyphrase extraction.**
    `[USER] User: …` / `[ASSISTANT] Assistant: …` / `[SYSTEM] …`
    prefixes are stripped before tokenisation, so artefact tokens
    like `"user user"` or `"assistant"` no longer surface as
    top-ranked keyphrases or simple keywords.
  - **Member excerpt size raised 80 → 150 characters** and made
    configurable via the new `DreamingConfig.member_excerpt_max_chars`
    field (positive integer; YAML key
    `extensions.dreaming.member_excerpt_max_chars`).  The bump
    gives downstream LLM judges a meaningfully longer window into
    each member while keeping the cluster content well within the
    2 KB structural budget.

  All three changes are deterministic, LLM-free, and additive — the
  v2 dispatch parser, the 12 v2 schema keys, and existing v1
  consumers are unchanged.

- **Opt-in content-hash deduplication on `create_thought`.** The
  persistence layer now exposes a new keyword-only argument
  `deduplicate: bool = False` on
  `SqliteEngravaCore.create_thought`.  When `True`, identical
  `content` (SHA-256 hash collision over the UTF-8 bytes, no
  normalization) collapses into a single thought whose
  `confirmation_count` is incremented and `updated_at` refreshed,
  instead of producing a duplicate row.  Default behavior is
  unchanged (`deduplicate=False` preserves the legacy create-on-every-
  call semantic).

  Configuration: a new top-level `IngestConfig` value object exposes
  `deduplication_enabled: bool = True`, accessible via
  `EngravaConfig.ingest`.  YAML callers can flip it via
  ```yaml
  ingest:
    deduplication_enabled: false
  ```
  Ingest-pipeline callers (e.g. benchmark adapters, bulk-import
  tooling) should read `config.ingest.deduplication_enabled` and pass
  it through to `create_thought(..., deduplicate=...)`.

  Schema: the `thought` table gains a nullable `content_hash TEXT`
  column and a supporting `idx_thought_content_hash` index.
  `PRAGMA user_version` is bumped from 9 to 10.  A new
  `_migrate_core_v9_to_v10` helper participates in the existing
  `ensure_schema` upgrade cascade, so DBs at any prior supported
  version (v3 through v9) upgrade in a single call.  The migration
  is idempotent (ALTER TABLE tolerates duplicate-column errors,
  CREATE INDEX uses `IF NOT EXISTS`).

  Backfill: pre-v10 thoughts retain `content_hash IS NULL` until the
  bundled `scripts/backfill_content_hashes.py` utility populates them
  in batches; running benchmarks against a freshly-bootstrapped DB
  (e.g. AMB PersonaMem fixtures) is unaffected because the column is
  filled on every insert.

  Empirical motivation: AMB PersonaMem benchmark runs (with full
  ingest tracing, two independent runs in DREAM and NODREAM modes)
  reproduced ~38.5% duplicate observation thoughts, with persona
  intros multiplied 12-13x per session; clustering and
  reflection-of-duplicates pollution amplified the waste downstream.

### Breaking Changes

- **Removed observability hook surface from the public package.** The
  `EngravaObservabilityHooksProtocol`, `ObservabilityDispatcher`,
  `DefaultObservabilityHooks`, `ObservabilityGates`, all 12 event
  dataclasses (`QueryStartEvent`, `QueryEndEvent`, `CandidatesFusedEvent`,
  `CandidateRecord`, `IngestStartEvent`, `IngestEndEvent`,
  `ThoughtCreatedEvent`, `EmbeddingComputedEvent`, `EdgeCreatedEvent`,
  `CycleStartEvent`, `CycleEndEvent`, `ClusterDecisionEvent`,
  `ReflectionCreatedEvent`), the `register_observability_hook()` method
  on the store, and the `obs_dispatcher` / `obs_gates` constructor
  parameters are gone.  The `observability:` section in the YAML config
  is no longer parsed.

- **`benchmarks/` directory removed** — benchmark runners and their
  adapters / checkpointing / LLM-judge helpers are no longer included
  in the open-source distribution. A public reproducibility benchmark
  remains available via `python -m engrava.benchmarks.synthetic`
  (see `docs/benchmarks.md`).

### Database Changes

- Upgrade-path validation is now part of release preparation for minor bumps.
- Automatic schema migration still runs on first connection via `ensure_schema()`.
- Releases that change schema behavior must document the change here explicitly.

### Changed

- **Search: `default_graph_weight` reverted to `0.0`** — graph-neighbour signal
  disabled by default. Empirical evidence from two independent AMB PersonaMem
  benchmark runs showed `graph_weight=0.3` caused a −8 pp accuracy regression
  (Chi² p=0.045). The score-adjustment mode cascade promotes short REFLECTION
  summaries over detail-rich source OBSERVATIONs. Graph signal remains available
  as an explicit opt-in via `search.default_graph_weight` in YAML config.
  Until candidate-expansion via `CONSOLIDATED_FROM` ships, the graph backend
  stays disabled.

### Added

- **`store.metrics()` snapshot API** — added `EngravaMetrics`
  with `thoughts`, `edges`, `storage`, and rolling-window `search_latency`
  percentiles (`p50` / `p95` / `p99`). `engrava info` now renders this
  contract instead of ad-hoc SQL counts, and the new `metrics:` YAML section
  configures the latency window size and opt-out behaviour.

- **Dreaming — clustering + REFLECTION thoughts** — `run_consolidation()`
  now builds thought clusters from the ASSOCIATED edge graph (Label Propagation
  Algorithm, with agglomerative cosine-similarity fallback) and creates
  `ThoughtType.REFLECTION` thoughts that summarise each qualifying cluster.
  A centroid embedding (mean of member vectors, L2-normalised) is stored for
  each REFLECTION.  `CONSOLIDATED_FROM` edges link the REFLECTION back to each
  cluster member.  Idempotent: re-runs skip clusters whose content-hash already
  exists.  Opt-out via `DreamingGates.enable_reflections = False`.
  `ConsolidationResult` gains a new `reflections_created` counter.
- **`DreamingGates` cluster fields** — `min_cluster_size` (default `3`),
  `cluster_similarity_threshold` (default `0.7`), `cluster_algorithm`
  (`"lpa"` | `"agglomerative"`, default `"lpa"`), and `enable_reflections`
  (default `True`).
- **`search_hybrid()` — `include_reflections` + `reflection_boost`** — callers
  can now pass `include_reflections=False` to exclude REFLECTION thoughts from
  hybrid search results, or a custom `reflection_boost` multiplier to re-rank
  them.  The default boost (`SearchConfig.reflection_boost = 1.2`) gives
  REFLECTION thoughts a mild up-ranking.
- **`search_reflections_only()`** — convenience method on `SqliteEngravaCore`
  that returns only `ThoughtType.REFLECTION` thoughts ranked by hybrid score.
- **`list_edges()`** — new `SqliteEngravaCore` method for querying edges by
  optional `edge_type` / `source` filters with configurable `limit`.
- **`SearchConfig.reflection_boost`** — new field (default `1.2`) parsed from
  YAML `search.reflection_boost`.

- **Dream-created edges** — `run_consolidation()` now creates
  `ASSOCIATED` edges between promoted thoughts and their nearest neighbours.
  Controlled via `DreamingConfig.edges` (`EdgeCreationConfig`). Edges use
  `source=KnowledgeSource.DREAMING` for attribution. Idempotent: re-runs do
  not create duplicate edges.
- **Graph-aware hybrid search** — `search_hybrid()` supports an
  optional 5th scoring signal (`graph_weight`) using 1-hop-weighted neighbour
  boost. Disabled by default (`graph_weight=0.0`, opt-in). Controlled via
  `SearchConfig.default_graph_weight`, `graph_edge_decay`, and
  `max_neighbors_per_candidate`.
- **`KnowledgeSource.DREAMING`** — new enum value for dream-originated edges.
- **`EdgeCreationConfig`** — frozen dataclass for edge-creation parameters
  (`enabled`, `top_k`, `min_similarity`, `edge_weight_factor`).
- **Priority signal in hybrid search** — `search_hybrid()` now supports an optional
  4th scoring signal based on thought `priority` (P1–P4). Controlled via
  `SearchConfig.default_priority_weight` (default `0.05`) and per-priority boost
  multipliers (`priority_boost_p1` through `priority_boost_p4`). Higher-priority
  thoughts receive a proportionally higher score contribution.
- **`DreamingGates.allow_zero_confirmation`** — new boolean field (default `True`)
  that bypasses the `min_confirmations` gate, allowing freshly ingested thoughts
  with zero confirmations to be eligible for dreaming promotion.
- `examples/config.yaml` — out-of-the-box configuration with dreaming enabled and
  sensible defaults.
- **Upgrade documentation and release template** — added `docs/upgrade.md`,
  `.github/release-notes-template.md`, and a README upgrade entry so release
  communication now has a stable place for compatibility guidance.

### Changed

- **`DreamingGates.min_age_cycles`** default changed from `10` to `1`. Freshly
  ingested thoughts become eligible for consolidation after a single cycle instead
  of ten.
- **`SearchConfig.default_vector_weight`** changed from `0.6` to `0.55` to
  accommodate the new priority signal while keeping weights summing to `1.0`.

### Fixed

- **Dreaming consolidation now actually runs on fresh batches.** Previously, the
  `min_confirmations=2` gate combined with `min_age_cycles=10` prevented any
  promotion in single-write batch-ingest scenarios (confirmation count = 0, age = 0).
  With `allow_zero_confirmation=True` (new default) and `min_age_cycles=1`, a
  single `run_consolidation()` call on a fresh batch can now promote qualifying
  thoughts.

## [0.2.0] — 2026-04-12

### Breaking Changes

- None.

### Database Changes

- Schema version bumped to core-5.
- Added `access_count`, `last_accessed_at`, `confirmation_count`,
  `consolidated_from`, and `visibility` to the thought table.
- Existing databases upgrade automatically through `ensure_schema()`.

### Added

- **Full-text search (FTS5)** — `search_fts()` method with BM25 ranking on `essence`
  and `content` fields. Hybrid search combines vector similarity, text relevance, and
  recency scoring via configurable `SearchConfig` weights.
- **Extension system** — `EngravaHooksProtocol` with 5 hook points (`on_store`,
  `on_retrieve`, `score_function`, `decay_function`, `mindql_extension_registry`).
  `DefaultEngravaHooks` provides no-op defaults.
- **Dreaming / memory consolidation** — `DreamingExtension` with 5 pluggable signal
  types (`ConfidenceSignal`, `ConfirmationSignal`, `FrequencySignal`, `RecencySignal`,
  `StalenessSignal`) and configurable gate thresholds.
- **sqlite-vec backend** — optional `SqliteVecSearchBackend` for hardware-accelerated
  vector search via the `vec` extra.
- **Multi-service isolation** — `EngravaManager` for running multiple independent
  databases, each with its own schema, embeddings, and FTS index.
- **YAML configuration** — `load_config()` factory, `EngravaConfig` with `SearchConfig`,
  `DreamingConfig`, `EmbeddingConfig`, and `ServicesConfig` sections.
- **5 embedding providers** — `SentenceTransformerProvider`, `OpenAICompatibleProvider`,
  `OllamaProvider`, `HuggingFaceProvider`, `CallbackProvider`.
- **MindQL enhancements** — `COUNT`, `SELECT` with `WHERE` clauses, extensible command
  registry via hooks.
- **CLI enhancements** — `export`, `import`, `gc`, `migrate` subcommands. Multi-service
  `--service` flag for `snapshot`/`restore`/`export`.
- **Read-only store** — `ReadOnlyEngrava` wrapper that raises `ReadOnlyViolationError`
  on write attempts.
- **`ExtensionManifest`** value object for extension discovery and registration.
- **`HybridSearchResult`** model combining vector, FTS, and recency scores.
- **`EmbeddingModelMismatchError`** exception for restore-time model validation.
- Open source release: standalone repository, MIT license, GitHub Actions CI/CD,
  PyPI publishing.

### Changed

- Bumped version from 0.1.0 to 0.2.0.
- `pyproject.toml` URLs now point to the standalone GitHub repository.
- Description updated to "Thought-graph database for AI agents".

### Fixed

- `--re-embed` flag now raises an error when no embedding provider is configured
  instead of silently succeeding.
- `snapshot --service` validates service existence before attempting export.
- `EngravaManager.get_store()` uses `asyncio.Lock` to prevent race conditions
  during concurrent lazy initialization.

## [0.1.0] — 2026-04-01

### Breaking Changes

- Initial release.

### Database Changes

- Initial SQLite schema introduced.
- No downgrade guarantees; follow forward-only upgrade policy from later releases.

### Added

- Initial release (internal).
- `SqliteEngravaCore` — async thought/edge/embedding/action CRUD.
- `ThoughtRecord`, `EdgeRecord`, `EmbeddingRecord`, `ActionRecord` frozen Pydantic models.
- 9 domain enums (`ThoughtType`, `Priority`, `LifecycleStatus`, `EdgeType`, etc.).
- Brute-force cosine similarity embedding search.
- `MindQLParser` and `MindQLExecutor` — `FIND` and basic query support.
- CLI with `info`, `query`, `snapshot`, `restore` subcommands.
- Schema migration support via `ensure_schema()`.
