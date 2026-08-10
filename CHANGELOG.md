# Changelog

All notable changes to engrava will be documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

## [0.6.0](https://github.com/sovantica/engrava/compare/v0.5.0...v0.6.0) (2026-08-10)

### Added

* add derived-records extension seam ([c9d55cb](https://github.com/sovantica/engrava/commit/c9d55cb1cb6dbcbd980078994b59819f07daacca))
* **core:** add derive_existing() backfill for the derived-records seam ([1c79611](https://github.com/sovantica/engrava/commit/1c79611490a179b9b141ca685a48e76c22a4c3b9))
* **core:** add opt-in cycle-provider seam and max_cycle accessor ([29d46e5](https://github.com/sovantica/engrava/commit/29d46e58afcbd371229654aeba9605ff3f6f88ea))
* **edges:** add generic metadata carrier with schema v19 migration ([08a4cb3](https://github.com/sovantica/engrava/commit/08a4cb3c9923f304bce3011050c95846598308a1))
* **extensions:** add zero-dependency split modes to StructuralSplitProducer ([320b88e](https://github.com/sovantica/engrava/commit/320b88e0cb9f66981f51abd5b48c4f389167ead8))
* **hygiene:** add a wall-clock restore window before permanent GC ([35c2769](https://github.com/sovantica/engrava/commit/35c276997a47b1091e56a62425132329c8e5eca9))
* **hygiene:** guard archival behind a minimum inactivity age and a usage-signal gate ([0ce64b7](https://github.com/sovantica/engrava/commit/0ce64b74e91d3ee270c9ecb4d480d6b621f7491d))
* **search:** add transaction-time recency axis with caller-supplied now ([49f8c9b](https://github.com/sovantica/engrava/commit/49f8c9b8141bbff07f4fb6a7bbfb377d5f069abe))
* **search:** exclude archived thoughts from default retrieval ([963e60c](https://github.com/sovantica/engrava/commit/963e60c32e222f52f2d05d9eb443fc2e4d214e87))
* **search:** reject wrong-dimension query vectors and count vector-arm degradation ([fe128f1](https://github.com/sovantica/engrava/commit/fe128f1161309d700ccb6e3860daaac489d521e4))

### Fixed

* **cli:** cover the expiry sweep and report an unreadable snapshot ([4ab31e4](https://github.com/sovantica/engrava/commit/4ab31e4a01aa174ecb1432d152ae22342d79c304))
* **cli:** give an invalid --service name a distinct, clean error ([5f5372d](https://github.com/sovantica/engrava/commit/5f5372d4408b7836ce72004e4f824e0d60cd2556))
* **cli:** purge the vector index when gc collects a thought ([44c6ffc](https://github.com/sovantica/engrava/commit/44c6ffc956340387b675a431458c7863bf1106a3))
* **cli:** validate a resolved empty or default service name ([6318f98](https://github.com/sovantica/engrava/commit/6318f98c564bd20a3cc1cf40be45fdc450aeedc1))
* **cli:** validate snapshot-restore input against a typed model and restore atomically ([d35dc5b](https://github.com/sovantica/engrava/commit/d35dc5b4bd56635f766f10198b7cbfbb484ba679))
* close Free audit source follow-ups ([d919d85](https://github.com/sovantica/engrava/commit/d919d85e43de1eaae5fce953b684446aa97f58fd))
* **config:** enforce uniform validation across sections and construction paths ([97a9af7](https://github.com/sovantica/engrava/commit/97a9af7d0ddd31fe050e033bea5b7a23c4ffa2b5))
* **config:** make the validated value the value that gets used ([e1d7d00](https://github.com/sovantica/engrava/commit/e1d7d003b4cb5bbebdc0d2a2e1801fdd1378d56f))
* **config:** use assign-to-variable message in unknown-key rejection ([69bce4e](https://github.com/sovantica/engrava/commit/69bce4e0b205cfa36c040def917296b573775097))
* **core:** reject inverted valid_from/valid_until intervals ([dd9305f](https://github.com/sovantica/engrava/commit/dd9305febd4e2d85a4c4c06f0f90d1fc504fa6b0))
* **dreaming:** propagate real integrity failures during edge creation ([79241c2](https://github.com/sovantica/engrava/commit/79241c239f06f17a12430fa8097d467b20fad9a7))
* **embeddings:** name the provider member a search needs instead of failing on it ([5e461f6](https://github.com/sovantica/engrava/commit/5e461f65ae905c2ba2c56947834e41e4c5140790))
* **infra:** harden core bootstrap and edge integrity classification ([be054f6](https://github.com/sovantica/engrava/commit/be054f665422580a032dc4b928d61ca73cd14add))
* **infra:** keep foreign-key enforcement safe on every swap failure path ([c393980](https://github.com/sovantica/engrava/commit/c39398044774293c6ce1ac34ce3bd74a656571b7))
* **infra:** make the v11->v12 child-table swap atomic via a savepoint ([7b9a1d0](https://github.com/sovantica/engrava/commit/7b9a1d0603d69c99912569b0ca2c229952225169))
* **infra:** write only the fields an update owns ([9d72125](https://github.com/sovantica/engrava/commit/9d721252d6408b6fc4e4eb64af1b05c28079dba4))
* **journal:** reclaim per-connection append locks with a weak-key registry ([412dcd8](https://github.com/sovantica/engrava/commit/412dcd8cca67bcff33fd3764cc765b0371fb0840))
* **mindql:** build the passthrough guard on a value the module owns ([6e8db02](https://github.com/sovantica/engrava/commit/6e8db02c1b98b8f30a76d205f016e114f836edb8))
* **mindql:** validate identifiers where the query is executed ([d4d8111](https://github.com/sovantica/engrava/commit/d4d811130cd1e0f3bd57ab2707164fc930a10be6))
* **read-only:** capability-separate the read-only view from the core protocol ([1033a2e](https://github.com/sovantica/engrava/commit/1033a2e8932d92d2944e305d819a1d46a0cd64cd))
* **search:** keep FTS5 MATCH valid so quoted and wildcard queries never silently drop BM25 ([5c99b88](https://github.com/sovantica/engrava/commit/5c99b88a5ec4582c099ed9e767b48b90038a65a0))
* **search:** quote exposed FTS5 boolean operators so bare queries never lose BM25 ([ad7ec85](https://github.com/sovantica/engrava/commit/ad7ec8583e68ed4cba530aca3de31e6f2383302d))
* **sqlite:** harden core migration registry ([5c9ec91](https://github.com/sovantica/engrava/commit/5c9ec9189f27f3a9a7996560ac839af66cae15ee))
* **sqlite:** harden extension migration identity and statement splitting ([cfc5e95](https://github.com/sovantica/engrava/commit/cfc5e9533c869ab96e2c5b7dc701de6eaa1049dc))

## [0.5.0](https://github.com/sovantica/engrava/compare/v0.4.0...v0.5.0) (2026-07-08)

### ⚠ BREAKING CHANGES

* **changelog:** the in-tree MCP server is removed from engrava. The engrava[mcp]
optional-dependency extra, the in-engrava engrava-mcp console script, and the
in-tree server module are gone; a plain 'pip install engrava' is unaffected. The
server moved to the standalone engrava-mcp package (uvx engrava-mcp), which
consumes engrava's public API. Migrate per the docs/upgrade.md 0.4 -> 0.5 notes.

### Added

* **core:** action-outcome feedback loop and mutable action lifecycle ([2008a3f](https://github.com/sovantica/engrava/commit/2008a3f73b474cdf644585ac44c0855b9e0dc21a))
* **core:** add opt-in deterministic memory-hygiene forgetting loop ([9167948](https://github.com/sovantica/engrava/commit/9167948f141cbf3b4dae26a87dfd4d17b96701f5))
* **core:** batch/get-or-create write primitives and embed-failure visibility ([e9c2021](https://github.com/sovantica/engrava/commit/e9c202199c248ea1b0dccde6c89e32ebab39bd03))
* **core:** opt-in typed provenance-context capture at create_thought ([b8b8fa3](https://github.com/sovantica/engrava/commit/b8b8fa3a69ddf062b49882578ebb25f126dd4d83))
* **dreaming:** activate consolidation — reachable scoring + live access substrate ([5114f81](https://github.com/sovantica/engrava/commit/5114f81e6f0b485c8b580da0e3018fe1ebc27ea7))
* **embeddings:** opt-in asymmetric query/document prefixes ([12e61fb](https://github.com/sovantica/engrava/commit/12e61fbd0b8c1d5897fdf7f9186f53336a6cb85e))
* **journal:** expose hash-chain verification via API, CLI, and on-open gate ([b26f519](https://github.com/sovantica/engrava/commit/b26f5191f6dac0985a552c6333e788186e6f3cf7))
* **mindql:** read-surface ergonomics — IN, boolean WHERE, ORDER BY, OFFSET, EXPLAIN, bound SELECT ([eca5a74](https://github.com/sovantica/engrava/commit/eca5a743017b07c44caebbc23ae6994c0ec2c7fb))
* remove the in-tree MCP server (now the standalone engrava-mcp package) ([1417e91](https://github.com/sovantica/engrava/commit/1417e91680298c650b8454e44e948057fa17cc30))
* **search:** add collapse_key de-fragmentation to hybrid retrieval ([6c80277](https://github.com/sovantica/engrava/commit/6c8027759a5073d96a9d73578af5980203efe4a8))
* **search:** add metadata and visibility filters to ranked retrieval ([a56fcc7](https://github.com/sovantica/engrava/commit/a56fcc7bdceb792ea714aaca7e8f447381f5efc6))
* **search:** add opt-in per-unit retention depth for collapse backfill ([e72cd22](https://github.com/sovantica/engrava/commit/e72cd22f5ca2f12724ddd6cdbe70502153f84381))
* **search:** batch read-path decode, inbound edge index, eviction visibility ([0e1b81f](https://github.com/sovantica/engrava/commit/0e1b81f4701ae362c3c49b9481914cc049560dcd))

### Fixed

* **dreaming:** add opt-in cold-start clustering fallback ([35aedab](https://github.com/sovantica/engrava/commit/35aedabe843c43ef7f4c334335c8593cf88c32e4))
* **lifecycle:** make archived thoughts restorable to ACTIVE ([88c2e3e](https://github.com/sovantica/engrava/commit/88c2e3eb3c5bc7a9baaffb513b7e46627a6f5a13))
* **search:** apply filters/visibility in the query-less fallback arm ([2f0640e](https://github.com/sovantica/engrava/commit/2f0640ee76d0947fcbd5708d5f67dbf8709c5c34))
* **search:** correct the sqlite-vec backend — purge deleted vectors, fill top_k ([d884052](https://github.com/sovantica/engrava/commit/d884052e871ddcc4d576471ffcee12cb6fee06ef))
* **search:** neutral midpoint for the degenerate min-max fusion case ([815f23b](https://github.com/sovantica/engrava/commit/815f23bcbeff1b45ffaccf82f7a289e06ca4837b))
* use absolute GitHub links in README so PyPI does not 404 ([acf56bf](https://github.com/sovantica/engrava/commit/acf56bfce59eafa9101295dec753cad7deb356d8))

### Documentation

* **changelog:** curate 0.5.0 Unreleased block and mark MCP removal breaking ([5cf3a78](https://github.com/sovantica/engrava/commit/5cf3a78f2108b84836d1b1a0c1b72a82450f1241))

## [0.4.0](https://github.com/sovantica/engrava/compare/v0.3.1...v0.4.0) (2026-06-18)

### Added

* add bi-temporal valid-time to thoughts and edges ([456bcb6](https://github.com/sovantica/engrava/commit/456bcb6f66710ce9a51a5761dd7e406d15e1d177))
* add temporal query predicates and invalidate primitive ([86de77f](https://github.com/sovantica/engrava/commit/86de77fc11ab4322927648d8c7a51ea6d13da91a))
* **api:** add remember() and recall() convenience methods on the store ([be9b110](https://github.com/sovantica/engrava/commit/be9b110d24494f3641bd504a00c2fc838b55c867))
* **mcp:** add delete_thought and delete_edge tools ([84b87f6](https://github.com/sovantica/engrava/commit/84b87f6569958052605a87fbed1aebb3fe734043))
* **mcp:** add guided memory prompts ([1f6fa36](https://github.com/sovantica/engrava/commit/1f6fa36d1b1fef6d63fa2f949c6f18441c84f25b))
* **mcp:** add MCP server with read tools (engrava[mcp] extra) ([68a8085](https://github.com/sovantica/engrava/commit/68a8085c601eb6343ee21622ff422f0234cdc294))
* **mcp:** add memory filters and pagination ([57357b7](https://github.com/sovantica/engrava/commit/57357b711b584e4b3b5a2d206344d33230f35a9e))
* **mcp:** add write tools, opt-in read-only mode, and per-tool safety annotations ([79d7604](https://github.com/sovantica/engrava/commit/79d7604aa492aeab8b2e8dacbbaab1231738d5c7))
* **mcp:** expose memory as resources (thought, stats, recent) ([c54dcf7](https://github.com/sovantica/engrava/commit/c54dcf77180a31c7bcb06d815de316ae1c93d488))
* **mcp:** map known failures to typed, actionable tool errors ([8b615cc](https://github.com/sovantica/engrava/commit/8b615cc58c5b81adf33a29de16be8060f2bf9bfe))
* **mindql:** add store-level execute_mindql entry point ([1c8ffb4](https://github.com/sovantica/engrava/commit/1c8ffb4b18c3ed817f6e229744e943262608413c))
* reflections inherit temporal extent from members ([8fba769](https://github.com/sovantica/engrava/commit/8fba76984c59e5c9424d68a4bba21be53b72d9a6))

### Fixed

* assert plan-shape invariant for temporal queries, not scan-vs-index ([0e4e176](https://github.com/sovantica/engrava/commit/0e4e1764ac770e837012852099c90c7c26b0c4a4))
* embed full thought content without duplication or silent truncation ([36e08e7](https://github.com/sovantica/engrava/commit/36e08e773f6a48d88b9875e1e42ce7051640e7f9))
* **embeddings:** retry transient errors with bounded backoff ([897c46f](https://github.com/sovantica/engrava/commit/897c46fd9300c4b8b9530852d414043593ffffbe))
* keep quoted MindQL values as strings and reject malformed conditions ([d88043d](https://github.com/sovantica/engrava/commit/d88043df45ab05047c20517a5882f071ad7cab92))
* let natural-language queries reach the full-text index ([bb6b729](https://github.com/sovantica/engrava/commit/bb6b7290d546185c7d01789e204338482e39e7cf))
* match exact table token in query-plan helpers ([cd4ecc2](https://github.com/sovantica/engrava/commit/cd4ecc264dff0e425a0dc0708deb5b68f83fab52))
* **mcp:** keep query_memory parse errors FIND-only ([5f4ea20](https://github.com/sovantica/engrava/commit/5f4ea2009d4e4527c3af1fc429055c61f38e509c))
* **mcp:** map write-tool errors and complete the 0.4.0 documentation ([6794436](https://github.com/sovantica/engrava/commit/6794436cf4eec8e16e01a67091d00435b29d1ca5))

### Changed

* tune sqlite pragmas and add hot-path indexes ([1256303](https://github.com/sovantica/engrava/commit/12563030ebab86c481fae341eaccabd8db2223eb))

## [0.3.1](https://github.com/sovantica/engrava/compare/v0.3.0...v0.3.1) (2026-06-02)

### Fixed

* **vector:** load sqlite-vec extension on the connection's worker thread ([457d2f7](https://github.com/sovantica/engrava/commit/457d2f724f877e74509ca711a122293a151e3b01))
* **vector:** re-disable extension loading in finally after load attempt ([e9ab267](https://github.com/sovantica/engrava/commit/e9ab2679c51508f1c2e0ec350c1635bbf2906be7))

## [0.3.0](https://github.com/sovantica/engrava/compare/v0.2.0...v0.3.0) (2026-06-02)

### Added

* graph memory database — dreaming consolidation, hybrid search, audit trail ([ed82259](https://github.com/sovantica/engrava/commit/ed822599f6e2922ec5b3eed8dc14e4a7bb9b024a))

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
