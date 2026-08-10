# Commit scopes — engrava

Allowed Conventional Commits scopes for this repository.

The vocabulary itself lives in one machine-readable file, [`commit-scopes.json`](../commit-scopes.json),
in two tiers. `.commitlintrc.js` reads that file and spreads both tiers into its `scope-enum`; no scope
name is written in the config at all. This page is the prose view of the same two tiers.

Two checks keep the three in agreement, and they prove different things:

- **`tests/test_commit_scope_vocabulary.py`**, in the normal test suite, compares the scopes documented
  here against the JSON tiers in both directions, checks that the two wiring lines of `.commitlintrc.js`
  are **present** in the file, and checks that the word `canonical` occurs in every deprecated
  description. It rejects duplicate entries in either tier. The wiring check is presence only: it catches
  someone replacing the wiring with a literal list, and it cannot show that no second, contradicting line
  exists — proving that over text needs a parser. Descriptions are otherwise unchecked; they are prose.
- **The `Verify commitlint admits the same scope set as commit-scopes.json` step** in `.github/workflows/commitlint.yml` asks commitlint for
  its effective configuration and fails the job unless the enforced `scope-enum` **admits the same set of
  scopes** as the two tiers of `commit-scopes.json`. That covers `extends`, an alternate rc file, a
  `commitlint` key in a `package.json`, and a rule rebuilt after the config is constructed. Its limits:
  it compares sets, so a duplicated or reordered entry passes it — the suite test rejects duplicates, and
  ordering is checked by nothing, since both sides of every comparison become sets — and it
  verifies the rule resolved by its **own default-root invocation** — it does not prove that the lint
  commands beside it in the same job resolve the same configuration if their flags, working directory,
  environment or parser differ.

Two things to know before relying on the vocabulary being enforced at all:

- **`.commitlintrc.js` skips whole messages before any rule runs**, and the exemptions are broader than
  they look. Any message *containing the substring* `Signed-off-by: dependabot[bot]` anywhere in it is
  skipped entirely, with no check that the commit is actually from the bot — so adding that line lets any
  author bypass every rule, scope allowlist included. Separately, messages *beginning* with `merge` or
  `revert` followed by whitespace or a colon are skipped (case-insensitively), which covers `Merge branch
  …` and `Revert "…"` but **not** `revert(hidden): undo`, which is linted normally. Both exemptions are
  deliberate — dependency-bot bodies and the 4-tier merge flow — and neither is authenticated.
- **`main` has no required status checks**, so a red Commitlint job reports and does not block. The scopes
  below are a convention these checks keep honest, not a gate that stops a merge.

Format in commits: `<type>(<scope>): <description>`
Example: `feat(dreaming): add priority signal to hybrid search`

## Scopes

### Canonical — use these in new commits

- `core` — domain models, core engine (`src/engrava/domain/`, `src/engrava/infrastructure/sqlite/`)
- `domain` — domain-layer models and protocols (`src/engrava/domain/`)
- `infra` — infrastructure layer: SQLite backend, persistence, connection handling
- `dreaming` — consolidation cycle, reflection, priority
- `search` — hybrid search, FTS5, vector arm, recency
- `embeddings` — embedding provider abstraction, implementations, role prefixes
- `mindql` — query language (FIND, COUNT, SELECT, extensions)
- `extensions` — extension system, hook contracts (`EngravaHooksProtocol`)
- `cli` — command-line interface
- `config` — YAML config, env vars
- `docs` — documentation (prose, not code)
- `bench` — benchmarks, performance testing
- `deps` — dependency updates
- `release` — release tooling, semantic-release config
- `ci` — CI workflows (also: commit type `ci:`)
- `build` — build tooling (also: commit type `build:`)
- `journal` — tamper-evident thought/edge journal + integrity verification
- `lifecycle` — thought / action lifecycle and state transitions
- `vec0` — sqlite-vec (`vec0`) vector backend specifics
- `actions` — Action Records + action-outcome feedback
- `perf` — performance-oriented changes (as a scope; `perf:` is also a type)
- `architecture` — layering and boundary invariants (tier boundaries, protocol seams)
- `hygiene` — memory hygiene: archival, restore window, deterministic forgetting / GC
- `migration` — schema migrations and the upgrade ladder (singular chosen because it is the spelling
  the current release head uses; history is split 2 plural / 1 singular)
- `read-only` — the read-only store view and its capability boundary
- `readme` — `README.md`
- `pyproject` — `pyproject.toml` (package metadata)
- `changelog` — `CHANGELOG.md`
- `gitignore` — `.gitignore`
- `api-reference` — `docs/api-reference.md`

### Deprecated — commitlint accepts these, but do not use them in new commits

Commitlint lints **every commit** in a pull request, not only its title, and this repository does not
rewrite merged history. The spellings below therefore stay in the `scope-enum` so that `git log` lints
clean — but each is a duplicate of a canonical scope above, or names code this repository no longer
contains. Using one in a new commit passes commitlint and is still wrong; reviewers should reject it.

- `sqlite` — in history; canonical is `infra` (the layer), or `migration` for migration-registry work
- `vector` — in history; canonical is `vec0` for backend specifics, `search` for the vector arm
- `migrations` — in history; canonical is the singular `migration`
- `types` — in history; no canonical of its own — name the layer being retyped (`core`, `domain`, `infra`)
- `tests` — in history; no canonical of its own — name the area under test, or omit the scope
- `edges` — in history; canonical is `core` for the edge record, `dreaming` for edges consolidation creates
- `api` — in history; canonical is `core` for the public store surface (`api-reference` is the docs file)
- `mcp` — in history; nothing canonical here — the MCP server left this repo for the standalone
  `engrava-mcp` package, so no new commit in this repository has that subject
- `performance` — in history; canonical is the short `perf`

## Adding a new scope

1. Open a PR adding the scope to **both** `commit-scopes.json` and this page, with a 1-line rationale
2. `tests/test_commit_scope_vocabulary.py` fails if you update only one of them
3. Do not add scopes to `.commitlintrc.js` — it holds no scope names, and the same test pins that
4. Merge before using the scope in another commit

Prefer an existing scope. A second spelling for a subject area that already has one makes the whole
vocabulary meaningless — the deprecated tier above is the record of that happening. Two never-used
duplicates (`storage`, `embedding`) were removed outright rather than deprecated, because a scope
commitlint accepts for no commit in the repository's history is accepted for nothing.

## Multi-scope commits

Use comma: `feat(dreaming,search): priority affects retrieval ranking`

## No scope

Omit for truly cross-cutting changes: `chore: upgrade Python to 3.13`
