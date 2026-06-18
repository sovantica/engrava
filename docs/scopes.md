# Commit scopes — engrava

Allowed Conventional Commits scopes for this repository. Commitlint uses this list as the authoritative scope-enum.

Format in commits: `<type>(<scope>): <description>`
Example: `feat(dreaming): add priority signal to hybrid search`

## Scopes

- `core` — domain models, core engine (`src/engrava/domain/`, `src/engrava/infrastructure/sqlite/`)
- `domain` — domain-layer models and protocols (`src/engrava/domain/`)
- `infra` — infrastructure layer (SQLite, persistence)
- `dreaming` — consolidation cycle, reflection, edges, priority
- `search` — hybrid search, FTS5, vector, recency
- `embedding` — embedding provider abstraction and implementations
- `mindql` — query language (FIND, COUNT, SELECT, extensions)
- `storage` — SQLite, sqlite-vec backend
- `extensions` — extension system, hook contracts (`EngravaHooksProtocol`)
- `cli` — command-line interface
- `config` — YAML config, env vars
- `docs` — documentation (prose, not code)
- `bench` — benchmarks, performance testing
- `deps` — dependency updates
- `release` — release tooling, semantic-release config
- `ci` — CI workflows (also: commit type `ci:`)
- `build` — build tooling (also: commit type `build:`)

## Adding a new scope

1. Open a PR adding the scope here with 1-line rationale
2. Commitlint will reject unknown scopes (safety net)
3. Merge before using the scope in another commit

## Multi-scope commits

Use comma: `feat(dreaming,search): priority affects retrieval ranking`

## No scope

Omit for truly cross-cutting changes: `chore: upgrade Python to 3.13`
