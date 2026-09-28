# Contributing to engrava

Thank you for your interest in contributing to **engrava**! This document
explains how to set up a development environment, submit changes, and what
we look for in contributions.

## Scope

engrava is **the memory database for AI agents**. Contributions should
stay within this scope:

**In scope:**
- Thought/edge/embedding CRUD improvements
- MindQL query language enhancements
- New embedding providers
- Extension system improvements
- Performance optimizations
- Documentation and examples
- Bug fixes and test coverage

**Out of scope:**
- Application-layer logic (planners, reasoners, cognitive architectures)
- Web UI or REST API (use engrava as a library)
- Non-SQLite storage backends (this is an SQLite-first project)

## Development Setup

```bash
# Clone the repository
git clone https://github.com/sovantica/engrava.git
cd engrava

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\Activate.ps1  # Windows

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

## Quality Standards

This project maintains strict code quality. All contributions must pass:

### Linting (ruff)

```bash
ruff check src/ tests/ examples/ scripts/
ruff format --check src/ tests/ examples/ scripts/
```

All ruff rules are enabled (`select = ["ALL"]`). Fix any violations before
submitting.

### Type Checking (mypy)

```bash
mypy --strict src/
```

Zero errors required. Use proper type annotations — no `Any`, no `type: ignore`
without justification.

### Tests (pytest)

```bash
pytest --cov --cov-fail-under=90
```

- Coverage must stay at or above **90%**.
- Use `async def test_*` directly — `asyncio_mode = "auto"` is configured.
- Prefer real implementations over mocks.
- New features require corresponding tests.

`make gate` runs everything above except the test suite (lint, format check,
type check, and the generated-goldens drift check) — the same static checks
CI runs, plus a check that the local git hooks are still wired. Measured at
~70 seconds on a reference machine, dominated by `mypy --strict`. `make
check` runs the full set, tests included.

## Git Hooks

`make install` (in addition to installing dependencies) points git at this
repository's own `.githooks/` directory and installs commitlint locally so
the hook can run it. Both steps live in `scripts/install_hooks.sh`, which you
can also run directly -- **from the primary checkout only**. It refuses to
run from a linked worktree, naming the primary checkout to run it from
instead: `core.hooksPath` lives in the shared `.git/config`, so installing
from a worktree would repoint every checkout's hooks at that worktree's own
`.githooks/`, which disappears the moment the worktree is removed.

- **`.githooks/commit-msg`** lints the message of the commit being created —
  including the commit you make after `git merge --squash`, whose message is
  hand-written. It skips the grammar check only for a commit created while a
  merge is in progress (`MERGE_HEAD` exists), including a merge whose subject
  was given by hand with `git merge --no-ff -m`.
- `make install` sets `core.hooksPath` to an **absolute** path, not the
  literal string `.githooks`. A relative value is resolved separately by
  every checkout, so a linked worktree on a branch laid out differently — or
  lacking `.githooks/` entirely — would silently run a different copy, or no
  hook at all, with the commit still succeeding. An absolute path is the same
  file from every worktree of this repository.
- **commitlint itself is installed once, at the primary checkout, and every
  worktree resolves it from there** (`NODE_PATH`, set by the hook) — only the
  *configuration* it enforces (`.commitlintrc.js`, `commit-scopes.json`)
  stays branch-local, because those are tracked files and are supposed to
  differ per branch. A commit made from any linked worktree is linted
  against that worktree's own scope enum, using the one shared install.
- **If commitlint cannot be resolved at all, the commit is refused**, with a
  message naming the checkout to run `make install` from. It does not warn
  and let the commit through: a hook that reports its health by existing is
  the exact failure this repository's local gates exist to remove, and a
  missing linter is not an exception to that.
- The hook chains to a locally installed hook, if present, at the standard
  git hooks path shared by every worktree — **before** the grammar check, so
  a linter problem can never cost this repository that separate layer too —
  and can never silently disable one that was already protecting this
  repository.
- `make gate` checks that `core.hooksPath` is set to a working absolute path
  **and** that commitlint actually resolves from the current checkout, not
  merely that a hook file is present and executable. Wiring has gone silently
  unset in this project before, and a hook that runs but cannot check
  anything reads exactly like one that passed.

**What this does not cover.** A commit created by `git rebase -i` with a
`squash` action fires no `commit-msg` hook at all — git's own sequencer passes
`--no-verify` to the `git commit` it runs internally. A `post-rewrite` hook is
given the rewritten commit ids and can read the squashed message, but it runs
after the rebase has rewritten the commits and cannot stop it; this repository
installs none. On a branch that goes through a pull request, CI's
`commitlint.yml` still lints that commit as an ordinary part of the branch. On
a branch that never becomes a pull request — such as this project's
`release/*` branches, which are merged locally — a rebase-squashed commit
message is linted by neither the hook nor `commitlint.yml`. Prefer
`git merge --squash` over `git rebase -i` + `squash` on a branch that will not
go through a pull request, for exactly this reason.

## Pull Request Guidelines

1. **One feature per PR** — keep changes focused and reviewable.
2. **Run all checks locally** before submitting:
   ```bash
   ruff check src/ tests/ examples/ scripts/
   ruff format --check src/ tests/ examples/ scripts/
   mypy --strict src/
   pytest --cov --cov-fail-under=90
   ```
3. **Write clear commit messages** — use imperative mood ("Add FTS5 support",
   not "Added FTS5 support").
4. **Update CHANGELOG.md** — add your changes under `[Unreleased]`.
5. **Update documentation** if your change affects public API or behavior.
6. **Add docstrings** — Google-style docstrings on all public symbols.

## Purity Invariant

Public Engrava artifacts must stay free of internal tier references.

- Do not mention internal product or tier names in `src/engrava/`, `docs/`, or public-facing contributor docs.
- Avoid internal branded example names in comments, docstrings, snippets, and tests.
- Prefer neutral names like `ThirdPartyHooks`, `CustomPlugin`, or `ConsumerApp`.

Before opening a PR, re-read your diff against the three rules above and remove any internal
reference. Maintainers verify this invariant during review.

## Code Style

- **Frozen Pydantic models** for all domain objects.
- **Async-first** — no sync wrappers around async operations.
- **Parameterized SQL** — never interpolate user input into queries.
- **StrEnum** for all categorical values.
- **Protocol-based abstractions** at extension boundaries.

## Reporting Issues

- Use [GitHub Issues](https://github.com/sovantica/engrava/issues).
- Include Python version, OS, and a minimal reproduction.
- For security vulnerabilities, email directly instead of filing a public issue.

## License

By contributing, you agree that your contributions will be licensed under the
MIT License.
