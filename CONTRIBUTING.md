# Contributing to engrava

Thank you for your interest in contributing to **engrava**! This document
explains how to set up a development environment, submit changes, and what
we look for in contributions.

## Scope

engrava is a **thought-graph database for AI agents**. Contributions should
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
ruff check src/ tests/
ruff format --check src/ tests/
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

## Pull Request Guidelines

1. **One feature per PR** — keep changes focused and reviewable.
2. **Run all checks locally** before submitting:
   ```bash
   ruff check src/ tests/
   ruff format --check src/ tests/
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

Run the purity check before opening a PR:

```bash
python scripts/assert_purity.py
```

CI also runs the same check and will fail if a forbidden reference leaks into the public tree.

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
