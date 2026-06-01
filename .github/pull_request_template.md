<!--
Thanks for opening a PR!

PR title MUST follow Conventional Commits:
  <type>(<scope>): <description>

Examples:
  feat(search): add fuzzy matching to FTS5 query builder
  fix(cli): handle missing config file gracefully
  docs(readme): add installation troubleshooting

Commitlint CI will enforce the format. See BRANCHING.md and CLAUDE.md
for branch and commit conventions.
-->

## What

<!-- Short summary of the change. One or two sentences. User-facing language. -->

## Why

<!-- Motivation / problem being solved. Link GitHub issues only (Closes #NNN). -->

## How

<!-- Technical approach. Mention tradeoffs or design decisions. -->

## How tested

- [ ] Unit tests
- [ ] Integration tests (if applicable)
- [ ] Manual verification (describe):

## Checklist

**Quality gates:**

- [ ] Tests added/updated (coverage ≥ 90% for new code)
- [ ] `ruff check` + `ruff format --check` clean
- [ ] `mypy --strict` clean
- [ ] Docs updated (README / docs / docstrings) if user-facing
- [ ] `## [Unreleased]` in `CHANGELOG.md` updated if change is non-trivial *(optional — semantic-release will auto-fill from commits)*

**Discipline (CI will block if violated):**

- [ ] Commit messages follow Conventional Commits and use only public-facing terminology
- [ ] Branch name follows `<type>/<kebab-description>` (`BRANCHING.md`)
- [ ] PR title and body describe the change in public-domain terms
- [ ] No breaking changes — OR — `!` in type AND `BREAKING CHANGE:` footer

## Breaking changes

<!-- If this PR has breaking changes, describe them here AND ensure commit message has `!` + `BREAKING CHANGE:` footer. -->

None.

## AI-agent disclosure

<!-- If an AI agent (Claude, Copilot, Cursor, Aider, etc.) substantively authored this PR, disclose. -->

- [ ] AI-agent used: <none / Claude / Copilot / Cursor / other>
