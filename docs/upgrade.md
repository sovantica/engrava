# Upgrade Guide

Most users can run `pip install --upgrade engrava` safely. Database migration is
automatic on first connection, and the upgrade path is validated in CI before
minor releases.

## What Happens During Upgrade

- Core schema migration runs automatically on the first `ensure_schema()` call.
- Existing data is preserved; migrations are forward-only.
- Extension schema migrations are also applied when an installed extension
  declares them.

In practice, most applications do not need a separate migration step. If your
app already calls `ensure_schema()` during startup, that call performs the
upgrade.

## Before You Upgrade

These steps are recommended, not required:

```bash
cp my-data.db my-data.db.bak
pip install --upgrade engrava
```

- Create a copy of the SQLite database file before the upgrade.
- Review [CHANGELOG.md](../CHANGELOG.md) for breaking changes and database notes.
- If you ship custom extensions, make sure their schema migrations are included
  in the version you are about to install.

## After You Upgrade

Use the CLI to confirm the upgraded database opens correctly:

```bash
engrava --db my-data.db info
engrava --db my-data.db migrate
```

- `engrava info` confirms the database is readable and reports current counts.
- `engrava migrate` is safe to run after upgrade; it re-checks that schema is up to date.
- `engrava gc` is optional if you want to compact archived or expired data after
  the upgrade.

## If Migration Fails

Migration errors should include the failing SQL or the extension responsible for
the failure.

Recommended recovery order:

1. Restore from your `.bak` copy.
2. Re-run the upgrade in a clean virtual environment.
3. Open an issue with the error message, `engrava info` output, and whether the
   failure happened in core schema migration or an extension migration.

When reporting the problem, redact file paths and application-specific content
if needed, but keep the SQL error and schema version details intact.

## Downgrade Policy

Downgrades are not supported for `0.x` releases. Migrations are forward-only.

If you must move data into an older version, use an export/import flow instead
of opening the upgraded database file directly:

```bash
engrava --db my-data.db snapshot -o backup.snapshot.jsonl
engrava --db new-old-version.db restore -i backup.snapshot.jsonl
```

## Compatibility Matrix

| From | To | Supported | Notes |
|---|---|---|---|
| 0.2.0 | 0.2.2 | Yes | Patch-level upgrade, no dedicated new extension migration layer |
| 0.2.2 | 0.3.0 | Yes | Minor upgrade with extension migration tracking and upgrade CI coverage |

## Version Notes

### 0.2.2 -> 0.3.0

- Extension schema migration tracking is now part of the upgrade path.
- Upgrade-path CI validates the `0.2.2 -> main` flow before release.
- Release notes and `CHANGELOG.md` now carry a dedicated `Database Changes`
  section for schema-affecting releases.

### Dreaming Defaults

Future releases that change dreaming defaults should document them here. For
example, a benchmark-facing default such as `dreaming_cycles=1` belongs in this
guide once it becomes part of a shipped release.

## Release Communication Rule

Any release that changes schema behavior must include a `Database Changes`
section in [CHANGELOG.md](../CHANGELOG.md) and in GitHub release notes.