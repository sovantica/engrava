"""The CLI schema-version gate.

A plain connection never learns whether the database it is about to act on is
even at the version it understands. The gate checks the database's stamped
``user_version`` before a command acts on it, and classifies the command as
destructive or read:

* a **destructive** command (``gc``, ``restore``) refuses outright on any
  schema that is not exactly head — below head because deleting through an
  engine that does not understand the schema it is deleting from is how the
  vector-resurrection defect (see ``test_referential_integrity.py``'s
  ``TestDeletionOnAPreCascadeSchema`` and ``test_sqlite_vec.py``'s
  ``TestVectorOwnershipIsTheThoughtNotTheEmbeddingRow``) reached a user in
  the first place, and above head because this build cannot understand it
  either;
* a **read** command (``info``, ``verify``, ``export``, ``snapshot``,
  ``recall`` without ``--config``, and a ``query`` that parses as
  ``FIND``/``COUNT``/``SELECT``) warns on stderr and proceeds on a behind
  schema, but still refuses above head. ``recall`` run through ``--config``
  refuses a behind schema too, because opening a configured store applies
  pending core migrations and a read must not;
* ``remember`` and ``link`` refuse a pre-existing database that is not at
  head, and create one that is absent or zero-byte at head;
* ``query`` classifies the *parsed* command, not the CLI command name — an
  ``EXTENSION`` command can write, so it is refused on a behind schema like
  any other destructive operation (there is today no read-only accessor for
  it to run under).

``gc``'s own behind-schema refusal is covered directly in
``test_deletion_blast_radius.py`` (alongside the blast-radius corpus it
otherwise shares), so this module does not repeat it; it covers gc's
above-head refusal and the rest of the gate.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
import uuid
from typing import TYPE_CHECKING

import aiosqlite
import pytest
from click.testing import CliRunner

from engrava import LifecycleStatus, Priority, SqliteEngravaCore, ThoughtRecord, ThoughtType
from engrava.cli.main import (
    _ahead_schema_refusal,
    _behind_schema_refusal,
    _behind_schema_warning,
    cli,
)
from engrava.domain.protocols.hooks import MindQLExtension
from engrava.infrastructure.sqlite.engrava_core import CORE_SCHEMA_HEAD_VERSION
from tests.test_migration_upgrade_chains import _bootstrap_core_at_version

if TYPE_CHECKING:
    from pathlib import Path

#: The schema version the "behind" fixture is stamped at — well below head,
#: and (per test_deletion_blast_radius.py's own fixture) genuinely
#: pre-cascade, so it is representative of the real defect shape rather than
#: an arbitrary "not head" number.
_BEHIND_VERSION = 11

_AHEAD_VERSION = CORE_SCHEMA_HEAD_VERSION + 1


def _make_thought(tid: str = "t-1") -> ThoughtRecord:
    return ThoughtRecord(
        thought_id=tid,
        thought_type=ThoughtType.OBSERVATION,
        essence=f"essence-{tid}",
        content=f"content-{tid}",
        priority=Priority.P3,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=1,
        updated_cycle=1,
        source="test",
    )


def _behind_db(tmp_path: Path, name: str = "behind.db") -> Path:
    """A real pre-cascade (core-11) single-file database with one thought."""
    db_path = tmp_path / name

    async def _setup() -> None:
        conn = await aiosqlite.connect(str(db_path))
        conn.row_factory = aiosqlite.Row
        await _bootstrap_core_at_version(conn, _BEHIND_VERSION)
        await conn.execute(
            "INSERT INTO thought (thought_id, thought_type, essence, content, "
            "priority, lifecycle_status) VALUES ('t-1', 'OBSERVATION', 'e', 'c', 'P2', 'ACTIVE')"
        )
        await conn.commit()
        await conn.close()

    asyncio.run(_setup())
    return db_path


def _head_db(tmp_path: Path, name: str = "head.db") -> Path:
    """A real head-schema single-file database with one thought."""
    db_path = tmp_path / name

    async def _setup() -> None:
        conn = await aiosqlite.connect(str(db_path))
        conn.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(conn, embedding_provider=None, auto_embed=False)
        store._owns_connection = True
        await store.ensure_schema()
        await store.create_thought(_make_thought())
        await store.close()

    asyncio.run(_setup())
    return db_path


def _ahead_db(tmp_path: Path, name: str = "ahead.db") -> Path:
    """A real head-schema database subsequently stamped above this build's head."""
    db_path = _head_db(tmp_path, name=name)

    async def _bump() -> None:
        conn = await aiosqlite.connect(str(db_path))
        await conn.execute(f"PRAGMA user_version = {_AHEAD_VERSION}")
        await conn.commit()
        await conn.close()

    asyncio.run(_bump())
    return db_path


async def _stamped_version(db_path: Path) -> int:
    conn = await aiosqlite.connect(str(db_path))
    try:
        cursor = await conn.execute("PRAGMA user_version")
        row = await cursor.fetchone()
        return int(row[0]) if row else 0
    finally:
        await conn.close()


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# ---------------------------------------------------------------------------
# Read-classified commands: warn and attempt on behind, refuse on ahead
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("command_args", [["info"], ["verify"], ["export"]])
def test_read_command_warns_and_attempts_on_a_behind_schema(
    runner: CliRunner, tmp_path: Path, command_args: list[str]
) -> None:
    db_path = _behind_db(tmp_path)
    result = runner.invoke(cli, ["--db", str(db_path), *command_args])

    assert result.exit_code == 0, result.output
    assert "behind" in result.output.lower()
    # A warn-and-attempt must not itself migrate the database.
    assert asyncio.run(_stamped_version(db_path)) == _BEHIND_VERSION


@pytest.mark.parametrize("command_args", [["info"], ["verify"], ["export"]])
def test_read_command_refuses_on_an_ahead_schema(
    runner: CliRunner, tmp_path: Path, command_args: list[str]
) -> None:
    db_path = _ahead_db(tmp_path)
    result = runner.invoke(cli, ["--db", str(db_path), *command_args])

    assert result.exit_code != 0, result.output
    assert "newer" in result.output.lower()
    assert asyncio.run(_stamped_version(db_path)) == _AHEAD_VERSION


def test_single_file_snapshot_warns_and_attempts_on_a_behind_schema(
    runner: CliRunner, tmp_path: Path
) -> None:
    db_path = _behind_db(tmp_path)
    out = tmp_path / "snap.jsonl"
    result = runner.invoke(cli, ["--db", str(db_path), "snapshot", "-o", str(out)])

    assert result.exit_code == 0, result.output
    assert "behind" in result.output.lower()
    assert out.exists()
    assert asyncio.run(_stamped_version(db_path)) == _BEHIND_VERSION


def test_single_file_snapshot_refuses_on_an_ahead_schema(runner: CliRunner, tmp_path: Path) -> None:
    db_path = _ahead_db(tmp_path)
    out = tmp_path / "snap.jsonl"
    result = runner.invoke(cli, ["--db", str(db_path), "snapshot", "-o", str(out)])

    assert result.exit_code != 0, result.output
    assert "newer" in result.output.lower()
    assert not out.exists()


# ---------------------------------------------------------------------------
# query: classifies the parsed command, not the CLI command name
# ---------------------------------------------------------------------------


def test_query_find_warns_and_attempts_on_a_behind_schema(
    runner: CliRunner, tmp_path: Path
) -> None:
    db_path = _behind_db(tmp_path)
    result = runner.invoke(
        cli, ["--db", str(db_path), "query", "FIND thoughts WHERE lifecycle_status = 'ACTIVE'"]
    )

    assert result.exit_code == 0, result.output
    assert "behind" in result.output.lower()
    assert "t-1" in result.output


def test_query_select_refuses_on_an_ahead_schema(runner: CliRunner, tmp_path: Path) -> None:
    db_path = _ahead_db(tmp_path)
    result = runner.invoke(cli, ["--db", str(db_path), "query", "SELECT * FROM thought"])

    assert result.exit_code != 0, result.output
    assert "newer" in result.output.lower()


def test_query_extension_refuses_on_a_behind_schema(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An EXTENSION command is refused like a destructive one on a behind schema.

    There is today no read-only accessor an extension handler could be
    restricted to, so the gate treats any EXTENSION command as capable of
    writing and refuses it exactly like a destructive built-in — never mind
    that this particular handler only reads.
    """

    async def _handler(conn: object, args: object) -> list[dict[str, object]]:
        del conn, args
        return [{"ok": True}]

    fake_extensions = {
        "PING": MindQLExtension(command_name="PING", handler=_handler, description="echo")
    }
    monkeypatch.setattr(
        "engrava.cli.main._load_mindql_extensions",
        lambda: fake_extensions,
    )

    db_path = _behind_db(tmp_path)
    result = runner.invoke(cli, ["--db", str(db_path), "query", "PING"])

    assert result.exit_code != 0, result.output
    assert "migrate" in result.output.lower()


# ---------------------------------------------------------------------------
# gc: destructive, refuses on an ahead schema too (behind is covered in
# test_deletion_blast_radius.py alongside the shared blast-radius corpus)
# ---------------------------------------------------------------------------


def test_gc_refuses_on_an_ahead_schema(runner: CliRunner, tmp_path: Path) -> None:
    db_path = _ahead_db(tmp_path)
    result = runner.invoke(cli, ["--db", str(db_path), "gc"])

    assert result.exit_code != 0, result.output
    assert "newer" in result.output.lower()


# ---------------------------------------------------------------------------
# restore: destructive; a pre-existing behind or ahead target refuses, a
# fresh target still bootstraps exactly as before
# ---------------------------------------------------------------------------


def test_restore_refuses_on_a_pre_existing_behind_target(runner: CliRunner, tmp_path: Path) -> None:
    db_path = _behind_db(tmp_path)
    snap = tmp_path / "snap.jsonl"
    snap.write_text("", encoding="utf-8")

    result = runner.invoke(cli, ["--db", str(db_path), "restore", "-i", str(snap)])

    assert result.exit_code != 0, result.output
    assert "migrate" in result.output.lower()
    assert asyncio.run(_stamped_version(db_path)) == _BEHIND_VERSION


def test_restore_refuses_on_a_pre_existing_ahead_target(runner: CliRunner, tmp_path: Path) -> None:
    db_path = _ahead_db(tmp_path)
    snap = tmp_path / "snap.jsonl"
    snap.write_text("", encoding="utf-8")

    result = runner.invoke(cli, ["--db", str(db_path), "restore", "-i", str(snap)])

    assert result.exit_code != 0, result.output
    assert "newer" in result.output.lower()
    assert asyncio.run(_stamped_version(db_path)) == _AHEAD_VERSION


def test_restore_still_bootstraps_a_fresh_target(runner: CliRunner, tmp_path: Path) -> None:
    """A target that does not exist yet is a fresh restore, not a behind one."""
    db_path = tmp_path / "does-not-exist-yet.db"
    snap = tmp_path / "snap.jsonl"
    snap.write_text("", encoding="utf-8")

    result = runner.invoke(cli, ["--db", str(db_path), "restore", "-i", str(snap)])

    assert result.exit_code == 0, result.output
    assert asyncio.run(_stamped_version(db_path)) == CORE_SCHEMA_HEAD_VERSION


# ---------------------------------------------------------------------------
# Service mode: the same classification, applied to a named service database
# ---------------------------------------------------------------------------


def _behind_service_db(data_dir: Path, name: str) -> Path:
    """A real pre-cascade (core-11) service database at ``<data_dir>/<name>.db``."""
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / f"{name}.db"

    async def _setup() -> None:
        conn = await aiosqlite.connect(str(db_path))
        conn.row_factory = aiosqlite.Row
        await _bootstrap_core_at_version(conn, _BEHIND_VERSION)
        await conn.execute(
            "INSERT INTO thought (thought_id, thought_type, essence, content, "
            "priority, lifecycle_status) VALUES ('t-1', 'OBSERVATION', 'e', 'c', 'P2', 'ACTIVE')"
        )
        await conn.commit()
        await conn.close()

    asyncio.run(_setup())
    return db_path


def test_service_snapshot_warns_and_attempts_without_migrating(
    runner: CliRunner, tmp_path: Path
) -> None:
    """snapshot --service on a behind target warns, exports, and never migrates it.

    ``EngravaManager.get_store`` normally calls ``ensure_schema()`` as part
    of opening a service — which would itself be exactly the implicit
    migration this warn-and-attempt read must not cause. The
    ``migrate=False`` path (``peek_schema_version`` + ``get_store(...,
    migrate=False)``) is what this test actually pins.
    """
    data_dir = tmp_path / "services"
    db_path = _behind_service_db(data_dir, "target")
    out = tmp_path / "svc.jsonl"

    result = runner.invoke(
        cli,
        [
            "--db",
            str(data_dir / "placeholder.db"),
            "snapshot",
            "--service",
            "target",
            "-o",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "behind" in result.output.lower()
    assert out.exists()
    assert asyncio.run(_stamped_version(db_path)) == _BEHIND_VERSION


def test_service_restore_refuses_on_a_pre_existing_behind_target(
    runner: CliRunner, tmp_path: Path
) -> None:
    data_dir = tmp_path / "services"
    db_path = _behind_service_db(data_dir, "target")
    snap = tmp_path / "snap.jsonl"
    snap.write_text("", encoding="utf-8")

    result = runner.invoke(
        cli,
        [
            "--db",
            str(data_dir / "placeholder.db"),
            "restore",
            "-i",
            str(snap),
            "--service",
            "target",
        ],
    )

    assert result.exit_code != 0, result.output
    assert "migrate" in result.output.lower()
    assert asyncio.run(_stamped_version(db_path)) == _BEHIND_VERSION


# ---------------------------------------------------------------------------
# migrate: calls ensure_schema() unconditionally
# ---------------------------------------------------------------------------
#
# migrate is deliberately not gated by _apply_*_schema_gate — running the
# pending migrations (or refusing to, per ensure_schema's own rules) is its
# entire job. But ensure_schema() can still raise SchemaVersionError itself
# (a populated sub-floor database, or one stamped above this build's head
# version), and the CLI must turn that into a clean, non-zero-exit message
# rather than an uncaught traceback.


def test_migrate_reports_a_clean_message_on_an_ahead_schema(
    runner: CliRunner, tmp_path: Path
) -> None:
    db_path = _ahead_db(tmp_path)

    result = runner.invoke(cli, ["--db", str(db_path), "migrate"])

    assert result.exit_code != 0, result.output
    assert result.exception is None or isinstance(result.exception, SystemExit), (
        "an ahead schema must be a clean refusal, not an uncaught exception"
    )
    assert "newer" in result.output.lower()
    assert asyncio.run(_stamped_version(db_path)) == _AHEAD_VERSION


def test_migrate_reports_a_clean_message_on_a_populated_sub_floor_schema(
    runner: CliRunner, tmp_path: Path
) -> None:
    """A real legacy shape: below the bootstrap floor, but not empty.

    Built by hand rather than via ``_bootstrap_core_at_version`` (which only
    reconstructs versions 2 and up): a bare ``thought`` table with one row,
    stamped at the SQLite default ``user_version`` of 0, is exactly the
    shape ``ensure_schema`` cannot tell apart from an empty file without
    checking for a row.
    """
    db_path = tmp_path / "populated-sub-floor.db"

    async def _setup() -> None:
        conn = await aiosqlite.connect(str(db_path))
        await conn.execute("CREATE TABLE thought (thought_id TEXT PRIMARY KEY, essence TEXT)")
        await conn.execute(
            "INSERT INTO thought (thought_id, essence) VALUES ('legacy-1', 'predates the ladder')"
        )
        await conn.commit()
        await conn.close()

    asyncio.run(_setup())

    result = runner.invoke(cli, ["--db", str(db_path), "migrate"])

    assert result.exit_code != 0, result.output
    assert result.exception is None or isinstance(result.exception, SystemExit), (
        "a populated sub-floor schema must be a clean refusal, not an uncaught exception"
    )
    assert "populated" in result.output.lower() or "carries a core table" in result.output.lower()
    assert asyncio.run(_stamped_version(db_path)) == 0, "a refusal must not stamp any version"


def test_migrate_still_bootstraps_a_genuinely_empty_database(
    runner: CliRunner, tmp_path: Path
) -> None:
    """The ordinary case is unaffected: an empty file still migrates to head."""
    db_path = tmp_path / "empty.db"

    result = runner.invoke(cli, ["--db", str(db_path), "migrate"])

    assert result.exit_code == 0, result.output
    assert asyncio.run(_stamped_version(db_path)) == CORE_SCHEMA_HEAD_VERSION


# ---------------------------------------------------------------------------
# recall: warns and runs on a behind schema (bare tiers), refuses on a behind
# schema under --config, and refuses above head on every tier
# ---------------------------------------------------------------------------
#
# ``--db`` and the default path open a bare connection, which never migrates,
# so a read can warn and run against the schema as stored. ``--config`` builds
# the store through ``SqliteEngravaCore.from_config``, which applies pending
# core migrations as it opens: a read there would rewrite the file, so a schema
# below head is refused (naming ``engrava migrate``) instead of warned about.

_TIERS = ["db", "config"]


def _target_args(tier: str, tmp_path: Path, db_path: Path) -> list[str]:
    """The global options that point a command at *db_path* through one store tier."""
    if tier == "db":
        return ["--db", str(db_path)]
    config_path = tmp_path / "engrava.yaml"
    config_path.write_text(f"database:\n  path: {db_path}\n", encoding="utf-8")
    return ["--config", str(config_path)]


_LONG_AGO_NS = 946_684_800 * 10**9


def _files_of(db_path: Path) -> dict[str, tuple[bytes, int]]:
    """The database file and any side file beside it, as (contents, modification time)."""
    return {
        entry.name: (entry.read_bytes(), entry.stat().st_mtime_ns)
        for entry in sorted(db_path.parent.iterdir())
        if entry.is_file() and entry.name.startswith(db_path.name)
    }


def _pinned_files_of(db_path: Path) -> dict[str, tuple[bytes, int]]:
    """Set every file's modification time far in the past, then read them.

    A command that touches a file it refuses would otherwise leave a modification
    time that differs from the one recorded only by the width of a clock tick.
    """
    for entry in db_path.parent.iterdir():
        if entry.is_file() and entry.name.startswith(db_path.name):
            os.utime(entry, ns=(_LONG_AGO_NS, _LONG_AGO_NS))
    return _files_of(db_path)


@pytest.mark.parametrize("as_json", [False, True], ids=["text", "json"])
@pytest.mark.parametrize("tier", _TIERS)
def test_recall_refuses_an_ahead_schema(
    runner: CliRunner, tmp_path: Path, tier: str, as_json: bool
) -> None:
    db_path = _ahead_db(tmp_path)
    before = _pinned_files_of(db_path)

    args = [*_target_args(tier, tmp_path, db_path), "recall", "essence"]
    result = runner.invoke(cli, [*args, "--json"] if as_json else args)

    assert _files_of(db_path) == before, (
        "a refused database must be left alone: same bytes, same modification time"
    )
    assert result.exit_code == 1, result.output
    assert result.stdout == "", "a refusal must print no results and no JSON object"
    assert result.stderr.strip() == _ahead_schema_refusal(_AHEAD_VERSION, command="recall")


@pytest.mark.parametrize("tier", _TIERS)
def test_a_database_under_a_path_with_uri_characters_is_still_checked(
    runner: CliRunner, tmp_path: Path, tier: str
) -> None:
    """``#``, ``?`` and ``%`` in a directory name must reach the check as part of the path."""
    odd_dir = tmp_path / "a#b?c%41"
    odd_dir.mkdir()
    db_path = _ahead_db(odd_dir)
    before = _pinned_files_of(db_path)

    result = runner.invoke(cli, [*_target_args(tier, tmp_path, db_path), "recall", "essence"])

    assert _files_of(db_path) == before, (
        "a refused database must be left alone: same bytes, same modification time"
    )
    assert result.exit_code == 1, result.output
    assert result.stderr.strip() == _ahead_schema_refusal(_AHEAD_VERSION, command="recall")


@pytest.mark.parametrize("tier", _TIERS)
def test_a_database_named_by_a_relative_path_is_still_checked(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tier: str
) -> None:
    """A relative ``--db`` or ``database.path`` is read from the current directory."""
    db_path = _ahead_db(tmp_path)
    monkeypatch.chdir(tmp_path)
    before = _pinned_files_of(db_path)

    args = [*_target_args(tier, tmp_path, db_path.relative_to(tmp_path)), "recall", "essence"]
    result = runner.invoke(cli, args)

    assert _files_of(db_path) == before, (
        "a refused database must be left alone: same bytes, same modification time"
    )
    assert result.exit_code == 1, result.output
    assert result.stderr.strip() == _ahead_schema_refusal(_AHEAD_VERSION, command="recall")


@pytest.mark.parametrize("as_json", [False, True], ids=["text", "json"])
def test_recall_warns_and_runs_on_a_behind_schema_on_the_bare_tier(
    runner: CliRunner, tmp_path: Path, as_json: bool
) -> None:
    db_path = _behind_db(tmp_path)

    args = ["--db", str(db_path), "recall", "c"]
    result = runner.invoke(cli, [*args, "--json"] if as_json else args)

    assert asyncio.run(_stamped_version(db_path)) == _BEHIND_VERSION, "a read must not migrate"
    assert result.exit_code == 0, result.output
    assert result.stderr.strip() == _behind_schema_warning(_BEHIND_VERSION, command="recall")
    assert result.stdout.strip(), "the read must still answer (rows, or '(no rows)')"
    if as_json:
        assert json.loads(result.stdout)["schema"] == "engrava.cli.recall.v1"


@pytest.mark.parametrize("as_json", [False, True], ids=["text", "json"])
def test_recall_refuses_a_behind_schema_on_the_config_tier(
    runner: CliRunner, tmp_path: Path, as_json: bool
) -> None:
    db_path = _behind_db(tmp_path)
    before = _pinned_files_of(db_path)

    args = [*_target_args("config", tmp_path, db_path), "recall", "c"]
    result = runner.invoke(cli, [*args, "--json"] if as_json else args)

    assert asyncio.run(_stamped_version(db_path)) == _BEHIND_VERSION, "a read must not migrate"
    assert _files_of(db_path) == before, (
        "a refused database must be left alone: same bytes, same modification time"
    )
    assert result.exit_code == 1, result.output
    assert result.stdout == "", "a refusal must print no results and no JSON object"
    assert result.stderr.strip() == _behind_schema_refusal(_BEHIND_VERSION, command="recall")


def test_recall_does_not_initialise_a_zero_byte_file_it_reaches_through_the_config(
    runner: CliRunner, tmp_path: Path
) -> None:
    """``recall`` refuses a zero-byte file instead of initialising it."""
    db_path = tmp_path / "empty.db"
    db_path.touch()
    before = _pinned_files_of(db_path)

    result = runner.invoke(cli, [*_target_args("config", tmp_path, db_path), "recall", "c"])

    assert _files_of(db_path) == before, "a refused file must be left alone"
    assert result.exit_code == 1, result.output
    assert result.stdout == "", "a refusal must print no results"
    assert result.stderr.strip() == _behind_schema_refusal(0, command="recall")


@pytest.mark.parametrize(
    ("command_args", "schema"),
    [
        (["info"], "ahead"),
        (["recall", "essence"], "ahead"),
        (["gc"], "ahead"),
        (["gc"], "behind"),
        (["remember", "a new thought"], "ahead"),
        (["remember", "a new thought"], "behind"),
    ],
)
def test_a_schema_refusal_is_plain_text_under_the_root_json_format(
    runner: CliRunner, tmp_path: Path, command_args: list[str], schema: str
) -> None:
    db_path = _ahead_db(tmp_path) if schema == "ahead" else _behind_db(tmp_path)
    expected = (
        _ahead_schema_refusal(_AHEAD_VERSION, command=command_args[0])
        if schema == "ahead"
        else _behind_schema_refusal(_BEHIND_VERSION, command=command_args[0])
    )

    result = runner.invoke(cli, ["--db", str(db_path), "--format", "json", *command_args])

    assert result.exit_code == 1, result.output
    assert result.stdout == "", "a refusal must print no JSON object"
    assert result.stderr.strip() == expected


@pytest.mark.parametrize("tier", _TIERS)
def test_recall_on_a_head_schema_runs_without_a_warning(
    runner: CliRunner, tmp_path: Path, tier: str
) -> None:
    db_path = _head_db(tmp_path)

    result = runner.invoke(cli, [*_target_args(tier, tmp_path, db_path), "recall", "essence"])

    assert asyncio.run(_stamped_version(db_path)) == CORE_SCHEMA_HEAD_VERSION
    assert result.exit_code == 0, result.output
    assert result.stderr == ""
    assert result.stdout.strip(), "the read must answer (rows, or '(no rows)')"


@pytest.mark.parametrize("command_args", [["recall", "c"], ["remember", "a new thought"]])
def test_a_configured_store_applies_a_pending_extension_migration_to_a_head_database(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command_args: list[str],
) -> None:
    """The gate covers the core schema; a configured store still runs extension migrations."""
    module = f"extension_migration_probe_{uuid.uuid4().hex}"
    (tmp_path / f"{module}.py").write_text(
        "from pathlib import Path\n"
        "from engrava.domain.manifest import ExtensionManifest\n"
        "from engrava.domain.protocols.hooks import DefaultEngravaHooks\n"
        "MANIFEST = ExtensionManifest(\n"
        "    name='probe',\n"
        "    version='1.0.0',\n"
        "    hooks_class=DefaultEngravaHooks,\n"
        "    schema_migrations=[Path(__file__).with_name('001_probe.sql')],\n"
        ")\n",
        encoding="utf-8",
    )
    (tmp_path / "001_probe.sql").write_text(
        "CREATE TABLE probe_extension_table (id TEXT PRIMARY KEY);\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    db_path = _head_db(tmp_path)
    config_path = tmp_path / "engrava.yaml"
    config_path.write_text(
        f"database:\n  path: {db_path}\nmanifests:\n  - {module}:MANIFEST\n", encoding="utf-8"
    )
    try:
        result = runner.invoke(cli, ["--config", str(config_path), *command_args])
    finally:
        sys.modules.pop(module, None)

    assert result.exit_code == 0, result.output
    conn = sqlite3.connect(f"{db_path.absolute().as_uri()}?mode=ro", uri=True)
    try:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
    finally:
        conn.close()
    assert "probe_extension_table" in tables, (
        "a store built from --config applies its extensions' pending migrations as it opens"
    )


# ---------------------------------------------------------------------------
# remember / link: never migrate an existing database's core schema on their own
# ---------------------------------------------------------------------------
#
# ``remember`` and ``link`` are allowed to create a database that is not there
# yet. A file that already exists is different: below head it needs
# ``engrava migrate`` (the one command that is asked to migrate), and above head
# it is newer than this build understands. Either way the database file is left
# exactly as it was found, on ``--db`` and under ``--config`` alike.

_WRITE_VERBS = ["remember", "link"]

_COUNT_SQL = {
    "thought": "SELECT COUNT(*) FROM thought",
    "edge": "SELECT COUNT(*) FROM edge",
}


def _write_args(verb: str, *, as_json: bool = False) -> list[str]:
    """One valid invocation of *verb*; ``link`` joins the two thoughts the fixtures hold."""
    args = (
        ["remember", "a new thought"]
        if verb == "remember"
        else [
            "link",
            "t-1",
            "t-2",
            "--type",
            "ASSOCIATED",
        ]
    )
    return [*args, "--json"] if as_json else args


def _rows_written_by(verb: str) -> str:
    """The table *verb* adds a row to."""
    return "thought" if verb == "remember" else "edge"


def _row_count(db_path: Path, table: str) -> int:
    conn = sqlite3.connect(f"{db_path.absolute().as_uri()}?mode=ro", uri=True)
    try:
        row = conn.execute(_COUNT_SQL[table]).fetchone()
        return int(row[0])
    finally:
        conn.close()


def _add_thought_row(db_path: Path, thought_id: str) -> None:
    """Add a second thought to a fixture, so ``link t-1 t-2`` has both endpoints."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "INSERT INTO thought (thought_id, thought_type, essence, content, "
            "priority, lifecycle_status) VALUES (?, 'OBSERVATION', 'e', 'c', 'P2', 'ACTIVE')",
            (thought_id,),
        )
        conn.commit()
    finally:
        conn.close()


@pytest.mark.parametrize("as_json", [False, True], ids=["text", "json"])
@pytest.mark.parametrize("tier", _TIERS)
@pytest.mark.parametrize("verb", _WRITE_VERBS)
def test_write_verb_refuses_an_ahead_schema(
    runner: CliRunner, tmp_path: Path, verb: str, tier: str, as_json: bool
) -> None:
    db_path = _ahead_db(tmp_path)
    _add_thought_row(db_path, "t-2")
    before = _pinned_files_of(db_path)

    result = runner.invoke(
        cli, [*_target_args(tier, tmp_path, db_path), *_write_args(verb, as_json=as_json)]
    )

    assert _files_of(db_path) == before, (
        "a refused database must be left alone: same bytes, same modification time"
    )
    assert result.exit_code == 1, result.output
    assert result.stdout == ""
    assert result.stderr.strip() == _ahead_schema_refusal(_AHEAD_VERSION, command=verb)


@pytest.mark.parametrize("as_json", [False, True], ids=["text", "json"])
@pytest.mark.parametrize("tier", _TIERS)
@pytest.mark.parametrize("verb", _WRITE_VERBS)
def test_write_verb_refuses_a_behind_schema_and_leaves_it_alone(
    runner: CliRunner, tmp_path: Path, verb: str, tier: str, as_json: bool
) -> None:
    """The refusal names ``engrava migrate``, for ``link`` too."""
    db_path = _behind_db(tmp_path)
    _add_thought_row(db_path, "t-2")
    table = _rows_written_by(verb)
    rows_before = _row_count(db_path, table)
    before = _pinned_files_of(db_path)

    result = runner.invoke(
        cli, [*_target_args(tier, tmp_path, db_path), *_write_args(verb, as_json=as_json)]
    )

    assert asyncio.run(_stamped_version(db_path)) == _BEHIND_VERSION, "must not migrate"
    assert _row_count(db_path, table) == rows_before, "must not add a row"
    assert _files_of(db_path) == before, (
        "a refused database must be left alone: same bytes, same modification time"
    )
    assert result.exit_code == 1, result.output
    assert result.stdout == ""
    assert result.stderr.strip() == _behind_schema_refusal(_BEHIND_VERSION, command=verb)


@pytest.mark.parametrize("tier", _TIERS)
@pytest.mark.parametrize("schema", ["ahead", "behind"])
def test_link_checks_the_schema_before_it_looks_for_its_endpoints(
    runner: CliRunner, tmp_path: Path, tier: str, schema: str
) -> None:
    """``t-2`` is not in the database: the schema refusal comes first, not exit 4."""
    db_path = _ahead_db(tmp_path) if schema == "ahead" else _behind_db(tmp_path)
    before = _pinned_files_of(db_path)

    result = runner.invoke(cli, [*_target_args(tier, tmp_path, db_path), *_write_args("link")])

    assert _files_of(db_path) == before, (
        "a refused database must be left alone: same bytes, same modification time"
    )
    assert result.exit_code == 1, result.output
    assert result.stdout == ""
    expected = (
        _ahead_schema_refusal(_AHEAD_VERSION, command="link")
        if schema == "ahead"
        else _behind_schema_refusal(_BEHIND_VERSION, command="link")
    )
    assert result.stderr.strip() == expected


@pytest.mark.parametrize("tier", _TIERS)
@pytest.mark.parametrize("verb", _WRITE_VERBS)
def test_write_verb_runs_on_a_head_schema(
    runner: CliRunner, tmp_path: Path, verb: str, tier: str
) -> None:
    db_path = _head_db(tmp_path)
    _add_thought_row(db_path, "t-2")
    table = _rows_written_by(verb)
    rows_before = _row_count(db_path, table)

    result = runner.invoke(cli, [*_target_args(tier, tmp_path, db_path), *_write_args(verb)])

    assert _row_count(db_path, table) == rows_before + 1
    assert asyncio.run(_stamped_version(db_path)) == CORE_SCHEMA_HEAD_VERSION
    assert result.exit_code == 0, result.output
    assert result.stderr == ""
    assert result.stdout.strip(), "the new thought or edge id is printed"


@pytest.mark.parametrize("tier", _TIERS)
def test_remember_creates_an_absent_database_at_head(
    runner: CliRunner, tmp_path: Path, tier: str
) -> None:
    db_path = tmp_path / "nested" / "new.db"

    result = runner.invoke(cli, [*_target_args(tier, tmp_path, db_path), *_write_args("remember")])

    assert asyncio.run(_stamped_version(db_path)) == CORE_SCHEMA_HEAD_VERSION
    assert _row_count(db_path, "thought") == 1
    assert result.exit_code == 0, result.output
    assert result.stderr.strip() == f"Created database: {db_path}"


@pytest.mark.parametrize("tier", _TIERS)
def test_link_creates_an_absent_database_at_head(
    runner: CliRunner, tmp_path: Path, tier: str
) -> None:
    """The path is created and stamped at head; the endpoints it names do not exist yet."""
    db_path = tmp_path / "nested" / "new.db"

    result = runner.invoke(cli, [*_target_args(tier, tmp_path, db_path), *_write_args("link")])

    assert asyncio.run(_stamped_version(db_path)) == CORE_SCHEMA_HEAD_VERSION
    assert _row_count(db_path, "edge") == 0
    assert result.exit_code == 4, result.output
    assert f"Created database: {db_path}" in result.stderr


@pytest.mark.parametrize("tier", _TIERS)
def test_remember_initialises_a_zero_byte_file(
    runner: CliRunner, tmp_path: Path, tier: str
) -> None:
    """A file with no schema object and no stamp holds nothing to migrate."""
    db_path = tmp_path / "empty.db"
    db_path.touch()

    result = runner.invoke(cli, [*_target_args(tier, tmp_path, db_path), *_write_args("remember")])

    assert asyncio.run(_stamped_version(db_path)) == CORE_SCHEMA_HEAD_VERSION
    assert _row_count(db_path, "thought") == 1
    assert result.exit_code == 0, result.output
    assert result.stderr == "", "an existing path is not reported as created"


@pytest.mark.parametrize("tier", _TIERS)
def test_link_initialises_a_zero_byte_file(runner: CliRunner, tmp_path: Path, tier: str) -> None:
    db_path = tmp_path / "empty.db"
    db_path.touch()

    result = runner.invoke(cli, [*_target_args(tier, tmp_path, db_path), *_write_args("link")])

    assert asyncio.run(_stamped_version(db_path)) == CORE_SCHEMA_HEAD_VERSION
    assert result.exit_code == 4, result.output
    assert "does not reference an existing thought" in result.stderr


@pytest.mark.parametrize("tier", _TIERS)
def test_remember_refuses_an_unstamped_file_that_holds_a_table(
    runner: CliRunner, tmp_path: Path, tier: str
) -> None:
    """Only a file with no schema object at all is treated as new.

    A file stamped 0 that already carries a table may hold someone's rows; it is
    a schema below head like any other, and ``engrava migrate`` is what decides
    whether it can be brought forward.
    """
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE thought (thought_id TEXT PRIMARY KEY, essence TEXT)")
    conn.execute("INSERT INTO thought (thought_id, essence) VALUES ('legacy-1', 'kept')")
    conn.commit()
    conn.close()
    before = _pinned_files_of(db_path)

    result = runner.invoke(cli, [*_target_args(tier, tmp_path, db_path), *_write_args("remember")])

    assert _files_of(db_path) == before, (
        "a refused database must be left alone: same bytes, same modification time"
    )
    assert result.exit_code == 1, result.output
    assert result.stderr.strip() == _behind_schema_refusal(0, command="remember")


@pytest.mark.parametrize("tier", _TIERS)
def test_link_checks_its_arguments_before_it_looks_at_the_database(
    runner: CliRunner, tmp_path: Path, tier: str
) -> None:
    db_path = _behind_db(tmp_path)
    before = _pinned_files_of(db_path)

    result = runner.invoke(
        cli,
        [
            *_target_args(tier, tmp_path, db_path),
            *["link", "t-1", "t-2", "--type", "ASSOCIATED", "--weight", "2"],
        ],
    )

    assert _files_of(db_path) == before
    assert result.exit_code == 2, result.output
    assert "--weight" in result.stderr


# ---------------------------------------------------------------------------
# The default path (no --db, no --config) is checked like an explicit one
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("verb", ["recall", *_WRITE_VERBS])
def test_the_default_path_refuses_an_ahead_schema(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verb: str
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ENGRAVA_DB", raising=False)
    db_path = _ahead_db(tmp_path, name="engrava.db")
    _add_thought_row(db_path, "t-2")
    before = _pinned_files_of(db_path)

    result = runner.invoke(cli, ["recall", "essence"] if verb == "recall" else _write_args(verb))

    assert _files_of(db_path) == before, "a refused database must be left alone"
    assert result.exit_code == 1, result.output
    assert result.stdout == ""
    assert result.stderr.strip() == _ahead_schema_refusal(_AHEAD_VERSION, command=verb)


@pytest.mark.parametrize("verb", _WRITE_VERBS)
def test_the_default_path_refuses_a_behind_schema_for_a_write(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verb: str
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ENGRAVA_DB", raising=False)
    db_path = _behind_db(tmp_path, name="engrava.db")
    _add_thought_row(db_path, "t-2")
    before = _pinned_files_of(db_path)

    result = runner.invoke(cli, _write_args(verb))

    assert asyncio.run(_stamped_version(db_path)) == _BEHIND_VERSION, "must not migrate"
    assert _files_of(db_path) == before, "a refused database must be left alone"
    assert result.exit_code == 1, result.output
    assert result.stdout == ""
    assert result.stderr.strip() == _behind_schema_refusal(_BEHIND_VERSION, command=verb)


def test_the_default_path_warns_and_runs_recall_on_a_behind_schema(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ENGRAVA_DB", raising=False)
    db_path = _behind_db(tmp_path, name="engrava.db")

    result = runner.invoke(cli, ["recall", "c"])

    assert asyncio.run(_stamped_version(db_path)) == _BEHIND_VERSION, "a read must not migrate"
    assert result.exit_code == 0, result.output
    assert result.stderr.strip() == _behind_schema_warning(_BEHIND_VERSION, command="recall")
