"""The CLI schema-version gate.

``ensure_schema`` is reached by only three built-in command names
(``migrate``, ``restore``, and ``snapshot`` in service mode); every other
built-in opens through a plain connection and never learns whether the
database it is about to act on is even at the version it understands. The
gate classifies every built-in as destructive or read and checks it against
the database's stamped ``user_version`` accordingly:

* a **destructive** command (``gc``, ``restore``) refuses outright on any
  schema that is not exactly head — below head because deleting through an
  engine that does not understand the schema it is deleting from is how the
  vector-resurrection defect (see ``test_referential_integrity.py``'s
  ``TestDeletionOnAPreCascadeSchema`` and ``test_sqlite_vec.py``'s
  ``TestVectorOwnershipIsTheThoughtNotTheEmbeddingRow``) reached a user in
  the first place, and above head because this build cannot understand it
  either;
* a **read** command (``info``, ``verify``, ``export``, ``snapshot``, and a
  ``query`` that parses as ``FIND``/``COUNT``/``SELECT``) warns on stderr and
  proceeds on a behind schema, but still refuses above head;
* ``query`` classifies the *parsed* command, not the CLI command name — an
  ``EXTENSION`` command can write, so it is refused on a behind schema like
  any other destructive operation (there is today no read-only accessor for
  it to run under).

``gc``'s own behind-schema refusal is covered directly in
``test_deletion_blast_radius.py`` (alongside the blast-radius corpus it
otherwise shares), so this module does not repeat it; it covers gc's
above-head refusal plus every other built-in command.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import aiosqlite
import pytest
from click.testing import CliRunner

from engrava import LifecycleStatus, Priority, SqliteEngravaCore, ThoughtRecord, ThoughtType
from engrava.cli.main import cli
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
    """A real head-schema (v20) single-file database with one thought."""
    db_path = tmp_path / name

    async def _setup() -> None:
        conn = await aiosqlite.connect(str(db_path))
        conn.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(conn, embedding_provider=None, auto_embed=False)
        store._owns_connection = True  # noqa: SLF001
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


def test_single_file_snapshot_refuses_on_an_ahead_schema(
    runner: CliRunner, tmp_path: Path
) -> None:
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


def test_restore_refuses_on_a_pre_existing_behind_target(
    runner: CliRunner, tmp_path: Path
) -> None:
    db_path = _behind_db(tmp_path)
    snap = tmp_path / "snap.jsonl"
    snap.write_text("", encoding="utf-8")

    result = runner.invoke(cli, ["--db", str(db_path), "restore", "-i", str(snap)])

    assert result.exit_code != 0, result.output
    assert "migrate" in result.output.lower()
    assert asyncio.run(_stamped_version(db_path)) == _BEHIND_VERSION


def test_restore_refuses_on_a_pre_existing_ahead_target(
    runner: CliRunner, tmp_path: Path
) -> None:
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
# migrate: the one built-in that calls ensure_schema() unconditionally
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
