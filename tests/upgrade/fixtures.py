"""Helpers for end-to-end upgrade-path validation."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path


def venv_python_path(venv_dir: Path) -> Path:
    """Return the Python executable path for a virtual environment."""
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def isolated_environment() -> dict[str, str]:
    """Return an environment with the interpreter-path variables removed.

    The whole point of this fixture is that the child processes run the engrava
    the throwaway venv has installed — the previous release first, the candidate
    after the upgrade. An ambient PYTHONPATH defeats that silently: it is
    inherited by every child and outranks the venv's site-packages, so the
    baseline install becomes inert, the fixture database is created at the head
    schema instead of the released one, and the run passes without migrating
    anything. Exporting PYTHONPATH at the working tree is normal practice when
    testing from a git worktree, so this cannot be left to the caller's shell.

    PYTHONHOME is removed defensively, for a different reason: it relocates the
    standard library wholesale. CPython's venv activation clears it too, but
    activation does *not* clear PYTHONPATH — which is why the removal above is
    load-bearing rather than a duplicate of what the venv already does.

    Deleting is chosen for robustness, not because emptying is broken: on the
    CPython versions this suite targets, an unset PYTHONPATH and PYTHONPATH=""
    give an identical sys.path. Only an empty component inside a non-empty
    value (PYTHONPATH=":") contributes an entry. An absent variable is the
    stronger and simpler guarantee, and it does not depend on that behaviour
    staying the same.
    """
    environment = dict(os.environ)
    for variable in ("PYTHONPATH", "PYTHONHOME"):
        environment.pop(variable, None)
    return environment


def run_command(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run a subprocess command in an isolated environment; fail loudly on non-zero exit."""
    return subprocess.run(  # noqa: S603 — trusted test fixture invocation
        command,
        cwd=str(cwd) if cwd is not None else None,
        check=True,
        capture_output=True,
        text=True,
        env=isolated_environment(),
    )


def pip_command(python_executable: Path | str, *arguments: str) -> list[str]:
    """Build a pip invocation the caller's environment cannot redirect.

    This test's claim is that the baseline came from PyPI, as that release
    published it. PIP_INDEX_URL, PIP_EXTRA_INDEX_URL, PIP_FIND_LINKS and
    PIP_CONFIG_FILE are all inherited from the caller and would change where the
    artifact is resolved from with no visible symptom in a passing run — the
    install still succeeds, the version string still reads 0.5.0, and the test
    would be asserting about a different build.

    `--isolated` makes pip ignore environment variables and per-user
    configuration. Note the boundary: it does not disable global or system-wide
    pip configuration, so a machine configured at that level can still point
    these installs elsewhere. It removes the ambient, per-invocation channel,
    not every channel.
    """
    return [str(python_executable), "-m", "pip", "--isolated", *arguments]


def create_venv(venv_dir: Path) -> Path:
    """Create an isolated virtual environment and return its Python executable."""
    run_command([sys.executable, "-m", "venv", str(venv_dir)])
    python_executable = venv_python_path(venv_dir)
    run_command(pip_command(python_executable, "install", "--upgrade", "pip"))
    return python_executable


def install_package(
    python_executable: Path,
    package_spec: str,
    *,
    editable: bool,
    cwd: Path | None = None,
    force_reinstall: bool = False,
) -> None:
    """Install a package spec into the given virtual environment.

    Args:
        python_executable: Interpreter of the target virtual environment.
        package_spec: A PEP 508 requirement, a local path, or (for the
            candidate build this upgrade path installs second) a built
            wheel's path.
        editable: Pass ``-e`` to pip.
        cwd: Working directory for the ``pip install`` subprocess.
        force_reinstall: Pass ``--force-reinstall``. Load-bearing for the
            candidate wheel: this repository's version bump happens inside
            the release pipeline itself, so a wheel built from an
            unreleased working tree carries the *same* version number as
            the last published release until that pipeline actually runs.
            Without ``--force-reinstall``, pip treats that version match as
            "nothing to do" -- confirmed via ``pip install`` printing
            "engrava is already installed with the same version as the
            provided wheel" -- and silently keeps the previously installed
            (real PyPI) build instead of installing this one, which would
            make the upgrade path a no-op upgrade to itself.

    """
    command = pip_command(python_executable, "install")
    if force_reinstall:
        command.append("--force-reinstall")
    if editable:
        command.extend(["-e", package_spec])
    else:
        command.append(package_spec)
    run_command(command, cwd=cwd)


def _populate_fixture_script(
    db_path: Path, pre_snapshot_path: Path, pre_journal_state_path: Path
) -> str:
    return textwrap.dedent(
        f"""
        import asyncio
        import json
        from pathlib import Path

        import aiosqlite

        from engrava import (
            ActionRecord,
            ActionStatus,
            ActionType,
            EdgeRecord,
            EdgeType,
            LifecycleStatus,
            Priority,
            SqliteEngravaCore,
            ThoughtRecord,
            ThoughtType,
            VerificationStatus,
        )
        from engrava.cli.main import _export_db_to_jsonl

        DB_PATH = r"{db_path}"
        PRE_SNAPSHOT_PATH = r"{pre_snapshot_path}"
        PRE_JOURNAL_STATE_PATH = r"{pre_journal_state_path}"

        async def main() -> None:
            # Closed in a finally: for the same reason as the verifier. aiosqlite
            # runs the connection on a non-daemon thread that stops only on
            # close(), so a failure anywhere in the schema creation or the writes
            # below would leave this process unable to exit — a hang at shutdown
            # instead of the error that caused it.
            conn = await aiosqlite.connect(DB_PATH)
            try:
                conn.row_factory = aiosqlite.Row
                # Journaling on so the upgrade path has a real hash chain to
                # preserve, not just the four core tables -- see verify_data()
                # in _verify_upgraded_db_script, which checks the chain survives
                # the migration unchanged rather than only the row counts.
                store = SqliteEngravaCore(conn, journal_enabled=True)
                await store.ensure_schema()

                for index in range(6):
                    thought = ThoughtRecord(
                        thought_id=f"thought-{{index:03d}}",
                        essence=f"Upgrade thought {{index}}",
                        content=f"Representative upgrade fixture thought {{index}}",
                        thought_type=ThoughtType.OBSERVATION,
                        source="upgrade-fixture",
                        lifecycle_status=(
                            LifecycleStatus.ARCHIVED if index == 5 else LifecycleStatus.ACTIVE
                        ),
                        priority=Priority.P1 if index == 0 else Priority.P2,
                        created_cycle=index,
                        updated_cycle=index,
                    )
                    created = await store.create_thought(thought)
                    await store.store_embedding(created.thought_id, [float(index + 1)] * 8)

                edge = EdgeRecord(
                    edge_id="edge-001",
                    from_thought_id="thought-000",
                    to_thought_id="thought-001",
                    edge_type=EdgeType.ASSOCIATED,
                    weight=0.9,
                    created_cycle=1,
                )
                await store.create_edge(edge)

                # Timestamps without an offset, in shapes the FROM release's
                # validator accepted and stored exactly as written: one
                # non-canonical value in every column the upgrade normalises.
                # A release that already normalises on write stores them
                # canonical instead; verify_data() reads which case this was
                # from the pre-upgrade snapshot.
                await store.create_thought(
                    ThoughtRecord(
                        thought_id="thought-naive-timestamps",
                        essence="Upgrade thought with naive timestamps",
                        content="Timestamps written without an offset, in several shapes",
                        thought_type=ThoughtType.OBSERVATION,
                        source="upgrade-fixture",
                        lifecycle_status=LifecycleStatus.ACTIVE,
                        priority=Priority.P2,
                        created_cycle=6,
                        updated_cycle=6,
                        created_at="2026-01-02 03:04:05",
                        updated_at="20260102T030405",
                        last_accessed_at="2026-W01-5T03:04:05",
                        expires_at="2099-12-31 23:00:00.5",
                        valid_from="2026-01-01",
                        valid_until="2099-07-01T00:00:00",
                        archived_at="2026-09-25 12:00:00",
                    )
                )
                await store.create_edge(
                    EdgeRecord(
                        edge_id="edge-naive-timestamps",
                        from_thought_id="thought-001",
                        to_thought_id="thought-002",
                        edge_type=EdgeType.ASSOCIATED,
                        weight=0.5,
                        created_cycle=2,
                        valid_from="2026-01-01 00:00:00",
                        valid_until="20990701T000000",
                    )
                )

                action = ActionRecord(
                    action_id="action-001",
                    source_thought_id="thought-000",
                    action_type=ActionType.CLI_OUTPUT,
                    intent="Representative upgrade fixture action",
                    status=ActionStatus.CONFIRMED,
                    verification_status=VerificationStatus.PENDING,
                )
                await store.create_action(action)

                await conn.commit()

                # Capture the pre-upgrade content and journal chain state, still
                # on the FROM release, for verify_data() to diff against after
                # the upgrade -- reusing the same export function `snapshot`
                # itself calls, rather than inventing a second dump format.
                await _export_db_to_jsonl(conn, Path(PRE_SNAPSHOT_PATH))

                integrity = await store.verify_journal()
                journal_state = {{
                    "valid": integrity.valid,
                    "entries_checked": integrity.entries_checked,
                }}
                if store.journal is not None:
                    entries = await store.journal.get_entries(limit=1000)
                    if entries:
                        last = max(entries, key=lambda e: e.sequence_number)
                        journal_state["last_sequence_number"] = last.sequence_number
                        journal_state["last_entry_hash"] = last.entry_hash
                with open(PRE_JOURNAL_STATE_PATH, "w", encoding="utf-8") as f:
                    json.dump(journal_state, f)
            finally:
                await conn.close()

        asyncio.run(main())
        """
    )


def populate_fixture_db(
    python_executable: Path,
    db_path: Path,
    pre_snapshot_path: Path,
    pre_journal_state_path: Path,
) -> None:
    """Create a representative fixture database using the installed package.

    Also captures a pre-upgrade content snapshot and journal chain state at
    ``pre_snapshot_path`` / ``pre_journal_state_path`` -- still on the FROM
    release -- for :func:`verify_upgraded_db` to diff against once the
    upgrade has run, so the upgrade path is checked for content and chain
    preservation, not only for post-upgrade row counts.
    """
    run_command(
        [
            str(python_executable),
            "-c",
            _populate_fixture_script(db_path, pre_snapshot_path, pre_journal_state_path),
        ]
    )


#: Maps each snapshot ``_type`` to the column identifying one of its records,
#: shared between the pre- and post-upgrade snapshot so records can be paired
#: up regardless of row order.
_SNAPSHOT_PRIMARY_KEYS = {
    "thought": "thought_id",
    "edge": "edge_id",
    "embedding": "embedding_id",
    "action": "action_id",
}

#: The timestamp columns the v20 -> v21 upgrade rewrites into the canonical UTC
#: form. A value in one of them may change its text across the upgrade, but
#: never its instant.
_NORMALISED_TIMESTAMP_COLUMNS = {
    "thought": (
        "created_at",
        "updated_at",
        "last_accessed_at",
        "expires_at",
        "valid_from",
        "valid_until",
        "archived_at",
    ),
    "edge": ("valid_from", "valid_until"),
}

#: The last schema version whose write path could store a non-canonical
#: timestamp: the upgrade from it is the one that normalises them.
_LAST_NON_CANONICAL_SCHEMA_VERSION = 20

#: Tables whose structure the migrated database must share with a fresh one.
_SCHEMA_PARITY_TABLES = (
    "thought",
    "edge",
    "embedding",
    "action",
    "_metadata",
    "journal_entry",
    "extension_schema_versions",
)


def _verify_upgraded_db_script(
    db_path: Path,
    snapshot_path: Path,
    pre_snapshot_path: Path,
    pre_journal_state_path: Path,
) -> str:
    return textwrap.dedent(
        f"""
        import asyncio
        import datetime
        import json
        import re
        import subprocess
        import sys
        from pathlib import Path

        import aiosqlite

        from engrava import SqliteEngravaCore
        from engrava.cli.main import _export_db_to_jsonl
        from engrava.config import DreamingConfig, DreamingGates, EdgeCreationConfig
        from engrava.extensions.dreaming import DreamingExtension

        DB_PATH = r"{db_path}"
        SNAPSHOT_PATH = r"{snapshot_path}"
        PRE_SNAPSHOT_PATH = r"{pre_snapshot_path}"
        PRE_JOURNAL_STATE_PATH = r"{pre_journal_state_path}"
        POST_MIGRATION_SNAPSHOT_PATH = str(Path(SNAPSHOT_PATH).with_suffix(".post-migration.jsonl"))

        _SNAPSHOT_PRIMARY_KEYS = {_SNAPSHOT_PRIMARY_KEYS!r}
        _NORMALISED_TIMESTAMP_COLUMNS = {_NORMALISED_TIMESTAMP_COLUMNS!r}
        _LAST_NON_CANONICAL_SCHEMA_VERSION = {_LAST_NON_CANONICAL_SCHEMA_VERSION!r}
        _SCHEMA_PARITY_TABLES = {_SCHEMA_PARITY_TABLES!r}

        # What ``datetime.isoformat()`` writes for a UTC instant, and nothing
        # else. Written here rather than imported, so the check shares no code
        # with the build under test.
        _CANONICAL_SHAPE = re.compile(
            r"[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}T[0-9]{{2}}:[0-9]{{2}}:[0-9]{{2}}"
            r"(\\.[0-9]{{6}})?\\+00:00"
        )

        def _is_canonical(value) -> bool:
            if not isinstance(value, str) or _CANONICAL_SHAPE.fullmatch(value) is None:
                return False
            try:
                return datetime.datetime.fromisoformat(value).isoformat() == value
            except ValueError:
                return False

        def _utc_instant(value):
            parsed = datetime.datetime.fromisoformat(value)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=datetime.timezone.utc)
            return parsed.astimezone(datetime.timezone.utc)

        def _snapshot_schema_version(path) -> int:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    if row.get("_type") == "metadata":
                        return int(row["schema_version"])
            raise AssertionError(f"no metadata header in {{path}}")

        def _norm_sql(sql: str) -> str:
            return re.sub(r"\\s*([(),])\\s*", r"\\1", " ".join(sql.split()))

        async def _schema_shape(conn):
            \"\"\"Column definitions, foreign keys, index and trigger DDL, FTS config.

            Column order is not compared: ALTER ... ADD COLUMN can only append,
            and the fresh DDL declares migration-added columns last for that
            reason, but the order is not what this check is about.
            \"\"\"
            shape = {{}}
            for table in _SCHEMA_PARITY_TABLES:
                cursor = await conn.execute(f"PRAGMA table_info({{table}})")
                shape["columns", table] = sorted(
                    (str(r[1]), str(r[2]), int(r[3]), None if r[4] is None else str(r[4]),
                     int(r[5]))
                    for r in await cursor.fetchall()
                )
                cursor = await conn.execute(f"PRAGMA foreign_key_list({{table}})")
                shape["foreign keys", table] = sorted(
                    (str(r[2]), str(r[3]), str(r[4]), str(r[6])) for r in await cursor.fetchall()
                )
            cursor = await conn.execute(
                "SELECT type, name, sql FROM sqlite_master "
                "WHERE type IN ('index', 'trigger') AND sql IS NOT NULL"
            )
            shape["indexes and triggers"] = sorted(
                (str(r[0]), str(r[1]), _norm_sql(str(r[2]))) for r in await cursor.fetchall()
            )
            cursor = await conn.execute("SELECT sql FROM sqlite_master WHERE name = 'thought_fts'")
            shape["fts"] = [_norm_sql(str(r[0])) for r in await cursor.fetchall()]
            return shape

        async def _assert_schema_equals_a_fresh_one(conn) -> None:
            fresh = await aiosqlite.connect(":memory:")
            try:
                await SqliteEngravaCore(fresh).ensure_schema()
                expected = await _schema_shape(fresh)
            finally:
                await fresh.close()
            actual = await _schema_shape(conn)
            differing = sorted(str(k) for k in expected.keys() | actual.keys()
                               if expected.get(k) != actual.get(k))
            if differing:
                raise AssertionError(
                    f"migrated schema differs from a fresh bootstrap in: {{differing}}"
                )

        async def _assert_timestamps_canonical(conn, pre_records, pre_version) -> None:
            \"\"\"Every stored value in a normalised column is canonical after the upgrade.

            From a release whose write path stored values as written, the fixture
            planted a non-canonical value in every such column; that is checked
            too, so the canonical check cannot pass on a fixture with nothing to
            normalise.
            \"\"\"
            for table, columns in _NORMALISED_TIMESTAMP_COLUMNS.items():
                key = _SNAPSHOT_PRIMARY_KEYS[table]
                for column in columns:
                    cursor = await conn.execute(
                        f"SELECT {{key}}, {{column}} FROM {{table}} WHERE {{column}} IS NOT NULL"
                    )
                    offenders = [
                        (row[0], row[1])
                        for row in await cursor.fetchall()
                        if not _is_canonical(row[1])
                    ]
                    if offenders:
                        raise AssertionError(
                            f"{{table}}.{{column}} holds non-canonical timestamps after "
                            f"the upgrade: {{offenders!r}}"
                        )
                    if pre_version > _LAST_NON_CANONICAL_SCHEMA_VERSION:
                        continue
                    planted = [
                        fields[column]
                        for fields in pre_records[table].values()
                        if fields.get(column) is not None and not _is_canonical(fields[column])
                    ]
                    if not planted:
                        raise AssertionError(
                            f"the fixture left no non-canonical value in {{table}}.{{column}} "
                            "before the upgrade, so the check above proves nothing"
                        )

        def _load_snapshot_records(path):
            \"\"\"Return ``{{table: {{record_id: fields}}}}`` from a snapshot JSONL file.\"\"\"
            records = {{table: {{}} for table in _SNAPSHOT_PRIMARY_KEYS}}
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    table = row.get("_type")
                    key_field = _SNAPSHOT_PRIMARY_KEYS.get(table)
                    if key_field is None:
                        continue
                    data = row["data"]
                    records[table][data[key_field]] = data
            return records

        def _assert_content_preserved(pre_path, post_path) -> None:
            \"\"\"Every pre-upgrade record must still be present, byte-identical
            on every field the pre-upgrade snapshot itself declared.

            A field that only exists in the post-upgrade snapshot (e.g. a
            migration-added column with a default value) is not compared --
            the pre-upgrade snapshot could not have declared an opinion about
            it. This asserts preservation, not that nothing was ever added.

            The one exception is a timestamp column the upgrade normalises: its
            text may change into the canonical form, but it must still name
            the same instant.
            \"\"\"
            pre = _load_snapshot_records(pre_path)
            post = _load_snapshot_records(post_path)
            for table, pre_rows in pre.items():
                if not pre_rows:
                    continue
                post_rows = post[table]
                normalised = _NORMALISED_TIMESTAMP_COLUMNS.get(table, ())
                for record_id, pre_fields in pre_rows.items():
                    if record_id not in post_rows:
                        raise AssertionError(
                            f"{{table}} {{record_id}} present before the upgrade "
                            "is missing after it"
                        )
                    post_fields = post_rows[record_id]
                    for field, pre_value in pre_fields.items():
                        post_value = post_fields.get(field)
                        if post_value == pre_value:
                            continue
                        if (
                            field in normalised
                            and isinstance(pre_value, str)
                            and _is_canonical(post_value)
                            and _utc_instant(pre_value) == _utc_instant(post_value)
                        ):
                            continue
                        raise AssertionError(
                            f"{{table}} {{record_id}} field {{field!r}} changed "
                            f"across the upgrade: {{pre_value!r}} -> {{post_value!r}}"
                        )

        async def verify_data() -> None:
            # aiosqlite runs its connection on a NON-daemon thread that only
            # stops once close() sends it the stop sentinel. Anything raising
            # between connect() and close() therefore leaves the interpreter
            # unable to exit: the process hangs at shutdown instead of
            # reporting the failure. The finally: block is what turns a broken
            # assertion in here back into a fast, readable error.
            conn = await aiosqlite.connect(DB_PATH)
            try:
                conn.row_factory = aiosqlite.Row
                # journal_enabled=True only to read the chain via store.journal
                # below (verify_journal() itself works regardless) -- it does
                # not change what the migration above already wrote.
                store = SqliteEngravaCore(conn, journal_enabled=True)
                await store.ensure_schema()

                pre_records = _load_snapshot_records(PRE_SNAPSHOT_PATH)
                for table, pre_rows in pre_records.items():
                    cursor = await conn.execute(f"SELECT COUNT(*) FROM {{table}}")
                    (count,) = await cursor.fetchone()
                    if count != len(pre_rows):
                        raise AssertionError(
                            f"{{table}} row count changed across the upgrade: "
                            f"{{len(pre_rows)}} -> {{count}}"
                        )
                await _assert_timestamps_canonical(
                    conn, pre_records, _snapshot_schema_version(PRE_SNAPSHOT_PATH)
                )
                await _assert_schema_equals_a_fresh_one(conn)

                metrics = await store.metrics()
                if metrics.thoughts.total < 6:
                    raise AssertionError(
                        f"expected >= 6 thoughts after upgrade, got {{metrics.thoughts.total}}"
                    )
                if metrics.edges.total < 1:
                    raise AssertionError(
                        f"expected >= 1 edge after upgrade, got {{metrics.edges.total}}"
                    )

                fts_results = await store.search_fts("Upgrade")
                if not fts_results:
                    raise AssertionError("expected FTS results after upgrade")

                # Captured here -- after the migration ensure_schema() just ran,
                # before dreaming consolidation below deliberately changes the
                # store -- so this is a clean "did the migration itself
                # preserve everything" snapshot, not confounded by later,
                # unrelated writes.
                await _export_db_to_jsonl(conn, Path(POST_MIGRATION_SNAPSHOT_PATH))

                post_integrity = await store.verify_journal()
                with open(PRE_JOURNAL_STATE_PATH, encoding="utf-8") as f:
                    pre_journal_state = json.load(f)
                if not post_integrity.valid:
                    raise AssertionError(
                        f"journal chain no longer verifies after upgrade: "
                        f"{{post_integrity.error_message}}"
                    )
                if post_integrity.entries_checked != pre_journal_state["entries_checked"]:
                    raise AssertionError(
                        "journal entry count changed across the upgrade: "
                        f"{{pre_journal_state['entries_checked']}} -> "
                        f"{{post_integrity.entries_checked}}"
                    )
                if "last_entry_hash" in pre_journal_state and store.journal is not None:
                    post_entries = await store.journal.get_entries(limit=1000)
                    post_last = max(post_entries, key=lambda e: e.sequence_number)
                    if post_last.sequence_number != pre_journal_state["last_sequence_number"]:
                        raise AssertionError(
                            "journal chain tail sequence number changed across "
                            f"the upgrade: {{pre_journal_state['last_sequence_number']}} -> "
                            f"{{post_last.sequence_number}}"
                        )
                    if post_last.entry_hash != pre_journal_state["last_entry_hash"]:
                        raise AssertionError(
                            "journal chain tail hash changed across the upgrade "
                            "-- same sequence number, different chain"
                        )

                dreaming = DreamingExtension(
                    config=DreamingConfig(
                        enabled=True,
                        promote_threshold=0.0,
                        gates=DreamingGates(
                            min_confirmations=0,
                            min_age_cycles=0,
                            allow_zero_confirmation=True,
                            enable_reflections=False,
                        ),
                        edges=EdgeCreationConfig(enabled=False),
                    )
                )
                await dreaming.run_consolidation(store, current_cycle=10)
            finally:
                await conn.close()

        asyncio.run(verify_data())

        _assert_content_preserved(PRE_SNAPSHOT_PATH, POST_MIGRATION_SNAPSHOT_PATH)

        snapshot_cmd = [
            sys.executable,
            "-m",
            "engrava.cli.main",
            "--db",
            DB_PATH,
            "snapshot",
            "-o",
            SNAPSHOT_PATH,
        ]
        snapshot = subprocess.run(snapshot_cmd, check=True, capture_output=True, text=True)
        if not SNAPSHOT_PATH:
            raise AssertionError("snapshot path missing")
        if "Exported" not in snapshot.stdout:
            raise AssertionError(f"unexpected snapshot output: {{snapshot.stdout!r}}")

        gc_cmd = [sys.executable, "-m", "engrava.cli.main", "--db", DB_PATH, "gc"]
        gc_result = subprocess.run(gc_cmd, check=True, capture_output=True, text=True)
        if "Collected" not in gc_result.stdout and "No archived" not in gc_result.stdout:
            raise AssertionError(f"unexpected gc output: {{gc_result.stdout!r}}")

        migrate_cmd = [sys.executable, "-m", "engrava.cli.main", "--db", DB_PATH, "migrate"]
        migrate_result = subprocess.run(migrate_cmd, check=True, capture_output=True, text=True)
        if "Schema up to date" not in migrate_result.stdout:
            raise AssertionError(f"unexpected migrate output: {{migrate_result.stdout!r}}")
        """  # noqa: S608 - a generated test script; it interpolates only this module's constants and temp paths
    )


def verify_upgraded_db(
    python_executable: Path,
    db_path: Path,
    snapshot_path: Path,
    pre_snapshot_path: Path,
    pre_journal_state_path: Path,
) -> None:
    """Verify upgraded DB behavior using the installed target package.

    Checks the post-upgrade counts/FTS/CLI behaviour the same as before, and
    also diffs the migration's own effect against ``pre_snapshot_path`` /
    ``pre_journal_state_path`` (captured by :func:`populate_fixture_db`
    while still on the FROM release): every pre-upgrade thought, edge,
    embedding and action must still be present with its pre-upgrade field
    values intact, and the journal chain must still verify with the same
    entry count and the same chain tail.
    """
    run_command(
        [
            str(python_executable),
            "-c",
            _verify_upgraded_db_script(
                db_path, snapshot_path, pre_snapshot_path, pre_journal_state_path
            ),
        ]
    )


def run_upgrade_path(
    *,
    from_spec: str,
    to_spec: str,
    repository_root: Path,
    from_editable: bool,
    to_editable: bool,
    db_path: Path,
    snapshot_path: Path,
    pre_snapshot_path: Path,
    pre_journal_state_path: Path,
) -> None:
    """Run an end-to-end upgrade validation in an isolated virtual environment."""
    with tempfile.TemporaryDirectory(prefix="engrava-upgrade-") as temp_dir:
        venv_dir = Path(temp_dir) / "venv"
        python_executable = create_venv(venv_dir)
        install_package(
            python_executable,
            from_spec,
            editable=from_editable,
            cwd=repository_root,
        )
        populate_fixture_db(python_executable, db_path, pre_snapshot_path, pre_journal_state_path)
        install_package(
            python_executable,
            to_spec,
            editable=to_editable,
            cwd=repository_root,
            # See install_package's docstring: the candidate build can carry
            # the same version number as the FROM release until the release
            # pipeline itself bumps it, so this must force the reinstall
            # rather than let pip treat a version match as a no-op.
            force_reinstall=True,
        )
        verify_upgraded_db(
            python_executable, db_path, snapshot_path, pre_snapshot_path, pre_journal_state_path
        )
