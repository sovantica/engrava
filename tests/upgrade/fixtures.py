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
) -> None:
    """Install a package spec into the given virtual environment."""
    command = pip_command(python_executable, "install")
    if editable:
        command.extend(["-e", package_spec])
    else:
        command.append(package_spec)
    run_command(command, cwd=cwd)


def _populate_fixture_script(db_path: Path) -> str:
    return textwrap.dedent(
        f"""
        import asyncio
        import aiosqlite

        from engrava import (
            EdgeRecord,
            EdgeType,
            LifecycleStatus,
            Priority,
            SqliteEngravaCore,
            ThoughtRecord,
            ThoughtType,
        )

        DB_PATH = r"{db_path}"

        async def main() -> None:
            # Closed in a finally: for the same reason as the verifier. aiosqlite
            # runs the connection on a non-daemon thread that stops only on
            # close(), so a failure anywhere in the schema creation or the writes
            # below would leave this process unable to exit — a hang at shutdown
            # instead of the error that caused it.
            conn = await aiosqlite.connect(DB_PATH)
            try:
                conn.row_factory = aiosqlite.Row
                store = SqliteEngravaCore(conn)
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

                await conn.commit()
            finally:
                await conn.close()

        asyncio.run(main())
        """
    )


def populate_fixture_db(python_executable: Path, db_path: Path) -> None:
    """Create a representative fixture database using the installed package."""
    run_command([str(python_executable), "-c", _populate_fixture_script(db_path)])


def _verify_upgraded_db_script(db_path: Path, snapshot_path: Path) -> str:
    return textwrap.dedent(
        f"""
        import asyncio
        import json
        import subprocess
        import sys

        import aiosqlite

        from engrava import SqliteEngravaCore
        from engrava.config import DreamingConfig, DreamingGates, EdgeCreationConfig
        from engrava.extensions.dreaming import DreamingExtension

        DB_PATH = r"{db_path}"
        SNAPSHOT_PATH = r"{snapshot_path}"

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
                store = SqliteEngravaCore(conn)
                await store.ensure_schema()

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
        """
    )


def verify_upgraded_db(python_executable: Path, db_path: Path, snapshot_path: Path) -> None:
    """Verify upgraded DB behavior using the installed target package."""
    run_command([str(python_executable), "-c", _verify_upgraded_db_script(db_path, snapshot_path)])


def run_upgrade_path(
    *,
    from_spec: str,
    to_spec: str,
    repository_root: Path,
    from_editable: bool,
    to_editable: bool,
    db_path: Path,
    snapshot_path: Path,
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
        populate_fixture_db(python_executable, db_path)
        install_package(
            python_executable,
            to_spec,
            editable=to_editable,
            cwd=repository_root,
        )
        verify_upgraded_db(python_executable, db_path, snapshot_path)
