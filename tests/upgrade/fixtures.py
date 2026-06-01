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


def run_command(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run a subprocess command and fail loudly on non-zero exit."""
    return subprocess.run(  # noqa: S603 — trusted test fixture invocation
        command,
        cwd=str(cwd) if cwd is not None else None,
        check=True,
        capture_output=True,
        text=True,
    )


def create_venv(venv_dir: Path) -> Path:
    """Create an isolated virtual environment and return its Python executable."""
    run_command([sys.executable, "-m", "venv", str(venv_dir)])
    python_executable = venv_python_path(venv_dir)
    run_command([str(python_executable), "-m", "pip", "install", "--upgrade", "pip"])
    return python_executable


def install_package(
    python_executable: Path,
    package_spec: str,
    *,
    editable: bool,
    cwd: Path | None = None,
) -> None:
    """Install a package spec into the given virtual environment."""
    command = [str(python_executable), "-m", "pip", "install"]
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
            conn = await aiosqlite.connect(DB_PATH)
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
            conn = await aiosqlite.connect(DB_PATH)
            conn.row_factory = aiosqlite.Row
            store = SqliteEngravaCore(conn)
            await store.ensure_schema()

            info = await store.info()
            if info["thoughts"] < 6:
                raise AssertionError(
                    f"expected >= 6 thoughts after upgrade, got {{info['thoughts']}}"
                )
            if info["edges"] < 1:
                raise AssertionError(f"expected >= 1 edge after upgrade, got {{info['edges']}}")

            fts_results = await store.search_fts("Upgrade")
            if not fts_results:
                raise AssertionError("expected FTS results after upgrade")

            metrics = await store.metrics()
            if metrics.thoughts.total < 6:
                raise AssertionError("metrics() returned incoherent thought count")

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
