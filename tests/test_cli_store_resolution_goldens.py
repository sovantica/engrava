"""Golden stdout/exit-code coverage for the shared store-resolution refactor.

``remember`` / ``recall`` / ``link`` (see ``engrava.cli.memory_commands``) are
built on a new shared store-resolution helper
(:mod:`engrava.cli.store_resolution`). The refactor that introduced it must
not change what ``info``, ``verify``, ``query``, or ``gc`` do — those four
commands' own code was not touched by this change, but "I didn't touch the
code" is a claim about the diff, not about behaviour. This test makes it an
empirical one: the exact invocation matrix below was run against the
pre-refactor worktree (commit ``b97b209``, the base this branch was cut from)
via the identical ``CliRunner`` harness, its stdout and exit codes captured
into ``tests/data/bare_command_goldens.json`` after normalizing the two
sources of incidental noise (the per-run temp directory path, and the
wall-clock timestamps ``info``'s metrics snapshot embeds). Running the same
matrix here, against the current code, and normalizing the same way, must
reproduce that file byte-for-byte.

The two ``info`` entries are the deliberate exception: ``info`` used to print
one ``schema_version`` — the metrics snapshot's own shape version — under a
label an operator had every reason to read as the database's. Both entries
were regenerated once, on purpose, to capture ``info`` now naming the two
numbers apart (``metrics_schema_version`` / ``database_schema_version`` in
JSON; both spelled out in the text line), not to loosen this test's guarantee
against an *unintended* change from the store-resolution refactor itself.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

import aiosqlite
from click.testing import CliRunner

from engrava import (
    EdgeRecord,
    EdgeType,
    LifecycleStatus,
    Priority,
    SqliteEngravaCore,
    ThoughtRecord,
    ThoughtType,
)
from engrava.cli.main import cli

if TYPE_CHECKING:
    from collections.abc import Sequence

_GOLDEN_PATH = Path(__file__).parent / "data" / "bare_command_goldens.json"

_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}T[0-9:.]+\+00:00")
_SNAPSHOT_TS_RE = re.compile(r'"snapshot_timestamp": [0-9.]+')


def _build_populated_db(db_path: Path) -> None:
    """Build the exact two-thought, one-edge database the goldens were captured against."""

    async def _setup() -> None:
        # Everything from schema setup through the commit runs under conn, so
        # it all belongs inside the try/finally below -- a failure partway
        # through (schema setup, either create call) would otherwise leak the
        # connection's non-daemon worker thread instead of closing it.
        conn = await aiosqlite.connect(str(db_path))
        try:
            conn.row_factory = aiosqlite.Row
            store = SqliteEngravaCore(conn, journal_enabled=True)
            await store.ensure_schema()
            for i, tid in enumerate(("t-old-0", "t-old-1")):
                await store.create_thought(
                    ThoughtRecord(
                        thought_id=tid,
                        essence=f"Essence for {tid}",
                        content=f"Content for {tid}",
                        thought_type=ThoughtType.OBSERVATION,
                        source="test",
                        lifecycle_status=LifecycleStatus.ACTIVE,
                        priority=Priority.P2,
                        created_cycle=i + 1,
                        updated_cycle=i + 1,
                    )
                )
            await store.create_edge(
                EdgeRecord(
                    edge_id="edge-001",
                    from_thought_id="t-old-0",
                    to_thought_id="t-old-1",
                    edge_type=EdgeType.ASSOCIATED,
                    weight=0.9,
                    created_cycle=1,
                )
            )
            await conn.commit()
        finally:
            await conn.close()

    asyncio.run(_setup())


def _normalize(args: Sequence[str], output: str, workdir: Path) -> tuple[list[str], str]:
    """Replace the two sources of incidental, run-to-run noise.

    Args:
        args: The raw CLI argv used for one invocation.
        output: That invocation's captured combined stdout.
        workdir: The temp directory the paths in ``args``/``output`` are under.

    Returns:
        ``(normalized_args, normalized_output)``, directly comparable across
        two different runs (and two different temp directories).

    """
    normalized_args = [a.replace(str(workdir), "WORKDIR") for a in args]
    normalized_output = output.replace(str(workdir), "WORKDIR")
    normalized_output = _SNAPSHOT_TS_RE.sub('"snapshot_timestamp": 0.0', normalized_output)
    normalized_output = _TIMESTAMP_RE.sub("1970-01-01T00:00:00.000000+00:00", normalized_output)
    return normalized_args, normalized_output


def _invocation_matrix(db: Path, missing: Path) -> list[list[str]]:
    return [
        ["--db", str(db), "info"],
        ["--db", str(db), "--format", "json", "info"],
        ["--db", str(missing), "info"],
        ["--db", str(db), "verify"],
        ["--db", str(db), "--format", "json", "verify"],
        ["--db", str(missing), "verify"],
        ["--db", str(db), "query", "SELECT thought_id, essence FROM thought ORDER BY thought_id"],
        ["--db", str(db), "--format", "json", "query", "COUNT thoughts WHERE priority = 'P2'"],
        ["--db", str(db), "query", "FIND thoughts WHERE lifecycle_status = 'ACTIVE'"],
        ["--db", str(missing), "query", "SELECT 1"],
        ["--db", str(db), "gc", "--dry-run"],
        ["--db", str(missing), "gc"],
    ]


def test_bare_commands_unchanged_by_the_store_resolution_refactor(tmp_path: Path) -> None:
    """``info`` / ``verify`` / ``query`` / ``gc`` reproduce the pre-refactor goldens exactly."""
    golden = json.loads(_GOLDEN_PATH.read_text(encoding="utf-8"))

    db = tmp_path / "populated.db"
    missing = tmp_path / "does-not-exist.db"
    _build_populated_db(db)

    runner = CliRunner()
    actual = []
    for args in _invocation_matrix(db, missing):
        result = runner.invoke(cli, args)
        norm_args, norm_output = _normalize(args, result.output, tmp_path)
        actual.append({"args": norm_args, "exit_code": result.exit_code, "output": norm_output})

    assert actual == golden


class TestConfigNeverReachesTheBareBuiltins:
    """A ``--config`` -- even a broken one -- must be a no-op for these four commands.

    ``info`` / ``verify`` / ``query`` / ``gc`` never call
    :func:`engrava.cli.memory_commands._resolve_for_command`; only
    ``remember`` / ``recall`` / ``link`` go through the ``--config``-loading
    gate that validates a caller-named file unconditionally (see
    ``resolve_store_target``'s docstring). The golden-replay matrix above
    proves these four commands are unchanged by the *store-resolution
    refactor*, but its own invocation matrix never supplies ``--config`` at
    all -- so it says nothing about the config-loading gate specifically.
    A future change that mistakenly wired these four into that same gate
    (routing their target through ``resolve_store_target`` instead of
    ``cfg.db_path`` directly) would leave every one of the existing focused
    CLI tests green, because none of them ever pass ``--config`` to one of
    these four commands. This closes that gap directly: a ``--config`` that
    cannot even parse must still leave each command's behaviour on its
    ``--db`` target byte-identical to not having named ``--config`` at all.
    """

    def test_a_broken_config_does_not_change_info_verify_query_gc(self, tmp_path: Path) -> None:
        db = tmp_path / "populated.db"
        missing = tmp_path / "does-not-exist.db"
        _build_populated_db(db)

        bad_config = tmp_path / "broken.yaml"
        bad_config.write_text("not: valid: yaml: [", encoding="utf-8")

        runner = CliRunner()
        for args in _invocation_matrix(db, missing):
            db_flag, db_value, *rest = args
            with_config_args = [db_flag, db_value, "--config", str(bad_config), *rest]

            bare_result = runner.invoke(cli, args)
            with_config_result = runner.invoke(cli, with_config_args)

            _, bare_output = _normalize(args, bare_result.output, tmp_path)
            _, with_config_output = _normalize(
                with_config_args, with_config_result.output, tmp_path
            )

            assert with_config_result.exit_code == bare_result.exit_code, args
            assert with_config_output == bare_output, args

    def test_a_missing_config_does_not_change_info_verify_query_gc(self, tmp_path: Path) -> None:
        db = tmp_path / "populated.db"
        missing = tmp_path / "does-not-exist.db"
        _build_populated_db(db)

        missing_config = tmp_path / "does-not-exist.yaml"

        runner = CliRunner()
        for args in _invocation_matrix(db, missing):
            db_flag, db_value, *rest = args
            with_config_args = [db_flag, db_value, "--config", str(missing_config), *rest]

            bare_result = runner.invoke(cli, args)
            with_config_result = runner.invoke(cli, with_config_args)

            _, bare_output = _normalize(args, bare_result.output, tmp_path)
            _, with_config_output = _normalize(
                with_config_args, with_config_result.output, tmp_path
            )

            assert with_config_result.exit_code == bare_result.exit_code, args
            assert with_config_output == bare_output, args
