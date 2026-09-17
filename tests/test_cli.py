"""CLI smoke tests for ``engrava`` command.

Tests all subcommands against an in-memory (temp file) SQLite database.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, cast

import click
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable
from click.testing import CliRunner

from engrava.cli.config import EngravaCLIConfig
from engrava.cli.main import _close_quietly, _import_records_to_db, cli

# Literal SQL, never interpolated: a read-back that assembles its own query
# cannot be trusted to disagree with the schema the command wrote to.
_CORE_ID_QUERIES: tuple[tuple[str, str], ...] = (
    ("thought", "SELECT thought_id FROM thought"),
    ("edge", "SELECT edge_id FROM edge"),
    ("embedding", "SELECT embedding_id FROM embedding"),
)


def _missing_snapshot(tmp_path: Path) -> Path:
    """Build a ``--input`` path that is not there — the likeliest typo."""
    return tmp_path / "no-such-snapshot.jsonl"


def _directory_snapshot(tmp_path: Path) -> Path:
    """Build a ``--input`` path naming a directory — the backup folder, not the file."""
    directory = tmp_path / "backups"
    directory.mkdir()
    return directory


def _binary_snapshot(tmp_path: Path) -> Path:
    """Build a ``--input`` path that opens but holds no UTF-8 text — e.g. a database."""
    binary = tmp_path / "looks-like-a-snapshot.jsonl"
    binary.write_bytes(b"\xff\xfe\x00\x01not text at all\n")
    return binary


#: The unusable ``--input`` paths reachable from the command line, declared once
#: so the tests that assert the message and the test that asserts the error type
#: cannot come to disagree about what they are feeding the command. A read that
#: fails after a successful open is not reachable this way and is covered where
#: the iterator itself is tested.
_UNUSABLE_SNAPSHOTS: tuple[Callable[[Path], Path], ...] = (
    _missing_snapshot,
    _directory_snapshot,
    _binary_snapshot,
)


def _stored_core_ids(db_path: Path) -> dict[str, set[str]]:
    """Read the stored core-row ids back from the database file itself.

    The CLI owns and closes its own connection, so what it left behind is read
    over an independent one rather than taken from the command's report.

    Args:
        db_path: Path to the database the CLI operated on.

    Returns:
        Every stored id, per core table.

    """
    conn = sqlite3.connect(db_path)
    try:
        return {
            table: {str(row[0]) for row in conn.execute(query)} for table, query in _CORE_ID_QUERIES
        }
    finally:
        conn.close()


def _journal_entry_count(db_path: Path) -> int:
    """Read the number of rows currently in ``journal_entry``.

    A plain, independent connection, for the same reason as
    :func:`_stored_core_ids`: the CLI owns and closes its own connection, so
    what it actually left on disk is read back rather than inferred from the
    command's own report.

    Args:
        db_path: Path to the database the CLI operated on.

    Returns:
        The row count of ``journal_entry``.

    """
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("SELECT COUNT(*) FROM journal_entry").fetchone()
        return int(row[0])
    finally:
        conn.close()


def _journal_entry_deltas(db_path: Path, target_id: str) -> list[dict[str, object]]:
    """Read every ``journal_entry.delta`` recorded for one ``target_id``, in order.

    Same rationale as :func:`_journal_entry_count`: an independent connection
    reads back what the CLI actually left on disk, rather than trusting the
    command's own report.

    Args:
        db_path: Path to the database the CLI operated on.
        target_id: The ``journal_entry.target_id`` to filter on.

    Returns:
        Each matching entry's ``delta``, parsed from JSON, oldest first.

    """
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT delta FROM journal_entry WHERE target_id = ? ORDER BY sequence_number",
            (target_id,),
        ).fetchall()
        return [cast("dict[str, object]", json.loads(row[0])) for row in rows]
    finally:
        conn.close()


def _write_journalled_thoughts(db_path: Path, thought_ids: list[str]) -> None:
    """Create a database whose thoughts were each recorded through the journal.

    Unlike ``populated_db``, this builds the store with ``journal_enabled=True``
    so ``create_thought`` writes one hash-linked ``journal_entry`` row per
    thought -- the CLI itself never enables journaling (there is no CLI flag
    for it), so a store that already carries journal history has to be built
    directly against the domain API, exactly as it would be by an application
    embedding engrava as a library.

    Args:
        db_path: Path to create the database at. Must not already exist.
        thought_ids: The thought ids to create, in order.

    """
    import asyncio

    import aiosqlite

    from engrava import (
        LifecycleStatus,
        Priority,
        SqliteEngravaCore,
        ThoughtRecord,
        ThoughtType,
    )

    async def _setup() -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(str(db_path))
        conn.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(conn, journal_enabled=True)
        await store.ensure_schema()
        for i, thought_id in enumerate(thought_ids):
            await store.create_thought(
                ThoughtRecord(
                    thought_id=thought_id,
                    essence=f"Essence for {thought_id}",
                    content=f"Content for {thought_id}",
                    thought_type=ThoughtType.OBSERVATION,
                    source="test",
                    lifecycle_status=LifecycleStatus.ACTIVE,
                    priority=Priority.P2,
                    created_cycle=i + 1,
                    updated_cycle=i + 1,
                )
            )
        await conn.commit()
        await conn.close()

    asyncio.run(_setup())


def _write_journalled_thought_pair_with_edge(db_path: Path) -> None:
    """Create two journalled thoughts (``t-old-0``, ``t-old-1``) joined by a journalled edge.

    Same construction rationale as :func:`_write_journalled_thoughts`: the CLI
    has no flag to enable journaling, so a store that already carries journal
    history for both a thought mutation and an edge mutation has to be built
    directly against the domain API. This backs the cascade-collision test,
    where deleting one endpoint's thought row cascades an ``ON DELETE CASCADE``
    foreign-key delete onto the edge.

    Args:
        db_path: Path to create the database at. Must not already exist.

    """
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

    async def _setup() -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(str(db_path))
        conn.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(conn, journal_enabled=True)
        await store.ensure_schema()
        for i, thought_id in enumerate(("t-old-0", "t-old-1")):
            await store.create_thought(
                ThoughtRecord(
                    thought_id=thought_id,
                    essence=f"Essence for {thought_id}",
                    content=f"Content for {thought_id}",
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
        await conn.close()

    asyncio.run(_setup())


@pytest.fixture
def runner() -> CliRunner:
    """Create a Click test runner."""
    return CliRunner()


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Return a temporary DB path."""
    return tmp_path / "test.db"


@pytest.fixture
def populated_db(db_path: Path) -> Path:
    """Create a DB with schema and sample data, return its path."""
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

    async def _setup() -> None:
        conn = await aiosqlite.connect(str(db_path))
        conn.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(conn)
        await store.ensure_schema()

        for i in range(3):
            t = ThoughtRecord(
                thought_id=f"thought-{i:03d}",
                essence=f"Test thought {i}",
                content=f"Test thought number {i}",
                thought_type=ThoughtType.OBSERVATION,
                source="test",
                lifecycle_status=LifecycleStatus.ACTIVE if i < 2 else LifecycleStatus.ARCHIVED,
                priority=Priority.P2,
                created_cycle=i + 1,
                updated_cycle=i + 1,
            )
            await store.create_thought(t)
            await store.store_embedding(f"thought-{i:03d}", [float(i)] * 16)

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

    asyncio.run(_setup())
    return db_path


@pytest.fixture
def journalled_db(tmp_path: Path) -> Path:
    """A database with three thoughts, each recorded through the journal.

    Distinct from ``populated_db``, whose store is built without
    ``journal_enabled`` and so leaves ``journal_entry`` empty.
    """
    db_path = tmp_path / "journalled.db"
    _write_journalled_thoughts(db_path, ["t-old-0", "t-old-1", "t-old-2"])
    return db_path


@pytest.fixture
def journalled_db_with_edge(tmp_path: Path) -> Path:
    """Two journalled thoughts (``t-old-0``, ``t-old-1``) joined by a journalled edge.

    Distinct from ``journalled_db``, which has no edges. Backs the
    cascade-collision known-defect test: ``edge`` carries an ``ON DELETE
    CASCADE`` foreign key to ``thought`` on both endpoints, so replacing
    ``t-old-0`` also removes this edge.
    """
    db_path = tmp_path / "journalled_with_edge.db"
    _write_journalled_thought_pair_with_edge(db_path)
    return db_path


@pytest.fixture
def unrelated_snapshot(runner: CliRunner, tmp_path: Path) -> Path:
    """A one-thought snapshot with no relation to ``journalled_db`` or ``populated_db``.

    Built through the CLI (a fresh source database, then ``engrava
    snapshot``) so the snapshot line format is exactly what real restores
    consume, not a hand-assembled JSONL fixture.
    """
    source_db = tmp_path / "source.db"
    result = runner.invoke(cli, ["--db", str(source_db), "migrate"])
    assert result.exit_code == 0, result.output

    import asyncio

    import aiosqlite

    from engrava import LifecycleStatus, Priority, SqliteEngravaCore, ThoughtRecord, ThoughtType

    async def _seed() -> None:
        conn = await aiosqlite.connect(str(source_db))
        conn.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(conn)
        await store.create_thought(
            ThoughtRecord(
                thought_id="t-src",
                essence="Essence for t-src",
                content="Content for t-src",
                thought_type=ThoughtType.OBSERVATION,
                source="test",
                lifecycle_status=LifecycleStatus.ACTIVE,
                priority=Priority.P2,
                created_cycle=1,
                updated_cycle=1,
            )
        )
        await conn.commit()
        await conn.close()

    asyncio.run(_seed())

    snap = tmp_path / "unrelated-snapshot.jsonl"
    result = runner.invoke(cli, ["--db", str(source_db), "snapshot", "-o", str(snap)])
    assert result.exit_code == 0, result.output
    return snap


@pytest.fixture
def colliding_snapshot(runner: CliRunner, tmp_path: Path) -> Path:
    """A one-thought snapshot whose ID collides with ``t-old-0`` in ``journalled_db``
    and ``journalled_db_with_edge``.

    Built through the CLI, same rationale as ``unrelated_snapshot``: the
    snapshot line format must be exactly what a real restore consumes, not a
    hand-assembled JSONL fixture. The colliding thought's essence/content
    differ from the original so a stored-content check can tell replacement
    from a no-op.
    """
    source_db = tmp_path / "collide-source.db"
    result = runner.invoke(cli, ["--db", str(source_db), "migrate"])
    assert result.exit_code == 0, result.output

    import asyncio

    import aiosqlite

    from engrava import LifecycleStatus, Priority, SqliteEngravaCore, ThoughtRecord, ThoughtType

    async def _seed() -> None:
        conn = await aiosqlite.connect(str(source_db))
        conn.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(conn)
        await store.create_thought(
            ThoughtRecord(
                thought_id="t-old-0",
                essence="Replacement essence for t-old-0",
                content="Replacement content for t-old-0",
                thought_type=ThoughtType.OBSERVATION,
                source="test",
                lifecycle_status=LifecycleStatus.ACTIVE,
                priority=Priority.P2,
                created_cycle=1,
                updated_cycle=1,
            )
        )
        await conn.commit()
        await conn.close()

    asyncio.run(_seed())

    snap = tmp_path / "colliding-snapshot.jsonl"
    result = runner.invoke(cli, ["--db", str(source_db), "snapshot", "-o", str(snap)])
    assert result.exit_code == 0, result.output
    return snap


@pytest.fixture
def colliding_snapshot_with_a_leading_new_record(runner: CliRunner, tmp_path: Path) -> Path:
    """A two-thought snapshot: a brand-new record first, then one colliding with ``t-old-0``.

    Backs the case that ``TestRestoreRefusesCollisionAgainstAJournalledStore``'s
    own docstring describes -- the whole-transaction rollback discarding a
    record inserted *before* the one that collides -- which none of that
    class's other scenarios actually exercise: they each carry only the one
    colliding record, with nothing successfully written ahead of it.
    ``t-brand-new`` is created first in the source database, so
    ``SELECT * FROM thought`` (no ``ORDER BY``, see the ``snapshot`` command)
    returns it before ``t-old-0`` in the exported snapshot, and a plain
    ``INSERT`` accepts it with no complaint before reaching the colliding
    second record.
    """
    source_db = tmp_path / "collide-source-leading.db"
    result = runner.invoke(cli, ["--db", str(source_db), "migrate"])
    assert result.exit_code == 0, result.output

    import asyncio

    import aiosqlite

    from engrava import LifecycleStatus, Priority, SqliteEngravaCore, ThoughtRecord, ThoughtType

    async def _seed() -> None:
        conn = await aiosqlite.connect(str(source_db))
        conn.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(conn)
        await store.create_thought(
            ThoughtRecord(
                thought_id="t-brand-new",
                essence="Essence for t-brand-new",
                content="Content for t-brand-new",
                thought_type=ThoughtType.OBSERVATION,
                source="test",
                lifecycle_status=LifecycleStatus.ACTIVE,
                priority=Priority.P2,
                created_cycle=1,
                updated_cycle=1,
            )
        )
        await store.create_thought(
            ThoughtRecord(
                thought_id="t-old-0",
                essence="Replacement essence for t-old-0",
                content="Replacement content for t-old-0",
                thought_type=ThoughtType.OBSERVATION,
                source="test",
                lifecycle_status=LifecycleStatus.ACTIVE,
                priority=Priority.P2,
                created_cycle=2,
                updated_cycle=2,
            )
        )
        await conn.commit()
        await conn.close()

    asyncio.run(_seed())

    snap = tmp_path / "colliding-with-leading-snapshot.jsonl"
    result = runner.invoke(cli, ["--db", str(source_db), "snapshot", "-o", str(snap)])
    assert result.exit_code == 0, result.output
    return snap


@pytest.fixture
def colliding_snapshot_for_populated_db(runner: CliRunner, tmp_path: Path) -> Path:
    """A one-thought snapshot whose ID collides with ``thought-000`` in ``populated_db``.

    Backs the negative control for the journalled-merge collision gate:
    ``populated_db`` never enables journaling, so this collision must still
    succeed and still replace, exactly as restore always behaved -- the gate
    exists only once a journal has rows.
    """
    source_db = tmp_path / "collide-source-populated.db"
    result = runner.invoke(cli, ["--db", str(source_db), "migrate"])
    assert result.exit_code == 0, result.output

    import asyncio

    import aiosqlite

    from engrava import LifecycleStatus, Priority, SqliteEngravaCore, ThoughtRecord, ThoughtType

    async def _seed() -> None:
        conn = await aiosqlite.connect(str(source_db))
        conn.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(conn)
        await store.create_thought(
            ThoughtRecord(
                thought_id="thought-000",
                essence="Replacement essence for thought-000",
                content="Replacement content for thought-000",
                thought_type=ThoughtType.OBSERVATION,
                source="test",
                lifecycle_status=LifecycleStatus.ACTIVE,
                priority=Priority.P2,
                created_cycle=1,
                updated_cycle=1,
            )
        )
        await conn.commit()
        await conn.close()

    asyncio.run(_seed())

    snap = tmp_path / "colliding-populated-snapshot.jsonl"
    result = runner.invoke(cli, ["--db", str(source_db), "snapshot", "-o", str(snap)])
    assert result.exit_code == 0, result.output
    return snap


@pytest.fixture
def custom_mutation_db(tmp_path: Path) -> Path:
    """A store carrying one journal entry whose ``mutation_type`` is an arbitrary string.

    ``JournalWriter.append()`` validates nothing about ``mutation_type`` -- it
    is unconstrained ``TEXT`` -- so this writes ``CUSTOM_MUTATION``, a value no
    other part of the codebase ever emits, directly through the writer
    (bypassing ``create_thought``'s own journalling) to prove the gate keys
    off "``journal_entry`` has rows", never off a specific recognised
    ``mutation_type``.
    """
    import asyncio

    import aiosqlite

    from engrava import LifecycleStatus, Priority, SqliteEngravaCore, ThoughtRecord, ThoughtType
    from engrava.infrastructure.sqlite.journal_writer import JournalWriter

    db_path = tmp_path / "custom_mutation.db"

    async def _setup() -> None:
        conn = await aiosqlite.connect(str(db_path))
        conn.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(conn, journal_enabled=False)
        await store.ensure_schema()
        await store.create_thought(
            ThoughtRecord(
                thought_id="t-cm-0",
                essence="Essence for t-cm-0",
                content="Content for t-cm-0",
                thought_type=ThoughtType.OBSERVATION,
                source="test",
                lifecycle_status=LifecycleStatus.ACTIVE,
                priority=Priority.P2,
                created_cycle=1,
                updated_cycle=1,
            )
        )
        writer = JournalWriter(conn)
        await writer.append(
            mutation_type="CUSTOM_MUTATION",
            target_id="t-cm-0",
            delta={"before": None, "after": {"content": "Content for t-cm-0"}},
        )
        await conn.commit()
        await conn.close()

    asyncio.run(_setup())
    return db_path


@pytest.fixture
def colliding_snapshot_for_custom_mutation_db(runner: CliRunner, tmp_path: Path) -> Path:
    """A one-thought snapshot whose ID collides with ``t-cm-0`` in ``custom_mutation_db``."""
    source_db = tmp_path / "collide-source-cm.db"
    result = runner.invoke(cli, ["--db", str(source_db), "migrate"])
    assert result.exit_code == 0, result.output

    import asyncio

    import aiosqlite

    from engrava import LifecycleStatus, Priority, SqliteEngravaCore, ThoughtRecord, ThoughtType

    async def _seed() -> None:
        conn = await aiosqlite.connect(str(source_db))
        conn.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(conn)
        await store.create_thought(
            ThoughtRecord(
                thought_id="t-cm-0",
                essence="Replacement essence for t-cm-0",
                content="Replacement content for t-cm-0",
                thought_type=ThoughtType.OBSERVATION,
                source="test",
                lifecycle_status=LifecycleStatus.ACTIVE,
                priority=Priority.P2,
                created_cycle=1,
                updated_cycle=1,
            )
        )
        await conn.commit()
        await conn.close()

    asyncio.run(_seed())

    snap = tmp_path / "colliding-cm-snapshot.jsonl"
    result = runner.invoke(cli, ["--db", str(source_db), "snapshot", "-o", str(snap)])
    assert result.exit_code == 0, result.output
    return snap


@pytest.fixture
def identical_snapshot_of_journalled_db(
    runner: CliRunner, journalled_db: Path, tmp_path: Path
) -> Path:
    """A byte-for-byte snapshot of ``journalled_db`` itself, restorable back into it.

    Restoring this into ``journalled_db`` collides every thought on its own
    unchanged primary key and content. Used to show that even an identical
    merge is refused: with the old ``INSERT OR REPLACE`` behavior this
    silently changed every row's ``rowid`` (SQLite resolves the primary-key
    conflict by deleting then re-inserting, even when column values match).
    """
    snap = tmp_path / "identical-snapshot.jsonl"
    result = runner.invoke(cli, ["--db", str(journalled_db), "snapshot", "-o", str(snap)])
    assert result.exit_code == 0, result.output
    return snap


def _thought_rowids(db_path: Path) -> dict[str, int]:
    """Read each thought's implicit ``rowid``, keyed by ``thought_id``.

    An ``INSERT OR REPLACE`` that resolves a primary-key collision by
    deleting and re-inserting changes a row's ``rowid`` even when every
    column value is unchanged -- a plain ``SELECT thought_id, essence, ...``
    comparison would never see that, which is exactly the first fatal finding
    against the abandoned detection-based branch this gate replaces.
    """
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT thought_id, rowid FROM thought").fetchall()
        return {str(row[0]): int(row[1]) for row in rows}
    finally:
        conn.close()


def _thought_fts_match_count(db_path: Path, term: str) -> int:
    """Count real ``thought_fts`` index entries matching ``term`` via a ``MATCH`` query.

    ``SELECT COUNT(*) FROM thought_fts`` (no ``MATCH``) is USELESS here and
    must not be used: ``thought_fts`` is an external-content FTS5 table
    (``content='thought'``, ``content_rowid='rowid'``, schema_core.sql), and a
    bare, unfiltered ``COUNT(*)`` over an external-content table is satisfied
    by reading through to the row count of the backing ``thought`` table
    itself. It reports the number of thoughts, by construction, no matter how
    desynchronised the FTS shadow tables actually are -- it cannot observe
    this defect at all.

    A ``MATCH`` query, by contrast, scans the real inverted index and returns
    one hit per indexed entry, including a stale entry whose rowid no longer
    exists in ``thought`` at all. That is exactly the shape of the hazard this
    probe exists to see: ``PRAGMA recursive_triggers`` defaults to ``0`` and
    nothing under ``src/`` sets it, and SQLite only fires a table's ``DELETE``
    trigger for the row an ``INSERT OR REPLACE`` conflict removes when
    recursive triggers are enabled. So ``thought_fts_insert`` fires for the
    new rowid while ``thought_fts_delete`` never fires for the old one, and
    the stale entry for the removed rowid survives in the index, pointing at
    a row that no longer exists.

    Args:
        db_path: Path to the database to inspect.
        term: An FTS5 query term expected to match every thought under test
            (e.g. a word common to every fixture's ``essence``/``content``).

    Returns:
        The number of ``thought_fts`` rows matching ``term`` -- real index
        entries, not thoughts.

    """
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM thought_fts WHERE thought_fts MATCH ?", (term,)
        ).fetchone()
        return int(row[0])
    finally:
        conn.close()


@pytest.fixture
def fresh_edge_id_duplicate_triple_snapshot(runner: CliRunner, tmp_path: Path) -> Path:
    """A snapshot carrying only an edge record: a fresh ``edge_id``, ``edge-001``'s own triple.

    Isolated to just the edge line -- the thought rows that satisfied this
    edge's own foreign key when it was created in the source database are
    stripped back out -- so restoring this into ``journalled_db_with_edge``
    collides *only* on the edge table's composite
    ``UNIQUE(from_thought_id, to_thought_id, edge_type)`` (schema_core.sql),
    never on a thought primary key. A probe keyed on primary ids alone would
    never see this collision at all, since ``edge_id`` itself (``edge-999``)
    is brand new.
    """
    source_db = tmp_path / "edge-source.db"

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

    async def _seed() -> None:
        conn = await aiosqlite.connect(str(source_db))
        conn.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(conn)
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
                edge_id="edge-999",
                from_thought_id="t-old-0",
                to_thought_id="t-old-1",
                edge_type=EdgeType.ASSOCIATED,
                weight=0.5,
                created_cycle=1,
            )
        )
        await conn.commit()
        await conn.close()

    asyncio.run(_seed())

    full_snap = tmp_path / "edge-source-full.jsonl"
    result = runner.invoke(cli, ["--db", str(source_db), "snapshot", "-o", str(full_snap)])
    assert result.exit_code == 0, result.output

    edge_only_lines = [
        line
        for line in full_snap.read_text(encoding="utf-8").splitlines()
        if json.loads(line).get("_type") == "edge"
    ]
    assert len(edge_only_lines) == 1
    edge_only_snap = tmp_path / "edge-only.jsonl"
    edge_only_snap.write_text("\n".join(edge_only_lines) + "\n", encoding="utf-8")
    return edge_only_snap


class TestGlobalControls:
    """Tests for extension isolation and verbose logging controls."""

    def test_no_extensions_skips_cli_discovery_for_help(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _unexpected_discovery() -> list[object]:
            msg = "CLI extension discovery must stay disabled"
            raise AssertionError(msg)

        monkeypatch.setattr(
            "engrava.cli.main._discover_extension_commands",
            _unexpected_discovery,
        )

        result = runner.invoke(cli, ["--no-extensions", "--help"])

        assert result.exit_code == 0
        assert "--no-extensions" in result.output

    def test_disable_extensions_environment_variable_skips_cli_discovery(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _unexpected_discovery() -> list[object]:
            msg = "CLI extension discovery must stay disabled"
            raise AssertionError(msg)

        monkeypatch.setattr(
            "engrava.cli.main._discover_extension_commands",
            _unexpected_discovery,
        )

        result = runner.invoke(
            cli,
            ["--help"],
            env={"ENGRAVA_DISABLE_EXTENSIONS": "1"},
        )

        assert result.exit_code == 0

    def test_no_extensions_skips_mindql_discovery(
        self,
        runner: CliRunner,
        populated_db: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _unexpected_discovery() -> dict[str, object]:
            msg = "MindQL extension discovery must stay disabled"
            raise AssertionError(msg)

        monkeypatch.setattr(
            "engrava.cli.main._load_mindql_extensions",
            _unexpected_discovery,
        )

        result = runner.invoke(
            cli,
            [
                "--db",
                str(populated_db),
                "--no-extensions",
                "query",
                "SELECT thought_id FROM thought LIMIT 1",
            ],
        )

        assert result.exit_code == 0
        assert "thought-000" in result.output

    def test_verbose_emits_debug_logging(
        self,
        runner: CliRunner,
        populated_db: Path,
    ) -> None:
        records: list[logging.LogRecord] = []

        class _CaptureHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        capture_handler = _CaptureHandler()
        package_logger = logging.getLogger("engrava")
        package_logger.addHandler(capture_handler)
        try:
            result = runner.invoke(
                cli,
                ["--db", str(populated_db), "--verbose", "info"],
            )
        finally:
            package_logger.removeHandler(capture_handler)

        assert result.exit_code == 0
        assert any(
            record.levelno == logging.DEBUG and record.getMessage() == "Verbose logging enabled"
            for record in records
        )


class TestInfo:
    """Tests for ``engrava info``."""

    def test_info_table_format(self, runner: CliRunner, populated_db: Path) -> None:
        result = runner.invoke(cli, ["--db", str(populated_db), "info"])
        assert result.exit_code == 0
        assert "Thoughts: 3" in result.output
        assert "Edges: 1" in result.output

    def test_info_json_format(self, runner: CliRunner, populated_db: Path) -> None:
        result = runner.invoke(cli, ["--db", str(populated_db), "--format", "json", "info"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["thoughts"]["total"] == 3
        assert data["edges"]["total"] == 1
        assert data["schema_version"] == 2
        assert data["search_latency"]["sample_count"] == 0

    def test_info_missing_db(self, runner: CliRunner, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.db"
        result = runner.invoke(cli, ["--db", str(missing), "info"])
        assert result.exit_code != 0


class TestQuery:
    """Tests for ``engrava query``."""

    def test_query_select(self, runner: CliRunner, populated_db: Path) -> None:
        result = runner.invoke(
            cli,
            ["--db", str(populated_db), "query", "SELECT thought_id FROM thought"],
        )
        assert result.exit_code == 0
        assert "thought-000" in result.output

    def test_query_json(self, runner: CliRunner, populated_db: Path) -> None:
        result = runner.invoke(
            cli,
            [
                "--db",
                str(populated_db),
                "--format",
                "json",
                "query",
                "SELECT thought_id, content FROM thought LIMIT 1",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 1
        assert "thought_id" in data[0]

    def test_query_rejects_non_select_sql(self, runner: CliRunner, populated_db: Path) -> None:
        result = runner.invoke(
            cli,
            ["--db", str(populated_db), "query", "DELETE FROM thought"],
        )
        assert result.exit_code != 0

    def test_query_csv(self, runner: CliRunner, populated_db: Path) -> None:
        result = runner.invoke(
            cli,
            [
                "--db",
                str(populated_db),
                "--format",
                "csv",
                "query",
                "SELECT thought_id FROM thought LIMIT 2",
            ],
        )
        assert result.exit_code == 0
        assert "thought_id" in result.output
        assert "thought-000" in result.output

    def test_query_mindql_find(self, runner: CliRunner, populated_db: Path) -> None:
        result = runner.invoke(
            cli,
            ["--db", str(populated_db), "query", "FIND thoughts WHERE lifecycle_status = 'ACTIVE'"],
        )
        assert result.exit_code == 0
        assert "thought-000" in result.output

    def test_query_mindql_count(self, runner: CliRunner, populated_db: Path) -> None:
        result = runner.invoke(
            cli,
            [
                "--db",
                str(populated_db),
                "--format",
                "json",
                "query",
                "COUNT thoughts",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data[0]["count"] == 3


class TestSnapshot:
    """Tests for ``engrava snapshot``."""

    def test_snapshot_creates_file(
        self,
        runner: CliRunner,
        populated_db: Path,
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "snap.jsonl"
        result = runner.invoke(
            cli,
            ["--db", str(populated_db), "snapshot", "-o", str(out)],
        )
        assert result.exit_code == 0
        assert out.exists()
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) >= 3  # 3 thoughts minimum
        for line in lines:
            record = json.loads(line)
            assert "_type" in record

    def test_snapshot_default_path(self, runner: CliRunner, populated_db: Path) -> None:
        result = runner.invoke(cli, ["--db", str(populated_db), "snapshot"])
        assert result.exit_code == 0
        default_out = populated_db.with_suffix(".snapshot.jsonl")
        assert default_out.exists()

    def test_snapshot_invalid_service_name_is_clean_error(
        self,
        runner: CliRunner,
        db_path: Path,
    ) -> None:
        """An invalid --service value is a clean CLI error, never a traceback."""
        result = runner.invoke(
            cli,
            ["--db", str(db_path), "snapshot", "--service", "../escape"],
        )
        assert result.exit_code != 0
        assert "Invalid --service value" in result.output
        # It was handled as a ClickException (clean exit), not an uncaught error.
        assert isinstance(result.exception, SystemExit)

    @pytest.mark.parametrize("bad_service", ["", "   "])
    def test_snapshot_empty_service_name_is_clean_error(
        self,
        runner: CliRunner,
        db_path: Path,
        bad_service: str,
    ) -> None:
        """An explicit empty/whitespace --service is validated, not routed away.

        Such a value is falsy, so it must be rejected on `is not None` rather
        than a truthiness check that would fall through to the single-database
        path.
        """
        result = runner.invoke(
            cli,
            ["--db", str(db_path), "snapshot", "--service", bad_service],
        )
        assert result.exit_code != 0
        assert "Invalid --service value" in result.output
        assert isinstance(result.exception, SystemExit)


class TestRestore:
    """Tests for ``engrava restore``."""

    def test_restore_roundtrip(self, runner: CliRunner, populated_db: Path, tmp_path: Path) -> None:
        snap = tmp_path / "snap.jsonl"
        runner.invoke(cli, ["--db", str(populated_db), "snapshot", "-o", str(snap)])

        new_db = tmp_path / "restored.db"
        result = runner.invoke(
            cli,
            ["--db", str(new_db), "restore", "-i", str(snap)],
        )
        assert result.exit_code == 0
        assert "Restored" in result.output

        # Verify restored data
        check = runner.invoke(
            cli,
            ["--db", str(new_db), "--format", "json", "info"],
        )
        data = json.loads(check.output)
        assert data["thoughts"]["total"] == 3

    def test_restore_with_clear(
        self,
        runner: CliRunner,
        populated_db: Path,
        tmp_path: Path,
    ) -> None:
        snap = tmp_path / "snap.jsonl"
        runner.invoke(cli, ["--db", str(populated_db), "snapshot", "-o", str(snap)])

        result = runner.invoke(
            cli,
            ["--db", str(populated_db), "restore", "-i", str(snap), "--clear"],
        )
        assert result.exit_code == 0

    def test_restore_with_clear_on_empty_journal_reports_zero_discarded(
        self,
        runner: CliRunner,
        populated_db: Path,
        tmp_path: Path,
    ) -> None:
        """``--clear`` on a store whose journal was never enabled discards nothing.

        ``populated_db`` never enables journaling, so ``journal_entry`` starts
        (and stays) empty. The count printed must say so honestly rather than
        the CLI staying silent about a table it now also clears.
        """
        assert _journal_entry_count(populated_db) == 0
        snap = tmp_path / "snap.jsonl"
        runner.invoke(cli, ["--db", str(populated_db), "snapshot", "-o", str(snap)])

        result = runner.invoke(
            cli,
            ["--db", str(populated_db), "restore", "-i", str(snap), "--clear"],
        )

        assert result.exit_code == 0
        assert _journal_entry_count(populated_db) == 0
        assert "Discarded 0 journal entries" in result.output

    def test_restore_with_clear_discards_the_journal(
        self,
        runner: CliRunner,
        journalled_db: Path,
        unrelated_snapshot: Path,
    ) -> None:
        """``--clear`` from an unrelated snapshot must not leave a journal that
        describes thoughts the clear just removed.

        Before the fix, ``journal_entry`` was not in the table list ``--clear``
        iterates and no foreign key reaches it, so it survived untouched:
        ``thought`` held only the restored ``t-src`` row while ``journal_entry``
        kept all three ``t-old-*`` entries, and ``verify_journal()`` reported
        that mismatched chain as ``valid``.
        """
        assert _journal_entry_count(journalled_db) == 3

        result = runner.invoke(
            cli,
            ["--db", str(journalled_db), "restore", "-i", str(unrelated_snapshot), "--clear"],
        )

        assert result.exit_code == 0, result.output
        assert _stored_core_ids(journalled_db)["thought"] == {"t-src"}
        assert _journal_entry_count(journalled_db) == 0
        assert "Discarded 3 journal entries" in result.output

        verify_result = runner.invoke(
            cli,
            ["--db", str(journalled_db), "--format", "json", "verify"],
        )
        assert verify_result.exit_code == 0, verify_result.output
        verify_data = json.loads(verify_result.output)
        assert verify_data["valid"] is True
        assert verify_data["entries_checked"] == 0

    def test_restore_without_clear_leaves_the_journal_untouched_for_disjoint_ids(
        self,
        runner: CliRunner,
        journalled_db: Path,
        unrelated_snapshot: Path,
    ) -> None:
        """A merge restore (no ``--clear``) leaves the journal alone when IDs don't collide.

        ``unrelated_snapshot`` carries a single thought (``t-src``) whose ID is
        disjoint from every ID already in ``journalled_db``, so this only
        establishes that ``journal_entry`` survives untouched in that disjoint
        case -- it keeps describing exactly the three pre-existing thoughts and
        nothing about the merged-in ``t-src``. It does **not** establish that a
        merge is safe for a *colliding* ID: restore inserts every record with
        ``INSERT OR REPLACE``, so an incoming ID that matches an existing
        journalled thought, edge, or action instead replaces (or, through a
        cascading foreign-key delete, removes) that row while its journal
        entry is left describing content that is no longer there. See the
        known-defect tests immediately below for that case.
        """
        assert _journal_entry_count(journalled_db) == 3

        result = runner.invoke(
            cli,
            ["--db", str(journalled_db), "restore", "-i", str(unrelated_snapshot)],
        )

        assert result.exit_code == 0, result.output
        assert "Discarded" not in result.output
        assert _journal_entry_count(journalled_db) == 3
        assert _stored_core_ids(journalled_db)["thought"] == {
            "t-old-0",
            "t-old-1",
            "t-old-2",
            "t-src",
        }

        verify_result = runner.invoke(
            cli,
            ["--db", str(journalled_db), "--format", "json", "verify"],
        )
        assert verify_result.exit_code == 0, verify_result.output
        verify_data = json.loads(verify_result.output)
        assert verify_data["valid"] is True
        assert verify_data["entries_checked"] == 3

    def test_restore_service_with_clear_discards_the_journal(
        self,
        runner: CliRunner,
        tmp_path: Path,
        unrelated_snapshot: Path,
    ) -> None:
        """The ``--service`` restore path clears the journal exactly like the
        single-database path.

        Both branches route through the same ``_import_records_to_db``, but
        that is an implementation detail this test does not assume -- it
        drives the ``--service`` restore through the CLI and inspects the
        resulting service database file directly.
        """
        services_dir = tmp_path / "services"
        service_db = services_dir / "svc.db"
        _write_journalled_thoughts(service_db, ["t-old-0", "t-old-1", "t-old-2"])
        assert _journal_entry_count(service_db) == 3

        result = runner.invoke(
            cli,
            [
                "--db",
                str(services_dir / "ignored.db"),
                "restore",
                "-i",
                str(unrelated_snapshot),
                "--service",
                "svc",
                "--clear",
            ],
        )

        assert result.exit_code == 0, result.output
        assert _stored_core_ids(service_db)["thought"] == {"t-src"}
        assert _journal_entry_count(service_db) == 0
        assert "Discarded 3 journal entries" in result.output

    def test_restore_service_refuses_collision_against_a_journalled_store(
        self,
        runner: CliRunner,
        tmp_path: Path,
        colliding_snapshot: Path,
    ) -> None:
        """The ``--service`` restore path is defended by the journalled-merge collision gate too.

        Every collision and override test elsewhere in this module exercises
        only the single-database restore path. ``_restore_service_snapshot``
        forwards ``orphan_journal_entries`` to the same
        ``_import_records_to_db`` the single-database path uses, but nothing
        pinned that a default ``--service`` restore into a journalled target
        is actually refused rather than merging silently.
        ``colliding_snapshot`` carries a thought whose ID (``t-old-0``)
        matches one already in the service database, with different
        essence/content.
        """
        services_dir = tmp_path / "services"
        service_db = services_dir / "svc.db"
        _write_journalled_thoughts(service_db, ["t-old-0", "t-old-1", "t-old-2"])
        assert _journal_entry_count(service_db) == 3

        result = runner.invoke(
            cli,
            [
                "--db",
                str(services_dir / "ignored.db"),
                "restore",
                "-i",
                str(colliding_snapshot),
                "--service",
                "svc",
            ],
        )

        assert result.exit_code != 0
        assert "--orphan-journal-entries" in result.output
        assert "journal_entry" in result.output

        assert _stored_core_ids(service_db)["thought"] == {"t-old-0", "t-old-1", "t-old-2"}
        conn = sqlite3.connect(service_db)
        try:
            live_content = conn.execute(
                "SELECT content FROM thought WHERE thought_id = ?",
                ("t-old-0",),
            ).fetchone()[0]
        finally:
            conn.close()
        assert live_content == "Content for t-old-0"
        assert _journal_entry_count(service_db) == 3

    def test_restore_service_orphan_journal_entries_overrides_the_gate(
        self,
        runner: CliRunner,
        tmp_path: Path,
        colliding_snapshot: Path,
    ) -> None:
        """``--orphan-journal-entries`` under ``--service`` restores the merge too.

        Complements the refusal above: the same override flag
        ``_restore_service_snapshot`` forwards must let this collision
        through and replace in service mode -- the flag's entire purpose --
        exactly as it does on the single-database path.
        """
        services_dir = tmp_path / "services"
        service_db = services_dir / "svc.db"
        _write_journalled_thoughts(service_db, ["t-old-0", "t-old-1", "t-old-2"])

        result = runner.invoke(
            cli,
            [
                "--db",
                str(services_dir / "ignored.db"),
                "restore",
                "-i",
                str(colliding_snapshot),
                "--service",
                "svc",
                "--orphan-journal-entries",
            ],
        )

        assert result.exit_code == 0, result.output
        conn = sqlite3.connect(service_db)
        try:
            live_content = conn.execute(
                "SELECT content FROM thought WHERE thought_id = ?",
                ("t-old-0",),
            ).fetchone()[0]
        finally:
            conn.close()
        assert live_content == "Replacement content for t-old-0"
        # The journal is left exactly as before: the override does not touch
        # journal_entry, it only changes which INSERT form is used.
        assert _journal_entry_count(service_db) == 3

    def test_restore_invalid_service_name_is_distinct_clean_error(
        self,
        runner: CliRunner,
        db_path: Path,
        tmp_path: Path,
    ) -> None:
        """An invalid --service value is a clean, distinct error on restore.

        It must not be mislabelled as an embedding-provider initialisation
        failure, and must not print a traceback. Validation happens before any
        snapshot file is read, so the input path need not exist.
        """
        snap = tmp_path / "missing.jsonl"
        result = runner.invoke(
            cli,
            ["--db", str(db_path), "restore", "-i", str(snap), "--service", "../escape"],
        )
        assert result.exit_code != 0
        assert "Invalid --service value" in result.output
        assert "embedding provider" not in result.output
        assert isinstance(result.exception, SystemExit)

    @pytest.mark.parametrize("bad_service", ["", "   "])
    def test_restore_empty_service_name_is_clean_error(
        self,
        runner: CliRunner,
        db_path: Path,
        tmp_path: Path,
        bad_service: str,
    ) -> None:
        """An explicit empty/whitespace --service on restore is rejected cleanly.

        Such a value is falsy; validating on `is not None` gives it the same
        clean ClickException as any other malformed name instead of silently
        falling through to the single-database path.
        """
        snap = tmp_path / "missing.jsonl"
        result = runner.invoke(
            cli,
            ["--db", str(db_path), "restore", "-i", str(snap), "--service", bad_service],
        )
        assert result.exit_code != 0
        assert "Invalid --service value" in result.output
        assert "embedding provider" not in result.output
        assert isinstance(result.exception, SystemExit)

    def test_restore_missing_input_file_is_clean_error(
        self,
        runner: CliRunner,
        db_path: Path,
        tmp_path: Path,
    ) -> None:
        """A ``-i`` path that is not there is a clean CLI error, never a traceback.

        Mistyping the snapshot path is the likeliest way to get this command
        wrong, so it is held to the same standard as an invalid ``--service``:
        a message naming the path and the option, not an ``OSError`` the user
        has to read a stack trace to understand.
        """
        missing = _missing_snapshot(tmp_path)

        result = runner.invoke(cli, ["--db", str(db_path), "restore", "-i", str(missing)])

        assert result.exit_code != 0
        assert str(missing) in result.output
        assert "--input" in result.output
        # Handled as a ClickException (clean exit), not an uncaught OSError.
        assert isinstance(result.exception, SystemExit)

    def test_restore_missing_input_file_in_service_mode_is_clean_error(
        self,
        runner: CliRunner,
        db_path: Path,
        tmp_path: Path,
    ) -> None:
        """The ``--service`` restore path rejects the same bad ``-i`` as cleanly.

        ``restore`` reaches its snapshot through two entry points — the
        single-database path and the service path — and the second one opens a
        service store first. Asserted here in its own right so neither can be
        clean only because the other is.
        """
        missing = _missing_snapshot(tmp_path)

        result = runner.invoke(
            cli,
            ["--db", str(db_path), "restore", "-i", str(missing), "--service", "svc"],
        )

        assert result.exit_code != 0
        assert str(missing) in result.output
        assert "--input" in result.output
        assert isinstance(result.exception, SystemExit)

    def test_restore_input_directory_is_clean_error(
        self,
        runner: CliRunner,
        db_path: Path,
        tmp_path: Path,
    ) -> None:
        """A ``-i`` path naming a directory is rejected the same clean way.

        Passing the backup *folder* instead of the file inside it fails at the
        same open, with a different ``OSError`` — so it is the same defect
        unless the guard covers the whole class, and it earns the same
        actionable message rather than only a tidy exit code.
        """
        directory = _directory_snapshot(tmp_path)

        result = runner.invoke(cli, ["--db", str(db_path), "restore", "-i", str(directory)])

        assert result.exit_code != 0
        assert str(directory) in result.output
        assert "--input" in result.output
        assert isinstance(result.exception, SystemExit)

    def test_restore_non_utf8_input_is_clean_error(
        self,
        runner: CliRunner,
        db_path: Path,
        tmp_path: Path,
    ) -> None:
        """A ``-i`` path that is not UTF-8 text is a clean CLI error too.

        Pointing ``restore`` at the database instead of at the snapshot opens
        successfully and then fails mid-read while decoding, which is a
        different failure from an unopenable path and needs its own guard.
        """
        binary = _binary_snapshot(tmp_path)

        result = runner.invoke(cli, ["--db", str(db_path), "restore", "-i", str(binary)])

        assert result.exit_code != 0
        assert str(binary) in result.output
        assert "UTF-8" in result.output
        assert "--input" in result.output
        assert isinstance(result.exception, SystemExit)

    @pytest.mark.parametrize(
        "service_args",
        [[], ["--service", "svc"]],
        ids=["single-db", "service"],
    )
    @pytest.mark.parametrize("make_input", _UNUSABLE_SNAPSHOTS)
    def test_restore_unusable_input_raises_the_typed_cli_error(
        self,
        runner: CliRunner,
        db_path: Path,
        tmp_path: Path,
        make_input: Callable[[Path], Path],
        service_args: list[str],
    ) -> None:
        """The guard raises ``ClickException`` itself, on either restore path.

        The tests above assert what the user sees, and Click renders every
        error it handles the same way — a usage error, an explicit ``sys.exit``
        and an abort all reach ``CliRunner`` as ``SystemExit``. Running with
        ``standalone_mode=False`` stops Click catching the exception, so the
        type is pinned here: that is what makes this a typed boundary rather
        than a tidy exit code.
        """
        bad_input = make_input(tmp_path)

        result = runner.invoke(
            cli,
            ["--db", str(db_path), "restore", "-i", str(bad_input), *service_args],
            standalone_mode=False,
        )

        assert isinstance(result.exception, click.ClickException), result.exception
        assert str(bad_input) in str(result.exception)

    def test_restore_missing_input_with_clear_keeps_every_stored_row(
        self,
        runner: CliRunner,
        populated_db: Path,
        tmp_path: Path,
    ) -> None:
        """``--clear`` must not empty the database when the snapshot is unusable.

        ``--clear`` deletes every core row inside the restore transaction,
        before the snapshot is opened. A failure at the open is therefore only
        harmless because the transaction rolls back, so the rows are re-read
        from SQLite here rather than trusted to the command's exit.
        """
        missing = tmp_path / "no-such-snapshot.jsonl"
        before = _stored_core_ids(populated_db)
        assert all(before.values()), f"corpus precondition failed: {before}"

        result = runner.invoke(
            cli,
            ["--db", str(populated_db), "restore", "-i", str(missing), "--clear"],
        )

        assert _stored_core_ids(populated_db) == before
        # Only once the stored rows are settled does the reported failure matter.
        assert result.exit_code != 0
        assert isinstance(result.exception, SystemExit)


class TestRestoreRefusesCollisionAgainstAJournalledStore:
    """A merge restore (no ``--clear``) refuses a collision once the target is journalled.

    Formerly ``TestRestoreWithoutClearKnownJournalCollisionDefects``: these
    same two scenarios used to pin the KNOWN DEFECT that ``INSERT OR REPLACE``
    silently replaced (or cascade-deleted) a journalled row, leaving its
    journal entry describing content that was no longer there while ``verify``
    kept reporting the chain valid. The journalled-merge collision gate in
    ``_import_records_to_db`` (``cli/main.py``) closes that gap: once the
    target's ``journal_entry`` table is non-empty, every incoming record is
    written with a plain ``INSERT`` instead, so SQLite itself refuses the
    collision. Nothing here still passes with the old ``INSERT OR REPLACE``
    behavior -- these tests must now show the refusal and an entirely
    untouched database, which is also the atomicity guarantee: the existing
    ``finally: await conn.rollback()`` in ``_import_records_to_db`` discards
    the whole transaction, including any record inserted before the one that
    collided.
    """

    def test_colliding_thought_id_is_refused_content_and_journal_untouched(
        self,
        runner: CliRunner,
        journalled_db: Path,
        colliding_snapshot: Path,
    ) -> None:
        """A colliding thought ID is refused; live content and the journal are both untouched.

        ``colliding_snapshot`` carries a thought whose ID (``t-old-0``) matches
        one already in ``journalled_db``, with different essence/content --
        exactly the mismatch a reviewer once reported as silently accepted.
        """
        assert _journal_entry_count(journalled_db) == 3
        before_deltas = _journal_entry_deltas(journalled_db, "t-old-0")
        assert len(before_deltas) == 1

        result = runner.invoke(
            cli,
            ["--db", str(journalled_db), "restore", "-i", str(colliding_snapshot)],
        )

        assert result.exit_code != 0
        assert "--orphan-journal-entries" in result.output
        assert "journal_entry" in result.output

        # Nothing was written: the live row keeps its original content ...
        assert _stored_core_ids(journalled_db)["thought"] == {"t-old-0", "t-old-1", "t-old-2"}
        conn = sqlite3.connect(journalled_db)
        try:
            live_content = conn.execute(
                "SELECT content FROM thought WHERE thought_id = ?",
                ("t-old-0",),
            ).fetchone()[0]
        finally:
            conn.close()
        assert live_content == "Content for t-old-0"

        # ... and the journal is exactly as it was.
        assert _journal_entry_count(journalled_db) == 3
        assert _journal_entry_deltas(journalled_db, "t-old-0") == before_deltas

        verify_result = runner.invoke(
            cli,
            ["--db", str(journalled_db), "--format", "json", "verify"],
        )
        assert verify_result.exit_code == 0, verify_result.output
        verify_data = json.loads(verify_result.output)
        assert verify_data["valid"] is True
        assert verify_data["entries_checked"] == 3

    def test_colliding_thought_id_is_refused_before_any_cascade_can_fire(
        self,
        runner: CliRunner,
        journalled_db_with_edge: Path,
        colliding_snapshot: Path,
    ) -> None:
        """A colliding thought ID is refused before its cascading edge delete can fire.

        ``edge`` carries an ``ON DELETE CASCADE`` foreign key to ``thought`` on
        both endpoints (schema_core.sql). Under the old ``INSERT OR REPLACE``
        behavior, resolving the primary-key collision on ``t-old-0`` deleted
        the pre-existing row first, and with ``PRAGMA foreign_keys = ON``
        (always on for restore, see ``_open_db``) that cascaded onto
        ``edge-001``. A plain ``INSERT`` has no delete half, so that cascade
        path is now unreachable rather than merely mitigated: it never gets
        the chance to fire.
        """
        assert _journal_entry_count(journalled_db_with_edge) == 3  # 2 thoughts + 1 edge

        def _edge_count() -> int:
            conn = sqlite3.connect(journalled_db_with_edge)
            try:
                row = conn.execute("SELECT COUNT(*) FROM edge").fetchone()
                return int(row[0])
            finally:
                conn.close()

        assert _edge_count() == 1

        result = runner.invoke(
            cli,
            ["--db", str(journalled_db_with_edge), "restore", "-i", str(colliding_snapshot)],
        )

        assert result.exit_code != 0
        assert "--orphan-journal-entries" in result.output

        assert _edge_count() == 1  # the cascade never fired: nothing was deleted
        assert _journal_entry_count(journalled_db_with_edge) == 3
        edge_deltas = _journal_entry_deltas(journalled_db_with_edge, "edge-001")
        assert len(edge_deltas) == 1

        verify_result = runner.invoke(
            cli,
            ["--db", str(journalled_db_with_edge), "--format", "json", "verify"],
        )
        assert verify_result.exit_code == 0, verify_result.output
        verify_data = json.loads(verify_result.output)
        assert verify_data["valid"] is True
        assert verify_data["entries_checked"] == 3

    def test_a_record_inserted_before_the_collision_is_also_rolled_back(
        self,
        runner: CliRunner,
        journalled_db: Path,
        colliding_snapshot_with_a_leading_new_record: Path,
    ) -> None:
        """The rollback discards a record inserted before the collision, too.

        ``colliding_snapshot_with_a_leading_new_record`` carries
        ``t-brand-new`` first, which the plain ``INSERT`` accepts with no
        complaint, followed by ``t-old-0``, which collides with the
        journalled target and aborts the whole restore. This is the class
        docstring's own claim: the whole-transaction rollback in
        ``_import_records_to_db`` must discard ``t-brand-new`` along with
        refusing ``t-old-0``, not leave the earlier, otherwise-successful
        insert sitting on disk.
        """
        # This test's whole premise is that ``t-brand-new`` precedes the
        # colliding ``t-old-0`` in the snapshot below. The exporter's
        # ``SELECT * FROM thought`` (cli/main.py) carries no ``ORDER BY``, so
        # SQL does not guarantee that order -- it only happens to match
        # creation order today. Pin the precondition here: if export order
        # ever changes, this fails loudly instead of leaving the assertions
        # below passing for the wrong reason (``t-brand-new`` absent because
        # it was never attempted, not because the rollback discarded it).
        snapshot_lines = [
            json.loads(line)
            for line in colliding_snapshot_with_a_leading_new_record.read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        thought_ids_in_snapshot_order = [
            line["data"]["thought_id"] for line in snapshot_lines if line.get("_type") == "thought"
        ]
        assert thought_ids_in_snapshot_order == ["t-brand-new", "t-old-0"]

        assert _journal_entry_count(journalled_db) == 3

        result = runner.invoke(
            cli,
            [
                "--db",
                str(journalled_db),
                "restore",
                "-i",
                str(colliding_snapshot_with_a_leading_new_record),
            ],
        )

        assert result.exit_code != 0
        assert "--orphan-journal-entries" in result.output

        stored_ids = _stored_core_ids(journalled_db)["thought"]
        assert "t-brand-new" not in stored_ids
        assert stored_ids == {"t-old-0", "t-old-1", "t-old-2"}

        conn = sqlite3.connect(journalled_db)
        try:
            live_content = conn.execute(
                "SELECT content FROM thought WHERE thought_id = ?",
                ("t-old-0",),
            ).fetchone()[0]
        finally:
            conn.close()
        assert live_content == "Content for t-old-0"
        assert _journal_entry_count(journalled_db) == 3


class TestJournalledMergeCollisionGate:
    """The rest of the journalled-merge collision gate's required verification.

    Complements ``TestRestoreRefusesCollisionAgainstAJournalledStore`` with the
    cases that class does not cover: an identical-content restore (no value
    differs, only the primary key collides), a journal entry recorded with an
    unrecognised ``mutation_type``, the composite ``UNIQUE`` on ``edge`` that a
    primary-id probe would miss, the negative control proving the gate is
    scoped to a non-empty journal, and the ``--orphan-journal-entries``
    override itself.
    """

    def test_identical_restore_is_refused_and_rowids_never_move(
        self,
        runner: CliRunner,
        journalled_db: Path,
        identical_snapshot_of_journalled_db: Path,
    ) -> None:
        """Restoring a store's own snapshot back into itself is refused.

        Every value in the incoming record matches what is already stored --
        only the primary key collides. Under the old ``INSERT OR REPLACE``
        behavior SQLite still resolves that collision by deleting and
        re-inserting the row, which silently changes its ``rowid`` (and, with
        it, desynchronises anything keyed on ``rowid``, such as the
        ``thought_fts`` external-content index or a persisted sqlite-vec
        table) even though no column value differs. A plain ``INSERT`` never
        reaches that delete-and-recreate at all.
        """
        before_rowids = _thought_rowids(journalled_db)
        # "content" matches every journalled thought's own content column
        # (`_write_journalled_thoughts` writes "Content for {thought_id}"), so
        # this is a real per-entry count of the FTS index, not of `thought`.
        before_fts = _thought_fts_match_count(journalled_db, "content")
        assert before_fts == 3
        assert _journal_entry_count(journalled_db) == 3

        result = runner.invoke(
            cli,
            ["--db", str(journalled_db), "restore", "-i", str(identical_snapshot_of_journalled_db)],
        )

        assert result.exit_code != 0
        assert "--orphan-journal-entries" in result.output

        assert _thought_rowids(journalled_db) == before_rowids
        # The refused restore never inserted anything, so the index carries
        # exactly the entries it started with -- neither a stale leftover from
        # a delete-and-recreate nor a duplicate.
        assert _thought_fts_match_count(journalled_db, "content") == before_fts
        assert _journal_entry_count(journalled_db) == 3

    def test_unrecognised_mutation_type_does_not_bypass_the_gate(
        self,
        runner: CliRunner,
        custom_mutation_db: Path,
        colliding_snapshot_for_custom_mutation_db: Path,
    ) -> None:
        """A ``CUSTOM_MUTATION`` journal entry gates a collision exactly like any other.

        The gate only asks whether ``journal_entry`` has rows; it never reads
        ``mutation_type``. A detector that instead tried to interpret the
        journal's content could be bypassed by a value it did not recognise --
        the second fatal finding against the abandoned branch this gate
        replaces -- and this is unreachable here for the same reason the first
        finding is: nothing about this record's insert depends on what
        ``mutation_type`` says.
        """
        assert _journal_entry_count(custom_mutation_db) == 1
        conn = sqlite3.connect(custom_mutation_db)
        try:
            before_content = conn.execute(
                "SELECT content FROM thought WHERE thought_id = 't-cm-0'"
            ).fetchone()[0]
        finally:
            conn.close()
        assert before_content == "Content for t-cm-0"

        result = runner.invoke(
            cli,
            [
                "--db",
                str(custom_mutation_db),
                "restore",
                "-i",
                str(colliding_snapshot_for_custom_mutation_db),
            ],
        )

        assert result.exit_code != 0
        assert "--orphan-journal-entries" in result.output

        conn = sqlite3.connect(custom_mutation_db)
        try:
            after_content = conn.execute(
                "SELECT content FROM thought WHERE thought_id = 't-cm-0'"
            ).fetchone()[0]
        finally:
            conn.close()
        assert after_content == before_content
        assert _journal_entry_count(custom_mutation_db) == 1

    def test_fresh_edge_id_with_duplicate_triple_is_refused(
        self,
        runner: CliRunner,
        journalled_db_with_edge: Path,
        fresh_edge_id_duplicate_triple_snapshot: Path,
    ) -> None:
        """A brand-new ``edge_id`` sharing an existing edge's triple is refused.

        The incoming record does not collide with ``edge-001`` on
        ``edge_id`` -- it has a different one (``edge-999``) -- so a probe
        keyed on primary ids would see no collision at all. It collides on
        ``edge``'s composite ``UNIQUE(from_thought_id, to_thought_id,
        edge_type)`` (schema_core.sql), which the plain ``INSERT`` leaves to
        SQLite itself to catch.
        """
        conn = sqlite3.connect(journalled_db_with_edge)
        try:
            before_edge_id = conn.execute(
                "SELECT edge_id FROM edge WHERE from_thought_id = 't-old-0' "
                "AND to_thought_id = 't-old-1'"
            ).fetchone()[0]
        finally:
            conn.close()
        assert before_edge_id == "edge-001"

        result = runner.invoke(
            cli,
            [
                "--db",
                str(journalled_db_with_edge),
                "restore",
                "-i",
                str(fresh_edge_id_duplicate_triple_snapshot),
            ],
        )

        assert result.exit_code != 0
        assert "--orphan-journal-entries" in result.output

        conn = sqlite3.connect(journalled_db_with_edge)
        try:
            row = conn.execute("SELECT COUNT(*) FROM edge").fetchone()
            edge_count = int(row[0])
            surviving_edge_id = conn.execute(
                "SELECT edge_id FROM edge WHERE from_thought_id = 't-old-0' "
                "AND to_thought_id = 't-old-1'"
            ).fetchone()[0]
        finally:
            conn.close()
        assert edge_count == 1
        assert surviving_edge_id == "edge-001"  # the original edge, not edge-999

    def test_colliding_restore_into_an_empty_journal_still_replaces(
        self,
        runner: CliRunner,
        populated_db: Path,
        colliding_snapshot_for_populated_db: Path,
    ) -> None:
        """The negative control: an empty ``journal_entry`` keeps the original merge behavior.

        ``populated_db`` is never journalled, which is also the overwhelmingly
        common case in practice -- the CLI has no flag that enables
        journaling. This restore must still succeed and still replace,
        proving the gate is scoped to a non-empty journal rather than having
        quietly changed the default merge behavior for everyone.
        """
        assert _journal_entry_count(populated_db) == 0

        result = runner.invoke(
            cli,
            ["--db", str(populated_db), "restore", "-i", str(colliding_snapshot_for_populated_db)],
        )

        assert result.exit_code == 0, result.output
        conn = sqlite3.connect(populated_db)
        try:
            live_content = conn.execute(
                "SELECT content FROM thought WHERE thought_id = 'thought-000'"
            ).fetchone()[0]
        finally:
            conn.close()
        assert live_content == "Replacement content for thought-000"

    def test_orphan_journal_entries_overrides_the_gate(
        self,
        runner: CliRunner,
        journalled_db: Path,
        colliding_snapshot: Path,
    ) -> None:
        """``--orphan-journal-entries`` restores the original merge behavior on request.

        The same collision ``TestRestoreRefusesCollisionAgainstAJournalledStore``
        shows refused now succeeds and replaces once the override is passed,
        which is the flag's entire purpose: a caller who has weighed the gap
        and wants the merge anyway.
        """
        result = runner.invoke(
            cli,
            [
                "--db",
                str(journalled_db),
                "restore",
                "-i",
                str(colliding_snapshot),
                "--orphan-journal-entries",
            ],
        )

        assert result.exit_code == 0, result.output
        conn = sqlite3.connect(journalled_db)
        try:
            live_content = conn.execute(
                "SELECT content FROM thought WHERE thought_id = 't-old-0'"
            ).fetchone()[0]
        finally:
            conn.close()
        assert live_content == "Replacement content for t-old-0"
        # The journal is left exactly as before: the override does not touch
        # journal_entry, it only changes which INSERT form is used.
        assert _journal_entry_count(journalled_db) == 3

    def test_restore_help_names_the_override_flag(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["restore", "--help"])
        assert result.exit_code == 0, result.output
        assert "--orphan-journal-entries" in result.output

    def test_foreign_key_violation_still_propagates_unchanged_under_the_gate(
        self,
        runner: CliRunner,
        journalled_db: Path,
    ) -> None:
        """A foreign-key violation is a different, pre-existing error and keeps its own behavior.

        An incoming edge whose endpoints do not exist in the target violates
        ``edge``'s foreign keys to ``thought`` -- a ``sqlite3.IntegrityError``
        with ``sqlite_errorcode`` ``787`` (``SQLITE_CONSTRAINT_FOREIGNKEY``),
        never ``1555`` or ``2067``. The gate must not catch this: it is not a
        collision the gate is scoped to, so it has to propagate exactly as it
        always did (an uncaught ``IntegrityError``, not the gate's
        ``click.ClickException``) whether or not the journalled-merge
        collision gate is active for this restore.
        """
        snap = journalled_db.parent / "fk-violation.jsonl"
        edge_data = {
            "edge_id": "edge-fk-1",
            "from_thought_id": "missing-a",
            "to_thought_id": "missing-b",
            "edge_type": "ASSOCIATED",
            "weight": 0.5,
            "created_cycle": 1,
        }
        snap.write_text(json.dumps({"_type": "edge", "data": edge_data}) + "\n", encoding="utf-8")

        result = runner.invoke(
            cli,
            ["--db", str(journalled_db), "restore", "-i", str(snap)],
            standalone_mode=False,
        )

        assert isinstance(result.exception, sqlite3.IntegrityError), result.exception
        assert result.exception.sqlite_errorcode == 787
        assert "orphan-journal-entries" not in str(result.exception)

    async def test_a_refused_collision_leaves_the_connection_out_of_a_transaction(
        self,
        colliding_snapshot: Path,
        tmp_path: Path,
    ) -> None:
        """A refused collision leaves ``in_transaction`` false on the caller's own open connection.

        Every other test in this module drives restore through the CLI,
        which always closes its connection afterward -- closing an
        ``aiosqlite`` connection implicitly rolls back any open transaction,
        so those tests cannot tell an explicit ``await conn.rollback()`` in
        ``_import_records_to_db``'s ``finally`` block apart from one that was
        silently removed. This calls ``_import_records_to_db`` directly on a
        connection it keeps open across the call, so only the explicit
        rollback -- not connection teardown -- can account for the result.
        """
        import aiosqlite

        from engrava import (
            LifecycleStatus,
            Priority,
            SqliteEngravaCore,
            ThoughtRecord,
            ThoughtType,
        )

        db_path = tmp_path / "direct-target.db"
        conn = await aiosqlite.connect(str(db_path))
        conn.row_factory = aiosqlite.Row
        try:
            # Built directly against the domain API, in-line, rather than via
            # ``_write_journalled_thoughts`` -- that helper's own ``asyncio.run()``
            # cannot be called from inside this test's already-running event loop.
            store = SqliteEngravaCore(conn, journal_enabled=True)
            await store.ensure_schema()
            await store.create_thought(
                ThoughtRecord(
                    thought_id="t-old-0",
                    essence="Essence for t-old-0",
                    content="Content for t-old-0",
                    thought_type=ThoughtType.OBSERVATION,
                    source="test",
                    lifecycle_status=LifecycleStatus.ACTIVE,
                    priority=Priority.P2,
                    created_cycle=1,
                    updated_cycle=1,
                )
            )
            await conn.commit()

            with pytest.raises(click.ClickException):
                await _import_records_to_db(conn, colliding_snapshot, orphan_journal_entries=False)

            assert not conn.in_transaction
        finally:
            await conn.close()


class TestGc:
    """Tests for ``engrava gc``."""

    def test_gc_removes_archived(self, runner: CliRunner, populated_db: Path) -> None:
        result = runner.invoke(cli, ["--db", str(populated_db), "gc"])
        assert result.exit_code == 0
        assert "Collected 1" in result.output

        # Verify only 2 remain
        check = runner.invoke(
            cli,
            ["--db", str(populated_db), "--format", "json", "info"],
        )
        data = json.loads(check.output)
        assert data["thoughts"]["total"] == 2

    def test_gc_dry_run(self, runner: CliRunner, populated_db: Path) -> None:
        result = runner.invoke(cli, ["--db", str(populated_db), "gc", "--dry-run"])
        assert result.exit_code == 0
        # The dry run is read immediately before the destructive run, so it must
        # name every row category the real run deletes, not just the thoughts
        # its count covers.
        assert "Would delete 1 archived thoughts" in result.output
        assert "edges, embeddings, and actions" in result.output
        assert "orphaned edges" not in result.output

        # Verify nothing actually deleted
        check = runner.invoke(
            cli,
            ["--db", str(populated_db), "--format", "json", "info"],
        )
        data = json.loads(check.output)
        assert data["thoughts"]["total"] == 3

    def test_gc_help_names_full_blast_radius(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["gc", "--help"])
        assert result.exit_code == 0
        assert "edges, embeddings, and actions" in result.output
        assert "orphaned edges" not in result.output

    def test_gc_nothing_to_collect(self, runner: CliRunner, populated_db: Path) -> None:
        # First gc removes the archived one
        runner.invoke(cli, ["--db", str(populated_db), "gc"])
        # Second gc should find nothing
        result = runner.invoke(cli, ["--db", str(populated_db), "gc"])
        assert result.exit_code == 0
        assert "No archived" in result.output


class TestMigrate:
    """Tests for ``engrava migrate``."""

    def test_migrate_creates_schema(self, runner: CliRunner, tmp_path: Path) -> None:
        new_db = tmp_path / "fresh.db"
        result = runner.invoke(cli, ["--db", str(new_db), "migrate"])
        assert result.exit_code == 0
        assert "Schema up to date" in result.output
        assert new_db.exists()


class TestExport:
    """Tests for ``engrava export``."""

    def test_export_all(self, runner: CliRunner, populated_db: Path, tmp_path: Path) -> None:
        out = tmp_path / "export.json"
        result = runner.invoke(
            cli,
            ["--db", str(populated_db), "export", "-o", str(out)],
        )
        assert result.exit_code == 0
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["format"] == "engrava-export"
        assert data["version"] == "0.1.0"
        assert len(data["thoughts"]) == 3
        assert len(data["edges"]) == 1

    def test_export_with_status_filter(
        self,
        runner: CliRunner,
        populated_db: Path,
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "active.json"
        result = runner.invoke(
            cli,
            ["--db", str(populated_db), "export", "-o", str(out), "--status", "ACTIVE"],
        )
        assert result.exit_code == 0
        data = json.loads(out.read_text(encoding="utf-8"))
        assert len(data["thoughts"]) == 2

    def test_export_default_path(self, runner: CliRunner, populated_db: Path) -> None:
        result = runner.invoke(cli, ["--db", str(populated_db), "export"])
        assert result.exit_code == 0
        default_out = populated_db.with_suffix(".export.json")
        assert default_out.exists()


# ------------------------------------------------------------------
# Corrupt-database hang guard
# ------------------------------------------------------------------
#
# ``sqlite3.DatabaseError: file is not a database`` raised from the first
# ``PRAGMA`` against a corrupt/truncated file used to leave the aiosqlite
# connection open. aiosqlite's connection worker thread is not a daemon and
# stops only when ``Connection.close()`` sends it the shutdown sentinel, so a
# leaked connection blocks ``threading._shutdown`` and the process never
# exits — it prints a traceback (or nothing, depending on buffering) and then
# hangs forever rather than returning any exit code.
#
# The hang happens at interpreter shutdown, which an in-process
# ``CliRunner`` invocation never reaches, so these run the real CLI as a
# subprocess with a hard wall-clock timeout: a regression here fails these
# tests promptly instead of wedging the whole suite.

# Long enough that a fixed, promptly-erroring command never gets close on a
# loaded CI host; far short of "wedge the test worker" if the fix regresses.
_CLI_SUBPROCESS_TIMEOUT_S = 20.0

# A command that returns fast enough for a live process to still be a
# meaningful "promptly" — as opposed to merely "before the hard cap fired".
_PROMPT_CEILING_S = 10.0


def _run_python_subprocess(argv: list[str]) -> tuple[subprocess.CompletedProcess[str], float]:
    """Run ``python *argv*`` as a real, separate process with a hard timeout.

    Uses this test file's own ``src`` on ``PYTHONPATH`` rather than whatever
    ``sys.path`` the test runner happened to start with, so the subprocess
    always exercises the same worktree's code the test itself was collected
    from — a shared, editable-installed ``engrava`` elsewhere on the host
    must not shadow it.

    Args:
        argv: Full argv after the interpreter, e.g.
            ``["-m", "engrava.cli.main", "--db", str(db_path), "info"]`` or
            ``["-c", some_source, "--db", str(db_path), "info"]``.

    Returns:
        The completed process and the wall-clock seconds it took.

    Raises:
        Failed test: via ``pytest.fail``, if the process does not exit
            within :data:`_CLI_SUBPROCESS_TIMEOUT_S` — the hang this guards
            against — instead of letting ``subprocess.TimeoutExpired``
            propagate as an error or, worse, blocking forever.

    """
    repo_src = str(Path(__file__).resolve().parent.parent / "src")
    env = {**os.environ, "PYTHONPATH": repo_src}
    start = time.monotonic()
    try:
        completed = subprocess.run(  # noqa: S603 -- fixed argv, no shell, our own source
            [sys.executable, *argv],
            capture_output=True,
            text=True,
            check=False,
            env=env,
            timeout=_CLI_SUBPROCESS_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"`python {' '.join(argv)}` did not exit within "
            f"{_CLI_SUBPROCESS_TIMEOUT_S:.0f}s — this is the hang this test "
            "guards against, not a slow environment."
        )
    return completed, time.monotonic() - start


def _run_engrava_subprocess(args: list[str]) -> tuple[subprocess.CompletedProcess[str], float]:
    """Run ``python -m engrava.cli.main *args*`` as a real, separate process.

    Args:
        args: Full CLI argv after the interpreter and ``-m`` module name,
            e.g. ``["--db", str(db_path), "info"]``.

    Returns:
        The completed process and the wall-clock seconds it took.

    """
    return _run_python_subprocess(["-m", "engrava.cli.main", *args])


def _assert_completed_fails_fast(
    completed: subprocess.CompletedProcess[str], elapsed: float
) -> None:
    """Assert a completed subprocess exited non-zero, promptly, not via the hang."""
    assert completed.returncode != 0, (
        f"expected a non-zero exit, got {completed.returncode}\n"
        f"stdout={completed.stdout!r}\nstderr={completed.stderr!r}"
    )
    assert completed.returncode != 124, "124 is the timeout(1) sentinel — this hung"
    assert elapsed < _PROMPT_CEILING_S, (
        f"took {elapsed:.1f}s — expected a prompt failure, not one that only "
        "beat the hard subprocess timeout"
    )


def _assert_fails_fast_and_names_the_problem(args: list[str]) -> None:
    """Assert *args* exits non-zero, promptly, with the failure named."""
    completed, elapsed = _run_engrava_subprocess(args)
    _assert_completed_fails_fast(completed, elapsed)
    combined = (completed.stdout + completed.stderr).lower()
    assert "database" in combined, (
        f"expected the failure to name the database problem\nstdout={completed.stdout!r}\n"
        f"stderr={completed.stderr!r}"
    )


def _assert_succeeds(args: list[str]) -> None:
    """Known-good control: the same command against a valid database still works."""
    completed, _elapsed = _run_engrava_subprocess(args)
    assert completed.returncode == 0, (
        f"expected exit 0, got {completed.returncode}\n"
        f"stdout={completed.stdout!r}\nstderr={completed.stderr!r}"
    )


@pytest.fixture
def corrupt_db(tmp_path: Path) -> Path:
    """A file named like a database that is not one — text, truncated, whatever.

    Reproduces the reported shape: a plain text file where a SQLite database
    is expected, which opens fine (the file exists) but fails the first real
    read against it.
    """
    path = tmp_path / "corrupt.sqlite"
    path.write_text("this is not a sqlite database, just some text\n", encoding="utf-8")
    return path


@pytest.fixture
def valid_snapshot(runner: CliRunner, populated_db: Path, tmp_path: Path) -> Path:
    """A real JSONL snapshot of ``populated_db`` — a valid ``restore`` input."""
    snap = tmp_path / "valid-snapshot.jsonl"
    result = runner.invoke(cli, ["--db", str(populated_db), "snapshot", "-o", str(snap)])
    assert result.exit_code == 0
    return snap


class TestCorruptDatabaseExitsInsteadOfHanging:
    """Every built-in command that opens the database file must fail fast.

    One test per affected command, each with the known-good control the
    acceptance criteria require: the same command against ``populated_db``
    (or an equivalent valid target) still works. Covers all eight built-in
    commands found to route through a connection-opening call in
    ``cli/main.py`` — the originally reported four (``info``, ``gc``,
    ``migrate``, ``snapshot``) plus ``verify``, ``query``, ``export``, and
    the single-database branch of ``restore``, which shared the exact same
    unclosed-connection mechanism. The two commands that only ever go
    through :class:`~engrava.infrastructure.service_manager.EngravaManager`
    (the ``--service`` branch of ``snapshot``/``restore``) are not covered
    here — that path already closes on its own error, independently of this
    fix.
    """

    def test_info(self, corrupt_db: Path, populated_db: Path) -> None:
        _assert_fails_fast_and_names_the_problem(["--db", str(corrupt_db), "info"])
        _assert_succeeds(["--db", str(populated_db), "info"])

    def test_verify(self, corrupt_db: Path, populated_db: Path) -> None:
        _assert_fails_fast_and_names_the_problem(["--db", str(corrupt_db), "verify"])
        _assert_succeeds(["--db", str(populated_db), "verify"])

    def test_query(self, corrupt_db: Path, populated_db: Path) -> None:
        _assert_fails_fast_and_names_the_problem(
            ["--db", str(corrupt_db), "query", "COUNT thoughts"]
        )
        _assert_succeeds(["--db", str(populated_db), "query", "COUNT thoughts"])

    def test_gc(self, corrupt_db: Path, populated_db: Path) -> None:
        _assert_fails_fast_and_names_the_problem(["--db", str(corrupt_db), "gc", "--dry-run"])
        _assert_succeeds(["--db", str(populated_db), "gc", "--dry-run"])

    def test_migrate(self, corrupt_db: Path, tmp_path: Path) -> None:
        _assert_fails_fast_and_names_the_problem(["--db", str(corrupt_db), "migrate"])
        # migrate is the one built-in whose target need not exist yet — a
        # fresh path is its own "known good" control (see
        # TestMigrate.test_migrate_creates_schema above).
        _assert_succeeds(["--db", str(tmp_path / "fresh-for-migrate.db"), "migrate"])

    def test_snapshot(self, corrupt_db: Path, populated_db: Path, tmp_path: Path) -> None:
        bad_out = tmp_path / "bad.snapshot.jsonl"
        good_out = tmp_path / "good.snapshot.jsonl"
        _assert_fails_fast_and_names_the_problem(
            ["--db", str(corrupt_db), "snapshot", "-o", str(bad_out)]
        )
        _assert_succeeds(["--db", str(populated_db), "snapshot", "-o", str(good_out)])

    def test_export(self, corrupt_db: Path, populated_db: Path, tmp_path: Path) -> None:
        bad_out = tmp_path / "bad.export.json"
        good_out = tmp_path / "good.export.json"
        _assert_fails_fast_and_names_the_problem(
            ["--db", str(corrupt_db), "export", "-o", str(bad_out)]
        )
        _assert_succeeds(["--db", str(populated_db), "export", "-o", str(good_out)])

    def test_restore(self, corrupt_db: Path, valid_snapshot: Path, tmp_path: Path) -> None:
        _assert_fails_fast_and_names_the_problem(
            ["--db", str(corrupt_db), "restore", "-i", str(valid_snapshot)]
        )
        good_target = tmp_path / "restored-for-control.db"
        _assert_succeeds(["--db", str(good_target), "restore", "-i", str(valid_snapshot)])


# A failure injected into ``SqliteEngravaCore.ensure_schema`` before ``cli()``
# ever runs. Unlike ``corrupt_db``, this never touches ``_open_db`` at all —
# ``aiosqlite.connect()`` against a target that does not exist yet always
# succeeds, and the file it creates is a valid, empty SQLite database, so
# every ``PRAGMA`` in ``_open_db`` passes. The only way to fail *after* the
# connection is open and *before* the block that is supposed to protect it
# is to fail inside whatever runs in between — here, ``ensure_schema()``.
_ENSURE_SCHEMA_FAULT_INJECTION = """
from engrava.infrastructure.sqlite.engrava_core import SqliteEngravaCore


async def _boom(self, *args, **kwargs):
    raise RuntimeError("injected ensure_schema failure for a bootstrap-window test")


SqliteEngravaCore.ensure_schema = _boom

from engrava.cli.main import cli

cli()
"""


class TestRestoreBootstrapWindowClosesOnFailure:
    """A failure inside ``ensure_schema()`` -- not inside ``_open_db`` -- must
    also close the connection promptly.

    ``restore`` against a target that does not pre-exist calls
    ``store = SqliteEngravaCore(conn)`` and then ``await
    store.ensure_schema()`` to bootstrap it. Before ``_opened_db`` wrapped
    the whole body in one step, both of those ran ahead of the ``try`` that
    was supposed to guarantee the close, so a failure there leaked the
    connection exactly like the corrupt-file case -- just through a
    different call, with a target file that is itself perfectly valid. The
    ``corrupt_db`` tests above are blind to this: they all fail inside
    ``_open_db``, which was already closing correctly before this round of
    fixes even started.
    """

    def test_restore_bootstrap_failure_is_not_a_hang(self, tmp_path: Path) -> None:
        target = tmp_path / "fresh-target-that-fails-to-bootstrap.db"
        assert not target.exists()  # pre_existing must be False to reach ensure_schema()
        missing_input = tmp_path / "never-read.jsonl"  # ensure_schema() fails first

        completed, elapsed = _run_python_subprocess(
            [
                "-c",
                _ENSURE_SCHEMA_FAULT_INJECTION,
                "--db",
                str(target),
                "restore",
                "-i",
                str(missing_input),
            ]
        )
        _assert_completed_fails_fast(completed, elapsed)
        combined = completed.stdout + completed.stderr
        assert "injected ensure_schema failure" in combined, (
            f"expected the injected failure to surface, not something else\n"
            f"stdout={completed.stdout!r}\nstderr={completed.stderr!r}"
        )


# A failure injected into ``aiosqlite.Connection.close()`` itself, on a
# command that otherwise succeeds against a perfectly valid database. Proves
# ``_opened_db``'s success path does not route a close failure through
# ``_close_quietly`` -- which logs and swallows by design, correct only when
# an exception is already in flight.
_CLOSE_FAILS_ON_SUCCESS_INJECTION = """
import aiosqlite

_real_close = aiosqlite.Connection.close


async def _close_blows_up(self):
    await _real_close(self)
    raise RuntimeError("injected close failure on the success path")


aiosqlite.Connection.close = _close_blows_up

from engrava.cli.main import cli

cli()
"""


class TestSuccessPathCloseFailureIsNotSwallowed:
    """A close() failure on an otherwise-successful command must surface.

    ``_opened_db``'s cleanup used to be an unconditional
    ``finally: await _close_quietly(conn)`` -- including on the success
    path, where a close failure is not secondary to anything, it is the
    only error there is. ``_close_quietly`` logs and swallows by design
    (correct for the exception-in-flight case), so the unconditional call
    turned a genuine close failure into a command that printed its normal
    success output and exited 0 -- a worse outcome than the original hang,
    which was at least visible.
    """

    def test_close_failure_after_a_successful_command_is_not_silent(
        self, populated_db: Path
    ) -> None:
        completed, elapsed = _run_python_subprocess(
            [
                "-c",
                _CLOSE_FAILS_ON_SUCCESS_INJECTION,
                "--db",
                str(populated_db),
                "info",
            ]
        )
        _assert_completed_fails_fast(completed, elapsed)
        combined = completed.stdout + completed.stderr
        assert "injected close failure" in combined, (
            f"expected the close failure to surface, not be swallowed\n"
            f"stdout={completed.stdout!r}\nstderr={completed.stderr!r}"
        )

    def test_close_success_control_still_exits_zero(self, populated_db: Path) -> None:
        """Known-good control: an ordinary close on the success path still exits 0."""
        _assert_succeeds(["--db", str(populated_db), "info"])


class _FakeConnection:
    """Stand-in connection for testing the CLI's ``_close_quietly`` in isolation.

    ``started`` fires the instant ``close()`` begins running, so a test can
    wait for the close to genuinely be in flight before delivering a
    cancellation -- no wall-clock sleep needed to land the race. ``may_finish``
    then holds the close from completing until the test says so, which is
    what makes the outcome deterministic rather than sleep-tuned: the
    cancellation is guaranteed to land strictly before the close resolves,
    and the close is guaranteed not to resolve on its own before the test
    permits it.
    """

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.may_finish = asyncio.Event()
        self.finished = False

    async def close(self) -> None:
        self.started.set()
        await self.may_finish.wait()
        self.finished = True


class _FailingConnection:
    """A connection whose ``close()`` raises an ordinary exception."""

    async def close(self) -> None:
        msg = "close blew up"
        raise RuntimeError(msg)


class _SynchronouslyFailingConnection:
    """A connection whose ``close()`` raises before returning an awaitable.

    Not an ``async def`` -- calling ``conn.close()`` raises immediately,
    before ``asyncio.ensure_future`` ever gets a coroutine to schedule.
    Distinct from ``_FailingConnection``, whose exception only surfaces
    once the resulting coroutine is awaited.
    """

    def close(self) -> None:
        msg = "close blew up synchronously"
        raise RuntimeError(msg)


class TestCliCloseQuietlyCancellation:
    """The CLI's own ``_close_quietly`` copy needs the same cancellation fix.

    This module defines its own ``_close_quietly`` rather than importing the
    infrastructure layer's (see that function's docstring), so the
    escaped-``BaseException`` gap in ``await conn.close()`` had to be fixed
    here independently too. These mirror
    ``TestCloseQuietlyCancellation`` in ``tests/test_service_isolation.py``,
    which covers the infrastructure copy.
    """

    async def test_the_close_still_completes_when_cancelled_mid_close(self) -> None:
        """A cancellation mid-close must not abandon the close itself."""
        conn = _FakeConnection()
        task = asyncio.create_task(_close_quietly(conn))
        await conn.started.wait()
        task.cancel()
        conn.may_finish.set()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert conn.finished, (
            "conn.close() never ran to completion under cancellation -- the "
            "exact leak _close_quietly exists to prevent"
        )

    async def test_an_ordinary_exception_from_close_is_still_swallowed_and_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Control: the documented, uncancelled behaviour must be unchanged."""
        conn = _FailingConnection()
        with caplog.at_level(logging.WARNING):
            await _close_quietly(conn)

        assert "Error closing connection during cleanup" in caplog.text

    async def test_a_synchronous_close_failure_is_still_swallowed_and_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A close() that raises before returning an awaitable must not escape.

        ``close_task = asyncio.ensure_future(conn.close())`` evaluates
        ``conn.close()`` before scheduling anything -- if that line ever
        sits outside the ``try``, a synchronous failure there escapes
        instead of being logged and swallowed, changing the ordinary,
        uncancelled contract this helper exists to keep.
        """
        conn = _SynchronouslyFailingConnection()
        with caplog.at_level(logging.WARNING):
            await _close_quietly(conn)

        assert "Error closing connection during cleanup" in caplog.text


class _PermissionDeniedConnection:
    """A connection whose ``close()`` raises a real, ordinary ``PermissionError``."""

    async def close(self) -> None:
        raise PermissionError(13, "Permission denied")


class _KeyboardInterruptOnCloseStrError(Exception):
    """Its own ``__str__`` raises ``KeyboardInterrupt``, like a real Ctrl-C mid-read."""

    def __str__(self) -> str:
        raise KeyboardInterrupt


class _SystemExitOnCloseStrError(Exception):
    """Its own ``__str__`` raises ``SystemExit(37)`` instead of returning text."""

    def __str__(self) -> str:
        raise SystemExit(37)


class _KeyboardInterruptOnCloseStrConnection:
    """A connection whose ``close()`` raises an exception hostile in its own ``__str__``."""

    async def close(self) -> None:
        message = "close"
        raise _KeyboardInterruptOnCloseStrError(message)


class _SystemExitOnCloseStrConnection:
    """A connection whose ``close()`` raises an exception hostile in its own ``__str__``."""

    async def close(self) -> None:
        message = "close"
        raise _SystemExitOnCloseStrError(message)


class TestCliCloseQuietlyDisclosesWhyNotJustWhere:
    """A verification round found the bare/default store tier's ``_close_quietly``
    still passing ``exc_info=True`` after the ``--config`` tier's own cleanup log
    (``memory_commands._opened_full_store``) had already been fixed to stop doing
    that. ``exc_info=True`` asks the standard library's traceback formatter to
    render the close exception a second, unguarded way -- and, separately, an
    earlier fix at the ``--config`` tier had *also* removed the only place a
    close failure's own diagnosis reached the log at all, trading the
    ``exc_info=True`` defect for a real regression: frame metadata says
    *where* closing failed, never *why*, so an ordinary ``PermissionError``,
    a full disk, or a locked file all looked identical. ``_close_quietly``
    now calls :func:`~engrava.cli.exception_reporting._describe_exception`
    once on the close exception -- the same single, guarded, non-absorbing
    attempt the boundary already made for the original exception -- and logs
    its result alongside the existing frame-only stack, never
    ``exc_info=True``.
    """

    async def test_an_ordinary_close_failure_now_logs_why_not_just_where(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        conn = _PermissionDeniedConnection()
        with caplog.at_level(logging.WARNING):
            await _close_quietly(conn)

        assert "Error closing connection during cleanup" in caplog.text
        assert "PermissionError: [Errno 13] Permission denied" in caplog.text
        assert "in _close_quietly" in caplog.text, (
            "the frame-only stack must still be present alongside the new "
            "description, not replaced by it"
        )
        for record in caplog.records:
            assert record.exc_info is None, (
                "must not pass exc_info=True any more -- that is the second, "
                "unguarded render this fix removes"
            )

    async def test_a_keyboard_interrupt_from_the_close_exceptions_str_is_not_absorbed(
        self,
    ) -> None:
        """The close exception's own formatting is now read once -- a real
        interrupt raised during that read must still escape, not be
        swallowed the way the standard library's own traceback formatter
        used to swallow it under ``exc_info=True``.
        """
        conn = _KeyboardInterruptOnCloseStrConnection()

        with pytest.raises(KeyboardInterrupt):
            await _close_quietly(conn)

    async def test_a_system_exit_from_the_close_exceptions_str_is_not_absorbed(self) -> None:
        conn = _SystemExitOnCloseStrConnection()

        with pytest.raises(SystemExit) as exc_info:
            await _close_quietly(conn)

        assert exc_info.value.code == 37


class _FormatThatComparesAsAnother(str):
    """Reads as its real text; hashes and compares as ``json``."""

    __slots__ = ()

    def __hash__(self) -> int:
        return hash("json")

    def __eq__(self, other: object) -> bool:
        return other == "json"

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)


class TestCliConfigKeepsWhatItResolved:
    """The CLI config is a public entry point reached straight from the command line.

    ``output_format`` is admitted by a membership test and then selects a
    renderer by equality — both run the value's own methods — and the paths are
    built by joining text that a subclass could render differently from the
    text that was checked.
    """

    def test_the_resolved_format_is_an_exact_str(self) -> None:
        config = EngravaCLIConfig.resolve(output_format=_FormatThatComparesAsAnother("table"))
        assert type(config.output_format) is str
        assert config.output_format == "table"

    def test_an_unrecognised_format_still_falls_back_to_table(self) -> None:
        assert EngravaCLIConfig.resolve(output_format="nonsense").output_format == "table"

    def test_every_legitimate_format_survives(self) -> None:
        for fmt in ("json", "table", "csv"):
            assert EngravaCLIConfig.resolve(output_format=fmt).output_format == fmt

    def test_the_resolved_path_is_built_from_owned_text(self, tmp_path: Path) -> None:
        class _PathTextThatRendersDifferently(str):
            __slots__ = ()

            def __str__(self) -> str:
                return "/etc/escaped.db"

        declared = str(tmp_path / "real.db")
        config = EngravaCLIConfig.resolve(db_path=_PathTextThatRendersDifferently(declared))
        assert str(config.db_path) == declared

    def test_direct_construction_owns_its_fields(self, tmp_path: Path) -> None:
        config = EngravaCLIConfig(
            db_path=tmp_path / "t.db",
            output_format=_FormatThatComparesAsAnother("csv"),  # type: ignore[arg-type]  # a str subclass is what is under test
        )
        assert type(config.output_format) is str
        assert config.output_format == "csv"


class TestCliConfigChoosesTheSourceItWasGiven:
    """An explicitly supplied path is used, whatever the value says about itself.

    The resolution order is written with ``or``, which asks each candidate
    whether it is truthy — a method the value may define. A string subclass
    that answers ``False`` erases the ``--db`` the user typed and silently
    substitutes the environment's value or the built-in default.
    """

    def test_an_explicit_path_is_not_suppressed_by_the_value(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        class _PathThatDeniesItself(str):
            __slots__ = ()

            def __bool__(self) -> bool:
                return False

        monkeypatch.setenv("ENGRAVA_DB", str(tmp_path / "from-env.db"))
        supplied = str(tmp_path / "explicit.db")

        config = EngravaCLIConfig.resolve(db_path=_PathThatDeniesItself(supplied))

        assert str(config.db_path) == supplied

    def test_an_explicit_config_path_is_not_suppressed_by_the_value(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        class _PathThatDeniesItself(str):
            __slots__ = ()

            def __bool__(self) -> bool:
                return False

        monkeypatch.setenv("ENGRAVA_CONFIG", str(tmp_path / "from-env.yaml"))
        supplied = str(tmp_path / "explicit.yaml")

        config = EngravaCLIConfig.resolve(config_path=_PathThatDeniesItself(supplied))

        assert config.config_path is not None
        assert str(config.config_path) == supplied

    def test_the_environment_is_still_used_when_nothing_is_supplied(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("ENGRAVA_DB", str(tmp_path / "from-env.db"))
        assert str(EngravaCLIConfig.resolve().db_path) == str(tmp_path / "from-env.db")

    def test_an_empty_string_still_falls_through(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Genuine emptiness keeps its old meaning; only the lie is closed."""
        monkeypatch.setenv("ENGRAVA_DB", str(tmp_path / "from-env.db"))
        assert str(EngravaCLIConfig.resolve(db_path="").db_path) == str(tmp_path / "from-env.db")


class TestModuleEntryPointExposesTheMemoryVerbs:
    """``python -m engrava.cli.main`` must expose ``remember`` / ``recall`` / ``link``.

    Running this file directly makes it ``__main__`` -- a module object
    distinct from ``engrava.cli.main`` even though it is the same file.
    ``engrava.cli.memory_commands`` decorates the ``cli`` Group belonging to
    the dotted-path import, not the ``__main__`` one, so without the
    ``__main__`` guard re-entering through that dotted import, the three new
    commands would silently resolve as "No such command" under ``-m`` even
    though the installed ``engrava`` console-script entry point (which never
    runs this file as ``__main__``) works.
    """

    def test_remember_appears_in_module_entry_point_help(self) -> None:
        completed, _elapsed = _run_engrava_subprocess(["--help"])
        assert completed.returncode == 0, completed.stderr
        assert "remember" in completed.stdout
        assert "recall" in completed.stdout
        assert "link" in completed.stdout

    def test_remember_actually_runs_under_the_module_entry_point(self, tmp_path: Path) -> None:
        db = tmp_path / "module-entry.db"
        completed, _elapsed = _run_engrava_subprocess(
            ["--db", str(db), "remember", "stored via python -m"]
        )
        assert completed.returncode == 0, (
            f"stdout={completed.stdout!r}\nstderr={completed.stderr!r}"
        )
        assert "No such command" not in completed.stderr
        assert db.exists()
