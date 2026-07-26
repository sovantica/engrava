"""Blast-radius tests for engrava's destructive operations.

A destructive operation is correct only when it is **narrow**: the rows it
targets are gone *and* every other row is still there, unchanged. Asserting
what the call returned — a boolean, a count, a CLI line — proves at most the
first half. A predicate widened to a whole-table wipe keeps every one of those
return values correct while the store is destroyed, so a suite that only
inspects them cannot tell the two apart.

Each test here therefore snapshots the rows that must survive **from SQLite**
before the operation, runs it, and re-reads **from SQLite** afterwards:

* the target row is absent,
* the exact set of surviving ids is what it should be, and
* every survivor's row still holds the same value in every column.

Both the survivors *and* the rows meant to disappear are read through
:func:`_require` first, which fails the test if the row was not there to begin
with. Without it either half could pass vacuously: a snapshot of a row that was
never stored compares equal to the ``None`` read back from an emptied table,
and an "it is gone now" assertion is trivially true about a row a drifting
corpus never inserted.

Four operations are covered, one class each:

* :meth:`~engrava.SqliteEngravaCore.delete_edge` — one edge, not the edge table.
* :meth:`~engrava.SqliteEngravaCore.cleanup_expired` under the ``delete`` TTL
  strategy — the expired thoughts and their cascaded children, not every thought.
* ``engrava gc`` — the ARCHIVED thoughts and their children, never the edges,
  embeddings or actions belonging to ACTIVE thoughts.
* ``engrava gc --expired`` — the expiry sweep per ``ttl.strategy``, and whether
  the archived-collection pass that follows it runs at all. Under ``archive``
  the command must **stop** after archiving; running on would hard-delete the
  rows it has just soft-retired.

Both ``gc`` passes are additionally asserted against the ``embedding_vec`` vec0
index. That table is the one store in the database no foreign key reaches, so
nothing removes a deleted thought's vector for the command: on a store carrying
the index, a pass that forgets it leaves the vector behind to occupy a KNN slot
for a thought that no longer exists, and — once SQLite reuses the freed
``embedding`` rowid — to stand in for a different thought's embedding entirely.
It is also the one store an over-broad purge could empty without any foreign key
or ``embedding`` read-back noticing, so both halves are read from the index
itself.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import sqlite3
import struct
from typing import TYPE_CHECKING

import aiosqlite
import click
import pytest
from click.testing import CliRunner

from engrava import (
    ActionRecord,
    ActionStatus,
    ActionType,
    EdgeRecord,
    EdgeType,
    KnowledgeSource,
    LifecycleStatus,
    Priority,
    SqliteEngravaCore,
    ThoughtRecord,
    ThoughtType,
    VerificationStatus,
)
from engrava.cli.main import cli

# The extension's own load sequence, reused so the read-back connections in this
# module cannot come to load sqlite-vec differently from the code under test.
from engrava.extensions.vector_sqlite_vec import _load_sqlite_vec_sync

# The real-shape per-version schema builder the migration-ladder suite already
# maintains. Reused rather than re-derived so the pre-cascade fixture below
# cannot drift from the schema that version actually shipped.
from tests.test_migration_upgrade_chains import _bootstrap_core_at_version

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
    from pathlib import Path

    from click.testing import Result


# ---------------------------------------------------------------------------
# Read-back queries
# ---------------------------------------------------------------------------
# Literal SQL, never interpolated: the read-back path must not be able to
# disagree with the schema the operation under test writes to.

_THOUGHT_BY_ID = "SELECT * FROM thought WHERE thought_id = ?"
_EDGE_BY_ID = "SELECT * FROM edge WHERE edge_id = ?"
_EMBEDDING_BY_OWNER = "SELECT * FROM embedding WHERE owner_id = ?"
_ACTION_BY_ID = "SELECT * FROM action WHERE action_id = ?"

_ALL_THOUGHT_IDS = "SELECT thought_id FROM thought"
_ALL_EDGE_IDS = "SELECT edge_id FROM edge"
_ALL_EMBEDDING_OWNERS = "SELECT owner_id FROM embedding"
_ALL_ACTION_IDS = "SELECT action_id FROM action"
_ARCHIVED_THOUGHT_IDS = "SELECT thought_id FROM thought WHERE lifecycle_status = 'ARCHIVED'"

#: The vec0 index is addressed by ``rowid`` alone — it holds no thought id — so
#: it is read as a set of rowids and mapped back through ``embedding``.
_ALL_VEC_ROWIDS = "SELECT rowid FROM embedding_vec"
_THOUGHT_EMBEDDING_ROWIDS = "SELECT owner_id, rowid FROM embedding WHERE owner_type = 'THOUGHT'"

#: Whether the database carries a persisted vec0 index at all. Readable without
#: sqlite-vec loaded (``sqlite_master`` is an ordinary table), unlike every
#: statement that names ``embedding_vec`` itself.
_VEC_INDEX_IN_SCHEMA = (
    "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'embedding_vec'"
)

#: sqlite-vec ships as an optional extra (``engrava[vec]``); the tests that need
#: a real vec0 index skip without it, exactly as the other vec0 suites do.
sqlite_vec_required = pytest.mark.skipif(
    importlib.util.find_spec("sqlite_vec") is None,
    reason="sqlite-vec package not installed",
)


def _as_mapping(row: sqlite3.Row | None) -> dict[str, object] | None:
    """Shape a fetched row as a plain mapping so two reads can be compared.

    Args:
        row: A row fetched from SQLite, or ``None`` when absent.

    Returns:
        The row's columns as a plain dict, or ``None``.

    """
    return None if row is None else dict(row)


def _require(row: dict[str, object] | None, key: str) -> dict[str, object]:
    """Return a pre-operation read, failing if the row was never there.

    This is the guard against a vacuous assertion in either direction: the
    snapshot of a row that does not exist is ``None``, and ``None == None``
    would hold just as well against a table the operation had wiped, while
    "the target is gone" is trivially true of a target that was never stored.

    Args:
        row: The row read back from SQLite before the operation.
        key: The id that was looked up, for the failure message.

    Returns:
        The row, guaranteed non-``None``.

    """
    assert row is not None, f"corpus precondition failed: nothing stored for {key!r}"
    return row


async def _row(db: aiosqlite.Connection, query: str, key: str) -> dict[str, object] | None:
    """Read one row back from an open store connection.

    Args:
        db: The store's own connection.
        query: One of the module-level single-row queries.
        key: The id to look up.

    Returns:
        The row as a plain mapping, or ``None`` when it is gone.

    """
    cursor = await db.execute(query, (key,))
    return _as_mapping(await cursor.fetchone())


async def _id_set(db: aiosqlite.Connection, query: str) -> set[str]:
    """Read a whole id column back from an open store connection.

    Args:
        db: The store's own connection.
        query: One of the module-level id-column queries.

    Returns:
        Every id currently stored in that column.

    """
    cursor = await db.execute(query)
    return {str(row[0]) for row in await cursor.fetchall()}


@contextlib.contextmanager
def _reopen(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Open a throwaway read-back connection to a database file.

    The CLI owns and closes its own connection, so its effects are asserted
    from an independent one over the same file.

    Args:
        db_path: Path to the database the CLI operated on.

    Yields:
        A connection with a row factory, closed on exit.

    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _sync_row(conn: sqlite3.Connection, query: str, key: str) -> dict[str, object] | None:
    """Read one row back over a plain :mod:`sqlite3` connection.

    Args:
        conn: A connection opened by :func:`_reopen`.
        query: One of the module-level single-row queries.
        key: The id to look up.

    Returns:
        The row as a plain mapping, or ``None`` when it is gone.

    """
    return _as_mapping(conn.execute(query, (key,)).fetchone())


def _sync_id_set(conn: sqlite3.Connection, query: str) -> set[str]:
    """Read a whole id column back over a plain :mod:`sqlite3` connection.

    Args:
        conn: A connection opened by :func:`_reopen`.
        query: One of the module-level id-column queries.

    Returns:
        Every id currently stored in that column.

    """
    return {str(row[0]) for row in conn.execute(query).fetchall()}


@contextlib.contextmanager
def _reopen_with_vector_index(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Open a read-back connection that can also read the vec0 index.

    ``embedding_vec`` is a virtual table, so every statement naming it fails
    with ``no such module: vec0`` unless sqlite-vec has been loaded into that
    particular connection first — :func:`_reopen` cannot see the index at all.

    Args:
        db_path: Path to the database the CLI operated on.

    Yields:
        A connection with sqlite-vec loaded and a row factory, closed on exit.

    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        _load_sqlite_vec_sync(conn)
        yield conn
    finally:
        conn.close()


def _sync_vec_rowids(conn: sqlite3.Connection) -> set[int]:
    """Read every rowid currently held in the vec0 index.

    Args:
        conn: A connection opened by :func:`_reopen_with_vector_index`.

    Returns:
        Every rowid in ``embedding_vec``.

    """
    return {int(row[0]) for row in conn.execute(_ALL_VEC_ROWIDS).fetchall()}


def _sync_embedding_rowids(conn: sqlite3.Connection) -> dict[str, int]:
    """Map each thought to the ``embedding`` rowid its vector is indexed under.

    The vec0 index shares the ``embedding`` table's rowid, which is the only
    thing that ties a vector back to a thought.

    Args:
        conn: A connection opened by either reopen helper.

    Returns:
        Thought id -> the rowid of its stored embedding.

    """
    rows = conn.execute(_THOUGHT_EMBEDDING_ROWIDS).fetchall()
    return {str(row["owner_id"]): int(row["rowid"]) for row in rows}


def _purge_that_removes_a_vector_then_fails(
    removed: list[int],
) -> Callable[[aiosqlite.Connection], Awaitable[int]]:
    """Build a stand-in purge that empties one index row and *then* raises.

    A stub that raises on entry proves only that ordinary rows roll back; the
    vec0 table keeps its data in shadow tables of its own, and whether those
    come back is the half worth asserting. Removing a vector first puts the
    index inside the failed transaction, so the read-back afterwards is a real
    test of the rollback rather than of a write that never happened.

    Args:
        removed: Collects the rowid the stub deleted, so a caller can tell a
            rollback that worked from a purge that was never reached.

    Returns:
        A drop-in for the module-level ``purge_orphan_vectors``.

    """

    async def _purge_then_fail(db: aiosqlite.Connection) -> int:
        cursor = await db.execute("SELECT MIN(rowid) FROM embedding_vec")
        row = await cursor.fetchone()
        lowest = None if row is None else row[0]
        assert lowest is not None, "the index was empty when the purge ran"
        await db.execute("DELETE FROM embedding_vec WHERE rowid = ?", (lowest,))
        removed.append(int(lowest))
        msg = "disk I/O error"
        raise sqlite3.OperationalError(msg)

    return _purge_then_fail


# ---------------------------------------------------------------------------
# Corpus builders
# ---------------------------------------------------------------------------


def _make_thought(
    tid: str,
    *,
    expires_at: str | None = None,
    lifecycle_status: LifecycleStatus = LifecycleStatus.ACTIVE,
) -> ThoughtRecord:
    """Build a deterministic thought for a blast-radius corpus."""
    return ThoughtRecord(
        thought_id=tid,
        thought_type=ThoughtType.OBSERVATION,
        essence=f"essence-{tid}",
        content=f"content-{tid}",
        priority=Priority.P3,
        lifecycle_status=lifecycle_status,
        created_cycle=1,
        updated_cycle=1,
        source="test",
        confidence=0.9,
        expires_at=expires_at,
    )


def _make_edge(
    eid: str,
    src: str,
    dst: str,
    *,
    edge_type: EdgeType = EdgeType.ASSOCIATED,
    weight: float = 0.5,
) -> EdgeRecord:
    """Build a deterministic edge for a blast-radius corpus."""
    return EdgeRecord(
        edge_id=eid,
        from_thought_id=src,
        to_thought_id=dst,
        edge_type=edge_type,
        weight=weight,
        created_cycle=1,
        source=KnowledgeSource.EXPERIENCE,
    )


def _make_action(aid: str, src: str) -> ActionRecord:
    """Build a deterministic action for a blast-radius corpus.

    The status is deliberately non-terminal: a terminal one would recompute the
    source thought's ``action_outcome_score``, coupling the parent row to how
    the corpus happens to be ordered. Keeping it out of the picture leaves the
    thought rows a function of the deletion under test and nothing else.
    """
    return ActionRecord(
        action_id=aid,
        source_thought_id=src,
        action_type=ActionType.CLI_OUTPUT,
        intent=f"intent-{aid}",
        status=ActionStatus.PLANNED,
        verification_status=VerificationStatus.PENDING,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def store() -> AsyncIterator[SqliteEngravaCore]:
    """A fresh in-memory core with the default (archive) TTL strategy."""
    async with aiosqlite.connect(":memory:") as db:
        db.row_factory = aiosqlite.Row
        core = SqliteEngravaCore(db=db, embedding_provider=None, auto_embed=False)
        await core.ensure_schema()
        yield core


@pytest.fixture
async def delete_store() -> AsyncIterator[SqliteEngravaCore]:
    """A fresh in-memory core configured with the ``delete`` TTL strategy."""
    async with aiosqlite.connect(":memory:") as db:
        db.row_factory = aiosqlite.Row
        core = SqliteEngravaCore(
            db=db,
            embedding_provider=None,
            auto_embed=False,
            ttl_strategy="delete",
        )
        await core.ensure_schema()
        yield core


@pytest.fixture
def runner() -> CliRunner:
    """A Click test runner for the ``gc`` command."""
    return CliRunner()


# ---------------------------------------------------------------------------
# The mixed-lifecycle corpus
# ---------------------------------------------------------------------------
# Declared once and materialised twice — through the public API onto a head
# schema, and through raw SQL onto a real pre-cascade one. What the two share is
# the **selection-relevant** state: identities, lifecycle statuses, relationships
# and vectors. Those are the fields a deletion predicate can select on and the
# fields these tests assert, so declaring them once is what stops the two
# fixtures describing different scenarios.
#
# The rest deliberately differs, and is not declared here: the API path writes
# cycles, source, confidence, computed content hashes, uuid5 embedding ids, the
# default model name and current timestamps, while the raw v11 path takes the
# schema defaults with ``emb-{tid}``, a fixed model name and a fixed timestamp.
# None of that can change which rows an operation picks.
#
# Rows are interleaved live · archived · live · archived · live. The corpus
# deliberately controls three ordering axes and brackets each doomed row between
# rows that must stay on all three: **id**, **rowid** (which follows the write
# order below) and, for edges, **weight**. On those three the doomed rows are
# additionally **non-adjacent**, so no contiguous range picks exactly them.
#
# Other columns — hashes, timestamps, model names, and on a head schema the
# uuid5 embedding ids — are not corpus-controlled; wherever they happen to
# bracket, that is incidental and nothing here relies on it.
#
# The non-adjacency property deliberately does NOT hold on two columns, and
# neither can be obtained by reordering the corpus:
#
# * ``thought.lifecycle_status`` is the command's own selection axis. Every
#   ARCHIVED row shares the value, so they are contiguous on it by definition —
#   a property of the predicate, not a weakness of the corpus. Selecting the
#   right *kind* of row but too few of them is what the second ARCHIVED thought
#   below is for.
# * ``embedding.vector_blob`` orders by packed float bytes. Interleaving it
#   would mean choosing vectors whose only justification is their memcmp order:
#   unreadable, and silently wrong again after any edit. No deletion predicate
#   orders by a float blob, so the corpus does not chase it.
#
# There are **two** ARCHIVED thoughts rather than one: with a single one,
# "collect the archived thoughts" and "collect the first archived thought"
# produce the same database.

_GC_THOUGHTS: tuple[tuple[str, LifecycleStatus], ...] = (
    ("t-active", LifecycleStatus.ACTIVE),
    ("t-archived", LifecycleStatus.ARCHIVED),
    ("t-kept", LifecycleStatus.ACTIVE),
    ("t-retired", LifecycleStatus.ARCHIVED),
    ("t-vital", LifecycleStatus.ACTIVE),
)

_GC_EDGES: tuple[tuple[str, str, str, EdgeType, float], ...] = (
    ("e-alpha-live", "t-active", "t-kept", EdgeType.ASSOCIATED, 0.2),
    ("e-bravo-archived", "t-archived", "t-active", EdgeType.ASSOCIATED, 0.3),
    ("e-charlie-live", "t-active", "t-kept", EdgeType.DEPENDS_ON, 0.4),
    ("e-delta-archived", "t-kept", "t-archived", EdgeType.ASSOCIATED, 0.5),
    ("e-echo-live", "t-kept", "t-active", EdgeType.ASSOCIATED, 0.6),
    ("e-foxtrot-archived", "t-retired", "t-vital", EdgeType.ASSOCIATED, 0.7),
    ("e-golf-live", "t-vital", "t-kept", EdgeType.ASSOCIATED, 0.8),
)

_GC_EMBEDDINGS: tuple[tuple[str, list[float]], ...] = (
    ("t-active", [0.1, 0.2, 0.3]),
    ("t-archived", [0.7, 0.8, 0.9]),
    ("t-kept", [0.4, 0.5, 0.6]),
    ("t-retired", [0.2, 0.3, 0.4]),
    ("t-vital", [0.5, 0.6, 0.7]),
)

_GC_ACTIONS: tuple[tuple[str, str], ...] = (
    ("a-alpha-live", "t-active"),
    ("a-bravo-archived", "t-archived"),
    ("a-charlie-live", "t-kept"),
    ("a-delta-archived", "t-retired"),
    ("a-echo-live", "t-vital"),
)

#: The schema version before ``ON DELETE CASCADE`` existed. The ``v11 -> v12``
#: migration is what introduced the foreign keys, so at v11 nothing cascades.
_PRE_CASCADE_VERSION = 11

#: Every corpus vector is three-dimensional, so the vec0 index is declared at
#: that width and indexes exactly the corpus the other fixtures materialise.
_VECTOR_DIMENSION = 3

#: The two backends a corpus can be written behind. ``numpy`` leaves the
#: database without a vec0 table at all — the shape most stores have — while
#: ``sqlite-vec`` persists one, which is the shape ``gc`` has to clean up after.
_NO_VECTOR_INDEX = "numpy"
_PERSISTED_VECTOR_INDEX = "sqlite-vec"


async def _materialise_mixed_lifecycle(db_path: Path, *, vector_backend: str) -> None:
    """Write the mixed-lifecycle corpus through the public API.

    Args:
        db_path: Where to create the database.
        vector_backend: ``_NO_VECTOR_INDEX`` or ``_PERSISTED_VECTOR_INDEX`` —
            the store creates and populates ``embedding_vec`` only while a
            sqlite-vec backend is configured.

    """
    conn = await aiosqlite.connect(str(db_path))
    conn.row_factory = aiosqlite.Row
    store = SqliteEngravaCore(db=conn, embedding_provider=None, auto_embed=False)
    await store.ensure_schema()
    await store._configure_vector_backend(
        backend_name=vector_backend,
        embedding_dimension=_VECTOR_DIMENSION,
    )

    for tid, status in _GC_THOUGHTS:
        await store.create_thought(_make_thought(tid, lifecycle_status=status))
    for eid, src, dst, edge_type, weight in _GC_EDGES:
        await store.create_edge(_make_edge(eid, src, dst, edge_type=edge_type, weight=weight))
    for tid, vector in _GC_EMBEDDINGS:
        await store.store_embedding(tid, vector)
    for aid, src in _GC_ACTIONS:
        await store.create_action(_make_action(aid, src))

    await conn.commit()
    await conn.close()


def _assert_index_mirrors_corpus(db_path: Path) -> None:
    """Fail unless the vec0 index holds exactly one row per stored embedding.

    A sqlite-vec backend that cannot load its extension falls back to numpy
    *silently*, leaving a database with no index and no vectors. Every
    assertion about what such an index still holds would then pass vacuously —
    "nothing was left behind" is trivially true of a table that was never
    created. The fixtures therefore refuse to hand one over.

    Args:
        db_path: The database just materialised.

    """
    with _reopen_with_vector_index(db_path) as conn:
        indexed = _sync_vec_rowids(conn)
        stored = set(_sync_embedding_rowids(conn).values())
        assert stored, "fixture precondition failed: no embeddings were stored"
        assert indexed == stored, "fixture precondition failed: the vec0 index is not the corpus"


@pytest.fixture
def mixed_lifecycle_db(tmp_path: Path) -> Path:
    """Materialise the mixed-lifecycle corpus on a head schema via the public API.

    Returns:
        Path to the populated database file.

    """
    db_path = tmp_path / "mixed-lifecycle.db"
    asyncio.run(_materialise_mixed_lifecycle(db_path, vector_backend=_NO_VECTOR_INDEX))
    return db_path


@pytest.fixture
def vec_indexed_mixed_lifecycle_db(tmp_path: Path) -> Path:
    """Materialise the same corpus behind a persisted vec0 index.

    Returns:
        Path to the populated database file, carrying an ``embedding_vec``
        table with one vector per stored embedding.

    """
    db_path = tmp_path / "mixed-lifecycle-vec.db"
    asyncio.run(_materialise_mixed_lifecycle(db_path, vector_backend=_PERSISTED_VECTOR_INDEX))
    _assert_index_mirrors_corpus(db_path)
    return db_path


@pytest.fixture
def pre_cascade_mixed_lifecycle_db(tmp_path: Path) -> Path:
    """Materialise the same corpus on a real pre-cascade (core-11) schema.

    ``gc`` opens its database through the CLI's plain connection helper, which
    sets PRAGMAs and **never migrates** — ``ensure_schema`` is reached only by
    ``restore`` and by the ``migrate`` command. A database last written before
    the ``v11 -> v12`` migration added the ``ON DELETE CASCADE`` foreign keys is
    therefore a shape ``engrava gc`` can genuinely be pointed at, and there
    ``PRAGMA foreign_keys = ON`` has no constraints to enforce.

    That is what makes this fixture necessary rather than redundant. On a head
    schema the parent delete cascades, so the three child deletes in
    ``_gc_archived`` cannot be observed to do anything: neutering them entirely
    leaves the resulting database unchanged. Here they are the *only* thing that
    removes an archived thought's edges, embeddings and actions.

    The schema is stamped by the migration suite's real-shape builder rather
    than hand-rolled, so it cannot drift from the shape v11 actually shipped.

    Returns:
        Path to the populated database file.

    """
    db_path = tmp_path / "pre-cascade.db"

    async def _setup() -> None:
        conn = await aiosqlite.connect(str(db_path))
        conn.row_factory = aiosqlite.Row
        await _bootstrap_core_at_version(conn, _PRE_CASCADE_VERSION)

        for tid, status in _GC_THOUGHTS:
            await conn.execute(
                "INSERT INTO thought (thought_id, thought_type, essence, content, "
                "priority, lifecycle_status) VALUES (?, 'OBSERVATION', ?, ?, 'P3', ?)",
                (tid, f"essence-{tid}", f"content-{tid}", status.value),
            )
        for eid, src, dst, edge_type, weight in _GC_EDGES:
            await conn.execute(
                "INSERT INTO edge (edge_id, from_thought_id, to_thought_id, edge_type, weight) "
                "VALUES (?, ?, ?, ?, ?)",
                (eid, src, dst, edge_type.value, weight),
            )
        for tid, vector in _GC_EMBEDDINGS:
            await conn.execute(
                "INSERT INTO embedding (embedding_id, owner_type, owner_id, model_name, "
                "dimension, vector_blob, created_at) "
                "VALUES (?, 'THOUGHT', ?, 'test-model', ?, ?, '2026-01-01T00:00:00+00:00')",
                (
                    f"emb-{tid}",
                    tid,
                    len(vector),
                    struct.pack(f"{len(vector)}f", *vector),
                ),
            )
        for aid, src in _GC_ACTIONS:
            await conn.execute(
                "INSERT INTO action (action_id, source_thought_id, action_type, intent) "
                "VALUES (?, ?, 'CLI_OUTPUT', ?)",
                (aid, src, f"intent-{aid}"),
            )

        await conn.commit()
        await conn.close()

    asyncio.run(_setup())
    return db_path


# ---------------------------------------------------------------------------
# delete_edge
# ---------------------------------------------------------------------------


class TestDeleteEdgeBlastRadius:
    """``delete_edge`` removes one edge, never the edge table."""

    #: Edges that must outlive a ``delete_edge("e-target")`` call — see
    #: :meth:`_seed` for how they bracket the target on every ordering axis.
    _SURVIVORS = ("e-adjacent-source", "e-reversed-pair", "e-variant-type")

    #: Thoughts the edge delete has no business touching either.
    _ENDPOINTS = ("t1", "t2", "t3")

    @staticmethod
    async def _seed(store: SqliteEngravaCore) -> None:
        """Seed a graph whose other edges overlap the target on every key.

        Each survivor shares something with ``e-target`` — its source thought,
        its endpoint pair, or its endpoint pair under another type — so a
        predicate widened along any of those axes, not only to the whole
        table, takes an edge it must not touch.

        The target is also **bracketed on every axis the table can be ordered
        by**, with a survivor sorting below it and one above:

        =============== ====================================================
        axis            order (survivor · **target** · survivor)
        =============== ====================================================
        ``edge_id``     adjacent-source · reversed-pair · **target** · variant-type
        ``rowid``       adjacent-source · **target** · reversed-pair · variant-type
        ``weight``      0.4 · **0.6** · 0.7 · 0.8
        =============== ====================================================

        Without that, a predicate that picks a row by *ordering* rather than by
        identity — ``ORDER BY <anything> LIMIT 1`` — could land on the correct
        row for the wrong reason and never be caught. ``rowid`` follows
        insertion order, which is why the target is created second.
        """
        for tid in ("t1", "t2", "t3"):
            await store.create_thought(_make_thought(tid))
        await store.create_edge(_make_edge("e-adjacent-source", "t1", "t3", weight=0.4))
        await store.create_edge(_make_edge("e-target", "t1", "t2", weight=0.6))
        await store.create_edge(_make_edge("e-reversed-pair", "t2", "t1", weight=0.7))
        await store.create_edge(
            _make_edge("e-variant-type", "t1", "t2", edge_type=EdgeType.DEPENDS_ON, weight=0.8),
        )

    async def test_delete_edge_removes_only_the_target(self, store: SqliteEngravaCore) -> None:
        """The target edge is gone; every other edge is still there, unchanged."""
        await self._seed(store)
        _require(await _row(store._db, _EDGE_BY_ID, "e-target"), "e-target")
        snapshots = {
            eid: _require(await _row(store._db, _EDGE_BY_ID, eid), eid) for eid in self._SURVIVORS
        }
        endpoints = {
            tid: _require(await _row(store._db, _THOUGHT_BY_ID, tid), tid)
            for tid in self._ENDPOINTS
        }

        deleted = await store.delete_edge("e-target")

        assert await _row(store._db, _EDGE_BY_ID, "e-target") is None
        assert await _id_set(store._db, _ALL_EDGE_IDS) == set(self._SURVIVORS)
        for eid, before in snapshots.items():
            assert await _row(store._db, _EDGE_BY_ID, eid) == before, f"edge {eid} was modified"
        assert await _id_set(store._db, _ALL_THOUGHT_IDS) == set(self._ENDPOINTS)
        for tid, before in endpoints.items():
            assert await _row(store._db, _THOUGHT_BY_ID, tid) == before, f"thought {tid} changed"
        # Only once the stored state is settled does the reported outcome matter.
        assert deleted is True

    async def test_delete_edge_miss_leaves_every_edge_intact(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """Deleting an id that is not stored removes nothing at all."""
        await self._seed(store)
        stored = ("e-target", *self._SURVIVORS)
        snapshots = {eid: _require(await _row(store._db, _EDGE_BY_ID, eid), eid) for eid in stored}

        deleted = await store.delete_edge("no-such-edge")

        assert await _id_set(store._db, _ALL_EDGE_IDS) == set(stored)
        for eid, before in snapshots.items():
            assert await _row(store._db, _EDGE_BY_ID, eid) == before, f"edge {eid} was modified"
        assert deleted is False


# ---------------------------------------------------------------------------
# cleanup_expired — delete strategy
# ---------------------------------------------------------------------------


class TestCleanupExpiredDeleteBlastRadius:
    """The ``delete`` TTL sweep removes the expired thoughts, not the table."""

    _EARLY_PAST = "2026-01-01T00:00:00+00:00"
    _LATE_PAST = "2026-03-01T00:00:00+00:00"
    _NOW = "2026-06-01T00:00:00+00:00"
    _NEAR_FUTURE = "2027-01-01T00:00:00+00:00"
    _FAR_FUTURE = "2028-01-01T00:00:00+00:00"

    #: Thoughts the sweep must leave alone: one with no TTL at all and two
    #: still inside theirs. They sort either side of both expired thoughts.
    _SURVIVORS = ("t-eternal", "t-fresh", "t-young")

    #: The two thoughts the sweep must remove. **Two**, deliberately: with only
    #: one, "delete the expired thoughts" and "delete the first expired thought"
    #: are indistinguishable, and a ``LIMIT``-shaped regression that half-empties
    #: the store passes. They are non-adjacent on every axis, so no contiguous
    #: range selects exactly them either.
    _DOOMED = ("t-expired", "t-lapsed")

    #: Edges that must outlive the sweep — interleaved with the doomed pair.
    _SURVIVING_EDGES = ("e-alpha-live", "e-charlie-live", "e-echo-live")

    #: Edges the cascade must take with their expired parent.
    _DOOMED_EDGES = ("e-bravo-doomed", "e-delta-doomed")

    #: Actions that must outlive the sweep — interleaved with the doomed pair.
    _SURVIVING_ACTIONS = ("a-alpha-live", "a-charlie-live", "a-echo-live")

    #: Actions the cascade must take with their expired parent.
    _DOOMED_ACTIONS = ("a-bravo-doomed", "a-delta-doomed")

    @classmethod
    async def _seed(cls, store: SqliteEngravaCore) -> None:
        """Seed two expired thoughts and three survivors, each carrying children.

        Rows are interleaved survivor · **doomed** · survivor · **doomed** ·
        survivor. The corpus deliberately controls four ordering axes and
        brackets each row that must disappear between rows that must stay on
        every one of them — **id**, **rowid**, ``edge.weight`` and
        ``thought.expires_at``:

        * ``thought``: ``t-eternal`` · **``t-expired``** · ``t-fresh`` ·
          **``t-lapsed``** · ``t-young``, matched by ``rowid``; ``expires_at``
          NULL · early past · late past · near future · far future (SQLite sorts
          NULL first), where the doomed pair is neither the lowest nor the
          highest two.
        * ``edge``: ``e-alpha-live`` · **``e-bravo-doomed``** ·
          ``e-charlie-live`` · **``e-delta-doomed``** · ``e-echo-live``, matched
          by ``rowid`` and by ``weight`` (0.2 · 0.3 · 0.4 · 0.5 · 0.6).
        * ``embedding``: ``owner_id`` and ``rowid`` follow the thought order.
        * ``action``: ``a-alpha-live`` · **``a-bravo-doomed``** ·
          ``a-charlie-live`` · **``a-delta-doomed``** · ``a-echo-live``, matched
          by ``rowid``.

        Columns the corpus does not control — content hashes, timestamps, model
        names, embedding ids — are left to whatever the write path produces;
        wherever they happen to bracket, that is incidental and nothing here
        relies on it.

        On the three identity and structural axes (id, ``rowid``, ``weight``)
        the doomed rows are additionally **non-adjacent**, so no contiguous
        range picks exactly them. That stronger property does not hold on two
        columns, and neither can be obtained by reordering:

        * ``thought.expires_at`` is the sweep's own selection axis. A doomed row
          has ``expires_at <= now`` and a survivor a later one or NULL, so
          survivors necessarily sort entirely before (NULL) or entirely after
          the doomed pair. Their contiguity there is forced by the predicate,
          not chosen by the corpus.
        * ``embedding.vector_blob`` orders by packed float bytes. Interleaving
          it would mean picking vectors whose only justification is their memcmp
          order: unreadable, and silently wrong again after any edit. No
          deletion predicate orders by a float blob.

        A predicate that selects by one of the controlled orderings instead of
        by expiry therefore cannot pick the right rows by coincidence, and one
        that selects the right *kind* of row but too few of them leaves a
        survivor set the tests compare exactly. Rows the seed leaves at equal
        values on some other column (``created_cycle``, ``confidence``) are ties
        that SQLite resolves in ``rowid`` order, which is interleaved.
        ``rowid`` follows insertion order, hence the write order below.
        """
        await store.create_thought(_make_thought("t-eternal"))
        await store.create_thought(_make_thought("t-expired", expires_at=cls._EARLY_PAST))
        await store.create_thought(_make_thought("t-fresh", expires_at=cls._NEAR_FUTURE))
        await store.create_thought(_make_thought("t-lapsed", expires_at=cls._LATE_PAST))
        await store.create_thought(_make_thought("t-young", expires_at=cls._FAR_FUTURE))

        await store.create_edge(_make_edge("e-alpha-live", "t-eternal", "t-fresh", weight=0.2))
        await store.create_edge(_make_edge("e-bravo-doomed", "t-expired", "t-fresh", weight=0.3))
        await store.create_edge(_make_edge("e-charlie-live", "t-fresh", "t-eternal", weight=0.4))
        await store.create_edge(_make_edge("e-delta-doomed", "t-lapsed", "t-eternal", weight=0.5))
        await store.create_edge(_make_edge("e-echo-live", "t-young", "t-eternal", weight=0.6))

        for tid, vector in (
            ("t-eternal", [0.7, 0.8, 0.9]),
            ("t-expired", [0.1, 0.2, 0.3]),
            ("t-fresh", [0.4, 0.5, 0.6]),
            ("t-lapsed", [0.2, 0.3, 0.4]),
            ("t-young", [0.5, 0.6, 0.7]),
        ):
            await store.store_embedding(tid, vector)

        await store.create_action(_make_action("a-alpha-live", "t-eternal"))
        await store.create_action(_make_action("a-bravo-doomed", "t-expired"))
        await store.create_action(_make_action("a-charlie-live", "t-fresh"))
        await store.create_action(_make_action("a-delta-doomed", "t-lapsed"))
        await store.create_action(_make_action("a-echo-live", "t-young"))

    async def test_sweep_removes_only_the_expired_thought(
        self,
        delete_store: SqliteEngravaCore,
    ) -> None:
        """Every expired thought is gone; the unexpired ones are stored unchanged."""
        await self._seed(delete_store)
        for tid in self._DOOMED:
            _require(await _row(delete_store._db, _THOUGHT_BY_ID, tid), tid)
        snapshots = {
            tid: _require(await _row(delete_store._db, _THOUGHT_BY_ID, tid), tid)
            for tid in self._SURVIVORS
        }

        result = await delete_store.cleanup_expired(now=self._NOW)

        for tid in self._DOOMED:
            assert await _row(delete_store._db, _THOUGHT_BY_ID, tid) is None, f"{tid} not swept"
        assert await _id_set(delete_store._db, _ALL_THOUGHT_IDS) == set(self._SURVIVORS)
        for tid, before in snapshots.items():
            after = await _row(delete_store._db, _THOUGHT_BY_ID, tid)
            assert after == before, f"thought {tid} was modified"
        # Only once the stored state is settled does the reported count matter.
        assert result.expired_count == len(self._DOOMED)

    async def test_sweep_keeps_the_children_of_surviving_thoughts(
        self,
        delete_store: SqliteEngravaCore,
    ) -> None:
        """The cascade follows the expired thoughts only, and stops there."""
        await self._seed(delete_store)
        db = delete_store._db
        doomed_children: tuple[tuple[str, str], ...] = (
            *((eid, _EDGE_BY_ID) for eid in self._DOOMED_EDGES),
            *((tid, _EMBEDDING_BY_OWNER) for tid in self._DOOMED),
            *((aid, _ACTION_BY_ID) for aid in self._DOOMED_ACTIONS),
        )
        for doomed, query in doomed_children:
            _require(await _row(db, query, doomed), doomed)
        edges_before = {
            eid: _require(await _row(db, _EDGE_BY_ID, eid), eid) for eid in self._SURVIVING_EDGES
        }
        embeddings_before = {
            tid: _require(await _row(db, _EMBEDDING_BY_OWNER, tid), tid) for tid in self._SURVIVORS
        }
        actions_before = {
            aid: _require(await _row(db, _ACTION_BY_ID, aid), aid)
            for aid in self._SURVIVING_ACTIONS
        }

        await delete_store.cleanup_expired(now=self._NOW)

        assert await _id_set(db, _ALL_EDGE_IDS) == set(self._SURVIVING_EDGES)
        assert await _id_set(db, _ALL_EMBEDDING_OWNERS) == set(self._SURVIVORS)
        assert await _id_set(db, _ALL_ACTION_IDS) == set(self._SURVIVING_ACTIONS)
        for eid, before in edges_before.items():
            assert await _row(db, _EDGE_BY_ID, eid) == before, f"edge {eid} was modified"
        for tid, before in embeddings_before.items():
            assert await _row(db, _EMBEDDING_BY_OWNER, tid) == before, f"embedding {tid} changed"
        for aid, before in actions_before.items():
            assert await _row(db, _ACTION_BY_ID, aid) == before, f"action {aid} was modified"


# ---------------------------------------------------------------------------
# engrava gc
# ---------------------------------------------------------------------------


class TestGcArchivedBlastRadius:
    """``engrava gc`` collects the ARCHIVED subtree, never a live thought's children."""

    #: Thoughts ``gc`` must leave in place, interleaved with the archived pair.
    _SURVIVORS = ("t-active", "t-kept", "t-vital")

    #: The two thoughts ``gc`` must collect. **Two**, so that collecting only
    #: the first of them is distinguishable from collecting them all.
    _DOOMED = ("t-archived", "t-retired")

    #: Edges ``gc`` must leave in place, interleaved with the archived edges so
    #: no contiguous range selects exactly the doomed ones.
    _SURVIVING_EDGES = ("e-alpha-live", "e-charlie-live", "e-echo-live", "e-golf-live")

    #: Edges ``gc`` must remove — every edge touching an ARCHIVED thought.
    _DOOMED_EDGES = ("e-bravo-archived", "e-delta-archived", "e-foxtrot-archived")

    #: Actions ``gc`` must leave in place, interleaved with the archived ones.
    _SURVIVING_ACTIONS = ("a-alpha-live", "a-charlie-live", "a-echo-live")

    #: Actions ``gc`` must remove — one per ARCHIVED thought.
    _DOOMED_ACTIONS = ("a-bravo-archived", "a-delta-archived")

    @staticmethod
    def _collect(runner: CliRunner, db_path: Path) -> None:
        """Run ``engrava gc`` over the corpus, asserting only that it ran.

        Deliberately does **not** assert the ``Collected N`` line: that is the
        command's own report of what it did, and checking it here would abort
        the test before a single row is read back — the very substitution of
        return value for state that these tests exist to rule out. Whether the
        collection happened at all is settled by the state assertions, which
        fail just as loudly on a no-op as on an over-broad delete.
        """
        result = runner.invoke(cli, ["--db", str(db_path), "gc"])
        assert result.exit_code == 0, result.output

    def test_gc_removes_every_archived_thought_and_nothing_else(
        self,
        runner: CliRunner,
        mixed_lifecycle_db: Path,
    ) -> None:
        """Both ARCHIVED thoughts are collected; the ACTIVE ones survive unchanged."""
        with _reopen(mixed_lifecycle_db) as conn:
            for doomed in self._DOOMED:
                _require(_sync_row(conn, _THOUGHT_BY_ID, doomed), doomed)
            snapshots = {
                tid: _require(_sync_row(conn, _THOUGHT_BY_ID, tid), tid) for tid in self._SURVIVORS
            }

        self._collect(runner, mixed_lifecycle_db)

        with _reopen(mixed_lifecycle_db) as conn:
            for doomed in self._DOOMED:
                assert _sync_row(conn, _THOUGHT_BY_ID, doomed) is None, f"{doomed} not collected"
            assert _sync_id_set(conn, _ALL_THOUGHT_IDS) == set(self._SURVIVORS)
            for tid, before in snapshots.items():
                assert _sync_row(conn, _THOUGHT_BY_ID, tid) == before, f"thought {tid} changed"

    def test_gc_keeps_edges_between_active_thoughts(
        self,
        runner: CliRunner,
        mixed_lifecycle_db: Path,
    ) -> None:
        """Every edge touching an ARCHIVED thought goes; the live ones stay."""
        with _reopen(mixed_lifecycle_db) as conn:
            for doomed in self._DOOMED_EDGES:
                _require(_sync_row(conn, _EDGE_BY_ID, doomed), doomed)
            snapshots = {
                eid: _require(_sync_row(conn, _EDGE_BY_ID, eid), eid)
                for eid in self._SURVIVING_EDGES
            }

        self._collect(runner, mixed_lifecycle_db)

        with _reopen(mixed_lifecycle_db) as conn:
            assert _sync_id_set(conn, _ALL_EDGE_IDS) == set(self._SURVIVING_EDGES)
            for eid, before in snapshots.items():
                assert _sync_row(conn, _EDGE_BY_ID, eid) == before, f"edge {eid} was modified"

    def test_gc_keeps_embeddings_of_active_thoughts(
        self,
        runner: CliRunner,
        mixed_lifecycle_db: Path,
    ) -> None:
        """Every ARCHIVED thought's embedding goes; the live ones stay."""
        with _reopen(mixed_lifecycle_db) as conn:
            for doomed in self._DOOMED:
                _require(_sync_row(conn, _EMBEDDING_BY_OWNER, doomed), doomed)
            snapshots = {
                tid: _require(_sync_row(conn, _EMBEDDING_BY_OWNER, tid), tid)
                for tid in self._SURVIVORS
            }

        self._collect(runner, mixed_lifecycle_db)

        with _reopen(mixed_lifecycle_db) as conn:
            assert _sync_id_set(conn, _ALL_EMBEDDING_OWNERS) == set(self._SURVIVORS)
            for tid, before in snapshots.items():
                after = _sync_row(conn, _EMBEDDING_BY_OWNER, tid)
                assert after == before, f"embedding of {tid} was modified"

    def test_gc_keeps_actions_of_active_thoughts(
        self,
        runner: CliRunner,
        mixed_lifecycle_db: Path,
    ) -> None:
        """Every ARCHIVED thought's action goes; the live ones stay."""
        with _reopen(mixed_lifecycle_db) as conn:
            for doomed in self._DOOMED_ACTIONS:
                _require(_sync_row(conn, _ACTION_BY_ID, doomed), doomed)
            snapshots = {
                aid: _require(_sync_row(conn, _ACTION_BY_ID, aid), aid)
                for aid in self._SURVIVING_ACTIONS
            }

        self._collect(runner, mixed_lifecycle_db)

        with _reopen(mixed_lifecycle_db) as conn:
            assert _sync_id_set(conn, _ALL_ACTION_IDS) == set(self._SURVIVING_ACTIONS)
            for aid, before in snapshots.items():
                assert _sync_row(conn, _ACTION_BY_ID, aid) == before, f"action {aid} was modified"

    def test_gc_collects_the_archived_subtree_without_a_cascade(
        self,
        runner: CliRunner,
        pre_cascade_mixed_lifecycle_db: Path,
    ) -> None:
        """On a pre-cascade schema the child deletes are the only thing that runs.

        ``gc`` never migrates, so a database written before the ``v11 -> v12``
        foreign keys existed is a shape it can be pointed at. There the parent
        delete cannot cascade, and the three child statements in
        ``_gc_archived`` are solely responsible for removing an archived
        thought's edges, embeddings and actions — the same statements whose
        effect is entirely masked on a head schema.

        The blast radius is asserted in both directions on all three child
        tables, exactly as on a head schema.
        """
        with _reopen(pre_cascade_mixed_lifecycle_db) as conn:
            # Corpus precondition: this really is a pre-cascade schema, so a
            # cascade cannot be doing the work the child deletes are credited
            # with. Without it the fixture could silently become a second head
            # -schema test and prove nothing new.
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            assert version == _PRE_CASCADE_VERSION
            for table in ("edge", "embedding", "action"):
                foreign_keys = conn.execute(f"PRAGMA foreign_key_list({table})").fetchall()
                assert foreign_keys == [], f"{table} already carries a cascade at v{version}"

            for doomed, query in (
                *((eid, _EDGE_BY_ID) for eid in self._DOOMED_EDGES),
                *((tid, _EMBEDDING_BY_OWNER) for tid in self._DOOMED),
                *((aid, _ACTION_BY_ID) for aid in self._DOOMED_ACTIONS),
            ):
                _require(_sync_row(conn, query, doomed), doomed)
            edges_before = {
                eid: _require(_sync_row(conn, _EDGE_BY_ID, eid), eid)
                for eid in self._SURVIVING_EDGES
            }
            embeddings_before = {
                tid: _require(_sync_row(conn, _EMBEDDING_BY_OWNER, tid), tid)
                for tid in self._SURVIVORS
            }
            actions_before = {
                aid: _require(_sync_row(conn, _ACTION_BY_ID, aid), aid)
                for aid in self._SURVIVING_ACTIONS
            }

        self._collect(runner, pre_cascade_mixed_lifecycle_db)

        with _reopen(pre_cascade_mixed_lifecycle_db) as conn:
            assert _sync_id_set(conn, _ALL_THOUGHT_IDS) == set(self._SURVIVORS)
            assert _sync_id_set(conn, _ALL_EDGE_IDS) == set(self._SURVIVING_EDGES)
            assert _sync_id_set(conn, _ALL_EMBEDDING_OWNERS) == set(self._SURVIVORS)
            assert _sync_id_set(conn, _ALL_ACTION_IDS) == set(self._SURVIVING_ACTIONS)
            for eid, before in edges_before.items():
                assert _sync_row(conn, _EDGE_BY_ID, eid) == before, f"edge {eid} was modified"
            for tid, before in embeddings_before.items():
                after = _sync_row(conn, _EMBEDDING_BY_OWNER, tid)
                assert after == before, f"embedding of {tid} was modified"
            for aid, before in actions_before.items():
                assert _sync_row(conn, _ACTION_BY_ID, aid) == before, f"action {aid} was modified"

    @sqlite_vec_required
    def test_gc_purges_only_the_collected_thoughts_vectors(
        self,
        runner: CliRunner,
        vec_indexed_mixed_lifecycle_db: Path,
    ) -> None:
        """The collected thoughts' vectors leave the index; the survivors' stay.

        ``embedding_vec`` is reached by no foreign key, so the parent delete
        cannot take a vector with it: unless the command removes it itself, the
        vector stays in the index for a thought that no longer exists.

        The corpus is interleaved live · archived · live · archived · live on
        ``rowid``, the only key the index is addressed by, so the doomed rowids
        are bracketed by surviving ones and are not adjacent — no purge that
        picks by rowid position can land on exactly them. Set equality, not
        containment: containment would hold just as well for a purge that had
        emptied the index, and this is the one store where that would go
        unnoticed by any foreign key or ``embedding`` read-back.
        """
        with _reopen_with_vector_index(vec_indexed_mixed_lifecycle_db) as conn:
            rowids = _sync_embedding_rowids(conn)
            assert set(rowids) == {*self._DOOMED, *self._SURVIVORS}, (
                f"corpus precondition failed: embeddings stored for {sorted(rowids)}"
            )
            written = [rowids[tid] for tid, _vector in _GC_EMBEDDINGS]
            assert written == sorted(written), (
                "corpus precondition failed: rowids do not follow the write order"
            )
            doomed = {rowids[tid] for tid in self._DOOMED}
            surviving = {rowids[tid] for tid in self._SURVIVORS}
            assert _sync_vec_rowids(conn) == doomed | surviving

        self._collect(runner, vec_indexed_mixed_lifecycle_db)

        with _reopen_with_vector_index(vec_indexed_mixed_lifecycle_db) as conn:
            remaining = _sync_vec_rowids(conn)
            assert not doomed & remaining, "a collected thought's vector survived"
            assert remaining == surviving
            assert _sync_id_set(conn, _ALL_THOUGHT_IDS) == set(self._SURVIVORS)

    @sqlite_vec_required
    def test_gc_dry_run_leaves_every_vector_in_the_index(
        self,
        runner: CliRunner,
        vec_indexed_mixed_lifecycle_db: Path,
    ) -> None:
        """``--dry-run`` reports the collection and leaves every vector stored.

        The purge is a write like any other, so it owes the same guarantee as
        the row deletes it follows. Every rowid is asserted, the doomed ones
        included: under ``--dry-run`` they are non-targets too.

        What this pins is the **committed** state. The dry-run path reaches no
        ``commit``, so a purge wrongly attempted there would still be rolled
        back when the command closes its connection; the test that no purge is
        attempted at all is the one below, which withholds the module and
        expects the command to run anyway.
        """
        with _reopen_with_vector_index(vec_indexed_mixed_lifecycle_db) as conn:
            before = _sync_vec_rowids(conn)
            assert before, "corpus precondition failed: the index is empty"

        result = runner.invoke(
            cli, ["--db", str(vec_indexed_mixed_lifecycle_db), "gc", "--dry-run"]
        )
        assert result.exit_code == 0, result.output

        with _reopen_with_vector_index(vec_indexed_mixed_lifecycle_db) as conn:
            assert _sync_vec_rowids(conn) == before
            assert _sync_id_set(conn, _ALL_THOUGHT_IDS) == {*self._DOOMED, *self._SURVIVORS}
        assert "Would delete 2 archived thoughts" in result.output

    @sqlite_vec_required
    def test_dry_run_does_not_need_sqlite_vec_on_an_indexed_store(
        self,
        runner: CliRunner,
        vec_indexed_mixed_lifecycle_db: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A run that deletes nothing is never refused for want of the module.

        The refusal above exists because a collection that cannot reach the
        index would strand vectors. A dry run deletes nothing, so it strands
        nothing, and demanding the optional extra to *preview* a collection
        would be a refusal with no guarantee behind it. Withholding the module
        is what makes this observable — without it the command loads sqlite-vec
        successfully and the question is never asked.
        """

        async def _refuse_to_load(db: object) -> bool:
            del db
            return False

        monkeypatch.setattr(
            "engrava.extensions.vector_sqlite_vec.load_sqlite_vec",
            _refuse_to_load,
        )
        with _reopen_with_vector_index(vec_indexed_mixed_lifecycle_db) as conn:
            vectors_before = _sync_vec_rowids(conn)
            assert vectors_before, "corpus precondition failed: the index is empty"

        result = runner.invoke(
            cli, ["--db", str(vec_indexed_mixed_lifecycle_db), "gc", "--dry-run"]
        )
        assert result.exit_code == 0, result.output

        with _reopen_with_vector_index(vec_indexed_mixed_lifecycle_db) as conn:
            assert _sync_vec_rowids(conn) == vectors_before
            assert _sync_id_set(conn, _ALL_THOUGHT_IDS) == {*self._DOOMED, *self._SURVIVORS}
        assert "Would delete 2 archived thoughts" in result.output

    def test_gc_without_a_vector_index_collects_and_creates_none(
        self,
        runner: CliRunner,
        mixed_lifecycle_db: Path,
    ) -> None:
        """A store that never carried a vec0 index collects exactly as before.

        Most stores have no ``embedding_vec`` table, and on those the purge has
        nothing to address: it must be a no-op rather than an error, and must
        not bring the index into existence. No sqlite-vec is needed to assert
        either — ``sqlite_master`` is an ordinary table.
        """
        with _reopen(mixed_lifecycle_db) as conn:
            assert conn.execute(_VEC_INDEX_IN_SCHEMA).fetchone() is None, (
                "corpus precondition failed: this fixture carries a vec0 index"
            )

        result = runner.invoke(cli, ["--db", str(mixed_lifecycle_db), "gc"])
        # An execution precondition, not the command's account of itself: it
        # fires only when there was no effect to read back at all.
        assert result.exit_code == 0, result.output

        with _reopen(mixed_lifecycle_db) as conn:
            assert _sync_id_set(conn, _ALL_THOUGHT_IDS) == set(self._SURVIVORS)
            assert _sync_id_set(conn, _ALL_EMBEDDING_OWNERS) == set(self._SURVIVORS)
            assert conn.execute(_VEC_INDEX_IN_SCHEMA).fetchone() is None

    @sqlite_vec_required
    def test_gc_refuses_to_collect_when_the_index_cannot_be_purged(
        self,
        runner: CliRunner,
        vec_indexed_mixed_lifecycle_db: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Without sqlite-vec loaded the command collects nothing at all.

        A vec0 table is unwritable — unreadable, even — on a connection that
        has not loaded the module, so a store carrying the index cannot be
        collected without it. Continuing anyway would delete the rows and leave
        their vectors stranded until something else reconciles the index — and
        permanently, in the vector's own terms, once SQLite hands the freed
        rowid to a new embedding, after which the stale vector reads as owned
        and no reconciliation can tell. The command therefore refuses
        **before** it deletes anything: the store is asserted to be exactly as
        it was, and only then the raised error.
        """

        async def _refuse_to_load(db: object) -> bool:
            del db
            return False

        monkeypatch.setattr(
            "engrava.extensions.vector_sqlite_vec.load_sqlite_vec",
            _refuse_to_load,
        )
        with _reopen_with_vector_index(vec_indexed_mixed_lifecycle_db) as conn:
            vectors_before = _sync_vec_rowids(conn)
            thoughts_before = _sync_id_set(conn, _ALL_THOUGHT_IDS)
            assert thoughts_before == {*self._DOOMED, *self._SURVIVORS}

        result = runner.invoke(
            cli,
            ["--db", str(vec_indexed_mixed_lifecycle_db), "gc"],
            standalone_mode=False,
        )

        with _reopen_with_vector_index(vec_indexed_mixed_lifecycle_db) as conn:
            assert _sync_id_set(conn, _ALL_THOUGHT_IDS) == thoughts_before
            assert _sync_vec_rowids(conn) == vectors_before
        assert isinstance(result.exception, click.ClickException), result.exception
        assert "engrava[vec]" in str(result.exception)

    @sqlite_vec_required
    def test_gc_collects_nothing_when_the_purge_itself_fails(
        self,
        runner: CliRunner,
        vec_indexed_mixed_lifecycle_db: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A purge that fails takes the whole collection down with it.

        The purge runs last, once the rows it cleans up after are already
        deleted. Were it outside their transaction, failing there would leave
        behind precisely the stranded vectors it exists to remove — the
        ``pytest.raises``-shaped hole where an operation validates after it has
        written. The deletes and the purge therefore share one transaction, and
        what proves it is the stored state after the failure, not the error.

        The stub removes a vector *before* it raises, so the rollback under
        test is the one that has to reach the vec0 table's own shadow storage,
        not merely the ordinary rows. Every canonical table is read back: a
        transaction that gave up half way would show there first.
        """
        removed_by_the_stub: list[int] = []
        monkeypatch.setattr(
            "engrava.extensions.vector_sqlite_vec.purge_orphan_vectors",
            _purge_that_removes_a_vector_then_fails(removed_by_the_stub),
        )
        with _reopen_with_vector_index(vec_indexed_mixed_lifecycle_db) as conn:
            vectors_before = _sync_vec_rowids(conn)
            thoughts_before = _sync_id_set(conn, _ALL_THOUGHT_IDS)
            edges_before = _sync_id_set(conn, _ALL_EDGE_IDS)
            actions_before = _sync_id_set(conn, _ALL_ACTION_IDS)
            assert thoughts_before == {*self._DOOMED, *self._SURVIVORS}

        result = runner.invoke(
            cli,
            ["--db", str(vec_indexed_mixed_lifecycle_db), "gc"],
            standalone_mode=False,
        )
        # An instrumentation precondition: without it, a purge that was never
        # reached would read as a rollback that worked.
        assert removed_by_the_stub, "the purge never reached the index"

        with _reopen_with_vector_index(vec_indexed_mixed_lifecycle_db) as conn:
            assert _sync_id_set(conn, _ALL_THOUGHT_IDS) == thoughts_before
            assert _sync_id_set(conn, _ALL_EDGE_IDS) == edges_before
            assert _sync_id_set(conn, _ALL_EMBEDDING_OWNERS) == thoughts_before
            assert _sync_id_set(conn, _ALL_ACTION_IDS) == actions_before
            assert _sync_vec_rowids(conn) == vectors_before
        assert isinstance(result.exception, sqlite3.OperationalError), result.exception


# ---------------------------------------------------------------------------
# engrava gc --expired
# ---------------------------------------------------------------------------
# ``gc --expired`` runs the TTL sweep first and then decides whether to run the
# archived-collection pass at all. That decision is the whole point of this
# corpus: under ``ttl.strategy: archive`` the command must stop after archiving,
# because collecting archived rows in the same pass would hard-delete the very
# rows it had just soft-retired. Under ``delete`` the sweep removes them
# outright and the collection pass then runs as usual.
#
# The corpus carries a row of each kind the decision distinguishes:
#
# * two **expired** thoughts — two rather than one, so "sweep the expired rows"
#   and "sweep the first expired row" leave different databases;
# * one thought that was **already ARCHIVED** before the command ran. It is what
#   makes the two directions of the decision observable on stored state rather
#   than only on the report: under ``archive`` it must still be there afterwards
#   (the collection pass never ran), under ``delete`` it must be gone (it did);
# * three **unexpired** thoughts — one with no TTL at all and two still inside
#   theirs — which no strategy may touch.
#
# Rows are interleaved unexpired · archived · expired · unexpired · expired ·
# unexpired, so on ``thought_id`` and on ``rowid`` (which follows the write order
# below) the rows that must change are bracketed by rows that must not, and no
# contiguous range over either axis selects exactly the rows any one pass
# removes. ``expires_at`` cannot be interleaved — it is the sweep's own
# selection axis, so NULL and future values necessarily sort away from past ones
# — and ``embedding.vector_blob`` is not chased for the reason given for the
# corpus above.

#: Expiry timestamps. ``gc --expired`` reads the wall clock itself — unlike
#: ``cleanup_expired`` it takes no injectable "now" — so the corpus brackets any
#: plausible run time by centuries in both directions instead of freezing it.
_LONG_PAST = "2020-01-01T00:00:00+00:00"
_LATER_PAST = "2020-06-01T00:00:00+00:00"
_NEAR_FUTURE = "2999-01-01T00:00:00+00:00"
_FAR_FUTURE = "3000-01-01T00:00:00+00:00"

_TTL_THOUGHTS: tuple[tuple[str, LifecycleStatus, str | None], ...] = (
    ("t-eternal", LifecycleStatus.ACTIVE, None),
    ("t-boxed", LifecycleStatus.ARCHIVED, None),
    ("t-expired", LifecycleStatus.ACTIVE, _LONG_PAST),
    ("t-fresh", LifecycleStatus.ACTIVE, _NEAR_FUTURE),
    ("t-lapsed", LifecycleStatus.ACTIVE, _LATER_PAST),
    ("t-young", LifecycleStatus.ACTIVE, _FAR_FUTURE),
)

#: Every id in the corpus, derived from it so the two cannot drift apart.
_TTL_THOUGHT_IDS: tuple[str, ...] = tuple(tid for tid, _status, _expiry in _TTL_THOUGHTS)

_TTL_EDGES: tuple[tuple[str, str, str, float], ...] = (
    ("e-alpha-live", "t-eternal", "t-fresh", 0.2),
    ("e-bravo-boxed", "t-boxed", "t-eternal", 0.3),
    ("e-charlie-expired", "t-expired", "t-fresh", 0.4),
    ("e-delta-live", "t-fresh", "t-eternal", 0.5),
    ("e-echo-lapsed", "t-lapsed", "t-eternal", 0.6),
    ("e-foxtrot-live", "t-young", "t-eternal", 0.7),
)

_TTL_EMBEDDINGS: tuple[tuple[str, list[float]], ...] = (
    ("t-eternal", [0.7, 0.8, 0.9]),
    ("t-boxed", [0.3, 0.4, 0.5]),
    ("t-expired", [0.1, 0.2, 0.3]),
    ("t-fresh", [0.4, 0.5, 0.6]),
    ("t-lapsed", [0.2, 0.3, 0.4]),
    ("t-young", [0.5, 0.6, 0.7]),
)

_TTL_ACTIONS: tuple[tuple[str, str], ...] = (
    ("a-alpha-live", "t-eternal"),
    ("a-bravo-boxed", "t-boxed"),
    ("a-charlie-expired", "t-expired"),
    ("a-delta-live", "t-fresh"),
    ("a-echo-lapsed", "t-lapsed"),
    ("a-foxtrot-live", "t-young"),
)


async def _materialise_ttl_lifecycle(db_path: Path, *, vector_backend: str) -> None:
    """Write the TTL corpus through the public API.

    Args:
        db_path: Where to create the database.
        vector_backend: ``_NO_VECTOR_INDEX`` or ``_PERSISTED_VECTOR_INDEX``.

    """
    conn = await aiosqlite.connect(str(db_path))
    conn.row_factory = aiosqlite.Row
    store = SqliteEngravaCore(db=conn, embedding_provider=None, auto_embed=False)
    await store.ensure_schema()
    await store._configure_vector_backend(
        backend_name=vector_backend,
        embedding_dimension=_VECTOR_DIMENSION,
    )

    for tid, status, expires_at in _TTL_THOUGHTS:
        await store.create_thought(
            _make_thought(tid, expires_at=expires_at, lifecycle_status=status),
        )
    for eid, src, dst, weight in _TTL_EDGES:
        await store.create_edge(_make_edge(eid, src, dst, weight=weight))
    for tid, vector in _TTL_EMBEDDINGS:
        await store.store_embedding(tid, vector)
    for aid, src in _TTL_ACTIONS:
        await store.create_action(_make_action(aid, src))

    await conn.commit()
    await conn.close()


@pytest.fixture
def ttl_lifecycle_db(tmp_path: Path) -> Path:
    """Materialise the TTL corpus on a head schema via the public API.

    Returns:
        Path to the populated database file.

    """
    db_path = tmp_path / "ttl-lifecycle.db"
    asyncio.run(_materialise_ttl_lifecycle(db_path, vector_backend=_NO_VECTOR_INDEX))
    return db_path


@pytest.fixture
def vec_indexed_ttl_lifecycle_db(tmp_path: Path) -> Path:
    """Materialise the same TTL corpus behind a persisted vec0 index.

    Returns:
        Path to the populated database file, carrying an ``embedding_vec``
        table with one vector per stored embedding.

    """
    db_path = tmp_path / "ttl-lifecycle-vec.db"
    asyncio.run(_materialise_ttl_lifecycle(db_path, vector_backend=_PERSISTED_VECTOR_INDEX))
    _assert_index_mirrors_corpus(db_path)
    return db_path


def _write_ttl_config(config_path: Path, db_path: Path, strategy: str) -> Path:
    """Write a real ``engrava.yaml`` that selects a TTL strategy.

    ``gc --expired`` reads the strategy from the config file and nowhere else,
    so the tests set it the way a user does — in YAML through ``--config`` —
    rather than by handing the command a config object. ``database.path`` is
    written because the loader requires it; the command still takes its database
    from ``--db``.

    Args:
        config_path: Where to write the YAML.
        db_path: Database path recorded in the file.
        strategy: ``"archive"`` or ``"delete"``.

    Returns:
        The path that was written.

    """
    config_path.write_text(
        f"database:\n  path: {db_path}\nttl:\n  strategy: {strategy}\n",
        encoding="utf-8",
    )
    return config_path


class TestGcExpiredBlastRadius:
    """``engrava gc --expired`` sweeps by strategy, and stops where it must."""

    #: The two thoughts past their expiry when the command runs.
    _EXPIRED = ("t-expired", "t-lapsed")

    #: Archived before the command ran — the witness for whether the
    #: archived-collection pass executed.
    _ALREADY_ARCHIVED = ("t-boxed",)

    #: Thoughts no strategy may touch: one without a TTL, two inside theirs.
    _UNEXPIRED = ("t-eternal", "t-fresh", "t-young")

    #: Children of the unexpired thoughts — they outlive every pass.
    _UNEXPIRED_EDGES = ("e-alpha-live", "e-delta-live", "e-foxtrot-live")
    _UNEXPIRED_ACTIONS = ("a-alpha-live", "a-delta-live", "a-foxtrot-live")

    #: Children of the expired thoughts — they go only when their parent is
    #: physically deleted, never when it is archived.
    _EXPIRED_EDGES = ("e-charlie-expired", "e-echo-lapsed")
    _EXPIRED_ACTIONS = ("a-charlie-expired", "a-echo-lapsed")

    #: Children of the already-archived thought.
    _ARCHIVED_EDGES = ("e-bravo-boxed",)
    _ARCHIVED_ACTIONS = ("a-bravo-boxed",)

    @staticmethod
    def _sweep(
        runner: CliRunner,
        db_path: Path,
        *,
        config_path: Path | None = None,
        dry_run: bool = False,
    ) -> Result:
        """Run ``engrava gc --expired`` over the corpus and hand back the result.

        Asserts only that the command ran: the ``Cleaned up N`` / ``Collected N``
        lines are the command's own account of what it did and would abort a
        test before a single row was read back. They are left to the caller to
        check *after* the stored state, which fails just as loudly on a no-op as
        on an over-broad delete.

        Args:
            runner: The Click test runner.
            db_path: Database to operate on.
            config_path: Optional ``engrava.yaml`` selecting the TTL strategy.
            dry_run: Pass ``--dry-run``.

        Returns:
            The Click result, for the caller's report assertions.

        """
        args = ["--db", str(db_path)]
        if config_path is not None:
            args += ["--config", str(config_path)]
        args += ["gc", "--expired"]
        if dry_run:
            args.append("--dry-run")
        result = runner.invoke(cli, args)
        assert result.exit_code == 0, result.output
        return result

    @classmethod
    def _archived_row(cls, before: dict[str, object]) -> dict[str, object]:
        """Return the row a TTL archival must leave behind, given the row before it.

        Spelled out as the whole row rather than as the few columns the test
        happens to care about, so an archival that also rewrote something else —
        content, a timestamp, a cycle counter — is caught here rather than
        silently accepted.

        Args:
            before: The thought row as stored before the sweep.

        Returns:
            The exact row expected afterwards.

        """
        return {
            **before,
            "lifecycle_status": LifecycleStatus.ARCHIVED.value,
            # TTL archival clears the expiry (the row is no longer subject to
            # TTL) and the hygiene-archival markers (a TTL archival is not a
            # hygiene archival).
            "expires_at": None,
            "archived_at_cycle": None,
            "archived_at": None,
        }

    @classmethod
    def _assert_corpus_preconditions(cls, before: dict[str, dict[str, object]]) -> None:
        """Fail if the corpus cannot tell the outcomes apart to begin with.

        Every assertion in this class is about a row *changing* or *not
        changing*. If the expired pair were already ARCHIVED, "it is archived
        now" would hold without the sweep running; if the archived witness were
        not archived, "the collection pass left it alone" would hold because
        there was nothing to collect.

        Args:
            before: Every corpus thought row, as stored before the command.

        """
        for tid in cls._EXPIRED:
            assert before[tid]["lifecycle_status"] == LifecycleStatus.ACTIVE.value
            assert before[tid]["expires_at"] is not None, f"{tid} carries no expiry"
        for tid in cls._ALREADY_ARCHIVED:
            assert before[tid]["lifecycle_status"] == LifecycleStatus.ARCHIVED.value
        for tid in cls._UNEXPIRED:
            assert before[tid]["lifecycle_status"] == LifecycleStatus.ACTIVE.value

    @staticmethod
    def _read_thoughts(db_path: Path) -> dict[str, dict[str, object]]:
        """Read every corpus thought row back from the database file.

        Args:
            db_path: Database the command operated on.

        Returns:
            Each corpus id mapped to its stored row.

        """
        with _reopen(db_path) as conn:
            return {
                tid: _require(_sync_row(conn, _THOUGHT_BY_ID, tid), tid) for tid in _TTL_THOUGHT_IDS
            }

    def test_archive_strategy_archives_the_expired_rows_and_deletes_nothing(
        self,
        runner: CliRunner,
        ttl_lifecycle_db: Path,
        tmp_path: Path,
    ) -> None:
        """Under ``archive`` the pass stops: not one thought row is removed.

        This is the interlock. The command archives the expired rows and must
        then decline to run the archived-collection pass — otherwise it would
        hard-delete the rows it has just archived, plus every row that was
        already archived, in a single invocation the user asked to *retire*
        data. Every corpus row must therefore still be stored afterwards, which
        is what the documented behaviour promises (``docs/data-lifecycle.md``:
        the pass "stops there — it does not also garbage-collect archived rows
        in the same run").
        """
        config = _write_ttl_config(tmp_path / "archive.yaml", ttl_lifecycle_db, "archive")
        before = self._read_thoughts(ttl_lifecycle_db)
        self._assert_corpus_preconditions(before)

        result = self._sweep(runner, ttl_lifecycle_db, config_path=config)

        with _reopen(ttl_lifecycle_db) as conn:
            assert _sync_id_set(conn, _ALL_THOUGHT_IDS) == set(_TTL_THOUGHT_IDS)
            for tid in self._EXPIRED:
                after = _sync_row(conn, _THOUGHT_BY_ID, tid)
                assert after == self._archived_row(before[tid]), f"{tid} was not archived as such"
            for tid in (*self._ALREADY_ARCHIVED, *self._UNEXPIRED):
                assert _sync_row(conn, _THOUGHT_BY_ID, tid) == before[tid], f"{tid} was modified"
        # Only once the stored state is settled does the reported outcome matter.
        assert "Cleaned up 2 expired thoughts (strategy: archive)." in result.output
        assert "Collected" not in result.output

    def test_archive_strategy_leaves_every_child_row_in_place(
        self,
        runner: CliRunner,
        ttl_lifecycle_db: Path,
        tmp_path: Path,
    ) -> None:
        """Archiving is an ``UPDATE``: no edge, embedding or action goes with it.

        The children of the archived rows are the ones a collection pass would
        take, so they are asserted here in their own right — including those of
        the thought that was already archived before the command ran.

        The parents are re-read afterwards as well, because "the children are
        all still there" is equally true of a command that never swept
        anything: without that read the test would hold for a ``--expired``
        flag that had stopped doing its job.
        """
        config = _write_ttl_config(tmp_path / "archive.yaml", ttl_lifecycle_db, "archive")
        all_edges = tuple(eid for eid, _src, _dst, _weight in _TTL_EDGES)
        all_actions = tuple(aid for aid, _src in _TTL_ACTIONS)
        with _reopen(ttl_lifecycle_db) as conn:
            edges_before = {
                eid: _require(_sync_row(conn, _EDGE_BY_ID, eid), eid) for eid in all_edges
            }
            embeddings_before = {
                tid: _require(_sync_row(conn, _EMBEDDING_BY_OWNER, tid), tid)
                for tid in _TTL_THOUGHT_IDS
            }
            actions_before = {
                aid: _require(_sync_row(conn, _ACTION_BY_ID, aid), aid) for aid in all_actions
            }

        self._sweep(runner, ttl_lifecycle_db, config_path=config)

        with _reopen(ttl_lifecycle_db) as conn:
            for tid in self._EXPIRED:
                parent = _require(_sync_row(conn, _THOUGHT_BY_ID, tid), tid)
                assert parent["lifecycle_status"] == LifecycleStatus.ARCHIVED.value, (
                    f"{tid} was never archived, so its children survived vacuously"
                )
            assert _sync_id_set(conn, _ALL_EDGE_IDS) == set(all_edges)
            assert _sync_id_set(conn, _ALL_EMBEDDING_OWNERS) == set(_TTL_THOUGHT_IDS)
            assert _sync_id_set(conn, _ALL_ACTION_IDS) == set(all_actions)
            for eid, before in edges_before.items():
                assert _sync_row(conn, _EDGE_BY_ID, eid) == before, f"edge {eid} was modified"
            for tid, before in embeddings_before.items():
                after = _sync_row(conn, _EMBEDDING_BY_OWNER, tid)
                assert after == before, f"embedding of {tid} was modified"
            for aid, before in actions_before.items():
                assert _sync_row(conn, _ACTION_BY_ID, aid) == before, f"action {aid} was modified"

    def test_default_strategy_without_a_config_file_archives(
        self,
        runner: CliRunner,
        ttl_lifecycle_db: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With no config at all the command takes the documented default.

        Without ``--config`` the strategy comes from a ``TTLConfig()`` built in
        the command rather than from the loader, which is a second way for the
        default to be wrong. ``ENGRAVA_CONFIG`` is cleared so an environment
        that happens to point at a config file cannot silently supply one.
        """
        monkeypatch.delenv("ENGRAVA_CONFIG", raising=False)
        before = self._read_thoughts(ttl_lifecycle_db)
        self._assert_corpus_preconditions(before)

        result = self._sweep(runner, ttl_lifecycle_db)

        with _reopen(ttl_lifecycle_db) as conn:
            assert _sync_id_set(conn, _ALL_THOUGHT_IDS) == set(_TTL_THOUGHT_IDS)
            for tid in self._EXPIRED:
                after = _sync_row(conn, _THOUGHT_BY_ID, tid)
                assert after == self._archived_row(before[tid]), f"{tid} was not archived as such"
            for tid in (*self._ALREADY_ARCHIVED, *self._UNEXPIRED):
                assert _sync_row(conn, _THOUGHT_BY_ID, tid) == before[tid], f"{tid} was modified"
        assert "(strategy: archive)" in result.output

    def test_delete_strategy_removes_the_expired_rows_and_collects_the_archived_one(
        self,
        runner: CliRunner,
        ttl_lifecycle_db: Path,
        tmp_path: Path,
    ) -> None:
        """Under ``delete`` the expired rows go — and the collection pass runs.

        The other direction of the interlock. The sweep deletes outright, so
        there is nothing freshly archived to protect and the pass that collects
        pre-existing archived rows must not be skipped: ``t-boxed`` was archived
        before the command ran and must be gone afterwards. The unexpired rows
        are untouched by either half.
        """
        config = _write_ttl_config(tmp_path / "delete.yaml", ttl_lifecycle_db, "delete")
        before = self._read_thoughts(ttl_lifecycle_db)
        self._assert_corpus_preconditions(before)

        result = self._sweep(runner, ttl_lifecycle_db, config_path=config)

        with _reopen(ttl_lifecycle_db) as conn:
            for tid in (*self._EXPIRED, *self._ALREADY_ARCHIVED):
                assert _sync_row(conn, _THOUGHT_BY_ID, tid) is None, f"{tid} is still stored"
            assert _sync_id_set(conn, _ALL_THOUGHT_IDS) == set(self._UNEXPIRED)
            for tid in self._UNEXPIRED:
                assert _sync_row(conn, _THOUGHT_BY_ID, tid) == before[tid], f"{tid} was modified"
        assert "Cleaned up 2 expired thoughts (strategy: delete)." in result.output
        assert "Collected 1 archived thoughts." in result.output

    def test_delete_strategy_keeps_the_children_of_surviving_thoughts(
        self,
        runner: CliRunner,
        ttl_lifecycle_db: Path,
        tmp_path: Path,
    ) -> None:
        """The cascade follows the swept and collected thoughts, and stops there."""
        config = _write_ttl_config(tmp_path / "delete.yaml", ttl_lifecycle_db, "delete")
        doomed_children: tuple[tuple[str, str], ...] = (
            *((eid, _EDGE_BY_ID) for eid in (*self._EXPIRED_EDGES, *self._ARCHIVED_EDGES)),
            *((tid, _EMBEDDING_BY_OWNER) for tid in (*self._EXPIRED, *self._ALREADY_ARCHIVED)),
            *((aid, _ACTION_BY_ID) for aid in (*self._EXPIRED_ACTIONS, *self._ARCHIVED_ACTIONS)),
        )
        with _reopen(ttl_lifecycle_db) as conn:
            for doomed, query in doomed_children:
                _require(_sync_row(conn, query, doomed), doomed)
            edges_before = {
                eid: _require(_sync_row(conn, _EDGE_BY_ID, eid), eid)
                for eid in self._UNEXPIRED_EDGES
            }
            embeddings_before = {
                tid: _require(_sync_row(conn, _EMBEDDING_BY_OWNER, tid), tid)
                for tid in self._UNEXPIRED
            }
            actions_before = {
                aid: _require(_sync_row(conn, _ACTION_BY_ID, aid), aid)
                for aid in self._UNEXPIRED_ACTIONS
            }

        self._sweep(runner, ttl_lifecycle_db, config_path=config)

        with _reopen(ttl_lifecycle_db) as conn:
            assert _sync_id_set(conn, _ALL_EDGE_IDS) == set(self._UNEXPIRED_EDGES)
            assert _sync_id_set(conn, _ALL_EMBEDDING_OWNERS) == set(self._UNEXPIRED)
            assert _sync_id_set(conn, _ALL_ACTION_IDS) == set(self._UNEXPIRED_ACTIONS)
            for eid, before in edges_before.items():
                assert _sync_row(conn, _EDGE_BY_ID, eid) == before, f"edge {eid} was modified"
            for tid, before in embeddings_before.items():
                after = _sync_row(conn, _EMBEDDING_BY_OWNER, tid)
                assert after == before, f"embedding of {tid} was modified"
            for aid, before in actions_before.items():
                assert _sync_row(conn, _ACTION_BY_ID, aid) == before, f"action {aid} was modified"

    @pytest.mark.parametrize(
        ("strategy", "reported"),
        [
            ("archive", "Would archive 2 expired thoughts."),
            ("delete", "Would delete 2 expired thoughts."),
        ],
    )
    def test_dry_run_changes_nothing_under_either_strategy(
        self,
        runner: CliRunner,
        ttl_lifecycle_db: Path,
        tmp_path: Path,
        strategy: str,
        reported: str,
    ) -> None:
        """``--dry-run`` reports the pass and writes nothing, on either strategy.

        Asserted over all four tables: a dry run that archived instead of
        deleting would leave every row present and still be wrong.
        """
        config = _write_ttl_config(tmp_path / f"{strategy}.yaml", ttl_lifecycle_db, strategy)
        all_edges = tuple(eid for eid, _src, _dst, _weight in _TTL_EDGES)
        all_actions = tuple(aid for aid, _src in _TTL_ACTIONS)
        thoughts_before = self._read_thoughts(ttl_lifecycle_db)
        self._assert_corpus_preconditions(thoughts_before)
        with _reopen(ttl_lifecycle_db) as conn:
            edges_before = {
                eid: _require(_sync_row(conn, _EDGE_BY_ID, eid), eid) for eid in all_edges
            }
            embeddings_before = {
                tid: _require(_sync_row(conn, _EMBEDDING_BY_OWNER, tid), tid)
                for tid in _TTL_THOUGHT_IDS
            }
            actions_before = {
                aid: _require(_sync_row(conn, _ACTION_BY_ID, aid), aid) for aid in all_actions
            }

        result = self._sweep(runner, ttl_lifecycle_db, config_path=config, dry_run=True)

        with _reopen(ttl_lifecycle_db) as conn:
            # Read back without _require here: a dry run that swept anyway must
            # report as a row that is now ``None``, not as a missing precondition.
            assert _sync_id_set(conn, _ALL_THOUGHT_IDS) == set(_TTL_THOUGHT_IDS)
            after = {tid: _sync_row(conn, _THOUGHT_BY_ID, tid) for tid in _TTL_THOUGHT_IDS}
            assert after == thoughts_before
            assert _sync_id_set(conn, _ALL_EDGE_IDS) == set(all_edges)
            assert _sync_id_set(conn, _ALL_EMBEDDING_OWNERS) == set(_TTL_THOUGHT_IDS)
            assert _sync_id_set(conn, _ALL_ACTION_IDS) == set(all_actions)
            for eid, before in edges_before.items():
                assert _sync_row(conn, _EDGE_BY_ID, eid) == before, f"edge {eid} was modified"
            for tid, before in embeddings_before.items():
                after = _sync_row(conn, _EMBEDDING_BY_OWNER, tid)
                assert after == before, f"embedding of {tid} was modified"
            for aid, before in actions_before.items():
                assert _sync_row(conn, _ACTION_BY_ID, aid) == before, f"action {aid} was modified"
        assert reported in result.output

    def test_a_second_pass_collects_what_the_archive_pass_left(
        self,
        runner: CliRunner,
        ttl_lifecycle_db: Path,
        tmp_path: Path,
    ) -> None:
        """The archived rows are removed by the *next* pass, not the one that archived them.

        This is the workflow the documentation prescribes for actually deleting
        expired data under the ``archive`` strategy — "run a separate
        ``engrava gc`` afterwards". By the second invocation there is nothing
        left to expire (the first cleared ``expires_at``), so the collection
        pass runs and takes all three archived thoughts with their children.
        Nothing that was never expired is touched by either pass.
        """
        config = _write_ttl_config(tmp_path / "archive.yaml", ttl_lifecycle_db, "archive")
        collected_later = (*self._EXPIRED, *self._ALREADY_ARCHIVED)
        with _reopen(ttl_lifecycle_db) as conn:
            for tid in collected_later:
                _require(_sync_row(conn, _THOUGHT_BY_ID, tid), tid)
            survivors_before = {
                tid: _require(_sync_row(conn, _THOUGHT_BY_ID, tid), tid) for tid in self._UNEXPIRED
            }
            edges_before = {
                eid: _require(_sync_row(conn, _EDGE_BY_ID, eid), eid)
                for eid in self._UNEXPIRED_EDGES
            }

        self._sweep(runner, ttl_lifecycle_db, config_path=config)
        result = self._sweep(runner, ttl_lifecycle_db, config_path=config)

        with _reopen(ttl_lifecycle_db) as conn:
            for tid in collected_later:
                assert _sync_row(conn, _THOUGHT_BY_ID, tid) is None, f"{tid} is still stored"
            assert _sync_id_set(conn, _ALL_THOUGHT_IDS) == set(self._UNEXPIRED)
            assert _sync_id_set(conn, _ALL_EDGE_IDS) == set(self._UNEXPIRED_EDGES)
            assert _sync_id_set(conn, _ALL_EMBEDDING_OWNERS) == set(self._UNEXPIRED)
            assert _sync_id_set(conn, _ALL_ACTION_IDS) == set(self._UNEXPIRED_ACTIONS)
            for tid, before in survivors_before.items():
                assert _sync_row(conn, _THOUGHT_BY_ID, tid) == before, f"{tid} was modified"
            for eid, before in edges_before.items():
                assert _sync_row(conn, _EDGE_BY_ID, eid) == before, f"edge {eid} was modified"
        assert "No expired thoughts to cleanup." in result.output
        assert "Collected 3 archived thoughts." in result.output

    @sqlite_vec_required
    def test_delete_strategy_purges_only_the_swept_thoughts_vectors(
        self,
        runner: CliRunner,
        vec_indexed_ttl_lifecycle_db: Path,
        tmp_path: Path,
    ) -> None:
        """The expiry sweep removes its own thoughts' vectors, on its own.

        The sweep deletes through the core rather than through the collection
        pass's statements, so it is a second, independent route to a stranded
        vector. Isolating it takes a first plain ``gc``: that collects the one
        pre-archived thought, after which the collection pass has nothing left
        to do and returns before reaching any purge of its own. Whatever the
        vectors of the two swept thoughts do next is therefore the sweep's
        doing and nothing else's — which is asserted, not assumed, by reading
        the lifecycle column back in between.

        The corpus is interleaved unexpired · archived · expired · unexpired ·
        expired · unexpired on ``rowid``, the index's own key, so no purge
        picking by rowid position selects exactly the swept pair.
        """
        config = _write_ttl_config(tmp_path / "delete.yaml", vec_indexed_ttl_lifecycle_db, "delete")
        with _reopen_with_vector_index(vec_indexed_ttl_lifecycle_db) as conn:
            rowids = _sync_embedding_rowids(conn)
            assert set(rowids) == set(_TTL_THOUGHT_IDS), (
                f"corpus precondition failed: embeddings stored for {sorted(rowids)}"
            )
            written = [rowids[tid] for tid, _vector in _TTL_EMBEDDINGS]
            assert written == sorted(written), (
                "corpus precondition failed: rowids do not follow the write order"
            )
            swept = {rowids[tid] for tid in self._EXPIRED}
            surviving = {rowids[tid] for tid in self._UNEXPIRED}

        collect = runner.invoke(cli, ["--db", str(vec_indexed_ttl_lifecycle_db), "gc"])
        assert collect.exit_code == 0, collect.output

        with _reopen(vec_indexed_ttl_lifecycle_db) as conn:
            assert _sync_id_set(conn, _ARCHIVED_THOUGHT_IDS) == set(), (
                "isolation precondition failed: the collection pass still has work to do"
            )

        self._sweep(runner, vec_indexed_ttl_lifecycle_db, config_path=config)

        with _reopen_with_vector_index(vec_indexed_ttl_lifecycle_db) as conn:
            remaining = _sync_vec_rowids(conn)
            assert not swept & remaining, "a swept thought's vector survived"
            assert remaining == surviving
            assert _sync_id_set(conn, _ALL_THOUGHT_IDS) == set(self._UNEXPIRED)

    @sqlite_vec_required
    def test_delete_strategy_purges_both_passes_vectors_in_one_run(
        self,
        runner: CliRunner,
        vec_indexed_ttl_lifecycle_db: Path,
        tmp_path: Path,
    ) -> None:
        """One ``--expired`` run leaves the index holding exactly the survivors.

        The test above isolates the sweep by emptying the collection pass
        first; this is the shape a user actually gets, where both passes delete
        in the same invocation. Each reaches the index independently — the
        module is loaded twice on the one connection, which must be harmless —
        and between them they must account for all three departing vectors: the
        two the sweep expires and the one the collection reaps.
        """
        config = _write_ttl_config(tmp_path / "delete.yaml", vec_indexed_ttl_lifecycle_db, "delete")
        collected = (*self._EXPIRED, *self._ALREADY_ARCHIVED)
        with _reopen_with_vector_index(vec_indexed_ttl_lifecycle_db) as conn:
            rowids = _sync_embedding_rowids(conn)
            assert set(rowids) == set(_TTL_THOUGHT_IDS), (
                f"corpus precondition failed: embeddings stored for {sorted(rowids)}"
            )
            doomed = {rowids[tid] for tid in collected}
            surviving = {rowids[tid] for tid in self._UNEXPIRED}
            assert _sync_vec_rowids(conn) == doomed | surviving

        result = self._sweep(runner, vec_indexed_ttl_lifecycle_db, config_path=config)

        with _reopen_with_vector_index(vec_indexed_ttl_lifecycle_db) as conn:
            remaining = _sync_vec_rowids(conn)
            assert not doomed & remaining, "a deleted thought's vector survived"
            assert remaining == surviving
            assert _sync_id_set(conn, _ALL_THOUGHT_IDS) == set(self._UNEXPIRED)
        assert "Cleaned up 2 expired thoughts (strategy: delete)." in result.output
        assert "Collected 1 archived thoughts." in result.output

    @sqlite_vec_required
    def test_archive_strategy_leaves_every_vector_in_the_index(
        self,
        runner: CliRunner,
        vec_indexed_ttl_lifecycle_db: Path,
        tmp_path: Path,
    ) -> None:
        """Under ``archive`` nothing is deleted, so no vector may be either.

        The strategy that archives stops before the collection pass, so this
        run removes no row from any table — including the pre-archived
        ``t-boxed``, whose vector is the non-target that tells a purge which
        correctly did nothing apart from one which swept the index anyway.

        This pins the resulting state, not the branch. An archiving sweep
        orphans no vector, so calling the reconciliation here would be
        observationally identical to skipping it; what would show is a purge
        that reached wider than an orphan.
        """
        config = _write_ttl_config(
            tmp_path / "archive.yaml", vec_indexed_ttl_lifecycle_db, "archive"
        )
        with _reopen_with_vector_index(vec_indexed_ttl_lifecycle_db) as conn:
            rowids = _sync_embedding_rowids(conn)
            assert set(rowids) == set(_TTL_THOUGHT_IDS), (
                f"corpus precondition failed: embeddings stored for {sorted(rowids)}"
            )
            before = _sync_vec_rowids(conn)
            assert before == set(rowids.values())

        result = self._sweep(runner, vec_indexed_ttl_lifecycle_db, config_path=config)

        with _reopen_with_vector_index(vec_indexed_ttl_lifecycle_db) as conn:
            assert _sync_vec_rowids(conn) == before
            assert _sync_id_set(conn, _ALL_THOUGHT_IDS) == set(_TTL_THOUGHT_IDS)
        assert "(strategy: archive)" in result.output

    @sqlite_vec_required
    def test_archive_strategy_does_not_need_sqlite_vec_on_an_indexed_store(
        self,
        runner: CliRunner,
        vec_indexed_ttl_lifecycle_db: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An archiving sweep is never refused for want of the module.

        It deletes no row, so it strands no vector and needs no access to the
        index. Withholding the module is what makes that observable: a command
        that asked for it before knowing whether it would delete anything would
        refuse to archive here, for a guarantee that is not at stake.
        """
        config = _write_ttl_config(
            tmp_path / "archive.yaml", vec_indexed_ttl_lifecycle_db, "archive"
        )

        async def _refuse_to_load(db: object) -> bool:
            del db
            return False

        monkeypatch.setattr(
            "engrava.extensions.vector_sqlite_vec.load_sqlite_vec",
            _refuse_to_load,
        )
        before = self._read_thoughts(vec_indexed_ttl_lifecycle_db)
        self._assert_corpus_preconditions(before)

        result = self._sweep(runner, vec_indexed_ttl_lifecycle_db, config_path=config)

        with _reopen_with_vector_index(vec_indexed_ttl_lifecycle_db) as conn:
            for tid in self._EXPIRED:
                after = _sync_row(conn, _THOUGHT_BY_ID, tid)
                assert after == self._archived_row(before[tid]), f"{tid} was not archived as such"
            for tid in (*self._ALREADY_ARCHIVED, *self._UNEXPIRED):
                assert _sync_row(conn, _THOUGHT_BY_ID, tid) == before[tid], f"{tid} was modified"
        assert "(strategy: archive)" in result.output

    @sqlite_vec_required
    def test_delete_strategy_sweeps_nothing_when_the_purge_itself_fails(
        self,
        runner: CliRunner,
        vec_indexed_ttl_lifecycle_db: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A purge that fails takes the expiry sweep down with it.

        The sweep runs through the core, which commits per operation, so its
        deletions are durable the moment it returns. A purge failing after that
        would leave the swept thoughts' vectors stranded — and would open a
        window in which another writer can claim a freed ``embedding`` rowid,
        after which the stale vector is owned and no reconciliation can tell.
        The sweep and its purge therefore share one transaction, and what
        proves it is the stored state after the failure, not the error.

        The stub removes a vector *before* it raises, so the rollback under
        test is the one that has to reach the vec0 table's own shadow storage.
        Every canonical table is read back, the children included: the sweep
        cascades to them, so a transaction that gave up half way would show
        there even where the thought rows had been restored.
        """
        config = _write_ttl_config(tmp_path / "delete.yaml", vec_indexed_ttl_lifecycle_db, "delete")
        removed_by_the_stub: list[int] = []
        monkeypatch.setattr(
            "engrava.extensions.vector_sqlite_vec.purge_orphan_vectors",
            _purge_that_removes_a_vector_then_fails(removed_by_the_stub),
        )
        all_edges = {eid for eid, _src, _dst, _weight in _TTL_EDGES}
        all_actions = {aid for aid, _src in _TTL_ACTIONS}
        with _reopen_with_vector_index(vec_indexed_ttl_lifecycle_db) as conn:
            vectors_before = _sync_vec_rowids(conn)
            thoughts_before = {
                tid: _require(_sync_row(conn, _THOUGHT_BY_ID, tid), tid) for tid in _TTL_THOUGHT_IDS
            }
            assert _sync_id_set(conn, _ALL_EDGE_IDS) == all_edges
            assert _sync_id_set(conn, _ALL_ACTION_IDS) == all_actions

        result = runner.invoke(
            cli,
            ["--db", str(vec_indexed_ttl_lifecycle_db), "--config", str(config), "gc", "--expired"],
            standalone_mode=False,
        )
        # An instrumentation precondition: without it, a purge that was never
        # reached would read as a rollback that worked.
        assert removed_by_the_stub, "the purge never reached the index"

        with _reopen_with_vector_index(vec_indexed_ttl_lifecycle_db) as conn:
            assert _sync_id_set(conn, _ALL_THOUGHT_IDS) == set(_TTL_THOUGHT_IDS)
            after = {tid: _sync_row(conn, _THOUGHT_BY_ID, tid) for tid in _TTL_THOUGHT_IDS}
            assert after == thoughts_before
            assert _sync_id_set(conn, _ALL_EDGE_IDS) == all_edges
            assert _sync_id_set(conn, _ALL_EMBEDDING_OWNERS) == set(_TTL_THOUGHT_IDS)
            assert _sync_id_set(conn, _ALL_ACTION_IDS) == all_actions
            assert _sync_vec_rowids(conn) == vectors_before
        assert isinstance(result.exception, sqlite3.OperationalError), result.exception
