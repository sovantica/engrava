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

Three operations are covered, one class each:

* :meth:`~engrava.SqliteEngravaCore.delete_edge` — one edge, not the edge table.
* :meth:`~engrava.SqliteEngravaCore.cleanup_expired` under the ``delete`` TTL
  strategy — the expired thoughts and their cascaded children, not every thought.
* ``engrava gc`` — the ARCHIVED thoughts and their children, never the edges,
  embeddings or actions belonging to ACTIVE thoughts.
"""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3
import struct
from typing import TYPE_CHECKING

import aiosqlite
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

# The real-shape per-version schema builder the migration-ladder suite already
# maintains. Reused rather than re-derived so the pre-cascade fixture below
# cannot drift from the schema that version actually shipped.
from tests.test_migration_upgrade_chains import _bootstrap_core_at_version

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator
    from pathlib import Path


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


@pytest.fixture
def mixed_lifecycle_db(tmp_path: Path) -> Path:
    """Materialise the mixed-lifecycle corpus on a head schema via the public API.

    Returns:
        Path to the populated database file.

    """
    db_path = tmp_path / "mixed-lifecycle.db"

    async def _setup() -> None:
        conn = await aiosqlite.connect(str(db_path))
        conn.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(db=conn, embedding_provider=None, auto_embed=False)
        await store.ensure_schema()

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

    asyncio.run(_setup())
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
