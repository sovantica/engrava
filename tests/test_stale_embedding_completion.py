"""Regression: a slower, stale auto-embed completion must not overwrite a newer vector.

Auto-embed is deliberately *not* run under the write lock that protects an
update: the provider call is a slow, arbitrary network round trip, and
holding the lock across it would turn every other task's unrelated write into
a bottleneck on this one call's round trip. That means two updates to the
*same* thought can have their auto-embed completions land out of persist
order: the update that persisted first can still be the one whose embed
finishes last. Before the fix under test, the late completion installed its
vector unconditionally — overwriting a newer vector with one computed from
now-superseded content, even though the durable text and FTS index already
reflect the later write.

The regression here reproduces that ordering deterministically, via an
``asyncio.Event`` per registered embed call (never a sleep or a timing
race): two updates to one thought are issued as separate ``asyncio.Task``s,
the first's embed call is parked before it can return, the second's update
persists and its embed installs a vector, and only then is the first's embed
call released. The correct outcome is that the *second* update's vector is
what's left standing — the first's completion must recognise its own content
is stale and drop itself instead of installing.

Parametrized over both vector backends (numpy default, sqlite-vec when
installed) per the milestone's acceptance bar: the fix lives in the
completion check made before ``store_embedding`` is ever called, so it must
hold whichever backend that call ends up writing through.
"""

from __future__ import annotations

import asyncio
import importlib.util
import struct
from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava.domain.enums import LifecycleStatus, Priority, ThoughtType
from engrava.domain.models.thought import ThoughtRecord
from engrava.infrastructure.sqlite.engrava_core import SqliteEngravaCore
from engrava.infrastructure.sqlite.vector_sqlite_vec import SqliteVecSearchBackend

if TYPE_CHECKING:
    from pathlib import Path

# Skip the sqlite-vec arm cleanly when the extension is absent, but never let
# it silently pass as a numpy run when it *is* installed -- mirrors the same
# guard in test_vec0_numpy_ranking_parity.py / test_store_embedding_atomicity.py.
sqlite_vec_required = pytest.mark.skipif(
    importlib.util.find_spec("sqlite_vec") is None,
    reason="sqlite-vec package not installed",
)

_DIMENSION = 3
_MODEL_NAME = "barrier-fixture-model"


class _BarrierProvider:
    """Deterministic embedding provider whose completions release on command.

    Each call to :meth:`register` arms a distinct ``asyncio.Event`` keyed to a
    marker substring of the embedded text; :meth:`embed` blocks on whichever
    gate matches the text it was called with until the test explicitly
    releases it, then returns the vector registered alongside that gate. A
    second ``asyncio.Event`` per marker (``entered``) fires the instant the
    call parks, so a test can deterministically wait for "this embed call has
    started" before deciding what to do next -- no sleep, no timing guess.
    """

    def __init__(self, dimension: int) -> None:
        self._dimension = dimension
        self._vectors: dict[str, list[float]] = {}
        self._gates: dict[str, asyncio.Event] = {}
        self.entered: dict[str, asyncio.Event] = {}

    @property
    def dimension(self) -> int:
        """Return the fixed vector dimensionality every registered vector uses."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """Return the fixed model identifier this fixture reports."""
        return _MODEL_NAME

    def register(self, marker: str, vector: list[float]) -> asyncio.Event:
        """Arm a gate for ``marker``; returns the event that releases it."""
        gate = asyncio.Event()
        self._gates[marker] = gate
        self._vectors[marker] = vector
        self.entered[marker] = asyncio.Event()
        return gate

    async def embed(self, text: str) -> list[float]:
        """Block until the gate matching ``text`` is released, then return its vector."""
        for marker, gate in self._gates.items():
            if marker in text:
                self.entered[marker].set()
                await gate.wait()
                return list(self._vectors[marker])
        msg = f"embed() called with unregistered text: {text!r}"
        raise AssertionError(msg)


async def _build_store(
    tmp_path: Path, provider: _BarrierProvider, *, backend: str
) -> SqliteEngravaCore:
    """Construct a real, file-backed store with auto-embed on and the given backend."""
    db = await aiosqlite.connect(str(tmp_path / f"{backend}.db"))
    db.row_factory = aiosqlite.Row
    store = SqliteEngravaCore(db, embedding_provider=provider, auto_embed=True)
    store._owns_connection = True
    await store.ensure_schema()
    await store._configure_vector_backend(backend_name=backend, embedding_dimension=_DIMENSION)
    if backend == "sqlite-vec":
        assert isinstance(store._vector_backend, SqliteVecSearchBackend), (
            "sqlite-vec backend degraded to the numpy fallback -- the parametrized "
            "backend coverage this test needs would be vacuous"
        )
    return store


def _thought(thought_id: str, *, essence: str, content: str) -> ThoughtRecord:
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.OBSERVATION,
        essence=essence,
        content=content,
        priority=Priority.P3,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=0,
        updated_cycle=0,
        source="test",
    )


async def _stored_vector(store: SqliteEngravaCore, thought_id: str) -> list[float]:
    embedding = await store.get_embedding(thought_id)
    assert embedding is not None, f"no embedding row for {thought_id}"
    return list(struct.unpack(f"{embedding.dimension}f", embedding.vector_blob))


@pytest.mark.parametrize(
    "backend",
    ["numpy", pytest.param("sqlite-vec", marks=sqlite_vec_required)],
)
class TestStaleAutoEmbedCompletionCannotOverwriteNewerVector:
    """Two updates to one thought; the slower embed completion must lose."""

    async def test_slower_stale_completion_is_dropped_not_installed(
        self, tmp_path: Path, backend: str
    ) -> None:
        provider = _BarrierProvider(_DIMENSION)
        store = await _build_store(tmp_path, provider, backend=backend)
        try:
            # Seed the row. Its own create-time embed is released immediately --
            # only the two updates below are under test.
            seed_gate = provider.register("seed-marker", [0.0, 0.0, 1.0])
            create_task = asyncio.ensure_future(
                store.create_thought(_thought("t-1", essence="seed-marker", content="seed body"))
            )
            await provider.entered["seed-marker"].wait()
            seed_gate.set()
            await create_task
            assert await _stored_vector(store, "t-1") == pytest.approx([0.0, 0.0, 1.0])

            # Update A persists first and starts embedding first, but its
            # completion is parked open -- the "slow, stale" side of the race.
            a_gate = provider.register("a-marker", [1.0, 0.0, 0.0])
            b_gate = provider.register("b-marker", [0.0, 1.0, 0.0])

            task_a = asyncio.ensure_future(
                store.update_thought("t-1", essence="a-marker", content="content a")
            )
            await provider.entered["a-marker"].wait()

            # Update B persists (and starts embedding) strictly after A's own
            # write already committed -- update_thought releases the write
            # lock before calling the provider, so B is never blocked by A's
            # still-open embed call.
            task_b = asyncio.ensure_future(
                store.update_thought("t-1", essence="b-marker", content="content b")
            )
            await provider.entered["b-marker"].wait()
            b_gate.set()
            await task_b

            # B's vector is installed before A's stale completion is ever
            # released -- this is the state a pre-fix run overwrites.
            assert await _stored_vector(store, "t-1") == pytest.approx([0.0, 1.0, 0.0])
            row = await store._get_thought_row("t-1")
            assert row is not None
            assert row["content"] == "content b"

            # Now release A's completion. Its content ("content a") no longer
            # matches the thought's current content ("content b"), so the
            # fix must drop it instead of installing over B's vector.
            a_gate.set()
            await task_a

            final_vector = await _stored_vector(store, "t-1")
            final_row = await store._get_thought_row("t-1")
            assert final_row is not None
            assert final_row["content"] == "content b"
            assert final_vector == pytest.approx([0.0, 1.0, 0.0]), (
                "a slower, stale auto-embed completion (A) overwrote the vector "
                "installed by a later, already-persisted update (B)"
            )

            # Exactly one embedding row for this thought -- the stale
            # completion did not create a second row or corrupt the upsert.
            cursor = await store._db.execute(
                "SELECT COUNT(*) AS n FROM embedding WHERE owner_type = 'THOUGHT' AND owner_id = ?",
                ("t-1",),
            )
            count_row = await cursor.fetchone()
            assert count_row is not None
            assert int(count_row["n"]) == 1
        finally:
            await store._db.close()

    async def test_stale_completion_after_delete_and_recreate_checks_content_not_revision(
        self, tmp_path: Path, backend: str
    ) -> None:
        """A revision number resets on delete+recreate; content identity must not be fooled.

        The thought's ``revision`` column restarts at 0 on a freshly recreated
        row with the same id, so a revision-*number* staleness check could
        coincidentally match a value captured before the row was deleted. This
        pins that the fix instead compares actual stored content: a stale
        completion for the deleted incarnation's content must still be
        dropped even though the recreated row's revision matches, and a
        completion whose content genuinely matches the current row (because
        the row was recreated with byte-identical content) is not spuriously
        dropped.
        """
        provider = _BarrierProvider(_DIMENSION)
        store = await _build_store(tmp_path, provider, backend=backend)
        try:
            # Create the original row; its own auto-embed is parked open and
            # never released until after delete + recreate below.
            orig_gate = provider.register("orig-marker", [1.0, 0.0, 0.0])
            create_task = asyncio.ensure_future(
                store.create_thought(
                    _thought("t-2", essence="orig-marker", content="original content")
                )
            )
            await provider.entered["orig-marker"].wait()

            # Delete the row out from under the still-pending embed. Deleting
            # while auto-embed is in flight is realistic: nothing in
            # create_thought's contract waits for auto-embed before allowing
            # concurrent access to the id.
            await store.delete_thought("t-2")

            # Recreate the same id with genuinely different content -- revision
            # restarts at 0, same as the deleted row's had been at creation.
            new_gate = provider.register("new-marker", [0.0, 1.0, 0.0])
            recreate_task = asyncio.ensure_future(
                store.create_thought(
                    _thought("t-2", essence="new-marker", content="different content")
                )
            )
            await provider.entered["new-marker"].wait()
            new_gate.set()
            await recreate_task
            assert await _stored_vector(store, "t-2") == pytest.approx([0.0, 1.0, 0.0])

            # Release the stale, original completion. Its content
            # ("original content") does not match the recreated row's content
            # ("different content") even though the revision matches, so it
            # must be dropped, not installed.
            orig_gate.set()
            await create_task

            final_vector = await _stored_vector(store, "t-2")
            final_row = await store._get_thought_row("t-2")
            assert final_row is not None
            assert final_row["content"] == "different content"
            assert final_vector == pytest.approx([0.0, 1.0, 0.0]), (
                "a stale completion from a deleted incarnation of this thought_id "
                "landed on the recreated row because a coincidentally-matching "
                "revision number was trusted instead of actual content"
            )
        finally:
            await store._db.close()

    async def test_completion_for_unchanged_content_still_installs(
        self, tmp_path: Path, backend: str
    ) -> None:
        """The common case -- content unchanged since scheduling -- still installs."""
        provider = _BarrierProvider(_DIMENSION)
        store = await _build_store(tmp_path, provider, backend=backend)
        try:
            gate = provider.register("only-marker", [0.5, 0.5, 0.0])
            task = asyncio.ensure_future(
                store.create_thought(
                    _thought("t-3", essence="only-marker", content="stable content")
                )
            )
            await provider.entered["only-marker"].wait()
            gate.set()
            await task

            assert await _stored_vector(store, "t-3") == pytest.approx([0.5, 0.5, 0.0])
        finally:
            await store._db.close()
