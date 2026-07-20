"""Uniform cross-arm liveness invariant: no retrieval arm surfaces dead rows.

Two row classes must never reach a caller through *any* retrieval path: a
thought whose ``expires_at`` has passed (TTL expiry), and a retired REFLECTION
(a ``REFLECTION`` whose ``lifecycle_status`` is no longer ``ACTIVE`` — an orphan
archived once its cluster left the active set, which must not over-recall on its
now-stale centroid). Each arm enforces this independently — the FTS and numpy
vector arms in their SQL ``WHERE``, the ``vec0`` arm in a post-``MATCH`` filter,
the query-less fallback in its own ``WHERE`` — so a single shared gate is not
what keeps them out.

:class:`TestCrossArmLiveness` pins the invariant uniformly: a live decoy sharing
the same vocabulary is returned by every arm (proof the arm executed), while the
expired thought and the retired REFLECTION are returned by none — not the FTS
arm, not the vector arm, not hybrid fusion, and not the all-signals-off fallback.

:class:`TestLivenessDiscriminatingPower` verifies the invariant's power on the
``vec0`` arm, whose liveness gate is an isolable seam
(:meth:`SqliteEngravaCore._filter_expired_results`): reverting it to an identity
makes the dead rows leak through the vector arm *only*, while the FTS arm — whose
gate is independent — still excludes them.
"""

from __future__ import annotations

import datetime
import hashlib
from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import CallbackProvider, SqliteEngravaCore
from engrava.domain.enums import (
    KnowledgeSource,
    LifecycleStatus,
    Priority,
    ThoughtType,
    ThoughtVisibility,
)
from engrava.domain.models.thought import ThoughtRecord
from tests.test_sqlite_vec import sqlite_vec_required

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


_EMBED_DIM = 32

# All three rows share this vocabulary, so every arm has a genuine reason to
# surface each of them — and the only thing keeping the dead rows out is the
# liveness gate, not a vocabulary mismatch.
_SHARED_VOCAB = "alpha beta gamma shared retrieval vocabulary"
_QUERY_TEXT = "alpha beta gamma shared"

_LIVE_ID = "t-live"
_EXPIRED_ID = "t-expired"
_RETIRED_ID = "t-retired"


def _embed(text: str) -> list[float]:
    """Embed text as an L2-normalized bag-of-words hashing vector.

    Args:
        text: Input text to embed.

    Returns:
        An ``_EMBED_DIM``-length unit vector.
    """
    vector = [0.0] * _EMBED_DIM
    for token in text.lower().split():
        digest = hashlib.sha1(token.encode("utf-8")).digest()  # noqa: S324
        vector[int.from_bytes(digest[:4], "big") % _EMBED_DIM] += 1.0
    norm = sum(value * value for value in vector) ** 0.5
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def _past_iso() -> str:
    """Return an ISO-8601 UTC timestamp a day in the past (already expired)."""
    return (datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=1)).isoformat()


def _thought(
    thought_id: str,
    *,
    thought_type: ThoughtType = ThoughtType.OBSERVATION,
    lifecycle_status: LifecycleStatus = LifecycleStatus.ACTIVE,
    expires_at: str | None = None,
) -> ThoughtRecord:
    """Build a stored thought sharing the suite's vocabulary.

    Args:
        thought_id: Stable identifier used to assert retrieval.
        thought_type: Classification (a REFLECTION for the retired-orphan case).
        lifecycle_status: Lifecycle state (non-ACTIVE for the retired case).
        expires_at: Optional ISO-8601 expiry (a past value for the expired case).

    Returns:
        A fully populated :class:`ThoughtRecord` ready for ``create_thought``.
    """
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=thought_type,
        essence=_SHARED_VOCAB,
        content=_SHARED_VOCAB,
        priority=Priority.P2,
        lifecycle_status=lifecycle_status,
        created_cycle=0,
        updated_cycle=0,
        source="test",
        confidence=0.8,
        source_type=KnowledgeSource.EXPERIENCE,
        visibility=ThoughtVisibility.SELECTIVE,
        expires_at=expires_at,
    )


@pytest.fixture
async def liveness_store() -> AsyncIterator[SqliteEngravaCore]:
    """Return a store holding a live decoy, an expired thought, and a retired REFLECTION.

    All three share the same vocabulary and (via ``auto_embed``) carry an
    embedding, so both the FTS and the vector arm have a genuine reason to
    surface each — leaving the liveness gate as the sole thing that keeps the
    dead rows out.

    Yields:
        A :class:`SqliteEngravaCore` with a deterministic vector arm.
    """
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode = WAL")
    await conn.execute("PRAGMA foreign_keys = ON")
    provider = CallbackProvider(
        callback=_embed,
        dimension=_EMBED_DIM,
        model_name="bag-of-words-liveness",
    )
    store = SqliteEngravaCore(conn, embedding_provider=provider, auto_embed=True)
    await store.ensure_schema()
    await store.create_thought(_thought(_LIVE_ID))
    await store.create_thought(_thought(_EXPIRED_ID, expires_at=_past_iso()))
    await store.create_thought(
        _thought(
            _RETIRED_ID,
            thought_type=ThoughtType.REFLECTION,
            lifecycle_status=LifecycleStatus.ARCHIVED,
        )
    )
    yield store
    await conn.close()


def _ids(results: list[tuple[str, float]]) -> set[str]:
    """Collect thought ids from scored results.

    Args:
        results: ``(thought_id, score)`` pairs.

    Returns:
        The set of returned thought ids.
    """
    return {thought_id for thought_id, _ in results}


class TestCrossArmLiveness:
    """No arm — FTS, vector, hybrid, or the all-off fallback — surfaces a dead row.

    The live decoy is present everywhere (the arm executed); the expired thought
    and the retired REFLECTION are present nowhere.
    """

    async def test_fts_arm_excludes_dead_rows(self, liveness_store: SqliteEngravaCore) -> None:
        """The FTS arm returns the live decoy and neither dead row."""
        ids = _ids(await liveness_store.search_fts(_QUERY_TEXT, top_k=50))
        assert _LIVE_ID in ids, "FTS arm must surface the live decoy (arm did not execute)"
        assert _EXPIRED_ID not in ids
        assert _RETIRED_ID not in ids

    async def test_vector_arm_excludes_dead_rows(self, liveness_store: SqliteEngravaCore) -> None:
        """The vector arm returns the live decoy and neither dead row."""
        ids = _ids(await liveness_store.search_similar(_embed(_QUERY_TEXT), top_k=50))
        assert _LIVE_ID in ids, "vector arm must surface the live decoy (arm did not execute)"
        assert _EXPIRED_ID not in ids
        assert _RETIRED_ID not in ids

    async def test_hybrid_excludes_dead_rows(self, liveness_store: SqliteEngravaCore) -> None:
        """Hybrid fusion returns the live decoy and neither dead row, both arms live."""
        result = await liveness_store.search_hybrid(_QUERY_TEXT, top_k=50)
        assert "fts5" in result.backends_used
        assert "vector" in result.backends_used
        ids = _ids(result.results)
        assert _LIVE_ID in ids, "hybrid must surface the live decoy"
        assert _EXPIRED_ID not in ids
        assert _RETIRED_ID not in ids

    async def test_all_off_fallback_excludes_dead_rows(
        self,
        liveness_store: SqliteEngravaCore,
    ) -> None:
        """The all-signals-off fallback (list-thoughts path) also excludes dead rows.

        With no query text, no vector, and every non-lexical weight zeroed,
        ``search_hybrid`` falls back to the query-less ranked window — which must
        enforce the same liveness contract as the arms.
        """
        result = await liveness_store.search_hybrid(
            "",
            None,
            top_k=50,
            recency_weight=0.0,
            priority_weight=0.0,
            graph_weight=0.0,
        )
        ids = _ids(result.results)
        assert _LIVE_ID in ids, "fallback must still surface the live row"
        assert _EXPIRED_ID not in ids
        assert _RETIRED_ID not in ids


@sqlite_vec_required
class TestLivenessDiscriminatingPower:
    """Reverting the vec0 liveness gate makes dead rows leak through that arm only."""

    async def _build_vec0_store(self, tmp_path: Path) -> SqliteEngravaCore:
        """Build a real ``vec0``-backed store seeded with live + dead rows.

        Mirrors the ``from_config`` bootstrap (schema then
        ``_configure_vector_backend``). Each row is FTS-indexed via
        ``create_thought`` and given a 2-D embedding placed near the query axis
        so the ``vec0`` KNN surfaces all three before the liveness filter runs.

        Args:
            tmp_path: Pytest temp directory for the on-disk database.

        Returns:
            The seeded ``vec0``-backed store.
        """
        db = await aiosqlite.connect(str(tmp_path / "vec0-liveness.db"))
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA foreign_keys = ON")
        store = SqliteEngravaCore(db)
        store._owns_connection = True
        await store.ensure_schema()
        await store._configure_vector_backend(backend_name="sqlite-vec", embedding_dimension=2)

        await store.create_thought(_thought(_LIVE_ID))
        await store.create_thought(_thought(_EXPIRED_ID, expires_at=_past_iso()))
        await store.create_thought(
            _thought(
                _RETIRED_ID,
                thought_type=ThoughtType.REFLECTION,
                lifecycle_status=LifecycleStatus.ARCHIVED,
            )
        )
        # All three sit right on the query axis so the KNN returns each; only the
        # post-MATCH liveness filter then removes the dead ones.
        await store.store_embedding(thought_id=_LIVE_ID, vector=[1.0, 0.0], model_name="m2")
        await store.store_embedding(thought_id=_EXPIRED_ID, vector=[0.999, 0.001], model_name="m2")
        await store.store_embedding(thought_id=_RETIRED_ID, vector=[0.998, 0.002], model_name="m2")
        return store

    async def test_reverting_vec0_filter_leaks_through_vector_arm_only(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Stripping ``_filter_expired_results`` leaks dead rows through the vector arm only."""
        store = await self._build_vec0_store(tmp_path)
        try:
            query = [1.0, 0.0]

            # Baseline: both arms exclude the dead rows.
            vec_ids = _ids(await store.search_similar(query, top_k=10))
            fts_ids = _ids(await store.search_fts(_QUERY_TEXT, top_k=10))
            assert _LIVE_ID in vec_ids
            assert _LIVE_ID in fts_ids
            assert {_EXPIRED_ID, _RETIRED_ID} & (vec_ids | fts_ids) == set()

            # Revert the vec0 liveness gate to a no-op post-filter.
            async def _identity(
                results: list[tuple[str, float]],
            ) -> list[tuple[str, float]]:
                return results

            monkeypatch.setattr(store, "_filter_expired_results", _identity)

            leaked_vec = _ids(await store.search_similar(query, top_k=10))
            still_clean_fts = _ids(await store.search_fts(_QUERY_TEXT, top_k=10))

            # The dead rows now leak through the vector arm...
            assert _EXPIRED_ID in leaked_vec
            assert _RETIRED_ID in leaked_vec
            # ...but the FTS arm's independent gate still excludes them.
            assert _EXPIRED_ID not in still_clean_fts
            assert _RETIRED_ID not in still_clean_fts
        finally:
            await store.close()
