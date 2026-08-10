"""Uniform cross-arm liveness invariant: no retrieval arm surfaces dead rows.

Three row classes must never reach a caller through *any* default retrieval
path: a thought whose ``expires_at`` has passed (TTL expiry); a retired
REFLECTION (a ``REFLECTION`` whose ``lifecycle_status`` is no longer ``ACTIVE``
— an orphan archived once its cluster left the active set, which must not
over-recall on its now-stale centroid); and an **archived** regular thought (a
non-REFLECTION whose ``lifecycle_status`` is ``ARCHIVED`` — forgotten by the
hygiene loop or TTL-archived, excluded from the default candidate set but
reversibly, via :meth:`SqliteEngravaCore.restore_thought`). Each arm enforces
this independently — the FTS and numpy vector arms in their SQL ``WHERE``, the
``vec0`` arm in a post-``MATCH`` filter, the query-less fallback in its own
``WHERE`` — so a single shared gate is not what keeps them out.

The archived exclusion, unlike expiry and the retired-REFLECTION floor, is
reversible per-call: passing ``include_archived=True`` re-admits archived rows
across every arm without restoring them (the "search my archive" escape hatch).
The retired-REFLECTION floor is *not* relaxed by that flag — it is an
independent gate.

:class:`TestCrossArmLiveness` pins the invariant uniformly: a live decoy sharing
the same vocabulary is returned by every arm (proof the arm executed), while the
expired thought, the retired REFLECTION, and the archived thought are returned by
none — not the FTS arm, not the vector arm, not hybrid fusion, and not the
all-signals-off fallback — yet the archived row is re-admitted under
``include_archived=True``.

:class:`TestLivenessDiscriminatingPower` verifies the invariant's power on the
``vec0`` arm, whose liveness gate is an isolable seam
(:meth:`SqliteEngravaCore._filter_expired_results`): reverting it to an identity
makes the dead rows leak through the vector arm *only*, while the FTS arm — whose
gate is independent — still excludes them.

:class:`TestArchivedExclusionDiscriminatingPower` proves the archived clause is
load-bearing per path: forcing an individual arm to skip its archived clause
(via ``include_archived=True`` on that one path) leaks the archived row through
that arm alone, while the others still exclude it.
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
_ARCHIVED_ID = "t-archived"


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
    """Return a store holding a live decoy plus three excluded rows.

    The excluded rows are an expired thought, a retired REFLECTION, and an
    archived regular OBSERVATION. All four share the same vocabulary and (via
    ``auto_embed``) carry an embedding, so both the FTS and the vector arm have a
    genuine reason to surface each — leaving the liveness / archived gate as the
    sole thing that keeps the excluded rows out. The archived row is created
    ACTIVE (so it is embedded) and then transitioned to ``ARCHIVED`` through the
    real ``ACTIVE -> ARCHIVED`` lifecycle edge, mirroring how the hygiene loop
    forgets a thought.

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
    await store.create_thought(_thought(_ARCHIVED_ID))
    await store.update_thought(_ARCHIVED_ID, lifecycle_status=LifecycleStatus.ARCHIVED)
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
        assert _ARCHIVED_ID not in ids


class TestArchivedExclusionAcrossArms:
    """Archived regular thoughts leave every default arm — and return on opt-in.

    A non-REFLECTION thought whose ``lifecycle_status`` is ``ARCHIVED`` shares the
    suite vocabulary and carries an embedding, so each arm has a genuine reason
    to surface it; only the archived-exclusion clause keeps it out. Passing
    ``include_archived=True`` re-admits it on the same arm.
    """

    async def test_fts_arm_excludes_archived(self, liveness_store: SqliteEngravaCore) -> None:
        """The FTS arm drops the archived row by default and re-admits it on opt-in."""
        default_ids = _ids(await liveness_store.search_fts(_QUERY_TEXT, top_k=50))
        assert _LIVE_ID in default_ids
        assert _ARCHIVED_ID not in default_ids
        opt_in_ids = _ids(
            await liveness_store.search_fts(_QUERY_TEXT, top_k=50, include_archived=True)
        )
        assert _ARCHIVED_ID in opt_in_ids

    async def test_vector_arm_excludes_archived(self, liveness_store: SqliteEngravaCore) -> None:
        """The vector arm drops the archived row by default and re-admits it on opt-in."""
        query = _embed(_QUERY_TEXT)
        default_ids = _ids(await liveness_store.search_similar(query, top_k=50))
        assert _LIVE_ID in default_ids
        assert _ARCHIVED_ID not in default_ids
        opt_in_ids = _ids(
            await liveness_store.search_similar(query, top_k=50, include_archived=True)
        )
        assert _ARCHIVED_ID in opt_in_ids

    async def test_hybrid_excludes_archived(self, liveness_store: SqliteEngravaCore) -> None:
        """Hybrid fusion drops the archived row by default and re-admits it on opt-in."""
        default_ids = _ids((await liveness_store.search_hybrid(_QUERY_TEXT, top_k=50)).results)
        assert _LIVE_ID in default_ids
        assert _ARCHIVED_ID not in default_ids
        opt_in_ids = _ids(
            (
                await liveness_store.search_hybrid(_QUERY_TEXT, top_k=50, include_archived=True)
            ).results
        )
        assert _ARCHIVED_ID in opt_in_ids

    async def test_recall_excludes_archived(self, liveness_store: SqliteEngravaCore) -> None:
        """``recall`` (the ergonomic hybrid shorthand) honours the archived exclusion."""
        default_ids = _ids((await liveness_store.recall(_QUERY_TEXT, top_k=50)).results)
        assert _LIVE_ID in default_ids
        assert _ARCHIVED_ID not in default_ids
        opt_in_ids = _ids(
            (await liveness_store.recall(_QUERY_TEXT, top_k=50, include_archived=True)).results
        )
        assert _ARCHIVED_ID in opt_in_ids

    async def test_fallback_excludes_archived(self, liveness_store: SqliteEngravaCore) -> None:
        """The all-signals-off fallback excludes the archived row and re-admits it on opt-in."""
        default = await liveness_store.search_hybrid(
            "", None, top_k=50, recency_weight=0.0, priority_weight=0.0, graph_weight=0.0
        )
        assert _LIVE_ID in _ids(default.results)
        assert _ARCHIVED_ID not in _ids(default.results)
        opt_in = await liveness_store.search_hybrid(
            "",
            None,
            top_k=50,
            recency_weight=0.0,
            priority_weight=0.0,
            graph_weight=0.0,
            include_archived=True,
        )
        assert _ARCHIVED_ID in _ids(opt_in.results)

    async def test_retired_reflection_floor_survives_include_archived(
        self,
        liveness_store: SqliteEngravaCore,
    ) -> None:
        """``include_archived=True`` re-admits the archived thought but NOT the retired REFLECTION.

        The retired-REFLECTION freshness floor is an independent gate; the
        archived escape hatch must not relax it, or a stale synthesis would
        over-recall on its now-dead centroid.
        """
        ids = _ids(
            (
                await liveness_store.search_hybrid(_QUERY_TEXT, top_k=50, include_archived=True)
            ).results
        )
        assert _ARCHIVED_ID in ids
        assert _RETIRED_ID not in ids
        assert _EXPIRED_ID not in ids


class TestArchivedExclusionDiscriminatingPower:
    """Skipping the archived clause on ONE arm leaks the archived row through that arm only.

    Each arm's archived clause is conditional on ``include_archived``. Forcing an
    individual arm to run with ``include_archived=True`` (while the caller asked
    for the default) reproduces "this arm forgot its archived clause": the
    archived row leaks through that arm, and only that arm.
    """

    async def test_forcing_fts_arm_leaks_archived_through_fts_only(
        self,
        liveness_store: SqliteEngravaCore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Dropping the FTS archived clause leaks the archived row through the FTS arm only."""
        original = liveness_store.search_fts

        async def _leaky_fts(
            query: str,
            top_k: int = 10,
            *,
            include_archived: bool = False,
            _filter_clause: object = None,
        ) -> list[tuple[str, float]]:
            return await original(
                query,
                top_k=top_k,
                include_archived=True,
                _filter_clause=_filter_clause,  # type: ignore[arg-type]
            )

        monkeypatch.setattr(liveness_store, "search_fts", _leaky_fts)

        leaked_fts = _ids(await liveness_store.search_fts(_QUERY_TEXT, top_k=50))
        clean_vec = _ids(await liveness_store.search_similar(_embed(_QUERY_TEXT), top_k=50))
        assert _ARCHIVED_ID in leaked_fts
        assert _ARCHIVED_ID not in clean_vec

    async def test_forcing_numpy_vector_arm_leaks_archived_through_vector_only(
        self,
        liveness_store: SqliteEngravaCore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Dropping the numpy vector arm's archived clause leaks the row through that arm only."""
        original = liveness_store._search_similar_numpy

        async def _leaky_numpy(
            query_vector: list[float],
            top_k: int = 10,
            threshold: float = 0.0,
            *,
            include_archived: bool = False,
            _filter_clause: object = None,
        ) -> list[tuple[str, float]]:
            return await original(
                query_vector,
                top_k,
                threshold,
                include_archived=True,
                _filter_clause=_filter_clause,  # type: ignore[arg-type]
            )

        monkeypatch.setattr(liveness_store, "_search_similar_numpy", _leaky_numpy)

        leaked_vec = _ids(await liveness_store.search_similar(_embed(_QUERY_TEXT), top_k=50))
        clean_fts = _ids(await liveness_store.search_fts(_QUERY_TEXT, top_k=50))
        assert _ARCHIVED_ID in leaked_vec
        assert _ARCHIVED_ID not in clean_fts


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
        await store.create_thought(_thought(_ARCHIVED_ID))
        await store.update_thought(_ARCHIVED_ID, lifecycle_status=LifecycleStatus.ARCHIVED)
        # All four sit right on the query axis so the KNN returns each; only the
        # post-MATCH liveness / archived filter then removes the excluded ones.
        await store.store_embedding(thought_id=_LIVE_ID, vector=[1.0, 0.0], model_name="m2")
        await store.store_embedding(thought_id=_EXPIRED_ID, vector=[0.999, 0.001], model_name="m2")
        await store.store_embedding(thought_id=_RETIRED_ID, vector=[0.998, 0.002], model_name="m2")
        await store.store_embedding(thought_id=_ARCHIVED_ID, vector=[0.997, 0.003], model_name="m2")
        return store

    async def test_reverting_vec0_filter_leaks_through_vector_arm_only(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Stripping ``_filter_expired_results`` leaks dead + archived rows through vector only."""
        store = await self._build_vec0_store(tmp_path)
        try:
            query = [1.0, 0.0]

            # Baseline: both arms exclude the dead and archived rows.
            vec_ids = _ids(await store.search_similar(query, top_k=10))
            fts_ids = _ids(await store.search_fts(_QUERY_TEXT, top_k=10))
            assert _LIVE_ID in vec_ids
            assert _LIVE_ID in fts_ids
            assert {_EXPIRED_ID, _RETIRED_ID, _ARCHIVED_ID} & (vec_ids | fts_ids) == set()

            # Revert the vec0 liveness + archived gate to a no-op post-filter.
            async def _identity(
                results: list[tuple[str, float]],
                *,
                include_archived: bool = False,
            ) -> list[tuple[str, float]]:
                return results

            monkeypatch.setattr(store, "_filter_expired_results", _identity)

            leaked_vec = _ids(await store.search_similar(query, top_k=10))
            still_clean_fts = _ids(await store.search_fts(_QUERY_TEXT, top_k=10))

            # The dead and archived rows now leak through the vector arm...
            assert _EXPIRED_ID in leaked_vec
            assert _RETIRED_ID in leaked_vec
            assert _ARCHIVED_ID in leaked_vec
            # ...but the FTS arm's independent gate still excludes them.
            assert _EXPIRED_ID not in still_clean_fts
            assert _RETIRED_ID not in still_clean_fts
            assert _ARCHIVED_ID not in still_clean_fts
        finally:
            await store.close()
