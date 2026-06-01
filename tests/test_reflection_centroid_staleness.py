"""REFLECTION centroid staleness tests — re-bind on evolve + recall freshness floor.

A REFLECTION's centroid is the L2-normalized mean of its members' vectors. It
must track the *current* cluster: when a source's essence/content evolves and
the source is re-embedded, the dependent REFLECTION centroids are recomputed so
recall reflects the present state (``score_after < score_before`` once members
drift). Metadata-only edits never re-embed the source, so they must leave the
REFLECTION centroid byte-identical. A retired REFLECTION must not over-recall.

Centroid bytes are read back through raw SQLite where the contract is about the
persisted embedding blob.
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import CallbackProvider, SqliteEngravaCore
from engrava.config import (
    DreamingConfig,
    DreamingGates,
    EdgeCreationConfig,
)
from engrava.domain.enums import (
    LifecycleStatus,
    Priority,
    ThoughtType,
)
from engrava.domain.models.thought import ThoughtRecord
from engrava.extensions.dreaming import DreamingExtension
from engrava.infrastructure.sqlite.centroid import CENTROID_MODEL_NAME, compute_centroid

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


# ---------------------------------------------------------------------------
# Topic-aware deterministic embedding callback
# ---------------------------------------------------------------------------

_PYTHON_VEC = [1.0, 0.0, 0.0]
_COOKING_VEC = [0.0, 1.0, 0.0]
_NEUTRAL_VEC = [0.0, 0.0, 1.0]


def _topic_embed(text: str) -> list[float]:
    """Map text to a topic vector by keyword (deterministic, dim=3)."""
    lowered = text.lower()
    if "cooking" in lowered or "sourdough" in lowered:
        return list(_COOKING_VEC)
    if "python" in lowered or "asyncio" in lowered:
        return list(_PYTHON_VEC)
    return list(_NEUTRAL_VEC)


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[SqliteEngravaCore]:
    """Store with a topic-aware embedding provider and auto-embed enabled."""
    db = await aiosqlite.connect(str(tmp_path / "staleness.db"))
    db.row_factory = aiosqlite.Row
    provider = CallbackProvider(callback=_topic_embed, dimension=3, model_name="topic-3")
    s = SqliteEngravaCore(db=db, embedding_provider=provider, auto_embed=True)
    await s.ensure_schema()
    yield s
    await db.close()


def _obs(tid: str, *, essence: str, content: str = "body") -> ThoughtRecord:
    """Build an ACTIVE OBSERVATION (auto-embedded on create)."""
    return ThoughtRecord(
        thought_id=tid,
        thought_type=ThoughtType.OBSERVATION,
        essence=essence,
        content=content,
        priority=Priority.P3,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=0,
        updated_cycle=0,
        source="test",
        confirmation_count=5,
        confidence=0.9,
    )


def _reflection_cfg() -> DreamingConfig:
    """Config that clusters python-topic sources into one REFLECTION."""
    return DreamingConfig(
        enabled=True,
        promote_threshold=0.0,
        max_p1_fraction=1.0,
        promote_targets="ALL",
        gates=DreamingGates(
            min_age_cycles=0,
            allow_zero_confirmation=True,
            max_promoted_per_run=50,
            enable_reflections=True,
            cluster_algorithm="agglomerative",
            min_cluster_size=2,
            cluster_quality_gating_enabled=False,
        ),
        edges=EdgeCreationConfig(
            enabled=True,
            top_k=5,
            min_similarity=0.5,
            edge_weight_factor=0.5,
        ),
    )


async def _raw_centroid(store: SqliteEngravaCore, reflection_id: str) -> list[float]:
    """Read the stored centroid blob back through raw SQLite and unpack it."""
    cursor = await store._db.execute(
        "SELECT dimension, vector_blob, model_name FROM embedding "
        "WHERE owner_type = 'THOUGHT' AND owner_id = ?",
        (reflection_id,),
    )
    row = await cursor.fetchone()
    assert row is not None, "centroid embedding row missing"
    assert str(row["model_name"]) == CENTROID_MODEL_NAME
    return list(struct.unpack(f"{int(row['dimension'])}f", row["vector_blob"]))


async def _raw_centroid_blob(store: SqliteEngravaCore, reflection_id: str) -> bytes:
    """Read the raw centroid blob bytes (for byte-identity checks)."""
    cursor = await store._db.execute(
        "SELECT vector_blob FROM embedding WHERE owner_type = 'THOUGHT' AND owner_id = ?",
        (reflection_id,),
    )
    row = await cursor.fetchone()
    assert row is not None
    return bytes(row["vector_blob"])


async def _seed_python_reflection(store: SqliteEngravaCore) -> tuple[str, list[str]]:
    """Create three python-topic sources and dream a REFLECTION over them."""
    ids = []
    for i in range(3):
        t = await store.create_thought(_obs(f"py-{i}", essence=f"python async topic {i}"))
        ids.append(t.thought_id)

    ext = DreamingExtension(config=_reflection_cfg())
    result = await ext.run_consolidation(store, current_cycle=1)
    assert result.reflections_created >= 1

    reflections = await store.list_thoughts(thought_type=ThoughtType.REFLECTION)
    rid = reflections[0].thought_id
    sources = await store.consolidated_member_ids(rid)
    assert len(sources) >= 3
    return rid, sources


def _cosine(a: list[float], b: list[float]) -> float:
    """Plain cosine similarity (vectors are unit-ish; guard zero norm)."""
    import math

    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# D2 — centroid re-bind on essence/content evolve
# ---------------------------------------------------------------------------


class TestCentroidRebindOnEvolve:
    """The centroid is recomputed when sources are re-embedded on evolve."""

    async def test_centroid_changes_after_majority_drift(self, store: SqliteEngravaCore) -> None:
        """Drifting a majority of sources to a disjoint topic moves the centroid."""
        rid, sources = await _seed_python_reflection(store)
        before = await _raw_centroid(store, rid)

        # Drift 2 of 3 sources from python -> cooking (essence change re-embeds).
        await store.update_thought(sources[0], essence="cooking sourdough bread")
        await store.update_thought(sources[1], essence="cooking sourdough starter")

        after = await _raw_centroid(store, rid)
        # Beyond float32 noise: the centroid vector actually moved.
        delta = max(abs(a - b) for a, b in zip(before, after, strict=True))
        assert delta > 1e-3

    async def test_rebound_centroid_equals_fresh_mean(self, store: SqliteEngravaCore) -> None:
        """Re-bound centroid equals a freshly computed mean over current members."""
        rid, sources = await _seed_python_reflection(store)
        await store.update_thought(sources[0], essence="cooking sourdough bread")

        # Independently recompute the expected centroid from current members.
        member_vectors: list[list[float]] = []
        for sid in sources:
            emb = await store.get_embedding(sid)
            assert emb is not None
            member_vectors.append(list(struct.unpack(f"{emb.dimension}f", emb.vector_blob)))
        expected = compute_centroid(member_vectors)

        stored = await _raw_centroid(store, rid)
        for got, exp in zip(stored, expected, strict=True):
            assert got == pytest.approx(exp, abs=1e-6)

    async def test_rebind_overwrites_in_place_no_duplicate_row(
        self, store: SqliteEngravaCore
    ) -> None:
        """Re-bind upserts the same centroid row — no duplicate embedding row."""
        rid, sources = await _seed_python_reflection(store)
        await store.update_thought(sources[0], essence="cooking sourdough bread")

        cursor = await store._db.execute(
            "SELECT COUNT(*) AS n FROM embedding WHERE owner_type = 'THOUGHT' AND owner_id = ?",
            (rid,),
        )
        row = await cursor.fetchone()
        assert row is not None
        assert int(row["n"]) == 1


# ---------------------------------------------------------------------------
# D2 guard — metadata-only edits MUST NOT re-bind
# ---------------------------------------------------------------------------


class TestMetadataOnlyDoesNotRebind:
    """Metadata/priority churn leaves the REFLECTION centroid byte-identical."""

    async def test_priority_only_edit_keeps_centroid_byte_identical(
        self, store: SqliteEngravaCore
    ) -> None:
        """Changing only a source's priority leaves the centroid blob identical."""
        rid, sources = await _seed_python_reflection(store)
        before = await _raw_centroid_blob(store, rid)

        # Priority-only edit: no essence/content change -> source not re-embedded.
        await store.update_thought(sources[0], priority="P1")

        after = await _raw_centroid_blob(store, rid)
        assert after == before  # byte-identical

    async def test_metadata_only_edit_keeps_centroid_byte_identical(
        self, store: SqliteEngravaCore
    ) -> None:
        """Changing only a source's metadata leaves the centroid blob identical."""
        rid, sources = await _seed_python_reflection(store)
        before = await _raw_centroid_blob(store, rid)

        await store.update_thought(sources[0], metadata={"note": "annotation only"})

        after = await _raw_centroid_blob(store, rid)
        assert after == before


# ---------------------------------------------------------------------------
# D3 / 009 — recall reflects current state; fresh not suppressed
# ---------------------------------------------------------------------------


class TestRecallReflectsCurrentState:
    """Drift drops recall on the original topic; fresh REFLECTIONs keep surfacing."""

    async def test_score_after_less_than_before_for_original_topic(
        self, store: SqliteEngravaCore
    ) -> None:
        """After majority drift, recall for the original (python) topic drops."""
        rid, sources = await _seed_python_reflection(store)
        before = await _raw_centroid(store, rid)
        score_before = _cosine(before, _PYTHON_VEC)

        await store.update_thought(sources[0], essence="cooking sourdough bread")
        await store.update_thought(sources[1], essence="cooking sourdough starter")

        after = await _raw_centroid(store, rid)
        score_after = _cosine(after, _PYTHON_VEC)

        assert score_after < score_before

    async def test_fresh_reflection_not_suppressed(self, store: SqliteEngravaCore) -> None:
        """A REFLECTION whose sources did NOT drift keeps its recall score."""
        rid, sources = await _seed_python_reflection(store)
        before = await _raw_centroid(store, rid)
        score_before = _cosine(before, _PYTHON_VEC)

        # A metadata-only edit (no drift) must not change recall.
        await store.update_thought(sources[0], priority="P1")

        after = await _raw_centroid(store, rid)
        score_after = _cosine(after, _PYTHON_VEC)

        assert score_after == pytest.approx(score_before, abs=1e-9)
        # And it still surfaces in hybrid search for its topic.
        hybrid = await store.search_hybrid("python", _PYTHON_VEC, top_k=10)
        assert rid in {tid for tid, _ in hybrid.results}

    async def test_reflection_boost_semantics_untouched(self, store: SqliteEngravaCore) -> None:
        """The freshness floor adds no scoring axis: boost still multiplies fresh scores."""
        rid, _ = await _seed_python_reflection(store)

        plain = await store.search_hybrid("python", _PYTHON_VEC, top_k=10, reflection_boost=1.0)
        boosted = await store.search_hybrid("python", _PYTHON_VEC, top_k=10, reflection_boost=2.0)

        plain_score = dict(plain.results).get(rid)
        boosted_score = dict(boosted.results).get(rid)
        assert plain_score is not None
        assert boosted_score is not None
        assert boosted_score > plain_score


# ---------------------------------------------------------------------------
# compute_centroid unit tests (the shared create + re-bind math)
# ---------------------------------------------------------------------------


class TestComputeCentroid:
    """The shared centroid helper is a deterministic L2-normalized mean."""

    def test_mean_then_normalized(self) -> None:
        """Two orthonormal vectors average to the normalized diagonal."""
        result = compute_centroid([[1.0, 0.0], [0.0, 1.0]])
        inv = 1.0 / (2.0**0.5)
        assert result[0] == pytest.approx(inv)
        assert result[1] == pytest.approx(inv)

    def test_unit_length_when_nonzero(self) -> None:
        """A non-zero centroid is returned at unit length."""
        result = compute_centroid([[3.0, 4.0], [3.0, 4.0]])
        norm = sum(x * x for x in result) ** 0.5
        assert norm == pytest.approx(1.0)

    def test_zero_mean_returns_zeros(self) -> None:
        """Members that cancel out yield the zero vector (no div-by-zero)."""
        result = compute_centroid([[1.0, 0.0], [-1.0, 0.0]])
        assert result == [0.0, 0.0]

    def test_empty_raises(self) -> None:
        """An empty member set is a programming error, not a silent zero."""
        with pytest.raises(ValueError, match="at least one member vector"):
            compute_centroid([])
