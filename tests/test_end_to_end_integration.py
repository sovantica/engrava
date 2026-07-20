"""True end-to-end integration flows over a real on-disk SQLite database.

The per-PR ``tests/`` suite is overwhelmingly unit + contract work on an
in-memory database with mocked embedders; the real backends (the
sentence-transformer embedder, the ``sqlite-vec`` ``vec0`` index, the extension
migrations) are exercised in isolation but never *composed*. These tests close
that gap: each one runs against a real ``tmp_path`` **file** database and drives
several subsystems together across the boundaries the unit tests mock.

Three flows are covered:

* :func:`test_full_pipeline_derive_embed_recall` — ingest → derived-records seam
  → auto-embed with the cached MiniLM → hybrid recall of the derived child.
* :func:`test_reopen_is_noop_and_query_is_identical` — durability across a close
  and a fresh reopen: ``ensure_schema`` is a no-op and the identical query
  returns the identical ranked list (both numpy and ``vec0`` arms).
* :func:`test_multi_extension_load_order_converges` — the ``vec0`` backend, the
  structural-split seam and the dreaming extension composed on one store, run
  through a consolidation cycle that must converge without drift.

Real model loads are gated behind ``sentence-transformers`` / ``sqlite-vec``
availability and stay offline-when-cached (enforced by the top-level
``conftest``), so a warm machine never reaches the network.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import (
    CallbackProvider,
    DeriveGates,
    EdgeType,
    LifecycleStatus,
    Priority,
    SqliteEngravaCore,
    StructuralSplitProducer,
    ThoughtType,
)
from engrava.config import DreamingConfig, DreamingGates, EdgeCreationConfig
from engrava.domain.models.thought import ThoughtRecord
from engrava.extensions.dreaming import DreamingExtension
from engrava.infrastructure.sqlite.engrava_core import _derived_thought_id

if TYPE_CHECKING:
    from pathlib import Path

    from engrava.domain.protocols.embedding_provider import (
        EmbeddingProviderProtocol,
    )


# ---------------------------------------------------------------------------
# Availability gates + shared fixtures
# ---------------------------------------------------------------------------

requires_local_embeddings = pytest.mark.skipif(
    importlib.util.find_spec("sentence_transformers") is None,
    reason="sentence-transformers not installed (engrava[embeddings-local] extra)",
)
requires_sqlite_vec = pytest.mark.skipif(
    importlib.util.find_spec("sqlite_vec") is None,
    reason="sqlite-vec package not installed",
)

#: The cached MiniLM the whole suite already loads (see ``conftest``); using the
#: L6 variant keeps the real-embedder flows meaningful without a second model.
_REAL_MODEL_NAME = "all-MiniLM-L6-v2"

#: Minimum fused-score separation between the gold-relevant rows and the
#: unrelated distractors in the full-pipeline recall. Chosen well below the
#: observed gap (~0.33 on L6) so the assertion is a clear-margin check that
#: survives model-minor-version score jitter, not a brittle hair-tie on rank.
_GOLD_MARGIN = 0.1


@pytest.fixture(scope="module")
def real_minilm_provider() -> EmbeddingProviderProtocol:
    """Return the cached MiniLM-L6 provider, loaded once per module.

    Module scope amortises the one-shot cold load across every real-embedder
    flow. The provider loads its model lazily (on first embed), and the
    top-level ``conftest`` forces offline mode when the model is cached, so this
    never reaches the network on a warm machine.
    """
    from engrava.embeddings.sentence_transformer import SentenceTransformerProvider

    return SentenceTransformerProvider(model_name=_REAL_MODEL_NAME)


def _thought(
    thought_id: str,
    *,
    essence: str,
    content: str,
    priority: Priority = Priority.P2,
) -> ThoughtRecord:
    """Build a realistic ACTIVE source thought for the integration flows."""
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.OBSERVATION,
        essence=essence,
        content=content,
        priority=priority,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=0,
        updated_cycle=0,
        source="test",
        confidence=0.9,
        confirmation_count=5,
    )


async def _user_version(conn: aiosqlite.Connection) -> int:
    """Return the SQLite ``user_version`` pragma for the connection."""
    cursor = await conn.execute("PRAGMA user_version")
    row = await cursor.fetchone()
    assert row is not None
    return int(row[0])


async def _scalar_count(conn: aiosqlite.Connection, sql: str) -> int:
    """Return the first column of the first row of a ``COUNT(*)`` query."""
    cursor = await conn.execute(sql)
    row = await cursor.fetchone()
    assert row is not None
    return int(row[0])


async def _embedding_created_at(
    store: SqliteEngravaCore,
    thought_ids: list[str],
) -> dict[str, str]:
    """Map each thought id to its embedding ``created_at`` fingerprint."""
    fingerprint: dict[str, str] = {}
    for thought_id in thought_ids:
        embedding = await store.get_embedding(thought_id)
        assert embedding is not None, f"missing embedding for {thought_id}"
        fingerprint[thought_id] = embedding.created_at
    return fingerprint


# ---------------------------------------------------------------------------
# 1. Full pipeline: derive -> auto-embed (real MiniLM) -> hybrid recall
# ---------------------------------------------------------------------------


@requires_local_embeddings
async def test_full_pipeline_derive_embed_recall(
    tmp_path: Path,
    real_minilm_provider: EmbeddingProviderProtocol,
) -> None:
    """Ingest → derived-records seam → auto-embed → hybrid recall on a file DB.

    Crosses every boundary the unit tests mock in isolation: the structural
    split derives a child per paragraph, each child auto-embeds through the real
    cached MiniLM, and a natural-language recall surfaces the gold derived child
    over unrelated noise while the source survives.
    """
    db_path = tmp_path / "pipeline.db"
    conn = await aiosqlite.connect(str(db_path))
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA foreign_keys = ON")
    store = SqliteEngravaCore(
        conn,
        hooks=StructuralSplitProducer(),
        embedding_provider=real_minilm_provider,
        auto_embed=True,
        derive_gates=DeriveGates(enabled=True),
    )
    await store.ensure_schema()
    try:
        # The gold fact lives ONLY in paragraph A; paragraph B is filler. The
        # split derives one child per paragraph, so the gold answer becomes its
        # own retrievable derived thought.
        gold_para = "The Larkspur roadside diner served warm blueberry pie at sunset."
        filler_para = "We topped up the tank and checked the tyre pressure before leaving."
        source = _thought(
            "src-roadtrip",
            essence="Road trip diary",
            content=f"{gold_para}\n\n{filler_para}",
        )
        await store.create_thought(source)
        # Single-paragraph distractors derive nothing and must not win recall.
        await store.create_thought(
            _thought(
                "noise-tax",
                essence="Finance",
                content="I filed the quarterly tax estimate on the accountant portal.",
            ),
        )
        await store.create_thought(
            _thought(
                "noise-gym",
                essence="Fitness",
                content="The gym swapped the treadmills for rowing machines this week.",
            ),
        )

        gold_child_id = _derived_thought_id(gold_para)
        filler_child_id = _derived_thought_id(filler_para)

        # The derived children crossed the derive + embed boundaries: a child row
        # exists, carries the segment content, and auto-embedded via real MiniLM.
        gold_child = await store.get_thought(gold_child_id)
        assert gold_child is not None
        assert gold_child.content == gold_para
        gold_child_embedding = await store.get_embedding(gold_child_id)
        assert gold_child_embedding is not None
        assert gold_child_embedding.model_name == _REAL_MODEL_NAME

        # Both children link back to the source with a DERIVED_FROM edge.
        in_edges = await store.get_edges("src-roadtrip", direction="IN")
        assert {edge.from_thought_id for edge in in_edges} == {
            gold_child_id,
            filler_child_id,
        }
        assert all(edge.edge_type == EdgeType.DERIVED_FROM for edge in in_edges)

        # The source itself survives (durable alongside its children).
        assert await store.get_thought("src-roadtrip") is not None

        # Hybrid recall (real FTS arm + real vector arm) surfaces the gold child.
        result = await store.recall(
            "where did we stop for warm blueberry pie on the road trip",
            top_k=5,
        )
        scores = dict(result.results)
        # The gold answer is recalled: both the derived child (per the WS) and
        # its source thought are retrievable.
        assert gold_child_id in scores, scores
        assert "src-roadtrip" in scores, scores
        # The gold-relevant rows clear the unrelated distractors by a clear
        # margin. Only the gold-vs-noise *separation* is asserted, never the
        # internal order of the two near-duplicate relevant rows (child vs its
        # own source): that ordering is not portably stable across platforms or
        # model minor versions, but the semantic separation from the noise is.
        relevant_floor = min(scores[gold_child_id], scores["src-roadtrip"])
        distractor_ceiling = max(
            (scores[tid] for tid in ("noise-tax", "noise-gym") if tid in scores),
            default=0.0,
        )
        assert relevant_floor > distractor_ceiling + _GOLD_MARGIN, scores
        # The real vector arm actually participated in the fusion.
        assert "vector" in result.backends_used
        assert "fts5" in result.backends_used
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# 2. Reopen durability: no-op ensure_schema + identical ranked list
# ---------------------------------------------------------------------------

#: A small deterministic corpus; ``d-car`` is the gold answer for the query.
_DURABILITY_CORPUS: tuple[tuple[str, str, str], ...] = (
    (
        "d-car",
        "Vehicle issue",
        "The mechanic said the alternator is failing and the timing belt is due soon.",
    ),
    (
        "d-paris",
        "Travel plan",
        "We are flying to Paris in October and staying near the Montmartre district.",
    ),
    (
        "d-guitar",
        "Hobby update",
        "I finally started weekly guitar lessons and I am learning fingerpicking now.",
    ),
    (
        "d-budget",
        "Finance task",
        "I updated the quarterly budget spreadsheet and the travel line is over by 1200 dollars.",
    ),
    (
        "d-lasagna",
        "Cooking note",
        "Grandma's lasagna recipe uses three cheeses and a slow simmered tomato ragu.",
    ),
    (
        "d-camera",
        "Photography",
        "I rented a wide angle lens for the canyon shoot and the dynamic range was stunning.",
    ),
)
_DURABILITY_QUERY = "what did the mechanic say about the alternator"
_DURABILITY_IDS = [row[0] for row in _DURABILITY_CORPUS]


@requires_local_embeddings
@pytest.mark.parametrize(
    "backend",
    ["numpy", pytest.param("sqlite-vec", marks=requires_sqlite_vec)],
)
async def test_reopen_is_noop_and_query_is_identical(
    tmp_path: Path,
    real_minilm_provider: EmbeddingProviderProtocol,
    backend: str,
) -> None:
    """A closed-and-reopened file DB re-migrates nothing and re-ranks identically.

    Proves FTS5 + embedding (and, for ``sqlite-vec``, the ``vec0`` index)
    durability: after a full close and a fresh ``SqliteEngravaCore`` on the same
    path, ``ensure_schema`` leaves ``user_version`` untouched, no row is
    re-embedded, and the identical hybrid query returns the identical ranked list
    (exact thought-id order; scores within a tight ``atol``).
    """
    db_path = tmp_path / f"durable-{backend}.db"

    # --- Session 1: bootstrap, ingest, embed, capture the ranked list. --------
    conn1 = await aiosqlite.connect(str(db_path))
    conn1.row_factory = aiosqlite.Row
    await conn1.execute("PRAGMA foreign_keys = ON")
    store1 = SqliteEngravaCore(
        conn1,
        embedding_provider=real_minilm_provider,
        auto_embed=True,
    )
    await store1.ensure_schema()
    for thought_id, essence, content in _DURABILITY_CORPUS:
        await store1.create_thought(_thought(thought_id, essence=essence, content=content))

    sample = await store1.get_embedding("d-car")
    assert sample is not None
    dimension = sample.dimension
    # ``_configure_vector_backend`` is the only in-test seam that swaps the real
    # vec0 backend onto a manually-constructed store: the public path
    # (``from_config``) is YAML-only and would both re-own the connection and
    # subsume the discrete ``ensure_schema`` no-op this reopen test observes.
    # This mirrors the established real-vec0 pattern in ``test_sqlite_vec.py``.
    await store1._configure_vector_backend(backend_name=backend, embedding_dimension=dimension)

    corpus_size = len(_DURABILITY_CORPUS)
    version_session1 = await _user_version(conn1)
    embedding_count_session1 = await _scalar_count(conn1, "SELECT COUNT(*) FROM embedding")
    created_at_session1 = await _embedding_created_at(store1, _DURABILITY_IDS)
    if backend == "sqlite-vec":
        vec_count_session1 = await _scalar_count(conn1, "SELECT COUNT(*) FROM embedding_vec")

    ranked_session1 = (await store1.recall(_DURABILITY_QUERY, top_k=corpus_size)).results
    assert "d-car" in {thought_id for thought_id, _ in ranked_session1}
    await conn1.close()

    # --- Session 2: reopen a fresh store on the same file. --------------------
    conn2 = await aiosqlite.connect(str(db_path))
    conn2.row_factory = aiosqlite.Row
    await conn2.execute("PRAGMA foreign_keys = ON")
    store2 = SqliteEngravaCore(
        conn2,
        embedding_provider=real_minilm_provider,
        auto_embed=True,
    )

    # ensure_schema is a genuine no-op on an already-head database.
    version_before_ensure = await _user_version(conn2)
    await store2.ensure_schema()
    version_after_ensure = await _user_version(conn2)
    assert version_before_ensure == version_after_ensure == version_session1
    # No extension migration re-ran (there are none registered).
    assert await _scalar_count(conn2, "SELECT COUNT(*) FROM extension_schema_versions") == 0

    # Reconfigure the same backend on the reopened store (see the session-1 note
    # on why the private seam is the only in-test path to the real vec0 backend).
    await store2._configure_vector_backend(backend_name=backend, embedding_dimension=dimension)

    # Nothing was re-embedded: identical row count and identical per-row
    # created_at fingerprints survive the reopen.
    assert await _scalar_count(conn2, "SELECT COUNT(*) FROM embedding") == embedding_count_session1
    assert await _embedding_created_at(store2, _DURABILITY_IDS) == created_at_session1
    if backend == "sqlite-vec":
        # The vec0 index persisted on disk and was reused, not rebuilt/duplicated.
        vec_count_session2 = await _scalar_count(conn2, "SELECT COUNT(*) FROM embedding_vec")
        assert vec_count_session2 == vec_count_session1

    ranked_session2 = (await store2.recall(_DURABILITY_QUERY, top_k=corpus_size)).results
    try:
        # The identical ranked list: exact thought-id order, scores within a
        # tight absolute tolerance.
        assert [tid for tid, _ in ranked_session2] == [tid for tid, _ in ranked_session1]
        for (id_1, score_1), (id_2, score_2) in zip(
            ranked_session1,
            ranked_session2,
            strict=True,
        ):
            assert id_2 == id_1
            assert score_2 == pytest.approx(score_1, abs=1e-9)
    finally:
        await conn2.close()


# ---------------------------------------------------------------------------
# 3. Multi-extension load order: vec0 + structural-split + dreaming
# ---------------------------------------------------------------------------


def _topic_embed(text: str) -> list[float]:
    """Deterministic content-keyed embedder: co-topic texts share a unit vector.

    Every text carrying the ``alpha`` marker maps to the same axis, so the
    ``alpha`` thoughts (and their derived children) are cosine-identical and
    cluster deterministically, while a ``beta`` text stays orthogonal. This keeps
    the dreaming cycle convergent without any timing/seed sensitivity.
    """
    lowered = text.lower()
    if "alpha" in lowered:
        return [1.0, 0.0, 0.0]
    if "beta" in lowered:
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]


def _converging_dreaming_config() -> DreamingConfig:
    """A dreaming config tuned to promote + link + reflect deterministically."""
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
            cluster_algorithm="lpa",
            min_cluster_size=2,
            # These synthetic fixtures pre-date the content-quality gates and use
            # contrived member content; exercising the gates here would only add
            # flakiness (they have dedicated coverage elsewhere).
            cluster_quality_gating_enabled=False,
        ),
        edges=EdgeCreationConfig(
            enabled=True,
            top_k=3,
            min_similarity=0.5,
            edge_weight_factor=0.5,
        ),
    )


@requires_sqlite_vec
async def test_multi_extension_load_order_converges(tmp_path: Path) -> None:
    """vec0 + structural-split + dreaming, composed on one file store, converge.

    Loads the ``vec0`` backend, the structural-split derived-records seam and the
    dreaming extension on a single on-disk store, then runs one consolidation
    cycle. Asserts the cycle converges — no crash, the expected derived children,
    ASSOCIATED edges and REFLECTION are present, the vec0 index stays in sync,
    and the schema does not drift — catching extension-migration interactions the
    isolated runners miss.
    """
    db_path = tmp_path / "multi_extension.db"
    conn = await aiosqlite.connect(str(db_path))
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA foreign_keys = ON")

    provider = CallbackProvider(_topic_embed, dimension=3, model_name="topic-3")
    store = SqliteEngravaCore(
        conn,
        hooks=StructuralSplitProducer(),
        embedding_provider=provider,
        auto_embed=True,
        derive_gates=DeriveGates(enabled=True),
    )
    await store.ensure_schema()
    head_version = await _user_version(conn)
    # ``_configure_vector_backend`` is the only in-test seam to the real vec0
    # backend here: the public ``from_config`` path is YAML-only and cannot wire
    # the runtime ``CallbackProvider`` this deterministic cluster relies on. Same
    # established real-vec0 pattern as ``test_sqlite_vec.py``.
    await store._configure_vector_backend(backend_name="sqlite-vec", embedding_dimension=3)
    try:
        # Two co-topic alpha sources cluster; a beta source stays isolated. One
        # alpha source carries two paragraphs so the split derives two children.
        await store.create_thought(
            _thought(
                "alpha-1",
                essence="alpha topic",
                content="alpha subject one, notes about alpha matters",
            ),
        )
        await store.create_thought(
            _thought(
                "alpha-2",
                essence="alpha topic",
                content=(
                    "alpha subject two, notes about alpha matters"
                    "\n\n"
                    "alpha subject two, further alpha detail recorded"
                ),
            ),
        )
        await store.create_thought(
            _thought(
                "beta-1",
                essence="beta topic",
                content="beta unrelated subject, notes about beta matters",
            ),
        )

        # Structural-split composed with vec0: alpha-2 derived two linked,
        # embedded children.
        alpha2_children = await store.get_edges("alpha-2", direction="IN")
        assert len(alpha2_children) == 2
        assert all(edge.edge_type == EdgeType.DERIVED_FROM for edge in alpha2_children)
        for edge in alpha2_children:
            assert await store.get_embedding(edge.from_thought_id) is not None

        # Every embedded row is mirrored into the live vec0 index.
        embedding_count_pre = await _scalar_count(conn, "SELECT COUNT(*) FROM embedding")
        vec_count_pre = await _scalar_count(conn, "SELECT COUNT(*) FROM embedding_vec")
        assert vec_count_pre == embedding_count_pre

        # Run one deterministic consolidation cycle at a fixed cycle number.
        extension = DreamingExtension(config=_converging_dreaming_config())
        result = await extension.run_consolidation(store, current_cycle=5)

        # Convergence: the alpha cluster produced at least one ASSOCIATED edge and
        # one REFLECTION, and the reflection is materialised in the store.
        assert result.edges_created >= 1
        assert result.reflections_created >= 1
        reflections = await store.list_thoughts(thought_type=ThoughtType.REFLECTION)
        assert len(reflections) >= 1

        # The vec0 index stayed consistent with the embedding table across the
        # whole cycle (including the reflection-centroid write).
        embedding_count_post = await _scalar_count(conn, "SELECT COUNT(*) FROM embedding")
        assert (
            await _scalar_count(conn, "SELECT COUNT(*) FROM embedding_vec") == embedding_count_post
        )

        # The schema converged: no migration drift from the composed extensions.
        assert await _user_version(conn) == head_version
        assert await _scalar_count(conn, "SELECT COUNT(*) FROM extension_schema_versions") == 0

        # The composed store still answers a hybrid query through vec0.
        post_cycle = await store.search_hybrid("alpha", [1.0, 0.0, 0.0], top_k=10)
        assert post_cycle.results
        assert "vector" in post_cycle.backends_used
    finally:
        await conn.close()
