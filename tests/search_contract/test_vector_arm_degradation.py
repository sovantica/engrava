"""No-silent-degradation invariant for the vector (cosine) retrieval arm.

This suite is the vector-arm sibling of ``test_fts_expert_syntax_fallback.py``.
Where that suite pins the FTS arm's ``fts_match_failure_count`` health counter
and its never-silently-degrade contract, this one pins the same shape for the
vector arm:

* **Typed health counter** — :attr:`SqliteEngravaCore.vector_arm_degradation_count`
  (read-only) increments once whenever :meth:`search_similar` degrades to an
  empty result because the query vector has no usable cosine direction (empty,
  all-zero, or non-finite). It is the vector-arm mirror of
  ``fts_match_failure_count``.
* **Typed rejection for a structural error** — a query vector whose *length*
  differs from the store's embedding dimension is a caller-contract violation,
  not a benign miss, so it raises
  :class:`~engrava.domain.exceptions.VectorDimensionMismatchError` instead of
  silently returning ``[]``. (Before this contract, a wrong-length *non-zero*
  vector raised an opaque numpy ``ValueError`` and a wrong-length *all-zero*
  vector silently returned ``[]`` — the latent silent-empty this suite closes.)

The standing safety invariant (:class:`TestVectorArmSafetyInvariant`) drives a
deterministic, enumerated adversarial query-vector corpus through a real store
and asserts, per case: the call never raises (except the dimension error, where
raising *is* the contract); the degradation counter moves by the predicted
delta; and a shared-vocabulary vector still returns real neighbours (proving the
arm executed rather than swallowing the input). Its discriminating power is
verified by :class:`TestVectorArmDiscriminatingPower`: reverting the degeneracy
guard stops the counter from moving, and reverting the dimension guard turns a
wrong-length vector back into a silent degradation.
"""

from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING, NamedTuple

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
from engrava.domain.exceptions import VectorDimensionMismatchError
from engrava.domain.models.thought import ThoughtRecord

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# ---------------------------------------------------------------------------
# Deterministic, network-free embedder (a self-contained bag-of-words hashing
# provider, dimension _EMBED_DIM). Cosine similarity between two such vectors
# grows with the fraction of shared vocabulary, so a query text sharing a stored
# thought's words yields a genuine, predictable neighbour.
# ---------------------------------------------------------------------------

_EMBED_DIM = 32


def _embed(text: str) -> list[float]:
    """Embed text as an L2-normalized bag-of-words hashing vector.

    Args:
        text: Input text to embed.

    Returns:
        An ``_EMBED_DIM``-length unit vector (all-zero only for empty / token-
        less text).
    """
    vector = [0.0] * _EMBED_DIM
    for token in text.lower().split():
        digest = hashlib.sha1(token.encode("utf-8")).digest()  # noqa: S324
        vector[int.from_bytes(digest[:4], "big") % _EMBED_DIM] += 1.0
    norm = sum(value * value for value in vector) ** 0.5
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


# Content whose distinctive tokens a shared-vocabulary query vector will match.
_MATCH_ID = "t-match"
_MATCH_CONTENT = "alpha beta gamma distinctive vocabulary"
_SHARED_QUERY_TEXT = "alpha beta gamma"


def _thought(thought_id: str, content: str) -> ThoughtRecord:
    """Build a stored thought for the vector-arm suite.

    Args:
        thought_id: Stable identifier used to assert retrieval.
        content: Full text, embedded by the deterministic provider.

    Returns:
        A fully populated :class:`ThoughtRecord` ready for ``create_thought``.
    """
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.OBSERVATION,
        essence=content,
        content=content,
        priority=Priority.P2,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=0,
        updated_cycle=0,
        source="test",
        confidence=0.8,
        source_type=KnowledgeSource.EXPERIENCE,
        visibility=ThoughtVisibility.SELECTIVE,
    )


@pytest.fixture
async def vector_store() -> AsyncIterator[SqliteEngravaCore]:
    """Return a store with a deterministic vector arm and a small corpus.

    Yields:
        A :class:`SqliteEngravaCore` (``auto_embed`` on, dimension
        ``_EMBED_DIM``) holding a thought whose vocabulary a shared-vocabulary
        query vector matches, plus two decoys.
    """
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode = WAL")
    await conn.execute("PRAGMA foreign_keys = ON")
    provider = CallbackProvider(
        callback=_embed,
        dimension=_EMBED_DIM,
        model_name="bag-of-words-vector-arm",
    )
    store = SqliteEngravaCore(conn, embedding_provider=provider, auto_embed=True)
    await store.ensure_schema()
    await store.create_thought(_thought(_MATCH_ID, _MATCH_CONTENT))
    await store.create_thought(_thought("t-decoy-1", "delta epsilon zeta unrelated topic"))
    await store.create_thought(_thought("t-decoy-2", "eta theta iota another cluster"))
    yield store
    await conn.close()


# ---------------------------------------------------------------------------
# The enumerated adversarial query-vector corpus
# ---------------------------------------------------------------------------


class VectorFuzzCase(NamedTuple):
    """One adversarial query vector and its expected vector-arm behaviour.

    Args:
        label: Stable, unique name of the adversarial dimension this case
            witnesses (asserted present by the corpus meta-test).
        vector: The raw query embedding driven through ``search_similar``.
        expect_error: The typed error the case must raise, or ``None`` when the
            case must not raise.
        expect_delta: The expected increment of ``vector_arm_degradation_count``
            for a non-raising case (``1`` for a degenerate vector, ``0`` for a
            healthy one). Ignored for raising cases (a typed rejection never
            touches the degradation counter).
        expect_hits: For a non-raising case, whether ``search_similar`` must
            return at least one neighbour. ``True`` only for the shared-
            vocabulary anchors — the proof the arm executed rather than
            swallowing the query.
    """

    label: str
    vector: list[float]
    expect_error: type[Exception] | None
    expect_delta: int
    expect_hits: bool


_SHARED_UNIT = _embed(_SHARED_QUERY_TEXT)
_EMPTY_TEXT_EMBEDDING = _embed("")  # token-less text embeds to the all-zero vector


def _build_vector_fuzz_corpus() -> tuple[VectorFuzzCase, ...]:
    """Build the deterministic adversarial query-vector corpus.

    Every adversarial dimension the vector-arm contract must cover is
    enumerated exactly once: a shared-vocabulary unit vector, its non-unit
    rescaling (cosine is scale-invariant, so it is *not* a degradation), an
    explicit all-zero vector, the all-zero vector produced by embedding empty
    text, a wrong-length (``D-1`` and ``D+1``) vector, and vectors carrying
    ``NaN`` / ``+inf`` / ``-inf`` components.

    Returns:
        The immutable tuple of fuzz cases.
    """
    non_unit = [value * 3.0 for value in _SHARED_UNIT]
    nan_vec = [math.nan, *([0.1] * (_EMBED_DIM - 1))]
    pos_inf_vec = [math.inf, *([0.1] * (_EMBED_DIM - 1))]
    neg_inf_vec = [-math.inf, *([0.1] * (_EMBED_DIM - 1))]
    return (
        # Healthy vectors: the arm executes and returns real neighbours.
        VectorFuzzCase("shared_vocab_unit", _SHARED_UNIT, None, 0, expect_hits=True),
        VectorFuzzCase("shared_vocab_non_unit", non_unit, None, 0, expect_hits=True),
        # Degenerate vectors: no cosine direction -> counted degradation to [].
        VectorFuzzCase("zero_vector", [0.0] * _EMBED_DIM, None, 1, expect_hits=False),
        VectorFuzzCase("empty_text_embedding", _EMPTY_TEXT_EMBEDDING, None, 1, expect_hits=False),
        VectorFuzzCase("nan_component", nan_vec, None, 1, expect_hits=False),
        VectorFuzzCase("pos_inf_component", pos_inf_vec, None, 1, expect_hits=False),
        VectorFuzzCase("neg_inf_component", neg_inf_vec, None, 1, expect_hits=False),
        # Wrong-dimension vectors: a structural contract violation -> typed raise.
        VectorFuzzCase(
            "dimension_minus_one",
            [0.2] * (_EMBED_DIM - 1),
            VectorDimensionMismatchError,
            0,
            expect_hits=False,
        ),
        VectorFuzzCase(
            "dimension_plus_one",
            [0.2] * (_EMBED_DIM + 1),
            VectorDimensionMismatchError,
            0,
            expect_hits=False,
        ),
    )


_VECTOR_FUZZ_CORPUS: tuple[VectorFuzzCase, ...] = _build_vector_fuzz_corpus()


# ---------------------------------------------------------------------------
# The read-only health counter
# ---------------------------------------------------------------------------


class TestDegradationCounter:
    """``vector_arm_degradation_count`` is a typed, read-only health counter."""

    async def test_counter_starts_at_zero(self, vector_store: SqliteEngravaCore) -> None:
        """A fresh store has never degraded."""
        assert vector_store.vector_arm_degradation_count == 0

    async def test_counter_increments_on_degenerate_vector(
        self,
        vector_store: SqliteEngravaCore,
    ) -> None:
        """A single all-zero query increments the counter by exactly one."""
        assert await vector_store.search_similar([0.0] * _EMBED_DIM) == []
        assert vector_store.vector_arm_degradation_count == 1

    async def test_counter_stays_zero_for_healthy_vectors(
        self,
        vector_store: SqliteEngravaCore,
    ) -> None:
        """A healthy shared-vocabulary query never touches the counter."""
        await vector_store.search_similar(_SHARED_UNIT, top_k=10)
        await vector_store.search_similar([value * 3.0 for value in _SHARED_UNIT], top_k=10)
        assert vector_store.vector_arm_degradation_count == 0

    async def test_counter_property_is_read_only(
        self,
        vector_store: SqliteEngravaCore,
    ) -> None:
        """``vector_arm_degradation_count`` is a read-only property (no setter)."""
        prop = type(vector_store).vector_arm_degradation_count
        assert isinstance(prop, property)
        assert prop.fset is None
        with pytest.raises(AttributeError):
            vector_store.vector_arm_degradation_count = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# The shared-vocabulary anchor — proof the arm genuinely ranks
# ---------------------------------------------------------------------------


class TestSharedVocabularyAnchor:
    """A shared-vocabulary vector ranks the matching thought with a real score.

    Non-empty alone is a weak proof (a zero-threshold search returns every row
    at score 0.0). Asserting the matching thought is present *with a positive
    score* proves the vector arm executed a genuine cosine ranking.
    """

    async def test_shared_vocab_ranks_the_match(
        self,
        vector_store: SqliteEngravaCore,
    ) -> None:
        """The shared-vocabulary vector returns the match at a positive score."""
        results = await vector_store.search_similar(_SHARED_UNIT, top_k=10)
        scored = dict(results)
        assert _MATCH_ID in scored
        assert scored[_MATCH_ID] > 0.0

    async def test_non_unit_rescaling_is_not_a_degradation(
        self,
        vector_store: SqliteEngravaCore,
    ) -> None:
        """A non-unit rescaling ranks identically to the unit vector (cosine scale-free)."""
        unit = dict(await vector_store.search_similar(_SHARED_UNIT, top_k=10))
        rescaled = dict(
            await vector_store.search_similar([v * 3.0 for v in _SHARED_UNIT], top_k=10)
        )
        assert unit.keys() == rescaled.keys()
        assert unit[_MATCH_ID] == pytest.approx(rescaled[_MATCH_ID], abs=1e-9)
        assert vector_store.vector_arm_degradation_count == 0


# ---------------------------------------------------------------------------
# Standing safety invariant over the adversarial corpus
# ---------------------------------------------------------------------------


class TestVectorArmSafetyInvariant:
    """No adversarial query vector silently degrades: it degrades *observably*.

    For every case: a healthy or degenerate vector returns a list (never
    raising), and the degradation counter moves exactly as the case predicts (0
    for healthy, 1 for degenerate); a wrong-dimension vector raises the typed
    error and leaves the counter untouched. A shared-vocabulary case must return
    real hits — proof the arm executed rather than swallowing the input.
    """

    async def test_invariant_over_corpus(self, vector_store: SqliteEngravaCore) -> None:
        """Drive the whole adversarial query-vector corpus through the store."""
        for case in _VECTOR_FUZZ_CORPUS:
            before = vector_store.vector_arm_degradation_count
            if case.expect_error is not None:
                with pytest.raises(case.expect_error):
                    await vector_store.search_similar(case.vector, top_k=10)
                assert vector_store.vector_arm_degradation_count == before, (
                    f"a typed rejection must not touch the degradation counter for {case.label!r}"
                )
                continue
            results = await vector_store.search_similar(case.vector, top_k=10)
            assert isinstance(results, list), f"non-list result for {case.label!r}"
            delta = vector_store.vector_arm_degradation_count - before
            assert delta == case.expect_delta, (
                f"degradation delta {delta} != expected {case.expect_delta} for {case.label!r}"
            )
            if case.expect_hits:
                assert results, f"expected neighbours (arm did not execute) for {case.label!r}"
            else:
                assert results == [], f"degenerate vector must degrade to [] for {case.label!r}"

    def test_corpus_crosses_the_adversarial_space(self) -> None:
        """Meta-test: the corpus witnesses every adversarial dimension.

        Guards against a corpus that silently shrinks to a benign subset (e.g.
        only healthy vectors, so the counter/typed-error assertions never fire).
        Each documented adversarial dimension must be present by label, and both
        partitions must be non-trivially populated.
        """
        labels = {case.label for case in _VECTOR_FUZZ_CORPUS}
        required = {
            "shared_vocab_unit",
            "shared_vocab_non_unit",
            "zero_vector",
            "empty_text_embedding",
            "nan_component",
            "pos_inf_component",
            "neg_inf_component",
            "dimension_minus_one",
            "dimension_plus_one",
        }
        assert required <= labels, f"missing adversarial dimensions: {required - labels}"

        degenerate = [
            c for c in _VECTOR_FUZZ_CORPUS if c.expect_error is None and c.expect_delta == 1
        ]
        healthy = [c for c in _VECTOR_FUZZ_CORPUS if c.expect_error is None and c.expect_hits]
        raising = [c for c in _VECTOR_FUZZ_CORPUS if c.expect_error is VectorDimensionMismatchError]
        assert len(degenerate) >= 4, "corpus must exercise several degenerate shapes"
        assert len(healthy) >= 2, "corpus must exercise several healthy shapes"
        assert len(raising) >= 2, "corpus must exercise both wrong-dimension shapes"

        # The witnessed shapes are genuinely what they claim, so the invariant
        # cannot be satisfied by benign vectors wearing adversarial labels: the
        # empty-text embedding really is the all-zero vector (the canonical auto-
        # embed-of-empty-text origin), and every non-finite case really carries a
        # non-finite component.
        by_label = {case.label: case.vector for case in _VECTOR_FUZZ_CORPUS}
        assert _EMPTY_TEXT_EMBEDDING == [0.0] * _EMBED_DIM
        assert by_label["empty_text_embedding"] == [0.0] * _EMBED_DIM
        for label in ("nan_component", "pos_inf_component", "neg_inf_component"):
            assert any(not math.isfinite(v) for v in by_label[label]), (
                f"{label!r} must carry a non-finite component"
            )
        # The wrong-dimension cases are genuinely off by exactly one on each side.
        assert len(by_label["dimension_minus_one"]) == _EMBED_DIM - 1
        assert len(by_label["dimension_plus_one"]) == _EMBED_DIM + 1


# ---------------------------------------------------------------------------
# Discriminating power — reverting a guard breaks the invariant
# ---------------------------------------------------------------------------


class TestVectorArmDiscriminatingPower:
    """The invariant is only meaningful if reverting a guard breaks it.

    Each test reverts one guard in-process to its pre-fix (buggy) form and
    asserts the observable the invariant relies on changes — proving the
    corpus/counter carry real discriminating power rather than passing vacuously.
    """

    async def test_reverting_degeneracy_guard_stops_the_counter(
        self,
        vector_store: SqliteEngravaCore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With degeneracy detection reverted, an all-zero query stops counting.

        Baseline: an all-zero vector degrades observably (counter +1). Revert
        the boundary degeneracy detector to the pre-fix ``always False`` and the
        same vector slips past the counter, degrading to ``[]`` *silently* (delta
        0) via the numpy arm's zero-norm guard — exactly the silent degradation
        the counter exists to surface.
        """
        import engrava.infrastructure.sqlite.engrava_core as core_mod

        zero = [0.0] * _EMBED_DIM

        before = vector_store.vector_arm_degradation_count
        assert await vector_store.search_similar(zero) == []
        assert vector_store.vector_arm_degradation_count == before + 1

        monkeypatch.setattr(core_mod, "_query_vector_is_degenerate", lambda _vector: False)

        before = vector_store.vector_arm_degradation_count
        assert await vector_store.search_similar(zero) == []
        assert vector_store.vector_arm_degradation_count == before, (
            "reverting the degeneracy guard must make the all-zero query degrade "
            "silently (counter unchanged), breaking the invariant's delta prediction"
        )

    async def test_reverting_dimension_guard_hides_a_wrong_length_vector(
        self,
        vector_store: SqliteEngravaCore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With the declared dimension hidden, a wrong-length all-zero vector no longer raises.

        Baseline: a wrong-length all-zero vector raises the typed dimension error
        (dimension is checked *before* degeneracy, so magnitude is irrelevant).
        Revert the store's declared dimension to ``None`` — the pre-fix state
        where ``search_similar`` could not know the expected length up front — and
        the same vector is now swallowed as a degenerate degradation (counter +1,
        ``[]``) instead of a loud rejection: the latent silent-empty this contract
        closes.
        """
        wrong_length_zero = [0.0] * (_EMBED_DIM - 1)

        with pytest.raises(VectorDimensionMismatchError):
            await vector_store.search_similar(wrong_length_zero)

        monkeypatch.setattr(vector_store, "_declared_embedding_dimension", lambda: None)

        before = vector_store.vector_arm_degradation_count
        results = await vector_store.search_similar(wrong_length_zero)
        assert results == []
        assert vector_store.vector_arm_degradation_count == before + 1, (
            "reverting the declared dimension must turn the typed rejection into a "
            "silent degradation, proving the dimension guard is load-bearing"
        )


# ---------------------------------------------------------------------------
# Undeclared-dimension store — the numpy arm still rejects wrong lengths typed
# ---------------------------------------------------------------------------


class TestUndeclaredDimensionStore:
    """A store with no declared dimension still rejects a wrong-length vector typed.

    When neither a vector backend nor an embedding provider is configured (raw
    embeddings written straight through ``store_embedding``),
    ``search_similar`` cannot check the query length at its boundary. The numpy
    arm then validates against the *stored* embedding dimension and raises the
    same :class:`VectorDimensionMismatchError` — never an opaque numpy shape
    error — so no query path leaks an untyped crash.
    """

    async def _bare_store(self) -> tuple[SqliteEngravaCore, aiosqlite.Connection]:
        """Build a provider-less, backend-less store with one 4-D embedding.

        Returns:
            The store and its connection (closed by the caller).
        """
        conn = await aiosqlite.connect(":memory:")
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode = WAL")
        await conn.execute("PRAGMA foreign_keys = ON")
        store = SqliteEngravaCore(conn)
        await store.ensure_schema()
        await store.create_thought(_thought("t-raw", "raw embedding row"))
        await store.store_embedding(
            thought_id="t-raw", vector=[0.1, 0.2, 0.3, 0.4], model_name="m4"
        )
        return store, conn

    async def test_no_declared_dimension(self) -> None:
        """The store declares no dimension without a backend or provider."""
        store, conn = await self._bare_store()
        try:
            assert store._declared_embedding_dimension() is None
        finally:
            await conn.close()

    async def test_matching_length_vector_ranks_the_row(self) -> None:
        """A correctly-sized raw vector still retrieves the stored row."""
        store, conn = await self._bare_store()
        try:
            results = await store.search_similar([0.1, 0.2, 0.3, 0.4], top_k=5)
            assert "t-raw" in {tid for tid, _ in results}
        finally:
            await conn.close()

    async def test_wrong_length_vector_raises_typed_from_numpy_arm(self) -> None:
        """A wrong-length non-zero vector raises the typed error, counter untouched."""
        store, conn = await self._bare_store()
        try:
            before = store.vector_arm_degradation_count
            with pytest.raises(VectorDimensionMismatchError):
                await store.search_similar([0.1, 0.2, 0.3], top_k=5)
            assert store.vector_arm_degradation_count == before
        finally:
            await conn.close()

    async def test_empty_vector_is_a_counted_degradation(self) -> None:
        """An empty query vector (no declared dimension to reject it) degrades, counted.

        With no declared dimension the boundary cannot reject ``[]`` as a length
        mismatch, so it is classified as the degenerate empty vector it is:
        counted and returned as ``[]``, never raising.
        """
        store, conn = await self._bare_store()
        try:
            before = store.vector_arm_degradation_count
            assert await store.search_similar([], top_k=5) == []
            assert store.vector_arm_degradation_count == before + 1
        finally:
            await conn.close()
