"""Tests for LongMemEval evaluation modes (substring + cosine).

The LLM mode lives in :func:`engrava.benchmarks.longmemeval.evaluate.evaluate_llm`
and is exercised in a separate test module.
"""

from __future__ import annotations

import hashlib

import pytest

from engrava.benchmarks.longmemeval.evaluate import (
    DEFAULT_COSINE_THRESHOLD,
    EvaluationOutcome,
    evaluate_cosine,
    evaluate_substring,
)


class _DeterministicEmbeddingProvider:
    """Hash-based deterministic embedding provider for tests.

    Embeds each input to a unit vector derived from its SHA-256 hash.
    Identical inputs return identical vectors → cosine similarity 1.0;
    different inputs return uncorrelated vectors.
    """

    def __init__(self, dimensions: int = 16) -> None:
        self._dim = dimensions

    async def embed(self, text: str) -> list[float]:
        return self._hash_to_vector(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_to_vector(t) for t in texts]

    async def verify_embedding_model(self) -> None:
        return None

    @property
    def model_name(self) -> str:
        return "test-deterministic"

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def _hash_to_vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # Map each pair of bytes to a signed value in [-1, 1] so unrelated
        # inputs produce vectors close to orthogonal rather than all-positive
        # (which would float every pairwise cosine into the high-similarity
        # band and defeat the threshold semantics tests rely on).
        raw = [
            (int.from_bytes(digest[i : i + 2], "big") / 32767.5) - 1.0
            for i in range(0, self._dim * 2, 2)
        ]
        return raw[: self._dim]


class TestEvaluateSubstring:
    """Substring containment checks."""

    def test_hit_when_chunk_contains_answer(self) -> None:
        """Default mode is case-insensitive containment."""
        outcome = evaluate_substring(
            "Pepper",
            ["Yesterday I adopted a tabby cat named pepper from the shelter."],
        )
        assert outcome == EvaluationOutcome(score=1.0, raw_signal=1.0)

    def test_miss_when_no_chunk_contains_answer(self) -> None:
        outcome = evaluate_substring(
            "Lisbon",
            ["I adopted a tabby cat.", "My passport just arrived."],
        )
        assert outcome.score == 0.0
        assert outcome.raw_signal == 0.0

    def test_empty_answer_misses(self) -> None:
        outcome = evaluate_substring("   ", ["anything"])
        assert outcome.score == 0.0

    def test_empty_retrieved_misses(self) -> None:
        outcome = evaluate_substring("Pepper", [])
        assert outcome.score == 0.0

    def test_case_sensitive_flag_blocks_lower_case_match(self) -> None:
        """``case_sensitive=True`` only matches an exact-case substring."""
        outcome = evaluate_substring(
            "Pepper",
            ["named pepper"],
            case_sensitive=True,
        )
        assert outcome.score == 0.0

    def test_match_short_circuits_after_first_hit(self) -> None:
        """The first matching chunk wins; later chunks need not be inspected."""
        chunks = ["irrelevant", "contains Pepper here", "irrelevant"]
        outcome = evaluate_substring("Pepper", chunks)
        assert outcome.score == 1.0


class TestEvaluateCosine:
    """Cosine-similarity scoring against an embedding provider."""

    @pytest.fixture
    def provider(self) -> _DeterministicEmbeddingProvider:
        return _DeterministicEmbeddingProvider()

    async def test_identical_chunk_hits(self, provider: _DeterministicEmbeddingProvider) -> None:
        """An exact-text chunk yields cosine 1.0 → score 1.0."""
        outcome = await evaluate_cosine(
            "tabby cat named Pepper",
            ["tabby cat named Pepper"],
            embedding_provider=provider,
        )
        assert outcome.score == 1.0
        assert outcome.raw_signal == pytest.approx(1.0)

    async def test_unrelated_chunks_miss(self, provider: _DeterministicEmbeddingProvider) -> None:
        """Hash-derived embeddings of unrelated text fall below threshold."""
        outcome = await evaluate_cosine(
            "tabby cat named Pepper",
            [
                "Mathematical analysis of complex variables.",
                "Yesterday I rode the train to Berlin.",
            ],
            embedding_provider=provider,
        )
        assert outcome.score == 0.0

    async def test_threshold_is_configurable(
        self,
        provider: _DeterministicEmbeddingProvider,
    ) -> None:
        """A threshold at the observed similarity floor flips a miss into a hit."""
        text = "tabby cat named Pepper"
        chunks = ["totally different sentence"]
        baseline = await evaluate_cosine(
            text,
            chunks,
            embedding_provider=provider,
        )
        assert baseline.score == 0.0
        # Re-running with a threshold below the observed raw signal must hit.
        relaxed = await evaluate_cosine(
            text,
            chunks,
            embedding_provider=provider,
            threshold=baseline.raw_signal - 1e-9,
        )
        assert relaxed.score == 1.0

    async def test_empty_inputs_miss(
        self,
        provider: _DeterministicEmbeddingProvider,
    ) -> None:
        outcome_empty_answer = await evaluate_cosine(
            "",
            ["anything"],
            embedding_provider=provider,
        )
        outcome_empty_chunks = await evaluate_cosine(
            "Pepper",
            [],
            embedding_provider=provider,
        )
        assert outcome_empty_answer.score == 0.0
        assert outcome_empty_chunks.score == 0.0

    async def test_default_threshold_constant(
        self,
        provider: _DeterministicEmbeddingProvider,
    ) -> None:
        """Default-threshold call matches a literal call with the constant."""
        outcome_default = await evaluate_cosine(
            "Pepper",
            ["Pepper"],
            embedding_provider=provider,
        )
        outcome_literal = await evaluate_cosine(
            "Pepper",
            ["Pepper"],
            embedding_provider=provider,
            threshold=DEFAULT_COSINE_THRESHOLD,
        )
        assert outcome_default == outcome_literal
