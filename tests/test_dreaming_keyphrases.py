"""Unit tests for ``engrava.extensions.dreaming_keyphrases``.

Two function surfaces are exercised:

* :func:`extract_simple_keywords` — frequency-ranked single tokens.
  These tests carry over the previous unit and cognitive-
  boundary tests for the now-deleted ``_extract_keywords`` helper in
  :mod:`engrava.extensions.dreaming` (the body migrated; the contract
  is preserved byte-for-byte).
* :func:`top_keyphrases_tfidf` — TF-IDF-scored 2-3 word n-grams over
  a cluster, with the corpus baseline supplied by the caller.

The migrated tests deliberately keep the same input fixtures as the
original location so behavioural parity is provable from the diff
alone.
"""

from __future__ import annotations

import inspect

from engrava.domain.enums import (
    KnowledgeSource,
    LifecycleStatus,
    Priority,
    ThoughtType,
    ThoughtVisibility,
)
from engrava.domain.models.thought import ThoughtRecord
from engrava.extensions.dreaming_keyphrases import (
    extract_simple_keywords,
    top_keyphrases_tfidf,
)


def _thought(content: str, thought_id: str = "t-x") -> ThoughtRecord:
    """Build a realistic ``ThoughtRecord`` for keyphrase tests."""
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.OBSERVATION,
        essence="essence",
        content=content,
        priority=Priority.P2,
        lifecycle_status=LifecycleStatus.CREATED,
        created_cycle=0,
        updated_cycle=0,
        source="test",
        confidence=0.9,
        source_type=KnowledgeSource.EXPERIENCE,
        visibility=ThoughtVisibility.SELECTIVE,
    )


# ---------------------------------------------------------------------------
# Migrated tests (4) from former tests/test_dreaming_clusters.py:155-185
# ---------------------------------------------------------------------------


class TestExtractSimpleKeywords:
    """Behavioural parity with the legacy ``_extract_keywords`` helper."""

    def test_returns_at_most_top_n(self) -> None:
        """Returns at most top_n keywords."""
        texts = ["alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"]
        kw = extract_simple_keywords(texts, top_n=5)
        assert len(kw) <= 5

    def test_filters_short_tokens(self) -> None:
        """Words shorter than 3 chars are filtered."""
        texts = ["to be or not to be that is the question"]
        kw = extract_simple_keywords(texts, top_n=10)
        assert all(len(w) >= 3 for w in kw)

    def test_orders_by_frequency_descending(self) -> None:
        """Most frequent keyword ranks first."""
        texts = ["python python python java java rust"]
        kw = extract_simple_keywords(texts, top_n=3)
        assert kw[0] == "python"

    def test_returns_list(self) -> None:
        """Result is a list of strings (no model objects)."""
        result = extract_simple_keywords(["machine learning deep learning"], top_n=5)
        assert isinstance(result, list)
        assert all(isinstance(w, str) for w in result)


# ---------------------------------------------------------------------------
# Migrated tests (3) from the former tier-boundary guard test.
# (cognitive-boundary properties of the legacy helper)
# ---------------------------------------------------------------------------


class TestExtractSimpleKeywordsCognitiveBoundary:
    """Cognitive-boundary properties: synchronous, deterministic, plain strings."""

    def test_is_synchronous(self) -> None:
        """``extract_simple_keywords`` is synchronous (no LLM round-trips)."""
        assert not inspect.iscoroutinefunction(extract_simple_keywords), (
            "extract_simple_keywords must be synchronous — LLM calls would be async"
        )

    def test_is_deterministic(self) -> None:
        """Repeated calls produce identical output (no LLM sampling)."""
        texts = ["neural networks deep learning machine learning"]
        r1 = extract_simple_keywords(texts, top_n=5)
        r2 = extract_simple_keywords(texts, top_n=5)
        assert r1 == r2, "extract_simple_keywords must be deterministic (no LLM sampling)"

    def test_returns_strings(self) -> None:
        """Returns a list of plain strings (no model objects)."""
        result = extract_simple_keywords(["alpha beta gamma delta"], top_n=4)
        assert isinstance(result, list)
        assert all(isinstance(w, str) for w in result)


# ---------------------------------------------------------------------------
# New tests for top_keyphrases_tfidf
# ---------------------------------------------------------------------------


class TestTopKeyphrasesTfidf:
    """Determinism, scoring shape, n-gram width, corpus-baseline behaviour."""

    def test_returns_two_or_three_word_phrases(self) -> None:
        cluster = [_thought("monday standup decisions and quarterly goals")]
        corpus = ["unrelated content"]
        result = top_keyphrases_tfidf(cluster, corpus=corpus, top_n=10)
        for entry in result:
            phrase = entry["phrase"]
            assert isinstance(phrase, str)
            word_count = len(phrase.split())
            assert word_count in {2, 3}, f"phrase {phrase!r} has {word_count} words"

    def test_deterministic_across_calls(self) -> None:
        cluster = [_thought("monday standup retrospective notes monday standup")]
        corpus = ["one document", "another document"]
        a = top_keyphrases_tfidf(cluster, corpus=corpus, top_n=3)
        b = top_keyphrases_tfidf(cluster, corpus=corpus, top_n=3)
        assert a == b

    def test_rare_phrase_outranks_common_phrase(self) -> None:
        """Phrases appearing in every corpus document score lower than rare phrases."""
        cluster = [_thought("alpha beta gamma delta epsilon zeta")]
        # "alpha beta" appears in every corpus document (high df → low idf);
        # "gamma delta" never appears outside the cluster (df=0 → high idf).
        corpus = ["alpha beta scenario one", "alpha beta scenario two"]
        result = top_keyphrases_tfidf(cluster, corpus=corpus, top_n=10)
        scores = {entry["phrase"]: entry["score"] for entry in result}
        assert scores["gamma delta"] > scores["alpha beta"]

    def test_empty_cluster_returns_empty(self) -> None:
        result = top_keyphrases_tfidf([], corpus=["doc"], top_n=3)
        assert result == []

    def test_empty_corpus_degenerates_gracefully(self) -> None:
        """Empty corpus is allowed; output is non-empty for non-trivial cluster."""
        cluster = [_thought("alpha beta gamma delta")]
        result = top_keyphrases_tfidf(cluster, corpus=[], top_n=5)
        assert len(result) > 0
        for entry in result:
            assert isinstance(entry["score"], float)

    def test_top_n_caps_result_length(self) -> None:
        cluster = [_thought("alpha beta gamma delta epsilon zeta eta theta")]
        result = top_keyphrases_tfidf(cluster, corpus=[], top_n=2)
        assert len(result) <= 2

    def test_score_is_float_and_phrase_is_string(self) -> None:
        cluster = [_thought("alpha beta gamma")]
        result = top_keyphrases_tfidf(cluster, corpus=[], top_n=1)
        assert len(result) == 1
        assert isinstance(result[0]["phrase"], str)
        assert isinstance(result[0]["score"], float)
