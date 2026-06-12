"""Functional contract suite for user-level search behavior.

This suite asserts the *behavioral* search contract — what a user should be
able to find — against a realistic, hand-authored conversational corpus, rather
than asserting that particular lines of code execute. Line coverage alone once
missed a dead lexical arm, a guaranteed-miss sanitizer, and a query crash,
because coverage measures executed lines, not asserted behaviors, and the
inputs in narrower tests happened to encode the very assumption that was wrong.

Each test class below pins one observable property of search:

* :class:`TestFindabilityInvariant` — every stored turn is retrievable from a
  query built from its own distinctive terms plus arbitrary function words.
* :class:`TestNoCrashInvariant` — adversarial query strings never raise.
* :class:`TestSanitizerRoundTrip` — contractions and clitics survive
  normalization and still match.
* :class:`TestArmLiveness` — natural-language questions return non-empty
  full-text candidates (guards a silent single-arm degradation).
* :class:`TestFusionSanity` — hybrid fusion ranks a strongly-matched turn above
  a weakly-matched one.
* :class:`TestEndToEndLifecycle` — each gold-labelled question retrieves its
  gold answer in the hybrid top-k.

PROCESS RULE: every future change to a retrieval mechanism (the FTS query
normalizer, the vector arm, the fusion scorer, or any new search backend) MUST
extend this suite with its own contract test asserting the user-visible
behavior the change is meant to preserve or introduce. A change that passes the
existing tests but silently breaks search must be made to fail here in seconds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from engrava import SqliteEngravaCore
    from tests.search_contract.conftest import CorpusTurn, GoldQuestion


# Arbitrary function words appended to findability queries. None of these
# should ever block a match — they carry no distinctive information.
_FUNCTION_WORDS = ("what", "did", "my", "about", "the", "was")

# Adversarial query strings that must never raise. Mix of pasted URLs, colon
# tokens, apostrophes, quotes, parens, emoji, CJK, empty/whitespace, and a very
# long string.
_ADVERSARIAL_QUERIES = (
    "",
    "   ",
    "\t\n  \t",
    "http://example.com/path?q=1&x=2",
    "https://docs.example.com/onboarding#section",
    "see http://example.com docs",
    "essence:something content:else",
    "weird:colon other:token",
    "12:30",
    "meeting at 12:30 pm",
    "don't worry about it",
    "sister's dog's leash",
    'an "unterminated phrase',
    'a "balanced phrase" here',
    "group (of tokens) here",
    "mismatched ) paren (",
    "emoji query 😀🚀 search",
    "中文 查询 测试",
    "café déjà vu naïve",
    "AND OR NOT",
    "a AND b OR c",
    "*",
    "***",
    "-",
    "--flag",
    "$5 ##tag @handle",
    "word " * 1000,
    "x" * 5000,
)


def _ids(results: list[tuple[str, float]]) -> set[str]:
    """Collect the thought ids from a list of scored search results.

    Args:
        results: A list of ``(thought_id, score)`` tuples.

    Returns:
        The set of thought ids present in the results.
    """
    return {thought_id for thought_id, _ in results}


class TestFindabilityInvariant:
    """Every stored turn is findable from its own distinctive terms.

    This is the property the old implicit-AND normalizer broke: appending
    ordinary function words to a few distinctive content terms must not stop a
    turn from being returned.
    """

    async def test_every_turn_found_by_its_distinctive_terms(
        self,
        fts_store: SqliteEngravaCore,
        corpus: tuple[CorpusTurn, ...],
    ) -> None:
        """Each turn is returned for a query of its terms plus function words."""
        missing: list[str] = []
        for turn in corpus:
            if not turn.distinctive_terms:
                continue
            query = " ".join((*_FUNCTION_WORDS[:3], *turn.distinctive_terms))
            results = await fts_store.search_fts(query, top_k=50)
            if turn.thought_id not in _ids(results):
                missing.append(turn.thought_id)
        assert missing == [], f"distinctive-term query failed to find: {missing}"

    async def test_single_distinctive_term_plus_function_words(
        self,
        fts_store: SqliteEngravaCore,
        corpus: tuple[CorpusTurn, ...],
    ) -> None:
        """A single distinctive term plus function words still finds the turn."""
        missing: list[str] = []
        for turn in corpus:
            if not turn.distinctive_terms:
                continue
            query = f"what about my {turn.distinctive_terms[0]}"
            results = await fts_store.search_fts(query, top_k=50)
            if turn.thought_id not in _ids(results):
                missing.append(turn.thought_id)
        assert missing == [], f"single-term query failed to find: {missing}"


class TestNoCrashInvariant:
    """Adversarial query strings return a list, never raise."""

    @pytest.mark.parametrize("query", _ADVERSARIAL_QUERIES)
    async def test_search_fts_never_raises(
        self,
        fts_store: SqliteEngravaCore,
        query: str,
    ) -> None:
        """``search_fts`` returns a list for every adversarial input."""
        results = await fts_store.search_fts(query)
        assert isinstance(results, list)

    @pytest.mark.parametrize("query", _ADVERSARIAL_QUERIES)
    async def test_search_hybrid_never_raises(
        self,
        hybrid_store: SqliteEngravaCore,
        query: str,
    ) -> None:
        """``search_hybrid`` returns results for every adversarial input."""
        result = await hybrid_store.search_hybrid(query, top_k=10)
        assert isinstance(result.results, list)


class TestSanitizerRoundTrip:
    """Contractions and clitics survive normalization and still match."""

    async def test_english_possessive_matches(
        self,
        fts_store: SqliteEngravaCore,
    ) -> None:
        """A possessive query (``sister's``) finds the turn about a sister."""
        results = await fts_store.search_fts("sister's")
        assert "turn-sister-dog" in _ids(results)

    async def test_english_possessive_in_question_matches(
        self,
        fts_store: SqliteEngravaCore,
    ) -> None:
        """The possessive embedded in a full question still finds the turn."""
        results = await fts_store.search_fts("what about my sister's dog")
        assert "turn-sister-dog" in _ids(results)

    async def test_negation_contraction_does_not_block(
        self,
        fts_store: SqliteEngravaCore,
    ) -> None:
        """A ``don't`` contraction in the query does not block a content match."""
        results = await fts_store.search_fts("I don't recall the tenkeyless keyboard")
        assert "turn-keyboard-don't" in _ids(results)

    async def test_french_elision_matches(
        self,
        fts_store: SqliteEngravaCore,
    ) -> None:
        """A French elision query (``l'école``) splits on the clitic and matches."""
        results = await fts_store.search_fts("l'école")
        assert "turn-french-school" in _ids(results)


class TestArmLiveness:
    """Natural-language questions return non-empty full-text candidates.

    A silently degraded lexical arm would return nothing for these questions
    while still passing single-token tests; asserting non-empty hits for the
    whole gold-question set is the observable proxy that guards against it.
    """

    async def test_every_gold_question_returns_fts_candidates(
        self,
        fts_store: SqliteEngravaCore,
        gold_questions: tuple[GoldQuestion, ...],
    ) -> None:
        """Each gold question yields at least one full-text candidate."""
        empty: list[str] = []
        for question in gold_questions:
            results = await fts_store.search_fts(question.question, top_k=50)
            if not results:
                empty.append(question.question)
        assert empty == [], f"FTS arm returned no candidates for: {empty}"


class TestFusionSanity:
    """Hybrid fusion ranks a strongly-matched turn above a weakly-matched one.

    A turn that shares every distinctive query term (matched by both the
    lexical and the deterministic vector arm) must outrank a turn that shares
    only one low-information generic term. This is the weaker, deterministic
    analogue of the "both arms agree beats one weak arm" property.
    """

    async def test_all_terms_doc_outranks_single_term_doc(
        self,
        hybrid_store: SqliteEngravaCore,
    ) -> None:
        """A doc with all query terms outranks one sharing a single generic term."""
        # "turn-job-marketing" contains marketing, specialist, and startup.
        # "turn-guitar-lessons" shares only the generic word "lessons".
        result = await hybrid_store.search_hybrid(
            "marketing specialist startup lessons",
            top_k=10,
        )
        ranked_ids = [thought_id for thought_id, _ in result.results]
        assert "turn-job-marketing" in ranked_ids
        assert "turn-guitar-lessons" in ranked_ids
        assert ranked_ids.index("turn-job-marketing") < ranked_ids.index("turn-guitar-lessons")

    async def test_both_arms_fire_for_a_distinctive_query(
        self,
        hybrid_store: SqliteEngravaCore,
    ) -> None:
        """A distinctive query engages both the lexical and the vector arm."""
        result = await hybrid_store.search_hybrid(
            "the hazelnut coffee creamer coupon",
            top_k=10,
        )
        assert "fts5" in result.backends_used
        assert "vector" in result.backends_used
        assert "turn-coffee-creamer" in _ids(result.results)


class TestEndToEndLifecycle:
    """Each gold-labelled question retrieves its gold answer in hybrid top-k.

    The deterministic, network-free analogue of an answer-turn-in-context
    benchmark: store the corpus, ask each natural-language question, and require
    the labelled answer turn in the returned top-k.
    """

    async def test_gold_answer_in_hybrid_top_k(
        self,
        hybrid_store: SqliteEngravaCore,
        gold_questions: tuple[GoldQuestion, ...],
    ) -> None:
        """Every gold question's answer turn appears in the hybrid top-k."""
        top_k = 10
        misses: list[str] = []
        for question in gold_questions:
            result = await hybrid_store.search_hybrid(question.question, top_k=top_k)
            if question.gold_thought_id not in _ids(result.results):
                misses.append(f"{question.question!r} -> {question.gold_thought_id}")
        assert misses == [], f"gold answer missing from top-{top_k} for: {misses}"
