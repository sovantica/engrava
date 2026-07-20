"""Golden-parity contract for retrieval *semantics* (not merely liveness).

A retrieval rewrite can stay non-empty yet return the WRONG answer. The
motivating regression normalized ``essence:"a b"`` to an unscoped
``essence a b`` — still valid FTS5, still returning documents — so every
findability / never-raises / arm-liveness test stayed green while the answer was
semantically wrong. Only a byte-identical normalizer golden or a frozen
ranked-result golden tells "different answer" apart from "an answer". This module
pins both:

* :class:`TestExpertNormalizationGolden` — every genuine expert query (the full
  column-filter x phrase x boolean cross-product) normalizes byte-identically to
  a checked-in golden.
* :class:`TestHybridRankedGolden` — the hybrid search over the deterministic
  corpus produces a frozen ``query -> [thought_id, rounded_score]`` list.
* :class:`TestGoldenDiscriminatingPower` — reverting the column-filter drop
  in-process makes BOTH goldens fail, proving they discriminate a wrong answer
  from an answer rather than passing vacuously.

The goldens are checked-in fixtures under ``goldens/``. The tests only *read*
them; regeneration is an explicit, reviewed command
(``python scripts/regenerate_search_goldens.py``). A test that rewrote its own
golden on mismatch would be coverage-padding, not a check — so a genuine drift
surfaces here as a failing assertion, never as a silently-overwritten fixture.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

from engrava.infrastructure.sqlite import engrava_core
from engrava.infrastructure.sqlite.engrava_core import (
    _normalize_fts_query,
    _query_is_expert_syntax,
)
from tests.search_contract.golden_fixtures import (
    HYBRID_RANKED_GOLDEN_PATH,
    HYBRID_SCORE_NDIGITS,
    HYBRID_TOP_K,
    LEGACY_EXPERT_PARITY_QUERIES,
    compute_expert_normalizations,
    load_expert_normalization_cases,
    load_golden,
    load_hybrid_ranked_cases,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from engrava import SqliteEngravaCore

# Loaded at collection time so the byte-identity check can parametrize per case.
_EXPERT_CASES: dict[str, str] = load_expert_normalization_cases()
_HYBRID_CASES: dict[str, list[list[str | float]]] = load_hybrid_ranked_cases()

# A column-filter query whose scope drop the WS calls out and whose ranked list
# visibly reshuffles end-to-end — the discriminating hybrid case.
_HYBRID_DISCRIMINATOR_QUERY = 'content:"three cheeses"'

# ``essence:"office plant"`` etc.: a column filter directly wrapping a phrase —
# the exact shape whose scope the rejected rewrite dropped.
_COLUMN_FILTER_PHRASE_RE = re.compile(r'(?:essence|content):"', re.IGNORECASE)


def _make_column_filter_dropping_normalizer(
    original: Callable[[str], str],
) -> Callable[[str], str]:
    """Build a normalizer that reproduces the rejected column-filter drop.

    The regression normalized ``essence:"a b"`` to an unscoped ``essence a b`` —
    valid FTS5 that still returns documents, so it slipped past liveness tests.
    This reproduces the drop surgically: only a genuine column-filter *phrase*
    query loses its ``:`` scope and quotes (then re-normalizes as a bare query);
    every other query is delegated to the real normalizer unchanged.

    Args:
        original: The real ``_normalize_fts_query`` captured before patching.

    Returns:
        A drop-in normalizer that mis-scopes column-filter phrase queries.
    """

    def _reverted(query: str) -> str:
        if _COLUMN_FILTER_PHRASE_RE.search(query):
            descoped = query.replace(":", " ").replace('"', " ")
            return original(descoped)
        return original(query)

    return _reverted


class TestExpertNormalizationGolden:
    """Every genuine expert query normalizes byte-identically to the golden."""

    @pytest.mark.parametrize(("query", "expected"), sorted(_EXPERT_CASES.items()))
    def test_normalization_is_byte_identical(self, query: str, expected: str) -> None:
        """Each expert query classifies expert and normalizes to the golden MATCH."""
        assert _query_is_expert_syntax(query) is True
        assert _normalize_fts_query(query) == expected

    def test_golden_matches_live_normalizer_exactly(self) -> None:
        """The whole golden equals the live normalizer over the canonical set.

        Catches both a stale golden and a query set that drifted from the
        checked-in file without a reviewed regeneration.
        """
        assert compute_expert_normalizations() == _EXPERT_CASES

    def test_golden_is_superset_of_prior_inline_cases(self) -> None:
        """Nothing lost: every previously-inline parity case is still covered."""
        assert LEGACY_EXPERT_PARITY_QUERIES.issubset(_EXPERT_CASES)
        # A strict superset of the five cases that used to live inline.
        assert len(_EXPERT_CASES) > len(LEGACY_EXPERT_PARITY_QUERIES)

    def test_golden_spans_the_cross_product(self) -> None:
        """Meta-test: the golden genuinely spans column-filter x phrase x boolean.

        Guards against a golden that silently shrank to a trivial shape.
        """
        queries = list(_EXPERT_CASES)

        def any_query(predicate: Callable[[str], bool]) -> bool:
            return any(predicate(query) for query in queries)

        # Phrase, and each boolean operator.
        assert any_query(lambda q: q.count('"') >= 2)
        assert any_query(lambda q: " AND " in q)
        assert any_query(lambda q: " OR " in q)
        assert any_query(lambda q: " NOT " in q)
        # Both indexed column filters.
        assert any_query(lambda q: q.startswith("essence:"))
        assert any_query(lambda q: q.startswith("content:"))
        # The column-filter-phrase shape (the dropped-scope bug class).
        assert any_query(lambda q: _COLUMN_FILTER_PHRASE_RE.search(q) is not None)
        # At least one non-identity rewrite (hyphenated identifier -> phrase).
        assert any_query(lambda q: _EXPERT_CASES[q] != q)

    async def test_expert_queries_execute_without_fallback(
        self,
        fts_store: SqliteEngravaCore,
    ) -> None:
        """Every golden expert query drives a valid MATCH with no fallback.

        Preserves the semantic of the retired ``TestGenuineExpertParity``: a
        genuine expert query is valid FTS5 as written, so the primary-``MATCH``
        failure counter never moves across the whole golden set.
        """
        before = fts_store.fts_match_failure_count
        for query in _EXPERT_CASES:
            assert isinstance(await fts_store.search_fts(query), list)
        assert fts_store.fts_match_failure_count == before


class TestHybridRankedGolden:
    """The hybrid ranked list is frozen to a deterministic checked-in golden."""

    async def test_ranked_results_match_golden(
        self,
        hybrid_store: SqliteEngravaCore,
    ) -> None:
        """Every hybrid query reproduces its frozen ordered ranked result."""
        mismatches: list[str] = []
        for query, expected in _HYBRID_CASES.items():
            result = await hybrid_store.search_hybrid(query, top_k=HYBRID_TOP_K)
            actual = [
                [thought_id, round(score, HYBRID_SCORE_NDIGITS)]
                for thought_id, score in result.results
            ]
            if actual != expected:
                mismatches.append(query)
        assert mismatches == [], f"hybrid ranking drifted from golden for: {mismatches}"

    def test_golden_declares_the_precision_it_was_generated_with(self) -> None:
        """The on-disk golden pins the same precision and depth the test asserts."""
        document = load_golden(HYBRID_RANKED_GOLDEN_PATH)
        assert document["score_ndigits"] == HYBRID_SCORE_NDIGITS
        assert document["top_k"] == HYBRID_TOP_K

    def test_golden_includes_the_column_filter_discriminator(self) -> None:
        """The end-to-end discriminator (a column-filter phrase) is frozen here."""
        assert _HYBRID_DISCRIMINATOR_QUERY in _HYBRID_CASES
        assert _COLUMN_FILTER_PHRASE_RE.search(_HYBRID_DISCRIMINATOR_QUERY) is not None


class TestGoldenDiscriminatingPower:
    """Reverting the column-filter drop must break BOTH goldens.

    A single in-process revert — the exact rewrite the WS rejected — is applied
    below. It must make the expert-normalizer golden AND the frozen hybrid
    golden fail, proving each golden discriminates a wrong answer from an answer
    rather than passing vacuously.
    """

    def test_revert_breaks_the_expert_normalization_golden(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The column-filter drop makes every phrase-filter case miss its golden."""
        original = engrava_core._normalize_fts_query
        monkeypatch.setattr(
            engrava_core,
            "_normalize_fts_query",
            _make_column_filter_dropping_normalizer(original),
        )

        column_filter_phrase = {
            query for query in _EXPERT_CASES if _COLUMN_FILTER_PHRASE_RE.search(query)
        }
        broken = {
            query
            for query in column_filter_phrase
            if engrava_core._normalize_fts_query(query) != _EXPERT_CASES[query]
        }
        # Every column-filter phrase case now diverges from its golden value...
        assert broken == column_filter_phrase
        # ...and the set is non-empty, so the golden really carries the bug class.
        assert broken
        # Non-column-filter cases are untouched by the surgical revert.
        for query in _EXPERT_CASES.keys() - column_filter_phrase:
            assert engrava_core._normalize_fts_query(query) == _EXPERT_CASES[query]
        # The canonical WS case: scope dropped to a bare OR query.
        assert engrava_core._normalize_fts_query('essence:"a b"') == "essence OR a OR b"
        assert _EXPERT_CASES['essence:"a b"'] == 'essence:"a b"'

    async def test_revert_breaks_the_hybrid_ranked_golden(
        self,
        hybrid_store: SqliteEngravaCore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The same drop reshuffles EVERY column-filter phrase query's ranking.

        The discriminator must cover the column-filter class in the hybrid
        golden, not a single instance: every column-filter phrase query the
        golden freezes must re-rank end-to-end when its scope is dropped, or the
        golden could not see that query's regression.
        """
        original = engrava_core._normalize_fts_query
        monkeypatch.setattr(
            engrava_core,
            "_normalize_fts_query",
            _make_column_filter_dropping_normalizer(original),
        )

        column_filter_queries = [
            query for query in _HYBRID_CASES if _COLUMN_FILTER_PHRASE_RE.search(query)
        ]
        # The golden carries more than one column-filter phrase query, so the
        # discriminator proves the class, not just the single strong instance.
        assert len(column_filter_queries) >= 2

        unchanged: list[str] = []
        for query in column_filter_queries:
            result = await hybrid_store.search_hybrid(query, top_k=HYBRID_TOP_K)
            actual = [
                [thought_id, round(score, HYBRID_SCORE_NDIGITS)]
                for thought_id, score in result.results
            ]
            if actual == _HYBRID_CASES[query]:
                unchanged.append(query)
        assert unchanged == [], (
            "dropping the column filter must re-rank every column-filter phrase "
            f"query in the hybrid golden; these did not change: {unchanged}"
        )
