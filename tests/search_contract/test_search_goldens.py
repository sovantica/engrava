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

from engrava import SqliteEngravaCore
from engrava.infrastructure.sqlite import engrava_core
from engrava.infrastructure.sqlite.engrava_core import (
    _normalize_fts_query,
    _query_is_expert_syntax,
)
from tests.search_contract.conftest import make_embedding_provider, populate_corpus
from tests.search_contract.golden_fixtures import (
    FROM_CONFIG_ASSET_PATH,
    HYBRID_CURRENT_CYCLE,
    HYBRID_GRAPH_WEIGHT,
    HYBRID_RANKED_FROM_CONFIG_GOLDEN_PATH,
    HYBRID_RANKED_GOLDEN_PATH,
    HYBRID_RECENCY_WEIGHT,
    HYBRID_SCORE_NDIGITS,
    HYBRID_TOP_K,
    LEGACY_EXPERT_PARITY_QUERIES,
    compute_expert_normalizations,
    load_expert_normalization_cases,
    load_golden,
    load_hybrid_ranked_cases,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

# Loaded at collection time so the byte-identity check can parametrize per case.
_EXPERT_CASES: dict[str, str] = load_expert_normalization_cases()
_HYBRID_CASES: dict[str, list[list[str | float]]] = load_hybrid_ranked_cases()
_HYBRID_FROM_CONFIG_CASES: dict[str, list[list[str | float]]] = load_hybrid_ranked_cases(
    HYBRID_RANKED_FROM_CONFIG_GOLDEN_PATH
)


async def _search_direct_golden(store: SqliteEngravaCore, query: str) -> list[list[str | float]]:
    """Run one query the same way the directly-constructed golden was built.

    A directly-constructed store needs explicit ``current_cycle`` /
    ``recency_weight`` / ``graph_weight`` to give the recency and graph
    signals a baseline at all — see the comment above ``HYBRID_CURRENT_CYCLE``
    in golden_fixtures.py. Every live re-query against the
    ``HYBRID_RANKED_GOLDEN_PATH`` golden must reproduce them exactly, or a
    live call that silently drifted from how the golden was generated would
    "pass" by comparing two different queries rather than catching a
    regression.

    Args:
        store: A hybrid-search-ready store built via direct construction.
        query: The query to run.

    Returns:
        The rounded ``[thought_id, score]`` pairs, in ranked order.
    """
    result = await store.search_hybrid(
        query,
        top_k=HYBRID_TOP_K,
        current_cycle=HYBRID_CURRENT_CYCLE,
        recency_weight=HYBRID_RECENCY_WEIGHT,
        graph_weight=HYBRID_GRAPH_WEIGHT,
    )
    return [
        [thought_id, round(score, HYBRID_SCORE_NDIGITS)] for thought_id, score in result.results
    ]


@pytest.fixture
async def hybrid_store_from_config() -> AsyncIterator[SqliteEngravaCore]:
    """Return a store built through ``from_config``, populated with the corpus.

    Mirrors ``conftest.hybrid_store`` but for the config-file construction
    path: loads ``goldens/from_config_search.yaml`` (in-memory database, one
    explicit search weight), then swaps in the same deterministic
    bag-of-words provider the direct-construction fixtures use — ``from_config``
    has no network-free, deterministic built-in embedding provider — before
    writing the corpus, so the vector arm stays comparable between the two
    goldens.

    Yields:
        A :class:`SqliteEngravaCore` built via ``from_config`` with both the
        FTS and vector arms live.
    """
    store = await SqliteEngravaCore.from_config(FROM_CONFIG_ASSET_PATH)
    store._embedding_provider = make_embedding_provider()
    store._auto_embed = True
    await populate_corpus(store)
    yield store
    await store.close()


# A column-filter query whose ranked list visibly reshuffles end-to-end when its
# scope is dropped — the discriminating hybrid case.
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
            actual = await _search_direct_golden(hybrid_store, query)
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

    @pytest.mark.parametrize(
        "query",
        [
            "lighthouse keeper aurora Kelso Sound",
            "quail migration route survey near the delta",
            "apprentice welder night shift inspection fire drill",
        ],
    )
    def test_golden_includes_the_signal_discriminator_pairs(self, query: str) -> None:
        """The three dedicated priority/cycle/graph pairs are frozen here.

        Each carries the two control/target ids the mutation-testing
        procedure reorders; a corpus edit that dropped one silently would
        otherwise slip past the byte-identity check above (which would just
        freeze new, still-vacuous scores on the next regeneration).
        """
        assert query in _HYBRID_CASES
        ids = {str(thought_id) for thought_id, _ in _HYBRID_CASES[query]}
        control_and_target = {tid for tid in ids if tid.endswith(("-control", "-target"))}
        assert len(control_and_target) == 2, (
            f"query {query!r} must freeze both the control and target id: got {ids}"
        )


class TestHybridRankedFromConfigGolden:
    """The ``from_config``-built hybrid ranked list has its own frozen golden.

    Mirrors :class:`TestHybridRankedGolden` exactly, but against a store built
    through ``SqliteEngravaCore.from_config`` — see
    ``golden_fixtures.compute_hybrid_rankings_from_config`` for why that
    construction path needs an independent baseline rather than inheriting
    the directly-constructed golden's assumption that the two never diverge.
    """

    async def test_ranked_results_match_golden(
        self,
        hybrid_store_from_config: SqliteEngravaCore,
    ) -> None:
        """Every hybrid query reproduces its frozen ordered ranked result."""
        mismatches: list[str] = []
        for query, expected in _HYBRID_FROM_CONFIG_CASES.items():
            result = await hybrid_store_from_config.search_hybrid(
                query, top_k=HYBRID_TOP_K, current_cycle=HYBRID_CURRENT_CYCLE
            )
            actual = [
                [thought_id, round(score, HYBRID_SCORE_NDIGITS)]
                for thought_id, score in result.results
            ]
            if actual != expected:
                mismatches.append(query)
        assert mismatches == [], f"hybrid ranking drifted from golden for: {mismatches}"

    def test_golden_declares_the_precision_it_was_generated_with(self) -> None:
        """The on-disk golden pins the same precision and depth the test asserts."""
        document = load_golden(HYBRID_RANKED_FROM_CONFIG_GOLDEN_PATH)
        assert document["score_ndigits"] == HYBRID_SCORE_NDIGITS
        assert document["top_k"] == HYBRID_TOP_K

    def test_golden_covers_the_same_queries_as_the_direct_golden(self) -> None:
        """The two goldens are generated from the same query set.

        Not an assertion that the *scores* agree (a genuine, intended
        divergence between the two construction paths is a legitimate
        outcome) — only that neither golden silently dropped a query the
        other still carries.
        """
        assert set(_HYBRID_FROM_CONFIG_CASES) == set(_HYBRID_CASES)


class TestGoldenDiscriminatingPower:
    """Reverting the column-filter drop must break BOTH goldens.

    A single in-process revert — the rewrite that drops the column-filter scope —
    is applied below. It must make the expert-normalizer golden AND the frozen hybrid
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
        # The canonical case: scope dropped to a bare OR query.
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
            actual = await _search_direct_golden(hybrid_store, query)
            if actual == _HYBRID_CASES[query]:
                unchanged.append(query)
        assert unchanged == [], (
            "dropping the column filter must re-rank every column-filter phrase "
            f"query in the hybrid golden; these did not change: {unchanged}"
        )
