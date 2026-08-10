"""Shared query sets and generation logic for the checked-in search goldens.

Two goldens defend retrieval *semantics* (not merely liveness) against the
column-filter-drop regression class: a rewrite once normalized ``essence:"a b"``
to an unscoped ``essence a b`` — still valid FTS5, still returning documents — so
every findability / never-raises / arm-liveness test stayed green while the
answer was semantically wrong. Only a byte-identical normalizer golden or a
frozen ranked-result golden tells "different answer" apart from "an answer".

This module is the single source of truth for BOTH the golden tests
(``test_search_goldens.py``) and the reviewed regeneration entry point
(``scripts/regenerate_search_goldens.py``): the query sets, the store
construction (reused from :mod:`tests.search_contract.conftest`), the
score-rounding precision, and the on-disk golden format all live here, so a
regenerated golden is byte-identical to what the tests read. The tests only
*read* these goldens; they never rewrite them — regeneration is an explicit,
reviewed command, so a genuine semantic drift surfaces as a failing assertion
rather than a silently-overwritten fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

from engrava.infrastructure.sqlite.engrava_core import _normalize_fts_query
from tests.search_contract.conftest import make_embedding_provider, open_populated_store

# ---------------------------------------------------------------------------
# On-disk golden layout
# ---------------------------------------------------------------------------

GOLDENS_DIR = Path(__file__).parent / "goldens"
EXPERT_NORMALIZATION_GOLDEN_PATH = GOLDENS_DIR / "fts_expert_normalization.json"
HYBRID_RANKED_GOLDEN_PATH = GOLDENS_DIR / "hybrid_ranked_results.json"

#: Command a maintainer runs to regenerate the goldens after an *intended*
#: retrieval-semantics change (recorded inside each golden file for provenance).
REGEN_COMMAND = "python scripts/regenerate_search_goldens.py"

#: Rounding precision for the frozen hybrid scores. Six digits is far tighter
#: than any legitimate fusion change yet survives JSON round-trip exactly (the
#: bag-of-words arm is deterministic, so there is no float jitter to absorb).
HYBRID_SCORE_NDIGITS = 6
#: Result depth frozen per hybrid query.
HYBRID_TOP_K = 10

# A ranked entry is ``[thought_id, rounded_score]`` — a JSON array, since the
# goldens round-trip through JSON where tuples are indistinguishable from lists.
RankedEntry = list[str | float]

# ---------------------------------------------------------------------------
# Golden 1 — expert-normalizer parity query set
# ---------------------------------------------------------------------------
# The full column-filter x phrase x boolean cross-product. Every entry is a
# *genuine* expert query (``_query_is_expert_syntax`` is True) whose normalized
# MATCH must stay byte-identical release-to-release. The set is a strict superset
# of the five cases that previously lived inline (see
# :data:`LEGACY_EXPERT_PARITY_QUERIES`), and deliberately spans the exact
# column-filter phrase shape (``essence:"a b"``) whose scope a prior rewrite
# dropped, plus non-identity rewrites (hyphenated identifiers) so the golden pins
# real normalization behaviour rather than a pure pass-through.

EXPERT_NORMALIZATION_QUERIES: tuple[str, ...] = (
    # Single-column filters, single token.
    "essence:memory",
    "content:memory",
    # Single-column filters wrapping a phrase (the dropped-scope bug class).
    'essence:"a b"',
    'content:"machine learning"',
    'content:"fiddle leaf"',
    'essence:"office plant"',
    # Boolean of two column filters.
    "content:foo AND essence:bar",
    "content:foo OR essence:bar",
    "content:foo NOT essence:bar",
    # Column filter combined with a bare token via a boolean.
    "content:memory AND relevant",
    "essence:body OR forum",
    # Bare boolean queries (no column filter).
    "cats AND dogs",
    "cats OR dogs",
    "cats NOT dogs",
    # Multi-operator boolean chains.
    "a AND b OR c",
    "foo AND bar NOT baz",
    # Standalone phrase queries.
    '"machine learning"',
    '"final answer"',
    # Phrase combined with a boolean.
    '"machine learning" AND relevant',
    '"machine learning" OR "deep learning"',
    '"final answer" NOT draft',
    # Phrase + column filter + boolean together.
    'content:"machine learning" AND essence:summary',
    'essence:"a b" OR content:"c d"',
    # Parenthesised phrase grouping.
    '("machine learning")',
    '("machine learning") AND cats',
    # Column-filter phrase trailed by a bare token.
    'content:"machine learning" relevant',
    # Hyphenated identifiers — expert normalization rewrites these to the
    # accepted phrase-quoted form, so the golden captures a real transformation.
    "content:memory AND REQ-FUNC*",
    "REQ-FUNC AND well-known",
    "essence:body AND req-func",
    "cats AND req-func*",
    '"machine learning" AND req-func',
)

#: The five cases that previously lived inline; the externalized golden must
#: remain a superset of them so nothing is lost in the move.
LEGACY_EXPERT_PARITY_QUERIES: frozenset[str] = frozenset(
    {
        'essence:"a b"',
        'content:"machine learning"',
        "content:foo AND essence:bar",
        "cats AND dogs",
        '"machine learning" AND relevant',
    }
)

# ---------------------------------------------------------------------------
# Golden 2 — frozen ranked hybrid result query set
# ---------------------------------------------------------------------------
# Driven against the deterministic ``hybrid_store`` corpus (bag-of-words vector
# arm, no network, no model). Includes column-filter phrase queries whose ranked
# list changes end-to-end if the column filter is dropped — the regression a
# liveness-only test cannot see.

HYBRID_RANKING_QUERIES: tuple[str, ...] = (
    # Column-filter phrase queries: the scope gate is load-bearing. Dropping it
    # (``content:"three cheeses"`` -> ``content OR three OR cheeses``) pulls in
    # unrelated "three ..." turns and reshuffles the ranked list.
    'content:"three cheeses"',
    'essence:"office plant"',
    # Natural-language gold questions (function words must not block a match).
    "what did I say about the marketing specialist job",
    "who gave the compression talk at the conference",
    # A distinctive multi-term query that engages both arms.
    "marketing specialist startup lessons",
    # A near-duplicate cluster query (ranking among close variants).
    "office fiddle leaf fig",
    # Both-arms-fire distinctive phrase.
    "the hazelnut coffee creamer coupon",
)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def compute_expert_normalizations() -> dict[str, str]:
    """Return each expert query mapped to its live normalized MATCH.

    Returns:
        An insertion-ordered mapping ``query -> _normalize_fts_query(query)`` for
        every entry in :data:`EXPERT_NORMALIZATION_QUERIES`.
    """
    return {query: _normalize_fts_query(query) for query in EXPERT_NORMALIZATION_QUERIES}


async def compute_hybrid_rankings() -> dict[str, list[RankedEntry]]:
    """Return each hybrid query mapped to its frozen ranked result.

    Builds the deterministic hybrid store, runs every query in
    :data:`HYBRID_RANKING_QUERIES`, and rounds each score to
    :data:`HYBRID_SCORE_NDIGITS`.

    Returns:
        An insertion-ordered mapping ``query -> [[thought_id, score], ...]``.
    """
    store, conn = await open_populated_store(
        embedding_provider=make_embedding_provider(),
        auto_embed=True,
    )
    try:
        rankings: dict[str, list[RankedEntry]] = {}
        for query in HYBRID_RANKING_QUERIES:
            result = await store.search_hybrid(query, top_k=HYBRID_TOP_K)
            rankings[query] = [
                [thought_id, round(score, HYBRID_SCORE_NDIGITS)]
                for thought_id, score in result.results
            ]
        return rankings
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# Serialization (used by both the tests, for loading, and the regen script)
# ---------------------------------------------------------------------------


def load_golden(path: Path) -> dict[str, object]:
    """Load and parse a checked-in golden file.

    Args:
        path: Absolute path to the golden JSON file.

    Returns:
        The parsed golden document (a ``description`` / ``regenerate`` / ...
        header plus a ``cases`` mapping).
    """
    with path.open(encoding="utf-8") as handle:
        parsed: dict[str, object] = json.load(handle)
    return parsed


def _cases(path: Path) -> dict[str, object]:
    """Return the ``cases`` mapping of a golden, validating its shape.

    Args:
        path: Absolute path to the golden JSON file.

    Returns:
        The raw ``cases`` mapping.

    Raises:
        TypeError: If the golden has no object-valued ``cases`` member.
    """
    cases = load_golden(path).get("cases")
    if not isinstance(cases, dict):
        msg = f"golden {path.name!r} must contain an object-valued 'cases' member"
        raise TypeError(msg)
    return cases


def load_expert_normalization_cases() -> dict[str, str]:
    """Return the expert-normalizer golden as a ``query -> MATCH`` mapping."""
    cases = _cases(EXPERT_NORMALIZATION_GOLDEN_PATH)
    return {str(query): str(match) for query, match in cases.items()}


def load_hybrid_ranked_cases() -> dict[str, list[RankedEntry]]:
    """Return the hybrid golden as a ``query -> [[thought_id, score], ...]`` map.

    Raises:
        TypeError: If any ranked entry is not a ``[thought_id, score]`` pair.
    """
    parsed: dict[str, list[RankedEntry]] = {}
    for query, entries in _cases(HYBRID_RANKED_GOLDEN_PATH).items():
        if not isinstance(entries, list):
            msg = f"hybrid golden case {query!r} must be a list of ranked entries"
            raise TypeError(msg)
        ranked: list[RankedEntry] = []
        for entry in entries:
            if not isinstance(entry, list) or len(entry) != 2:
                msg = f"hybrid golden case {query!r} has a malformed entry: {entry!r}"
                raise TypeError(msg)
            ranked.append([str(entry[0]), float(entry[1])])
        parsed[str(query)] = ranked
    return parsed


def _render(document: dict[str, object]) -> str:
    """Serialize a golden document to its canonical on-disk form.

    Args:
        document: The golden document to serialize.

    Returns:
        Pretty-printed JSON with a trailing newline (stable, review-friendly,
        and byte-reproducible so ``--check`` can diff it exactly).
    """
    return json.dumps(document, indent=2, ensure_ascii=True) + "\n"


def render_expert_normalization_golden() -> str:
    """Render the expert-normalizer parity golden from the live normalizer."""
    document: dict[str, object] = {
        "description": (
            "Byte-identical FTS5 expert-normalizer parity. Maps each genuine "
            "expert query (column filter / phrase / boolean cross-product) to "
            "its normalized MATCH. A change here means a query's semantics "
            "changed; regenerate only when that change is intended."
        ),
        "regenerate": REGEN_COMMAND,
        "cases": compute_expert_normalizations(),
    }
    return _render(document)


async def render_hybrid_ranked_golden() -> str:
    """Render the frozen hybrid ranked-result golden from the live search."""
    document: dict[str, object] = {
        "description": (
            "Frozen hybrid ranked results over the deterministic search-contract "
            "corpus (bag-of-words vector arm; no model, no network). Maps each "
            "query to its ordered [thought_id, rounded_score] list. Discriminates "
            "'different ranked answer' from 'an answer'; regenerate only when a "
            "ranking change is intended."
        ),
        "regenerate": REGEN_COMMAND,
        "score_ndigits": HYBRID_SCORE_NDIGITS,
        "top_k": HYBRID_TOP_K,
        "cases": await compute_hybrid_rankings(),
    }
    return _render(document)
