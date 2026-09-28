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

from engrava import SqliteEngravaCore
from engrava.infrastructure.sqlite.engrava_core import _normalize_fts_query
from tests.search_contract.conftest import (
    make_embedding_provider,
    open_populated_store,
    populate_corpus,
)

# ---------------------------------------------------------------------------
# On-disk golden layout
# ---------------------------------------------------------------------------

GOLDENS_DIR = Path(__file__).parent / "goldens"
EXPERT_NORMALIZATION_GOLDEN_PATH = GOLDENS_DIR / "fts_expert_normalization.json"
HYBRID_RANKED_GOLDEN_PATH = GOLDENS_DIR / "hybrid_ranked_results.json"
HYBRID_RANKED_FROM_CONFIG_GOLDEN_PATH = GOLDENS_DIR / "hybrid_ranked_results_from_config.json"
#: The YAML asset the ``from_config``-built golden loads. Ships a database
#: path of ``:memory:`` and only the one search weight that needs an explicit
#: value (see the file's own header comment for why).
FROM_CONFIG_ASSET_PATH = GOLDENS_DIR / "from_config_search.yaml"

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
    # Priority-signal discriminator (see conftest._CORPUS): isolated
    # vocabulary shared by exactly two byte-identical-content turns that
    # differ only in priority.
    "lighthouse keeper aurora Kelso Sound",
    # Cycle-signal discriminator: isolated vocabulary shared by two
    # byte-identical-content turns that differ in cycle x priority.
    "quail migration route survey near the delta",
    # Graph-signal discriminator: isolated vocabulary shared by the
    # control/target twins plus their connected neighbour.
    "apprentice welder night shift inspection fire drill",
)

# ---------------------------------------------------------------------------
# Hybrid query-time overrides
# ---------------------------------------------------------------------------
# A directly-constructed store (no ``SearchConfig``) resolves an unspecified
# ``recency_weight`` to ``0.0`` (see
# ``SqliteEngravaCore._resolve_hybrid_defaults``) — NOT the ``0.10`` documented
# default, which only applies once a ``SearchConfig`` exists. And every
# construction path resolves an unspecified ``graph_weight`` to ``0.0`` (the
# graph signal is opt-in). Without these two explicit overrides, the recency
# and graph signals would be structurally silent in this golden regardless of
# what the corpus contains — corpus variety alone does not activate them. The
# recency weight (0.10), the half-life (left at its 50-cycle default) and the
# edge decay (left at its 0.5 default) match the documented defaults. The 0.1
# graph weight does not: the documented default is 0.0, and 0.1 is an explicit
# choice that switches the graph signal on for this golden.
HYBRID_CURRENT_CYCLE = 100
# Deliberately always positive: a resolved recency weight of ``0.0`` takes a
# different code path on the query-less fallback (flat scores instead of
# cycle decay), so this frozen corpus has no coverage of that configuration.
# Do not lower this to ``0.0`` to "improve" coverage — add a dedicated,
# non-golden test for the zero-weight case instead.
HYBRID_RECENCY_WEIGHT = 0.10
HYBRID_GRAPH_WEIGHT = 0.1


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


async def _rank_queries(
    store: SqliteEngravaCore,
    *,
    pass_recency_and_graph_overrides: bool,
) -> dict[str, list[RankedEntry]]:
    """Run every query in :data:`HYBRID_RANKING_QUERIES` against a store.

    Shared by both golden-generation paths so the only difference between
    them is how the store itself was built (direct construction vs.
    ``from_config``) — never a difference in how the queries are issued.

    Args:
        store: A store already populated with the shared corpus.
        pass_recency_and_graph_overrides: When ``True``, pass explicit
            ``recency_weight`` / ``graph_weight`` per-call overrides (the
            directly-constructed store has no ``SearchConfig``, so these
            signals would otherwise be silent — see the module-level
            comment above :data:`HYBRID_CURRENT_CYCLE`). When ``False``,
            leave them unset so they resolve from the store's own
            ``SearchConfig`` (the ``from_config`` path already activates
            them through the loaded YAML). ``current_cycle`` is always
            passed explicitly either way — it is a per-call argument on
            every construction path, never config-driven.

    Returns:
        An insertion-ordered mapping ``query -> [[thought_id, score], ...]``.
    """
    rankings: dict[str, list[RankedEntry]] = {}
    for query in HYBRID_RANKING_QUERIES:
        if pass_recency_and_graph_overrides:
            result = await store.search_hybrid(
                query,
                top_k=HYBRID_TOP_K,
                current_cycle=HYBRID_CURRENT_CYCLE,
                recency_weight=HYBRID_RECENCY_WEIGHT,
                graph_weight=HYBRID_GRAPH_WEIGHT,
            )
        else:
            result = await store.search_hybrid(
                query,
                top_k=HYBRID_TOP_K,
                current_cycle=HYBRID_CURRENT_CYCLE,
            )
        rankings[query] = [
            [thought_id, round(score, HYBRID_SCORE_NDIGITS)] for thought_id, score in result.results
        ]
    return rankings


async def compute_hybrid_rankings() -> dict[str, list[RankedEntry]]:
    """Return each hybrid query mapped to its frozen ranked result.

    Builds the deterministic hybrid store (direct construction, no
    ``SearchConfig``), runs every query in :data:`HYBRID_RANKING_QUERIES` with
    the explicit recency/graph overrides those signals need on this
    construction path, and rounds each score to :data:`HYBRID_SCORE_NDIGITS`.

    Returns:
        An insertion-ordered mapping ``query -> [[thought_id, score], ...]``.
    """
    store, conn = await open_populated_store(
        embedding_provider=make_embedding_provider(),
        auto_embed=True,
    )
    try:
        return await _rank_queries(store, pass_recency_and_graph_overrides=True)
    finally:
        await conn.close()


async def compute_hybrid_rankings_from_config() -> dict[str, list[RankedEntry]]:
    """Return each hybrid query's ranked result, built through ``from_config``.

    Exercises the config-file construction path (:meth:`SqliteEngravaCore.
    from_config`) end to end — YAML parsing, ``SearchConfig`` resolution, an
    in-memory database — rather than the direct constructor the other golden
    uses, so a wiring bug specific to that path (e.g. a ``_parse_search``
    field silently mapped wrong) has a baseline that can catch it. The only
    piece ``from_config`` cannot resolve deterministically and network-free is
    the embedding provider (its built-in providers all call out to a real
    model or API), so this swaps in the same deterministic bag-of-words
    provider immediately after construction, before any thought is written —
    the same provider the direct-construction golden uses, so the vector arm
    stays comparable between the two.

    Returns:
        An insertion-ordered mapping ``query -> [[thought_id, score], ...]``.
    """
    store = await SqliteEngravaCore.from_config(FROM_CONFIG_ASSET_PATH)
    try:
        store._embedding_provider = make_embedding_provider()
        store._auto_embed = True
        await populate_corpus(store)
        return await _rank_queries(store, pass_recency_and_graph_overrides=False)
    finally:
        await store.close()


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


def load_hybrid_ranked_cases(
    path: Path = HYBRID_RANKED_GOLDEN_PATH,
) -> dict[str, list[RankedEntry]]:
    """Return a hybrid golden as a ``query -> [[thought_id, score], ...]`` map.

    Args:
        path: Which hybrid golden to load. Defaults to the directly-constructed
            golden; pass :data:`HYBRID_RANKED_FROM_CONFIG_GOLDEN_PATH` for the
            ``from_config``-built one.

    Raises:
        TypeError: If any ranked entry is not a ``[thought_id, score]`` pair.
    """
    parsed: dict[str, list[RankedEntry]] = {}
    for query, entries in _cases(path).items():
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


async def render_hybrid_ranked_from_config_golden() -> str:
    """Render the ``from_config``-built hybrid ranked-result golden.

    Same corpus and query set as :func:`render_hybrid_ranked_golden`, but the
    store is built through ``SqliteEngravaCore.from_config`` instead of the
    direct constructor — see :func:`compute_hybrid_rankings_from_config` for
    why that path needs its own frozen baseline rather than inheriting the
    directly-constructed golden's assumption that the two are equivalent.
    """
    document: dict[str, object] = {
        "description": (
            "Frozen hybrid ranked results over the same deterministic "
            "search-contract corpus as hybrid_ranked_results.json, but built "
            "through SqliteEngravaCore.from_config (YAML config parsing + "
            "SearchConfig resolution) instead of direct construction — see "
            "goldens/from_config_search.yaml. Exercises the config-file wiring "
            "path independently; regenerate only when a ranking change is "
            "intended."
        ),
        "regenerate": REGEN_COMMAND,
        "score_ndigits": HYBRID_SCORE_NDIGITS,
        "top_k": HYBRID_TOP_K,
        "cases": await compute_hybrid_rankings_from_config(),
    }
    return _render(document)
