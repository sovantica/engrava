"""Tests for previously-uncovered branches in SqliteEngravaCore.

Covers:
- list_thoughts() filter branches: thought_type, min_cycle, max_cycle,
  visibility, exclude_visibility.
- get_edges() direction="IN" and direction="BOTH".
- _cosine_similarity() with zero-magnitude vectors.
- _resolve_hybrid_defaults() validation errors (negative weights, bad half_life).
- _normalize_fts_token() edge cases: AND/OR/NOT passthrough, parentheses,
  trailing '*' without hyphen.
- _encode_consolidated() / _decode_consolidated() with non-None values.
- create_thought() persists consolidated_from and retrieves it correctly.
- _load_recency_scores() early-return on empty thought_ids set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import (
    CoreThoughtRecord,
    EdgeRecord,
    EdgeType,
    KnowledgeSource,
    LifecycleStatus,
    Priority,
    SqliteEngravaCore,
    ThoughtType,
    ThoughtVisibility,
)
from engrava.infrastructure.sqlite.engrava_core import (
    _cosine_similarity,
    _decode_consolidated,
    _encode_consolidated,
    _normalize_fts_query,
    _normalize_fts_token,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def db() -> AsyncIterator[aiosqlite.Connection]:
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode = WAL")
    await conn.execute("PRAGMA foreign_keys = ON")
    store = SqliteEngravaCore(conn)
    await store.ensure_schema()
    yield conn
    await conn.close()


@pytest.fixture
async def store(db: aiosqlite.Connection) -> SqliteEngravaCore:
    s = SqliteEngravaCore(db)
    await s._probe_fts()
    return s


def _make_thought(
    thought_id: str = "t-001",
    *,
    thought_type: ThoughtType = ThoughtType.TASK,
    lifecycle_status: LifecycleStatus = LifecycleStatus.ACTIVE,
    priority: Priority = Priority.P2,
    visibility: ThoughtVisibility = ThoughtVisibility.SELECTIVE,
    updated_cycle: int = 0,
    consolidated_from: list[str] | None = None,
) -> CoreThoughtRecord:
    return CoreThoughtRecord(
        thought_id=thought_id,
        thought_type=thought_type,
        essence="Test thought",
        content="Test content",
        priority=priority,
        lifecycle_status=lifecycle_status,
        created_cycle=0,
        updated_cycle=updated_cycle,
        source="test",
        confidence=0.8,
        source_type=KnowledgeSource.EXPERIENCE,
        visibility=visibility,
        consolidated_from=consolidated_from,
    )


# ---------------------------------------------------------------------------
# list_thoughts() — filter branches
# ---------------------------------------------------------------------------


class TestListThoughtsFilters:
    """Verify all filter branches inside list_thoughts()."""

    async def test_filter_thought_type(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_make_thought("t-task", thought_type=ThoughtType.TASK))
        await store.create_thought(_make_thought("t-belief", thought_type=ThoughtType.BELIEF))
        tasks = await store.list_thoughts(thought_type="TASK")
        assert len(tasks) == 1
        assert tasks[0].thought_id == "t-task"

        beliefs = await store.list_thoughts(thought_type="BELIEF")
        assert len(beliefs) == 1
        assert beliefs[0].thought_id == "t-belief"

    async def test_filter_min_cycle(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_make_thought("t-old", updated_cycle=0))
        await store.create_thought(_make_thought("t-new", updated_cycle=10))
        results = await store.list_thoughts(min_cycle=5)
        assert len(results) == 1
        assert results[0].thought_id == "t-new"

    async def test_filter_max_cycle(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_make_thought("t-old", updated_cycle=0))
        await store.create_thought(_make_thought("t-new", updated_cycle=10))
        results = await store.list_thoughts(max_cycle=5)
        assert len(results) == 1
        assert results[0].thought_id == "t-old"

    async def test_filter_min_and_max_cycle(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_make_thought("t-0", updated_cycle=0))
        await store.create_thought(_make_thought("t-5", updated_cycle=5))
        await store.create_thought(_make_thought("t-10", updated_cycle=10))
        results = await store.list_thoughts(min_cycle=3, max_cycle=7)
        assert len(results) == 1
        assert results[0].thought_id == "t-5"

    async def test_filter_visibility_include(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_make_thought("t-pub", visibility=ThoughtVisibility.PUBLIC))
        await store.create_thought(_make_thought("t-priv", visibility=ThoughtVisibility.PRIVATE))
        results = await store.list_thoughts(visibility="public")
        assert len(results) == 1
        assert results[0].thought_id == "t-pub"

    async def test_filter_exclude_visibility(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_make_thought("t-pub", visibility=ThoughtVisibility.PUBLIC))
        await store.create_thought(_make_thought("t-priv", visibility=ThoughtVisibility.PRIVATE))
        results = await store.list_thoughts(exclude_visibility="private")
        assert len(results) == 1
        assert results[0].thought_id == "t-pub"

    async def test_filter_thought_type_returns_empty(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_make_thought("t-task", thought_type=ThoughtType.TASK))
        results = await store.list_thoughts(thought_type="BELIEF")
        assert results == []


# ---------------------------------------------------------------------------
# get_edges() — direction branches
# ---------------------------------------------------------------------------


class TestGetEdgesDirections:
    """Verify get_edges() with all three direction values."""

    @pytest.fixture
    async def graph(self, store: SqliteEngravaCore) -> SqliteEngravaCore:
        """Create two thoughts and one directed edge a→b."""
        await store.create_thought(_make_thought("t-a"))
        await store.create_thought(_make_thought("t-b"))
        edge = EdgeRecord(
            edge_id="e-001",
            from_thought_id="t-a",
            to_thought_id="t-b",
            edge_type=EdgeType.DEPENDS_ON,
            weight=1.0,
            created_cycle=0,
        )
        await store.create_edge(edge)
        return store

    async def test_direction_out(self, graph: SqliteEngravaCore) -> None:
        edges = await graph.get_edges("t-a", direction="OUT")
        assert len(edges) == 1
        assert edges[0].from_thought_id == "t-a"

    async def test_direction_in(self, graph: SqliteEngravaCore) -> None:
        edges = await graph.get_edges("t-b", direction="IN")
        assert len(edges) == 1
        assert edges[0].to_thought_id == "t-b"

    async def test_direction_in_source_has_no_incoming(self, graph: SqliteEngravaCore) -> None:
        edges = await graph.get_edges("t-a", direction="IN")
        assert edges == []

    async def test_direction_both_from_perspective_of_source(
        self, graph: SqliteEngravaCore
    ) -> None:
        edges = await graph.get_edges("t-a", direction="BOTH")
        assert len(edges) == 1
        assert edges[0].from_thought_id == "t-a"

    async def test_direction_both_from_perspective_of_target(
        self, graph: SqliteEngravaCore
    ) -> None:
        edges = await graph.get_edges("t-b", direction="BOTH")
        assert len(edges) == 1
        assert edges[0].to_thought_id == "t-b"

    async def test_direction_both_bidirectional(self, store: SqliteEngravaCore) -> None:
        """Node with both incoming and outgoing edges returns all."""
        await store.create_thought(_make_thought("t-x"))
        await store.create_thought(_make_thought("t-y"))
        await store.create_thought(_make_thought("t-z"))
        await store.create_edge(
            EdgeRecord(
                edge_id="e-xy",
                from_thought_id="t-x",
                to_thought_id="t-y",
                edge_type=EdgeType.ASSOCIATED,
                weight=1.0,
                created_cycle=0,
            )
        )
        await store.create_edge(
            EdgeRecord(
                edge_id="e-zy",
                from_thought_id="t-z",
                to_thought_id="t-y",
                edge_type=EdgeType.ASSOCIATED,
                weight=1.0,
                created_cycle=0,
            )
        )
        edges = await store.get_edges("t-y", direction="BOTH")
        assert len(edges) == 2


# ---------------------------------------------------------------------------
# _cosine_similarity() — zero-magnitude vectors
# ---------------------------------------------------------------------------


class TestCosineSimilarityZeroVector:
    """_cosine_similarity() must return 0.0 for zero-magnitude vectors."""

    def test_zero_a_returns_zero(self) -> None:
        result = _cosine_similarity([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        assert result == 0.0

    def test_zero_b_returns_zero(self) -> None:
        result = _cosine_similarity([1.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        assert result == 0.0

    def test_both_zero_returns_zero(self) -> None:
        result = _cosine_similarity([0.0, 0.0], [0.0, 0.0])
        assert result == 0.0

    def test_nonzero_vectors_returns_nonzero(self) -> None:
        result = _cosine_similarity([1.0, 0.0], [1.0, 0.0])
        assert abs(result - 1.0) < 1e-9

    async def test_search_similar_with_zero_embedding(self, store: SqliteEngravaCore) -> None:
        """Zero-magnitude stored embedding produces 0.0 similarity score."""
        await store.create_thought(_make_thought("t-001"))
        await store.store_embedding("t-001", [0.0, 0.0, 0.0])
        results = await store.search_similar([1.0, 0.0, 0.0])
        assert len(results) == 1
        assert results[0][1] == 0.0


# ---------------------------------------------------------------------------
# _resolve_hybrid_defaults() — validation errors
# ---------------------------------------------------------------------------


class TestResolveHybridDefaultsValidation:
    """_resolve_hybrid_defaults() raises ValueError for invalid weights."""

    async def test_negative_fts_weight_raises(self, store: SqliteEngravaCore) -> None:
        with pytest.raises(ValueError, match="fts_weight"):
            await store.search_hybrid("test", fts_weight=-0.1)

    async def test_negative_vector_weight_raises(self, store: SqliteEngravaCore) -> None:
        with pytest.raises(ValueError, match="vector_weight"):
            await store.search_hybrid("test", vector_weight=-1.0)

    async def test_negative_recency_weight_raises(self, store: SqliteEngravaCore) -> None:
        with pytest.raises(ValueError, match="recency_weight"):
            await store.search_hybrid("test", recency_weight=-0.5)

    async def test_zero_recency_half_life_raises(self, store: SqliteEngravaCore) -> None:
        with pytest.raises(ValueError, match="recency_half_life"):
            await store.search_hybrid("test", recency_half_life=0)

    async def test_negative_recency_half_life_raises(self, store: SqliteEngravaCore) -> None:
        with pytest.raises(ValueError, match="recency_half_life"):
            await store.search_hybrid("test", recency_half_life=-5)


# ---------------------------------------------------------------------------
# _normalize_fts_token() — edge cases
# ---------------------------------------------------------------------------


class TestNormalizeFtsToken:
    """Edge cases in the FTS5 token normalizer.

    ``_normalize_fts_token`` returns the list of FTS5 terms a single token
    expands to. A token may yield zero terms (all punctuation), one term
    (a plain word), or several (a contraction split on its clitic). The
    ``expert`` flag mirrors the surrounding query: expert syntax preserves
    operators/phrases/column filters, bare syntax sanitizes everything.
    """

    # --- Expert-mode passthrough -------------------------------------------

    def test_and_keyword_passthrough(self) -> None:
        assert _normalize_fts_token("AND", expert=True) == ["AND"]

    def test_or_keyword_passthrough(self) -> None:
        assert _normalize_fts_token("OR", expert=True) == ["OR"]

    def test_not_keyword_passthrough(self) -> None:
        assert _normalize_fts_token("NOT", expert=True) == ["NOT"]

    def test_token_with_quotes_passthrough(self) -> None:
        assert _normalize_fts_token('"already-quoted"', expert=True) == ['"already-quoted"']

    def test_whitelisted_column_filter_passthrough(self) -> None:
        assert _normalize_fts_token("essence:value", expert=True) == ["essence:value"]
        assert _normalize_fts_token("content:value", expert=True) == ["content:value"]

    def test_unknown_column_filter_is_sanitized(self) -> None:
        # A non-whitelisted column would crash FTS5 ("no such column: field"),
        # so it is split on the colon into bare OR-terms instead.
        assert _normalize_fts_token("field:value", expert=True) == ["field", "value"]

    def test_natural_language_colon_is_sanitized(self) -> None:
        assert _normalize_fts_token("events:", expert=True) == ["events"]

    def test_empty_token_yields_no_terms(self) -> None:
        assert _normalize_fts_token("", expert=True) == []
        assert _normalize_fts_token("", expert=False) == []

    def test_no_hyphen_passthrough(self) -> None:
        assert _normalize_fts_token("simple", expert=True) == ["simple"]

    def test_prefix_star_no_hyphen(self) -> None:
        """Trailing '*' on a non-hyphenated token is preserved."""
        assert _normalize_fts_token("prefix*", expert=True) == ["prefix*"]

    def test_hyphen_token_normalized(self) -> None:
        assert _normalize_fts_token("REQ-FUNC", expert=True) == ['"REQ-FUNC"']

    def test_hyphen_token_with_star(self) -> None:
        assert _normalize_fts_token("REQ-FUNC*", expert=True) == ['"REQ-FUNC"*']

    def test_parenthesized_hyphen_token(self) -> None:
        assert _normalize_fts_token("(REQ-001)", expert=True) == ['("REQ-001")']

    def test_leading_paren_only(self) -> None:
        assert _normalize_fts_token("(REQ-001", expert=True) == ['("REQ-001"']

    def test_trailing_paren_only(self) -> None:
        assert _normalize_fts_token("REQ-001)", expert=True) == ['"REQ-001")']

    # --- Bare natural-language mode ----------------------------------------

    def test_bare_no_hyphen_word(self) -> None:
        assert _normalize_fts_token("simple", expert=False) == ["simple"]

    def test_bare_contraction_splits_into_terms(self) -> None:
        assert _normalize_fts_token("sister's", expert=False) == ["sister", "s"]

    def test_bare_hyphen_token_is_quoted(self) -> None:
        assert _normalize_fts_token("REQ-FUNC", expert=False) == ['"REQ-FUNC"']

    def test_bare_url_splits_into_fragments(self) -> None:
        # A pasted URL is never a column filter in bare mode; it splits on the
        # colon, slashes and dots into useful OR-terms.
        assert _normalize_fts_token("http://example.com", expert=False) == [
            "http",
            "example",
            "com",
        ]

    def test_bare_only_punctuation_yields_no_terms(self) -> None:
        assert _normalize_fts_token("!!!", expert=False) == []

    # --- Whole-query normalization -----------------------------------------

    def test_expert_query_joins_with_spaces(self) -> None:
        # Uppercase operators trigger expert mode → space-joined implicit AND.
        result = _normalize_fts_query("REQ-001 AND simple-word OR OR")
        assert '"REQ-001"' in result
        assert '"simple-word"' in result
        assert " AND " in result
        assert result.endswith("OR")

    def test_bare_query_joins_with_or(self) -> None:
        result = _normalize_fts_query("coffee creamer coupon")
        assert result == "coffee OR creamer OR coupon"

    def test_currency_token_loses_dollar_prefix(self) -> None:
        assert _normalize_fts_token("$5", expert=False) == ["5"]

    def test_question_with_currency_and_punctuation_normalizes(self) -> None:
        result = _normalize_fts_query("Where did I redeem a $5 coupon on coffee creamer?")
        assert "$" not in result
        assert "?" not in result
        assert "5" in result
        # Bare query → OR-joined.
        assert " OR " in result


# ---------------------------------------------------------------------------
# _encode_consolidated() / _decode_consolidated() — non-None values
# ---------------------------------------------------------------------------


class TestConsolidatedEncoding:
    """Module-level JSON helpers for consolidated_from field."""

    def test_encode_none_returns_none(self) -> None:
        assert _encode_consolidated(None) is None

    def test_encode_empty_list(self) -> None:
        result = _encode_consolidated([])
        assert result == "[]"

    def test_encode_list_with_ids(self) -> None:
        result = _encode_consolidated(["t-001", "t-002"])
        assert result is not None
        assert "t-001" in result
        assert "t-002" in result

    def test_decode_none_returns_none(self) -> None:
        assert _decode_consolidated(None) is None

    def test_decode_json_string(self) -> None:
        result = _decode_consolidated('["t-001", "t-002"]')
        assert result == ["t-001", "t-002"]

    def test_roundtrip(self) -> None:
        original = ["src-a", "src-b", "src-c"]
        encoded = _encode_consolidated(original)
        decoded = _decode_consolidated(encoded)
        assert decoded == original

    async def test_create_thought_with_consolidated_from_persisted(
        self, store: SqliteEngravaCore
    ) -> None:
        """consolidated_from field survives a round-trip through SQLite."""
        await store.create_thought(_make_thought("t-src-1"))
        await store.create_thought(_make_thought("t-src-2"))
        consolidated = _make_thought(
            "t-consolidated",
            consolidated_from=["t-src-1", "t-src-2"],
        )
        await store.create_thought(consolidated)
        retrieved = await store.get_thought("t-consolidated")
        assert retrieved is not None
        assert retrieved.consolidated_from == ["t-src-1", "t-src-2"]


# ---------------------------------------------------------------------------
# _load_recency_scores() — empty thought_ids early return
# ---------------------------------------------------------------------------


class TestLoadRecencyScores:
    """_load_recency_scores() returns {} immediately for empty input."""

    async def test_empty_thought_ids_returns_empty_dict(self, store: SqliteEngravaCore) -> None:
        result = await store._load_recency_scores(
            thought_ids=set(),
            current_cycle=10,
            recency_half_life=50,
        )
        assert result == {}

    async def test_nonempty_thought_ids_returns_scores(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_make_thought("t-001", updated_cycle=0))
        result = await store._load_recency_scores(
            thought_ids={"t-001"},
            current_cycle=10,
            recency_half_life=50,
        )
        assert "t-001" in result
        assert 0.0 <= result["t-001"] <= 1.0
