"""Tests for hybrid search enhancement.

Covers:
- Recency scoring (exp decay)
- Auto-embed query integration
- Graceful degradation matrix (all combinations)
- SearchConfig YAML parsing
- Weight redistribution
- Backward compatibility with legacy call patterns
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import aiosqlite
import pytest

from engrava import (
    CallbackProvider,
    SearchConfig,
    SqliteEngravaCore,
)
from engrava.config import (
    ConfigError,
    _parse_search,
)
from engrava.domain.enums import (
    KnowledgeSource,
    LifecycleStatus,
    Priority,
    ThoughtType,
    ThoughtVisibility,
)
from engrava.domain.models.thought import ThoughtRecord

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make(
    thought_id: str,
    *,
    essence: str = "Test",
    content: str = "Content",
    created_cycle: int = 0,
    updated_cycle: int = 0,
    priority: Priority = Priority.P2,
) -> ThoughtRecord:
    """Minimal thought for search tests."""
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.TASK,
        essence=essence,
        content=content,
        priority=priority,
        lifecycle_status=LifecycleStatus.CREATED,
        created_cycle=created_cycle,
        updated_cycle=updated_cycle,
        source="test",
        confidence=0.8,
        source_type=KnowledgeSource.EXPERIENCE,
        visibility=ThoughtVisibility.SELECTIVE,
    )


def _dummy_embed(text: str) -> list[float]:
    """Deterministic dummy embedding (dim=4)."""
    return [float(len(text) % 10) / 10.0] * 4


@pytest.fixture
async def db() -> AsyncIterator[aiosqlite.Connection]:
    """In-memory SQLite connection with core schema."""
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
    """Plain store without embedding provider."""
    return SqliteEngravaCore(db)


@pytest.fixture
def embed_provider() -> CallbackProvider:
    """A CallbackProvider for auto-embed tests."""
    return CallbackProvider(callback=_dummy_embed, dimension=4, model_name="test-4")


@pytest.fixture
async def store_embed(
    db: aiosqlite.Connection,
    embed_provider: CallbackProvider,
) -> SqliteEngravaCore:
    """Store with embedding provider and auto_embed=True."""
    return SqliteEngravaCore(
        db,
        embedding_provider=embed_provider,
        auto_embed=True,
    )


# ---------------------------------------------------------------------------
# SearchConfig parsing
# ---------------------------------------------------------------------------


class TestSearchConfigParsing:
    """Tests for _parse_search YAML parser."""

    def test_parse_none_returns_defaults(self) -> None:
        cfg = _parse_search(None)
        assert cfg == SearchConfig()

    def test_parse_valid_config(self) -> None:
        raw = {
            "default_fts_weight": 0.2,
            "default_vector_weight": 0.5,
            "default_recency_weight": 0.3,
            "recency_half_life": 100,
        }
        cfg = _parse_search(raw)
        assert cfg.default_fts_weight == pytest.approx(0.2)
        assert cfg.default_vector_weight == pytest.approx(0.5)
        assert cfg.default_recency_weight == pytest.approx(0.3)
        assert cfg.recency_half_life == 100

    def test_parse_invalid_type_raises(self) -> None:
        with pytest.raises(ConfigError, match="must be a mapping"):
            _parse_search("bad")

    def test_parse_negative_weight_raises(self) -> None:
        with pytest.raises(ConfigError, match="default_fts_weight"):
            _parse_search({"default_fts_weight": -0.1})

    def test_parse_invalid_half_life_raises(self) -> None:
        with pytest.raises(ConfigError, match="recency_half_life"):
            _parse_search({"recency_half_life": 0})

    def test_parse_non_int_half_life_raises(self) -> None:
        with pytest.raises(ConfigError, match="recency_half_life"):
            _parse_search({"recency_half_life": 3.5})

    def test_defaults_match_dataclass(self) -> None:
        cfg = _parse_search({})
        assert cfg.default_fts_weight == pytest.approx(0.3)
        assert cfg.default_vector_weight == pytest.approx(0.55)
        assert cfg.default_recency_weight == pytest.approx(0.1)
        assert cfg.default_priority_weight == pytest.approx(0.05)
        assert cfg.recency_half_life == 50


# ---------------------------------------------------------------------------
# Recency scoring
# ---------------------------------------------------------------------------


class TestRecencyScoring:
    """Tests for recency signal in hybrid search."""

    async def test_recency_score_favors_recent(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """More recently updated thoughts should score higher with recency."""
        old = await store.create_thought(
            _make("t-old", essence="Topic alpha", content="Alpha info", updated_cycle=10),
        )
        new = await store.create_thought(
            _make("t-new", essence="Topic alpha", content="Alpha info", updated_cycle=90),
        )
        # Store identical embeddings so vector score is the same.
        vec = [0.5] * 4
        await store.store_embedding(old.thought_id, vec, model_name="test")
        await store.store_embedding(new.thought_id, vec, model_name="test")

        result = await store.search_hybrid(
            "alpha",
            vec,
            fts_weight=0.0,
            vector_weight=0.5,
            recency_weight=0.5,
            recency_half_life=50,
            current_cycle=100,
        )
        assert len(result.results) == 2
        ids = [tid for tid, _ in result.results]
        assert ids[0] == "t-new"
        assert "recency" in result.backends_used

    async def test_recency_exp_decay_formula(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """Verify the exponential decay formula at half_life boundary."""
        await store.create_thought(
            _make("t-hl", essence="Decay test", content="Content", updated_cycle=50),
        )
        vec = [0.5] * 4
        await store.store_embedding("t-hl", vec, model_name="test")

        # At current_cycle=100; age=50; half_life=50 → score ≈ 0.5.
        result = await store.search_hybrid(
            "decay",
            vec,
            fts_weight=0.0,
            vector_weight=0.0,
            recency_weight=1.0,
            recency_half_life=50,
            current_cycle=100,
        )
        assert len(result.results) == 1
        _, score = result.results[0]
        expected = math.exp(-math.log(2) / 50 * 50)
        assert score == pytest.approx(expected, abs=0.01)

    async def test_recency_skipped_when_no_current_cycle(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """Recency is skipped when current_cycle=None."""
        await store.create_thought(_make("t-skip", essence="Skip test", content="Content"))
        vec = [0.5] * 4
        await store.store_embedding("t-skip", vec, model_name="test")

        result = await store.search_hybrid(
            "skip",
            vec,
            recency_weight=0.5,
            current_cycle=None,
        )
        assert "recency" not in result.backends_used


# ---------------------------------------------------------------------------
# Auto-embed query
# ---------------------------------------------------------------------------


class TestAutoEmbedQuery:
    """Tests for auto-embedding the query when query_vector is None."""

    async def test_auto_embed_with_provider(
        self,
        store_embed: SqliteEngravaCore,
    ) -> None:
        """search_hybrid auto-embeds query when provider configured."""
        await store_embed.create_thought(
            _make("t-ae", essence="Auto embed test", content="Search content"),
        )
        # Thought has auto-embedded vector; search with None vector.
        result = await store_embed.search_hybrid(
            "auto embed",
            None,
        )
        assert "vector" in result.backends_used
        assert len(result.results) >= 1

    async def test_no_provider_no_vector_skips_vector(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """Without provider and vector, vector search is skipped."""
        await store.create_thought(
            _make("t-nv", essence="No vector test", content="Text only"),
        )
        result = await store.search_hybrid("no vector", None)
        assert "vector" not in result.backends_used

    async def test_explicit_vector_overrides_auto_embed(
        self,
        store_embed: SqliteEngravaCore,
    ) -> None:
        """When caller passes explicit vector, auto-embed is not triggered."""
        await store_embed.create_thought(
            _make("t-eo", essence="Explicit override", content="Override test"),
        )
        explicit_vec = [0.9] * 4
        provider_mock = AsyncMock()
        store_embed._embedding_provider = provider_mock  # type: ignore[assignment]

        await store_embed.search_hybrid("explicit", explicit_vec)
        provider_mock.embed.assert_not_called()


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    """Tests for all degradation scenarios."""

    async def test_no_fts_no_vector_no_recency_fallback(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """All signals off → fallback to list_thoughts."""
        await store.create_thought(_make("t-fb", essence="Fallback", content="Test"))

        result = await store.search_hybrid(
            "",
            None,
            fts_weight=0.3,
            vector_weight=0.7,
            recency_weight=0.0,
            priority_weight=0.0,
        )
        # Should fallback and still return the thought.
        assert len(result.results) == 1
        assert result.results[0][0] == "t-fb"
        assert result.results[0][1] == 0.0

    async def test_recency_only_fallback_ranks_latest_thoughts(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """When FTS and vector are unavailable, recency fallback still returns results."""
        await store.create_thought(
            _make("t-old-fb", essence="Old fallback", content="Test", updated_cycle=10),
        )
        await store.create_thought(
            _make("t-new-fb", essence="New fallback", content="Test", updated_cycle=90),
        )

        result = await store.search_hybrid(
            "",
            None,
            recency_weight=1.0,
            priority_weight=0.0,
            current_cycle=100,
        )

        assert [thought_id for thought_id, _score in result.results] == [
            "t-new-fb",
            "t-old-fb",
        ]
        assert result.backends_used == frozenset({"recency"})

    async def test_vector_only_when_no_fts_text(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """Empty query_text → FTS skipped; vector only."""
        t = await store.create_thought(
            _make("t-vo", essence="Vector only", content="Stuff"),
        )
        vec = [0.5] * 4
        await store.store_embedding(t.thought_id, vec, model_name="test")

        result = await store.search_hybrid(
            "",
            vec,
            fts_weight=0.3,
            vector_weight=0.7,
        )
        assert "fts5" not in result.backends_used
        assert "vector" in result.backends_used

    async def test_fts_only_when_no_vector(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """query_vector=None + no provider → FTS only."""
        await store.create_thought(
            _make("t-fo", essence="FTS only test", content="Keyword search"),
        )
        result = await store.search_hybrid(
            "keyword",
            None,
            fts_weight=0.7,
            vector_weight=0.3,
        )
        assert "fts5" in result.backends_used
        assert "vector" not in result.backends_used

    async def test_weight_redistribution(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """Weights for disabled signals get redistributed."""
        t = await store.create_thought(
            _make("t-wr", essence="Redistribution test", content="Content"),
        )
        vec = [0.5] * 4
        await store.store_embedding(t.thought_id, vec, model_name="test")

        # recency disabled (no current_cycle); fts=0.3 vec=0.7 → effective 0.3/1.0, 0.7/1.0.
        result = await store.search_hybrid(
            "redistribution",
            vec,
            fts_weight=0.3,
            vector_weight=0.7,
            recency_weight=0.5,
            current_cycle=None,
        )
        assert len(result.results) >= 1
        assert "recency" not in result.backends_used
        # Score should be positive (weights redistributed to active signals).
        assert result.results[0][1] > 0.0


# ---------------------------------------------------------------------------
# Backward compatibility (legacy call patterns)
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    """Ensure legacy call patterns still work unchanged."""

    async def test_positional_query_vector(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """Old-style call: search_hybrid(text, vector) still works."""
        r = await store.search_hybrid("test", [0.1, 0.2, 0.3])
        assert r.results == []

    async def test_default_recency_weight_is_zero(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """Default recency_weight=0.0 means no recency impact."""
        await store.create_thought(
            _make("t-bc", essence="Compat test", content="Old style"),
        )
        vec = [0.5] * 4
        await store.store_embedding("t-bc", vec, model_name="test")

        result = await store.search_hybrid("compat", vec)
        assert "recency" not in result.backends_used


# ---------------------------------------------------------------------------
# from_config with search section
# ---------------------------------------------------------------------------


class TestFromConfigSearch:
    """Integration tests for from_config wiring search config."""

    async def test_from_config_with_search_section(self, tmp_path: Path) -> None:
        """from_config parses search: YAML section."""
        import yaml

        db_path = tmp_path / "test.db"
        cfg_path = tmp_path / "engrava.yaml"
        cfg_path.write_text(
            yaml.dump(
                {
                    "database": {"path": str(db_path)},
                    "search": {
                        "default_fts_weight": 0.2,
                        "default_vector_weight": 0.5,
                        "default_recency_weight": 0.3,
                        "recency_half_life": 100,
                    },
                }
            ),
            encoding="utf-8",
        )

        from engrava.config import load_config

        config = load_config(cfg_path)
        assert config.search.default_fts_weight == pytest.approx(0.2)
        assert config.search.default_vector_weight == pytest.approx(0.5)
        assert config.search.default_recency_weight == pytest.approx(0.3)
        assert config.search.recency_half_life == 100

    async def test_from_config_applies_search_defaults_at_runtime(
        self,
        tmp_path: Path,
    ) -> None:
        """Configured search defaults are used when per-call overrides are omitted."""
        import yaml

        db_path = tmp_path / "runtime.db"
        cfg_path = tmp_path / "engrava.yaml"
        cfg_path.write_text(
            yaml.dump(
                {
                    "database": {"path": str(db_path)},
                    "search": {
                        "default_fts_weight": 0.2,
                        "default_vector_weight": 0.3,
                        "default_recency_weight": 0.5,
                        "default_priority_weight": 0.0,
                        "default_graph_weight": 0.0,
                        "recency_half_life": 100,
                    },
                }
            ),
            encoding="utf-8",
        )

        async with await SqliteEngravaCore.from_config(cfg_path) as store:
            await store.create_thought(
                _make("t-runtime", essence="Runtime", content="Runtime", updated_cycle=100),
            )
            store.search_fts = AsyncMock(return_value=[("t-runtime", 10.0)])
            store.search_similar = AsyncMock(return_value=[("t-runtime", 0.4)])

            result = await store.search_hybrid("runtime", [0.1], current_cycle=100)

        # The single mocked FTS hit is the degenerate min-max case (hi == lo):
        # it normalizes to the neutral 0.5 (not the old 1.0). Fused score =
        # fts 0.5*0.2 + vector 0.4*0.3 + recency 1.0*0.5 = 0.10 + 0.12 + 0.50.
        assert result.results[0] == ("t-runtime", pytest.approx(0.72))

    async def test_from_config_without_search_uses_defaults(self, tmp_path: Path) -> None:
        """Missing search: section uses SearchConfig defaults."""
        import yaml

        db_path = tmp_path / "test.db"
        cfg_path = tmp_path / "engrava.yaml"
        cfg_path.write_text(
            yaml.dump({"database": {"path": str(db_path)}}),
            encoding="utf-8",
        )

        from engrava.config import load_config

        config = load_config(cfg_path)
        assert config.search == SearchConfig()


# ---------------------------------------------------------------------------
# Priority signal tests
# ---------------------------------------------------------------------------


class TestPrioritySignal:
    """Verify that priority acts as a 4th scoring signal in hybrid search."""

    async def test_priority_weight_contributes_to_score(self) -> None:
        """Priority weight produces a non-zero score contribution."""
        conn = await aiosqlite.connect(":memory:")
        conn.row_factory = aiosqlite.Row
        try:
            search_cfg = SearchConfig(
                default_fts_weight=0.0,
                default_vector_weight=0.0,
                default_recency_weight=0.0,
                default_priority_weight=1.0,
            )
            provider = CallbackProvider(callback=_dummy_embed, dimension=4, model_name="test-4")
            store = SqliteEngravaCore(
                conn,
                embedding_provider=provider,
                auto_embed=True,
                search_config=search_cfg,
            )
            await store.ensure_schema()

            t = _make("t-pri", essence="priority test", priority=Priority.P1)
            await store.create_thought(t)

            result = await store.search_hybrid(
                "priority test",
                priority_weight=1.0,
                fts_weight=0.0,
                vector_weight=0.0,
                recency_weight=0.0,
            )

            assert len(result.results) > 0
            assert "priority" in result.backends_used
            # P1 boost=1.0 → full priority score
            assert result.results[0][1] > 0.0
        finally:
            await conn.close()

    async def test_p1_outranks_p3_same_similarity(self) -> None:
        """P1 thought outranks P3 thought when vector similarity is identical."""
        conn = await aiosqlite.connect(":memory:")
        conn.row_factory = aiosqlite.Row
        try:
            search_cfg = SearchConfig(
                default_vector_weight=0.5,
                default_priority_weight=0.5,
                default_fts_weight=0.0,
                default_recency_weight=0.0,
            )
            provider = CallbackProvider(callback=_dummy_embed, dimension=4, model_name="test-4")
            store = SqliteEngravaCore(
                conn,
                embedding_provider=provider,
                auto_embed=True,
                search_config=search_cfg,
            )
            await store.ensure_schema()

            # Both have same essence/content → same embedding → same vector score
            t_p1 = _make("t-p1", essence="same text", content="same content", priority=Priority.P1)
            t_p3 = _make("t-p3", essence="same text", content="same content", priority=Priority.P3)
            await store.create_thought(t_p1)
            await store.create_thought(t_p3)

            result = await store.search_hybrid("same text")

            scores = dict(result.results)
            assert scores["t-p1"] > scores["t-p3"]
        finally:
            await conn.close()

    async def test_priority_disabled_identity_scoring(self) -> None:
        """When priority_weight=0.0, priority signal is not used."""
        conn = await aiosqlite.connect(":memory:")
        conn.row_factory = aiosqlite.Row
        try:
            search_cfg = SearchConfig(default_priority_weight=0.0)
            provider = CallbackProvider(callback=_dummy_embed, dimension=4, model_name="test-4")
            store = SqliteEngravaCore(
                conn,
                embedding_provider=provider,
                auto_embed=True,
                search_config=search_cfg,
            )
            await store.ensure_schema()

            t = _make("t-nopri", essence="no priority", priority=Priority.P1)
            await store.create_thought(t)

            result = await store.search_hybrid("no priority", priority_weight=0.0)

            assert "priority" not in result.backends_used
        finally:
            await conn.close()

    def test_priority_boost_config_validation(self) -> None:
        """Negative priority boost raises ConfigError."""
        with pytest.raises(ConfigError, match="priority_boost_p1"):
            _parse_search({"priority_boost_p1": -0.5})
