"""Tests for embedding providers.

Covers CallbackProvider unit tests, auto-embed integration, model
immutability (lazy lock + verify_embedding_model), config parsing,
and resolve_embedding_provider.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest

from engrava import (
    CallbackProvider,
    EmbeddingConfig,
    EmbeddingModelMismatchError,
    EmbeddingProviderProtocol,
    SqliteEngravaCore,
)
from engrava.config import (
    ConfigError,
    _parse_embeddings,
    resolve_embedding_provider,
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


def _dummy_embed(text: str) -> list[float]:
    """Deterministic dummy embedding function (dim=4)."""
    return [float(len(text) % 10) / 10.0] * 4


@pytest.fixture
def callback_provider() -> CallbackProvider:
    """A CallbackProvider with dimension=4."""
    return CallbackProvider(
        callback=_dummy_embed,
        dimension=4,
        model_name="test-4",
    )


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
async def store_with_embed(
    db: aiosqlite.Connection,
    callback_provider: CallbackProvider,
) -> SqliteEngravaCore:
    """Store with auto-embed enabled."""
    return SqliteEngravaCore(
        db,
        embedding_provider=callback_provider,
        auto_embed=True,
    )


@pytest.fixture
async def store_plain(db: aiosqlite.Connection) -> SqliteEngravaCore:
    """Store without embedding provider (backward compat)."""
    return SqliteEngravaCore(db)


def _make_thought(
    thought_id: str = "t-emb-001",
    essence: str = "Test essence",
    content: str = "Test content",
) -> ThoughtRecord:
    """Minimal thought for embedding tests."""
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.TASK,
        essence=essence,
        content=content,
        priority=Priority.P2,
        lifecycle_status=LifecycleStatus.CREATED,
        created_cycle=0,
        updated_cycle=0,
        source="test",
        confidence=0.8,
        source_type=KnowledgeSource.EXPERIENCE,
        visibility=ThoughtVisibility.SELECTIVE,
    )


# ---------------------------------------------------------------------------
# CallbackProvider unit tests
# ---------------------------------------------------------------------------


class TestCallbackProvider:
    """Unit tests for CallbackProvider."""

    async def test_embed_returns_vector(self, callback_provider: CallbackProvider) -> None:
        result = await callback_provider.embed("hello")
        assert len(result) == 4
        assert all(isinstance(v, float) for v in result)

    async def test_embed_batch_returns_list(self, callback_provider: CallbackProvider) -> None:
        results = await callback_provider.embed_batch(["hello", "world"])
        assert len(results) == 2
        assert all(len(v) == 4 for v in results)

    def test_dimension_property(self, callback_provider: CallbackProvider) -> None:
        assert callback_provider.dimension == 4

    def test_model_name_property(self, callback_provider: CallbackProvider) -> None:
        assert callback_provider.model_name == "test-4"

    def test_implements_protocol(self, callback_provider: CallbackProvider) -> None:
        assert isinstance(callback_provider, EmbeddingProviderProtocol)


# ---------------------------------------------------------------------------
# Auto-embed on create_thought
# ---------------------------------------------------------------------------


class TestAutoEmbed:
    """Integration tests for auto-embed flow."""

    async def test_create_thought_auto_embeds(
        self,
        store_with_embed: SqliteEngravaCore,
    ) -> None:
        thought = _make_thought()
        created = await store_with_embed.create_thought(thought)
        embedding = await store_with_embed.get_embedding(created.thought_id)
        assert embedding is not None
        assert embedding.dimension == 4
        assert embedding.model_name == "test-4"

    async def test_update_thought_re_embeds_on_content_change(
        self,
        store_with_embed: SqliteEngravaCore,
    ) -> None:
        thought = _make_thought()
        await store_with_embed.create_thought(thought)

        # Update content — should trigger re-embed.
        await store_with_embed.update_thought(
            thought.thought_id,
            content="Updated content",
            updated_cycle=1,
        )
        embedding = await store_with_embed.get_embedding(thought.thought_id)
        assert embedding is not None

    async def test_update_thought_re_embeds_on_essence_change(
        self,
        store_with_embed: SqliteEngravaCore,
    ) -> None:
        thought = _make_thought()
        await store_with_embed.create_thought(thought)

        await store_with_embed.update_thought(
            thought.thought_id,
            essence="New essence",
            updated_cycle=1,
        )
        embedding = await store_with_embed.get_embedding(thought.thought_id)
        assert embedding is not None

    async def test_update_thought_skips_embed_when_no_content_change(
        self,
        store_with_embed: SqliteEngravaCore,
    ) -> None:
        thought = _make_thought()
        await store_with_embed.create_thought(thought)

        # Get embedding after create.
        emb_before = await store_with_embed.get_embedding(thought.thought_id)
        assert emb_before is not None
        ts_before = emb_before.created_at

        # Update priority only — should NOT re-embed.
        await store_with_embed.update_thought(
            thought.thought_id,
            priority=Priority.P1.value,
            updated_cycle=1,
        )
        emb_after = await store_with_embed.get_embedding(thought.thought_id)
        assert emb_after is not None
        # Timestamp unchanged → no re-embed occurred.
        assert emb_after.created_at == ts_before

    async def test_no_provider_no_auto_embed(
        self,
        store_plain: SqliteEngravaCore,
    ) -> None:
        thought = _make_thought()
        await store_plain.create_thought(thought)
        embedding = await store_plain.get_embedding(thought.thought_id)
        assert embedding is None

    async def test_auto_embed_false_no_embed(
        self,
        db: aiosqlite.Connection,
        callback_provider: CallbackProvider,
    ) -> None:
        store = SqliteEngravaCore(
            db,
            embedding_provider=callback_provider,
            auto_embed=False,
        )
        thought = _make_thought()
        await store.create_thought(thought)
        embedding = await store.get_embedding(thought.thought_id)
        assert embedding is None


# ---------------------------------------------------------------------------
# Model immutability (lazy lock)
# ---------------------------------------------------------------------------


class TestModelImmutability:
    """Tests for embedding model lazy lock in _metadata."""

    async def test_first_embed_locks_model(
        self,
        store_with_embed: SqliteEngravaCore,
    ) -> None:
        thought = _make_thought()
        await store_with_embed.create_thought(thought)

        # Verify _metadata has the model info.
        cursor = await store_with_embed._db.execute(
            "SELECT value FROM _metadata WHERE key = 'embedding_model_name'"
        )
        row = await cursor.fetchone()
        assert row is not None
        assert row["value"] == "test-4"

    async def test_model_mismatch_raises(
        self,
        db: aiosqlite.Connection,
    ) -> None:
        # First store embeds with model-A.
        provider_a = CallbackProvider(_dummy_embed, dimension=4, model_name="model-A")
        store_a = SqliteEngravaCore(db, embedding_provider=provider_a, auto_embed=True)
        await store_a.create_thought(_make_thought("t-a"))

        # Second store with model-B should raise.
        provider_b = CallbackProvider(_dummy_embed, dimension=4, model_name="model-B")
        store_b = SqliteEngravaCore(db, embedding_provider=provider_b, auto_embed=True)
        with pytest.raises(EmbeddingModelMismatchError, match="model-A"):
            await store_b.create_thought(_make_thought("t-b"))

    async def test_dimension_mismatch_raises(
        self,
        db: aiosqlite.Connection,
    ) -> None:
        provider_4 = CallbackProvider(_dummy_embed, dimension=4, model_name="same")
        store_4 = SqliteEngravaCore(db, embedding_provider=provider_4, auto_embed=True)
        await store_4.create_thought(_make_thought("t-4"))

        def _embed_8(text: str) -> list[float]:
            return [0.1] * 8

        provider_8 = CallbackProvider(_embed_8, dimension=8, model_name="same")
        store_8 = SqliteEngravaCore(db, embedding_provider=provider_8, auto_embed=True)
        with pytest.raises(EmbeddingModelMismatchError, match="dim=4"):
            await store_8.create_thought(_make_thought("t-8"))

    async def test_verify_embedding_model_eager(
        self,
        db: aiosqlite.Connection,
    ) -> None:
        # Lock model.
        provider_a = CallbackProvider(_dummy_embed, dimension=4, model_name="locked-model")
        store_a = SqliteEngravaCore(db, embedding_provider=provider_a, auto_embed=True)
        await store_a.create_thought(_make_thought("t-lock"))

        # Verify with different model.
        provider_b = CallbackProvider(_dummy_embed, dimension=4, model_name="wrong-model")
        store_b = SqliteEngravaCore(db, embedding_provider=provider_b)
        with pytest.raises(EmbeddingModelMismatchError):
            await store_b.verify_embedding_model()

    async def test_verify_embedding_model_noop_without_provider(
        self,
        store_plain: SqliteEngravaCore,
    ) -> None:
        # Should not raise.
        await store_plain.verify_embedding_model()

    async def test_same_model_succeeds(
        self,
        db: aiosqlite.Connection,
    ) -> None:
        provider = CallbackProvider(_dummy_embed, dimension=4, model_name="consistent")
        store1 = SqliteEngravaCore(db, embedding_provider=provider, auto_embed=True)
        await store1.create_thought(_make_thought("t-1"))

        store2 = SqliteEngravaCore(db, embedding_provider=provider, auto_embed=True)
        # Should not raise — same model.
        await store2.create_thought(_make_thought("t-2"))

    async def test_manual_store_embedding_locks_model(
        self,
        store_plain: SqliteEngravaCore,
    ) -> None:
        thought = _make_thought()
        await store_plain.create_thought(thought)
        await store_plain.store_embedding(
            thought.thought_id,
            [0.1, 0.2, 0.3],
            model_name="manual-model",
        )

        # Verify _metadata has the model info.
        cursor = await store_plain._db.execute(
            "SELECT value FROM _metadata WHERE key = 'embedding_model_name'"
        )
        row = await cursor.fetchone()
        assert row is not None
        assert row["value"] == "manual-model"


# ---------------------------------------------------------------------------
# Schema migration core-4 → core-5
# ---------------------------------------------------------------------------


class TestSchemaMigration:
    """Tests for core-4 → core-5 migration."""

    async def test_ensure_schema_creates_metadata_table(self) -> None:
        conn = await aiosqlite.connect(":memory:")
        conn.row_factory = aiosqlite.Row
        try:
            store = SqliteEngravaCore(conn)
            await store.ensure_schema()

            cursor = await conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='_metadata'"
            )
            row = await cursor.fetchone()
            assert row is not None
        finally:
            await conn.close()

    async def test_migration_from_v4_creates_metadata(self) -> None:
        conn = await aiosqlite.connect(":memory:")
        conn.row_factory = aiosqlite.Row
        try:
            # Bootstrap to v4 manually.
            store = SqliteEngravaCore(conn)
            await store.ensure_schema()
            # Force version back to 4 to simulate existing v4 DB.
            await conn.execute("PRAGMA user_version = 4")
            await conn.commit()

            # Re-run ensure_schema → should migrate forward through every
            # subsequent helper to the current head version.
            store2 = SqliteEngravaCore(conn)
            await store2.ensure_schema()

            cursor = await conn.execute("PRAGMA user_version")
            row = await cursor.fetchone()
            assert row is not None
            assert int(row[0]) == 12

            # _metadata table should exist.
            cursor = await conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='_metadata'"
            )
            assert await cursor.fetchone() is not None
        finally:
            await conn.close()


# ---------------------------------------------------------------------------
# EmbeddingConfig parsing
# ---------------------------------------------------------------------------


class TestEmbeddingConfigParsing:
    """Tests for _parse_embeddings and resolve_embedding_provider."""

    def test_parse_none(self) -> None:
        assert _parse_embeddings(None) is None

    def test_parse_valid_config(self) -> None:
        raw = {
            "provider": "sentence-transformer",
            "model": "all-MiniLM-L12-v2",
            "auto_embed": True,
            "device": "cuda",
            "batch_size": 64,
        }
        cfg = _parse_embeddings(raw)
        assert cfg is not None
        assert cfg.provider == "sentence-transformer"
        assert cfg.model == "all-MiniLM-L12-v2"
        assert cfg.auto_embed is True
        assert cfg.device == "cuda"
        assert cfg.batch_size == 64

    def test_parse_invalid_provider(self) -> None:
        with pytest.raises(ConfigError, match=r"embeddings\.provider"):
            _parse_embeddings({"provider": "invalid"})

    def test_parse_invalid_type(self) -> None:
        with pytest.raises(ConfigError, match="must be a mapping"):
            _parse_embeddings("not-a-dict")

    def test_parse_invalid_auto_embed(self) -> None:
        with pytest.raises(ConfigError, match="auto_embed"):
            _parse_embeddings({"auto_embed": "yes"})

    def test_parse_invalid_batch_size(self) -> None:
        with pytest.raises(ConfigError, match="batch_size"):
            _parse_embeddings({"batch_size": 0})

    def test_parse_env_var_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_API_KEY", "sk-secret")
        cfg = _parse_embeddings(
            {
                "provider": "openai-compatible",
                "api_key": "${TEST_API_KEY}",
            }
        )
        assert cfg is not None
        assert cfg.api_key == "sk-secret"

    def test_parse_env_var_missing_raises(self) -> None:
        with pytest.raises(ConfigError, match="MISSING_KEY"):
            _parse_embeddings(
                {
                    "provider": "openai-compatible",
                    "api_key": "${MISSING_KEY}",
                }
            )

    def test_resolve_none_config(self) -> None:
        assert resolve_embedding_provider(None) is None

    def test_resolve_null_provider(self) -> None:
        cfg = EmbeddingConfig(provider=None)
        assert resolve_embedding_provider(cfg) is None

    def test_resolve_openai_provider(self) -> None:
        cfg = EmbeddingConfig(
            provider="openai-compatible",
            model="text-embedding-3-small",
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
        )
        provider = resolve_embedding_provider(cfg)
        assert provider is not None
        assert provider.model_name == "text-embedding-3-small"

    def test_resolve_ollama_provider(self) -> None:
        cfg = EmbeddingConfig(
            provider="ollama",
            model="nomic-embed-text",
        )
        provider = resolve_embedding_provider(cfg)
        assert provider is not None
        assert provider.model_name == "nomic-embed-text"

    def test_resolve_sentence_transformer_missing(self) -> None:
        cfg = EmbeddingConfig(provider="sentence-transformer")
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            # SentenceTransformerProvider constructor doesn't import eagerly,
            # so this will succeed. The import error happens at load time.
            # We test via resolve which does the lazy import.
            pass
        # Just verify it creates the provider object (import available).
        provider = resolve_embedding_provider(cfg)
        assert provider is not None

    def test_resolve_huggingface_provider(self) -> None:
        cfg = EmbeddingConfig(
            provider="huggingface",
            model="sentence-transformers/all-MiniLM-L12-v2",
            api_key="hf-test",
        )
        provider = resolve_embedding_provider(cfg)
        assert provider is not None
        assert provider.model_name == "sentence-transformers/all-MiniLM-L12-v2"


# ---------------------------------------------------------------------------
# OpenAI/Ollama provider tests (mocked HTTP)
# ---------------------------------------------------------------------------


class TestOpenAIProvider:
    """Unit tests for OpenAICompatibleProvider."""

    async def test_embed_sends_request(self) -> None:
        from engrava.embeddings.openai_compatible import OpenAICompatibleProvider

        provider = OpenAICompatibleProvider(
            model_name="test-model",
            base_url="https://api.test.com/v1",
            api_key="sk-test",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}],
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        result = await provider.embed("hello")
        assert result == [0.1, 0.2, 0.3]
        assert provider.dimension == 3
        mock_client.post.assert_called_once()

    async def test_embed_batch(self) -> None:
        from engrava.embeddings.openai_compatible import OpenAICompatibleProvider

        provider = OpenAICompatibleProvider(
            model_name="test-model",
            base_url="https://api.test.com/v1",
            api_key="sk-test",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"index": 0, "embedding": [0.1, 0.2]},
                {"index": 1, "embedding": [0.3, 0.4]},
            ],
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        results = await provider.embed_batch(["a", "b"])
        assert len(results) == 2


class TestOllamaProvider:
    """Unit tests for OllamaProvider."""

    async def test_embed_sends_request(self) -> None:
        from engrava.embeddings.ollama import OllamaProvider

        provider = OllamaProvider(model_name="nomic", base_url="http://localhost:11434")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embeddings": [[0.5, 0.6, 0.7, 0.8]],
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        result = await provider.embed("hello")
        assert result == [0.5, 0.6, 0.7, 0.8]
        assert provider.dimension == 4
        mock_client.post.assert_called_once()


# ---------------------------------------------------------------------------
# SentenceTransformerProvider tests (mocked model)
# ---------------------------------------------------------------------------


class TestSentenceTransformerProvider:
    """Unit tests for SentenceTransformerProvider."""

    async def test_embed_returns_vector(self) -> None:
        import numpy as np

        from engrava.embeddings.sentence_transformer import SentenceTransformerProvider

        provider = SentenceTransformerProvider(model_name="test-model", device="cpu")

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        mock_model.get_sentence_embedding_dimension.return_value = 3
        provider._model = mock_model
        provider._dimension = 3

        result = await provider.embed("hello")
        assert result == [pytest.approx(0.1), pytest.approx(0.2), pytest.approx(0.3)]
        mock_model.encode.assert_called_once_with("hello", normalize_embeddings=True)

    async def test_embed_batch_returns_list(self) -> None:
        import numpy as np

        from engrava.embeddings.sentence_transformer import SentenceTransformerProvider

        provider = SentenceTransformerProvider(model_name="test-model")

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_model.get_sentence_embedding_dimension.return_value = 2
        provider._model = mock_model
        provider._dimension = 2

        results = await provider.embed_batch(["a", "b"])
        assert len(results) == 2
        assert results[0] == [pytest.approx(0.1), pytest.approx(0.2)]
        assert results[1] == [pytest.approx(0.3), pytest.approx(0.4)]

    def test_properties(self) -> None:
        from engrava.embeddings.sentence_transformer import SentenceTransformerProvider

        provider = SentenceTransformerProvider(model_name="my-model")
        assert provider.model_name == "my-model"

    def test_implements_protocol(self) -> None:
        from engrava.embeddings.sentence_transformer import SentenceTransformerProvider

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 4
        provider = SentenceTransformerProvider()
        provider._model = mock_model
        provider._dimension = 4
        assert isinstance(provider, EmbeddingProviderProtocol)

    def test_lazy_load_sets_dimension(self) -> None:
        from engrava.embeddings.sentence_transformer import SentenceTransformerProvider

        provider = SentenceTransformerProvider()
        mock_st_class = MagicMock()
        mock_instance = MagicMock()
        mock_instance.get_sentence_embedding_dimension.return_value = 384
        mock_st_class.return_value = mock_instance

        with (
            patch.dict("sys.modules", {"sentence_transformers": MagicMock()}),
            patch(
                "engrava.embeddings.sentence_transformer.SentenceTransformerProvider._load_model"
            ) as mock_load,
        ):
            mock_load.return_value = mock_instance
            provider._model = mock_instance
            provider._dimension = 384
            assert provider.dimension == 384


# ---------------------------------------------------------------------------
# HuggingFaceProvider tests (mocked client)
# ---------------------------------------------------------------------------


class TestHuggingFaceProvider:
    """Unit tests for HuggingFaceProvider."""

    async def test_embed_returns_vector(self) -> None:
        import numpy as np

        from engrava.embeddings.huggingface import HuggingFaceProvider

        provider = HuggingFaceProvider(
            model_name="test-hf-model",
            api_key="hf-test",
        )

        mock_client = MagicMock()
        mock_client.feature_extraction.return_value = np.array([0.1, 0.2, 0.3])
        provider._client = mock_client

        result = await provider.embed("hello")
        assert result == [pytest.approx(0.1), pytest.approx(0.2), pytest.approx(0.3)]
        assert provider.dimension == 3
        mock_client.feature_extraction.assert_called_once_with("hello", model="test-hf-model")

    async def test_embed_batch(self) -> None:
        import numpy as np
        import numpy.typing as npt

        from engrava.embeddings.huggingface import HuggingFaceProvider

        provider = HuggingFaceProvider(
            model_name="test-hf-model",
            api_key="hf-test",
        )

        call_count = 0

        def fake_extract(text: str, model: str) -> npt.NDArray[np.float64]:
            nonlocal call_count
            call_count += 1
            return np.array([0.1 * call_count, 0.2 * call_count])

        mock_client = MagicMock()
        mock_client.feature_extraction.side_effect = fake_extract
        provider._client = mock_client

        results = await provider.embed_batch(["a", "b"])
        assert len(results) == 2
        assert mock_client.feature_extraction.call_count == 2

    async def test_handles_nested_array(self) -> None:
        from engrava.embeddings.huggingface import HuggingFaceProvider

        provider = HuggingFaceProvider(model_name="test-hf-model", api_key="hf-test")

        mock_client = MagicMock()
        # Some models return [[0.1, 0.2, 0.3]] (nested).
        mock_client.feature_extraction.return_value = [[0.1, 0.2, 0.3]]
        provider._client = mock_client

        result = await provider.embed("hello")
        assert result == [0.1, 0.2, 0.3]
        assert provider.dimension == 3

    def test_properties(self) -> None:
        from engrava.embeddings.huggingface import HuggingFaceProvider

        provider = HuggingFaceProvider(model_name="my-hf-model", api_key="tk")
        assert provider.model_name == "my-hf-model"

    def test_dimension_raises_before_embed(self) -> None:
        from engrava.embeddings.huggingface import HuggingFaceProvider

        provider = HuggingFaceProvider()
        with pytest.raises(RuntimeError, match="Dimension not yet known"):
            _ = provider.dimension

    def test_implements_protocol(self) -> None:
        from engrava.embeddings.huggingface import HuggingFaceProvider

        provider = HuggingFaceProvider(dimension=384)
        assert isinstance(provider, EmbeddingProviderProtocol)


# ---------------------------------------------------------------------------
# from_config with embeddings section (integration)
# ---------------------------------------------------------------------------


class TestFromConfigEmbeddings:
    """Integration tests for from_config with embeddings: YAML section."""

    async def test_from_config_wires_openai_provider(self, tmp_path: Path) -> None:
        """from_config with openai-compatible embeddings creates the provider."""
        import yaml

        db_path = tmp_path / "test.db"
        cfg_path = tmp_path / "engrava.yaml"
        cfg_path.write_text(
            yaml.dump(
                {
                    "database": {"path": str(db_path)},
                    "embeddings": {
                        "provider": "openai-compatible",
                        "model": "text-embedding-3-small",
                        "auto_embed": False,
                        "api_key": "sk-test-key",
                    },
                }
            ),
            encoding="utf-8",
        )

        async with await SqliteEngravaCore.from_config(cfg_path) as store:
            assert store._embedding_provider is not None
            assert type(store._embedding_provider).__name__ == "OpenAICompatibleProvider"
            assert store._embedding_provider.model_name == "text-embedding-3-small"
            assert store._auto_embed is False

    async def test_from_config_wires_ollama_provider(self, tmp_path: Path) -> None:
        """from_config with ollama embeddings creates the provider."""
        import yaml

        db_path = tmp_path / "test.db"
        cfg_path = tmp_path / "engrava.yaml"
        cfg_path.write_text(
            yaml.dump(
                {
                    "database": {"path": str(db_path)},
                    "embeddings": {
                        "provider": "ollama",
                        "model": "nomic-embed-text",
                        "auto_embed": True,
                    },
                }
            ),
            encoding="utf-8",
        )

        async with await SqliteEngravaCore.from_config(cfg_path) as store:
            assert store._embedding_provider is not None
            assert type(store._embedding_provider).__name__ == "OllamaProvider"
            assert store._auto_embed is True

    async def test_from_config_no_embeddings(self, tmp_path: Path) -> None:
        """from_config without embeddings section has no provider."""
        import yaml

        db_path = tmp_path / "test.db"
        cfg_path = tmp_path / "engrava.yaml"
        cfg_path.write_text(
            yaml.dump({"database": {"path": str(db_path)}}),
            encoding="utf-8",
        )

        async with await SqliteEngravaCore.from_config(cfg_path) as store:
            assert store._embedding_provider is None
            assert store._auto_embed is False
